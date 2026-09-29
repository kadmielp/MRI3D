from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product
from pathlib import Path
from threading import Event
from typing import Callable
import hashlib
import numpy as np
import pydicom
from pydicom.uid import ImplicitVRLittleEndian
import SimpleITK as sitk


class Cancelled(Exception):
    pass


class ImagingError(ValueError):
    pass


class Job:
    def __init__(self, report: Callable[[int, str], None] | None = None):
        self.cancelled = Event()
        self.report = report or (lambda value, message: None)

    def check(self):
        if self.cancelled.is_set():
            raise Cancelled("Operation cancelled")

    def progress(self, value: int, message: str):
        self.check()
        self.report(value, message)


@dataclass
class Slice:
    path: str
    position: tuple
    sop_uid: str


@dataclass
class Series:
    key: str
    uid: str
    description: str
    rows: int
    columns: int
    pixel_spacing: tuple
    orientation: tuple
    slices: list[Slice] = field(default_factory=list)
    thickness: float = 0.0
    issues: list[str] = field(default_factory=list)
    spacing_z: float = 0.0
    localizer: bool = False

    @property
    def usable(self):
        return not self.issues and len(self.slices) >= 2 and not self.localizer

    def validate(self):
        row, col = np.array(self.orientation).reshape(2, 3)
        if not (np.isclose(np.linalg.norm(row), 1, atol=1e-4) and
                np.isclose(np.linalg.norm(col), 1, atol=1e-4) and
                abs(np.dot(row, col)) < 1e-4):
            self.issues.append("Invalid direction cosines")
            return
        normal = np.cross(row, col)
        self.slices.sort(key=lambda s: np.dot(s.position, normal))
        if len(self.slices) < 2:
            self.issues.append("At least two parallel slices are needed")
            return
        positions = np.array([s.position for s in self.slices])
        differences = np.diff(positions @ normal)
        self.spacing_z = float(np.median(differences))
        if np.any(differences < 0.001):
            self.issues.append("Duplicate slice positions; select a single acquisition")
        if self.spacing_z > 0 and not np.allclose(differences, self.spacing_z, rtol=0.01, atol=0.01):
            self.issues.append("Irregular spacing or missing slices; cannot safely reconstruct")
        expected = positions[0] + np.outer((positions - positions[0]) @ normal, normal)
        if not np.allclose(positions, expected, atol=0.05):
            self.issues.append("In-plane slice shifts are unsupported")
        if self.localizer:
            self.issues.append("Localizer / scout: excluded from 3D reconstruction")


@dataclass
class Scan:
    series: list[Series]
    ignored: int
    warnings: list[str]


@dataclass
class Volume:
    image: sitk.Image
    data: np.ndarray
    series: Series
    source_geometry: dict
    invert: bool = False
    # Source series of a fused volume; None for a single-series volume.
    components: list[Series] | None = None

    @property
    def spacing(self):
        return self.image.GetSpacing()

    @property
    def origin(self):
        return self.image.GetOrigin()

    def world(self, xyz):
        return self.image.TransformContinuousIndexToPhysicalPoint(tuple(float(v) for v in xyz))


def geometry(image):
    return dict(size=list(image.GetSize()), spacing=list(image.GetSpacing()),
                origin=list(image.GetOrigin()), direction=list(image.GetDirection()))


def read_dicom(path, pixels=False):
    ds = pydicom.dcmread(path, stop_before_pixels=not pixels, force=True)
    # Some older scanner exports omit the Part 10 header entirely.
    if "TransferSyntaxUID" not in ds.file_meta:
        if ds.original_encoding != (True, True):
            raise ImagingError("DICOM without transfer syntax must use implicit little endian")
        ds.file_meta.TransferSyntaxUID = ImplicitVRLittleEndian
    return ds


def scan_folder(folder: str, job: Job | None = None) -> Scan:
    job = job or Job()
    root = Path(folder)
    if not root.is_dir():
        raise ImagingError("Choose an existing DICOM folder")
    files = []
    for path in root.rglob("*"):
        job.check()
        if path.is_file():
            files.append(path)
    groups = {}
    ignored = 0
    warnings = []
    for index, path in enumerate(files):
        job.progress(int(index * 95 / max(1, len(files))), f"Reading scan headers · {index + 1}/{len(files)}")
        try:
            ds = read_dicom(path)
            if not getattr(ds, "Modality", None):
                raise ImagingError("Missing DICOM modality")
            if getattr(ds, "Modality", None) != "MR":
                ignored += 1
                continue
            if int(getattr(ds, "NumberOfFrames", 1)) != 1:
                warnings.append("Enhanced / multiframe MRI is unsupported in this version")
                ignored += 1
                continue
            uid = str(ds.SeriesInstanceUID)
            orient = tuple(float(v) for v in ds.ImageOrientationPatient)
            spacing = tuple(float(v) for v in ds.PixelSpacing)
            pos = tuple(float(v) for v in ds.ImagePositionPatient)
            if len(orient) != 6 or len(spacing) != 2 or len(pos) != 3:
                raise ImagingError("Incomplete spatial metadata")
            if not np.isfinite(orient + spacing + pos).all() or min(spacing) <= 0:
                raise ImagingError("Invalid spatial metadata")
            group = (uid, int(ds.Rows), int(ds.Columns), tuple(round(v, 5) for v in orient),
                     tuple(round(v, 5) for v in spacing), str(getattr(ds, "EchoNumbers", "")),
                     str(getattr(ds, "TemporalPositionIdentifier", "")),
                     str(getattr(ds, "ComplexImageComponent", "")))
            if group not in groups:
                description = str(getattr(ds, "SeriesDescription", "Unnamed MRI series"))
                image_type = " ".join(getattr(ds, "ImageType", [])).upper()
                scout = any(s in (image_type + " " + description.upper()) for s in
                            ("LOCALIZER", "SCOUT", "3-PLANE", "3 PLANE"))
                groups[group] = Series(hashlib.sha256(repr(group).encode()).hexdigest()[:20], uid,
                                       description, int(ds.Rows), int(ds.Columns), spacing, orient,
                                       thickness=float(getattr(ds, "SliceThickness", 0)), localizer=scout)
            groups[group].slices.append(Slice(str(path.resolve()), pos, str(ds.SOPInstanceUID)))
        except Exception as exc:
            if isinstance(exc, Cancelled):
                raise
            ignored += 1
            if path.suffix.lower() == ".dcm":
                warnings.append("Unreadable DICOM or missing required metadata: " + type(exc).__name__)
    result = sorted(groups.values(), key=lambda s: (s.description.casefold(), s.key))
    for series in result:
        series.validate()
    job.progress(100, f"Found {len(result)} compatible series groups")
    return Scan(result, ignored, sorted(set(warnings)))


def canonicalize(image: sitk.Image, job: Job) -> sitk.Image:
    oriented = sitk.DICOMOrient(image, "LPS")
    if np.allclose(oriented.GetDirection(), np.eye(3).ravel(), atol=1e-6):
        oriented.SetDirection(tuple(np.eye(3).ravel()))
        return oriented
    corners = np.array([oriented.TransformContinuousIndexToPhysicalPoint(tuple(float(x) for x in corner))
                        for corner in product(*[(-0.5, n - 0.5) for n in oriented.GetSize()])])
    spacing = np.array(oriented.GetSpacing())
    low, high = corners.min(axis=0), corners.max(axis=0)
    size = np.ceil((high - low) / spacing).astype(int)
    if np.prod(size.astype(float)) > 160_000_000:
        raise ImagingError("Reoriented volume exceeds the 160 million voxel memory limit")
    filt = sitk.ResampleImageFilter()
    filt.SetSize([int(n) for n in size])
    filt.SetOutputSpacing(tuple(spacing))
    filt.SetOutputOrigin(tuple(low + spacing / 2))
    filt.SetOutputDirection(tuple(np.eye(3).ravel()))
    filt.SetInterpolator(sitk.sitkLinear)
    filt.SetDefaultPixelValue(float(sitk.GetArrayViewFromImage(image).min()))
    filt.AddCommand(sitk.sitkProgressEvent, lambda: filt.Abort() if job.cancelled.is_set() else None)
    result = filt.Execute(oriented)
    job.check()
    return result


def load_series(series: Series, job: Job | None = None) -> Volume:
    job = job or Job()
    job.check()
    if not series.usable:
        raise ImagingError("; ".join(series.issues) or "This series cannot be reconstructed")
    if series.rows * series.columns * len(series.slices) > 160_000_000:
        raise ImagingError("Series exceeds the 160 million voxel memory limit")
    arrays = []
    photometric = None
    for i, item in enumerate(series.slices):
        job.progress(int(i / len(series.slices) * 75), f"Decoding slice {i + 1}/{len(series.slices)}")
        ds = read_dicom(item.path, pixels=True)
        if str(ds.SOPInstanceUID) != item.sop_uid or str(ds.SeriesInstanceUID) != series.uid:
            raise ImagingError("Source DICOM changed after import; scan the folder again")
        if (not np.allclose(tuple(float(v) for v in ds.ImagePositionPatient), item.position, atol=1e-5, rtol=0)
                or not np.allclose(tuple(float(v) for v in ds.ImageOrientationPatient), series.orientation, atol=1e-4, rtol=0)
                or not np.allclose(tuple(float(v) for v in ds.PixelSpacing), series.pixel_spacing, atol=1e-5, rtol=0)):
            raise ImagingError("Source geometry changed after import; scan the folder again")
        if int(getattr(ds, "SamplesPerPixel", 1)) != 1 or getattr(ds, "PhotometricInterpretation", "") not in ("MONOCHROME1", "MONOCHROME2"):
            raise ImagingError("Only monochrome MRI pixel data is supported")
        if photometric is not None and photometric != ds.PhotometricInterpretation:
            raise ImagingError("Mixed photometric interpretations in one stack")
        photometric = ds.PhotometricInterpretation
        try:
            pixels = ds.pixel_array.astype(np.float32)
        except Exception as exc:
            raise ImagingError(f"Unable to decode pixel data. This transfer syntax may need another decoder. {exc}") from exc
        if pixels.shape != (series.rows, series.columns):
            raise ImagingError("Pixel dimensions do not match the series")
        pixels = pixels * float(getattr(ds, "RescaleSlope", 1)) + float(getattr(ds, "RescaleIntercept", 0))
        if not np.isfinite(pixels).all():
            raise ImagingError("Non-finite voxel values are unsupported")
        arrays.append(pixels)
    image = sitk.GetImageFromArray(np.stack(arrays))
    image.SetSpacing((series.pixel_spacing[1], series.pixel_spacing[0], series.spacing_z))
    image.SetOrigin(series.slices[0].position)
    row, col = np.array(series.orientation).reshape(2, 3)
    image.SetDirection(tuple(np.column_stack([row, col, np.cross(row, col)]).ravel()))
    source_geometry = geometry(image)
    job.progress(80, "Aligning views to patient LPS coordinates")
    image = canonicalize(image, job)
    data = sitk.GetArrayFromImage(image)
    job.progress(100, "Volume ready")
    return Volume(image, data, series, source_geometry, photometric == "MONOCHROME1")


def fuse_series(series_list: list[Series], job: Job | None = None) -> Volume:
    job = job or Job()
    job.check()
    if not series_list:
        raise ImagingError("Select at least one series to open or fuse")
    if len(series_list) == 1:
        return load_series(series_list[0], job)

    for s in series_list:
        if not s.usable:
            raise ImagingError(f"Series '{s.description}' cannot be reconstructed: {'; '.join(s.issues)}")

    # 1. Load and canonicalize each series
    volumes = []
    n_series = len(series_list)
    for idx, s in enumerate(series_list):
        job.progress(int(idx / n_series * 40), f"Decoding series {idx + 1}/{n_series}: {s.description}")
        vol = load_series(s, job)
        volumes.append(vol)

    job.progress(45, "Calculating unified physical LPS bounding box")
    bounds = []
    for vol in volumes:
        img = vol.image
        sz = np.array(img.GetSize())
        sp = np.array(img.GetSpacing())
        orig = np.array(img.GetOrigin())
        low = orig - sp / 2.0
        high = orig + (sz - 0.5) * sp
        bounds.append((low, high))

    all_lows = np.array([b[0] for b in bounds])
    all_highs = np.array([b[1] for b in bounds])
    union_low = all_lows.min(axis=0)
    union_high = all_highs.max(axis=0)

    # Determine target isotropic spacing based on finest in-plane resolution
    min_in_plane = min(min(s.pixel_spacing) for s in series_list)
    target_sp = float(np.clip(min_in_plane, 0.25, 0.6))
    
    extent = union_high - union_low
    target_size = np.ceil(extent / target_sp).astype(int)
    
    # Safety: clamp size so total voxel count <= 40 million voxels
    total_voxels = np.prod(target_size.astype(float))
    if total_voxels > 40_000_000:
        scale_factor = (total_voxels / 40_000_000) ** (1.0 / 3.0)
        target_sp *= scale_factor
        target_size = np.ceil(extent / target_sp).astype(int)

    ref = sitk.Image([int(n) for n in target_size], sitk.sitkFloat32)
    ref.SetSpacing((target_sp, target_sp, target_sp))
    ref.SetOrigin(tuple(union_low + target_sp / 2.0))
    ref.SetDirection(tuple(np.eye(3).ravel()))

    def get_taper_1d(length, margin=6):
        if length <= 2 * margin:
            return np.ones(length, dtype=np.float32)
        w = np.ones(length, dtype=np.float32)
        ramp = 0.5 * (1.0 - np.cos(np.linspace(0, np.pi, margin, dtype=np.float32)))
        w[:margin] = ramp
        w[-margin:] = ramp[::-1]
        return w

    job.progress(55, "Resampling orthogonal series onto isotropic grid")
    resampled_arrays = []
    gradient_arrays = []
    mask_arrays = []
    
    for idx, vol in enumerate(volumes):
        job.progress(55 + int(idx / n_series * 25), f"Resampling plane {idx + 1}/{n_series} ({vol.series.description})")
        sz = vol.image.GetSize()
        mx = min(6, max(1, sz[0] // 8))
        my = min(6, max(1, sz[1] // 8))
        mz = min(4, max(1, sz[2] // 6))
        wx = get_taper_1d(sz[0], mx)
        wy = get_taper_1d(sz[1], my)
        wz = get_taper_1d(sz[2], mz)
        taper_3d = (wz[:, None, None] * wy[None, :, None] * wx[None, None, :]).astype(np.float32)
        
        native_mask = sitk.GetImageFromArray(taper_3d)
        native_mask.CopyInformation(vol.image)

        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(ref)
        resampler.SetInterpolator(sitk.sitkLinear)
        resampler.SetDefaultPixelValue(0.0)
        
        resampled_img = resampler.Execute(vol.image)
        resampled_mask = resampler.Execute(native_mask)
        
        arr = sitk.GetArrayFromImage(resampled_img).astype(np.float32)
        mask_arr = sitk.GetArrayFromImage(resampled_mask).astype(np.float32)
        mask_arr = np.clip(mask_arr, 0.0, 1.0)
        
        # Robust intensity normalization using voxels inside scan coverage
        core_voxels = arr[mask_arr > 0.6]
        if len(core_voxels) > 100:
            p2, p98 = np.percentile(core_voxels, [2, 98])
            if p98 > p2:
                arr = (arr - p2) / (p98 - p2) * 1000.0
                arr = np.clip(arr, 0.0, 2000.0)
        
        # Compute gradient on the resampled volume
        grad_img = sitk.GradientMagnitude(resampled_img)
        grad_arr = sitk.GetArrayFromImage(grad_img).astype(np.float32)
        
        # Suppress artificial boundary gradients at the FOV box edge
        grad_arr = grad_arr * mask_arr

        resampled_arrays.append(arr)
        gradient_arrays.append(grad_arr)
        mask_arrays.append(mask_arr)

    job.progress(85, "Performing gradient-weighted multi-planar fusion")
    # Dynamic weights: mask_arr ensures zero contribution outside FOV,
    # and gradient magnitude prioritizes sharp in-plane details inside the FOV.
    weights = [m * ((g ** 1.2) + 1.0) for m, g in zip(mask_arrays, gradient_arrays)]
    total_weight = sum(weights)
    
    # Avoid divide by zero in areas outside all scans
    safe_weight = np.maximum(total_weight, 1e-6)
    fused_arr = sum(w * a for w, a in zip(weights, resampled_arrays)) / safe_weight
    
    # Zero out background voxels where no scan had coverage
    no_coverage = (total_weight < 1e-3)
    fused_arr[no_coverage] = 0.0
    
    fused_image = sitk.GetImageFromArray(fused_arr)
    fused_image.CopyInformation(ref)

    labels = [s.description.split(':')[-1].strip() for s in series_list]
    desc = f"Fused Isotropic 3D ({len(series_list)} planes: {', '.join(labels)})"
    fused_series = Series(
        key=hashlib.sha256(desc.encode()).hexdigest()[:20],
        uid=series_list[0].uid,
        description=desc,
        rows=int(target_size[1]),
        columns=int(target_size[0]),
        pixel_spacing=(target_sp, target_sp),
        orientation=tuple(np.eye(3)[:2].ravel()),
        slices=[Slice("", (0, 0, 0), "") for _ in range(int(target_size[2]))],
        thickness=target_sp,
        spacing_z=target_sp,
    )
    
    source_geometry = geometry(fused_image)
    job.progress(100, "Multi-planar isotropic fusion complete")
    return Volume(fused_image, fused_arr, fused_series, source_geometry, invert=False, components=list(series_list))


def mask_image(mask: np.ndarray, volume: Volume):
    image = sitk.GetImageFromArray(mask.astype(np.uint8))
    image.CopyInformation(volume.image)
    return image


def segment(volume: Volume, operation: str, mask: np.ndarray, lower=0., upper=1., seed=None,
            job: Job | None = None):
    job = job or Job()
    job.progress(5, f"Processing {operation}")
    if operation in ("grow", "threshold") and lower > upper:
        raise ImagingError("Lower threshold must not exceed upper threshold")
    if operation == "threshold":
        result = (volume.data >= lower) & (volume.data <= upper)
    else:
        source = mask_image(mask, volume)
        if operation == "grow":
            if seed is None:
                raise ImagingError("Click a slice to place a seed first")
            x, y, z = map(int, seed)
            if not lower <= volume.data[z, y, x] <= upper:
                raise ImagingError("The seed intensity is outside the threshold range")
            source = volume.image
            filt = sitk.ConnectedThresholdImageFilter()
            filt.SetLower(float(lower))
            filt.SetUpper(float(upper))
            filt.SetSeedList([(x, y, z)])
        elif operation == "fill":
            filt = sitk.BinaryFillholeImageFilter()
            filt.SetForegroundValue(1)
        elif operation == "island":
            filt = sitk.ConnectedComponentImageFilter()
        else:
            raise ImagingError("Unknown segmentation operation")
        filt.AddCommand(sitk.sitkProgressEvent, lambda: filt.Abort() if job.cancelled.is_set() else None)
        image = filt.Execute(source)
        job.check()
        if operation == "island":
            image = sitk.RelabelComponent(image, sortByObjectSize=True) == 1
        result = sitk.GetArrayFromImage(image) > 0
    job.progress(100, "Selection ready")
    return result
