from __future__ import annotations

from dataclasses import dataclass, field
import uuid
import numpy as np
import SimpleITK as sitk
import vtk
from vtk.util.numpy_support import numpy_to_vtk
from .imaging import Volume, Job, ImagingError


@dataclass
class Layer:
    name: str
    color: tuple
    mask: np.ndarray
    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    visible: bool = True
    opacity: float = 0.85
    smooth: bool = False
    representation: str = "surface"
    mesh: object = None
    undo: list = field(default_factory=list)
    redo: list = field(default_factory=list)

    def remember(self):
        self.undo.append(np.packbits(self.mask, axis=None))
        self.undo = self.undo[-20:]
        self.redo.clear()
        self.mesh = None

    def restore(self, redo=False):
        source, target = (self.redo, self.undo) if redo else (self.undo, self.redo)
        if not source:
            return False
        target.append(np.packbits(self.mask, axis=None))
        self.mask = np.unpackbits(source.pop(), count=self.mask.size).reshape(self.mask.shape).astype(bool)
        self.mesh = None
        return True


def vtk_image(data, spacing, origin):
    result = vtk.vtkImageData()
    result.SetDimensions(*data.shape[::-1])
    result.SetSpacing(*spacing)
    result.SetOrigin(*origin)
    result.GetPointData().SetScalars(numpy_to_vtk(np.ascontiguousarray(data).ravel(), deep=True))
    return result


def make_mesh(volume: Volume, mask: np.ndarray, smooth=False, job: Job | None = None):
    job = job or Job()
    job.progress(5, "Building surface in patient coordinates")
    if not mask.any():
        raise ImagingError("This layer is empty. Mark tissue before generating a surface.")
    # Padding closes surfaces that reach a volume boundary, without moving voxel centers.
    padded = np.pad(mask.astype(np.uint8), 1)
    origin = np.array(volume.origin) - np.array(volume.spacing)
    data = vtk_image(padded, volume.spacing, origin)
    contour = vtk.vtkFlyingEdges3D()
    contour.SetInputData(data)
    contour.SetValue(0, 0.5)
    contour.AddObserver("ProgressEvent", lambda o, e: o.SetAbortExecute(job.cancelled.is_set()))
    contour.Update()
    job.progress(65, "Finishing surface")
    output = contour.GetOutput()
    if smooth:
        smoothing = vtk.vtkWindowedSincPolyDataFilter()
        smoothing.SetInputData(output)
        smoothing.SetNumberOfIterations(15)
        smoothing.SetPassBand(0.1)
        smoothing.BoundarySmoothingOff()
        smoothing.FeatureEdgeSmoothingOff()
        smoothing.NormalizeCoordinatesOn()
        smoothing.AddObserver("ProgressEvent", lambda o, e: o.SetAbortExecute(job.cancelled.is_set()))
        smoothing.Update()
        output = smoothing.GetOutput()
    normals = vtk.vtkPolyDataNormals()
    normals.SetInputData(output)
    normals.SplittingOff()
    normals.ConsistencyOn()
    normals.Update()
    mesh = vtk.vtkPolyData()
    mesh.DeepCopy(normals.GetOutput())
    job.progress(100, "Surface ready")
    return mesh


def export_mesh(mesh, path, color):
    from pathlib import Path
    import json
    target = Path(path)
    if mesh is None or mesh.GetNumberOfPoints() == 0:
        raise ImagingError("Generate a non-empty surface first")
    if target.suffix.lower() == ".stl":
        writer = vtk.vtkSTLWriter()
        writer.SetFileTypeToBinary()
        writer.SetHeader("MRI3D | units: mm | coordinates: DICOM LPS | illustrative segmentation")
    elif target.suffix.lower() == ".ply":
        writer = vtk.vtkPLYWriter()
        writer.SetFileTypeToBinary()
        writer.SetColorModeToUniformColor()
        writer.SetColor(*[int(c) for c in color])
        writer.AddComment("MRI3D units: millimeters; coordinate system: DICOM LPS")
    else:
        raise ImagingError("Choose STL or PLY export")
    writer.SetInputData(mesh)
    writer.SetFileName(str(target))
    if writer.Write() != 1 or writer.GetErrorCode() != 0:
        raise ImagingError("Could not write the surface; check the destination and disk space")
    target.with_suffix(target.suffix + ".json").write_text(json.dumps({
        "units": "mm", "coordinates": "DICOM LPS", "purpose": "Personal exploration",
        "bounds_mm": list(mesh.GetBounds()), "color_rgb": list(color),
        "note": "User-reviewed segmentation; smoothing does not recover missing anatomy."
    }, indent=2), encoding="utf-8")


def slice_array(array, axis, xyz):
    x, y, z = map(int, xyz)
    if axis == 2:
        return array[z, :, :]
    if axis == 1:
        return array[:, y, :][::-1, :]
    return array[:, :, x][::-1, :]


def slice_to_voxel(axis, u, v, xyz, shape):
    x, y, z = map(int, xyz)
    if axis == 2:
        x, y = u, v
    elif axis == 1:
        x, z = u, shape[0] - 1 - v
    else:
        y, z = u, shape[0] - 1 - v
    return tuple(int(np.clip(a, 0, n - 1)) for a, n in zip((x, y, z), shape[::-1]))


def paint_mask(mask, axis, xyz, radius_mm, spacing, value):
    # A disk in the displayed physical plane; slice thickness never stretches the brush.
    plane = slice_array(mask, axis, xyz)
    x, y, z = xyz
    if axis == 2:
        u, v, su, sv = x, y, spacing[0], spacing[1]
    elif axis == 1:
        u, v, su, sv = x, mask.shape[0] - 1 - z, spacing[0], spacing[2]
    else:
        u, v, su, sv = y, mask.shape[0] - 1 - z, spacing[1], spacing[2]
    ur = int(np.ceil(radius_mm / su))
    vr = int(np.ceil(radius_mm / sv))
    x0, x1 = max(0, u - ur), min(plane.shape[1], u + ur + 1)
    y0, y1 = max(0, v - vr), min(plane.shape[0], v + vr + 1)
    yy, xx = np.ogrid[y0:y1, x0:x1]
    circle = ((xx - u) * su) ** 2 + ((yy - v) * sv) ** 2 <= radius_mm ** 2
    plane[y0:y1, x0:x1][circle] = value


def filter_clusters(mask: np.ndarray, min_voxels: int = 25) -> np.ndarray:
    """Remove disconnected voxel clusters smaller than min_voxels (cluster-extent thresholding)."""
    if not mask.any() or min_voxels <= 1:
        return mask.copy()
    img = sitk.GetImageFromArray(mask.astype(np.uint8))
    cc = sitk.ConnectedComponent(img)
    relabeled = sitk.RelabelComponent(cc, minimumObjectSize=int(min_voxels))
    return sitk.GetArrayFromImage(relabeled) > 0

