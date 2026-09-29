from pathlib import Path
import numpy as np
import pydicom
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import MRImageStorage, ExplicitVRLittleEndian, generate_uid
import pytest
import SimpleITK as sitk

from mri3d.imaging import scan_folder, load_series, Job, Cancelled, ImagingError, Volume, Series, canonicalize, fuse_series
from mri3d.models import Layer, make_mesh, export_mesh, paint_mask, slice_array, slice_to_voxel
from mri3d.projects import save_project, open_project, read_manifest, missing_sources


def write_stack(folder, positions=(0, 3, 6, 9), orientation=(1, 0, 0, 0, 1, 0), uid=None):
    folder.mkdir(parents=True, exist_ok=True)
    uid = uid or generate_uid()
    study = generate_uid()
    for index, z in enumerate(positions):
        meta = FileMetaDataset()
        meta.TransferSyntaxUID = ExplicitVRLittleEndian
        meta.MediaStorageSOPClassUID = MRImageStorage
        meta.MediaStorageSOPInstanceUID = generate_uid()
        # Reverse filenames intentionally: geometry must determine ordering.
        path = folder / f"{len(positions) - index:02}.dcm"
        ds = FileDataset(str(path), {}, file_meta=meta, preamble=b"\0" * 128)
        ds.SOPClassUID = MRImageStorage
        ds.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
        ds.StudyInstanceUID = study
        ds.SeriesInstanceUID = uid
        ds.SeriesDescription = "Synthetic geometry fixture"
        ds.Modality = "MR"
        ds.ImageOrientationPatient = list(orientation)
        row, col = np.array(orientation).reshape(2, 3)
        normal = np.cross(row, col)
        ds.ImagePositionPatient = list(np.array([-10., 20., 0.]) + normal * float(z))
        ds.Rows = 12
        ds.Columns = 16
        ds.PixelSpacing = [2., 1.]
        ds.SliceThickness = 2.5
        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = "MONOCHROME2"
        ds.BitsAllocated = 16
        ds.BitsStored = 16
        ds.HighBit = 15
        ds.PixelRepresentation = 0
        ds.RescaleSlope = 2
        ds.RescaleIntercept = -10
        pixels = np.zeros((12, 16), dtype=np.uint16)
        pixels[3:9, 4:12] = index + 50
        ds.PixelData = pixels.tobytes()
        ds.save_as(path, enforce_file_format=True)
    return uid


@pytest.fixture
def stack(tmp_path):
    folder = tmp_path / "dicom"
    write_stack(folder)
    return folder


@pytest.fixture
def volume(stack):
    return load_series(scan_folder(str(stack)).series[0])


def test_geometry_order_scale_and_rescale(volume):
    assert volume.data.shape == (4, 12, 16)
    assert volume.spacing == (1., 2., 3.)
    assert volume.world((4, 3, 2)) == (-6., 26., 6.)
    assert list(volume.data[:, 4, 5]) == [90, 92, 94, 96]
    assert np.allclose(volume.image.GetDirection(), np.eye(3).ravel())


@pytest.mark.parametrize("positions,issue", [((0, 3, 3, 6), "Duplicate"), ((0, 3, 9, 12), "Irregular")])
def test_invalid_stacks(tmp_path, positions, issue):
    write_stack(tmp_path, positions)
    series = scan_folder(str(tmp_path)).series[0]
    assert not series.usable
    assert any(issue in s for s in series.issues)
    with pytest.raises(ImagingError):
        load_series(series)


def test_mixed_orientations_split(stack):
    path = next(stack.glob("*.dcm"))
    ds = pydicom.dcmread(path)
    ds.ImageOrientationPatient = [0, 1, 0, 0, 0, 1]
    ds.save_as(path, enforce_file_format=True)
    scan = scan_folder(str(stack))
    assert len(scan.series) == 2
    assert sum(len(s.slices) for s in scan.series) == 4


def test_malformed_and_non_mri(stack):
    (stack / "bad.dcm").write_bytes(b"not a dicom file")
    scan = scan_folder(str(stack))
    assert len(scan.series) == 1
    assert scan.ignored == 1 and scan.warnings
    for path in stack.glob("0*.dcm"):
        ds = pydicom.dcmread(path)
        ds.Modality = "CT"
        ds.save_as(path, enforce_file_format=True)
    assert scan_folder(str(stack)).series == []


def test_cancellation(stack, volume):
    job = Job()
    job.cancelled.set()
    with pytest.raises(Cancelled):
        scan_folder(str(stack), job)
    with pytest.raises(Cancelled):
        load_series(volume.series, job)


def test_reorientation_preserves_physical_values():
    data = np.arange(4 * 5 * 6, dtype=np.float32).reshape(4, 5, 6)
    image = sitk.GetImageFromArray(data)
    image.SetSpacing((1, 2, 3))
    image.SetOrigin((100, -50, 20))
    image.SetDirection((-1., 0., 0., 0., -1., 0., 0., 0., 1.))
    output = canonicalize(image, Job())
    for xyz in [(0, 0, 0), (5, 4, 3), (1, 3, 2)]:
        world = image.TransformIndexToPhysicalPoint(xyz)
        mapped = output.TransformPhysicalPointToIndex(world)
        assert output[mapped] == image[xyz]
    assert np.allclose(output.GetDirection(), np.eye(3).ravel())


def test_oblique_resampling_physical_linear_field():
    image = sitk.Image([20, 20, 20], sitk.sitkFloat32)
    angle = np.deg2rad(15)
    image.SetDirection((np.cos(angle), -np.sin(angle), 0, np.sin(angle), np.cos(angle), 0, 0, 0, 1))
    image.SetOrigin((-5, 8, 20))
    image.SetSpacing((1, 2, 3))
    for z in range(20):
        for y in range(20):
            for x in range(20):
                world = image.TransformIndexToPhysicalPoint((x, y, z))
                image[x, y, z] = world[0] + 2 * world[1] + 3 * world[2]
    result = canonicalize(image, Job())
    center = image.TransformIndexToPhysicalPoint((10, 10, 10))
    ix = result.TransformPhysicalPointToIndex(center)
    point = result.TransformIndexToPhysicalPoint(ix)
    assert result[ix] == pytest.approx(point[0] + 2 * point[1] + 3 * point[2], abs=0.001)


@pytest.mark.parametrize("axis", [0, 1, 2])
def test_slice_mapping_and_brush(volume, axis):
    xyz = (7, 5, 2)
    mask = np.zeros(volume.data.shape, bool)
    paint_mask(mask, axis, xyz, 0.4, volume.spacing, True)
    assert mask.sum() == 1 and mask[2, 5, 7]
    plane = slice_array(mask, axis, xyz)
    v, u = np.argwhere(plane)[0]
    assert slice_to_voxel(axis, int(u), int(v), xyz, mask.shape) == xyz


def test_segmentation_and_history(volume):
    from mri3d.imaging import segment
    empty = np.zeros(volume.data.shape, bool)
    result = segment(volume, "grow", empty, 89, 97, (5, 4, 1))
    assert result.sum() == 4 * 6 * 8
    with pytest.raises(ImagingError, match="outside"):
        segment(volume, "grow", empty, 200, 300, (5, 4, 1))
    layer = Layer("Tissue", (1, 2, 3), empty.copy())
    layer.remember()
    layer.mask = result
    assert layer.restore() and not layer.mask.any()
    assert layer.restore(True) and np.array_equal(layer.mask, result)
    hole = np.ones(volume.data.shape, bool)
    hole[1, 5, 6] = False
    assert segment(volume, "fill", hole).all()
    islands = result.copy()
    islands[0, 0, 0] = True
    assert np.array_equal(segment(volume, "island", islands), result)


def test_mesh_world_bounds_and_exports(volume, tmp_path):
    mask = np.zeros(volume.data.shape, bool)
    mask[1:3, 3:8, 4:10] = True
    mesh = make_mesh(volume, mask)
    assert mesh.GetBounds() == pytest.approx((-6.5, -0.5, 25., 35., 1.5, 7.5))
    original = mask.copy()
    assert make_mesh(volume, mask, smooth=True).GetNumberOfCells() > 0
    assert np.array_equal(mask, original)
    import vtk
    for suffix, reader_type in [("stl", vtk.vtkSTLReader), ("ply", vtk.vtkPLYReader)]:
        path = tmp_path / ("mesh." + suffix)
        export_mesh(mesh, str(path), (50, 160, 220))
        reader = reader_type()
        reader.SetFileName(str(path))
        reader.Update()
        assert reader.GetOutput().GetBounds() == pytest.approx(mesh.GetBounds(), abs=1e-5)
        if suffix == "ply":
            assert reader.GetOutput().GetPointData().GetScalars().GetTuple3(0) == (50, 160, 220)
        assert path.with_suffix(path.suffix + ".json").is_file()
    with pytest.raises(ImagingError, match="empty"):
        make_mesh(volume, np.zeros_like(mask))


def test_project_roundtrip_and_relink(volume, tmp_path, stack):
    layer = Layer("Bone", (244, 203, 141), volume.data > 0)
    path = tmp_path / "example.mri3d"
    save_project(path, volume, [layer], {"xyz": [5, 4, 2]})
    restored, layers, state = open_project(path)
    assert np.array_equal(restored.data, volume.data)
    assert np.array_equal(layers[0].mask, layer.mask)
    assert layers[0].id == layer.id and state["xyz"] == [5, 4, 2]
    moved = tmp_path / "moved"
    stack.rename(moved)
    assert missing_sources(read_manifest(path))
    with pytest.raises(ImagingError, match="moved"):
        open_project(path)
    restored, layers, _ = open_project(path, relink=str(moved))
    assert np.array_equal(restored.data, volume.data)
    unrelated = tmp_path / "unrelated"
    write_stack(unrelated)
    with pytest.raises(ImagingError, match="same series"):
        open_project(path, relink=str(unrelated))


def test_cancelled_save_preserves_previous_manifest(volume, tmp_path):
    path = tmp_path / "test.mri3d"
    layer = Layer("Bone", (1, 2, 3), volume.data > 0)
    save_project(path, volume, [layer], {})
    previous = path.read_bytes()
    job = Job()
    job.cancelled.set()
    with pytest.raises(Cancelled):
        save_project(path, volume, [layer], {}, job)
    assert path.read_bytes() == previous


def test_mask_geometry_mismatch_rejected(volume, tmp_path):
    path = tmp_path / "test.mri3d"
    save_project(path, volume, [Layer("Tissue", (1, 2, 3), volume.data > 0)], {})
    manifest = read_manifest(path)
    mask_path = tmp_path / manifest["layers"][0]["mask"]
    image = sitk.ReadImage(str(mask_path))
    image.SetOrigin((999, 999, 999))
    sitk.WriteImage(image, str(mask_path))
    with pytest.raises(ImagingError, match="geometry"):
        open_project(path)


def test_fuse_series_orthogonal(tmp_path):
    folder_axi = tmp_path / "axi"
    write_stack(folder_axi, positions=(0, 2, 4, 6), orientation=(1, 0, 0, 0, 1, 0))
    folder_cor = tmp_path / "cor"
    write_stack(folder_cor, positions=(0, 2, 4, 6), orientation=(1, 0, 0, 0, 0, -1))

    scan_axi = scan_folder(str(folder_axi))
    scan_cor = scan_folder(str(folder_cor))

    assert len(scan_axi.series) == 1
    assert len(scan_cor.series) == 1

    fused = fuse_series([scan_axi.series[0], scan_cor.series[0]])
    assert np.isclose(fused.image.GetSpacing()[0], fused.image.GetSpacing()[1])
    assert np.isclose(fused.image.GetSpacing()[1], fused.image.GetSpacing()[2])
    assert "Fused Isotropic 3D" in fused.series.description
    assert fused.data.ndim == 3



def test_fused_project_roundtrip_and_relink(tmp_path):
    scans = tmp_path / "scans"
    write_stack(scans / "axi", positions=(0, 2, 4, 6), orientation=(1, 0, 0, 0, 1, 0))
    write_stack(scans / "cor", positions=(0, 2, 4, 6), orientation=(1, 0, 0, 0, 0, -1))
    fused = fuse_series(scan_folder(str(scans)).series)
    layer = Layer("Tissue", (1, 2, 3), fused.data > fused.data.mean())
    path = tmp_path / "fused.mri3d"
    save_project(path, fused, [layer], {"xyz": [1, 1, 1]})
    assert not missing_sources(read_manifest(path))
    restored, layers, _ = open_project(path)
    assert restored.components is not None and len(restored.components) == 2
    assert np.allclose(restored.data, fused.data)
    assert np.array_equal(layers[0].mask, layer.mask)
    moved = tmp_path / "moved"
    scans.rename(moved)
    assert missing_sources(read_manifest(path))
    restored, layers, _ = open_project(path, relink=str(moved))
    assert np.allclose(restored.data, fused.data)
