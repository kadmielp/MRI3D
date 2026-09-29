import numpy as np
import pydicom
import pytest
from .test_imaging import write_stack
from mri3d.imaging import scan_folder, load_series, ImagingError, Job, Cancelled, segment


def test_localizer_and_multiframe(tmp_path):
    write_stack(tmp_path)
    for path in tmp_path.glob("*.dcm"):
        ds = pydicom.dcmread(path)
        ds.ImageType = ["ORIGINAL", "PRIMARY", "LOCALIZER"]
        ds.save_as(path, enforce_file_format=True)
    result = scan_folder(str(tmp_path))
    assert not result.series[0].usable
    assert result.series[0].localizer
    for path in tmp_path.glob("*.dcm"):
        ds = pydicom.dcmread(path)
        ds.NumberOfFrames = 2
        ds.save_as(path, enforce_file_format=True)
    result = scan_folder(str(tmp_path))
    assert not result.series and result.ignored == 4
    assert any("multiframe" in w for w in result.warnings)


def test_legacy_implicit_vr_without_preamble(tmp_path):
    write_stack(tmp_path)
    for path in tmp_path.glob("*.dcm"):
        ds = pydicom.dcmread(path)
        ds.preamble = None
        ds.file_meta = pydicom.dataset.FileMetaDataset()
        pydicom.dcmwrite(path, ds, implicit_vr=True, little_endian=True, enforce_file_format=False)
    result = scan_folder(str(tmp_path))
    assert len(result.series) == 1
    assert load_series(result.series[0]).data[1, 4, 5] == 92


def test_source_geometry_changed_since_scan(tmp_path):
    write_stack(tmp_path)
    series = scan_folder(str(tmp_path)).series[0]
    path = series.slices[0].path
    ds = pydicom.dcmread(path)
    ds.ImagePositionPatient = [0, 0, 0]
    ds.save_as(path, enforce_file_format=True)
    with pytest.raises(ImagingError, match="geometry changed"):
        load_series(series)


def test_in_plane_shift_rejected(tmp_path):
    write_stack(tmp_path)
    path = next(tmp_path.glob("*.dcm"))
    ds = pydicom.dcmread(path)
    ds.ImagePositionPatient[0] += 2
    ds.save_as(path, enforce_file_format=True)
    series = scan_folder(str(tmp_path)).series[0]
    assert not series.usable
    assert any("shifts" in issue for issue in series.issues)


def test_threshold_bounds_and_cancelled_mesh(tmp_path):
    write_stack(tmp_path)
    volume = load_series(scan_folder(str(tmp_path)).series[0])
    with pytest.raises(ImagingError, match="Lower"):
        segment(volume, "threshold", volume.data > 0, 100, 0)
    from mri3d.models import make_mesh
    job = Job()
    job.cancelled.set()
    with pytest.raises(Cancelled):
        make_mesh(volume, volume.data > 0, job=job)
