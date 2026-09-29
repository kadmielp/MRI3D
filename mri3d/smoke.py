"""Self-contained app smoke workflow; usable from the packaged executable."""
from pathlib import Path
import json
import sys
import time
import traceback
import numpy as np
import SimpleITK as sitk
import vtk
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication
from .imaging import scan_folder, load_series, Volume, Series, geometry
from .models import Layer, make_mesh, export_mesh, paint_mask
from .projects import save_project, open_project


def synthetic_volume():
    zz, yy, xx = np.mgrid[:40, :64, :64]
    distance = ((xx - 32) / 18) ** 2 + ((yy - 32) / 21) ** 2 + ((zz - 20) / 13) ** 2
    data = (np.exp(-distance * 2) * 1000).astype(np.float32)
    image = sitk.GetImageFromArray(data)
    image.SetSpacing((0.8, 0.8, 1.5))
    image.SetOrigin((-25, -25, -30))
    series = Series("synthetic", "synthetic", "Synthetic smoke volume", 64, 64, (0.8, 0.8), (1, 0, 0, 0, 1, 0), thickness=1.5, spacing_z=1.5)
    return Volume(image, data, series, geometry(image))


def run_smoke(window, output, folder=None):
    root = Path(output).resolve()
    root.mkdir(parents=True, exist_ok=True)
    report = {"packaged": bool(getattr(sys, "frozen", False)), "executable": sys.executable}
    started = time.perf_counter()
    timer = QTimer(window)
    timer.setInterval(100)
    window.smoke_timer = timer
    state = {"phase": 0}

    def fail(exc):
        timer.stop()
        report["error"] = str(exc)
        report["traceback"] = traceback.format_exc()
        (root / "smoke.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        window.dirty = False
        window.close()
        QApplication.exit(1)

    def loaded(volume):
        window.set_volume(volume)
        report["shape_zyx"] = list(volume.data.shape)
        report["spacing_xyz"] = list(volume.spacing)
        state["phase"] = 1

    def initial(job):
        if folder:
            scan = scan_folder(folder, job)
            report["series_groups"] = len(scan.series)
            report["usable_groups"] = sum(s.usable for s in scan.series)
            report["files"] = sum(len(s.slices) for s in scan.series)
            series = next((s for s in scan.series if s.usable and s.description == "Cor T1 FSE"), next(s for s in scan.series if s.usable))
            return load_series(series, job)
        return synthetic_volume()

    def tick():
        try:
            if time.perf_counter() - started > 180:
                raise RuntimeError("GUI smoke test timed out")
            if window.thread is not None:
                return
            volume = window.volume
            if state["phase"] == 1:
                # This is a technical selection fixture, not a claimed anatomical segmentation.
                mask = np.zeros(volume.data.shape, bool)
                xyz = tuple(n // 2 for n in volume.image.GetSize())
                layer = Layer("QA selection · not anatomical", (88, 216, 188), mask)
                window.layers.append(layer)
                window.refresh_layer_list(0)
                window.radius.setValue(8)
                window.tool.setCurrentIndex(1)
                window.pick(2, xyz, True)
                window.end_stroke()
                assert layer.mask.any()
                marked = layer.mask.copy()
                window.undo()
                assert not layer.mask.any()
                window.undo(True)
                assert np.array_equal(layer.mask, marked)
                # Add thickness for a useful closed test mesh.
                for dz in range(-2, 3):
                    paint_mask(layer.mask, 2, (xyz[0], xyz[1], xyz[2] + dz), 8, volume.spacing, True)
                report["brush_undo_redo"] = True
                window.generate()
                state["phase"] = 2
            elif state["phase"] == 2:
                layer = window.layers[0]
                assert layer.mesh is not None
                export_mesh(layer.mesh, str(root / "qa-surface.ply"), layer.color)
                export_mesh(layer.mesh, str(root / "qa-surface.stl"), layer.color)
                report["mesh_points"] = layer.mesh.GetNumberOfPoints()
                report["mesh_bounds_mm"] = list(layer.mesh.GetBounds())
                window.clip.setChecked(True)
                window.clip_axis.setCurrentIndex(1)
                window.clip_position.setValue(45)
                window.scene.reset_camera()
                camera = window.scene.renderer.GetActiveCamera()
                camera.Azimuth(25)
                camera.Elevation(10)
                window.scene.renderer.ResetCameraClippingRange()
                window.scene.render()
                for axis in range(3):
                    window.set_index(axis, volume.image.GetSize()[axis] // 2)
                report["linked_views_and_clipping"] = True
                if folder:
                    path = root / "qa-project.mri3d"
                    state["path"] = path
                    views = window.view_state()
                    window.start_job(lambda job: save_project(path, volume, window.layers, views, job), lambda result: None)
                state["phase"] = 3
            elif state["phase"] == 3:
                if folder:
                    def reopened(result):
                        restored, layers, views = result
                        assert np.array_equal(layers[0].mask, window.layers[0].mask)
                        assert np.array_equal(restored.data, window.volume.data)
                        report["project_roundtrip"] = True
                    window.start_job(lambda job: open_project(state["path"], job=job), reopened)
                state["phase"] = 4
            elif state["phase"] == 4:
                state["phase"] = 5
                # Wait a paint cycle before collecting the rendered app.
            elif state["phase"] == 5:
                window.sidebar.setMinimumWidth(300)
                window.grab().save(str(root / "app.png"))
                capture = vtk.vtkWindowToImageFilter()
                capture.SetInput(window.scene.widget.GetRenderWindow())
                capture.ReadFrontBufferOff()
                capture.Update()
                writer = vtk.vtkPNGWriter()
                writer.SetFileName(str(root / "scene.png"))
                writer.SetInputConnection(capture.GetOutputPort())
                writer.Write()
                durations = []
                for _ in range(10):
                    begin = time.perf_counter()
                    window.scene.renderer.GetActiveCamera().Azimuth(2)
                    window.scene.render()
                    durations.append(time.perf_counter() - begin)
                report["render_median_ms"] = float(np.median(durations) * 1000)
                report["elapsed_seconds"] = time.perf_counter() - started
                report["success"] = True
                (root / "smoke.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
                timer.stop()
                window.dirty = False
                window.close()
                QApplication.exit(0)
        except Exception as exc:
            fail(exc)

    # Avoid modal error dialogs in unattended QA; report failures instead.
    window.error = lambda message: fail(RuntimeError(message))
    timer.timeout.connect(tick)
    window.start_job(initial, loaded)
    timer.start()
