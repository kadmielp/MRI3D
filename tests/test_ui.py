import numpy as np
import pytest
from PySide6.QtCore import Qt, QPoint
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication
from mri3d.views import SliceCanvas
from mri3d.smoke import synthetic_volume


@pytest.mark.parametrize("axis", [0, 1, 2])
def test_mouse_location_matches_displayed_voxel(axis):
    app = QApplication.instance() or QApplication([])
    volume = synthetic_volume()
    view = SliceCanvas(axis)
    view.resize(450, 400)
    view.refresh(volume, [32, 32, 20], [], 500, 1000)
    view.show()
    app.processEvents()
    picked = []
    view.picked.connect(lambda axis, xyz, first: picked.append((axis, xyz, first)))
    plane_width = view.pix.width()
    plane_height = view.pix.height()
    u, v = plane_width // 3, plane_height // 3
    px = int(view.target.x() + (u + 0.5) / plane_width * view.target.width())
    py = int(view.target.y() + (v + 0.5) / plane_height * view.target.height())
    QTest.mouseClick(view, Qt.MouseButton.LeftButton, pos=QPoint(px, py))
    assert len(picked) == 1 and picked[0][0] == axis and picked[0][2]
    xyz = picked[0][1]
    assert xyz[axis] == [32, 32, 20][axis]
    from mri3d.models import slice_to_voxel
    assert xyz == slice_to_voxel(axis, u, v, [32, 32, 20], volume.data.shape)
    view.close()


def test_dynamic_clipping_with_slice_navigation():
    from mri3d.app import MainWindow
    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    volume = synthetic_volume()
    window.set_volume(volume)

    # Enable clipping on Axial (axis 2: Inferior / superior)
    window.clip.setChecked(True)
    window.clip_axis.setCurrentIndex(2)
    assert window.scene.clip_enabled

    # Check clipping plane origin along Z matches current slice
    curr_z = window.xyz[2]
    expected_world = volume.world(window.xyz)
    assert np.isclose(window.scene.clip_plane.GetOrigin()[2], expected_world[2])

    # Scroll / step Axial slice
    window.step(2, 5)
    assert window.xyz[2] == curr_z + 5
    expected_world = volume.world(window.xyz)
    assert np.isclose(window.scene.clip_plane.GetOrigin()[2], expected_world[2])

    # Move slider on equivalent slice panel (Axial panel with axis 2)
    axial_panel = next(p for p in window.panels if p.axis == 2)
    axial_panel.slider.setValue(10)
    assert window.xyz[2] == 10
    expected_world = volume.world(window.xyz)
    assert np.isclose(window.scene.clip_plane.GetOrigin()[2], expected_world[2])

    # Switch Cut axis to Coronal (axis 1: Anterior / posterior)
    window.clip_axis.setCurrentIndex(1)
    expected_world = volume.world(window.xyz)
    assert np.isclose(window.scene.clip_plane.GetOrigin()[1], expected_world[1])

    # Move coronal slice
    coronal_panel = next(p for p in window.panels if p.axis == 1)
    coronal_panel.slider.setValue(15)
    assert window.xyz[1] == 15
    expected_world = volume.world(window.xyz)
    assert np.isclose(window.scene.clip_plane.GetOrigin()[1], expected_world[1])

    # Reverse cut
    normal_before = window.scene.clip_plane.GetNormal()
    window.clip_flip.setChecked(True)
    normal_after = window.scene.clip_plane.GetNormal()
    assert normal_after[1] == -normal_before[1]

    # Disable clipping
    window.clip.setChecked(False)
    assert not window.scene.clip_enabled

    window.close()


def test_filter_clusters():
    from mri3d.models import filter_clusters
    mask = np.zeros((20, 20, 20), dtype=bool)
    # Small noise island (3 voxels)
    mask[2:5, 2, 2] = True
    # Larger anatomical island (4x4x4 = 64 voxels)
    mask[10:14, 10:14, 10:14] = True

    filtered = filter_clusters(mask, min_voxels=25)
    assert not filtered[2:5, 2, 2].any()
    assert filtered[10:14, 10:14, 10:14].all()


def test_3d_visual_enhancements(tmp_path):
    from mri3d.app import MainWindow
    from mri3d.models import Layer, make_mesh
    from mri3d.projects import save_project, read_manifest
    from mri3d.smoke import synthetic_volume

    app = QApplication.instance() or QApplication([])
    volume = synthetic_volume()
    window = MainWindow()
    window.set_volume(volume)

    # Turntable Orbit
    assert not window.scene.turntable_timer.isActive()
    window.scene.toggle_turntable(True)
    assert window.scene.turntable_timer.isActive()
    assert window.scene.btn_orbit.isChecked()
    window.scene._turntable_step()
    window.scene.toggle_turntable(False)
    assert not window.scene.turntable_timer.isActive()

    # Cube Axes (Physical Scale)
    assert not window.scene.cube_axes.GetVisibility()
    window.scene.toggle_axes(True)
    assert window.scene.cube_axes.GetVisibility()
    assert window.scene.btn_scale.isChecked()
    window.scene.toggle_axes(False)
    assert not window.scene.cube_axes.GetVisibility()

    # Stereoscopic Anaglyph 3D
    rw = window.scene.widget.GetRenderWindow()
    assert not rw.GetStereoRender()
    window.scene.toggle_anaglyph(True)
    assert rw.GetStereoRender()
    assert window.scene.btn_anaglyph.isChecked()
    window.scene.toggle_anaglyph(False)
    assert not rw.GetStereoRender()

    # Add layer and generate surface
    mask = np.zeros(volume.data.shape, dtype=bool)
    mask[5:12, 5:12, 5:12] = True
    layer = Layer("Bone", (244, 203, 141), mask)
    layer.mesh = make_mesh(volume, layer.mask)
    window.layers.append(layer)
    window.refresh_layer_list(0)
    window.scene.sync_layers(window.layers)

    actor = window.scene.surface_actors[layer.id]
    assert actor.GetProperty().GetRepresentation() == 2  # VTK_SURFACE = 2

    # Switch to wireframe
    window.layer_rep.setCurrentIndex(1)
    assert layer.representation == "wireframe"
    assert actor.GetProperty().GetRepresentation() == 1  # VTK_WIREFRAME = 1

    # Switch to points
    window.layer_rep.setCurrentIndex(2)
    assert layer.representation == "points"
    assert actor.GetProperty().GetRepresentation() == 0  # VTK_POINTS = 0

    # Test project persistence of representation
    proj_file = tmp_path / "test_rep.mri3d"
    save_project(proj_file, volume, window.layers, window.view_state())
    manifest = read_manifest(proj_file)
    assert len(manifest["layers"]) == 1
    assert manifest["layers"][0]["representation"] == "points"

    window.dirty = False
    window.close()


def test_ui_multi_series_selection_and_fusion(tmp_path):
    from mri3d.app import MainWindow
    from mri3d.imaging import scan_folder
    from tests.test_imaging import write_stack

    app = QApplication.instance() or QApplication([])
    combined_folder = tmp_path / "combined"
    write_stack(combined_folder / "axi", positions=(0, 2, 4, 6), orientation=(1, 0, 0, 0, 1, 0))
    write_stack(combined_folder / "cor", positions=(0, 2, 4, 6), orientation=(1, 0, 0, 0, 0, -1))

    window = MainWindow()
    scan = scan_folder(str(combined_folder))
    window.show_scan(scan)
    assert window.tree.topLevelItemCount() == 2

    # Select both
    window.tree.topLevelItem(0).setSelected(True)
    window.tree.topLevelItem(1).setSelected(True)
    window.series_selected()

    assert "Fuse 2 orthogonal planes" in window.load_button.text()
    assert window.load_button.isEnabled()

    window.close()

