from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt, Signal, QRectF, QPointF, QTimer
from PySide6.QtGui import QColor, QImage, QPainter, QPen
from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QSlider, QPushButton
import vtk
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
from .models import slice_array, slice_to_voxel, vtk_image


ACCENTS = {2: "#5ddbc6", 1: "#7fa9ff", 0: "#f0bb74"}


class SliceCanvas(QWidget):
    picked = Signal(int, object, bool)
    strokeEnded = Signal()
    stepped = Signal(int, int)

    def __init__(self, axis, parent=None):
        super().__init__(parent)
        self.axis = axis
        self.volume = None
        self.xyz = [0, 0, 0]
        self.layers = []
        self.level = 0
        self.window = 1
        self.invert = False
        self.zoom = 1.0
        self.pan = QPointF()
        self.last_pan = None
        self.dragging = False
        self.target = QRectF()
        self.pix = None
        self.hover_val = None
        self.setMinimumSize(200, 150)
        self.setMouseTracking(True)

    def refresh(self, volume, xyz, layers, level, window, invert=False):
        self.volume, self.xyz, self.layers = volume, list(xyz), layers
        self.level, self.window, self.invert = level, max(0.001, window), invert
        if volume is not None:
            plane = slice_array(volume.data, self.axis, xyz)
            grey = np.clip((plane - (level - window / 2)) / self.window * 255, 0, 255).astype(np.uint8)
            if invert:
                grey = 255 - grey
            rgb = np.repeat(grey[:, :, None], 3, axis=2)
            for layer in layers:
                if layer.visible:
                    mask = slice_array(layer.mask, self.axis, xyz)
                    alpha = layer.opacity * 0.55
                    rgb[mask] = (rgb[mask] * (1 - alpha) + np.array(layer.color) * alpha).astype(np.uint8)
            rgb = np.ascontiguousarray(rgb)
            self.pix = QImage(rgb.data, rgb.shape[1], rgb.shape[0], rgb.strides[0], QImage.Format.Format_RGB888).copy()
        self.update()

    def reset_view(self):
        self.zoom = 1.0
        self.pan = QPointF()
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.fillRect(self.rect(), QColor("#080f19"))
        if self.volume is None or self.pix is None:
            p.setPen(QColor("#73859a"))
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "Import a DICOM series to begin")
            return
        sx, sy, sz = self.volume.spacing
        su, sv = (sx, sy) if self.axis == 2 else ((sx, sz) if self.axis == 1 else (sy, sz))
        pw, ph = self.pix.width() * su, self.pix.height() * sv
        scale = min((self.width() - 44) / pw, (self.height() - 44) / ph) * self.zoom
        w, h = pw * scale, ph * scale
        self.target = QRectF((self.width() - w) / 2 + self.pan.x(), (self.height() - h) / 2 + self.pan.y(), w, h)
        p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
        p.drawImage(self.target, self.pix)
        x, y, z = self.xyz
        u, v = (x, y) if self.axis == 2 else ((x, self.volume.data.shape[0] - 1 - z) if self.axis == 1 else (y, self.volume.data.shape[0] - 1 - z))
        cx = self.target.left() + (u + 0.5) / self.pix.width() * w
        cy = self.target.top() + (v + 0.5) / self.pix.height() * h
        p.save()
        p.setClipRect(self.target)
        p.setPen(QPen(QColor(ACCENTS[self.axis]), 1, Qt.PenStyle.DashLine))
        p.drawLine(QPointF(cx, self.target.top()), QPointF(cx, self.target.bottom()))
        p.drawLine(QPointF(self.target.left(), cy), QPointF(self.target.right(), cy))
        p.restore()

        # Anatomical direction labels with subtle pill badges
        labels = ("R", "L", "A", "P") if self.axis == 2 else (("R", "L", "S", "I") if self.axis == 1 else ("A", "P", "S", "I"))
        rects = [
            QRectF(6, self.height() / 2 - 12, 22, 24),
            QRectF(self.width() - 28, self.height() / 2 - 12, 22, 24),
            QRectF(self.width() / 2 - 12, 4, 24, 22),
            QRectF(self.width() / 2 - 12, self.height() - 26, 24, 22),
        ]
        font = p.font()
        font.setBold(True)
        font.setPointSize(9)
        p.setFont(font)
        for text, r in zip(labels, rects):
            p.save()
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QColor(12, 20, 32, 180))
            p.drawRoundedRect(r, 4, 4)
            p.setPen(QColor(ACCENTS[self.axis]))
            p.drawText(r, Qt.AlignmentFlag.AlignCenter, text)
            p.restore()

        # Hover voxel intensity HUD overlay
        if self.hover_val is not None:
            p.save()
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QColor(12, 20, 32, 200))
            badge_r = QRectF(8, 8, 80, 20)
            p.drawRoundedRect(badge_r, 4, 4)
            p.setPen(QColor("#94a3b8"))
            font_small = p.font()
            font_small.setPointSize(8)
            p.setFont(font_small)
            p.drawText(badge_r, Qt.AlignmentFlag.AlignCenter, f"Val: {self.hover_val:.0f}")
            p.restore()

        # Zoom level HUD indicator
        if abs(self.zoom - 1.0) > 0.05:
            p.save()
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QColor(12, 20, 32, 200))
            zoom_r = QRectF(8, self.height() - 28, 60, 20)
            p.drawRoundedRect(zoom_r, 4, 4)
            p.setPen(QColor("#94a3b8"))
            font_small = p.font()
            font_small.setPointSize(8)
            p.setFont(font_small)
            p.drawText(zoom_r, Qt.AlignmentFlag.AlignCenter, f"{int(self.zoom * 100)}%")
            p.restore()

    def voxel_at(self, point):
        if self.volume is None or not self.target.contains(point):
            return None
        u = int((point.x() - self.target.left()) / self.target.width() * self.pix.width())
        v = int((point.y() - self.target.top()) / self.target.height() * self.pix.height())
        return slice_to_voxel(self.axis, u, v, self.xyz, self.volume.data.shape)

    def mousePressEvent(self, event):
        if event.button() in (Qt.MouseButton.RightButton, Qt.MouseButton.MiddleButton):
            self.last_pan = event.position()
        elif event.button() == Qt.MouseButton.LeftButton:
            xyz = self.voxel_at(event.position())
            if xyz is not None:
                self.dragging = True
                self.picked.emit(self.axis, xyz, True)

    def mouseMoveEvent(self, event):
        if self.last_pan is not None:
            self.pan += event.position() - self.last_pan
            self.last_pan = event.position()
            self.update()
        elif self.dragging:
            xyz = self.voxel_at(event.position())
            if xyz is not None:
                self.picked.emit(self.axis, xyz, False)
        else:
            vox = self.voxel_at(event.position())
            if vox is not None and self.volume is not None:
                self.hover_val = float(self.volume.data[vox[2], vox[1], vox[0]])
            else:
                self.hover_val = None
            self.update()

    def leaveEvent(self, event):
        self.hover_val = None
        self.update()

    def mouseReleaseEvent(self, event):
        if self.dragging:
            self.strokeEnded.emit()
        self.dragging = False
        self.last_pan = None

    def wheelEvent(self, event):
        direction = 1 if event.angleDelta().y() > 0 else -1
        if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            self.zoom = float(np.clip(self.zoom * (1.15 ** direction), 0.2, 15))
            self.update()
        else:
            self.stepped.emit(self.axis, direction)
        event.accept()


class SlicePanel(QWidget):
    indexChanged = Signal(int, int)
    maximizeRequested = Signal(int)

    def __init__(self, title, axis):
        super().__init__()
        self.axis = axis
        self.title_text = title.upper()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(1, 1, 1, 1)
        layout.setSpacing(0)

        # Header bar with accent badge and viewport controls
        self.header = QWidget()
        self.header.setObjectName("sliceHeader")
        h_layout = QHBoxLayout(self.header)
        h_layout.setContentsMargins(8, 4, 8, 4)
        h_layout.setSpacing(6)

        self.badge = QLabel(self.title_text)
        self.badge.setStyleSheet(f"""
            color: {ACCENTS[axis]};
            font-size: 11px;
            font-weight: 700;
            padding: 2px 7px;
            background: #18{ACCENTS[axis][1:]};
            border-radius: 4px;
            border: 1px solid #44{ACCENTS[axis][1:]};
        """)
        h_layout.addWidget(self.badge)

        self.counter = QLabel("—")
        self.counter.setStyleSheet("color: #94a3b8; font-size: 11px; font-weight: 500;")
        h_layout.addWidget(self.counter)

        h_layout.addStretch()

        self.btn_reset = QPushButton("⟲")
        self.btn_reset.setFixedSize(22, 22)
        self.btn_reset.setToolTip("Reset Zoom & Pan")
        self.btn_reset.setStyleSheet("""
            QPushButton { background: #16202e; border: 1px solid #27374d; border-radius: 4px; color: #8fa0b5; font-size: 12px; font-weight: bold; padding: 0; }
            QPushButton:hover { background: #263852; color: #ffffff; border-color: #3b5273; }
        """)
        self.btn_reset.clicked.connect(self.reset_view)
        h_layout.addWidget(self.btn_reset)

        self.btn_max = QPushButton("⛶")
        self.btn_max.setFixedSize(22, 22)
        self.btn_max.setToolTip("Maximize / Restore Viewport")
        self.btn_max.setStyleSheet("""
            QPushButton { background: #16202e; border: 1px solid #27374d; border-radius: 4px; color: #8fa0b5; font-size: 11px; font-weight: bold; padding: 0; }
            QPushButton:hover { background: #263852; color: #ffffff; border-color: #3b5273; }
        """)
        self.btn_max.clicked.connect(lambda: self.maximizeRequested.emit(self.axis))
        h_layout.addWidget(self.btn_max)

        layout.addWidget(self.header)

        self.canvas = SliceCanvas(axis)
        layout.addWidget(self.canvas, 1)
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setEnabled(False)
        self.slider.valueChanged.connect(lambda value: self.indexChanged.emit(axis, value))
        layout.addWidget(self.slider)

    def reset_view(self):
        self.canvas.reset_view()

    def refresh(self, volume, xyz, layers, level, window, invert=False):
        if volume is not None:
            size = volume.data.shape[2 - self.axis]
            self.slider.blockSignals(True)
            self.slider.setRange(0, size - 1)
            self.slider.setValue(xyz[self.axis])
            self.slider.setEnabled(True)
            self.slider.blockSignals(False)
            world_coord = volume.world(xyz)[self.axis]
            self.counter.setText(f"{xyz[self.axis] + 1} / {size}  ·  {world_coord:.1f} mm")
        self.canvas.refresh(volume, xyz, layers, level, window, invert)


class Scene(QWidget):
    maximizeRequested = Signal()

    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(1, 1, 1, 1)
        layout.setSpacing(0)

        # 3D Viewport Header
        self.header = QWidget()
        self.header.setObjectName("sceneHeader")
        h_layout = QHBoxLayout(self.header)
        h_layout.setContentsMargins(8, 4, 8, 4)
        h_layout.setSpacing(6)

        title = QLabel("3D EXPLORER")
        title.setStyleSheet("""
            color: #b8a0f4;
            font-size: 11px;
            font-weight: 700;
            padding: 2px 7px;
            background: #18b8a0f4;
            border-radius: 4px;
            border: 1px solid #44b8a0f4;
        """)
        h_layout.addWidget(title)

        units = QLabel("LPS / MM")
        units.setStyleSheet("color: #94a3b8; font-size: 10px; font-weight: 500;")
        h_layout.addWidget(units)

        h_layout.addStretch()

        for name, callback in [
            ("Top", lambda: self.set_view_preset("top")),
            ("Front", lambda: self.set_view_preset("front")),
            ("Side", lambda: self.set_view_preset("side")),
        ]:
            btn = QPushButton(name)
            btn.setFixedHeight(22)
            btn.setStyleSheet("""
                QPushButton { background: #16202e; border: 1px solid #27374d; border-radius: 4px; color: #8fa0b5; font-size: 10px; padding: 1px 6px; font-weight: 500; }
                QPushButton:hover { background: #263852; color: #ffffff; border-color: #3b5273; }
            """)
            btn.clicked.connect(callback)
            h_layout.addWidget(btn)

        self.btn_orbit = QPushButton("⟳ Orbit")
        self.btn_orbit.setCheckable(True)
        self.btn_orbit.setFixedHeight(22)
        self.btn_orbit.setToolTip("Continuous 360° Turntable Orbit (Key: O)")
        self.btn_orbit.setStyleSheet("""
            QPushButton { background: #16202e; border: 1px solid #27374d; border-radius: 4px; color: #8fa0b5; font-size: 10px; padding: 1px 6px; font-weight: 500; }
            QPushButton:hover { background: #263852; color: #ffffff; border-color: #3b5273; }
            QPushButton:checked { background: #112a3d; border-color: #38bdf8; color: #38bdf8; font-weight: bold; }
        """)
        self.btn_orbit.clicked.connect(lambda: self.toggle_turntable())
        h_layout.addWidget(self.btn_orbit)

        self.btn_scale = QPushButton("📏 Scale")
        self.btn_scale.setCheckable(True)
        self.btn_scale.setFixedHeight(22)
        self.btn_scale.setToolTip("Toggle 3D Physical LPS Millimeter Cube Axes (Key: C)")
        self.btn_scale.setStyleSheet("""
            QPushButton { background: #16202e; border: 1px solid #27374d; border-radius: 4px; color: #8fa0b5; font-size: 10px; padding: 1px 6px; font-weight: 500; }
            QPushButton:hover { background: #263852; color: #ffffff; border-color: #3b5273; }
            QPushButton:checked { background: #112a3d; border-color: #38bdf8; color: #38bdf8; font-weight: bold; }
        """)
        self.btn_scale.clicked.connect(lambda: self.toggle_axes())
        h_layout.addWidget(self.btn_scale)

        self.btn_anaglyph = QPushButton("👓 3D")
        self.btn_anaglyph.setCheckable(True)
        self.btn_anaglyph.setFixedHeight(22)
        self.btn_anaglyph.setToolTip("Toggle Red/Cyan Stereoscopic 3D Anaglyph")
        self.btn_anaglyph.setStyleSheet("""
            QPushButton { background: #16202e; border: 1px solid #27374d; border-radius: 4px; color: #8fa0b5; font-size: 10px; padding: 1px 6px; font-weight: 500; }
            QPushButton:hover { background: #263852; color: #ffffff; border-color: #3b5273; }
            QPushButton:checked { background: #112a3d; border-color: #38bdf8; color: #38bdf8; font-weight: bold; }
        """)
        self.btn_anaglyph.clicked.connect(lambda: self.toggle_anaglyph())
        h_layout.addWidget(self.btn_anaglyph)

        self.btn_reset_cam = QPushButton("⟲")
        self.btn_reset_cam.setFixedSize(22, 22)
        self.btn_reset_cam.setToolTip("Reset 3D Camera")
        self.btn_reset_cam.setStyleSheet("""
            QPushButton { background: #16202e; border: 1px solid #27374d; border-radius: 4px; color: #8fa0b5; font-size: 12px; font-weight: bold; padding: 0; }
            QPushButton:hover { background: #263852; color: #ffffff; border-color: #3b5273; }
        """)
        self.btn_reset_cam.clicked.connect(self.reset_camera)
        h_layout.addWidget(self.btn_reset_cam)

        self.btn_max = QPushButton("⛶")
        self.btn_max.setFixedSize(22, 22)
        self.btn_max.setToolTip("Maximize / Restore 3D Viewport")
        self.btn_max.setStyleSheet("""
            QPushButton { background: #16202e; border: 1px solid #27374d; border-radius: 4px; color: #8fa0b5; font-size: 11px; font-weight: bold; padding: 0; }
            QPushButton:hover { background: #263852; color: #ffffff; border-color: #3b5273; }
        """)
        self.btn_max.clicked.connect(lambda: self.maximizeRequested.emit())
        h_layout.addWidget(self.btn_max)

        layout.addWidget(self.header)

        self.widget = QVTKRenderWindowInteractor(self)
        layout.addWidget(self.widget, 1)
        self.renderer = vtk.vtkRenderer()
        self.renderer.SetBackground(0.031, 0.059, 0.098)
        self.renderer.SetBackground2(0.085, 0.125, 0.19)
        self.renderer.GradientBackgroundOn()
        self.widget.GetRenderWindow().AddRenderer(self.renderer)
        self.widget.GetRenderWindow().SetMultiSamples(0)
        self.widget.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())
        self.widget.Initialize()
        self.volume = None
        self.volume_actor = None
        self.mapper = None
        self.surface_actors = {}
        self.clip_plane = vtk.vtkPlane()
        self.clip_enabled = False
        self.opacity = 0.25
        self.level, self.window = 0, 1
        axes = vtk.vtkAxesActor()
        axes.SetXAxisLabelText("L")
        axes.SetYAxisLabelText("P")
        axes.SetZAxisLabelText("S")
        self.axes = vtk.vtkOrientationMarkerWidget()
        self.axes.SetOrientationMarker(axes)
        self.axes.SetInteractor(self.widget)
        self.axes.SetViewport(0, 0, 0.20, 0.25)
        self.axes.SetEnabled(1)
        self.axes.InteractiveOff()
        self.cursor = vtk.vtkCursor3D()
        self.cursor.AllOff()
        self.cursor.AxesOn()
        self.cursor_mapper = vtk.vtkPolyDataMapper()
        self.cursor_mapper.SetInputConnection(self.cursor.GetOutputPort())
        self.cursor_actor = vtk.vtkActor()
        self.cursor_actor.SetMapper(self.cursor_mapper)
        self.cursor_actor.GetProperty().SetColor(0.36, 0.86, 0.78)
        self.cursor_actor.GetProperty().SetLineWidth(1.5)
        self.renderer.AddActor(self.cursor_actor)
        self.cursor_actor.SetVisibility(False)

        self.turntable_timer = QTimer(self)
        self.turntable_timer.timeout.connect(self._turntable_step)

        self.cube_axes = vtk.vtkCubeAxesActor()
        self.cube_axes.SetCamera(self.renderer.GetActiveCamera())
        self.cube_axes.SetXTitle("L")
        self.cube_axes.SetYTitle("P")
        self.cube_axes.SetZTitle("S")
        self.cube_axes.SetXUnits("mm")
        self.cube_axes.SetYUnits("mm")
        self.cube_axes.SetZUnits("mm")
        self.cube_axes.SetFlyModeToStaticTriad()
        for i in range(3):
            self.cube_axes.GetTitleTextProperty(i).SetColor(0.22, 0.74, 0.97)
            self.cube_axes.GetLabelTextProperty(i).SetColor(0.70, 0.82, 0.92)
        self.cube_axes.GetProperty().SetColor(0.25, 0.35, 0.48)
        self.cube_axes.SetVisibility(False)
        self.renderer.AddActor(self.cube_axes)

    def toggle_turntable(self, enabled=None):
        if enabled is None:
            enabled = not self.turntable_timer.isActive()
        if enabled:
            self.turntable_timer.start(30)
        else:
            self.turntable_timer.stop()
        if hasattr(self, "btn_orbit"):
            self.btn_orbit.blockSignals(True)
            self.btn_orbit.setChecked(enabled)
            self.btn_orbit.blockSignals(False)
        return enabled

    def _turntable_step(self):
        self.renderer.GetActiveCamera().Azimuth(0.8)
        self.render()

    def toggle_axes(self, enabled=None):
        if self.cube_axes is None:
            return False
        if enabled is None:
            enabled = not self.cube_axes.GetVisibility()
        self.cube_axes.SetVisibility(bool(enabled))
        if hasattr(self, "btn_scale"):
            self.btn_scale.blockSignals(True)
            self.btn_scale.setChecked(enabled)
            self.btn_scale.blockSignals(False)
        self.render()
        return enabled

    def toggle_anaglyph(self, enabled=None):
        rw = self.widget.GetRenderWindow()
        if enabled is None:
            enabled = not rw.GetStereoRender()
        if enabled:
            rw.SetStereoTypeToAnaglyph()
            rw.StereoRenderOn()
        else:
            rw.StereoRenderOff()
        if hasattr(self, "btn_anaglyph"):
            self.btn_anaglyph.blockSignals(True)
            self.btn_anaglyph.setChecked(enabled)
            self.btn_anaglyph.blockSignals(False)
        self.render()
        return rw.GetStereoRender()

    def set_view_preset(self, preset):
        if self.volume is None:
            return
        camera = self.renderer.GetActiveCamera()
        center = self.volume.world((np.array(self.volume.image.GetSize()) - 1) / 2)
        camera.SetFocalPoint(*center)
        dist = 500
        if preset == "top":
            camera.SetPosition(center[0], center[1], center[2] + dist)
            camera.SetViewUp(0, 1, 0)
        elif preset == "front":
            camera.SetPosition(center[0], center[1] - dist, center[2])
            camera.SetViewUp(0, 0, 1)
        elif preset == "side":
            camera.SetPosition(center[0] + dist, center[1], center[2])
            camera.SetViewUp(0, 0, 1)
        self.renderer.ResetCameraClippingRange()
        self.render()

    def set_volume(self, volume):
        self.volume = volume
        if self.volume_actor:
            self.renderer.RemoveVolume(self.volume_actor)
        for actor in self.surface_actors.values():
            self.renderer.RemoveActor(actor)
        self.surface_actors.clear()
        image = vtk_image(volume.data, volume.spacing, volume.origin)
        self.mapper = vtk.vtkSmartVolumeMapper()
        self.mapper.SetInputData(image)
        self.mapper.SetInteractiveAdjustSampleDistances(True)
        self.mapper.SetInteractiveUpdateRate(20)
        self.mapper.SetBlendModeToComposite()
        self.mapper.SetMaxMemoryInBytes(768 * 1024 * 1024)
        self.property = vtk.vtkVolumeProperty()
        self.property.SetInterpolationTypeToLinear()
        self.property.ShadeOn()
        self.property.SetAmbient(0.2)
        self.property.SetDiffuse(0.7)
        self.property.SetSpecular(0.3)
        self.property.SetSpecularPower(20.0)
        self.volume_actor = vtk.vtkVolume()
        self.volume_actor.SetMapper(self.mapper)
        self.volume_actor.SetProperty(self.property)
        self.renderer.AddVolume(self.volume_actor)
        self.cursor.SetModelBounds(*image.GetBounds())
        self.cursor_actor.SetVisibility(True)
        self.cube_axes.SetBounds(*image.GetBounds())
        self.reset_camera()

    def reset_camera(self):
        camera = self.renderer.GetActiveCamera()
        if self.volume is not None:
            center = self.volume.world((np.array(self.volume.image.GetSize()) - 1) / 2)
            camera.SetFocalPoint(*center)
            camera.SetPosition(center[0], center[1] - 500, center[2] + 100)
            camera.SetViewUp(0, 0, 1)
        self.renderer.ResetCamera()
        self.render()

    def appearance(self, level, window, opacity, visible, invert=False):
        if self.volume_actor is None:
            return
        self.level, self.window, self.opacity = level, window, opacity
        low, high = level - window / 2, level + window / 2
        colors = vtk.vtkColorTransferFunction()
        palette = [(0.02, 0.04, 0.08), (0.4, 0.54, 0.64), (0.93, 0.86, 0.72)]
        if invert:
            palette.reverse()
        for t, color in zip([low, level, high], palette):
            colors.AddRGBPoint(t, *color)
        alpha = vtk.vtkPiecewiseFunction()
        alpha.AddPoint(low, 0)
        alpha.AddPoint(level, opacity * 0.15)
        alpha.AddPoint(high, opacity)
        self.property.SetColor(colors)
        self.property.SetScalarOpacity(alpha)
        self.property.SetScalarOpacityUnitDistance(min(self.volume.spacing))
        self.volume_actor.SetVisibility(visible)
        self.render()

    def set_cursor(self, xyz, render=True):
        if self.volume is not None:
            self.cursor.SetFocalPoint(*self.volume.world(xyz))
            if render:
                self.render()

    def sync_layers(self, layers):
        ids = {l.id for l in layers if l.mesh is not None}
        for ident in list(self.surface_actors):
            if ident not in ids:
                self.renderer.RemoveActor(self.surface_actors.pop(ident))
        for layer in layers:
            if layer.mesh is None:
                continue
            if layer.id not in self.surface_actors:
                mapper = vtk.vtkPolyDataMapper()
                mapper.ScalarVisibilityOff()
                actor = vtk.vtkActor()
                actor.SetMapper(mapper)
                self.renderer.AddActor(actor)
                self.surface_actors[layer.id] = actor
            actor = self.surface_actors[layer.id]
            actor.GetMapper().SetInputData(layer.mesh)
            prop = actor.GetProperty()
            rep = getattr(layer, "representation", "surface")
            if rep == "wireframe":
                prop.SetRepresentationToWireframe()
                prop.SetLineWidth(1.4)
            elif rep == "points":
                prop.SetRepresentationToPoints()
                prop.SetPointSize(3.0)
            else:
                prop.SetRepresentationToSurface()
            prop.SetColor(*[c / 255 for c in layer.color])
            prop.SetOpacity(layer.opacity)
            prop.SetAmbient(0.18)
            prop.SetDiffuse(0.75)
            prop.SetSpecular(0.35)
            prop.SetSpecularPower(25.0)
            actor.SetVisibility(layer.visible)
            actor.GetMapper().RemoveAllClippingPlanes()
            if self.clip_enabled:
                actor.GetMapper().AddClippingPlane(self.clip_plane)
        self.render()

    def clipping(self, enabled, axis, fraction, flipped, render=True):
        self.clip_enabled = enabled
        if self.volume is None:
            return
        normal = [0., 0., 0.]
        normal[axis] = -1 if flipped else 1
        self.clip_plane.SetNormal(*normal)
        size = np.array(self.volume.image.GetSize()) - 1
        xyz = size / 2.0
        xyz[axis] = size[axis] * fraction
        self.clip_plane.SetOrigin(*self.volume.world(xyz))
        for mapper in [self.mapper] + [a.GetMapper() for a in self.surface_actors.values()]:
            mapper.RemoveAllClippingPlanes()
            if enabled:
                mapper.AddClippingPlane(self.clip_plane)
        if render:
            self.render()

    def render(self):
        self.widget.GetRenderWindow().Render()

    def close(self):
        if hasattr(self, "turntable_timer"):
            self.turntable_timer.stop()
        self.axes.SetEnabled(0)
        self.widget.Finalize()
        super().close()
