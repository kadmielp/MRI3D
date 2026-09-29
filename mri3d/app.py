from __future__ import annotations

import argparse
from pathlib import Path
import sys
import traceback
import numpy as np
from PySide6.QtCore import Qt, QObject, QThread, Signal, Slot, QTimer
from PySide6.QtGui import QAction, QColor, QKeySequence
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QSplitter, QScrollArea, QLabel, QPushButton, QFileDialog, QTreeWidget,
    QTreeWidgetItem, QHeaderView, QGroupBox, QFormLayout, QDoubleSpinBox,
    QComboBox, QCheckBox, QSlider, QListWidget, QListWidgetItem, QInputDialog,
    QColorDialog, QMessageBox, QProgressBar, QToolBar, QAbstractItemView,
    QTabWidget, QButtonGroup, QFrame, QDialog,
)
from .imaging import scan_folder, load_series, segment, Job, Cancelled, ImagingError, fuse_series
from .models import Layer, make_mesh, export_mesh, paint_mask, filter_clusters
from .projects import save_project, open_project, read_manifest, missing_sources
from .views import SlicePanel, Scene, ACCENTS


DEFAULT_FOLDER = r"G:\Vibing\MRI3D\DICOMIMAGES"
COLORS = [(244, 203, 141), (88, 216, 188), (127, 170, 255), (210, 155, 247), (245, 138, 144)]
STYLE = """
QMainWindow, QDialog {
    background: #090e17;
    color: #e2e8f0;
    font-family: 'Segoe UI', 'Inter', -apple-system, sans-serif;
    font-size: 12px;
}
QWidget {
    color: #cbd5e1;
    font-family: 'Segoe UI', 'Inter', -apple-system, sans-serif;
    font-size: 12px;
}
QToolBar {
    background: #0f1724;
    border-bottom: 1px solid #1c2b3d;
    padding: 5px 10px;
    spacing: 6px;
}
QToolButton {
    padding: 6px 12px;
    border-radius: 5px;
    background: #172233;
    border: 1px solid #23354d;
    color: #e2e8f0;
    font-weight: 500;
}
QToolButton:hover {
    background: #23344d;
    border-color: #3b5275;
    color: #ffffff;
}
QPushButton {
    background: #152030;
    border: 1px solid #24354c;
    border-radius: 5px;
    padding: 6px 12px;
    color: #e2e8f0;
    font-weight: 500;
}
QPushButton:hover {
    background: #20314a;
    border-color: #38537a;
    color: #ffffff;
}
QPushButton:pressed {
    background: #111a26;
}
QPushButton#primary {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #0d9488, stop:1 #14b8a6);
    border: 1px solid #2dd4bf;
    color: #ffffff;
    font-weight: 600;
}
QPushButton#primary:hover {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #0f766e, stop:1 #0d9488);
}
QPushButton#toolBtn {
    background: #121c2b;
    border: 1px solid #203047;
    border-radius: 5px;
    padding: 7px 10px;
    font-weight: 600;
    color: #94a3b8;
}
QPushButton#toolBtn:hover {
    background: #1b2a40;
    color: #ffffff;
    border-color: #334a6d;
}
QPushButton#toolBtn:checked {
    background: #0284c7;
    border: 1px solid #38bdf8;
    color: #ffffff;
}
QPushButton#navBtn {
    background: #141f2e;
    border: 1px solid #22344a;
    border-radius: 4px;
    padding: 4px 9px;
    color: #94a3b8;
    font-size: 11px;
    font-weight: 600;
}
QPushButton#navBtn:hover {
    background: #203148;
    color: #ffffff;
    border-color: #38bdf8;
}
QPushButton#navBtn:checked {
    background: #0369a1;
    color: #ffffff;
    border: 1px solid #38bdf8;
}
QPushButton:disabled, QToolButton:disabled {
    color: #475569;
    background: #0f1622;
    border-color: #192331;
}
QTabWidget::pane {
    border: 1px solid #1c2b3d;
    background: #0d1420;
    border-radius: 6px;
    top: -1px;
}
QTabBar::tab {
    background: #131d2a;
    color: #8fa0b5;
    border: 1px solid #1c2b3d;
    border-bottom: none;
    padding: 7px 14px;
    font-size: 11px;
    font-weight: 600;
    border-top-left-radius: 5px;
    border-top-right-radius: 5px;
    margin-right: 3px;
}
QTabBar::tab:selected {
    background: #0d1420;
    color: #38bdf8;
    border-color: #1c2b3d;
    border-bottom: 2px solid #38bdf8;
}
QTabBar::tab:hover:!selected {
    background: #1a2738;
    color: #cbd5e1;
}
QGroupBox {
    border: 1px solid #1c2b3d;
    border-radius: 6px;
    margin-top: 10px;
    padding: 10px 8px 8px;
    background: #111a26;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 10px;
    color: #7dd3fc;
    padding: 0 4px;
    font-weight: 700;
    font-size: 10px;
    letter-spacing: 0.5px;
}
QTreeWidget, QListWidget {
    background: #090e17;
    border: 1px solid #1c2a3d;
    border-radius: 5px;
    color: #e2e8f0;
}
QTreeWidget::item, QListWidget::item {
    padding: 6px 8px;
    border-radius: 4px;
}
QTreeWidget::item:hover, QListWidget::item:hover {
    background: #141e2c;
}
QTreeWidget::item:selected, QListWidget::item:selected {
    background: #1a2e45;
    color: #38bdf8;
}
QHeaderView::section {
    background: #131d2b;
    color: #94a3b8;
    padding: 6px 8px;
    border: none;
    border-bottom: 1px solid #1e2c3e;
    font-weight: 600;
    font-size: 11px;
}
QComboBox, QDoubleSpinBox, QLineEdit {
    background: #090e17;
    border: 1px solid #223247;
    border-radius: 5px;
    padding: 5px 8px;
    color: #f1f5f9;
}
QComboBox:hover, QDoubleSpinBox:hover, QLineEdit:hover {
    border-color: #364d6e;
}
QComboBox:focus, QDoubleSpinBox:focus, QLineEdit:focus {
    border-color: #38bdf8;
}
QComboBox::drop-down {
    border: none;
    padding-right: 6px;
}
QSlider::groove:horizontal {
    height: 4px;
    background: #1c2838;
    border-radius: 2px;
}
QSlider::sub-page:horizontal {
    background: #14b8a6;
    border-radius: 2px;
}
QSlider::handle:horizontal {
    width: 14px;
    height: 14px;
    background: #2dd4bf;
    margin: -5px 0;
    border-radius: 7px;
    border: 1px solid #0d9488;
}
QSlider::handle:horizontal:hover {
    background: #5eead4;
    border-color: #14b8a6;
}
QSlider {
    min-height: 20px;
}
QScrollArea {
    border: none;
    background: transparent;
}
QScrollBar:vertical {
    background: #090e17;
    width: 7px;
    margin: 0;
    border-radius: 3px;
}
QScrollBar::handle:vertical {
    background: #223247;
    min-height: 25px;
    border-radius: 3px;
}
QScrollBar::handle:vertical:hover {
    background: #364d6e;
}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    height: 0;
}
QStatusBar {
    background: #0c121c;
    color: #8fa0b5;
    border-top: 1px solid #182433;
    padding: 2px 6px;
}
QProgressBar {
    border: 1px solid #223247;
    border-radius: 4px;
    text-align: center;
    max-height: 16px;
    background: #090e17;
    font-size: 10px;
    font-weight: 600;
    color: #f1f5f9;
}
QProgressBar::chunk {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #0d9488, stop:1 #2dd4bf);
    border-radius: 3px;
}
QCheckBox {
    spacing: 7px;
    color: #cbd5e1;
}
QCheckBox::indicator {
    width: 15px;
    height: 15px;
    border: 1px solid #26384f;
    border-radius: 3px;
    background: #090e17;
}
QCheckBox::indicator:checked {
    background: #14b8a6;
    border-color: #2dd4bf;
}
QSplitter::handle {
    background: #182433;
    width: 3px;
    height: 3px;
}
QSplitter::handle:hover {
    background: #38bdf8;
}
QToolTip {
    background: #182333;
    color: #f8fafc;
    border: 1px solid #364d6e;
    border-radius: 4px;
    padding: 4px 8px;
}
"""


class ShortcutsDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("MRI3D · Shortcuts & Ergonomics Guide")
        self.resize(680, 560)
        self.setStyleSheet(STYLE)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        header = QLabel("KEYBOARD SHORTCUTS & ERGONOMICS")
        header.setStyleSheet("font-size: 15px; font-weight: 700; color: #38bdf8; letter-spacing: 0.5px;")
        layout.addWidget(header)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        content = QWidget()
        c_layout = QVBoxLayout(content)
        c_layout.setSpacing(12)

        sections = [
            ("SLICE NAVIGATION & 2D TOOLS", [
                ("Left Click / Drag", "Navigate to slice, place seed, or paint mask stroke"),
                ("Right / Middle Drag", "Pan active 2D slice view"),
                ("Mouse Wheel", "Scroll through consecutive slices"),
                ("Ctrl + Mouse Wheel", "Zoom in / out on slice"),
                ("V", "Activate Navigate / Seed tool"),
                ("B", "Activate Brush painting tool"),
                ("E", "Activate Erase tool"),
                ("[  /  ]", "Decrease / Increase brush radius by 1 mm"),
                ("R", "Reset pan and zoom on all slice views"),
            ]),
            ("3D EXPLORATION & CLIPPING", [
                ("Left Drag in 3D", "Rotate 3D volume in patient LPS space"),
                ("Mouse Wheel in 3D", "Zoom 3D camera in and out"),
                ("Top / Front / Side", "Instant anatomical view orientation presets"),
                ("O", "Toggle 360° Turntable Orbit (continuous motion parallax)"),
                ("C", "Toggle 3D Physical LPS Millimeter Cube Axes"),
                ("👓 3D Button", "Toggle Red/Cyan Stereoscopic 3D Anaglyph"),
                ("Clip 3D View", "Dynamic cut plane linked to active slice slider / scroll"),
                ("Reverse Cut", "Invert the cut direction along active axis"),
            ]),
            ("VIEWPORT LAYOUTS", [
                ("Space", "Cycle between Quad (2x2), 3D Focused, and Single viewports"),
                ("Double-Click Header", "Maximize or restore the clicked viewport"),
                ("Escape", "Instantly restore default 2x2 Quad layout"),
            ]),
            ("SEGMENTATION WORKFLOW", [
                ("Ctrl + Z  /  Ctrl + Y", "Undo / Redo up to 20 brush and mask operations"),
                ("Render Mode", "Toggle Shaded Surface, Wireframe, or Point Cloud"),
                ("Auto-fit ±15%", "Automatically set lower & upper intensity bounds around seed"),
                ("Grow from Seed", "Connected region growing within intensity limits"),
                ("Keep Largest", "Isolate main anatomical component and discard noise"),
                ("Filter Noise", "Drop disconnected noise clusters (< 25 voxels)"),
                ("Fill Holes", "Close internal 3D cavities"),
                ("✦ Generate Surface", "Construct 3D polygon surface with Flying Edges"),
                ("Ctrl + S  /  Ctrl + E", "Save project / Export surface mesh (STL or PLY)"),
            ])
        ]

        for sec_title, items in sections:
            grp = QGroupBox(sec_title)
            g_layout = QGridLayout(grp)
            g_layout.setContentsMargins(10, 10, 10, 10)
            g_layout.setHorizontalSpacing(16)
            g_layout.setVerticalSpacing(5)
            for row, (key, desc) in enumerate(items):
                lbl_key = QLabel(key)
                lbl_key.setStyleSheet("color: #38bdf8; font-weight: 700; font-family: 'Consolas', monospace;")
                lbl_desc = QLabel(desc)
                lbl_desc.setStyleSheet("color: #cbd5e1;")
                g_layout.addWidget(lbl_key, row, 0)
                g_layout.addWidget(lbl_desc, row, 1)
            g_layout.setColumnStretch(1, 1)
            c_layout.addWidget(grp)

        scroll.setWidget(content)
        layout.addWidget(scroll, 1)

        btn_close = button("Close", self.accept, True)
        btn_close.setFixedWidth(100)
        b_row = QHBoxLayout()
        b_row.addStretch()
        b_row.addWidget(btn_close)
        layout.addLayout(b_row)


class Worker(QObject):
    progress = Signal(int, str)
    result = Signal(object)
    error = Signal(str)
    finished = Signal()

    def __init__(self, fn):
        super().__init__()
        self.fn = fn
        self.job = Job(self.progress.emit)

    @Slot()
    def run(self):
        try:
            self.result.emit(self.fn(self.job))
        except Cancelled:
            self.error.emit("")
        except Exception as exc:
            if self.job.cancelled.is_set():
                self.error.emit("")
            else:
                traceback.print_exc()
                self.error.emit(str(exc))
        finally:
            self.finished.emit()


def button(text, callback, primary=False):
    result = QPushButton(text)
    if primary:
        result.setObjectName("primary")
    result.clicked.connect(callback)
    return result


def spin(value=0, low=-1e9, high=1e9, decimals=1):
    result = QDoubleSpinBox()
    result.setRange(low, high)
    result.setDecimals(decimals)
    result.setValue(value)
    result.setKeyboardTracking(False)
    return result


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("MRI3D · Local MRI Explorer")
        self.resize(1500, 950)
        self.volume = None
        self.layers = []
        self.xyz = [0, 0, 0]
        self.seed = None
        self.folder = DEFAULT_FOLDER
        self.project_path = None
        self.dirty = False
        self.thread = None
        self.worker = None
        self.success_callback = None
        self.stroke_previous = None
        self.updating = False
        self.layout_mode = "quad"
        self.maximized_widget = None

        self.content = QWidget()
        outer = QVBoxLayout(self.content)
        outer.setContentsMargins(10, 8, 10, 6)
        outer.setSpacing(6)

        # Command Center / Top Navigation Header
        top_bar = QWidget()
        top_bar.setObjectName("topBar")
        top_bar.setStyleSheet("""
            QWidget#topBar {
                background: #0d1522;
                border: 1px solid #1a2738;
                border-radius: 8px;
                padding: 4px;
            }
        """)
        top_layout = QHBoxLayout(top_bar)
        top_layout.setContentsMargins(12, 5, 12, 5)
        top_layout.setSpacing(10)

        # Brand / Logo
        brand_layout = QVBoxLayout()
        brand_layout.setSpacing(0)
        title = QLabel("MRI3D")
        title.setStyleSheet("font-size: 19px; font-weight: 800; color: #38bdf8; letter-spacing: 1px;")
        brand_layout.addWidget(title)
        subtitle = QLabel("LOCAL WORKSTATION")
        subtitle.setStyleSheet("color: #64748b; font-size: 9px; font-weight: 700; letter-spacing: 1.5px;")
        brand_layout.addWidget(subtitle)
        top_layout.addLayout(brand_layout)

        # Dataset Status Chip
        self.series_chip = QLabel("No scan series loaded · Click 'Import DICOM' to begin")
        self.series_chip.setStyleSheet("""
            background: #131d2b;
            color: #94a3b8;
            border: 1px solid #1f2e42;
            border-radius: 5px;
            padding: 5px 12px;
            font-size: 11px;
        """)
        top_layout.addWidget(self.series_chip, 1)

        # Viewport Layout Switcher
        layout_box = QHBoxLayout()
        layout_box.setSpacing(3)
        lbl_layout = QLabel("LAYOUT:")
        lbl_layout.setStyleSheet("color: #64748b; font-size: 10px; font-weight: 700;")
        layout_box.addWidget(lbl_layout)

        self.layout_buttons = {}
        for name, mode in [("⊞ Quad", "quad"), ("◰ 3D", "3d"), ("⬚ Axial", "axial"), ("⬚ Coronal", "coronal"), ("⬚ Sagittal", "sagittal")]:
            btn = QPushButton(name)
            btn.setObjectName("navBtn")
            btn.setCheckable(True)
            btn.clicked.connect(lambda checked=False, m=mode: self.set_layout_mode(m))
            layout_box.addWidget(btn)
            self.layout_buttons[mode] = btn
        self.layout_buttons["quad"].setChecked(True)
        top_layout.addLayout(layout_box)

        # Main Actions
        act_box = QHBoxLayout()
        act_box.setSpacing(5)
        for label, fn, key in [
            ("📁 Import DICOM", self.import_folder, "Ctrl+I"),
            ("📂 Open Project", self.open_dialog, "Ctrl+O"),
            ("💾 Save", self.save_dialog, "Ctrl+S"),
            ("⟲ Reset Views", self.reset_views, "R"),
            ("⌨ Shortcuts (F1)", self.guide, "F1"),
        ]:
            btn = QPushButton(label)
            btn.setToolTip(f"Shortcut: {key}")
            btn.clicked.connect(fn)
            act_box.addWidget(btn)
        top_layout.addLayout(act_box)

        outer.addWidget(top_bar)

        # Banner kept for compatibility
        self.banner = QLabel("Import a scan folder, choose a series, then explore or mark tissues to create a model.")
        self.banner.setWordWrap(True)
        self.banner.setStyleSheet("color: #64748b; font-size: 11px; padding: 2px 6px;")
        self.banner.hide()
        outer.addWidget(self.banner)

        # Main Horizontal Splitter
        splitter = QSplitter(Qt.Orientation.Horizontal)
        self.sidebar = self.build_sidebar()
        splitter.addWidget(self.sidebar)

        # Grid host for viewports
        grid_host = QWidget()
        grid_host.setObjectName("gridHost")
        grid_host.setStyleSheet("QWidget#gridHost { background: #070c14; border-radius: 8px; border: 1px solid #141f2e; }")
        self.grid = QGridLayout(grid_host)
        self.grid.setContentsMargins(4, 4, 4, 4)
        self.grid.setSpacing(4)

        self.panels = [SlicePanel("Axial", 2), SlicePanel("Coronal", 1), SlicePanel("Sagittal", 0)]
        self.scene = Scene()
        self.view_widgets = [self.panels[0], self.panels[1], self.panels[2], self.scene]

        self.grid.addWidget(self.panels[0], 0, 0)
        self.grid.addWidget(self.panels[1], 0, 1)
        self.grid.addWidget(self.panels[2], 1, 0)
        self.grid.addWidget(self.scene, 1, 1)
        self.grid.setRowStretch(0, 1)
        self.grid.setRowStretch(1, 1)
        self.grid.setColumnStretch(0, 1)
        self.grid.setColumnStretch(1, 1)

        # Connect maximize signals
        self.panels[0].maximizeRequested.connect(lambda a: self.toggle_maximize(self.panels[0]))
        self.panels[1].maximizeRequested.connect(lambda a: self.toggle_maximize(self.panels[1]))
        self.panels[2].maximizeRequested.connect(lambda a: self.toggle_maximize(self.panels[2]))
        self.scene.maximizeRequested.connect(lambda: self.toggle_maximize(self.scene))

        splitter.addWidget(grid_host)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([340, 1160])
        outer.addWidget(splitter, 1)

        footer = QLabel("Shortcuts: V (Seed) · B (Brush) · E (Erase) · [ / ] (Radius) · Space (Layout) · Esc (Quad) · F1 (Shortcuts)")
        footer.setStyleSheet("color: #64748b; font-size: 11px; padding: 2px 4px;")
        outer.addWidget(footer)

        self.setCentralWidget(self.content)
        self.build_toolbar()

        for panel in self.panels:
            panel.indexChanged.connect(self.set_index)
            panel.canvas.picked.connect(self.pick)
            panel.canvas.strokeEnded.connect(self.end_stroke)
            panel.canvas.stepped.connect(self.step)

        self.progress = QProgressBar()
        self.progress.setFixedWidth(160)
        self.cancel_button = button("Cancel", self.cancel_job)
        self.statusBar().addPermanentWidget(self.progress)
        self.statusBar().addPermanentWidget(self.cancel_button)
        self.progress.hide()
        self.cancel_button.hide()
        self.statusBar().showMessage("Ready · Precision local MRI workstation")

    def build_toolbar(self):
        self.toolbar = QToolBar()
        self.toolbar.setMovable(False)
        self.addToolBar(self.toolbar)
        for label, fn, key in [("Import DICOM", self.import_folder, "Ctrl+I"),
                               ("Open project", self.open_dialog, "Ctrl+O"),
                               ("Save project", self.save_dialog, "Ctrl+S"),
                               ("Save as", lambda: self.save_dialog(save_as=True), "Ctrl+Shift+S"),
                               ("Export model", self.export_dialog, "Ctrl+E"),
                               ("Reset views", self.reset_views, "R"),
                               ("Guide", self.guide, "F1")]:
            action = QAction(label, self)
            action.setShortcut(QKeySequence(key))
            action.triggered.connect(fn)
            self.toolbar.addAction(action)
        for shortcut, fn in [("Ctrl+Z", lambda: self.undo(False)), ("Ctrl+Y", lambda: self.undo(True))]:
            action = QAction(self)
            action.setShortcut(QKeySequence(shortcut))
            action.triggered.connect(fn)
            self.addAction(action)
        self.toolbar.hide()

    def build_sidebar(self):
        self.tabs = QTabWidget()
        self.tabs.setObjectName("sidebarTabs")
        self.tabs.setDocumentMode(True)

        # Tab 1: Scans
        tab_scans = QWidget()
        layout_scans = QVBoxLayout(tab_scans)
        layout_scans.setContentsMargins(8, 8, 8, 8)
        layout_scans.setSpacing(8)
        layout_scans.addWidget(button("📁 Choose DICOM folder…", self.import_folder))

        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(["Series / status", "Slices", "Spacing mm"])
        self.tree.setRootIsDecorated(False)
        self.tree.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.tree.setMinimumHeight(170)
        self.tree.header().setSectionResizeMode(0, QHeaderView.ResizeMode.Interactive)
        self.tree.setColumnWidth(0, 160)
        self.tree.setColumnWidth(1, 45)
        self.tree.setColumnWidth(2, 110)
        self.tree.itemSelectionChanged.connect(self.series_selected)
        self.tree.itemDoubleClicked.connect(lambda item, col: self.load_selected())
        layout_scans.addWidget(self.tree, 1)

        self.series_info = QLabel("No scans imported")
        self.series_info.setWordWrap(True)
        self.series_info.setStyleSheet("color: #94a3b8; font-size: 11px; padding: 4px;")
        layout_scans.addWidget(self.series_info)

        btn_row = QHBoxLayout()
        self.auto_pair_button = button("⚡ Auto-select orthogonal trio", self.auto_select_orthogonal)
        self.auto_pair_button.setToolTip("Automatically find and select matching Axial, Coronal, and Sagittal planes for super-resolution fusion")
        btn_row.addWidget(self.auto_pair_button)
        layout_scans.addLayout(btn_row)

        self.load_button = button("Open selected series", self.load_selected, True)
        self.load_button.setEnabled(False)
        layout_scans.addWidget(self.load_button)
        self.tabs.addTab(tab_scans, "📂 Scans")

        # Tab 2: Display & 3D
        tab_display = QWidget()
        layout_display = QVBoxLayout(tab_display)
        layout_display.setContentsMargins(8, 8, 8, 8)
        layout_display.setSpacing(10)

        # Contrast group
        grp_contrast = QGroupBox("CONTRAST & WINDOWING")
        box_contrast = QVBoxLayout(grp_contrast)
        box_contrast.setSpacing(6)

        # Presets row
        preset_row = QHBoxLayout()
        preset_row.setSpacing(4)
        for name, preset_id in [("Auto", "auto"), ("Soft Tissue", "soft"), ("Bone", "bone"), ("Full", "full")]:
            btn = QPushButton(name)
            btn.setStyleSheet("padding: 4px 6px; font-size: 11px;")
            btn.clicked.connect(lambda checked=False, p=preset_id: self.apply_contrast_preset(p))
            preset_row.addWidget(btn)
        box_contrast.addLayout(preset_row)

        form_c = QFormLayout()
        self.window_spin = spin(1000, 0.001)
        self.level_spin = spin(500)
        self.window_spin.valueChanged.connect(self.display_changed)
        self.level_spin.valueChanged.connect(self.display_changed)
        form_c.addRow("Window", self.window_spin)
        form_c.addRow("Level", self.level_spin)
        self.invert = QCheckBox("Invert grayscale")
        self.invert.toggled.connect(self.display_changed)
        form_c.addRow(self.invert)
        box_contrast.addLayout(form_c)
        layout_display.addWidget(grp_contrast)

        # 3D Volume Display group
        grp_vol = QGroupBox("3D VOLUME RENDERING")
        form_v = QFormLayout(grp_vol)
        self.show_volume = QCheckBox("Show MRI volume in 3D")
        self.show_volume.setChecked(True)
        self.show_volume.toggled.connect(self.display_changed)
        form_v.addRow(self.show_volume)
        self.volume_opacity = QSlider(Qt.Orientation.Horizontal)
        self.volume_opacity.setRange(0, 100)
        self.volume_opacity.setValue(25)
        self.volume_opacity.valueChanged.connect(self.display_changed)
        form_v.addRow("Opacity", self.volume_opacity)
        layout_display.addWidget(grp_vol)

        # 3D Clipping group
        grp_clip = QGroupBox("DYNAMIC 3D CLIPPING")
        form_clip = QFormLayout(grp_clip)
        self.clip = QCheckBox("Clip 3D view")
        self.clip.toggled.connect(self.clip_changed)
        form_clip.addRow(self.clip)
        self.clip_axis = QComboBox()
        self.clip_axis.addItems(["Left / right (Sagittal)", "Anterior / posterior (Coronal)", "Inferior / superior (Axial)"])
        self.clip_axis.currentIndexChanged.connect(self.clip_changed)
        form_clip.addRow("Cut axis", self.clip_axis)
        self.clip_label = QLabel("—")
        self.clip_label.setStyleSheet("color: #94a3b8;")
        form_clip.addRow("Cut plane", self.clip_label)
        self.clip_flip = QCheckBox("Reverse cut")
        self.clip_flip.toggled.connect(self.clip_changed)
        form_clip.addRow(self.clip_flip)
        # Retain hidden clip_position for compatibility
        self.clip_position = QSlider(Qt.Orientation.Horizontal)
        self.clip_position.setRange(0, 100)
        self.clip_position.setValue(50)
        self.clip_position.valueChanged.connect(self.clip_position_changed)
        layout_display.addWidget(grp_clip)
        layout_display.addStretch()
        self.tabs.addTab(tab_display, "🎛 Display")

        # Tab 3: Segmentation
        tab_seg = QWidget()
        scroll_seg = QScrollArea()
        scroll_seg.setWidgetResizable(True)
        content_seg = QWidget()
        layout_seg = QVBoxLayout(content_seg)
        layout_seg.setContentsMargins(8, 8, 8, 8)
        layout_seg.setSpacing(10)

        # Slice Tool Segmented Buttons
        grp_tool = QGroupBox("ACTIVE SLICE TOOL")
        box_tool = QVBoxLayout(grp_tool)
        tool_row = QHBoxLayout()
        tool_row.setSpacing(4)
        self.tool_buttons = [
            QPushButton("🎯 Seed (V)"),
            QPushButton("🖌 Brush (B)"),
            QPushButton("⌫ Erase (E)")
        ]
        self.tool_group = QButtonGroup(self)
        self.tool_group.setExclusive(True)
        for i, btn in enumerate(self.tool_buttons):
            btn.setCheckable(True)
            btn.setObjectName("toolBtn")
            self.tool_group.addButton(btn, i)
            tool_row.addWidget(btn)
        self.tool_buttons[0].setChecked(True)
        self.tool_group.idClicked.connect(self.set_tool_index)
        box_tool.addLayout(tool_row)

        # Mirrored combo box for backward compatibility
        self.tool = QComboBox()
        self.tool.addItems(["Navigate / seed", "Brush", "Erase"])
        self.tool.currentIndexChanged.connect(self.tool_index_changed)

        # Brush radius with quick presets
        form_r = QFormLayout()
        self.radius = spin(3, 0.2, 50)
        self.radius.setSuffix(" mm")
        r_row = QHBoxLayout()
        r_row.addWidget(self.radius, 1)
        for r_val in [1, 3, 6, 12]:
            b_r = QPushButton(f"{r_val}mm")
            b_r.setStyleSheet("padding: 3px 5px; font-size: 10px;")
            b_r.clicked.connect(lambda checked=False, val=r_val: self.radius.setValue(val))
            r_row.addWidget(b_r)
        form_r.addRow("Radius", r_row)
        box_tool.addLayout(form_r)
        layout_seg.addWidget(grp_tool)

        # Tissue layers
        grp_layers = QGroupBox("TISSUE LAYERS")
        box_layers = QVBoxLayout(grp_layers)
        self.layer_list = QListWidget()
        self.layer_list.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.layer_list.setMaximumHeight(110)
        self.layer_list.currentRowChanged.connect(self.layer_selected)
        self.layer_list.itemChanged.connect(self.layer_visibility)
        box_layers.addWidget(self.layer_list)

        l_row = QHBoxLayout()
        l_row.addWidget(button("+ Layer", self.add_layer))
        l_row.addWidget(button("Rename", self.rename_layer))
        l_row.addWidget(button("Remove", self.remove_layer))
        box_layers.addLayout(l_row)

        form_l = QFormLayout()
        self.color_button = button("Choose color", self.change_color)
        form_l.addRow("Color", self.color_button)
        self.layer_opacity = QSlider(Qt.Orientation.Horizontal)
        self.layer_opacity.setRange(0, 100)
        self.layer_opacity.setValue(85)
        self.layer_opacity.valueChanged.connect(self.layer_appearance)
        form_l.addRow("Opacity", self.layer_opacity)
        self.layer_rep = QComboBox()
        self.layer_rep.addItems(["Shaded Surface", "Wireframe Cage", "Point Cloud"])
        self.layer_rep.currentIndexChanged.connect(self.layer_representation_changed)
        form_l.addRow("Render mode", self.layer_rep)
        self.smooth = QCheckBox("Smooth surface (illustrative)")
        self.smooth.toggled.connect(self.smoothing_changed)
        form_l.addRow(self.smooth)
        box_layers.addLayout(form_l)
        self.layer_info = QLabel("Create a layer to begin selecting tissue.")
        self.layer_info.setWordWrap(True)
        self.layer_info.setStyleSheet("color: #94a3b8; font-size: 11px;")
        box_layers.addWidget(self.layer_info)
        layout_seg.addWidget(grp_layers)

        # Region & intensity
        grp_region = QGroupBox("REGION & THRESHOLD")
        form_reg = QFormLayout(grp_region)
        self.seed_label = QLabel("Click a slice to place a seed.")
        self.seed_label.setWordWrap(True)
        self.seed_label.setStyleSheet("color: #94a3b8; font-size: 11px;")
        form_reg.addRow(self.seed_label)

        self.lower = spin(0)
        self.upper = spin(1000)
        form_reg.addRow("Lower limit", self.lower)
        form_reg.addRow("Upper limit", self.upper)

        btn_fit = QPushButton("⚡ Auto-fit limits around seed (±15%)")
        btn_fit.setStyleSheet("padding: 4px; font-size: 11px; background: #1a2738;")
        btn_fit.clicked.connect(self.auto_fit_seed)
        form_reg.addRow(btn_fit)

        self.merge_mode = QComboBox()
        self.merge_mode.addItems(["Add selection to layer", "Replace layer selection"])
        form_reg.addRow(self.merge_mode)

        act_row = QHBoxLayout()
        act_row.addWidget(button("Grow from seed", lambda: self.process("grow"), True))
        act_row.addWidget(button("Select range", lambda: self.process("threshold")))
        form_reg.addRow(act_row)
        layout_seg.addWidget(grp_region)

        # Refine & Mesh
        grp_refine = QGroupBox("REFINE & EXPORT")
        box_refine = QVBoxLayout(grp_refine)
        ref_row = QHBoxLayout()
        ref_row.addWidget(button("Keep largest", lambda: self.process("island")))
        ref_row.addWidget(button("Fill holes", lambda: self.process("fill")))
        ref_row.addWidget(button("Filter noise", self.filter_noise))
        box_refine.addLayout(ref_row)

        undo_row = QHBoxLayout()
        undo_row.addWidget(button("↶ Undo (Ctrl+Z)", lambda: self.undo(False)))
        undo_row.addWidget(button("↷ Redo (Ctrl+Y)", lambda: self.undo(True)))
        box_refine.addLayout(undo_row)

        box_refine.addWidget(button("✦ Generate / update surface", self.generate, True))
        box_refine.addWidget(button("💾 Export selected surface…", self.export_dialog))
        layout_seg.addWidget(grp_refine)

        scroll_seg.setWidget(content_seg)
        v_seg = QVBoxLayout(tab_seg)
        v_seg.setContentsMargins(0, 0, 0, 0)
        v_seg.addWidget(scroll_seg)
        self.tabs.addTab(tab_seg, "🏷 Segmentation")

        return self.tabs

    def set_layout_mode(self, mode):
        self.layout_mode = mode
        if mode == "quad":
            for w in self.view_widgets:
                w.show()
            self.maximized_widget = None
        elif mode == "axial":
            self._show_only_view(self.panels[0])
        elif mode == "coronal":
            self._show_only_view(self.panels[1])
        elif mode == "sagittal":
            self._show_only_view(self.panels[2])
        elif mode == "3d":
            self._show_only_view(self.scene)
        self.update_layout_buttons()

    def _show_only_view(self, target):
        for w in self.view_widgets:
            w.setVisible(w is target)
        self.maximized_widget = target

    def toggle_maximize(self, target):
        if self.maximized_widget is target:
            self.set_layout_mode("quad")
        else:
            self._show_only_view(target)
            if target is self.panels[0]:
                self.layout_mode = "axial"
            elif target is self.panels[1]:
                self.layout_mode = "coronal"
            elif target is self.panels[2]:
                self.layout_mode = "sagittal"
            elif target is self.scene:
                self.layout_mode = "3d"
            self.update_layout_buttons()

    def cycle_layout(self):
        modes = ["quad", "3d", "axial", "coronal", "sagittal"]
        curr_idx = modes.index(self.layout_mode) if self.layout_mode in modes else 0
        next_mode = modes[(curr_idx + 1) % len(modes)]
        self.set_layout_mode(next_mode)

    def update_layout_buttons(self):
        if hasattr(self, "layout_buttons"):
            for mode, btn in self.layout_buttons.items():
                btn.blockSignals(True)
                btn.setChecked(mode == self.layout_mode)
                btn.blockSignals(False)

    def apply_contrast_preset(self, preset):
        if self.volume is None:
            return
        if preset == "auto":
            self.auto_contrast()
            return
        data = self.volume.data
        d_min = float(data.min())
        d_max = float(data.max())
        d_range = max(1.0, d_max - d_min)
        if preset == "soft":
            window = d_range * 0.4
            level = d_min + d_range * 0.3
        elif preset == "bone":
            window = d_range * 0.75
            level = d_min + d_range * 0.55
        elif preset == "full":
            window = d_range
            level = d_min + d_range / 2
        else:
            return
        self.window_spin.setValue(max(0.001, window))
        self.level_spin.setValue(level)

    def auto_fit_seed(self):
        if self.volume is None or self.seed is None:
            self.error("Click a slice to place a seed point first.")
            return
        val = float(self.volume.data[self.seed[2], self.seed[1], self.seed[0]])
        margin = max(20.0, abs(val) * 0.15)
        self.lower.setValue(max(float(self.volume.data.min()), val - margin))
        self.upper.setValue(min(float(self.volume.data.max()), val + margin))
        self.statusBar().showMessage(f"Bounds fit to seed: [{val - margin:.1f} .. {val + margin:.1f}]")

    def set_tool_index(self, index):
        self.tool.setCurrentIndex(index)

    def tool_index_changed(self, index):
        if hasattr(self, "tool_buttons") and 0 <= index < len(self.tool_buttons):
            self.tool_buttons[index].setChecked(True)
        names = ["Navigate / Seed", "Brush", "Erase"]
        if 0 <= index < len(names):
            self.statusBar().showMessage(f"Active tool: {names[index]}")

    def keyPressEvent(self, event):
        from PySide6.QtWidgets import QLineEdit, QAbstractSpinBox
        focus = QApplication.focusWidget()
        if isinstance(focus, (QLineEdit, QAbstractSpinBox)):
            super().keyPressEvent(event)
            return
        key = event.key()
        if key == Qt.Key.Key_V:
            self.set_tool_index(0)
            event.accept()
            return
        elif key == Qt.Key.Key_B:
            self.set_tool_index(1)
            event.accept()
            return
        elif key == Qt.Key.Key_E:
            self.set_tool_index(2)
            event.accept()
            return
        elif key == Qt.Key.Key_BracketLeft:
            self.radius.setValue(max(0.2, self.radius.value() - 1.0))
            event.accept()
            return
        elif key == Qt.Key.Key_BracketRight:
            self.radius.setValue(min(50.0, self.radius.value() + 1.0))
            event.accept()
            return
        elif key == Qt.Key.Key_Space:
            self.cycle_layout()
            event.accept()
            return
        elif key == Qt.Key.Key_Escape:
            self.set_layout_mode("quad")
            event.accept()
            return
        elif key == Qt.Key.Key_O:
            self.scene.toggle_turntable()
            event.accept()
            return
        elif key == Qt.Key.Key_C:
            self.scene.toggle_axes()
            event.accept()
            return
        super().keyPressEvent(event)

    def start_job(self, fn, success):
        if self.thread is not None:
            return
        self.success_callback = success
        self.thread = QThread(self)
        self.worker = Worker(fn)
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.progress.connect(self.job_progress)
        self.worker.result.connect(self.job_result)
        self.worker.error.connect(self.job_error)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.job_finished)
        self.content.setEnabled(False)
        self.toolbar.setEnabled(False)
        self.progress.setValue(0)
        self.progress.show()
        self.cancel_button.setEnabled(True)
        self.cancel_button.show()
        self.thread.start()

    @Slot(int, str)
    def job_progress(self, value, message):
        self.progress.setValue(value)
        self.statusBar().showMessage(message)

    @Slot(object)
    def job_result(self, result):
        try:
            self.success_callback(result)
        except Exception as exc:
            traceback.print_exc()
            self.error(str(exc))

    @Slot(str)
    def job_error(self, message):
        if message:
            self.error(message)
        else:
            self.statusBar().showMessage("Cancelled · existing work retained")

    @Slot()
    def job_finished(self):
        self.thread.deleteLater()
        self.thread = None
        self.worker = None
        self.success_callback = None
        self.content.setEnabled(True)
        self.toolbar.setEnabled(True)
        self.progress.hide()
        self.cancel_button.hide()

    def cancel_job(self):
        if self.worker is not None:
            self.worker.job.cancelled.set()
            self.cancel_button.setEnabled(False)
            self.statusBar().showMessage("Cancelling at the next safe processing point…")

    def error(self, text):
        QMessageBox.warning(self, "MRI3D", text)

    def can_replace(self):
        if self.thread is not None:
            return False
        if not self.dirty:
            return True
        answer = QMessageBox.question(self, "Unsaved project", "This project has unsaved changes.\nSave it first to keep them, or discard them to continue.",
                                      QMessageBox.StandardButton.Discard | QMessageBox.StandardButton.Cancel,
                                      QMessageBox.StandardButton.Cancel)
        return answer == QMessageBox.StandardButton.Discard

    def mark_dirty(self):
        self.dirty = True
        name = Path(self.project_path).stem if self.project_path else "Untitled"
        self.setWindowTitle(f"MRI3D · {name} *")

    def import_folder(self):
        if self.thread is not None:
            return
        folder = QFileDialog.getExistingDirectory(self, "Import MRI DICOM folder", self.folder)
        if folder:
            self.scan(folder)

    def scan(self, folder):
        self.folder = str(folder)
        self.start_job(lambda job: scan_folder(folder, job), self.show_scan)

    def show_scan(self, scan):
        self.tree.clear()
        for series in scan.series:
            spacing = f"{series.pixel_spacing[1]:.3g} × {series.pixel_spacing[0]:.3g} × {series.spacing_z:.3g}"
            item = QTreeWidgetItem([series.description, str(len(series.slices)), spacing])
            item.setData(0, Qt.ItemDataRole.UserRole, series)
            item.setToolTip(0, "\n".join(series.issues) if series.issues else f"{series.columns} × {series.rows} pixels · thickness {series.thickness:g} mm")
            if not series.usable:
                item.setForeground(0, QColor("#c49b76"))
                item.setText(0, "[Excluded] " + series.description)
            self.tree.addTopLevelItem(item)
        usable = sum(s.usable for s in scan.series)
        self.series_info.setText(f"{len(scan.series)} groups · {usable} reconstructable · {scan.ignored} skipped")
        self.statusBar().showMessage(f"Import complete · {len(scan.series)} groups · {usable} reconstructable")
        if scan.warnings:
            self.error("Some files were skipped:\n" + "\n".join(scan.warnings))
        if not scan.series:
            self.error("No supported MRI series found. Choose a folder containing single-frame MRI DICOM files.")

    def selected_series_list(self):
        items = self.tree.selectedItems()
        result = []
        for it in items:
            s = it.data(0, Qt.ItemDataRole.UserRole)
            if s and s not in result:
                result.append(s)
        if not result:
            curr = self.tree.currentItem()
            if curr:
                s = curr.data(0, Qt.ItemDataRole.UserRole)
                if s:
                    result.append(s)
        return result

    def series_selected(self):
        series_list = self.selected_series_list()
        if not series_list:
            self.load_button.setEnabled(False)
            self.load_button.setText("Open selected series")
            return
        if len(series_list) == 1:
            series = series_list[0]
            self.load_button.setEnabled(series.usable)
            self.load_button.setText("Open selected series")
            self.series_info.setText("; ".join(series.issues) if series.issues else
                                     f"{series.columns} × {series.rows} × {len(series.slices)} · slice spacing {series.spacing_z:.2f} mm · thickness {series.thickness:g} mm")
        else:
            usable = [s for s in series_list if s.usable]
            all_usable = len(usable) == len(series_list)
            self.load_button.setEnabled(all_usable and len(usable) >= 2)
            self.load_button.setText(f"⚡ Fuse {len(series_list)} orthogonal planes into Isotropic 3D")
            labels = [s.description.split(':')[-1].strip() for s in series_list]
            self.series_info.setText(f"Selected {len(series_list)} series for super-resolution fusion:\n" + ", ".join(labels))

    def load_selected(self):
        series_list = self.selected_series_list()
        if not series_list or not self.can_replace():
            return
        if any(not s.usable for s in series_list):
            self.error("One or more selected series cannot be reconstructed.")
            return
        if len(series_list) == 1:
            self.start_job(lambda job: load_series(series_list[0], job), self.set_volume)
        else:
            self.start_job(lambda job: fuse_series(series_list, job), self.set_volume)

    def auto_select_orthogonal(self):
        current = self.tree.currentItem()
        if current is None:
            for i in range(self.tree.topLevelItemCount()):
                item = self.tree.topLevelItem(i)
                s = item.data(0, Qt.ItemDataRole.UserRole)
                if s and s.usable:
                    current = item
                    break
        if current is None:
            self.error("Import a scan folder first.")
            return
        curr_series = current.data(0, Qt.ItemDataRole.UserRole)
        desc_lower = curr_series.description.lower()
        key_tokens = [tok for tok in desc_lower.replace(":", " ").split() if tok not in ("axi", "cor", "sag", "axial", "coronal", "sagittal")]
        
        matches = []
        for i in range(self.tree.topLevelItemCount()):
            item = self.tree.topLevelItem(i)
            s = item.data(0, Qt.ItemDataRole.UserRole)
            if not s or not s.usable:
                continue
            s_desc = s.description.lower()
            if all(tok in s_desc for tok in key_tokens) or (":" in curr_series.description and s.description.startswith(curr_series.description.split(":")[0])):
                matches.append(item)
                
        if len(matches) < 2:
            self.statusBar().showMessage("Could not automatically find orthogonal partner series for this scan.")
            return
            
        self.tree.clearSelection()
        for it in matches[:3]:
            it.setSelected(True)
        self.series_selected()
        self.statusBar().showMessage(f"Selected {len(self.tree.selectedItems())} matching orthogonal series. Click 'Fuse' to combine.")

    def set_volume(self, volume, layers=None, views=None):
        self.updating = True
        self.volume = volume
        self.layers = layers if layers is not None else []
        self.xyz = [n // 2 for n in volume.image.GetSize()]
        self.seed = None
        self.seed_label.setText("Click a slice to place a seed.")
        self.scene.set_volume(volume)
        self.invert.setChecked(volume.invert)
        self.auto_contrast()
        self.lower.setValue(self.level_spin.value())
        self.upper.setValue(float(volume.data.max()))
        self.clip.setChecked(False)
        self.tool.setCurrentIndex(0)
        if views:
            self.restore_views(views)
        self.updating = False
        self.refresh_layer_list()
        for panel in self.panels:
            panel.canvas.reset_view()
        self.refresh()
        self.display_changed()
        self.clip_changed()
        self.project_path = None
        self.dirty = False
        self.setWindowTitle("MRI3D · " + volume.series.description)
        self.banner.setText(f"{volume.series.description}  ·  {len(volume.series.slices)} source slices  ·  "
                            f"{volume.series.spacing_z:.2f} mm slice spacing  ·  LPS orientation\n"
                            "Separate tissue layers are user selections. Interpolated views and smoothed surfaces do not add scan detail.")
        if hasattr(self, "series_chip"):
            self.series_chip.setText(f"📁 {volume.series.description} ({len(volume.series.slices)} slices)")
            self.series_chip.setStyleSheet("background: #112638; border: 1px solid #0284c7; color: #38bdf8; padding: 3px 10px; border-radius: 12px; font-weight: bold; font-size: 11px;")
        if hasattr(self, "tabs"):
            self.tabs.setCurrentIndex(1)
        self.statusBar().showMessage("Volume ready · click a slice to navigate or place a seed")

    def refresh(self):
        if self.volume is None:
            return
        for panel in self.panels:
            panel.refresh(self.volume, self.xyz, self.layers, self.level_spin.value(), self.window_spin.value(), self.invert.isChecked())
        if self.clip.isChecked():
            self.update_clipping(render=False)
        else:
            self.update_clip_label()
        self.scene.set_cursor(self.xyz, render=True)

    def set_index(self, axis, value):
        if self.volume is None or self.thread is not None:
            return
        self.xyz[axis] = int(value)
        self.refresh()

    def step(self, axis, direction):
        if self.volume is not None:
            self.set_index(axis, int(np.clip(self.xyz[axis] + direction, 0, self.volume.image.GetSize()[axis] - 1)))

    def pick(self, axis, xyz, first):
        if self.volume is None or self.thread is not None:
            return
        tool = self.tool.currentIndex()
        layer = self.current_layer()
        if tool and layer is None:
            if first:
                self.error("Add a tissue layer before painting.")
            return
        if tool:
            if first:
                layer.remember()
                self.stroke_previous = None
            # Interpolate pointer motion to avoid gaps during fast strokes.
            previous = self.stroke_previous if self.stroke_previous is not None else xyz
            steps = max(1, max(abs(a - b) for a, b in zip(previous, xyz)))
            for t in np.linspace(0, 1, steps + 1):
                point = tuple(np.rint(np.array(previous) * (1 - t) + np.array(xyz) * t).astype(int))
                paint_mask(layer.mask, axis, point, self.radius.value(), self.volume.spacing, tool == 1)
            self.stroke_previous = xyz
            layer.mesh = None
            self.mark_dirty()
        else:
            self.seed = tuple(xyz)
            value = self.volume.data[xyz[2], xyz[1], xyz[0]]
            world = self.volume.world(xyz)
            self.seed_label.setText(f"Seed: {value:.1f} intensity\nLPS mm: {world[0]:.1f}, {world[1]:.1f}, {world[2]:.1f}")
        self.xyz = list(xyz)
        self.refresh()

    def end_stroke(self):
        self.stroke_previous = None
        self.scene.sync_layers(self.layers)
        self.layer_selected()

    def current_layer(self):
        index = self.layer_list.currentRow()
        return self.layers[index] if 0 <= index < len(self.layers) else None

    def refresh_layer_list(self, select=None):
        old = self.layer_list.currentRow() if select is None else select
        self.layer_list.blockSignals(True)
        self.layer_list.clear()
        for layer in self.layers:
            item = QListWidgetItem(layer.name)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Checked if layer.visible else Qt.CheckState.Unchecked)
            item.setForeground(QColor(*layer.color))
            self.layer_list.addItem(item)
        if self.layers:
            self.layer_list.setCurrentRow(max(0, min(old, len(self.layers) - 1)))
        self.layer_list.blockSignals(False)
        self.layer_selected()

    def layer_selected(self, *_):
        layer = self.current_layer()
        if layer is None:
            self.layer_info.setText("Create a layer to begin selecting tissue.")
            return
        self.layer_opacity.blockSignals(True)
        self.smooth.blockSignals(True)
        self.layer_opacity.setValue(round(layer.opacity * 100))
        self.smooth.setChecked(layer.smooth)
        self.layer_opacity.blockSignals(False)
        self.smooth.blockSignals(False)
        if hasattr(self, "layer_rep"):
            rep_map = {"surface": 0, "wireframe": 1, "points": 2}
            self.layer_rep.blockSignals(True)
            self.layer_rep.setCurrentIndex(rep_map.get(getattr(layer, "representation", "surface"), 0))
            self.layer_rep.blockSignals(False)
        self.color_button.setStyleSheet(f"color: rgb{layer.color};")
        count = int(np.count_nonzero(layer.mask))
        status = f"{layer.mesh.GetNumberOfCells():,} surface faces" if layer.mesh is not None else "surface not generated / needs update"
        ml = count * float(np.prod(self.volume.spacing)) / 1000 if self.volume else 0
        self.layer_info.setText(f"{count:,} selected voxels · {ml:.2f} mL\n{status}")

    def layer_representation_changed(self, index):
        layer = self.current_layer()
        if layer is None:
            return
        modes = ["surface", "wireframe", "points"]
        if 0 <= index < len(modes):
            layer.representation = modes[index]
            self.scene.sync_layers(self.layers)
            self.mark_dirty()

    def filter_noise(self):
        layer = self.current_layer()
        if layer is None:
            self.error("Select a tissue layer first.")
            return
        layer.remember()
        layer.mask = filter_clusters(layer.mask, min_voxels=25)
        layer.mesh = None
        self.mark_dirty()
        self.refresh()
        self.scene.sync_layers(self.layers)
        self.layer_selected()
        self.statusBar().showMessage("Filtered small noise clusters (< 25 voxels) · update surface when ready")

    def add_layer(self):
        if self.volume is None:
            self.error("Open a scan series first.")
            return
        name, ok = QInputDialog.getText(self, "New tissue layer", "Layer name", text="Bone" if not self.layers else "Tissue " + str(len(self.layers) + 1))
        if ok and name.strip():
            self.layers.append(Layer(name.strip(), COLORS[len(self.layers) % len(COLORS)], np.zeros(self.volume.data.shape, dtype=bool)))
            self.refresh_layer_list(len(self.layers) - 1)
            self.mark_dirty()

    def rename_layer(self):
        layer = self.current_layer()
        if layer:
            name, ok = QInputDialog.getText(self, "Rename layer", "Layer name", text=layer.name)
            if ok and name.strip():
                layer.name = name.strip()
                self.refresh_layer_list()
                self.mark_dirty()

    def remove_layer(self):
        layer = self.current_layer()
        if layer and QMessageBox.question(self, "Remove layer", f"Remove '{layer.name}' and its selection?", QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No, QMessageBox.StandardButton.No) == QMessageBox.StandardButton.Yes:
            del self.layers[self.layer_list.currentRow()]
            self.refresh_layer_list()
            self.scene.sync_layers(self.layers)
            self.refresh()
            self.mark_dirty()

    def change_color(self):
        layer = self.current_layer()
        if layer:
            color = QColorDialog.getColor(QColor(*layer.color), self, "Layer color")
            if color.isValid():
                layer.color = (color.red(), color.green(), color.blue())
                self.refresh_layer_list()
                self.layer_appearance()

    def layer_visibility(self, item):
        index = self.layer_list.row(item)
        self.layers[index].visible = item.checkState() == Qt.CheckState.Checked
        self.scene.sync_layers(self.layers)
        self.refresh()
        self.mark_dirty()

    def layer_appearance(self, *_):
        layer = self.current_layer()
        if layer:
            layer.opacity = self.layer_opacity.value() / 100
            self.scene.sync_layers(self.layers)
            self.refresh()
            self.mark_dirty()

    def smoothing_changed(self, checked):
        layer = self.current_layer()
        if layer:
            layer.smooth = checked
            layer.mesh = None
            self.scene.sync_layers(self.layers)
            self.layer_selected()
            self.mark_dirty()

    def process(self, operation):
        layer = self.current_layer()
        if layer is None or self.volume is None:
            self.error("Open a series and add a tissue layer first.")
            return
        lower, upper, seed = self.lower.value(), self.upper.value(), self.seed
        merge = self.merge_mode.currentIndex() == 0 and operation in ("grow", "threshold")
        def apply(mask):
            layer.remember()
            layer.mask = layer.mask | mask if merge else mask
            self.scene.sync_layers(self.layers)
            self.layer_selected()
            self.refresh()
            self.mark_dirty()
            self.statusBar().showMessage("Selection updated · review the overlays before generating a surface")
        self.start_job(lambda job: segment(self.volume, operation, layer.mask, lower, upper, seed, job), apply)

    def undo(self, redo=False):
        if self.thread is not None:
            return
        layer = self.current_layer()
        if layer and layer.restore(redo):
            self.scene.sync_layers(self.layers)
            self.layer_selected()
            self.refresh()
            self.mark_dirty()

    def generate(self):
        layer = self.current_layer()
        if layer is None or self.volume is None:
            self.error("Add and mark a tissue layer first.")
            return
        def apply(mesh):
            layer.mesh = mesh
            self.scene.sync_layers(self.layers)
            self.layer_selected()
            self.statusBar().showMessage("Surface ready · export in millimeters or keep editing the layer")
        self.start_job(lambda job: make_mesh(self.volume, layer.mask, layer.smooth, job), apply)

    def auto_contrast(self):
        if self.volume is None:
            return
        lo, hi = np.percentile(self.volume.data[::2, ::2, ::2], [2, 99.5])
        self.window_spin.blockSignals(True)
        self.level_spin.blockSignals(True)
        self.window_spin.setValue(max(1, hi - lo))
        self.level_spin.setValue((hi + lo) / 2)
        self.window_spin.blockSignals(False)
        self.level_spin.blockSignals(False)
        self.display_changed()

    def display_changed(self, *_):
        if self.updating or self.volume is None:
            return
        self.refresh()
        self.scene.appearance(self.level_spin.value(), self.window_spin.value(), self.volume_opacity.value() / 100,
                              self.show_volume.isChecked(), self.invert.isChecked())
        self.mark_dirty()

    def clip_position_changed(self, value):
        if self.volume is not None and not self.updating:
            axis = self.clip_axis.currentIndex()
            size = max(1, self.volume.image.GetSize()[axis] - 1)
            self.set_index(axis, int(round(value / 100 * size)))

    def update_clip_label(self):
        if self.volume is None:
            self.clip_label.setText("—")
            self.clip_label.setStyleSheet("color: #91a4b9;")
            return
        axis = self.clip_axis.currentIndex()
        size = self.volume.image.GetSize()[axis]
        curr = self.xyz[axis] + 1
        accent = ACCENTS.get(axis, "#91a4b9")
        status = f"Slice {curr} / {size}"
        if not self.clip.isChecked():
            status += " (off)"
            self.clip_label.setStyleSheet("color: #91a4b9;")
        else:
            self.clip_label.setStyleSheet(f"color: {accent}; font-weight: 600;")
        self.clip_label.setText(status)

    def update_clipping(self, render=True):
        if self.volume is None:
            return
        axis = self.clip_axis.currentIndex()
        size = max(1, self.volume.image.GetSize()[axis] - 1)
        fraction = float(self.xyz[axis]) / size
        self.clip_position.blockSignals(True)
        self.clip_position.setValue(int(round(fraction * 100)))
        self.clip_position.blockSignals(False)
        self.update_clip_label()
        self.scene.clipping(self.clip.isChecked(), axis, fraction, self.clip_flip.isChecked(), render=render)

    def clip_changed(self, *_):
        if self.updating or self.volume is None:
            return
        self.update_clipping(render=True)
        self.mark_dirty()

    def reset_views(self):
        for panel in self.panels:
            panel.canvas.reset_view()
        self.scene.reset_camera()

    def view_state(self):
        camera = self.scene.renderer.GetActiveCamera()
        return dict(xyz=self.xyz, level=self.level_spin.value(), window=self.window_spin.value(), invert=self.invert.isChecked(),
                    volume_opacity=self.volume_opacity.value(), show_volume=self.show_volume.isChecked(),
                    clip=self.clip.isChecked(), clip_axis=self.clip_axis.currentIndex(), clip_position=self.clip_position.value(),
                    clip_flip=self.clip_flip.isChecked(), lower=self.lower.value(), upper=self.upper.value(),
                    camera=dict(position=camera.GetPosition(), focal=camera.GetFocalPoint(), up=camera.GetViewUp(),
                                scale=camera.GetParallelScale(), parallel=bool(camera.GetParallelProjection())))

    def restore_views(self, views):
        self.xyz = [int(np.clip(v, 0, n - 1)) for v, n in zip(views.get("xyz", self.xyz), self.volume.image.GetSize())]
        if len(self.xyz) != 3:
            self.xyz = [n // 2 for n in self.volume.image.GetSize()]
        for key, control in [("level", self.level_spin), ("window", self.window_spin), ("volume_opacity", self.volume_opacity),
                             ("clip_position", self.clip_position), ("lower", self.lower), ("upper", self.upper)]:
            if key in views:
                control.setValue(views[key])
        for key, control in [("invert", self.invert), ("show_volume", self.show_volume), ("clip", self.clip), ("clip_flip", self.clip_flip)]:
            if key in views:
                control.setChecked(views[key])
        self.clip_axis.setCurrentIndex(int(np.clip(views.get("clip_axis", 0), 0, 2)))
        if "camera" in views:
            values = views["camera"]
            camera = self.scene.renderer.GetActiveCamera()
            camera.SetPosition(*values["position"])
            camera.SetFocalPoint(*values["focal"])
            camera.SetViewUp(*values["up"])
            camera.SetParallelScale(values["scale"])
            camera.SetParallelProjection(values["parallel"])
            self.scene.renderer.ResetCameraClippingRange()

    def save_dialog(self, _checked=False, save_as=False):
        if self.volume is None:
            self.error("Open a scan series first.")
            return
        if self.thread is not None:
            return
        path = self.project_path
        if path is None or save_as:
            path, _ = QFileDialog.getSaveFileName(self, "Save MRI3D project", str(Path.home() / "Documents" / "My MRI.mri3d"), "MRI3D project (*.mri3d)")
        if not path:
            return
        if not path.lower().endswith(".mri3d"):
            path += ".mri3d"
        state = self.view_state()
        def saved(result):
            self.project_path = result
            self.dirty = False
            self.setWindowTitle("MRI3D · " + Path(result).stem)
            self.statusBar().showMessage("Project saved · keep its companion -data folder and source DICOM scans")
        self.start_job(lambda job: save_project(path, self.volume, self.layers, state, job), saved)

    def open_dialog(self):
        if not self.can_replace():
            return
        path, _ = QFileDialog.getOpenFileName(self, "Open MRI3D project", str(Path.home() / "Documents"), "MRI3D project (*.mri3d)")
        if path:
            self.load_project(path)

    def load_project(self, path):
        try:
            manifest = read_manifest(path)
            relink = None
            if missing_sources(manifest):
                relink = QFileDialog.getExistingDirectory(self, "Source scans moved — select their new DICOM folder", self.folder)
                if not relink:
                    return
            def loaded(result):
                volume, layers, views = result
                self.set_volume(volume, layers, views)
                self.project_path = str(path)
                self.setWindowTitle("MRI3D · " + Path(path).stem)
                if relink:
                    self.mark_dirty()
                self.statusBar().showMessage("Project opened · masks restored; generate surfaces when needed")
            self.start_job(lambda job: open_project(path, relink, job), loaded)
        except Exception as exc:
            self.error(str(exc))

    def export_dialog(self):
        layer = self.current_layer()
        if self.thread is not None:
            return
        if layer is None or layer.mesh is None:
            self.error("Select a layer and generate its surface first.")
            return
        path, selected = QFileDialog.getSaveFileName(self, "Export surface — millimeters / DICOM LPS", str(Path.home() / "Documents" / "MRI-model.ply"),
                                                    "Colored PLY (*.ply);;STL (*.stl)")
        if not path:
            return
        if Path(path).suffix.lower() not in (".stl", ".ply"):
            path += ".stl" if "STL" in selected else ".ply"
        def export(job):
            job.progress(10, "Exporting surface")
            export_mesh(layer.mesh, path, layer.color)
            job.report(100, "Surface exported")
            return path
        self.start_job(export, lambda result: self.statusBar().showMessage("Exported " + result + " · units: mm; coordinate metadata in companion JSON"))

    def guide(self):
        ShortcutsDialog(self).exec()

    def closeEvent(self, event):
        if self.thread is not None:
            self.cancel_job()
            self.statusBar().showMessage("Cancelling processing… close the window again once it finishes.")
            event.ignore()
            return
        if not self.can_replace():
            event.ignore()
            return
        self.scene.close()
        event.accept()


def main():
    parser = argparse.ArgumentParser(description="MRI3D local MRI explorer")
    parser.add_argument("--folder", help="Scan a DICOM folder on launch")
    parser.add_argument("--project", help="Open a saved MRI3D project")
    parser.add_argument("--smoke-test", metavar="OUTPUT", help="Run an automated GUI smoke test and write its results")
    args = parser.parse_args()
    app = QApplication(sys.argv[:1])
    app.setApplicationName("MRI3D")
    app.setStyle("Fusion")
    app.setStyleSheet(STYLE)
    window = MainWindow()
    window.show()
    if args.smoke_test:
        from .smoke import run_smoke
        QTimer.singleShot(500, lambda: run_smoke(window, args.smoke_test, args.folder))
    elif args.project:
        QTimer.singleShot(100, lambda: window.load_project(args.project))
    elif args.folder:
        QTimer.singleShot(100, lambda: window.scan(args.folder))
    return app.exec()
