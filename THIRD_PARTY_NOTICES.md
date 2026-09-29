# Third-party notices

MRI3D uses independently distributed open-source components. Their license files and package metadata are retained in the packaged `_internal` directory where provided by their wheels.

| Component | Upstream | License family |
| --- | --- | --- |
| Python | https://www.python.org/ | PSF |
| PySide6 / Shiboken6 / Qt | https://www.qt.io/qt-for-python | LGPLv3 / GPLv3 / commercial, according to component |
| NumPy | https://numpy.org/ | BSD |
| pydicom | https://pydicom.github.io/ | MIT |
| SimpleITK | https://simpleitk.org/ | Apache 2.0 |
| VTK | https://vtk.org/ | BSD |
| pylibjpeg, libjpeg plugin, OpenJPEG plugin | https://github.com/pydicom/pylibjpeg | MIT and bundled codec licenses |
| PyInstaller (packaging) | https://pyinstaller.org/ | GPL with bootloader exception |

PySide6 and Qt are dynamically linked in the one-folder distribution. The build specification and Python source are included in this repository so the application can be rebuilt against replacement compatible libraries. Retain upstream notices when redistributing. This file is an index, not a substitute for the full licenses shipped by those components.
