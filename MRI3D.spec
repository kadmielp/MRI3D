# Build a self-contained Windows folder: dist/MRI3D/MRI3D.exe.
from PyInstaller.utils.hooks import collect_submodules, copy_metadata

datas = []
for package in ['pydicom', 'pylibjpeg', 'pylibjpeg-libjpeg', 'pylibjpeg-openjpeg', 'numpy', 'vtk', 'SimpleITK', 'PySide6', 'shiboken6']:
    datas += copy_metadata(package)
datas += [('README.md', '.'), ('THIRD_PARTY_NOTICES.md', '.')]
hidden = collect_submodules('pydicom.pixels') + collect_submodules('pylibjpeg')
hidden += collect_submodules('libjpeg') + collect_submodules('openjpeg')
hidden += ['vtkmodules.util.data_model', 'vtkmodules.util.execution_model']

a = Analysis(['main.py'], pathex=[], binaries=[], datas=datas, hiddenimports=hidden,
             hookspath=[], hooksconfig={}, runtime_hooks=[],
             excludes=['tkinter', 'matplotlib', 'pytest', 'PySide6.QtWebEngineCore', 'PySide6.QtWebEngineWidgets'],
             noarchive=False)
pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name='MRI3D', debug=False,
          bootloader_ignore_signals=False, strip=False, upx=False, console=False,
          disable_windowed_traceback=False)
coll = COLLECT(exe, a.binaries, a.datas, strip=False, upx=False, name='MRI3D')
