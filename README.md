# MRI3D

A local Windows MRI explorer with linked anatomical slices, volume rendering, user-guided tissue segmentation and physical-scale mesh export. Version 1.0.0 is for personal exploration.

## Run the Windows app

Open **`dist\MRI3D\MRI3D.exe`**. Keep the entire `MRI3D` folder together: its `_internal` folder contains the Python runtime and imaging dependencies. No separate Python installation, account, server or network connection is required. The executable is unsigned.

Choose **Import DICOM**, select a scan folder, select a compatible series and click **Open selected series**. The folder picker starts at `G:\Vibing\MRI3D\DICOMIMAGES`. It does not import or modify source files automatically.

## Explore and create a model

1. Adjust **Window** and **Level**, or use **Auto contrast**. Scroll over a slice to move through it. Ctrl + wheel zooms; right-drag pans. Clicking a slice moves all crosshairs and places a seed.
2. Rotate the 3D view by dragging; use the wheel to zoom. Volume opacity and clipping help reveal structures. This initial view represents MRI intensity, not automatically identified anatomy.
3. Scroll down the left panel to **Tissue layers**. Add a layer such as Bone, Muscle or a descriptive label. Each layer has its own color, visibility and opacity.
4. Click **Navigate / seed**, place a seed inside a region, read its intensity, and set lower and upper intensity limits. **Grow from seed** selects a connected region; **Select threshold range** selects matching voxels throughout the volume. MRI intensities have no universal tissue thresholds. Review the result in all planes.
5. Use **Brush** or **Erase** to refine the selection. Brush radius is measured in millimeters and applies to the current displayed slice. **Keep largest** removes disconnected components; **Fill holes** fills enclosed 3D cavities. Undo / redo retains up to 20 changes per layer.
6. **Generate / update surface** creates a mesh. Changing the mask or smoothing option invalidates its old surface. Smoothing affects the surface, never the saved mask. Rebuild before exporting.
7. **Save project** stores source references, masks, colors and display settings. **Export model** exports the selected generated surface as colored PLY or binary STL. Exports use DICOM LPS coordinates in millimeters. Clipping changes the display only; exported meshes remain complete.

**LPS axes:** +X points toward the patient's left, +Y posterior, +Z superior. Slice orientation letters describe the patient, not the screen. Exports include a companion JSON file documenting units, coordinates and bounds; STL itself does not encode units.

## Layouts, 3D tools and fusion

- **Layouts:** switch between Quad, 3D and single-plane layouts from the header, press Space to cycle, Esc to return to Quad, or double-click a view to maximize it. F1 lists all shortcuts.
- **3D view:** Top / Front / Side presets, **Orbit** (O) for a continuous turntable, **Scale** (C) for millimeter LPS cube axes, and **3D** for red/cyan anaglyph stereo. Each layer can be drawn as a shaded surface, wireframe or point cloud; the choice is saved in the project.
- **Contrast presets:** Auto, Soft Tissue, Bone and Full are starting points derived from the volume's intensity range, not calibrated tissue windows.
- **Segmentation helpers:** **Auto-fit limits** sets lower / upper limits to ±15% (at least ±20) around the seed intensity. **Filter noise** removes disconnected components smaller than 25 voxels.
- **Fusion:** select two or three orthogonal series (Ctrl-click, or **Auto-select orthogonal trio**) and click **Fuse**. Each series is resampled onto a shared isotropic LPS grid (0.25–0.6 mm, capped at 40 million voxels) and blended with gradient-weighted averaging (sharper in-plane detail counts more) after per-series intensity normalization. The result looks sharper than any single stack, but it is still interpolation and not acquired resolution; series from different sequences or with patient motion between them will not align. Fused projects store their component series and rebuild the fused volume when reopened.

## Projects and backups

A project consists of a `.mri3d` JSON manifest and its neighboring `.mri3d-data` folder containing compressed `.mha` masks. Keep both, along with the original DICOM scans. Saving commits the manifest atomically only after all new masks have been written. Previous mask generations are retained so interrupted saves cannot destroy the last saved project; repeated saves therefore consume additional disk space. A Save As to a new filename creates a compact current copy.

If source scans move, opening the project asks for their new folder. Relinking requires the same series and SOP instance identifiers and checks geometry before restoring masks. Save the project to keep updated source paths. Masks are tied to their source series; loading a different sequence does not carry labels across. Meshes are regenerated from masks after reopening, so a stale surface cannot be reused accidentally.

No scans, screenshots or projects are sent anywhere. Project files contain local paths and DICOM identifiers and should be treated as private. Test artifacts using your scans are excluded from Git and the application package.

## Supported inputs and limits

- Single-frame monochrome MRI DICOM stacks, including the supplied older implicit-VR files without a Part 10 preamble. JPEG / JPEG-LS and JPEG 2000 decoder plugins are bundled; availability also depends on the precise codec variant.
- Grouping uses series UID, orientation, dimensions, pixel spacing, echo, temporal position and image component. Slice order comes from physical positions. Scout/localizer groups are excluded; duplicate positions, nonuniform slice spacing and in-plane shifts are rejected with an explanation.
- Oblique stacks are resampled into a patient-aligned LPS working grid using linear interpolation. Original geometry is retained in projects. Label masks use the working grid. A 160-million-voxel safety limit applies to source and resampled volumes.
- Enhanced multiframe MRI, CT, registration between series (fusion assumes they are already aligned), anatomical AI, DICOM SEG export, PACS and clinical validation are outside this version.
- The supplied 320 files produce 18 geometry groups: 15 reconstructable sequences and three localizer orientations. Reconstructable slice spacing is approximately 2.7–4.6 mm. The gaps and thickness limit surface fidelity; interpolation is not recovered detail.
- Dark MRI regions near metal are not reliable screw geometry. User labels should describe visible evidence without asserting accurate implant shape. This app does not diagnose conditions or automatically reconstruct screws.
- Long operations run in workers. Cancel takes effect at the next safe processing point; some native decoder, file-write and mesh stages finish their current stage first. An already committed save or export reports success rather than pretending it was cancelled.

## Development

Use Python 3.11 x64 on Windows. From this directory:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\run.ps1
.\.venv\Scripts\python.exe -m pytest -q
.\build.ps1
```

Command-line entry points work from source or the packaged executable:

```powershell
.\dist\MRI3D\MRI3D.exe --folder 'G:\Vibing\MRI3D\DICOMIMAGES'
.\dist\MRI3D\MRI3D.exe --project 'C:\path\My MRI.mri3d'
.\dist\MRI3D\MRI3D.exe --folder 'G:\Vibing\MRI3D\DICOMIMAGES' --smoke-test artifacts\packaged-smoke
```

The smoke-test option creates a clearly labeled technical selection, exercises editing / undo, linked views, clipping, surface export, project roundtrip and rendering, writes screenshots and a JSON report, then exits. It does not claim its test selection is anatomical. Omitting `--folder` uses a synthetic volume and skips DICOM project roundtrip.

## Code organization

- `mri3d/imaging.py`: DICOM validation, reconstruction, LPS geometry and segmentation.
- `mri3d/models.py`, `projects.py`: layer history, surface generation, exports and atomic project saves.
- `mri3d/views.py`, `app.py`: slice canvases, VTK rendering, desktop workflow and background jobs.
- `tests/`: synthetic geometry, failure handling, segmentation, export and persistence checks.

## Third-party components

See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). Dependency versions are pinned in `requirements.txt`; build dependencies are pinned in `requirements-dev.txt`.
