from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import json
import os
import uuid
import numpy as np
import SimpleITK as sitk
from .imaging import Series, Slice, Volume, Job, ImagingError, geometry, mask_image, load_series, fuse_series, scan_folder
from .models import Layer


def save_project(path, volume: Volume, layers: list[Layer], views: dict, job: Job | None = None):
    job = job or Job()
    path = Path(path).resolve()
    # A new generation keeps the old project usable even if saving is interrupted.
    generation = uuid.uuid4().hex
    assets = path.parent / (path.name + "-data") / generation
    assets.mkdir(parents=True, exist_ok=False)
    items = []
    for i, layer in enumerate(layers):
        job.progress(int(i * 85 / max(1, len(layers))), f"Saving layer {i + 1}/{len(layers)}")
        filename = f"{i}.mha"
        sitk.WriteImage(mask_image(layer.mask, volume), str(assets / filename), True)
        items.append(dict(id=layer.id, name=layer.name, color=list(layer.color), visible=layer.visible,
                          opacity=layer.opacity, smooth=layer.smooth,
                          representation=getattr(layer, "representation", "surface"),
                          mask=str((assets / filename).relative_to(path.parent))))
    manifest = dict(format="MRI3D", version=1, series=asdict(volume.series),
                    source_geometry=volume.source_geometry, working_geometry=geometry(volume.image),
                    layers=items, views=views, coordinates="LPS", units="mm")
    if volume.components:
        manifest["components"] = [asdict(s) for s in volume.components]
    temp = path.with_name(path.name + "." + generation + ".tmp")
    job.check()
    temp.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    job.check()
    os.replace(temp, path)
    # Once committed, report success even if cancellation arrives at this instant.
    job.report(100, "Project saved")
    return str(path)


def read_manifest(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data.get("format") != "MRI3D" or data.get("version") != 1:
        raise ImagingError("Unsupported project format or version")
    return data


def source_series(data):
    # Fused projects are rebuilt from their component series.
    return data.get("components") or [data["series"]]


def missing_sources(data):
    return any(not Path(s["path"]).is_file() for series in source_series(data) for s in series["slices"])


def series_from_dict(item):
    fields = dict(item)
    fields["slices"] = [Slice(**s) for s in fields["slices"]]
    return Series(**fields)


def open_project(path, relink=None, job: Job | None = None):
    job = job or Job()
    manifest = read_manifest(path)
    fused = bool(manifest.get("components"))
    series_list = [series_from_dict(item) for item in source_series(manifest)]
    if relink:
        found = scan_folder(relink, job)
        matched = []
        for series in series_list:
            match = next((s for s in found.series if s.key == series.key and
                          {f.sop_uid for f in s.slices} == {f.sop_uid for f in series.slices}), None)
            if match is None:
                raise ImagingError("That folder does not contain the same series and source images")
            matched.append(match)
        series_list = matched
    if any(not Path(s.path).is_file() for series in series_list for s in series.slices):
        raise ImagingError("Source scans have moved. Use Open project and relink the DICOM folder.")
    volume = fuse_series(series_list, job) if fused else load_series(series_list[0], job)
    current = geometry(volume.image)
    for key in ("size", "origin", "direction", "spacing"):
        if not np.allclose(current[key], manifest["working_geometry"][key], atol=1e-5, rtol=1e-6):
            raise ImagingError("Source geometry differs from the saved project; masks cannot be aligned")
    layers = []
    root = Path(path).resolve().parent
    for item in manifest["layers"]:
        job.check()
        mask_path = (root / item["mask"]).resolve()
        if not mask_path.is_relative_to(root):
            raise ImagingError("Invalid project mask path")
        image = sitk.ReadImage(str(mask_path))
        if image.GetDimension() != 3 or image.GetNumberOfComponentsPerPixel() != 1:
            raise ImagingError("Invalid segmentation image")
        for key, value in geometry(image).items():
            if not np.allclose(value, current[key], atol=1e-5, rtol=1e-6):
                raise ImagingError("Saved mask geometry does not match the source volume")
        color = tuple(int(c) for c in item["color"])
        if len(color) != 3 or any(c < 0 or c > 255 for c in color):
            raise ImagingError("Invalid layer color")
        layer = Layer(item["name"], color, sitk.GetArrayFromImage(image) > 0, id=item["id"],
                      visible=bool(item["visible"]), opacity=float(np.clip(item["opacity"], 0, 1)),
                      smooth=bool(item["smooth"]),
                      representation=str(item.get("representation", "surface")))
        layers.append(layer)
    job.progress(100, "Project opened")
    return volume, layers, manifest["views"]
