"""Pre-rendered views of CAD parts: write them to disk once, then load them fast for training.

Layout written by render_split() and read by ViewDataset:

    <out>/meta.json                              image size, field of view, camera matrix K
    <out>/index.csv                              one row per image (see COLUMNS)
    <out>/<class>/<part>_e<elev>_a<azim>.png     BGR image, part centred and filling the frame

Poses are world -> camera (OpenCV convention), with the part centred at the origin and scaled
into the unit sphere, exactly as cad_views.load_triangles() prepares it.
"""
import csv
import json
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from aruco_tools.cad import MeshRenderer
from aruco_tools.types import CameraIntrinsics

from .cad_obj_dataset import CadObjDataset
from .cad_views import load_triangles, render_view

COLUMNS = ["file", "class", "part", "elev", "azim", "rx", "ry", "rz", "tx", "ty", "tz"]


def _render_part(job) -> list[list]:
    """Worker: render every view of one part, write the PNGs, return their index rows."""
    path, out_root, elevs, azims, size, fov = job
    cv2.setNumThreads(1)  # parallelism comes from the process pool
    cls, part = path.parent.name, path.stem
    (out_root / cls).mkdir(parents=True, exist_ok=True)
    renderer = MeshRenderer(CameraIntrinsics.from_fov(size, size, fov), load_triangles(path))
    rows = []
    for elev in elevs:
        for azim in azims:
            img, pose = render_view(renderer, elev, azim)
            rel = f"{cls}/{part}_e{elev:g}_a{azim:03g}.png"
            cv2.imwrite(str(out_root / rel), img)
            rows.append([rel, cls, part, elev, azim, *pose.rvec.ravel(), *pose.tvec.ravel()])
    return rows


def render_split(src: CadObjDataset, out_root: Path, elevs=(20, 45), azims=tuple(range(0, 360, 45)),
                 size: int = 224, fov: float = 40, workers: int | None = None,
                 limit: int | None = None) -> Path:
    """Render len(elevs) x len(azims) views of every part in `src` (or the first `limit`
    parts) into `out_root`, using `workers` processes. Returns the path of index.csv."""
    out_root = Path(out_root).expanduser()
    out_root.mkdir(parents=True, exist_ok=True)
    files = src.files[:limit] if limit else src.files
    jobs = [(f, out_root, elevs, azims, size, fov) for f in files]

    index = out_root / "index.csv"
    with Pool(workers) as pool, index.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(COLUMNS)
        for k, rows in enumerate(pool.imap(_render_part, jobs, chunksize=16), 1):
            writer.writerows(rows)
            if k % 1000 == 0 or k == len(jobs):
                print(f"  {k}/{len(jobs)} parts", flush=True)

    meta = {"size": size, "fov_deg": fov, "elevs": list(elevs), "azims": list(azims),
            "K": CameraIntrinsics.from_fov(size, size, fov).K.tolist(), "classes": src.classes}
    (out_root / "meta.json").write_text(json.dumps(meta, indent=1))
    return index


class ViewDataset(Dataset):
    """index.csv rows -> (image float32 [3, H, W] RGB in 0-1, label int, pose float32 [6]).

    pose is (rvec, tvec): the camera's view of the part, world -> camera. Pass the train set's
    `classes` to the test set so both use the same label numbers. `transform` is applied to
    the image tensor (augmentation, ImageNet normalisation, ...)."""

    def __init__(self, root, classes: list[str] | None = None, transform=None):
        self.root = Path(root).expanduser()
        self.meta = json.loads((self.root / "meta.json").read_text())
        with (self.root / "index.csv").open() as fh:
            self.rows = list(csv.DictReader(fh))
        self.classes = classes or self.meta["classes"]
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        self.K = np.array(self.meta["K"])
        self.transform = transform

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, int, torch.Tensor]:
        row = self.rows[i]
        bgr = cv2.imread(str(self.root / row["file"]), cv2.IMREAD_COLOR)
        image = torch.from_numpy(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)).permute(2, 0, 1).float() / 255
        if self.transform:
            image = self.transform(image)
        pose = torch.tensor([float(row[k]) for k in ("rx", "ry", "rz", "tx", "ty", "tz")])
        return image, self.class_to_idx[row["class"]], pose
