"""PyTorch Dataset for folders of CAD meshes laid out as root/<class name>/<id>.obj (e.g. MCB).

Each sample is a fixed-size point cloud sampled from the mesh surface, centred and scaled to fit
in the unit sphere, plus the class index taken from the parent folder name.
"""
import random
from pathlib import Path

import numpy as np
import open3d as o3d
import torch
from torch.utils.data import Dataset

MCB_ROOT = Path.home() / ".cache" / "mcb_cad_data" / "MCB_A"


class CadObjDataset(Dataset):
    """root/<class name>/<id>.obj -> (points float32 [num_points, 3], label int)."""

    def __init__(self, root, num_points: int = 2048, classes: list[str] | None = None):
        self.root = Path(root).expanduser()
        self.files = sorted(self.root.glob("*/*.obj"))  # only .obj, so .DS_Store is skipped
        if not self.files:
            raise FileNotFoundError(f"no .obj files in {self.root}/<class>/")
        # pass the train set's classes to the test set so both use the same label numbers
        self.classes = classes or sorted({f.parent.name for f in self.files})
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        self.num_points = num_points

    def pick_index(self, cls: str | None = None) -> int:
        """A random sample index, optionally from one class folder."""
        if cls is None:
            return random.randrange(len(self))
        if cls not in self.class_to_idx:
            raise ValueError(f"unknown class {cls!r}; choose from: {', '.join(self.classes)}")
        return random.choice([i for i, f in enumerate(self.files) if f.parent.name == cls])

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, int]:
        path = self.files[i]
        mesh = o3d.io.read_triangle_mesh(str(path))
        pts = np.asarray(mesh.sample_points_uniformly(self.num_points).points, dtype=np.float32)
        pts -= pts.mean(axis=0)                                # centre
        pts /= np.linalg.norm(pts, axis=1).max() + 1e-8       # fit in the unit sphere
        return torch.from_numpy(pts), self.class_to_idx[path.parent.name]
