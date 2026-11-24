"""Load an STL or OBJ part and draw it from a camera pose with OpenCV (no 3D library needed)."""
from __future__ import annotations

import re
from pathlib import Path

import cv2
import numpy as np

from .types import CameraIntrinsics, Pose, marker_corners_3d


def load_stl(path: str | Path, scale: float = 0.001) -> np.ndarray:
    """Triangles of a binary or ASCII STL as an (n, 3, 3) array, multiplied by `scale`
    (default: file in millimetres -> metres)."""
    data = Path(path).expanduser().read_bytes()
    if data[:5] == b"solid" and b"facet" in data[:1024]:  # ASCII
        tris = np.array(re.findall(rb"vertex\s+(\S+)\s+(\S+)\s+(\S+)", data), float).reshape(-1, 3, 3)
    else:  # binary: 80-byte header, uint32 count, then 50-byte records
        n = int.from_bytes(data[80:84], "little")
        record = np.dtype([("normal", "<f4", 3), ("v", "<f4", (3, 3)), ("attr", "<u2")])
        tris = np.frombuffer(data, record, count=n, offset=84)["v"].astype(float)
    return tris * scale


def load_obj(path: str | Path) -> np.ndarray:
    """Triangles of a Wavefront OBJ as an (n, 3, 3) array, in the file's own units.
    Reads only `v` and `f` lines; polygons are split into triangle fans."""
    verts, faces = [], []
    for line in Path(path).expanduser().read_text(errors="ignore").splitlines():
        if line.startswith("v "):
            verts.append([float(x) for x in line.split()[1:4]])
        elif line.startswith("f "):
            idx = [int(tok.split("/")[0]) for tok in line.split()[1:]]  # "f 1/2/3" -> 1
            idx = [i - 1 if i > 0 else len(verts) + i for i in idx]      # 1-based or negative
            faces += [(idx[0], idx[k], idx[k + 1]) for k in range(1, len(idx) - 1)]
    return np.asarray(verts, float)[np.asarray(faces, int)]


def load_mesh(path: str | Path, size_m: float = 0.15) -> np.ndarray:
    """Triangles in metres from an STL (assumed millimetres) or an OBJ. OBJ files have no
    standard unit (MCB parts are about 1 unit across), so they are scaled so their longest
    side is `size_m`."""
    if Path(path).suffix.lower() == ".obj":
        tris = load_obj(path)
        pts = tris.reshape(-1, 3)
        return tris * (size_m / np.ptp(pts, axis=0).max())
    return load_stl(path)


def place_on_marker(tris: np.ndarray) -> np.ndarray:
    """Centre the part on the marker (x, y) and rest its bottom on the marker plane (z = 0)."""
    pts = tris.reshape(-1, 3)
    centre = (pts.min(0) + pts.max(0)) / 2
    return tris - [centre[0], centre[1], pts[:, 2].min()]


class MeshRenderer:
    """Draws triangles given in the marker frame, seen from a marker pose (marker -> camera).

    Flat shading with a fixed light above the marker, back-face culling, and far-to-near
    drawing order. Good enough for simple parts; not a full z-buffer."""

    def __init__(self, camera: CameraIntrinsics, tris: np.ndarray,
                 color_bgr=(80, 150, 230), light=(0.3, 0.5, 1.0)):
        self.camera, self.tris = camera, tris
        self.color = np.array(color_bgr, float)
        self.light = np.asarray(light, float) / np.linalg.norm(light)
        pts = tris.reshape(-1, 3)
        self.centre = (pts.min(0) + pts.max(0)) / 2
        self.radius = float(np.linalg.norm(pts - self.centre, axis=1).max())

    def framed_pose(self, pose: Pose, fill: float = 0.8) -> Pose:
        """Same viewing direction and roll as `pose`, but with the camera moved so the part
        sits in the middle of the image and fills about `fill` of its height."""
        R, t = pose.R, pose.tvec.ravel()
        eye_dir = -R.T @ t - self.centre                      # part centre -> camera, marker frame
        eye_dir /= np.linalg.norm(eye_dir)
        tan_half = (self.camera.height / 2) / self.camera.K[1, 1]
        eye = self.centre + eye_dir * self.radius / (tan_half * fill)

        forward = -eye_dir
        right = np.cross(forward, -R[1])                      # keep the camera's own "up" (-y)
        right /= np.linalg.norm(right)
        R_new = np.stack([right, np.cross(forward, right), forward])
        return Pose.from_rt(R_new, -R_new @ eye)

    def render(self, pose: Pose | None, marker_length_m: float | None = None,
               image: np.ndarray | None = None) -> np.ndarray:
        """Draw onto a copy of `image` (e.g. the camera frame), or onto a plain dark panel."""
        img = np.full((self.camera.height, self.camera.width, 3), 40, np.uint8) if image is None \
            else image.copy()
        if pose is None:
            return img
        R, t, K = pose.R, pose.tvec.ravel(), self.camera.K

        if marker_length_m:  # outline of the marker under the part, for reference
            square = marker_corners_3d(marker_length_m) @ R.T + t
            if (square[:, 2] > 1e-3).all():
                px = square @ K.T
                cv2.polylines(img, [np.round(px[:, :2] / px[:, 2:] * 16).astype(np.int32)], True,
                              (160, 160, 160), 2, cv2.LINE_AA, 4)

        v = self.tris @ R.T + t                                   # vertices in camera frame
        normals = np.cross(v[:, 1] - v[:, 0], v[:, 2] - v[:, 0])
        normals /= np.linalg.norm(normals, axis=1, keepdims=True) + 1e-12
        facing = np.einsum("ij,ij->i", normals, v[:, 0]) < 0     # front faces only
        keep = facing & (v[..., 2] > 1e-3).all(1)                 # and fully in front of camera
        v, normals = v[keep], normals[keep]

        shade = 0.3 + 0.7 * np.clip(normals @ (R @ self.light), 0, 1)
        px = v @ K.T
        pts = np.round(px[..., :2] / px[..., 2:] * 16).astype(np.int32)  # 4-bit subpixel
        for i in np.argsort(-v[..., 2].mean(1)):                  # far to near
            cv2.fillConvexPoly(img, pts[i], (self.color * shade[i]).tolist(), cv2.LINE_AA, 4)
        return img
