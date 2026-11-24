"""Render CAD meshes and point clouds as OpenCV images (BGR uint8) from cameras placed around them.

Each view is a pinhole camera on a sphere around the part, given by elevation and azimuth in
degrees, looking at the centre. The mesh is drawn with aruco_tools.cad.MeshRenderer.
"""
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d

from aruco_tools.cad import MeshRenderer
from aruco_tools.sim import OrbitCamera
from aruco_tools.types import CameraIntrinsics, Pose


def load_triangles(path: Path) -> np.ndarray:
    """Mesh triangles [T, 3, 3], with the bounding-box centre at the origin (the point every view
    is framed on) and scaled into the unit sphere."""
    mesh = o3d.io.read_triangle_mesh(str(path))
    v = np.asarray(mesh.vertices, dtype=np.float64)
    v = v - (v.min(axis=0) + v.max(axis=0)) / 2
    v /= np.linalg.norm(v, axis=1).max() + 1e-8
    return v[np.asarray(mesh.triangles)]


def view_pose(elev: float, azim: float, dist: float = 3.0) -> Pose:
    """Camera on a sphere of radius `dist` at (elev, azim) degrees, looking at the origin.
    Returned as the world -> camera pose that MeshRenderer expects."""
    el, az = np.radians(np.clip(elev, -89, 89)), np.radians(azim)  # avoid the poles, up is +z
    eye = dist * np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
    R = OrbitCamera._look_at(eye, np.zeros(3))
    return Pose.from_rt(R, -R @ eye)


def render_view(renderer: MeshRenderer, elev: float, azim: float) -> tuple[np.ndarray, Pose]:
    """BGR image of the mesh from (elev, azim), framed so the part fills the image, and the
    camera pose actually used (world -> camera; the camera is moved in or out for framing)."""
    pose = renderer.framed_pose(view_pose(elev, azim))
    return renderer.render(pose), pose


def create_view(renderer: MeshRenderer, elev: float, azim: float, label: bool = True) -> np.ndarray:
    """One BGR image of the mesh from (elev, azim), framed so the part fills the image."""
    img, _ = render_view(renderer, elev, azim)
    if label:
        cv2.putText(img, f"el {elev:.0f}  az {azim:.0f}", (8, 20), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (220, 220, 220), 1, cv2.LINE_AA)
    return img


def create_views(triangles: np.ndarray, views: int = 4, elev: float = 25, size: int = 320,
                 fov_deg: float = 40, label: bool = True) -> list[np.ndarray]:
    """`views` BGR images from cameras spaced evenly around the part at the same elevation.
    Use label=False for images that go into a model (no text drawn on them)."""
    renderer = MeshRenderer(CameraIntrinsics.from_fov(size, size, fov_deg), triangles)
    return [create_view(renderer, elev, -60 + k * 360 / views, label) for k in range(views)]


def tile(images: list[np.ndarray], cols: int = 4, fill: int = 40) -> np.ndarray:
    """Same-size images in a grid, `cols` per row; empty cells are filled with `fill` grey."""
    blank = np.full_like(images[0], fill)
    cells = images + [blank] * (-len(images) % cols)
    return cv2.vconcat([cv2.hconcat(cells[r:r + cols]) for r in range(0, len(cells), cols)])


def create_points_view(points: np.ndarray, elev: float = 25, azim: float = -60, size: int = 320,
                       fov_deg: float = 40) -> np.ndarray:
    """BGR image of a point cloud [N, 3] (in the unit sphere), near points bright, far ones dark."""
    camera = CameraIntrinsics.from_fov(size, size, fov_deg)
    tan_half = np.tan(np.radians(fov_deg) / 2)
    pose = view_pose(elev, azim, dist=1.0 / (tan_half * 0.85))   # unit sphere fills ~85%
    px, _ = cv2.projectPoints(points.astype(np.float64), pose.rvec, pose.tvec, camera.K, None)
    depth = (points @ pose.R.T + pose.tvec.ravel())[:, 2]
    near = (depth.max() - depth) / (np.ptp(depth) + 1e-8)          # 1 = nearest
    img = np.full((size, size, 3), 40, np.uint8)
    for i in np.argsort(-depth):                                  # far to near
        color = cv2.applyColorMap(np.uint8([[255 * near[i]]]), cv2.COLORMAP_VIRIDIS)[0, 0]
        cv2.circle(img, tuple(np.round(px[i, 0]).astype(int)), 2, color.tolist(), -1, cv2.LINE_AA)
    cv2.putText(img, f"points ({len(points)})", (8, 20), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (220, 220, 220), 1, cv2.LINE_AA)
    return img
