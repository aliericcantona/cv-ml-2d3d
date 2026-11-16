"""Synthetic test scenes: a marker lying on a textured ground plane (z = 0, marker
centred at the origin) seen by a camera orbiting around it."""
from __future__ import annotations

import os
from dataclasses import dataclass
from enum import Enum

import cv2
import numpy as np

from .types import CameraIntrinsics, MarkerCard, Pose, marker_corners_3d


class Background(Enum):
    GRASS = "grass"
    NOISE = "noise"
    CHECKER = "checker"   # adversarial: lots of marker-like black/white squares
    WHITE = "white"
    DARK = "dark"


def make_ground(background: Background | str, size_px: int = 2048, seed: int = 0) -> np.ndarray:
    """Ground texture from a builtin Background or an image path."""
    n, rng = size_px, np.random.default_rng(seed)
    if isinstance(background, str):
        img = cv2.imread(os.path.expanduser(background))
        if img is None:
            raise ValueError(f"cannot read background image {background!r}")
        return cv2.resize(img, (n, n), interpolation=cv2.INTER_AREA)
    if background is Background.GRASS:
        return _grass(rng, n)
    if background is Background.NOISE:
        return rng.integers(0, 256, (n, n, 3), dtype=np.uint8)
    if background is Background.CHECKER:
        yy, xx = np.mgrid[0:n, 0:n] // 64
        return cv2.cvtColor((((yy + xx) % 2) * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    if background is Background.WHITE:
        return np.full((n, n, 3), 235, np.uint8)
    return np.full((n, n, 3), 20, np.uint8)  # DARK


def _grass(rng, n):
    def lowfreq(cells):
        return cv2.resize(rng.random((cells, cells)).astype(np.float32), (n, n),
                          interpolation=cv2.INTER_CUBIC)
    shade = 0.6 * lowfreq(8) + 0.4 * lowfreq(40)
    img = np.stack([30 + 25 * shade, 90 + 70 * shade, 40 + 35 * shade], -1).astype(np.uint8)
    for _ in range(60000):  # short blades in random green shades
        x, y = rng.integers(0, n, 2)
        ang, ln, g = rng.uniform(0, np.pi), rng.integers(6, 22), int(rng.integers(70, 200))
        col = (int(g * rng.uniform(0.2, 0.45)), g, int(g * rng.uniform(0.3, 0.6)))
        cv2.line(img, (int(x), int(y)), (int(x + ln * np.cos(ang)), int(y + ln * np.sin(ang))),
                 col, 1, cv2.LINE_AA)
    return cv2.GaussianBlur(img, (3, 3), 0)


@dataclass(frozen=True)
class CameraState:
    R: np.ndarray           # world -> camera rotation
    position: np.ndarray    # camera centre in world, metres

    @property
    def marker_pose(self) -> Pose:
        """Ground-truth pose of the marker (world origin) in the camera frame."""
        return Pose.from_rt(self.R, -self.R @ self.position)

    @property
    def elevation_deg(self) -> float:
        return float(np.degrees(np.arcsin(self.position[2] / np.linalg.norm(self.position))))

    def sees_marker(self, camera: CameraIntrinsics, marker_length_m: float) -> bool:
        """True if all four marker corners project inside the image."""
        pose = self.marker_pose
        px, _ = cv2.projectPoints(marker_corners_3d(marker_length_m), pose.rvec, pose.tvec, camera.K, camera.dist)
        x, y = px.reshape(-1, 2).T
        return bool(((x >= 0) & (x < camera.width) & (y >= 0) & (y < camera.height)).all())


class OrbitCamera:
    """Circles the marker while bobbing in height (22..82 deg) and distance (0.5..1.7 m),
    aiming slightly off-centre so the marker drifts around the frame."""

    def at(self, t: float) -> CameraState:
        az = 0.35 * t
        el = np.radians(52 + 30 * np.sin(0.23 * t))
        dist = 1.1 + 0.6 * np.sin(0.17 * t + 1.0)
        pos = dist * np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
        target = 0.12 * np.array([np.sin(0.5 * t), np.cos(0.41 * t), 0.0])
        return CameraState(self._look_at(pos, target), pos)

    @staticmethod
    def _look_at(pos, target, up=(0.0, 0.0, 1.0)):
        f = target - pos
        f /= np.linalg.norm(f)
        r = np.cross(f, up)
        r /= np.linalg.norm(r)
        return np.stack([r, np.cross(f, r), f])  # rows: x right, y down, z forward


class GroundPlaneRenderer:
    """Ray-casts every pixel onto the ground plane and samples the ground texture
    and marker card there. Renders at `supersample`x and downsizes for anti-aliasing."""

    def __init__(self, camera: CameraIntrinsics, ground: np.ndarray, card: MarkerCard,
                 ground_size_m: float = 4.0, supersample: int = 2):
        self.camera, self.ground, self.card = camera, ground, card
        self.ground_size_m = ground_size_m
        w, h = camera.width * supersample, camera.height * supersample
        self._shape = (h, w)
        K = camera.K.copy()
        K[:2] *= supersample
        u, v = np.meshgrid(np.arange(w) + 0.5, np.arange(h) + 0.5)
        rays = np.stack([u, v, np.ones_like(u)], -1) @ np.linalg.inv(K).T
        self._rays = rays.reshape(-1, 3).astype(np.float32)
        top = np.linspace(1, 0, h, dtype=np.float32)[:, None, None]
        self._sky = np.array([235, 205, 170], np.float32) * top + np.array([250, 245, 235], np.float32) * (1 - top)

    def render(self, cam: CameraState) -> np.ndarray:
        d = self._rays @ cam.R.astype(np.float32)   # ray directions in world
        hit = d[:, 2] < -1e-6
        t = np.where(hit, -cam.position[2] / np.where(hit, d[:, 2], -1), 0)
        X = cam.position[0] + t * d[:, 0]
        Y = cam.position[1] + t * d[:, 1]

        ground = self._sample(self.ground, X, Y, self.ground_size_m, cv2.BORDER_REFLECT)
        card = self._sample(self.card.image, X, Y, self.card.card_length_m, cv2.BORDER_CONSTANT)
        alpha = self._sample(np.full(self.card.image.shape[:2], 1, np.float32), X, Y,
                             self.card.card_length_m, cv2.BORDER_CONSTANT)[..., None]

        img = ground * (1 - alpha) + card * alpha
        img = np.where(hit.reshape(self._shape)[..., None], img, self._sky)
        img = cv2.resize(img, (self.camera.width, self.camera.height), interpolation=cv2.INTER_AREA)
        return np.clip(img, 0, 255).astype(np.uint8)

    def _sample(self, tex, X, Y, size_m, border):
        """Sample a square texture centred at the origin and spanning size_m metres."""
        n = tex.shape[0]
        ppm = n / size_m
        mx = (X * ppm + n / 2).reshape(self._shape).astype(np.float32)
        my = (-Y * ppm + n / 2).reshape(self._shape).astype(np.float32)
        return cv2.remap(tex, mx, my, cv2.INTER_LINEAR, borderMode=border).astype(np.float32)


def add_sensor_effects(img: np.ndarray, rng: np.random.Generator,
                       noise_sigma: float = 4.0, blur_px: int = 0) -> np.ndarray:
    if blur_px > 1:
        k = np.zeros((blur_px, blur_px), np.float32)
        k[blur_px // 2, :] = 1.0 / blur_px
        img = cv2.filter2D(img, -1, k)
    if noise_sigma > 0:
        img = np.clip(img + rng.normal(0, noise_sigma, img.shape), 0, 255).astype(np.uint8)
    return img
