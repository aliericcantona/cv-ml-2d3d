"""Shared enums and data classes for ArUco generation, detection and pose."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import cv2
import numpy as np


class ArucoDictType(Enum):
    """Supported marker dictionaries (NxN bits, number of ids)."""
    DICT_4X4_50 = cv2.aruco.DICT_4X4_50
    DICT_4X4_100 = cv2.aruco.DICT_4X4_100
    DICT_4X4_250 = cv2.aruco.DICT_4X4_250
    DICT_4X4_1000 = cv2.aruco.DICT_4X4_1000
    DICT_5X5_50 = cv2.aruco.DICT_5X5_50
    DICT_5X5_100 = cv2.aruco.DICT_5X5_100
    DICT_5X5_250 = cv2.aruco.DICT_5X5_250
    DICT_5X5_1000 = cv2.aruco.DICT_5X5_1000
    DICT_6X6_50 = cv2.aruco.DICT_6X6_50
    DICT_6X6_100 = cv2.aruco.DICT_6X6_100
    DICT_6X6_250 = cv2.aruco.DICT_6X6_250
    DICT_6X6_1000 = cv2.aruco.DICT_6X6_1000
    DICT_7X7_50 = cv2.aruco.DICT_7X7_50
    DICT_7X7_100 = cv2.aruco.DICT_7X7_100
    DICT_7X7_250 = cv2.aruco.DICT_7X7_250
    DICT_7X7_1000 = cv2.aruco.DICT_7X7_1000
    ARUCO_ORIGINAL = cv2.aruco.DICT_ARUCO_ORIGINAL
    APRILTAG_36H11 = cv2.aruco.DICT_APRILTAG_36H11

    def dictionary(self) -> cv2.aruco.Dictionary:
        return cv2.aruco.getPredefinedDictionary(self.value)


def marker_corners_3d(marker_length_m: float) -> np.ndarray:
    """The marker's 4 corners in its own frame (centred, z = 0, z pointing out of the marker),
    in ArUco's corner order TL, TR, BR, BL. The one definition of the marker geometry:
    detection, visibility checks and drawing all use it, so they can't disagree."""
    h = marker_length_m / 2
    return np.array([[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]], np.float64)


@dataclass(frozen=True)
class CameraIntrinsics:
    K: np.ndarray                       # 3x3 camera matrix
    dist: np.ndarray | None = None      # distortion coeffs, None = no distortion
    width: int = 0
    height: int = 0

    @classmethod
    def from_fov(cls, width: int, height: int, fov_deg: float = 60.0) -> CameraIntrinsics:
        """Pinhole camera with square pixels and the principal point at the centre."""
        f = (width / 2.0) / np.tan(np.radians(fov_deg) / 2.0)
        K = np.array([[f, 0, width / 2.0], [0, f, height / 2.0], [0, 0, 1]], np.float64)
        return cls(K, None, width, height)


@dataclass(frozen=True)
class Pose:
    """Marker pose in the camera frame (OpenCV convention)."""
    rvec: np.ndarray    # 3x1 Rodrigues rotation
    tvec: np.ndarray    # 3x1 translation, metres

    @classmethod
    def from_rt(cls, R: np.ndarray, t: np.ndarray) -> Pose:
        rvec, _ = cv2.Rodrigues(np.asarray(R, np.float64))
        return cls(rvec, np.asarray(t, np.float64).reshape(3, 1))

    @property
    def R(self) -> np.ndarray:
        return cv2.Rodrigues(self.rvec)[0]

    @property
    def distance_m(self) -> float:
        return float(np.linalg.norm(self.tvec))

    def error_to(self, other: Pose) -> tuple[float, float]:
        """(rotation error in degrees, translation error in mm) vs another pose."""
        cos = (np.trace(self.R @ other.R.T) - 1) / 2
        rot_deg = float(np.degrees(np.arccos(np.clip(cos, -1, 1))))
        trans_mm = float(1000 * np.linalg.norm(self.tvec.ravel() - other.tvec.ravel()))
        return rot_deg, trans_mm


@dataclass(frozen=True)
class MarkerCard:
    """A generated marker image plus its physical size when printed."""
    image: np.ndarray           # BGR, square
    dict_type: ArucoDictType
    marker_id: int
    marker_length_m: float      # side of the black square (what the detector measures)
    card_length_m: float        # side incl. the white quiet-zone border

    @property
    def has_border(self) -> bool:
        return self.card_length_m > self.marker_length_m


@dataclass(frozen=True)
class MarkerDetection:
    marker_id: int
    corners: np.ndarray         # 4x2 pixel coords: TL, TR, BR, BL
    pose: Pose | None = None    # set when intrinsics + marker length are known


@dataclass(frozen=True)
class DetectionResult:
    markers: list[MarkerDetection] = field(default_factory=list)
    rejected: list[np.ndarray] = field(default_factory=list)  # candidate quads that failed decoding

    def get(self, marker_id: int) -> MarkerDetection | None:
        return next((m for m in self.markers if m.marker_id == marker_id), None)
