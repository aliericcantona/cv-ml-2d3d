from __future__ import annotations

import cv2
import numpy as np

from .types import (ArucoDictType, CameraIntrinsics, DetectionResult, MarkerDetection, Pose,
                    marker_corners_3d)


class ArucoMarkerDetector:
    """Finds markers in an image. If camera intrinsics and the marker side length
    are given, it also estimates each marker's pose."""

    def __init__(self, dict_type: ArucoDictType = ArucoDictType.DICT_4X4_50,
                 camera: CameraIntrinsics | None = None,
                 marker_length_m: float | None = None):
        params = cv2.aruco.DetectorParameters()
        params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        self._detector = cv2.aruco.ArucoDetector(dict_type.dictionary(), params)
        self.camera = camera
        self.marker_length_m = marker_length_m

    def detect(self, image: np.ndarray) -> DetectionResult:
        corners, ids, rejected = self._detector.detectMarkers(image)
        ids = [] if ids is None else ids.flatten()
        markers = []
        for c, i in zip(corners, ids):
            c = c.reshape(4, 2)
            pose = self._estimate_pose(c)
            rmse = self.reprojection_rmse(c, pose) if pose else None
            markers.append(MarkerDetection(int(i), c, pose, rmse))
        return DetectionResult(markers, [r.reshape(4, 2) for r in rejected])

    def _estimate_pose(self, corners: np.ndarray) -> Pose | None:
        if self.camera is None or self.marker_length_m is None:
            return None
        # This corner order is required by SOLVEPNP_IPPE_SQUARE and matches aruco's TL, TR, BR, BL
        obj = marker_corners_3d(self.marker_length_m).astype(np.float32)
        ok, rvec, tvec = cv2.solvePnP(obj, corners, self.camera.K, self.camera.dist,
                                      flags=cv2.SOLVEPNP_IPPE_SQUARE)
        return Pose(rvec, tvec) if ok else None

    def reprojection_rmse(self, corners: np.ndarray, pose: Pose) -> float:
        """Root-mean-square distance in pixels between the detected corners and the marker's
        3D corners projected with `pose`. Needs no ground truth, so it works on real images too."""
        projected, _ = cv2.projectPoints(marker_corners_3d(self.marker_length_m), pose.rvec, pose.tvec,
                                         self.camera.K, self.camera.dist)
        return float(np.sqrt(np.mean(np.sum((projected.reshape(4, 2) - corners) ** 2, axis=1))))

    def draw(self, image: np.ndarray, result: DetectionResult, show_rejected: bool = False) -> np.ndarray:
        out = image.copy()
        if show_rejected and result.rejected:
            cv2.polylines(out, [r.astype(np.int32) for r in result.rejected], True, (0, 0, 255), 1)
        for m in result.markers:
            cv2.aruco.drawDetectedMarkers(out, [m.corners[None]], np.array([[m.marker_id]]), (255, 0, 255))
            if m.pose is not None:
                cv2.drawFrameAxes(out, self.camera.K, self.camera.dist, m.pose.rvec, m.pose.tvec,
                                  self.marker_length_m * 0.8, 3)
        return out
