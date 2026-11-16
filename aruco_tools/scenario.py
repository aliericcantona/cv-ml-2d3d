"""One simulated setup (camera orbiting a marker on a background) and its per-frame loop:
render -> add noise -> detect -> score. Used by both the demo and the benchmark."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .detector import ArucoMarkerDetector
from .evaluator import DetectionStats, FrameScore
from .generator import ArucoMarkerGenerator
from .sim import Background, CameraState, GroundPlaneRenderer, OrbitCamera, add_sensor_effects, make_ground
from .types import CameraIntrinsics, DetectionResult, MarkerDetection, Pose

MARKER_ID = 0
MARKER_SIZE_M = 0.1875
IMAGE_SIZE = (960, 720)
BORDER_RATIO = 1 / 6  # white margin per side, as a fraction of the marker side


@dataclass(frozen=True)
class Frame:
    """Everything produced for one time step."""
    t: float
    truth: CameraState               # true camera (ground truth)
    image: np.ndarray                # rendered, noisy camera image
    result: DetectionResult          # all detections and rejected candidates
    detection: MarkerDetection | None  # our marker, if found
    score: FrameScore

    @property
    def pose(self) -> Pose | None:
        """Detected marker pose, or None."""
        return self.detection.pose if self.detection else None


class Scenario:
    """A camera orbiting one marker on a ground texture.

    `background` is a Background, its name ("checker"), an image path, or a ready ground
    texture array. `step(t)` renders, detects and scores one frame; totals build up in `stats`."""

    def __init__(self, background: Background | str | np.ndarray = Background.GRASS,
                 border: bool = True, seed: int = 0):
        self.camera = CameraIntrinsics.from_fov(*IMAGE_SIZE)
        self.marker_id, self.marker_size_m = MARKER_ID, MARKER_SIZE_M
        card = ArucoMarkerGenerator().generate(MARKER_ID, MARKER_SIZE_M,
                                               border_ratio=BORDER_RATIO if border else 0)
        self.renderer = GroundPlaneRenderer(self.camera, _ground(background), card)
        self.detector = ArucoMarkerDetector(camera=self.camera, marker_length_m=MARKER_SIZE_M)
        self.orbit = OrbitCamera()
        self.rng = np.random.default_rng(seed)
        self.stats = DetectionStats()

    def step(self, t: float) -> Frame:
        truth = self.orbit.at(t)
        image = add_sensor_effects(self.renderer.render(truth), self.rng)
        result = self.detector.detect(image)
        detection = result.get(self.marker_id)
        visible = truth.sees_marker(self.camera, self.marker_size_m)
        score = self.stats.add(detection, truth.marker_pose, visible)
        return Frame(t, truth, image, result, detection, score)


def _ground(background: Background | str | np.ndarray) -> np.ndarray:
    if isinstance(background, np.ndarray):
        return background
    if isinstance(background, str) and background in {b.value for b in Background}:
        background = Background(background)
    return make_ground(background)  # a Background, or an image path
