"""Scores detections against ground truth: detection rate and pose error."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

from .types import MarkerDetection, Pose


class Status(Enum):
    SKIPPED = "skipped"  # marker not fully in view: not scored
    MISSED = "missed"    # in view, but no pose found
    GOOD = "good"        # pose within the error limits
    WRONG = "wrong"      # pose found, but worse than the limits


@dataclass(frozen=True)
class FrameScore:
    """The outcome of one frame. Errors are set only for GOOD and WRONG."""
    status: Status
    rot_err_deg: float | None = None
    pos_err_mm: float | None = None
    reproj_rmse_px: float | None = None

    def describe(self) -> str:
        if self.status is Status.SKIPPED:
            return "marker out of view (not scored)"
        if self.status is Status.MISSED:
            return "NOT DETECTED"
        text = f"rot err {self.rot_err_deg:.2f} deg   pos err {self.pos_err_mm:.1f} mm"
        if self.reproj_rmse_px is not None:
            text += f"   reproj {self.reproj_rmse_px:.2f} px"
        return text + "   WRONG POSE" if self.status is Status.WRONG else text


class DetectionStats:
    """Feed it one frame at a time with `add`; read the running totals.

    A detection whose pose is off by more than `wrong_rot_deg` or `wrong_pos_mm` is counted
    as a *wrong* pose, not a good one. Its errors are still kept, so p95/max show it."""

    def __init__(self, wrong_rot_deg: float = 5.0, wrong_pos_mm: float = 20.0):
        self.wrong_rot_deg, self.wrong_pos_mm = wrong_rot_deg, wrong_pos_mm
        self.frames = 0      # frames where the marker was fully in view
        self.skipped = 0     # frames where it wasn't (not scored)
        self.wrong = 0       # detected, but pose badly off
        self.rot_errors_deg: list[float] = []
        self.pos_errors_mm: list[float] = []
        self.reproj_rmse_px: list[float] = []

    def add(self, detection: MarkerDetection | None, truth: Pose, visible: bool = True) -> FrameScore:
        """Score one frame and add it to the totals."""
        if not visible:
            self.skipped += 1
            return FrameScore(Status.SKIPPED)
        self.frames += 1
        if detection is None or detection.pose is None:
            return FrameScore(Status.MISSED)

        rot, pos = detection.pose.error_to(truth)
        rmse = detection.reproj_rmse_px
        self.rot_errors_deg.append(rot)
        self.pos_errors_mm.append(pos)
        if rmse is not None:
            self.reproj_rmse_px.append(rmse)
        if rot > self.wrong_rot_deg or pos > self.wrong_pos_mm:
            self.wrong += 1
            return FrameScore(Status.WRONG, rot, pos, rmse)
        return FrameScore(Status.GOOD, rot, pos, rmse)

    @property
    def detected(self) -> int:
        return len(self.rot_errors_deg)

    @property
    def good(self) -> int:
        return self.detected - self.wrong

    @property
    def rate(self) -> float:
        """Fraction of in-view frames with a *correct* pose (0..1)."""
        return self.good / self.frames if self.frames else 0.0

    @property
    def wrong_rate(self) -> float:
        return self.wrong / self.frames if self.frames else 0.0

    def rot_percentile(self, q: float) -> float | None:
        return float(np.percentile(self.rot_errors_deg, q)) if self.detected else None

    def pos_percentile(self, q: float) -> float | None:
        return float(np.percentile(self.pos_errors_mm, q)) if self.detected else None

    def reproj_percentile(self, q: float) -> float | None:
        return float(np.percentile(self.reproj_rmse_px, q)) if self.reproj_rmse_px else None

    def summary(self) -> str:
        text = f"good {self.good}/{self.frames} ({100 * self.rate:.0f}%)   wrong {self.wrong}"
        if self.detected:
            text += (f"   median rot {self.rot_percentile(50):.2f} deg"
                     f"   pos {self.pos_percentile(50):.1f} mm")
        if self.reproj_rmse_px:
            text += f"   reproj {self.reproj_percentile(50):.2f} px"
        return text
