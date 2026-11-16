"""Reprojection RMSE over time, drawn with OpenCV only: a live strip for the demo and a PNG to save."""
from __future__ import annotations

import math

import cv2
import numpy as np

from .evaluator import FrameScore, Status

# BGR colours (dark chart surface)
SURFACE = (25, 26, 26)      # #1a1a19
GRID = (52, 54, 54)
AXIS = (90, 92, 92)
TEXT = (236, 239, 240)      # primary ink
TEXT_2 = (183, 194, 195)    # secondary ink
SERIES = (229, 135, 57)     # blue #3987e5
CRITICAL = (59, 59, 208)    # red #d03b3b, wrong-pose frames
FONT = cv2.FONT_HERSHEY_SIMPLEX


def _nice_ceil(v: float, steps: tuple[float, ...] = (1, 2, 5, 10)) -> float:
    """Smallest step x 10^k that is >= v."""
    k = 10 ** math.floor(math.log10(v))
    return next(m * k for m in steps if m * k >= v - 1e-12)


# y-axis tops whose quarters are round numbers (e.g. 16 -> 4, 8, 12, 16)
Y_STEPS = (1, 1.2, 1.6, 2, 2.4, 4, 6, 8, 10)


class RmsePlot:
    """Collects one RMSE per frame (`add`) and draws it against time (`render`).

    Frames with no pose leave a gap in the line; frames whose pose is WRONG get a red dot."""

    def __init__(self, window_s: float = 20.0):
        self.window_s = window_s            # live view shows the last `window_s` seconds
        self.t: list[float] = []
        self.rmse: list[float] = []         # NaN when there is no pose
        self.wrong: list[bool] = []

    def add(self, t: float, score: FrameScore) -> None:
        self.t.append(t)
        self.rmse.append(math.nan if score.reproj_rmse_px is None else score.reproj_rmse_px)
        self.wrong.append(score.status is Status.WRONG)

    def render(self, width: int, height: int, full: bool = False) -> np.ndarray:
        """The chart as a BGR image. `full=True` shows the whole run instead of the last window."""
        img = np.full((height, width, 3), SURFACE, np.uint8)
        left, right, top, bottom = 70, 100, 76, 40
        pw, ph = width - left - right, height - top - bottom
        t, y, wrong = np.array(self.t), np.array(self.rmse), np.array(self.wrong, bool)

        t_last = t[-1] if len(t) else 0.0
        if full and len(t):
            t0, t1 = t[0], max(t_last, t[0] + 1e-6)
        else:
            t0 = max(0.0, t_last - self.window_s) if len(t) else 0.0
            t1 = t0 + self.window_s
        keep = (t >= t0) & (t <= t1)
        t, y, wrong = t[keep], y[keep], wrong[keep]
        finite = np.isfinite(y)
        y_max = _nice_ceil(max(0.4, float(y[finite].max()) * 1.05 if finite.any() else 0.4), Y_STEPS)

        def px(tv, yv):
            return (int(round(left + (tv - t0) / (t1 - t0) * pw)),
                    int(round(top + ph - min(yv, y_max) / y_max * ph)))

        # title + stats over the shown range
        cv2.putText(img, "Reprojection RMSE (px)", (left, 28), FONT, 0.7, TEXT, 2, cv2.LINE_AA)
        if finite.any():
            v = y[finite]
            span = "whole run" if full else f"last {self.window_s:g} s"
            stats = (f"{span}:  median {np.median(v):.2f}   p95 {np.percentile(v, 95):.2f}"
                     f"   max {v.max():.2f}   gaps = no pose")
            cv2.putText(img, stats, (left, 56), FONT, 0.55, TEXT_2, 1, cv2.LINE_AA)
        if wrong.any():  # key for the red dots
            kx = width - right - 60
            cv2.circle(img, (kx, 22), 6, CRITICAL, -1, cv2.LINE_AA)
            cv2.putText(img, "wrong pose", (kx + 14, 28), FONT, 0.55, TEXT_2, 1, cv2.LINE_AA)

        # y grid + labels
        for i in range(5):
            val = y_max * i / 4
            _, gy = px(t0, val)
            cv2.line(img, (left, gy), (left + pw, gy), AXIS if i == 0 else GRID, 1)
            cv2.putText(img, f"{val:g}", (left - 50, gy + 5), FONT, 0.5, TEXT_2, 1, cv2.LINE_AA)

        # x ticks
        step = _nice_ceil((t1 - t0) / 6)
        for tv in np.arange(math.ceil(t0 / step) * step, t1 + 1e-9, step):
            gx, _ = px(tv, 0)
            cv2.line(img, (gx, top + ph), (gx, top + ph + 5), AXIS, 1)
            cv2.putText(img, f"{tv:g}s", (gx - 14, top + ph + 24), FONT, 0.5, TEXT_2, 1, cv2.LINE_AA)

        # line, broken where there is no pose
        run: list[tuple[int, int]] = []
        for tv, yv in [*zip(t, y), (None, math.nan)]:
            if tv is not None and math.isfinite(yv):
                run.append(px(tv, yv))
                continue
            if len(run) > 1:
                cv2.polylines(img, [np.array(run, np.int32)], False, SERIES, 2, cv2.LINE_AA)
            elif run:
                cv2.circle(img, run[0], 2, SERIES, -1, cv2.LINE_AA)
            run = []

        # wrong-pose dots, with a surface ring so they stand off the line
        for tv, yv in zip(t[wrong & finite], y[wrong & finite]):
            p = px(tv, yv)
            cv2.circle(img, p, 8, SURFACE, -1, cv2.LINE_AA)
            cv2.circle(img, p, 6, CRITICAL, -1, cv2.LINE_AA)

        # current value at the end of the line
        if finite.any():
            i = np.flatnonzero(finite)[-1]
            ex, ey = px(t[i], y[i])
            cv2.circle(img, (ex, ey), 4, SERIES, -1, cv2.LINE_AA)
            cv2.putText(img, f"{y[i]:.2f} px", (ex + 10, ey + 5), FONT, 0.55, TEXT, 1, cv2.LINE_AA)
        return img

    def save(self, path: str, size: tuple[int, int] = (1600, 500)) -> None:
        """Write the whole run as a PNG (or any format cv2.imwrite understands)."""
        cv2.imwrite(path, self.render(*size, full=True))
