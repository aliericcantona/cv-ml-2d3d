"""Charts drawn with OpenCV only (no plotting library), so they can go straight into the demo window
and its video.

ErrorPlot: rotation error, position error and reprojection RMSE over time, as three stacked
charts sharing one time axis (one scale each, never two scales on one chart)."""
from __future__ import annotations

import math
from dataclasses import dataclass

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
AA = cv2.LINE_AA

# axis tops whose halves are round numbers
Y_STEPS = (1, 1.2, 1.6, 2, 2.4, 4, 6, 8, 10)


def nice_ceil(v: float, steps: tuple[float, ...] = (1, 2, 5, 10)) -> float:
    """Smallest step x 10^k that is >= v."""
    k = 10 ** math.floor(math.log10(v))
    return next(m * k for m in steps if m * k >= v - 1e-12)


def dashed_hline(img: np.ndarray, x0: int, x1: int, y: int, color, dash: int = 8) -> None:
    for x in range(x0, x1, 2 * dash):
        cv2.line(img, (x, y), (min(x + dash, x1), y), color, 1, AA)


def up_triangle(img: np.ndarray, x: int, y: int, color, size: int = 6) -> None:
    pts = np.array([[x, y - size], [x - size, y + size // 2], [x + size, y + size // 2]], np.int32)
    cv2.fillPoly(img, [pts], color, AA)


@dataclass
class Series:
    title: str
    unit: str
    fmt: str                    # value format, e.g. ".2f"
    floor: float                # smallest axis top
    limit: float | None = None  # wrong-pose limit, drawn as a dashed line


class ErrorPlot:
    """Collects the per-frame errors (`add`) and draws them against time (`render`).

    Frames with no pose leave a gap; frames whose pose is WRONG get a red dot. If a few huge
    errors (a flipped pose) would squash everything else, the axis is capped and those
    values are drawn as triangles at the top ("off scale")."""

    def __init__(self, wrong_rot_deg: float = 5.0, wrong_pos_mm: float = 20.0, window_s: float = 20.0):
        self.window_s = window_s            # live view shows the last `window_s` seconds
        self.series = [Series("Rotation error", "deg", ".2f", 1.0, wrong_rot_deg),
                       Series("Position error", "mm", ".1f", 10.0, wrong_pos_mm),
                       Series("Reprojection RMSE", "px", ".2f", 0.4)]
        self.t: list[float] = []
        self.values: list[list[float]] = [[], [], []]   # NaN when there is no pose
        self.wrong: list[bool] = []

    def add(self, t: float, score: FrameScore) -> None:
        self.t.append(t)
        for vals, v in zip(self.values, (score.rot_err_deg, score.pos_err_mm, score.reproj_rmse_px)):
            vals.append(math.nan if v is None else v)
        self.wrong.append(score.status is Status.WRONG)

    def render(self, width: int, height: int, full: bool = False) -> np.ndarray:
        """The charts as one BGR image. `full=True` shows the whole run instead of the last window."""
        img = np.full((height, width, 3), SURFACE, np.uint8)
        t = np.array(self.t)
        if full and len(t):
            t0, t1 = t[0], max(t[-1], t[0] + 1e-6)
        else:
            t0 = max(0.0, t[-1] - self.window_s) if len(t) else 0.0
            t1 = t0 + self.window_s
        keep = (t >= t0) & (t <= t1)
        wrong = np.array(self.wrong, bool)[keep]
        span = "whole run" if full else f"last {self.window_s:g} s"

        # header: what the marks mean
        cv2.putText(img, f"Pose error vs ground truth ({span})", (16, 30), FONT, 0.7, TEXT, 2, AA)
        x, y = 16, 58
        cv2.circle(img, (x + 6, y - 5), 5, CRITICAL, -1, AA)
        cv2.putText(img, "wrong pose", (x + 18, y), FONT, 0.5, TEXT_2, 1, AA)
        x += 130
        up_triangle(img, x + 6, y - 5, SERIES)
        cv2.putText(img, "off scale", (x + 18, y), FONT, 0.5, TEXT_2, 1, AA)
        x += 115
        dashed_hline(img, x, x + 24, y - 5, TEXT_2, 4)
        cv2.putText(img, "wrong-pose limit", (x + 30, y), FONT, 0.5, TEXT_2, 1, AA)

        # three strips; the RMSE one is shorter, it is the supporting number
        top, bottom = 76, 34
        weights = (0.4, 0.4, 0.2)
        avail = height - top - bottom
        y0 = top
        for i, (s, vals) in enumerate(zip(self.series, self.values)):
            h = int(avail * weights[i])
            last = i == len(self.series) - 1
            self._strip(img, y0, h + (bottom if last else 0), t[keep], np.array(vals)[keep], wrong,
                        t0, t1, s, x_labels=last)
            y0 += h
        return img

    @staticmethod
    def _strip(img, y0, h, t, y, wrong, t0, t1, s: Series, x_labels: bool) -> None:
        width = img.shape[1]
        left, right = 70, 100
        top = y0 + 34                   # room for the strip title
        ph = h - 34 - (44 if x_labels else 12)
        pw = width - left - right
        finite = np.isfinite(y)
        v = y[finite]

        # axis top: fit the data, but don't let a few flips squash the rest
        y_max = max(s.floor, (s.limit or 0) * 1.5)
        if len(v):
            p95 = float(np.percentile(v, 95))
            y_max = max(y_max, float(v.max()) * 1.05 if v.max() <= 4 * max(p95, s.floor) else p95 * 2)
        y_max = nice_ceil(y_max, Y_STEPS)

        def px(tv, yv):
            return (int(round(left + (tv - t0) / (t1 - t0) * pw)),
                    int(round(top + ph - min(yv, y_max) / y_max * ph)))

        title = f"{s.title} ({s.unit})"
        cv2.putText(img, title, (left, top - 12), FONT, 0.6, TEXT, 1, AA)
        if len(v):
            stats = (f"median {np.median(v):{s.fmt}}   p95 {np.percentile(v, 95):{s.fmt}}"
                     f"   max {v.max():{s.fmt}}")
            (tw, _), _ = cv2.getTextSize(title, FONT, 0.6, 1)
            cv2.putText(img, stats, (left + tw + 20, top - 12), FONT, 0.5, TEXT_2, 1, AA)

        for i in range(3):              # 0, half, top
            val = y_max * i / 2
            _, gy = px(t0, val)
            cv2.line(img, (left, gy), (left + pw, gy), AXIS if i == 0 else GRID, 1)
            cv2.putText(img, f"{val:g}", (left - 50, gy + 5), FONT, 0.45, TEXT_2, 1, AA)
        if s.limit is not None and s.limit < y_max:
            _, ly = px(t0, s.limit)
            dashed_hline(img, left, left + pw, ly, TEXT_2)

        if x_labels:
            step = nice_ceil((t1 - t0) / 6)
            for tv in np.arange(math.ceil(t0 / step) * step, t1 + 1e-9, step):
                gx, _ = px(tv, 0)
                cv2.line(img, (gx, top + ph), (gx, top + ph + 5), AXIS, 1)
                cv2.putText(img, f"{tv:g}s", (gx - 14, top + ph + 24), FONT, 0.5, TEXT_2, 1, AA)

        # line, broken where there is no pose
        run: list[tuple[int, int]] = []
        for tv, yv in [*zip(t, y), (None, math.nan)]:
            if tv is not None and math.isfinite(yv):
                run.append(px(tv, yv))
                continue
            if len(run) > 1:
                cv2.polylines(img, [np.array(run, np.int32)], False, SERIES, 2, AA)
            elif run:
                cv2.circle(img, run[0], 2, SERIES, -1, AA)
            run = []

        off = finite & (y > y_max)
        for tv in t[off]:
            up_triangle(img, px(tv, 0)[0], top + 2, SERIES)
        if s.limit is not None:         # wrong-pose dots only on the charts that define "wrong"
            for tv, yv in zip(t[wrong & finite], y[wrong & finite]):
                p = px(tv, yv)
                cv2.circle(img, p, 5, SURFACE, -1, AA)
                cv2.circle(img, p, 4, CRITICAL, -1, AA)

        if finite.any():                # current value at the end of the line
            i = np.flatnonzero(finite)[-1]
            ex, ey = px(t[i], y[i])
            cv2.circle(img, (ex, ey), 4, SERIES, -1, AA)
            cv2.putText(img, f"{y[i]:{s.fmt}}", (ex + 10, ey + 5), FONT, 0.5, TEXT, 1, AA)

    def save(self, path: str, size: tuple[int, int] = (1600, 900)) -> None:
        """Write the whole run as a PNG (or any format cv2.imwrite understands)."""
        cv2.imwrite(path, self.render(*size, full=True))
