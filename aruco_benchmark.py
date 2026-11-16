"""Detection quality per background, with and without the white border.

    uv run aruco_benchmark.py [n_frames]

Only frames with the marker fully in view are scored.
  good   = detected with a correct pose (rot <= 5 deg and pos <= 20 mm)
  wrong  = detected, but the pose is worse than that
  rot/pos columns: median / 95th percentile / worst, over all detections (wrong ones included)
"""
import sys

import numpy as np

from aruco_tools.evaluator import DetectionStats
from aruco_tools.scenario import Scenario
from aruco_tools.sim import Background, make_ground


def spread(stats: DetectionStats, values: str, fmt: str) -> str:
    """'median / p95 / max' of rot or pos errors, or '-' if nothing was detected."""
    if not stats.detected:
        return "-"
    get = stats.rot_percentile if values == "rot" else stats.pos_percentile
    return " / ".join(format(get(q), fmt) for q in (50, 95, 100))


def main(n_frames: int = 120) -> None:
    times = np.linspace(0, 60, n_frames)
    print(f"{'background':<11}{'border':<8}{'good':>6}{'wrong':>7}{'skipped':>9}"
          f"   {'rot err deg (med/p95/max)':<27}{'pos err mm (med/p95/max)'}")
    for bg in Background:
        ground = make_ground(bg)  # build once, share between both border variants
        for border in (True, False):
            scenario = Scenario(ground, border, seed=1)
            for t in times:
                scenario.step(t)
            stats = scenario.stats
            print(f"{bg.value:<11}{'on' if border else 'off':<8}"
                  f"{100 * stats.rate:>5.0f}%{stats.wrong:>7}{stats.skipped:>9}"
                  f"   {spread(stats, 'rot', '.2f'):<27}{spread(stats, 'pos', '.1f')}")


if __name__ == "__main__":
    main(*map(int, sys.argv[1:2]))
