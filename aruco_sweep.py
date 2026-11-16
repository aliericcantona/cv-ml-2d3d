"""Where does the marker pose break? Sweeps the viewing angle at a few distances and plots
rotation error, position error and failure rate against it.

    python aruco_sweep.py [background] [--no-border] [--distances 1,2,3] [--noise 4] [--blur 0]
                          [--views 24] [--max-angle 80] [--step 5] [--out sweep.png]

Viewing angle = angle between the camera's line of sight and the marker's normal
(0 = looking straight down at it, 90 = edge-on). For every distance and angle the camera is
placed at `--views` evenly spaced azimuths around the marker, aimed at its centre.

A frame is a *failure* if the marker is not detected or its pose is wrong
(rotation > 5 deg or position > 20 mm, as in the benchmark). A *flip* is a wrong pose where the
other planar solution (IPPE returns two) would have been closer to the truth: the classic
square-marker ambiguity.

Needs matplotlib for the plot.
"""
import argparse

import cv2
import numpy as np

from aruco_tools.evaluator import DetectionStats
from aruco_tools.scenario import Scenario
from aruco_tools.sim import CameraState, OrbitCamera, add_sensor_effects
from aruco_tools.types import Pose, marker_corners_3d

COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]  # categorical, fixed order


def camera_at(dist: float, angle_deg: float, az: float) -> CameraState:
    """Camera `dist` m from the marker centre, `angle_deg` off its normal, at azimuth `az`, aimed at it."""
    a = np.radians(angle_deg)
    pos = dist * np.array([np.sin(a) * np.cos(az), np.sin(a) * np.sin(az), np.cos(a)])
    # straight down, "world up" is parallel to the view: use the azimuth direction as up instead
    up = (0.0, 0.0, 1.0) if angle_deg > 1 else (np.cos(az), np.sin(az), 0.0)
    return CameraState(OrbitCamera._look_at(pos, np.zeros(3), up), pos)


def is_flip(sc: Scenario, corners: np.ndarray, truth: Pose, chosen_rot_err: float) -> bool:
    """True if the other IPPE solution is closer to the truth than the one we got."""
    obj = marker_corners_3d(sc.marker_size_m).astype(np.float32)
    n, rvecs, tvecs, _ = cv2.solvePnPGeneric(obj, corners, sc.camera.K, sc.camera.dist,
                                             flags=cv2.SOLVEPNP_IPPE_SQUARE)
    errs = [Pose(r, t).error_to(truth)[0] for r, t in zip(rvecs, tvecs)]
    return n == 2 and min(errs) < chosen_rot_err - 1.0


def sweep(sc: Scenario, dist: float, angles, views: int, noise: float, blur: int) -> list[dict]:
    rows = []
    for angle in angles:
        stats, flips = DetectionStats(), 0
        for az in np.linspace(0, 2 * np.pi, views, endpoint=False):
            cam = camera_at(dist, angle, az)
            image = add_sensor_effects(sc.renderer.render(cam), sc.rng, noise_sigma=noise, blur_px=blur)
            det = sc.detector.detect(image).get(sc.marker_id)
            score = stats.add(det, cam.marker_pose, cam.sees_marker(sc.camera, sc.marker_size_m))
            if score.rot_err_deg is not None and score.rot_err_deg > stats.wrong_rot_deg:
                flips += is_flip(sc, det.corners, cam.marker_pose, score.rot_err_deg)
        rows.append(dict(angle=angle, frames=stats.frames, detected=stats.detected, wrong=stats.wrong,
                         flips=flips, failure=1 - stats.rate if stats.frames else np.nan,
                         rot_med=stats.rot_percentile(50), rot_p95=stats.rot_percentile(95),
                         pos_med=stats.pos_percentile(50), pos_p95=stats.pos_percentile(95)))
    return rows


def plot(results: dict[float, list[dict]], marker_px: dict[float, float], title: str, out: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter

    fig, (ax1, ax3, ax2) = plt.subplots(3, 1, figsize=(10, 10), sharex=True, height_ratios=(3, 3, 2))
    for ax in (ax1, ax3, ax2):
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color="#e5e4e0", linewidth=0.8)
        ax.tick_params(colors="#52514e")
    end_labels = []  # (x, y, text) for the failure-rate line ends, de-overlapped below
    for (dist, rows), color in zip(results.items(), COLORS):
        a = np.array([r["angle"] for r in rows], float)
        col = lambda k: np.array([np.nan if r[k] is None else r[k] for r in rows], float)
        med, p95 = col("rot_med"), col("rot_p95")
        pmed, pp95 = col("pos_med"), col("pos_p95")
        fail = 100 * np.array([r["failure"] for r in rows])
        flips = np.array([r["flips"] for r in rows])
        label = f"{dist:g} m (marker {marker_px[dist]:.0f} px)"
        ax1.fill_between(a, med, p95, color=color, alpha=0.12, linewidth=0)
        ax1.plot(a, med, color=color, linewidth=2, marker="o", markersize=4, label=label)
        ax1.plot(a[flips > 0], med[flips > 0], linestyle="none", marker="X", markersize=9,
                 color=color, markeredgecolor="white", markeredgewidth=1)
        ax3.fill_between(a, pmed, pp95, color=color, alpha=0.12, linewidth=0)
        ax3.plot(a, pmed, color=color, linewidth=2, marker="o", markersize=4)
        ax2.plot(a, fail, color=color, linewidth=2, marker="o", markersize=4, label=label)
        last = np.flatnonzero(np.isfinite(fail))
        if len(last):
            end_labels.append([a[last[-1]], fail[last[-1]], f"{dist:g} m"])
    end_labels.sort(key=lambda e: e[1])
    for prev, cur in zip(end_labels, end_labels[1:]):  # keep labels >= 9 points apart
        cur[1] = max(cur[1], prev[1] + 9)
    for x, y, text in end_labels:
        ax2.annotate(text, (x, y), xytext=(8, 0), textcoords="offset points", va="center",
                     color="#0b0b0b", fontsize=9, annotation_clip=False)

    ax1.set_yscale("log")
    ax1.axhline(5, color="#52514e", linestyle="--", linewidth=1)
    ax1.annotate("wrong-pose limit 5°", (1, 5), xycoords=("axes fraction", "data"), ha="right",
                 va="bottom", color="#52514e", fontsize=9)
    ax1.set_ylabel("rotation error (deg, log)\nline = median, band = to p95")
    ax1.plot([], [], linestyle="none", marker="X", color="#52514e", markersize=8, label="has pose flips")
    ax1.legend(frameon=False, fontsize=9, ncol=4, loc="lower left", bbox_to_anchor=(0, 1.0))
    ax3.set_yscale("log")
    ax3.axhline(20, color="#52514e", linestyle="--", linewidth=1)
    ax3.annotate("wrong-pose limit 20 mm", (1, 20), xycoords=("axes fraction", "data"), ha="right",
                 va="bottom", color="#52514e", fontsize=9)
    for ax in (ax1, ax3):  # plain numbers on the log axes (20, not 2x10^1)
        ax.yaxis.set_major_locator(LogLocator(subs=(1, 2, 5)))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        ax.yaxis.set_minor_formatter(NullFormatter())
    ax3.set_ylabel("position error (mm, log)\nline = median, band = to p95")
    ax2.set_ylabel("failure rate (%)\nmissed or wrong pose")
    ax2.set_ylim(-3, 103)
    ax2.set_xlabel("viewing angle from marker normal (deg)   0 = head-on, 90 = edge-on")
    fig.suptitle(title, x=0.01, ha="left", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"saved {out}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("background", nargs="?", default="grass")
    p.add_argument("--no-border", action="store_true", help="marker without the white margin")
    p.add_argument("--distances", default="1,2,3", help="camera distances in metres, comma-separated")
    p.add_argument("--noise", type=float, default=4.0, help="sensor noise sigma (grey levels)")
    p.add_argument("--blur", type=int, default=0, help="horizontal motion blur length in pixels")
    p.add_argument("--views", type=int, default=24, help="azimuths per angle and distance")
    p.add_argument("--max-angle", type=float, default=80)
    p.add_argument("--step", type=float, default=5)
    p.add_argument("--out", default="sweep.png", help="plot file")
    args = p.parse_args()

    sc = Scenario(args.background, border=not args.no_border, seed=1)
    distances = [float(d) for d in args.distances.split(",")]
    angles = np.arange(0, args.max_angle + 1e-9, args.step)
    marker_px = {d: sc.camera.K[0, 0] * sc.marker_size_m / d for d in distances}  # head-on width

    results = {}
    for d in distances:
        print(f"\n{d:g} m  (marker {marker_px[d]:.0f} px wide head-on)")
        print(f"{'angle':>6}{'detected':>10}{'wrong':>7}{'flips':>7}{'fail':>7}   rot err med / p95 (deg)")
        results[d] = sweep(sc, d, angles, args.views, args.noise, args.blur)
        for r in results[d]:
            rot = "-" if r["rot_med"] is None else f"{r['rot_med']:.2f} / {r['rot_p95']:.2f}"
            print(f"{r['angle']:>6.0f}{r['detected']:>6}/{r['frames']:<3}{r['wrong']:>7}{r['flips']:>7}"
                  f"{100 * r['failure']:>6.0f}%   {rot}")

    title = (f"Pose error vs viewing angle: {args.background}, margin {'off' if args.no_border else 'on'}, "
             f"noise σ={args.noise:g}" + (f", blur {args.blur}px" if args.blur else "")
             + f", {args.views} views per point")
    plot(results, marker_px, title, args.out)


if __name__ == "__main__":
    main()
