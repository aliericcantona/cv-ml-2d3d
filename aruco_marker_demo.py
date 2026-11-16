"""Camera orbits an ArUco marker on the ground; shows detected pose error vs ground truth.

    uv run aruco_marker_demo.py [grass|noise|checker|white|dark|path/to/image.jpg] [--no-border] [--cad bunny,armadillo] [--cad-every 3] [--overlay] [--error-plot [errors.png]] [--save out.mp4]

Right panel: the CAD part drawn on the marker at the *detected* pose, seen from the camera.
--cad takes model names from assets/ (bunny, armadillo, knot), STL files or folders, comma-separated;
with several models the part changes every --cad-every seconds. --cad alone, or no --cad,
runs every STL in assets/.
--overlay also draws it on top of the camera image (augmented-reality style).
--error-plot (alias --rmse-plot) adds a third panel with live charts of the rotation error,
position error (both against ground truth) and reprojection RMSE, and labels the errors on the
marker; with a file name it also saves the charts of the whole run when you quit.
--save records every shown frame (all panels) to an MP4 (30 fps) until you quit.

Press q or ESC to quit.
"""
import argparse
from pathlib import Path

import cv2
import numpy as np

from aruco_tools.cad import MeshRenderer, load_stl, place_on_marker
from aruco_tools.plot import ErrorPlot
from aruco_tools.scenario import Frame, Scenario

WINDOW = "ArUco demo"
FPS = 30


def put_lines(img: np.ndarray, lines: list[str]) -> None:
    for i, text in enumerate(lines):
        cv2.putText(img, text, (10, 30 + 30 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)


def label_errors(view: np.ndarray, frame: Frame) -> None:
    """Write the frame's pose error just above the marker."""
    s = frame.score
    if frame.detection is None or s.rot_err_deg is None:
        return
    x, y = frame.detection.corners.min(axis=0).astype(int)
    org = (int(x), max(20, int(y) - 12))
    text = f"rot {s.rot_err_deg:.2f} deg  pos {s.pos_err_mm:.1f} mm"
    cv2.putText(view, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(view, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)


MODEL_DIRS = (Path("assets"),)  # where bare model names and "every STL" come from


def stl_files(paths: list[str]) -> list[Path]:
    """Turn --cad arguments into STL files, in the order given. Each argument may hold several
    comma-separated items; an item is a folder (all .stl inside, sorted), a file, or a bare
    model name looked up in MODEL_DIRS ("bunny" -> assets/bunny.stl). No arguments = every
    STL in MODEL_DIRS."""
    if not paths:
        paths = [str(d) for d in MODEL_DIRS]
    files: list[Path] = []
    for item in (s.strip() for arg in paths for s in arg.split(",") if s.strip()):
        p = Path(item).expanduser()
        if p.is_dir():
            files += sorted(p.glob("*.stl"))
        elif p.is_file():
            files.append(p)
        else:
            found = [d / f"{item}.stl" for d in MODEL_DIRS if (d / f"{item}.stl").is_file()]
            if not found:
                known = sorted({f.stem for d in MODEL_DIRS for f in d.glob("*.stl")})
                raise SystemExit(f"--cad: no file or model named {item!r} (known models: {', '.join(known)})")
            files.append(found[0])
    if not files:
        raise SystemExit(f"no .stl files found in {paths}")
    return files


def draw(frame: Frame, scenario: Scenario, cad: MeshRenderer, overlay: bool,
         plot: ErrorPlot | None = None, cad_name: str = "") -> np.ndarray:
    """Camera view, CAD panel and (with `plot`) the RMSE chart, side by side."""
    view = scenario.detector.draw(frame.image, frame.result)
    if overlay and frame.pose:
        view = cad.render(frame.pose, image=view)
    if plot is not None:
        label_errors(view, frame)
    put_lines(view, [frame.score.describe(), scenario.stats.summary()])

    # right panel: zoomed in on the part, looking from the same direction as the camera
    panel = cad.render(cad.framed_pose(frame.pose) if frame.pose else None, scenario.marker_size_m)
    put_lines(panel, [f"{cad_name} at detected pose" if frame.pose else "no pose"])
    panels = [view, panel]
    if plot is not None:
        panels.append(plot.render(scenario.camera.width, scenario.camera.height))
    return np.hstack(panels)


class Recorder:
    """MP4 writer that opens on the first frame, once the frame size is known."""

    def __init__(self, path: str):
        self.path, self._video = path, None

    def write(self, image: np.ndarray) -> None:
        if self._video is None:
            h, w = image.shape[:2]
            self._video = cv2.VideoWriter(self.path, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (w, h))
        self._video.write(image)

    def close(self) -> None:
        if self._video is not None:
            self._video.release()
            print(f"saved {self.path}")


def main(background: str = "grass", border: bool = True, cad_paths: list[str] | str | None = None,
         cad_every: float = 3.0,
         overlay: bool = False, save_path: str | None = None, error_plot: str | None = None) -> None:
    """`error_plot`: None = off, "" = live charts only, a file name = live charts + saved PNG."""
    scenario = Scenario(background, border)
    files = stl_files([cad_paths] if isinstance(cad_paths, str) else cad_paths or [])
    models = [(f.stem, MeshRenderer(scenario.camera, place_on_marker(load_stl(f)))) for f in files]

    plot = (ErrorPlot(scenario.stats.wrong_rot_deg, scenario.stats.wrong_pos_mm)
            if error_plot is not None else None)
    n_panels = 3 if plot is not None else 2
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)  # resizable: panels side by side
    width = 2160 if plot is not None else 1600  # 3 panels: 2160 x 540
    cv2.resizeWindow(WINDOW, width, round(width * scenario.camera.height / (n_panels * scenario.camera.width)))
    video = Recorder(save_path) if save_path else None

    t = 0.0
    while cv2.waitKey(1) & 0xFF not in (ord("q"), 27):  # q or ESC quits
        t += 1 / FPS
        frame = scenario.step(t)
        if plot is not None:
            plot.add(t, frame.score)
        name, cad = models[int(t // cad_every) % len(models)]  # next model every `cad_every` s
        image = draw(frame, scenario, cad, overlay, plot, name)
        cv2.imshow(WINDOW, image)
        if video:
            video.write(image)

    if video:
        video.close()
    if plot is not None and error_plot:
        plot.save(error_plot)
        print(f"saved {error_plot}")
    cv2.destroyAllWindows()
    print(scenario.stats.summary())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ArUco marker demo")
    parser.add_argument("background", nargs="?", default="grass",
                        help="grass, noise, checker, white, dark, or an image path")
    parser.add_argument("--no-border", action="store_true", help="marker without the white border")
    parser.add_argument("--cad", nargs="*", default=[], metavar="MODELS",
                        help="comma-separated model names (bunny,armadillo), STL files or folders; alone = every STL")
    parser.add_argument("--cad-every", type=float, default=3.0, metavar="SECONDS",
                        help="with several models, switch to the next one every SECONDS")
    parser.add_argument("--overlay", action="store_true", help="also draw the part on the camera image")
    parser.add_argument("--error-plot", "--rmse-plot", dest="error_plot", nargs="?", const="",
                        metavar="PATH.png",
                        help="live rotation/position error and RMSE charts as a third panel; with a path, also save them on quit")
    parser.add_argument("--save", metavar="PATH.mp4", help="record the window to an MP4 file")
    args = parser.parse_args()
    main(args.background, border=not args.no_border, cad_paths=args.cad, cad_every=args.cad_every, overlay=args.overlay,
         save_path=args.save, error_plot=args.error_plot)
