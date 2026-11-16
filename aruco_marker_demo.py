"""Camera orbits an ArUco marker on the ground; shows detected pose error vs ground truth.

    uv run aruco_marker_demo.py [grass|noise|checker|white|dark|path/to/image.jpg] [--no-border] [--cad examples/part.stl] [--overlay] [--save out.mp4]

Right panel: the CAD part drawn on the marker at the *detected* pose, seen from the camera.
--overlay also draws it on top of the camera image (augmented-reality style).
--save records every shown frame to an MP4 (30 fps) until you quit.

Press q or ESC to quit.
"""
import argparse

import cv2
import numpy as np

from aruco_tools.cad import MeshRenderer, load_stl, place_on_marker
from aruco_tools.scenario import Frame, Scenario

WINDOW = "ArUco demo"
FPS = 30


def put_lines(img: np.ndarray, lines: list[str]) -> None:
    for i, text in enumerate(lines):
        cv2.putText(img, text, (10, 30 + 30 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)


def draw(frame: Frame, scenario: Scenario, cad: MeshRenderer, overlay: bool) -> np.ndarray:
    """Camera view (left) and CAD panel (right), side by side."""
    view = scenario.detector.draw(frame.image, frame.result)
    if overlay and frame.pose:
        view = cad.render(frame.pose, image=view)
    put_lines(view, [frame.score.describe(), scenario.stats.summary()])

    # right panel: zoomed in on the part, looking from the same direction as the camera
    panel = cad.render(cad.framed_pose(frame.pose) if frame.pose else None, scenario.marker_size_m)
    put_lines(panel, ["CAD at detected pose" if frame.pose else "no pose"])
    return np.hstack([view, panel])


def main(background: str = "grass", border: bool = True, cad_path: str = "examples/part.stl",
         overlay: bool = False, save_path: str | None = None) -> None:
    scenario = Scenario(background, border)
    cad = MeshRenderer(scenario.camera, place_on_marker(load_stl(cad_path)))

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)  # resizable: both panels side by side
    cv2.resizeWindow(WINDOW, 1600, 600)
    video = None  # opened on the first frame, once the frame size is known

    t = 0.0
    while cv2.waitKey(1) & 0xFF not in (ord("q"), 27):  # q or ESC quits
        t += 1 / FPS
        image = draw(scenario.step(t), scenario, cad, overlay)
        cv2.imshow(WINDOW, image)
        if save_path:
            if video is None:
                h, w = image.shape[:2]
                video = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (w, h))
            video.write(image)

    if video is not None:
        video.release()
        print(f"saved {save_path}")
    cv2.destroyAllWindows()
    print(scenario.stats.summary())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ArUco marker demo")
    parser.add_argument("background", nargs="?", default="grass",
                        help="grass, noise, checker, white, dark, or an image path")
    parser.add_argument("--no-border", action="store_true", help="marker without the white border")
    parser.add_argument("--cad", default="examples/part.stl", help="STL file (millimetres) to show on the marker")
    parser.add_argument("--overlay", action="store_true", help="also draw the part on the camera image")
    parser.add_argument("--save", metavar="PATH.mp4", help="record the window to an MP4 file")
    args = parser.parse_args()
    main(args.background, border=not args.no_border, cad_path=args.cad, overlay=args.overlay,
         save_path=args.save)
