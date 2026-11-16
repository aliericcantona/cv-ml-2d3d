# cv-2d3d-ml

A simulated camera orbits an ArUco marker on the ground. Each frame the marker is detected,
its 3D pose is compared with the true pose, and a CAD part is drawn on it.

How it works, class by class: [docs/architecture.md](docs/architecture.md)

## Requirements

Python 3.12+ with `opencv-python` and `numpy`. `open3d` is only needed for `make_models.py`.

## Run

```
# live demo (q or ESC to quit)
uv run aruco_marker_demo.py
uv run aruco_marker_demo.py checker --cad assets/bunny.stl --overlay --save /tmp/aruco_bunny_checker.mp4
uv run aruco_marker_demo.py noise --no-border --cad bunny --overlay --error-plot errors.png
uv run aruco_marker_demo.py --overlay --error-plot --save run.mp4               # every STL in assets/ takes a turn: armadillo, bunny, knot
uv run aruco_marker_demo.py --cad bunny,armadillo --overlay --error-plot         # just these two

# accuracy table for every background, with and without the white margin
uv run aruco_benchmark.py

# where does the pose break? error and failure rate vs viewing angle (0-80 deg) at 1, 2, 3 m
# (needs matplotlib; ~2.5 min)
uv run aruco_sweep.py --out sweep.png
uv run aruco_sweep.py noise --no-border --blur 5 --distances 1,2 --out sweep_hard.png

# export bunny / armadillo / knot STL files to assets/ (once)
uv run make_models.py
```

### Demo options

| Option | What it does | Default |
|---|---|---|
| `checker` | Background: `grass`, `noise`, `checker`, `white`, `dark`, or an image path | `grass` |
| `--no-border` | Marker without the white margin | margin on |
| `--cad bunny,armadillo` | Models to show on the marker, comma-separated: names from `assets/` (`bunny`, `armadillo`, `knot`), STL files (in mm) or folders; several models take turns. `--cad` alone runs every STL | every STL in `assets/` |
| `--cad-every SECONDS` | With several models, switch to the next one this often | `3` |
| `--overlay` | Also draw the part on the camera image | off |
| `--error-plot [FILE.png]` | Adds a third panel with live charts of rotation error, position error (vs ground truth) and reprojection RMSE, and writes the errors on the marker (`--rmse-plot` also works); with a file name, also saves the whole-run chart on quit | off |
| `--save FILE.mp4` | Record the window; the file is completed when you press q. Includes the error panel when `--error-plot` is on | off |

## Layout

```
aruco_tools/            the library
  types.py              camera, pose, marker and detection types
  generator.py          makes marker images (with or without a white margin)
  detector.py           finds markers and estimates their pose
  sim.py                backgrounds, orbiting camera, renderer, sensor noise
  evaluator.py          DetectionStats: detection rate and pose error vs ground truth
  scenario.py           Scenario.step(t): render → noise → detect → score one frame
  cad.py                loads STL files and draws them at a pose
  plot.py               ErrorPlot: pose errors and RMSE over time (live charts / PNG)
aruco_marker_demo.py    live demo
aruco_benchmark.py      benchmark table
aruco_sweep.py          error vs viewing angle and distance, plotted
make_models.py          exports test models to assets/
assets/                 STL models
docs/                   documentation
```
