# ArUco prototype: how it works

A simulated camera circles an ArUco marker lying on the ground. Every frame we render
the image, detect the marker, estimate its 3D pose, and compare that pose with the
known true pose. The detected pose is also used to draw a CAD part on the marker.

## One frame, step by step

```
OrbitCamera ──► true camera pose
                    │
                    ▼
GroundPlaneRenderer ──► clean image ──► add_sensor_effects ──► noisy image
                                                                    │
                                                                    ▼
                                                     ArucoMarkerDetector
                                                     (corners + pose)
                                                                    │
                       ┌────────────────────────────────────────────┤
                       ▼                                            ▼
              DetectionStats                                 MeshRenderer
       (compare with true pose)                     (draw CAD part at the pose)
```

## Classes

All classes live in the `aruco_tools/` package.

### Basic types (`types.py`)

| Class | What it is |
|---|---|
| `ArucoDictType` | Which family of ArUco markers to use (e.g. `DICT_4X4_50`: 4×4 bit codes, 50 ids). |
| `CameraIntrinsics` | The camera's lens model: focal length, image centre, image size. `from_fov(w, h)` builds one from a field of view. |
| `marker_corners_3d` | The marker's 4 corner positions in 3D. The one definition used by detection, visibility checks and drawing. |
| `Pose` | Where the marker is relative to the camera: a rotation and a translation in metres. `error_to(other)` returns the rotation error (degrees) and position error (mm) between two poses. |
| `MarkerCard` | A generated marker image plus its real-world size, with or without a white margin. |
| `MarkerDetection` | One detected marker: its id, its 4 corner pixels, its pose, and the pose's reprojection RMSE in pixels. |
| `DetectionResult` | Everything found in one image: the detected markers plus rejected candidate shapes. |

### Making markers (`generator.py`)

**`ArucoMarkerGenerator`** draws a marker image for a given id and size. `border_ratio`
sets the white margin around it (`1/6` by default, `0` = no margin). The margin is what
lets the detector separate the marker's black edge from the background.

### Detecting markers (`detector.py`)

**`ArucoMarkerDetector`** finds markers and estimates their pose.

1. **Find candidates:** turn the image black/white and look for dark four-sided shapes.
2. **Read the code:** check the bit pattern inside each shape against the dictionary.
   Shapes that don't match are *rejected*.
3. **Refine corners** to sub-pixel accuracy.
4. **Estimate the pose:** from the 4 corners, the known marker size and the camera
   model, OpenCV's `solvePnP` (IPPE_SQUARE method) computes the rotation and position.
5. **Check the fit:** `reprojection_rmse()` projects the marker's 3D corners back into the
   image with that pose and returns the root-mean-square distance (pixels) to the detected
   corners. It needs no ground truth, so it is the quality number you would also have on a
   real camera. A low RMSE means the pose explains the corners well; it does not prove the
   pose is right. With only 4 corners and a 6-DoF pose, PnP fits them almost exactly, so the
   RMSE stays tiny even when the pose is off: on `grass` it is unrelated to the true rotation
   error (rank correlation -0.13). Use the rotation/position error against ground truth as the
   real metric; the RMSE only flags grossly misplaced corners.

`draw()` draws the outline and the 3D axes on the image.

### Simulating the scene (`sim.py`)

| Piece | What it does |
|---|---|
| `Background` / `make_ground` | Ground textures: grass, noise, checker, white, dark, or any image file. |
| `OrbitCamera` | A smooth camera path: circles the marker while moving up/down (22°–82°) and in/out (0.5–1.7 m). `at(t)` gives the camera at time `t`. |
| `CameraState` | One camera position on that path. `marker_pose` is the **true** pose (ground truth). `sees_marker()` says whether the whole marker is in the image. |
| `GroundPlaneRenderer` | Draws what the camera sees: for each pixel, finds where its ray hits the ground and samples the ground texture or the marker there. |
| `add_sensor_effects` | Adds random noise (and optional motion blur), like a real camera sensor. |

### Scoring (`evaluator.py`)

**`DetectionStats`** collects results frame by frame:

- Frames where the marker is not fully visible are **skipped** (not scored).
- **good:** detected, and the pose is within 5° and 20 mm of the truth.
- **wrong:** detected, but the pose is worse than that.
- Reports median, 95th percentile and worst errors, plus the same for the reprojection RMSE.

`add()` returns a **`FrameScore`**: the frame's status (`SKIPPED`, `MISSED`, `GOOD`, `WRONG`)
its errors and reprojection RMSE. `describe()` turns it into the one-line text shown in the demo.

### Running a setup (`scenario.py`)

**`Scenario`** puts the pieces together for one setup (background + margin on/off): camera,
marker, renderer, detector, orbit, noise and stats. `step(t)` runs one frame
(render → noise → detect → score) and returns a **`Frame`** with the true camera, the image,
the detections, the detected pose and the score. The demo draws frames; the benchmark only
counts them. The marker id, marker size and image size are defined here.

### Drawing the CAD part (`cad.py`)

| Piece | What it does |
|---|---|
| `load_stl` | Reads an STL file (binary or text) into a list of triangles; converts mm to metres. |
| `place_on_marker` | Centres the part on the marker and rests it on the ground. |
| `MeshRenderer` | Draws the triangles from a pose: hides back faces, shades each face by a fixed light, paints far triangles first. `framed_pose()` zooms in so the part fills the view while keeping the same viewing direction. |

## Scripts

| Script | What it does |
|---|---|
| `aruco_marker_demo.py` | Live demo: camera view on the left, CAD part on the right. Options: background, `--no-border`, `--cad`, `--overlay`, `--save`. |
| `aruco_benchmark.py` | Runs every background with and without the margin and prints a table of good/wrong rates and errors. |
| `make_models.py` | Exports the bunny, armadillo and knot models to `assets/` as STL files. |

## Why the pose shakes a little

Each frame is solved on its own from 4 noisy corners. Tilt is hardest to measure when
the camera looks almost straight down, so small corner errors become small rotation
errors that change every frame. The true camera path is perfectly smooth; the shaking
is the detection error. Smoothing over time would reduce it.
