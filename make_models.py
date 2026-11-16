"""Export classic 3D test models (from Open3D's sample data) as STL files for the demo.

    uv run make_models.py            # writes assets/bunny.stl, assets/armadillo.stl, assets/knot.stl
    uv run aruco_marker_demo.py --cad assets/bunny.stl

Open3D downloads each model the first time. Every model is reduced to ~4000 triangles (so the
demo stays fast), turned to stand upright on the marker, and scaled to SIZE_MM across, in mm.
"""
from pathlib import Path

import numpy as np
import open3d as o3d

SIZE_MM = 150         # longest side of each model; the marker is 187.5 mm
TRIANGLES = 4000
OUT_DIR = Path("assets")

Y_UP_TO_Z_UP = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], float)  # these scans are y-up

MODELS = {  # name: (Open3D dataset, rotation to stand it on the marker)
    "bunny": (o3d.data.BunnyMesh, Y_UP_TO_Z_UP),
    "armadillo": (o3d.data.ArmadilloMesh, Y_UP_TO_Z_UP),
    "knot": (o3d.data.KnotMesh, np.eye(3)),
}


def export(name: str, dataset, rotation: np.ndarray) -> Path:
    mesh = o3d.io.read_triangle_mesh(dataset().path)
    mesh = mesh.simplify_quadric_decimation(TRIANGLES)

    v = np.asarray(mesh.vertices) @ rotation.T
    v = (v - v.min(0)) * (SIZE_MM / np.ptp(v, axis=0).max())
    mesh.vertices = o3d.utility.Vector3dVector(v)
    mesh.compute_triangle_normals()  # STL stores one normal per triangle

    path = OUT_DIR / f"{name}.stl"
    o3d.io.write_triangle_mesh(str(path), mesh)
    print(f"{path}: {len(mesh.triangles)} triangles")
    return path


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)
    for name, (dataset, rotation) in MODELS.items():
        export(name, dataset, rotation)


if __name__ == "__main__":
    main()
