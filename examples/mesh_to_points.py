import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt


def mesh_to_pointcloud(mesh, n_points=5000):
    points = np.asarray(mesh.vertices)

    rng = np.random.default_rng(100)
    indices = rng.choice(
        len(points),
        size=n_points,
        replace=False
    )

    return points[indices]


def myLambertian(
    mesh,
    light_dir=(0., 0., 1.),
    ka=0.2,
    Ia=1.0,
    kd=0.8,
    Id=1.0
):
    mesh.compute_vertex_normals()
    normals = np.asarray(mesh.vertex_normals)

    light_dir = np.asarray(light_dir, dtype=float)
    light_dir /= np.linalg.norm(light_dir)

    # Lambertian diffuse shading
    dot_products = normals @ light_dir
    diffuse = np.maximum(0.0, dot_products)

    intensity = np.clip(
        ka * Ia + kd * Id * diffuse,
        0.0,
        1.0
    )

    # Grayscale RGB colors
    colors = np.repeat(intensity[:, None], 3, axis=1)

    mesh.vertex_colors = o3d.utility.Vector3dVector(colors)

    return mesh


def shade_mesh(mesh, mode="lambertian", color=(0.7, 0.7, 0.7)):
    # Keep original mesh unchanged
    shaded_mesh = o3d.geometry.TriangleMesh(mesh)
    shaded_mesh.compute_vertex_normals()

    material = o3d.visualization.rendering.MaterialRecord()

    if mode == "lambertian":
        # Our own Lambertian implementation
        myLambertian(
            shaded_mesh,
            light_dir=(1., 1., 2.),
            ka=0.2,
            kd=0.8
        )

        # Unlit rendering: display our computed vertex colors
        material.shader = "defaultUnlit"

    elif mode == "open3d":
        # Open3D computes the lighting
        shaded_mesh.paint_uniform_color(color)
        material.shader = "defaultLit"

    else:
        raise ValueError("mode must be 'lambertian' or 'open3d'")

    o3d.visualization.draw([
        {
            "name": f"Stanford Bunny - {mode}",
            "geometry": shaded_mesh,
            "material": material
        }
    ])

    return shaded_mesh


def plot_pointcloud(points):
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")

    ax.scatter(
        points[:, 0],
        points[:, 1],
        points[:, 2],
        s=1
    )

    ax.set_box_aspect(np.ptp(points, axis=0))
    plt.show()


# -----------------------------
# MAIN
# -----------------------------

dataset = o3d.data.BunnyMesh()
mesh = o3d.io.read_triangle_mesh(dataset.path)

points = mesh_to_pointcloud(mesh, n_points=1000)

plot_pointcloud(points)

# SWITCH HERE:
shading_mode = "lambertian"  # "lambertian" or "open3d"

shaded_mesh = shade_mesh(mesh, mode=shading_mode)