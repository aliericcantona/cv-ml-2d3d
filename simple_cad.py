import cadquery as cq
import open3d as o3d

part = (
    cq.Workplane("XY")
    .box(100, 60, 10)
    .faces(">Z")
    .workplane()
    .hole(20)
)

cq.exporters.export(part, "examples/part.stl")

mesh = o3d.io.read_triangle_mesh("examples/part.stl")
mesh.compute_vertex_normals()

o3d.visualization.draw_geometries([mesh])