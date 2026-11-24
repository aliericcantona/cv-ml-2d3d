"""Build three mechanical test parts with CadQuery and export them as STL files for the demo.

    uv run make_parts.py     # writes assets/spur_gear.stl, assets/mounting_bracket.stl, assets/bearing_housing.stl
    uv run aruco_marker_demo.py --cad assets/spur_gear.stl

Each part tests something different in pose estimation:
    spur_gear        - rotational symmetry (24 teeth: yaw is ambiguous every 15 deg)
    mounting_bracket - sharp edges and flat faces, no symmetry: a clean 6D-pose target
    bearing_housing  - complex geometry: bore, flange, ribs, bolt holes

The parts are made here instead of downloaded, so their dimensions are exact and known (mm),
they sit on z = 0, and they are centred on the origin in x and y.
"""
from pathlib import Path

import cadquery as cq
import numpy as np

OUT_DIR = Path("assets")
TOLERANCE_MM = 0.05     # STL chord tolerance; smaller = more triangles
ANGULAR_TOL = 0.2       # radians


def involute_tooth_profile(module: float, teeth: int, pressure_deg: float = 20.0,
                           pts_per_flank: int = 8) -> list[tuple[float, float]]:
    """Closed 2D outline of a standard involute spur gear (no profile shift)."""
    alpha = np.radians(pressure_deg)
    r_pitch = module * teeth / 2
    r_base = r_pitch * np.cos(alpha)
    r_tip = r_pitch + module
    r_root = r_pitch - 1.25 * module

    def inv(a):  # involute function
        return np.tan(a) - a

    # Half the tooth's angular thickness at the pitch circle, then at any radius r.
    half_pitch_thk = np.pi / (2 * teeth)

    def half_angle(r):
        a = np.arccos(np.clip(r_base / r, -1, 1))
        return half_pitch_thk + inv(alpha) - inv(a)

    r_start = max(r_base, r_root)
    radii = np.linspace(r_start, r_tip, pts_per_flank)
    pitch_angle = 2 * np.pi / teeth

    outline = []
    for i in range(teeth):
        c = i * pitch_angle
        # root point before the tooth, rising flank, tip, falling flank
        lead = c - half_angle(r_start)
        outline.append((r_root * np.cos(lead), r_root * np.sin(lead)))
        for r in radii:
            t = c - half_angle(r)
            outline.append((r * np.cos(t), r * np.sin(t)))
        for r in radii[::-1]:
            t = c + half_angle(r)
            outline.append((r * np.cos(t), r * np.sin(t)))
        trail = c + half_angle(r_start)
        outline.append((r_root * np.cos(trail), r_root * np.sin(trail)))
    return [(float(x), float(y)) for x, y in outline]


def spur_gear() -> cq.Workplane:
    """Module 5, 24 teeth -> 120 mm pitch diameter, 130 mm tip diameter, 20 mm face width."""
    m, z, width = 5.0, 24, 20.0
    gear = cq.Workplane("XY").polyline(involute_tooth_profile(m, z)).close().extrude(width)
    hub = cq.Workplane("XY").circle(25).extrude(width + 10)          # raised hub
    gear = gear.union(hub)
    gear = gear.faces(">Z").workplane().hole(20)                     # shaft bore
    # keyway: 6 mm wide, cut into the bore
    key = cq.Workplane("XY").center(10 + 1.5, 0).rect(5, 6).extrude(width + 10)
    gear = gear.cut(key)
    # four lightening holes in the web
    gear = (gear.faces("<Z").workplane()
            .polarArray(37, 45, 360, 4).hole(16, width))
    return gear


def mounting_bracket() -> cq.Workplane:
    """L-bracket: 120 x 80 base, 90 mm tall upright, 10 mm plate, with a gusset and slots."""
    t = 10.0
    base = cq.Workplane("XY").box(120, 80, t, centered=(True, True, False))
    upright = (cq.Workplane("XY").box(120, t, 90, centered=(True, False, False))
               .translate((0, -40, 0)))
    bracket = base.union(upright)
    # triangular gusset off-centre so the part has no mirror symmetry
    gusset = (cq.Workplane("YZ").polyline([(-40 + t, t), (-40 + t, 70), (20, t)]).close()
              .extrude(8).translate((25, 0, 0)))
    bracket = bracket.union(gusset)
    # two mounting holes in the base, one slot in the base
    bracket = (bracket.faces("<Z").workplane(centerOption="CenterOfBoundBox")
               .pushPoints([(-40, -15), (40, -15)]).hole(10))
    slot = (cq.Workplane("XY").center(-15, 20).slot2D(40, 10).extrude(t))
    bracket = bracket.cut(slot)
    # two holes of different sizes in the upright
    for x, z, d in [(-35, 60, 12), (30, 45, 8)]:
        hole = cq.Workplane("XZ").center(x, z).circle(d / 2).extrude(-t).translate((0, -40, 0))
        bracket = bracket.cut(hole)
    return bracket


def bearing_housing() -> cq.Workplane:
    """Flanged pillow-block style housing for a 6205 bearing (52 mm OD)."""
    flange_t, body_h = 12.0, 45.0
    flange = (cq.Workplane("XY").rect(140, 90).extrude(flange_t)
              .edges("|Z").fillet(15))
    body = cq.Workplane("XY").circle(40).extrude(body_h)
    housing = flange.union(body)
    # stepped bore: 52 mm bearing seat, 40 mm through bore, 30 mm shaft shoulder
    housing = housing.cut(cq.Workplane("XY").workplane(offset=body_h - 15).circle(26).extrude(15))
    housing = housing.cut(cq.Workplane("XY").circle(20).extrude(body_h))
    # four ribs between the body and the flange
    for ang in (0, 90, 180, 270):
        rib = (cq.Workplane("XZ").polyline([(38, flange_t), (60, flange_t), (38, body_h - 8)])
               .close().extrude(3).translate((0, 1.5, 0))
               .rotate((0, 0, 0), (0, 0, 1), ang + 45))
        housing = housing.union(rib)
    # bolt holes with counterbores on the flange
    bolts = [(-55, -30), (55, -30), (-55, 30), (55, 30)]
    housing = housing.cut(cq.Workplane("XY").pushPoints(bolts).circle(5.5).extrude(flange_t))
    housing = housing.cut(cq.Workplane("XY").workplane(offset=flange_t - 4)
                          .pushPoints(bolts).circle(9).extrude(4))
    # grease nipple boss on one side breaks the 4-fold symmetry
    boss = (cq.Workplane("YZ").workplane(offset=30).center(0, 30).circle(7).extrude(15))
    housing = housing.union(boss).cut(
        cq.Workplane("YZ").workplane(offset=30).center(0, 30).circle(2.5).extrude(16))
    return housing


PARTS = {
    "spur_gear": spur_gear,
    "mounting_bracket": mounting_bracket,
    "bearing_housing": bearing_housing,
}


def export(name: str, part: cq.Workplane) -> Path:
    bb = part.val().BoundingBox()
    part = part.translate((-(bb.xmin + bb.xmax) / 2, -(bb.ymin + bb.ymax) / 2, -bb.zmin))
    path = OUT_DIR / f"{name}.stl"
    cq.exporters.export(part, str(path), tolerance=TOLERANCE_MM, angularTolerance=ANGULAR_TOL)
    bb = part.val().BoundingBox()
    print(f"{path}: {bb.xlen:.0f} x {bb.ylen:.0f} x {bb.zlen:.0f} mm")
    return path


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)
    for name, build in PARTS.items():
        export(name, build())


if __name__ == "__main__":
    main()
