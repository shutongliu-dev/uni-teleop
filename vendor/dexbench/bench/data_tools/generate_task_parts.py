"""Generate and CPU-audit the original task asset library; no simulator import.

Run from the project root: .venv/bin/python -m bench.data_tools.generate_task_parts
Meshes are authored in metres, with explicit convex pieces preserving bores.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from scipy.spatial import ConvexHull
import trimesh

from bench.paths import DATA_ROOT
from bench.simulation.assets.task_parts import DEFAULT_ASSET_ROOT

IQUAT = [0., 0., 0., 1.]
COLORS = {"brass": [.68, .43, .12], "brass_edge": [.83, .63, .28],
          "steel": [.54, .60, .64], "dark": [.13, .17, .20],
          "blue": [.13, .34, .47], "blue_light": [.22, .48, .59],
          "rubber": [.065, .075, .078], "cream": [.84, .83, .73],
          "orange": [.80, .36, .105], "green": [.29, .48, .33]}


def mesh(vertices, faces):
    result = trimesh.Trimesh(np.asarray(vertices), np.asarray(faces), process=True)
    result.update_faces(result.nondegenerate_faces())
    result.remove_unreferenced_vertices()
    result.fix_normals(multibody=True)
    return result


def loft(rings, *, close_profile=False, caps=False):
    """Join equally sampled perimeter rings with outward winding."""
    rings = np.asarray(rings, dtype=float)
    n = rings.shape[1]
    vertices = rings.reshape(-1, 3).tolist()
    faces = []
    for k in range(len(rings) if close_profile else len(rings)-1):
        nk = (k+1) % len(rings)
        for j in range(n):
            nj = (j+1) % n
            faces.extend([[k*n+j, k*n+nj, nk*n+nj], [k*n+j, nk*n+nj, nk*n+j]])
    if caps:
        for k, reverse in ((0, True), (len(rings)-1, False)):
            center = len(vertices)
            vertices.append(rings[k].mean(axis=0).tolist())
            for j in range(n):
                a, b = k*n+j, k*n+(j+1) % n
                faces.append([center, b, a] if reverse else [center, a, b])
    return mesh(vertices, faces)


def lathe(profile, segments=96):
    """Closed radial-height boundary; positive inner radii produce open bores."""
    angles = np.arange(segments)*2*math.pi/segments
    rings = [np.c_[r*np.cos(angles), r*np.sin(angles), np.full(segments, z)] for r, z in profile]
    return loft(rings, close_profile=True)


def rounded_rectangle(hx, hy, bevel):
    # Eight corners make an actual flat chamfer rather than a shading illusion.
    b = min(bevel, hx*.45, hy*.45)
    return np.array([[hx-b, hy], [-hx+b, hy], [-hx, hy-b], [-hx, -hy+b],
                     [-hx+b, -hy], [hx-b, -hy], [hx, -hy+b], [hx, hy-b]])


def bevel_box(size, bevel=.001):
    hx, hy, hz = np.array(size)/2
    b = min(bevel, hz*.45, hx*.3, hy*.3)
    rings = []
    for z, inset in ((-hz, b), (-hz+b, 0), (hz-b, 0), (hz, b)):
        p = rounded_rectangle(hx-inset, hy-inset, max(b-inset, .00001))
        rings.append(np.c_[p, np.full(len(p), z)])
    return loft(rings, caps=True)


def convex(points):
    return trimesh.convex.convex_hull(np.asarray(points))


def annulus(outer, inner, height, bevel=.0007, segments=96):
    h, b = height/2, min(bevel, height*.24)
    return lathe([(inner+b, -h), (outer-b, -h), (outer, -h+b),
                  (outer, h-b), (outer-b, h), (inner+b, h),
                  (inner, h-b), (inner, -h+b)], segments)


def ring_wedges(outer, inner, height, count=24, z=0, bevel=.0003):
    """A union of convex ring sectors; no piece crosses the inner tangent circle."""
    result = []
    half, b = height/2, min(bevel, height*.2)
    # Inner endpoint radii are expanded so each straight inner chord is tangent
    # to the required clear bore, never an inscribed chord that narrows it.
    inner_vertex = inner/math.cos(math.pi/count)
    for i in range(count):
        angles = np.array([i, i+1])*2*math.pi/count
        points = []
        for zz, inset in ((-half, b), (-half+b, 0), (half-b, 0), (half, b)):
            for r in (inner_vertex+inset, outer-inset):
                points.extend(np.c_[r*np.cos(angles), r*np.sin(angles), np.full(2, zz+z)])
        result.append(convex(points))
    return result


def gear_outline(teeth, module=.004, pressure_deg=20):
    rp = teeth*module/2
    rb = rp*math.cos(math.radians(pressure_deg))
    ra, rr = rp+module, rp-1.25*module
    inv_pitch = math.tan(math.radians(pressure_deg))-math.radians(pressure_deg)

    def half_width(r):
        a = math.acos(min(1, rb/r))
        return math.pi/(2*teeth)+inv_pitch-(math.tan(a)-a)

    all_points, tooth_points = [], []
    for tooth in range(teeth):
        phi = tooth*2*math.pi/teeth
        width_base, width_top = half_width(rb), half_width(ra)
        # A 0.7 mrad root transition separates angular samples and avoids
        # collinear, zero-area triangles where the annular face meets a flank.
        polar = [(rr, phi-math.pi/teeth), (rr, phi-width_base-.0007)]
        polar.extend((r, phi-half_width(r)) for r in np.linspace(rb, ra, 7))
        polar.extend((ra, phi+a) for a in np.linspace(-width_top, width_top, 5)[1:])
        polar.extend((r, phi+half_width(r)) for r in np.linspace(ra, rb, 7)[1:])
        polar.append((rr, phi+width_base+.0007))
        pts = np.array([[r*math.cos(a), r*math.sin(a)] for r, a in polar])
        tooth_points.append(pts)
        all_points.extend(pts)
    return np.asarray(all_points), tooth_points, {"teeth": teeth, "module_m": module,
        "pressure_angle_deg": pressure_deg, "pitch_radius_m": rp,
        "addendum_radius_m": ra, "root_radius_m": rr}


def gear_mesh(teeth, inner=.011, height=.018):
    outer, tooth_points, params = gear_outline(teeth)
    r = np.linalg.norm(outer, axis=1)
    unit = outer/r[:, None]
    # Match the inner angular samples to the tooth outline. The annular band
    # is stitched directly to an inner wall; there is no polygon-fan end cap.
    h, b = height/2, .00075
    rings = []
    for radius, z in ((r-b, -h), (r, -h+b), (r, h-b), (r-b, h),
                      (np.full(len(r), inner+b), h), (np.full(len(r), inner), h-b),
                      (np.full(len(r), inner), -h+b), (np.full(len(r), inner+b), -h)):
        rings.append(np.c_[unit*radius[:, None], np.full(len(r), z)])
    display = loft(rings, close_profile=True)
    pieces = []
    # One convex sector per tooth keeps contact counts tractable. At the bore,
    # chords are circumscribed. Small root fillets are conservatively filled.
    for i, pts in enumerate(tooth_points):
        a0, a1 = (i-.5)*2*math.pi/teeth, (i+.5)*2*math.pi/teeth
        ri = inner/math.cos(math.pi/teeth)
        rr = params["root_radius_m"]
        boundary = np.vstack([pts, [rr*math.cos(a1), rr*math.sin(a1)],
                              [ri*math.cos(a1), ri*math.sin(a1)],
                              [ri*math.cos(a0), ri*math.sin(a0)]])
        points = []
        for z in (-h, h):
            points.extend(np.c_[boundary, np.full(len(boundary), z)])
        pieces.append(convex(points))
    return display, pieces, params


class Asset:
    def __init__(self, root, asset_id, dynamic, mass=0):
        self.root, self.name, self.dynamic, self.mass = root, asset_id, dynamic, mass
        self.visual = []
        self.collision = []
        self.data = {}

    def visual_mesh(self, value, color, label, position=(0, 0, 0)):
        value = value.copy()
        value.apply_translation(position)
        self.visual.append((label, value, COLORS.get(color, color)))

    def collision_mesh(self, value, label, position=(0, 0, 0)):
        self.collision.append((label, value, {"geometry": "convex_mesh", "position_m": list(position)}))

    def box(self, position, size, color, label, bevel=.001, collision=True):
        self.visual_mesh(bevel_box(size, bevel), color, label, position)
        if collision:
            self.collision.append((label, None, {"geometry": "box", "position_m": list(position),
                                               "half_extents_m": (np.asarray(size)/2).tolist()}))

    def cylinder(self, position, radius, height, color, label, collision=True):
        value = trimesh.creation.cylinder(radius, height, sections=64)
        self.visual_mesh(value, color, label, position)
        if collision:
            # MJWarp does not provide multicontact support for native
            # CYLINDER/MESH pairs. A faceted convex mesh supports flat shoulders.
            self.collision_mesh(convex(value.vertices), label, position)

    def write(self):
        dest = self.root/self.name
        (dest/"visual").mkdir(parents=True, exist_ok=True)
        (dest/"collision").mkdir(parents=True, exist_ok=True)

        def save(value, path):
            # Fixed precision preserves sub-micron task dimensions and stable hashes.
            path.write_text(trimesh.exchange.obj.export_obj(value, digits=10, include_normals=True))
            return {"mesh_path": str(path.relative_to(self.root)),
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "aabb_m": value.bounds.tolist(), "vertices": len(value.vertices),
                    "triangles": len(value.faces), "watertight": bool(value.is_watertight)}

        visuals, collisions = [], []
        for i, (label, value, color) in enumerate(self.visual):
            entry = save(value, dest/"visual"/f"{i:02d}_{label}.obj")
            entry.update(label=label, color=color, classification="visual_only", position_m=[0, 0, 0],
                         quaternion_xyzw=IQUAT, scale=[1, 1, 1])
            visuals.append(entry)
        combined = trimesh.util.concatenate([m for _, m, _ in self.visual])
        entry = save(combined, dest/"visual.obj")
        for i, (label, value, component) in enumerate(self.collision):
            component = dict(component, label=label, classification="physics",
                quaternion_xyzw=component.get("quaternion_xyzw", IQUAT), scale=[1, 1, 1])
            if value is not None:
                component.update(save(value, dest/"collision"/f"{i:02d}_{label}.obj"))
            collisions.append(component)
        size = combined.extents
        inertia = self.mass/12*np.array([size[1]**2+size[2]**2,
            size[0]**2+size[2]**2, size[0]**2+size[1]**2])
        result = {"mesh_path": entry["mesh_path"], "sha256": entry["sha256"], "aabb_m": combined.bounds.tolist(),
            "size_m": size.tolist(), "dynamic": self.dynamic, "mass_kg": self.mass,
            "center_of_mass_m": [0, 0, 0], "inertia_diag_kg_m2": inertia.tolist(),
            "inertia_method": "Declared uniform bounding-box approximation, not measured inertia" if self.dynamic else "Static fixture; no rigid mass",
            "origin": "symmetry centre / declared centre of mass" if self.dynamic else "base bottom centre",
            "visual_parts": visuals, "collision_parts": collisions, "friction": .65,
            "provenance": {"source": "Original procedural geometry authored for this sequence workcell",
                "source_kind": "original_procedural", "license": "CC0-1.0", "generator": "bench/data_tools/generate_task_parts.py",
                "dimension_status": "simulation task design; not manufacturer CAD"}}
        result.update(self.data)
        return result


def make_gear(root, name, teeth, mass, color):
    a = Asset(root, name, True, mass)
    visual, pieces, parameters = gear_mesh(teeth)
    a.visual_mesh(visual, color, "chamfered_involute_gear")
    # Shallow concentric machining witness band on each face, leaving the bore open.
    for side in (-1, 1):
        band = annulus(.017, .0158, .00018, .00003)
        a.visual_mesh(band, "brass_edge" if color == "brass" else "blue_light", f"witness_ring_{side}", [0, 0, side*.00902])
    for i, piece in enumerate(pieces):
        a.collision_mesh(piece, f"tooth_sector_{i:02d}")
    a.data.update(gear=parameters, bore={"axis": "z", "radius_m": .011,
        "guaranteed_collision_clear_radius_m": .011, "through": True},
        collision_approximation="One convex sector per tooth; root concavities and 0.75 mm face bevel filled conservatively; bore circumscribed to preserve clearance")
    if name == "gear_small":
        # Redundant analytic support lies wholly inside the existing nominal
        # annulus: inner square half-width 11.2 mm exceeds the circumscribed
        # 17-gon bore vertex radius (11.19 mm), while its 28.284 mm outer corner
        # radius is inside the root chord radius (28.50 mm). It adds no mass,
        # seals no bore, and makes flat support robust against native CCD
        # omissions for deeply intersecting thin mesh sectors on this backend.
        for sign in (-1, 1):
            for label, position, half in ((f"support_y_{sign}", [0, sign*.0156, 0], [.020, .0044, .009]),
                                          (f"support_x_{sign}", [sign*.0156, 0, 0], [.0044, .0112, .009])):
                a.collision.append((label, None, {"geometry": "box", "position_m": position, "half_extents_m": half}))
        a.data["collision_approximation"] += "; four redundant analytic box supports contained inside the existing annulus preserve the through bore and exterior envelope; density is zero and declared body mass/inertia remain unchanged"
        a.data["interior_support_frame"] = {"boxes": 4, "inner_square_half_width_m": .0112,
            "outer_square_half_width_m": .020, "half_height_m": .009,
            "maximum_outer_corner_radius_m": math.sqrt(2)*.020,
            "minimum_existing_root_chord_radius_m": parameters["root_radius_m"]*math.cos(math.pi/teeth),
            "maximum_existing_bore_vertex_radius_m": .011/math.cos(math.pi/teeth)}
    return a


def make_washer(root):
    a = Asset(root, "washer", True, .085)
    a.visual_mesh(annulus(.033, .011, .008, .00055), "steel", "beveled_washer")
    for i, p in enumerate(ring_wedges(.033, .011, .008, count=24, bevel=.0005)):
        a.collision_mesh(p, f"ring_sector_{i:02d}")
    a.data["bore"] = {"axis": "z", "radius_m": .011, "guaranteed_collision_clear_radius_m": .011, "through": True}
    return a


def make_spool(root):
    a = Asset(root, "spool", True, .17)
    profile = [(.0116, -.024), (.0288, -.024), (.030, -.0228), (.030, -.018),
        (.0288, -.017), (.023, -.017), (.022, -.016), (.022, .016),
        (.023, .017), (.0288, .017), (.030, .018), (.030, .0228),
        (.0288, .024), (.0116, .024), (.011, .0234), (.011, -.0234)]
    a.visual_mesh(lathe(profile), "orange", "hollow_spool")
    for sign in (-1, 1):
        a.visual_mesh(annulus(.027, .0245, .0002, .00002), "cream", f"flange_mark_{sign}", [0, 0, sign*.02402])
    # Three axial annuli retain the centre waist and the continuous shaft bore.
    for section, (radius, height, z) in enumerate(((.022, .034, 0), (.030, .007, -.0205), (.030, .007, .0205))):
        for i, p in enumerate(ring_wedges(radius, .011, height, 12, z, .0003)):
            a.collision_mesh(p, f"section_{section}_sector_{i:02d}")
    a.data["bore"] = {"axis": "z", "radius_m": .011, "guaranteed_collision_clear_radius_m": .011, "through": True}
    return a


def make_bar(root):
    a = Asset(root, "press_bar", True, .16)
    a.box([0, 0, -.001], [.136, .022, .018], "steel", "grip_core", .0014)
    # Real recessed grip bands: raised convex ribs leave 1.4 mm gaps, with the
    # core recessed by 1 mm underneath. Separate end caps are easy to grasp.
    for i in range(11):
        x = (i-5)*.011
        a.box([x, 0, 0], [.0096, .024, .020], "dark", f"grip_rib_{i:02d}", .0006)
    for sign in (-1, 1):
        a.box([sign*.071, 0, 0], [.018, .028, .024], "orange", f"end_cap_{sign}", .0022)
        a.box([sign*.071, 0, .0121], [.007, .014, .0003], "cream", f"cap_inlay_{sign}", .00005, collision=False)
    # Cloth cells must not tunnel between the narrow visual grip ribs or end
    # caps. One chamfered envelope provides a smooth, continuous contact shell;
    # grip details remain display geometry and no functional bore is removed.
    a.collision = []
    a.collision_mesh(convex(bevel_box([.160, .028, .024], .002).vertices), "cloth_safe_shell")
    a.data["grasp_regions"] = [{"center_m": [0, 0, 0], "size_m": [.110, .024, .020], "approach": "top or either long side"}]
    a.data["collision_approximation"] = "Cloth-safe single convex 160 x 28 x 24 mm envelope with 2 mm edge chamfers; visual grip grooves and spaces between narrow end-cap strips are filled; no functional through hole is removed"
    return a


def make_wrap_cylinder(root):
    a = Asset(root, "wrap_cylinder", True, .20)
    # Rotate its longitudinal local z axis onto x; it rests sideways for wrapping.
    value = lathe([(.00001, -.050), (.0225, -.050), (.024, -.0485), (.024, .0485),
                   (.0225, .050), (.00001, .050)], 96)
    rotation = trimesh.transformations.rotation_matrix(math.pi/2, [0, 1, 0])
    value.apply_transform(rotation)
    a.visual_mesh(value, "blue", "chamfered_workpiece")
    for sign in (-1, 1):
        end = trimesh.creation.cylinder(.018, .0003, sections=64)
        end.apply_transform(rotation)
        a.visual_mesh(end, "steel", f"end_face_{sign}", [sign*.0501, 0, 0])
    # A convex, faceted cylinder supports multi-CCD on the pinned GPU backend;
    # native CYLINDER/MESH pairs only generate one contact. The x-axis and real
    # bevel are preserved. The solver must explicitly enable multi-CCD.
    a.collision_mesh(convex(value.vertices), "chamfered_cylinder")
    a.data["long_axis"] = "x"
    return a


def make_tray(root, name, size, divided=False):
    a = Asset(root, name, False)
    sx, sy = size
    floor, wall, height = .006, .004, .024
    a.box([0, 0, floor/2], [sx, sy, floor], "blue", "floor", .0012)
    for sign in (-1, 1):
        a.box([sign*(sx-wall)/2, 0, (floor+height)/2], [wall, sy, height-floor], "blue", f"end_wall_{sign}", .0009)
        a.box([0, sign*(sy-wall)/2, (floor+height)/2], [sx-2*wall, wall, height-floor], "blue", f"side_wall_{sign}", .0009)
    if divided:
        for x in (-.075, .075):
            a.box([x, 0, .011], [wall, sy-2*wall, .010], "blue_light", f"divider_{x}", .0008)
        for i, (x, r, color) in enumerate(((-.15, .050, "brass_edge"), (0, .035, "steel"), (.15, .032, "orange"))):
            a.visual_mesh(annulus(r, r-.0012, .00016, .00002), color, f"placement_outline_{i}", [x, 0, floor+.0001])
            # Legible one/two/three embossed dots identify matching bays.
            for d in range(i+1):
                a.cylinder([x+(d-i/2)*.006, -sy/2+.012, floor+.0005], .0016, .0006, "cream", f"bay_{i}_dot_{d}", collision=False)
        a.data["slot_centers_m"] = [[-.15, 0, floor], [0, 0, floor], [.15, 0, floor]]
        a.data["slot_labels"] = ["gear_large", "washer", "spool"]
        a.data["slot_clear_size_xy_m"] = [[.146, sy-2*wall], [.146, sy-2*wall], [.146, sy-2*wall]]
    else:
        a.data["interior_bounds_m"] = [[-sx/2+wall, -sy/2+wall, floor], [sx/2-wall, sy/2-wall, height]]
    a.data.update(floor_height_m=floor, rim_top_m=height, wall_thickness_m=wall,
                  collision_approximation="Box components; submillimetre edge chamfers are display detail")
    return a


def make_fixture(root):
    a = Asset(root, "dual_shaft_fixture", False)
    a.box([0, 0, .007], [.225, .135, .014], "steel", "chamfered_base", .0024)
    for ix in (-1, 1):
        for iy in (-1, 1):
            p = [ix*.096, iy*.052, .016]
            # Hex bolt heads and contrasting washers are actual separate meshes.
            a.visual_mesh(annulus(.008, .0035, .0018, .0002), "dark", f"mount_washer_{ix}_{iy}", [p[0], p[1], .0149])
            head = trimesh.creation.cylinder(.0056, .004, sections=6)
            a.visual_mesh(head, "steel", f"hex_bolt_{ix}_{iy}", p)
            a.cylinder(p, .0056, .004, "steel", f"bolt_collision_{ix}_{iy}", collision=True)
    centers = [[-.041, 0], [.041, 0]]
    for i, (x, y) in enumerate(centers):
        a.cylinder([x, y, .037], .006, .046, "steel", f"shaft_{i}")
        tip = lathe([(.00001, 0), (.006, 0), (.0045, .003), (.00001, .003)], 48)
        a.visual_mesh(tip, "steel", f"tapered_tip_{i}", [x, y, .060])
        a.collision_mesh(convex(tip.vertices), f"shaft_tip_{i}", [x, y, .060])
    # The large gear uses a loose 8 mm washer; the small gear uses this fixed
    # 8 mm shoulder, yielding the same final centre height of 31 mm.
    a.cylinder([.041, 0, .018], .018, .008, "dark", "small_gear_shoulder")
    a.data.update(shaft_centers_m=[[x, y, .014] for x, y in centers], shaft_radius_m=.006,
        shaft_top_m=.063, gear_center_distance_m=.082, nominal_pitch_center_distance_m=.078,
        center_distance_clearance_m=.004, assembly_surface_z_m=.014,
        assembled_poses={"washer": {"position_m": [-.041, 0, .018], "quaternion_xyzw": IQUAT},
            "gear_large": {"position_m": [-.041, 0, .031], "quaternion_xyzw": IQUAT},
            "gear_small": {"position_m": [.041, 0, .031], "quaternion_xyzw": IQUAT}},
        design_note="Loose insertion and visual gear engagement task, not a power-transmission or precision-fit model; shaft diameter 12 mm versus bore 22 mm; gear pitch centre distance expanded by 4 mm")
    return a


def make_cradle(root):
    a = Asset(root, "cylinder_cradle", False)
    a.box([0, 0, .003], [.140, .090, .006], "steel", "base", .0014)
    for sign in (-1, 1):
        # These rounded low guides allow loose insertion and cloth draping.
        a.box([0, sign*.029, .008], [.110, .004, .004], "dark", f"lateral_guide_{sign}", .0013)
        a.box([sign*.057, 0, .0085], [.004, .054, .005], "dark", f"axial_stop_{sign}", .0016)
        for ysign in (-1, 1):
            # Open annular counterbore mouths are display detail above a base;
            # they mark blind mounting bores, rather than pretending through holes.
            p = [sign*.061, ysign*.035, .0061]
            a.visual_mesh(annulus(.0036, .0022, .0002, .00003), "dark", f"blind_mount_bore_{sign}_{ysign}", p)
    a.data.update(contact_floor_z_m=.006, guide_top_z_m=.011,
        lateral_clear_width_m=.054, axial_clear_length_m=.110,
        target_poses={"wrap_cylinder": {"position_m": [0, 0, .030], "quaternion_xyzw": IQUAT}},
        design_note="Loose cradle with 6 mm base and 4/5 mm guide heights above the contact floor; blind mount-hole mouth marks are visual only")
    return a


def make_pickup_pedestal(root):
    a = Asset(root, "pickup_pedestal", False)
    profile = [(.00001, 0), (.0212, 0), (.022, .0008), (.022, .0022),
        (.0212, .003), (.020, .003), (.020, .0075), (.0194, .008),
        (.0194, .0088), (.020, .0093), (.020, .0133), (.0194, .0138),
        (.0194, .0146), (.020, .0151), (.020, .0195), (.0195, .020),
        (.00001, .020)]
    value = lathe(profile)
    a.visual_mesh(value, "dark", "turned_support")
    a.collision_mesh(convex(value.vertices), "convex_support")
    a.visual_mesh(annulus(.0218, .0205, .00018, .00003), "blue_light", "base_witness_ring", [0, 0, .0025])
    a.box([0, -.02015, .0112], [.010, .00035, .0038], "cream", "height_label", .00005, collision=False)
    for x in (-.002, .002):
        a.box([x, -.0204, .0112], [.0007, .0002, .0024], "blue", f"label_mark_{x}", .00004, collision=False)
    a.data.update(support_plane_z_m=.020, nominal_support_diameter_m=.040,
        flat_support_diameter_m=.039, base_diameter_m=.044,
        placement_poses={"gear_large": {"position_m": [0, 0, .029], "quaternion_xyzw": IQUAT},
            "gear_small": {"position_m": [0, 0, .029], "quaternion_xyzw": IQUAT},
            "washer": {"position_m": [0, 0, .024], "quaternion_xyzw": IQUAT}},
        qualified_supported_assets=["gear_large", "gear_small", "washer"],
        collision_approximation="Single convex mesh preserves support face and base chamfers; two shallow turned witness grooves are visual detail",
        purpose="Lift source parts 20 mm for side grasping; a 66 mm washer overhangs the 39 mm flat support by 13.5 mm")
    return a


def _collision_vertices(part, root):
    q = part.get("quaternion_xyzw", IQUAT)
    transform = trimesh.transformations.quaternion_matrix([q[3], *q[:3]])
    transform[:3, 3] = part["position_m"]
    if part["geometry"] == "convex_mesh":
        v = trimesh.load(root/part["mesh_path"], force="mesh", process=False).vertices
    elif part["geometry"] == "box":
        v = trimesh.creation.box(np.array(part["half_extents_m"])*2).vertices
    else:
        v = trimesh.creation.cylinder(part["radius_m"], part["half_height_m"]*2, sections=48).vertices
    return trimesh.transform_points(v, transform)


def _bore_clearance(points):
    # Distance of the origin to a convex polygon in XY. If the origin lies
    # inside the hull, the piece obstructs the bore and the clearance is zero.
    points = np.unique(np.asarray(points)[:, :2], axis=0)
    hull = ConvexHull(points)
    if np.max(hull.equations[:, 2]) <= 1e-12:
        return 0.
    p = points[hull.vertices]
    a, b = p, np.roll(p, -1, axis=0)
    direction = b-a
    t = np.clip(-np.sum(a*direction, axis=1)/np.sum(direction**2, axis=1), 0, 1)
    return float(np.min(np.linalg.norm(a+t[:, None]*direction, axis=1)))


def _polygon_overlap(a, b, tolerance=1e-9):
    a = a[ConvexHull(a).vertices]
    b = b[ConvexHull(b).vertices]
    for p in (a, b):
        edges = np.roll(p, -1, axis=0)-p
        axes = np.c_[-edges[:, 1], edges[:, 0]]
        axes /= np.linalg.norm(axes, axis=1)[:, None]
        for axis in axes:
            aa, bb = a@axis, b@axis
            if aa.max() <= bb.min()+tolerance or bb.max() <= aa.min()+tolerance:
                return False
    return True


def audit(root, assets):
    results = {}
    for name, asset in assets.items():
        bad_visual, bad_collision, total_triangles = [], [], 0
        for part in asset["visual_parts"]:
            m = trimesh.load(root/part["mesh_path"], force="mesh", process=True)
            total_triangles += len(m.faces)
            if not m.is_watertight or not m.is_winding_consistent or not np.isfinite(m.vertices).all() or m.volume <= 0:
                bad_visual.append(part["label"])
        clearance = math.inf
        for part in asset["collision_parts"]:
            pts = _collision_vertices(part, root)
            if "bore" in asset:
                clearance = min(clearance, _bore_clearance(pts))
            if part["geometry"] == "convex_mesh":
                m = trimesh.load(root/part["mesh_path"], force="mesh", process=True)
                if not m.is_watertight or not m.is_convex or m.volume <= 0:
                    bad_collision.append(part["label"])
        if bad_visual or bad_collision:
            raise ValueError(f"Invalid topology in {name}: visuals={bad_visual}, collisions={bad_collision}")
        if "bore" in asset and clearance < asset["bore"]["radius_m"]-1e-8:
            raise ValueError(f"Blocked/narrowed bore in {name}: {clearance}")
        results[name] = {"visual_closed_manifold_and_positive_volume": True, "collision_all_convex": True,
            "visual_triangles": total_triangles, "collision_parts": len(asset["collision_parts"]),
            "minimum_collision_bore_clear_radius_m": None if math.isinf(clearance) else clearance}
    # Check that the proposed final gear pose contains no collision-sector
    # overlaps in XY. Both gears occupy the same Z interval after assembly.
    poses = assets["dual_shaft_fixture"]["assembled_poses"]
    left = [_collision_vertices(p, root)[:, :2]+np.array(poses["gear_large"]["position_m"][:2]) for p in assets["gear_large"]["collision_parts"]]
    right = [_collision_vertices(p, root)[:, :2]+np.array(poses["gear_small"]["position_m"][:2]) for p in assets["gear_small"]["collision_parts"]]
    overlap = [(i, j) for i, a in enumerate(left) for j, b in enumerate(right) if _polygon_overlap(a, b)]
    if overlap:
        raise ValueError(f"Assembled gear collision sectors overlap: {overlap}")
    return {"method": "CPU mesh topology, convex hull checks, exact XY convex-polygon bore clearance, and gear-sector separating-axis assembly test",
        "dynamic_simulation_performed": False, "units": "metres", "assets": results,
        "assembled_gears": {"overlapping_collision_sector_pairs": overlap, "yaw_rad": [0, 0],
            "shaft_to_bore_radial_clearance_m": .005, "gear_bottom_z_m": .022,
            "washer_bottom_z_m": .014, "washer_top_z_m": .022,
            "small_shoulder_top_z_m": .022, "status": "geometrically_clear"}}


def generate(root):
    root.mkdir(parents=True, exist_ok=True)
    candidates = [make_gear(root, "gear_large", 22, .22, "brass"),
        make_gear(root, "gear_small", 17, .14, "blue"), make_washer(root), make_spool(root),
        make_bar(root), make_wrap_cylinder(root),
        make_tray(root, "tray_sorting", [.45, .15], True),
        make_tray(root, "tray_lining", [.34, .30]), make_fixture(root), make_cradle(root),
        make_pickup_pedestal(root)]
    assets = {a.name: a.write() for a in candidates}
    report = audit(root, assets)
    (root/"audit.json").write_text(json.dumps(report, indent=2)+"\n")
    manifest = {"schema_version": 1, "asset_version": "sequence_workcell/v1", "units": "metres",
        "quaternion_order": "xyzw", "license": "CC0-1.0", "created_date": "2026-10-09",
        "generator": {"path": str(Path(__file__).resolve()), "sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "numpy_version": np.__version__, "trimesh_version": trimesh.__version__},
        "physics_requirements": {"rigid_contact_gap_m": .001, "rigid_multiccd": True,
            "rigid_contact_capacity": 8192, "rigid_constraint_capacity": 16384,
            "qualification_dt_s": .0005, "backend": "pinned Newton 1.6.1 / MuJoCo"},
        "assets": assets, "fixed_layouts": {
            "sort_tray": {"asset_id": "tray_sorting", "slots": assets["tray_sorting"]["slot_centers_m"]},
            "assemble": {"asset_id": "dual_shaft_fixture", "poses": assets["dual_shaft_fixture"]["assembled_poses"]}},
        "audit": {"path": "audit.json", "sha256": hashlib.sha256((root/"audit.json").read_bytes()).hexdigest()}}
    (root/"manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    (root/"LICENSE.txt").write_text("CC0 1.0 Universal\n\nOriginal procedural geometry in this asset directory is dedicated to the public domain under CC0 1.0.\nNo downloaded third-party meshes are incorporated.\nCanonical legal code: https://creativecommons.org/publicdomain/zero/1.0/legalcode\n")
    (root/"README.txt").write_text(f"Sequence workcell task parts v1\n\nOriginal procedural metre-scale meshes and independent convex collision components.\nSource generator: {DATA_ROOT}/interface/bench/data_tools/generate_task_parts.py\nLoader: bench.simulation.assets.task_parts.load_task_asset\nEnvironment override: SEQUENCE_TASK_ASSETS\n\nNever convexify a complete gear, washer, or spool: doing so blocks its through bore.\nUse every collision_parts entry on one body. Visual_parts have collision disabled.\nDynamic mass/inertia values are explicit simulation approximations, not measurements.\nStatic tray and fixture origins are base-bottom centres; dynamic origins are centres of mass.\nThe 22/17-tooth gears use module 4 mm, 20-degree pressure-angle involute silhouettes.\nCollision roots and edge bevels are conservative approximations. Each gear uses one\nconvex sector per tooth; every inner chord clears the specified 11 mm bore radius.\nDual shafts have radius 6 mm and 82 mm centre spacing (4 mm extra clearance).\nAssembly poses are geometrically checked, not a demonstrated manipulation rollout.\n\nSHA-256, topology, AABB, provenance, dimensions, mass, COM, inertia, per-part\nvisual/physics classification, slot centres, and assembly poses are in manifest.json.\nDetailed CPU checks are in audit.json. Assets are CC0-1.0; no external mesh source.\n")
    return {"root": str(root), "assets": sorted(assets), "manifest_sha256": hashlib.sha256((root/"manifest.json").read_bytes()).hexdigest(), "audit": report}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset-root", type=Path, default=DEFAULT_ASSET_ROOT)
    args = parser.parse_args()
    print(json.dumps(generate(args.asset_root), indent=2))


if __name__ == "__main__":
    main()
