"""Third-floor residential view: private balcony, lower court, opposite flats.

The room's z=0 floor is 6.2 m above the courtyard. Architecture, openings,
balconies and courtyard are real dimensional scenery; the HDR supplies sky only.
Original shared materials and plants are referenced without source mutation.
"""
from pathlib import Path

import numpy as np
from pxr import Usd, UsdGeom, UsdPhysics

from bench.paths import load_json


HDR_RELATIVE = (
    "skies/polyhaven/v2/kloofendal_48d_partly_cloudy_puresky/"
    "kloofendal_48d_partly_cloudy_puresky_2k.hdr"
)
HDR_ROTATE_XYZ_DEG = (0., 0., 0.)
COURTYARD_Z = -6.20
BALCONY_Z = -.018
OPPOSITE_FRONT_X = -16.0


def _materials(b):
    root = b.assets_root / "materials/ambientcg/v1/Concrete048"
    b.dependency(root / "SOURCE.json")
    texture = str(root / "Concrete048_1K-JPG_Color.jpg")
    normal = str(root / "Concrete048_1K-JPG_NormalGL.jpg")
    rough = str(root / "Concrete048_1K-JPG_Roughness.jpg")
    for index, tint in enumerate(((.72, .71, .66), (.69, .69, .65),
                                  (.75, .74, .69), (.70, .71, .68))):
        b.material("exterior_paving_" + str(index), (.47, .47, .43), .87,
                   texture=texture, normal=normal, roughness_texture=rough,
                   texture_tint=tint)
    b.material("exterior_concrete", (.36, .37, .34), .89, normal=normal)
    b.material("exterior_galvanized", (.16, .18, .18), .40, .82)
    b.material("exterior_graphite", (.055, .064, .059), .53, .62)
    b.material("exterior_soil", (.072, .065, .042), .99, normal=normal)
    b.material("exterior_grout", (.26, .28, .26), .95)
    plaster = b.assets_root / "materials/polyhaven/v1/white_plaster_02"
    b.dependency(plaster / "SOURCE.json")
    for name, color in (("render", (.69, .68, .62)),
                        ("side_render", (.55, .57, .53)),
                        ("sill", (.49, .50, .46)),
                        ("curtain_0", (.47, .46, .39)),
                        ("curtain_1", (.37, .40, .38)),
                        ("curtain_2", (.60, .57, .49))):
        b.material("exterior_" + name, color, .85,
                   normal=str(plaster / "white_plaster_02_nor_gl_1k.jpg"))
    b.material("exterior_window", (.12, .17, .19), .17, .32, opacity=.44)
    b.material("exterior_interior", (.055, .060, .056), .94)
    b.material("exterior_clay", (.38, .28, .20), .85, normal=normal)


def _box(b, name, center, size, material, bevel=.002, rotation=(0, 0, 0)):
    return b.box("exterior_" + name, center, size, material, bevel,
                 rotation=rotation, collision=False)


def _pavers(b, name, bounds, top, *, pitch=.60):
    xmin, xmax, ymin, ymax = bounds
    for i, x0 in enumerate(np.arange(xmin, xmax, pitch)):
        x1 = min(x0 + pitch, xmax)
        if x1 - x0 <= .004:
            continue
        for j, y0 in enumerate(np.arange(ymin, ymax, pitch)):
            y1 = min(y0 + pitch, ymax)
            if y1 - y0 <= .004:
                continue
            _box(b, f"{name}_{i}_{j}", ((x0 + x1) / 2, (y0 + y1) / 2, top - .011),
                 (x1 - x0 - .004, y1 - y0 - .004, .022),
                 "exterior_paving_" + str((i * 3 + j * 5) % 4), .001)


def _front_rail(b, name, x, ymin, ymax, floor, *, height=1.10, detail=True):
    count = max(2, int(np.ceil((ymax - ymin) / 1.20)) + 1)
    for i, y in enumerate(np.linspace(ymin, ymax, count)):
        _box(b, f"{name}_post_{i}", (x, y, floor + height / 2),
             (.032, .032, height), "exterior_graphite")
        if detail:
            _box(b, f"{name}_foot_{i}", (x, y, floor + .004),
                 (.085, .070, .008), "exterior_graphite", .001)
            for sign in (-1, 1):
                b.cylinder(f"exterior_{name}_bolt_{i}_{sign}",
                           (x + sign * .027, y, floor + .011),
                           .004, .006, "exterior_galvanized", collision=False)
    _box(b, name + "_top", (x, (ymin + ymax) / 2, floor + height - .017),
         (.043, ymax - ymin + .032, .034), "exterior_graphite", .004)
    _box(b, name + "_bottom", (x, (ymin + ymax) / 2, floor + .10),
         (.024, ymax - ymin, .025), "exterior_graphite")
    for i, y in enumerate(np.arange(ymin + .10, ymax, .10)):
        _box(b, f"{name}_baluster_{i}", (x, float(y), floor + (height + .10) / 2),
             (.010, .010, height - .134), "exterior_graphite", .001)


def _source(b, slug):
    if slug == "potted_plant_01":
        folder = b.assets_root / "objects/polyhaven/derived/v1" / slug
        metadata_path = folder / "DERIVATION.json"
        metadata = load_json(metadata_path)
        source = folder / (slug + ".usdc")
        dependencies = [metadata_path, metadata["source"],
                        metadata["source_provenance"],
                        Path(metadata["source_provenance"]).with_name("LICENSE.txt"),
                        *metadata["texture_files"]]
        hidden_children = ("node_1", "node_2")
    else:
        folder = b.assets_root / "objects/polyhaven/v1" / slug
        metadata_path = folder / "SOURCE.json"
        metadata = load_json(metadata_path)
        source = folder / metadata["usd_path"]
        dependencies = [metadata_path] + [folder / item["path"]
                                         for item in metadata["files"]
                                         if item["kind"] in ("usd", "texture")]
        hidden_children = (slug + "_pot", slug + "_dirt")
    for dependency in (source, *dependencies):
        if not Path(dependency).is_file():
            raise FileNotFoundError(dependency)
        b.dependency(dependency)
    return source, metadata_path, hidden_children


def _plant(b, name, slug, position, yaw, scale):
    source, provenance, hidden_children = _source(b, slug)
    path = b.path("exterior_" + name)
    root = UsdGeom.Xform.Define(b.stage, path)
    reference = b.stage.DefinePrim(path + "/Asset", "Xform")
    reference.GetReferences().AddReference(str(source))
    # Reuse the source's real leaves and branches as planted vegetation.
    # Removing a source flowerpot is a layer override, never a source mutation.
    for child in hidden_children:
        prim = b.stage.GetPrimAtPath(path + "/Asset/" + child)
        if not prim:
            raise ValueError("Missing authored plant part: " + child)
        prim.SetActive(False)
    b.xform(root.GetPrim(), (0, 0, 0), rotation=(0, 0, yaw), scale=(scale,) * 3)
    for prim in Usd.PrimRange(reference):
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            UsdPhysics.CollisionAPI(prim).CreateCollisionEnabledAttr(False)
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            UsdPhysics.RigidBodyAPI(prim).CreateRigidBodyEnabledAttr(False)
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render"])
    bounds = cache.ComputeWorldBound(root.GetPrim()).ComputeAlignedRange()
    lo, hi = np.array(bounds.GetMin()), np.array(bounds.GetMax())
    translation = np.array(position) - np.array(((lo[0] + hi[0]) / 2,
                                                 (lo[1] + hi[1]) / 2, lo[2]))
    b.xform(root.GetPrim(), translation, rotation=(0, 0, yaw), scale=(scale,) * 3)
    record = {"name": "exterior_" + name, "kind": "asset",
              "source": str(source), "source_manifest": str(provenance),
              "position": list(position), "bounds_size": (hi - lo).tolist(),
              "yaw_deg": yaw, "scale": scale, "license": "CC0-1.0",
              "hidden_source_parts": list(hidden_children),
              "static_scenery": True, "collision": False}
    b.records.append(record)
    return record


def _planter(b, name, x, y, length, *, z=BALCONY_Z, depth=.38, height=.32):
    t = .028
    _box(b, name + "_bottom", (x, y, z + t / 2),
         (depth, length, t), "exterior_clay")
    for side in (-1, 1):
        _box(b, name + "_long_" + str(side), (x + side * (depth - t) / 2, y, z + height / 2),
             (t, length, height), "exterior_clay", .004)
        _box(b, name + "_end_" + str(side), (x, y + side * (length - t) / 2, z + height / 2),
             (depth - 2 * t, t, height), "exterior_clay", .004)
    soil = z + height - .045
    _box(b, name + "_soil", (x, y, soil - .012),
         (depth - 2 * t, length - 2 * t, .024), "exterior_soil")
    return soil


def _private_balcony(b):
    xmin, xmax, ymin, ymax = -3.40, -2.07, -1.25, 2.35
    center = ((xmin + xmax) / 2, (ymin + ymax) / 2)
    # Cantilever slab joins the building wall, with tile joints and a continuous
    # structural edge. Lower floors make the elevated apartment legible.
    for level in (0, -1, -2):
        top = BALCONY_Z + 3.0 * level
        _box(b, f"home_balcony_slab_{level}", (*center, top - .102),
             (xmax - xmin, ymax - ymin, .16), "exterior_concrete", .006)
        if level == 0:
            _pavers(b, "private_tile", (xmin, xmax, ymin, ymax), top, pitch=.45)
        else:
            _box(b, f"home_balcony_finish_{level}", (*center, top - .011),
                 (xmax - xmin, ymax - ymin, .022), "exterior_paving_0", .001)
        _front_rail(b, f"home_rail_{level}", -3.35, -1.18, 2.28, top,
                    detail=level == 0)
    _box(b, "home_upper_slab", (*center, 2.98),
         (xmax - xmin, ymax - ymin, .16), "exterior_render", .004)
    for index, y in enumerate((ymin, ymax)):
        _box(b, f"balcony_privacy_wall_{index}", (center[0], y, 1.441),
             (xmax - xmin, .10, 2.918), "exterior_render", .004)
        # These walls continue to the courtyard; there is no unsupported slab
        # or visually detached garden platform beneath the apartment.
        _box(b, f"lower_balcony_side_{index}", (center[0], y, -3.10),
             (xmax - xmin, .10, 6.20), "exterior_side_render", .004)
    _box(b, "home_lower_facade", (-2.12, center[1], -3.15),
         (.10, ymax - ymin, 6.10), "exterior_side_render", .002)
    # The existing door rail is +.18 m. This masonry step is supported directly
    # on the balcony, leaving two roughly 10 cm rises instead of a floating lip.
    _box(b, "door_landing_step", (-2.32, 1.2625, .031),
         (.36, 1.425, .098), "exterior_sill", .004)
    _box(b, "door_step_nosing", (-2.487, 1.2625, .076),
         (.018, 1.425, .008), "exterior_paving_2", .001)
    # Drainage is at the outer edge, away from the waterproof threshold.
    _box(b, "drain_bed", (-3.255, .55, BALCONY_Z + .002),
         (.065, 3.31, .006), "exterior_graphite", 0)
    for sign in (-1, 1):
        _box(b, "drain_rim_" + str(sign), (-3.255 + sign * .031, .55, BALCONY_Z + .005),
             (.004, 3.31, .006), "exterior_galvanized", .001)
    for i, y in enumerate(np.arange(-1.09, 2.20, .045)):
        _box(b, "drain_grate_" + str(i), (-3.255, float(y), BALCONY_Z + .005),
             (.060, .008, .006), "exterior_galvanized", .001)
    b.cylinder("exterior_home_rainwater_pipe", (-3.34, 2.40, -1.54),
               .035, 9.32, "exterior_galvanized", collision=False)
    for i, z in enumerate((-5.4, -2.4, .6, 2.5)):
        _box(b, "home_pipe_clip_" + str(i), (-3.34, 2.36, z),
             (.09, .09, .025), "exterior_graphite", .002)
    plants = []
    for name, y, slug, yaw, scale in (
        ("entry_planter", -.96, "potted_plant_02", 24., .51),
        ("rear_planter", 2.03, "potted_plant_01", -47., .43),
    ):
        soil = _planter(b, name, -2.98, y, .42)
        plants.append(_plant(b, name + "_plant", slug, (-2.98, y, soil), yaw, scale))
    return plants


def _residential_window(b, name, y, floor, *, door=False, variant=0):
    x, width = OPPOSITE_FRONT_X, 1.90
    low, high = floor + (.05 if door else .85), floor + 2.45
    height, middle = high - low, (high + low) / 2
    # The facade is assembled around this hole. A recessed reveal, dark inner
    # room, glass and separate curtains create actual depth behind each frame.
    _box(b, name + "_recess", (x - .34, y, middle),
         (.025, width - .06, height - .06), "exterior_interior", 0)
    for side in (-1, 1):
        _box(b, name + "_reveal_" + str(side), (x - .15, y + side * .96, middle),
             (.30, .04, height), "exterior_sill", .001)
        _box(b, name + "_jamb_" + str(side), (x + .008, y + side * .916, middle),
             (.075, .068, height), "exterior_graphite", .002)
    for label, z in (("head", high - .034), ("bottom", low + .034)):
        _box(b, name + "_" + label, (x + .008, y, z),
             (.075, width, .068), "exterior_graphite", .002)
    _box(b, name + "_mullion", (x + .014, y + .12, middle),
         (.08, .055, height - .10), "exterior_graphite", .002)
    _box(b, name + "_glass", (x - .035, y, middle),
         (.008, width - .14, height - .14), "exterior_window", 0)
    _box(b, name + "_sill", (x + .035, y, low - .022),
         (.36, width + .14, .045), "exterior_sill", .003)
    curtain = "exterior_curtain_" + str(variant % 3)
    # Different open widths avoid a repeated identical wallpaper appearance.
    curtain_width = (.34, .49, .65, .26)[variant % 4]
    for side in (-1, 1):
        cy = y + side * (width / 2 - curtain_width / 2 - .06)
        _box(b, name + "_curtain_" + str(side), (x - .23, cy, middle),
             (.020, curtain_width, height - .11), curtain, .001)
        for fold in range(3):
            _box(b, f"{name}_curtain_fold_{side}_{fold}",
                 (x - .208, cy - curtain_width / 2 + (fold + .5) * curtain_width / 3, middle),
                 (.026, .018, height - .11), curtain, .002)
    if variant % 5 == 1:
        _box(b, name + "_blind", (x - .12, y, high - .31),
             (.025, width - .15, .49), "exterior_curtain_2", .002)
        for i in range(7):
            _box(b, name + "_blind_seam_" + str(i), (x - .102, y, high - .11 - .06 * i),
                 (.009, width - .16, .006), "exterior_sill", .001)
    return {"name": name, "front_x_m": x, "y_bounds_m": [y - .95, y + .95],
            "z_bounds_m": [low, high], "balcony_door": door}


def _opposite_residence(b):
    front, rear, ymin, ymax = OPPOSITE_FRONT_X, -21.20, -3.70, 18.70
    roof = COURTYARD_Z + 12.0
    _box(b, "residence_rear", (rear, (ymin + ymax) / 2, (COURTYARD_Z + roof) / 2),
         (.24, ymax - ymin, 12.), "exterior_side_render", .004)
    for i, y in enumerate((ymin, ymax)):
        _box(b, f"residence_side_{i}", ((front + rear) / 2, y, (COURTYARD_Z + roof) / 2),
             (front - rear, .24, 12.), "exterior_side_render", .004)
    windows, balcony_count = [], 0
    for level in range(5):
        z = COURTYARD_Z + 3.0 * level
        _box(b, f"residence_floor_{level}", ((front + rear) / 2, (ymin + ymax) / 2, z - .09),
             (front - rear, ymax - ymin, .18), "exterior_concrete", .003)
        if level == 4:
            continue
        # Structural piers leave real 1.90 m apertures, never a textured box.
        for pier in range(8):
            py = ymin + 3.2 * pier
            width = .65 if pier in (0, 7) else 1.30
            py += .325 if pier == 0 else (-.325 if pier == 7 else 0.)
            _box(b, f"residence_pier_{level}_{pier}", (front - .12, py, z + 1.50),
                 (.24, width, 3.0), "exterior_render", .003)
        _box(b, f"residence_header_{level}", (front - .12, (ymin + ymax) / 2, z + 2.725),
             (.24, ymax - ymin, .55), "exterior_render", .003)
        _box(b, f"residence_joint_{level}", (front + .004, (ymin + ymax) / 2, z + 2.985),
             (.012, ymax - ymin, .022), "exterior_sill", .001)
        for bay in range(7):
            cy = ymin + 1.60 + 3.20 * bay
            door = level > 0 and bay in (1, 3, 5)
            sill = .05 if door else .85
            _box(b, f"residence_apron_{level}_{bay}", (front - .12, cy, z + sill / 2),
                 (.24, 1.90, sill), "exterior_render", .002)
            name = f"window_{level}_{bay}"
            windows.append(_residential_window(b, name, cy, z, door=door,
                                               variant=bay + 3 * level))
            if not door:
                continue
            balcony_count += 1
            _box(b, f"opposite_balcony_{level}_{bay}", (front + .525, cy, z - .08),
                 (1.05, 2.20, .16), "exterior_sill", .005)
            _front_rail(b, f"opposite_rail_{level}_{bay}", front + 1.0,
                        cy - 1.05, cy + 1.05, z, height=1.05, detail=False)
            for side in (-1, 1):
                _box(b, f"opposite_balcony_side_{level}_{bay}_{side}",
                     (front + .525, cy + side * 1.05, z + .50),
                     (1.05, .06, 1.0), "exterior_render", .003)
    for side, y in enumerate((ymin + .17, ymax - .17)):
        b.cylinder(f"exterior_residence_downpipe_{side}", (front + .055, y, -.15),
                   .045, 12.10, "exterior_galvanized", collision=False)
    _box(b, "residence_roof_parapet", (front - .12, (ymin + ymax) / 2, roof + .19),
         (.24, ymax - ymin, .38), "exterior_render", .003)
    _box(b, "residence_roof_coping", (front - .12, (ymin + ymax) / 2, roof + .40),
         (.32, ymax - ymin + .10, .04), "exterior_sill", .004)
    return windows, balcony_count


def _courtyard(b):
    # A small paved residential court is six metres below, not level with the
    # living room. Low planting stays close to the opposite building entrance.
    _box(b, "court_base", (-9.52, 7.50, COURTYARD_Z - .10),
         (14.90, 30., .20), "exterior_paving_1", .002)
    _pavers(b, "court_entrance_paver", (-15.9, -13.75, -3.5, 18.5), COURTYARD_Z + .012,
            pitch=1.0)
    for i, y in enumerate((1.40, 8.0, 14.60)):
        soil = _planter(b, "court_bed_" + str(i), -12.95, y, 2.20,
                        z=COURTYARD_Z, depth=1.05, height=.25)
        for j, (offset, scale, yaw) in enumerate(((-.60, .90, 18.), (.05, .72, 132.), (.65, .81, 269.))):
            _plant(b, f"court_shrub_{i}_{j}", "potted_plant_02",
                   (-12.95, y + offset, soil), yaw + i * 31, scale)
    # Narrow paving joints give the lower level a scale without busy park props.
    for i, x in enumerate(np.arange(-12., -2.2, 1.5)):
        _box(b, "court_joint_" + str(i), (float(x), 7.5, COURTYARD_Z + .002),
             (.012, 30., .004), "exterior_grout", 0)


def build_exterior(b):
    """Author a private balcony and an opposite four-storey residential block."""
    _materials(b)
    plants = _private_balcony(b)
    windows, balconies = _opposite_residence(b)
    _courtyard(b)
    hdr = b.assets_root / HDR_RELATIVE
    b.dependency(hdr)
    b.dependency(hdr.parent / "SOURCE.json")
    return {
        "schema": "scence.living_exterior.v2",
        "near_field": "Third-floor private balcony, waterproof threshold step, drainage, guardrail and two small planted containers",
        "floor_assumption": {"apartment_level": 3, "room_floor_z_m": 0.,
                             "courtyard_z_m": COURTYARD_Z, "storey_pitch_m": 3.0},
        "balcony": {"xy_bounds_m": [-3.40, -2.07, -1.25, 2.35],
                    "surface_z_m": BALCONY_Z, "rail_height_m": 1.10,
                    "ceiling_underside_z_m": 2.90,
                    "door_rail_z_m": .18, "supported_step_top_z_m": .08},
        "opposite_residence": {"front_x_m": OPPOSITE_FRONT_X,
                               "y_bounds_m": [-3.70, 18.70], "storeys": 4,
                               "balconies": balconies, "window_openings": windows,
                               "complete_shell": True},
        "private_balcony_plants": plants,
        "plants": [record for record in b.records
                   if record.get("name", "").startswith("exterior_")
                   and record.get("kind") == "asset"],
        "far_field": {
            "kind": "CC0 pure-sky HDR; residential buildings and courtyard are dimensional geometry",
            "source": str(hdr), "source_manifest": str(hdr.parent / "SOURCE.json"),
            "sha256": b.dependencies[str(hdr.resolve())]["sha256"],
            "texture_format": "latlong", "rotate_xyz_deg": list(HDR_ROTATE_XYZ_DEG),
            "license": "CC0-1.0",
            "orientation_basis": "Isaac RTX handles stage up-axis; no extra X rotation or photographic ground panorama.",
        },
        "collision": False,
        "scope": "Static residential scenery and visual support only; existing source files preserved; no structural, dynamics or manipulation qualification",
    }
