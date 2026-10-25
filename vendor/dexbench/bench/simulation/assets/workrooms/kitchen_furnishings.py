"""Detailed fixed kitchen fittings; all dressing is static, never task payload.

A cut-out countertop surrounds an actual double sink.  The central cooking
zone and island remain clear so future tasks can own their movable contents.
All coordinates are metres, Z up; imported sources remain unchanged on disk.
"""
from __future__ import annotations

import math
from pxr import UsdGeom, UsdPhysics


COUNTER_TOP = .90
COUNTER_Y = .94
COUNTER_DEPTH = .74
BAR_CENTER = (.45, -.95)
BAR_SIZE = (2.80, .80, .045)


def _materials(b):
    for name, color, roughness, metallic in (
        ("kitchen_sage", (.27, .36, .29), .40, .0),
        ("kitchen_sage_recess", (.18, .25, .20), .48, .0),
        ("kitchen_quartz", (.83, .82, .76), .27, .0),
        ("kitchen_grout", (.61, .61, .55), .78, .0),
        ("kitchen_tile_cream", (.79, .81, .75), .26, .0),
        ("kitchen_tile_sage", (.70, .75, .66), .30, .0),
        ("kitchen_brass", (.43, .29, .11), .27, .83),
        ("kitchen_hob_glass", (.013, .018, .019), .13, .16),
        ("kitchen_hob_ring", (.25, .29, .28), .34, .42),
        ("kitchen_oven_glass", (.025, .037, .043), .16, .24),
        ("kitchen_linen", (.66, .69, .57), .95, .0),
        ("kitchen_terracotta", (.52, .23, .11), .76, .0),
        ("kitchen_leaf", (.13, .28, .09), .74, .0),
        ("kitchen_soil", (.055, .032, .017), 1., .0),
        ("kitchen_oil", (.26, .30, .055), .27, .0),
        ("kitchen_red", (.57, .11, .045), .47, .0),
        ("kitchen_yellow", (.80, .53, .10), .48, .0),
        ("kitchen_display", (.46, .70, .60), .25, .0),
    ):
        b.material(name, color, roughness, metallic)


def _handle(b, name, x, y, z, width=.20, material="kitchen_brass"):
    """Three-dimensional pull with two stand-offs, facing the robot aisle."""
    for side in (-1, 1):
        b.cylinder(name + f"_mount_{side}", (x + side * width / 2, y + .017, z),
                   .006, .034, material, rotation=(90, 0, 0), collision=False)
    b.cylinder(name + "_grip", (x, y, z), .007, width + .015, material,
               rotation=(0, 90, 0), collision=False)


def _door(b, name, x, width, z=.49, height=.69, y=.592):
    b.box(name, (x, y, z), (width, .026, height), "kitchen_sage", .003)
    # Slim stile-and-rail geometry catches light and leaves a real recessed panel.
    b.box(name + "_recess", (x, y - .014, z),
          (width - .09, .004, height - .095), "kitchen_sage_recess", .002,
          collision=False)
    b.box(name + "_panel", (x, y - .0165, z),
          (width - .11, .006, height - .115), "kitchen_sage", .002,
          collision=False)
    _handle(b, name + "_pull", x, y - .044, z + height / 2 - .075,
            min(.23, width - .14))


def _static_source(b, name, relative_path, position, rotation=(0, 0, 0)):
    """Reference licensed scenery and author all physics overrides locally."""
    source = b.assets_root / "objects" / "simready" / "v2" / relative_path
    if not source.is_file():
        raise FileNotFoundError(source)
    # Preserve hashes for source layers, textures, license and attribution.
    for dependency in sorted(source.parent.rglob("*")):
        if dependency.is_file():
            b.dependency(dependency)
    root = b.asset(name, source, position, rotation=rotation)
    # Instances hide their mesh children from ordinary PrimRange traversal.
    # Uninstance in this scene layer before applying static mesh overrides.
    pending = [b.stage.GetPrimAtPath(root)]
    while pending:
        prim = pending.pop()
        if prim.IsInstance():
            prim.SetInstanceable(False)
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            prim.RemoveAPI(UsdPhysics.ArticulationRootAPI)
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            UsdPhysics.RigidBodyAPI(prim).CreateRigidBodyEnabledAttr(False)
        if prim.IsA(UsdPhysics.Joint):
            UsdPhysics.Joint(prim).CreateJointEnabledAttr(False)
        if prim.IsA(UsdGeom.Mesh):
            UsdPhysics.CollisionAPI.Apply(prim).CreateCollisionEnabledAttr(True)
            # A static triangle surface preserves sink cavities and openings.
            UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr("none")
        pending.extend(prim.GetChildren())
    return {
        "prim": root, "source_usd": str(source), "source_version": "simready/v2",
        "license": "CC-BY-4.0", "attribution": "NVIDIA Corporation",
        "position_m": list(position), "rotation_deg": list(rotation),
        "scale": 1., "static_dressing": True,
        "source_sha256": b.dependencies[str(source.resolve())]["sha256"],
    }


def _counter(b):
    supports = []

    def slab(name, x0, x1, y0=.57, y1=1.31):
        path = b.box(name, ((x0 + x1) / 2, (y0 + y1) / 2, .881),
                     (x1 - x0, y1 - y0, .038), "kitchen_quartz", .0025)
        supports.append(path + "/Collider")

    # Sink opening: x [-1.1975,-.5025], y [.7075,1.1725].
    # Source rim overlaps each edge, but nothing spans either sink cavity.
    slab("kitchen_counter_left", -1.70, -1.1975)
    slab("kitchen_counter_sink_front", -1.1975, -.5025, .57, .7075)
    slab("kitchen_counter_sink_rear", -1.1975, -.5025, 1.1725, 1.31)
    slab("kitchen_counter_middle", -.5025, .32)
    # Embedded hob has its own thin support surface, flush with the stone.
    slab("kitchen_counter_hob_front", .32, .98, .57, .645)
    slab("kitchen_counter_hob_rear", .32, .98, 1.235, 1.31)
    slab("kitchen_counter_right", .98, 1.85)
    # Cabinet shells are not solid blocks: the sink has space below its bowls.
    b.box("kitchen_base_toekick", (.075, 1.00, .065),
          (3.55, .57, .13), "kitchen_sage_recess", .002)
    b.box("kitchen_base_floor", (.075, .96, .143),
          (3.55, .67, .026), "oak", .002)
    b.box("kitchen_base_back", (.075, 1.295, .50),
          (3.55, .026, .71), "kitchen_sage", .002)
    for i, x in enumerate((-1.688, -1.22, -.48, .308, .992, 1.838)):
        b.box(f"kitchen_cabinet_side_{i}", (x, .95, .50),
              (.024, .69, .71), "kitchen_sage", .002)
    _door(b, "kitchen_cabinet_left", -1.452, .437)
    for i, x in enumerate((-1.031, -.669)):
        _door(b, f"kitchen_sink_door_{i}", x, .350)
    for i, (z, height) in enumerate(((.245, .183), (.466, .246), (.729, .258))):
        _door(b, f"kitchen_prep_drawer_{i}", -.087, .758, z, height)
    for i, x in enumerate((1.205, 1.625)):
        _door(b, f"kitchen_right_door_{i}", x, .401)
    return supports


def _sink(b):
    sink = _static_source(
        b, "kitchen_double_sink", "Sink_Basin/rc_fixture_sink_basin_a.usd",
        (-.85, .94, .679813058))
    # Goose-neck mixer: the outlet is over the left bowl, never over the rim.
    x, y, z = -.91, 1.24, .90
    b.cylinder("kitchen_faucet_foot", (x, y, z + .004), .028, .008, "chrome")
    b.cylinder("kitchen_faucet_mixer", (x, y, z + .052), .022, .095, "chrome")
    points = [(x, y, z + .075), (x, y, z + .20)]
    for i in range(21):
        a = math.pi * i / 20
        points.append((x, y - .09 + .09 * math.cos(a), z + .20 + .09 * math.sin(a)))
    points.append((x, y - .18, z + .155))
    b.tube("kitchen_faucet_spout", points, .012, "chrome", collision=True)
    b.cylinder("kitchen_faucet_aerator", (x, y - .18, z + .154),
               .014, .018, "steel", collision=False)
    b.cylinder("kitchen_faucet_lever_hub", (x + .035, y, z + .063),
               .014, .025, "chrome", rotation=(0, 90, 0), collision=False)
    b.tube("kitchen_faucet_lever", [(x + .044, y, z + .065),
                                    (x + .044, y, z + .13),
                                    (x + .044, y - .035, z + .145)],
           .006, "chrome", collision=False)
    # A compact ceramic dispenser at the outer edge is fixed dressing.
    b.lathe("kitchen_soap_bottle", (-1.37, 1.22, .90),
            [(0, 0), (.027, 0), (.033, .009), (.031, .10),
             (.019, .118), (.012, .118), (.012, .13), (0, .13)], "ceramic")
    b.tube("kitchen_soap_pump", [(-1.37, 1.22, 1.026), (-1.37, 1.22, 1.052),
                                (-1.37, 1.18, 1.052)], .005, "chrome")
    b.box("kitchen_sponge_tray", (-1.44, .78, .906), (.13, .08, .012),
          "ceramic", .007, collision=False)
    b.box("kitchen_sponge", (-1.44, .78, .923), (.095, .055, .024),
          "kitchen_linen", .006, collision=False)
    return sink


def _cooking(b):
    support = b.box("kitchen_cooktop", (.65, .94, .891), (.66, .59, .018),
                    "kitchen_hob_glass", .005) + "/Collider"
    # Four flush induction zones have two fine rings and center registration.
    for i, (x, y, radius) in enumerate(((.485, .815, .102), (.815, .815, .085),
                                       (.485, 1.095, .084), (.815, 1.095, .100))):
        for j, r in enumerate((radius, radius - .005)):
            b.torus(f"kitchen_hob_zone_{i}_ring_{j}", (x, y, .9004), r, .00065,
                    "kitchen_hob_ring")
        for axis in range(2):
            size = (.016, .0011, .0007) if axis == 0 else (.0011, .016, .0007)
            b.box(f"kitchen_hob_cross_{i}_{axis}", (x, y, .9005), size,
                  "kitchen_hob_ring", .0001, collision=False)
    for i in range(4):
        b.box(f"kitchen_hob_touch_{i}", (.56 + .06 * i, .672, .9005),
              (.018, .0015, .0007), "kitchen_hob_ring", .0001, collision=False)
    b.torus("kitchen_hob_power", (.43, .674, .9005), .007, .0007, "kitchen_hob_ring")
    # Built-in oven: shadow gap, steel frame, dark double-glazed door and vents.
    b.box("kitchen_oven_case", (.65, .946, .50), (.626, .66, .709), "black", .003)
    b.box("kitchen_oven_face", (.65, .601, .50), (.620, .022, .696), "steel", .004)
    b.box("kitchen_oven_control_fascia", (.65, .586, .786),
          (.603, .014, .105), "kitchen_hob_glass", .002)
    b.box("kitchen_oven_door_gasket", (.65, .583, .459),
          (.572, .012, .505), "rubber", .006, collision=False)
    b.box("kitchen_oven_door_frame", (.65, .572, .459),
          (.555, .020, .484), "steel", .005)
    b.box("kitchen_oven_window", (.65, .559, .446),
          (.480, .008, .352), "kitchen_oven_glass", .008, collision=False)
    b.box("kitchen_oven_window_inner", (.65, .554, .442),
          (.410, .001, .278), "kitchen_hob_glass", .005, collision=False)
    _handle(b, "kitchen_oven_handle", .65, .530, .659, .43, "steel")
    for i, x in enumerate((.43, .87)):
        b.cylinder(f"kitchen_oven_dial_{i}", (x, .564, .786),
                   .023, .027, "steel", rotation=(90, 0, 0), collision=False)
        b.box(f"kitchen_oven_dial_mark_{i}", (x, .548, .798),
              (.002, .001, .010), "black", .0001, collision=False)
    b.box("kitchen_oven_display", (.65, .577, .786),
          (.130, .003, .039), "black", .001, collision=False)
    # Seven-segment geometry keeps the small clock readable without a decal.
    for digit, mask in enumerate((0b0110000, 0b1101101, 0b1111110, 0b1111110)):
        cx = .609 + digit * .025
        for segment, (dx, dz, sx, sz) in enumerate(
            ((0, .012, .012, .0014), (.007, .006, .0014, .010),
             (.007, -.006, .0014, .010), (0, -.012, .012, .0014),
             (-.007, -.006, .0014, .010), (-.007, .006, .0014, .010),
             (0, 0, .012, .0014))):
            if mask & (1 << (6 - segment)):
                b.box(f"kitchen_clock_{digit}_{segment}", (cx + dx, .574, .786 + dz),
                      (sx, .0008, sz), "kitchen_display", .0001, collision=False)
    for i in range(17):
        b.box(f"kitchen_oven_bottom_vent_{i}", (.41 + i * .030, .584, .157),
              (.018, .002, .009), "black", .001, collision=False)
    # Compact wall chimney hood with visible filter mesh, lip and controls.
    b.box("kitchen_hood_canopy", (.65, 1.045, 1.615), (.81, .53, .065),
          "steel", .007)
    b.box("kitchen_hood_lower_rim", (.65, 1.045, 1.580), (.76, .48, .012),
          "black", .003, collision=False)
    b.box("kitchen_hood_chimney", (.65, 1.185, 2.126), (.32, .25, .96),
          "steel", .005)
    for panel, x in enumerate((.46, .84)):
        b.box(f"kitchen_hood_filter_{panel}", (x, 1.06, 1.571),
              (.33, .36, .006), "steel", .001, collision=False)
        for i in range(17):
            b.box(f"kitchen_hood_filter_slit_{panel}_{i}", (x - .15 + i * .018, 1.06, 1.567),
                  (.0025, .325, .001), "black", .0001, collision=False)
        b.cylinder(f"kitchen_hood_light_{panel}", (x, .845, 1.573),
                   .018, .004, "white", collision=False)
    for i in range(4):
        b.cylinder(f"kitchen_hood_button_{i}", (.60 + i * .034, .776, 1.618),
                   .006, .003, "black", rotation=(90, 0, 0), collision=False)
    return support


def _backsplash(b):
    b.box("kitchen_backsplash_base", (.075, 1.327, 1.265), (3.55, .018, .73),
          "kitchen_grout", .001)
    # Actual glazed subway tiles with recessed, 3 mm grout joints.
    for row in range(6):
        for col in range(16):
            x0 = -1.70 + col * .235 - (row % 2) * .1175
            x1 = min(x0 + .232, 1.85)
            x0 = max(x0, -1.70)
            if x1 <= x0:
                continue
            b.box(f"kitchen_backsplash_tile_{row}_{col}",
                  ((x0 + x1) / 2, 1.314, .963 + row * .119),
                  (x1 - x0, .008, .116),
                  "kitchen_tile_sage" if (row * 3 + col) % 5 == 0 else "kitchen_tile_cream",
                  .0017, collision=False)
    # Double socket with recessed contacts, set away from the wet sink zone.
    b.box("kitchen_power_plate", (1.63, 1.304, 1.205), (.15, .012, .085),
          "white", .006, collision=False)
    for side in (-1, 1):
        x = 1.63 + .038 * side
        b.box(f"kitchen_socket_{side}", (x, 1.296, 1.205), (.058, .006, .058),
              "ceramic", .003, collision=False)
        for k in (-1, 1):
            b.box(f"kitchen_socket_hole_{side}_{k}", (x + k * .012, 1.292, 1.211),
                  (.003, .002, .012), "black", .0005, collision=False)


def _cup(b, name, x, y, z, material="ceramic"):
    b.lathe(name, (x, y, z), [(0, 0), (.028, 0), (.034, .007),
             (.039, .080), (.038, .086), (.034, .086), (.029, .009), (0, .009)], material)
    angles = [math.pi * (-.60 + 1.20 * i / 24) for i in range(25)]
    b.tube(name + "_handle", [(x + .037 + .024 * math.cos(a), y,
                               z + .046 + .027 * math.sin(a)) for a in angles],
           .005, material)


def _plant(b, name, x, y, z, scale=1.):
    b.lathe(name + "_pot", (x, y, z),
            [(r * scale, h * scale) for r, h in
             ((0, 0), (.052, 0), (.069, .12), (.075, .12), (.075, .13),
              (.064, .13), (.058, .018), (0, .018))], "kitchen_terracotta")
    b.cylinder(name + "_soil", (x, y, z + .117 * scale), .063 * scale,
               .008 * scale, "kitchen_soil", collision=False)
    for i in range(9):
        angle = i * 2.39996
        tip = (x + math.cos(angle) * .075 * scale,
               y + math.sin(angle) * .075 * scale,
               z + (.23 + .028 * (i % 3)) * scale)
        b.tube(name + f"_stem_{i}", [(x, y, z + .12 * scale),
                 (x + .025 * math.cos(angle) * scale,
                  y + .025 * math.sin(angle) * scale, z + .20 * scale), tip],
               .0017 * scale, "kitchen_leaf")
        for side in (-1, 1):
            leaf = (tip[0] + math.cos(angle + side) * .019 * scale,
                    tip[1] + math.sin(angle + side) * .019 * scale, tip[2] - .015 * scale)
            b.sphere(name + f"_leaf_{i}_{side}", leaf,
                     (.031 * scale, .014 * scale, .004 * scale), "kitchen_leaf")


def _shelves_and_dressing(b):
    for level, z in enumerate((1.70, 2.04)):
        b.box(f"kitchen_open_shelf_{level}", (-.80, 1.185, z),
              (1.69, .25, .038), "oak", .003)
        for side, x in enumerate((-1.46, -.14)):
            b.box(f"kitchen_shelf_bracket_wall_{level}_{side}", (x, 1.302, z - .095),
                  (.021, .017, .19), "kitchen_brass", .001, collision=False)
            b.box(f"kitchen_shelf_bracket_arm_{level}_{side}", (x, 1.19, z - .026),
                  (.021, .23, .014), "kitchen_brass", .001, collision=False)
    for i in range(4):
        b.lathe(f"kitchen_plate_stack_{i}", (-1.35, 1.18, 1.719 + i * .011),
                [(0, 0), (.060, 0), (.103, .010), (.108, .019),
                 (.102, .021), (.070, .009), (0, .008)], "ceramic")
    for i, x in enumerate((-.94, -.77, -.60)):
        _cup(b, f"kitchen_shelf_mug_{i}", x, 1.18, 1.719,
             "ceramic" if i != 1 else "kitchen_sage")
    for i, x in enumerate((-1.40, -1.18, -.96)):
        height = (.19, .22, .16)[i]
        b.lathe(f"kitchen_storage_jar_{i}", (x, 1.18, 2.059),
                [(0, 0), (.063, 0), (.071, .01), (.071, height - .015),
                 (.065, height), (0, height)], "ceramic")
        b.cylinder(f"kitchen_storage_lid_{i}", (x, 1.18, 2.064 + height),
                   .072, .012, "oak", collision=False)
    _plant(b, "kitchen_shelf_herb", -.25, 1.18, 1.719, .70)
    # A rail and softly folded tea towel occupy the narrow strip below shelves.
    b.cylinder("kitchen_rail", (-.62, 1.25, 1.435), .008, .85,
               "kitchen_brass", rotation=(0, 90, 0), collision=False)
    for side, x in enumerate((-1.045, -.195)):
        b.cylinder(f"kitchen_rail_mount_{side}", (x, 1.283, 1.435),
                   .014, .066, "kitchen_brass", rotation=(90, 0, 0), collision=False)
    vertices, faces = [], []
    nx, nz = 18, 25
    for j in range(nz + 1):
        t = j / nz
        for i in range(nx + 1):
            s = i / nx
            vertices.append((-.86 + .26 * s,
                             1.232 - .012 * math.sin(s * math.pi * 6) - .009 * t,
                             1.432 - .28 * t + .005 * math.sin(s * math.pi * 4) * t))
    for j in range(nz):
        for i in range(nx):
            a = j * (nx + 1) + i
            faces.append((a, a + 1, a + nx + 2, a + nx + 1))
    b.mesh(b.path("kitchen_tea_towel"), vertices, faces, "kitchen_linen")
    for i, x in enumerate((-.72, -.58, -.44)):
        b.lathe(f"kitchen_spice_jar_{i}", (x, 1.205, 2.059),
                [(0, 0), (.026, 0), (.030, .005), (.030, .10), (.023, .11), (0, .11)],
                "ceramic")
        b.cylinder(f"kitchen_spice_lid_{i}", (x, 1.205, 2.173),
                   .030, .015, "oak", collision=False)
    b.lathe("kitchen_olive_oil", (-.14, 1.19, 2.059),
            [(0, 0), (.036, 0), (.039, .013), (.039, .17),
             (.017, .197), (.014, .24), (0, .24)], "kitchen_oil")
    b.cylinder("kitchen_olive_oil_cap", (-.14, 1.19, 2.307), .016, .019,
               "black", collision=False)
    # An end-wall chopping board hangs above the right preparation surface.
    b.box("kitchen_display_board", (1.40, 1.295, 1.44), (.23, .024, .31),
          "oak", .018, collision=False)
    b.box("kitchen_display_board_handle", (1.40, 1.295, 1.615),
          (.065, .024, .065), "oak", .012, collision=False)
    b.torus("kitchen_display_board_hole", (1.40, 1.279, 1.625), .012, .003,
            "walnut", rotation=(90, 0, 0))
    return [
        _static_source(b, "kitchen_toaster", "Toaster_A01/sm_appliance_toaster_a01_01.usd",
                       (-1.485, .995, .90)),
    ]


def _bar_stool(b, index, x):
    y, seat_z = -1.83, .63
    b.cylinder(f"kitchen_stool_{index}_seat", (x, y, seat_z), .19, .038,
               "oak", collision=True)
    b.cylinder(f"kitchen_stool_{index}_seat_cushion", (x, y, seat_z + .023),
               .17, .016, "kitchen_linen", collision=False)
    for leg, (sx, sy) in enumerate(((-1, -1), (-1, 1), (1, -1), (1, 1))):
        top = (x + sx * .115, y + sy * .115, seat_z - .018)
        foot = (x + sx * .17, y + sy * .17, .018)
        b.tube(f"kitchen_stool_{index}_leg_{leg}", [foot, top], .017,
               "walnut", collision=True)
        b.sphere(f"kitchen_stool_{index}_foot_{leg}", foot, (.019, .019, .018), "rubber")
    for side in (-1, 1):
        b.tube(f"kitchen_stool_{index}_rung_x_{side}",
               [(x - .15, y + side * .15, .235), (x + .15, y + side * .15, .235)],
               .012, "walnut", collision=False)
        b.tube(f"kitchen_stool_{index}_rung_y_{side}",
               [(x + side * .15, y - .15, .235), (x + side * .15, y + .15, .235)],
               .012, "walnut", collision=False)
    # Low curved backrest keeps the island and robot visible in overview views.
    for side in (-1, 1):
        b.tube(f"kitchen_stool_{index}_back_post_{side}",
               [(x + side * .145, y - .115, .63), (x + side * .16, y - .16, .84)],
               .013, "walnut", collision=False)
    b.tube(f"kitchen_stool_{index}_back", [(x - .17 + .34 * i / 28,
           y - .17 - .025 * math.sin(math.pi * i / 28), .84) for i in range(29)],
           .025, "oak", collision=False)


def _bar(b):
    bar = b.box("worktop", (BAR_CENTER[0], BAR_CENTER[1], COUNTER_TOP - BAR_SIZE[2] / 2),
                BAR_SIZE, "kitchen_quartz", .006)
    # End trestles support a genuinely open aisle-facing knee bay.
    for side, x in enumerate((-.866, 1.766)):
        b.box(f"kitchen_bar_end_{side}", (x, -.97, .426),
              (.080, .63, .852), "oak", .005)
        b.box(f"kitchen_bar_end_foot_{side}", (x, -.97, .016),
              (.096, .65, .032), "black", .003)
    b.box("kitchen_bar_outer_rail", (.45, -1.265, .735),
          (2.56, .055, .22), "walnut", .003)
    for i in range(43):
        b.box(f"kitchen_bar_reed_{i}", (-.808 + i * .060, -1.300, .735),
              (.033, .019, .218), "oak", .004, collision=False)
    for index, x in enumerate((-.32, 1.13)):
        _bar_stool(b, index, x)
    # Peripheral food and cookware arrangements belong to kitchen_dressing.
    return bar + "/Collider"


def _refrigerator(b):
    # Source front is +X; -90 degrees puts its doors toward the robot aisle.
    asset = _static_source(b, "kitchen_refrigerator",
                           "Refrigerator_A01/sm_rc_appliance_refrigerator_a01_01.usd",
                           (2.35, .94, 0.), (0, 0, -90))
    for side, x in enumerate((1.866, 2.834)):
        b.box(f"kitchen_fridge_tall_side_{side}", (x, 1.005, 1.065),
              (.032, .68, 2.13), "oak", .003)
    b.box("kitchen_fridge_overhead_case", (2.35, 1.005, 1.946),
          (.936, .68, .35), "kitchen_sage", .003)
    for side, x in enumerate((2.112, 2.588)):
        _door(b, f"kitchen_fridge_overhead_door_{side}", x, .465,
              z=1.946, height=.328, y=.650)
    b.box("kitchen_fridge_top_trim", (2.35, 1.005, 2.145),
          (1.01, .72, .034), "oak", .003)
    return asset


def build_furnishings(b):
    """Author the fixed galley and island; return exact support and asset data."""
    _materials(b)
    countertop_paths = _counter(b)
    imported = [_sink(b)]
    cooking = _cooking(b)
    _backsplash(b)
    imported.extend(_shelves_and_dressing(b))
    bar = _bar(b)
    imported.append(_refrigerator(b))
    return {
        "support_colliders": {
            "bar": bar,
            "preparation": "/World/Geometry/kitchen_counter_middle/Collider",
            "cooking": cooking,
        },
        "countertop_support_colliders": countertop_paths,
        "layout": {
            "back_counter": {"x_range_m": [-1.70, 1.85], "center_y_m": COUNTER_Y,
                             "depth_m": COUNTER_DEPTH, "top_z_m": COUNTER_TOP,
                             "aisle_edge_y_m": .57},
            "bar": {"center_xy_m": list(BAR_CENTER), "size_xyz_m": list(BAR_SIZE),
                    "top_z_m": COUNTER_TOP, "aisle_edge_y_m": -.55,
                    "clear_task_rectangle_xy_m": [-.10, 1.10, -1.20, -.60]},
            "cooktop": {"center_xy_m": [.65, .94], "size_xy_m": [.66, .59],
                        "top_z_m": .90, "zones": 4, "type": "induction"},
            "sink": {"center_xy_m": [-.85, .94], "rim_top_z_m": .904,
                     "counter_cutout_xy_m": [-1.1975, -.5025, .7075, 1.1725],
                     "open_basin": True, "collision": "static_triangle_surfaces"},
            "robot_aisle_width_m": 1.12,
        },
        "assets": imported,
        "scope": "fixed furniture and static kitchen dressing; no dynamic task contents",
    }
