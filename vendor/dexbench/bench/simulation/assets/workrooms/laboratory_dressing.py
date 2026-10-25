"""Fixed safety, documentation and building-service assets for the laboratory.

All dimensions are metres. These are static environment assets, not movable
task props; the central island and its robot approach remain free of dressing.
"""
import math

from .interiors import _Frame


ASSET_CLASSES = (
    "emergency_eyewash", "eyewash_sign", "fire_extinguisher", "first_aid_cabinet",
    "spill_kit", "glove_dispenser", "mask_dispenser", "goggle_rack",
    "hand_sanitizer", "sharps_container", "ppe_instruction_sign",
    "laboratory_noticeboard", "laboratory_log_clipboard", "marker_tray",
    "wall_clock", "temperature_humidity_monitor", "safety_data_binder_holder",
    "segregated_pedal_bins", "adjustable_laboratory_stools",
    "ceiling_supply_diffusers", "ceiling_return_grille",
    "smoke_detector", "sprinkler_heads", "occupancy_sensor",
    "emergency_luminaire",
)


def _materials(b):
    for name, color, roughness, metallic in (
        ("ld_white", (.86, .90, .89), .38, 0),
        ("ld_teal", (.075, .30, .32), .43, 0),
        ("ld_gray", (.18, .23, .25), .50, .10),
        ("ld_screen", (.025, .065, .075), .22, 0),
        ("ld_green", (.04, .37, .22), .48, 0),
        ("ld_red", (.60, .045, .025), .36, 0),
        ("ld_yellow", (.85, .57, .055), .46, 0),
        ("ld_blue", (.07, .25, .46), .50, 0),
        ("ld_paper", (.91, .93, .91), .82, 0),
        ("ld_cork", (.37, .44, .43), .94, 0),
        ("ld_pad", (.035, .085, .09), .78, 0),
    ):
        b.material(name, color, roughness, metallic)
    b.material("ld_lens", (.65, .85, .87), .12, opacity=.34)


def _cross(f, name, xyz, size, material="ld_white"):
    x, y, z = xyz
    f.box(name + "_upright", (x, y, z), (size * .30, .003, size),
          material, .001, collision=False)
    f.box(name + "_crossbar", (x, y - .001, z), (size, .003, size * .30),
          material, .001, collision=False)


def _eyewash(b):
    f = _Frame(b, (-2.44, -1.24, 0), 90)
    f.box("ld_eyewash_wall_bracket", (0, -.025, .88), (.20, .05, .31),
          "steel", .008)
    f.tube("ld_eyewash_supply", [(0, -.06, .28), (0, -.06, .74),
                              (0, -.26, .74), (0, -.26, .86)],
           .019, "steel", collision=True)
    # Revolved double wall leaves the bowl visibly open, with a real drain.
    f.lathe("ld_eyewash_bowl", (0, -.26, .79),
            [(0, 0), (.055, 0), (.10, .018), (.176, .098),
             (.176, .112), (.166, .112), (.095, .030), (.045, .012), (0, .012)],
            "steel", collision=True)
    f.cylinder("ld_eyewash_drain", (0, -.26, .804), .024, .003,
               "black", collision=False)
    for i, x in enumerate((-.082, .082)):
        f.tube(f"ld_eyewash_head_pipe_{i}",
               [(x, -.16, .90), (x, -.16, .99), (x, -.22, 1.005)],
               .012, "chrome", collision=True)
        f.lathe(f"ld_eyewash_head_{i}", (x, -.22, .985),
                [(.020, 0), (.027, .018), (.025, .035), (0, .035)],
                "ld_green", collision=True)
        f.cylinder(f"ld_eyewash_head_cover_{i}", (x, -.22, 1.025),
                   .027, .009, "ld_white", collision=False)
    f.box("ld_eyewash_push_paddle", (.22, -.10, .86), (.13, .025, .16),
          "ld_green", .015)
    f.label("ld_eyewash_push_text", "PUSH", (.22, -.114, .86),
            (.10, .045), rotation=(90, 0, 0), font_size=52)
    f.box("ld_eyewash_sign", (0, -.023, 1.61), (.55, .024, .29),
          "ld_green", .006, collision=False)
    _cross(f, "ld_eyewash_sign_cross", (-.195, -.038, 1.65), .085)
    f.label("ld_eyewash_sign_text", "EMERGENCY\nEYE WASH",
            (.045, -.039, 1.66), (.35, .14), rotation=(90, 0, 0), font_size=56)
    f.label("ld_eyewash_inspection_tag", "WEEKLY CHECK\nDATE  /  INITIALS",
            (0, -.039, 1.51), (.40, .045), rotation=(90, 0, 0), font_size=32)
    f.box("ld_sharps_bracket", (-.63, -.032, 1.23), (.20, .06, .22),
          "steel", .004)
    f.box("ld_sharps_container", (-.63, -.13, 1.30), (.22, .20, .28),
          "ld_yellow", .023)
    f.box("ld_sharps_lid", (-.63, -.13, 1.45), (.236, .212, .027),
          "ld_red", .008)
    f.box("ld_sharps_aperture", (-.63, -.13, 1.465), (.10, .055, .003),
          "black", .008, collision=False)
    f.label("ld_sharps_label", "SHARPS\nNO RECAP",
            (-.63, -.232, 1.31), (.15, .09), rotation=(90, 0, 0), font_size=40)


def _safety_wall(b):
    # Front-facing local assets look into the room; yaw 180 maps local -Y to +Y.
    f = _Frame(b, (0, -2.45, 0), 180)
    x = 2.25
    f.box("ld_extinguisher_mount", (x, -.018, .91), (.16, .035, .43),
          "ld_gray", .008)
    f.lathe("ld_extinguisher_body", (x, -.13, .58),
            [(0, 0), (.060, 0), (.079, .025), (.079, .36),
             (.066, .395), (.026, .42), (.026, .45), (0, .45)],
            "ld_red", collision=True)
    f.box("ld_extinguisher_handle", (x, -.13, 1.055), (.13, .025, .025),
          "black", .004)
    f.box("ld_extinguisher_trigger", (x + .025, -.13, 1.08),
          (.09, .025, .012), "steel", .003)
    f.cylinder("ld_extinguisher_gauge", (x, -.174, 1.015), .022, .016,
               "steel", rotation=(90, 0, 0), collision=False)
    f.cylinder("ld_extinguisher_gauge_face", (x, -.183, 1.015), .017, .002,
               "ld_white", rotation=(90, 0, 0), collision=False)
    f.tube("ld_extinguisher_hose",
           [(x-.055, -.12, 1.045), (x-.11, -.13, 1.02),
            (x-.12, -.13, .79), (x-.10, -.15, .73)], .009, "rubber")
    f.label("ld_extinguisher_label", "CO2\nFIRE", (x, -.211, .81),
            (.10, .14), rotation=(90, 0, 0), font_size=42)
    f.box("ld_fire_sign", (x, -.02, 1.39), (.24, .025, .19),
          "ld_red", .005, collision=False)
    f.label("ld_fire_sign_text", "FIRE\nEXTINGUISHER",
            (x, -.034, 1.39), (.21, .14), rotation=(90, 0, 0), font_size=40)

    f.box("ld_first_aid_body", (1.70, -.075, 1.74), (.40, .15, .35),
          "ld_white", .014)
    f.box("ld_first_aid_door", (1.70, -.157, 1.74), (.378, .018, .328),
          "ld_green", .007)
    _cross(f, "ld_first_aid_symbol", (1.70, -.168, 1.79), .12)
    f.label("ld_first_aid_text", "FIRST AID", (1.70, -.171, 1.66),
            (.29, .055), rotation=(90, 0, 0), font_size=48)
    f.box("ld_first_aid_latch", (1.865, -.174, 1.74), (.018, .018, .065),
          "steel", .003, collision=False)
    f.box("ld_spill_kit_case", (1.70, -.075, 1.23), (.42, .15, .30),
          "ld_yellow", .013)
    f.tube("ld_spill_kit_handle", [(1.64, -.075, 1.38),
                                 (1.64, -.075, 1.415),
                                 (1.76, -.075, 1.415),
                                 (1.76, -.075, 1.38)], .009, "black")
    f.label("ld_spill_kit_text", "SPILL KIT\nABSORB  /  CONTAIN",
            (1.70, -.152, 1.23), (.34, .13), rotation=(90, 0, 0), font_size=40)

    for name, x, z, title, color in (
        ("gloves", 1.10, 1.77, "NITRILE  /  M", "ld_teal"),
        ("masks", 1.10, 1.48, "CLEAN MASKS", "ld_blue"),
    ):
        f.box("ld_" + name + "_holder", (x, -.06, z), (.36, .12, .21),
              "ld_white", .009)
        f.box("ld_" + name + "_box", (x, -.13, z), (.32, .025, .17),
              color, .006)
        f.label("ld_" + name + "_label", title, (x, -.145, z+.042),
                (.28, .05), rotation=(90, 0, 0), font_size=40)
        f.box("ld_" + name + "_slot", (x, -.147, z-.033),
              (.16, .003, .032), "black", .012, collision=False)
        f.box("ld_" + name + "_protruding_fold", (x, -.153, z-.036),
              (.06, .006, .024), "ld_white", .008, collision=False)
    f.box("ld_goggle_rack_back", (.55, -.025, 1.75), (.40, .05, .24),
          "ld_white", .006)
    for i, xx in enumerate((.45, .65)):
        for j, dx in enumerate((-.038, .038)):
            f.box(f"ld_goggle_frame_{i}_{j}", (xx+dx, -.066, 1.77),
                  (.072, .019, .057), "ld_teal", .014, collision=False)
            f.box(f"ld_goggle_lens_{i}_{j}", (xx+dx, -.078, 1.77),
                  (.060, .004, .042), "ld_lens", .010, collision=False)
        f.box(f"ld_goggle_bridge_{i}", (xx, -.069, 1.77),
              (.018, .020, .010), "ld_teal", .003, collision=False)
        f.tube(f"ld_goggle_strap_{i}", [(xx-.072, -.056, 1.77),
                                      (xx-.065, -.02, 1.82),
                                      (xx+.065, -.02, 1.82),
                                      (xx+.072, -.056, 1.77)], .004, "rubber")
    f.label("ld_goggle_tag", "EYE PROTECTION", (.55, -.053, 1.67),
            (.32, .045), rotation=(90, 0, 0), font_size=34)
    f.box("ld_sanitizer_body", (.55, -.07, 1.38), (.16, .14, .23),
          "ld_white", .018)
    f.box("ld_sanitizer_window", (.55, -.143, 1.40), (.065, .004, .09),
          "ld_teal", .010, collision=False)
    f.box("ld_sanitizer_lever", (.55, -.16, 1.295), (.12, .07, .024),
          "steel", .004)
    f.label("ld_sanitizer_tag", "SANITIZE", (.55, -.145, 1.33),
            (.13, .032), rotation=(90, 0, 0), font_size=32)
    f.box("ld_ppe_sign_back", (1.10, -.017, 2.11), (1.42, .025, .19),
          "ld_blue", .005, collision=False)
    f.label("ld_ppe_sign_text", "LABORATORY   /   PPE REQUIRED\nCOAT   ·   GLOVES   ·   EYE PROTECTION",
            (1.10, -.031, 2.11), (1.34, .15), rotation=(90, 0, 0), font_size=48)


def _documentation(b):
    f = _Frame(b, (0, -2.45, 0), 180)
    f.box("ld_noticeboard_frame", (-.40, -.025, 1.70), (1.16, .05, .77),
          "steel", .005)
    f.box("ld_noticeboard_surface", (-.40, -.054, 1.70), (1.12, .010, .73),
          "ld_paper", .002, collision=False)
    f.label("ld_noticeboard_title", "LAB 02   /   DAILY WORK",
            (-.40, -.061, 1.997), (1.04, .078), rotation=(90, 0, 0), font_size=48)
    f.label("ld_noticeboard_schedule",
            "EQUIPMENT       STATUS       OPERATOR\n"
            "FUME HOOD       CHECKED      ______\n"
            "BALANCE         ZEROED       ______\n"
            "MICROSCOPE      CLEAN        ______\n"
            "BENCHES         CLEAR        ______",
            (-.62, -.062, 1.71), (.63, .41), rotation=(90, 0, 0), font_size=36)
    f.box("ld_noticeboard_cork", (.005, -.062, 1.70), (.27, .006, .44),
          "ld_cork", .002, collision=False)
    for i, z in enumerate((1.79, 1.61)):
        f.label(f"ld_notice_{i}",
                ("SAMPLE LOG\nID / DATE / USER", "END OF DAY\nCLEAN + RECORD")[i],
                (.005, -.067, z), (.23, .135),
                rotation=(90, 0, 0), font_size=35)
        f.sphere(f"ld_notice_pin_{i}", (.005, -.073, z+.055),
                 .008, "ld_red", collision=False)
    f.box("ld_marker_tray", (-.40, -.09, 1.286), (1.09, .14, .024),
          "steel", .003)
    for i, mat in enumerate(("ld_blue", "ld_red", "black")):
        f.cylinder(f"ld_board_marker_{i}", (-.73+i*.20, -.10, 1.310),
                   .009, .14, "ld_white", rotation=(0, 90, 0), collision=False)
        f.cylinder(f"ld_board_marker_cap_{i}", (-.65+i*.20, -.10, 1.310),
                   .010, .027, mat, rotation=(0, 90, 0), collision=False)
    f.box("ld_board_eraser", (-.08, -.10, 1.318), (.13, .055, .035),
          "ld_teal", .007, collision=False)

    f.box("ld_log_clipboard", (-1.36, -.032, 1.56), (.30, .036, .40),
          "ld_gray", .006)
    f.label("ld_log_sheet", "ROOM CHECK\n\nDATE: __________\nTEMP: __________\nRH: ____________\nSIGNED: _________",
            (-1.36, -.052, 1.55), (.265, .35), rotation=(90, 0, 0), font_size=34)
    f.box("ld_log_clip", (-1.36, -.058, 1.74), (.11, .012, .035),
          "chrome", .004, collision=False)
    f.box("ld_environment_monitor", (-1.36, -.045, 2.02), (.26, .09, .16),
          "ld_white", .010)
    f.box("ld_environment_screen", (-1.36, -.092, 2.025), (.22, .005, .11),
          "ld_screen", .005, collision=False)
    f.label("ld_environment_readout", "22.4 °C   45 % RH",
            (-1.36, -.096, 2.025), (.205, .064), rotation=(90, 0, 0), font_size=32)
    f.box("ld_sds_holder_back", (-1.94, -.025, 1.48), (.40, .05, .44),
          "ld_gray", .006)
    f.box("ld_sds_holder_bottom", (-1.94, -.10, 1.27), (.40, .19, .025),
          "steel", .003)
    for i, xx in enumerate((-2.075, -1.94, -1.805)):
        f.box(f"ld_sds_binder_{i}", (xx, -.11, 1.49), (.115, .13, .37),
              ("ld_teal", "ld_blue", "ld_gray")[i], .004)
        f.label(f"ld_sds_binder_tag_{i}", ("SDS", "SOP", "LOG")[i],
                (xx, -.177, 1.53), (.09, .055), rotation=(90, 0, 0), font_size=40)
    f.box("ld_sds_holder_front", (-1.94, -.20, 1.32), (.40, .02, .12),
          "ld_white", .004)
    f.label("ld_sds_holder_label", "SAFETY DATA", (-1.94, -.212, 1.32),
            (.34, .057), rotation=(90, 0, 0), font_size=34)

    f.cylinder("ld_clock_case", (-.42, -.045, 2.40), .17, .06,
               "ld_gray", rotation=(90, 0, 0), collision=False)
    f.cylinder("ld_clock_face", (-.42, -.078, 2.40), .154, .008,
               "ld_white", rotation=(90, 0, 0), collision=False)
    for i in range(12):
        a = 2*math.pi*i/12
        f.tube(f"ld_clock_tick_{i}",
               [(-.42+.135*math.sin(a), -.085, 2.40+.135*math.cos(a)),
                (-.42+.119*math.sin(a), -.085, 2.40+.119*math.cos(a))],
               .0027, "ld_gray")
    f.tube("ld_clock_hour", [(-.42, -.089, 2.40), (-.478, -.089, 2.434)],
           .004, "ld_gray")
    f.tube("ld_clock_minute", [(-.42, -.091, 2.40), (-.368, -.091, 2.49)],
           .003, "ld_gray")
    f.sphere("ld_clock_spindle", (-.42, -.094, 2.40), .009, "ld_red")


def _bins(b):
    for i, (x, title, accent) in enumerate((
        (-1.92, "GENERAL\nWASTE", "ld_gray"),
        (-1.43, "CLEAN\nRECYCLING", "ld_blue"),
        (-.94, "LABORATORY\nSOLID WASTE", "ld_yellow"),
    )):
        n = f"ld_pedal_bin_{i}"
        b.box(n+"_base", (x, -2.12, .030), (.34, .34, .06), "rubber", .025)
        b.box(n+"_body", (x, -2.12, .285), (.38, .37, .49), "ld_white", .035)
        b.box(n+"_rim", (x, -2.12, .541), (.40, .39, .032), accent, .015)
        b.box(n+"_lid", (x, -2.12, .570), (.405, .395, .028), accent, .019)
        b.box(n+"_hinge", (x, -2.322, .555), (.18, .025, .039), "steel", .006)
        b.box(n+"_pedal", (x, -1.904, .045), (.16, .10, .023), "steel", .006)
        b.label(n+"_label", title, (x, -1.932, .34), (.27, .15),
                rotation=(90, 0, 180), font_size=43)


def _stool(b, name, x, y, seat_height):
    for i in range(5):
        a = 2*math.pi*i/5
        dx, dy = math.cos(a), math.sin(a)
        b.tube(name+f"_base_arm_{i}",
               [(x, y, .155), (x+.19*dx, y+.19*dy, .10),
                (x+.265*dx, y+.265*dy, .071)], .019, "ld_gray", collision=True)
        b.cylinder(name+f"_glide_{i}", (x+.265*dx, y+.265*dy, .026),
                   .030, .052, "rubber")
    b.cylinder(name+"_gas_body", (x, y, .27), .038, .27, "ld_gray")
    b.cylinder(name+"_gas_piston", (x, y, (seat_height+.36)/2),
               .023, seat_height-.36, "chrome")
    b.torus(name+"_foot_ring", (x, y, .265), .215, .013, "chrome", collision=True)
    for i in range(3):
        a = 2*math.pi*i/3
        b.tube(name+f"_ring_spoke_{i}",
               [(x, y, .265), (x+.215*math.cos(a), y+.215*math.sin(a), .265)],
               .010, "chrome", collision=True)
    b.cylinder(name+"_seat_pan", (x, y, seat_height-.021), .189, .028, "ld_gray")
    b.lathe(name+"_seat", (x, y, seat_height-.008),
            [(0, 0), (.185, 0), (.193, .012), (.193, .032),
             (.177, .046), (0, .046)], "ld_pad", collision=True)
    b.tube(name+"_height_lever",
           [(x+.05, y, seat_height-.033),
            (x+.18, y-.045, seat_height-.033)], .007, "chrome")
    b.box(name+"_lever_grip", (x+.195, y-.05, seat_height-.033),
          (.065, .030, .022), "black", .007, collision=False)


def _ceiling_services(b):
    # Flush services complement the room's separate lighting grid.
    for i, (x, y) in enumerate(((-1.35, .55), (2.40, 2.15))):
        n = f"ld_supply_diffuser_{i}"
        b.box(n+"_flange", (x, y, 2.971), (.56, .56, .025),
              "ld_white", .004, collision=False)
        b.box(n+"_recess", (x, y, 2.953), (.49, .49, .009),
              "ld_gray", .003, collision=False)
        for j, size in enumerate((.46, .36, .26, .16)):
            z = 2.943-j*.007
            for side in (-1, 1):
                b.box(n+f"_blade_x_{j}_{side}", (x+side*size/2, y, z),
                      (.024, size+.024, .012), "ld_white", .002, collision=False)
                b.box(n+f"_blade_y_{j}_{side}", (x, y+side*size/2, z),
                      (size-.024, .024, .012), "ld_white", .002, collision=False)
        b.box(n+"_center", (x, y, 2.914), (.12, .12, .014),
              "ld_white", .002, collision=False)
    x, y = .10, -.35
    b.box("ld_return_flange", (x, y, 2.970), (.90, .43, .028),
          "ld_white", .004, collision=False)
    b.box("ld_return_recess", (x, y, 2.950), (.83, .36, .008),
          "ld_gray", .002, collision=False)
    for i in range(16):
        b.box(f"ld_return_louver_{i}", (x-.39+i*.052, y, 2.940),
              (.019, .34, .016), "ld_white", .002,
              rotation=(0, 20, 0), collision=False)
    b.cylinder("ld_smoke_detector_base", (1.48, -.95, 2.978), .082, .030,
               "ld_white", collision=False)
    b.lathe("ld_smoke_detector_dome", (1.48, -.95, 2.923),
            [(0, 0), (.055, 0), (.073, .025), (.078, .050), (0, .050)],
            "ld_white")
    for i in range(14):
        a = 2*math.pi*i/14
        b.box(f"ld_detector_slot_{i}",
              (1.48+.068*math.cos(a), -.95+.068*math.sin(a), 2.947),
              (.007, .010, .012), "ld_gray", .001,
              rotation=(0, 0, math.degrees(a)), collision=False)
    b.sphere("ld_detector_led", (1.48, -1.005, 2.928), .004, "ld_green")
    for i, (x, y) in enumerate(((-.80, -.75), (1.75, 1.65))):
        b.cylinder(f"ld_sprinkler_escutcheon_{i}", (x, y, 2.985),
                   .043, .025, "chrome", collision=False)
        b.cylinder(f"ld_sprinkler_stem_{i}", (x, y, 2.954),
                   .010, .044, "chrome", collision=False)
        b.cylinder(f"ld_sprinkler_deflector_{i}", (x, y, 2.928),
                   .027, .005, "chrome", collision=False)
        b.cylinder(f"ld_sprinkler_bulb_{i}", (x, y, 2.950),
                   .0035, .022, "ld_red", collision=False)
    b.cylinder("ld_occupancy_sensor_base", (2.75, -1.75, 2.984),
               .065, .025, "ld_white", collision=False)
    b.sphere("ld_occupancy_sensor_lens", (2.75, -1.75, 2.957),
             (.048, .048, .026), "ld_white")
    # Slim emergency luminaire sits above, and does not reduce, the exit path.
    b.box("ld_emergency_light_body", (3.445, -1.75, 2.50),
          (.070, .46, .12), "ld_white", .009, collision=False)
    b.box("ld_emergency_light_diffuser", (3.405, -1.75, 2.49),
          (.010, .40, .075), "ld_paper", .004, collision=False)
    b.sphere("ld_emergency_light_indicator", (3.398, -1.94, 2.53),
             .004, "ld_green")


def add_laboratory_dressing(b):
    """Add perimeter detail for the 6.0 × 5.2 m laboratory, returning inventory."""
    _materials(b)
    _eyewash(b)
    _safety_wall(b)
    _documentation(b)
    _bins(b)
    _stool(b, "ld_stool_0", -1., 1.05, .66)
    _stool(b, "ld_stool_1", 2.28, 1.05, .69)
    _ceiling_services(b)
    return {
        "schema": "laboratory.dressing.v1",
        "asset_classes": list(ASSET_CLASSES),
        "asset_class_count": len(ASSET_CLASSES),
        "scope": "static environment dressing; no task manipulation qualification",
        "eyewash_location": [-2.18, -1.24, .90],
        "stool_centers": [[-1., 1.05, .66], [2.28, 1.05, .69]],
        "front_wall_storage_depth_m": .24,
        "unoccupied_island_top_bounds": [-.55, 1.85, -.625, .625, .90],
        "source": "original procedural metric geometry",
    }
