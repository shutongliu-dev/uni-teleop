"""Domestic linen storage and entrance cleaning kit for the laundry scene.

All contents are fixed background scenery, kept below the folding worktop or
inside existing cabinet openings. No task objects or runtime bodies are added.
"""
import math

import numpy as np

from .interiors import _Frame
from .laundry_details import (
    _bottle, _folded_towel, _grid, _jar, _source_asset,
    define_detail_materials,
)


def _fabric_box(b, name, xyz, text, color):
    x, y, z = xyz
    w, d, h = .34, .355, .235
    b.box(name + "_shell", (x, y, z + h / 2), (w, d, h), color, .017)
    b.box(name + "_lid", (x, y, z + h + .009), (w + .008, d + .008, .018), color, .009)
    # Sewing around the lid and metal-reinforced pull distinguish fabric bins.
    for side in (-1, 1):
        b.tube(name + "_seam_" + str(side),
               [(x - w / 2 - .001, y + side * (d / 2 - .018), z + .012),
                (x - w / 2 - .001, y + side * (d / 2 - .018), z + h - .008)],
               .0008, "laundry_linen_cream")
    b.box(name + "_pull_mount", (x - w / 2 - .003, y, z + .165),
          (.006, .10, .036), "laundry_ivory", .004, collision=False)
    b.tube(name + "_pull", [(x - w / 2 - .011, y - .042, z + .168),
                             (x - w / 2 - .025, y - .035, z + .155),
                             (x - w / 2 - .025, y + .035, z + .155),
                             (x - w / 2 - .011, y + .042, z + .168)],
           .004, "laundry_pegs")
    b.label(name + "_label", text, (x - w / 2 - .004, y, z + .085),
            (.17, .045), rotation=(90, 0, -90), font_size=60)


def _tote(b):
    name, x, y, z = "laundry_entrance_tote", -2.475, -1.62, 1.43
    def panel(u, v, side):
        bulge = .027 * math.sin(math.pi * u) * math.sin(math.pi * v)
        wrinkle = .005 * math.sin(u * 26 + v * 9) * math.sin(math.pi * v)
        return (x + side * (.022 + bulge) + wrinkle,
                y + (u - .5) * (.27 - .018 * v),
                z - .32 * v + .008 * math.sin(u * 13) * v)
    for side in (-1, 1):
        fn = lambda u, v, side=side: panel(u, v, side)
        _grid(b, name + "_panel_" + str(side), fn, 30, 36, "laundry_linen_cream", (1, 1.3))
        for edge in (0, 1):
            b.tube(name + f"_hem_{side}_{edge}", [fn(u, edge) for u in np.linspace(0, 1, 30)],
                   .0021, "laundry_linen_cream")
        handle = [(x + side * .023, y + .085 * math.cos(a),
                   z + .25 * math.sin(a)) for a in np.linspace(0, math.pi, 26)]
        b.tube(name + "_handle_" + str(side), handle, .006, "laundry_linen_cream")
    for edge in (0, 1):
        b.tube(name + "_side_seam_" + str(edge),
               [panel(edge, v, 1) for v in np.linspace(0, 1, 30)], .0015, "laundry_linen_ochre")


def _entrance_tools(b):
    # Everything hugs the wall to the left of the room camera, below eye level.
    b.box("laundry_entrance_tool_rail", (-2.561, -1.65, 1.69),
          (.04, .59, .075), "oak", .008)
    for j, y in enumerate((-1.85, -1.62, -1.43)):
        b.tube(f"laundry_entrance_hook_{j}", [(-2.54, y, 1.69), (-2.48, y, 1.69),
                                              (-2.47, y, 1.73)], .005, "laundry_brass")
    # Broom bristles are individual bundles with a lightly splayed silhouette.
    b.tube("laundry_broom_handle", [(-2.515, -1.86, .25), (-2.515, -1.86, 1.65)],
           .012, "laundry_pegs", collision=True)
    b.box("laundry_broom_head", (-2.515, -1.86, .23), (.065, .25, .055),
          "oak", .012)
    for j in range(26):
        for row in range(4):
            yy = -1.975 + j * .0092
            xx = -2.541 + row * .017
            b.tube(f"laundry_broom_bristle_{j}_{row}",
                   [(xx, yy, .207), (xx + (row - 1.5) * .003,
                                     yy + (j - 12.5) * .0014, .11 + .003 * (j % 3))],
                   .002, "laundry_linen_cream")
    b.torus("laundry_broom_hanging_loop", (-2.515, -1.86, 1.658), .021, .003,
             "laundry_caps", rotation=(0, 90, 0))
    # Flat mop pad has a terry edge and a narrow swivel neck.
    b.box("laundry_mop_pad", (-2.485, -1.43, .065), (.14, .23, .025),
          "laundry_linen_sage", .012)
    b.box("laundry_mop_plate", (-2.485, -1.43, .083), (.125, .205, .019),
          "laundry_sage", .008)
    b.tube("laundry_mop_handle", [(-2.485, -1.43, .095), (-2.52, -1.43, .18),
                                 (-2.52, -1.43, 1.65)], .010, "chrome", collision=True)
    b.tube("laundry_mop_grip", [(-2.52, -1.43, 1.40), (-2.52, -1.43, 1.63)],
           .014, "laundry_sage")
    # Dustpan hangs with the scoop opening toward the room.
    b.box("laundry_dustpan_back", (-2.51, -1.43, .82), (.018, .23, .22),
          "laundry_sage", .012)
    b.box("laundry_dustpan_lip", (-2.455, -1.43, .715), (.12, .24, .015),
          "laundry_sage", .005)
    for side in (-1, 1):
        b.box("laundry_dustpan_side_" + str(side), (-2.475, -1.43 + side * .112, .76),
              (.08, .015, .10), "laundry_sage", .005)
    b.tube("laundry_dustpan_grip", [(-2.51, -1.43, .91), (-2.51, -1.43, 1.09)],
           .012, "laundry_caps")
    b.torus("laundry_dustpan_loop", (-2.51, -1.43, 1.11), .018, .004,
             "laundry_caps", rotation=(0, 90, 0))
    b.tube("laundry_dustpan_wall_hook", [(-2.56, -1.43, 1.12), (-2.49, -1.43, 1.12)],
           .004, "laundry_brass")
    _tote(b)


def add_storage_details(b):
    """Return an explicit inventory of 24 fixed storage / housekeeping objects."""
    define_detail_materials(b)
    inventory = []
    def item(name, kind, xyz):
        inventory.append({"name": name, "class": kind, "position": list(xyz), "count": 1})
    # Real woven baskets are fully inside the two island cubbies.
    for i, x in enumerate((.25, 1.05)):
        name = f"laundry_island_basket_{i}"
        _source_asset(b, name, "wicker_basket_02", (x, -.56, .20), scale=1.40)
        item(name, "under_island_woven_basket", (x, -.56, .20))
        for j in range(3):
            name = f"laundry_island_linen_{i}_{j}"
            xyz = (x, -.56, .365 + j * .036)
            _folded_towel(b, name, xyz, (.36, .255, .036),
                          ("laundry_linen_cream", "laundry_linen_sage")[i], seed=j + i)
            item(name, "stored_folded_linen", xyz)
    # Cabinet bottom: linen-covered boxes; middle: spare wash supplies.
    for i, (y, color, label) in enumerate(((.39, "laundry_linen_cream", "BED LINEN"),
                                           (.87, "laundry_linen_sage", "SPARE COVERS"))):
        xyz = (2.49, y, .114)
        name = f"laundry_cabinet_fabric_box_{i}"
        _fabric_box(b, name, xyz, label, color)
        item(name, "fabric_storage_box", xyz)
    for i, (y, color, text) in enumerate(((.30, "laundry_ivory", "LINEN\nRINSE"),
                                         (.48, "laundry_sage", "DELICATE\nWASH"))):
        name = f"laundry_cabinet_spare_product_{i}"
        f = _Frame(b, (2.48, y, 0), -90)
        _bottle(f, name, (0, 0, .474), color, text, height=.23)
        item(name, "spare_laundry_product", (2.48, y, .474))
    f = _Frame(b, (2.48, .74, 0), -90)
    _jar(f, "laundry_cabinet_peg_tin", (0, 0, .474), "CLOTHES\nPEGS", radius=.073, height=.16)
    item("laundry_cabinet_peg_tin", "clothespin_tin", (2.48, .74, .474))
    f = _Frame(b, (2.48, .95, 0), -90)
    _jar(f, "laundry_cabinet_soap_tin", (0, 0, .474), "SOAP\nFLAKES", radius=.07, height=.19)
    item("laundry_cabinet_soap_tin", "spare_soap_tin", (2.48, .95, .474))
    for i, y in enumerate((.39, .84)):
        for j in range(3):
            name = f"laundry_cabinet_folded_towel_{i}_{j}"
            xyz = (2.49, y, .844 + j * .042)
            _folded_towel(b, name, xyz, (.34, .33, .042),
                          ("laundry_linen_white", "laundry_linen_blue")[i], seed=i + j)
            item(name, "cabinet_folded_towel", xyz)
    _entrance_tools(b)
    for name, kind, xyz in (
        ("laundry_broom", "broom", (-2.515, -1.86, .11)),
        ("laundry_mop", "flat_mop", (-2.485, -1.43, .05)),
        ("laundry_dustpan", "dustpan", (-2.51, -1.43, .71)),
        ("laundry_entrance_tote", "canvas_tote", (-2.475, -1.62, 1.11)),
    ):
        item(name, kind, xyz)
    return {"items": inventory, "semantic_object_count": len(inventory),
            "scope": "fixed domestic storage, below worktop and outside the task volume"}
