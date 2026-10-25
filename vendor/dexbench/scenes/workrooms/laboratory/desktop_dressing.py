"""Task-owned, fixed desktop supplies for the five laboratory tasks.

The island is supplied by the pinned room. This opt-in layer occupies its ends
and a narrow service-side strip, leaving a shared central approach and all rear
task targets clear. Supplies have real static collision outlines by default;
they are scenery, not dynamic task objects or scoring supports. No room, robot,
physics world, controller, external texture or runtime state is authored here.
"""
from __future__ import annotations

import math

import numpy as np
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

from bench.simulation.assets.workrooms.builder import Builder, safe
from .object_geometry import attr, schema


SCHEMA = "laboratory_desktop_dressing.v1"
TABLE_Z_M = .9
CONTACT_PRIOR = {"static_friction": .30, "dynamic_friction": .25, "restitution": .01,
                 "contact_offset_m": .0002, "rest_offset_m": 0.,
                 "source": "Declared synthetic dry-contact engineering prior, not measured calibration."}
ZONES = {
    "west_supplies": [[-.47, -.34, TABLE_Z_M], [.33, .52, 1.16]],
    "east_supplies": [[1.16, -.34, TABLE_Z_M], [1.78, .51, 1.16]],
    "service_strip": [[.36, -.395, TABLE_Z_M], [1.04, -.305, 1.02]],
}
KEEP_CLEAR = {
    "central_task_approach": [[.35, -.30, TABLE_Z_M], [1.12, .635, 1.60]],
    "rear_task_targets": [[.40, 1.99, TABLE_Z_M], [1.04, 2.20, 1.60]],
    "robot_aisle": [[-.55, .64, .0], [1.85, 1.98, 1.80]],
}


class _DesktopBuilder(Builder):
    """Use existing rounded visual recipes entirely within a task subtree."""

    def __init__(self, stage, root_path, collision):
        self.stage, self.root = stage, root_path
        self.materials, self.records, self._names = {}, [], set()
        self.dependencies = {}
        self.collision = bool(collision)
        self.items, self.active = [], None
        UsdGeom.Xform.Define(stage, root_path)
        for name, color, roughness, metal, opacity in (
            ("paper", (.89, .90, .84), .85, 0., 1.),
            ("white", (.77, .84, .83), .32, 0., 1.),
            ("teal", (.035, .29, .31), .42, 0., 1.),
            ("blue", (.045, .22, .46), .37, 0., 1.),
            ("cyan", (.17, .55, .60), .48, 0., 1.),
            ("amber", (.33, .13, .035), .20, 0., 1.),
            ("steel", (.48, .54, .55), .22, .78, 1.),
            ("ink", (.035, .055, .06), .56, 0., 1.),
            ("rubber", (.065, .09, .09), .78, 0., 1.),
            ("red", (.58, .12, .075), .47, 0., 1.),
            ("yellow", (.83, .60, .16), .51, 0., 1.),
            ("lens", (.46, .75, .78), .12, 0., .42),
        ):
            material = UsdShade.Material.Define(stage, root_path+"/Materials/"+name)
            shader = UsdShade.Shader.Define(stage, str(material.GetPath())+"/Surface")
            shader.CreateIdAttr("UsdPreviewSurface")
            shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
            for key, value in (("roughness", roughness), ("metallic", metal), ("opacity", opacity)):
                shader.CreateInput(key, Sdf.ValueTypeNames.Float).Set(value)
            material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
            self.materials[name] = material
        self.contact = UsdShade.Material.Define(stage, root_path+"/Materials/DryContact")
        physical = UsdPhysics.MaterialAPI.Apply(self.contact.GetPrim())
        physical.CreateStaticFrictionAttr(CONTACT_PRIOR["static_friction"])
        physical.CreateDynamicFrictionAttr(CONTACT_PRIOR["dynamic_friction"])
        physical.CreateRestitutionAttr(CONTACT_PRIOR["restitution"])

    def path(self, name):
        path = self.active["path"]+"/Visual/"+safe(name)
        if path in self._names:
            raise ValueError("Duplicate dressing mesh: "+path)
        self._names.add(path)
        return path

    def item(self, name, zone, center, *, yaw=0., description, asset_count=1):
        path = self.root+"/Items/"+name
        if self.stage.GetPrimAtPath(path):
            raise ValueError("Duplicate desktop item: "+name)
        prim = UsdGeom.Xform.Define(self.stage, path)
        self.xform(prim.GetPrim(), center, (0., 0., yaw))
        prim.GetPrim().SetCustomDataByKey("laboratory:role", "fixed_background_supply")
        self.active = {"id": name, "path": path, "zone": zone,
            "description": description, "visible_asset_count": asset_count,
            "collider_paths": [], "position_m": list(center), "yaw_deg": yaw,
            "dynamic": False, "task_object": False, "scoring_support": False}
        self.items.append(self.active)

    def solid_box(self, name, center, size, material, bevel=.002, rotation=(0, 0, 0)):
        path = self.box(name, center, size, material, bevel, rotation, collision=False)
        if self.collision:
            collider = UsdGeom.Cube.Define(self.stage, path+"/Collider")
            collider.CreateSizeAttr(1.)
            self.xform(collider.GetPrim(), (0, 0, 0), scale=size)
            self._collider(collider.GetPrim())
        return path

    def solid_cylinder(self, name, center, radius, height, material, rotation=(0, 0, 0)):
        path = self.cylinder(name, center, radius, height, material, rotation, collision=False)
        if self.collision:
            collider = UsdGeom.Cylinder.Define(self.stage, path+"/Collider")
            collider.CreateRadiusAttr(radius)
            collider.CreateHeightAttr(height)
            collider.CreateAxisAttr("Z")
            self._collider(collider.GetPrim())
        return path

    def _collider(self, prim, *, invisible=True):
        if invisible:
            UsdGeom.Imageable(prim).CreateVisibilityAttr().Set("invisible")
        UsdPhysics.CollisionAPI.Apply(prim)
        schema(prim, "PhysxCollisionAPI")
        attr(prim, "physxCollision:contactOffset", Sdf.ValueTypeNames.Float, CONTACT_PRIOR["contact_offset_m"])
        attr(prim, "physxCollision:restOffset", Sdf.ValueTypeNames.Float, CONTACT_PRIOR["rest_offset_m"])
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(self.contact, materialPurpose="physics")
        self.active["collider_paths"].append(str(prim.GetPath()))

    def outline_box(self, name, center, size):
        """Add a static silhouette to a detailed compound visual, in item frame."""
        if not self.collision:
            return
        prim = UsdGeom.Cube.Define(self.stage, self.active["path"]+"/Collision/"+name)
        prim.CreateSizeAttr(1.)
        self.xform(prim.GetPrim(), center, scale=size)
        self._collider(prim.GetPrim())

    def outline_cylinder(self, name, center, radius, height):
        if not self.collision:
            return
        prim = UsdGeom.Cylinder.Define(self.stage, self.active["path"]+"/Collision/"+name)
        prim.CreateRadiusAttr(radius)
        prim.CreateHeightAttr(height)
        prim.CreateAxisAttr("Z")
        self.xform(prim.GetPrim(), center)
        self._collider(prim.GetPrim())

    def tag(self, name, center, size, *, code=1, rotation=(0, 0, 0), color="paper"):
        """Embossed label stock, colored header, ruled batch fields and barcode."""
        w, h = size
        # Label lies on local XY, then its group rotates to a bottle/box face.
        path = self.path(name)
        group = UsdGeom.Xform.Define(self.stage, path)
        self.xform(group.GetPrim(), center, rotation)
        previous = self.active
        self.active = {"path": path}
        self.box("stock", (0, 0, 0), (w, h, .00035), color, .0001, collision=False)
        self.box("header", (0, h*.32, .00024), (w*.87, h*.14, .00012),
                 "teal", .00005, collision=False)
        for row, width in enumerate((.64, .79, .48)):
            self.box("field_"+str(row), (-w*(.87-width)/2, h*(.09-.16*row), .00025),
                     (w*width, h*.028, .0001), "ink", 0., collision=False)
        for i in range(13):
            width = w*(.012 if (i+code)%3 else .024)
            self.box("barcode_"+str(i), (w*(-.40+i*.065), -h*.36, .00025),
                     (width, h*.11, .0001), "ink", 0., collision=False)
        self.active = previous


def _tray(b, size, *, color="white", floor=.004, wall=.003, height=.018):
    w, d = size
    b.solid_box("tray_floor", (0, 0, floor/2), (w, d, floor), color, .002)
    for sign in (-1, 1):
        b.solid_box("tray_side_"+str(sign+1), (sign*(w-wall)/2, 0, height/2),
                    (wall, d, height), color, .001)
        b.solid_box("tray_end_"+str(sign+1), (0, sign*(d-wall)/2, height/2),
                    (w-2*wall, wall, height), color, .001)


def _bottle(b, prefix, xy, *, radius=.032, height=.15, color="amber", cap="blue", base=.005):
    x, y = xy
    b.lathe(prefix+"_body", (x, y, base),
            [(0, 0), (radius*.83, 0), (radius, .004), (radius, height*.71),
             (radius*.91, height*.78), (radius*.52, height*.85),
             (radius*.49, height*.97), (0, height*.97)], color)
    # The shoulder is bounded by a separate small cylinder; no tall box fills
    # surrounding air or extends up to the full radius around the narrow neck.
    b.outline_cylinder(prefix+"_body_outline", (x, y, base+height*.37), radius, height*.74)
    b.solid_cylinder(prefix+"_neck", (x, y, base+height*.85), radius*.53, height*.24, color)
    b.solid_cylinder(prefix+"_cap", (x, y, base+height), radius*.59, .022, cap)
    b.torus(prefix+"_tamper_ring", (x, y, base+height-.013), radius*.52, .0015, "white")
    for i in range(24):
        a = 2*math.pi*i/24
        b.box(prefix+"_cap_rib_"+str(i),
              (x+radius*.586*math.cos(a), y+radius*.586*math.sin(a), base+height),
              (.0013, .0017, .017), cap, .0003, rotation=(0, 0, math.degrees(a)), collision=False)
    def patch(name, angle, arc_width, z, h, material, offset=.00015):
        vertices, normals, faces = [], [], []
        for zz in (z-h/2, z+h/2):
            for i in range(25):
                a = angle+arc_width*(i/24-.5)
                vertices.append((x+(radius+offset)*math.cos(a), y+(radius+offset)*math.sin(a), zz))
                normals.append((math.cos(a), math.sin(a), 0.))
        for i in range(24):
            faces.append((i, i+1, i+26, i+25))
        b.mesh(b.path(prefix+"_label_"+name), vertices, faces, material, normals=normals)

    label_z, label_h = base+height*.43, height*.44
    patch("paper", math.pi/2, 1.55, label_z, label_h, "paper")
    patch("header", math.pi/2, 1.36, label_z+label_h*.32, label_h*.14, "teal", .00020)
    for i, span in enumerate((1.13, 1.28, .90)):
        patch("field_"+str(i), math.pi/2+.04, span, label_z+label_h*(.09-i*.16),
              label_h*.028, "ink", .00021)
    for i in range(13):
        patch("barcode_"+str(i), math.pi/2-.60+i*.10, .018 if i%3 else .035,
              label_z-label_h*.36, label_h*.11, "ink", .00022)


def _west(b):
    b.item("reagent_station", "west_supplies", (-.26, -.19, TABLE_Z_M),
           description="Fixed spill tray with three sealed, labelled stock bottles", asset_count=4)
    _tray(b, (.36, .26), color="steel", height=.023)
    for i, (x, y, radius, height) in enumerate(((-.108, -.030, .033, .155),
            (0, .032, .032, .177), (.108, -.023, .030, .138))):
        _bottle(b, "stock_"+str(i), (x, y), radius=radius, height=height,
                cap=("blue", "white", "teal")[i])

    b.item("wash_bottle", "west_supplies", (.055, -.235, TABLE_Z_M),
           description="Fixed labelled wash bottle with bent dispensing spout")
    _bottle(b, "wash", (0, 0), radius=.041, height=.155, color="white", cap="cyan", base=0.)
    b.tube("bent_spout", [(0, 0, .168), (0, 0, .215), (.025, 0, .235), (.065, 0, .235),
                          (.072, 0, .223)], .004, "white")
    b.outline_box("spout_upright", (0, 0, .192), (.009, .009, .052))
    b.outline_box("spout_tip", (.041, 0, .226), (.070, .009, .019))

    b.item("bench_wipes", "west_supplies", (.18, .02, TABLE_Z_M),
           description="Fixed low-lint wipe dispenser with folded sheets")
    b.solid_box("carton", (0, 0, .037), (.22, .13, .074), "teal", .006)
    b.box("top_seam", (0, 0, .0742), (.20, .11, .0006), "cyan", .001, collision=False)
    b.box("dispensing_slot", (0, 0, .0748), (.115, .024, .001), "ink", .006, collision=False)
    for i in range(4):
        b.box("fold_"+str(i), (-.014+i*.011, 0, .086+i*.001), (.019, .040, .025),
              "paper", .0015, rotation=(0, 12*(-1)**i, 0), collision=False)
    b.outline_box("folded_tissue", (.003, 0, .088), (.058, .041, .028))
    b.tag("batch_label", (0, .0653, .037), (.158, .043), rotation=(-90, 0, 0))

    b.item("open_logbook", "west_supplies", (.125, .305, TABLE_Z_M), yaw=-3.,
           description="Fixed open ruled laboratory logbook, page stack, binding and ribbon")
    b.solid_box("cover", (0, 0, .004), (.28, .305, .008), "blue", .003)
    for side in (-1, 1):
        x = side*.069
        b.solid_box("page_block_"+str(side+1), (x, 0, .011), (.132, .292, .008), "paper", .002)
        for layer in range(5):
            b.box("page_edge_%s_%s" % (side+1, layer), (x, .1462, .008+layer*.0014),
                  (.129, .0003, .0002), "white", 0., collision=False)
        for line in range(15):
            yy = -.121+line*.016
            b.box("rule_%s_%s" % (side+1, line), (x, yy, .0152), (.116, .00055, .00012),
                  "cyan", 0., collision=False)
            if line in (2, 3, 5, 7, 10):
                points = [(x-.047+j*.009, yy+.003+.0013*math.sin(j*2.2+line), .0154)
                          for j in range(5+(line%4))]
                b.tube("notes_%s_%s" % (side+1, line), points, .00035, "ink")
        b.box("margin_"+str(side+1), (x-.047, 0, .0153), (.0005, .282, .0001), "red", 0., collision=False)
    b.box("binding", (0, 0, .011), (.004, .29, .005), "blue", .001, collision=False)
    b.box("bookmark", (.021, -.110, .0156), (.007, .088, .0005), "yellow", .0001, collision=False)

    b.item("logbook_pen", "west_supplies", (.303, .29, TABLE_Z_M),
           description="Fixed capped laboratory marker with barrel clip")
    b.solid_cylinder("barrel", (0, 0, .007), .006, .138, "white", (90, 0, 0))
    b.solid_cylinder("cap", (0, .073, .007), .007, .029, "blue", (90, 0, 0))
    b.box("clip", (0, .067, .013), (.003, .036, .0025), "blue", .0008, collision=False)
    for y in (-.035, -.028, -.021):
        b.box("barrel_mark_"+str(round(y*1000)), (0, y, .0121), (.004, .0012, .0002), "ink", 0., collision=False)

    b.item("safety_glasses", "west_supplies", (-.285, .12, TABLE_Z_M), yaw=5.,
           description="Fixed safety spectacles with translucent lenses and folded temples")
    for side in (-1, 1):
        x = side*.041
        b.box("frame_"+str(side+1), (x, .0, .018), (.074, .045, .012), "teal", .009, collision=False)
        b.box("lens_"+str(side+1), (x, .0, .025), (.063, .035, .002), "lens", .006, collision=False)
        b.outline_box("lens_outline_"+str(side+1), (x, 0, .020), (.074, .045, .018))
        points = [(side*.075, .0, .018), (side*.086, -.036, .009),
                  (side*.060, -.067, .003), (side*.042, -.066, .003)]
        b.tube("temple_"+str(side+1), points, .003, "teal")
        b.outline_box("temple_outline_"+str(side+1), (side*.069, -.044, .008), (.039, .051, .011))
        b.cylinder("hinge_"+str(side+1), (side*.075, .003, .027), .0022, .0015, "steel", collision=False)
    b.solid_box("bridge", (0, .006, .016), (.015, .013, .009), "teal", .003)

    b.item("petri_stack", "west_supplies", (-.285, .335, TABLE_Z_M),
           description="Three fixed empty lidded sample dishes with concentric rims", asset_count=3)
    for i in range(3):
        z = i*.013
        b.solid_cylinder("dish_"+str(i), (0, 0, z+.005), .057, .010, "white")
        b.lathe("rim_"+str(i), (0, 0, z), [(.054, .002), (.059, .002), (.059, .013),
                (.0565, .013), (.0565, .009), (.054, .009)], "lens")
        b.torus("lid_seam_"+str(i), (0, 0, z+.012), .058, .001, "white")
    b.tag("lot_label", (0, 0, .0395), (.068, .026), code=5)

    b.item("interval_timer", "west_supplies", (-.135, .484, TABLE_Z_M),
           description="Fixed idle bench timer, LCD and separate buttons")
    b.solid_box("case", (0, 0, .014), (.116, .050, .028), "white", .006)
    b.box("bezel", (-.017, 0, .0282), (.064, .032, .001), "rubber", .003, collision=False)
    b.box("screen", (-.017, 0, .0289), (.057, .025, .0006), "cyan", .001, collision=False)
    # Empty LCD segments denote an idle display; it is never a live clock.
    for i in range(4):
        x = -.038+i*.014
        for yy in (-.008, .008):
            b.box("digit_%s_%s" % (i, int(yy*1000)), (x, yy, .0294), (.008, .0008, .0002), "ink", 0., collision=False)
        for dx in (-.0045, .0045):
            b.box("digit_side_%s_%s" % (i, int(dx*10000)), (x+dx, 0, .0294), (.0008, .015, .0002), "ink", 0., collision=False)
    for i, color in enumerate(("teal", "red")):
        b.cylinder("button_"+str(i), (.040, -.011+i*.022, .0295), .006, .003, color, collision=False)


def _east(b):
    b.item("sample_archive_box", "east_supplies", (1.39, -.19, TABLE_Z_M),
           description="Fixed closed archive box with grid index, lid seam and latches")
    b.solid_box("body", (0, 0, .039), (.28, .22, .078), "cyan", .008)
    b.solid_box("lid", (0, 0, .083), (.286, .226, .012), "white", .006)
    b.box("lid_inset", (0, 0, .0893), (.244, .184, .001), "paper", .003, collision=False)
    for i in range(6):
        b.box("grid_col_"+str(i), (-.11+i*.044, 0, .090), (.00065, .15, .0002), "cyan", 0., collision=False)
    for i in range(4):
        b.box("grid_row_"+str(i), (0, -.075+i*.05, .090), (.22, .00065, .0002), "cyan", 0., collision=False)
    for side in (-1, 1):
        b.box("latch_"+str(side+1), (side*.085, .113, .073), (.034, .005, .024), "teal", .002, collision=False)
    b.tag("archive_label", (0, .1104, .040), (.158, .038), rotation=(-90, 0, 0), code=7)

    b.item("spare_tube_rack", "east_supplies", (1.435, .085, TABLE_Z_M),
           description="Fixed spare capped-tube rack with rails, open slots and individual tubes", asset_count=13)
    b.solid_box("base", (0, 0, .004), (.31, .16, .008), "teal", .003)
    for side in (-1, 1):
        b.solid_box("rack_end_"+str(side+1), (side*.150, 0, .038), (.010, .15, .068), "teal", .002)
    for row in range(2):
        y = -.040+row*.080
        for side in (-1, 1):
            b.solid_box("rack_rail_%s_%s" % (row, side+1), (0, y+side*.016, .047),
                        (.29, .006, .007), "cyan", .001)
        for col in range(6):
            x = -.115+col*.046
            name = "spare_%s_%s" % (row, col)
            b.solid_cylinder(name+"_tube", (x, y, .044), .010, .068, "white")
            b.solid_cylinder(name+"_cap", (x, y, .080), .012, .012, "blue" if row else "teal")
            b.torus(name+"_lip", (x, y, .075), .011, .0011, "white")
            b.box(name+"_label", (x, y+.0102, .047), (.010, .0007, .025), "paper", .0002, collision=False)

    b.item("glove_carton", "east_supplies", (1.46, .36, TABLE_Z_M),
           description="Fixed nitrile glove carton, open dispensing slot and cuff folds")
    b.solid_box("carton", (0, 0, .038), (.33, .20, .076), "blue", .006)
    b.box("top_print", (0, 0, .0763), (.30, .17, .0006), "cyan", .002, collision=False)
    b.box("slot", (.040, 0, .077), (.145, .048, .0008), "ink", .012, collision=False)
    b.tag("carton_label", (-.096, 0, .0775), (.082, .128), code=8)
    for i in range(3):
        b.box("cuff_"+str(i), (.032+i*.015, .003, .082+i*.001), (.029, .060, .010),
              "cyan", .004, rotation=(0, 12*(-1)**i, -4), collision=False)
    b.outline_box("cuffs", (.047, .003, .084), (.080, .064, .016))

    b.item("sterile_swab_jar", "east_supplies", (1.69, -.23, TABLE_Z_M),
           description="Fixed sealed sterilization jar with brushed cap and batch label")
    _bottle(b, "jar", (0, 0), radius=.041, height=.120, color="white", cap="steel", base=0.)

    b.item("label_roll", "east_supplies", (1.69, -.035, TABLE_Z_M),
           description="Fixed hollow roll of laboratory identification labels")
    # Separate annular segments preserve the actual core opening.
    from .object_geometry import ring
    for index, shape in enumerate(ring(.020, .045, 0, .034, 32)):
        name = "roll_segment_"+str(index)
        path = b.path(name)
        prim = b.mesh(path, shape.vertices, shape.faces, "paper")
        if b.collision:
            UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr("convexHull")
            b._collider(prim, invisible=False)
    b.torus("core_top", (0, 0, .0343), .0205, .001, "yellow")
    for radius in (.026, .033, .040, .044):
        b.torus("winding_"+str(round(radius*1000)), (0, 0, .0344), radius, .00025, "white")
    b.solid_box("label_tail", (0, .061, .0006), (.032, .038, .0012), "paper", .0002)
    b.tag("tail_print", (0, .061, .0014), (.025, .028), code=3)

    b.item("cap_organizer", "east_supplies", (1.69, .17, TABLE_Z_M),
           description="Fixed small parts tray containing spare color-coded sealing caps", asset_count=7)
    _tray(b, (.115, .165), color="white", height=.014)
    for row in range(3):
        for col in range(2):
            x, y = -.026+col*.052, -.050+row*.050
            color = ("blue", "teal", "red")[row]
            b.solid_cylinder("cap_%s_%s" % (row, col), (x, y, .011), .017, .014, color)
            b.torus("cap_lip_%s_%s" % (row, col), (x, y, .018), .015, .001, "white")

    b.item("benchtop_brush", "east_supplies", (1.69, .405, TABLE_Z_M),
           description="Fixed compact anti-static cleanup brush with bristle rows")
    b.solid_box("handle", (0, 0, .011), (.055, .13, .016), "white", .006)
    b.solid_box("bristle_body", (0, -.035, .004), (.052, .050, .008), "rubber", .001)
    for col in range(6):
        for row in range(3):
            b.box("bristle_%s_%s" % (row, col), (-.022+col*.009, -.059-row*.004, .003),
                  (.004, .020, .006), "rubber", .0005, collision=False)

    b.item("spare_marker", "east_supplies", (1.19, .13, TABLE_Z_M),
           description="Fixed fine-tip marker beside consumable supplies")
    b.solid_cylinder("body", (0, 0, .007), .006, .15, "white", (90, 0, 0))
    b.solid_cylinder("cap", (0, .080, .007), .007, .020, "teal", (90, 0, 0))


def _service_strip(b):
    b.item("sealed_tip_cassette", "service_strip", (.70, -.351, TABLE_Z_M),
           description="Fixed sealed narrow consumable cassette along service edge", asset_count=11)
    b.solid_box("cassette", (0, 0, .012), (.61, .070, .024), "teal", .003)
    for i in range(10):
        x = -.267+i*.0593
        b.solid_box("packet_"+str(i), (x, 0, .030), (.052, .056, .012), "paper", .002)
        b.box("packet_seal_"+str(i), (x, -.018, .0362), (.046, .006, .0006),
              "cyan", .0003, collision=False)
        for line in range(3):
            b.box("packet_line_%s_%s" % (i, line), (x-.008+line*.007, .004, .0363),
                  (.0013, .019, .0002), "blue", 0., collision=False)


def author_desktop_dressing(stage, root_path="/World/LaboratoryTask/DesktopDressing", *, collision=True):
    """Append a self-contained background layer and return its frozen inventory.

    Call this once when building a new task asset and retain the returned value
    as ``metadata['desktop_dressing']``. Collision is opt-out for image-only
    diagnostics; task releases use the default. The caller must not add these
    paths to its dynamic actors or referee's task-support identities.
    """
    if not root_path.startswith("/World/LaboratoryTask/") or root_path.endswith("/"):
        raise ValueError("Desktop dressing must belong to a laboratory task subtree")
    if stage.GetPrimAtPath(root_path):
        raise ValueError("Desktop dressing already exists; author a fresh task version")
    b = _DesktopBuilder(stage, root_path, collision)
    _west(b)
    _east(b)
    _service_strip(b)
    # Include invisible collision outlines when reserving space, not only the
    # slightly smaller rounded visual surfaces.
    bbox = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render"],
                            useExtentsHint=False, ignoreVisibility=True)
    for item in b.items:
        bound = bbox.ComputeWorldBound(stage.GetPrimAtPath(item["path"])).ComputeAlignedRange()
        item["world_bounds_m"] = [list(bound.GetMin()), list(bound.GetMax())]
        lower, upper = np.asarray(item["world_bounds_m"])
        zone_low, zone_high = np.asarray(ZONES[item["zone"]])
        if np.any(lower < zone_low-1e-6) or np.any(upper > zone_high+1e-6):
            raise ValueError("Desktop item escapes its reserved zone: "+item["id"])
        for name, bounds in KEEP_CLEAR.items():
            keep_low, keep_high = np.asarray(bounds)
            if np.all(upper > keep_low) and np.all(lower < keep_high):
                raise ValueError("Desktop dressing blocks "+name+": "+item["id"])
    return {"schema": SCHEMA, "root_path": root_path, "collision_enabled": bool(collision),
        "ownership": "task_owned_fixed_background", "dynamic_actor_count": 0,
        "fixed_group_count": len(b.items),
        "visible_asset_count": sum(item["visible_asset_count"] for item in b.items),
        "collider_count": sum(len(item["collider_paths"]) for item in b.items),
        "items": b.items, "zones_world_bounds_m": ZONES, "keep_clear_world_bounds_m": KEEP_CLEAR,
        "source": {"kind": "original_procedural", "license": "CC0-1.0", "external_assets": []},
        "physics_prior": dict(CONTACT_PRIOR),
        "collision_model": "Static primitive outlines; open label core uses convex annular pieces. Fine visual ribs and print have no independent colliders.",
        "function_boundary": "Fixed background supplies only: not movable task objects, not valid scoring supports, no functioning instruments or consumable chemistry.",
        "qualification": "Authored geometry and static keep-clear checks only; native clearance/contact evidence belongs to each new release."}
