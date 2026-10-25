"""Render-only apparatus finishes attached to the original physical bodies.

All dimensions are metres. Labels and markings are original procedural meshes,
so authoring in an anonymous USD stage needs no temporary image files or fonts.
These additions do not participate in collision, inertia or task evaluation.
"""
from __future__ import annotations

import math

import numpy as np
import trimesh
from pxr import Gf, Sdf, UsdGeom, UsdShade

from scenes.workrooms.laboratory.object_geometry import box, cylinder, ring


# Original compact block-letter alphabet; each row is five binary pixels.
# Geometry stays legible at oblique views without texture-filtering dependencies.
GLYPHS = {
    " ": [0, 0, 0, 0, 0, 0, 0],
    "A": [14, 17, 17, 31, 17, 17, 17], "B": [30, 17, 17, 30, 17, 17, 30],
    "C": [14, 17, 16, 16, 16, 17, 14], "D": [30, 17, 17, 17, 17, 17, 30],
    "E": [31, 16, 16, 30, 16, 16, 31], "F": [31, 16, 16, 30, 16, 16, 16],
    "G": [14, 17, 16, 23, 17, 17, 15], "H": [17, 17, 17, 31, 17, 17, 17],
    "I": [14, 4, 4, 4, 4, 4, 14], "J": [7, 2, 2, 2, 18, 18, 12],
    "K": [17, 18, 20, 24, 20, 18, 17], "L": [16, 16, 16, 16, 16, 16, 31],
    "M": [17, 27, 21, 21, 17, 17, 17], "N": [17, 25, 21, 19, 17, 17, 17],
    "O": [14, 17, 17, 17, 17, 17, 14], "P": [30, 17, 17, 30, 16, 16, 16],
    "Q": [14, 17, 17, 17, 21, 18, 13], "R": [30, 17, 17, 30, 20, 18, 17],
    "S": [15, 16, 16, 14, 1, 1, 30], "T": [31, 4, 4, 4, 4, 4, 4],
    "U": [17, 17, 17, 17, 17, 17, 14], "V": [17, 17, 17, 17, 17, 10, 4],
    "W": [17, 17, 17, 21, 21, 21, 10], "X": [17, 17, 10, 4, 10, 17, 17],
    "Y": [17, 17, 10, 4, 4, 4, 4], "Z": [31, 1, 2, 4, 8, 16, 31],
    "0": [14, 17, 19, 21, 25, 17, 14], "1": [4, 12, 4, 4, 4, 4, 14],
    "2": [14, 17, 1, 2, 4, 8, 31], "3": [30, 1, 1, 14, 1, 1, 30],
    "4": [2, 6, 10, 18, 31, 2, 2], "5": [31, 16, 16, 30, 1, 1, 30],
    "6": [14, 16, 16, 30, 17, 17, 14], "7": [31, 1, 2, 4, 8, 8, 8],
    "8": [14, 17, 17, 14, 17, 17, 14], "9": [14, 17, 17, 15, 1, 1, 14],
    "-": [0, 0, 0, 31, 0, 0, 0], ".": [0, 0, 0, 0, 0, 6, 6],
    "/": [1, 1, 2, 4, 8, 16, 16], "+": [0, 4, 4, 31, 4, 4, 0],
    # Unit letters deliberately distinguish grams and millilitres from G/ML.
    "g": [0, 0, 15, 17, 15, 1, 14], "m": [0, 0, 26, 21, 21, 21, 21],
}

FINISHES = {
    "frosted_polymer": ((.78, .87, .87), .31, .0),
    "warm_polymer": ((.86, .82, .68), .39, .0),
    "white_polymer": ((.91, .92, .87), .32, .0),
    "label": ((.94, .95, .90), .76, .0),
    "ink": ((.025, .055, .066), .55, .0),
    "blue": ((.035, .24, .52), .28, .0),
    "amber": ((.85, .41, .055), .33, .0),
    "teal": ((.025, .31, .30), .34, .0),
    "rubber": ((.035, .046, .05), .76, .0),
    "steel": ((.53, .61, .64), .23, .80),
    "brushed_steel": ((.38, .45, .48), .38, .72),
}


class DetailAuthor:
    """Author visual children and record their attachment and label semantics."""

    def __init__(self, stage, metadata):
        self.stage, self.meta = stage, metadata
        self.root = metadata["root_path"]
        self.materials, self.paths, self.labels = {}, [], []
        for name, (color, roughness, metallic) in FINISHES.items():
            path = self.root + "/VisualMaterials/" + name
            material = UsdShade.Material.Define(stage, path)
            shader = UsdShade.Shader.Define(stage, path + "/Surface")
            shader.CreateIdAttr("UsdPreviewSurface")
            shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
            shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(roughness)
            shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(metallic)
            material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
            self.materials[name] = material

    def path(self, name):
        return self.meta["actors"].get(name) or self.meta["fixtures"][name]

    def mass_text(self, name, prefix=""):
        mass = next(row["mass_kg"] for row in self.meta["objects"] if row["id"] == name)
        return prefix + format(mass*1000, "g") + " g"

    def bind(self, path, finish):
        """Bind only the display material; retain purpose=physics bindings."""
        UsdShade.MaterialBindingAPI.Apply(self.stage.GetPrimAtPath(path)).Bind(self.materials[finish])

    def mesh(self, body, name, shape, finish):
        return self.faces(body, name, np.asarray(shape.vertices), np.asarray(shape.faces), finish)

    def faces(self, body, name, points, faces, finish):
        root = self.path(body) + "/VisualDetails"
        visual = UsdGeom.Xform.Define(self.stage, root)
        visual.GetPrim().CreateAttribute("laboratory:visualOnly", Sdf.ValueTypeNames.Bool).Set(True)
        path = root + "/" + name
        mesh = UsdGeom.Mesh.Define(self.stage, path)
        mesh.CreatePointsAttr([Gf.Vec3f(*p) for p in points])
        mesh.CreateFaceVertexCountsAttr([len(f) for f in faces])
        mesh.CreateFaceVertexIndicesAttr([int(i) for face in faces for i in face])
        mesh.CreateSubdivisionSchemeAttr("none")
        mesh.CreateDoubleSidedAttr(True)
        self.bind(path, finish)
        self.paths.append(path)
        return path

    def boxes(self, body, name, specifications, finish):
        return self.mesh(body, name, trimesh.util.concatenate([
            box(size, center) for size, center in specifications]), finish)

    def band(self, body, name, inner, outer, z, height, finish, *, x=0., y=0.):
        shape = trimesh.util.concatenate(ring(inner, outer, z-height/2, z+height/2, 64))
        shape.apply_translation((x, y, 0.))
        return self.mesh(body, name, shape, finish)

    def knurl(self, body, radius, z0, z1, finish, *, count=40):
        ribs = []
        for index in range(count):
            angle = 2*math.pi*index/count
            rib = box((.0003, .00065, z1-z0), (radius, 0., (z0+z1)/2))
            rib.apply_transform(trimesh.transformations.rotation_matrix(angle, (0, 0, 1)))
            ribs.append(rib)
        self.mesh(body, "CapGripRibs", trimesh.util.concatenate(ribs), finish)

    @staticmethod
    def surface(center, facing="front", radius=None):
        """Coordinates: front=-Y, rear=+Y, top=+Z; curved text follows a wall."""
        center = np.asarray(center, dtype=float)
        if facing == "top":
            u, v, normal = np.array([1., 0, 0]), np.array([0., 1., 0]), np.array([0., 0., 1.])
        elif facing == "front":
            u, v, normal = np.array([1., 0, 0]), np.array([0., 0., 1.]), np.array([0., -1., 0])
        elif facing == "rear":
            u, v, normal = np.array([-1., 0, 0]), np.array([0., 0., 1.]), np.array([0., 1., 0])
        else:
            raise ValueError("Unknown label facing: " + facing)

        def point(x, y, offset=0.):
            if radius is None:
                return center + x*u + y*v + offset*normal
            angle = x/radius
            return center + (radius+offset)*(math.sin(angle)*u+math.cos(angle)*normal) + y*v
        return point

    def panel(self, body, name, center, size, finish="label", *, facing="front", radius=None):
        point = self.surface(center, facing, radius)
        width, height = size
        segments = 32 if radius else 1
        points = [point(-width/2+width*i/segments, y) for i in range(segments+1)
                  for y in (-height/2, height/2)]
        faces = [[2*i, 2*i+2, 2*i+3, 2*i+1] for i in range(segments)]
        self.faces(body, name, points, faces, finish)

    def text(self, body, name, lines, center, size, *, facing="front", radius=None,
             finish="ink", background=True):
        """One mesh per label, with quad strokes following the bottle surface."""
        width, height = size
        if isinstance(lines, str):
            lines = lines.split("\n")
        if background:
            self.panel(body, name+"Paper", center, size, facing=facing, radius=radius)
        scale = min(width*.89/max(6*len(line)-1 for line in lines), height*.84/(9*len(lines)-2))
        point = self.surface(center, facing, radius)
        points, faces = [], []
        for line_index, line in enumerate(lines):
            x0 = -(6*len(line)-1)*scale/2
            y0 = (9*len(lines)-2)*scale/2-line_index*9*scale
            for char_index, char in enumerate(line):
                for row_index, bits in enumerate(GLYPHS[char]):
                    # Coalesce a contiguous run into one quad to limit mesh size.
                    col = 0
                    while col < 5:
                        if not bits & (1 << (4-col)):
                            col += 1
                            continue
                        start = col
                        while col < 5 and bits & (1 << (4-col)):
                            col += 1
                        x, y = x0+(char_index*6+start)*scale, y0-(row_index+1)*scale
                        quad = [point(x, y, .000035), point(x+(col-start)*scale, y, .000035),
                                point(x+(col-start)*scale, y+scale, .000035), point(x, y+scale, .000035)]
                        faces.append(list(range(len(points), len(points)+4)))
                        points.extend(quad)
        self.faces(body, name+"Ink", points, faces, finish)
        self.labels.append({"body_id": body, "name": name, "text": "\n".join(lines),
                            "facing": facing, "local_center_m": list(center), "size_m": list(size)})

    def finish(self):
        self.meta["visual_detail"] = {
            "schema": "laboratory_apparatus_visual_details.v1", "visual_prim_paths": self.paths,
            "labels": self.labels, "source": "Original procedural geometry and block-letter markings, CC0-1.0.",
            "scope": "Noncollision cosmetic geometry; existing physical shapes, mass, inertia and joints retained.",
            "mass_labels": "Declared nominal object dry masses; labels are not live scale readings.",
            "qualification": "Appearance authoring only; no new manipulation qualification.",
        }
        return self.meta


def _sample_sorting(a):
    for record in a.meta["objects"]:
        name, category = record["id"], record["category"]
        base = a.path(name)
        a.bind(base+"/Shape_000", "frosted_polymer" if category == "blue" else "warm_polymer")
        a.bind(base+"/Shape_001", category)
        a.knurl(name, .0139, .048, .059, category)
        a.band(name, "TamperBand", .0127, .0132, .0455, .0013, category)
        a.band(name, "MouldedHeel", .0127, .01315, .003, .0016, "white_polymer")
        for facing in ("front", "rear"):
            a.text(name, "Identity_"+facing, [name[-2:]+" / "+category.upper(), "SEALED", a.mass_text(name)],
                   (0., 0., .026), (.026, .028), facing=facing, radius=.01316)
            a.panel(name, "Category_"+facing, (0., 0., .0415), (.026, .0027), category,
                    facing=facing, radius=.01318)
        a.text(name, "CapNumber", name[-2:], (0., 0., .06012), (.015, .010), facing="top")
    for category in ("blue", "amber"):
        name = "tray_"+category
        a.text(name, "Classification", category.upper(), (0., -.06012, .011), (.080, .011))
        for side in (-1, 1):
            a.boxes(name, "Rim_"+str(side+1), [((.0012, .112, .0004), (side*.047, 0., .01805))],
                    "white_polymer")


def _vial_racking(a):
    for record in a.meta["objects"]:
        name = record["id"]
        a.bind(a.path(name)+"/Shape_000", "frosted_polymer")
        a.bind(a.path(name)+"/Shape_001", "blue")
        a.knurl(name, .0094, .083, .089, "blue", count=32)
        a.band(name, "SealBand", .0088, .00915, .0795, .0015, "blue")
        a.band(name, "RoundedBaseBand", .0088, .00915, .002, .0014, "white_polymer")
        for facing in ("front", "rear"):
            a.text(name, "VialIdentity_"+facing, ["V"+name[-2:], a.mass_text(name), "SEALED"],
                   (0., 0., .047), (.018, .029), facing=facing, radius=.00916)
            # A clear writing field distinguishes the sample ID from a scale.
            a.panel(name, "WritingStrip_"+facing, (0., 0., .069), (.016, .005),
                    facing=facing, radius=.00916)
        a.text(name, "CapNumber", name[-2:], (0., 0., .09012), (.011, .007), facing="top")
    for index, x in enumerate((-.08, 0., .08), 1):
        a.band("rack", "SocketRim_"+str(index), .0121, .0218, .06608, .00016, "white_polymer", x=x)
        # Keep the whole number plate ahead of the post at y=[-.033,-.021].
        a.text("rack", "SocketNumber_"+str(index), str(index), (x, -.043, .00615), (.018, .014), facing="top")
        a.text("supply_cradle", "LaneNumber_"+str(index), "V"+str(index).zfill(2),
               (x, -.065, .00415), (.035, .012), facing="top")
    a.text("rack", "Identification", "OPEN VIAL RACK", (0., -.05515, .003), (.20, .0045))


def _cap_placement(a):
    a.bind(a.path("sample_bottle")+"/Shape_000", "white_polymer")
    a.bind(a.path("sample_bottle")+"/Shape_001", "frosted_polymer")
    # Existing helical contacts are recolored only, preserving their SDF topology.
    for index in range(2, 11):
        a.bind(a.path("sample_bottle")+f"/Shape_{index:03}", "warm_polymer")
    for facing, y in (("front", -.02215), ("rear", .02215)):
        a.text("sample_bottle", "Identity_"+facing, ["SAMPLE", "B-01", a.mass_text("sample_bottle", "BODY ")],
               (0., y, .027), (.036, .037), facing=facing)
    a.knurl("cap", .0269, -.008, .010, "blue", count=56)
    a.band("cap", "LowerRim", .0266, .02715, -.0083, .0012, "blue")
    a.text("cap", "CapIdentity", ["TURN", a.mass_text("cap")], (0., 0., .01415), (.032, .021), facing="top")
    a.text("bottle_holder", "HolderTag", "BOTTLE", (0., -.03515, .022), (.052, .012))
    a.text("cap_tray", "CapTrayTag", "CAP", (0., -.05015, .010), (.065, .010))
    # Alignment marks are painted on the closure, not a kinematic constraint.
    a.boxes("cap", "ClosureIndex", [((.0015, .00018, .011), (0., -.02714, -.001))], "label")
    a.boxes("sample_bottle", "BottleIndex", [((.0015, .00018, .007), (0., -.02215, .047))], "blue")


def _solid_weighing(a):
    for record in a.meta["objects"]:
        if record["id"].startswith("pellet_"):
            continue
        for path in record["collider_paths"]:
            a.bind(path, "white_polymer" if record["id"] == "weighing_cup" else "brushed_steel")
    for name in ("source_tray", "scoop_rest", "balance_platform"):
        for path in a.meta["fixture_metadata"][name]["collider_paths"]:
            a.bind(path, "teal" if name == "scoop_rest" else "brushed_steel")
    a.bind(a.path("balance_platform")+"/Collision/base", "ink")
    a.bind(a.path("balance_platform")+"/Collision/pan", "steel")
    a.text("source_tray", "GranuleLabel", ["MODEL GRANULES", "24 X 0.5 g"],
           (0., -.05515, .014), (.118, .018))
    a.text("weighing_cup", "BoatLabel", ["BOAT", a.mass_text("weighing_cup", "TARE ")],
           (0., -.03265, .020), (.060, .021))
    a.text("weighing_cup", "BoatLabelRear", ["BOAT", a.mass_text("weighing_cup", "TARE ")], (0., .03265, .020),
           (.060, .021), facing="rear")
    a.boxes("weighing_cup", "RolledRim", [
        ((.075, .0012, .0004), (0., side*.0317, .03505)) for side in (-1, 1)], "white_polymer")
    a.text("scoop", "HandleStamp", a.mass_text("scoop"), (-.085, 0., .01315),
           (.038, .009), facing="top", background=False)
    a.boxes("scoop", "HandleGrip", [((.001, .013, .00025), (-.133+i*.004, 0., .01310))
                                      for i in range(7)], "rubber")
    a.text("scoop_rest", "RestTag", "SCOOP", (.050, -.023, .00615), (.050, .011), facing="top")
    # The -Y top strip and vertical face are reserved for the live mass display.
    a.text("balance_platform", "ScaleRearTag", ["LAB", "NOMINAL"], (0., .06265, .0115),
           (.080, .018), facing="rear")
    for index, x in enumerate((-.058, .058)):
        a.mesh("balance_platform", "CaseScrew_"+str(index), cylinder(.002, .0003, (x, .052, .02205)), "steel")
        a.boxes("balance_platform", "ScrewSlot_"+str(index), [((.0025, .00045, .00012), (x, .052, .02224))], "ink")


def _pipette_transfer(a):
    containers = a.meta["task_parameters"]["hydraulics"]["containers"]
    for name, container in containers.items():
        source = name == "source"
        radius, height = (.036, .080) if source else (.025, .060)
        # Continuous outer lip/base highlights never bridge the open vessel.
        a.band(name, "RimHighlight", radius-.0007, radius+.00012, height-.0006, .0012, "white_polymer")
        a.band(name, "FootRing", radius-.0005, radius+.00012, .0025, .0010, "white_polymer")
        identity = ["SOURCE" if source else name[-1].upper(), a.mass_text(name, "DRY ")]
        a.text(name, "VesselIdentity", identity, (0., 0., height-.011),
               (.040 if source else .030, .016), radius=radius+.00016)
        volume_marks = (20, 40, 60, 80, 100, 120, 160, 200) if source else (5, 10, 15, 20, 30, 40, 50)
        for ml in volume_marks:
            z = container["floor_z_m"]+ml*1.e-6/(math.pi*container["radius_m"]**2)
            # The zero follows the interior floor and spacing follows actual bore area.
            a.panel(name, "VolumeTick_"+str(ml), (0., 0., z), (.010, .00045), "blue",
                    facing="rear", radius=radius+.00017)
            a.text(name, "VolumeNumber_"+str(ml), str(ml), (0., 0., z+.00155), (.010, .0025),
                   facing="rear", radius=radius+.00019, finish="blue", background=False)
        a.text(name, "VolumeUnit", "mL", (0., 0., height-.004), (.014, .005),
               facing="rear", radius=radius+.00019, finish="blue", background=False)
        # Earlier unnumbered marks were not tied to volume; retain their disabled
        # collider definitions but suppress their appearance in the new asset.
        for index in range(1, 4):
            UsdGeom.Imageable(a.stage.GetPrimAtPath(a.path(name)+f"/Graduation_{index}")).CreateVisibilityAttr("invisible")
    area = a.meta["task_parameters"]["hydraulics"]["piston_area_m2"]
    for ml in (0, 5, 10, 15, 20):
        z = .028+ml*1.e-6/area
        for facing in ("front", "rear"):
            a.panel("pipette", f"StrokeTick_{ml}_{facing}", (0., 0., z), (.010, .0005), "blue",
                    facing=facing, radius=.01416)
            a.text("pipette", f"StrokeValue_{ml}_{facing}", str(ml), (0., 0., z+.0024), (.011, .0032),
                   facing=facing, radius=.01419, finish="blue", background=False)
    a.text("pipette", "ToolName", "mL", (0., 0., .082), (.011, .004), radius=.01419,
           finish="blue", background=False)
    a.text("pipette", "DryMass", a.mass_text("pipette"), (.015, 0., .08915),
           (.016, .007), facing="top")
    a.text("plunger", "ThumbPadLabel", ["PUSH", a.mass_text("plunger")], (0., 0., .08515),
           (.019, .013), facing="top")
    a.band("plunger", "SealLipTop", .0113, .01178, .0011, .0005, "rubber")
    a.band("plunger", "SealLipBottom", .0113, .01178, -.0011, .0005, "rubber")
    a.text("pipette_cradle", "CradleLabel", "PISTON TOOL", (.83, .48185, .903), (.076, .0045))


def add_task_visual_details(stage, metadata):
    """Apply the explicitly versioned visual finish to a freshly authored task."""
    author = DetailAuthor(stage, metadata)
    {"sample_sorting": _sample_sorting, "vial_racking": _vial_racking,
     "cap_placement": _cap_placement, "solid_weighing": _solid_weighing,
     "pipette_transfer": _pipette_transfer}[metadata["task_id"]](author)
    return author.finish()
