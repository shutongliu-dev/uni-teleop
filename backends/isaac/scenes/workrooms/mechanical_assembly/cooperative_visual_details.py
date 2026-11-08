"""Original, label-free machining details for the cooperative insertion props.

Only display geometry and display bindings change. The original colliders stay
active and retain every physical attribute. Small cosmetic bevels stay inside
the original solid; shallow surface finishes extend by at most 0.15 mm.
"""
from __future__ import annotations

import math
import hashlib
import json
from pathlib import Path

import numpy as np
import trimesh
from pxr import Gf, Sdf, UsdGeom, UsdShade

from bench.paths import DATA_ROOT, load_json
from scenes.workrooms.laboratory.object_geometry import box, cylinder, tray_parts


FINISHES = {
    "anodized_teal": ((.030, .235, .235), .31, .72),
    "machined_aluminum": ((.59, .65, .67), .26, .87),
    "shaft_steel": ((.47, .53, .57), .23, .91),
    "steel_grain": ((.38, .43, .46), .37, .83),
    "gunmetal": ((.075, .090, .105), .30, .78),
    "recess": ((.014, .020, .024), .60, .25),
    "brass_grip": ((.56, .35, .105), .32, .75),
    "brass_face": ((.67, .46, .17), .26, .80),
    "grip_inset": ((.046, .063, .066), .65, .12),
    "grip_ridge": ((.080, .103, .108), .56, .18),
    "tray_blue": ((.025, .130, .245), .40, .48),
    "tray_inside": ((.048, .186, .285), .49, .35),
    "tray_grain": ((.051, .196, .296), .53, .31),
}


def _beveled_box(size, center=(0., 0., 0.), bevel=.0003):
    """Cut only the corners; preserve original face locations and solid bounds."""
    half = np.asarray(size, float)/2
    bevel = min(bevel, float(half.min())*.4)
    points = []
    for axis in range(3):
        for signs in np.ndindex(2, 2, 2):
            point = half-bevel
            point[axis] = half[axis]
            points.append(point*(2*np.asarray(signs)-1)+center)
    return trimesh.convex.convex_hull(np.asarray(points))


def _lathe(profile, sections=128):
    points = [[r*math.cos(a), r*math.sin(a), z]
              for z, r in profile for a in np.arange(sections)*2*math.pi/sections]
    faces = []
    for row in range(len(profile)-1):
        for i in range(sections):
            j = (i+1) % sections
            a, b = row*sections+i, row*sections+j
            c, d = (row+1)*sections+j, (row+1)*sections+i
            faces.extend(((a, b, c), (a, c, d)))
    for i in range(1, sections-1):
        faces.append((0, i+1, i))
        k = (len(profile)-1)*sections
        faces.append((k, k+i, k+i+1))
    return trimesh.Trimesh(vertices=points, faces=faces, process=False)


def _annulus(inner, outer, z, sections=128):
    points = [[r*math.cos(a), r*math.sin(a), z]
              for r in (inner, outer) for a in np.arange(sections)*2*math.pi/sections]
    faces = []
    for i in range(sections):
        j = (i+1) % sections
        faces.extend(((i, sections+i, sections+j), (i, sections+j, j)))
    return trimesh.Trimesh(vertices=points, faces=faces, process=False)


class _Details:
    def __init__(self, stage, metadata):
        self.stage, self.meta = stage, metadata
        self.root = metadata['root_path']
        self.materials, self.paths, self.hidden = {}, [], []
        if stage.GetPrimAtPath(self.root+'/CooperativeVisualMaterials'):
            raise ValueError('Cooperative visual details have already been applied')
        for name, (color, roughness, metallic) in FINISHES.items():
            path = self.root+'/CooperativeVisualMaterials/'+name
            material = UsdShade.Material.Define(stage, path)
            shader = UsdShade.Shader.Define(stage, path+'/Surface')
            shader.CreateIdAttr('UsdPreviewSurface')
            shader.CreateInput('diffuseColor', Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
            shader.CreateInput('roughness', Sdf.ValueTypeNames.Float).Set(roughness)
            shader.CreateInput('metallic', Sdf.ValueTypeNames.Float).Set(metallic)
            material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), 'surface')
            self.materials[name] = material

    def path(self, body):
        return self.meta['actors'].get(body) or self.meta['fixtures'][body]

    def mesh(self, body, name, shape, finish):
        root = self.path(body)+'/VisualDetails'
        group = UsdGeom.Xform.Define(self.stage, root)
        group.GetPrim().CreateAttribute('mechanical:visualOnly', Sdf.ValueTypeNames.Bool).Set(True)
        path = root+'/'+name
        mesh = UsdGeom.Mesh.Define(self.stage, path)
        mesh.CreatePointsAttr([Gf.Vec3f(*point) for point in shape.vertices])
        mesh.CreateFaceVertexCountsAttr([3]*len(shape.faces))
        mesh.CreateFaceVertexIndicesAttr(np.asarray(shape.faces).reshape(-1).tolist())
        mesh.CreateSubdivisionSchemeAttr('none')
        # Flat normals preserve the actual shaft facets, crisp machined rims
        # and the intentionally small bevel facets.
        mesh.CreateNormalsAttr([Gf.Vec3f(*n) for n in np.repeat(shape.face_normals, 3, axis=0)])
        mesh.SetNormalsInterpolation(UsdGeom.Tokens.faceVarying)
        UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(self.materials[finish])
        self.paths.append(path)

    def shapes(self, body, name, shapes, finish):
        self.mesh(body, name, trimesh.util.concatenate(shapes), finish)

    def hide_original_display(self, body):
        # Visibility does not disable a collider or alter a physics binding.
        records = self.meta['objects']+list(self.meta['fixture_metadata'].values())
        record = next(row for row in records if row['id'] == body)
        for path in record['collider_paths']:
            UsdGeom.Imageable(self.stage.GetPrimAtPath(path)).CreateVisibilityAttr().Set('invisible')
            self.hidden.append(path)

    def screws(self, body, centers):
        rings, cores = [], []
        for x, y, z in centers:
            rings.append(cylinder(.0026, .00012, (x, y, z), sections=32))
            cores.append(cylinder(.00125, .000025, (x, y, z+.000075), sections=6))
        self.shapes(body, 'FlushFastenerFaces', rings, 'gunmetal')
        self.shapes(body, 'HexSocketShading', cores, 'recess')


def _sleeve(a):
    a.hide_original_display('sleeve')
    walls = [((.018, .064, .052), (-.023, 0., .034)),
             ((.018, .064, .052), (.023, 0., .034)),
             ((.028, .018, .052), (0., -.023, .034)),
             ((.028, .018, .052), (0., .023, .034))]
    a.shapes('sleeve', 'AnodizedHousing', [_beveled_box(s, c) for s, c in walls], 'anodized_teal')
    a.mesh('sleeve', 'MachinedFoot', _beveled_box((.074, .070, .008), (0., 0., .004)),
           'machined_aluminum')
    a.mesh('sleeve', 'SideGripBody', _beveled_box((.048, .028, .018), (0., -.048, .025)),
           'anodized_teal')
    # Four independent rim surfaces surround the *real* 28 mm opening. They
    # never cap it, narrow it or create a second apparent insertion target.
    rim = [box((.004, .036, .00006), (x, 0., .06002)) for x in (-.016, .016)]
    rim += [box((.028, .004, .00006), (0., y, .06002)) for y in (-.016, .016)]
    a.shapes('sleeve', 'MachinedMouthRim', rim, 'machined_aluminum')
    a.screws('sleeve', [(x, y, .06004) for x in (-.025, .025) for y in (-.024, .024)])
    a.mesh('sleeve', 'GripInset', _beveled_box((.040, .022, .00010), (0., -.048, .03402), .00002),
           'grip_inset')
    ribs = [box((.00065, .018, .00004), (x, -.048, .03409))
            for x in np.linspace(-.017, .017, 13)]
    a.shapes('sleeve', 'FineGripRibs', ribs, 'grip_ridge')
    # Shallow machining grain on the visible base flange, not a new seam or
    # simulated fastener operation.
    grain = [box((.000045, .060, .00002), (x, 0., .00802))
             for x in np.linspace(-.0355, -.033, 12)]
    grain += [box((.000045, .060, .00002), (x, 0., .00802))
              for x in np.linspace(.033, .0355, 12)]
    a.shapes('sleeve', 'FootMachiningGrain', grain, 'steel_grain')


def _peg(a):
    a.hide_original_display('peg')
    a.mesh('peg', 'GroundSquareShaft', _beveled_box((.020, .020, .080), (0., 0., .040), .00018),
           'shaft_steel')
    # The native head is a 32-sided cylinder. Every cosmetic radius is within
    # its inscribed radius, including the knurl crests.
    profile = [(.080, .02440), (.08045, .02485), (.083, .02485),
               (.08325, .02458), (.08365, .02458), (.08390, .02485),
               (.09610, .02485), (.09635, .02458), (.09675, .02458),
               (.097, .02485), (.09955, .02485), (.100, .02440)]
    a.mesh('peg', 'TurnedHead', _lathe(profile), 'brass_grip')
    # Fine cut grip grooves are represented as darker facets on the surface;
    # they do not imply protruding teeth or alter grasp collision.
    grooves = []
    for i in range(96):
        angle = 2*math.pi*i/96
        strip = box((.00002, .00019, .0108), (.024855, 0., .090))
        strip.apply_transform(trimesh.transformations.rotation_matrix(angle, (0, 0, 1)))
        grooves.append(strip)
    a.shapes('peg', 'HeadGripCutLines', grooves, 'gunmetal')
    a.mesh('peg', 'SatinHeadFace', cylinder(.0224, .000025, (0., 0., .10002), sections=128), 'brass_face')
    face_rings = []
    for radius in (.012, .016, .019, .0215):
        face_rings.append(_annulus(radius-.000025, radius+.000025, .100035))
    a.shapes('peg', 'ConcentricTurningGrain', face_rings, 'brass_grip')
    shaft_lines = []
    for side in (-1, 1):
        for v in np.linspace(-.008, .008, 17):
            shaft_lines.append(box((.000025, .000025, .068), (v, side*.01001, .039)))
            shaft_lines.append(box((.000025, .000025, .068), (side*.01001, v, .039)))
    a.shapes('peg', 'LongitudinalGrindingGrain', shaft_lines, 'steel_grain')


def _tray(a):
    a.hide_original_display('collection_tray')
    pieces = tray_parts((.26, .28, .026), floor=.004, wall=.004)
    a.shapes('collection_tray', 'FoldedTrayShell',
             [_beveled_box(part.extents, part.bounds.mean(0), .0003) for part in pieces], 'tray_blue')
    a.mesh('collection_tray', 'SatinInnerFloor',
           _beveled_box((.250, .270, .00003), (0., 0., .00402), .000005), 'tray_inside')
    # A narrow exposed folded edge highlights the existing rim; the full
    # opening and 26 mm wall height remain unchanged.
    edges = [box((.0011, .276, .00004), (x, 0., .02602)) for x in (-.128, .128)]
    edges += [box((.256, .0011, .00004), (0., y, .02602)) for y in (-.138, .138)]
    a.shapes('collection_tray', 'FoldedRimHighlight', edges, 'machined_aluminum')
    grain = [box((.242, .000055, .000012), (0., y, .004045))
             for y in np.linspace(-.127, .127, 92)]
    a.shapes('collection_tray', 'FineFloorFinish', grain, 'tray_grain')
    # Pressed corner and perimeter lines stay directly on the existing floor.
    border = [box((.0010, .254, .000025), (x, 0., .00404)) for x in (-.1195, .1195)]
    border += [box((.240, .0010, .000025), (0., y, .00404)) for y in (-.127, .127)]
    a.shapes('collection_tray', 'PressedFloorBorder', border, 'tray_blue')


def _apply_v1(stage, metadata):
    """Attach finishes to current body frames, independent of robot layout."""
    a = _Details(stage, metadata)
    _sleeve(a)
    _peg(a)
    _tray(a)
    metadata['visual_detail'] = {
        'schema': 'mechanical_cooperative_visual_details.v1',
        'source': 'Original procedural geometry and finishes; CC0-1.0.',
        'visual_prim_paths': a.paths,
        'hidden_collider_display_paths': a.hidden,
        'labels': [],
        'scope': 'Render-only machining, grip and folded-edge details; all native physical properties unchanged.',
        'maximum_surface_overlay_m': .00015,
        'maximum_cosmetic_corner_inset_m': .0006,
        'mating_geometry': 'Original 20 mm square shaft, 28 mm square socket and 52 mm usable depth retained.',
        'qualification': 'Appearance authoring, not new manipulation or force-calibration evidence.',
    }
    return metadata


# A separate profile keeps the historical v1 authoring byte-for-byte stable.
MATERIAL_ROOT_V2 = Path(f'{DATA_ROOT}/assets/task_parts/mechanical_assembly/'
                        'cooperative_insertion/visual_materials/v2')
FINISHES_V2 = {
    'v2_milled_silver': ((.58, .64, .68), .29, .90),
    'v2_ground_steel': ((.44, .50, .55), .24, .94),
    'v2_teal_enamel': ((.025, .115, .120), .38, .50),
    'v2_dark_grip': ((.028, .038, .043), .62, .18),
    'v2_grip_land': ((.070, .095, .100), .52, .26),
    'v2_brass_knurl': ((.53, .34, .105), .34, .84),
    'v2_brass_face': ((.64, .44, .155), .29, .87),
    'v2_blue_enamel': ((.026, .125, .205), .42, .42),
    'v2_tray_floor': ((.055, .170, .235), .53, .34),
    'v2_seam': ((.030, .042, .050), .55, .40),
}


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _visual_materials_v2():
    """Create original CC0 maps once, then verify their immutable provenance.

    All relief is shading-only: no displacement or physics material is used.
    The maps are generated here so the asset has no downloaded texture source.
    """
    from PIL import Image

    root = MATERIAL_ROOT_V2
    if not root.exists():
        root.mkdir(parents=True)
        n, seed = 1024, 20261021
        rng = np.random.default_rng(seed)
        y, x = np.mgrid[0:n, 0:n] / n
        noise = rng.standard_normal((n, n))
        # Periodic fields avoid seams when a physically scaled material repeats.
        grain = (.60*np.sin(2*np.pi*y*173) + .25*np.sin(2*np.pi*y*359)
                 + .15*np.sin(2*np.pi*(y*61+x*3)))
        satin = .42*np.sin(2*np.pi*x*251)*np.sin(2*np.pi*y*249)+.20*noise
        scuffs = np.zeros((n, n))
        for _ in range(23):
            cx, cy = rng.uniform(.04, .96, 2)
            angle = rng.uniform(-.7, .7)
            u = (x-cx)*np.cos(angle)+(y-cy)*np.sin(angle)
            v = -(x-cx)*np.sin(angle)+(y-cy)*np.cos(angle)
            scuffs += np.exp(-(v/.00065)**2)*np.exp(-(u/rng.uniform(.015, .085))**8)
        files = {}
        for name, (color, roughness, _) in FINISHES_V2.items():
            metal = name in ('v2_milled_silver', 'v2_ground_steel', 'v2_brass_face')
            floor = name == 'v2_tray_floor'
            field = grain if metal else satin
            modulation = 1+.022*field + (.030*scuffs if floor else 0)
            linear = np.clip(np.asarray(color)[None, None, :]*modulation[:, :, None], 0, 1)
            srgb = np.where(linear <= .0031308, linear*12.92,
                            1.055*np.power(linear, 1/2.4)-.055)
            maps = {
                'color': np.uint8(np.clip(srgb*255, 0, 255)),
                'roughness': np.uint8(np.clip(roughness+.023*field
                                     + (.085*scuffs if floor else 0), .08, .9)*255),
            }
            # Very shallow finish grain, never a raised functional feature.
            normal = np.empty((n, n, 3))
            normal[:, :, 0] = .018*np.sin(2*np.pi*x*251) if not metal else .004*noise
            normal[:, :, 1] = .028*np.cos(2*np.pi*y*173) if metal else .018*np.cos(2*np.pi*y*249)
            normal[:, :, 2] = 1.
            normal /= np.linalg.norm(normal, axis=2, keepdims=True)
            maps['normal'] = np.uint8(np.clip((normal*.5+.5)*255, 0, 255))
            for role, pixels in maps.items():
                path = root/f'{name}_{role}.png'
                Image.fromarray(pixels).save(path, compress_level=9)
                files[path.name] = _sha256(path)
        license_path = root/'LICENSE.txt'
        license_path.write_text('Original procedural material maps for Mechanical Task10.\n'
                                'Copyright 2026. Dedicated to the public domain under CC0 1.0.\n'
                                'https://creativecommons.org/publicdomain/zero/1.0/\n'
                                'No photographs, external source images or third-party textures were used.\n')
        files[license_path.name] = _sha256(license_path)
        provenance = {
            'schema': 'mechanical_cooperative_materials.v2', 'license': 'CC0-1.0',
            'source': 'Original deterministic NumPy procedural color, roughness and tangent-normal maps.',
            'generator': 'scenes/workrooms/mechanical_assembly/cooperative_visual_details.py',
            'generator_sha256': _sha256(__file__), 'seed': seed, 'resolution': [n, n],
            'external_sources': [], 'displacement': False, 'files_sha256': files,
            'physical_effect': 'None; visual surface finish only.',
        }
        (root/'provenance.json').write_text(json.dumps(provenance, indent=2, sort_keys=True)+'\n')
    provenance = load_json(root/'provenance.json')
    for name, expected in provenance['files_sha256'].items():
        if _sha256(root/name) != expected:
            raise ValueError(f'Immutable cooperative material changed: {root/name}')
    return {str(root/name): value for name, value in provenance['files_sha256'].items()} | {
        str(root/'provenance.json'): _sha256(root/'provenance.json')}


class _DetailsV2(_Details):
    def __init__(self, stage, metadata):
        super().__init__(stage, metadata)
        self.dependencies = _visual_materials_v2()
        for name, (color, roughness, metallic) in FINISHES_V2.items():
            path = self.root+'/CooperativeVisualMaterials/'+name
            material = UsdShade.Material.Define(stage, path)
            shader = UsdShade.Shader.Define(stage, path+'/Surface')
            shader.CreateIdAttr('UsdPreviewSurface')
            shader.CreateInput('metallic', Sdf.ValueTypeNames.Float).Set(metallic)
            uv = UsdShade.Shader.Define(stage, path+'/UV')
            uv.CreateIdAttr('UsdPrimvarReader_float2')
            uv.CreateInput('varname', Sdf.ValueTypeNames.Token).Set('st')
            uv.CreateOutput('result', Sdf.ValueTypeNames.Float2)
            for role, target, value_type, output in (
                ('color', 'diffuseColor', Sdf.ValueTypeNames.Color3f, 'rgb'),
                ('roughness', 'roughness', Sdf.ValueTypeNames.Float, 'r'),
                ('normal', 'normal', Sdf.ValueTypeNames.Normal3f, 'rgb'),
            ):
                tex = UsdShade.Shader.Define(stage, path+'/'+role.title())
                tex.CreateIdAttr('UsdUVTexture')
                tex.CreateInput('file', Sdf.ValueTypeNames.Asset).Set(
                    Sdf.AssetPath(str(MATERIAL_ROOT_V2/f'{name}_{role}.png')))
                tex.CreateInput('sourceColorSpace', Sdf.ValueTypeNames.Token).Set('sRGB' if role == 'color' else 'raw')
                tex.CreateInput('wrapS', Sdf.ValueTypeNames.Token).Set('repeat')
                tex.CreateInput('wrapT', Sdf.ValueTypeNames.Token).Set('repeat')
                tex.CreateInput('st', Sdf.ValueTypeNames.Float2).ConnectToSource(uv.ConnectableAPI(), 'result')
                if role == 'normal':
                    tex.CreateInput('scale', Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(2, 2, 2, 1))
                    tex.CreateInput('bias', Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(-1, -1, -1, 0))
                tex.CreateOutput(output, Sdf.ValueTypeNames.Float if output == 'r' else Sdf.ValueTypeNames.Float3)
                shader.CreateInput(target, value_type).ConnectToSource(tex.ConnectableAPI(), output)
            material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), 'surface')
            self.materials[name] = material

    def mesh(self, body, name, shape, finish):
        super().mesh(body, name, shape, finish)
        # Face-varying physical-scale projections keep the bore and shaft grain
        # readable without smoothing away the true square section.
        faces = np.asarray(shape.vertices)[np.asarray(shape.faces)]
        normals = np.asarray(shape.face_normals)
        uv = np.empty((len(faces), 3, 2))
        for axis, axes in ((0, (1, 2)), (1, (0, 2)), (2, (0, 1))):
            mask = np.argmax(np.abs(normals), axis=1) == axis
            uv[mask] = faces[mask][:, :, axes]
        scale = .28 if body == 'collection_tray' else .080
        uv = uv.reshape(-1, 2)/scale+.5
        mesh = UsdGeom.Mesh(self.stage.GetPrimAtPath(self.paths[-1]))
        UsdGeom.PrimvarsAPI(mesh).CreatePrimvar('st', Sdf.ValueTypeNames.TexCoord2fArray,
                                              UsdGeom.Tokens.faceVarying).Set(
            [Gf.Vec2f(*value) for value in uv])


def _oriented_face(shape, center, normal):
    transform = trimesh.geometry.align_vectors((0, 0, 1), normal)
    shape.apply_transform(transform)
    shape.apply_translation(center)
    return shape


def _flush_fasteners_v2(a, body, name, centers, normal=(0, 0, 1)):
    """Shallow inlaid fastener faces: a real open hex ring over a dark recess."""
    rings, recesses, lands = [], [], []
    for center in centers:
        rings.append(_oriented_face(_annulus(.0012, .0025, .000055, 6), center, normal))
        lands.append(_oriented_face(_annulus(.0025, .0029, .000045, 32), center, normal))
        recesses.append(_oriented_face(cylinder(.0012, .00002, (0, 0, .00002), sections=6), center, normal))
    a.shapes(body, name+'Sockets', rings, 'v2_ground_steel')
    a.shapes(body, name+'Seats', lands, 'v2_milled_silver')
    a.shapes(body, name+'Recesses', recesses, 'v2_seam')


def _sleeve_v2(a):
    a.hide_original_display('sleeve')
    walls = [((.018, .064, .052), (-.023, 0., .034)),
             ((.018, .064, .052), (.023, 0., .034)),
             ((.028, .018, .052), (0., -.023, .034)),
             ((.028, .018, .052), (0., .023, .034))]
    # The underlying silver machining is also the actual visible bore wall.
    # Every edge cut removes display material; nothing enters the 28 mm bore.
    a.shapes('sleeve', 'MilledHousingAndSquareBore',
             [_beveled_box(s, c, .00055) for s, c in walls], 'v2_milled_silver')
    a.mesh('sleeve', 'MachinedFoot', _beveled_box((.074, .070, .008), (0, 0, .004), .00065), 'v2_milled_silver')
    panels = [box((.00006, .058, .042), (x, 0, .034)) for x in (-.03202, .03202)]
    panels += [box((.058, .00006, .042), (0, y, .034)) for y in (-.03202, .03202)]
    a.shapes('sleeve', 'RecessedTealSideFinish', panels, 'v2_teal_enamel')
    # Two millimetre frame lands and substantial finish regions remain readable
    # in the task camera; they are coatings, not extra colliding cover plates.
    seams = [box((.000035, .057, .0011), (x, 0, z))
             for x in (-.032065, .032065) for z in (.014, .054)]
    seams += [box((.057, .000035, .0011), (0, y, z))
              for y in (-.032065, .032065) for z in (.014, .054)]
    a.shapes('sleeve', 'HousingAssemblySeams', seams, 'v2_seam')
    # The top annular land deliberately exposes both the 28 mm square opening
    # and its silver 52 mm-deep interior instead of suggesting a blind hole.
    rim = [box((.0032, .036, .00004), (x, 0, .060015)) for x in (-.0156, .0156)]
    rim += [box((.028, .0032, .00004), (0, y, .060015)) for y in (-.0156, .0156)]
    a.shapes('sleeve', 'GroundSquareMouthLand', rim, 'v2_ground_steel')
    _flush_fasteners_v2(a, 'sleeve', 'TopRetainers',
                        [(x, y, .060) for x in (-.025, .025) for y in (-.024, .024)])
    a.mesh('sleeve', 'GripMetalSubstrate', _beveled_box((.048, .028, .018), (0, -.048, .025), .00065), 'v2_milled_silver')
    a.mesh('sleeve', 'GripTopInlay', box((.043, .023, .00006), (0, -.048, .03402)), 'v2_dark_grip')
    a.mesh('sleeve', 'GripFrontInlay', box((.042, .00006, .013), (0, -.06202, .025)), 'v2_dark_grip')
    ribs = [box((.0013, .018, .00004), (x, -.048, .03408)) for x in np.linspace(-.017, .017, 11)]
    ribs += [box((.0013, .00004, .010), (x, -.06208, .025)) for x in np.linspace(-.017, .017, 11)]
    a.shapes('sleeve', 'BroadGripTextureLands', ribs, 'v2_grip_land')
    _flush_fasteners_v2(a, 'sleeve', 'GripEndRetainers',
                        [(x, -.048, .034) for x in (-.020, .020)])


def _diamond_knurl(z0=.0833, z1=.0967, sections=48, rows=10):
    """Cut diamond facets inside the original 32-sided cylindrical collider.

    Approximately 3.25 mm-wide diamonds are deliberately readable at task
    resolution. Crests stay inside the collider's 24.8796 mm inscribed radius.
    """
    points, faces = [], []
    theta_step, dz = 2*math.pi/sections, (z1-z0)/(rows-1)
    for row in range(rows):
        z = z0+row*dz
        for col in range(sections):
            theta = (col+.5*(row % 2))*theta_step
            # A diamond is clipped at the two end rings without crossing them.
            corners = [(theta-theta_step/2, z), (theta, min(z1, z+dz)),
                       (theta+theta_step/2, z), (theta, max(z0, z-dz))]
            base = len(points)
            points.append([.02484*math.cos(theta), .02484*math.sin(theta), z])
            for angle, height in corners:
                points.append([.02462*math.cos(angle), .02462*math.sin(angle), height])
            for i in range(4):
                faces.append((base, base+1+i, base+1+(i+1) % 4))
    # Diamond ordering is clockwise as seen from outside; fix winding once.
    mesh = trimesh.Trimesh(vertices=points, faces=np.asarray(faces)[:, ::-1], process=False)
    return mesh


def _peg_v2(a):
    a.hide_original_display('peg')
    a.mesh('peg', 'GroundSquareShaft', _beveled_box((.020, .020, .080), (0, 0, .040), .00020), 'v2_ground_steel')
    # The head is one physical component. Colour/finish bands describe turned
    # machining and knurling, without adding a separate movable knob or cap.
    a.mesh('peg', 'TurnedHeadLowerShoulder',
           _lathe([(.080, .0241), (.0806, .02484), (.0825, .02484), (.0833, .02462)]), 'v2_milled_silver')
    a.mesh('peg', 'KnurlRoot', _lathe([(.0833, .02460), (.0967, .02460)]), 'v2_brass_knurl')
    a.mesh('peg', 'DiamondCutGrip', _diamond_knurl(), 'v2_brass_knurl')
    a.mesh('peg', 'TurnedHeadUpperShoulder',
           _lathe([(.0967, .02462), (.0975, .02484), (.0993, .02484), (.100, .0241)]), 'v2_milled_silver')
    a.mesh('peg', 'SatinTurnedEndFace', cylinder(.0226, .00002, (0, 0, .100015), sections=128), 'v2_brass_face')
    rings = [_annulus(inner, outer, .10003) for inner, outer in
             ((.0226, .0238), (.0198, .02015), (.0157, .0159), (.0107, .01085))]
    a.shapes('peg', 'ConcentricFaceMachining', rings, 'v2_ground_steel')
    # Visible but flush grinding passes show square shaft flats; the insertion
    # section is unchanged and no surface reaches beyond ±10.02 mm.
    passes = [box((.0018, .00002, .069), (v, sign*.01001, .040))
              for sign in (-1, 1) for v in (-.006, .006)]
    passes += [box((.00002, .0018, .069), (sign*.01001, v, .040))
               for sign in (-1, 1) for v in (-.006, .006)]
    a.shapes('peg', 'FlatGrindingPasses', passes, 'v2_milled_silver')


def _tray_v2(a):
    a.hide_original_display('collection_tray')
    pieces = tray_parts((.26, .28, .026), floor=.004, wall=.004)
    a.shapes('collection_tray', 'FoldedSteelTrayShell',
             [_beveled_box(p.extents, p.bounds.mean(0), .00065) for p in pieces], 'v2_blue_enamel')
    a.mesh('collection_tray', 'SatinNonSlipInnerFloor',
           box((.250, .270, .000035), (0, 0, .004025)), 'v2_tray_floor')
    rims = [box((.0030, .274, .00004), (x, 0, .026025)) for x in (-.128, .128)]
    rims += [box((.254, .0030, .00004), (0, y, .026025)) for y in (-.138, .138)]
    a.shapes('collection_tray', 'PressedSteelTopHem', rims, 'v2_milled_silver')
    border = [box((.0025, .264, .000035), (x, 0, .00407)) for x in (-.122, .122)]
    border += [box((.2415, .0025, .000035), (0, y, .00407)) for y in (-.132, .132)]
    a.shapes('collection_tray', 'ShallowPressedFloorLand', border, 'v2_blue_enamel')
    # Four corner laps and inset seam strips explain how the sheet-metal sides
    # join. They lie on the existing wall, not across the open tray interior.
    corner_laps = [box((.010, .000045, .017), (x, y, .0145))
                   for x in (-.119, .119) for y in (-.14002, .14002)]
    a.shapes('collection_tray', 'FoldedCornerLaps', corner_laps, 'v2_ground_steel')
    seams = [box((.00075, .000025, .016), (x, y, .0145))
             for x in (-.1138, .1138) for y in (-.14006, .14006)]
    a.shapes('collection_tray', 'CornerJointLines', seams, 'v2_seam')
    for sign, label in ((-1, 'Front'), (1, 'Rear')):
        _flush_fasteners_v2(a, 'collection_tray', label+'CornerRivets',
                            [(x, sign*.140045, .015) for x in (-.119, .119)], (0, sign, 0))
    # Thin metal bands expose the folded wall return, within the original wall.
    returns = [box((.00003, .260, .0015), (x, 0, .0237)) for x in (-.126025, .126025)]
    returns += [box((.245, .00003, .0015), (0, y, .0237)) for y in (-.136025, .136025)]
    a.shapes('collection_tray', 'InnerHemReturn', returns, 'v2_milled_silver')


def apply_visual_details(stage, metadata, *, profile='v1'):
    """Attach versioned display-only finishes, independent of robot layout."""
    if profile == 'v1':
        return _apply_v1(stage, metadata)
    if profile != 'v2':
        raise ValueError(f'Unknown cooperative visual profile: {profile}')
    a = _DetailsV2(stage, metadata)
    _sleeve_v2(a)
    _peg_v2(a)
    _tray_v2(a)
    metadata['visual_detail'] = {
        'schema': 'mechanical_cooperative_visual_details.v2', 'profile': 'v2',
        'source': 'Original procedural display geometry and PBR material maps; CC0-1.0.',
        'visual_prim_paths': a.paths, 'hidden_collider_display_paths': a.hidden,
        'dependencies': a.dependencies, 'labels': [],
        'scope': 'Render-only machined housing, square bore, diamond knurl and folded tray; native physics unchanged.',
        'maximum_surface_overlay_m': .00015,
        'maximum_cosmetic_corner_inset_m': .0010,
        'corner_inset_scope': 'Non-mating housing/foot/tray edges <=0.65 mm; turned head edge <=0.90 mm; shaft edge <=0.20 mm.',
        'mating_geometry': 'Original 20 mm square shaft, 28 mm square socket and 52 mm usable depth retained.',
        'material_displacement': False,
        'qualification': 'Appearance authoring only; no new manipulation or force-calibration claim.',
    }
    return metadata
