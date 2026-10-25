"""Task-owned wooden tower scenery, with independent rigid blocks.

The explicit dimensions below are a scene candidate, not a user-approved or
native-qualified task release. Rule/configuration factories remain authoritative.
No room, robot, light, PhysicsScene, grasp constraint or object drive is authored.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np
from pxr import Gf, Sdf, UsdGeom, UsdPhysics, UsdShade

from .tower_geometry import block_ids, initial_tower, validate_geometry
from bench.paths import DATA_ROOT, load_json


CANDIDATE_GEOMETRY = {'block_size_m': [.15, .045, .025], 'layers': 6,
                      'in_layer_gap_m': .001}
ASSETS_ROOT = Path(os.environ.get('LIVING_TASK_ASSETS_ROOT', f'{DATA_ROOT}/assets'))
MATERIAL_RELATIVE = Path('materials/polyhaven/v1/kitchen_wood')
PHYSICS_PRIOR = {
    'density_kg_m3': 650., 'static_friction': .45, 'dynamic_friction': .35,
    'restitution': 0., 'contact_offset_m': .0001, 'rest_offset_m': 0.,
    'source': 'Engineering prior for a homogeneous dry wooden toy block; not measured or calibrated.',
    'inertia': 'Exact uniform rectangular-solid formula for the conservative box collider.',
    'collision': 'One full-size primitive box per block; visual edge radius 0.4 mm lies inside it.',
    'visual_edge_radius_m': .0004, 'native_accepted': False,
}


def _dependency(path):
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'bytes': path.stat().st_size}


def _material_source():
    folder = ASSETS_ROOT / MATERIAL_RELATIVE
    source_path = folder / 'SOURCE.json'
    source = load_json(source_path)
    files = {item['path']: item for item in source['files']}
    maps = {'diffuseColor': 'kitchen_wood_diff_1k.jpg',
            'normal': 'kitchen_wood_nor_gl_1k.jpg',
            'roughness': 'kitchen_wood_rough_1k.jpg'}
    dependencies = [_dependency(source_path)]
    for name in maps.values():
        item = _dependency(folder / name)
        if item['sha256'] != files[name]['sha256']:
            raise ValueError('Shared wood texture differs from its provenance: ' + name)
        dependencies.append(item)
    return folder, maps, dependencies, {
        'source': source['source'], 'source_url': source['source_url'],
        'license': source['license'], 'license_url': source['license_url'],
        'source_manifest': str(source_path), 'source_files_modified': False,
        'texture_scale_m': .60, 'grain_axis': 'Block local X on long faces',
    }


def _plain_material(stage, path, color, roughness=.65):
    material = UsdShade.Material.Define(stage, path)
    surface = UsdShade.Shader.Define(stage, path + '/Surface')
    surface.CreateIdAttr('UsdPreviewSurface')
    surface.CreateInput('diffuseColor', Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    surface.CreateInput('roughness', Sdf.ValueTypeNames.Float).Set(roughness)
    surface.CreateInput('metallic', Sdf.ValueTypeNames.Float).Set(0.)
    material.CreateSurfaceOutput().ConnectToSource(surface.ConnectableAPI(), 'surface')
    return material


def _materials(stage, root, folder, maps):
    looks = root + '/Looks'
    UsdGeom.Scope.Define(stage, looks)
    materials = {}
    for index, bias in enumerate(((.15, .12, .075), (.18, .145, .095), (.13, .10, .060))):
        name = 'Wood_' + str(index)
        material = _plain_material(stage, looks + '/' + name, (.58, .40, .20))
        surface = UsdShade.Shader(stage.GetPrimAtPath(looks + '/' + name + '/Surface'))
        st = UsdShade.Shader.Define(stage, looks + '/' + name + '/ST')
        st.CreateIdAttr('UsdPrimvarReader_float2')
        st.CreateInput('varname', Sdf.ValueTypeNames.Token).Set('st')
        for role, filename in maps.items():
            tex = UsdShade.Shader.Define(stage, looks + '/' + name + '/' + role)
            tex.CreateIdAttr('UsdUVTexture')
            tex.CreateInput('file', Sdf.ValueTypeNames.Asset).Set(str(folder / filename))
            tex.CreateInput('sourceColorSpace', Sdf.ValueTypeNames.Token).Set('sRGB' if role == 'diffuseColor' else 'raw')
            tex.CreateInput('st', Sdf.ValueTypeNames.Float2).ConnectToSource(st.ConnectableAPI(), 'result')
            for axis in ('S', 'T'):
                tex.CreateInput('wrap' + axis, Sdf.ValueTypeNames.Token).Set('repeat')
            if role == 'normal':
                tex.CreateInput('scale', Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(.45, .45, 2., 1.))
                tex.CreateInput('bias', Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(-.225, -.225, -1., 0.))
            elif role == 'diffuseColor':
                tex.CreateInput('scale', Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(1.05, .98, .86, 1.))
                tex.CreateInput('bias', Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(*bias, 0.))
            else:
                tex.CreateInput('scale', Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(.45, .45, .45, 1.))
                tex.CreateInput('bias', Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(.33, .33, .33, 0.))
            value_type = (Sdf.ValueTypeNames.Normal3f if role == 'normal' else
                          Sdf.ValueTypeNames.Color3f if role == 'diffuseColor' else Sdf.ValueTypeNames.Float)
            surface.CreateInput(role, value_type).ConnectToSource(tex.ConnectableAPI(), 'r' if role == 'roughness' else 'rgb')
        materials[name] = material
    for name, color in (('Ink', (.11, .065, .030)), ('Target', (.24, .32, .24)),
                        ('Supply', (.43, .39, .31)), ('Front', (.16, .33, .48)),
                        ('Rear', (.52, .25, .16))):
        materials[name] = _plain_material(stage, looks + '/' + name, color)
    physics = UsdShade.Material.Define(stage, looks + '/WoodPhysics')
    properties = UsdPhysics.MaterialAPI.Apply(physics.GetPrim())
    properties.CreateStaticFrictionAttr(PHYSICS_PRIOR['static_friction'])
    properties.CreateDynamicFrictionAttr(PHYSICS_PRIOR['dynamic_friction'])
    properties.CreateRestitutionAttr(PHYSICS_PRIOR['restitution'])
    materials['Physics'] = physics
    return materials


def _mesh(stage, path, vertices, faces, material, *, uv=None, normals=None):
    vertices = np.asarray(vertices, dtype=np.float32)
    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr(vertices.tolist())
    mesh.CreateFaceVertexCountsAttr([len(face) for face in faces])
    mesh.CreateFaceVertexIndicesAttr([i for face in faces for i in face])
    mesh.CreateSubdivisionSchemeAttr('none')
    mesh.CreateDoubleSidedAttr(True)
    mesh.CreateExtentAttr([Gf.Vec3f(*map(float, vertices.min(axis=0))),
                          Gf.Vec3f(*map(float, vertices.max(axis=0)))])
    if uv is not None:
        UsdGeom.PrimvarsAPI(mesh).CreatePrimvar('st', Sdf.ValueTypeNames.TexCoord2fArray,
                                              'vertex').Set(uv)
    if normals is not None:
        mesh.CreateNormalsAttr(normals)
        mesh.SetNormalsInterpolation('vertex')
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)
    return mesh


def _block_visual(stage, path, size, material, index):
    # Rounded display is contained by the full nominal box. Broad support faces
    # stay at exactly +/- half-size; the collider does not shrink support height.
    half, radius = np.asarray(size) / 2, PHYSICS_PRIOR['visual_edge_radius_m']
    vertices, faces, normals, uv = [], [], [], []
    offset = ((index * .173) % .75, (index * .317) % .65)
    for axis in range(3):
        u, v = (axis + 1) % 3, (axis + 2) % 3
        coordinates = lambda a: [-half[a], -half[a] + radius * .293, -half[a] + radius,
                                  half[a] - radius, half[a] - radius * .293, half[a]]
        cu, cv = coordinates(u), coordinates(v)
        for sign in (-1, 1):
            start = len(vertices)
            for b in cv:
                for a in cu:
                    point = np.zeros(3)
                    point[axis], point[u], point[v] = sign * half[axis], a, b
                    inside = np.clip(point, -half + radius, half - radius)
                    normal = point - inside
                    normal /= np.linalg.norm(normal)
                    point = inside + radius * normal
                    vertices.append(point.tolist())
                    normals.append(normal.tolist())
                    # Source grain runs along texture V, hence along block X.
                    uv.append([float(point[1 if axis == 2 else 2] / .60 + offset[0]),
                               float(point[0 if axis != 0 else 1] / .60 + offset[1])])
            for j in range(len(cv) - 1):
                for i in range(len(cu) - 1):
                    a = start + j * len(cu) + i
                    face = [a, a + 1, a + len(cu) + 1, a + len(cu)]
                    faces.append(face if sign > 0 else face[::-1])
    _mesh(stage, path, vertices, faces, material, uv=uv, normals=normals)


_DIGITS = ('abcedf', 'bc', 'abged', 'abgcd', 'fgbc',
           'afgcd', 'afgecd', 'abc', 'abcdefg', 'abfgcd')


def _number_mark(stage, path, value, size, material):
    """Small flat two-digit ink marks on the top and both short ends; no collider."""
    # Original seven-stroke vector glyphs avoid fonts, bitmap generation and
    # external font licensing. All marks travel with their physical block.
    segments = {'a': (.5, .94, .78, .12), 'g': (.5, .5, .78, .12),
                'd': (.5, .06, .78, .12), 'f': (.10, .73, .14, .40),
                'b': (.90, .73, .14, .40), 'e': (.10, .27, .14, .40),
                'c': (.90, .27, .14, .40)}
    vertices, faces = [], []
    for face in ('top', 'end_positive', 'end_negative'):
        height = .010 if face == 'top' else .012
        width = height * .56
        for digit_index, digit in enumerate(f'{value:02d}'):
            for segment in _DIGITS[int(digit)]:
                cx, cy, sx, sy = segments[segment]
                points = []
                for xx, yy in ((-.5, -.5), (.5, -.5), (.5, .5), (-.5, .5)):
                    a = ((cx + xx * sx) + digit_index * 1.25 - 1.125) * width
                    c = (cy + yy * sy - .5) * height
                    if face == 'top':
                        points.append((size[0] * .26 + a, c, size[2] / 2 + .00002))
                    else:
                        sign = 1 if face == 'end_positive' else -1
                        points.append((sign * (size[0] / 2 + .00002), sign * a, c))
                start = len(vertices)
                vertices.extend(points)
                faces.append(list(range(start, start + 4)))
    _mesh(stage, path, vertices, faces, material)


def _block(stage, root, name, pose, size, materials, index):
    path = root + '/Blocks/' + name
    body = UsdGeom.Xform.Define(stage, path)
    transform = UsdGeom.Xformable(body)
    transform.AddTranslateOp().Set(Gf.Vec3d(*pose['position_m']))
    q = pose['quaternion_wxyz']
    transform.AddOrientOp().Set(Gf.Quatf(q[0], Gf.Vec3f(*q[1:])))
    rigid = UsdPhysics.RigidBodyAPI.Apply(body.GetPrim())
    rigid.CreateRigidBodyEnabledAttr(True)
    rigid.CreateKinematicEnabledAttr(False)
    rigid.CreateVelocityAttr(Gf.Vec3f(0))
    rigid.CreateAngularVelocityAttr(Gf.Vec3f(0))
    mass = float(np.prod(size) * PHYSICS_PRIOR['density_kg_m3'])
    inertia = [mass * (size[(axis + 1) % 3] ** 2 + size[(axis + 2) % 3] ** 2) / 12
               for axis in range(3)]
    properties = UsdPhysics.MassAPI.Apply(body.GetPrim())
    properties.CreateMassAttr(mass)
    properties.CreateCenterOfMassAttr(Gf.Vec3f(0))
    properties.CreateDiagonalInertiaAttr(Gf.Vec3f(*inertia))
    properties.CreatePrincipalAxesAttr(Gf.Quatf(1))
    _block_visual(stage, path + '/Visual', size, materials['Wood_' + str(index % 3)], index)
    _number_mark(stage, path + '/Number', index + 1, size, materials['Ink'])
    collider = UsdGeom.Cube.Define(stage, path + '/Collider')
    collider.CreateSizeAttr(1.)
    UsdGeom.Xformable(collider).AddScaleOp().Set(Gf.Vec3f(*size))
    collider.CreateVisibilityAttr('invisible')
    UsdPhysics.CollisionAPI.Apply(collider.GetPrim()).CreateCollisionEnabledAttr(True)
    collider.GetPrim().AddAppliedSchema('PhysxCollisionAPI')
    for field, value in (('contactOffset', PHYSICS_PRIOR['contact_offset_m']), ('restOffset', 0.)):
        collider.GetPrim().CreateAttribute('physxCollision:' + field, Sdf.ValueTypeNames.Float).Set(value)
    UsdShade.MaterialBindingAPI.Apply(collider.GetPrim()).Bind(materials['Physics'], materialPurpose='physics')
    body.GetPrim().SetCustomDataByKey('living_task_block_id', name)
    body.GetPrim().SetCustomDataByKey('scene_candidate', True)
    yaw = 2 * math.atan2(q[3], q[0])
    extents = [abs(math.cos(yaw)) * size[0] / 2 + abs(math.sin(yaw)) * size[1] / 2,
               abs(math.sin(yaw)) * size[0] / 2 + abs(math.cos(yaw)) * size[1] / 2, size[2] / 2]
    bounds = [[pose['position_m'][i] + sign * extents[i] for i in range(3)] for sign in (-1, 1)]
    return {'block_id': name, 'number': index + 1, 'prim_path': path,
            'collider_paths': [path + '/Collider'], **deepcopy(pose),
            'size_m': list(size), 'initial_bounds_m': bounds,
            'mass_kg': mass, 'inertia_diagonal_kg_m2': inertia,
            'dynamic': True, 'kinematic': False}


def _marker(stage, root, name, bounds, material, *, width=.0012, dash=.020, z_m=.78006):
    # A zero-thickness decal mesh 60 microns above the tabletop. It neither
    # raises the support plane nor supplies a collider/hidden placement fixture.
    xmin, ymin = bounds[0]
    xmax, ymax = bounds[1]
    vertices, faces = [], []
    for horizontal, length, a0, fixed in ((True, xmax - xmin, xmin, ymin),
                                          (True, xmax - xmin, xmin, ymax),
                                          (False, ymax - ymin, ymin, xmin),
                                          (False, ymax - ymin, ymin, xmax)):
        count = max(1, int(math.ceil(length / (dash * 1.5))))
        step = length / count
        for index in range(count):
            start, end = a0 + index * step, min(a0 + index * step + dash, a0 + length)
            points = [(start, fixed - width / 2), (end, fixed - width / 2),
                      (end, fixed + width / 2), (start, fixed + width / 2)]
            if not horizontal:
                points = [(y, x) for x, y in points]
            begin = len(vertices)
            vertices.extend((x, y, z_m) for x, y in points)
            faces.append(list(range(begin, begin + 4)))
    path = root + '/Markers/' + name
    _mesh(stage, path, vertices, faces, material)
    return {'prim_path': path, 'bounds_xy_m': bounds, 'z_m': z_m,
            'collision': False, 'purpose': 'Visual table marking only'}


def _supply_layout(geometry):
    # Each side has 3 x 3 parallel blocks. The long axis runs along Y; 35 mm
    # lateral and 50 mm end clearances leave practical initial finger access.
    ids, layout = block_ids(geometry), {}
    for bank in range(2):
        for row in range(3):
            for column in range(3):
                index = bank * 9 + row * 3 + column
                x = (.285 if bank == 0 else .855) + column * .080
                y = -.280 + row * .200
                layout[ids[index]] = {'position_m': [x, y, .78 + geometry['block_size_m'][2] / 2],
                                      'quaternion_wxyz': [math.sqrt(.5), 0., 0., math.sqrt(.5)]}
    return layout


def author_tower_task(stage, task_id, root_path='/World/LivingTask', *, presentation=False, profile=None):
    """Compose one task under a fresh root and return JSON-safe scene metadata."""
    if task_id not in ('tower_building', 'tower_extraction'):
        raise ValueError('Unknown living-room tower task')
    if type(presentation) is not bool:
        raise ValueError('presentation must be boolean')
    if profile not in (None, 'trays_v1'):
        raise ValueError('Unknown tower asset profile')
    root = Sdf.Path(root_path)
    if not root.IsAbsolutePath() or not root.IsPrimPath() or root == Sdf.Path.absoluteRootPath:
        raise ValueError('root_path must be an absolute USD prim path')
    if stage.GetPrimAtPath(root):
        raise ValueError('Task root already exists; author into a new stage or root')
    geometry = validate_geometry(deepcopy(CANDIDATE_GEOMETRY))
    if profile == 'trays_v1':
        from . import tower_trays
        geometry = validate_geometry(deepcopy(tower_trays.BUILDING_GEOMETRY if task_id == 'tower_building'
                                              else tower_trays.EXTRACTION_GEOMETRY))
    if task_id == 'tower_building':
        from .tasks.tower_building.task import task_configuration
        if profile == 'trays_v1':
            configuration = task_configuration(geometry, inventory_count=36,
                supply_trays=tower_trays.building_supply_trays(),
                supply_layout=tower_trays.building_supply_layout(), scoring_mode='freeform_height')
        else:
            configuration = task_configuration(geometry, supply_layout=_supply_layout(geometry))
        poses = deepcopy(configuration['initial_objects'])
        placed_ids = []
        if presentation:
            # Three levels describe the action in a still, without fabricating a
            # reset or success. Nine remaining blocks stay in their supply slots.
            placed_ids = ([f'block_{index:02d}' for index in range(9, 18)] if profile else
                          ['block_00', 'block_01', 'block_02', 'block_09', 'block_10',
                           'block_11', 'block_03', 'block_04', 'block_05'])
            from .tower_geometry import tower_slots
            display_slots = tower_slots(geometry) if profile else configuration['target_slots']
            for name, slot in zip(placed_ids, display_slots[:9], strict=True):
                yaw = slot['yaw_rad']
                poses[name] = {'position_m': list(slot['position_m']),
                               'quaternion_wxyz': [math.cos(yaw / 2), 0., 0., math.sin(yaw / 2)]}
    else:
        from .tasks.tower_extraction.task import task_configuration
        configuration = task_configuration(geometry, **({'collection_trays': tower_trays.extraction_collection_trays()}
                                                         if profile else {}))
        poses = {slot['block_id']: {'position_m': slot['position_m'],
                  'quaternion_wxyz': [math.cos(slot['yaw_rad'] / 2), 0., 0., math.sin(slot['yaw_rad'] / 2)]}
                 for slot in initial_tower(geometry)}
        placed_ids = list(poses)
    folder, maps, dependencies, material_source = _material_source()
    UsdGeom.Xform.Define(stage, root)
    UsdGeom.Scope.Define(stage, root_path + '/Blocks')
    UsdGeom.Scope.Define(stage, root_path + '/Markers')
    materials = _materials(stage, root_path, folder, maps)
    if profile:
        materials['TrayFloor'] = _plain_material(stage, root_path + '/Looks/TrayFloor', (.57, .43, .27), .72)
    objects = [_block(stage, root_path, name, poses[name], geometry['block_size_m'], materials, index)
               for index, name in enumerate(poses)]
    fixtures = tower_trays.author_trays(stage, root_path, task_id, materials) if profile else {}
    markers = []
    if task_id == 'tower_building' and profile is None:
        markers.append(_marker(stage, root_path, 'TowerFootprint', [[.562, -.088], [.738, .088]], materials['Target']))
        for name, x in (('SupplyLeft', .245), ('SupplyRight', .815)):
            markers.append(_marker(stage, root_path, name, [[x, -.377], [x + .24, .217]], materials['Supply']))
    elif task_id == 'tower_extraction':
        for side, slots in configuration['collection_slots'].items():
            for slot in slots:
                center, size = slot['position_m'], geometry['block_size_m']
                bounds = [[center[0] - size[0] / 2, center[1] - size[1] / 2],
                          [center[0] + size[0] / 2, center[1] + size[1] / 2]]
                z = configuration['collection_trays'][side]['floor_z_m'] + .00006 if profile else .78006
                markers.append(_marker(stage, root_path, slot['slot_id'], bounds, materials[side.title()],
                                       width=.0010, dash=.012, z_m=z))
    metadata = {
        'schema': 'living_room_tower_scene_candidate.v1', 'task_id': task_id,
        'root_prim_path': root_path, 'geometry': geometry, 'configuration': configuration,
        'configuration_sha256': configuration['configuration_sha256'],
        'dimension_status': '150 x 45 x 25 mm visual/physics scene candidate; user dimensions remain unfrozen.',
        'objects': objects, 'dynamic_prim_paths': [item['prim_path'] for item in objects],
        'layout': {'table_top_z_m': .78, 'table_bounds_xy_m': [[.15, -.5], [1.15, .5]],
                   'tower_center_xy_m': [.65, 0.], 'authored_tower_block_ids': placed_ids,
                   'authored_supply_block_ids': [name for name in poses if name not in placed_ids],
                   'collection_slots': configuration.get('collection_slots', {}),
                   'source': 'Existing task configuration factories with explicit candidate dimensions'},
        'markers': markers, 'physics_prior': deepcopy(PHYSICS_PRIOR),
        'material_source': material_source, 'dependencies': dependencies,
        'procedural_geometry': {'generator': __name__ + '.author_tower_task',
                                'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                                'license': 'CC0-1.0', 'external_geometry_used': False},
        'presentation_only': bool(presentation and task_id == 'tower_building'),
        'is_task_reset': not (presentation and task_id == 'tower_building'),
        'presentation_note': ('Three assembled layers plus nine supply blocks; authored illustration, not task reset, robot action or success.'
                              if presentation and task_id == 'tower_building' else 'Authored candidate reset; settling and manipulation remain unqualified.'),
        'success_claimed': False, 'native_qualified': False,
    }
    if profile:
        metadata.update(profile=profile, **fixtures)
        length, width, height = geometry['block_size_m']
        metadata['dimension_status'] = f'{length * 1000:g} x {width * 1000:g} x {height * 1000:g} mm task-owned scene candidate.'
        metadata['layout']['source'] = 'Versioned trays_v1 configuration; original table and center preserved'
        metadata['layout']['build_area_has_fixed_target_markers'] = False
        metadata['procedural_geometry']['layout_source_sha256'] = hashlib.sha256(Path(tower_trays.__file__).read_bytes()).hexdigest()
        source_files = [Path(__file__), Path(tower_trays.__file__),
                        Path(__file__).parent / 'tasks' / task_id / 'assets.py']
        metadata['procedural_geometry']['source_files_sha256'] = {
            'scenes/workrooms/living_room/' + str(path.relative_to(Path(__file__).parent)):
                hashlib.sha256(path.read_bytes()).hexdigest() for path in source_files}
        if task_id == 'tower_extraction':
            crossing_width = 3 * width + 2 * geometry['in_layer_gap_m']
            metadata['geometric_clearance_audit'] = {
                'parallel_block_gap_m': geometry['in_layer_gap_m'],
                'block_end_overhang_beyond_crossing_layer_m': (length - crossing_width) / 2,
                'legacy_parallel_block_gap_m': CANDIDATE_GEOMETRY['in_layer_gap_m'],
                'legacy_block_end_overhang_beyond_crossing_layer_m':
                    (CANDIDATE_GEOMETRY['block_size_m'][0] - 3 * CANDIDATE_GEOMETRY['block_size_m'][1]
                     - 2 * CANDIDATE_GEOMETRY['in_layer_gap_m']) / 2,
                'tower_to_nearest_tray_outer_edge_m': .17 - tower_trays.WALL_THICKNESS_M - length / 2,
                'lowest_extractable_block_bottom_above_tray_rim_m': .78 + height - .798,
                'collection_slot_gaps_xy_m': [.24 - length, .06 - width],
                'scope': 'Authored geometric clearances only; no finger access, trajectory or extraction success qualification.'}
        else:
            metadata['geometric_clearance_audit'] = {
                'supply_columns_lateral_gap_m': .080 - width,
                'supply_columns_end_gap_m': .200 - length,
                'supply_stack_layers': 2, 'inventory_count': len(objects),
                'empty_center_x_interval_m': [.495 + tower_trays.WALL_THICKNESS_M,
                                              .805 - tower_trays.WALL_THICKNESS_M],
                'scope': 'Authored tray packing and empty center only; no required build footprint or robot access qualification.'}
        if presentation and task_id == 'tower_building':
            metadata['presentation_note'] = 'Illustrative stack plus 27 supply blocks; not a required tower form, task reset, robot action or score.'
    # No callers should need pxr/numpy encoders to freeze the authoring receipt.
    json.dumps(metadata, allow_nan=False)
    return metadata
