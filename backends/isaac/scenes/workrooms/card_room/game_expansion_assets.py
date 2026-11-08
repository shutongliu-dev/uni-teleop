"""Self-contained USD assets for four card-room game scene candidates.

Only /World/CardGame is authored. The immutable heritage room, robot, controller
and PhysicsScene are owned by their existing layers. Each body has explicit
mass, COM and inertia from its actual collision solids, with documented uniform
density priors. Geometry and material checks are not native physics evidence.
"""
from __future__ import annotations

import math

import numpy as np

from .game_designs import (BOARD_TOP_Z_M, CONVEX_PRISM_SEGMENTS, MEMORY_RESET_SEED, TASK_IDS,
                           WOOD_DENSITY_KG_M3, design_for, memory_pairs_for_seed)
from .scene import environment_reference

ROOT = '/World/CardGame'
COLORS = {
    'walnut': [.23, .115, .052], 'edge': [.105, .047, .022],
    'maple': [.68, .48, .255], 'black': [.035, .042, .038],
    'white': [.86, .81, .68], 'ink': [.105, .062, .03],
    'green': [.15, .25, .18], 'gold': [.64, .42, .13],
    'red': [.51, .12, .09], 'blue': [.09, .25, .43],
    'teal': [.06, .34, .29], 'purple': [.34, .16, .37],
}


def _box_mass(size, position, density=WOOD_DENSITY_KG_M3):
    size, center = np.asarray(size, float), np.asarray(position, float)
    mass = float(np.prod(size)*density)
    tensor = mass/12*np.diag([size[1]**2+size[2]**2,
                              size[0]**2+size[2]**2, size[0]**2+size[1]**2])
    return mass, center, tensor


def _prism_mass(radius, height, position, segments=CONVEX_PRISM_SEGMENTS, density=WOOD_DENSITY_KG_M3):
    """Exact mass properties of the authored prism; native cooking is separate."""
    angle = 2*math.pi/segments
    mass = .5*segments*radius**2*math.sin(angle)*height*density
    iz = mass*radius**2*(2+math.cos(angle))/6
    ix = iz/2+mass*height**2/12
    return mass, np.asarray(position, float), np.diag([ix, ix, iz])


def _mass_properties(parts):
    from scipy.spatial.transform import Rotation
    mass = sum(part[0] for part in parts)
    center = sum(m*c for m, c, _ in parts)/mass
    tensor = np.zeros((3, 3))
    for m, c, inertia in parts:
        d = c-center
        tensor += inertia+m*(np.dot(d, d)*np.eye(3)-np.outer(d, d))
    diagonal, basis = np.linalg.eigh(tensor)
    if np.linalg.det(basis) < 0:
        basis[:, 0] *= -1
    xyzw = Rotation.from_matrix(basis).as_quat()
    return {'mass_kg': float(mass), 'center_of_mass_m': center.tolist(),
            'inertia_tensor_kg_m2': tensor.tolist(),
            'diagonal_inertia_kg_m2': diagonal.tolist(),
            'principal_axes_wxyz': [float(xyzw[3]), *map(float, xyzw[:3])],
            'source': 'Analytic non-overlapping collision solids, uniform 650 kg/m3 design prior.'}


def _material(stage, name, color, friction, *, roughness=.55):
    from pxr import Gf, Sdf, UsdPhysics, UsdShade
    material = UsdShade.Material.Define(stage, ROOT+'/Materials/'+name)
    shader = UsdShade.Shader.Define(stage, material.GetPath().AppendChild('Surface'))
    shader.CreateIdAttr('UsdPreviewSurface')
    shader.CreateInput('diffuseColor', Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    shader.CreateInput('roughness', Sdf.ValueTypeNames.Float).Set(roughness)
    shader.CreateInput('metallic', Sdf.ValueTypeNames.Float).Set(0.)
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), 'surface')
    physical = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
    physical.CreateStaticFrictionAttr(float(friction[0]))
    physical.CreateDynamicFrictionAttr(float(friction[1]))
    physical.CreateRestitutionAttr(0.)
    # These native schema opinions are kept explicit even in CPU usd-core,
    # which does not ship the optional PhysxSchema Python module.
    material.GetPrim().AddAppliedSchema('PhysxMaterialAPI')
    material.GetPrim().CreateAttribute('physxMaterial:frictionCombineMode', Sdf.ValueTypeNames.Token).Set('average')
    material.GetPrim().CreateAttribute('physxMaterial:restitutionCombineMode', Sdf.ValueTypeNames.Token).Set('average')
    return material


def _bind(prim, material, *, collision=False):
    from pxr import UsdPhysics, UsdShade
    binding = UsdShade.MaterialBindingAPI.Apply(prim)
    binding.Bind(material)
    if collision:
        binding.Bind(material, materialPurpose='physics')
        UsdPhysics.CollisionAPI.Apply(prim).CreateCollisionEnabledAttr(True)


def _box(stage, path, size, position, material, *, collision=False):
    from pxr import Gf, UsdGeom
    shape = UsdGeom.Cube.Define(stage, path)
    shape.CreateSizeAttr(1.)
    shape.AddTranslateOp().Set(Gf.Vec3d(*position))
    shape.AddScaleOp().Set(Gf.Vec3f(*size))
    _bind(shape.GetPrim(), material, collision=collision)
    return str(shape.GetPath())


def _mesh(stage, path, points, counts, indices, material, *, collision=False):
    from pxr import Gf, UsdGeom, UsdPhysics
    mesh = UsdGeom.Mesh.Define(stage, path)
    values = np.asarray(points, float)
    mesh.CreatePointsAttr([Gf.Vec3f(*p) for p in values])
    mesh.CreateFaceVertexCountsAttr(counts)
    mesh.CreateFaceVertexIndicesAttr(indices)
    mesh.CreateSubdivisionSchemeAttr('none')
    mesh.CreateDoubleSidedAttr(True)
    mesh.CreateExtentAttr([Gf.Vec3f(*values.min(axis=0)), Gf.Vec3f(*values.max(axis=0))])
    _bind(mesh.GetPrim(), material, collision=collision)
    if collision:
        UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr('convexHull')
    return str(mesh.GetPath())


def _prism(stage, path, radius, height, position, material, *, segments=CONVEX_PRISM_SEGMENTS, collision=True):
    points = []
    for z in (-height/2, height/2):
        points.extend([[radius*math.cos(2*math.pi*i/segments)+position[0],
                        radius*math.sin(2*math.pi*i/segments)+position[1], z+position[2]]
                       for i in range(segments)])
    counts = [segments, segments]+[4]*segments
    indices = list(reversed(range(segments)))+list(range(segments, 2*segments))
    for i in range(segments):
        j = (i+1)%segments
        indices.extend([i, j, j+segments, i+segments])
    return _mesh(stage, path, points, counts, indices, material, collision=collision)


def _flat_polygon(stage, path, xy, z, material):
    points = [[x, y, z] for x, y in xy]
    return _mesh(stage, path, points, [len(points)], list(range(len(points))), material)


def _ring(stage, path, center, inner, outer, z, material, segments=96):
    points = [[center[0]+r*math.cos(2*math.pi*i/segments),
               center[1]+r*math.sin(2*math.pi*i/segments), z]
              for r in (inner, outer) for i in range(segments)]
    indices = []
    for i in range(segments):
        j = (i+1)%segments
        indices.extend([i, i+segments, j+segments, j])
    return _mesh(stage, path, points, [4]*segments, indices, material)


def _markings(stage, design, materials):
    from pxr import UsdGeom
    root = ROOT+'/Fixtures/Markings'
    UsdGeom.Xform.Define(stage, root)
    b = design['board']; z = b['top_z_m']+.00004
    size = b['size_xy_m']
    court = materials['green'] if design['task_id'] == 'disk_curling' else materials['maple']
    _box(stage, root+'/PlayingSurface', [*size, .00004], [.65, 0., z], court)
    # A simple dark rim and warm veneer match the fixed room without labels.
    for axis in (0, 1):
        for sign in (-1, 1):
            dims = [size[0]+.012, .006, .00004] if axis == 1 else [.006, size[1], .00004]
            p = [.65, 0., z+.00004]; p[axis] += sign*(size[axis]/2+.003)
            _box(stage, root+f'/Frame_{axis}_{"p" if sign > 0 else "n"}', dims, p, materials['edge'])
    if design['task_id'] in ('polyomino_territory', 'barrier_race'):
        pitch = b['pitch_m']; nx, ny = b['columns'], b['rows']
        for i in range(nx+1):
            _box(stage, root+f'/GridX_{i}', [.0018, ny*pitch, .00004],
                 [.65+(i-nx/2)*pitch, 0., z+.00005], materials['ink'])
        for i in range(ny+1):
            _box(stage, root+f'/GridY_{i}', [nx*pitch, .0018, .00004],
                 [.65, (i-ny/2)*pitch, z+.00005], materials['ink'])
        for owner, sign in (('black', -1), ('white', 1)):
            _box(stage, root+'/'+owner+'_Edge', [nx*pitch, .006, .00004],
                 [.65, sign*(ny*pitch/2+.006), z+.00007], materials[owner])
    elif design['task_id'] == 'memory_pairs':
        for owner, zone in design['collection_zones'].items():
            _box(stage, root+'/'+owner+'_Collection', [*zone['size_xy_m'], .00004],
                 [*zone['center_xy_m'], z], materials[owner])
    else:
        center = b['target_center_xy_m']
        for i, radius in enumerate(b['scoring_radii_m']):
            _ring(stage, root+f'/ScoringRing_{i}', center, radius-.0015, radius+.0015,
                  z+.00006, materials['gold'] if i == 0 else materials['white'])
        _ring(stage, root+'/TargetCenter', center, .0, .004, z+.00006, materials['gold'])
        for owner, y in b['launch_lines_y_m'].items():
            _box(stage, root+'/'+owner+'_LaunchLine', [.44, .003, .00004],
                 [.65, y, z+.00006], materials[owner])


def _memory_pattern(stage, actor_path, pair_id, materials):
    """Printed geometry on the underside; identical back and opaque edge stock."""
    index = int(pair_id.rsplit('_', 1)[-1]); z = -.01004
    colors = ('red', 'blue', 'teal', 'purple', 'gold', 'black')
    material = materials[colors[index]]
    if index == 0:
        xy = [[.020*math.cos(i*2*math.pi/48), .020*math.sin(i*2*math.pi/48)] for i in range(48)]
    elif index == 1:
        xy = [[-.018, -.018], [.018, -.018], [.018, .018], [-.018, .018]]
    elif index == 2:
        xy = [[0., .024], [-.022, -.016], [.022, -.016]]
    elif index == 3:
        xy = [[0., .027], [-.020, 0.], [0., -.027], [.020, 0.]]
    elif index == 4:
        xy = [[(.024 if i%2 == 0 else .011)*math.cos(math.pi/2+i*math.pi/5),
               (.024 if i%2 == 0 else .011)*math.sin(math.pi/2+i*math.pi/5)] for i in range(10)]
    else:
        xy = [[-.007, -.025], [.007, -.025], [.007, -.007], [.022, -.007],
              [.022, .007], [.007, .007], [.007, .025], [-.007, .025],
              [-.007, .007], [-.022, .007], [-.022, -.007], [-.007, -.007]]
    # Render concave emblems as a triangle fan around the interior origin.
    points = [[0., 0., z]]+[[x, y, z] for x, y in xy]
    faces = [v for i in range(len(xy)) for v in (0, 1+(i+1)%len(xy), i+1)]
    _mesh(stage, actor_path+'/FrontEmblem', points, [3]*len(xy), faces, material)
    _ring(stage, actor_path+'/BackMedallion', [0., 0.], .015, .017, .01004, materials['edge'])
    _ring(stage, actor_path+'/BackInset', [0., 0.], .004, .0055, .01004, materials['edge'])


def _author_body(stage, piece, materials, private_pairs):
    from pxr import Gf, UsdGeom, UsdPhysics
    path = ROOT+'/Actors/'+piece['id']
    body = UsdGeom.Xform.Define(stage, path)
    body.AddTranslateOp().Set(Gf.Vec3d(*piece['initial_position_m']))
    body.AddOrientOp().Set(Gf.Quatf(1., Gf.Vec3f(0., 0., 0.)))
    physical = UsdPhysics.RigidBodyAPI.Apply(body.GetPrim())
    physical.CreateRigidBodyEnabledAttr(True)
    physical.CreateKinematicEnabledAttr(False)
    material = materials[piece.get('owner') or 'maple']
    collisions, solids = [], []
    if piece['kind'] in ('polyomino', 'barrier'):
        for i, box in enumerate(piece['collision_boxes']):
            collisions.append(_box(stage, path+f'/Solid_{i}', box['dimensions_m'], box['position_m'], material, collision=True))
            solids.append(_box_mass(box['dimensions_m'], box['position_m']))
    elif piece['kind'] in ('pawn', 'disk'):
        cylinders = piece.get('cylinders', [{'radius_m': piece.get('radius_m'),
                                           'height_m': piece.get('height_m'), 'position_m': [0., 0., 0.]}])
        for i, cylinder in enumerate(cylinders):
            radius, height, position = cylinder['radius_m'], cylinder['height_m'], cylinder['position_m']
            collisions.append(_prism(stage, path+f'/Solid_{i}', radius, height, position, material))
            solids.append(_prism_mass(radius, height, position))
        if piece['kind'] == 'disk':
            _ring(stage, path+'/WoodInlay', [0., 0.], .023, .0245, piece['height_m']/2+.00004,
                  materials['gold'])
    else:
        collisions.append(_box(stage, path+'/Solid_0', piece['dimensions_m'], [0., 0., 0.], material, collision=True))
        solids.append(_box_mass(piece['dimensions_m'], [0., 0., 0.]))
        _memory_pattern(stage, path, private_pairs[piece['id']], materials)
    props = _mass_properties(solids)
    mass_api = UsdPhysics.MassAPI.Apply(body.GetPrim())
    mass_api.CreateMassAttr(props['mass_kg'])
    mass_api.CreateCenterOfMassAttr(Gf.Vec3f(*props['center_of_mass_m']))
    mass_api.CreateDiagonalInertiaAttr(Gf.Vec3f(*props['diagonal_inertia_kg_m2']))
    q = props['principal_axes_wxyz']
    mass_api.CreatePrincipalAxesAttr(Gf.Quatf(q[0], Gf.Vec3f(*q[1:])))
    return path, collisions, props


def _probes(task_id):
    def place(actor, target, axis='z', angle=0.):
        return {'actor': actor, 'target_position_m': target, 'relative_rotation_axis': axis,
                'relative_rotation_rad': angle, 'clearance_z_m': 1.04}
    common = {'scope': 'Private engineering object-load recipe; never Agent observations, task reset or operation credit.'}
    if task_id == 'polyomino_territory':
        common.update(primary_actor='black_tet_L', secondary_actor='black_mono', placements=[
            place('black_tet_L', [.676, 0., .8176], angle=math.pi/2),
            place('black_mono', [.676, .026, .8176])],
            nominal_neighbor_clearance_m=.002)
    elif task_id == 'barrier_race':
        common.update(primary_actor='black_barrier_0', secondary_actor='black_pawn', placements=[
            place('black_pawn', [.65, -.075, .8491]),
            place('black_barrier_0', [.6125, -.0375, .8341], angle=math.pi/2)])
    elif task_id == 'memory_pairs':
        common.update(primary_actor='card_00', placements=[place('card_00', [.65, -.355, .8166], 'x', math.pi)],
                      return_to_initial_after_placements=True)
    else:
        common.update(primary_actor='black_disk_0', secondary_actor='white_disk_0',
            diagnostic_reset_positions_m={'black_disk_0': [.65, -.20, .8156], 'white_disk_0': [.65, .04, .8156]},
            pusher_start_position_m=[.65, -.28, .826], push_direction_world=[0., 1., 0.],
            pusher_dimensions_m=[.045, .025, .04], target_position_m=[.65, .10, .8156])
    return common


def author_task(stage, task_id):
    """Author one candidate and return its private, JSON-compatible metadata.

    No PhysicsScene, room reference, robot, lights, native runtime or outcome
    gate is added. A fresh root is required; existing assets are never replaced.
    """
    from pxr import UsdGeom
    design = design_for(task_id)
    if stage.GetPrimAtPath(ROOT):
        raise ValueError('CardGame root already exists; use a new stage')
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.)
    if not stage.GetPrimAtPath('/World'):
        world = UsdGeom.Xform.Define(stage, '/World')
        stage.SetDefaultPrim(world.GetPrim())
    UsdGeom.Xform.Define(stage, ROOT)
    UsdGeom.Xform.Define(stage, ROOT+'/Fixtures')
    UsdGeom.Xform.Define(stage, ROOT+'/Actors')
    friction = (.24, .18) if task_id == 'disk_curling' else (.55, .42)
    materials = {name: _material(stage, name, color, friction, roughness=.42 if task_id == 'disk_curling' else .6)
                 for name, color in COLORS.items()}
    board = design['board']
    fixture = _box(stage, ROOT+'/Fixtures/Deck', [*board['deck_size_xy_m'], board['deck_thickness_m']],
                   [.65, 0., BOARD_TOP_Z_M-board['deck_thickness_m']/2], materials['walnut'], collision=True)
    _markings(stage, design, materials)
    private_pairs = memory_pairs_for_seed(MEMORY_RESET_SEED) if task_id == 'memory_pairs' else {}
    actors, collisions, mass_properties = {}, {}, {}
    for piece in design['pieces']:
        actor, shapes, props = _author_body(stage, piece, materials, private_pairs)
        actors[piece['id']] = actor
        collisions[piece['id']] = shapes
        mass_properties[piece['id']] = props
    metadata = {
        'schema': 'card_game_candidate_assets.v1', 'task_id': task_id,
        'version': 'scene_v1', 'root_prim': ROOT, 'parent_environment': environment_reference(),
        'actors': actors, 'initial_positions_m': {p['id']: p['initial_position_m'] for p in design['pieces']},
        'initial_orientations_wxyz': {p['id']: p['initial_orientation_wxyz'] for p in design['pieces']},
        'masses_kg': {identity: prop['mass_kg'] for identity, prop in mass_properties.items()},
        'mass_properties': mass_properties, 'board_top_z_m': BOARD_TOP_Z_M,
        'fixture_collision_paths': [fixture], 'actor_collision_paths': collisions,
        'actor_geometry': {p['id']: {**p, 'local_bottom_z_m': -p['dimensions_m'][2]/2,
            'half_height_m': p['dimensions_m'][2]/2} for p in design['pieces']},
        'dimensions': design, 'engineering_probe': _probes(task_id),
        'cameras': {
            # Same interior view as the pinned room; y=-2.25 is inside its wall.
            'room': {'eye_m': [2.72, -2., 2.02], 'target_m': [.4, .56, 1.2],
                     'focal_length_mm': 22., 'resolution': [1600, 1200]},
            'tabletop': {'eye_m': [.65, -.001, 2.05], 'target_m': [.65, 0., .806],
                        'focal_length_mm': 30., 'resolution': [1600, 1200]},
            'oblique': {'eye_m': [1.8, -1.1, 1.65], 'target_m': [.65, 0., .84],
                       'focal_length_mm': 38., 'resolution': [1600, 1200]},
        },
        'physics_priors': {
            'density_kg_m3': WOOD_DENSITY_KG_M3, 'material_model': 'Uniform wood/composite design prior',
            'static_friction': friction[0], 'dynamic_friction': friction[1], 'restitution': 0.,
            'friction_combine_mode': 'average', 'restitution_combine_mode': 'average',
            'cylinder_collision': 'Authored 32-sided convex prism with 64 vertices; native cooked hull identity is unconfirmed',
            'cylinder_source_segments': CONVEX_PRISM_SEGMENTS,
            'cylinder_source_vertices': 2*CONVEX_PRISM_SEGMENTS,
            'native_cooked_hull_identity_confirmed': False,
            'polyomino_collision': 'Non-overlapping boxes with inset exterior faces, preserving open concavities',
            'contact_offset_policy': 'Native defaults; no task-specific contact/solver/drive override',
            'calibrated': False,
        },
        'provenance': {'geometry': 'Original procedural geometry authored for this project',
                       'external_textures': [], 'external_meshes': [],
                       'fixed_room_mutated': False, 'physical_input_hashes_required': True},
        'scope': 'Private candidate asset metadata. Native tests and reports are separate; no runtime admission or robot-operation claim.',
    }
    if private_pairs:
        metadata['private_reset_pairs'] = private_pairs
        metadata['private_reset_seed'] = MEMORY_RESET_SEED
        metadata['private_pair_emblems'] = {f'pair_{i}': name for i, name in enumerate(
            ('red_circle', 'blue_square', 'teal_triangle', 'purple_diamond', 'gold_star', 'black_cross'))}
    return metadata


def audit_stage(stage, metadata):
    """Lightweight USD checks; returned development findings are not task inputs."""
    from pxr import Usd, UsdGeom, UsdPhysics, UsdShade
    task_id = metadata['task_id']
    expected = {'polyomino_territory': 12, 'barrier_race': 10, 'memory_pairs': 12, 'disk_curling': 8}[task_id]
    actors = metadata['actors']
    if len(actors) != expected:
        raise ValueError('Unexpected dynamic actor inventory')
    physics_scenes = [str(p.GetPath()) for p in stage.Traverse() if p.IsA(UsdPhysics.Scene)]
    if physics_scenes:
        raise ValueError('Task assets must not own a PhysicsScene')
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ['default'])
    boxes = {}
    for identity, path in actors.items():
        prim = stage.GetPrimAtPath(path)
        if not prim or not prim.HasAPI(UsdPhysics.RigidBodyAPI):
            raise ValueError('Missing rigid body: '+identity)
        properties = metadata['mass_properties'][identity]
        mass = UsdPhysics.MassAPI(prim)
        np.testing.assert_allclose(mass.GetMassAttr().Get(), properties['mass_kg'], rtol=1.e-6)
        if min(mass.GetDiagonalInertiaAttr().Get()) <= 0:
            raise ValueError('Inertia must be positive')
        bounds = cache.ComputeWorldBound(prim).ComputeAlignedRange()
        low, high = np.array(bounds.GetMin()), np.array(bounds.GetMax())
        if not np.isfinite([low, high]).all():
            raise ValueError('Nonfinite body geometry')
        if np.any(low[:2] < [.17, -.48]) or np.any(high[:2] > [1.13, .48]):
            raise ValueError('Actor starts outside support deck: '+identity)
        if abs(low[2]-(BOARD_TOP_Z_M+.0006)) > .00015:
            raise ValueError('Actor reset does not rest just above deck: '+identity)
        boxes[identity] = (low, high)
        for path in metadata['actor_collision_paths'][identity]:
            collider = stage.GetPrimAtPath(path)
            if not collider.HasAPI(UsdPhysics.CollisionAPI):
                raise ValueError('Missing real collision geometry: '+path)
            material, _ = UsdShade.MaterialBindingAPI(collider).ComputeBoundMaterial(materialPurpose='physics')
            if not material or not material.GetPrim().HasAPI(UsdPhysics.MaterialAPI):
                raise ValueError('Collision material is not explicitly bound')
    identities = list(boxes)
    for i, a in enumerate(identities):
        for b in identities[i+1:]:
            low_a, high_a = boxes[a]; low_b, high_b = boxes[b]
            if np.all(np.minimum(high_a, high_b)-np.maximum(low_a, low_b) > 1.e-7):
                raise ValueError(f'Initial actor bounds overlap: {a}/{b}')
    probe_lattice_cells = {}
    if task_id == 'polyomino_territory':
        board = metadata['dimensions']['board']
        pitch = board['pitch_m']; occupied = set()
        for placement in metadata['engineering_probe']['placements']:
            identity = placement['actor']
            cells = np.asarray(metadata['actor_geometry'][identity]['cells'], float)
            nominal_local_xy = (cells-(cells.min(axis=0)+cells.max(axis=0))/2)*pitch
            angle = placement['relative_rotation_rad']
            if placement['relative_rotation_axis'] != 'z':
                raise ValueError('Polyomino lattice probe requires in-plane rotation')
            rotation = np.array([[math.cos(angle), -math.sin(angle)],
                                 [math.sin(angle), math.cos(angle)]])
            world_xy = nominal_local_xy@rotation.T+placement['target_position_m'][:2]
            indices = ((world_xy-np.asarray(board['center_xy_m']))/pitch
                       +(np.asarray([board['columns'], board['rows']])-1)/2)
            rounded = np.rint(indices).astype(int)
            if (not np.allclose(indices, rounded, rtol=0., atol=1.e-8)
                    or np.any(rounded < 0)
                    or np.any(rounded >= [board['columns'], board['rows']])):
                raise ValueError('Engineering placement is off the legal board lattice: '+identity)
            placement_cells = {tuple(map(int, cell)) for cell in rounded}
            if occupied & placement_cells:
                raise ValueError('Engineering polyomino placements overlap occupied cells')
            occupied.update(placement_cells)
            probe_lattice_cells[identity] = sorted(map(list, placement_cells))
    return {'schema': 'card_game_asset_structural_audit.v1', 'task_id': task_id,
            'actor_count': len(actors), 'collider_count': sum(map(len, metadata['actor_collision_paths'].values()))+1,
            'actor_bounds_overlap': False, 'explicit_mass_inertia_and_materials': True,
            'engineering_probe_lattice_cells': probe_lattice_cells,
            'scope': 'CPU USD geometry and parameters only; no native dynamics, contact or robot manipulation evidence.'}
