"""Finite position/finish-only derivatives of the three remaining study tasks.

Only the intended loose item moves. Original room, fixtures, articulations,
geometry, mass properties, colors, labels and scoring thresholds are preserved.
This author is separate from the historical size-changing study seed author.
"""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import shutil

import numpy as np

from . import rules
from .seed_variants import (ROOT, SCHEMA, ASSET_ROOT, file_hash, _checked, _write,
                            collision_bounds, _appearance)
from bench.paths import load_json, usd_value_text

TASKS = ('book_retrieval', 'drawer_retrieval', 'reading_stand')
GENERATOR_VERSION = 'study_position_texture_v1'
VARIANT_SCOPE = 'initial_xy_translation_and_surface_texture_only'
MOVABLE = {'book_retrieval': 'robotics', 'drawer_retrieval': 'notebook', 'reading_stand': 'book'}
OFFSETS_XY_M = ((0., 0.), (-.006, -.006), (-.006, 0.), (-.006, .006),
                (0., -.006), (0., .006), (.006, -.006), (.006, 0.), (.006, .006), (.003, .003))


def _seed(task_id, seed):
    if task_id not in TASKS or type(seed) is not int or seed not in range(10):
        raise ValueError('Study placement requires a supported task and integer seed 0 through 9')


def plan_case(task_id, seed, configuration):
    _seed(task_id, seed)
    rules.validate_configuration(configuration)
    if configuration['task_id'] != task_id or MOVABLE[task_id] not in configuration['objects']:
        raise ValueError('Placement parent task differs')
    plan = {'schema': 'study_placement_plan.v1', 'generator_version': GENERATOR_VERSION,
            'task_id': task_id, 'seed': seed, 'movable_object_id': MOVABLE[task_id],
            'translation_xy_m': list(OFFSETS_XY_M[seed]),
            'parent_configuration_sha256': configuration['configuration_sha256'],
            'scope': 'Loose item XY and visual roughness only; no yaw, Z, geometry, mass or mechanism change'}
    plan['plan_sha256'] = rules.canonical_hash(plan)
    return plan


def _configuration(parent, plan):
    value = deepcopy(parent)
    if plan['seed']:
        pos = value['objects'][plan['movable_object_id']]['initial_pose']['position_m']
        pos[:2] = [p+d for p, d in zip(pos[:2], plan['translation_xy_m'])]
        value['configuration_sha256'] = rules.canonical_hash({k: v for k, v in value.items() if k != 'configuration_sha256'})
    return value


def _stage_values(stage):
    """Full composed authored properties, including joints and material physics."""
    # dexbench: asset paths compare by the file they resolve to (bench.paths.usd_value_text).
    return {str(p.GetPath()): {'type': p.GetTypeName(), 'apis': list(p.GetAppliedSchemas()),
            'attributes': {a.GetName(): (usd_value_text(a.Get()), tuple(map(str, a.GetConnections())))
                           for a in p.GetAttributes() if a.HasAuthoredValue() or a.GetConnections()},
            'relationships': {r.GetName(): tuple(map(str, r.GetTargets())) for r in p.GetRelationships()}}
            for p in stage.Traverse()}


def assert_translation_only(before, after, movable_path, expected_position):
    """Reject any accidental physical/visual modification before finish authoring."""
    original, derived = _stage_values(before), _stage_values(after)
    key = 'xformOp:translate'
    if movable_path not in derived or key not in derived[movable_path]['attributes']:
        raise ValueError('Expected original explicit translation op')
    from pxr import UsdGeom
    actual = [float(x) for x in UsdGeom.Xformable(after.GetPrimAtPath(movable_path)).GetLocalTransformation().ExtractTranslation()]
    if not np.allclose(actual, expected_position, atol=1e-12, rtol=0):
        raise ValueError('Actual translated pose differs')
    derived[movable_path]['attributes'][key] = original[movable_path]['attributes'][key]
    if original != derived:
        raise ValueError('Placement derivative changed protected geometry, mass, mechanism, material or other pose')
    return rules.canonical_hash(original)


def assert_finish_only(before_properties, stage, appearance):
    """Only PreviewSurface roughness and the two declared texture nodes vary."""
    from pxr import UsdShade
    after = _stage_values(stage)
    allowed_new = {material+'/'+suffix for material in appearance.get('changed_materials', [])
                   for suffix in ('SeedFinishUV', 'SeedFinish')}
    shaders = {str(UsdShade.Material(stage.GetPrimAtPath(material)).ComputeSurfaceSource()[0].GetPath())
               for material in appearance.get('changed_materials', [])}
    if set(after)-set(before_properties) != allowed_new or set(before_properties)-set(after):
        raise ValueError('Finish changed unrelated prim inventory')
    for path, original in before_properties.items():
        row = deepcopy(after[path]); original = deepcopy(original)
        if path in shaders:
            row['attributes'].pop('inputs:roughness', None)
            original['attributes'].pop('inputs:roughness', None)
        if row != original:
            raise ValueError('Finish changed protected geometry, colors, labels or physics')


def _surface(stage, body_path, collider_path, direction):
    """An actual primitive surface point, expressed in the owning body's frame."""
    from pxr import Gf, UsdGeom, UsdPhysics
    prim = stage.GetPrimAtPath(collider_path)
    if not prim or not prim.HasAPI(UsdPhysics.CollisionAPI) or UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get() is False:
        raise ValueError('Interaction reference must use an enabled collider')
    cache = UsdGeom.XformCache()
    matrix = cache.GetLocalToWorldTransform(prim)*cache.GetLocalToWorldTransform(stage.GetPrimAtPath(body_path)).GetInverse()
    linear = np.asarray(matrix)[:3, :3].T
    local_dir = linear.T @ np.asarray(direction, dtype=float)
    if prim.IsA(UsdGeom.Cube):
        point = np.sign(local_dir)*float(UsdGeom.Cube(prim).GetSizeAttr().Get())/2
    elif prim.IsA(UsdGeom.Cylinder):
        shape = UsdGeom.Cylinder(prim); axis = {'X': 0, 'Y': 1, 'Z': 2}[str(shape.GetAxisAttr().Get())]
        radial = local_dir.copy(); radial[axis] = 0
        point = radial/np.linalg.norm(radial)*float(shape.GetRadiusAttr().Get()) if np.linalg.norm(radial) else radial
        point[axis] = np.sign(local_dir[axis])*float(shape.GetHeightAttr().Get())/2
    else:
        raise ValueError('Study placement reference expects original cube/cylinder collider')
    return list(matrix.Transform(Gf.Vec3d(*point)))


def _features(stage, config):
    rows = []
    def add(label, path, point, direction, group, *, state='initial', kind='actual_surface', body=None, distance=.06):
        row = {'label': label, 'prim_path': path, 'local_position_m': list(map(float, point)),
               'approach_direction_local': list(map(float, direction)), 'approach_distance_m': distance,
               'type': kind, 'reference_state': state, 'operation_group': group, 'required': True,
               'scope': 'geometric_reference_not_grasp_center'}
        if body: row['body_id'] = body
        rows.append(row)
    def surface(name, collider, direction, label, group, state='initial'):
        path = ROOT+'/Objects/'+name
        add(label, path, _surface(stage, path, path+'/'+collider, direction), direction, group, body=name, state=state)
    task = config['task_id']
    if task == 'book_retrieval':
        for direction, label in (([1,0,0], 'spine'), ([0,-1,0], 'negative_cover'), ([0,1,0], 'positive_cover')):
            surface('robotics', 'Collider', direction, 'robotics_'+label, 'robotics_pickup')
            if label == 'spine': rows[-1]['approach'] = 'shelf_front'
        goal = config['targets']['robotics']; p = np.mean(goal['bounds_m'], axis=0); p[2] = goal['support_z_m']
        add('robotics_desk_goal', ROOT, p, [0,0,1], 'robotics_placement', kind='goal')
    elif task == 'drawer_retrieval':
        surface('drawer', 'Geometry/handle_grip/Collider', [-1,0,0], 'drawer_handle_front', 'drawer_open')
        surface('drawer', 'Geometry/handle_grip/Collider', [0,0,1], 'drawer_handle_top', 'drawer_open')
        path = ROOT+'/Objects/notebook'
        local = _surface(stage, path, path+'/Geometry/cover_front/Collider', [0,0,1])
        # The notebook rides the opened drawer; this is explicitly a conditional
        # geometry reference, never a claim that the closed cabinet is clear.
        pos = np.array(config['objects']['notebook']['initial_pose']['position_m']) + local
        pos[0] -= config['mechanism']['travel_m']
        add('notebook_top_after_opening', ROOT, pos, [0,0,1], 'notebook_pickup', state='opened')
        rows[-1]['reference_assumption'] = 'Notebook carried with drawer to its authored full-travel position; actual opening trajectory and opened-scene clearance require separate validation.'
        goal = config['targets']['notebook']; p = np.mean(goal['bounds_m'], axis=0); p[2] = goal['support_z_m']
        add('notebook_tray_goal', ROOT, p, [0,0,1], 'notebook_placement', kind='goal')
        # Closing uses the same physical handle after extraction and opening.
        p = np.array(config['objects']['drawer']['initial_pose']['position_m'])
        p += _surface(stage, ROOT+'/Objects/drawer', ROOT+'/Objects/drawer/Geometry/handle_grip/Collider', [-1,0,0])
        p[0] -= config['mechanism']['travel_m']
        add('drawer_handle_opened', ROOT, p, [-1,0,0], 'drawer_close', state='opened')
    else:
        surface('panel', 'Geometry/handle_grip/Visual', [0,0,1], 'panel_handle_folded', 'panel_open')
        surface('support_pin', 'Geometry/handle/Visual', [0,1,0], 'support_pin_handle_withdrawn', 'pin_insert')
        surface('book', 'Geometry/cover_front/Collider', [0,0,1], 'book_source_top', 'book_pickup')
        from .reading_stand_rules import on_panel, THETA
        add('book_reading_goal_deployed', ROOT, on_panel([.11,0.,.006]), [-np.sin(THETA),0,np.cos(THETA)],
            'book_placement', state='opened', kind='goal')
    return rows


def _fixed_supports(stage):
    from pxr import UsdPhysics
    rows = []
    for p in stage.Traverse():
        if not str(p.GetPath()).startswith(ROOT+'/Fixtures/') or not p.HasAPI(UsdPhysics.CollisionAPI): continue
        if UsdPhysics.CollisionAPI(p).GetCollisionEnabledAttr().Get() is False: continue
        ancestor = p
        while ancestor and not ancestor.IsPseudoRoot():
            if ancestor.HasAPI(UsdPhysics.RigidBodyAPI): break
            ancestor = ancestor.GetParent()
        else:
            rows.append(str(p.GetPath()))
    return sorted(rows)


def derive_stage(parent_stage, parent_metadata, seed):
    """Short CPU authoring helper. No live state, solver or controller exists."""
    from pxr import Gf, Usd, UsdGeom, UsdPhysics
    from scenes.workrooms.living_room import seed_geometry as geometry
    task_id = parent_metadata['task_id']; plan = plan_case(task_id, seed, parent_metadata['configuration'])
    stage = Usd.Stage.Open(parent_stage.Flatten()); md = deepcopy(parent_metadata)
    config = _configuration(md['configuration'], plan)
    actors = {name: ROOT+'/Objects/'+name for name in config['objects']}
    if {str(p.GetPath()) for p in stage.Traverse() if p.HasAPI(UsdPhysics.RigidBodyAPI)} != set(actors.values()):
        raise ValueError('Study actor inventory differs')
    name = MOVABLE[task_id]; path = actors[name]; wanted = config['objects'][name]['initial_pose']['position_m']
    if seed:
        xf = UsdGeom.Xformable(stage.GetPrimAtPath(path)); ops = xf.GetOrderedXformOps()
        if len(ops) != 2 or ops[0].GetOpName() != 'xformOp:translate' or ops[1].GetOpName() != 'xformOp:orient':
            raise ValueError('Unexpected loose body pose representation')
        ops[0].Set(Gf.Vec3d(*wanted))
    physics_hash = assert_translation_only(parent_stage, stage, path, wanted)
    bounds = {name: collision_bounds(stage, path) for name, path in actors.items()}
    poses = {name: geometry.pose(stage.GetPrimAtPath(path)) for name, path in actors.items()}
    masses = geometry.mass_inventory(stage, actors)
    for name, row in md['objects'].items():
        row['initial_pose'] = deepcopy(config['objects'][name]['initial_pose'])
        if 'position_m' in row: row['position_m'] = deepcopy(poses[name]['position_m'])
    md.update(configuration=config, actors=actors, actual_body_poses_m=poses, mass_properties=masses,
              static_geometry={'bodies': [{'object_id': name, 'local_bounds_m': bounds[name],
                  'nominal_mass_kg': masses[name]['mass_kg']} for name in actors]},
              interaction_targets=_features(stage, config), fixed_support_paths=_fixed_supports(stage),
              seed_plan=plan, protected_parent_properties_sha256=physics_hash,
              placement_scope={'geometry_mass_material_joints_unchanged': True,
                  'moved_object_id': MOVABLE[task_id], 'robot_manipulation_qualified': False},
              physics_accepted=False, robot_policy_success_claimed=False)
    return stage, md


def _parent(parent_release):
    root = Path(parent_release).resolve(strict=True)
    manifest = load_json(root/'release_manifest.json')
    if manifest.get('schema') not in ('study_task_release.v1', 'workroom_robot_configuration_release.v1') or len(manifest['tasks']) != 1:
        raise ValueError('Expected original study release')
    for name, expected in manifest['files_sha256'].items():
        path = (root/name).resolve(strict=True)
        if not path.is_relative_to(root): raise ValueError('Parent file escapes release')
        _checked(path, expected)
    entry = manifest['tasks'][0]; values = {}
    for name in ('scene', 'task', 'initial_state'):
        path = (root/entry[name]).resolve(strict=True)
        if not path.is_relative_to(root): raise ValueError('Parent input escapes release')
        values[name] = load_json(path)
    _seed(values['task']['id'], 0)
    return root, values


def evaluator_sources(parent_root, task_id):
    """Only two immutable-reset validators need a finite-position extension.

The original validators still validate every other field, including rules and
thresholds. Original scoring and observer source bytes are never modified.
All ten seeds receive the same sources; the original release is untouched.
"""
    root = Path(parent_root)/'evaluator'
    sources = {str(p.relative_to(root)): p.read_text() for p in sorted(root.rglob('*.py'))}
    if not sources or 'sequence_evaluator.py' not in sources: raise ValueError('Missing frozen evaluator')
    if task_id == 'book_retrieval': return sources
    name = 'drawer_rules.py' if task_id == 'drawer_retrieval' else 'reading_stand_rules.py'
    if name not in sources: raise ValueError('Missing original mechanism rules')
    sources[name] += f'''

# Finite position-only derivative. Original configuration/rules remain authoritative.
_placement_original_validate = validate_configuration
_PLACEMENT_OFFSETS_XY_M = {OFFSETS_XY_M!r}
_PLACEMENT_OBJECT_ID = {MOVABLE[task_id]!r}

def validate_configuration(value):
    expected = configuration()
    if not isinstance(value, dict):
        raise ValueError('Expected frozen placement configuration')
    original = expected['objects'][_PLACEMENT_OBJECT_ID]['initial_pose']['position_m']
    allowed = [[original[0]+dx, original[1]+dy, original[2]]
               for dx, dy in _PLACEMENT_OFFSETS_XY_M]
    actual = value.get('objects', {{}}).get(_PLACEMENT_OBJECT_ID, {{}}).get('initial_pose', {{}}).get('position_m')
    if canonical_hash(actual) not in {{canonical_hash(row) for row in allowed}}:
        raise ValueError('Position is not one of the ten frozen placements')
    if value.get('configuration_sha256') != canonical_hash({{k:v for k,v in value.items() if k != 'configuration_sha256'}}):
        raise ValueError('Frozen placement configuration hash differs')
    normalized = deepcopy(value)
    normalized['objects'][_PLACEMENT_OBJECT_ID]['initial_pose']['position_m'] = original
    normalized['configuration_sha256'] = canonical_hash({{k:v for k,v in normalized.items() if k != 'configuration_sha256'}})
    if canonical_hash(normalized) != canonical_hash(expected):
        raise ValueError('Placement changed protected configuration fields')
    _placement_original_validate(normalized)
    return deepcopy(value)
'''
    return sources


def prepare_asset(parent_release: Path, seed: int, output: Path, *, profile=None, attempt=0) -> Path:
    if profile == "parameterized_physics_v1":
        from scenes.workrooms.parameterized.workroom import prepare_asset as prepare_parameterized
        return prepare_parameterized(parent_release, seed, output, profile=profile, attempt=attempt)
    from pxr import Usd
    from scenes.workrooms.living_room import seed_geometry as geometry
    from . import seed_variants, drawer_rules, reading_stand_rules
    from scripts.maintenance.freeze_project_sources import verify_if_snapshot
    source = verify_if_snapshot(Path(__file__).resolve().parents[3],
        required_modules=(__name__, geometry.__name__, seed_variants.__name__, rules.__name__,
                          drawer_rules.__name__, reading_stand_rules.__name__), entrypoint_file=Path(__file__))
    if not source or not source.get('verified'): raise ValueError('Placement publication requires complete frozen source')
    root, values = _parent(parent_release); task_id = values['task']['id']; _seed(task_id, seed)
    layer = next(row for row in values['scene']['task_layers'] if row['id'] == 'task')
    candidate = values['scene']['study_task_provenance']['candidate']
    usd = _checked(layer['usd_path'], layer['sha256'])
    cm = _checked(candidate['manifest_path'], candidate['manifest_sha256']); original = load_json(cm)
    metadata_path = _checked(cm.parent/original['metadata_file'], original['metadata_sha256'])
    metadata = load_json(metadata_path)
    if (metadata['configuration'] != values['task']['evaluation']['configuration'] or
            original['task_id'] != task_id or original['initial_usd_sha256'] != layer['sha256']):
        raise ValueError('Parent configuration/layer/asset differs')
    deps = dict(layer['dependencies_sha256'])
    for path, expected in deps.items(): _checked(path, expected)
    output = Path(output).resolve()
    if not output.is_relative_to(ASSET_ROOT): raise ValueError('Study placement assets require shared versioned study root')
    stage, metadata = derive_stage(Usd.Stage.Open(str(usd)), metadata, seed)
    output.mkdir(parents=True, exist_ok=False)
    before_finish = _stage_values(stage)
    appearance = _appearance(stage, seed, output)
    assert_finish_only(before_finish, stage, appearance)
    target = output/'initial.usda'
    if seed == 0: shutil.copyfile(usd, target)
    else: stage.GetRootLayer().Export(str(target))
    refs = {str(root/'release_manifest.json'): file_hash(root/'release_manifest.json'),
            str(usd): file_hash(usd), str(cm): file_hash(cm), str(metadata_path): file_hash(metadata_path)}
    metadata.update(source_snapshot=source, appearance_variant=appearance)
    _write(output/'metadata.json', metadata)
    _write(output/'SOURCE.json', {'schema': 'study_placement_source.v1', 'generator_version': GENERATOR_VERSION,
        'source_snapshot': source, 'parent_references': refs,
        'license': 'Original roughness derivative CC0-1.0; original asset/font/artwork licenses retained',
        'scope': 'XY and visual finish only; no native, reachability or manipulation qualification'})
    shutil.copyfile(cm.parent/'LICENSE.txt', output/'LICENSE.txt')
    deps.update(refs); deps.update(appearance['dependencies_sha256'])
    deps.update({str(p): file_hash(p) for p in output.iterdir() if p.is_file()})
    manifest = {'schema': SCHEMA, 'family': 'study', 'task_id': task_id, 'seed': seed,
        'generator_version': GENERATOR_VERSION, 'variant_scope': VARIANT_SCOPE, 'source_snapshot': source,
        'parent_references': refs, 'dependencies_sha256': deps,
        'initial_usd': str(target), 'initial_usd_sha256': file_hash(target),
        'metadata': str(output/'metadata.json'), 'metadata_sha256': file_hash(output/'metadata.json'),
        'parent_initial_usd_sha256': layer['sha256'],
        'parent_configuration_sha256': values['task']['evaluation']['configuration']['configuration_sha256'],
        'plan_sha256': metadata['seed_plan']['plan_sha256'], 'license_file': str(output/'LICENSE.txt'),
        'physics_accepted': False, 'native_qualified': False, 'robot_manipulation_qualified': False}
    _write(output/'seed_manifest.json', manifest)
    return output/'seed_manifest.json'


def compose_case(parent_release: Path, asset_manifest_path: Path, configuration_id: str) -> dict:
    from scenes.workrooms.parameterized.workroom import recognizes, compose_case as compose_parameterized
    if recognizes(asset_manifest_path):
        return compose_parameterized(parent_release, asset_manifest_path, configuration_id)
    root, values = _parent(parent_release)
    path = Path(asset_manifest_path).resolve(strict=True); manifest = load_json(path)
    task_id, seed = manifest['task_id'], manifest['seed']; _seed(task_id, seed)
    if (manifest.get('schema') != SCHEMA or manifest.get('family') != 'study'
            or manifest.get('generator_version') != GENERATOR_VERSION or manifest.get('variant_scope') != VARIANT_SCOPE
            or task_id != values['task']['id']):
        raise ValueError('Placement asset identity differs')
    scene, task = values['scene'], values['task']
    if configuration_id not in ('single_arm', 'dual_same_side', 'dual_opposed') or scene.get('robot_configuration', {}).get('id', 'single_arm') != configuration_id:
        raise ValueError('Requested robot configuration differs from parent')
    for file, expected in {**manifest['parent_references'], **manifest['dependencies_sha256']}.items(): _checked(file, expected)
    _checked(manifest['initial_usd'], manifest['initial_usd_sha256'])
    md = load_json(_checked(manifest['metadata'], manifest['metadata_sha256']))
    parent = task['evaluation']['configuration']; plan = plan_case(task_id, seed, parent)
    layer = next(row for row in scene['task_layers'] if row['id'] == 'task')
    if (manifest['parent_initial_usd_sha256'] != layer['sha256'] or manifest['parent_configuration_sha256'] != parent['configuration_sha256']
            or md['seed_plan'] != plan or manifest['plan_sha256'] != plan['plan_sha256']
            or md['configuration'] != _configuration(parent, plan)):
        raise ValueError('Placement parent/configuration/plan differs')
    actors = {name: ROOT+'/Objects/'+name for name in parent['objects']}
    if md['actors'] != actors or set(md['actual_body_poses_m']) != set(actors):
        raise ValueError('Placement actor inventory differs')
    for name, prim_path in actors.items():
        actual = md['actual_body_poses_m'][name]
        expected = md['configuration']['objects'][name]['initial_pose']
        if (actual['prim_path'] != prim_path or not np.allclose(actual['position_m'], expected['position_m'], rtol=0, atol=1e-12)
                or not np.allclose(rules.rotation(actual['quaternion_wxyz']), rules.rotation(expected['quaternion_wxyz']), rtol=0, atol=1e-12)):
            raise ValueError('Placement recorded pose differs from declared initial configuration')
    files = {**manifest['dependencies_sha256'], str(path): file_hash(path)}
    layer.update(usd_path=manifest['initial_usd'], sha256=manifest['initial_usd_sha256'], dependencies_sha256=deepcopy(files))
    task['episode_seed'] = seed; scene['scenario']['seed'] = seed
    task['evaluation']['configuration'] = deepcopy(md['configuration'])
    # Rules/goal/thresholds remain byte-equivalent public content. No private
    # reset poses or conditional geometry enter the ordinary public task fields.
    scene['study_task_provenance']['seed_derivative'] = {'manifest_path': str(path),
        'manifest_sha256': file_hash(path), 'seed': seed, 'native_qualified': False}
    return {**values, 'evaluator_sources': evaluator_sources(root, task_id),
        'actual_body_poses_m': deepcopy(md['actual_body_poses_m']), 'interaction_targets': deepcopy(md['interaction_targets']),
        'static_geometry': deepcopy(md['static_geometry']), 'fixed_support_paths': deepcopy(md['fixed_support_paths']),
        'asset_files_sha256': files}
