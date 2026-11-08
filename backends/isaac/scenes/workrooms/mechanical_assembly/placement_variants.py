"""Finite mechanical supply translations and render-only surface finishes.

The original task assets, fitting geometry and evaluator code remain inputs.
This module deliberately excludes the earlier L1 geometry-randomization family.
USD authoring is lazy so ordinary catalog/episode composition remains pre-Kit.
"""
from copy import deepcopy
import json
import math
from pathlib import Path

from bench.paths import load_json, undistributed_bytecode
from scenes.workrooms.seed_geometry import (canonical_hash, checked_seed,
    check_parent_release, file_hash, write_json)

VERSION = 'mechanical_supply_xy_texture_v1'
BELT_SAFE_VERSION = 'mechanical_belt_supply_xy_texture_v2'
TASK_IDS = ('peg_insertion', 'shaft_spacer_assembly', 'fastener_assembly',
    'gear_train_assembly', 'nist_connector_insertion', 'nist_belt_drive_assembly',
    'nist_cable_routing', 'nist_wire_harness', 'cooperative_insertion')
DIRECTIONS = ((0., 0.), (-1., -1.), (1., -1.), (-1., 1.), (1., 1.),
              (-1., 0.), (1., 0.), (0., -1.), (0., 1.), (.5, -.5))
MAX_TRANSLATION_M = .01
# The fixed idler shaft lies to the positive-X side of the original belt.
# Keep all nine new placements on the negative-X / non-positive-Y side.
BELT_SAFE_TRANSLATIONS_M = ((0., 0.),) + tuple(
    (x, y) for x in (-.008, -.006, -.004) for y in (-.01, -.005, 0.))
SCOPE = 'Initial XY translation and surface texture only; no geometry, mass, contact, solver or numerical rule changes.'
VARIANT_SCOPE = 'initial_xy_translation_and_surface_texture_only'


def _read(path):
    return load_json(Path(path))


def _check(path, expected):
    if undistributed_bytecode(path):
        return  # pinned bytecode cache, not shipped
    if file_hash(path) != expected:
        raise ValueError('Changed frozen input: '+str(path))


def _inputs(parent):
    parent = Path(parent).resolve()
    check_parent_release(parent)
    scene, task, initial = [_read(parent/(name+'.json')) for name in
                            ('scene', 'task', 'initial_state')]
    if task['id'] not in TASK_IDS:
        raise ValueError('Only the nine declared mechanical placement tasks')
    return scene, task, initial


def _task_fixtures(scene):
    return [row for row in scene.get('fixtures', []) if 'robot_stand' not in row['id']]


def catalog_body_path(spec, *, dynamic=True):
    """Original native USD loader's pinned anchored-body identity (no USD import)."""
    from bench.simulation.isaacsim_backend.asset_catalog import load_task_asset
    root='/World/'+('Objects/' if dynamic else 'Fixtures/')+spec['id']
    asset=load_task_asset(spec['asset_id'],spec.get('asset_root'),verify_hashes=True)
    if asset.get('backend')!='usd_reference' or asset.get('placement_form','root')=='root':
        return root
    if asset['placement_form']!='anchored':raise ValueError('Unsupported original USD placement form')
    default=asset['usd_default_prim'];body=asset['source_rigid_body_prim']
    if body!=default and not body.startswith(default+'/'):
        raise ValueError('Anchored body escapes original USD default prim')
    return root+'/'+default.rsplit('/',1)[-1]+body[len(default):]


def physical_parent_identity(scene, task):
    """Same original apparatus may be shared by the existing robot installations."""
    bundle = scene.get('native_task_bundle')
    if bundle:
        return canonical_hash({'usd_path': bundle['usd_path'], 'usd_sha256': bundle['usd_sha256'],
            'rigid_bodies': bundle['rigid_bodies'], 'joints': bundle.get('joints', {}),
            'metadata': task['asset_metadata']})
    return canonical_hash({'objects': scene['objects'], 'fixtures': _task_fixtures(scene)})


def plan_case(scene, task, seed, *, generator_version=None):
    checked_seed(seed)
    ident = task['id']
    if ident not in TASK_IDS:
        raise ValueError('Unsupported mechanical placement task')
    version = generator_version
    if version is None:
        version = BELT_SAFE_VERSION if ident == 'nist_belt_drive_assembly' else VERSION
    if version not in (VERSION, BELT_SAFE_VERSION) or (
            version == BELT_SAFE_VERSION and ident != 'nist_belt_drive_assembly'):
        raise ValueError('Unsupported mechanical placement generator version for task')
    groups = []
    if 'native_task_bundle' not in scene:
        actors = [row['id'] for row in scene['objects']]
        fixtures = ({'peg_insertion': ['supply_cradle_0', 'supply_cradle_1'],
            'shaft_spacer_assembly': ['spacer_supply_rest', 'wide_ring_supply_rest', 'top_ring_supply_rest'],
            'fastener_assembly': ['bolt_supply_rest', 'tool_supply_rest'],
            'gear_train_assembly': []})[ident]
        groups = [{'id': 'loose_supply', 'actor_ids': actors, 'fixture_ids': fixtures}]
    else:
        meta = task['asset_metadata']; actors = list(meta['actors'])
        flexible = meta['task_parameters'].get('flexible_groups', [])
        grouped = set()
        for group in flexible:
            ids = list(group['actor_ids'])
            if not ids or not set(ids) <= set(actors) or grouped.intersection(ids):
                raise ValueError('Invalid original articulated group')
            grouped.update(ids)
            groups.append({'id': group['id'], 'actor_ids': ids, 'fixture_ids': []})
        loose = [name for name in actors if name not in grouped]
        if ident == 'nist_belt_drive_assembly':
            # The original drive pulley starts seated on its fixed shaft.
            loose.remove('drive_pulley')
            original = next(row for row in meta['objects'] if row['id'] == 'drive_pulley')
            goal = next(row for row in meta['task_parameters']['seating_goals'] if row['actor'] == 'drive_pulley')
            if original['initial_position_m'] != goal['target_position_m']:
                raise ValueError('Unexpected original pre-seated drive pulley')
        if loose:
            groups.append({'id': 'loose_supply', 'actor_ids': loose, 'fixture_ids': []})
    moved = [name for g in groups for name in g['actor_ids']]
    if len(set(moved)) != len(moved):
        raise ValueError('A body cannot belong to multiple translation groups')
    delta = (list(BELT_SAFE_TRANSLATIONS_M[seed]) if version == BELT_SAFE_VERSION
             else [v*MAX_TRANSLATION_M for v in DIRECTIONS[seed]])
    return {'schema': 'mechanical_supply_xy_plan.v1', 'generator_version': version,
        'task_id': ident, 'seed': seed, 'translation_xy_m': delta,
        'groups': groups, 'fixed_actor_ids': sorted(set(actors)-set(moved)),
        'physical_parent_identity': physical_parent_identity(scene, task),
        'rotation_changed': False, 'height_changed': False, 'scope': SCOPE}


def _translate(position, delta):
    return [position[0]+delta[0], position[1]+delta[1], position[2]]


def _compose_numeric(scene, task, plan, data):
    """Only authored reset fields move. Targets and all tolerances remain fixed."""
    scene, task = deepcopy(scene), deepcopy(task)
    seed = plan['seed']; delta = plan['translation_xy_m']
    moved = {name for g in plan['groups'] for name in g['actor_ids']}
    rests = {name for g in plan['groups'] for name in g['fixture_ids']}
    if not seed:
        return scene, task
    if 'native_task_bundle' in scene:
        meta = deepcopy(task['asset_metadata'])
        poses = data['actual_body_poses_m']
        for row in meta['objects']:
            if row['id'] in moved:
                p = poses[row['id']]['position_m']
                row['initial_position_m'] = deepcopy(p)
                row['initial_pose_xyzw'][:3] = p
        task['asset_metadata'] = meta
        # The original native observer and evaluator use this same metadata.
        if 'metadata' in task['evaluation']:
            task['evaluation']['metadata'] = deepcopy(meta)
        task['evaluation']['episode_seed']=seed
        task['episode_seed']=seed
        if 'objects' in task['evaluation']:
            task['evaluation']['objects'] = deepcopy(meta['objects'])
        bundle = scene['native_task_bundle']
        bundle.update(usd_path=data['bundle_usd']['path'], usd_sha256=data['bundle_usd']['sha256'])
        bundle['dependencies_sha256'] = {**bundle.get('dependencies_sha256', {}), **data['bundle_dependencies']}
        if bundle.get('observer'):
            bundle['observer']['configuration'] = deepcopy(task)
    else:
        for row in scene['objects']:
            if row['id'] in moved:
                row['position_m'] = _translate(row['position_m'], delta)
            if row['id'] in data['presentations']:
                row['presentation'] = deepcopy(data['presentations'][row['id']])
        for row in scene['fixtures']:
            if row['id'] in rests:
                row['position_m'] = _translate(row['position_m'], delta)
        task['evaluation']['initial_objects'] = {row['id']: {k: deepcopy(row.get(k, [1.,0.,0.,0.]))
            for k in ('position_m', 'quaternion_wxyz')} for row in scene['objects']}
        reset = task.get('reset_conditions', {})
        if 'supply_positions_m' in reset:
            reset['supply_positions_m'] = {name: _translate(p, delta) for name,p in reset['supply_positions_m'].items()}
        for key in ('bolt_supply_position_m', 'tool_supply_position_m'):
            if key in reset:
                reset[key] = _translate(reset[key], delta)
        for container in (task, scene):
            lane = container.get('task_layout', {}).get('supply_lane')
            if lane:
                for key in ('positions_m', 'part_positions_m'):
                    lane[key] = {name: _translate(p, delta) for name,p in lane[key].items()}
        supply = task.get('supply_layout', {})
        if 'gear_supply_positions_xy_m' in supply:
            supply['gear_supply_positions_xy_m'] = [[p[0]+delta[0],p[1]+delta[1]]
                for p in supply['gear_supply_positions_xy_m']]
    return scene, task


def prepare_asset(parent_release: Path, seed: int, output: Path, *, generator_version=None, profile=None, attempt=0) -> Path:
    if profile == "parameterized_physics_v1":
        from scenes.workrooms.parameterized.workroom import prepare_asset as prepare_parameterized
        return prepare_parameterized(parent_release, seed, output, profile=profile, attempt=attempt)
    from .placement_authoring import author_asset
    parent = Path(parent_release).resolve()
    scene, task, initial = _inputs(parent)
    return author_asset(parent, scene, task, initial,
        plan_case(scene, task, seed, generator_version=generator_version), Path(output).resolve())


def _check_reset_projections(scene, task, plan, data):
    """Re-derive poses from the pinned parent; never trust a rehashed sidecar."""
    bundle=scene.get('native_task_bundle')
    if bundle:
        rows=task['asset_metadata']['objects']
        original={r['id']:{'position_m':r['initial_position_m'],
            'quaternion_wxyz':[r['orientation_xyzw'][3],*r['orientation_xyzw'][:3]],
            'prim_path':bundle['rigid_bodies'][r['id']]} for r in rows}
    else:
        # compose_case already checked the frozen parent/dependency file hashes.
        from scenes.workrooms.initial_randomization.physical_inputs import object_actor_path
        original={r['id']:{'position_m':r['position_m'],
            'quaternion_wxyz':r.get('quaternion_wxyz',[1.,0.,0.,0.]),
            'prim_path':object_actor_path(r)} for r in scene['objects']}
    poses=data['actual_body_poses_m']
    if set(poses)!=set(original):raise ValueError('Reset body inventory differs from original')
    moved={n for group in plan['groups'] for n in group['actor_ids']}
    for name,source in original.items():
        actual=poses[name]
        expected=_translate(source['position_m'],plan['translation_xy_m']) if name in moved else source['position_m']
        if actual['prim_path']!=source['prim_path']:
            raise ValueError('Reset body path changed')
        # USD stores orient as quatf. Gf's matrix -> axis-angle readback keeps
        # the float32 scalar and reconstructs its axis component: the actual
        # original belt has up to 3.285e-7 component rounding near identity.
        # This is input-identity precision, never a referee pose threshold.
        for key,values,tolerance in (('position_m',expected,1e-12),
                                    ('quaternion_wxyz',source['quaternion_wxyz'],5e-7)):
            got=actual[key]
            if (len(got)!=len(values) or any(type(x) not in (int,float) or not math.isfinite(x) for x in got)
                    or min(max(abs(a-b) for a,b in zip(got,values)),
                           max(abs(a+b) for a,b in zip(got,values)) if key=='quaternion_wxyz' else math.inf)>tolerance):
                raise ValueError('Reset pose exceeds original XY-only plan: '+name)
    bodies=data['static_geometry']['bodies']
    if len(bodies)!=len(original) or {r['object_id']for r in bodies}!=set(original):
        raise ValueError('Nominal geometry inventory differs from actual bodies')
    if not plan['seed'] and (data['presentations'] or data.get('bundle_usd') or data['appearance']):
        raise ValueError('Seed zero must preserve canonical appearance and asset paths')


def compose_case(parent_release, asset_manifest_path, configuration_id):
    from scenes.workrooms.parameterized.workroom import recognizes, compose_case as compose_parameterized
    if recognizes(asset_manifest_path):
        return compose_parameterized(parent_release, asset_manifest_path, configuration_id)
    parent = Path(parent_release).resolve()
    scene, task, initial = _inputs(parent)
    manifest = _read(asset_manifest_path)
    if (manifest.get('schema') != 'workroom_seed_asset.v1'
            or manifest.get('generator_version') not in (VERSION, BELT_SAFE_VERSION)
            or manifest.get('family') != 'mechanical_assembly' or manifest.get('task_id') != task['id']):
        raise ValueError('Wrong mechanical placement asset identity')
    if configuration_id not in (('dual_same_side',) if task['id']=='cooperative_insertion'
                                 else ('single_arm','dual_same_side','dual_opposed')):
        raise ValueError('Unsupported original robot installation')
    original_configuration=('dual_same_side' if task['id']=='cooperative_insertion'
                            else scene.get('robot_configuration',{}).get('id','single_arm'))
    if configuration_id!=original_configuration:
        raise ValueError('Requested installation differs from the selected parent robot')
    for path, sha in {**manifest['parent_references'], **manifest['dependencies_sha256']}.items():
        _check(path, sha)
    provenance=manifest['author_source'];_check(provenance['manifest_path'],provenance['manifest_sha256'])
    snapshot=_read(provenance['manifest_path']);source_root=Path(provenance['snapshot_root'])
    if (snapshot.get('schema')!='complete_benchmark_source_snapshot_v1'
            or Path(snapshot['snapshot_root']).resolve()!=source_root.resolve()
            or Path(provenance['manifest_path']).resolve()!=(source_root/'source_snapshot_manifest.json').resolve()):
        raise ValueError('Original author snapshot identity is invalid')
    required={'scenes/workrooms/mechanical_assembly/placement_variants.py',
        'scenes/workrooms/mechanical_assembly/placement_authoring.py','scenes/workrooms/seed_geometry.py',
        'bench/simulation/isaacsim_backend/asset_catalog.py','bench/simulation/isaacsim_backend/presentation.py'}
    files=provenance['source_files_sha256']
    if {str(Path(p).relative_to(source_root))for p in files}!=required:
        raise ValueError('Incomplete original author source identity')
    for path,sha in files.items():
        _check(path,sha)
        if snapshot['files_sha256'].get(str(Path(path).relative_to(source_root)))!=sha:
            raise ValueError('Author source differs from its frozen snapshot')
    data = _read(manifest['metadata'])
    _check(manifest['metadata'], manifest['metadata_sha256'])
    expected = plan_case(scene, task, manifest['seed'], generator_version=manifest['generator_version'])
    if data['plan'] != expected or manifest['variant_scope'] != VARIANT_SCOPE:
        raise ValueError('Frozen supply translation differs from its original apparatus')
    if any(data[k] != manifest[k] for k in ('actual_body_poses_m','interaction_targets','static_geometry','fixed_support_paths')):
        raise ValueError('Frozen seed projections disagree')
    if data.get('protected_physics_before_sha256') != data.get('protected_physics_after_sha256'):
        raise ValueError('Mechanical seed changed protected physical inputs')
    _check_reset_projections(scene, task, expected, data)
    scene, task = _compose_numeric(scene, task, expected, data)
    sources = {str(p.relative_to(parent/'evaluator')): p.read_text()
               for p in (parent/'evaluator').rglob('*.py')}
    return {'scene':scene,'task':task,'initial_state':initial,'evaluator_sources':sources,
        **{key:deepcopy(data[key]) for key in ('actual_body_poses_m','interaction_targets','static_geometry','fixed_support_paths')},
        'asset_files_sha256':{**manifest['parent_references'], **manifest['dependencies_sha256'],
            **files,provenance['manifest_path']:provenance['manifest_sha256'],
            str(Path(asset_manifest_path).resolve()):file_hash(asset_manifest_path)}}
