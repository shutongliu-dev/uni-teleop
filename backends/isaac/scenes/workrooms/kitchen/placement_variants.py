"""Finite world-X placements for the three remaining kitchen tasks.

This adapter does not change the earlier kitchen geometry variants. USD work
is confined to the lazy author; ordinary release composition is pre-Kit.
"""
from copy import deepcopy
import importlib
import json
import math
from pathlib import Path

from bench.paths import load_json, undistributed_bytecode
from scenes.workrooms.seed_geometry import canonical_hash, checked_seed, file_hash
from .composition import bind_episode_task, load_asset_manifest

TASK_IDS = ('breakfast_serving', 'pot_lid_placement', 'surface_wiping')
VERSION = 'kitchen_supply_world_x_texture_v1'
VARIANT_SCOPE = 'initial_xy_translation_and_surface_texture_only'
ROOT = '/World/KitchenTask'
OFFSETS_X_M = (0., -.01, -.0075, -.005, -.0025, .0025, .005, .0075, .01, .00125)
SOURCE_FILES = (
    'scenes/workrooms/kitchen/placement_variants.py',
    'scenes/workrooms/kitchen/placement_authoring.py',
    'scenes/workrooms/kitchen/composition.py',
    'scenes/workrooms/seed_geometry.py',
    'scenes/workrooms/mechanical_assembly/placement_authoring.py',
    'scenes/workrooms/mechanical_assembly/placement_variants.py',
)


def _read(path):
    return load_json(Path(path))


def _check(path, expected):
    if undistributed_bytecode(path):
        return  # pinned bytecode cache, not shipped
    if file_hash(path) != expected:
        raise ValueError('Changed frozen kitchen input: '+str(path))


def _parent(release):
    root = Path(release).resolve(strict=True)
    manifest = _read(root/'release_manifest.json')
    if (manifest.get('schema') != 'kitchen_manipulation_task_release.v1'
            or manifest.get('task_id') not in TASK_IDS):
        raise ValueError('Expected an original remaining kitchen task release')
    references = {str(root/'release_manifest.json'): file_hash(root/'release_manifest.json')}
    for name, sha in manifest['files_sha256'].items():
        path = (root/name).resolve(strict=True)
        if not path.is_relative_to(root):
            raise ValueError('Parent file escapes release')
        references[str(path)] = sha
    references.update(manifest['asset_files_sha256'])
    for path, sha in references.items():
        _check(path, sha)
    scene, task, initial = [_read(root/(name+'.json')) for name in ('scene', 'task', 'initial_state')]
    candidate, metadata = load_asset_manifest(manifest['task_id'], scene['kitchen_task_provenance']['asset_candidate'])
    if (task['id'] != manifest['task_id'] or task['asset_metadata'] != metadata
            or scene['native_task_bundle']['usd_path'] != candidate['usd_path']
            or scene['native_task_bundle']['usd_sha256'] != candidate['usd_sha256']
            or scene['native_task_bundle']['rigid_bodies'] != metadata['actors']
            or scene['native_task_bundle']['joints'] != metadata.get('joints', {})):
        raise ValueError('Original kitchen scene/metadata identity differs')
    for path, sha in ((candidate['manifest_path'], candidate['manifest_sha256']),
                      (candidate['usd_path'], candidate['usd_sha256']),
                      (candidate['metadata_path'], candidate['metadata_sha256'])):
        references[path] = sha
    references.update(candidate['auxiliary_files_sha256'])
    references.update({p: row['sha256'] for p, row in candidate['dependencies'].items()})
    return scene, task, initial, candidate, references


def plan_case(scene, task, seed):
    checked_seed(seed)
    task_id = task['id']
    if task_id not in TASK_IDS:
        raise ValueError('Unsupported kitchen placement task')
    md = task['asset_metadata']
    actors = md['actors']
    expected = {'breakfast_serving': {'tray', 'plate', 'cup', 'bread_0', 'bread_1'},
                'pot_lid_placement': {'pot', 'lid'}}
    if task_id in expected and set(actors) != expected[task_id]:
        raise ValueError('Original kitchen inventory differs')
    if task_id == 'surface_wiping':
        composite = md['elastic_composites']['sponge']
        if (set(actors) != set(composite['member_ids']) or len(actors) != 28
                or len(composite['links']) != 63 or len(md['joints']) != 63
                or scene['native_task_bundle']['elastic_composites'] != md['elastic_composites']):
            raise ValueError('All 28 original sponge members and 63 passive links are required')
    fixtures = {'breakfast_serving': [], 'pot_lid_placement': ['source_mat'],
                'surface_wiping': ['sponge_tray']}[task_id]
    if set(fixtures) != set(md['fixtures']):
        raise ValueError('Unexpected original supply fixture inventory')
    return {'schema': 'kitchen_supply_world_x_plan.v1', 'generator_version': VERSION,
        'task_id': task_id, 'seed': seed, 'allowed_axis': 'world_x',
        'translation_xy_m': [OFFSETS_X_M[seed], 0.],
        'groups': [{'id': 'original_supply', 'actor_ids': sorted(actors), 'fixture_ids': fixtures}],
        'parent_metadata_sha256': canonical_hash(md),
        'physical_parent_identity': canonical_hash({'usd_sha256': scene['native_task_bundle']['usd_sha256'],
            'metadata': md}),
        'scope': 'Initial world-X group translation and light albedo texture only; original Y/Z/orientation, geometry, mass, joints, material physics, room, robot, control and numerical thresholds unchanged.'}


def _metadata(parent, plan):
    """Only the declared reset/reference coordinates change, never local geometry."""
    md = deepcopy(parent)
    if not plan['seed']:
        return md
    dx = plan['translation_xy_m'][0]
    for row in md['objects']:
        row['initial_pose_xyzw'][0] += dx
    for row in md['fixture_metadata'].values():
        row['initial_pose_xyzw'][0] += dx
    goals = md['goals']
    if plan['task_id'] == 'breakfast_serving':
        goals['source_center_xy_m'][0] += dx
        for point in goals['source_region_xy_m']:
            point[0] += dx
        goals['source_tray_pose_xyzw'][0] += dx
        goals['nominal_transport_distance_m'] = goals['destination_center_xy_m'][0]-goals['source_center_xy_m'][0]
    elif plan['task_id'] == 'surface_wiping':
        for row in md['elastic_composites']['sponge']['cells']:
            row['initial_pose_xyzw'][0] += dx
        for pose in goals['body_poses_xyzw'].values():
            pose[0] += dx
    return md


def prepare_asset(parent_release, seed, output, *, profile=None, attempt=0):
    if profile == "parameterized_physics_v1":
        from scenes.workrooms.parameterized.workroom import prepare_asset as prepare_parameterized
        return prepare_parameterized(parent_release, seed, output, profile=profile, attempt=attempt)
    from .placement_authoring import author_asset
    parent = Path(parent_release).resolve(strict=True)
    scene, task, initial, candidate, references = _parent(parent)
    return author_asset(parent, scene, task, candidate, references,
                        plan_case(scene, task, seed), Path(output).resolve())


def _author_provenance(proof):
    _check(proof['manifest_path'], proof['manifest_sha256'])
    root = Path(proof['snapshot_root']).resolve(strict=True)
    snapshot = _read(proof['manifest_path'])
    if (snapshot.get('schema') != 'complete_benchmark_source_snapshot_v1'
            or Path(snapshot['snapshot_root']).resolve() != root
            or Path(proof['manifest_path']).resolve() != root/'source_snapshot_manifest.json'):
        raise ValueError('Original kitchen author snapshot identity is invalid')
    files = proof['source_files_sha256']
    if {str(Path(p).relative_to(root)) for p in files} != set(SOURCE_FILES):
        raise ValueError('Incomplete kitchen author source identity')
    for path, sha in files.items():
        _check(path, sha)
        if snapshot['files_sha256'].get(str(Path(path).relative_to(root))) != sha:
            raise ValueError('Original author differs from its source snapshot')
    return files


def _verify_projections(md, data):
    rows = {r['id']: r for r in md['objects']}
    if set(data['actual_body_poses_m']) != set(rows):
        raise ValueError('Actual kitchen body inventory differs')
    for name, original in rows.items():
        pose = data['actual_body_poses_m'][name]
        expected = original['initial_pose_xyzw']
        q = [expected[6], *expected[3:6]]
        if pose['prim_path'] != original['body_path']:
            raise ValueError('Actual kitchen body path differs')
        for key, values, atol in (('position_m', expected[:3], 1e-12), ('quaternion_wxyz', q, 5e-7)):
            actual = pose[key]
            if (len(actual) != len(values) or any(type(v) not in (int, float) or not math.isfinite(v) for v in actual)
                    or min(max(abs(a-b) for a,b in zip(actual, values)),
                           max(abs(a+b) for a,b in zip(actual, values)) if key == 'quaternion_wxyz' else math.inf) > atol):
                raise ValueError('Actual reset exceeds the world-X-only plan: '+name)
    bodies = data['static_geometry']['bodies']
    if len(bodies) != len(rows) or {r['object_id'] for r in bodies} != set(rows):
        raise ValueError('Static geometry differs from actual body inventory')
    for row in bodies:
        if abs(row['nominal_mass_kg']-rows[row['object_id']]['mass_kg']) > 1e-7:
            raise ValueError('Nominal mass differs from original')


def compose_case(parent_release, asset_manifest_path, configuration_id):
    from scenes.workrooms.parameterized.workroom import recognizes, compose_case as compose_parameterized
    if recognizes(asset_manifest_path):
        return compose_parameterized(parent_release, asset_manifest_path, configuration_id)
    if configuration_id != 'single_arm':
        raise ValueError('These original kitchen tasks support only single_arm')
    parent = Path(parent_release).resolve(strict=True)
    scene, task, initial, candidate, references = _parent(parent)
    path = Path(asset_manifest_path).resolve(strict=True)
    manifest = _read(path)
    if (manifest.get('schema') != 'workroom_seed_asset.v1' or manifest.get('family') != 'kitchen'
            or manifest.get('task_id') != task['id'] or manifest.get('generator_version') != VERSION
            or manifest.get('variant_scope') != VARIANT_SCOPE or manifest['parent_references'] != references):
        raise ValueError('Kitchen placement asset differs from selected parent/task/profile')
    plan = plan_case(scene, task, manifest['seed'])
    for filename, sha in manifest['dependencies_sha256'].items():
        _check(filename, sha)
    sources = _author_provenance(manifest['author_source'])
    _check(manifest['metadata'], manifest['metadata_sha256'])
    data = _read(manifest['metadata'])
    expected_md = _metadata(task['asset_metadata'], plan)
    if data['plan'] != plan or data['asset_metadata'] != expected_md:
        raise ValueError('Kitchen reset metadata differs from its finite parent-bound recipe')
    projection_keys = ('actual_body_poses_m', 'interaction_targets', 'static_geometry', 'fixed_support_paths')
    if any(data[k] != manifest[k] for k in projection_keys):
        raise ValueError('Kitchen frozen projections disagree')
    if data['protected_physics_before_sha256'] != data['protected_physics_after_sha256']:
        raise ValueError('Kitchen placement changed protected physical inputs')
    _verify_projections(expected_md, data)
    if not plan['seed']:
        if data['bundle_usd'] != {'path': candidate['usd_path'], 'sha256': candidate['usd_sha256']} or data['appearance']:
            raise ValueError('Seed zero must preserve canonical physical inputs and appearance')
    else:
        from bench.isaacsim.release import _frozen_evaluator_module
        task['asset_metadata'] = expected_md
        module = _frozen_evaluator_module(parent)
        rules = importlib.import_module(module.__package__+'.rules')
        task = bind_episode_task(task, plan['seed'], rules)
        bundle = scene['native_task_bundle']
        bundle.update(usd_path=data['bundle_usd']['path'], usd_sha256=data['bundle_usd']['sha256'])
        bundle['dependencies_sha256'] = {**bundle.get('dependencies_sha256', {}), **data['bundle_dependencies']}
        if 'elastic_composites' in bundle:
            bundle['elastic_composites'] = deepcopy(expected_md['elastic_composites'])
        bundle['observer']['configuration'] = deepcopy(task)
        scene['kitchen_task_provenance'].update(physical_reset_seeded=True,
            seed_scope='Frozen world-X placement and render-only texture; no runtime pose writes',
            placement_asset_manifest={'path': str(path), 'sha256': file_hash(path)})
    _check(data['bundle_usd']['path'], data['bundle_usd']['sha256'])
    for filename, sha in data['bundle_dependencies'].items():
        _check(filename, sha)
    return {'scene': scene, 'task': task, 'initial_state': initial,
        'evaluator_sources': {str(p.relative_to(parent/'evaluator')): p.read_text()
                              for p in (parent/'evaluator').rglob('*.py')},
        **{key: deepcopy(data[key]) for key in projection_keys},
        'asset_files_sha256': {**references, **manifest['dependencies_sha256'], **sources,
            manifest['author_source']['manifest_path']: manifest['author_source']['manifest_sha256'],
            str(path): file_hash(path)}}
