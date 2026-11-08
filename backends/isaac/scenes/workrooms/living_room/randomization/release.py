"""Prepare and load immutable, reset-only ordinary living-room case releases."""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
import shutil
import tempfile
import types

from . import TASK_IDS, SEEDS, CONFIGURATIONS
from .cases import validate_plan
from .common import (canonical_hash, file_hash, read_json, write_json, checked_file,
                     local_path, selection, checked_poses, vector)
from bench.paths import DATA_ROOT

SCHEMA = 'living_room_randomized_release.v1'
VERSION = 'isaacsim-living-room-randomized-release-v1'
ASSET_SCHEMA = 'living_room_seeded_asset_variant.v1'
DATASETS = Path(f'{DATA_ROOT}/datasets')
PARENT_SCHEMAS = frozenset({'living_room_manipulation_task_release.v1',
                          'workroom_robot_configuration_release.v1'})


def _metadata(task):
    return task.get('asset_metadata', task['evaluation'].get('candidate_metadata'))


def _source_files(parent, task_id, configuration_id):
    """Only new remote releases adopt the reviewed dimensional parameterization."""
    sources = {str(p.relative_to(parent/'evaluator')): p.read_text()
               for p in (parent/'evaluator').rglob('*.py')}
    if task_id not in ('remote_operation', 'storage_box'):
        return sources
    folder = Path(__file__).parents[1]/'tasks'/task_id
    sources = {'sequence_evaluator.py': (folder/'evaluator.py').read_text(),
               'rules.py': (folder/'rules.py').read_text()}
    if task_id == 'remote_operation' and configuration_id != 'single_arm':
        from scenes.workrooms.robot_configuration_adaptation import adapt_payload
        manifest = read_json(parent/'release_manifest.json')
        single = Path(manifest['base_release']['path'])
        original_task = read_json(single/'task.json')
        _, sources, _ = adapt_payload('living_room', task_id, original_task, sources)
    return sources


def _rules(sources):
    module = types.ModuleType('_living_seeded_rules_'+canonical_hash(sources))
    exec(compile(sources['rules.py'], '<frozen living-room rules>', 'exec'), module.__dict__)
    return module


def _check_sources(parent, task_id, configuration_id, sources):
    from .source_pins import REMOTE_SINGLE, REMOTE_DUAL, STORAGE
    if task_id == 'remote_operation':
        expected = REMOTE_SINGLE if configuration_id == 'single_arm' else REMOTE_DUAL
    elif task_id == 'storage_box':
        expected = STORAGE
    else:
        expected = {str(p.relative_to(parent/'evaluator')): file_hash(p)
                    for p in (parent/'evaluator').rglob('*.py')}
    if {name: _text_hash(text) for name, text in sources.items()} != expected:
        raise ValueError('Unreviewed frozen geometry/scoring adapter source')


def _parent_asset(scene):
    candidate = scene['living_room_provenance']['asset_candidate']
    manifest = read_json(checked_file(candidate['manifest_path'], candidate['manifest_sha256']))
    return {'manifest_path': candidate['manifest_path'], 'manifest_sha256': candidate['manifest_sha256'],
        'initial_usd': candidate['initial_usd'], 'initial_usd_sha256': candidate['initial_usd_sha256'],
        'metadata': str(Path(candidate['initial_usd']).with_name('metadata.json')),
        'metadata_sha256': manifest['metadata_sha256']}


def _asset(path, *, task_id, seed, original_metadata, parent_asset, assets=True):
    path = Path(path).resolve()
    value = read_json(path)
    if (value.get('schema') != ASSET_SCHEMA or value.get('task_id') != task_id
            or type(value.get('seed')) is not int or value['seed'] != seed):
        raise ValueError('Expected a versioned living-room reset asset')
    plan = value['case_plan']
    validate_plan(plan, original_metadata)
    if (plan['task_id'], plan['seed']) != (task_id, seed) or value['case_plan_sha256'] != plan['plan_sha256']:
        raise ValueError('Asset belongs to a different deterministic reset case')
    metadata_path = Path(value['metadata']).resolve()
    checked_file(metadata_path, value['metadata_sha256'])
    metadata = read_json(metadata_path)
    if metadata.get('task_id', task_id) != task_id:
        raise ValueError('Asset metadata has a different task identity')
    if value['parent_asset_reference'] != parent_asset:
        raise ValueError('Reset asset has a different original physical parent')
    if plan.get('generator_version') == 'parameterized_physics_v1':
        recipe = value.get('parameterized_physics', {})
        if (recipe.get('parameters') != plan['parameters']
                or recipe.get('applied') != metadata.get('parameterized_physics', {'canonical': True, 'applied': False})):
            raise ValueError('Parameterized physical recipe differs from the actual metadata')
        hashes = recipe.get('source_files_sha256', {})
        snapshot = value.get('source_snapshot', {})
        source_root = Path(snapshot.get('snapshot_root', ''))
        expected_paths = {str(source_root/name) for name in (
            'scenes/workrooms/living_room/randomization/parameterized.py',
            'scenes/workrooms/parameterized/spec.py', 'scenes/workrooms/parameterized/appearance.py')}
        if not source_root.is_absolute() or set(hashes) != expected_paths:
            raise ValueError('Parameterized living author sources are incomplete')
        for source_path, expected in hashes.items():
            checked_file(source_path, expected)
    checked_poses(value['actual_body_poses_m'])
    if value['interaction_features'] != metadata['interaction_targets']:
        raise ValueError('Asset feature inventories disagree')
    inventory = {str(path): file_hash(path), str(metadata_path): value['metadata_sha256'],
                 str(Path(value['initial_usd']).resolve()): value['initial_usd_sha256'],
                 **value['dependencies_sha256']}
    if plan.get('generator_version') == 'parameterized_physics_v1':
        inventory.update(value['parameterized_physics']['source_files_sha256'])
    if assets:
        for p, digest in inventory.items():
            checked_file(p, digest)
    for name in ('source_physics_identity', 'visual_variant', 'parent_asset_reference'):
        if not isinstance(value.get(name), dict):
            raise ValueError('Missing asset physical/visual provenance: '+name)
    body_geometry = metadata.get('static_body_geometry')
    if (not isinstance(body_geometry, list) or len(body_geometry) != len(value['actual_body_poses_m'])
            or {row.get('object_id') for row in body_geometry} != set(value['actual_body_poses_m'])):
        raise ValueError('Authored body-local dimensions and masses are required')
    for row in body_geometry:
        bounds = row['local_bounds_m']
        if not isinstance(bounds, list) or len(bounds) != 2:
            raise ValueError('Body-local bounds must contain two corners')
        lo, hi = [vector(p, 3, 'body-local bound') for p in bounds]
        mass, = vector([row['nominal_mass_kg']], 1, 'nominal mass')
        if mass <= 0 or any(lo[i] >= hi[i] for i in range(3)):
            raise ValueError('Positive physical mass and body extent required')
    value['_manifest_path'] = str(path)
    value['_manifest_sha256'] = file_hash(path)
    return value, metadata, inventory


def _body_paths(scene):
    if 'native_task_bundle' in scene:
        return scene['native_task_bundle']['rigid_bodies']
    return scene['task_layers'][0]['rigid_bodies']


def _geometry_public(task_id, metadata, task):
    """Only nominal dimensions and static goals, never reset body truth."""
    bodies = [{k: deepcopy(row[k]) for k in ('object_id', 'local_bounds_m', 'nominal_mass_kg')}
              for row in metadata['static_body_geometry']]
    if task_id == 'remote_operation':
        return {'dimensions_m': {'remote': metadata['remote']['envelope_xyz_m'],
            'holder_opening': metadata['holder']['opening_xy_m']}, 'bodies': bodies,
            'goal': {'operation': 'remote_return_to_holder'}}
    if task_id == 'water_pouring':
        return {'dimensions_m': {name+'_inner_profile': row['inner_profile_z_radius_spout_m']
                for name, row in metadata['containers'].items()}, 'bodies': bodies,
            'handle_geometry': {'capsule_radius_m': metadata['handle_geometry']['capsule_radius_m']},
            'goal': {'operation': 'pour_and_return', 'return_position_m': metadata['containers']['kettle']['initial_position_m'],
                     'target_interval_ml': task['public_rules']['goal']['target_interval_ml']}}
    if task_id == 'storage_box':
        return {'dimensions_m': {'item': metadata['item_size_m']}, 'bodies': bodies,
            'goal': {'operation': 'collect_and_close', 'tray_inside_xy_m': metadata['tray_inside_xy_m'],
                     'tray_floor_z_m': metadata['tray_floor_z_m']}}
    raise ValueError('Unsupported randomized ordinary task')


def _compose(parent, configuration_id, seed, asset, metadata, sources):
    scene, task, initial = [read_json(parent/(n+'.json')) for n in ('scene', 'task', 'initial_state')]
    ident = task['id']
    selection(ident, seed, configuration_id)
    source_metadata = _metadata(task)
    geometry_metadata = deepcopy(metadata)
    expected = _body_paths(scene)
    actual = asset['actual_body_poses_m']
    if set(actual) != set(expected) or any(actual[n]['prim_path'] != p for n, p in expected.items()):
        raise ValueError('Randomization must retain every original native actor identity')
    if seed == 0:
        original_usd = (scene['native_task_bundle']['usd_sha256'] if 'native_task_bundle' in scene
                        else scene['task_layers'][0]['sha256'])
        if asset['initial_usd_sha256'] != original_usd:
            raise ValueError('Seed zero must preserve the original USD bytes')
        # Feature/provenance fields may be appended to the separate asset record;
        # the canonical task's scoring and public inputs remain identical.
        metadata = deepcopy(source_metadata)
    else:
        from scenes.workrooms.living_room.runtime_task import bind_episode_task
        task['asset_metadata'] = deepcopy(metadata)
        task = bind_episode_task(task, seed, _rules(sources))
        if ident == 'storage_box':
            rules = _rules(sources)
            task['public_rules'] = rules.public_rules(asset_metadata=metadata)
            task['public_rules_sha256'] = canonical_hash(task['public_rules'])
            task['public_referee_contract_sha256'] = canonical_hash(task['public_rules']['referee'])
            bounds = metadata['tray_inside_xy_m']
            task['public_goal']['target_center_xy_m'] = [(bounds[0][i]+bounds[1][i])/2 for i in (0, 1)]
    if 'scenario' in scene:
        scene['scenario']['seed'] = seed
    if 'seed' in task:
        task['seed'] = seed
    if 'episode_seed' in task:
        task['episode_seed'] = seed
    if seed != 0:
        if 'native_task_bundle' in scene:
            bundle = scene['native_task_bundle']
            bundle.update(usd_path=asset['initial_usd'], usd_sha256=asset['initial_usd_sha256'])
        else:
            scene['task_layers'][0].update(usd_path=asset['initial_usd'], sha256=asset['initial_usd_sha256'],
                dependencies_sha256=deepcopy(asset['dependencies_sha256']))
        provenance = scene['living_room_provenance']
        provenance['parent_asset_candidate'] = deepcopy(provenance['asset_candidate'])
        provenance['asset_candidate'].update(manifest_path=asset['_manifest_path'],
            manifest_sha256=asset['_manifest_sha256'], initial_usd=asset['initial_usd'],
            initial_usd_sha256=asset['initial_usd_sha256'],
            evidence_scope='Versioned reset-only geometry/layout inputs; no inherited native qualification.')
    content = {'task_id': ident, 'configuration_id': configuration_id,
        'parent_release_sha256': file_hash(parent/'release_manifest.json'),
        'asset_usd_sha256': asset['initial_usd_sha256'],
        'actual_body_poses_m': deepcopy(actual), 'geometry': _geometry_public(ident, geometry_metadata, task),
        'source_physics_identity': deepcopy(asset['source_physics_identity']),
        'visual_variant': deepcopy(asset['visual_variant'])}
    case = {'schema': 'living_room_reset_case.v1', 'task_id': ident, 'configuration_id': configuration_id,
        'seed': seed, 'parent_release_sha256': content['parent_release_sha256'],
        'case_fingerprint': canonical_hash(content), 'case_plan': deepcopy(asset['case_plan']),
        'actual_body_poses_m': deepcopy(actual), 'interaction_targets': deepcopy(asset['interaction_features']),
        'asset_reference': {'initial_usd': asset['initial_usd'], 'initial_usd_sha256': asset['initial_usd_sha256'],
            'metadata': asset['metadata'], 'metadata_sha256': asset['metadata_sha256']},
        'static_geometry': content['geometry'], 'source_physics_identity': content['source_physics_identity'],
        'visual_variant': content['visual_variant'], 'reset_only': True, 'runtime_object_writes': False}
    if asset['case_plan'].get('generator_version') == 'parameterized_physics_v1':
        case['parameterized_physics'] = deepcopy(asset['parameterized_physics'])
    case['case_sha256'] = canonical_hash(case)
    task['reset_case'] = deepcopy(case)
    scene['living_room_reset_case'] = {k: case[k] for k in ('schema', 'seed', 'case_sha256', 'case_fingerprint')}
    for name in ('observer', 'reduced_dynamics'):
        hook = scene.get('native_task_bundle', {}).get(name)
        if hook:
            hook['configuration'] = deepcopy(task)
            hook['files'] = {n: _text_hash(text) for n, text in sources.items()}
            hook['sha256'] = hook['files'][str(Path(hook['path']).relative_to('evaluator'))]
    return scene, task, initial, case


def _text_hash(value):
    import hashlib
    return hashlib.sha256(value.encode()).hexdigest()


def prepare_release(parent_release, output, *, configuration_id, assets_by_seed):
    """Create a fresh ten-case derivative; no old source/asset/release is edited."""
    from bench.isaacsim.release import validate
    parent, output = Path(parent_release).resolve(), Path(output).resolve()
    original = validate(parent)
    ident = original['task_id']
    selection(ident, 0, configuration_id)
    if original['schema'] not in PARENT_SCHEMAS or original.get('task_kind') == 'turn_game':
        raise ValueError('Only original ordinary living-room releases can be derived')
    if (output.exists() or not output.is_relative_to(DATASETS) or
            set(assets_by_seed) != set(SEEDS) or any(type(k) is not int for k in assets_by_seed)):
        raise ValueError('Choose a fresh shared-dataset release and exactly ten integer seed assets')
    parent_scene, parent_task = read_json(parent/'scene.json'), read_json(parent/'task.json')
    actual_config = parent_scene.get('robot_configuration', {}).get('id', 'single_arm')
    if actual_config != configuration_id:
        raise ValueError('Parent robot installation differs from requested configuration')
    sources = _source_files(parent, ident, configuration_id)
    _check_sources(parent, ident, configuration_id, sources)
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix='.'+output.name+'.preparing-', dir=output.parent))
    try:
        inventory = read_json(parent/'asset_file_hashes.json')
        # Parent dependencies remain bound, even when a new task USD replaces one.
        inventory[str(parent/'release_manifest.json')] = {'sha256': file_hash(parent/'release_manifest.json')}
        cases = {}
        for seed in SEEDS:
            asset, metadata, deps = _asset(assets_by_seed[seed], task_id=ident, seed=seed,
                                         original_metadata=_metadata(parent_task), parent_asset=_parent_asset(parent_scene))
            inventory.update({p: {'sha256': h} for p, h in deps.items()})
            scene, task, initial, case = _compose(parent, configuration_id, seed, asset, metadata, sources)
            prefix = f'cases/seed_{seed:02d}'
            for name, value in (('scene', scene), ('task', task), ('initial_state', initial), ('case', case)):
                write_json(staging/prefix/(name+'.json'), value)
                if seed == 0 and name != 'case':
                    write_json(staging/(name+'.json'), value)
            cases[str(seed)] = {'scene': prefix+'/scene.json', 'task': prefix+'/task.json',
                'initial_state': prefix+'/initial_state.json', 'case': prefix+'/case.json',
                'case_sha256': case['case_sha256'], 'case_fingerprint': case['case_fingerprint'],
                'asset_manifest': str(Path(assets_by_seed[seed]).resolve()),
                'asset_manifest_sha256': file_hash(assets_by_seed[seed])}
        for name, text in sources.items():
            path = staging/'evaluator'/name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text)
        shutil.copy2(parent/'physics_profile.json', staging/'physics_profile.json')
        write_json(staging/'rules.json', read_json(staging/'task.json')['public_rules'])
        write_json(staging/'asset_file_hashes.json', inventory)
        write_json(staging/'source_provenance.json', {'schema': 'living_room_seed_release_sources.v1',
            'sources_sha256': {str(p): file_hash(p) for p in sorted(Path(__file__).parent.glob('*.py'))},
            'original_assets_modified': False, 'original_releases_modified': False,
            'shared_robot_controller_unchanged': True, 'runtime_object_writes': False,
            'evaluator_sources_sha256': {n: _text_hash(t) for n, t in sources.items()}})
        files = {str(p.relative_to(staging)): {'sha256': file_hash(p)} for p in sorted(staging.rglob('*')) if p.is_file()}
        entry = deepcopy(original['tasks'][0])
        entry['task_kind'] = 'single_agent'
        for name in ('scene', 'task', 'initial_state'):
            entry[name] = name+'.json'
            entry[name+'_sha256'] = files[name+'.json']['sha256']
        manifest = {'schema': SCHEMA, 'schema_version': VERSION, 'task_id': ident,
            'task_kind': 'single_agent', 'configuration_id': configuration_id,
            'release_id': ident+'_seeded_'+output.name, 'prepared_at_utc': datetime.now(timezone.utc).isoformat(),
            'status': 'prepared_unvalidated', 'physics_accepted': False, 'task_policy_success_claimed': False,
            'original_release_modified': False, 'source_release': None,
            'base_release': {'path': str(parent), 'manifest_sha256': file_hash(parent/'release_manifest.json'),
                             'schema': original['schema']},
            'seed_values': list(SEEDS), 'cases': cases, 'tasks': [entry], 'files': files,
            'files_sha256': {n: r['sha256'] for n, r in files.items()},
            'asset_files_sha256': {n: r['sha256'] for n, r in inventory.items()},
            'physics_profile': 'physics_profile.json', 'physics_profile_sha256': files['physics_profile.json']['sha256'],
            'qualification_scope': 'Versioned physical reset inputs; IK and native feasibility are separate evidence.'}
        write_json(staging/'release_manifest.json', manifest)
        validate_release(staging, manifest)
        staging.rename(output)
        return manifest
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def validate_release(root, manifest, *, assets=True):
    from bench.isaacsim.release import validate, _frozen_evaluator_module
    root = Path(root).resolve()
    selection(manifest['task_id'], 0, manifest['configuration_id'])
    if (manifest.get('schema') != SCHEMA or manifest.get('schema_version') != VERSION
            or manifest.get('task_kind') != 'single_agent' or manifest.get('seed_values') != list(SEEDS)
            or manifest.get('status') != 'prepared_unvalidated' or manifest.get('physics_accepted') is not False
            or manifest.get('task_policy_success_claimed') is not False or manifest.get('original_release_modified') is not False):
        raise ValueError('Malformed ordinary randomized release')
    base = manifest['base_release']; parent = Path(base['path']).resolve()
    if parent == root:
        raise ValueError('Randomized release must bind an independent original parent')
    checked_file(parent/'release_manifest.json', base['manifest_sha256'])
    original = validate(parent, assets=assets)
    if original['schema'] not in PARENT_SCHEMAS or original['schema'] != base['schema'] or original['task_id'] != manifest['task_id']:
        raise ValueError('Changed original parent task')
    parent_scene = read_json(parent/'scene.json')
    if parent_scene.get('robot_configuration', {}).get('id', 'single_arm') != manifest['configuration_id']:
        raise ValueError('Parent robot installation differs from the randomized release')
    entries = manifest.get('tasks')
    if (not isinstance(entries, list) or len(entries) != 1 or entries[0].get('id') != manifest['task_id']
            or entries[0].get('task_kind') != 'single_agent'):
        raise ValueError('Randomized release requires one ordinary task entry')
    files = manifest['files']
    if manifest['files_sha256'] != {n: r['sha256'] for n, r in files.items()}:
        raise ValueError('Frozen file inventory indexes disagree')
    actual = {str(p.relative_to(root)) for p in root.rglob('*') if p.is_file()
              and not ('__pycache__' in p.parts and p.suffix == '.pyc')}
    if actual != set(files) | {'release_manifest.json'}:
        raise ValueError('Frozen release contains missing or undeclared files')
    for name, row in files.items():
        checked_file(local_path(root, name), row['sha256'])
    for name in ('scene', 'task', 'initial_state'):
        if (entries[0].get(name) != name+'.json' or
                entries[0].get(name+'_sha256') != files[name+'.json']['sha256']):
            raise ValueError('Canonical task entry differs from frozen inputs')
    inventory = read_json(root/'asset_file_hashes.json')
    if manifest['asset_files_sha256'] != {n: r['sha256'] for n, r in inventory.items()}:
        raise ValueError('Frozen external input inventory differs')
    if assets:
        for name, row in inventory.items():
            checked_file(name, row['sha256'])
    if (manifest['physics_profile'] != 'physics_profile.json' or
            manifest['physics_profile_sha256'] != file_hash(parent/'physics_profile.json') or
            file_hash(root/'physics_profile.json') != manifest['physics_profile_sha256']):
        raise ValueError('Randomization changed the shared physical profile')
    cases = manifest['cases']
    if set(cases) != {str(s) for s in SEEDS} or len({r['case_fingerprint'] for r in cases.values()}) != len(SEEDS):
        raise ValueError('Exactly ten distinct physical cases are required')
    sources = {str(p.relative_to(root/'evaluator')): p.read_text() for p in (root/'evaluator').rglob('*.py')}
    _check_sources(parent, manifest['task_id'], manifest['configuration_id'], sources)
    parent_task = read_json(parent/'task.json')
    parent_asset = _parent_asset(parent_scene)
    evaluator = _frozen_evaluator_module(root).SequenceEvaluator
    for seed in SEEDS:
        record = cases[str(seed)]
        checked_file(record['asset_manifest'], record['asset_manifest_sha256'])
        asset, metadata, deps = _asset(record['asset_manifest'], task_id=manifest['task_id'], seed=seed,
                                     original_metadata=_metadata(parent_task), parent_asset=parent_asset, assets=assets)
        if any(inventory.get(p, {}).get('sha256') != h for p, h in deps.items()):
            raise ValueError('Case dependencies are absent from the release inventory')
        expected = _compose(parent, manifest['configuration_id'], seed, asset, metadata, sources)
        for name, value in zip(('scene', 'task', 'initial_state', 'case'), expected):
            if read_json(local_path(root, record[name])) != value:
                raise ValueError('Frozen case differs from bound source inputs: '+name)
        if (record['case_sha256'], record['case_fingerprint']) != (expected[3]['case_sha256'], expected[3]['case_fingerprint']):
            raise ValueError('Case identity differs')
        evaluator(expected[1])
        if seed == 0:
            for name, value in zip(('scene', 'task', 'initial_state'), expected):
                if read_json(root/(name+'.json')) != value:
                    raise ValueError('Discovery inputs must select canonical seed zero')
    if read_json(root/'rules.json') != read_json(root/'task.json')['public_rules']:
        raise ValueError('Canonical public rules differ')
    return manifest


def task_inputs(root, task_id, manifest, *, seed=0, phase='nominal'):
    selection(task_id, seed, manifest['configuration_id'])
    if task_id != manifest['task_id'] or phase != 'nominal':
        raise ValueError('Randomized tasks support only their frozen physical reset cases')
    root = Path(root).resolve()
    record = manifest['cases'][str(seed)]
    scene, task, initial = [read_json(local_path(root, record[n])) for n in ('scene', 'task', 'initial_state')]
    for name in ('observer', 'reduced_dynamics'):
        hook = scene.get('native_task_bundle', {}).get(name)
        if hook:
            hook['path'] = str(local_path(root, hook['path']))
    return deepcopy(manifest['tasks'][0]), scene, task, initial


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent-release', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--configuration', choices=CONFIGURATIONS, required=True)
    parser.add_argument('--asset-index', type=Path, required=True,
                        help='JSON mapping decimal seed strings to frozen asset manifest paths')
    args = parser.parse_args()
    manifest = prepare_release(args.parent_release, args.output, configuration_id=args.configuration,
        assets_by_seed={int(k): v for k, v in read_json(args.asset_index).items()})
    print(manifest['release_id'])


if __name__ == '__main__':
    main()
