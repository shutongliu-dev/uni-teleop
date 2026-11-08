"""Frozen per-task physical resets, with original robots and explicit lineage."""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import importlib
from pathlib import Path
import shutil
import tempfile

from bench.paths import DATA_ROOT, OUTPUT_ROOT
from bench.protocol.living_reset import StaticGeometry
from . import (TASK_FAMILIES, PLACEMENT_TASK_FAMILIES, PLACEMENT_SCOPE,
               SEEDS, SCHEMA, VERSION, selection)
from .common import (canonical_hash, checked_file, checked_poses, file_hash,
                     local_path, read_json, text_hash, vector, write_json)
from .physical_inputs import check_actor_inventory, check_canonical_physics

ASSET_SCHEMA = 'workroom_seed_asset.v1'
PARENT_SCHEMAS = {
    'study': {'study_task_release.v1', 'workroom_robot_configuration_release.v1'},
    'kitchen': {'kitchen_manipulation_task_release.v1'},
    'laboratory': {'laboratory_manipulation_task_release.v1', 'workroom_robot_configuration_release.v1'},
    'mechanical_assembly': {'mechanical_assembly_task_release.v1', 'workroom_robot_configuration_release.v1',
                            'mechanical_fastener_task_release.v1', 'mechanical_nist_task_release.v1',
                            'mechanical_cooperative_task_release.v1'},
}


def adapter(task_id):
    name = 'placement_variants' if task_id in PLACEMENT_TASK_FAMILIES else 'seed_variants'
    return importlib.import_module(f'scenes.workrooms.{TASK_FAMILIES[task_id]}.{name}')


def _asset(path, task_id, seed, *, assets):
    path = Path(path).resolve()
    if not path.is_relative_to(f'{DATA_ROOT}/assets'):
        raise ValueError('Reusable reset assets belong in shared assets')
    value = read_json(path)
    if (value.get('schema') != ASSET_SCHEMA or value.get('family') != TASK_FAMILIES[task_id]
            or value.get('task_id') != task_id or type(value.get('seed')) is not int
            or value['seed'] != seed or not value.get('generator_version')):
        raise ValueError('Frozen asset has another task/seed or missing author identity')
    from scenes.workrooms.parameterized.spec import VERSION as PARAMETERIZED_VERSION, validate_parameters
    if value.get('generator_version') == PARAMETERIZED_VERSION:
        validate_parameters(value['parameter_instance'])
        if value.get('variant_scope') != PARAMETERIZED_VERSION:
            raise ValueError('Parameterized asset scope differs')
    elif task_id in PLACEMENT_TASK_FAMILIES and value.get('variant_scope') != PLACEMENT_SCOPE:
        raise ValueError('Position/texture cases must explicitly preserve physical geometry')
    for name in ('parent_references', 'dependencies_sha256'):
        if not isinstance(value.get(name), dict) or not value[name]:
            raise ValueError('Asset must bind parent and actual dependencies')
        for source, expected in value[name].items():
            if not Path(source).is_absolute():
                raise ValueError('Asset dependencies must be absolute')
            if assets:
                checked_file(source, expected)
    return value


def _parent_configuration(scene, task_id=None):
    if task_id == 'cooperative_insertion':
        if (scene.get('robot_configuration') is not None
                or scene.get('robot', {}).get('hand_id') != 'dual_fr3_sharpa_w01'):
            raise ValueError('Cooperative insertion must retain its original fixed dual robot')
        return 'dual_same_side'
    return scene.get('robot_configuration', {}).get('id', 'single_arm')


def _features(values):
    if not isinstance(values, list) or not values or len({x['label'] for x in values}) != len(values):
        raise ValueError('Unique explicit interaction features required')
    for row in values:
        if (row.get('type') not in ('actual_surface', 'goal', 'pouring_edge')
                or row.get('reference_state') not in ('initial', 'opened')
                or type(row.get('required')) is not bool or not row.get('operation_group')
                or not isinstance(row.get('prim_path'), str) or not row['prim_path'].startswith('/World/')):
            raise ValueError('Invalid frozen interaction feature')
        vector(row['local_position_m'], 3)
        direction = vector(row['approach_direction_local'], 3)
        distance, = vector([row['approach_distance_m']], 1)
        if abs(sum(v*v for v in direction)-1.) > 1e-6 or not 0 < distance <= .15:
            raise ValueError('Feature requires a unit approach and bounded distance')


def _preserved(parent_scene, parent_initial, scene, initial, seed):
    if initial != parent_initial:
        raise ValueError('Randomization must preserve the original robot reset')
    for key in ('robot', 'workroom', 'isaacsim', 'physics', 'controller', 'policy',
                'policies', 'robot_configuration', 'cameras', 'observation_profile'):
        if scene.get(key) != parent_scene.get(key):
            raise ValueError('Randomization changed protected scene input: '+key)
    if 'scenario' in parent_scene:
        expected = deepcopy(parent_scene['scenario']); expected['seed'] = seed
        if scene.get('scenario') != expected:
            raise ValueError('Scenario changes exceed the declared seed')


def _compose(parent, asset_path, task_id, seed, configuration_id):
    selection(task_id, seed, configuration_id)
    values = adapter(task_id).compose_case(parent, asset_path, configuration_id)
    required = {'scene', 'task', 'initial_state', 'evaluator_sources', 'actual_body_poses_m',
                'interaction_targets', 'static_geometry', 'asset_files_sha256'}
    if not isinstance(values, dict) or not required <= values.keys():
        raise ValueError('Family adapter omitted required physical inputs')
    scene, task, initial = [deepcopy(values[k]) for k in ('scene', 'task', 'initial_state')]
    original_scene, original_task, original_initial = [read_json(parent/(n+'.json'))
                                                     for n in ('scene', 'task', 'initial_state')]
    if task.get('id') != task_id or task.get('schema_version') != original_task.get('schema_version'):
        raise ValueError('Randomization changed the task identity/schema')
    if 'scenario' in scene:
        scene['scenario']['seed'] = seed
    for key in ('seed', 'episode_seed'):
        if key in task:
            task[key] = seed
    if 'parameterized_physics' in values and 'episode_seed' in task.get('evaluation', {}):
        # The shared physical asset is authored from canonical seed zero. Bind
        # the evaluator's episode identity alongside the task identity; its
        # frozen constructor still checks every rule and geometry field.
        task['evaluation']['episode_seed'] = seed
    _preserved(original_scene, original_initial, scene, initial, seed)
    poses = deepcopy(values['actual_body_poses_m']); checked_poses(poses)
    check_actor_inventory(original_scene, scene, poses)
    if seed == 0:
        check_canonical_physics(original_scene, scene)
    features = deepcopy(values['interaction_targets']); _features(features)
    geometry = StaticGeometry.model_validate(values['static_geometry']).model_dump(mode='json')
    if set(poses) != {row['object_id'] for row in geometry['bodies']}:
        raise ValueError('Public nominal bodies differ from private actor inventory')
    sources = values['evaluator_sources']
    if (not isinstance(sources, dict) or 'sequence_evaluator.py' not in sources
            or any(not isinstance(text, str) or Path(n).is_absolute() or '..' in Path(n).parts
                   or not n.endswith('.py') for n, text in sources.items())):
        raise ValueError('Complete relative frozen evaluator Python sources required')
    dependencies = values['asset_files_sha256']
    if not isinstance(dependencies, dict) or not dependencies:
        raise ValueError('Actual geometry/material dependency hashes required')
    record = read_json(asset_path)
    dependencies = {**dependencies, **record['parent_references'], **record['dependencies_sha256'],
                    str(Path(asset_path).resolve()): file_hash(asset_path)}
    for path, expected in dependencies.items():
        if not Path(path).is_absolute() or not isinstance(expected, str) or len(expected) != 64:
            raise ValueError('Absolute hashed asset inventory required')
    physical = {'actual_body_poses_m': poses, 'static_geometry': geometry}
    case = {'schema': 'workroom_reset_case.v1', 'family': TASK_FAMILIES[task_id],
        'task_id': task_id, 'configuration_id': configuration_id, 'seed': seed,
        'parent_release_sha256': file_hash(parent/'release_manifest.json'),
        'asset_reference': {'path': str(Path(asset_path).resolve()), 'sha256': file_hash(asset_path)},
        'physical_geometry_fingerprint': canonical_hash(physical),
        'case_fingerprint': canonical_hash({**physical, 'asset_content_hashes': sorted(set(dependencies.values())),
                                            'configuration_id': configuration_id}),
        'actual_body_poses_m': poses, 'interaction_targets': features, 'static_geometry': geometry,
        'reset_only': True, 'runtime_object_writes': False}
    if 'fixed_support_paths' in values:
        case['fixed_support_paths'] = deepcopy(values['fixed_support_paths'])
    if 'parameterized_physics' in values:
        case['parameterized_physics'] = deepcopy(values['parameterized_physics'])
    case['case_sha256'] = canonical_hash(case)
    task['reset_case'] = deepcopy(case)
    scene['workroom_reset_case'] = {k: case[k] for k in ('schema', 'family', 'seed', 'case_sha256', 'case_fingerprint')}
    for container in [scene.get('native_task_bundle', {}), *scene.get('task_layers', [])]:
        for name in ('observer', 'reduced_dynamics'):
            hook = container.get(name)
            if hook:
                name_in_sources = str(Path(hook['path'])).split('evaluator/')[-1]
                if name_in_sources not in sources:
                    raise ValueError('Native hook is absent from frozen evaluator source')
                hook.update(path='evaluator/'+name_in_sources, configuration=deepcopy(task),
                            files={n: text_hash(t) for n, t in sources.items()},
                            sha256=text_hash(sources[name_in_sources]))
    return scene, task, initial, case, sources, dependencies


def prepare_release(parent_release, output, *, configuration_id, assets_by_seed):
    from bench.isaacsim.release import validate
    from scripts.maintenance.freeze_project_sources import verify_if_snapshot
    source_root = Path(__file__).resolve().parents[3]
    source = verify_if_snapshot(source_root, required_modules=(__name__,), entrypoint_file=Path(__file__))
    if not source or not source.get('runtime_imports_verified'):
        raise ValueError('Physical reset publication requires complete frozen source imports')
    parent, output = Path(parent_release).resolve(), Path(output).resolve()
    original = validate(parent)
    task_id = original['task_id']; selection(task_id, 0, configuration_id)
    family = TASK_FAMILIES[task_id]
    if (original.get('schema') not in PARENT_SCHEMAS[family]
            or _parent_configuration(read_json(parent/'scene.json'), task_id) != configuration_id):
        raise ValueError('Wrong original task family/installation')
    if (output.exists() or not output.is_relative_to(OUTPUT_ROOT)
            or set(assets_by_seed) != set(SEEDS) or any(type(k) is not int for k in assets_by_seed)):
        raise ValueError('Fresh shared release directory and ten integer seed assets required')
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix='.'+output.name+'.preparing-', dir=output.parent))
    try:
        inventory = read_json(parent/'asset_file_hashes.json')
        inventory[str(parent/'release_manifest.json')] = {'sha256': file_hash(parent/'release_manifest.json')}
        cases, frozen_sources = {}, None
        for seed in SEEDS:
            path = Path(assets_by_seed[seed]).resolve()
            _asset(path, task_id, seed, assets=True)
            scene, task, initial, case, sources, deps = _compose(parent, path, task_id, seed, configuration_id)
            if frozen_sources is not None and sources != frozen_sources:
                raise ValueError('All seeds must use the same frozen evaluator implementation')
            frozen_sources = sources
            inventory.update({p: {'sha256': h} for p, h in deps.items()})
            prefix = f'cases/seed_{seed:02d}'
            values = {'scene': scene, 'task': task, 'initial_state': initial, 'case': case}
            for name, value in values.items():
                write_json(staging/prefix/(name+'.json'), value)
                if seed == 0 and name != 'case':
                    write_json(staging/(name+'.json'), value)
            cases[str(seed)] = {**{n: prefix+'/'+n+'.json' for n in values},
                'case_sha256': case['case_sha256'], 'case_fingerprint': case['case_fingerprint'],
                'asset_manifest': str(path), 'asset_manifest_sha256': file_hash(path)}
        for name, text in frozen_sources.items():
            path = staging/'evaluator'/name; path.parent.mkdir(parents=True, exist_ok=True); path.write_text(text)
        shutil.copy2(parent/'physics_profile.json', staging/'physics_profile.json')
        write_json(staging/'rules.json', read_json(staging/'task.json')['public_rules'])
        write_json(staging/'asset_file_hashes.json', inventory)
        write_json(staging/'source_provenance.json', {'schema': 'workroom_seed_sources.v1',
            'family_adapter': adapter(task_id).__name__, 'family_adapter_sha256': file_hash(adapter(task_id).__file__),
            'source_snapshot': source, 'source_manifest_sha256': file_hash(source_root/'source_snapshot_manifest.json'),
            'evaluator_sources_sha256': {n: text_hash(t) for n, t in frozen_sources.items()},
            'original_releases_modified': False, 'original_assets_modified': False,
            'shared_robot_controller_unchanged': True, 'runtime_object_writes': False})
        files = {str(p.relative_to(staging)): {'sha256': file_hash(p)} for p in sorted(staging.rglob('*')) if p.is_file()}
        entry = deepcopy(original['tasks'][0]); entry['task_kind'] = 'single_agent'
        for name in ('scene', 'task', 'initial_state'):
            entry[name] = name+'.json'; entry[name+'_sha256'] = files[name+'.json']['sha256']
        manifest = {'schema': SCHEMA, 'schema_version': VERSION, 'family': family, 'task_id': task_id,
            'task_kind': 'single_agent', 'configuration_id': configuration_id, 'release_id': task_id+'_seeded_'+output.name,
            'prepared_at_utc': datetime.now(timezone.utc).isoformat(), 'status': 'prepared_unvalidated',
            'physics_accepted': False, 'task_policy_success_claimed': False, 'original_release_modified': False,
            'base_release': {'path': str(parent), 'manifest_sha256': file_hash(parent/'release_manifest.json'), 'schema': original['schema']},
            'seed_values': list(SEEDS), 'cases': cases, 'tasks': [entry], 'files': files,
            'files_sha256': {n: r['sha256'] for n, r in files.items()},
            'asset_files_sha256': {n: r['sha256'] for n, r in inventory.items()},
            'physics_profile': 'physics_profile.json', 'physics_profile_sha256': files['physics_profile.json']['sha256'],
            'qualification_scope': 'Frozen physical inputs only; independent workspace, native and HTTP evidence required.'}
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
    task_id = manifest['task_id']; configuration_id = manifest['configuration_id']
    selection(task_id, 0, configuration_id)
    if (manifest.get('schema') != SCHEMA or manifest.get('schema_version') != VERSION
            or manifest.get('family') != TASK_FAMILIES[task_id] or manifest.get('seed_values') != list(SEEDS)
            or manifest.get('task_kind') != 'single_agent' or manifest.get('status') != 'prepared_unvalidated'
            or any(manifest.get(k) is not False for k in ('physics_accepted', 'task_policy_success_claimed', 'original_release_modified'))):
        raise ValueError('Malformed ordinary physical reset release')
    base = manifest['base_release']; parent = Path(base['path']).resolve()
    if parent == root or base['schema'] not in PARENT_SCHEMAS[TASK_FAMILIES[task_id]]:
        raise ValueError('Randomized release requires an independent original parent')
    checked_file(parent/'release_manifest.json', base['manifest_sha256'])
    original = validate(parent, assets=assets)
    if (original['schema'] != base['schema'] or original['task_id'] != task_id
            or _parent_configuration(read_json(parent/'scene.json'), task_id) != configuration_id):
        raise ValueError('Original task/installation changed')
    files = manifest['files']
    if manifest['files_sha256'] != {n: r['sha256'] for n, r in files.items()}:
        raise ValueError('Frozen file indexes disagree')
    actual = {str(p.relative_to(root)) for p in root.rglob('*') if p.is_file()
              and not ('__pycache__' in p.parts and p.suffix == '.pyc')}
    if actual != set(files) | {'release_manifest.json'}:
        raise ValueError('Frozen release contains missing or undeclared files')
    for name, row in files.items():
        checked_file(local_path(root, name), row['sha256'])
    entries = manifest.get('tasks', [])
    if len(entries) != 1 or entries[0]['id'] != task_id or entries[0].get('task_kind') != 'single_agent':
        raise ValueError('Exactly one ordinary task entry required')
    for name in ('scene', 'task', 'initial_state'):
        if entries[0].get(name) != name+'.json' or entries[0].get(name+'_sha256') != files[name+'.json']['sha256']:
            raise ValueError('Canonical entry differs from frozen files')
    inventory = read_json(root/'asset_file_hashes.json')
    if manifest['asset_files_sha256'] != {n: r['sha256'] for n, r in inventory.items()}:
        raise ValueError('External input indexes disagree')
    if assets:
        for name, row in inventory.items():
            checked_file(name, row['sha256'])
    if (manifest.get('physics_profile') != 'physics_profile.json'
            or manifest.get('physics_profile_sha256') != file_hash(parent/'physics_profile.json')
            or file_hash(root/'physics_profile.json') != manifest['physics_profile_sha256']):
        raise ValueError('Shared physics profile changed')
    cases = manifest['cases']
    if set(cases) != {str(s) for s in SEEDS}:
        raise ValueError('Exactly ten frozen cases required')
    sources = {str(p.relative_to(root/'evaluator')): p.read_text() for p in (root/'evaluator').rglob('*.py')}
    provenance = read_json(root/'source_provenance.json')
    author_module = adapter(task_id).__name__
    if (provenance.get('schema') != 'workroom_seed_sources.v1'
            or provenance.get('family_adapter') != author_module
            or provenance.get('evaluator_sources_sha256') != {n: text_hash(t) for n, t in sources.items()}):
        raise ValueError('Frozen evaluator/author provenance differs')
    if (not isinstance(provenance.get('source_snapshot'), dict)
            or provenance['source_snapshot'].get('runtime_imports_verified') is not True):
        raise ValueError('Frozen publication must bind its verified author snapshot')
    source_manifest = Path(provenance['source_snapshot']['snapshot_root'])/'source_snapshot_manifest.json'
    checked_file(source_manifest, provenance.get('source_manifest_sha256'))
    author_snapshot = read_json(source_manifest)
    author_relative = author_module.replace('.', '/')+'.py'
    if (author_snapshot.get('schema') != 'complete_benchmark_source_snapshot_v1'
            or author_snapshot.get('snapshot_root') != str(source_manifest.parent)
            or author_snapshot.get('files_sha256', {}).get(author_relative) != provenance.get('family_adapter_sha256')):
        raise ValueError('Original author module differs from its complete source snapshot')
    checked_file(source_manifest.parent/author_relative, provenance.get('family_adapter_sha256'))
    # The author and input consumer can run under different Python/Kit import
    # lifecycles. Bind the original author bytes above, then strictly reproduce
    # every scene, reset, evaluator and dependency with the current consumer.
    evaluator = _frozen_evaluator_module(root).SequenceEvaluator
    physical_fingerprints = set()
    for seed in SEEDS:
        record = cases[str(seed)]
        checked_file(record['asset_manifest'], record['asset_manifest_sha256'])
        _asset(record['asset_manifest'], task_id, seed, assets=assets)
        values = _compose(parent, record['asset_manifest'], task_id, seed, configuration_id)
        if values[4] != sources:
            raise ValueError('Frozen evaluator differs from the declared family adapter')
        if any(inventory.get(p, {}).get('sha256') != h for p, h in values[5].items()):
            raise ValueError('Case dependency omitted from release inventory')
        for name, value in zip(('scene', 'task', 'initial_state', 'case'), values[:4]):
            if read_json(local_path(root, record[name])) != value:
                raise ValueError('Case differs from its bound original inputs: '+name)
        if (record['case_sha256'], record['case_fingerprint']) != (values[3]['case_sha256'], values[3]['case_fingerprint']):
            raise ValueError('Frozen case identity differs')
        physical_fingerprints.add(values[3]['physical_geometry_fingerprint'])
        evaluator(values[1])
        if seed == 0:
            for name, value in zip(('scene', 'task', 'initial_state'), values[:3]):
                if read_json(root/(name+'.json')) != value:
                    raise ValueError('Discovery inputs do not match canonical seed zero')
    if len(physical_fingerprints) != 10:
        raise ValueError('Seeds must select ten distinct physical layouts/geometries')
    if read_json(root/'rules.json') != read_json(root/'task.json')['public_rules']:
        raise ValueError('Canonical public rules differ')
    return manifest


def task_inputs(root, task_id, manifest, *, seed=0, phase='nominal'):
    selection(task_id, seed, manifest['configuration_id'])
    if task_id != manifest['task_id'] or phase != 'nominal':
        raise ValueError('Only frozen ordinary initial cases are supported')
    root = Path(root).resolve(); record = manifest['cases'][str(seed)]
    scene, task, initial = [read_json(local_path(root, record[n])) for n in ('scene', 'task', 'initial_state')]
    for container in [scene.get('native_task_bundle', {}), *scene.get('task_layers', [])]:
        for name in ('observer', 'reduced_dynamics'):
            if name in container:
                container[name]['path'] = str(local_path(root, container[name]['path']))
    return deepcopy(manifest['tasks'][0]), scene, task, initial
