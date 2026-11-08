"""Strict laundry finite-case releases, preserving original native cloth logic."""
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
import shutil
import tempfile
import types

from . import RESET_SCHEMA, SCHEMA, SEEDS, VERSION, PARAMETERIZED_VERSION, selection
from . import common
from .common import (canonical_hash, checked_file, checked_inventory, file_hash,
                     local_path, parent_inputs, read_json, text_hash, verify_provenance, write_json)
from .author import load_asset

OLD_DESCRIPTION = 'Frozen canonical reset; seed is provenance only.'
NEW_DESCRIPTION = 'Frozen finite initial XY placement and render-only fabric texture case; cloth shape, material, target and robot parameters are unchanged.'
PARAMETERIZED_DESCRIPTION = 'Frozen finite initial XY/yaw placement, fabric appearance and local cloth mass/density/friction case; rest mesh, thickness, solver, robot and numerical task thresholds are unchanged.'
EVALUATORS = ('sequence_evaluator.py', 'rules.py', 'native_observer.py')


def evaluator_sources(parent, *, generator_version=None):
    sources = {name: (Path(parent)/'evaluator'/name).read_text() for name in EVALUATORS}
    if sources['rules.py'].count(OLD_DESCRIPTION) != 1 or NEW_DESCRIPTION in sources['rules.py']:
        raise ValueError('Parent rules do not have the one supported provenance-only description')
    description = PARAMETERIZED_DESCRIPTION if generator_version == PARAMETERIZED_VERSION else NEW_DESCRIPTION
    sources['rules.py'] = sources['rules.py'].replace(OLD_DESCRIPTION, description)
    return sources


def _rule_module(sources):
    module = types.ModuleType('_laundry_seed_rules_projection')
    exec(compile(sources['rules.py'], '<pinned-laundry-rules-description-adapter>', 'exec'), module.__dict__)
    return module


def compose_case(parent, asset_path, seed, *, assets=True, parent_values=None, sources=None):
    parent = Path(parent).resolve()
    values = parent_values or parent_inputs(parent, assets=assets)
    original, original_scene, original_task, original_initial, _ = values
    selection(original_task['id'], seed, original['configuration_id'])
    asset, plan, metadata, inventory = load_asset(parent, asset_path, seed, assets=assets, parent_values=values)
    parameterized = asset['generator_version'] == PARAMETERIZED_VERSION
    sources = evaluator_sources(parent, generator_version=asset['generator_version']) if sources is None else sources
    scene, task, initial = deepcopy(original_scene), deepcopy(original_task), deepcopy(original_initial)
    positions = {row['id']: deepcopy(row['initial_vertices_m']) for row in metadata['cloths']}
    for row in scene['cloths']:
        row['initial_vertices_m'] = deepcopy(positions[row['id']])
        row['visual_profile'] = deepcopy(asset['visual_profiles'][row['id']])
        if parameterized and seed:
            current = next(value for value in metadata['cloths'] if value['id'] == row['id'])
            row['mass_kg'], row['friction'] = current['mass_kg'], current['friction']
    initial['cloths_vertices_m'] = deepcopy(positions)
    task['initial_state'] = deepcopy(initial)
    task['asset_metadata'] = deepcopy(metadata)
    task['evaluation']['metadata'] = deepcopy(metadata)
    task['episode_seed'] = seed
    rules = _rule_module(sources)
    task['public_rules'] = rules.task_public_rules(task)
    task['evaluation']['public_rules_sha256'] = rules.digest(task['public_rules'])
    task = rules.bind_task_rules(task)
    expected_referee = original_task['public_rules']['referee']
    if parameterized:
        # Existing public nominal-mass fields must match the real material.
        # Recompute the allowed reference using ONLY each original cloth mass;
        # every numerical task criterion and all other properties stay exact.
        nominal = deepcopy(original_task)
        actual_masses = {row['id']: row['mass_kg'] for row in metadata['cloths']}
        for owner in (nominal['asset_metadata'], nominal['evaluation']['metadata']):
            for row in owner['cloths']:
                row['mass_kg'] = actual_masses[row['id']]
        expected_referee = rules.task_public_rules(nominal)['referee']
    if (task['public_rules']['referee'] != expected_referee
            or task['public_goal'] != original_task['public_goal']
            or task['evaluation']['thresholds'] != original_task['evaluation']['thresholds']):
        raise ValueError('Seed derivation changed the original public goal or numerical referee')
    pin = {'path': str(Path(asset_path).resolve()), 'sha256': file_hash(asset_path)}
    # Content identity excludes seed labels and filesystem paths; initial geometry,
    # original material/target identity and actual texture content determine it.
    physical = {'cloths_vertices_m': positions,
                'protected_parent_metadata': canonical_hash(original_task['asset_metadata']),
                'configuration_id': original['configuration_id'],
                'robot': original_scene['robot'], 'public_goal': original_task['public_goal']}
    if parameterized:
        physical['parameterized_physics'] = deepcopy(asset['parameterized_physics'])
    visual_content = {row['id']: read_json(asset['visual_profiles'][row['id']]['path'])['texture_sha256']
                      for row in scene['cloths']}
    parent_pin = {'path': str(parent), 'manifest_sha256': file_hash(parent/'release_manifest.json')}
    case = {'schema': RESET_SCHEMA, 'family': 'laundry_room', 'task_id': task['id'],
            'configuration_id': original['configuration_id'], 'seed': seed,
            'generator_version': asset['generator_version'], 'parent_release': parent_pin,
            'parent_release_sha256': parent_pin['manifest_sha256'], 'asset_reference': pin,
            'case_fingerprint': canonical_hash({'physical': physical, 'appearance': visual_content}),
            'initial_geometry_sha256': canonical_hash(positions), 'plan': deepcopy(plan),
            'rest_geometry_and_physics_unchanged': not (parameterized and seed != 0), 'target_coordinates_unchanged': True,
            'runtime_cloth_state_writes': False, 'reset_only': True}
    if parameterized:
        case['parameterized_physics'] = deepcopy(asset['parameterized_physics'])
    case['case_sha256'] = canonical_hash(case)
    task['reset_case'] = deepcopy(case)
    scene['laundry_reset_case'] = {k: deepcopy(case[k]) for k in
                                  ('schema', 'family', 'seed', 'case_sha256', 'case_fingerprint')}
    scene['laundry_task_provenance']['seed_asset'] = pin
    # Compare all unapproved fields, including the original joint reset, exactly.
    reverted = deepcopy(scene)
    reverted.pop('laundry_reset_case')
    reverted['laundry_task_provenance'].pop('seed_asset')
    for current, old in zip(reverted['cloths'], original_scene['cloths'], strict=True):
        current['initial_vertices_m'] = deepcopy(old['initial_vertices_m'])
        current['visual_profile'] = deepcopy(old['visual_profile'])
        if parameterized:
            for key in ('mass_kg', 'friction'):
                if key in old:
                    current[key] = deepcopy(old[key])
                else:
                    current.pop(key, None)
    if reverted != original_scene:
        raise ValueError('Scene changes exceed the declared cloth reset/material/appearance recipe')
    if ({k: v for k, v in initial.items() if k != 'cloths_vertices_m'} !=
            {k: v for k, v in original_initial.items() if k != 'cloths_vertices_m'}):
        raise ValueError('Robot reset changed')
    if seed == 0 and (metadata != original_task['asset_metadata'] or initial != original_initial
                     or scene['cloths'] != original_scene['cloths']):
        raise ValueError('Seed zero changed canonical physical or appearance inputs')
    return scene, task, initial, case, sources, inventory


def prepare_release(parent_release, output, *, assets_by_seed, configuration_id):
    parent, output = Path(parent_release).resolve(), Path(output).resolve()
    original = parent_inputs(parent)
    task_id = original[0]['task_id']; selection(task_id, 0, configuration_id)
    if (original[0]['configuration_id'] != configuration_id
            or set(assets_by_seed) != set(SEEDS) or any(type(s) is not int for s in assets_by_seed)
            or output.exists() or not output.is_relative_to(common.RELEASE_ROOT)):
        raise ValueError('Fresh shared release, original installation and all ten integer seeds required')
    versions = {read_json(path)['generator_version'] for path in assets_by_seed.values()}
    if len(versions) != 1:
        raise ValueError('One frozen generator version must cover every seed')
    generator_version = versions.pop()
    parameterized = generator_version == PARAMETERIZED_VERSION
    provenance = common.source_provenance(parameterized=True) if parameterized else common.source_provenance()
    sources = evaluator_sources(parent, generator_version=generator_version)
    description = PARAMETERIZED_DESCRIPTION if parameterized else NEW_DESCRIPTION
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix='.'+output.name+'.preparing-', dir=output.parent))
    cases, inventory = {}, {}
    for seed in SEEDS:
        values = compose_case(parent, assets_by_seed[seed], seed, parent_values=original, sources=sources)
        scene, task, initial, case, _, deps = values
        inventory.update(deps)
        prefix = f'cases/seed_{seed:02d}'
        for name, value in zip(('scene', 'task', 'initial_state', 'case'), values[:4]):
            write_json(staging/prefix/(name+'.json'), value)
            if seed == 0 and name != 'case':
                write_json(staging/(name+'.json'), value)
        cases[str(seed)] = {**{n: prefix+'/'+n+'.json' for n in ('scene', 'task', 'initial_state', 'case')},
            'asset_manifest': str(Path(assets_by_seed[seed]).resolve()),
            'asset_manifest_sha256': file_hash(assets_by_seed[seed]),
            'case_sha256': case['case_sha256'], 'case_fingerprint': case['case_fingerprint']}
    for name, source in sources.items():
        path = staging/'evaluator'/name; path.parent.mkdir(parents=True, exist_ok=True); path.write_text(source)
    shutil.copyfile(parent/'physics_profile.json', staging/'physics_profile.json')
    write_json(staging/'rules.json', read_json(staging/'task.json')['public_rules'])
    write_json(staging/'asset_file_hashes.json', {p: {'sha256': h} for p, h in sorted(inventory.items())})
    write_json(staging/'source_provenance.json', {
        'schema': 'laundry_seed_sources.v1', **provenance,
        'evaluator_sources_sha256': {n: text_hash(t) for n, t in sources.items()},
        'rules_description_derivation': {'schema': 'laundry_seed_rule_description.v1',
            'parent_rules_sha256': file_hash(parent/'evaluator/rules.py'),
            'old_literal': OLD_DESCRIPTION, 'new_literal': description,
            'replacement_count': 1, 'numerical_rules_changed': False},
        'runtime_cloth_state_writes': False, 'original_inputs_modified': False})
    files = {str(p.relative_to(staging)): {'sha256': file_hash(p)} for p in sorted(staging.rglob('*')) if p.is_file()}
    entry = deepcopy(original[0]['tasks'][0]); entry['task_kind'] = 'single_agent'
    for name in ('scene', 'task', 'initial_state'):
        entry[name] = name+'.json'; entry[name+'_sha256'] = files[name+'.json']['sha256']
    manifest = {'schema': SCHEMA, 'schema_version': VERSION, 'family': 'laundry_room',
        'task_id': task_id, 'task_kind': 'single_agent', 'configuration_id': configuration_id,
        'release_id': task_id+'_seeded_'+output.name, 'prepared_at_utc': datetime.now(timezone.utc).isoformat(),
        'status': 'prepared_unvalidated', 'physics_accepted': False, 'task_policy_success_claimed': False,
        'original_release_modified': False, 'base_release': {'path': str(parent),
            'manifest_sha256': file_hash(parent/'release_manifest.json'), 'schema': original[0]['schema']},
        'capabilities': deepcopy(original[0]['capabilities']), 'seed_values': list(SEEDS), 'cases': cases,
        'tasks': [entry], 'files': files, 'files_sha256': {n: r['sha256'] for n, r in files.items()},
        'asset_files_sha256': inventory, 'physics_profile': 'physics_profile.json',
        'physics_profile_sha256': files['physics_profile.json']['sha256'],
        'qualification_scope': 'Frozen XY and albedo inputs only; no workspace, native, interface or manipulation qualification inherited.'}
    if parameterized:
        manifest['generator_version'] = generator_version
        manifest['qualification_scope'] = 'Frozen XY/yaw, fabric appearance and actual local mass/density/friction inputs; workspace, native, interface and manipulation require separate evidence.'
    write_json(staging/'release_manifest.json', manifest)
    # A failed staging directory is retained; publication never overwrites inputs.
    validate_release(staging, manifest)
    staging.rename(output)
    return manifest


def validate_release(root, manifest, *, assets=True):
    from . import validation_cache
    root = Path(root).resolve()
    manifest = deepcopy(manifest)
    manifest_sha = validation_cache.manifest_input(root, manifest)
    key = (str(root), manifest_sha, bool(assets))
    if validation_cache.get(key) is not None:
        return deepcopy(manifest)
    try:
        proof = validation_cache.collect(root, manifest, manifest_sha)
        _validate_uncached(root, manifest, assets=assets)
        proof.verify()
    except BaseException:
        validation_cache.discard(key)
        raise
    validation_cache.store(key, proof)
    return deepcopy(manifest)


def _validate_uncached(root, manifest, *, assets=True):
    from bench.isaacsim.release import _frozen_evaluator_module
    root = Path(root).resolve()
    if (manifest.get('schema') != SCHEMA or manifest.get('schema_version') != VERSION
            or manifest.get('family') != 'laundry_room' or manifest.get('task_kind') != 'single_agent'
            or manifest.get('status') != 'prepared_unvalidated' or manifest.get('physics_accepted') is not False
            or manifest.get('task_policy_success_claimed') is not False
            or manifest.get('original_release_modified') is not False):
        raise ValueError('Invalid laundry randomized release or inherited qualification')
    task_id, configuration = manifest.get('task_id'), manifest.get('configuration_id')
    selection(task_id, 0, configuration)
    if manifest.get('seed_values') != list(SEEDS) or set(manifest.get('cases', {})) != {str(s) for s in SEEDS}:
        raise ValueError('Exactly ten finite laundry cases are required')
    base = manifest.get('base_release', {})
    parent = Path(base.get('path', '')).resolve()
    if parent == root or root.is_relative_to(parent) or parent.is_relative_to(root):
        raise ValueError('Derived release must be independent of its original')
    checked_file(parent/'release_manifest.json', base.get('manifest_sha256'))
    original = parent_inputs(parent, assets=assets)
    if (original[0]['task_id'] != task_id or original[0]['configuration_id'] != configuration
            or base.get('schema') != original[0]['schema']
            or manifest.get('capabilities') != original[0]['capabilities']):
        raise ValueError('Derived release changed its original task/robot capability')
    required = {'scene.json', 'task.json', 'initial_state.json', 'rules.json', 'physics_profile.json',
                'asset_file_hashes.json', 'source_provenance.json',
                *('evaluator/'+n for n in EVALUATORS),
                *(f'cases/seed_{s:02d}/{n}.json' for s in SEEDS for n in ('scene', 'task', 'initial_state', 'case'))}
    files = manifest.get('files', {})
    if set(files) != required or manifest.get('files_sha256') != {n: r.get('sha256') for n, r in files.items()}:
        raise ValueError('Incomplete frozen laundry files')
    for name, record in files.items():
        if set(record) != {'sha256'}:
            raise ValueError('Unexpected frozen file descriptor')
        checked_file(local_path(root, name), record['sha256'])
    actual = {str(p.relative_to(root)) for p in root.rglob('*') if p.is_file()
              and not ('__pycache__' in p.parts and p.suffix == '.pyc')}
    if actual != required | {'release_manifest.json'}:
        raise ValueError('Undeclared frozen laundry release files')
    generator_version = manifest.get('generator_version')
    if generator_version not in (None, PARAMETERIZED_VERSION):
        raise ValueError('Unknown explicit laundry release generator')
    sources = evaluator_sources(parent, generator_version=generator_version)
    for name, source in sources.items():
        if (root/'evaluator'/name).read_text() != source:
            raise ValueError('Frozen evaluator differs beyond its one descriptive literal')
    provenance = read_json(root/'source_provenance.json')
    verify_provenance(provenance)
    expected_derivation = {'schema': 'laundry_seed_rule_description.v1',
        'parent_rules_sha256': file_hash(parent/'evaluator/rules.py'), 'old_literal': OLD_DESCRIPTION,
        'new_literal': PARAMETERIZED_DESCRIPTION if generator_version == PARAMETERIZED_VERSION else NEW_DESCRIPTION,
        'replacement_count': 1, 'numerical_rules_changed': False}
    if (provenance.get('schema') != 'laundry_seed_sources.v1'
            or provenance.get('evaluator_sources_sha256') != {n: text_hash(t) for n, t in sources.items()}
            or provenance.get('rules_description_derivation') != expected_derivation
            or provenance.get('runtime_cloth_state_writes') is not False
            or provenance.get('original_inputs_modified') is not False):
        raise ValueError('Invalid frozen laundry author/evaluator derivation')
    if (root/'physics_profile.json').read_bytes() != (parent/'physics_profile.json').read_bytes():
        raise ValueError('Original physics profile bytes changed')
    if (manifest.get('physics_profile') != 'physics_profile.json'
            or manifest.get('physics_profile_sha256') != files['physics_profile.json']['sha256']):
        raise ValueError('Physics profile pin differs')
    entry = deepcopy(original[0]['tasks'][0]); entry['task_kind'] = 'single_agent'
    for name in ('scene', 'task', 'initial_state'):
        entry[name] = name+'.json'; entry[name+'_sha256'] = files[name+'.json']['sha256']
    if manifest.get('tasks') != [entry]:
        raise ValueError('Public task entry differs from its original identity')
    inventory = read_json(root/'asset_file_hashes.json')
    if any(not isinstance(row, dict) or set(row) != {'sha256'} for row in inventory.values()):
        raise ValueError('Invalid dependency descriptor')
    if manifest.get('asset_files_sha256') != {p: row['sha256'] for p, row in inventory.items()}:
        raise ValueError('Dependency manifest differs')
    checked_inventory(manifest['asset_files_sha256'], verify=assets)
    expected_inventory, fingerprints = {}, set()
    evaluator = _frozen_evaluator_module(root).SequenceEvaluator
    for seed in SEEDS:
        record = manifest['cases'][str(seed)]
        expected_paths = {n: f'cases/seed_{seed:02d}/{n}.json' for n in ('scene', 'task', 'initial_state', 'case')}
        if set(record) != {*expected_paths, 'asset_manifest', 'asset_manifest_sha256', 'case_sha256', 'case_fingerprint'}:
            raise ValueError('Unexpected laundry case descriptor')
        if any(record.get(n) != p for n, p in expected_paths.items()):
            raise ValueError('Case path does not match its seed')
        checked_file(record['asset_manifest'], record['asset_manifest_sha256'])
        values = compose_case(parent, record['asset_manifest'], seed, assets=assets,
                              parent_values=original, sources=sources)
        for name, expected in zip(expected_paths, values[:4]):
            if read_json(local_path(root, record[name])) != expected:
                raise ValueError('Frozen laundry '+name+' differs from exact parent derivation')
        case = values[3]
        if (case['generator_version'] == PARAMETERIZED_VERSION) != (generator_version == PARAMETERIZED_VERSION):
            raise ValueError('Release and case parameterized generator identities differ')
        if any(record[k] != case[k] for k in ('case_sha256', 'case_fingerprint')):
            raise ValueError('Frozen laundry case identity differs')
        expected_inventory.update(values[5]); fingerprints.add(case['case_fingerprint'])
        evaluator(values[1])
        if seed == 0:
            for name, expected in zip(('scene', 'task', 'initial_state'), values[:3]):
                if read_json(root/(name+'.json')) != expected:
                    raise ValueError('Discovery root must be the exact seed-zero case')
    if expected_inventory != manifest['asset_files_sha256'] or len(fingerprints) != 10:
        raise ValueError('Case dependency inventory or unique reset contents differ')
    if read_json(root/'rules.json') != read_json(root/'task.json')['public_rules']:
        raise ValueError('Root rules differ from canonical seed zero')
    return manifest


def task_inputs(root, task_id, manifest, *, seed=0, phase='nominal'):
    selection(task_id, seed, manifest['configuration_id'])
    if phase != 'nominal' or task_id != manifest['task_id']:
        raise ValueError('Only finite ordinary laundry reset cases are supported')
    root = Path(root).resolve(); record = manifest['cases'][str(seed)]
    values = []
    for name in ('scene', 'task', 'initial_state'):
        path = local_path(root, record[name])
        checked_file(path, manifest['files_sha256'][record[name]])
        values.append(read_json(path))
    return deepcopy(manifest['tasks'][0]), *values
