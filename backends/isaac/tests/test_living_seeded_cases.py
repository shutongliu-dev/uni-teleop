"""CPU reset contracts and source binding only; no native/robot qualification."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import types

import pytest

from bench.paths import load_json
from scenes.workrooms.catalog import task_definition
from scenes.workrooms.living_room.randomization import TASK_IDS, SEEDS, CONFIGURATIONS
from scenes.workrooms.living_room.randomization.cases import plan_case, validate_plan
from scenes.workrooms.living_room.randomization.common import canonical_hash, selection
from scenes.workrooms.living_room.randomization.public import project_reset_case
from scenes.workrooms.living_room.randomization.release import _source_files, _rules, _check_sources
from scenes.workrooms.living_room.tasks.remote_operation import rules as remote
from scenes.workrooms.living_room.tasks.remote_operation.evaluator import configuration as remote_configuration
from scenes.workrooms.living_room.tasks.storage_box import rules as storage
from scenes.workrooms.living_room.tasks.storage_box.evaluator import configuration as storage_configuration


def parent(task_id, configuration='single_arm'):
    roots = set()
    for profile in ('original', 'l_mount'):
        reference = task_definition('living_room', task_id, robot_profile=profile)[
            'robot_configurations'][configuration]['source_release']
        root = Path(reference['path'])
        manifest = load_json(root/'release_manifest.json')
        # Publishing a reset release or its L-mount robot revision changes the
        # catalog pointer, not the canonical configuration used to exercise
        # source adaptation here; both robot profiles share that parent.
        for wrapper in ('workroom_robot_revision_release.v1', 'living_room_randomized_release.v1'):
            if manifest['schema'] == wrapper:
                root = Path(manifest['base_release']['path'])
                manifest = load_json(root/'release_manifest.json')
        roots.add(root)
    root, = roots
    return root, load_json(root/'task.json')


@pytest.mark.parametrize('configuration', CONFIGURATIONS)
def test_parent_fixture_uses_original_configured_release_after_publication(tmp_path, monkeypatch, configuration):
    original, task = parent('remote_operation', configuration)
    derived = tmp_path/'derived'; derived.mkdir()
    (derived/'release_manifest.json').write_text(json.dumps({
        'schema': 'living_room_randomized_release.v1', 'base_release': {'path': str(original)}}))
    monkeypatch.setattr(sys.modules[__name__], 'task_definition', lambda *args, **kwargs: {
        'robot_configurations': {configuration: {'source_release': {'path': str(derived)}}}})
    assert parent('remote_operation', configuration) == (original, task)


@pytest.mark.parametrize('task_id', TASK_IDS)
def test_ten_deterministic_bounded_cases_keep_canonical_zero(task_id):
    _, task = parent(task_id)
    metadata = task['asset_metadata']
    before = deepcopy(metadata)
    plans = [plan_case(task_id, metadata, seed) for seed in SEEDS]
    assert metadata == before
    assert len({canonical_hash(p['group_transforms']) for p in plans}) == 10
    for p in plans:
        assert validate_plan(p, metadata) == p
        assert p['plan_sha256'] == canonical_hash({k: v for k, v in p.items() if k != 'plan_sha256'})
        assert p == plan_case(task_id, metadata, p['seed'])
    assert plans[0]['geometry_variant'] == plans[0]['appearance_variant'] == 'canonical'
    assert all(row['translation_xy_m'] == [0., 0.] and row['yaw_rad'] == 0.
               for row in plans[0]['group_transforms'].values())
    assert {p['geometry_variant'] for p in plans[1:]} == {'narrow', 'wide'}
    changed = deepcopy(plans[1]); next(iter(changed['group_transforms'].values()))['yaw_rad'] += .01
    with pytest.raises(ValueError, match='versioned'):
        validate_plan(changed, metadata)


@pytest.mark.parametrize('invalid', [-1, 10, True, 1.0, '1'])
def test_seed_domain_is_explicit_and_bounded(invalid):
    with pytest.raises(ValueError):
        selection('remote_operation', invalid)


@pytest.mark.parametrize('task_id', ['tower_building', 'tower_extraction', 'task0'])
def test_other_ordinary_and_competitive_tasks_are_excluded(task_id):
    with pytest.raises(ValueError):
        selection(task_id, 0)


def variant_remote(scale):
    _, task = parent('remote_operation')
    m = deepcopy(task['asset_metadata'])
    m['geometry_variant'] = {'schema': 'living_remote_geometry_variant.v1'}
    m['remote']['shell_size_xyz_m'][0] *= scale
    m['remote']['envelope_xyz_m'][0] *= scale
    m['remote']['local_bounds_m'] = [[-.0275*scale, -.08, -.0104], [.0275*scale, .08, .0135]]
    m['holder']['opening_xy_m'][0] *= scale
    m['holder']['liner_local_z_m'] = .007
    task['asset_metadata'] = m
    task['evaluation'] = remote.evaluation_configuration(asset_metadata=m)
    return task


@pytest.mark.parametrize('scale', [.94, 1., 1.06])
def test_remote_actual_envelope_is_bound_and_thresholds_unchanged(scale):
    task = variant_remote(scale)
    c, _ = remote_configuration(task)
    assert c['geometry']['remote_local_bounds_m'][1][0] == .0275*scale
    for key, value in remote.thresholds().items():
        assert c[key] == value
    other = deepcopy(task); other['evaluation']['geometry']['remote_local_bounds_m'][1][0] += .001
    with pytest.raises(ValueError, match='Geometry'):
        remote_configuration(other)


@pytest.mark.parametrize('scale', [.8, 1.2, float('nan'), float('inf')])
def test_remote_unknown_geometry_fails_closed(scale):
    with pytest.raises(ValueError):
        variant_remote(scale)


def test_default_remote_geometry_remains_exact():
    _, task = parent('remote_operation')
    assert remote.geometry_configuration(task['asset_metadata']) == task['evaluation']['geometry']
    assert remote_configuration(task)[0] == task['evaluation']


@pytest.mark.parametrize('configuration_id', CONFIGURATIONS)
def test_new_remote_sources_preserve_declared_hand_ownership(configuration_id):
    root, task = parent('remote_operation', configuration_id)
    sources = _source_files(root, 'remote_operation', configuration_id)
    _check_sources(root, 'remote_operation', configuration_id, sources)
    module = _rules(sources)
    expected = ['right_'] if configuration_id == 'single_arm' else ['left_', 'right_']
    assert module.evaluation_configuration(asset_metadata=task['asset_metadata'])['hand_link_prefixes'] == expected
    assert module.public_rules() == task['public_rules']
    package_name = '_test_living_variant_'+configuration_id
    package = types.ModuleType(package_name); package.__path__ = []
    module.__name__ = package_name+'.rules'
    evaluator = types.ModuleType(package_name+'.sequence_evaluator'); evaluator.__package__ = package_name
    try:
        sys.modules[package_name] = package; sys.modules[module.__name__] = module
        exec(compile(sources['sequence_evaluator.py'], '<test frozen evaluator>', 'exec'), evaluator.__dict__)
        assert evaluator.configuration(task)[0]['hand_link_prefixes'] == expected
    finally:
        sys.modules.pop(package_name, None); sys.modules.pop(module.__name__, None)


@pytest.mark.parametrize('task_id', TASK_IDS)
def test_rehashed_arbitrary_scorer_changes_are_not_accepted(task_id):
    root, _ = parent(task_id)
    sources = _source_files(root, task_id, 'single_arm')
    _check_sources(root, task_id, 'single_arm', sources)
    sources['sequence_evaluator.py'] += '\n# an unreviewed modification\n'
    with pytest.raises(ValueError, match='Unreviewed'):
        _check_sources(root, task_id, 'single_arm', sources)


@pytest.mark.parametrize('configuration_id', CONFIGURATIONS)
def test_runtime_selection_checks_new_release_configuration_id(tmp_path, monkeypatch, configuration_id):
    from bench.runtime import robot_configurations as bindings
    root = tmp_path/'release'; root.mkdir()
    robot = {'hand_id': 'sharpa_w01_right' if configuration_id == 'single_arm' else 'dual_fr3_sharpa_w01',
             'assembly_version': 'isaacsim_6_1_v2'}
    (root/'scene.json').write_text(json.dumps({'robot': robot}))
    manifest = {'schema': 'living_room_randomized_release.v1', 'configuration_id': configuration_id,
                'tasks': [{'id': 'remote_operation', 'scene': 'scene.json'}]}
    monkeypatch.setattr(bindings, 'validate', lambda *args, **kwargs: manifest)
    monkeypatch.setattr(bindings, 'digest', lambda *args: 'a'*64)
    binding = {'task_id': 'remote_operation', 'releases': {configuration_id: {'path': str(root), 'manifest_sha256': 'a'*64}},
               'robot_configurations': {configuration_id: {'robot_requirements': robot}}}
    assert bindings.checked_configuration_release(binding, configuration_id) == root
    manifest['configuration_id'] = 'dual_opposed' if configuration_id != 'dual_opposed' else 'single_arm'
    with pytest.raises(ValueError, match='installation'):
        bindings.checked_configuration_release(binding, configuration_id)


def test_storage_true_nominal_masses_are_public_without_changing_thresholds():
    _, task = parent('storage_box')
    original_rules = storage.public_rules()
    assert original_rules == task['public_rules']
    metadata = task['asset_metadata']
    metadata['geometry_variant'] = {'schema': 'living_storage_box_geometry_variant.v1'}
    metadata['masses_kg'] = {k: v*1.05 for k, v in metadata['masses_kg'].items()}
    task['public_rules'] = storage.public_rules(asset_metadata=metadata)
    task['evaluation'] = storage.evaluation_configuration(asset_metadata=metadata)
    config, _ = storage_configuration(task)
    assert config['thresholds'] == storage.THRESHOLDS
    assert {r['id']: r['nominal_mass_kg'] for r in task['public_rules']['referee']['object_properties']} == metadata['masses_kg']
    bad = deepcopy(task); bad['public_rules'] = original_rules
    with pytest.raises(ValueError, match='contract'):
        storage_configuration(bad)


def test_public_static_projection_excludes_all_private_pose_and_contact_fields():
    case = {'schema': 'living_room_reset_case.v1', 'task_id': 'remote_operation', 'seed': 1,
        'configuration_id': 'single_arm', 'actual_body_poses_m': {'private': 'never public'},
        'interaction_targets': [{'private': 'never public'}],
        'static_geometry': {'dimensions_m': {'remote': [.055, .16, .0239], 'holder_opening': [.067, .172]},
            'goal': {'operation': 'remote_return_to_holder', 'private_reset': [1, 2, 3]},
            'bodies': [{'object_id': 'remote', 'local_bounds_m': [[-.02, -.08, -.01], [.02, .08, .01]],
                        'nominal_mass_kg': .1, 'prim_path': '/private'}], 'raw_contacts': [1]}}
    case['case_sha256'] = canonical_hash(case)
    task = {'id': 'remote_operation', 'reset_case': case}
    out = project_reset_case(task, {'task_id': 'remote_operation', 'instruction': 'test'})
    encoded = json.dumps(out)
    assert 'never public' not in encoded and 'private_reset' not in encoded and '/private' not in encoded
    assert 'actual_body_poses' not in encoded and 'raw_contacts' not in encoded
    assert out['reset_case']['static_geometry']['bodies'][0]['nominal_mass_kg'] == .1
    case['seed'] = 2
    with pytest.raises(ValueError, match='identity'):
        project_reset_case(task, {'task_id': 'remote_operation'})


def synthetic_reset_task(task_id, seed):
    """Public transport fixture, with no claim about its physical realizability."""
    _, task = parent(task_id)
    geometry = {
        'remote_operation': {'dimensions_m': {'remote': [.055, .16, .0239], 'holder_opening': [.067, .172]},
                             'goal': {'operation': 'remote_return_to_holder'}},
        'water_pouring': {'dimensions_m': {'kettle_inner_profile': [[0., .05, .05], [.1, .05, .05]],
                                          'cup_inner_profile': [[0., .03, .03], [.1, .03, .03]]},
                          'handle_geometry': {'capsule_radius_m': .0065},
                          'goal': {'operation': 'pour_and_return', 'return_position_m': [.5, .1, .9],
                                   'target_interval_ml': [90., 110.]}},
        'storage_box': {'dimensions_m': {'item': [.04, .04, .04]},
                        'goal': {'operation': 'collect_and_close', 'tray_inside_xy_m': [[.4, .6], [-.1, .1]],
                                 'tray_floor_z_m': .8}},
    }[task_id]
    geometry['bodies'] = [{'object_id': 'fixture_body', 'local_bounds_m': [[-.02]*3, [.02]*3],
                           'nominal_mass_kg': .1}]
    case = {'schema': 'living_room_reset_case.v1', 'task_id': task_id, 'seed': seed,
            'configuration_id': 'single_arm', 'static_geometry': geometry}
    case['case_sha256'] = canonical_hash(case)
    task['reset_case'] = case
    return task


@pytest.mark.parametrize('task_id', ['remote_operation', 'storage_box'])
def test_get_task_typed_extension_preserves_legacy_and_checks_seed_geometry(task_id):
    from bench.protocol.functions import FUNCTION_OUTPUTS, REPLY_MODELS, RESPONSE_ADAPTER
    from bench.protocol.task_referee import project_task_rules, PublicTaskRequirements as Legacy
    task = synthetic_reset_task(task_id, 1)
    base = project_task_rules(task)
    base.update(referee={'schema': 'physical_task_referee_state.v1', 'state': 'ongoing',
        'phase': next(iter(base['public_rules']['phases'])),
        'reason_codes': [next(iter(base['public_rules']['reason_descriptions']))],
        'terminal': False, 'recoverable': True}, episode={'terminal': False, 'outcome': None})
    extended = project_reset_case(task, base)
    model = FUNCTION_OUTPUTS['get_task']
    assert model.model_validate(base).model_dump(mode='json') == Legacy.model_validate(base).model_dump(mode='json')
    assert model.model_validate(extended).model_dump(mode='json') == extended
    reply = REPLY_MODELS['get_task'](request_id='fixture', context=None, result=extended).model_dump(mode='json')
    assert RESPONSE_ADAPTER.validate_python(reply).result.reset_case.seed == 1
    for key, value in [('seed', True), ('seed', 10), ('actual_body_poses_m', {})]:
        bad = deepcopy(extended); bad['reset_case'][key] = value
        with pytest.raises(ValueError): model.model_validate(bad)
    bad = deepcopy(extended); bad['reset_case']['static_geometry']['bodies'][0]['prim_path'] = '/private'
    with pytest.raises(ValueError): model.model_validate(bad)
    other = 'storage_box' if task_id == 'remote_operation' else 'remote_operation'
    bad = deepcopy(extended)
    bad['reset_case'] = project_reset_case(synthetic_reset_task(other, 1), {'task_id': other})['reset_case']
    with pytest.raises(ValueError, match='differs from the task'): model.model_validate(bad)


@pytest.mark.parametrize('task_id', TASK_IDS)
def test_seeded_discovery_selects_public_contract_before_episode_without_worker(tmp_path, monkeypatch, task_id):
    from fastapi.testclient import TestClient
    from bench.isaacsim import release as dispatch
    from bench.runtime.server import create_app
    from scenes.workrooms.living_room.public_task import project_manipulation_task
    from bench.protocol.living_reset import PublicResetCase
    root = tmp_path/'release'; root.mkdir()
    manifest = {'schema': 'living_room_randomized_release.v1', 'configuration_id': 'single_arm',
                'tasks': [{'id': task_id, 'task': 'task.json'}]}
    (root/'release_manifest.json').write_text(json.dumps(manifest))
    loaded = []
    def inputs(root, selected, *, seed):
        loaded.append((selected, seed))
        return {}, {}, synthetic_reset_task(selected, seed), {}
    monkeypatch.setattr(dispatch, 'validate', lambda *args, **kwargs: manifest)
    monkeypatch.setattr(dispatch, 'task_inputs', inputs)
    manager = types.SimpleNamespace(release=root, shutdown=lambda: None,
        configuration_bindings=lambda: {}, configuration_release=lambda *args: root)
    with TestClient(create_app(service=manager)) as client:
        for selected in (None, '1', '9'):
            query = {'task_id': task_id, 'configuration_id': 'single_arm'}
            if selected is not None: query['seed'] = selected
            response = client.get('/tasks', params=query)
            assert response.status_code == 200, response.text
            row, = response.json()
            seed = 0 if selected is None else int(selected)
            case = PublicResetCase.model_validate(row['reset_case'])
            assert case.seed == seed and case.configuration_id == 'single_arm'
            assert row['task_release_sha256'] and row['configuration_id'] == 'single_arm'
            expected = project_reset_case(synthetic_reset_task(task_id, seed),
                project_manipulation_task(synthetic_reset_task(task_id, seed)))
            assert all(row[k] == value for k, value in expected.items())
            assert loaded[-1] == (task_id, seed)
        before = deepcopy(loaded)
        for query in ({'seed': '1'}, *({'task_id': task_id, 'seed': value}
                       for value in ('true', '-1', '10', '1.0', '01'))):
            assert client.get('/tasks', params=query).status_code == 422
        assert loaded == before
        manifest['schema'] = 'living_room_manipulation_task_release.v1'
        assert client.get('/tasks', params={'task_id': task_id, 'seed': '0'}).status_code == 422
