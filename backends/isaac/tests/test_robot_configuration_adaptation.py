"""CPU counterexamples for derivative-only frozen bimanual scoring contracts."""
from copy import deepcopy
import importlib
import json
from pathlib import Path
import sys
import types
import uuid

import pytest

from bench.paths import DATA_ROOT, load_json
from scenes.workrooms.catalog import task_definition
from scenes.workrooms.robot_configuration_adaptation import (
    ALLOWED_TASKS, FINGER_PREFIXES, HAND_PREFIXES, SOURCE_PINS, adapt_payload)


def frozen_payload(family, task_id):
    root = Path(task_definition(family, task_id)['source_release']['path'])
    task = load_json(root/'task.json')
    sources = {str(path.relative_to(root/'evaluator')): path.read_text()
               for path in (root/'evaluator').rglob('*.py')}
    return task, sources


@pytest.fixture
def load_sources(tmp_path):
    packages = []

    def load(sources):
        name = '_bimanual_frozen_test_'+uuid.uuid4().hex
        root = tmp_path/name
        root.mkdir()
        for relative, text in sources.items():
            destination = root/relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(text)
        package = types.ModuleType(name)
        package.__path__ = [str(root)]
        sys.modules[name] = package
        packages.append(name)
        return importlib.import_module(name+'.sequence_evaluator')

    yield load
    for name in list(sys.modules):
        if any(name == package or name.startswith(package+'.') for package in packages):
            del sys.modules[name]


@pytest.mark.parametrize('family,task_id', sorted(ALLOWED_TASKS))
def test_all_current_frozen_scorers_accept_minimal_adaptation(family, task_id, load_sources):
    task, sources = frozen_payload(family, task_id)
    original = deepcopy(task), dict(sources)
    adapted, new_sources, report = adapt_payload(family, task_id, task, sources)
    module = load_sources(new_sources)
    module.SequenceEvaluator(adapted)
    assert (task, sources) == original
    for key in task.keys()-{'public_rules', 'public_goal', 'evaluation',
                           'public_rules_sha256', 'public_referee_contract_sha256'}:
        if (family, task_id) == ('card_room', 'task0') and key == 'title_zh':
            assert adapted[key] == 'Task0 · 随机三子摆放'
            continue
        assert adapted[key] == task[key], key
    for key in task['evaluation'].keys()-{'hand_link_prefixes', 'finger_link_prefixes', 'public_rules_sha256'}:
        assert adapted['evaluation'][key] == task['evaluation'][key], key
    assert set(report['changed_sources']) == set(SOURCE_PINS.get((family, task_id), {}))
    if (family, task_id) not in SOURCE_PINS:
        assert adapted == task
        assert new_sources == sources
    assert report['numeric_thresholds_changed'] is False


@pytest.mark.parametrize('family,task_id', sorted(SOURCE_PINS))
def test_unreviewed_source_refused(family, task_id):
    task, sources = frozen_payload(family, task_id)
    for name in SOURCE_PINS[family, task_id]:
        tampered = dict(sources)
        tampered[name] += '\n# unreviewed edit\n'
        with pytest.raises(ValueError, match='SHA256'):
            adapt_payload(family, task_id, task, tampered)


@pytest.mark.parametrize('family,task_id', [
    ('kitchen', 'food_sorting'), ('laundry_room', 'towel_folding'),
    ('mechanical_assembly', 'cooperative_insertion'), ('living_room', 'tower_extraction'),
    ('card_room', 'tic_tac_toe_3x3')])
def test_excluded_rooms_and_multiplayer_tasks_refused(family, task_id):
    with pytest.raises(ValueError, match='allowlist'):
        adapt_payload(family, task_id, {}, {})


@pytest.mark.parametrize('name', ['../rules.py', '/rules.py', 'evaluator/rules.py', 'x//rules.py'])
def test_source_path_escape_refused(name):
    task, sources = frozen_payload('card_room', 'task0')
    sources[name] = '# invalid path'
    with pytest.raises(ValueError, match='relative Python paths'):
        adapt_payload('card_room', 'task0', task, sources)


@pytest.mark.parametrize('task_id', ['remote_operation', 'water_pouring'])
def test_episode_rebinding_uses_new_frozen_rules(task_id, load_sources):
    from scenes.workrooms.living_room.runtime_task import bind_episode_task
    task, sources = frozen_payload('living_room', task_id)
    adapted, sources, _ = adapt_payload('living_room', task_id, task, sources)
    module = load_sources(sources)
    rules = importlib.import_module(module.__package__+'.rules')
    for seed in (0, 17, 4294967295):
        rebound = bind_episode_task(adapted, seed, rules)
        module.SequenceEvaluator(rebound)
        assert '单臂' not in rebound['public_rules']['instruction_zh']
        if task_id == 'remote_operation':
            assert rebound['evaluation']['hand_link_prefixes'] == HAND_PREFIXES
            assert rebound['evaluation']['finger_link_prefixes'] == FINGER_PREFIXES
            assert '同一只手' not in json.dumps(rebound['public_rules'], ensure_ascii=False)
        for key in adapted['evaluation'].keys()-{'seed', 'contract_sha256'}:
            assert rebound['evaluation'][key] == adapted['evaluation'][key]


@pytest.mark.parametrize('version,envelope', [('v5', .006), ('v6', .006666666666666667+.001)])
def test_water_derivatives_preserve_each_frozen_spill_geometry(version, envelope, load_sources):
    from scenes.workrooms.living_room.runtime_task import bind_episode_task
    root = Path(f'{DATA_ROOT}/datasets/fr3_sharpa_bench/isaacsim/releases/living_room/water_pouring')/version
    task = load_json(root/'task.json')
    sources = {str(p.relative_to(root/'evaluator')): p.read_text()
               for p in (root/'evaluator').rglob('*.py')}
    adapted, sources, report = adapt_payload('living_room', 'water_pouring', task, sources)
    assert report == json.loads(json.dumps(report))
    from bench.isaacsim.release import digest
    assert report['source_sha_pins_checked'] == {'rules.py': digest(root/'evaluator/rules.py')}
    module = load_sources(sources)
    rules = importlib.import_module(module.__package__+'.rules')
    for seed in (0, 17, 1729):
        rebound = bind_episode_task(adapted, seed, rules)
        module.SequenceEvaluator(rebound)
        assert rebound['evaluation']['geometry_spill_envelope_m'] == pytest.approx(envelope)
        assert rebound['evaluation']['thresholds'] == task['evaluation']['thresholds']
        assert rebound['evaluation']['quantities'] == task['evaluation']['quantities']
        assert rebound['asset_metadata'] == task['asset_metadata']
    assert report['numeric_thresholds_changed'] is False


def test_card_seed_rebinding_keeps_random_cells_and_public_contract(load_sources):
    task, sources = frozen_payload('card_room', 'task0')
    adapted, sources, _ = adapt_payload('card_room', 'task0', task, sources)
    module = load_sources(sources)
    targets = []
    for seed in (0, 17, 4294967295):
        rebound = module.bind_episode_targets(adapted, seed)
        module.SequenceEvaluator(rebound)
        assert '用机器人' in rebound['public_rules']['instruction_zh']
        assert rebound['evaluation']['target_cells'] == rebound['public_goal']['target_cells']
        targets.append(rebound['public_goal']['target_cells'])
    assert targets[0] != targets[1]


@pytest.fixture
def remote(load_sources):
    task, sources = frozen_payload('living_room', 'remote_operation')
    task, sources, _ = adapt_payload('living_room', 'remote_operation', task, sources)
    module = load_sources(sources)
    return task, module, module.NativeTaskObserver(task)


def remote_contact(a, b, impulse=.001):
    return {'actor0': a, 'actor1': b, 'time_s': .02, 'point_count': 1,
            'impulse_magnitude_sum_ns': impulse, 'min_separation_m': -.00001}


@pytest.mark.parametrize('holding,pressing', [('left', 'right'), ('right', 'left')])
def test_remote_holding_and_pressing_can_use_either_hand(remote, holding, pressing):
    task, module, observer = remote
    actors = task['evaluation']['actors']
    contacts = [remote_contact(actors['remote'], '/World/Robot/'+holding+'_hand_shell'),
                remote_contact(actors['power'], '/World/Robot/'+pressing+'_index_DP')]
    hand, touching_robot, external, _, _, _, keys, touching = observer._contacts({'time_s': .02, 'contacts': contacts})
    assert hand and touching_robot and keys['power'] and touching['power']
    assert not external
    for row in contacts:
        row['impulse_magnitude_sum_ns'] = 0.
    hand, touching_robot, _, _, _, _, keys, touching = observer._contacts({'time_s': .02, 'contacts': contacts})
    assert not hand and not keys['power']
    assert touching_robot and touching['power']  # zero-load manifolds still prevent release


@pytest.mark.parametrize('side', ['left', 'right'])
def test_remote_arm_or_wrist_cannot_earn_hand_credit(remote, side):
    task, module, observer = remote
    for link in (side+'_hand_wrist', side+'_hand_flange', 'fr3_'+side+'_link7'):
        path = '/World/Robot/'+link
        assert not observer._hand(path)
        assert not observer._hand(path, finger=True)
        result = observer._contacts({'time_s': .02, 'contacts': [
            remote_contact(task['evaluation']['actors']['remote'], path)]})
        assert result[0] is False and result[2] is True


def test_remote_namespace_and_threshold_changes_still_rejected(remote):
    task, module, _ = remote
    for key, bad in [('hand_link_prefixes', ['left_', 'right_', 'fr3_']),
                     ('finger_link_prefixes', FINGER_PREFIXES+['left_hand_']),
                     ('stable_s', 0.)]:
        tampered = deepcopy(task)
        tampered['evaluation'][key] = bad
        with pytest.raises(ValueError):
            module.SequenceEvaluator(tampered)


@pytest.fixture
def fastener(load_sources):
    task, sources = frozen_payload('mechanical_assembly', 'fastener_assembly')
    task, sources, _ = adapt_payload('mechanical_assembly', 'fastener_assembly', task, sources)
    module = load_sources(sources)
    evaluator = module.SequenceEvaluator(task)
    # Isolate classification from the independently tested sampling protocol.
    evaluator._sampling = lambda observation, time, first: (0., 40, .02)
    return task, module, evaluator


def fastener_contact(a, b, impulse=.001):
    return {'actor0': a, 'actor1': b, 'time_s': .0005, 'point_count': 1,
        'nonzero_impulse_normal_count': int(impulse > 0), 'impulse_magnitude_sum_ns': impulse,
        'impulse_vector_ns': [0., 0., impulse], 'normal_length_max_abs_error': 0.,
        'min_separation_m': -.00001, 'impulse_components': 'normal_only'}


@pytest.mark.parametrize('side', ['left', 'right'])
def test_fastener_hand_tool_credit_and_all_robot_release(fastener, side):
    task, _, evaluator = fastener
    tool = task['evaluation']['actors']['screwdriver']
    hand = '/World/Robot/'+side+'_index_DP'
    for impulse in (.001, 0.):
        sets, _, _, violation, _ = evaluator._contacts({'contacts': {
            'rigid': [fastener_contact(tool, hand, impulse)]}}, .02, False)
        assert sets['robot_target'] == {1}
        assert sets['hand_tool'] == ({1} if impulse else set())
        assert violation is None
    arm = '/World/Robot/fr3_'+side+'_link7'
    sets, _, _, violation, _ = evaluator._contacts({'contacts': {
        'rigid': [fastener_contact(tool, arm)]}}, .02, False)
    assert not sets['hand_tool']
    assert violation == 'non_hand_robot_tool_contact'


@pytest.mark.parametrize('side', ['left', 'right'])
def test_fastener_setup_does_not_weaken_direct_bolt_prohibition(fastener, side):
    task, _, evaluator = fastener
    bolt = task['evaluation']['actors']['bolt']
    contact = fastener_contact(bolt, '/World/Robot/'+side+'_index_DP')
    assert evaluator.reset_variant == 'loose_bolt'
    result = evaluator._contacts({'contacts': {'rigid': [contact]}}, .02, False)
    assert result[3] is None
    evaluator.baseline = object()  # once engaged, either hand must use the tool
    result = evaluator._contacts({'contacts': {'rigid': [contact]}}, .02, False)
    assert result[3] == 'direct_robot_bolt_contact'


def test_fastener_namespace_and_threshold_changes_still_rejected(fastener):
    task, module, _ = fastener
    for key, bad in [('hand_link_prefixes', ['left_', 'right_', 'fr3_']),
                     ('minimum_pair_impulse_ns', 0.)]:
        tampered = deepcopy(task)
        tampered['evaluation'][key] = bad
        with pytest.raises(ValueError):
            module.SequenceEvaluator(tampered)


@pytest.fixture
def gear(load_sources):
    task, sources = frozen_payload('mechanical_assembly', 'gear_train_assembly')
    task, sources, _ = adapt_payload('mechanical_assembly', 'gear_train_assembly', task, sources)
    module = load_sources(sources)
    evaluator = module.SequenceEvaluator(task)
    evaluator.previous_time = 0.
    # Classifier unit counterexamples reuse the already validated native row
    # shape while keeping protocol chronology covered by the original suite.
    module.read_native_contact_window = lambda observation, **kwargs: (observation['rows'], 40, .02)
    return task, module, evaluator


def gear_rows(task, module, hand_a, hand_b):
    actors = task['evaluation']['actors']
    tool = module.TOOLS[0]
    rows = [fastener_contact(actors[name], task['evaluation']['fixture_actor']) for name in module.GEARS]
    rows += [fastener_contact(actors[a], actors[b]) for a, b in zip(module.GEARS, module.GEARS[1:])]
    rows += [fastener_contact(actors[tool], actors['input_gear'])]
    rows += [fastener_contact(actors[tool], '/World/Robot/'+hand_a),
             fastener_contact(actors[tool], '/World/Robot/'+hand_b)]
    rows[-2]['impulse_vector_ns'] = [.001, 0., 0.]
    rows[-1]['impulse_vector_ns'] = [-.001, 0., 0.]
    return rows


@pytest.mark.parametrize('side', ['left', 'right'])
def test_gear_both_hands_can_supply_full_contact_transmission(gear, side):
    task, module, evaluator = gear
    rows = gear_rows(task, module, side+'_thumb_DP', side+'_index_DP')
    result, _, _ = evaluator._contacts({'rows': rows}, .02, False)
    assert result['chain'] == 1/evaluator.substeps
    assert result['released'] is False
    assert result['direct_other_drive'] is False
    for row in rows[-2:]:
        row['impulse_magnitude_sum_ns'] = 0.
        row['impulse_vector_ns'] = [0., 0., 0.]
    result, _, _ = evaluator._contacts({'rows': rows}, .02, False)
    assert result['chain'] == 0.
    assert result['released'] is False


@pytest.mark.parametrize('side', ['left', 'right'])
def test_gear_either_arm_and_direct_gear_drive_remain_disallowed(gear, side):
    task, module, evaluator = gear
    rows = gear_rows(task, module, side+'_thumb_DP', side+'_index_DP')
    tool = task['evaluation']['actors'][module.TOOLS[0]]
    rows.append(fastener_contact(tool, '/World/Robot/fr3_'+side+'_link7'))
    result, _, _ = evaluator._contacts({'rows': rows}, .02, False)
    assert result['chain'] == 0.
    rows.pop()
    rows.append(fastener_contact(task['evaluation']['actors']['input_gear'],
                                 '/World/Robot/'+side+'_index_DP'))
    result, _, _ = evaluator._contacts({'rows': rows}, .02, False)
    assert result['direct_other_drive'] is True
    assert result['gear_released'] is False


@pytest.mark.parametrize('family,task_id,helper', [
    ('laboratory', 'sample_sorting', 'native_evidence'),
    ('mechanical_assembly', 'nist_connector_insertion', 'nist_native_evidence')])
@pytest.mark.parametrize('link', ['left_index_DP', 'right_index_DP', 'fr3_left_link7', 'fr3_right_link7'])
def test_unchanged_native_contact_helpers_cover_both_arms(family, task_id, helper, link, load_sources):
    task, sources = frozen_payload(family, task_id)
    module = load_sources(sources)
    # NIST is frozen under native_evidence.py in some release formats.
    name = helper if helper+'.py' in sources else 'native_evidence'
    evidence = importlib.import_module(module.__package__+'.'+name)
    row = fastener_contact('/World/Objects/test', '/World/Robot/'+link, 0.)
    contacts = evidence.Contacts([row], {'test': '/World/Objects/test'}, {'test'}, .0005, 1.e-10)
    assert contacts.robot == {'test'}  # no-load contact must still prevent release
    assert not contacts.upward


def test_study_observer_requires_contact_filters_for_both_arms(load_sources):
    task, sources = frozen_payload('study', 'book_stacking')
    module = load_sources(sources)
    observer_module = importlib.import_module(module.__package__+'.native_observer')
    config = task['evaluation']
    robot = {name: '/World/Robot/'+name for name in ('left_index_DP', 'right_index_DP', 'fr3_left_link7', 'fr3_right_link7')}
    bodies = config['body_paths']
    all_paths = [*bodies.values(), *config['support_paths'].values(), *robot.values()]
    environment = types.SimpleNamespace(dt=config['physics_dt_s'], scene_info={
        'objects': {name: {'path': path} for name, path in bodies.items()}, 'robot_link_paths': robot},
        _contact_sensor_paths=list(bodies.values()),
        _contact_filter_paths=[list(all_paths) for _ in bodies])
    observer = observer_module.NativeObserver(environment, task)
    for name, path in robot.items():
        assert observer.actor(path) == 'robot/'+name
    environment._contact_filter_paths[0].remove(robot['left_index_DP'])
    with pytest.raises(ValueError, match='filters omit'):
        observer_module.NativeObserver(environment, task)


def test_tower_contact_actor_ids_keep_both_arms_in_single_agent_mode(load_sources):
    _, sources = frozen_payload('living_room', 'tower_building')
    module = load_sources(sources)
    evidence = importlib.import_module(module.__package__+'.native_evidence')
    for name in ('left_index_DP', 'right_index_DP', 'fr3_left_link7', 'fr3_right_link7'):
        actor = evidence.actor('/World/Robot/'+name, {}, game=False)
        assert actor == 'robot/'+name
        assert evidence.touched([{'a': 'block', 'b': actor}], {'block'})
