"""CPU reset/layout and privileged-plan contracts for mechanical L3."""
from copy import deepcopy

import pytest

from bench.paths import load_json
from scenes.workrooms.mechanical_assembly.tasks.shaft_spacer_assembly import layout
from scenes.workrooms.mechanical_assembly.tasks.shaft_spacer_assembly.task import build_task_inputs
from scripts.validation import validate_mechanical_shaft_spacer as validator


def test_task3_layout_keeps_canonical_supply_reset_and_support_order():
    value = build_task_inputs()
    scene, task = value['scene'], value['task']
    report = layout.validate_layout(scene, task)
    assert report['passed']
    for low, high in report['supply_rest_bounds_m'].values():
        assert low[2] == pytest.approx(layout.TABLE_TOP_Z_M)
        assert high[2] == pytest.approx(layout.TABLE_TOP_Z_M + layout.PARAMETERS['supply_rest_height'])
    assert task['task_layout']['layout_id'] == layout.LAYOUT_ID
    assert task['evaluation']['part_order'] == list(layout.PART_ORDER)
    assert task['evaluation']['required_support_pairs'] == [
        ['spacer', 'fixture'], ['wide_ring', 'spacer'], ['top_ring', 'wide_ring']]
    assert task['reset_conditions']['supply_positions_m'] == {
        name: list(position) for name, position in layout.SUPPLY_POSITIONS_M.items()}
    assert scene['task_layout']['parent_scene_unchanged']
    assert scene['workroom']['version'] == 'mechanical_workshop_20261017_v3'


def test_task3_layout_rejects_supply_drift_or_overlap():
    value = build_task_inputs()
    changed = deepcopy(value['scene'])
    next(row for row in changed['fixtures'] if row['id'] == 'spacer_supply_rest')['position_m'][1] = .01
    with pytest.raises(ValueError, match='supply reset'):
        layout.validate_layout(changed, value['task'])

    changed = deepcopy(value['scene'])
    rests = [row for row in changed['fixtures'] if row['id'].endswith('_supply_rest')]
    rests[1]['position_m'] = rests[0]['position_m']
    with pytest.raises(ValueError, match='supply reset'):
        layout.validate_layout(changed, value['task'])


def test_task3_cpu_reachability_and_recovery_are_explicit():
    value = build_task_inputs()
    targets = value['task']['cpu_contract']['reachability_targets']
    assert set(targets['approach_points_m']) == set(layout.PART_ORDER)
    assert targets['shaft_approach_m'][2] > layout.TABLE_TOP_Z_M
    assert value['task']['cpu_contract']['ordinary_agent_validation'] is False
    recovery = value['task']['task_layout']['recovery']
    assert recovery['reset_is_in_episode_action'] is False
    assert recovery['allowed_operations'] == ['withdraw', 'regrasp', 'realign', 'retry']
    assert 'object_pose_playback' in recovery['forbidden_operations']


def test_privileged_plan_is_zero_model_and_fail_closed(tmp_path):
    plan = validator.build_plan()
    assert plan['schema'] == 'mechanical_l3_privileged_validation_plan.v1'
    assert plan['model_calls'] == 0
    assert plan['native_runtime_started'] is False
    assert plan['ordinary_agent_validation'] is False
    assert plan['object_state_writeback'] is False
    assert plan['layout_check']['passed']
    assert plan['privileged_contract_check']['passed']
    assert [phase['id'] for phase in plan['privileged_contract']['phases'][:6]] == [
        'reset', 'ready', 'place_spacer', 'place_wide_ring', 'place_top_ring', 'stability']
    output = tmp_path / 'l3_plan.json'
    validator.run(output)
    assert load_json(output)['task_id'] == 'shaft_spacer_assembly'
    with pytest.raises(FileExistsError):
        validator.run(output)


def test_native_probe_input_freezes_release_robot_phases_and_no_writeback():
    value = validator.build_native_input()
    assert value['schema'] == 'mechanical_l3_native_probe_input.v1'
    assert value['release_manifest_sha256'] == validator.definition.TASK['source_release']['manifest_sha256']
    assert value['model_calls'] == 0
    assert value['ordinary_agent_validation'] is False
    assert value['task_success_claimed'] is False
    assert value['physics_accepted'] is False
    contract = value['robot_phase_contract']
    assert contract['object_attachment'] is False
    assert contract['object_state_writeback'] is False
    assert contract['external_object_forces'] is False
    assert contract['phase_count'] == 8
    assert [phase['id'] for phase in contract['phases']] == [
        'ready', 'approach_spacer', 'shaft_approach_spacer',
        'approach_wide_ring', 'shaft_approach_wide_ring',
        'approach_top_ring', 'shaft_approach_top_ring', 'withdraw']
    assert all(case['duration_s'] == pytest.approx(sum(phase['duration_s'] for phase in case['robot_phases']))
               for case in value['cases'])
    assert all(phase['timing']['stop_on_endpoint_failure'] for phase in contract['phases'])
    assert all(len(case['robot_phases']) == contract['phase_count'] for case in value['cases'])
    assert {case['id'] for case in value['cases']} == {
        'nominal_supply_with_robot_targets', 'guided_stack_drop_with_robot_targets',
        'near_seat_with_robot_targets', 'wrong_order_recovery_probe'}
    for case in value['cases']:
        assert case['release_contract']['free_dynamic_objects'] is True
        assert case['release_contract']['gravity_only_after_reset'] is True
        assert case['release_contract']['object_pose_writeback_forbidden'] is True
        assert case['release_contract']['external_object_forces_forbidden'] is True
        receipt = case['phase_receipt_contract']
        assert receipt['record_raw_contacts'] is True
        assert receipt['task_success_credit'] == 0
        assert receipt['required_phase_ids'] == [phase['id'] for phase in case['robot_phases']]
    guided = next(case for case in value['cases'] if case['id'] == 'guided_stack_drop_with_robot_targets')
    assert guided['release_contract']['gravity_only_after_reset'] is True
    assert guided['release_contract']['object_pose_writeback_forbidden'] is True


def test_native_robot_phase_contract_rejects_incomplete_or_non_grid_targets():
    from scripts.validation.validate_assembly_contacts import (
        _validate_privileged_input_contract, _validate_robot_phases)
    initial = {'joint_a': 0.0, 'joint_b': 0.0}
    with pytest.raises(ValueError, match='complete finite joint targets'):
        _validate_robot_phases([{'id': 'bad', 'duration_s': .02,
                                 'joint_positions_rad': {'joint_a': 1.}}], initial)
    with pytest.raises(ValueError, match='20 ms durations'):
        _validate_robot_phases([{'id': 'bad', 'duration_s': .021,
                                 'joint_positions_rad': initial}], initial)
    value = validator.build_native_input()
    value['model_calls'] = 1
    with pytest.raises(ValueError, match='Frozen L3 privileged input contract failed'):
        _validate_privileged_input_contract(value)


def test_wrong_order_reset_physically_reverses_stack_without_changing_goal():
    nominal = validator._near_seat_case([])
    wrong = validator._wrong_order_case([])
    assert wrong['assertions'] == nominal['assertions']
    assert wrong['reset_objects'] != nominal['reset_objects']
    ordered = sorted(wrong['reset_objects'], key=lambda name: wrong['reset_objects'][name]['position_m'][2])
    assert ordered == ['top_ring', 'wide_ring', 'spacer']
    heights = {'top_ring': .025, 'wide_ring': .020, 'spacer': .030}
    for lower, upper in zip(ordered, ordered[1:]):
        gap = (wrong['reset_objects'][upper]['position_m'][2] -
               wrong['reset_objects'][lower]['position_m'][2] - heights[lower])
        assert gap == pytest.approx(.003)
    assert wrong['expected_functional_success'] is False


@pytest.fixture(scope='module')
def l3_native_input():
    return validator.build_native_input()


@pytest.mark.parametrize('missing', ['release_contract', 'phase_receipt_contract', 'contact_actor_paths'])
def test_l3_input_rejects_missing_case_evidence_contract(l3_native_input, missing):
    from scripts.validation.validate_assembly_contacts import _validate_privileged_input_contract
    changed = deepcopy(l3_native_input)
    case = changed['cases'][2]
    if missing == 'contact_actor_paths':
        del case['phase_receipt_contract'][missing]
    else:
        del case[missing]
    with pytest.raises(ValueError, match=missing):
        _validate_privileged_input_contract(changed)


@pytest.mark.parametrize('physical_passed,include_rows', [(True, True), (False, True), (True, False)])
def test_l3_case_keeps_physical_result_and_native_contacts(monkeypatch, physical_passed, include_rows):
    from scripts.validation import validate_assembly_contacts as runner
    phase = {'id': 'release', 'duration_s': .02, 'joint_positions_rad': {}}
    case = {'robot_phases': [phase], **validator._native_case_contract([phase], 'test')}
    row = {
        'phase': 'release', 'native_step': 1, 'time_s': .0005,
        'raw_contacts': [{
            'actor0': '/World/Objects/spacer',
            'actor1': '/World/Fixtures/shaft_fixture/collision_000',
            'impulse_magnitude_sum_ns': .001,
        }],
    }
    monkeypatch.setattr(runner, 'assess_case', lambda *args: {
        'passed': physical_passed, 'checks': {'support': physical_passed},
        'functional_contact_success': physical_passed,
    })
    result = runner._assess_contact_case([row] if include_rows else [], case, {}, .0005,
                                         schema=validator.NATIVE_INPUT_SCHEMA)
    assert result['passed'] is (physical_passed and include_rows)
    assert result['checks']['support'] is physical_passed
    assert result['phase_receipts']['passed'] is include_rows
    if include_rows:
        contact = result['phase_receipts']['receipts'][0]['contact_pairs'][0]
        assert contact['records'] == 1
        assert contact['positive_substeps'] == 1

