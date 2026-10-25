"""CPU contracts for the privileged L5 gear-train native probe."""
from copy import deepcopy

import pytest

from scripts.validation import validate_mechanical_gear as validator
from scripts.validation.validate_assembly_contacts import _validate_privileged_input_contract


def test_l5_native_input_binds_v34_and_zero_model_contract():
    value = validator.build_native_input()
    assert value['schema'] == 'mechanical_l5_gear_native_probe_input.v1'
    assert value['release'].endswith('/practical_rules_v4/release')
    assert value['release_manifest_sha256'] == '62bd44368d8d42a36885d29272d7ef0dae2582acf84da2f8faee6d6fe37c96f5'
    assert value['model_calls'] == 0
    assert value['ordinary_agent_validation'] is False
    assert value['task_success_claimed'] is False
    assert value['physics_accepted'] is False
    assert value['object_state_writeback_after_reset'] is False
    assert value['external_object_forces'] is False
    contract = value['robot_phase_contract']
    assert contract['phase_count'] == 28
    assert contract['object_attachment'] is False
    assert contract['object_state_writeback'] is False
    assert contract['external_object_forces'] is False
    assert all(set(phase['joint_positions_rad']) == set(value['initial_state']['joint_positions_rad'])
               for phase in contract['phases'])
    assert all(phase['duration_s'] >= .02 and
               abs(phase['duration_s']/.02-round(phase['duration_s']/.02)) < 1.e-7
               for phase in contract['phases'])
    assert validator._validate_privileged_input_contract(value)['passed'] is True
    assert _validate_privileged_input_contract(value)['passed'] is True


def test_l5_phase_sequence_covers_each_gear_release_and_tool_engagement():
    value = validator.build_native_input()
    phases = [phase['id'] for phase in value['cases'][0]['robot_phases']]
    assert phases == [
        'ready',
        'input_gear_approach', 'input_gear_pickup_close', 'input_gear_lift',
        'input_gear_shaft_approach', 'input_gear_lower', 'input_gear_release', 'input_gear_withdraw',
        'idler_gear_approach', 'idler_gear_pickup_close', 'idler_gear_lift',
        'idler_gear_shaft_approach', 'idler_gear_lower', 'idler_gear_release', 'idler_gear_withdraw',
        'output_gear_approach', 'output_gear_pickup_close', 'output_gear_lift',
        'output_gear_shaft_approach', 'output_gear_lower', 'output_gear_release', 'output_gear_withdraw',
        'tool_approach', 'tool_close', 'tool_lift', 'tool_engage', 'tool_release', 'withdraw',
    ]
    required = set(value['cases'][0]['phase_receipt_contract']['required_phase_ids'])
    assert {'input_gear_pickup_close', 'input_gear_lower', 'input_gear_release',
            'idler_gear_pickup_close', 'idler_gear_lower', 'idler_gear_release',
            'output_gear_pickup_close', 'output_gear_lower', 'output_gear_release',
            'tool_approach', 'tool_close', 'tool_engage', 'tool_release', 'withdraw'} <= required


def test_l5_layout_candidate_uses_actual_supply_coordinates():
    release = validator.definition.TASK['design']['supply_layout_candidate']['release']
    value = validator.build_native_input(release['path'])
    assert value['release_manifest_sha256'] == release['manifest_sha256']
    assert _validate_privileged_input_contract(value)['passed']
    supplies = {row['id']: row['position_m'][:2] for row in value['scene']['objects']}
    phases = {phase['id']: phase for phase in value['cases'][0]['robot_phases']}
    for gear_id in validator.GEAR_IDS:
        for suffix in ('_approach', '_pickup_close', '_lift'):
            assert phases[gear_id+suffix]['endpoint_ik']['position_m'][:2] == supplies[gear_id]


def test_l5_reset_and_phase_contract_reject_shortcuts():
    value = validator.build_native_input()
    broken = deepcopy(value)
    broken['cases'][0]['reset_objects']['input_gear']['position_m'][2] += .001
    with pytest.raises(ValueError, match='Frozen L5 privileged input contract failed'):
        validator._validate_privileged_input_contract(broken)

    broken = validator.build_native_input()
    broken['cases'][0]['release_contract']['external_object_forces_forbidden'] = False
    with pytest.raises(ValueError, match='Frozen L5 privileged input contract failed'):
        validator._validate_privileged_input_contract(broken)

    broken = validator.build_native_input()
    broken['robot_phase_contract']['object_attachment'] = True
    with pytest.raises(ValueError, match='Frozen L5 privileged input contract failed'):
        _validate_privileged_input_contract(broken)


def test_l5_cpu_plan_never_starts_native_or_agent_validation(tmp_path):
    output = tmp_path/'gear_plan.json'
    plan = validator.run(output)
    assert output.is_file()
    assert plan['model_calls'] == 0
    assert plan['ordinary_agent_validation'] is False
    assert plan['native_runtime_started'] is False
    assert plan['task_success_claimed'] is False
    assert plan['robot_phase_contract']['phase_count'] == 28
