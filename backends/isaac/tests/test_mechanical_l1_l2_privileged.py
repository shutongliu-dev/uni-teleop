"""CPU contracts for the privileged L1/L2 native input builders.

These tests freeze release/reset/robot-phase semantics only.  They never start
Isaac Sim and do not turn phase targets into manipulation or Agent scores.
"""
from copy import deepcopy

import pytest

from scripts.validation import validate_mechanical_peg as peg
from scripts.validation import validate_mechanical_sorting as sorting
from scripts.validation.validate_assembly_contacts import _validate_privileged_input_contract as validate_runner_contract


@pytest.mark.parametrize('builder,task_id,schema,phase_count', [
    (sorting, 'parts_sorting', 'mechanical_l1_native_probe_input.v1', 19),
    (peg, 'peg_insertion', 'mechanical_l2_native_probe_input.v1', 15),
])
def test_native_input_binds_release_zero_model_and_robot_targets(builder, task_id, schema, phase_count):
    value = builder.build_native_input()
    assert value['schema'] == schema
    assert value['task_id'] == task_id
    assert value['release_manifest_sha256']
    assert value['model_calls'] == 0
    assert value['ordinary_agent_validation'] is False
    assert value['task_success_claimed'] is False
    assert value['physics_accepted'] is False
    assert value['object_state_writeback_after_reset'] is False
    assert value['external_object_forces'] is False
    contract = value['robot_phase_contract']
    assert contract['phase_count'] == phase_count
    assert len(contract['phases']) == phase_count
    assert contract['object_attachment'] is False
    assert contract['object_state_writeback'] is False
    assert contract['external_object_forces'] is False
    assert all(set(phase['joint_positions_rad']) == set(value['initial_state']['joint_positions_rad'])
               for phase in contract['phases'])
    assert all(phase['duration_s'] >= .02 and
               abs(phase['duration_s']/.02-round(phase['duration_s']/.02)) < 1e-7
               for phase in contract['phases'])
    assert builder._validate_privileged_input_contract(value)['passed'] is True
    assert validate_runner_contract(value)['passed'] is True


def test_l1_requires_native_grasp_support_release_and_terminal_observations():
    value = sorting.build_native_input()
    assert value['required_native_observations'] == list(sorting.REQUIRED_NATIVE_OBSERVATIONS)
    case = value['cases'][0]
    assert case['reset_variant'] == 'original_supply'
    assert case['expected_functional_success'] is True
    assert case['phase_receipt_contract']['record_raw_contacts'] is True
    assert case['phase_receipt_contract']['task_success_credit'] == 0
    assert case['release_contract'] == {
        'free_dynamic_objects': True,
        'gravity_only_after_reset': True,
        'object_pose_writeback_forbidden': True,
        'external_object_forces_forbidden': True,
    }
    required = set(case['phase_receipt_contract']['required_phase_ids'])
    assert {'block_pickup_close', 'block_lift', 'block_release'} <= required
    assert {'cylinder_pickup_close', 'cylinder_lift', 'cylinder_release'} <= required
    assert {'ring_pickup_close', 'ring_lift', 'ring_release'} <= required


def test_l2_keeps_nominal_reset_separate_from_outside_socket_negative_control():
    value = peg.build_native_input()
    nominal, negative = value['cases']
    frozen = value['task']['evaluation']['initial_objects']
    assert nominal['reset_objects'] == frozen
    assert 'engineering_reset_override' not in nominal
    assert negative['engineering_reset_override'] is True
    assert negative['expected_functional_success'] is False
    assert negative['reset_objects'] != frozen
    assert negative['reset_variant'] == 'outside_socket_engineering_probe'
    required = set(nominal['phase_receipt_contract']['required_phase_ids'])
    assert {'peg_0_pickup_close', 'peg_0_guided_lower', 'peg_0_release'} <= required
    assert {'peg_1_pickup_close', 'peg_1_guided_lower', 'peg_1_release'} <= required


@pytest.mark.parametrize('builder', [sorting, peg])
def test_privileged_contract_rejects_release_drift_and_shortcuts(builder):
    value = builder.build_native_input()
    broken = deepcopy(value)
    broken['release_manifest_sha256'] = '0'*64
    with pytest.raises(ValueError, match='privileged input contract failed'):
        builder._validate_privileged_input_contract(broken)

    broken = builder.build_native_input()
    broken['robot_phase_contract']['object_state_writeback'] = True
    with pytest.raises(ValueError, match='privileged input contract failed'):
        builder._validate_privileged_input_contract(broken)

    broken = builder.build_native_input()
    broken['cases'][0]['phase_receipt_contract']['task_success_credit'] = 1
    with pytest.raises(ValueError, match='privileged input contract failed'):
        builder._validate_privileged_input_contract(broken)


@pytest.mark.parametrize('builder', [sorting, peg])
def test_cpu_plan_never_starts_native_or_agent_validation(builder, tmp_path):
    output = tmp_path/(builder.TASK_ID+'.json')
    plan = builder.run(output)
    assert output.is_file()
    assert plan['model_calls'] == 0
    assert plan['ordinary_agent_validation'] is False
    assert plan['native_runtime_started'] is False
    assert plan['task_success_claimed'] is False
    assert plan['contract_check']['passed'] is True
