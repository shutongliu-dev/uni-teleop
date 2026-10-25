"""CPU contracts for the privileged L4 loose-bolt native probe."""
from copy import deepcopy

import pytest

from scripts.validation import validate_mechanical_fastener as validator
from scripts.validation.validate_assembly_contacts import (
    _phase_receipts,
    _validate_privileged_input_contract,
)


def test_l4_native_input_binds_v10_pgs_and_zero_model_contract():
    value = validator.build_native_input()
    assert value['schema'] == 'mechanical_l4_loose_native_probe_input.v1'
    assert value['release'].endswith('/loose_bolt_candidate_v10_pgs')
    assert value['release_manifest_sha256'] == 'e3fdb6d0ce747bcc0f4b4786dbe9fe4ad11cfd69ab4266919551820357e40a44'
    assert value['model_calls'] == 0
    assert value['ordinary_agent_validation'] is False
    assert value['task_success_claimed'] is False
    assert value['physics_accepted'] is False
    assert value['object_state_writeback_after_reset'] is False
    assert value['external_object_forces'] is False
    contract = value['robot_phase_contract']
    assert contract['object_attachment'] is False
    assert contract['object_state_writeback'] is False
    assert contract['external_object_forces'] is False
    assert [phase['id'] for phase in value['cases'][0]['robot_phases']] == [
        'ready', 'pickup_approach', 'pickup_close', 'pickup_lift',
        'guide_approach', 'start_thread', 'release_bolt', 'tool_approach',
        'tool_close', 'tool_insertion', 'release', 'withdraw',
    ]


def test_l4_reset_and_receipt_contract_is_fail_closed():
    value = validator.build_native_input()
    assert _validate_privileged_input_contract(value)['passed'] is True
    broken = deepcopy(value)
    broken['cases'][0]['release_contract']['external_object_forces_forbidden'] = False
    with pytest.raises(ValueError, match='Frozen L4 privileged input contract failed'):
        _validate_privileged_input_contract(broken)
    moved = deepcopy(value)
    moved['cases'][0]['reset_objects']['bolt']['position_m'][2] += .001
    with pytest.raises(ValueError, match='Frozen L4 privileged input contract failed'):
        _validate_privileged_input_contract(moved)


def test_phase_receipts_keep_raw_contacts_and_do_not_claim_success():
    phase = {'id': 'pickup', 'duration_s': .02, 'joint_positions_rad': {}}
    rows = [{
        'phase': 'pickup', 'native_step': 1, 'time_s': .0005,
        'maximum_penetration_m': .0001, 'robot_interference': [],
        'raw_contacts': [{
            'actor0': '/World/Objects/bolt', 'actor1': '/World/Fixtures/bolt_supply_rest',
            'impulse_magnitude_sum_ns': 0.,
        }],
    }]
    result = _phase_receipts(rows, [phase], {
        'required_phase_ids': ['pickup'],
        'contact_actor_paths': {
            'bolt': '/World/Objects/bolt',
            'bolt_supply_rest': '/World/Fixtures/bolt_supply_rest',
        },
        'minimum_native_samples_per_phase': 1,
        'record_raw_contacts': True,
        'task_success_credit': 0,
    })
    assert result['passed'] is True
    assert result['task_success_credit'] == 0
    assert result['object_state_writeback_after_reset'] is False
    assert result['external_object_forces'] is False
    assert result['receipts'][0]['contact_pairs'][0]['records'] == 1


def test_phase_receipts_reject_missing_native_phase():
    result = _phase_receipts([], [{'id': 'pickup', 'duration_s': .02, 'joint_positions_rad': {}}], {
        'required_phase_ids': ['pickup'], 'contact_actor_paths': {},
        'minimum_native_samples_per_phase': 1,
    })
    assert result['passed'] is False
