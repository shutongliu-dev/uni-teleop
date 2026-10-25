"""Longer observations preserve the predeclared physical and scoring contract."""
from copy import deepcopy

import pytest

from scripts.validation.gear_transmission_checks import CONTRACT, probe_contract
from scripts.validation.validate_gear_transmission import make_probe_inputs, validate_probe_inputs


def test_extended_window_changes_only_two_durations():
    contract = probe_contract('extended_observation')
    assert [p['duration_s'] for p in contract['phases']] == [1., 8., .5, 9., 1.]
    for phase in contract['phases']:
        if phase['id'] in ('forward', 'reverse'):
            phase['duration_s'] -= 4.
    assert contract == CONTRACT
    assert probe_contract() == CONTRACT


@pytest.mark.parametrize('preset', ['standard', 'extended_observation', 'high_torque'])
def test_declared_contracts_are_accepted(preset):
    inputs = {'probe_contract': probe_contract(preset), 'supply_duration_s': 1.5}
    assert validate_probe_inputs(inputs) == inputs['probe_contract']


def test_high_torque_variant_is_explicitly_bounded_and_separate_from_standard():
    high = probe_contract('high_torque')
    assert high['engineering_variant'] == 'bounded_high_torque_v1'
    assert high['maximum_torque_nm'] == 1.6
    assert high['maximum_torque_nm'] != CONTRACT['maximum_torque_nm']


@pytest.mark.parametrize('change', ['torque', 'span', 'duration', 'private_hook'])
def test_unfrozen_thresholds_durations_and_solver_hook_are_rejected(change):
    inputs = {'probe_contract': probe_contract('extended_observation'), 'supply_duration_s': 1.5}
    if change == 'torque':
        inputs['probe_contract']['maximum_torque_nm'] = 1.
    elif change == 'span':
        inputs['probe_contract']['minimum_driven_input_span_rad'] = .1
    elif change == 'duration':
        inputs['probe_contract']['phases'][1]['duration_s'] = 7.
    else:
        inputs['diagnostic_solver_type'] = 'PGS'
    with pytest.raises(ValueError):
        validate_probe_inputs(inputs)


def test_canonical_pgs_and_extended_window_preserve_full_task_inputs():
    original = make_probe_inputs()
    selected = make_probe_inputs(contract_preset='extended_observation', solver_type='PGS')
    assert 'diagnostic_solver_type' not in selected
    assert validate_probe_inputs(selected) == probe_contract('extended_observation')
    same = deepcopy(selected)
    assert same['scene']['isaacsim'].pop('solver_type') == 'PGS'
    assert same['scene']['mechanical_task_provenance'].pop('solver_selection')['requested'] == 'PGS'
    same['probe_contract'] = probe_contract()
    assert same == original


def test_native_preflight_binds_transmission_geometry_to_current_task_contract():
    inputs = make_probe_inputs()
    assert inputs['transmission_geometry_contract']['gear_teeth'] == {
        'input_gear': 20, 'idler_gear': 32, 'output_gear': 20}
    assert inputs['transmission_geometry_contract']['shaft_centers_world_xy_m'] == [
        [.4, -.13], [.4, 0.], [.4, .13]]
    inputs['transmission_geometry_contract']['gear_teeth']['idler_gear'] = 20
    with pytest.raises(ValueError, match='geometry contract'):
        validate_probe_inputs(inputs)
