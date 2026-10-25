"""Independent public-schema attacks and all projection snapshots round-tripped."""
from copy import deepcopy
import importlib.util
from pathlib import Path

import pytest
from pydantic import ValidationError

from bench.sensors.contact_history import ContactHistory, REGIONS
from bench.sensors.contact_schema import ContactHistoryObservation

spec = importlib.util.spec_from_file_location('independent_contact_history_fixtures',
    Path(__file__).with_name('test_contact_history.py'))
projection = importlib.util.module_from_spec(spec)
spec.loader.exec_module(projection)
metadata, config = projection.metadata, projection.config


@pytest.fixture(autouse=True)
def every_snapshot_has_exact_public_schema(monkeypatch):
    original = ContactHistory.snapshot

    def checked_snapshot(self):
        result = original(self)
        assert ContactHistoryObservation.model_validate(result).model_dump(mode='json') == result
        return result

    monkeypatch.setattr(ContactHistory, 'snapshot', checked_snapshot)


# The existing29 test assertions remain unchanged; every snapshot additionally
# traverses the public schema. Original first-run XML remains independent.
for name, function in vars(projection).items():
    if name.startswith('test_'):
        globals()['test_schema_compatible_' + name[5:]] = function


@pytest.fixture
def available(config):
    history = ContactHistory(config)
    history.begin('public_a', 0.)
    history.observe(projection.window(rows=[projection.contact(config, tick=i) for i in range(1, 41)]))
    return history.snapshot()


@pytest.mark.parametrize('attack', ['counter_float', 'counter_bool', 'counter_negative',
    'count_exceeds_total', 'streak_exceeds_count', 'impulse_count_disagree', 'impulse_nan',
    'missing_region', 'missing_pair', 'extra_region', 'clock', 'sample_count', 'window_count',
    'coverage_true', 'coverage_zero', 'coverage_integer', 'private_extra', 'private_region_extra',
    'coactivity_above_zero_pad', 'action_path', 'reason_on_valid'])
def test_public_schema_rejects_false_available_claims(available, attack):
    result = deepcopy(available)
    thumb = result['regions']['thumb_pad']
    if attack == 'counter_float': thumb['contact_substeps'] = 40.
    elif attack == 'counter_bool': thumb['contact_substeps'] = True
    elif attack == 'counter_negative': thumb['contact_substeps'] = -1
    elif attack == 'count_exceeds_total': thumb['contact_substeps'] = 41
    elif attack == 'streak_exceeds_count': thumb['current_contact_substeps'] = 41
    elif attack == 'impulse_count_disagree': thumb['normal_impulse_sum_ns'] = 0.
    elif attack == 'impulse_nan': thumb['normal_impulse_sum_ns'] = float('nan')
    elif attack == 'missing_region': del result['regions']['arm']
    elif attack == 'missing_pair': del result['pad_coactivity']['thumb+index']
    elif attack == 'extra_region': result['regions']['private_actor'] = deepcopy(thumb)
    elif attack == 'clock': result['sim_time_s'] = .04
    elif attack == 'sample_count': result['sampled_substeps'] = 80
    elif attack == 'window_count': result['observed_control_windows'] = 2
    elif attack == 'coverage_true': result['coverage']['cloth'] = True
    elif attack == 'coverage_zero': result['coverage']['cloth'] = 0
    elif attack == 'coverage_integer': result['coverage']['all_declared_robot_rigid_links'] = 1
    elif attack == 'private_extra': result['actor0'] = '/World/private_canary'
    elif attack == 'private_region_extra': thumb['actor0'] = '/World/private_canary'
    elif attack == 'coactivity_above_zero_pad': result['pad_coactivity']['thumb+index'] = {'contact_substeps':1, 'current_contact_substeps':0, 'longest_contact_substeps':1}
    elif attack == 'action_path': result['action_id'] = '/World/private_canary'
    elif attack == 'reason_on_valid': result['reason'] = 'invalid_rigid_contact_window'
    with pytest.raises(ValidationError): ContactHistoryObservation.model_validate(result)


@pytest.mark.parametrize('state', ['not_sampled', 'unavailable', 'invalid'])
def test_unknown_summary_cannot_be_reported_as_measured_zero(config, state):
    history = ContactHistory(config)
    if state == 'unavailable': history = ContactHistory({**config, 'available':False})
    elif state == 'invalid': history.observe(projection.window(2))
    result = history.snapshot()
    result['regions'] = {name:{'contact_substeps':0,'current_contact_substeps':0,
        'longest_contact_substeps':0,'normal_impulse_sum_ns':0.} for name in REGIONS}
    with pytest.raises(ValidationError): ContactHistoryObservation.model_validate(result)


@pytest.mark.parametrize('attack', ['invented_windows', 'prefix_substep_mismatch', 'missing_configured_dt'])
def test_invalid_history_still_cannot_invent_observed_prefix(config, attack):
    history = ContactHistory(config)
    history.observe(projection.window())
    history.observe(projection.window(3))
    result = history.snapshot()
    assert result['status'] == 'invalid' and result['observed_control_windows'] == 1
    if attack == 'invented_windows':
        result['observed_control_windows'] = 500
        result['sampled_substeps'] = 20000
    elif attack == 'prefix_substep_mismatch': result['sampled_substeps'] = 41
    else: result['physics_dt_s'] = None
    with pytest.raises(ValidationError): ContactHistoryObservation.model_validate(result)
