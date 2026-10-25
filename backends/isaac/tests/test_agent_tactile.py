"""CPU-only contract tests for the identity-free native contact projection."""
from copy import deepcopy
import json
import math

import pytest

from bench.protocol.schemas import TactileObservation
from bench.sensors.tactile import (
    FINGER_LINKS, NATIVE_SOURCE, NORMAL_TOLERANCE, SOURCE,
    project_tactile, tactile_configuration,
)


@pytest.fixture
def metadata():
    paths = {link: '/World/Robot/' + link for link in FINGER_LINKS.values()}
    return {'native_isaacsim_app': True, 'physics_dt_s': .0005, 'control_dt_s': .02,
            'hand_spec': {'hand_id': 'sharpa_w01_right'},
            'scene_info': {'robot_link_paths': paths}, 'body_names': list(paths),
            'contact_evidence': {'source': NATIVE_SOURCE, 'impulse_components': 'normal_only',
                                 'aggregation': {'version': 'float64_point_normal_impulses_v1'},
                                 'sensor_paths': list(paths.values())}}


@pytest.fixture
def config(metadata):
    return tactile_configuration(metadata, 'sharpa_w01_right')


@pytest.fixture
def raw():
    return {'sim_time_s': .02, 'last_duration_s': .02,
            'contacts': {'source': SOURCE, 'errors': [], 'rigid': []}}


def row(config, *, finger='thumb', time=.0005, impulse=.0005,
        counterpart='/World/Objects/private_canary'):
    return {'time_s': time, 'actor0': config['sensor_paths'][finger], 'actor1': counterpart,
            'point_count': 1, 'nonzero_impulse_normal_count': int(impulse != 0),
            'normal_length_max_abs_error': 0., 'impulse_components': 'normal_only',
            'impulse_vector_ns': [0., 0., impulse], 'impulse_magnitude_sum_ns': impulse,
            'raw_scalar_abs_impulse_sum_ns': impulse, 'min_separation_m': -.001,
            'collider_identity_available': False}


def assert_missing(result, status):
    assert result['status'] == status
    assert result['window_start_sim_s'] is None
    assert result['window_duration_s'] is None
    assert result['window_completeness'] == 'not_available'
    for finger in result['fingers'].values():
        assert finger == {'status': status, 'rigid_contact_any': None,
                          'normal_impulse_sum_ns': None, 'mean_normal_load_n': None}
    TactileObservation.model_validate(result)


def test_native_metadata_requires_all_five_actual_elastomer_actors(metadata):
    original = deepcopy(metadata)
    config = tactile_configuration(metadata, 'sharpa_w01_right')
    assert config['available'] is True
    assert config['sensor_paths']['little'] == '/World/Robot/right_pinky_elastomer'
    assert config['native_physics_dt_s'] == .0005
    assert config['coverage'] == {'rigid': True, 'cloth': False, 'self_contacts': True,
                                  'region': 'whole_elastomer_collision_body',
                                  'taxels': False, 'shear': False, 'torque': False}
    assert metadata == original


@pytest.mark.parametrize('finger', list(FINGER_LINKS))
@pytest.mark.parametrize('missing', ['body', 'sensor', 'link', 'duplicate_sensor', 'duplicate_body', 'virtual_tip'])
def test_partial_or_ambiguous_mapping_disables_every_channel(metadata, raw, finger, missing):
    link = FINGER_LINKS[finger]
    path = metadata['scene_info']['robot_link_paths'][link]
    if missing == 'body':
        metadata['body_names'].remove(link)
    elif missing == 'sensor':
        metadata['contact_evidence']['sensor_paths'].remove(path)
    elif missing == 'link':
        del metadata['scene_info']['robot_link_paths'][link]
    elif missing == 'duplicate_sensor':
        metadata['contact_evidence']['sensor_paths'].append(path)
    elif missing == 'duplicate_body':
        metadata['body_names'].append(link)
    else:
        metadata['scene_info']['robot_link_paths'][link] = path.replace('elastomer', 'fingertip')
    config = tactile_configuration(metadata, 'sharpa_w01_right')
    assert config['available'] is False
    assert config['sensor_paths'] == {}
    assert config['coverage']['rigid'] is False
    assert_missing(project_tactile(raw, config), 'unavailable')


@pytest.mark.parametrize('field,value', [
    ('native_isaacsim_app', False), ('hand_spec', {'hand_id': 'sharpa_w02_right'}),
    ('physics_dt_s', 0), ('physics_dt_s', -.0005), ('physics_dt_s', float('nan')),
    ('physics_dt_s', .0007), ('control_dt_s', .01), ('control_dt_s', True),
    ('contact_evidence', {}), ('body_names', []), ('scene_info', {}),
])
def test_unsupported_or_missing_native_metadata_disables_proxy(metadata, field, value):
    metadata[field] = value
    assert tactile_configuration(metadata, 'sharpa_w01_right')['available'] is False


@pytest.mark.parametrize('robot_id', ['sharpa_w02_right', 'sharpa_w01_left', 'fake'])
def test_other_robot_does_not_inherit_hardware_capability(metadata, robot_id):
    assert tactile_configuration(metadata, robot_id)['available'] is False


@pytest.mark.parametrize('field,value', [('source', 'proximity'), ('impulse_components', 'full_wrench'),
                                       ('aggregation', {'version': 'legacy_float32'})])
def test_unsupported_contact_metadata_disables_proxy(metadata, field, value):
    metadata['contact_evidence'][field] = value
    assert tactile_configuration(metadata, 'sharpa_w01_right')['available'] is False


def test_reset_missing_and_measured_zero_are_different(raw, config):
    raw.update(sim_time_s=0., last_duration_s=0.)
    assert_missing(project_tactile(raw, config), 'not_sampled')
    raw.update(sim_time_s=.02, last_duration_s=.02)
    result = project_tactile(raw, config)
    assert result['status'] == 'available'
    assert result['window_start_sim_s'] == 0.
    assert result['window_duration_s'] == .02
    assert result['window_completeness'] == 'assumed_from_native_step_contract'
    assert result['coverage']['cloth'] is False
    assert result['force_calibrated'] is False
    for finger in result['fingers'].values():
        assert finger == {'status': 'valid', 'rigid_contact_any': False,
                          'normal_impulse_sum_ns': 0., 'mean_normal_load_n': 0.}
    TactileObservation.model_validate(result)
    del raw['contacts']
    assert_missing(project_tactile(raw, config), 'invalid')


def test_window_impulse_divides_by_control_period_and_never_exposes_identity(raw, config):
    raw['contacts']['rigid'] = [row(config, time=tick * .0005) for tick in range(1, 41)]
    before_raw, before_config = deepcopy(raw), deepcopy(config)
    result = project_tactile(raw, config)
    finger = result['fingers']['thumb']
    assert finger['normal_impulse_sum_ns'] == pytest.approx(.02)
    assert finger['mean_normal_load_n'] == pytest.approx(1.)
    assert finger['rigid_contact_any'] is True
    assert result['frame_id'] == 'scalar_invariant'
    serialized = json.dumps(result)
    for secret in ['private_canary', '/World', 'actor', 'collider', 'separation', 'point_count', 'sensor_paths']:
        assert secret not in serialized
    assert raw == before_raw and config == before_config
    TactileObservation.model_validate(result)


def test_last_control_interval_only_after_longer_agent_action(raw, config):
    raw['sim_time_s'] = .5
    raw['contacts']['rigid'] = [row(config, time=.5, impulse=.002)]
    result = project_tactile(raw, config)
    assert result['window_start_sim_s'] == .48
    assert result['window_duration_s'] == .02
    assert result['fingers']['thumb']['mean_normal_load_n'] == .1


@pytest.mark.parametrize('suffix', ['/collision', '_copy', '_private'])
def test_path_matching_is_exact_no_substring_sensor_aliases(raw, config, suffix):
    item = row(config)
    item['actor0'] += suffix
    raw['contacts']['rigid'] = [item]
    result = project_tactile(raw, config)
    assert result['status'] == 'available'
    assert all(f['normal_impulse_sum_ns'] == 0 for f in result['fingers'].values())


def test_pad_self_contact_counts_once_per_finger_and_is_orientation_invariant(raw, config):
    item = row(config, impulse=.004, counterpart=config['sensor_paths']['index'])
    raw['contacts']['rigid'] = [item]
    result = project_tactile(raw, config)
    for finger in ('thumb', 'index'):
        assert result['fingers'][finger]['mean_normal_load_n'] == .2
    assert result['fingers']['middle']['mean_normal_load_n'] == 0.
    item['actor0'], item['actor1'] = item['actor1'], item['actor0']
    item['impulse_vector_ns'] = [0., 0., -.004]
    assert project_tactile(raw, config) == result


@pytest.mark.parametrize('reverse', [False, True])
def test_same_pair_same_substep_duplicate_is_invalid_not_double_counted(raw, config, reverse):
    item = row(config)
    duplicate = deepcopy(item)
    if reverse:
        duplicate['actor0'], duplicate['actor1'] = duplicate['actor1'], duplicate['actor0']
    raw['contacts']['rigid'] = [item, duplicate]
    assert_missing(project_tactile(raw, config), 'invalid')


@pytest.mark.parametrize('field,value', [
    ('time_s', -.0005), ('time_s', 0.), ('time_s', .0205), ('time_s', .0006),
    ('time_s', float('nan')), ('impulse_magnitude_sum_ns', -1.),
    ('impulse_magnitude_sum_ns', float('inf')), ('impulse_magnitude_sum_ns', True),
    ('impulse_vector_ns', [0., 0., .001]), ('impulse_vector_ns', [0., 0., float('nan')]),
    ('impulse_vector_ns', [0., 0.]), ('impulse_vector_ns', [False, 0., 0.]),
    ('point_count', 0), ('point_count', True), ('point_count', 1.),
    ('nonzero_impulse_normal_count', 0), ('nonzero_impulse_normal_count', 2),
    ('normal_length_max_abs_error', NORMAL_TOLERANCE * 2),
    ('normal_length_max_abs_error', -.001), ('impulse_components', 'full_wrench'),
    ('actor1', None),
])
def test_malformed_native_row_is_missing_data_never_zero(raw, config, field, value):
    item = row(config)
    item[field] = value
    raw['contacts']['rigid'] = [item]
    assert_missing(project_tactile(raw, config), 'invalid')


@pytest.mark.parametrize('key', ['time_s', 'point_count', 'nonzero_impulse_normal_count',
                                'normal_length_max_abs_error', 'impulse_vector_ns',
                                'impulse_magnitude_sum_ns', 'impulse_components'])
def test_missing_integrity_fields_are_rejected(raw, config, key):
    item = row(config)
    del item[key]
    raw['contacts']['rigid'] = [item]
    assert_missing(project_tactile(raw, config), 'invalid')


@pytest.mark.parametrize('case', ['short_window', 'long_window', 'no_duration', 'source', 'errors',
                                 'missing_errors', 'missing_rows', 'bad_row', 'same_actor', 'reset_rows'])
def test_incomplete_or_errored_window_is_invalid(raw, config, case):
    if case in ('short_window', 'long_window'):
        raw['last_duration_s'] = .01 if case == 'short_window' else .04
    elif case == 'no_duration':
        del raw['last_duration_s']
    elif case == 'source':
        raw['contacts']['source'] = 'geometric proximity'
    elif case == 'errors':
        raw['contacts']['errors'] = ['overflow']
    elif case == 'missing_errors':
        del raw['contacts']['errors']
    elif case == 'missing_rows':
        del raw['contacts']['rigid']
    elif case == 'bad_row':
        raw['contacts']['rigid'] = [None]
    elif case == 'same_actor':
        raw['contacts']['rigid'] = [row(config, counterpart=config['sensor_paths']['thumb'])]
    else:
        raw.update(sim_time_s=0., last_duration_s=0.)
        raw['contacts']['rigid'] = [row(config)]
    assert_missing(project_tactile(raw, config), 'invalid')


def test_contact_candidates_with_zero_solved_impulse_are_not_loaded(raw, config):
    raw['contacts']['rigid'] = [row(config, impulse=0.)]
    result = project_tactile(raw, config)
    assert result['status'] == 'available'
    assert result['fingers']['thumb']['rigid_contact_any'] is False
    assert result['fingers']['thumb']['mean_normal_load_n'] == 0.


def test_force_overflow_is_invalid(raw, config):
    raw['contacts']['rigid'] = [row(config, impulse=1e308)]
    assert_missing(project_tactile(raw, config), 'invalid')


@pytest.mark.parametrize('period', [.01, .04, 0., math.inf, math.nan, True])
def test_caller_cannot_relabel_native_window(raw, config, period):
    assert_missing(project_tactile(raw, config, control_period_s=period), 'invalid')


@pytest.mark.parametrize('time', [None, -1., math.inf, math.nan, True])
def test_missing_observation_time_is_never_invented(raw, config, time):
    raw['sim_time_s'] = time
    with pytest.raises(ValueError, match='observation time'):
        project_tactile(raw, config)


@pytest.fixture
def available_tactile(raw, config):
    raw['contacts']['rigid'] = [row(config, impulse=.0002)]
    return project_tactile(raw, config)


@pytest.mark.parametrize('missing', [None, *FINGER_LINKS])
def test_public_schema_requires_exactly_five_channels(available_tactile, missing):
    if missing is None:
        available_tactile['fingers'] = {}
    else:
        del available_tactile['fingers'][missing]
    with pytest.raises(ValueError, match='exactly all five'):
        TactileObservation.model_validate(available_tactile)


def test_public_schema_rejects_unknown_channel(available_tactile):
    available_tactile['fingers']['pinky'] = available_tactile['fingers']['little']
    with pytest.raises(ValueError):
        TactileObservation.model_validate(available_tactile)


@pytest.mark.parametrize('status', ['unavailable', 'not_sampled', 'invalid'])
def test_public_schema_rejects_mixed_finger_status(available_tactile, status):
    available_tactile['fingers']['thumb'] = {'status': status}
    with pytest.raises(ValueError, match='statuses'):
        TactileObservation.model_validate(available_tactile)


@pytest.mark.parametrize('case', ['unavailable', 'invalid', 'not_sampled'])
@pytest.mark.parametrize('corruption', ['window_start', 'window_duration', 'completeness', 'coverage', 'valid_finger'])
def test_missing_readings_cannot_claim_measurements(raw, config, case, corruption):
    if case == 'unavailable':
        config['available'] = False
    elif case == 'invalid':
        raw['contacts']['errors'] = ['overflow']
    else:
        raw.update(sim_time_s=0., last_duration_s=0.)
    result = project_tactile(raw, config)
    assert_missing(result, case)
    if corruption == 'window_start':
        result['window_start_sim_s'] = 0.
    elif corruption == 'window_duration':
        result['window_duration_s'] = .02
    elif corruption == 'completeness':
        result['window_completeness'] = 'assumed_from_native_step_contract'
    elif corruption == 'coverage':
        result['coverage']['rigid'] = not result['coverage']['rigid']
    else:
        result['fingers']['thumb'] = {'status': 'valid', 'rigid_contact_any': False,
                                     'normal_impulse_sum_ns': 0., 'mean_normal_load_n': 0.}
    with pytest.raises(ValueError):
        TactileObservation.model_validate(result)


def test_not_sampled_is_reset_not_a_later_missing_window(raw, config):
    raw.update(sim_time_s=0., last_duration_s=0.)
    result = project_tactile(raw, config)
    result['sim_time_s'] = .02
    with pytest.raises(ValueError, match='reset'):
        TactileObservation.model_validate(result)


@pytest.mark.parametrize('status', ['available', 'unavailable', 'invalid', 'not_sampled'])
def test_public_http_omitted_nulls_remain_compatible(raw, config, status):
    if status == 'unavailable':
        config['available'] = False
    elif status == 'invalid':
        raw['contacts']['errors'] = ['overflow']
    elif status == 'not_sampled':
        raw.update(sim_time_s=0., last_duration_s=0.)
    projected = project_tactile(raw, config)
    assert projected['status'] == status

    def omit_nulls(value):
        return ({key: omit_nulls(item) for key, item in value.items() if item is not None}
                if isinstance(value, dict) else value)

    wire = omit_nulls(projected)
    parsed = TactileObservation.model_validate_json(json.dumps(wire))
    assert parsed.model_dump(mode='json', exclude_unset=True) == wire
    assert parsed.model_dump(mode='json') == projected


@pytest.mark.parametrize('key,value', [
    ('window_start_sim_s', None), ('window_duration_s', None),
    ('window_completeness', 'not_available'),
    ('window_start_sim_s', .001), ('window_start_sim_s', .03),
    ('window_duration_s', .021), ('window_duration_s', 0.), ('sim_time_s', 0.),
])
def test_available_window_matches_clock(available_tactile, key, value):
    available_tactile[key] = value
    with pytest.raises(ValueError, match='window'):
        TactileObservation.model_validate(available_tactile)


def test_available_requires_supported_coverage(available_tactile):
    available_tactile['coverage']['rigid'] = False
    with pytest.raises(ValueError, match='coverage'):
        TactileObservation.model_validate(available_tactile)


@pytest.mark.parametrize('time', [.02 - 5e-9, .02, .5, 3600.])
def test_schema_preserves_projector_clock_rounding_and_wire_bytes(raw, config, time):
    raw['sim_time_s'] = time
    result = project_tactile(raw, config)
    assert result['status'] == 'available'
    before = deepcopy(result)
    parsed = TactileObservation.model_validate(result)
    assert parsed.model_dump(mode='json') == before
    assert TactileObservation.model_validate_json(json.dumps(result)).model_dump(mode='json') == before
    assert result == before


@pytest.mark.parametrize('case', ['positive_false', 'zero_true', 'zero_load', 'zero_impulse', 'bad_units'])
def test_public_impulse_load_and_contact_are_consistent(available_tactile, case):
    finger = available_tactile['fingers']['thumb']
    if case == 'positive_false':
        finger['rigid_contact_any'] = False
    elif case == 'zero_true':
        finger.update(normal_impulse_sum_ns=0., mean_normal_load_n=0.)
    elif case == 'zero_load':
        finger['mean_normal_load_n'] = 0.
    elif case == 'zero_impulse':
        finger.update(normal_impulse_sum_ns=0., rigid_contact_any=False)
    else:
        finger['mean_normal_load_n'] = 100.
    with pytest.raises(ValueError):
        TactileObservation.model_validate(available_tactile)


@pytest.mark.parametrize('impulse', [0., 5e-324, .0002, 1e100])
def test_legal_zero_and_extreme_finite_readings_are_preserved(raw, config, impulse):
    raw['contacts']['rigid'] = [row(config, impulse=impulse)]
    result = project_tactile(raw, config)
    assert result['status'] == 'available'
    assert TactileObservation.model_validate(result).model_dump(mode='json') == result


def test_units_check_does_not_use_absolute_zero_force_slack(available_tactile):
    available_tactile['fingers']['thumb'].update(normal_impulse_sum_ns=1e-20, mean_normal_load_n=1e-10)
    with pytest.raises(ValueError, match='divided by window duration'):
        TactileObservation.model_validate(available_tactile)


def test_units_check_allows_small_relative_serialization_roundoff(available_tactile):
    available_tactile['fingers']['thumb']['mean_normal_load_n'] *= 1. + 1e-12
    TactileObservation.model_validate(available_tactile)


@pytest.mark.parametrize('key', ['sim_time_s', 'window_start_sim_s', 'window_duration_s',
                               'normal_impulse_sum_ns', 'mean_normal_load_n'])
@pytest.mark.parametrize('value', [math.nan, math.inf, -math.inf, -1., True, '0.02'])
def test_public_tactile_numbers_must_be_finite_and_not_coerced(available_tactile, key, value):
    target = available_tactile if key.endswith('_s') else available_tactile['fingers']['thumb']
    target[key] = value
    with pytest.raises(ValueError):
        TactileObservation.model_validate(available_tactile)


@pytest.mark.parametrize('key', ['rigid_contact_any', 'rigid', 'cloth', 'self_contacts',
                               'taxels', 'shear', 'torque', 'force_calibrated'])
@pytest.mark.parametrize('value', [0, 1, 0., 1., 'true', 'false'])
def test_public_tactile_flags_require_exact_boolean_types(available_tactile, key, value):
    target = (available_tactile['fingers']['thumb'] if key == 'rigid_contact_any'
              else available_tactile if key == 'force_calibrated' else available_tactile['coverage'])
    target[key] = value
    with pytest.raises(ValueError):
        TactileObservation.model_validate(available_tactile)
