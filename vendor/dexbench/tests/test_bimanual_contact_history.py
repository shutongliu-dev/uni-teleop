"""Native-shaped CPU evidence for identity-free bimanual action histories."""
from copy import deepcopy
import json

import pytest
from pydantic import ValidationError

from bench.protocol.schemas import ActionEvent
from bench.sensors.contact_history import BIMANUAL_PROTOCOL, ContactHistory, configure_contact_history
from bench.sensors.contact_schema import (BimanualContactHistoryObservation,
    ContactHistoryObservation, MAX_BIMANUAL_CONTACT_HISTORY_BYTES, parse_contact_history)
from bench.sensors.tactile import FINGER_LINKS, NATIVE_SOURCE
from test_contact_history import contact, window


@pytest.fixture
def dual_metadata():
    links = []
    for side in ('left', 'right'):
        links += [side + '_' + name.split('_', 1)[1] for name in FINGER_LINKS.values()]
        links += [side + suffix for suffix in ('_index_PP', '_index_DP', '_hand_flange', '_hand_wrist')]
        links += [f'fr3_{side}_link{i}' for i in range(3)]
    paths = {name: '/World/Robot/' + ('Left' if 'left' in name else 'Right') + '/private_robot_canary/' + name
             for name in links}
    sensors = list(paths.values()) + ['/World/Objects/private_object_canary']
    return {'native_isaacsim_app': True, 'physics_dt_s': .0005, 'control_dt_s': .02,
        'hand_spec': {'hand_id': 'dual_fr3_sharpa_w01'}, 'body_names': links + ['private_object_canary'],
        'object_names': ['private_object_canary'], 'scene_info': {'robot_link_paths': paths},
        'contact_evidence': {'source': NATIVE_SOURCE, 'impulse_components': 'normal_only',
            'aggregation': {'version': 'float64_point_normal_impulses_v1'}, 'sensor_paths': sensors,
            'filter_paths': [list(sensors) for _ in sensors]}}


@pytest.fixture
def dual_config(dual_metadata):
    result = configure_contact_history(dual_metadata, 'dual_fr3_sharpa_w01')
    assert result['available'] and result['protocol'] == BIMANUAL_PROTOCOL
    return result


def sampled_window(number=1, rows=()):
    value = window(number, rows)
    value['contacts']['sampling'] = {'protocol': 'native_rigid_contact_sampling_v1',
        'physics_dt_s': .0005, 'start_sim_time_s': (number-1)*.02, 'end_sim_time_s': number*.02,
        'native_start_time_s': 1.+(number-1)*.02, 'native_end_time_s': 1.+number*.02,
        'native_start_step_index': 2000+(number-1)*40, 'native_end_step_index': 2000+number*40,
        'sampled_substeps': 40, 'complete': True}
    return value


def path(config, link):
    return next(value for value in config['regions_by_path'] if value.endswith('/' + link))


def checked(history):
    value = history.snapshot()
    assert parse_contact_history(value).model_dump(mode='json') == value
    assert len(json.dumps(value).encode()) < MAX_BIMANUAL_CONTACT_HISTORY_BYTES
    return value


def test_sides_and_global_robot_pairs_count_once_with_overlapping_contacts(dual_config):
    config = dual_config
    rows = [contact(config, 'left_thumb_elastomer', impulse=.001),
            contact(config, 'right_thumb_elastomer', impulse=.002),
            contact(config, 'left_index_elastomer', tick=2, impulse=.003,
                    other=path(config, 'right_index_elastomer')),
            contact(config, 'left_index_PP', tick=2, impulse=.004,
                    other=path(config, 'left_index_DP')),
            contact(config, 'fr3_right_link1', tick=2, impulse=.005,
                    other=path(config, 'right_hand_wrist'))]
    history = ContactHistory(config); history.begin('dual', 0.); history.observe(sampled_window(rows=rows))
    result = checked(history)
    left, right = (result['hands'][side] for side in ('left', 'right'))
    assert left['regions']['thumb_pad']['normal_impulse_sum_ns'] == .001
    assert right['regions']['thumb_pad']['normal_impulse_sum_ns'] == .002
    assert left['regions']['index_pad']['normal_impulse_sum_ns'] == .003
    assert right['regions']['index_pad']['normal_impulse_sum_ns'] == .003
    assert left['regions']['index_shell']['normal_impulse_sum_ns'] == .004  # one region, not twice
    assert right['regions']['arm']['normal_impulse_sum_ns'] == .005
    assert left['self_contact']['normal_impulse_sum_ns'] == .004
    assert right['self_contact']['normal_impulse_sum_ns'] == .005
    assert result['cross_side_contact']['normal_impulse_sum_ns'] == .003
    assert result['self_contact']['normal_impulse_sum_ns'] == pytest.approx(.012)
    assert result['self_contact']['contact_substeps'] == 1  # simultaneous three robot pairs
    assert left['pad_coactivity']['thumb+index']['contact_substeps'] == 0
    assert right['pad_coactivity']['thumb+index']['contact_substeps'] == 0


def test_same_tick_pad_coactivity_does_not_infer_cross_hand_contact(dual_config):
    rows = [contact(dual_config, name, tick=10) for name in (
        'left_thumb_elastomer', 'left_index_elastomer', 'right_thumb_elastomer', 'right_index_elastomer')]
    history = ContactHistory(dual_config); history.observe(sampled_window(rows=rows)); result = checked(history)
    assert all(hand['pad_coactivity']['thumb+index']['contact_substeps'] == 1 for hand in result['hands'].values())
    assert result['cross_side_contact']['contact_substeps'] == result['self_contact']['contact_substeps'] == 0


def test_cross_arm_contact_is_named_cross_side_not_claimed_as_a_grasp(dual_config):
    row = contact(dual_config, 'fr3_left_link1', other=path(dual_config, 'fr3_right_link1'))
    history = ContactHistory(dual_config); history.observe(sampled_window(rows=[row])); result = checked(history)
    assert result['cross_side_contact']['contact_substeps'] == 1
    assert all(hand['regions']['arm']['contact_substeps'] == 1 for hand in result['hands'].values())
    assert all(hand['regions']['thumb_pad']['contact_substeps'] == 0 for hand in result['hands'].values())


def test_acknowledged_windows_preserve_cross_side_streak_and_reset_only_on_new_action(dual_config):
    history = ContactHistory(dual_config); history.begin('first', 0.)
    for number, ticks in [(1, range(21, 41)), (2, range(1, 11))]:
        rows = [contact(dual_config, 'left_thumb_elastomer', window=number, tick=tick,
                        other=path(dual_config, 'right_index_elastomer')) for tick in ticks]
        history.observe(sampled_window(number, rows))
    result = checked(history)
    assert result['sampled_substeps'] == 80 and result['observed_control_windows'] == 2
    assert result['cross_side_contact']['contact_substeps'] == 30
    assert result['cross_side_contact']['longest_contact_substeps'] == 30
    assert result['cross_side_contact']['current_contact_substeps'] == 0
    copy = history.snapshot(); copy['hands']['left']['regions']['thumb_pad']['contact_substeps'] = 99
    assert checked(history) == result
    history.begin('second', .04)
    assert checked(history)['status'] == 'not_sampled'
    history.observe(sampled_window(3))
    assert checked(history)['cross_side_contact']['contact_substeps'] == 0


@pytest.mark.parametrize('damage', ['missing_sampling', 'incomplete_sampling', 'wrong_sample_count',
    'bool_count', 'clock_end', 'clock_start', 'native_duration', 'native_index', 'native_reset',
    'missing_contacts', 'sensor_error', 'missing_window', 'repeated_window', 'unmapped_left',
    'unmapped_right', 'nan_impulse', 'zero_with_positive_count', 'reversed_duplicate'])
def test_invalid_or_missing_evidence_is_unknown_for_both_sides_until_next_action(dual_config, damage):
    history = ContactHistory(dual_config)
    history.observe(sampled_window(rows=[contact(dual_config, 'left_thumb_elastomer')]))
    raw = sampled_window(2, [contact(dual_config, 'right_thumb_elastomer', window=2)])
    sampling = raw['contacts']['sampling']
    if damage == 'missing_sampling': del raw['contacts']['sampling']
    elif damage == 'incomplete_sampling': sampling['complete'] = False
    elif damage == 'wrong_sample_count': sampling['sampled_substeps'] = 39
    elif damage == 'bool_count': sampling['sampled_substeps'] = True
    elif damage == 'clock_end': sampling['end_sim_time_s'] = .02
    elif damage == 'clock_start': sampling['start_sim_time_s'] = 0.
    elif damage == 'native_duration': sampling['native_end_time_s'] += .01
    elif damage == 'native_index': sampling['native_end_step_index'] -= 1
    elif damage == 'native_reset':
        sampling['native_start_step_index'] -= 40; sampling['native_end_step_index'] -= 40
    elif damage == 'missing_contacts': del raw['contacts']
    elif damage == 'sensor_error': raw['contacts']['errors'] = ['private_error_canary']
    elif damage == 'missing_window': raw = sampled_window(3)
    elif damage == 'repeated_window': raw = sampled_window(1)
    elif damage.startswith('unmapped_'):
        raw['contacts']['rigid'][0]['actor0'] = '/World/Robot/' + damage.split('_')[1].title() + '/unmapped'
    elif damage == 'nan_impulse': raw['contacts']['rigid'][0]['impulse_magnitude_sum_ns'] = float('nan')
    elif damage == 'zero_with_positive_count':
        raw['contacts']['rigid'][0]['impulse_magnitude_sum_ns'] = 0.
        raw['contacts']['rigid'][0]['impulse_vector_ns'] = [0., 0., 0.]
    else:
        extra = deepcopy(raw['contacts']['rigid'][0]); extra['actor0'], extra['actor1'] = extra['actor1'], extra['actor0']
        raw['contacts']['rigid'].append(extra)
    history.observe(raw)
    result = checked(history)
    assert result['status'] == 'invalid' and result['sampled_substeps'] == 40
    assert result['hands'] is result['self_contact'] is result['cross_side_contact'] is None
    history.observe(sampled_window(4)); assert checked(history)['status'] == 'invalid'
    history.begin('fresh', .08); history.observe(sampled_window(5)); assert checked(history)['status'] == 'available'


@pytest.mark.parametrize('damage', ['missing_pad', 'missing_body', 'unknown_link', 'side_path_swapped',
    'aliased_path', 'duplicate_sensor', 'missing_filter', 'missing_cross_side_filter', 'unknown_robot_sensor'])
def test_incomplete_or_ambiguous_dual_link_coverage_disables_entire_history(dual_metadata, damage):
    metadata = deepcopy(dual_metadata)
    paths = metadata['scene_info']['robot_link_paths']; evidence = metadata['contact_evidence']
    if damage == 'missing_pad': del paths['left_thumb_elastomer']
    elif damage == 'missing_body': metadata['body_names'].remove('right_index_PP')
    elif damage == 'unknown_link':
        paths['left_index_unknown'] = '/World/Robot/Left/left_index_unknown'
        metadata['body_names'].append('left_index_unknown')
    elif damage == 'side_path_swapped': paths['left_index_PP'] = paths['left_index_PP'].replace('/Left/', '/Right/')
    elif damage == 'aliased_path': paths['right_index_PP'] = paths['right_index_PP'].replace('/Right/', '/Right/../Right/')
    elif damage == 'duplicate_sensor': evidence['sensor_paths'].append(paths['right_index_PP'])
    elif damage == 'missing_filter': del evidence['filter_paths']
    elif damage == 'missing_cross_side_filter': evidence['filter_paths'][0].remove(paths['right_index_PP'])
    else: evidence['sensor_paths'].append('/World/Robot/Left/unknown')
    config = configure_contact_history(metadata, 'dual_fr3_sharpa_w01')
    assert not config['available']
    history = ContactHistory(config); history.observe(sampled_window())
    result = checked(history)
    assert result['status'] == 'unavailable' and result['hands'] is None


def test_public_projection_is_bounded_identity_free_and_does_not_mutate_evidence(dual_config):
    original = deepcopy(dual_config)
    raw = sampled_window(rows=[contact(dual_config, 'left_thumb_elastomer')])
    original_raw = deepcopy(raw)
    history = ContactHistory(dual_config); history.begin('action', 0.); history.observe(raw)
    value = checked(history)
    event = ActionEvent(action_id='action', status='completed', sim_time_s=.02, wall_time_s=.1,
                        contact_history=value).model_dump(mode='json')
    text = json.dumps(event)
    for secret in ('private_', '/World/', 'actor0', 'actor1', 'sensor_paths', 'object_id', 'filter_paths'):
        assert secret not in text
    assert original == dual_config and original_raw == raw
    assert isinstance(parse_contact_history(value), BimanualContactHistoryObservation)
    with pytest.raises(ValidationError): ContactHistoryObservation.model_validate(value)


@pytest.mark.parametrize('damage', ['unknown_protocol', 'private_field', 'private_region', 'missing_hand',
    'missing_region', 'false_zero', 'wrong_global_impulse', 'wrong_global_count', 'wrong_cross_count'])
def test_strict_v2_schema_rejects_partial_or_forged_public_summaries(dual_config, damage):
    history = ContactHistory(dual_config)
    history.observe(sampled_window(rows=[contact(dual_config, 'left_index_PP',
        other=path(dual_config, 'right_index_DP'))]))
    value = checked(history)
    if damage == 'unknown_protocol': value['protocol'] = 'future'
    elif damage == 'private_field': value['actor0'] = 'private'
    elif damage == 'private_region': value['hands']['left']['regions']['arm']['actor0'] = 'private'
    elif damage == 'missing_hand': del value['hands']['left']
    elif damage == 'missing_region': del value['hands']['left']['regions']['arm']
    elif damage == 'false_zero': value['status'] = 'invalid'; value['reason'] = 'invalid_rigid_contact_window'
    elif damage == 'wrong_global_impulse': value['self_contact']['normal_impulse_sum_ns'] *= 2
    elif damage == 'wrong_global_count': value['self_contact'].update(contact_substeps=2, longest_contact_substeps=2)
    else: value['cross_side_contact'].update(contact_substeps=41, longest_contact_substeps=41)
    with pytest.raises(ValidationError): parse_contact_history(value)


def test_public_history_size_limit_rejects_oversized_but_consistent_counters(dual_config):
    history = ContactHistory(dual_config); history.observe(sampled_window())
    value = checked(history)
    windows = 10**300
    samples = 40 * windows
    value.update(observed_control_windows=windows, sampled_substeps=samples,
                 duration_s=windows*.02, sim_time_s=windows*.02)
    for hand in value['hands'].values():
        for region in hand['regions'].values():
            region.update(contact_substeps=samples, current_contact_substeps=samples,
                          longest_contact_substeps=samples, normal_impulse_sum_ns=1.)
        for pair in hand['pad_coactivity'].values():
            pair.update(contact_substeps=samples, current_contact_substeps=samples,
                        longest_contact_substeps=samples)
    with pytest.raises(ValidationError, match='payload limit'):
        parse_contact_history(value)
