"""Independent CPU counterexamples for identity-free action contact history."""
from copy import deepcopy
import json

import pytest

from bench.sensors.contact_history import ContactHistory, configure_contact_history
from bench.sensors.tactile import FINGER_LINKS, NATIVE_SOURCE, SOURCE, project_tactile


@pytest.fixture
def metadata():
    links = list(FINGER_LINKS.values()) + [
        'right_thumb_DP', 'right_index_PP', 'right_index_DP', 'right_middle_PP',
        'right_ring_PP', 'right_pinky_PP', 'right_hand_wrist', 'fr3_link1', 'fr3_link2']
    paths = {name: '/World/Robot/private_robot_canary/' + name for name in links}
    return {'native_isaacsim_app': True, 'physics_dt_s': .0005, 'control_dt_s': .02,
            'hand_spec': {'hand_id': 'sharpa_w01_right'},
            'body_names': links + ['private_object_canary'],
            'object_names': ['private_object_canary'],
            'scene_info': {'robot_link_paths': paths},
            'contact_evidence': {'source': NATIVE_SOURCE, 'impulse_components': 'normal_only',
                'aggregation': {'version': 'float64_point_normal_impulses_v1'},
                'sensor_paths': list(paths.values())}}


@pytest.fixture
def config(metadata):
    result = configure_contact_history(metadata, 'sharpa_w01_right')
    assert result['available']
    return result


def contact(config, link='right_thumb_elastomer', *, tick=1, window=1, impulse=.001,
            other='/World/Objects/private_object_canary'):
    path = next(p for p in config['regions_by_path'] if p.rsplit('/', 1)[-1] == link)
    return {'time_s': ((window - 1) * 40 + tick) * .0005,
            'actor0': path, 'actor1': other, 'point_count': 1,
            'min_separation_m': 0.,
            'nonzero_impulse_normal_count': int(impulse > 0),
            'normal_length_max_abs_error': 0., 'impulse_components': 'normal_only',
            'impulse_vector_ns': [0., 0., impulse], 'impulse_magnitude_sum_ns': impulse,
            # Native rows always report solver separation; the online rigid
            # safety guard rejects rows without it.
            'min_separation_m': -.00005, 'private_pose_canary': [1., 2., 3.]}


def window(number=1, rows=()):
    return {'sim_time_s': number * .02, 'last_duration_s': .02,
            'contacts': {'source': SOURCE, 'errors': [], 'rigid': list(rows)},
            'private_evaluator_canary': {'success': True}}


def count(result, region):
    return result['regions'][region]


def assert_unknown(result, status):
    assert result['status'] == status
    assert result['regions'] is None
    assert result['pad_coactivity'] is None
    assert result['self_contact'] is None


def test_same_last_window_impulse_distinguishes_synchronous_and_alternating_pads(config):
    synchronized, alternating = [], []
    for tick in range(1, 21):
        synchronized += [contact(config, tick=tick), contact(config, 'right_index_elastomer', tick=tick)]
        alternating += [contact(config, tick=tick), contact(config, 'right_index_elastomer', tick=tick+20)]
    raws = [window(rows=synchronized), window(rows=alternating)]
    assert project_tactile(raws[0], config['tactile']) == project_tactile(raws[1], config['tactile'])
    snapshots = []
    for raw in raws:
        history = ContactHistory(config); history.begin('action_sync', 0.); history.observe(raw)
        snapshots.append(history.snapshot())
    assert snapshots[0]['regions'] != snapshots[1]['regions']  # Current streak differs.
    for result in snapshots:
        for region in ('thumb_pad', 'index_pad'):
            assert count(result, region)['normal_impulse_sum_ns'] == pytest.approx(.02)
            assert count(result, region)['contact_substeps'] == 20
    assert snapshots[0]['pad_coactivity']['thumb+index']['contact_substeps'] == 20
    assert snapshots[1]['pad_coactivity']['thumb+index']['contact_substeps'] == 0


def test_shell_contact_visible_when_all_five_pad_sums_are_zero(config):
    raw = window(rows=[contact(config, 'right_index_DP', tick=i) for i in range(1, 41)])
    history = ContactHistory(config); history.observe(raw); result = history.snapshot()
    assert all(row['normal_impulse_sum_ns'] == 0 for row in project_tactile(raw, config['tactile'])['fingers'].values())
    assert count(result, 'index_shell')['contact_substeps'] == 40
    assert count(result, 'index_shell')['normal_impulse_sum_ns'] == pytest.approx(.04)


def test_cross_window_continuity_and_read_does_not_reset(config):
    history = ContactHistory(config)
    history.begin('continuous', 0.)
    history.observe(window(rows=[contact(config, tick=i) for i in range(21, 41)]))
    first = history.snapshot(); assert history.snapshot() == first
    first['regions']['thumb_pad']['contact_substeps'] = -99
    assert count(history.snapshot(), 'thumb_pad')['contact_substeps'] == 20
    history.observe(window(2, [contact(config, window=2, tick=i) for i in range(1, 11)]))
    result = history.snapshot()
    assert count(result, 'thumb_pad') == {'contact_substeps': 30, 'current_contact_substeps': 0,
        'longest_contact_substeps': 30, 'normal_impulse_sum_ns': pytest.approx(.03)}
    assert result['observed_control_windows'] == 2 and result['sampled_substeps'] == 80
    history.begin('next', .04)
    assert_unknown(history.snapshot(), 'not_sampled')
    history.observe(window(3))
    assert count(history.snapshot(), 'thumb_pad')['contact_substeps'] == 0
    assert history.snapshot()['duration_s'] == pytest.approx(.02)


@pytest.mark.parametrize('damage', ['missing', 'gap', 'out_of_order', 'repeat_window',
    'duplicate_pair', 'reverse_duplicate_pair', 'nan_impulse', 'nan_time', 'nan_vector',
    'short_duration', 'missing_errors', 'unmapped_robot', 'not_dict'])
def test_any_bad_window_invalidates_entire_action_and_cannot_recover(config, damage):
    history = ContactHistory(config)
    history.observe(window(rows=[contact(config)]))
    raw = window(2, [contact(config, window=2)])
    if damage == 'missing': del raw['contacts']
    elif damage == 'gap': raw = window(3)
    elif damage == 'out_of_order': raw['sim_time_s'] = .01
    elif damage == 'repeat_window': raw = window(1)
    elif damage in ('duplicate_pair', 'reverse_duplicate_pair'):
        extra = deepcopy(raw['contacts']['rigid'][0])
        if damage.startswith('reverse'): extra['actor0'], extra['actor1'] = extra['actor1'], extra['actor0']
        raw['contacts']['rigid'].append(extra)
    elif damage == 'nan_impulse': raw['contacts']['rigid'][0]['impulse_magnitude_sum_ns'] = float('nan')
    elif damage == 'nan_time': raw['sim_time_s'] = float('nan')
    elif damage == 'nan_vector': raw['contacts']['rigid'][0]['impulse_vector_ns'][0] = float('nan')
    elif damage == 'short_duration': raw['last_duration_s'] = .01
    elif damage == 'missing_errors': del raw['contacts']['errors']
    elif damage == 'unmapped_robot': raw['contacts']['rigid'][0]['actor0'] = '/World/Robot/unmapped_canary'
    elif damage == 'not_dict': raw = None
    history.observe(raw)
    assert_unknown(history.snapshot(), 'invalid')
    history.observe(window(4))
    assert_unknown(history.snapshot(), 'invalid')
    assert history.snapshot()['sim_time_s'] == .08
    history.begin('fresh_action', .08); history.observe(window(5))
    assert history.snapshot()['status'] == 'available'


def test_missing_unknown_unavailable_and_valid_zero_are_distinct(config, metadata):
    history = ContactHistory(config)
    assert_unknown(history.snapshot(), 'not_sampled')
    history.observe(window()); assert history.snapshot()['status'] == 'available'
    assert all(row['contact_substeps'] == 0 for row in history.snapshot()['regions'].values())
    metadata['object_names'] = []
    unsupported = ContactHistory(configure_contact_history(metadata, 'sharpa_w01_right'))
    unsupported.observe(window())
    assert_unknown(unsupported.snapshot(), 'unavailable')


@pytest.mark.parametrize('links,region', [(('right_index_PP', 'right_index_DP'), 'index_shell'),
    (('fr3_link1', 'fr3_link2'), 'arm')])
def test_robot_self_pair_same_region_counts_once_not_twice(config, links, region):
    other = next(p for p in config['regions_by_path'] if p.endswith('/' + links[1]))
    raw = window(rows=[contact(config, links[0], other=other, impulse=.003)])
    history = ContactHistory(config); history.observe(raw); result = history.snapshot()
    assert count(result, region)['contact_substeps'] == 1
    assert count(result, region)['normal_impulse_sum_ns'] == .003
    assert result['self_contact']['contact_substeps'] == 1
    assert result['self_contact']['normal_impulse_sum_ns'] == .003


def test_two_pairs_one_region_same_tick_sum_impulses_but_one_contact_substep(config):
    raw = window(rows=[contact(config, 'right_index_PP', impulse=.003),
                       contact(config, 'right_index_DP', impulse=.002)])
    history = ContactHistory(config); history.observe(raw)
    assert count(history.snapshot(), 'index_shell')['contact_substeps'] == 1
    assert count(history.snapshot(), 'index_shell')['normal_impulse_sum_ns'] == .005


def test_zero_impulse_pair_is_valid_but_not_contact(config):
    history = ContactHistory(config); history.observe(window(rows=[contact(config, impulse=0.)]))
    assert history.snapshot()['status'] == 'available'
    assert count(history.snapshot(), 'thumb_pad')['contact_substeps'] == 0


def test_privacy_input_immutability_configuration_copy_and_row_order(config, metadata):
    original_meta, original_config = deepcopy(metadata), deepcopy(config)
    raw = window(rows=[contact(config, tick=40), contact(config, tick=1)])
    original_raw = deepcopy(raw)
    history = ContactHistory(config); history.begin('public_action_1', 0.); history.observe(raw)
    result = history.snapshot()
    serialized = json.dumps(result)
    for secret in ('private_robot_canary', 'private_object_canary', 'private_pose_canary',
                   'private_evaluator_canary', '/World/', 'actor0', 'sensor_paths'):
        assert secret not in serialized
    assert raw == original_raw and config == original_config and metadata == original_meta
    config['regions_by_path'].clear()
    history.observe(window(2, [contact(original_config, window=2)]))
    assert history.snapshot()['status'] == 'available'


@pytest.mark.parametrize('start', [float('nan'), float('inf'), True, -.001])
def test_begin_rejects_invalid_clock(config, start):
    with pytest.raises(ValueError): ContactHistory(config).begin('a', start)


@pytest.mark.parametrize('field', ['missing_robot_body', 'missing_sensor', 'unknown_link'])
def test_incomplete_declared_robot_coverage_disables_all_regions(metadata, field):
    if field == 'missing_robot_body': metadata['body_names'].remove('right_index_PP')
    elif field == 'missing_sensor': metadata['contact_evidence']['sensor_paths'].pop()
    else:
        metadata['scene_info']['robot_link_paths']['robot_unsupported'] = '/World/Robot/robot_unsupported'
        metadata['body_names'].append('robot_unsupported')
        metadata['contact_evidence']['sensor_paths'].append('/World/Robot/robot_unsupported')
    assert configure_contact_history(metadata, 'sharpa_w01_right')['available'] is False
