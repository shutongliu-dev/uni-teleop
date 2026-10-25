"""Adversarial CPU constraints; native acceptance is recorded separately."""
from copy import deepcopy
import json

import pytest

from scenes.manipulation.humanlike import HumanlikeGraspEvaluator, default_humanlike_config


@pytest.fixture
def config():
    return default_humanlike_config()


def observation(config, step=0, *, fingers=('thumb', 'index'), opposed=True, ticks=None):
    time = step * .02
    center = [.07, -.05, .85]
    q = {name: 0. for name in config['joint_limits_rad']}
    q.update(right_thumb_CMC_FE=.5, right_index_MCP_FE=.4, right_index_PIP=.5, right_index_DIP=.3,
             right_middle_MCP_FE=.4, right_middle_PIP=.5, right_middle_DIP=.3)
    rows = []
    if step:
        for tick in (range(1, 41) if ticks is None else ticks):
            for finger in fingers:
                sign = 1 if finger == 'thumb' or not opposed else -1
                rows.append({'time_s': time-.02+tick*.0005, 'actor0': '/World/Objects/cube',
                    'actor1': '/World/Robot/nested/right_' + finger + '_elastomer',
                    'point_count': 1, 'impulse_vector_ns': [0., sign * .001, 0.],
                    'impulse_magnitude_sum_ns': .001, 'nonzero_impulse_normal_count': 1,
                    'normal_length_max_abs_error': 0., 'min_separation_m': -.0001,
                    'impulse_components': 'normal_only'})
    return {'time_s': time, 'last_duration_s': .02 if step else 0., 'joint_positions_rad': q,
        'metrics': {'state_finite': True}, 'objects': {'cube': {'position_m': center}},
        'robot': {'hand_id': 'sharpa_w01_right',
            'right_hand_flange': {'position_m': [0., 0., 1.], 'quaternion_wxyz': [0., 1., 0., 0.]},
            'fingertips': {'right_thumb_fingertip': {'position_m': [.07, -.075, .85]},
                          'right_index_fingertip': {'position_m': [.07, -.025, .85]},
                          'right_middle_fingertip': {'position_m': [.075, -.024, .85]}}},
        'contacts': {'source': 'PhysX stage-bound rigid contact tensor; normal impulses',
                     'errors': [], 'rigid': rows}}


def scorer(config):
    h = HumanlikeGraspEvaluator(config)
    first = h.update(observation(config))
    assert first['valid'] and not first['passed'] and first['reason'] == 'baseline_recorded'
    return h


def test_opposing_pad_contact_and_pose_require_continuous_dwell(config):
    h = scorer(config)
    for i in range(1, 6):
        s = h.update(observation(config, i))
        assert s['valid'] and s['instantaneous_pass'] and not s['passed']
    s = h.update(observation(config, 6))
    assert s['passed'] and s['hold_elapsed_s'] == pytest.approx(.12)
    assert s['metrics']['simultaneous_opposing_fraction'] == 1.
    assert s['metrics']['pad_fingers_contacting'] == ['index', 'thumb']
    assert json.loads(json.dumps(s, allow_nan=False)) == s


@pytest.mark.parametrize('fingers', [(), ('thumb',), ('index',), ('ring', 'pinky')])
def test_single_finger_or_wrong_finger_cannot_pass(config, fingers):
    h = scorer(config)
    for i in range(1, 15):
        s = h.update(observation(config, i, fingers=fingers))
    assert s['valid'] and not s['passed'] and not s['instantaneous_pass']


def test_middle_is_supported_opposition(config):
    h = scorer(config)
    for i in range(1, 7):
        s = h.update(observation(config, i, fingers=('thumb', 'middle')))
    assert s['passed']


def test_same_direction_forces_do_not_establish_opposition(config):
    h = scorer(config)
    s = h.update(observation(config, 1, opposed=False))
    assert s['grasp_attempted'] and not s['instantaneous_pass']
    assert s['metrics']['simultaneous_opposing_fraction'] == 0.


def test_pair_orientation_is_normalized_before_comparing_normals(config):
    h = scorer(config)
    o = observation(config, 1)
    for row in o['contacts']['rigid']:
        if 'index' in row['actor1']:
            row['actor0'], row['actor1'] = row['actor1'], row['actor0']
            row['impulse_vector_ns'] = [-x for x in row['impulse_vector_ns']]
    assert h.update(o)['instantaneous_pass']


def test_non_simultaneous_finger_contacts_do_not_form_a_grasp(config):
    h = scorer(config)
    o = observation(config, 1)
    o['contacts']['rigid'] = [r for r in o['contacts']['rigid'] if
        ('thumb' in r['actor1'] and r['time_s'] <= .01) or
        ('index' in r['actor1'] and r['time_s'] > .01)]
    s = h.update(o)
    assert s['grasp_attempted'] and not s['instantaneous_pass']
    assert s['metrics']['simultaneous_opposing_fraction'] == 0.


@pytest.mark.parametrize('count, expected', [(19, False), (20, True), (40, True)])
def test_explicit_substep_coverage_threshold(config, count, expected):
    h = scorer(config)
    s = h.update(observation(config, 1, ticks=range(1, count+1)))
    assert s['instantaneous_pass'] is expected
    assert s['metrics']['simultaneous_opposing_fraction'] == count / 40


@pytest.mark.parametrize('replacement', ['right_index_DP', 'fr3_link8', 'right_ring_elastomer'])
def test_non_pad_or_non_opposing_finger_links_are_not_substitutes(config, replacement):
    h = scorer(config)
    o = observation(config, 1)
    for r in o['contacts']['rigid']:
        r['actor1'] = r['actor1'].replace('right_index_elastomer', replacement)
    assert not h.update(o)['instantaneous_pass']


def test_no_object_identity_substitution(config):
    h = scorer(config)
    o = observation(config, 1)
    for r in o['contacts']['rigid']:
        r['actor0'] = '/World/Objects/context_blue_cup'
    assert not h.update(o)['instantaneous_pass']


def test_zero_solved_impulse_does_not_count(config):
    h = scorer(config)
    o = observation(config, 1)
    for r in o['contacts']['rigid']:
        r.update(impulse_magnitude_sum_ns=0., impulse_vector_ns=[0., 0., 0.], nonzero_impulse_normal_count=0)
    s = h.update(o)
    assert s['valid'] and not s['instantaneous_pass']


def test_contact_loss_revokes_pass_and_restarts_dwell(config):
    h = scorer(config)
    for i in range(1, 7):
        s = h.update(observation(config, i))
    assert s['passed']
    s = h.update(observation(config, 7, fingers=()))
    assert not s['passed'] and s['hold_elapsed_s'] == 0.
    for i in range(8, 13):
        assert not h.update(observation(config, i))['passed']
    assert h.update(observation(config, 13))['passed']


def test_observation_gap_cannot_supply_hold_time(config):
    h = scorer(config)
    s = h.update(observation(config, 9))
    assert s['valid'] and not s['passed'] and not s['instantaneous_pass']
    assert s['reason'] == 'observation_gap:dwell_reset'


@pytest.mark.parametrize('mutate', [
    lambda o: o['contacts']['rigid'][0].update(time_s=-1.),
    lambda o: o['contacts']['rigid'][0].update(time_s=.04),
    lambda o: o['contacts']['rigid'][0].update(time_s=.0007),
    lambda o: o['contacts']['rigid'].append(deepcopy(o['contacts']['rigid'][0])),
    lambda o: o['contacts'].update(source='distance'),
    lambda o: o['contacts'].update(errors=['overflow']),
    lambda o: o['contacts']['rigid'][0].pop('impulse_vector_ns'),
    lambda o: o['contacts']['rigid'][0].update(impulse_vector_ns=[0., 2., 0.]),
    lambda o: o['contacts']['rigid'][0].update(min_separation_m=float('nan')),
    lambda o: o['contacts']['rigid'][0].update(point_count=True),
    lambda o: o['contacts']['rigid'][0].update(normal_length_max_abs_error=.01),
    lambda o: o['contacts']['rigid'][0].update(nonzero_impulse_normal_count=0),
    lambda o: o['contacts']['rigid'][0].update(impulse_components='combined'),
    lambda o: o['robot'].pop('right_hand_flange'),
    lambda o: o['robot']['fingertips'].pop('right_thumb_fingertip'),
    lambda o: o['robot'].update(hand_id='leap'),
    lambda o: o['joint_positions_rad'].pop('right_thumb_IP'),
    lambda o: o['joint_positions_rad'].update(right_thumb_IP=float('inf')),
    lambda o: o['metrics'].update(state_finite=False),
    lambda o: o.update(last_duration_s=.01),
    lambda o: o['robot']['right_hand_flange'].update(quaternion_wxyz=[0., 0., 0., 0.]),
])
def test_missing_or_corrupt_evidence_fails_closed_and_stays_invalid(config, mutate):
    h = scorer(config)
    o = observation(config, 1)
    mutate(o)
    s = h.update(o)
    assert not s['valid'] and not s['passed'] and s['reason'].startswith('invalid_observation:')
    assert not h.update(observation(config, 2))['valid']


@pytest.mark.parametrize('mutate, metric', [
    (lambda o: o['joint_positions_rad'].update(right_thumb_IP=2.), 'joints_within_limits'),
    (lambda o: o['joint_positions_rad'].update(right_index_PIP=0., right_index_DIP=1.3), 'finger_coupling_valid'),
    (lambda o: o['robot']['right_hand_flange'].update(quaternion_wxyz=[1., 0., 0., 0.]), 'wrist_approach_valid'),
    (lambda o: o['robot']['right_hand_flange'].update(position_m=[1., 0., 1.]), 'object_in_wrist_grasp_box'),
])
def test_wrong_posture_blocks_grasp_but_evidence_still_valid(config, mutate, metric):
    h = scorer(config)
    o = observation(config, 1)
    mutate(o)
    s = h.update(o)
    assert s['valid'] and not s['pose_valid'] and not s['instantaneous_pass']
    assert s['metrics'][metric] is False


def test_same_side_tips_cannot_pass_even_with_opposing_force_vectors(config):
    h = scorer(config)
    o = observation(config, 1)
    o['robot']['fingertips']['right_index_fingertip']['position_m'] = [.07, -.10, .85]
    o['robot']['fingertips']['right_middle_fingertip']['position_m'] = [.07, -.11, .85]
    s = h.update(o)
    assert s['valid'] and not s['pose_valid'] and not s['instantaneous_pass']


@pytest.mark.parametrize('kind', ['object', 'self', 'table'])
def test_deep_robot_penetration_blocks_grasp(config, kind):
    h = scorer(config)
    o = observation(config, 1)
    row = deepcopy(o['contacts']['rigid'][0])
    row['min_separation_m'] = -.004
    if kind == 'object':
        o['contacts']['rigid'][0] = row
    else:
        row['actor0'] = '/World/Robot/right_index_DP' if kind == 'self' else '/World/Table'
        o['contacts']['rigid'].append(row)
    s = h.update(o)
    assert s['valid'] and not s['instantaneous_pass'] and not s['metrics']['penetration_within_limit']


def test_no_input_mutation_and_reset_discards_all_history(config):
    c = deepcopy(config)
    h = scorer(config)
    o = observation(config, 1)
    before = deepcopy(o)
    result = h.update(o)
    assert o == before and config == c
    result['metrics'].clear()
    assert h.last_result['metrics']
    h.reset()
    assert h.previous_time is None and h.maximum_hold_s == 0. and h.invalid is None
    assert h.update(observation(config))['valid']


@pytest.mark.parametrize('mutate', [
    lambda c: c.update(version='unknown'),
    lambda c: c['joint_limits_rad'].pop('right_thumb_IP'),
    lambda c: c['joint_limits_rad'].update(right_thumb_IP=[1., 0.]),
    lambda c: c.update(physics_dt_s=.003),
    lambda c: c.update(minimum_opposed_substep_fraction=1.1),
    lambda c: c.update(maximum_normal_direction_cosine=.1),
    lambda c: c.update(hold_s=float('nan')),
    lambda c: c.update(wrist_approach_axis_local=[0., 0., 0.]),
])
def test_invalid_release_configuration_is_rejected(config, mutate):
    mutate(config)
    with pytest.raises(ValueError):
        HumanlikeGraspEvaluator(config)
