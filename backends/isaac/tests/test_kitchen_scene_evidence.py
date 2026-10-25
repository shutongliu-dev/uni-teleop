"""Saved evidence must not replace complete native support and quiet receipts."""
from copy import deepcopy
import math

import pytest

from scripts.validation.kitchen_scene_evidence import support_evidence


BODY = '/World/KitchenTask/Actors/tomato_0'
SUPPORT = '/World/KitchenTask/Fixtures/source_board'


def fixture(*, rolling=True):
    limits = dict(maximum_linear_speed_m_s=.025, maximum_angular_speed_rad_s=.25,
        maximum_stable_position_range_m=.004, maximum_stable_orientation_range_rad=.08)
    if rolling:
        limits['stability_pose_window_s'] = .020
    task = {'id': 'produce_transfer', 'evaluation': {'thresholds': limits, 'metadata': {
        'actors': {'tomato_0': BODY}, 'fixtures': {'source_board': SUPPORT},
        'objects': [{'id': 'tomato_0', 'mass_kg': .05}],
        'goals': {'support_by_actor': {'tomato_0': 'source_board'}},
        'source_support_by_actor': {'tomato_0': 'source_board'}}}}
    runtime = {'physics_dt_s': .0005, 'scene_info': {}}
    records = []
    for step in range(1, 31):
        time_s = step*.020
        obs = {'time_s': time_s,
            'objects': {'tomato_0': {'position_m': [0., 0., .9],
                'quaternion_wxyz': [1., 0., 0., 0.],
                'linear_velocity_m_s': [0., 0., 0.], 'angular_velocity_rad_s': [0., 0., 0.]}},
            'task_state': {'complete': True, 'errors': [], 'native_support_by_actor': {
                'tomato_0': {'supported': True, 'released_quiet_native_s': time_s,
                    'witnesses': [{'support_actor_path': SUPPORT}]}}},
            'contacts': {'errors': [], 'sampling': {'complete': True, 'sampled_substeps': 40,
                'native_start_step_index': (step-1)*40, 'native_end_step_index': step*40},
                'rigid': [{'actor0': BODY, 'actor1': SUPPORT, 'impulse_vector_ns': [0., 0., .0002],
                    'min_separation_m': -.0001, 'time_s': time_s, 'impulse_magnitude_sum_ns': .0002}]}}
        records.append({'kind': 'step', 'observation': obs})
    return runtime, records, task


def test_new_pose_contract_retains_tensor_peaks_while_legacy_contract_keeps_its_gate():
    runtime, records, task = fixture()
    for record in records:
        record['observation']['objects']['tomato_0']['angular_velocity_rad_s'] = [0., 2., 0.]
    result = support_evidence(runtime, records, task)
    body = result['objects']['tomato_0']
    assert result['passed']
    assert body['terminal_maximum_angular_speed_rad_s'] == 2.
    assert not body['tensor_velocity_gate_applied']
    assert not body['tensor_velocity_gate_would_pass']
    assert body['saved_pose_stability']['passed']
    del task['evaluation']['thresholds']['stability_pose_window_s']
    legacy = support_evidence(runtime, records, task)
    assert not legacy['passed']
    assert legacy['objects']['tomato_0']['tensor_velocity_gate_applied']


def test_saved_rotation_is_normalized_and_quaternion_sign_does_not_create_motion():
    runtime, records, task = fixture()
    for index, record in enumerate(records):
        record['observation']['objects']['tomato_0']['quaternion_wxyz'] = [(-1.)**index*1.001, 0., 0., 0.]
    result = support_evidence(runtime, records, task)
    assert result['passed']
    assert result['objects']['tomato_0']['saved_pose_stability']['maximum_angular_speed_rad_s'] == 0.


@pytest.mark.parametrize('motion', ['rotation', 'translation', 'position_envelope'])
def test_native_quiet_claim_cannot_hide_saved_pose_motion(motion):
    runtime, records, task = fixture()
    for record in records:
        obs = record['observation']
        t = obs['time_s']
        state = obs['objects']['tomato_0']
        if motion == 'rotation':
            # Total terminal rotation is only .12 rad, but sustained .3 rad/s
            # exceeds the speed limit even with zero native tensor velocities.
            angle = .3*t
            state['quaternion_wxyz'] = [math.cos(angle/2), 0., 0., math.sin(angle/2)]
        elif motion == 'translation':
            state['position_m'][0] = .03*t
        else:
            # All poses remain within 2.2 mm of the final pose. The 4.4 mm
            # all-pairs range still fails; every speed stays below 25 mm/s.
            state['position_m'][0] = -.0022+.022*(t-.2) if t <= .4 else .0022-.011*(t-.4)
    result = support_evidence(runtime, records, task)
    assert not result['passed']
    review = result['objects']['tomato_0']['saved_pose_stability']
    assert not review['passed']
    if motion == 'position_envelope':
        assert review['maximum_linear_speed_m_s'] < .025
        assert review['position_range_m'] == pytest.approx(.0044)
        assert result['objects']['tomato_0']['terminal_position_excursion_m'] < .004


@pytest.mark.parametrize('fault', ['native_quiet_gap', 'support_gap', 'downward_contact',
    'penetration', 'robot_contact', 'unexpected_contact', 'missing_step', 'native_step_gap',
    'incomplete_samples', 'zero_quaternion', 'nonfinite_tensor', 'unknown_window'])
def test_pose_stability_cannot_waive_missing_native_evidence_or_physical_checks(fault):
    runtime, records, task = fixture()
    middle = records[-8]['observation']
    last = records[-1]['observation']
    if fault == 'native_quiet_gap':
        last['task_state']['native_support_by_actor']['tomato_0']['released_quiet_native_s'] = .02
    elif fault == 'support_gap':
        middle['task_state']['native_support_by_actor']['tomato_0']['supported'] = False
    elif fault == 'downward_contact':
        for record in records:
            record['observation']['contacts']['rigid'][0]['impulse_vector_ns'][2] *= -1.
    elif fault == 'penetration':
        middle['contacts']['rigid'][0]['min_separation_m'] = -.005
    elif fault in ('robot_contact', 'unexpected_contact'):
        contact = deepcopy(middle['contacts']['rigid'][0])
        contact['actor1'] = '/World/Robot/finger' if fault == 'robot_contact' else '/World/UndeclaredSupport'
        middle['contacts']['rigid'].append(contact)
    elif fault == 'missing_step':
        del records[-8]
    elif fault == 'native_step_gap':
        middle['contacts']['sampling']['native_end_step_index'] += 1
    elif fault == 'incomplete_samples':
        middle['contacts']['sampling']['complete'] = False
    elif fault == 'zero_quaternion':
        middle['objects']['tomato_0']['quaternion_wxyz'] = [0.]*4
    elif fault == 'nonfinite_tensor':
        middle['objects']['tomato_0']['linear_velocity_m_s'][0] = float('nan')
    else:
        task['evaluation']['thresholds']['stability_pose_window_s'] = .04
    assert not support_evidence(runtime, records, task)['passed']
