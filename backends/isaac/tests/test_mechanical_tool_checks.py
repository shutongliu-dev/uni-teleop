"""Fail-closed regressions using explicit recorded native sample fixtures."""
import copy

import numpy as np
import pytest

from scripts.validation.mechanical_tool_checks import (
    preinserted_window, released_phase_window, contact_exchange_check)


DT = .0005
PATHS = {'screwdriver': '/World/Objects/tool', 'bolt': '/World/Objects/bolt/asset/body',
         'fixture': '/World/Fixtures/fixture/asset/body'}
HAND = '/World/Robot/right_index_DP'
INITIAL = {'screwdriver': {'position_m': [0., 0., .9], 'quaternion_wxyz': [1., 0., 0., 0.]},
           'bolt': {'position_m': [0., 0., .85], 'quaternion_wxyz': [1., 0., 0., 0.]}}
TARGETS = {**{f'fr3_joint{i}': .1 for i in range(1, 8)},
           **{f'right_joint{i}': .2 for i in range(22)}}


def contact(a, b, impulse=.001, separation=-.0001):
    return {'actor0': a, 'actor1': b, 'normal_impulses_ns': [impulse],
            'separations_m': [separation]}


def samples(phase, duration):
    template = {'phase': phase, 'object_poses_xyzw': {
        name: [*state['position_m'], 0., 0., 0., 1.] for name, state in INITIAL.items()},
        'object_velocities': {name: [0.]*6 for name in INITIAL},
        'joint_positions_rad': dict(TARGETS),
        'raw_contacts': [contact(PATHS['screwdriver'], PATHS['bolt']),
                         contact(PATHS['bolt'], PATHS['fixture'])]}
    return [{**copy.deepcopy(template), 'native_step_index': i+1,
             'simulation_time_s': (i+1)*DT} for i in range(round(duration/DT))]


def test_settle_requires_every_native_sample_and_exact_support_actor_pairs():
    rows = samples('settle', .2)
    assert preinserted_window(rows, INITIAL, PATHS, DT)['passed']
    rows[100]['raw_contacts'][1]['actor1'] = '/World/Fixtures/fixture/other_body'
    result = preinserted_window(rows, INITIAL, PATHS, DT)
    assert not result['passed']
    assert result['positive_contact_samples']['bolt_fixture'] == 399


@pytest.mark.parametrize('gap_steps,passed', [(1, True), (2, True), (3, False), (40, False)])
def test_settle_bounds_zero_impulse_gaps_instead_of_demanding_constant_pressure(gap_steps, passed):
    rows = samples('settle', .2)
    for row in rows[100:100+gap_steps]:
        row['raw_contacts'][0]['normal_impulses_ns'] = [0.]
    result = preinserted_window(rows, INITIAL, PATHS, DT)
    assert result['passed'] is passed
    assert result['maximum_pressure_gap_s']['tool_bolt'] == gap_steps*DT
    assert len(result['support_interval_impulses_ns']['tool_bolt']) == 10


@pytest.mark.parametrize('failure', ['angular_velocity', 'linear_velocity', 'rotation', 'transient_displacement', 'penetration', 'negative_contact'])
def test_reset_final_origin_alone_cannot_hide_unstable_window(failure):
    rows = samples('settle', .2)
    sample = rows[100]  # Last sample stays exactly at reset; older gate passed it.
    if failure == 'angular_velocity':
        sample['object_velocities']['screwdriver'][5] = .301
    elif failure == 'linear_velocity':
        sample['object_velocities']['bolt'][0] = .021
    elif failure == 'rotation':
        sample['object_poses_xyzw']['screwdriver'][3:] = [0., 0., np.sin(.02), np.cos(.02)]
    elif failure == 'transient_displacement':
        sample['object_poses_xyzw']['bolt'][0] += .0031
    elif failure == 'penetration':
        sample['raw_contacts'][0]['separations_m'] = [-.0021]
    else:
        for row in rows[100:103]:
            row['raw_contacts'][0]['normal_impulses_ns'] = [-.001]
    assert not preinserted_window(rows, INITIAL, PATHS, DT)['passed']


def test_settle_rejects_incomplete_phase_or_nonfinite_raw_evidence():
    rows = samples('settle', .2)
    assert not preinserted_window(rows[:-1], INITIAL, PATHS, DT)['passed']
    rows[0]['phase'] = 'approach'
    assert not preinserted_window(rows, INITIAL, PATHS, DT)['passed']
    rows[0]['phase'] = 'settle'
    rows[0]['raw_contacts'][0]['normal_impulses_ns'] = [float('nan')]
    with pytest.raises(ValueError, match='Invalid raw native contact'):
        preinserted_window(rows, INITIAL, PATHS, DT)


@pytest.mark.parametrize('phase', ['open', 'retreat', 'final_hold'])
def test_release_requires_measured_all_joint_endpoint_and_no_hand_tool_contact(phase):
    rows = samples(phase, .1)
    assert released_phase_window(rows, phase, TARGETS, {HAND}, PATHS['screwdriver'], DT)['passed']
    rows[100]['joint_positions_rad']['fr3_joint7'] += .0101
    assert not released_phase_window(rows, phase, TARGETS, {HAND}, PATHS['screwdriver'], DT)['passed']
    rows[100]['joint_positions_rad']['fr3_joint7'] = TARGETS['fr3_joint7']
    rows[100]['joint_positions_rad']['right_joint21'] += .0201
    assert not released_phase_window(rows, phase, TARGETS, {HAND}, PATHS['screwdriver'], DT)['passed']
    rows[100]['joint_positions_rad']['right_joint21'] = TARGETS['right_joint21']
    rows[50]['raw_contacts'].append(contact(HAND, PATHS['screwdriver']))
    result = released_phase_window(rows, phase, TARGETS, {HAND}, PATHS['screwdriver'], DT)
    assert not result['passed'] and result['hand_tool_contact_samples'] == 1


def test_release_rejects_missing_joint_and_stale_native_clock():
    rows = samples('open', .1)
    rows[2]['simulation_time_s'] = rows[1]['simulation_time_s']
    assert not released_phase_window(rows, 'open', TARGETS, {HAND}, PATHS['screwdriver'], DT)['passed']
    rows[2]['simulation_time_s'] = 3*DT
    rows[2]['joint_positions_rad'].pop('right_joint21')
    with pytest.raises(ValueError, match='joint inventory'):
        released_phase_window(rows, 'open', TARGETS, {HAND}, PATHS['screwdriver'], DT)


def test_exchange_close_false_or_count_mismatch_cannot_pass():
    good = {'all_samples_consumed_once': True, 'acquisitions': 1000, 'deliveries': 1000}
    assert contact_exchange_check(good, 1000, 1000)['passed']
    assert not contact_exchange_check({**good, 'all_samples_consumed_once': False}, 1000, 1000)['passed']
    assert not contact_exchange_check({**good, 'deliveries': 999}, 1000, 1000)['passed']
    assert not contact_exchange_check(good, 1000, 999)['passed']
