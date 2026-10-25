from copy import deepcopy
import numpy as np
import pytest
from scripts.validation.mechanical_fastener_passive_checks import (task_contact_penetration,
    passive_phase_window)
from scenes.workrooms.mechanical_assembly.tasks.fastener_assembly.rules import THRESHOLDS


PATHS = {'screwdriver': 'tool', 'bolt': 'bolt', 'fixture': 'fixture'}


def evidence(phase='final_passive', seconds=.6):
    config = deepcopy(THRESHOLDS)
    config['robot_actor_prefix'] = '/World/Robot/'
    config['initial_objects'] = {name: {'position_m': [0, 0, .8], 'quaternion_wxyz': [1, 0, 0, 0]}
                                 for name in ('screwdriver', 'bolt')}
    rows = []
    for i in range(round(seconds/.0005)):
        rows.append({'phase': phase, 'native_step_index': i+1, 'simulation_time_s': (i+1)*.0005,
            'object_poses_xyzw': {name: [0, 0, .8, 0, 0, 0, 1] for name in ('screwdriver', 'bolt')},
            'object_velocities': {name: [0]*6 for name in ('screwdriver', 'bolt')},
            'raw_contacts': [{'actor0': a, 'actor1': b, 'normal_impulses_ns': [.001], 'separations_m': [-1e-6]}
                             for a, b in [('tool', 'bolt'), ('bolt', 'fixture')]]})
    return rows, config


def loose_evidence(phase='final_passive', seconds=.6):
    rows, config = evidence(phase, seconds)
    for row in rows:
        row['raw_contacts'] = [
            {'actor0': 'tool', 'actor1': 'tool_supply/collision_001', 'normal_impulses_ns': [.001], 'separations_m': [-1e-6]},
            {'actor0': 'bolt', 'actor1': 'bolt_supply/collision_000', 'normal_impulses_ns': [.001], 'separations_m': [-1e-6]},
        ]
    paths = {**PATHS, 'tool_supply_rest': 'tool_supply', 'bolt_supply_rest': 'bolt_supply'}
    return rows, config, paths


@pytest.mark.parametrize('phase', ['settle', 'final_passive'])
def test_complete_frozen_passive_window(phase):
    rows, config = evidence(phase)
    assert passive_phase_window(rows, phase, PATHS, .0005, config)['passed']
    assert task_contact_penetration(rows, PATHS, .0005, config)['passed']


def test_loose_reset_accepts_declared_supply_support_pairs():
    rows, config, paths = loose_evidence()
    report = passive_phase_window(rows, 'final_passive', paths, .0005, config)
    assert report['passed']
    assert report['checks']['no_unexpected_pressure_object_contacts']
    assert min(report['support_interval_impulses_ns']['bolt_supply']) > 0.
    assert min(report['support_interval_impulses_ns']['tool_supply']) > 0.
    assert report['maximum_fixture_support_gap_s'] == 0.


def test_short_release_is_insufficient_even_with_stationary_objects():
    rows, config = evidence(seconds=.3)
    assert not passive_phase_window(rows, 'final_passive', PATHS, .0005, config)['passed']


def test_zero_impulse_deep_penetration_is_not_ignored():
    rows, config = evidence()
    rows[4]['raw_contacts'][1].update(normal_impulses_ns=[0.], separations_m=[-.003])
    assert not task_contact_penetration(rows, PATHS, .0005, config)['passed']


@pytest.mark.parametrize('fault', ['missing_tick', 'speed', 'drift', 'support_gap', 'observed_rotation'])
def test_passive_gate_rejects_each_independent_failure(fault):
    rows, config = evidence()
    if fault == 'missing_tick':
        del rows[-200]
    elif fault == 'speed':
        rows[-100]['object_velocities']['bolt'][0] = .003
    elif fault == 'drift':
        rows[-100]['object_poses_xyzw']['bolt'][0] = .00011
    elif fault == 'support_gap':
        for row in rows[-103:-100]:row['raw_contacts'][1]['normal_impulses_ns'] = [0.]
    else:
        rows[-1]['object_poses_xyzw']['bolt'][5:] = [np.sin(.0006/2), np.cos(.0006/2)]
    assert not passive_phase_window(rows, 'final_passive', PATHS, .0005, config)['passed']


def test_negative_signed_support_scalars_are_pressure_presence():
    rows, config = evidence()
    for row in rows:
        for contact in row['raw_contacts']:
            contact['normal_impulses_ns'] = [-.001]
    report = passive_phase_window(rows, 'final_passive', PATHS, .0005, config)
    assert report['passed']
    assert report['maximum_fixture_support_gap_s'] == 0.
    assert min(report['support_interval_impulses_ns']['bolt_fixture']) > 0.


@pytest.mark.parametrize('phase', ['settle', 'final_passive'])
@pytest.mark.parametrize('object_name', ['screwdriver', 'bolt'])
@pytest.mark.parametrize('impulse', [0., -.001])
def test_passive_robot_touching_is_rejected_even_without_positive_impulse(phase, object_name, impulse):
    rows, config = evidence(phase)
    rows[-100]['raw_contacts'].append({'actor0': '/World/Robot/right_thumb_link',
        'actor1': PATHS[object_name], 'normal_impulses_ns': [impulse], 'separations_m': [0.]})
    report = passive_phase_window(rows, phase, PATHS, .0005, config)
    assert not report['passed']
    assert not report['checks']['no_robot_object_manifolds']
    assert report['robot_object_manifold_pairs'] == [['/World/Robot/right_thumb_link', PATHS[object_name]]]


def test_negative_unexpected_pressure_is_not_clamped_away():
    rows, config = evidence()
    rows[-100]['raw_contacts'].append({'actor0': 'other_body', 'actor1': 'bolt',
        'normal_impulses_ns': [-.001], 'separations_m': [0.]})
    report = passive_phase_window(rows, 'final_passive', PATHS, .0005, config)
    assert not report['passed']
    assert not report['checks']['no_unexpected_pressure_object_contacts']
