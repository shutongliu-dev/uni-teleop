"""Counterexamples to treating the looser rig rest gate as canonical rest."""
from copy import deepcopy
import math

import pytest

from scenes.workrooms.mechanical_assembly.tasks.gear_train_assembly.rules import GEARS, TOOLS
from scripts.validation.gear_passive_checks import canonical_passive_window


DT = .0005
PATHS = {n: '/World/Objects/'+n for n in GEARS+TOOLS}
FIXTURE = '/World/Fixtures/gear_workplate'
INFO = {'objects': {n: {'path': p} for n, p in PATHS.items()}, 'fixtures': {'gear_workplate': {'path': FIXTURE}}}


@pytest.fixture
def rows():
    positions = {n: [.4, y, .802, 0., 0., 0., 1.] for n, y in zip(GEARS, (-.13, 0., .13))}
    positions.update({n: [.6, 0., .82, 0., 0., 0., 1.] for n in TOOLS})
    contacts = [{'actor0': PATHS[n], 'actor1': FIXTURE+'/thrust',
        'normal_impulses_ns': [-.001], 'normals': [[0., 0., -1.]]} for n in GEARS]
    return [{'native_step': i+1400, 'time_s': .7+i*DT, 'poses_xyzw': deepcopy(positions),
        'velocities': {n: [0.]*6 for n in PATHS}, 'raw_contacts': deepcopy(contacts),
        'maximum_penetration_m': 0., 'wrench': {'native_step': i+1400, 'body': 'socket_crank',
            'phase': 'settle', 'world_frame': True, 'force_n': [0., 0., 0.], 'torque_nm': [0., 0., 0.]}}
        for i in range(601)]


def assess(rows, phase='settle'):
    return canonical_passive_window(rows, INFO, end_time_s=1., phase=phase, dt=DT)


def test_exact_canonical_window_contains_baseline_and_fifteen_observed_intervals(rows):
    result = assess(rows)
    assert result['passed'], result
    assert result['native_intervals'] == 600 and result['observed_intervals'] == 15
    assert len(result['objects']) == 6 and not result['task_success_claimed']


@pytest.mark.parametrize('fault', ['loose_rig_speed', 'observed_angular', 'observed_linear', 'anchor_range',
                                 'native_angular', 'missing_baseline', 'missing_substep', 'support', 'external_wrench'])
def test_incomplete_or_noncanonical_rest_is_rejected(rows, fault):
    if fault == 'loose_rig_speed':
        rows[300]['velocities']['input_gear'][0] = .003
    elif fault == 'observed_angular':
        for i, row in enumerate(rows):
            theta = i*DT*.025
            row['poses_xyzw']['input_gear'][5:] = [math.sin(theta/2), math.cos(theta/2)]
    elif fault == 'observed_linear':
        rows[40]['poses_xyzw']['input_gear'][0] += .00005
    elif fault == 'anchor_range':
        for i, row in enumerate(rows):
            row['poses_xyzw']['input_gear'][0] += i/600*.00011
    elif fault == 'native_angular':
        rows[300]['velocities']['input_gear'][5] = .051
    elif fault == 'missing_baseline':
        rows.pop(0)
    elif fault == 'missing_substep':
        rows.pop(301)
    elif fault == 'support':
        for row in rows[1:41]:
            row['raw_contacts'][0]['normal_impulses_ns'] = [.001]
    elif fault == 'external_wrench':
        rows[300]['wrench']['torque_nm'][2] = .01
    result = assess(rows)
    assert not result['passed'], fault
    if fault not in ('missing_baseline', 'missing_substep'):
        assert 'error' not in result, result


def test_final_release_rejects_even_zero_pressure_robot_tool_touch(rows):
    for row in rows:
        row['wrench']['phase'] = 'final_hold'
    rows[300]['raw_contacts'].append({'actor0': PATHS['socket_crank'], 'actor1': '/World/Robot/right_index_MP',
        'normal_impulses_ns': [0.], 'normals': [[1., 0., 0.]]})
    result = assess(rows, 'final_hold')
    assert not result['passed'] and not result['checks']['released_without_robot_manifolds']


def test_downward_fixture_contacts_must_cancel_upward_support(rows):
    for row in rows:
        row['raw_contacts'].append({'actor0': FIXTURE+'/downward_stop', 'actor1': PATHS['input_gear'],
            'normal_impulses_ns': [.002], 'normals': [[0., 0., 1.]]})
    assert not assess(rows)['passed']
