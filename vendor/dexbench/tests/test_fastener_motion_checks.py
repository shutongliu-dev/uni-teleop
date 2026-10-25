"""Regression cases for native speed spikes and omitted control intervals."""
from copy import deepcopy

import pytest

from scenes.workrooms.mechanical_assembly.tasks.fastener_assembly.rules import THRESHOLDS
from scripts.validation.mechanical_fastener_motion_checks import full_motion_window


def stream():
    anchor = {'native_step_index': 2, 'simulation_time_s': 0.,
              'object_poses_xyzw': {name: [0., 0., .8, 0., 0., 0., 1.]
                                    for name in ('bolt', 'screwdriver')}}
    rows = [{**deepcopy(anchor), 'phase': 'settle' if i < 40 else 'loaded_turn',
             'native_step_index': i+3, 'simulation_time_s': (i+1)*.0005,
             'object_velocities': {name: [0.]*6 for name in ('bolt', 'screwdriver')}}
            for i in range(80)]
    return anchor, rows


@pytest.mark.parametrize('fault', ['native_spike', 'phase_boundary', 'first_interval', 'missing_step', 'nonfinite'])
def test_full_motion_cannot_hide_bad_substep_or_control_interval(fault):
    anchor, rows = stream()
    if fault == 'native_spike':
        rows[25]['object_velocities']['bolt'][5] = 1.25
    elif fault in ('phase_boundary', 'first_interval'):
        for row in rows[40 if fault == 'phase_boundary' else 0:]:
            row['object_poses_xyzw']['bolt'][0] += .01
    elif fault == 'missing_step':
        del rows[24]
    else:
        rows[24]['object_velocities']['bolt'][0] = float('nan')
    result = full_motion_window(rows, .0005, THRESHOLDS, expected_samples=80, anchor=anchor)
    assert not result['passed']
    if fault == 'native_spike':
        assert result['objects']['bolt']['maximum_native_angular_speed_rad_s'] == 1.25
    elif fault in ('phase_boundary', 'first_interval'):
        assert result['objects']['bolt']['maximum_observed_20ms_linear_speed_m_s'] == .5


def test_native_anchor_covers_first_interval_and_legacy_omission_is_explicit():
    anchor, rows = stream()
    anchored = full_motion_window(rows, .0005, THRESHOLDS, expected_samples=80, anchor=anchor)
    legacy = full_motion_window(rows, .0005, THRESHOLDS, expected_samples=80)
    assert anchored['passed'] and legacy['passed']
    assert anchored['first_20ms_pose_interval_checked']
    assert anchored['objects']['bolt']['observed_20ms_intervals'] == 2
    assert not legacy['first_20ms_pose_interval_checked']
    assert legacy['objects']['bolt']['observed_20ms_intervals'] == 1
    assert not anchored['phase_boundaries_reset_pose_history']


def test_stream_shorter_than_one_control_period_fails_without_extrema_error():
    _, rows = stream()
    result = full_motion_window(rows[:3], .0005, THRESHOLDS, expected_samples=80)
    assert not result['passed']
    assert result['objects']['bolt']['observed_20ms_intervals'] == 0
