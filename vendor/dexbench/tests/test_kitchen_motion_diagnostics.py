"""Separate real substep oscillation from sustained drift without waiving either."""
from copy import deepcopy
import json
import math

import pytest

from scripts.validation.kitchen_motion_diagnostics import NativeMotionDiagnostics


LIMITS = {'maximum_linear_speed_m_s': .025, 'maximum_angular_speed_rad_s': .25,
          'maximum_stable_position_range_m': .004, 'maximum_stable_orientation_range_rad': .08}


def frame(step, dt=.0005):
    t = step*dt
    objects = {}
    for name, x, a, v, w in (
        ('jitter', (-1)**step*.00005, (-1)**step*.0005, .2, 2.),
        ('drift', t*.1, t, .1, 1.)):
        # Deliberate source norm/sign variation must not create fake rotation.
        scale = (-1)**step*1.0001
        objects[name] = {'position_m': [x, 0., 1.],
            'quaternion_wxyz': [scale*math.cos(a/2), 0., 0., scale*math.sin(a/2)],
            'linear_velocity_m_s': [v, 0., 0.], 'angular_velocity_rad_s': [0., 0., w]}
    return {'native_step': step, 'time_s': t, 'native_time_s': t+10.,
            'physics_dt_s': dt, 'objects': objects}


def test_native_peaks_remain_visible_while_window_ranges_distinguish_oscillation(tmp_path):
    diagnostic = NativeMotionDiagnostics(LIMITS, .0005)
    for step in range(1201):
        raw = frame(step)
        original = deepcopy(raw)
        assert diagnostic.observe(raw)
        assert raw == original
    result = diagnostic.summary(.4)
    assert result['window_complete'] and result['window_frames'] == 801
    assert result['window_start_s'] == pytest.approx(.2)
    jitter, drift = result['objects']['jitter'], result['objects']['drift']
    assert jitter['native']['angular_over_limit_count'] == 801
    assert jitter['adjacent_pose']['angular_over_limit_count'] == 800
    assert jitter['adjacent_pose']['angular_speed_peak_rad_s'] == pytest.approx(2.)
    assert jitter['orientation_range_rad'] == pytest.approx(.001)
    assert not jitter['orientation_range_over_limit']
    assert jitter['pose_20ms']['angular_speed_peak_rad_s'] == pytest.approx(0.)
    assert jitter['pose_20ms']['linear_speed_peak_m_s'] == pytest.approx(0.)
    assert drift['orientation_range_rad'] == pytest.approx(.4)
    assert drift['position_range_m'] == pytest.approx(.04)
    assert drift['orientation_range_over_limit'] and drift['position_range_over_limit']
    assert drift['pose_20ms']['angular_speed_peak_rad_s'] == pytest.approx(1.)
    assert drift['net_rotation_rad'] == pytest.approx(.4)
    assert jitter['net_rotation_rad'] == pytest.approx(0.)
    path = tmp_path/'raw.jsonl'
    diagnostic.save_raw(path)
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(rows) == 1201
    assert rows[0]['objects']['jitter']['quaternion_norm_before_normalization'] == pytest.approx(1.0001)


def test_missing_or_invalid_frames_mark_diagnostic_incomplete_without_raising():
    diagnostic = NativeMotionDiagnostics(LIMITS, .0005)
    assert diagnostic.observe(frame(0))
    assert diagnostic.observe(frame(2))
    broken = frame(3)
    broken['objects']['jitter']['quaternion_wxyz'] = [0.]*4
    assert not diagnostic.observe(broken)
    broken['objects'] = None
    assert not diagnostic.observe(broken)
    result = diagnostic.summary()
    assert not result['window_complete']
    assert any('Missing' in error for error in result['errors'])
    assert any('Zero native quaternion' in error for error in result['errors'])
    assert result['objects']['jitter']['adjacent_pose']['samples'] == 0
