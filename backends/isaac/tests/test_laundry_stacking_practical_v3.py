"""Versioned Task3 practical rules; synthetic evidence, no native qualification."""
from copy import deepcopy

import numpy as np
import pytest

from scenes.workrooms.laundry_room.composition import build_task_inputs
from scenes.workrooms.laundry_room.evaluation import SequenceEvaluator
from scenes.workrooms.laundry_room.rules import (
    LEGACY_PROFILE, PRACTICAL_PROFILE, STACKING_PRACTICAL_PROFILE, thresholds,
)
from scenes.workrooms.laundry_room.terminal_window import terminal_window
from test_laundry_practical_rules import goal_vertices, sample, start


def task(profile=STACKING_PRACTICAL_PROFILE):
    return build_task_inputs('towel_stacking', rule_profile=profile,
        physics_options={'physics_dt_s': .002})['task']


def arrays(spec, poses=None, *, count=281):
    goal = goal_vertices(spec)
    result = {'native_step': np.arange(count), 'time_s': np.arange(count)*.002}
    for row in spec['asset_metadata']['cloths']:
        name = row['id']
        result[name+'_q'] = (np.broadcast_to(goal[name], (count, *goal[name].shape)).copy()
                            if poses is None else poses[name].copy())
        result[name+'_qd'] = np.zeros_like(result[name+'_q'])
        result[name+'_control_mask'] = np.zeros((count, len(goal[name])), bool)
        result[name+'_triangles'] = np.array(row['triangles'])
    return result


def test_profile_is_opt_in_and_task3_only():
    assert build_task_inputs('towel_stacking')['task']['evaluation']['rule_profile'] == PRACTICAL_PROFILE
    new = task()
    assert new['public_rules']['schema'] == 'laundry_rules.v5'
    assert new['public_rules']['rule_profile'] == STACKING_PRACTICAL_PROFILE
    old = thresholds('towel_stacking', PRACTICAL_PROFILE)
    changed = {k: v for k, v in new['evaluation']['thresholds'].items() if old.get(k) != v}
    assert changed == {'stable_node_drift_m': .008, 'stability_position_interval_s': .02,
        'folded_height_area_percentile': 99., 'folded_height_absolute_max_m': .045}
    for identifier in ('towel_flattening', 'towel_folding', 'shirt_folding'):
        with pytest.raises(ValueError):
            build_task_inputs(identifier, rule_profile=STACKING_PRACTICAL_PROFILE)


def test_millimetre_jitter_passes_but_raw_native_diagnostics_are_retained():
    spec = task()
    goal = goal_vertices(spec)
    scorer = start(spec, goal)
    a = arrays(spec)
    for step in range(2, 281):
        q = {k: v+[.0005*(-1)**step, 0, 0] for k, v in goal.items()}
        vel = {k: np.broadcast_to([.5*(-1)**step, 0, 0], v.shape).copy() for k, v in q.items()}
        result = scorer.observe(sample(spec, step, q, speeds=vel))
        for name in q:
            a[name+'_q'][step] = q[name]
            a[name+'_qd'][step] = vel[name]
    assert result['success'], result
    assert result['detail']['raw_speed_p95_m_s'] == pytest.approx(.5)
    assert result['detail']['pose_speed_p95_m_s'] == pytest.approx(.5)
    assert result['detail']['control_period_pose_speed_p95_m_s'] < 1e-10
    terminal = terminal_window(a, SequenceEvaluator(spec))
    assert terminal['passed'], terminal
    assert terminal['raw_speed_p95_m_s'] == pytest.approx(.5)
    assert terminal['pose_speed_p95_m_s'] == pytest.approx(.5)
    assert not terminal_window(a, SequenceEvaluator(task(PRACTICAL_PROFILE)))['passed']


@pytest.mark.parametrize('kind', ('translation', 'oscillation', 'unfolded', 'retained', 'crossing', 'robot_attachment'))
def test_real_motion_unfolded_and_unreleased_are_rejected(kind):
    spec = task()
    goal = goal_vertices(spec)
    scorer = start(spec, goal)
    for step in range(2, 281):
        q = deepcopy(goal)
        if kind == 'translation':
            q['towel'][:, 0] += (step-2)*.00004  # 2 cm/s but >8 mm/half-second.
        elif kind == 'oscillation':
            q['towel'][:, 0] += .012*np.sin(step*np.pi/10)  # 20 ms alias cannot hide amplitude.
        elif kind == 'unfolded':
            q['towel'] = np.asarray(spec['asset_metadata']['cloths'][0]['rest_vertices_m'])+[.65,-.55,.9105]
        physics = ({'self_intersections': 1, 'valid': False, 'validity_scope': 'strict_geometry'}
                   if kind == 'crossing' else {'hidden_constraints': 1} if kind == 'robot_attachment' else None)
        result = scorer.observe(sample(spec, step, q, controlled=kind == 'retained', physics=physics))
    assert not result['success'], result


def test_missing_native_frame_and_insufficient_20ms_history_never_score():
    spec = task()
    goal = goal_vertices(spec)
    scorer = start(spec, goal)
    for step in range(2, 20):
        result = scorer.observe(sample(spec, step, goal))
    assert result['released_stable_s'] < .02
    assert not scorer.observe(sample(spec, 21, goal))['success']
    assert scorer.invalid is not None
    a = arrays(spec, count=251)
    assert terminal_window(a, SequenceEvaluator(spec))['reason_codes'] == ['terminal_window_incomplete']
    a = arrays(spec)
    a['native_step'][-260] += 1  # Inside extra history, before the actual final .5 s window.
    with pytest.raises(ValueError, match='history'):
        terminal_window(a, SequenceEvaluator(spec))


def test_height_is_rest_area_weighted_with_absolute_guard():
    spec = task()
    scorer = SequenceEvaluator(spec)
    q = goal_vertices(spec)['towel']
    center = np.asarray(spec['asset_metadata']['target_center_xy_m'])
    base = q[:, 2].min()
    # Test isolated height metric without granting mesh/contact validity.
    q[0, 2] = base+.041
    good, detail = scorer._fold('towel', q, center, table_supported=False)
    assert good and detail['absolute_height_range_m'] > .04
    assert detail['rest_area_height_range_m'] < .04
    assert not SequenceEvaluator(task(PRACTICAL_PROFILE))._fold('towel', q, center, table_supported=False)[0]
    q[0, 2] = base+.046
    assert not scorer._fold('towel', q, center, table_supported=False)[0]
    q = goal_vertices(spec)['towel']
    # Choose cumulative rest area >1%; vertex count alone is not the rule.
    ids = np.argsort(scorer.meshes['towel']['weights'])[::-1]
    weights = scorer.meshes['towel']['weights']
    n = np.searchsorted(np.cumsum(weights[ids])/weights.sum(), .011)+1
    q[ids[:n], 2] = base+.041
    good, detail = scorer._fold('towel', q, center, table_supported=False)
    assert not good and detail['rest_area_height_range_m'] > .04


def test_terminal_speed_checks_each_cloth_each_frame_and_raw_is_diagnostic():
    spec = task()
    a = arrays(spec)
    a['towel_q'][-1, :, 0] += .002  # .1 m/s over20ms; cannot dilute by fixed base.
    result = terminal_window(a, SequenceEvaluator(spec))
    assert not result['passed']
    assert 'terminal_control_period_pose_speed_exceeded' in result['reason_codes']
    assert result['control_period_pose_speed_p95_m_s'] == pytest.approx(.1)
    a = arrays(spec)
    a['towel_control_mask'][-50, 0] = True
    assert not terminal_window(a, SequenceEvaluator(spec))['complete_release']


def test_native_timestep_must_divide_twenty_milliseconds():
    with pytest.raises(ValueError, match='compatible native time step'):
        build_task_inputs('towel_stacking', rule_profile=STACKING_PRACTICAL_PROFILE,
                          physics_options={'physics_dt_s': .0015})
    spec = task()
    spec['evaluation']['physics_dt_s'] = .0015
    with pytest.raises(ValueError, match='integral 20 ms'):
        SequenceEvaluator(spec)


def test_v3_keeps_all_robot_separation_and_operation_requirements():
    from scenes.workrooms.laundry_room.rules import ROBOT_CONTROL, public_rules, bind_task_rules, digest
    from test_laundry_robot_rules import robot_frame, operated, finish as finish_robot
    spec = task()
    spec['evaluation']['control_mode'] = ROBOT_CONTROL
    spec['public_rules'] = public_rules('towel_stacking', profile=STACKING_PRACTICAL_PROFILE,
                                       control_mode=ROBOT_CONTROL)
    spec['evaluation']['public_rules_sha256'] = digest(spec['public_rules'])
    spec = bind_task_rules(spec)
    scorer, goal = operated(spec)
    held = finish_robot(spec, scorer, goal, last=280, separated=False)
    assert not held['success'] and not held['detail']['all_control_released']
    assert held['released_stable_s'] == 0
    assert finish_robot(spec, scorer, goal, first=281, last=541)['success']
    assert not scorer.observe(robot_frame(spec, 542, goal, separated=False))['success']
    passive = SequenceEvaluator(spec)
    for step in range(281):
        result = passive.observe(robot_frame(spec, step, goal))
    assert not result['success'] and not result['detail']['operation_seen']
