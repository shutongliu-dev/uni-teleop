"""Practical task outcomes and counterexamples; synthetic CPU evidence only."""
from copy import deepcopy

import numpy as np
import pytest

from scenes.workrooms.laundry_room import assets
from scenes.workrooms.laundry_room.composition import build_task_inputs
from scenes.workrooms.laundry_room.evaluation import FRAME_SCHEMA, SequenceEvaluator
from scenes.workrooms.laundry_room.rules import LEGACY_PROFILE, PRACTICAL_PROFILE, TASK_IDS


def task_inputs(task_id='towel_flattening', *, profile=None):
    options = {} if profile is None else {'rule_profile': profile}
    return build_task_inputs(task_id, physics_options={'physics_dt_s': .002}, **options)['task']


def goal_vertices(task):
    result = {}
    for spec in task['asset_metadata']['cloths']:
        rest = np.asarray(spec['rest_vertices_m'])
        if task['id'] == 'towel_flattening':
            result[spec['id']] = rest+[.65, -.55, .9015]
        else:
            z = .9105 if task['id'] == 'towel_stacking' and spec['id'] == 'towel' else .9015
            result[spec['id']] = assets.folded_vertices(
                rest, task['asset_metadata']['target_center_xy_m'], lower_z=z)
    return result


def sample(task, step, vertices=None, *, controlled=False, physics=None, speeds=None):
    specs = task['asset_metadata']['cloths']
    p = {'complete': True, 'valid': True, 'gravity_enabled': True,
         'hidden_constraints': 0, 'self_intersections': 0,
         'intercloth_intersections': 0, 'max_penetration_m': 0.}
    p.update(physics or {})
    return {'schema': FRAME_SCHEMA, 'complete': True,
        'time_s': step*task['evaluation']['physics_dt_s'], 'native_step': step,
        'control_mode': 'engineering_local_grasp', 'physics': p,
        'controlled_vertices': {s['id']: [0, 1] if controlled and s['id'] == 'towel' else [] for s in specs},
        'cloths': {s['id']: {'enabled': True,
            'vertices_m': np.asarray(s['initial_vertices_m']) if vertices is None else vertices[s['id']],
            'velocities_m_s': np.zeros_like(s['rest_vertices_m'], dtype=float)
                if speeds is None else speeds[s['id']],
            'triangles': s['triangles']} for s in specs}}


def start(task, goal):
    scorer = SequenceEvaluator(task)
    assert not scorer.observe(sample(task, 0))['success']
    assert not scorer.observe(sample(task, 1, goal, controlled=True))['success']
    return scorer


def finish(task, scorer, goal, *, first=2, last=252, **kwargs):
    for step in range(first, last+1):
        score = scorer.observe(sample(task, step, goal, **kwargs))
    return score


@pytest.mark.parametrize('task_id', TASK_IDS)
def test_default_public_rules_and_evaluator_select_practical_profile(task_id):
    task = task_inputs(task_id)
    assert task['evaluation']['rule_profile'] == PRACTICAL_PROFILE
    assert task['public_rules']['rule_profile'] == PRACTICAL_PROFILE
    assert task['public_rules']['schema'] == 'laundry_rules.v2'
    assert task['public_rules']['referee']['thresholds'] == task['evaluation']['thresholds']
    assert SequenceEvaluator(task).profile == PRACTICAL_PROFILE


@pytest.mark.parametrize('task_id', TASK_IDS)
def test_slight_misalignment_wrinkles_and_one_edge_jitter_can_succeed(task_id):
    task = task_inputs(task_id)
    goal = goal_vertices(task)
    goal['towel'][:, 0] += .022 if task_id == 'towel_flattening' else .035
    # A shallow smooth wrinkle leaves support and material ordering intact.
    for v in goal.values():
        v[:, 2] += .002*(1+np.sin((v[:, 1]+.55)*np.pi/.4))
    scorer = start(task, goal)
    speeds = {name: np.zeros_like(v) for name, v in goal.items()}
    speeds['towel'][0, 0] = .2
    for step in range(2, 253):
        jittered = {name: v.copy() for name, v in goal.items()}
        jittered['towel'][0, 0] += .0001*(-1)**step
        score = scorer.observe(sample(task, step, jittered, speeds=speeds))
    assert score['success'], score
    assert score['released_stable_s'] == pytest.approx(.5)
    assert score['strict_diagnostics']['max_node_speed_m_s'] == pytest.approx(.2)
    assert score['detail']['raw_speed_p95_m_s'] == 0
    assert score['detail']['pose_speed_p95_m_s'] == 0
    assert not score['physics_accepted'] and not score['robot_manipulation_qualified']


@pytest.mark.parametrize('task_id', TASK_IDS)
def test_one_and_half_mm_contact_compression_does_not_block_success(task_id):
    task = task_inputs(task_id)
    goal = goal_vertices(task)
    for v in goal.values():
        v[:, 2] -= .0015
    scorer = start(task, goal)
    result = finish(task, scorer, goal, physics={
        'valid': False, 'validity_scope': 'strict_geometry', 'max_penetration_m': .0015})
    assert result['success'], result
    assert result['strict_diagnostics']['violating_frames'] == 251
    assert result['strict_diagnostics']['first_violation_step'] == 2


def test_stability_requires_elapsed_half_second_after_first_released_sample():
    task = task_inputs()
    goal = goal_vertices(task)
    scorer = start(task, goal)
    before = finish(task, scorer, goal, last=251)
    assert not before['success']
    assert before['released_stable_s'] == pytest.approx(.498)
    assert scorer.observe(sample(task, 252, goal))['success']


@pytest.mark.parametrize('task_id', TASK_IDS)
def test_retained_grip_and_reset_without_operation_do_not_succeed(task_id):
    task = task_inputs(task_id)
    goal = goal_vertices(task)
    scorer = start(task, goal)
    assert not finish(task, scorer, goal, controlled=True)['success']
    scorer = SequenceEvaluator(task)
    for step in range(253):
        score = scorer.observe(sample(task, step, goal))
    assert not score['success']
    assert score['detail']['operation_seen'] is False


@pytest.mark.parametrize('task_id', TASK_IDS)
@pytest.mark.parametrize('failure', ('floating', 'misplaced', 'collapsed'))
def test_obvious_goal_failures_remain_failures(task_id, failure):
    task = task_inputs(task_id)
    goal = goal_vertices(task)
    if failure == 'floating':
        goal['towel'][:, 2] += .06
    elif failure == 'misplaced':
        goal['towel'][:, 0] += .12
    elif task_id == 'towel_flattening':
        rest = np.asarray(task['asset_metadata']['cloths'][0]['rest_vertices_m'])
        goal['towel'] = assets.folded_vertices(rest, [.65, -.55])
    else:
        goal['towel'][:, 2] = .9105 if task_id == 'towel_stacking' else .9015
    scorer = start(task, goal)
    assert not finish(task, scorer, goal)['success']


def test_slow_slide_is_not_stable_even_when_each_native_velocity_is_small():
    task = task_inputs()
    goal = goal_vertices(task)
    scorer = start(task, goal)
    for step in range(2, 253):
        shifted = {'towel': goal['towel']+[(step-2)*.000075, 0., 0.]}
        score = scorer.observe(sample(task, step, shifted))
        assert not score['success']
    assert score['detail']['pose_speed_p95_m_s'] < .08
    assert score['released_stable_s'] < .5


@pytest.mark.parametrize('failure', ('self_crossing', 'intercloth_crossing', 'compression', 'stretch'))
def test_small_contact_or_deformation_failure_can_recover_with_new_stability_window(failure):
    task = task_inputs()
    goal = goal_vertices(task)
    scorer = start(task, goal)
    finish(task, scorer, goal, last=19)
    damaged = deepcopy(goal)
    p = {'valid': False, 'validity_scope': 'strict_geometry'}
    if failure == 'self_crossing':
        p['self_intersections'] = 1
    elif failure == 'intercloth_crossing':
        p['intercloth_intersections'] = 1
    elif failure == 'compression':
        p['max_penetration_m'] = .006
    else:
        center = np.array([.65, -.55])
        damaged['towel'][:, :2] = center+1.6*(damaged['towel'][:, :2]-center)
    bad = scorer.observe(sample(task, 20, damaged, physics=p))
    assert bad['status'] == 'ongoing', bad
    assert bad['reason'] == 'contact_recovery_required'
    assert bad['released_stable_s'] == 0
    # The recovered pose may itself have a one-frame finite-difference jump.
    recovered = finish(task, scorer, goal, first=21, last=272)
    assert recovered['success'], recovered
    assert recovered['strict_diagnostics']['violating_frames'] >= 1
    assert recovered['strict_diagnostics']['first_violation_step'] == 20


@pytest.mark.parametrize('failure', ('compression', 'stretch', 'crossings', 'unscoped_invalid_physics'))
def test_gross_failure_and_unexplained_engine_invalidity_stop_task(failure):
    task = task_inputs()
    goal = goal_vertices(task)
    scorer = start(task, goal)
    p = {'valid': False, 'validity_scope': 'strict_geometry'}
    if failure == 'compression':
        p['max_penetration_m'] = .011
    elif failure == 'stretch':
        center = np.array([.65, -.55])
        goal['towel'][:, :2] = center+2.1*(goal['towel'][:, :2]-center)
    elif failure == 'crossings':
        p.update(self_intersections=5, intercloth_intersections=4)
    else:
        p.pop('validity_scope')
    result = scorer.observe(sample(task, 2, goal, physics=p))
    assert result['status'] == 'safety_stop'
    assert not result['success']
    assert scorer.observe(sample(task, 3, goal)) == result


@pytest.mark.parametrize('failure', ('regrasp', 'move', 'hidden_constraint', 'missing_frame'))
def test_success_is_retracted_if_subsequent_evidence_breaks_completion(failure):
    task = task_inputs()
    goal = goal_vertices(task)
    scorer = start(task, goal)
    assert finish(task, scorer, goal)['success']
    if failure == 'move':
        goal['towel'][:, 0] += .1
    next_frame = sample(task, 254 if failure == 'missing_frame' else 253, goal,
        controlled=failure == 'regrasp',
        physics={'hidden_constraints': 1} if failure == 'hidden_constraint' else None)
    result = scorer.observe(next_frame)
    assert not result['success'], result
    assert result['status'] == ('unknown' if failure in ('hidden_constraint', 'missing_frame') else 'ongoing')


@pytest.mark.parametrize('kind', ('raw', 'pose'))
def test_motion_on_one_cloth_cannot_be_diluted_by_stationary_base(kind):
    task = task_inputs('towel_stacking')
    goal = goal_vertices(task)
    scorer = start(task, goal)
    finish(task, scorer, goal, last=251)
    speeds = {name: np.zeros_like(v) for name, v in goal.items()}
    # Six percent of one cloth exceeds P95; pooled two-cloth P95 would miss it.
    n = int(np.ceil(len(goal['towel'])*.06))
    if kind == 'raw':
        speeds['towel'][:n, 0] = .1
    else:
        goal['towel'][:n, 0] += .0002
    result = scorer.observe(sample(task, 252, goal, speeds=speeds))
    assert not result['success']
    assert result['released_stable_s'] == 0
    key = 'raw_speed_p95_m_s' if kind == 'raw' else 'pose_speed_p95_m_s'
    assert result['detail'][key] > .08


def test_zero_raw_velocities_do_not_hide_pose_motion():
    task = task_inputs()
    goal = goal_vertices(task)
    scorer = start(task, goal)
    finish(task, scorer, goal, last=251)
    goal['towel'][:, 0] += .0002
    result = scorer.observe(sample(task, 252, goal))
    assert not result['success']
    assert result['detail']['raw_speed_p95_m_s'] == 0
    assert result['detail']['pose_speed_p95_m_s'] == pytest.approx(.1)


def test_legacy_profile_keeps_max_speed_gate_and_latched_success():
    task = task_inputs(profile=LEGACY_PROFILE)
    assert 'rule_profile' not in task['evaluation']
    assert task['public_rules']['schema'] == 'laundry_rules.v1'
    goal = goal_vertices(task)
    scorer = start(task, goal)
    speeds = {'towel': np.zeros_like(goal['towel'])}
    speeds['towel'][0, 0] = .08
    result = finish(task, scorer, goal, speeds=speeds)
    assert not result['success']
    assert result['released_stable_s'] == 0
    assert scorer.profile == LEGACY_PROFILE
    scorer = start(task, goal)
    result = finish(task, scorer, goal)
    assert result['success']
    assert scorer.observe(sample(task, 253, goal, controlled=True)) == result


def test_legacy_profile_keeps_one_mm_physics_failure_latched():
    task = task_inputs(profile=LEGACY_PROFILE)
    scorer = SequenceEvaluator(task)
    result = scorer.observe(sample(task, 0, physics={
        'valid': False, 'validity_scope': 'strict_geometry', 'max_penetration_m': .0015}))
    assert result['status'] == 'safety_stop'
    assert scorer.observe(sample(task, 1)) == result
