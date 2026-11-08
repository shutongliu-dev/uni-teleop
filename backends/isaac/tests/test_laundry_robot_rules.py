"""Synthetic rule counterexamples; no native robot manipulation qualification."""
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest

from bench.paths import DATA_ROOT, load_json
from scenes.workrooms.laundry_room.evaluation import SequenceEvaluator
from scenes.workrooms.laundry_room.rules import (
    ROBOT_CONTROL, bind_task_rules, digest, public_rules)
from test_laundry_practical_rules import task_inputs, goal_vertices, sample


def robot_task(task_id='towel_folding'):
    task = task_inputs(task_id)
    task['evaluation']['control_mode'] = ROBOT_CONTROL
    task['public_rules'] = public_rules(task_id, control_mode=ROBOT_CONTROL)
    task['evaluation']['public_rules_sha256'] = digest(task['public_rules'])
    return bind_task_rules(task)


def robot_frame(task, step, vertices=None, *, near=False, x=0., separated=None):
    frame = sample(task, step, vertices)
    frame['control_mode'] = ROBOT_CONTROL
    frame['robot_interaction'] = {
        'schema': 'laundry_robot_interaction.v1', 'complete': True,
        'method': 'native_cooked_convex_geometry', 'hand_near_cloth': near,
        'fully_separated': not near if separated is None else separated,
        'hands': {side: {'wrist_position_m': [x, y, 1.1],
                        'near_towel': near and side == 'left',
                        'near_any_cloth': near and side == 'left'}
                  for side, y in (('left', -.37), ('right', -.73))}}
    return frame


def finish(task, scorer, goal, *, first=3, last=253, **kwargs):
    for step in range(first, last+1):
        score = scorer.observe(robot_frame(task, step, goal, **kwargs))
    return score


def operated(task):
    scorer = SequenceEvaluator(task)
    scorer.observe(robot_frame(task, 0))
    scorer.observe(robot_frame(task, 1, near=True))
    goal = goal_vertices(task)
    scorer.observe(robot_frame(task, 2, goal, near=True, x=.015))
    return scorer, goal


@pytest.mark.parametrize('task_id', ['towel_flattening', 'towel_folding', 'towel_stacking'])
def test_one_hand_operation_and_full_release_pass_without_stricter_geometry(task_id):
    task = robot_task(task_id)
    scorer, goal = operated(task)
    # Retain the original practical shape and placement tolerances.
    goal['towel'][:, 0] += .022 if task_id == 'towel_flattening' else .035
    result = finish(task, scorer, goal, last=254)
    assert result['success'], result
    assert result['control_mode'] == ROBOT_CONTROL
    assert 'engineering_control_mode' not in result
    assert not result['robot_manipulation_qualified']
    assert not scorer.observe(robot_frame(task, 255, goal, near=True))['success']


@pytest.mark.parametrize('case', ['reset', 'passive_fall', 'air_motion', 'static_near', 'tiny_jitter', 'broken_contact'])
def test_reset_passive_motion_and_insufficient_hand_evidence_do_not_earn_operation(case):
    task = robot_task()
    scorer = SequenceEvaluator(task)
    goal = goal_vertices(task)
    for step in range(254):
        vertices = goal if case == 'reset' or step else None
        near = case in ('static_near', 'tiny_jitter') or case == 'broken_contact' and step % 2 == 0
        x = .002*(-1)**step if case == 'tiny_jitter' else step*.002
        if case == 'static_near':
            x = 0.
        result = scorer.observe(robot_frame(task, step, vertices, near=near, x=x))
    assert not result['success']
    assert not result['detail']['operation_seen']


def test_arm_support_blocks_release_even_when_both_hands_are_clear():
    task = robot_task()
    scorer, goal = operated(task)
    result = finish(task, scorer, goal, separated=False)
    assert result['detail']['operation_seen']
    assert not result['detail']['all_control_released']
    assert result['released_stable_s'] == 0
    assert not result['success']
    assert finish(task, scorer, goal, first=254, last=504)['success']


@pytest.mark.parametrize('damage', ['absent', 'incomplete', 'wrong_method', 'wrong_side', 'nonfinite', 'contradiction', 'mask', 'wrong_mode', 'null_evidence', 'null_hand'])
def test_missing_or_false_native_interaction_evidence_never_scores(damage):
    task = robot_task()
    frame = robot_frame(task, 0)
    e = frame['robot_interaction']
    if damage == 'absent':
        del frame['robot_interaction']
    elif damage == 'incomplete':
        e['complete'] = False
    elif damage == 'wrong_method':
        e['method'] = 'wrist_distance_proxy'
    elif damage == 'wrong_side':
        e['hands']['front'] = e['hands'].pop('left')
    elif damage == 'nonfinite':
        e['hands']['left']['wrist_position_m'][0] = float('nan')
    elif damage == 'contradiction':
        e['hands']['left'].update(near_towel=True, near_any_cloth=True)
    elif damage == 'null_evidence':
        frame['robot_interaction'] = None
    elif damage == 'null_hand':
        e['hands']['left'] = None
    elif damage == 'mask':
        frame['controlled_vertices']['towel'] = [0]
    else:
        frame['control_mode'] = 'engineering_local_grasp'
    result = SequenceEvaluator(task).observe(frame)
    assert result['status'] == 'unknown' and not result['success']


@pytest.mark.parametrize('task_id', ['towel_flattening', 'towel_folding', 'towel_stacking'])
def test_existing_practical_engineering_rules_keep_exact_published_identity(task_id):
    root = Path(f'{DATA_ROOT}/datasets/fr3_sharpa_bench/development/laundry_tasks_20261021/releases')
    expected = load_json(root/task_id/'practical_v2/rules.json')
    assert public_rules(task_id) == expected


def test_unsupported_robot_modes_are_rejected():
    with pytest.raises(ValueError):
        public_rules('towel_folding', profile='laundry_strict_task.v1', control_mode=ROBOT_CONTROL)
