"""A later hand gesture cannot take credit for earlier passive cloth motion."""
import pytest

from scenes.workrooms.laundry_room.evaluation import SequenceEvaluator
from test_laundry_practical_rules import goal_vertices
from test_laundry_robot_rules import finish, robot_frame, robot_task


@pytest.mark.parametrize('task_id', ['towel_folding', 'towel_stacking'])
def test_passive_goal_before_first_touch_cannot_be_credited_by_a_later_hand_gesture(task_id):
    task = robot_task(task_id)
    scorer = SequenceEvaluator(task)
    goal = goal_vertices(task)
    assert not scorer.observe(robot_frame(task, 0))['success']
    # This synthetic motion reaches the goal with both hands clear. It must
    # stay uncredited even if a later nearby hand moves while the cloth rests.
    scorer.observe(robot_frame(task, 1, goal))
    scorer.observe(robot_frame(task, 2, goal, near=True))
    scorer.observe(robot_frame(task, 3, goal, near=True, x=.015))
    result = finish(task, scorer, goal, first=4, last=254)
    assert not result['success'], result
    assert result['detail']['operation_displacement_seen'] is False
    assert result['released_stable_s'] == 0.
