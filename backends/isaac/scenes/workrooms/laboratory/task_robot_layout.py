"""Reuse the reviewed two-bench installation for every laboratory task."""
from . import robot_layout
from .rules import TASK_IDS


def robot_configuration(task_id):
    if task_id not in TASK_IDS:
        raise ValueError('Unknown laboratory task')
    return robot_layout.robot_configuration()


def ready_configuration(task_id, robot):
    if task_id not in TASK_IDS:
        raise ValueError('Unknown laboratory task')
    return robot_layout.ready_configuration(robot)


def placement_transform(task_id):
    if task_id not in TASK_IDS:
        raise ValueError('Unknown laboratory task')
    return robot_layout.placement_transform()


def base_quaternion(task_id):
    if task_id not in TASK_IDS:
        raise ValueError('Unknown laboratory task')
    return robot_layout.ROBOT_BASE_QUATERNION_WXYZ
