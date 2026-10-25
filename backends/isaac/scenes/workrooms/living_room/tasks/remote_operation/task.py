"""Load the frozen executable remote_operation candidate without starting Isaac."""
from copy import deepcopy
from pathlib import Path

from .definition import TASK


def _published_configuration(seed):
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError('seed must be an unsigned 32-bit integer')
    from bench.isaacsim.release import checked, task_inputs
    reference = TASK['source_release']
    root = Path(reference['path'])
    checked(root/'release_manifest.json', reference['manifest_sha256'])
    _, scene, task, initial = task_inputs(root, TASK['task_id'], seed=seed)
    return {'schema': 'living_room_task_configuration.v1', 'task_id': TASK['task_id'],
        'task_number': TASK['task_number'], 'status': 'implemented', 'seed': seed,
        'environment': deepcopy(scene['workroom']),
        'robot_requirements': deepcopy(TASK['robot_requirements']),
        'rules': deepcopy(task['public_rules']), 'public_goal': deepcopy(task['public_goal']),
        'scene': scene, 'task': task, 'initial_state': initial,
        'native_execution_available': True, 'source_release': deepcopy(reference),
        'qualification': deepcopy(TASK['qualification'])}


def task_configuration(*, seed=0):
    """Return verified canonical scene/task/reset plus convenient public fields."""
    return _published_configuration(seed)


def build_task_inputs(*, seed=0):
    """Author inputs for preparing a new release; use task_configuration to load the pin."""
    from scenes.workrooms.living_room.runtime_task import build_task_inputs as build
    return build('remote_operation', seed=seed)
