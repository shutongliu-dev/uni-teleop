"""Load the published kitchen task candidate or prepare a new asset-bound input."""
from copy import deepcopy
from pathlib import Path

from .definition import TASK


def task_configuration(*, seed=0):
    from bench.isaacsim.release import checked, task_inputs
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError('seed must be an unsigned 32-bit integer')
    reference = TASK.get('source_release')
    if TASK.get('status') != 'implemented' or not reference:
        raise ValueError('Kitchen task has no published executable release')
    root = Path(reference['path'])
    checked(root/'release_manifest.json', reference['manifest_sha256'])
    _, scene, task, initial = task_inputs(root, TASK['task_id'], seed=seed)
    return {'schema': 'kitchen_task_configuration.v1', 'task_id': TASK['task_id'],
        'status': 'implemented', 'seed': seed, 'environment': deepcopy(scene['workroom']),
        'robot_requirements': deepcopy(TASK['robot_requirements']),
        'rules': deepcopy(task['public_rules']), 'public_goal': deepcopy(task['public_goal']),
        'scene': scene, 'task': task, 'initial_state': initial,
        'native_execution_available': True, 'source_release': deepcopy(reference),
        'qualification': deepcopy(TASK['qualification'])}


def build_task_inputs(*, asset_manifest, seed=0):
    from scenes.workrooms.kitchen.composition import build_task_inputs as build
    return build('breakfast_serving', asset_manifest=asset_manifest, seed=seed)
