"""Load the frozen executable water_pouring candidate without starting Isaac."""
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


def design_configuration(*, seed=0, quantities=None):
    """Inspect an unfrozen water proposal; never relabel it as native task input."""
    from scenes.workrooms.living_room.scene import environment_reference
    from .rules import public_rules
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError('seed must be an unsigned 32-bit integer')
    rules = public_rules(quantities=quantities)
    rules['status'] = 'design_only_proposal'
    rules['goal']['status'] = 'design_only_proposal'
    rules['qualification'] = 'Unfrozen quantity proposal; requires matching physical inventory, a new release and separate validation.'
    return {'schema': 'living_room_task_plan.v1', 'task_id': TASK['task_id'],
        'task_number': TASK['task_number'], 'status': 'planned', 'seed': seed,
        'configuration_scope': 'design_only_proposal',
        'environment': environment_reference(),
        'robot_requirements': deepcopy(TASK['robot_requirements']),
        'rules': rules, 'public_goal': deepcopy(rules['goal']),
        'initial_state': None, 'source_release': None, 'native_execution_available': False,
        'requires_new_release': True,
        'cannot_run_reason': 'Quantity changes are a design proposal; the runnable release fixes 250 mL initial water, 100 +/- 10 mL target and 5 mL spill budget.'}


def task_configuration(*, seed=0, quantities=None):
    """Load nominal frozen inputs, or explicitly mark nondefault quantities as design."""
    from .rules import public_rules
    proposed = public_rules(quantities=quantities)
    if proposed['goal'] != public_rules()['goal']:
        return design_configuration(seed=seed, quantities=quantities)
    return _published_configuration(seed)


def build_task_inputs(*, seed=0):
    """Author inputs for preparing a new release; use task_configuration to load the pin."""
    from scenes.workrooms.living_room.runtime_task import build_task_inputs as build
    return build('water_pouring', seed=seed)
