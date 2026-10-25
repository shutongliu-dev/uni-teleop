"""Operator-installed controller factories bound to frozen task robot geometry."""
from functools import partial
import json

from bench.protocol.policies import PolicyRegistry
from bench.control.factory import configured_policy, public_robot_geometry, robot_configuration
from bench.isaacsim.release import task_inputs, validate


def _configured_policy(configuration_json, board_game=False):
    """Reconstruct the operator-captured task configuration for one episode."""
    configuration = json.loads(configuration_json)
    if board_game:
        from .board_game_control import BoardGameFingertipPolicy
        policy = BoardGameFingertipPolicy(**configuration)
        policy.robot_configuration = json.loads(configuration_json)
        return policy
    return configured_policy(configuration)


def default_registry(release):
    """Pin task-specific construction without changing the controller algorithm.

    A release may vary robot placement between tasks. The factory captures only
    validated static robot configuration, never native state or task objectives.
    Each episode gets a new executor whose descriptor must equal its startup pin.
    """
    manifest = validate(release, assets=False)
    if not manifest['tasks']:
        raise ValueError('Default controller requires at least one frozen task')
    registry = PolicyRegistry()
    descriptors = {}
    for entry in manifest['tasks']:
        _, scene, task, _ = task_inputs(release, entry['id'])
        configuration = robot_configuration(scene)
        board_game = task.get('schema') == 'robot_board_game_task.v1' or entry.get('task_kind', manifest.get('task_kind')) == 'turn_game'
        captured = json.dumps(configuration, sort_keys=True, separators=(',', ':'), allow_nan=False)
        factory = partial(_configured_policy, captured, board_game)
        key = (captured, board_game)
        if key not in descriptors:
            descriptors[key] = factory().descriptor.model_copy(deep=True)
        registry.register_task_factory(entry['id'], descriptors[key], factory)
    return registry
