"""Build robot-only geometry and executors from static scene configuration.

No task catalog, release loader, native state or privileged object geometry is
used here. Runtime sessions and scene engineering tools share this boundary.
"""
from __future__ import annotations

from collections.abc import Mapping
import json
from pathlib import Path

from bench.paths import DATA_ROOT


def robot_configuration(scene: Mapping) -> dict:
    """Copy only static robot geometry, retaining historical default-key omission."""
    robot = scene['robot']
    configuration = {'assets_root': str(Path(robot.get('assets_root', f'{DATA_ROOT}/assets')).resolve()),
                     'base_position_m': list(robot.get('base_position_m', [0., 0., .77])),
                     'base_quaternion_wxyz': list(robot.get('base_quaternion_wxyz', [1., 0., 0., 0.]))}
    from bench.simulation.isaacsim_backend.robot_instances import ROBOT_ID, ASSEMBLY_VERSION, checked_instances
    if robot.get('hand_id') == ROBOT_ID:
        return {**configuration, 'hand_id': ROBOT_ID, 'assembly_version': ASSEMBLY_VERSION,
                'instances': checked_instances(robot)}
    if 'instances' in robot:
        raise ValueError('Instance declarations require the instance robot embodiment')
    if 'arm_base_poses' in robot:
        from bench.simulation.isaacsim_backend.robot_layout import arm_base_poses
        configuration['arm_base_poses'] = arm_base_poses(robot)
    # Only a non-default hand adds a key, so existing SHARPA configurations are unchanged.
    if robot.get('hand_id', 'sharpa_w01_right') != 'sharpa_w01_right':
        configuration['hand_id'] = robot['hand_id']
    # Likewise only an assembly other than the hand's v1-era default (a mount
    # convention change, e.g. flange_axis_centred_v2) adds the version key.
    from bench.simulation.isaacsim_backend.robot_registry import get_hand_spec
    version = robot.get('assembly_version')
    if version is not None and version != get_hand_spec(robot.get('hand_id', 'sharpa_w01_right'))['assembly_version']:
        configuration['assembly_version'] = version
    return configuration


def public_robot_geometry(configuration: Mapping):
    """Robot-only public FK for a pinned configuration (single arm or bimanual)."""
    configuration = _snapshot(configuration)
    from bench.control.fingertip import Fr3SharpaGeometry
    if configuration.get('hand_id') in ('dual_fr3_sharpa_w01', 'fr3_sharpa_instances'):
        from bench.control.bimanual import BimanualGeometry
        return BimanualGeometry(**configuration)
    return Fr3SharpaGeometry(**configuration)


def _snapshot(configuration: Mapping) -> dict:
    """Own an independent, finite JSON value instead of the caller's mapping."""
    return json.loads(json.dumps(dict(configuration), sort_keys=True, separators=(',', ':'), allow_nan=False))


def configured_policy(configuration: Mapping):
    """Create a fresh executor with the unchanged controller descriptor."""
    from bench.control.fingertip import ClassicalFingertipPolicy
    captured = _snapshot(configuration)
    if captured.get('hand_id') in ('dual_fr3_sharpa_w01', 'fr3_sharpa_instances'):
        from bench.control.bimanual import BimanualFingertipPolicy
        policy = BimanualFingertipPolicy(**captured)
    else:
        policy = ClassicalFingertipPolicy(**captured)
    policy.robot_configuration = _snapshot(captured)
    return policy
