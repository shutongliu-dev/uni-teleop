"""Explicit instances of unchanged FR3/right-SHARPA source assemblies.

Namespaces identify actuators, not hand chirality. Poses are relative to the
common robot base. No task, controller gain, mesh, mass or inertia lives here.
"""
from copy import deepcopy
import re

ROBOT_ID = 'fr3_sharpa_instances'
ASSEMBLY_VERSION = 'namespaced_v1'


def checked_namespace(value):
    if (not isinstance(value, str) or re.fullmatch(r'[a-z][a-z0-9]{0,23}', value) is None
            or value == 'fr3'):
        raise ValueError('Robot instance namespace must be 1–24 lowercase alphanumeric characters, not fr3')
    return value


def instance_name(namespace, source_name):
    """Rename source identifiers only; the source hand is always the right hand."""
    checked_namespace(namespace)
    if source_name == 'base':
        return f'fr3_{namespace}_base'
    if source_name.startswith('fr3_'):
        return f'fr3_{namespace}_' + source_name[4:]
    if source_name.startswith('right_'):
        return namespace + source_name[5:]
    raise ValueError(f'Unsupported original FR3/right-SHARPA identifier: {source_name!r}')


def checked_instances(robot):
    from .robot_layout import checked_pose
    from .robot_registry import get_hand_spec

    if robot.get('hand_id') != ROBOT_ID or robot.get('assembly_version') != ASSEMBLY_VERSION:
        raise ValueError('Explicit robot instances require the namespaced_v1 composition')
    if 'arm_base_poses' in robot:
        raise ValueError('Robot instances declare their own base poses, not arm_base_poses')
    instances = robot.get('instances')
    if not isinstance(instances, dict) or not 2 <= len(instances) <= 8:
        raise ValueError('Robot instances require 2–8 named source configurations')
    for name in instances:
        checked_namespace(name)
    result = {}
    required = {'hand_id', 'assembly_version', 'base_position_m', 'base_quaternion_wxyz'}
    for name in sorted(instances):
        item = instances[name]
        if not isinstance(item, dict) or set(item) != required:
            raise ValueError('Each instance requires only pinned hand_id, assembly_version and rigid base pose')
        if item['hand_id'] != 'sharpa_w01_right':
            raise ValueError('Instance composition currently supports original right SHARPA assemblies only')
        if not isinstance(item['assembly_version'], str):
            raise ValueError('Each instance must explicitly pin its source assembly version')
        get_hand_spec(item['hand_id'], item['assembly_version'])
        pose = checked_pose(item['base_position_m'], item['base_quaternion_wxyz'], name)
        result[name] = {'hand_id': item['hand_id'], 'assembly_version': item['assembly_version'],
                        'base_position_m': pose['position_m'],
                        'base_quaternion_wxyz': pose['quaternion_wxyz']}
    return result


def instance_spec(robot):
    """Resolve the full robot inventory from its declaration, never a task ID."""
    from .robot_registry import get_hand_spec
    instances = checked_instances(robot)
    sources = {name: get_hand_spec(item['hand_id'], item['assembly_version'])
               for name, item in instances.items()}
    first = next(iter(sources.values()))
    spec = deepcopy(first)
    spec.update(hand_id=ROBOT_ID, label='Independent FR3 + original right SHARPA instances',
                assembly_version=ASSEMBLY_VERSION, release_assembly_version=ASSEMBLY_VERSION,
                registered_assembly_versions=[ASSEMBLY_VERSION],
                instances=instances, source_hand_specs=sources, bimanual=True,
                arms={name: {'base_offset_m': item['base_position_m']} for name, item in instances.items()},
                arm_dof=7*len(instances), hand_dof=22*len(instances), joint_count=29*len(instances),
                actuated_joint_count=29*len(instances), prefix='', mimic_joints={},
                control_model=f'{len(instances)} independent original FR3/right-SHARPA robots; 29 position drives each',
                # This is a composition frame, not a replacement physical mount.
                mount_translation_m=[0., 0., 0.], mount_rpy_rad=[0., 0., 0.],
                mount_quaternion_wxyz=[1., 0., 0., 0.], mount_convention='per_instance_source',
                hand_frame_yaw_rad=0.,
                finger_names=list(first.get('finger_names') or first['fingertip_links']),
                fingertip_links={f'{name}_{finger}': instance_name(name, link)
                    for name, source in sources.items() for finger, link in source['fingertip_links'].items()},
                pinch_links={name: {finger: [instance_name(name, link) for link in links]
                    for finger, links in source['pinch_links'].items()} for name, source in sources.items()},
                palm_link=instance_name(next(iter(instances)), first['palm_link']),
                arm_home_rad_by_side={name: deepcopy(robot.get('arm_home_rad_by_side', {}).get(name,
                    robot.get('arm_home_rad', [0., -.785, 0., -2.356, 0., 1.57, .785]))) for name in instances})
    return spec
