"""Narrow input derivation, with no simulator access or task-object changes."""
from copy import deepcopy
import hashlib
import json
import math
import re

from bench.paths import portable
from . import REVISION_SCHEMA

# Physical robot assets and explicitly reviewed ready coordinates may change.
# World installations, contact/controller parameters and ownership may not.
ROBOT_MUTABLE = frozenset({'assembly_version', 'mount_position_m',
    'mount_quaternion_wxyz', 'arm_home_rad', 'arm_home_rad_by_side', 'instances', 'mounts_by_side'})
INSTANCE_MUTABLE = frozenset({'assembly_version', 'mount_position_m', 'mount_quaternion_wxyz'})
RESET_SCHEMAS = frozenset({'workroom_reset_case.v1', 'living_room_reset_case.v1',
                           'laundry_reset_case.v1'})


def canonical_hash(value):
    # Identities hash data:// references, never this machine's data root.
    return hashlib.sha256(json.dumps(portable(value), sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def _same(a, b):
    # Python's False == 0 must never make an input-preservation check pass.
    return canonical_hash(a) == canonical_hash(b)


def _vector(value, length):
    if (not isinstance(value, list) or len(value) != length
            or any(type(v) not in (int, float) or not math.isfinite(v) for v in value)):
        raise ValueError('Robot coordinates must be finite numeric vectors')


def _check_robot(original, replacement):
    if not isinstance(original, dict) or not isinstance(replacement, dict):
        raise ValueError('Original and replacement robot objects are required')
    if set(original)-set(replacement) or set(replacement)-set(original)-{'mounts_by_side'}:
        raise ValueError('Robot revision cannot add or remove undeclared robot fields')
    for name in original:
        if name not in ROBOT_MUTABLE and not _same(original[name], replacement[name]):
            raise ValueError('Robot revision changed a protected field: '+name)
    if 'instances' in original:
        before, after = original['instances'], replacement['instances']
        if set(before) != set(after):
            raise ValueError('Robot instance ownership changed')
        for side in before:
            if set(before[side]) != set(after[side]):
                raise ValueError('Robot instance field inventory changed')
            for name in before[side]:
                if name not in INSTANCE_MUTABLE and not _same(before[side][name], after[side][name]):
                    raise ValueError('Robot instance installation/chirality changed')
    if 'mounts_by_side' in replacement:
        mounts = replacement['mounts_by_side']
        sides = (set(original['instances']) if 'instances' in original else
                 {'left', 'right'} if original['hand_id'] == 'dual_fr3_sharpa_w01' else {'right'})
        if not isinstance(mounts, dict) or set(mounts) != sides:
            raise ValueError('Per-side mounts must preserve original robot ownership')
        for mount in mounts.values():
            if not isinstance(mount, dict) or set(mount) != {'translation_m', 'quaternion_wxyz', 'rpy_rad'}:
                raise ValueError('Per-side mount requires complete SE3 metadata')
            _vector(mount['translation_m'], 3); _vector(mount['rpy_rad'], 3)
            _vector(mount['quaternion_wxyz'], 4)
            if abs(sum(v*v for v in mount['quaternion_wxyz'])-1.) > 1e-6:
                raise ValueError('Per-side mount quaternion must be normalized')
    if 'mount_position_m' in replacement:
        _vector(replacement['mount_position_m'], 3)
    if 'mount_quaternion_wxyz' in replacement:
        q = replacement['mount_quaternion_wxyz']; _vector(q, 4)
        if abs(sum(v*v for v in q)-1.) > 1e-6:
            raise ValueError('Robot mount quaternion must be normalized')
    if 'arm_home_rad' in replacement:
        _vector(replacement['arm_home_rad'], 7)
    if 'arm_home_rad_by_side' in original:
        old, new = original['arm_home_rad_by_side'], replacement['arm_home_rad_by_side']
        if set(old) != set(new):
            raise ValueError('Ready coordinates changed robot ownership')
        for q in new.values():
            _vector(q, 7)


def checked_revision(revision, scene, initial):
    """Check an explicit per-installation descriptor before deriving any input.

    Asset dependency and frozen-source checks belong to the release publisher;
    this function only establishes the permitted semantic input differences.
    """
    required = {'schema', 'revision_id', 'parent_robot_sha256', 'robot',
                'joint_positions_rad', 'robot_asset_manifest_sha256'}
    if not isinstance(revision, dict) or set(revision) != required or revision['schema'] != REVISION_SCHEMA:
        raise ValueError('Invalid robot revision descriptor')
    if not isinstance(revision['revision_id'], str) or not revision['revision_id']:
        raise ValueError('Robot revision needs an explicit version identity')
    pin = revision['robot_asset_manifest_sha256']
    if not isinstance(pin, str) or len(pin) != 64 or any(c not in '0123456789abcdef' for c in pin):
        raise ValueError('Robot revision must bind its immutable asset manifest')
    if revision['parent_robot_sha256'] != canonical_hash(scene['robot']):
        raise ValueError('Robot revision belongs to another original installation')
    _check_robot(scene['robot'], revision['robot'])
    old = initial.get('joint_positions_rad'); new = revision['joint_positions_rad']
    if (not isinstance(old, dict) or not old or not isinstance(new, dict) or set(old) != set(new)
            or any(type(v) not in (int, float) or not math.isfinite(v) for v in new.values())):
        raise ValueError('Ready reset must preserve the complete original joint-name set')
    for name, value in old.items():
        if not re.fullmatch(r'fr3_(?:(?:left|right|front|rear)_)?joint[1-7]', name) and not _same(value,new[name]):
            raise ValueError('Robot ready derivation cannot change hand or non-arm joints')
    return deepcopy(revision)


def _reset_case(case, revision_hash):
    if (not isinstance(case, dict) or case.get('schema') not in RESET_SCHEMAS
            or case.get('case_sha256') != canonical_hash({k: v for k, v in case.items() if k != 'case_sha256'})
            or not isinstance(case.get('case_fingerprint'), str)):
        raise ValueError('Original reset case identity is invalid')
    if 'robot_revision' in case:
        raise ValueError('Nested robot reset revisions are not supported')
    result = deepcopy(case)
    result['robot_revision'] = {'schema': 'robot_revision_case_lineage.v1',
        'revision_sha256': revision_hash, 'parent_case_sha256': case['case_sha256'],
        'parent_case_fingerprint': case['case_fingerprint']}
    result['case_fingerprint'] = canonical_hash({'parent_case_fingerprint': case['case_fingerprint'],
                                                'robot_revision_sha256': revision_hash})
    result['case_sha256'] = canonical_hash({k: v for k, v in result.items() if k != 'case_sha256'})
    return result


def derive_inputs(scene, task, initial, revision):
    """Copy inputs and apply only declared robot/ready changes and new identity.

    Callers retain the original frozen evaluator and rules. No original object,
    cloth, fixture, room, numerical threshold or controller value is modified.
    Practice has a separate identity adapter and cannot use this ordinary path.
    """
    revision = checked_revision(revision, scene, initial)
    if 'practice_case' in task:
        raise ValueError('Practice requires its explicit case identity adapter')
    new_scene, new_task, new_initial = deepcopy(scene), deepcopy(task), deepcopy(initial)
    new_scene['robot'] = deepcopy(revision['robot'])
    new_initial['joint_positions_rad'] = deepcopy(revision['joint_positions_rad'])
    if 'initial_state' in task:
        if not _same(task['initial_state'], initial):
            raise ValueError('Task and original initial-state inputs disagree')
        new_task['initial_state'] = deepcopy(new_initial)
    revision_hash = canonical_hash(revision)
    if 'reset_case' in task:
        new_task['reset_case'] = _reset_case(task['reset_case'], revision_hash)
        case = new_task['reset_case']
        # Existing family summaries have different whitelists. Retain their keys.
        for name in ('workroom_reset_case', 'living_room_reset_case', 'laundry_reset_case'):
            if name in new_scene:
                summary = new_scene[name]
                for key in ('case_sha256', 'case_fingerprint'):
                    if key in summary:
                        if summary[key] != task['reset_case'][key]:
                            raise ValueError('Scene and task original case identities disagree')
                        summary[key] = case[key]
    # Native task hooks carry a duplicate task configuration; it must follow
    # identity/ready changes, with evaluator source and all physical settings intact.
    for container in [new_scene.get('native_task_bundle', {}), *new_scene.get('task_layers', [])]:
        for name in ('observer', 'reduced_dynamics'):
            hook = container.get(name)
            if hook and 'configuration' in hook and _same(hook['configuration'], task):
                hook['configuration'] = deepcopy(new_task)
    return new_scene, new_task, new_initial


def descriptor(scene, initial, *, joint_positions_rad, assembly_version,
               mounts_by_side, robot_asset_manifest_sha256):
    """Build an explicit input descriptor from a separately checked ready map.

    This does not solve IK, assert collision clearance or transfer qualification.
    The publisher separately binds assembly_version and mounts to actual assets.
    """
    robot = deepcopy(scene['robot']); joints=deepcopy(joint_positions_rad)
    if robot['hand_id']=='fr3_sharpa_instances':
        for instance in robot['instances'].values():instance['assembly_version']=assembly_version
    else:
        robot.update(assembly_version=assembly_version,
            mount_position_m=deepcopy(mounts_by_side['right']['translation_m']),
            mount_quaternion_wxyz=deepcopy(mounts_by_side['right']['quaternion_wxyz']),
            mounts_by_side=deepcopy(mounts_by_side if robot['hand_id']=='dual_fr3_sharpa_w01'
                                   else {'right':mounts_by_side['right']}))
    if 'arm_home_rad' in robot:
        robot['arm_home_rad']=[joints[f'fr3_joint{i}'] for i in range(1,8)]
    if 'arm_home_rad_by_side' in robot:
        robot['arm_home_rad_by_side']={side:[joints[f'fr3_{side}_joint{i}'] for i in range(1,8)]
                                       for side in robot['arm_home_rad_by_side']}
    value={'schema':REVISION_SCHEMA,'revision_id':assembly_version,
        'parent_robot_sha256':canonical_hash(scene['robot']),'robot':robot,
        'joint_positions_rad':joints,'robot_asset_manifest_sha256':robot_asset_manifest_sha256}
    return checked_revision(value,scene,initial)
