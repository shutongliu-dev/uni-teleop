"""Bind two physical robots to exactly one frozen ordinary task instance.

This is an input identity check, not IK, collision or manipulation evidence.
The L-mount release must be the existing narrow robot-only derivation of the
original-robot release. Merely having equal seed labels is insufficient.
"""
from copy import deepcopy
import json
from pathlib import Path

from bench.paths import load_json, OUTPUT_ROOT
from scenes.workrooms.robot_revision.inputs import (
    canonical_hash, derive_inputs, RESET_SCHEMAS)
from scenes.workrooms.robot_revision.assets import file_hash, checked_file

SCHEMA = 'parameterized_robot_pair.v1'
PROFILES = {'original': 'isaacsim_6_1_v2', 'l_mount': 'isaacsim_6_1_l_mount_v3'}
SEEDS = tuple(range(10))


def read(path):
    return load_json(Path(path))


def reference(root):
    root = Path(root).resolve()
    return {'path': str(root), 'manifest_sha256': file_hash(root/'release_manifest.json')}


def _same(left, right):
    return canonical_hash(left) == canonical_hash(right)


def _checked_case(case):
    if (not isinstance(case, dict) or case.get('schema') not in RESET_SCHEMAS
            or case.get('case_sha256') != canonical_hash(
                {key: value for key, value in case.items() if key != 'case_sha256'})):
        raise ValueError('Invalid frozen task instance identity')
    result = deepcopy(case)
    for name in ('case_sha256', 'case_fingerprint', 'robot_revision'):
        result.pop(name, None)
    return result


def physical_instance(scene, task, initial):
    """Retain all task/world/solver/appearance values, omit robot-bound copies.

    Removing the robot here only defines the shared physical-task identity;
    assert_same_task separately checks the complete allowed robot derivation,
    including protected controllers, installation bases and non-arm joints.
    """
    world, contract, physical_initial = deepcopy(scene), deepcopy(task), deepcopy(initial)
    world.pop('robot', None)
    physical_initial.pop('joint_positions_rad', None)
    if 'initial_state' in contract:
        if not _same(contract['initial_state'], initial):
            raise ValueError('Task has a different initial-state copy')
        contract['initial_state'] = deepcopy(physical_initial)
    case = _checked_case(contract['reset_case'])
    contract['reset_case'] = case
    for name in ('workroom_reset_case', 'living_room_reset_case', 'laundry_reset_case'):
        if name in world:
            for key in ('case_sha256', 'case_fingerprint'):
                world[name].pop(key, None)
    for container in [world.get('native_task_bundle', {}), *world.get('task_layers', [])]:
        for name in ('observer', 'reduced_dynamics'):
            hook = container.get(name)
            if hook and 'configuration' in hook and _same(hook['configuration'], task):
                # Normalize only the known duplicate. Other private hook
                # configurations remain intact and must compare exactly.
                hook['configuration'] = deepcopy(contract)
    return {'scene': world, 'task': contract, 'initial_state': physical_initial}


def assert_same_task(original, revised, revision):
    """Reject every difference outside the existing robot-only derivation."""
    if len(original) != 3 or len(revised) != 3:
        raise ValueError('Each profile requires scene, task and initial state')
    expected = derive_inputs(*original, revision)
    if not _same(expected, revised):
        raise ValueError('Robot profiles differ beyond robot assets/ready coordinates/lineage')
    old, new = physical_instance(*original), physical_instance(*revised)
    if not _same(old, new):
        raise ValueError('The two robots have different physical task instances')
    for profile, values in (('original', original), ('l_mount', revised)):
        robot = values[0]['robot']
        versions = ({row['assembly_version'] for row in robot['instances'].values()}
                    if 'instances' in robot else {robot['assembly_version']})
        if versions != {PROFILES[profile]}:
            raise ValueError('Wrong physical robot version for '+profile)
    return canonical_hash(old)


def _documents(root, manifest, seed):
    row = manifest['cases'][str(seed)]
    result = []
    for name in ('scene', 'task', 'initial_state'):
        relative = Path(row[name])
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('Case input escapes its frozen release')
        path = checked_file(root/relative, manifest['files_sha256'][str(relative)])
        result.append(read(path))
    case = result[1].get('reset_case', {})
    if type(case.get('seed')) is not int or case['seed'] != seed:
        raise ValueError('Case label differs from actual task seed')
    return result


def pair_record(original_release, revised_release, *, validate_assets=True):
    from bench.isaacsim.release import validate
    original, revised = Path(original_release).resolve(), Path(revised_release).resolve()
    old, new = [validate(path, assets=validate_assets) for path in (original, revised)]
    if (old.get('seed_values') != list(SEEDS) or new.get('seed_values') != list(SEEDS)
            or set(old.get('cases', {})) != set(map(str, SEEDS))
            or set(new.get('cases', {})) != set(map(str, SEEDS))
            or new.get('schema') != 'workroom_robot_revision_release.v1'
            or new.get('base_release') != {**reference(original), 'schema': old['schema']}
            or old.get('task_id') != new.get('task_id')
            or old.get('configuration_id') != new.get('configuration_id')):
        raise ValueError('Require ten exact task cases and their robot-only derivative')
    revision = read(checked_file(revised/new['robot_revision'],
                                new['files_sha256'][new['robot_revision']]))
    identities = {}
    for seed in SEEDS:
        before, after = _documents(original, old, seed), _documents(revised, new, seed)
        identities[str(seed)] = assert_same_task(before, after, revision)
    return {'schema': SCHEMA, 'family': new['family'], 'task_id': old['task_id'],
            'configuration_id': old['configuration_id'],
            'robot_profiles': {key: reference(path) for key, path in
                               (('original', original), ('l_mount', revised))},
            'robot_assembly_versions': deepcopy(PROFILES),
            'seed_values': list(SEEDS), 'physical_case_sha256': identities,
            'robot_revision_sha256': canonical_hash(revision),
            'scope': 'Exact shared task/world inputs; no physical acceptance or manipulation claim.'}


def write_pair(original_release, revised_release, output):
    output = Path(output).resolve()
    if not output.is_relative_to(OUTPUT_ROOT) or output.exists():
        raise ValueError('Use a fresh shared dataset path for the pair identity')
    record = pair_record(original_release, revised_release)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as stream:
        json.dump(record, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    return {'path': str(output), 'sha256': file_hash(output)}
