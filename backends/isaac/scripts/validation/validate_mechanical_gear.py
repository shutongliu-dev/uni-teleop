"""Build the zero-model privileged L5 gear-train native probe.

The input binds the prepared v3.4 20/32/20 candidate and describes robot-only
endpoint phases for placing each gear, releasing it on its shaft, engaging a
free tool and withdrawing.  Native execution records contacts and stability;
this builder never writes object poses, applies object forces or awards task
credit.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from scripts.validation import mechanical_robot_phase_timing as phase_timing

from scenes.workrooms.mechanical_assembly.tasks.gear_train_assembly import assets, definition, rules, task


SCHEMA = 'mechanical_l5_privileged_validation_plan.v1'
NATIVE_INPUT_SCHEMA = 'mechanical_l5_gear_native_probe_input.v1'
RELEASE_NAME = 'large_idler_v3_4_compact_pgs_release'
GEAR_IDS = ('input_gear', 'idler_gear', 'output_gear')
TOOL_IDS = ('socket_crank', 'open_end_wrench', 'straight_hex_screwdriver')
ALL_OBJECT_IDS = GEAR_IDS + TOOL_IDS


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _release_reference():
    candidate = definition.TASK['candidate_referee']['release']
    return Path(candidate['path']).resolve(), candidate['manifest_sha256']


def _check_release(release=None):
    expected_path, expected_hash = _release_reference()
    release = expected_path if release is None else Path(release).resolve()
    layout = definition.TASK['design']['supply_layout_candidate']['release']
    references = {expected_path: expected_hash, Path(layout['path']).resolve(): layout['manifest_sha256']}
    if release not in references:
        raise ValueError('L5 native probe requires a pinned geometry or supply-layout release')
    expected_hash = references[release]
    manifest = release/'release_manifest.json'
    if not manifest.is_file():
        raise ValueError('L5 native input requires a frozen release manifest')
    actual = digest(manifest)
    if actual != expected_hash:
        raise ValueError(f'L5 release manifest drift: {actual} != {expected_hash}')
    return release, actual


def _hand_postures(initial):
    from scenes.workrooms.mechanical_assembly.robot_layout import READY_FINGER_POSTURE_RAD
    from scenes.workrooms.mechanical_assembly.tasks.fastener_assembly.motion_plan import opposing_pinch_candidate

    candidate = opposing_pinch_candidate()
    hand_names = set(initial['joint_positions_rad']) - {
        f'fr3_joint{i}' for i in range(1, 8)}
    result = {
        'ready': dict(READY_FINGER_POSTURE_RAD),
        'open': dict(candidate['open_hand_rad']),
        'closed': dict(candidate['closed_hand_rad']),
    }
    for name, values in result.items():
        if set(values) != hand_names:
            raise ValueError('L5 hand posture has an incomplete 22-joint inventory: '+name)
    return result


def _robot_target(scene, initial, position_m, hand):
    from scenes.workrooms.mechanical_assembly.robot_layout import geometry_model, solve_wrist

    model = geometry_model(scene['robot'])
    solved = solve_wrist(model, position_m)
    if not solved.get('converged'):
        raise ValueError('L5 native endpoint IK did not converge: '+repr(position_m))
    target = dict(initial['joint_positions_rad'])
    target.update(solved['joint_positions_rad'])
    target.update(hand)
    model.checked_joints(target, enforce_limits=True)
    return target, {
        'position_m': list(position_m),
        'position_error_m': float(solved['position_error_m']),
        'rotation_error_rad': float(solved['rotation_error_rad']),
    }


def _robot_phases(scene, initial):
    """Build bounded endpoint targets for gear placement and tool engagement."""
    postures = _hand_postures(initial)
    stations = [
        ('ready', (.10, 0., 1.40), 'ready', 'Robot-only reset endpoint.'),
        ('input_gear_approach', (.21, -.20, 1.04), 'open', 'Approach the input gear supply rest; no attachment.'),
        ('input_gear_pickup_close', (.21, -.20, .98), 'closed', 'Close the hand at the input gear; contact is observed only.'),
        ('input_gear_lift', (.25, -.15, 1.08), 'closed', 'Lift endpoint for a possible gravity/contact transfer.'),
        ('input_gear_shaft_approach', (.40, -.13, 1.00), 'closed', 'Approach the input shaft with the gear held only by native contact.'),
        ('input_gear_lower', (.40, -.13, .94), 'closed', 'Lower toward the input shaft entry; no pose writeback.'),
        ('input_gear_release', (.40, -.13, .98), 'open', 'Release the input gear and observe passive support.'),
        ('input_gear_withdraw', (.10, 0., 1.40), 'open', 'Withdraw after input gear release.'),
        ('idler_gear_approach', (.25, .32, 1.04), 'open', 'Approach the idler gear supply rest; no attachment.'),
        ('idler_gear_pickup_close', (.25, .32, .98), 'closed', 'Close the hand at the idler gear; contact is observed only.'),
        ('idler_gear_lift', (.28, .22, 1.08), 'closed', 'Lift endpoint for a possible gravity/contact transfer.'),
        ('idler_gear_shaft_approach', (.40, 0., 1.00), 'closed', 'Approach the idler shaft with the gear held only by native contact.'),
        ('idler_gear_lower', (.40, 0., .94), 'closed', 'Lower toward the idler shaft entry; no pose writeback.'),
        ('idler_gear_release', (.40, 0., .98), 'open', 'Release the idler gear and observe passive support.'),
        ('idler_gear_withdraw', (.10, 0., 1.40), 'open', 'Withdraw after idler gear release.'),
        ('output_gear_approach', (.21, .10, 1.04), 'open', 'Approach the output gear supply rest; no attachment.'),
        ('output_gear_pickup_close', (.21, .10, .98), 'closed', 'Close the hand at the output gear; contact is observed only.'),
        ('output_gear_lift', (.25, .06, 1.08), 'closed', 'Lift endpoint for a possible gravity/contact transfer.'),
        ('output_gear_shaft_approach', (.40, .13, 1.00), 'closed', 'Approach the output shaft with the gear held only by native contact.'),
        ('output_gear_lower', (.40, .13, .94), 'closed', 'Lower toward the output shaft entry; no pose writeback.'),
        ('output_gear_release', (.40, .13, .98), 'open', 'Release the output gear and observe passive support.'),
        ('output_gear_withdraw', (.10, 0., 1.40), 'open', 'Withdraw after output gear release.'),
        ('tool_approach', (.50, -.28, 1.04), 'open', 'Approach the free socket-crank supply area; no attachment.'),
        ('tool_close', (.50, -.28, .98), 'closed', 'Close the hand at the tool supply area; contact is observed only.'),
        ('tool_lift', (.45, -.22, 1.08), 'closed', 'Lift endpoint for a possible gravity/contact transfer.'),
        ('tool_engage', (.40, -.13, 1.02), 'closed', 'Approach the input hub for native tool contact; no wrench is applied.'),
        ('tool_release', (.40, -.13, 1.08), 'open', 'Release the tool and observe passive stability.'),
        ('withdraw', (.10, 0., 1.40), 'open', 'Withdraw to the shared ready region.'),
    ]
    phases, previous = [], None
    supplies = {row['id']: row['position_m'] for row in scene['objects']}
    for name, position, posture, scope in stations:
        for gear_id in GEAR_IDS:
            if name in (gear_id+'_approach', gear_id+'_pickup_close', gear_id+'_lift'):
                position = (*supplies[gear_id][:2], position[2])
        target, ik = _robot_target(scene, initial, position, postures[posture])
        if previous is not None and target == previous:
            continue
        phases.append({
            'id': name,
            'duration_s': .24,
            'joint_positions_rad': target,
            'endpoint_ik': ik,
            'stage': name,
            'scope': scope+' No task score is assigned.',
        })
        if posture == 'closed':
            object_id = next((row['id'] for row in scene['objects'] if name.startswith(row['id']+'_')), None)
            if name.startswith('tool_'):
                object_id = 'socket_crank'
            if object_id is None:
                raise ValueError('Closed-hand diagnostic phase lacks its intended object: '+name)
            phases[-1]['hand_contact_actor_prefix'] = '/World/Objects/'+object_id
        previous = target
    return phase_timing.time_phases(phases, scene, initial)


def _paths(frozen_task):
    actors = frozen_task['evaluation']['actors']
    return {
        **{name: actors[name] for name in ALL_OBJECT_IDS},
        'gear_workplate': frozen_task['evaluation']['fixture_actor'],
    }


def _source_files():
    return [Path(__file__), Path(phase_timing.__file__), Path(task.__file__), Path(definition.__file__),
            Path(assets.__file__), Path(rules.__file__)]


def _validate_privileged_input_contract(inputs):
    from scripts.validation.validate_assembly_contacts import _validate_robot_phases

    mismatches = {}
    if inputs.get('schema') != NATIVE_INPUT_SCHEMA:
        mismatches['schema'] = inputs.get('schema')
    if inputs.get('task_id') != 'gear_train_assembly':
        mismatches['task_id'] = inputs.get('task_id')
    if inputs.get('release'):
        release = Path(inputs['release'])
        manifest = release/'release_manifest.json'
        actual = digest(manifest) if manifest.is_file() else None
        if actual != inputs.get('release_manifest_sha256'):
            mismatches['release_manifest_sha256'] = {'expected': inputs.get('release_manifest_sha256'), 'actual': actual}
        try:
            _check_release(release)
        except ValueError as error:
            mismatches['release'] = str(error)
    for key, expected in {
        'model_calls': 0, 'ordinary_agent_validation': False,
        'task_success_claimed': False, 'physics_accepted': False,
        'object_state_writeback_after_reset': False, 'external_object_forces': False,
    }.items():
        if inputs.get(key) != expected:
            mismatches[key] = {'expected': expected, 'actual': inputs.get(key)}
    contract = inputs.get('robot_phase_contract')
    if (not isinstance(contract, dict)
            or any(contract.get(key) is not False for key in
                   ('object_attachment', 'object_state_writeback', 'external_object_forces'))):
        mismatches['robot_phase_contract'] = 'L5 robot phase contract must forbid attachment, pose writeback and external object forces.'
    cases = inputs.get('cases')
    if not isinstance(cases, list) or len(cases) != 1:
        mismatches['cases'] = 'L5 requires one nominal gear/tool phase case.'
    initial = inputs.get('initial_state', {}).get('joint_positions_rad')
    for case in cases or []:
        if set(case.get('reset_objects', {})) != set(ALL_OBJECT_IDS):
            mismatches['reset_objects'] = sorted(case.get('reset_objects', {}))
        expected_objects = ((inputs.get('task') or {}).get('evaluation') or {}).get('initial_objects')
        if isinstance(expected_objects, dict) and case.get('reset_objects') != expected_objects:
            mismatches['reset_pose'] = 'L5 reset objects must match the frozen unassembled release byte-for-byte.'
        release_contract = case.get('release_contract') or {}
        required_release = {
            'free_dynamic_objects': True, 'gravity_only_after_reset': True,
            'object_pose_writeback_forbidden': True, 'external_object_forces_forbidden': True,
        }
        if any(release_contract.get(k) is not v for k, v in required_release.items()):
            mismatches['release_contract'] = 'L5 case must forbid reset shortcuts and external object forces.'
        receipt = case.get('phase_receipt_contract') or {}
        if receipt.get('record_raw_contacts') is not True or receipt.get('task_success_credit') != 0:
            mismatches['phase_receipt_contract'] = 'L5 phases must record raw contacts with zero task credit.'
        try:
            phases = _validate_robot_phases(case.get('robot_phases'), initial)
            required = receipt.get('required_phase_ids')
            ids = [phase['id'] for phase in phases]
            if not isinstance(required, list) or not required or len(required) != len(set(required)) or any(name not in ids for name in required):
                mismatches['required_phase_ids'] = required
            if abs(float(case.get('duration_s')) - sum(row['duration_s'] for row in phases)) > 1e-8:
                mismatches['phase_duration'] = True
        except (KeyError, TypeError, ValueError) as exc:
            mismatches['robot_phases'] = str(exc)
    if mismatches:
        raise ValueError('Frozen L5 privileged input contract failed: '+json.dumps(mismatches, sort_keys=True))
    return {'passed': True, 'checked': True, 'release_manifest_sha256': inputs['release_manifest_sha256'],
            'source_files_checked': len(inputs.get('source_sha256') or {}),
            'scope': 'L5 release, source, reset and no-shortcut contract checked before native runtime.'}


def build_plan(*, release=None):
    release, release_hash = _check_release(release)
    from bench.isaacsim.release import task_inputs
    from scenes.workrooms.mechanical_assembly.robot_layout import LAYOUT_ID

    _, scene, frozen_task, initial = task_inputs(release, 'gear_train_assembly')
    if frozen_task.get('task_id') != 'gear_train_assembly' or scene.get('isaacsim', {}).get('solver_type') != 'PGS':
        raise ValueError('L5 native input requires the pinned PGS gear candidate')
    phases = _robot_phases(scene, initial)
    return {
        'schema': SCHEMA,
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'family': 'mechanical_assembly', 'task_id': 'gear_train_assembly',
        'release': str(release), 'release_manifest_sha256': release_hash,
        'release_status': 'prepared_unvalidated', 'layout_id': LAYOUT_ID,
        'model_calls': 0, 'ordinary_agent_validation': False,
        'native_runtime_started': False, 'task_success_claimed': False,
        'physics_accepted': False, 'object_state_writeback': False,
        'external_object_forces': False,
        'reset_objects': deepcopy(frozen_task['evaluation']['initial_objects']),
        'contact_actor_paths': _paths(frozen_task),
        'robot_phase_contract': {
            'phase_count': len(phases),
            'duration_s': sum(float(p['duration_s']) for p in phases),
            'phases': phases, 'object_attachment': False,
            'object_state_writeback': False, 'external_object_forces': False,
            'scope': 'Privileged robot joint targets only; endpoint IK is not grasp or task qualification.',
        },
        'scope': 'CPU contract for a zero-model privileged L5 native run; no native manipulation acceptance.',
    }


def build_native_input(release=None):
    release, release_hash = _check_release(release)
    from bench.isaacsim.release import task_inputs

    _, scene, frozen_task, initial = task_inputs(release, 'gear_train_assembly')
    if frozen_task.get('task_id') != 'gear_train_assembly':
        raise ValueError('L5 native input requires the gear-train task')
    phases = _robot_phases(scene, initial)
    required = [
        'input_gear_approach', 'input_gear_pickup_close', 'input_gear_lower', 'input_gear_release',
        'idler_gear_approach', 'idler_gear_pickup_close', 'idler_gear_lower', 'idler_gear_release',
        'output_gear_approach', 'output_gear_pickup_close', 'output_gear_lower', 'output_gear_release',
        'tool_approach', 'tool_close', 'tool_engage', 'tool_release', 'withdraw',
    ]
    phase_ids = [phase['id'] for phase in phases]
    if any(name not in phase_ids for name in required):
        raise ValueError('L5 native phase list is missing a required stage')
    objects = deepcopy(frozen_task['evaluation']['initial_objects'])
    case = {
        'id': 'gear_supply_robot_phase_probe',
        'duration_s': sum(float(p['duration_s']) for p in phases),
        'reset_variant': 'unassembled_gears_and_tools',
        'reset_objects': objects,
        'robot_phases': phases,
        'phase_receipt_contract': {
            'required_phase_ids': required,
            'contact_actor_paths': _paths(frozen_task),
            'minimum_native_samples_per_phase': 1,
            'record_raw_contacts': True, 'task_success_credit': 0,
        },
        'release_contract': {
            'free_dynamic_objects': True, 'gravity_only_after_reset': True,
            'object_pose_writeback_forbidden': True,
            'external_object_forces_forbidden': True,
        },
        'scope': 'L5 gear/tool reset and robot phase receipt diagnostic; no task score.',
    }
    source_files = _source_files()
    value = {
        'schema': NATIVE_INPUT_SCHEMA, 'family': 'mechanical_assembly',
        'task_id': 'gear_train_assembly', 'release': str(release),
        'release_manifest_sha256': release_hash, 'release_status': 'prepared_unvalidated',
        'scene': scene, 'task': frozen_task, 'initial_state': initial, 'cases': [case],
        'robot_phase_contract': {
            'phase_count': len(phases),
            'duration_s': sum(float(p['duration_s']) for p in phases),
            'phases': phases, 'object_attachment': False,
            'object_state_writeback': False, 'external_object_forces': False,
            'scope': 'Privileged robot joint targets and native integration only.',
        },
        'model_calls': 0, 'ordinary_agent_validation': False,
        'task_success_claimed': False, 'physics_accepted': False,
        'object_state_writeback_after_reset': False, 'external_object_forces': False,
        'source_sha256': {str(path): digest(path) for path in source_files},
        'scope': 'Frozen L5 native reset and phase-observation input; native report cannot promote the release.',
    }
    _validate_privileged_input_contract(value)
    return value


def run_native_input(output, release=None):
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    value = build_native_input(release)
    output.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    return value


def run(output, release=None):
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    value = build_plan(release=release)
    output.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--native-input', type=Path)
    parser.add_argument('--release', type=Path)
    args = parser.parse_args()
    plan = run(args.output, args.release)
    native = run_native_input(args.native_input, args.release) if args.native_input else None
    print(json.dumps({
        'passed': True, 'output': str(args.output.resolve()),
        'model_calls': plan['model_calls'],
        'native_runtime_started': plan['native_runtime_started'],
        'phase_count': plan['robot_phase_contract']['phase_count'],
        'native_input': None if native is None else str(args.native_input.resolve()),
        'native_case_count': None if native is None else len(native['cases']),
        'task_success_claimed': False,
    }, ensure_ascii=False))


if __name__ == '__main__':
    main()
