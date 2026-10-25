"""Build the zero-model privileged L2 peg-insertion validation input.

The input binds the existing peg release and describes robot-only pickup,
alignment, guided lowering and release phases.  It is a native evidence
contract, not an evaluator shortcut: objects remain free dynamic bodies and
the builder never writes their poses or starts a simulator.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path

from scripts.validation import mechanical_robot_phase_timing as phase_timing

from scenes.workrooms.mechanical_assembly.tasks.peg_insertion import definition


SCHEMA = 'mechanical_l2_privileged_validation_plan.v1'
NATIVE_INPUT_SCHEMA = 'mechanical_l2_native_probe_input.v1'
TASK_ID = 'peg_insertion'
RELEASE_NAME = 'peg_insertion'
REQUIRED_NATIVE_OBSERVATIONS = (
    'fixed_room_and_task_release_load',
    'robot_reset_joint_positions',
    'task_object_initial_poses',
    'robot_endpoint_ik_for_each_cradle_and_socket',
    'native_grasp_contact_for_each_peg',
    'native_guided_bore_contact_for_each_peg',
    'native_bottom_seat_support_for_each_peg',
    'native_release_robot_separation',
    'peg_axis_and_bottom_result',
    'terminal_stability_window',
    'complete_native_substep_contact_stream',
    'source_asset_release_gpu_output_provenance',
)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _check_release(release=None):
    expected_path = Path(definition.TASK['source_release']['path']).resolve()
    expected_hash = definition.TASK['source_release']['manifest_sha256']
    release = expected_path if release is None else Path(release).resolve()
    if release.name != RELEASE_NAME:
        raise ValueError('L2 privileged input is pinned to the peg_insertion release')
    manifest = release/'release_manifest.json'
    if not manifest.is_file():
        raise ValueError('L2 native input requires a frozen release manifest')
    actual = digest(manifest)
    if actual != expected_hash:
        raise ValueError(f'L2 release manifest drift: {actual} != {expected_hash}')
    return release, actual


def _hand_postures(initial):
    from scenes.workrooms.mechanical_assembly.robot_layout import READY_FINGER_POSTURE_RAD
    from scenes.workrooms.mechanical_assembly.tasks.fastener_assembly.motion_plan import opposing_pinch_candidate

    candidate = opposing_pinch_candidate()
    hand_names = set(initial['joint_positions_rad']) - {f'fr3_joint{i}' for i in range(1, 8)}
    result = {'ready': dict(READY_FINGER_POSTURE_RAD),
              'open': dict(candidate['open_hand_rad']),
              'closed': dict(candidate['closed_hand_rad'])}
    if any(set(values) != hand_names for values in result.values()):
        raise ValueError('L2 hand posture requires all 22 hand joints')
    return result


def _robot_target(scene, initial, position_m, hand):
    from scenes.workrooms.mechanical_assembly.robot_layout import geometry_model, solve_wrist

    model = geometry_model(scene['robot'])
    solved = solve_wrist(model, position_m)
    if not solved.get('converged'):
        raise ValueError('L2 native endpoint IK did not converge: '+repr(position_m))
    target = dict(initial['joint_positions_rad'])
    target.update(solved['joint_positions_rad'])
    target.update(hand)
    model.checked_joints(target, enforce_limits=True)
    return target, {'position_m': list(position_m),
                    'position_error_m': float(solved['position_error_m']),
                    'rotation_error_rad': float(solved['rotation_error_rad'])}


def _robot_phases(scene, initial):
    postures = _hand_postures(initial)
    stations = [('ready', (.10, 0., 1.40), 'ready', 'Reset endpoint above the task table.')]
    for index, y in enumerate((-.25, .25)):
        name = f'peg_{index}'
        target_y = -.08 if index == 0 else .08
        stations.extend([
            (f'{name}_pickup_approach', (.25, y, 1.04), 'open', 'Approach the open supply cradle.'),
            (f'{name}_pickup_close', (.25, y, .90), 'closed', 'Close hand; native peg contact is observed only.'),
            (f'{name}_lift', (.25, y, 1.04), 'closed', 'Lift endpoint; no attachment or pose writeback.'),
            (f'{name}_socket_approach', (.40, target_y, 1.04), 'closed', 'Approach the matching socket entry.'),
            (f'{name}_guided_lower', (.40, target_y, .90), 'closed', 'Lower through the passive chamfered guide.'),
            (f'{name}_release', (.40, target_y, .86), 'open', 'Open hand and observe bottom-seat support.'),
            (f'{name}_withdraw', (.40, target_y, 1.04), 'open', 'Withdraw after release.'),
        ])
    phases, previous = [], None
    for name, position, posture, scope in stations:
        target, ik = _robot_target(scene, initial, position, postures[posture])
        if previous is not None and target == previous:
            continue
        phases.append({'id': name, 'duration_s': .24,
                       'joint_positions_rad': target, 'endpoint_ik': ik,
                       'stage': name, 'scope': scope+' No task score is assigned.'})
        if posture == 'closed':
            object_id = next((row['id'] for row in scene['objects'] if name.startswith(row['id']+'_')), None)
            if object_id is None:
                raise ValueError('Closed-hand diagnostic phase lacks its intended object: '+name)
            phases[-1]['hand_contact_actor_prefix'] = '/World/Objects/'+object_id
        previous = target
    return phase_timing.time_phases(phases, scene, initial)


def _release_contract():
    return {'free_dynamic_objects': True, 'gravity_only_after_reset': True,
            'object_pose_writeback_forbidden': True, 'external_object_forces_forbidden': True}


def _nominal_case(frozen_task, phases):
    required = [phase['id'] for phase in phases if phase['id'].endswith(('_pickup_close', '_guided_lower', '_release'))]
    return {
        'id': 'nominal_insertion_robot_phase_probe',
        'duration_s': sum(float(phase['duration_s']) for phase in phases),
        'reset_variant': 'original_supply',
        'reset_objects': deepcopy(frozen_task['evaluation']['initial_objects']),
        'robot_phases': phases,
        'phase_receipt_contract': {
            'required_phase_ids': required,
            'contact_actor_paths': frozen_task['evaluation']['actors'],
            'minimum_native_samples_per_phase': 1,
            'record_raw_contacts': True,
            'task_success_credit': 0,
        },
        'release_contract': _release_contract(),
        'expected_functional_success': True,
        'scope': 'Privileged peg reset, robot endpoint, grasp, guide, seat and release receipt; no task score.',
    }


def _outside_socket_case(frozen_task, phases):
    reset = deepcopy(frozen_task['evaluation']['initial_objects'])
    targets = frozen_task['evaluation']['target_bottom_centers_m']
    reset['peg_0']['position_m'] = [targets['peg_0'][0]+.045, targets['peg_0'][1], targets['peg_0'][2]+.003]
    return {
        'id': 'outside_socket_negative_control',
        'duration_s': sum(float(phase['duration_s']) for phase in phases),
        'reset_variant': 'outside_socket_engineering_probe',
        'reset_objects': reset,
        'robot_phases': phases,
        'phase_receipt_contract': {
            'required_phase_ids': ['peg_0_guided_lower', 'peg_0_release'],
            'contact_actor_paths': frozen_task['evaluation']['actors'],
            'minimum_native_samples_per_phase': 1,
            'record_raw_contacts': True,
            'task_success_credit': 0,
        },
        'release_contract': _release_contract(),
        'expected_functional_success': False,
        'engineering_reset_override': True,
        'scope': 'Negative native guide-boundary diagnostic; no task score or automatic reset correction.',
    }


def build_native_input(release=None):
    from bench.isaacsim.release import task_inputs
    release, release_hash = _check_release(release)
    _, scene, frozen_task, initial = task_inputs(release, TASK_ID)
    if frozen_task.get('task_id') != TASK_ID:
        raise ValueError('L2 native input requires the peg_insertion task')
    phases = _robot_phases(scene, initial)
    source_files = [Path(__file__), Path(definition.__file__), Path(phase_timing.__file__)]
    return {
        'schema': NATIVE_INPUT_SCHEMA, 'family': 'mechanical_assembly', 'task_id': TASK_ID,
        'release': str(release), 'release_manifest_sha256': release_hash,
        'release_status': 'prepared_unvalidated', 'scene': scene, 'task': frozen_task,
        'initial_state': initial,
        'cases': [_nominal_case(frozen_task, phases), _outside_socket_case(frozen_task, phases)],
        'required_native_observations': list(REQUIRED_NATIVE_OBSERVATIONS),
        'robot_phase_contract': {
            'phase_count': len(phases), 'duration_s': sum(float(p['duration_s']) for p in phases),
            'phases': phases, 'object_attachment': False,
            'object_state_writeback': False, 'external_object_forces': False,
            'scope': 'Privileged robot joint targets only; native contacts determine grasp, guide and release evidence.',
        },
        'model_calls': 0, 'ordinary_agent_validation': False,
        'task_success_claimed': False, 'physics_accepted': False,
        'object_state_writeback_after_reset': False, 'external_object_forces': False,
        'source_sha256': {str(path): digest(path) for path in source_files},
        'scope': 'Frozen L2 reset, negative control and robot phase input; native reports remain diagnostics.',
    }


def _validate_robot_phases(phases, initial):
    from scripts.validation.validate_assembly_contacts import _validate_robot_phases as validate_native_phases
    if not isinstance(phases, list) or not phases:
        raise ValueError('Robot phases must be nonempty')
    return validate_native_phases(phases, initial)


def _validate_privileged_input_contract(inputs):
    mismatches = {}
    if inputs.get('schema') != NATIVE_INPUT_SCHEMA:
        mismatches['schema'] = inputs.get('schema')
    required = {'task_id': TASK_ID, 'model_calls': 0, 'ordinary_agent_validation': False,
                'task_success_claimed': False, 'physics_accepted': False,
                'object_state_writeback_after_reset': False, 'external_object_forces': False}
    mismatches.update({key: {'expected': value, 'actual': inputs.get(key)}
                       for key, value in required.items() if inputs.get(key) != value})
    release = Path(inputs.get('release', ''))
    manifest = release/'release_manifest.json'
    actual_hash = digest(manifest) if manifest.is_file() else None
    if actual_hash != inputs.get('release_manifest_sha256'):
        mismatches['release_manifest_sha256'] = {'expected': inputs.get('release_manifest_sha256'), 'actual': actual_hash}
    if release.name != RELEASE_NAME:
        mismatches['release_name'] = release.name
    for raw_path, expected in (inputs.get('source_sha256') or {}).items():
        actual = digest(raw_path) if Path(raw_path).is_file() else None
        if actual != expected:
            mismatches['source_sha256:'+str(raw_path)] = {'expected': expected, 'actual': actual}
    contract = inputs.get('robot_phase_contract')
    if (not isinstance(contract, dict) or any(contract.get(key) is not False
            for key in ('object_attachment', 'object_state_writeback', 'external_object_forces'))):
        mismatches['robot_phase_contract'] = 'L2 phase contract must forbid attachment, pose writeback and external object forces.'
    try:
        phases = _validate_robot_phases(contract['phases'], inputs['initial_state']['joint_positions_rad'])
        if contract.get('phase_count') != len(phases):
            mismatches['phase_count'] = {'expected': len(phases), 'actual': contract.get('phase_count')}
        if abs(float(contract.get('duration_s'))-sum(float(p['duration_s']) for p in phases)) > 1e-8:
            mismatches['phase_duration'] = True
    except (KeyError, TypeError, ValueError) as exc:
        mismatches['robot_phases'] = str(exc)
    cases = inputs.get('cases')
    if not isinstance(cases, list) or not cases or len({c.get('id') for c in cases if isinstance(c, dict)}) != len(cases):
        mismatches['cases'] = 'L2 requires unique native cases'
    expected_objects = ((inputs.get('task') or {}).get('evaluation') or {}).get('initial_objects')
    for case in cases or []:
        if case.get('engineering_reset_override') is not True and case.get('reset_objects') != expected_objects:
            mismatches['reset:'+str(case.get('id'))] = 'L2 nominal reset must match frozen task object poses byte-for-byte.'
        receipt = case.get('phase_receipt_contract')
        if (not isinstance(receipt, dict) or receipt.get('record_raw_contacts') is not True
                or receipt.get('task_success_credit') != 0):
            mismatches['receipt:'+str(case.get('id'))] = 'L2 case must record raw contacts with zero task credit.'
        try:
            checked = _validate_robot_phases(case.get('robot_phases'), inputs['initial_state']['joint_positions_rad'])
            ids = {p['id'] for p in checked}
            required_ids = receipt.get('required_phase_ids', [])
            if not required_ids or len(required_ids) != len(set(required_ids)) or not set(required_ids) <= ids:
                mismatches['required_phases:'+str(case.get('id'))] = True
            if abs(float(case.get('duration_s'))-sum(float(p['duration_s']) for p in checked)) > 1e-8:
                mismatches['case_duration:'+str(case.get('id'))] = True
        except (KeyError, TypeError, ValueError) as exc:
            mismatches['case_phases:'+str(case.get('id'))] = str(exc)
        release_contract = case.get('release_contract')
        if (not isinstance(release_contract, dict)
                or any(release_contract.get(key) is not True for key in _release_contract())):
            mismatches['release_contract:'+str(case.get('id'))] = True
    observations = inputs.get('required_native_observations')
    if tuple(observations or ()) != REQUIRED_NATIVE_OBSERVATIONS:
        mismatches['required_native_observations'] = observations
    if mismatches:
        raise ValueError('Frozen L2 privileged input contract failed: '+json.dumps(mismatches, sort_keys=True))
    from scripts.validation.validate_assembly_contacts import _validate_privileged_input_contract as validate_native_contract
    validate_native_contract(inputs)
    return {'passed': True, 'checked': True, 'release_manifest_sha256': actual_hash,
            'scope': 'L2 release, reset, negative control, robot phase and no-shortcut contract checked before native runtime.'}


def build_plan(*, release=None):
    value = build_native_input(release)
    return {
        'schema': SCHEMA, 'family': value['family'], 'task_id': TASK_ID,
        'release': value['release'], 'release_manifest_sha256': value['release_manifest_sha256'],
        'release_status': value['release_status'], 'model_calls': 0,
        'ordinary_agent_validation': False, 'native_runtime_started': False,
        'task_success_claimed': False, 'physics_accepted': False,
        'object_state_writeback': False, 'external_object_forces': False,
        'robot_phase_contract': value['robot_phase_contract'],
        'required_native_observations': value['required_native_observations'],
        'cases': [{'id': case['id'], 'reset_variant': case['reset_variant'],
                   'expected_functional_success': case['expected_functional_success']}
                  for case in value['cases']],
        'contract_check': _validate_privileged_input_contract(value),
        'scope': 'CPU-only L2 privileged native plan; no native runtime or manipulation acceptance.',
    }


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
    native = None if args.native_input is None else run_native_input(args.native_input, args.release)
    print(json.dumps({'passed': plan['contract_check']['passed'], 'model_calls': 0,
                      'native_runtime_started': False,
                      'phase_count': plan['robot_phase_contract']['phase_count'],
                      'native_case_count': None if native is None else len(native['cases']),
                      'native_input': None if native is None else str(args.native_input.resolve())},
                     ensure_ascii=False))


if __name__ == '__main__':
    main()
