"""Build the zero-model privileged L4 loose-bolt native probe.

The probe binds the prepared loose-bolt v10 PGS release and describes a bounded
robot-only endpoint sequence.  Reset places the bolt and screwdriver at their
task-owned rests; all subsequent changes are native robot joint commands.  The
companion native contact runner records every substep and a receipt for each
phase, but never writes an object pose, applies an object force, or awards task
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

from scenes.workrooms.mechanical_assembly.tasks.fastener_assembly import definition, task


SCHEMA = 'mechanical_l4_privileged_validation_plan.v1'
NATIVE_INPUT_SCHEMA = 'mechanical_l4_loose_native_probe_input.v1'
RELEASE_NAME = 'loose_bolt_candidate_v10_pgs'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _release_reference():
    candidate = definition.TASK['design']['loose_bolt_candidate']
    release = Path(candidate['release']['path']).resolve()
    expected = candidate['release']['manifest_sha256']
    return release, expected


def _check_release(release=None):
    expected_path, expected_hash = _release_reference()
    release = expected_path if release is None else Path(release).resolve()
    if release.name != RELEASE_NAME:
        raise ValueError('L4 native probe is pinned to '+RELEASE_NAME)
    manifest = release/'release_manifest.json'
    if not manifest.is_file():
        raise ValueError('L4 native input requires a frozen release manifest')
    actual = digest(manifest)
    if actual != expected_hash:
        raise ValueError(f'L4 release manifest drift: {actual} != {expected_hash}')
    return release, actual


def _hand_postures(initial):
    """Return reset/open/closed maps without changing any source asset."""
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
            raise ValueError('L4 hand posture has an incomplete 22-joint inventory: '+name)
    return result


def _robot_target(scene, initial, position_m, hand):
    """Solve one endpoint target; no object transform is authored."""
    from scenes.workrooms.mechanical_assembly.robot_layout import geometry_model, solve_wrist

    model = geometry_model(scene['robot'])
    solved = solve_wrist(model, position_m)
    if not solved.get('converged'):
        raise ValueError('L4 native endpoint IK did not converge: '+repr(position_m))
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
    """Build endpoint phases for pickup, guide, thread start, insertion, release."""
    postures = _hand_postures(initial)
    stations = [
        ('ready', (.10, 0., 1.40), 'ready', 'Robot-only reset endpoint.'),
        ('pickup_approach', (.56, -.16, 1.04), 'open', 'Approach loose bolt rest; no attachment.'),
        ('pickup_close', (.56, -.16, 1.00), 'closed', 'Close hand at the bolt rest; contact is observed only.'),
        ('pickup_lift', (.50, -.10, 1.10), 'closed', 'Lift endpoint for a possible gravity/contact transfer.'),
        ('guide_approach', (.38, 0., .98), 'closed', 'Approach the task-owned entry guide.'),
        ('start_thread', (.38, 0., .94), 'closed', 'Lower endpoint for real thread lead-in; no pose writeback.'),
        ('release_bolt', (.38, 0., .98), 'open', 'Open hand and observe passive bolt support/recovery.'),
        ('tool_approach', (.29, .36, 1.04), 'open', 'Approach independent screwdriver rest.'),
        ('tool_close', (.29, .36, .98), 'closed', 'Close hand at screwdriver rest; contact is observed only.'),
        ('tool_insertion', (.38, 0., .98), 'closed', 'Endpoint at fixture mouth for tool insertion.'),
        ('release', (.38, 0., 1.04), 'open', 'Open tool hand and observe release stability.'),
        ('withdraw', (.10, 0., 1.40), 'open', 'Withdraw to the shared ready region.'),
    ]
    phases, previous = [], None
    for name, position, posture, scope in stations:
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
            object_id = 'screwdriver' if name.startswith('tool_') else 'bolt'
            phases[-1]['hand_contact_actor_prefix'] = '/World/Objects/'+object_id
        previous = target
    return phase_timing.time_phases(phases, scene, initial)


def _paths(task_value):
    evaluation = task_value['evaluation']
    actors = evaluation['actors']
    return {
        'bolt': actors['bolt'],
        'screwdriver': actors['screwdriver'],
        'threaded_fixture': actors['threaded_fixture'],
        'entry_guide': evaluation['entry_guide_actor'],
        'bolt_supply_rest': evaluation['supply_rest_actor'],
        'tool_supply_rest': evaluation['tool_rest_actor'],
    }


def build_plan(*, release=None):
    """Return a CPU-only contract; this function never starts Isaac Sim."""
    release, release_hash = _check_release(release)
    from bench.isaacsim.release import task_inputs
    from scenes.workrooms.mechanical_assembly.robot_layout import LAYOUT_ID

    _, scene, frozen_task, initial = task_inputs(release, 'fastener_assembly')
    if frozen_task.get('reset_conditions', {}).get('reset_variant') != 'loose_bolt':
        raise ValueError('L4 candidate release is not a loose-bolt reset')
    phases = _robot_phases(scene, initial)
    paths = _paths(frozen_task)
    contract = {
        'phase_count': len(phases),
        'duration_s': sum(float(p['duration_s']) for p in phases),
        'phases': phases,
        'object_attachment': False,
        'object_state_writeback': False,
        'external_object_forces': False,
        'required_stage_ids': [
            'pickup_approach', 'pickup_close', 'pickup_lift',
            'guide_approach', 'start_thread', 'tool_approach',
            'tool_insertion', 'release',
        ],
        'scope': 'Privileged robot joint targets only; endpoint IK is not grasp or task qualification.',
    }
    return {
        'schema': SCHEMA,
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'family': 'mechanical_assembly',
        'task_id': 'fastener_assembly',
        'reset_variant': 'loose_bolt',
        'release': str(release),
        'release_manifest_sha256': release_hash,
        'release_status': 'prepared_unvalidated',
        'layout_id': LAYOUT_ID,
        'model_calls': 0,
        'ordinary_agent_validation': False,
        'native_runtime_started': False,
        'task_success_claimed': False,
        'physics_accepted': False,
        'object_state_writeback': False,
        'external_object_forces': False,
        'reset_objects': deepcopy(frozen_task['evaluation']['initial_objects']),
        'contact_actor_paths': paths,
        'privileged_contract': contract,
        'source_sha256': {
            str(path): digest(path) for path in (
                Path(__file__), Path(phase_timing.__file__), Path(task.__file__), Path(definition.__file__),
            )
        },
        'scope': 'CPU contract for a zero-model privileged loose-bolt native run; no native manipulation acceptance.',
    }


def build_native_input(release=None):
    """Freeze one loose reset plus robot phase receipts for the native runner."""
    release, release_hash = _check_release(release)
    from bench.isaacsim.release import task_inputs

    _, scene, frozen_task, initial = task_inputs(release, 'fastener_assembly')
    if frozen_task.get('task_id') != 'fastener_assembly':
        raise ValueError('L4 native input requires the fastener task')
    if frozen_task.get('reset_conditions', {}).get('reset_variant') != 'loose_bolt':
        raise ValueError('L4 native input requires the loose-bolt reset variant')
    phases = _robot_phases(scene, initial)
    required = [
        'pickup_approach', 'pickup_close', 'pickup_lift',
        'guide_approach', 'start_thread', 'tool_approach',
        'tool_insertion', 'release',
    ]
    phase_ids = [phase['id'] for phase in phases]
    if any(name not in phase_ids for name in required):
        raise ValueError('L4 native phase list is missing a required stage')
    case = {
        'id': 'loose_supply_robot_phase_probe',
        'duration_s': sum(float(p['duration_s']) for p in phases),
        'reset_variant': 'loose_bolt',
        'reset_objects': deepcopy(frozen_task['evaluation']['initial_objects']),
        'robot_phases': phases,
        'phase_receipt_contract': {
            'required_phase_ids': required,
            'contact_actor_paths': _paths(frozen_task),
            'minimum_native_samples_per_phase': 1,
            'record_raw_contacts': True,
            'task_success_credit': 0,
        },
        'release_contract': {
            'free_dynamic_objects': True,
            'gravity_only_after_reset': True,
            'object_pose_writeback_forbidden': True,
            'external_object_forces_forbidden': True,
        },
        'scope': 'Loose-bolt reset/contact/phase receipt diagnostic; no task score.',
    }
    source_files = [Path(__file__), Path(phase_timing.__file__), Path(task.__file__), Path(definition.__file__)]
    return {
        'schema': NATIVE_INPUT_SCHEMA,
        'family': 'mechanical_assembly',
        'task_id': 'fastener_assembly',
        'reset_variant': 'loose_bolt',
        'release': str(release),
        'release_manifest_sha256': release_hash,
        'scene': scene,
        'task': frozen_task,
        'initial_state': initial,
        'cases': [case],
        'robot_phase_contract': {
            'phase_count': len(phases),
            'duration_s': sum(float(p['duration_s']) for p in phases),
            'phases': phases,
            'object_attachment': False,
            'object_state_writeback': False,
            'external_object_forces': False,
            'scope': 'Privileged robot joint targets and native integration only.',
        },
        'model_calls': 0,
        'ordinary_agent_validation': False,
        'task_success_claimed': False,
        'physics_accepted': False,
        'object_state_writeback_after_reset': False,
        'external_object_forces': False,
        'source_sha256': {str(path): digest(path) for path in source_files},
        'scope': 'Frozen loose-bolt native reset and phase-observation input; the native report cannot promote the release.',
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
    native = run_native_input(args.native_input, args.release) if args.native_input else None
    print(json.dumps({
        'passed': True,
        'output': str(args.output.resolve()),
        'model_calls': plan['model_calls'],
        'native_runtime_started': plan['native_runtime_started'],
        'phase_count': plan['privileged_contract']['phase_count'],
        'native_input': None if native is None else str(args.native_input.resolve()),
        'native_case_count': None if native is None else len(native['cases']),
        'task_success_claimed': False,
    }, ensure_ascii=False))


if __name__ == '__main__':
    main()
