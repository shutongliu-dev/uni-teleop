"""Privileged fastener script through real public bench functions and native IPC.

No model calls, object wrenches, attachments, or state playback. The task release
owns reset and scoring; the script owns its strategy and diagnostic gates.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
import math
from pathlib import Path
import time
import traceback

import numpy as np

from bench.control.factory import configured_policy, robot_configuration
from bench.geometry import quaternion_matrix
from bench.isaacsim.release import digest, dump, task_inputs
from bench.paths import load_json
from bench.protocol import PolicyRegistry
from bench.runtime.native import NativeWorker
from bench.runtime.functions import EpisodeFunctions  # Loaded before the frozen-import check.
from bench.runtime.service import CreateEpisodeRequest, EpisodeSession
from scripts.maintenance.freeze_project_sources import verify_if_snapshot


class ObservedNativeWorker(NativeWorker):
    worker_module = 'scripts.validation.mechanical_function_worker'


def episode_request(task_id, policy, plan_sha256):
    """Allow the slow approach/return paths within an explicit action budget."""
    return CreateEpisodeRequest(task_id=task_id, agent_id='privileged_fastener_script',
        agent_version='public_functions_v1', agent_config_sha256=plan_sha256,
        seed=0, track='free_tool_agent', observation_profile='vision_proprio',
        budgets={'sim_time_s': 90., 'wall_time_s': None, 'max_actions': 300,
                 'max_action_sim_s': 10.},
        policy_id=policy.descriptor.policy_id, policy_version=policy.descriptor.version)


def recoverable_close_failure(row, session):
    status = row['status']
    diagnostic = status.get('diagnostic') or {}
    return (session.result is None and status['status'] == 'failed'
            and diagnostic.get('code') in ('tracking_stalled', 'convergence_timeout'))


def acceptance_result(error, result, checks, *, turn_degrees, stop_after_grasp=False):
    """Keep short grasp diagnostics distinct from complete task qualification."""
    common = bool(error is None and result and result['physics_valid']
                  and checks.get('observer_seal', {}).get('passed')
                  and checks.get('final_source_verification', {}).get('passed')
                  and checks.get('opposed_grasp', {}).get('passed'))
    if stop_after_grasp:
        if turn_degrees:
            raise ValueError('A grasp-only diagnostic must explicitly disable turning')
        return {'acceptance_scope': 'grasp_diagnostic_only',
                'diagnostic_grip_passed': common, 'task_qualified': False, 'passed': common}
    complete_task = bool(common and turn_degrees and result['outcome'] == 'success')
    release_ok = all(checks.get(name+'_endpoint', {}).get('passed')
                     for name in ('open', 'retreat', 'final_hold'))
    return {'acceptance_scope': 'complete_task' if turn_degrees else 'grasp_and_release_diagnostic',
            'diagnostic_grip_passed': common, 'task_qualified': complete_task,
            'passed': complete_task if turn_degrees else bool(common and release_ok)}


def grip_execution_scope(mode, plan, policy):
    close = next(phase for phase in plan['phases'] if phase['name'] == 'close')
    direct = mode == 'direct'
    return {
        'close_function': 'move_joints' if direct else 'set_hand_targets',
        'plan_hand_speed_cap_rad_s': close.get('hand_speed_cap_rad_s', 1.),
        'plan_hand_speed_cap_applied_to_close': direct,
        'close_timing': ('minimum-jerk command retimed from acknowledged drive reference using '
                         'the actual 80% ramp and public-limit ramp extension'
                         if direct else 'bench feedback controller computes its own nominal duration'),
        'feedback_hand_velocity_limit_rad_s': policy.limits.hand_velocity_rad_s,
        'feedback_hand_acceleration_limit_rad_s2': policy.limits.hand_acceleration_rad_s2,
        'feedback_close_timeout_sim_s': None if direct else max(3., close['duration_s']),
        'timeout_controls_motion_speed': False,
        'reference_speed_is_measured_physical_speed': False,
        'recoverable_close_observation': 'hold existing drive reference, then use unchanged native grip gate',
    }


class Script:
    def __init__(self, session, output):
        self.session, self.output = session, Path(output)
        self.phases = []
        self.counter = 0

    def invoke(self, function, arguments):
        self.counter += 1
        reply = self.session.functions.invoke({'function': function,
            'request_id': f'call_{self.counter:04d}', 'arguments': arguments}).model_dump(mode='json')
        if not reply['ok']:
            raise RuntimeError(f'{function}: {reply["error"]}')
        return reply['result']

    def progress(self):
        path = self.session.directory / 'private/fastener_physics/progress.json'
        return load_json(path) if path.exists() else None

    def execute(self, name, function, arguments, *, allow_contact_failure=False, endpoint_targets=None):
        session = self.session
        if session.result is not None:
            raise RuntimeError('Episode ended before '+name)
        action_id = f'{len(self.phases):03d}_{name}'
        args = {**arguments, 'action_id': action_id, 'observation_sequence': session.sequence}
        targets = ({**args['arm_joint_positions_rad'], **args['hand_joint_positions_rad']}
                   if function == 'move_joints' else
                   dict(session.snapshot.proprioception.commanded_joint_positions_rad or {}))
        if endpoint_targets is not None:
            targets = endpoint_targets
        dump(session.directory / 'private/engineering_phase.json', {'phase': name, 'joint_targets_rad': targets})
        row = {'name': name, 'function': function, 'arguments': args,
               'start_sim_time_s': session.raw['sim_time_s']}
        self.phases.append(row)
        dump(self.output / 'phases.json', self.phases)
        print(json.dumps({'event': 'function_start', 'phase': name, 'function': function,
                          'simulation_time_s': row['start_sim_time_s']}), flush=True)
        row['submission'] = self.invoke(function, args)
        last_log = time.monotonic()
        while session.thread is not None and session.thread.is_alive():
            session.thread.join(.25)
            if time.monotonic() - last_log >= 20:
                progress = self.progress()
                print(json.dumps({'event': 'native_progress', 'phase': name,
                    'simulation_time_s': (progress or {}).get('simulation_time_s'),
                    'samples': (progress or {}).get('native_samples')}), flush=True)
                last_log = time.monotonic()
        row['status'] = self.invoke('get_action', {'action_id': action_id})
        row['end_sim_time_s'] = session.raw['sim_time_s']
        row['private_contact_evidence'] = self.progress()
        dump(self.output / 'phases.json', self.phases)
        print(json.dumps({'event': 'function_end', 'phase': name,
                          'status': row['status']['status'],
                          'simulation_time_s': row['end_sim_time_s']}), flush=True)
        if row['status']['status'] != 'completed':
            if allow_contact_failure and recoverable_close_failure(row, session):
                row['recoverable_contact_failure'] = True
                dump(self.output / 'phases.json', self.phases)
                return row
            raise RuntimeError(f'{name} action failed: {row["status"]}')
        return row

    def joints(self, phase, *, targets=None):
        from scenes.workrooms.mechanical_assembly.tasks.fastener_assembly.public_motion_plan import public_move_arguments
        session = self.session
        q = phase['joint_positions_rad'] if targets is None else targets
        start = session.snapshot.proprioception.commanded_joint_positions_rad
        if start is None:
            raise RuntimeError('Public snapshot has no acknowledged drive reference')
        arguments = public_move_arguments({**phase, 'joint_positions_rad': q}, session.capabilities,
            start, observation_sequence=session.sequence,
            action_id=f'{len(self.phases):03d}_{phase["name"]}')
        return self.execute(phase['name'], 'move_joints', arguments)

    def hold(self, name, duration, *, endpoint_targets=None):
        return self.execute(name, 'hold', {'duration_s': duration}, endpoint_targets=endpoint_targets)

    def close_hand(self, phase):
        q = phase['joint_positions_rad']
        return self.execute(phase['name'], 'set_hand_targets', {'hand': 'right',
            'hand_posture_rad': {n: q[n] for n in self.session.capabilities.hand_joints},
            'timeout_sim_s': max(3., phase['duration_s'])}, allow_contact_failure=True)

    def ensure_task_settled(self, checks):
        """Do not confuse the support diagnostic with the task's stable baseline."""
        row = {'passed': False, 'extra_wait_s': 0.,
               'scope': 'Canonical task evaluator baseline before any robot approach or closure'}
        checks['task_settlement_before_grasp'] = row
        for attempt in range(5):
            if self.session.score.get('settled_baseline_recorded') is True:
                row.update(passed=True, score=deepcopy(self.session.score))
                dump(self.output / 'checks.json', checks)
                return
            if self.session.result is not None or attempt == 4:
                break
            self.hold(f'await_task_settle_{attempt+1}', .5)
            row['extra_wait_s'] += .5
        row['score'] = deepcopy(self.session.score)
        dump(self.output / 'checks.json', checks)
        raise RuntimeError('Task has no passive settled baseline; robot approach was not attempted')

    def observe_grasp(self, phase, checks, *, close_row=None):
        """Observe physical grasp even when contact blocked the requested posture.

        The failed close remains failed. Only the two existing recoverable
        diagnostics allow this hold; the native grip gate is never replaced by
        posture tracking success or relaxed after a tracking failure.
        """
        recovery = None
        if close_row is not None and close_row['status']['status'] != 'completed':
            if not recoverable_close_failure(close_row, self.session):
                raise RuntimeError(f'Unsafe close recovery: {close_row["status"]}')
            recovery = {'close_action_id': close_row['arguments']['action_id'],
                        'close_status': deepcopy(close_row['status']),
                        'hold_action_id': f'{len(self.phases):03d}_{phase["name"]}',
                        'state': 'hold_pending', 'turn_allowed_by_grip_gate': False}
            checks['close_recovery'] = recovery
            dump(self.output / 'checks.json', checks)
        try:
            hold_row = self.hold(phase['name'], max(1., phase['duration_s']))
        except Exception as exc:
            if recovery is not None:
                held = next((row for row in self.phases
                             if row['arguments']['action_id'] == recovery['hold_action_id']), {})
                recovery.update(state='hold_failed', error=str(exc),
                    hold_status=deepcopy(held.get('status')))
                dump(self.output / 'checks.json', checks)
            raise
        progress = self.progress()
        checks['opposed_grasp'] = progress['grip_last_300ms']
        if recovery is not None:
            recovery.update(state='hold_observed',
                hold_action_id=hold_row['arguments']['action_id'],
                hold_status=deepcopy(hold_row['status']),
                opposed_grasp=deepcopy(checks['opposed_grasp']),
                turn_allowed_by_grip_gate=(self.session.result is None
                                           and checks['opposed_grasp']['passed'] is True))
        dump(self.output / 'checks.json', checks)
        return progress


def turn_path(model, joints, tool_pose, degrees):
    """Low-speed robot-only IK around the measured independent tool axis.

    This is an engineering strategy; it does not constrain or move the tool.
    The frozen task evaluator independently measures bolt rotation and travel.
    """
    from scenes.workrooms.mechanical_assembly.tasks.fastener_assembly.motion_plan import _solve
    if not 0 < degrees <= 45:
        raise ValueError('The engineering short-turn candidate is bounded to 45 degrees')
    wrist = model.public_geometry(joints)['wrist']
    point = np.asarray(wrist['position_m'])
    rotation = quaternion_matrix(wrist['quaternion_wxyz'])
    tip = np.asarray(tool_pose[:3])
    axis = quaternion_matrix(np.asarray(tool_pose)[[6, 3, 4, 5]])[:, 2]
    skew = np.array([[0., -axis[2], axis[1]], [axis[2], 0., -axis[0]], [-axis[1], axis[0], 0.]])
    phases, seed = [], {n: joints[n] for n in model.arm.names}
    # Small segments keep the curved contact path close to its joint trajectory.
    count = math.ceil(degrees / 3.)
    for index in range(1, count + 1):
        theta = -math.radians(degrees) * index / count
        rotate = np.eye(3) + math.sin(theta) * skew + (1.-math.cos(theta)) * (skew @ skew)
        target = tip + rotate @ (point-tip) + axis * (.003 * theta / (2*math.pi))
        arm, evidence = _solve(model, target, rotate @ rotation, seed)
        phases.append({'name': f'turn_{index:02d}', 'joint_positions_rad': {**joints, **arm},
                       'duration_s': .5, 'ik': evidence, 'nominal_angle_rad': theta})
        seed = arm
    return phases, {'position_m': target.tolist(), 'rotation': (rotate @ rotation).tolist()}


def run(args):
    if args.stop_after_grasp and args.turn_degrees:
        raise ValueError('--stop-after-grasp requires --turn-degrees 0')
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    session = None
    error = None
    checks = {}
    try:
        imports = verify_if_snapshot(Path(__file__).resolve().parents[2], required_modules=(
            'scripts.validation.validate_mechanical_functions', 'bench.runtime.service',
            'bench.runtime.functions', 'bench.runtime.native'), entrypoint_file=Path(__file__))
        dump(output / 'runtime_imports.json', imports)
        if digest(args.motion_plan) != args.motion_plan_sha256:
            raise ValueError('Motion plan changed')
        plan = load_json(args.motion_plan)
        dump(output / 'motion_plan.json', plan)
        entry, scene, task, initial = task_inputs(args.release, args.task_id)
        policy = configured_policy(robot_configuration(scene))
        registry = PolicyRegistry()
        registry.register(policy)
        request = episode_request(entry['id'], policy, digest(args.motion_plan))
        session = EpisodeSession(request, output / 'episode', release=args.release,
            device=args.device, registry=registry, worker_factory=ObservedNativeWorker)
        script = Script(session, output)
        dump(output / 'public_function_catalog.json', session.functions.catalog().model_dump(mode='json'))
        dump(output / 'robot_guide.json', script.invoke('get_robot', {}))
        dump(output / 'hand_presets.json', script.invoke('get_hand_presets', {}))
        dump(output / 'scope.json', {'privileged_script': True, 'model_calls': 0,
            'interface': 'bench_functions_v1 Python dispatcher with canonical session and native IPC',
            'http_transport_tested': False, 'grip_execution': args.grip_execution,
            'stop_after_grasp': args.stop_after_grasp,
            'task_qualification_attempted': bool(args.turn_degrees and not args.stop_after_grasp),
            'grip_execution_scope': grip_execution_scope(args.grip_execution, plan, policy),
            'task_release_sha256': digest(args.release / 'release_manifest.json'),
            'private_information_used': ['frozen reset/tool geometry', 'native grip gate and tool axis'],
            'object_state_writes_after_reset': 0, 'object_external_wrenches': 0})
        script.hold('settle', .5)
        checks['preinserted_reset_stability'] = script.progress()['preinserted_reset_stability']
        if not checks['preinserted_reset_stability']['passed']:
            raise RuntimeError('Fresh task reset failed passive native stability')
        if args.turn_degrees:
            script.ensure_task_settled(checks)
        turning = False
        turned = None
        final_arm = None
        close_row = None
        for phase in plan['phases']:
            name = phase['name']
            if name == 'close' and args.grip_execution == 'feedback':
                close_row = script.close_hand(phase)
            elif name == 'hold_grasp':
                progress = script.observe_grasp(phase, checks, close_row=close_row)
                if args.stop_after_grasp:
                    break
                if session.result is None and checks['opposed_grasp']['passed'] is True and args.turn_degrees:
                    q = dict(session.snapshot.proprioception.commanded_joint_positions_rad)
                    phases, turned = turn_path(policy.geometry, q,
                        progress['object_poses_xyzw']['screwdriver'], args.turn_degrees)
                    dump(output / 'turn_plan.json', phases)
                    for turn in phases:
                        script.joints(turn)
                        turning = True
                        final_arm = {n: turn['joint_positions_rad'][n] for n in session.capabilities.arm_joints}
                        if session.result is not None:
                            break
                    if session.result is None:
                        script.hold('hold_after_turn', .4)
                # A failed grasp skips turning but still executes ordinary release.
            elif name == 'final_hold':
                script.hold(name, max(.6, phase['duration_s']), endpoint_targets=phase['joint_positions_rad'])
            else:
                targets = phase['joint_positions_rad']
                if turning and name in ('release_preclose', 'open_thumb_fingers', 'open'):
                    targets = {**targets, **final_arm}
                elif turning and name in ('retreat', 'lift_clear'):
                    from scenes.workrooms.mechanical_assembly.tasks.fastener_assembly.motion_plan import _solve
                    rotation = np.asarray(turned['rotation'])
                    point = np.asarray(turned['position_m']) - .08 * rotation[:, 0]
                    if name == 'lift_clear':
                        point += [0., 0., .22]
                    final_arm, _ = _solve(policy.geometry, point, rotation, final_arm)
                    targets = {**targets, **final_arm}
                script.joints(phase, targets=targets)
            if name in ('open', 'retreat', 'final_hold'):
                checks[name+'_endpoint'] = (script.progress() or {}).get('release_endpoint', {'passed': False})
            if session.result is not None:
                break
        checks['turn_executed'] = turning
        checks['final_task_score'] = session.score
        if session.result is None:
            script.invoke('finish_episode', {})
    except BaseException as exc:
        error = {'type': type(exc).__name__, 'detail': str(exc), 'traceback': traceback.format_exc()}
        traceback.print_exc()
    finally:
        if session is not None and session.result is None:
            try:
                session.finish()
            except BaseException:
                traceback.print_exc()
                if session.worker is not None:
                    session.worker.close(force=True)
        result = session.result.model_dump(mode='json') if session is not None and session.result else None
        seal_path = output / 'episode/private/fastener_physics/seal.json'
        seal = load_json(seal_path) if seal_path.exists() else {}
        checks['observer_seal'] = {'passed': seal.get('passed') is True,
            'path': str(seal_path), 'sha256': digest(seal_path) if seal_path.exists() else None}
        try:
            dump(output / 'runtime_imports_final.json', verify_if_snapshot(
                Path(__file__).resolve().parents[2], required_modules=(
                    'scripts.validation.validate_mechanical_functions', 'bench.runtime.service',
                    'bench.runtime.functions'), entrypoint_file=Path(__file__)))
            checks['final_source_verification'] = {'passed': True}
        except BaseException as exc:
            checks['final_source_verification'] = {'passed': False, 'detail': str(exc)}
        report = {'schema': 'mechanical_public_functions_diagnostic.v1', 'error': error,
                  'checks': checks, 'episode_result': result, 'model_calls': 0,
                  'task_success': bool(result and result['outcome'] == 'success'),
                  **acceptance_result(error, result, checks, turn_degrees=args.turn_degrees,
                                      stop_after_grasp=args.stop_after_grasp)}
        dump(output / 'report.json', report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--release', required=True, type=Path)
    parser.add_argument('--task-id', default='fastener_assembly')
    parser.add_argument('--motion-plan', required=True, type=Path)
    parser.add_argument('--motion-plan-sha256', required=True)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--grip-execution', choices=('direct', 'feedback'), default='feedback')
    parser.add_argument('--turn-degrees', type=float, default=30.)
    parser.add_argument('--stop-after-grasp', action='store_true',
                        help='End after native grip hold; diagnostic only, requires --turn-degrees 0')
    args = parser.parse_args()
    if args.stop_after_grasp and args.turn_degrees:
        parser.error('--stop-after-grasp requires --turn-degrees 0')
    raise SystemExit(0 if run(args)['passed'] else 1)


if __name__ == '__main__':
    main()
