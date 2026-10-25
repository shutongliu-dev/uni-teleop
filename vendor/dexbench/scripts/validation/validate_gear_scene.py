"""Validate the current L5 supply and progressively assembled passive gear train.

Only the independent socket receives a recorded bounded engineering torque.
All object placement is reset-only. These probes do not execute an Agent or
award assembly, tool-selection, grasp, or robot task credit.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
import math
import os
from pathlib import Path
import traceback

import numpy as np

from bench.isaacsim.release import digest, dump
from bench.paths import load_json
from scripts.maintenance.freeze_project_sources import verify_if_snapshot
from scripts.validation.gear_transmission_checks import compact_native_row, axial_rotation
from scripts.validation.mechanical_checks import contact_filter_coverage
from scripts.validation.mechanical_convex_geometry import check_convex_geometry
from scripts.validation.mechanical_tool_checks import contact_exchange_check
from scripts.validation.validate_assembly_contacts import Observer
from scripts.validation.validate_gear_transmission import mixed_native_parameters
from scripts.validation.validate_scenario import seal_report_before_shutdown
from scripts.validation.validate_workrooms import capture, plain, stage_inventory


CASE_IDS = ('supply', 'input_only', 'input_idler', 'full_train')


class StageObserver(Observer):
    """Write original point evidence, retaining exact signed reductions in RAM."""

    def __init__(self, env, output, contract, *, loaded_case):
        self.contract = contract
        self.loaded_case = loaded_case
        self.phase_definition = {'id': 'supply', 'loaded': False, 'target_speed_rad_s': 0.}
        self.wrench_stream = (output/'applied_wrenches.jsonl').open('x')
        self.applied_wrench_calls = 0
        super().__init__(env, output)

    def step(self, *args, **kwargs):
        from bench.simulation.isaacsim_backend.env import _np
        env, contract, phase = self.env, self.contract, self.phase_definition
        self.phase = phase['id']
        force, torque = [0., 0., 0.], [0., 0., 0.]
        velocity = _np(env._object_views['socket_crank'].get_velocities()).reshape(6)
        if not np.isfinite(velocity).all() or np.linalg.norm(velocity[3:]) > contract['maximum_tool_speed_rad_s']:
            raise RuntimeError('Tool finite-state or angular-speed safety bound failed')
        applied = bool(self.loaded_case and phase['loaded'])
        if applied:
            torque[2] = float(np.clip(contract['speed_load_coefficient_nm_s_per_rad'] *
                (phase['target_speed_rad_s']-velocity[5]),
                -contract['maximum_torque_nm'], contract['maximum_torque_nm']))
            env._object_views['socket_crank'].apply_forces_and_torques_at_position(
                env._tensor(np.asarray([force], dtype=np.float32)),
                env._tensor(np.asarray([torque], dtype=np.float32)), None, env._one, True)
            self.applied_wrench_calls += 1
        result = super().step(*args, **kwargs)
        row = self.rows[-1]
        receipt = {'native_step': row['native_step'], 'time_s': row['time_s'],
            'phase': self.phase, 'body': 'socket_crank', 'world_frame': True,
            'force_n': force, 'torque_nm': torque,
            'prestep_tool_velocity': velocity.tolist(),
            'target_speed_rad_s': phase['target_speed_rad_s'],
            'wrench_api_called': applied, 'explicit_wake_called': False}
        if hasattr(self, 'goal_control'):
            receipt['goal_control'] = deepcopy(self.goal_control)
        row['wrench'] = receipt
        self.wrench_stream.write(json.dumps(receipt, separators=(',', ':'), allow_nan=False)+'\n')
        # The raw JSONL above remains untouched, including every signed point.
        self.rows[-1] = compact_native_row(row)
        if any(np.linalg.norm(v[:3]) > contract['abort_body_linear_speed_m_s'] or
               np.linalg.norm(v[3:]) > contract['abort_body_angular_speed_rad_s']
               for v in row['velocities'].values()):
            self.stream.flush()
            self.wrench_stream.flush()
            raise RuntimeError('Native motion safety bound exceeded; offending frame retained')
        return result

    def close(self):
        try:
            return super().close()
        finally:
            self.wrench_stream.close()


class GoalObserver(StageObserver):
    """Stop loading the real socket when the passive output reaches its target."""

    def __init__(self, env, output, contract, *, loaded_case, goal_contract):
        self.goal_contract = goal_contract
        self.goal_baseline_rad = None
        self.goal_reached = False
        self.goal_control = {}
        super().__init__(env, output, contract, loaded_case=loaded_case)

    def choose_goal_load(self):
        declared = self.phase_definition
        if declared['id'] == 'goal_drive':
            if not self.rows:
                raise RuntimeError('A settled output baseline is required before loading')
            last = self.rows[-1]
            angle = axial_rotation(last['poses_xyzw']['output_gear'])
            if self.goal_baseline_rad is None:
                self.goal_baseline_rad = angle
            turn = math.atan2(math.sin(angle-self.goal_baseline_rad),
                              math.cos(angle-self.goal_baseline_rad))
            error = self.goal_contract['target_rad']-turn
            if abs(error) <= self.goal_contract['cutoff_tolerance_rad']:
                self.goal_reached = True
            self.goal_control = {'baseline_output_rad': self.goal_baseline_rad,
                'measured_output_turn_rad': turn, 'remaining_rad': error,
                'measurement_time_s': last['time_s'], 'load_cutoff': self.goal_reached}
            self.phase_definition = dict(declared, loaded=not self.goal_reached,
                target_speed_rad_s=0. if self.goal_reached else
                    math.copysign(abs(declared['target_speed_rad_s']), error))
        return declared

    def step(self, *args, **kwargs):
        declared = self.choose_goal_load()
        try:
            return super().step(*args, **kwargs)
        finally:
            self.phase_definition = declared


def verify_source():
    return verify_if_snapshot(Path(__file__).resolve().parents[2],
        required_modules=('scripts.validation.validate_gear_scene',
                          'scripts.validation.gear_scene_contract'),
        entrypoint_file=Path(__file__))


def run(args):
    from scripts.validation.gear_scene_contract import build_inputs, assess_stage
    if args.goal:
        from scripts.validation.gear_goal_contract import build_goal_inputs, assess_goal
        build_inputs = lambda release: build_goal_inputs(release, practical=args.practical, tool_load=args.tool_load)
        assess_stage = assess_goal
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    env = observer = None
    report = {'schema': 'native_current_gear_scene.v1', 'passed': False,
        'physics_checks_passed': False, 'model_calls': 0,
        'task_success_claimed': False, 'physics_accepted': False,
        'robot_manipulation_qualified': False,
        'quantitative_contact_force_qualified': False,
        'object_state_writeback_after_reset': False,
        'selected_cases': args.cases, 'cases': {},
        'scope': 'Current frozen L5 scene; reset-only staged assembly and bounded torque on the independent socket. No Agent or robot manipulation.',
        'driver_override': os.environ.get('ISAAC_ALLOW_UNVALIDATED_DRIVER') == '1',
        'gpu_uuid': os.environ.get('CUDA_VISIBLE_DEVICES')}
    try:
        if verify_source() is None:
            raise ValueError('A complete source snapshot is required')
        if os.environ.get('ISAAC_BENCH_SINGLE_CUDA_DEVICE') != '1' or not report['gpu_uuid'].startswith('GPU-'):
            raise ValueError('Select one native GPU by UUID')
        if digest(args.input) != args.input_sha256:
            raise ValueError('Frozen probe input changed')
        inputs = load_json(args.input)
        report['observation_contract'] = inputs.get('observation_contract', {
            'scope': 'Full native-substep object states and raw contact points.'})
        report['input'] = {'path': str(args.input.resolve()), 'sha256': args.input_sha256}
        report['release'] = {'path': inputs['release'], 'manifest_sha256': inputs['release_manifest_sha256']}
        for path, expected in inputs['asset_files_sha256'].items():
            if digest(path) != expected:
                raise ValueError('Frozen asset changed: '+path)
        from bench.simulation.isaacsim_backend.env import FR3DexterousIsaacEnv
        env = FR3DexterousIsaacEnv(inputs['scene'], device='cuda:0', render_width=640, render_height=480)
        # Reproduce the exact input after native USD is initialized.
        if inputs != build_inputs(Path(inputs['release'])):
            raise ValueError('Input differs from the current pinned release and source contract')
        dump(out/'runtime_imports.json', verify_source())
        selected = [case for case in inputs['cases'] if case['id'] in args.cases]
        if [case['id'] for case in selected] != args.cases:
            raise ValueError('Selected cases must follow the declared staged order')
        for case in selected:
            directory = out/case['id']
            directory.mkdir()
            reset = deepcopy(inputs['initial_state'])
            reset['objects'] = deepcopy(case['reset_objects'])
            env.reset(initial_state=reset)
            if env.dt != inputs['scene']['isaacsim']['physics_dt_s']:
                raise ValueError('Native timestep differs from the frozen scene')
            metadata = env.metadata()
            checks = {'stage_inventory': stage_inventory(env),
                'contact_filters': contact_filter_coverage(env),
                'native_parameters': mixed_native_parameters(env),
                'native_all_convex_geometry': check_convex_geometry(env)}
            solver = metadata['physics_solver']
            expected_solver = inputs['scene']['isaacsim'].get('solver_type', 'TGS')
            checks['solver'] = {'passed': solver['profile_selection'] == expected_solver and
                solver['usd_readback'] == expected_solver, 'actual': solver, 'expected': expected_solver}
            dump(directory/'metadata.json', metadata)
            dump(directory/'native_checks.json', plain(checks))
            if not all(item['passed'] for item in checks.values()):
                raise RuntimeError('Native geometry, mass/material or scene configuration check failed')
            dump(directory/'initial_native_state.json', plain(env.capture_state()))
            observer_kwargs = {'loaded_case': case['mounted_tool']}
            if args.goal:
                observer_kwargs['goal_contract'] = inputs['goal_contract']
            observer_class = GoalObserver if args.goal else StageObserver
            if args.practical:
                from scripts.validation.gear_practical_observer import PracticalGoalObserver
                observer_class = PracticalGoalObserver
            observer = observer_class(
                env, directory, inputs['probe_contract'], **observer_kwargs)
            first_step = int(env.world.current_time_step_index)
            phases = case.get('phases') or [{'id': 'supply', 'duration_s': case['duration_s'],
                'loaded': False, 'target_speed_rad_s': 0.}]
            actual_phases = []
            for phase in phases:
                observer.phase_definition = phase
                phase_start = len(observer.rows)
                for index in range(round(phase['duration_s']/.02)):
                    observed = env.step({'joint_positions_rad': inputs['initial_state']['joint_positions_rad'], 'duration_s': .02})
                    if args.practical:
                        observer.after_control()
                    observer.stream.flush()
                    observer.wrench_stream.flush()
                    if observed['terminated'] or not observed['metrics']['state_finite']:
                        raise RuntimeError('Native physics terminated or state became nonfinite')
                    if index % 25 == 24:
                        last = observer.rows[-1]
                        print(json.dumps({'event': 'phase_progress', 'case': case['id'],
                            'phase': phase['id'], 'time_s': last['time_s'],
                            'yaw_degrees': {name: math.degrees(axial_rotation(pose))
                                for name, pose in last['poses_xyzw'].items()
                                if name in ('socket_crank', *case['active_gears'])}}), flush=True)
                    if args.goal and phase['id'] == 'goal_drive' and observer.goal_reached:
                        break
                actual_phases.append({'id': phase['id'],
                    'duration_s': (len(observer.rows)-phase_start)*(.02 if args.practical else env.dt)})
                print(json.dumps({'event': 'phase_completed', 'case': case['id'],
                    'phase': phase['id'], 'native_samples': int(env.world.current_time_step_index)-first_step,
                    'state_observations': len(observer.rows)}), flush=True)
            rows, wrench_calls = observer.rows, observer.applied_wrench_calls
            goal_reached = observer.goal_reached if args.goal else None
            exchange = observer.close()
            observer = None
            result = assess_stage(rows, env.scene_info, env.dt, case, inputs)
            native_samples = int(env.world.current_time_step_index)-first_step
            result['contact_exchange'] = exchange if args.practical else contact_exchange_check(
                exchange, native_samples, len(rows))
            if args.practical:
                result['contact_exchange']['passed'] &= exchange['sampled_native_steps'] == native_samples
            result['wrench_api_calls'] = wrench_calls
            result['actual_phases'] = actual_phases
            if args.goal:
                result['tool_load_cutoff_reached'] = goal_reached
            result['physics_checks_passed'] = bool(result['passed'] and result['contact_exchange']['passed'])
            try:
                result['rgbd'] = capture(env, directory, 'final')
            except Exception as exc:
                result['rgbd'] = {'passed': False, 'error': repr(exc)}
            result['passed'] = bool(result['physics_checks_passed'] and result['rgbd']['passed'])
            result['native_samples'] = native_samples
            result['state_observations'] = len(rows)
            result['scope'] = case.get('scope', report['scope'])
            dump(directory/'result.json', plain(result))
            report['cases'][case['id']] = result
            dump(out/'progress.json', plain(report))
            print(json.dumps({'event': 'case_completed', 'case': case['id'],
                'physics_checks_passed': result['physics_checks_passed'], 'passed': result['passed']}), flush=True)
            del rows
        guard = env.metadata()['static_mesh_contact_runtime']
        report['fixed_support_guard'] = guard
        fixed = guard['maximum_translation_m'] <= 1.e-6 and guard['maximum_angle_rad'] <= 1.e-5
        report['physics_checks_passed'] = bool(fixed and report['cases'] and
            all(item['physics_checks_passed'] for item in report['cases'].values()))
        report['passed'] = bool(report['physics_checks_passed'] and
            all(item['passed'] for item in report['cases'].values()))
        dump(out/'runtime_imports_final.json', verify_source())
    except BaseException as exc:
        report.update(passed=False, error={'type': type(exc).__name__, 'detail': str(exc),
            'traceback': traceback.format_exc()})
    finally:
        if observer is not None:
            try:
                report['interrupted_contact_exchange'] = observer.close()
            except Exception as exc:
                report['observer_close_error'] = repr(exc)
    report['artifacts_sha256'] = {str(p.relative_to(out)): digest(p)
        for p in out.rglob('*') if p.is_file()}
    return seal_report_before_shutdown(out, plain(report), env)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--release', type=Path)
    parser.add_argument('--input', type=Path)
    parser.add_argument('--input-sha256')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--goal', action='store_true', help='Drive the real socket to the current output goal and release')
    parser.add_argument('--practical', action='store_true', help='Observe the goal every 20 ms; retain backend native contact summaries and per-step tool loads')
    parser.add_argument('--tool-load', choices=('high_torque', 'lever_40n'), default='high_torque')
    parser.add_argument('--cases', nargs='+', choices=CASE_IDS)
    args = parser.parse_args()
    if args.practical and not args.goal:
        parser.error('--practical requires --goal')
    if args.tool_load != 'high_torque' and not args.practical:
        parser.error('The lever load requires --goal --practical')
    if args.goal and args.cases:
        parser.error('--goal selects its own target_goal case')
    args.cases = ['target_goal'] if args.goal else (args.cases or list(CASE_IDS))
    if args.prepare:
        from scripts.validation.gear_scene_contract import build_inputs
        if args.goal:
            from scripts.validation.gear_goal_contract import build_goal_inputs
            build_inputs = lambda release: build_goal_inputs(release, practical=args.practical, tool_load=args.tool_load)
        if args.output.exists():
            parser.error('Choose a fresh input path')
        dump(args.output, build_inputs(args.release))
        print(json.dumps({'input': str(args.output), 'sha256': digest(args.output)}))
        return 0
    if args.input is None or args.input_sha256 is None:
        parser.error('Native execution requires --input and --input-sha256')
    return 0 if run(args)['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
