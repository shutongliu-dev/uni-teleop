"""Insert, turn and withdraw one real L5 tool against three passive gears.

The selected tool alone is a prescribed native kinematic pose boundary. All
six original objects remain, and no gear state or force is written after reset.
This measures tool geometry/contact function, not robot grasp or task success.
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
from scripts.validation.gear_scene_contract import build_inputs
from scripts.validation.gear_transmission_checks import GEARS, axial_rotation
from scripts.validation.mechanical_iteration_readback import stage_iteration_settings
from scripts.validation.validate_gear_driven_tool import (
    BoundaryObserver, body_readback, boundary_api, make_environment, target_pose,
)
from scripts.validation.validate_scenario import seal_report_before_shutdown
from scripts.validation.validate_workrooms import capture, plain


CASES = {'socket_insert': 'socket_crank', 'wrench_insert': 'open_end_wrench'}
PERIOD = .02


def build_tool_inputs(case_id, release=None):
    if case_id not in CASES:
        raise ValueError('Select socket_insert or wrench_insert')
    inputs = build_inputs(release)
    tool = CASES[case_id]
    mounted = inputs['cases'][-1]['reset_objects']
    objects = deepcopy(inputs['task']['evaluation']['initial_objects'])
    for gear in GEARS:
        objects[gear] = deepcopy(mounted[gear])
    center = [.4, -.13, .824]
    if case_id == 'socket_insert':
        engaged, initial, insert_distance, withdraw_distance = [.4, -.13, .8202], [.4, -.13, .9002], .08, .08
        insertion_direction, exit_direction = [0., 0., -1.], [0., 0., 1.]
        minimum_exit = .06
    else:
        engaged, initial, insert_distance, withdraw_distance = [.392, -.13, .824], [.352, -.13, .824], .04, .05
        insertion_direction = [1., 0., 0.]
        exit_direction = [-math.cos(math.radians(20.)), -math.sin(math.radians(20.)), 0.]
        minimum_exit = .04
    speed = float(inputs['task']['evaluation']['maximum_drive_angular_speed_rad_s'])
    turn = math.radians(20.)
    quantize = lambda seconds: PERIOD*math.ceil(seconds/PERIOD-1.e-10)
    phases = [{'id': name, 'duration_s': duration} for name, duration in (
        ('settle', .2), ('insert', quantize(insert_distance/.08)),
        ('drive', quantize(turn/speed)), ('withdraw', quantize(withdraw_distance/.08)), ('final_hold', .3))]
    contract = {'case_id': case_id, 'body': tool, 'actor': '/World/Objects/'+tool,
        'native_api': boundary_api(tool), 'initial_position_m': initial,
        'engaged_position_m': engaged, 'rotation_center_m': center,
        'insertion_direction_world': insertion_direction, 'insert_distance_m': insert_distance,
        'linear_speed_m_s': .08, 'angular_speed_rad_s': speed, 'drive_turn_rad': turn,
        'withdraw_direction_world': exit_direction, 'withdraw_distance_m': withdraw_distance,
        'minimum_withdraw_distance_m': minimum_exit, 'minimum_output_turn_rad': math.radians(3.),
        'phases': phases, 'state_observation_period_s': PERIOD,
        'native_contact_period_s': inputs['scene']['isaacsim']['physics_dt_s'],
        'gear_state_writeback_after_reset': False, 'gear_force_or_constraint': False,
        'tool_pose_writeback_after_reset': True,
        'tool_velocity_writeback': False, 'tensor_kinematic_targets_used': False,
        'dynamic_tool_release_claimed': False, 'reaction_force_or_torque_calibrated': False,
        'boundary_semantics': 'Only the tested tool receives native poses before each integration step. Three gears and the two other tools remain dynamic.',
        'normal_startup_slack_is_allowed': True, 'scope': __doc__}
    objects[tool] = {'position_m': initial, 'quaternion_wxyz': [1., 0., 0., 0.]}
    inputs.update(schema='mechanical_gear_tool_function_inputs.v1',
        cases=[{'id': case_id, 'active_gears': list(GEARS), 'mounted_tool': True,
            'reset_objects': objects, 'duration_s': sum(p['duration_s'] for p in phases),
            'phases': deepcopy(phases), 'scope': __doc__}],
        tool_function_contract=contract, operator_boundary_contract=deepcopy(contract),
        object_state_writeback_after_reset=True, tool_pose_writeback_after_reset=True,
        gear_state_writeback_after_reset=False)
    return inputs


def target_for_phase(contract, phase, phase_time_s):
    """Analytic tool motion; clamp fractional control windows to exact endpoints."""
    if phase not in {p['id'] for p in contract['phases']} or not math.isfinite(phase_time_s):
        raise ValueError('A declared phase and finite elapsed time are required')
    elapsed = max(0., phase_time_s)
    position, yaw = np.asarray(contract['initial_position_m'], float), 0.
    if phase == 'insert':
        position += np.asarray(contract['insertion_direction_world'])*min(
            contract['insert_distance_m'], contract['linear_speed_m_s']*elapsed)
    elif phase in ('drive', 'withdraw', 'final_hold'):
        yaw = min(contract['drive_turn_rad'], contract['angular_speed_rad_s']*elapsed) if phase == 'drive' else contract['drive_turn_rad']
        center = np.asarray(contract['rotation_center_m'])
        offset = np.asarray(contract['engaged_position_m'])-center
        cosine, sine = math.cos(yaw), math.sin(yaw)
        rotation = np.array([[cosine, -sine, 0.], [sine, cosine, 0.], [0., 0., 1.]])
        position = center+rotation @ offset
        if phase in ('withdraw', 'final_hold'):
            distance = contract['withdraw_distance_m'] if phase == 'final_hold' else min(
                contract['withdraw_distance_m'], contract['linear_speed_m_s']*elapsed)
            position += distance*np.asarray(contract['withdraw_direction_world'])
    return target_pose(position, yaw)


class FunctionObserver(BoundaryObserver):
    def __init__(self, env, output, inputs):
        self.phase_steps = 0
        super().__init__(env, output, inputs)

    def start_phase(self, phase):
        self.phase, self.phase_steps = phase, 0

    def boundary_pose(self):
        self.phase_steps += 1
        pose = target_for_phase(self.contract, self.phase, self.phase_steps*self.env.dt)
        self.position = pose[0, :3].tolist()
        self.yaw = axial_rotation(pose[0])
        return pose


def assess_function(rows, inputs, flags):
    c, tool = inputs['tool_function_contract'], inputs['tool_function_contract']['body']
    expected_phases = [p['id'] for p in c['phases'] for _ in range(round(p['duration_s']/PERIOD))]
    contiguous = (len(rows) == len(expected_phases)
        and [r['phase'] for r in rows] == expected_phases
        and all(abs(r['time_s']-(i+1)*PERIOD) < 1.e-8
            and (not i or r['native_step']-rows[i-1]['native_step'] == round(PERIOD/c['native_contact_period_s']))
            for i, r in enumerate(rows)))
    drive_indices = [i for i, row in enumerate(rows) if row['phase'] == 'drive']
    baseline = drive_indices[0]-1
    turns = {}
    for name in (*GEARS, tool):
        angles = np.unwrap([axial_rotation(r['poses_xyzw'][name]) for r in rows])
        turns[name] = float(angles[-1]-angles[baseline])
    paths = {n: '/World/Objects/'+n for n in (*GEARS, tool)}
    pairs = ((tool, 'input_gear'), ('input_gear', 'idler_gear'), ('idler_gear', 'output_gear'))
    contact_counts = {a+'__'+b: len({r['time_s'] for i in drive_indices for r in rows[i]['native_contacts']
        if frozenset((r['actor0'], r['actor1'])) == frozenset((paths[a], paths[b]))
        and r['impulse_magnitude_sum_ns'] > 1.e-8}) for a, b in pairs}
    tail = rows[-round(.3/PERIOD)-1:]
    drive_end = rows[drive_indices[-1]]['poses_xyzw'][tool]
    exit_distance = min(float((np.asarray(r['poses_xyzw'][tool][:3])-drive_end[:3]) @ np.asarray(c['withdraw_direction_world'])) for r in tail)
    final_touch = [r for row in tail for r in row['native_contacts']
        if paths[tool] in (r['actor0'], r['actor1']) and r['impulse_magnitude_sum_ns'] > 1.e-8]
    rest, seated, supported = {}, {}, {}
    for name in GEARS:
        poses = np.asarray([r['poses_xyzw'][name] for r in tail], float)
        quaternions = poses[:, 3:]/np.linalg.norm(poses[:, 3:], axis=1)[:, None]
        rotations = 2.*np.arccos(np.clip(np.abs(quaternions @ quaternions[0]), 0., 1.))
        rest[name] = {'position_range_m': float(np.linalg.norm(poses[:, :3]-poses[0, :3], axis=1).max()),
            'angle_range_degrees': math.degrees(float(rotations.max()))}
        pose, expected = poses[-1], np.asarray(inputs['cases'][0]['reset_objects'][name]['position_m'])
        tilt = math.degrees(math.acos(float(np.clip(1.-2.*(pose[3]**2+pose[4]**2), -1., 1.))))
        seated[name] = bool(np.linalg.norm(pose[:2]-expected[:2]) <= .002 and abs(pose[2]-expected[2]) <= .003 and tilt <= 5.)
        body = paths[name]
        supported[name] = any((r['impulse_vector_ns'][2] if r['actor0'] == body else -r['impulse_vector_ns'][2]) > 1.e-8
            for row in tail for r in row['native_contacts'] if body in (r['actor0'], r['actor1'])
            and ((r['actor1'] if r['actor0'] == body else r['actor0']) == '/World/Fixtures/gear_workplate'
                 or (r['actor1'] if r['actor0'] == body else r['actor0']).startswith('/World/Fixtures/gear_workplate/')))
    maximum_penetration = max([0., *[-r['min_separation_m'] for row in rows for r in row['native_contacts']]])
    actual_insert = np.asarray(rows[baseline]['poses_xyzw'][tool][:3])
    follow_position = max(float(np.linalg.norm(np.asarray(r['poses_xyzw'][tool][:3])-r['boundary_command']['target_transform_xyzw'][:3])) for r in rows)
    checks = {'complete_native_contact_and_state_windows': contiguous,
        'all_six_original_body_modes': set(flags) == set(inputs['cases'][0]['reset_objects'])
            and all(v['rigid_body_enabled'] and v['kinematic_enabled'] == (n == tool) for n, v in flags.items()),
        'actual_tool_inserted': float(np.linalg.norm(actual_insert-c['engaged_position_m'])) <= .003,
        'actual_tool_turned': turns[tool] >= math.radians(15.),
        'actual_tool_and_two_mesh_contacts': all(contact_counts.values()),
        'correct_gear_directions': turns['input_gear'] > 0. and turns['idler_gear'] < 0. and turns['output_gear'] > 0.,
        'output_moves_at_least_three_degrees': turns['output_gear'] >= c['minimum_output_turn_rad'],
        'actual_tool_withdrawn_without_load': exit_distance >= c['minimum_withdraw_distance_m'] and not final_touch,
        'gears_remain_on_supported_shafts': all(seated.values()) and all(supported.values()),
        'passive_gears_can_stop': all(v['position_range_m'] <= .001 and v['angle_range_degrees'] <= .5 for v in rest.values()),
        'no_obvious_penetration': maximum_penetration <= inputs['task']['physical_geometry']['module_m']}
    return {'passed': all(checks.values()), 'checks': checks,
        'relative_turn_degrees': {n: math.degrees(v) for n, v in turns.items()},
        'contact_native_samples_during_drive': contact_counts,
        'actual_withdraw_distance_m': exit_distance, 'final_tool_positive_contacts': len(final_touch),
        'actual_insert_position_m': actual_insert.tolist(), 'maximum_tool_position_tracking_error_m': follow_position,
        'final_support': supported, 'final_seated': seated, 'released_gears_rest': rest,
        'maximum_reported_penetration_m': maximum_penetration,
        'normal_startup_slack_is_allowed': True, 'task_success_claimed': False,
        'robot_grasp_qualified': False, 'dynamic_tool_release_qualified': False, 'scope': __doc__}


def source_verification():
    return verify_if_snapshot(Path(__file__).resolve().parents[2], required_modules=(
        'scripts.validation.validate_gear_tool_function', 'scripts.validation.validate_gear_driven_tool'),
        entrypoint_file=Path(__file__))


def run(args):
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    env = observer = None
    report = {'schema': 'native_gear_tool_function_probe.v1', 'passed': False, 'model_calls': 0,
        'task_success_claimed': False, 'physics_accepted': False, 'operator_driven_tool_boundary': True,
        'gear_state_writeback_after_reset': False, 'tool_pose_writeback_after_reset': True,
        'dynamic_tool_release_qualified': False, 'scope': __doc__}
    try:
        if source_verification() is None or digest(args.input) != args.input_sha256:
            raise ValueError('Complete sealed source and exact frozen input are required')
        if os.environ.get('ISAAC_BENCH_SINGLE_CUDA_DEVICE') != '1' or not os.environ.get('CUDA_VISIBLE_DEVICES', '').startswith('GPU-'):
            raise ValueError('Select exactly one GPU by UUID')
        inputs = load_json(args.input)
        if inputs != build_tool_inputs(args.case, inputs['release']):
            raise ValueError('Input differs from the frozen new-layout tool-function contract')
        for path, expected in inputs['asset_files_sha256'].items():
            if digest(path) != expected:
                raise ValueError('Frozen asset changed: '+path)
        dump(output/'runtime_imports.json', source_verification())
        c = inputs['tool_function_contract']
        env, changes = make_environment(inputs['scene'], c['body'])
        reset = deepcopy(inputs['initial_state'])
        reset['objects'] = deepcopy(inputs['cases'][0]['reset_objects'])
        env.reset(initial_state=reset)
        dump(output/'mode_changes.json', changes)
        dump(output/'reset_receipts.json', env.operator_reset_receipts)
        dump(output/'iteration_settings.json', stage_iteration_settings(env))
        dump(output/'body_readback.json', plain(body_readback(env, c['body'], all_objects=True)))
        dump(output/'metadata.json', env.metadata())
        path = c['actor']
        if any(p == path or p.startswith(path+'/') for p in env.scene_info.get('kinematic_body_paths', [])):
            raise RuntimeError('The operator tool must not be a guarded fixed support')
        flags = body_readback(env, c['body'], all_objects=True)
        if any(v['kinematic_enabled'] != (n == c['body']) for n, v in flags.items()):
            raise RuntimeError('Only the tested tool may be kinematic')
        observer = FunctionObserver(env, output, inputs)
        for phase in c['phases']:
            observer.start_phase(phase['id'])
            start = None
            for i in range(round(phase['duration_s']/PERIOD)):
                obs = env.step({'joint_positions_rad': inputs['initial_state']['joint_positions_rad'], 'duration_s': PERIOD})
                observer.after_control()
                row = observer.rows[-1]
                if obs['terminated'] or not obs['metrics']['state_finite']:
                    raise RuntimeError('Native state became invalid')
                if start is None:
                    start = row['poses_xyzw'][c['body']]
                if phase['id'] in ('insert', 'drive') and i == min(24, round(phase['duration_s']/PERIOD)-1):
                    actual = row['poses_xyzw'][c['body']]
                    moved = (abs(axial_rotation(actual)-axial_rotation(start)) >= math.radians(1.) if phase['id'] == 'drive'
                        else np.linalg.norm(np.asarray(actual[:3])-start[:3]) >= .005)
                    if not moved:
                        raise RuntimeError('The selected tool boundary did not produce actual native motion')
                if i % 25 == 24 or i+1 == round(phase['duration_s']/PERIOD):
                    print(json.dumps({'event': 'tool_function_progress', 'case': args.case,
                        'phase': phase['id'], 'time_s': row['time_s'],
                        'tool_position_m': row['poses_xyzw'][c['body']][:3],
                        'yaw_degrees': {n: math.degrees(axial_rotation(row['poses_xyzw'][n])) for n in (*GEARS, c['body'])}}), flush=True)
        rows, exchange = observer.rows, observer.close()
        observer = None
        flags = body_readback(env, c['body'], all_objects=True)
        result = assess_function(rows, inputs, flags)
        report.update(result=result, contact_exchange=exchange, final_body_readback=flags,
            physics_checks_passed=result['passed'] and exchange['complete_control_windows'])
        report['rgbd'] = capture(env, output, 'final')
        report['passed'] = bool(report['physics_checks_passed'] and report['rgbd']['passed'])
        dump(output/'iteration_settings_final.json', stage_iteration_settings(env))
        dump(output/'runtime_imports_final.json', source_verification())
    except BaseException as exc:
        report['error'] = {'type': type(exc).__name__, 'detail': str(exc), 'traceback': traceback.format_exc()}
    finally:
        if observer is not None:
            report['interrupted_contact_exchange'] = observer.close()
    report['artifacts_sha256'] = {str(p.relative_to(output)): digest(p) for p in output.rglob('*') if p.is_file()}
    return seal_report_before_shutdown(output, plain(report), env)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--case', choices=tuple(CASES), required=True)
    parser.add_argument('--release', type=Path)
    parser.add_argument('--input', type=Path)
    parser.add_argument('--input-sha256')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.prepare:
        if args.output.exists():
            parser.error('Choose a fresh input path')
        dump(args.output, build_tool_inputs(args.case, args.release))
        return 0
    if args.input is None or args.input_sha256 is None:
        parser.error('Native mode requires a frozen input and hash')
    return 0 if run(args)['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
