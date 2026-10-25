"""Operator-driven actual socket boundary, with three passive native gears.

The socket alone is kinematic in this engineering variant. Its original meshes,
material and stored mass remain; its commanded motion replaces an operator's
motion boundary. No gear pose, velocity, joint or force is authored after reset.
This is neither a released dynamic-tool test nor robot task success.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
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
from scripts.validation.validate_scenario import seal_report_before_shutdown
from scripts.validation.validate_workrooms import capture, plain


TOOL = 'socket_crank'
BOUNDARY_API = 'RigidBodyView.set_transforms(socket_only)'


def checked_boundary_tool(tool):
    if tool not in (TOOL, 'open_end_wrench'):
        raise ValueError('Only the socket or open-end wrench may be an operator boundary')
    return tool


def boundary_api(tool=TOOL):
    return BOUNDARY_API if checked_boundary_tool(tool) == TOOL else 'RigidBodyView.set_transforms(open_end_wrench_only)'


def build_driven_inputs(release=None):
    inputs = build_inputs(release)
    case = deepcopy(inputs['cases'][-1])
    case.update(id='operator_driven_socket', scope=__doc__)
    inputs.update(schema='operator_driven_gear_probe.v1', cases=[case],
        operator_boundary_contract={
            'body': TOOL, 'actor': '/World/Objects/socket_crank',
            'native_api': BOUNDARY_API,
            'boundary_semantics': 'Prescribed native socket poses before each integration step; contacts move the three passive gears. This pose boundary does not qualify tool velocity or reaction forces.',
            'tensor_kinematic_targets_used': False,
            'tool_velocity_writeback': False,
            'angular_speed_rad_s': inputs['task']['evaluation']['maximum_drive_angular_speed_rad_s'],
            'maximum_drive_s': 8., 'settle_s': 1., 'stop_hold_s': .3,
            'withdraw_height_m': .08, 'withdraw_speed_m_s': .08, 'free_gear_hold_s': 1.,
            'minimum_actual_withdraw_height_m': .06,
            'target_rad': inputs['task']['evaluation']['nominal_output_turn_rad'],
            'cutoff_tolerance_rad': math.radians(3.),
            'minimum_output_turn_rad': inputs['task']['evaluation']['minimum_output_turn_rad'],
            'maximum_output_turn_rad': inputs['task']['evaluation']['maximum_output_turn_rad'],
            'gear_state_writeback_after_reset': False, 'gear_force_or_constraint': False,
            'dynamic_tool_release_claimed': False,
            'reaction_force_or_torque_calibrated': False,
            'state_observation_period_s': .02,
            'scope': __doc__})
    return inputs


@contextmanager
def kinematic_socket_authoring(changes, tool=TOOL):
    """Keep the normal object path and change this one body's mode pre-play."""
    from pxr import UsdPhysics
    from bench.simulation.isaacsim_backend import scene
    original = scene.add_task_asset
    checked_boundary_tool(tool)

    def author(stage, spec, *, dynamic):
        info = original(stage, spec, dynamic=dynamic)
        if spec['id'] == tool:
            if not dynamic or info['path'] != '/World/Objects/'+tool:
                raise ValueError('Expected the original independent tool object')
            body = UsdPhysics.RigidBodyAPI(stage.GetPrimAtPath(info['path']))
            if not body or body.GetKinematicEnabledAttr().Get() is True:
                raise ValueError('The tool must originate as a native dynamic body')
            before = body.GetKinematicEnabledAttr().Get()
            body.CreateKinematicEnabledAttr(True)
            changes.append({'path': info['path'], 'property': 'physics:kinematicEnabled',
                'before': before, 'after': True, 'time': 'before native play'})
        return info

    scene.add_task_asset = author
    try:
        yield
    finally:
        scene.add_task_asset = original


class ResetSocketView:
    """Avoid dynamic-velocity initialization for the kinematic socket only."""
    def __init__(self, view, env, tool=TOOL):
        self.view, self.env = view, env
        self.tool = checked_boundary_tool(tool)

    def __getattr__(self, name):
        return getattr(self.view, name)

    def set_transforms(self, data, indices):
        if not self.env.operator_reset_active:
            raise RuntimeError('Direct socket pose writes are reset-only')
        self.view.set_transforms(data, indices)
        self.env.operator_reset_receipts.append({'body': self.tool, 'pose_initialized_at_reset': True,
            'boundary_api': boundary_api(self.tool)})

    def set_velocities(self, data, indices):
        if not self.env.operator_reset_active:
            raise RuntimeError('No post-reset socket velocity write is permitted')
        self.env.operator_reset_receipts.append({'body': self.tool,
            'kinematic_velocity_initialization_skipped': True})


def make_environment(scene, tool=TOOL):
    checked_boundary_tool(tool)
    from bench.simulation.isaacsim_backend.env import FR3DexterousIsaacEnv

    class BoundaryEnvironment(FR3DexterousIsaacEnv):
        def __init__(self, *args, **kwargs):
            self.operator_reset_active = False
            self.operator_reset_receipts = []
            self.operator_body_mode_changes = []
            super().__init__(*args, **kwargs)

        def _initialize(self):
            # SimulationApp has started before this hook, so native USD and
            # scene imports resolve from the pinned Isaac runtime.
            with kinematic_socket_authoring(self.operator_body_mode_changes, tool):
                return super()._initialize()

        def _initialize_views(self):
            super()._initialize_views()
            self._object_views[tool] = ResetSocketView(self._object_views[tool], self, tool)

        def reset(self, *args, **kwargs):
            self.operator_reset_active = True
            try:
                return super().reset(*args, **kwargs)
            finally:
                self.operator_reset_active = False

    env = BoundaryEnvironment(scene, device='cuda:0', render_width=640, render_height=480)
    changes = env.operator_body_mode_changes
    if len(changes) != 1:
        raise RuntimeError('Exactly one pre-play tool mode change is required')
    return env, changes


def target_pose(position, yaw):
    return np.asarray([[*position, 0., 0., math.sin(yaw/2), math.cos(yaw/2)]], dtype=np.float32)


def submit_socket_target(env, pose, tool=TOOL):
    """Prescribe one allowed tool pose; gears receive no state writes.

    The pinned GPU runtime cannot set kinematic targets, and USD animation is
    not consumed by the manual integration path. This declared engineering
    boundary uses native pose writes at 0.5 ms instead. It tests geometric
    contact transmission, without claiming dynamic tool or force calibration.
    """
    checked_boundary_tool(tool)
    view = env._object_views[tool]
    if view.count != 1:
        raise ValueError('Kinematic boundary must address exactly one tool')
    pose = np.asarray(pose, dtype=np.float32)
    if pose.shape != (1, 7) or not np.isfinite(pose).all():
        raise ValueError('A finite single-body tool transform is required')
    path = env.scene_info['objects'][tool]['path']
    if path != '/World/Objects/'+tool:
        raise ValueError('The operator boundary can address only the original selected tool')
    native = view.view if isinstance(view, ResetSocketView) else view
    native.set_transforms(env._tensor(pose), env._one)


class BoundaryObserver:
    def __init__(self, env, output, inputs):
        self.env, self.original, self.inputs = env, env.world.step, inputs
        self.contract = inputs['operator_boundary_contract']
        self.tool = checked_boundary_tool(self.contract['body'])
        reset = inputs['cases'][0]['reset_objects'][self.tool]
        self.position, self.yaw = list(reset['position_m']), 0.
        self.phase, self.rows, self.window_commands = 'settle', [], []
        self.native_steps = 0
        self.stream = (output/'native_control_frames.jsonl').open('x')
        self.commands = (output/'operator_boundary_targets.jsonl').open('x')
        self.pairs = {frozenset(('/World/Objects/'+a, '/World/Objects/'+b)) for a, b in (
            (self.tool, 'input_gear'), ('input_gear', 'idler_gear'), ('idler_gear', 'output_gear'))}
        env.world.step = self.step

    def boundary_pose(self):
        env, c = self.env, self.contract
        if self.phase == 'drive':
            self.yaw += c['angular_speed_rad_s']*env.dt
        elif self.phase == 'withdraw':
            initial_z = self.inputs['cases'][0]['reset_objects'][self.tool]['position_m'][2]
            self.position[2] = min(initial_z+c['withdraw_height_m'],
                self.position[2]+c['withdraw_speed_m_s']*env.dt)
        return target_pose(self.position, self.yaw)

    def step(self, *args, **kwargs):
        if args or kwargs != {'render': False}:
            raise RuntimeError('Ordinary native integration is required')
        env = self.env
        target = self.boundary_pose()
        submit_socket_target(env, target, self.tool)
        result = self.original(*args, **kwargs)
        command = {'native_step': int(env.world.current_time_step_index),
            'time_s': float(env._current_step_time), 'phase': self.phase,
            'body': self.tool, 'native_api': boundary_api(self.tool),
            'target_transform_xyzw': target[0].tolist(),
            'target_yaw_rad': self.yaw, 'gear_state_setters_called': False,
            'external_wrench_api_called': False}
        self.commands.write(json.dumps(command, separators=(',', ':'), allow_nan=False)+'\n')
        self.window_commands.append(command)
        self.native_steps += 1
        return result

    def after_control(self):
        env = self.env
        sampling = deepcopy(env._contact_sampling)
        expected = round(.02/env.dt)
        if (sampling['complete'] is not True or sampling['sampled_substeps'] != expected
                or len(self.window_commands) != expected
                or any(b['native_step'] != a['native_step']+1 for a, b in zip(self.window_commands, self.window_commands[1:]))):
            raise RuntimeError('Native contact acquisition or boundary command stream is incomplete')
        state, contacts = env.capture_state(), deepcopy(env._contact_reports)
        if not all(np.isfinite(v).all() for v in state.values()):
            raise RuntimeError('Nonfinite native physical state')
        by_time = {}
        for contact in contacts:
            if contact['impulse_magnitude_sum_ns'] > 1.e-8:
                by_time.setdefault(contact['time_s'], set()).add(frozenset((contact['actor0'], contact['actor1'])))
        row = {'phase': self.phase, 'native_step': int(env.world.current_time_step_index),
            'time_s': float(env.time_s), 'native_contact_sampling': sampling,
            'poses_xyzw': {n: state['object_q'][i].tolist() for i, n in enumerate(env.object_names)},
            'velocities': {n: state['object_qd'][i].tolist() for i, n in enumerate(env.object_names)},
            'native_contacts': contacts, 'contact_source': 'original_native_backend_pair_summaries',
            'simultaneous_drive_chain_samples': sum(self.pairs <= pairs for pairs in by_time.values()),
            'boundary_command': self.window_commands[-1], 'boundary_command_count': len(self.window_commands)}
        self.rows.append(row)
        self.stream.write(json.dumps(plain(row), separators=(',', ':'), allow_nan=False)+'\n')
        self.stream.flush()
        self.commands.flush()
        self.window_commands.clear()
        limits = self.inputs['probe_contract']
        if any(np.linalg.norm(v[:3]) > limits['abort_body_linear_speed_m_s']
                or np.linalg.norm(v[3:]) > limits['abort_body_angular_speed_rad_s']
                for n, v in row['velocities'].items() if n in GEARS):
            raise RuntimeError('Passive gear crossed the existing engineering explosive-motion bound')

    def close(self):
        self.env.world.step = self.original
        self.stream.close()
        self.commands.close()
        return {'native_boundary_commands': self.native_steps, 'state_observations': len(self.rows),
            'complete_control_windows': not self.window_commands,
            'scope': 'Native contacts acquired every substep; physical poses observed every 20 ms.'}


def body_readback(env, tool=TOOL, *, all_objects=False):
    from pxr import UsdPhysics
    from bench.simulation.isaacsim_backend.env import _np
    result = {}
    checked_boundary_tool(tool)
    for name in (env.object_names if all_objects else (*GEARS, tool)):
        path = env.scene_info['objects'][name]['path']
        body = UsdPhysics.RigidBodyAPI(env.stage.GetPrimAtPath(path))
        result[name] = {'path': path, 'rigid_body_enabled': bool(body.GetRigidBodyEnabledAttr().Get()),
            'kinematic_enabled': bool(body.GetKinematicEnabledAttr().Get()),
            'native_mass_kg': _np(env._object_views[name].get_masses()).reshape(-1).tolist(),
            'native_inertia': _np(env._object_views[name].get_inertias()).reshape(-1).tolist()}
    return result


def assess(rows, inputs, flags):
    c = inputs['operator_boundary_contract']
    base_index = max(i for i, r in enumerate(rows) if r['phase'] == 'settle')
    angles = {n: np.unwrap([axial_rotation(r['poses_xyzw'][n]) for r in rows]) for n in GEARS}
    turns = {n: float(v[-1]-v[base_index]) for n, v in angles.items()}
    tail = rows[-round(.3/.02)-1:]
    rest = {}
    for name in GEARS:
        poses = np.asarray([r['poses_xyzw'][name] for r in tail])
        yaw = np.unwrap([axial_rotation(p) for p in poses])
        rest[name] = {'position_range_m': float(np.linalg.norm(poses[:, :3]-poses[0, :3], axis=1).max()),
            'yaw_range_degrees': math.degrees(float(np.ptp(yaw)))}
    contacts = [r for row in rows if row['phase'] == 'drive' for r in row['native_contacts']]
    expected_pairs = [frozenset(('/World/Objects/'+a, '/World/Objects/'+b)) for a, b in (
        (TOOL, 'input_gear'), ('input_gear', 'idler_gear'), ('idler_gear', 'output_gear'))]
    observed = {frozenset((r['actor0'], r['actor1'])) for r in contacts if r['impulse_magnitude_sum_ns'] > 1.e-8}
    final_tool_contacts = [r for row in tail for r in row['native_contacts']
        if '/World/Objects/socket_crank' in (r['actor0'], r['actor1'])]
    final_tool_positive_contacts = [r for r in final_tool_contacts if r['impulse_magnitude_sum_ns'] > 1.e-8]
    actual_withdraw_height = min(r['poses_xyzw'][TOOL][2] for r in tail)-rows[base_index]['poses_xyzw'][TOOL][2]
    simultaneous_chain_samples = sum(r['simultaneous_drive_chain_samples'] for r in rows if r['phase'] == 'drive')
    maximum_penetration = max([0., *[-r['min_separation_m'] for row in rows for r in row['native_contacts']]])
    ratios = {'idler_per_input': turns['idler_gear']/turns['input_gear'] if turns['input_gear'] else None,
              'output_per_input': turns['output_gear']/turns['input_gear'] if turns['input_gear'] else None}
    seated, supported = {}, {}
    for name in GEARS:
        pose = np.asarray(rows[-1]['poses_xyzw'][name])
        expected = np.asarray(inputs['cases'][0]['reset_objects'][name]['position_m'])
        tilt = math.degrees(math.acos(float(np.clip(1.-2.*(pose[3]**2+pose[4]**2), -1., 1.))))
        seated[name] = bool(np.linalg.norm(pose[:2]-expected[:2]) <= .002
            and abs(pose[2]-expected[2]) <= .003 and tilt <= 5.)
        body = '/World/Objects/'+name
        supported[name] = any((r['impulse_vector_ns'][2] if r['actor0'] == body else -r['impulse_vector_ns'][2]) > 1.e-8
            for row in tail for r in row['native_contacts']
            if body in (r['actor0'], r['actor1'])
            and ((r['actor1'] if r['actor0'] == body else r['actor0']) == '/World/Fixtures/gear_workplate'
                 or (r['actor1'] if r['actor0'] == body else r['actor0']).startswith('/World/Fixtures/gear_workplate/')))
    checks = {'three_gears_remain_free_dynamic': all(flags[n]['rigid_body_enabled'] and not flags[n]['kinematic_enabled'] for n in GEARS),
        'only_socket_is_kinematic_boundary': flags[TOOL]['kinematic_enabled'],
        'output_reaches_original_task_sector': c['minimum_output_turn_rad'] <= turns['output_gear'] <= c['maximum_output_turn_rad'],
        'correct_gear_directions': turns['input_gear'] > 0. and turns['idler_gear'] < 0. and turns['output_gear'] > 0.,
        'reasonable_net_gear_ratios': ratios['idler_per_input'] is not None
            and abs(ratios['idler_per_input']+.625) <= .2 and abs(ratios['output_per_input']-1.) <= .3,
        'actual_tool_and_two_mesh_contacts': all(pair in observed for pair in expected_pairs),
        'tool_withdrawn_without_final_touch': actual_withdraw_height >= c['minimum_actual_withdraw_height_m']
            and not final_tool_positive_contacts,
        'passive_gears_can_stop': all(r['position_range_m'] <= .001 and r['yaw_range_degrees'] <= .5 for r in rest.values()),
        'gears_remain_on_supported_shafts': all(seated.values()) and all(supported.values()),
        'no_obvious_penetration': maximum_penetration <= inputs['task']['physical_geometry']['module_m']}
    return {'passed': all(checks.values()), 'checks': checks,
        'relative_gear_turn_degrees': {n: math.degrees(v) for n, v in turns.items()},
        'final_turn_ratios': ratios, 'final_support': supported, 'final_seated': seated,
        'contact_diagnostics': {'actual_simultaneous_contact_chain': simultaneous_chain_samples > 0,
            'simultaneous_drive_chain_samples': simultaneous_chain_samples,
            'scope': 'Same-native-step three-interface contact is a reference; normal slack and intermittent physical transfer do not require it.'},
        'actual_tool_withdrawal': {'minimum_tail_height_increase_m': actual_withdraw_height,
            'minimum_required_height_increase_m': c['minimum_actual_withdraw_height_m'],
            'tail_positive_impulse_contacts': len(final_tool_positive_contacts),
            'tail_zero_load_contact_reports': len(final_tool_contacts)-len(final_tool_positive_contacts)},
        'released_gears_rest': rest, 'maximum_reported_penetration_m': maximum_penetration,
        'task_success_claimed': False, 'dynamic_tool_release_qualified': False,
        'scope': __doc__}


def source_verification():
    return verify_if_snapshot(Path(__file__).resolve().parents[2],
        required_modules=('scripts.validation.validate_gear_driven_tool',), entrypoint_file=Path(__file__))


def run(args):
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    env = observer = None
    report = {'schema': 'operator_driven_native_gear_probe.v1', 'passed': False,
        'model_calls': 0, 'task_success_claimed': False, 'physics_accepted': False,
        'gear_state_writeback_after_reset': False, 'operator_driven_tool_boundary': True,
        'dynamic_tool_release_qualified': False, 'scope': __doc__}
    try:
        if source_verification() is None or digest(args.input) != args.input_sha256:
            raise ValueError('Complete sealed source and exact frozen input are required')
        if os.environ.get('ISAAC_BENCH_SINGLE_CUDA_DEVICE') != '1' or not os.environ.get('CUDA_VISIBLE_DEVICES', '').startswith('GPU-'):
            raise ValueError('Select exactly one GPU by UUID')
        inputs = load_json(args.input)
        for path, expected in inputs['asset_files_sha256'].items():
            if digest(path) != expected:
                raise ValueError('Frozen asset changed: '+path)
        dump(output/'runtime_imports.json', source_verification())
        env, changes = make_environment(inputs['scene'])
        if inputs != build_driven_inputs(inputs['release']):
            raise ValueError('Input differs from the pinned new-layout boundary contract')
        reset = deepcopy(inputs['initial_state'])
        reset['objects'] = deepcopy(inputs['cases'][0]['reset_objects'])
        env.reset(initial_state=reset)
        dump(output/'mode_changes.json', changes)
        dump(output/'reset_receipts.json', env.operator_reset_receipts)
        dump(output/'iteration_settings.json', stage_iteration_settings(env))
        flags = body_readback(env)
        dump(output/'body_readback.json', plain(flags))
        dump(output/'metadata.json', env.metadata())
        fixed_paths = env.scene_info.get('kinematic_body_paths', [])
        if any(p == flags[TOOL]['path'] or p.startswith(flags[TOOL]['path']+'/') for p in fixed_paths):
            raise RuntimeError('The operator tool must not be a guarded fixed support')
        report['operator_tool_excluded_from_fixed_support_guard'] = True
        if not flags[TOOL]['kinematic_enabled'] or any(flags[n]['kinematic_enabled'] for n in GEARS):
            raise RuntimeError('Native body modes differ from the declared tool-only boundary')
        observer = BoundaryObserver(env, output, inputs)
        c, cutoff = inputs['operator_boundary_contract'], False
        def advance(phase, duration, stop_at_goal=False):
            nonlocal cutoff
            observer.phase = phase
            first = len(observer.rows)
            for i in range(round(duration/.02)):
                obs = env.step({'joint_positions_rad': inputs['initial_state']['joint_positions_rad'], 'duration_s': .02})
                observer.after_control()
                if obs['terminated'] or not obs['metrics']['state_finite']:
                    raise RuntimeError('Native state became invalid')
                current = observer.rows[-1]
                if i % 25 == 24:
                    print(json.dumps({'event': 'boundary_progress', 'phase': phase, 'time_s': current['time_s'],
                        'yaw_degrees': {n: math.degrees(axial_rotation(current['poses_xyzw'][n])) for n in (*GEARS, TOOL)}}), flush=True)
                if stop_at_goal:
                    base = observer.rows[first-1]
                    tool_turn = axial_rotation(current['poses_xyzw'][TOOL])-axial_rotation(base['poses_xyzw'][TOOL])
                    if i == 24 and abs(tool_turn) < math.radians(1.):
                        raise RuntimeError('The operator boundary did not produce actual native socket motion')
                    turn = axial_rotation(current['poses_xyzw']['output_gear'])-axial_rotation(base['poses_xyzw']['output_gear'])
                    if abs(c['target_rad']-turn) <= c['cutoff_tolerance_rad']:
                        cutoff = True
                        break
        advance('settle', c['settle_s'])
        advance('drive', c['maximum_drive_s'], stop_at_goal=True)
        advance('stop', c['stop_hold_s'])
        advance('withdraw', c['withdraw_height_m']/c['withdraw_speed_m_s'])
        advance('free_gear_hold', c['free_gear_hold_s'])
        rows = observer.rows
        exchange = observer.close()
        observer = None
        final_flags = body_readback(env)
        result = assess(rows, inputs, final_flags)
        result['goal_cutoff_reached'] = cutoff
        report.update(result=result, contact_exchange=exchange, final_body_readback=final_flags,
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
    parser.add_argument('--release', type=Path)
    parser.add_argument('--input', type=Path)
    parser.add_argument('--input-sha256')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.prepare:
        if args.output.exists():
            parser.error('Choose a fresh input path')
        dump(args.output, build_driven_inputs(args.release))
        return 0
    if args.input is None or args.input_sha256 is None:
        parser.error('Native mode requires a frozen input and hash')
    return 0 if run(args)['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
