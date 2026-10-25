"""Ordinary 20 ms state observation with unchanged native contact integration."""
from copy import deepcopy
import json

import numpy as np

from scripts.validation.validate_gear_scene import GoalObserver
from scripts.validation.validate_mechanical_dynamics import declared_mount_pairs


class PracticalGoalObserver(GoalObserver):
    """Apply only tool torque each native step; let the backend acquire contacts."""

    def __init__(self, env, output, contract, *, loaded_case, goal_contract):
        self.env, self.original = env, env.world.step
        self.contract, self.loaded_case, self.goal_contract = contract, loaded_case, goal_contract
        self.phase_definition = {'id': 'settle', 'loaded': False, 'target_speed_rad_s': 0.}
        self.goal_baseline_rad, self.goal_reached, self.goal_control = None, False, {}
        self.rows, self.window_wrenches = [], []
        self.applied_wrench_calls = self.sampled_native_steps = 0
        self.integral_tool_torque_nm = 0.
        self.complete = True
        self.stream = (output/'native_control_frames.jsonl').open('x')
        self.wrench_stream = (output/'applied_wrenches.jsonl').open('x')
        self.robot = set(env.scene_info['robot_link_paths'].values())
        self.mount = {tuple(sorted(p)) for p in declared_mount_pairs(env)}
        paths = env.scene_info['objects']
        self.drive_pairs = {frozenset((paths[a]['path'], paths[b]['path'])) for a, b in (
            ('socket_crank', 'input_gear'), ('input_gear', 'idler_gear'), ('idler_gear', 'output_gear'))}
        self.sleep_reader = None
        if contract.get('wake_loaded_tool'):
            import omni.usd
            from omni.physx import get_physx_simulation_interface
            from pxr import PhysicsSchemaTools
            interface = get_physx_simulation_interface()
            stage_id = omni.usd.get_context().get_stage_id()
            ids = {n: PhysicsSchemaTools.sdfPathToInt(paths[n]['path']) for n in env.object_names}
            self.sleep_reader = lambda: {n: bool(interface.is_sleeping(stage_id, body)) for n, body in ids.items()}
        env.world.step = self.step

    def step(self, *args, **kwargs):
        if args or kwargs != {'render': False}:
            raise RuntimeError('Ordinary native integration is required')
        from bench.simulation.isaacsim_backend.env import _np
        declared = self.choose_goal_load()
        try:
            phase, env, c = self.phase_definition, self.env, self.contract
            velocity = _np(env._object_views['socket_crank'].get_velocities()).reshape(6)
            if not np.isfinite(velocity).all() or np.linalg.norm(velocity[3:]) > c['maximum_tool_speed_rad_s']:
                raise RuntimeError('Tool finite-state or angular-speed safety bound failed')
            torque = [0., 0., 0.]
            applied = bool(self.loaded_case and phase['loaded'])
            wake = bool(applied and c.get('wake_loaded_tool'))
            if applied:
                if wake:
                    from scripts.validation.probe_gear_tool_wake import wake_single_body
                    wake_single_body(env._object_views['socket_crank'])
                error = phase['target_speed_rad_s']-velocity[5]
                self.integral_tool_torque_nm = float(np.clip(self.integral_tool_torque_nm +
                    c.get('integral_load_coefficient_nm_per_rad', 0.)*error*env.dt,
                    -c['maximum_torque_nm'], c['maximum_torque_nm']))
                torque[2] = float(np.clip(c['speed_load_coefficient_nm_s_per_rad'] * error +
                    self.integral_tool_torque_nm, -c['maximum_torque_nm'], c['maximum_torque_nm']))
                env._object_views['socket_crank'].apply_forces_and_torques_at_position(
                    env._tensor(np.zeros((1, 3), dtype=np.float32)),
                    env._tensor(np.asarray([torque], dtype=np.float32)), None, env._one, True)
                self.applied_wrench_calls += 1
            else:
                self.integral_tool_torque_nm = 0.
            result = self.original(*args, **kwargs)
            receipt = {'native_step': int(env.world.current_time_step_index),
                'time_s': float(env._current_step_time), 'phase': phase['id'],
                'body': 'socket_crank', 'world_frame': True, 'force_n': [0., 0., 0.],
                'torque_nm': torque, 'prestep_tool_velocity': velocity.tolist(),
                'target_speed_rad_s': phase['target_speed_rad_s'],
                'wrench_api_called': applied, 'explicit_wake_called': wake,
                'integral_tool_torque_nm': self.integral_tool_torque_nm,
                'goal_control': deepcopy(self.goal_control)}
            self.window_wrenches.append(receipt)
            self.wrench_stream.write(json.dumps(receipt, separators=(',', ':'), allow_nan=False)+'\n')
            return result
        finally:
            self.phase_definition = declared

    def after_control(self):
        env = self.env
        sampling = deepcopy(env._contact_sampling)
        expected = round(.02/env.dt)
        complete = (sampling['complete'] is True and sampling['sampled_substeps'] == expected
            and len(self.window_wrenches) == expected
            and self.window_wrenches[-1]['native_step'] == int(env.world.current_time_step_index)
            and all(r['native_step'] == self.window_wrenches[0]['native_step']+i
                for i, r in enumerate(self.window_wrenches)))
        self.complete &= complete
        if not complete:
            raise RuntimeError('Native backend contact or tool-load sampling missed an integration step')
        self.sampled_native_steps += expected
        state = env.capture_state()
        if not all(np.isfinite(v).all() for v in state.values()):
            raise RuntimeError('Nonfinite native state')
        contacts = deepcopy(env._contact_reports)
        by_time = {}
        for c in contacts:
            if c['impulse_magnitude_sum_ns'] > 1.e-8:
                by_time.setdefault(c['time_s'], set()).add(frozenset((c['actor0'], c['actor1'])))
        row = {'phase': self.phase_definition['id'], 'native_step': int(env.world.current_time_step_index),
            'time_s': float(env.time_s),
            'poses_xyzw': {n: state['object_q'][i].tolist() for i, n in enumerate(env.object_names)},
            'velocities': {n: state['object_qd'][i].tolist() for i, n in enumerate(env.object_names)},
            'robot_interference': any(c['impulse_magnitude_sum_ns'] > 1.e-8
                and self.robot.intersection((c['actor0'], c['actor1']))
                and tuple(sorted((c['actor0'], c['actor1']))) not in self.mount for c in contacts),
            'maximum_penetration_m': max([0., *[-c['min_separation_m'] for c in contacts]]),
            'contact_totals_source': 'native_backend_pair_sums_window.v1',
            'contact_totals': [dict(actor0=c['actor0'], actor1=c['actor1'],
                absolute_scalar_sum_ns=c['raw_scalar_abs_impulse_sum_ns'],
                normal_vector_magnitude_sum_ns=c['impulse_magnitude_sum_ns'],
                signed_actor0_impulse_vector_ns=c['impulse_vector_ns'], time_s=c['time_s']) for c in contacts],
            'simultaneous_drive_chain_samples': sum(self.drive_pairs <= pairs for pairs in by_time.values()),
            'native_contact_sampling': sampling, 'wrench': self.window_wrenches[-1]}
        if self.sleep_reader is not None:
            row['native_sleeping'] = self.sleep_reader()
        self.rows.append(row)
        self.stream.write(json.dumps(row, separators=(',', ':'), allow_nan=False)+'\n')
        self.window_wrenches.clear()
        if any(np.linalg.norm(v[:3]) > self.contract['abort_body_linear_speed_m_s'] or
               np.linalg.norm(v[3:]) > self.contract['abort_body_angular_speed_rad_s']
               for v in row['velocities'].values()):
            raise RuntimeError('Observed native motion safety bound exceeded; frame retained')

    def close(self):
        self.env.world.step = self.original
        self.stream.close()
        self.wrench_stream.close()
        return {'passed': self.complete and not self.window_wrenches,
            'sampled_native_steps': self.sampled_native_steps, 'state_observations': len(self.rows),
            'scope': 'Backend acquired each native substep; signed pair summaries retained per 20 ms window. Raw contact points and full-substep object states were not archived.'}
