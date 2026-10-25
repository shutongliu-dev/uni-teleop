"""Small socket-only force/wake comparison; no gear or task qualification."""
import argparse
from copy import deepcopy
import json
import math
import os
from pathlib import Path
import traceback

import numpy as np

from bench.isaacsim.release import dump, digest
from bench.paths import load_json
from scenes.workrooms.mechanical_assembly.tasks.gear_train_assembly.task import build_task_inputs
from scripts.maintenance.freeze_project_sources import verify_if_snapshot
from scripts.validation.gear_transmission_checks import axial_rotation
from scripts.validation.mechanical_convex_geometry import check_convex_geometry
from scripts.validation.validate_assembly_contacts import Observer, native_parameters
from scripts.validation.mechanical_tool_checks import contact_exchange_check
from scripts.validation.validate_scenario import seal_report_before_shutdown
from scripts.validation.validate_workrooms import plain


CONTRACT = {'schema': 'socket_wake_probe.v1', 'dt_s': .0005,
    'cases': [{'id': 'natural', 'wake_at_loaded_phase_start': False},
              {'id': 'explicit_wake', 'wake_at_loaded_phase_start': True}],
    'phases': [{'id': 'settle', 'duration_s': 1., 'target_rad_s': 0., 'loaded': False},
               {'id': 'forward', 'duration_s': .2, 'target_rad_s': math.radians(20.), 'loaded': True},
               {'id': 'coast', 'duration_s': .6, 'target_rad_s': 0., 'loaded': False},
               {'id': 'reverse', 'duration_s': .2, 'target_rad_s': -math.radians(20.), 'loaded': True},
               {'id': 'final_hold', 'duration_s': .3, 'target_rad_s': 0., 'loaded': False}],
    'coefficient_nm_s_per_rad': 2., 'maximum_torque_nm': .8,
    'maximum_linear_speed_m_s': .5, 'maximum_angular_speed_rad_s': 2.,
    'zero_wrench_api_calls_during_passive_phases': True,
    'scope': 'Only the original free socket and its original supply rest. The cases differ only by one explicit wake at each loaded phase start. No gears, controller changes or post-reset pose/velocity writes.'}


def wake_single_body(view):
    """Wake the one selected tool without a GPU index tensor.

    This native API expects host indices even for a GPU body view. Its
    documented no-index form selects every body in this checked singleton.
    """
    if view.count != 1:
        raise ValueError('Explicit wake must select exactly one tool body')
    view.wake_up()


def make_inputs():
    inputs = build_task_inputs(solver_type='PGS')
    inputs['scene']['objects'] = [s for s in inputs['scene']['objects'] if s['id'] == 'socket_crank']
    inputs['scene']['fixtures'] = [s for s in inputs['scene']['fixtures'] if s['id'] in ('robot_stand', 'socket_crank_rest')]
    inputs['initial_state']['objects'] = {n: s for n, s in inputs['initial_state'].get('objects', {}).items() if n == 'socket_crank'}
    inputs['scene']['scenario']['scope'] = CONTRACT['scope']
    inputs['wake_probe_contract'] = deepcopy(CONTRACT)
    return inputs


class WakeObserver(Observer):
    def __init__(self, env, output, case):
        import omni.usd
        from omni.physx import get_physx_simulation_interface
        from pxr import PhysicsSchemaTools
        self.interface = get_physx_simulation_interface()
        self.stage_id = omni.usd.get_context().get_stage_id()
        self.body_id = PhysicsSchemaTools.sdfPathToInt(env.scene_info['objects']['socket_crank']['path'])
        self.case = case
        self.phase = CONTRACT['phases'][0]
        self.previous_phase = None
        self.receipts = []
        self.receipt_stream = (output/'wrench_sleep_receipts.jsonl').open('x')
        super().__init__(env, output)

    def sleeping(self):
        try:
            return {'available': True, 'sleeping': bool(self.interface.is_sleeping(self.stage_id, self.body_id))}
        except Exception as exc:
            return {'available': False, 'error': type(exc).__name__+': '+str(exc)}

    def step(self, *args, **kwargs):
        from bench.simulation.isaacsim_backend.env import _np
        env = self.env
        view = env._object_views['socket_crank']
        before = self.sleeping()
        pose_before = _np(view.get_transforms()).reshape(7).tolist()
        velocity = _np(view.get_velocities()).reshape(6)
        if not np.isfinite(velocity).all() or np.linalg.norm(velocity[3:]) > CONTRACT['maximum_angular_speed_rad_s']:
            raise RuntimeError('Prestep tool speed limit')
        wake = bool(self.case['wake_at_loaded_phase_start'] and self.phase['loaded'] and self.previous_phase != self.phase['id'])
        if wake:
            wake_single_body(view)
        after_wake = self.sleeping()
        torque = float(np.clip(CONTRACT['coefficient_nm_s_per_rad']*(self.phase['target_rad_s']-velocity[5]),
            -CONTRACT['maximum_torque_nm'], CONTRACT['maximum_torque_nm'])) if self.phase['loaded'] else 0.
        view.apply_forces_and_torques_at_position(env._tensor(np.zeros((1, 3), dtype=np.float32)),
            env._tensor(np.array([[0., 0., torque]], dtype=np.float32)), None, env._one, True)
        after_force = self.sleeping()
        result = super().step(*args, **kwargs)
        row = self.rows[-1]
        receipt = {'native_step': row['native_step'], 'time_s': row['time_s'], 'phase': self.phase['id'],
            'body': 'socket_crank', 'world_frame': True, 'force_n': [0., 0., 0.], 'torque_nm': [0., 0., torque],
            'explicit_wake_called': wake, 'sleep_before': before, 'sleep_after_wake': after_wake,
            'sleep_after_force': after_force, 'sleep_after_step': self.sleeping(),
            'pose_before_xyzw': pose_before, 'velocity_before': velocity.tolist()}
        self.receipts.append(receipt)
        self.receipt_stream.write(json.dumps(receipt, allow_nan=False)+'\n')
        self.previous_phase = self.phase['id']
        v = row['velocities']['socket_crank']
        if np.linalg.norm(v[:3]) > CONTRACT['maximum_linear_speed_m_s'] or np.linalg.norm(v[3:]) > CONTRACT['maximum_angular_speed_rad_s']:
            raise RuntimeError('Poststep tool speed limit')
        return result

    def close(self):
        try:
            return super().close()
        finally:
            self.receipt_stream.close()


def run(args):
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[2]
    env = observer = None
    report = {'passed': False, 'task_success_claimed': False, 'physics_accepted': False,
        'model_calls': 0, 'post_reset_object_pose_velocity_writeback': False, 'cases': {}, 'scope': CONTRACT['scope']}
    try:
        imports = verify_if_snapshot(root, required_modules=('scripts.validation.probe_gear_tool_wake',), entrypoint_file=Path(__file__))
        if imports is None or digest(args.input) != args.input_sha256:
            raise ValueError('Complete sealed source and frozen input required')
        dump(out/'runtime_imports.json', imports)
        inputs = load_json(args.input)
        if inputs['wake_probe_contract'] != CONTRACT or inputs['scene']['isaacsim']['solver_type'] != 'PGS':
            raise ValueError('Undeclared wake comparison contract')
        if [s['id'] for s in inputs['scene']['objects']] != ['socket_crank']:
            raise ValueError('Only the real independent socket is permitted')
        if os.environ.get('ISAAC_BENCH_SINGLE_CUDA_DEVICE') != '1' or not os.environ.get('CUDA_VISIBLE_DEVICES', '').startswith('GPU-'):
            raise ValueError('Select one GPU by UUID')
        for path, expected in inputs['asset_files_sha256'].items():
            if digest(path) != expected:
                raise ValueError('Frozen asset changed: '+path)
        from bench.simulation.isaacsim_backend.env import FR3DexterousIsaacEnv
        env = FR3DexterousIsaacEnv(inputs['scene'], device='cuda:0', render_width=640, render_height=480)
        from pxr import PhysxSchema
        for case in CONTRACT['cases']:
            directory = out/case['id']
            directory.mkdir()
            env.reset(initial_state=inputs['initial_state'])
            if env.dt != CONTRACT['dt_s']:
                raise ValueError('Unexpected physics timestep')
            metadata = env.metadata()
            parameters = native_parameters(env)
            geometry = check_convex_geometry(env)
            body = PhysxSchema.PhysxRigidBodyAPI(env.stage.GetPrimAtPath(env.scene_info['objects']['socket_crank']['path']))
            scene_api = PhysxSchema.PhysxSceneAPI(env.stage.GetPrimAtPath(env.world.get_physics_context().prim_path))
            readback = {'body_sleep_threshold': body.GetSleepThresholdAttr().Get(),
                'body_stabilization_threshold': body.GetStabilizationThresholdAttr().Get(),
                'scene_stabilization_enabled': scene_api.GetEnableStabilizationAttr().Get(),
                'physics_solver': metadata['physics_solver']}
            dump(directory/'metadata.json', metadata)
            dump(directory/'parameters_geometry_sleep.json', plain({'parameters': parameters, 'geometry': geometry, 'sleep': readback}))
            if not parameters['passed'] or not geometry['passed'] or any(metadata['physics_solver'][k] != 'PGS' for k in ('profile_selection', 'usd_readback')):
                raise RuntimeError('Native parameter/geometry/solver evidence failed')
            observer = WakeObserver(env, directory, case)
            first_step = int(env.world.current_time_step_index)
            for phase in CONTRACT['phases']:
                observer.phase = phase
                for _ in range(round(phase['duration_s']/.02)):
                    observed = env.step({'joint_positions_rad': inputs['initial_state']['joint_positions_rad'], 'duration_s': .02})
                    observer.stream.flush()
                    observer.receipt_stream.flush()
                    if observed['terminated'] or not observed['metrics']['state_finite']:
                        raise RuntimeError('Native state failed')
                print(json.dumps({'case': case['id'], 'phase': phase['id'], 'sim_time_s': observed['sim_time_s']}), flush=True)
            rows, receipts = observer.rows, observer.receipts
            exchange = observer.close()
            observer = None
            phase_results = {}
            for phase in CONTRACT['phases']:
                indices = [i for i, r in enumerate(receipts) if r['phase'] == phase['id']]
                phase_results[phase['id']] = {'samples': len(indices),
                    'yaw_change_degrees': math.degrees(axial_rotation(rows[indices[-1]]['poses_xyzw']['socket_crank'])-axial_rotation(receipts[indices[0]]['pose_before_xyzw'])),
                    'before_sleep_true_samples': sum(receipts[i]['sleep_before'].get('sleeping') is True for i in indices),
                    'after_force_sleep_true_samples': sum(receipts[i]['sleep_after_force'].get('sleeping') is True for i in indices),
                    'after_step_sleep_true_samples': sum(receipts[i]['sleep_after_step'].get('sleeping') is True for i in indices)}
            report['cases'][case['id']] = {'phases': phase_results, 'native_samples': len(rows),
                'sleep_queries_available': all(r[k]['available'] for r in receipts for k in ('sleep_before', 'sleep_after_wake', 'sleep_after_force', 'sleep_after_step')),
                'explicit_wake_calls': sum(r['explicit_wake_called'] for r in receipts),
                'robot_interference': any(r['robot_interference'] for r in rows),
                'maximum_penetration_m': max(r['maximum_penetration_m'] for r in rows),
                'contact_exchange': contact_exchange_check(exchange, int(env.world.current_time_step_index)-first_step, len(rows)),
                'sleep_readback': readback}
            dump(directory/'result.json', plain(report['cases'][case['id']]))
            del rows, receipts
        report['passed'] = len(report['cases']) == 2 and all(c['native_samples'] == 4600 and c['contact_exchange']['passed'] and not c['robot_interference']
            and c['maximum_penetration_m'] <= .0005 for c in report['cases'].values())
        report['interpretation_limit'] = 'Completion only: absent sleep, unavailable sleeping query, or unchanged response leaves the original gear-stall cause unknown.'
    except BaseException as exc:
        report['error'] = {'type': type(exc).__name__, 'detail': str(exc), 'traceback': traceback.format_exc()}
    finally:
        if observer is not None:
            observer.close()
    report['artifact_sha256'] = {str(p.relative_to(out)): digest(p) for p in out.rglob('*') if p.is_file()}
    return seal_report_before_shutdown(out, plain(report), env)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--input-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    raise SystemExit(0 if run(args)['passed'] else 1)


if __name__ == '__main__':
    main()
