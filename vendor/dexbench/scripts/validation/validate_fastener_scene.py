"""Isolated L4 contact-load diagnostic in the frozen mechanical task scene.

The robot holds its ordinary ready joint targets. An explicitly privileged
fixture-independent load is applied to the tool COM only; the bolt is passive.
No pose/velocity setter, lead-following z target or artificial joint is used.
This is scene contact-function evidence, never a robot manipulation score.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import traceback
import numpy as np

from bench.geometry import quaternion_matrix
from bench.isaacsim.release import dump, task_inputs
from scripts.maintenance.freeze_project_sources import verify_if_snapshot
from scripts.validation.validate_mechanical_tool_motion import ToolObserver
from scripts.validation.validate_mechanical_dynamics import FixtureMonitor, native_task_parameters
from scripts.validation.mechanical_checks import contact_filter_coverage
from scripts.validation.mechanical_tool_checks import contact_exchange_check
from scripts.validation.mechanical_native_geometry import collect_task_geometry
from scripts.validation.mechanical_fastener_passive_checks import (task_contact_penetration,
    passive_phase_window)
from scripts.validation.mechanical_fastener_contact_reset import (prepare_contact_reset,
    passive_support_pairs)
from scripts.validation.mechanical_fastener_motion_checks import full_motion_window
from scripts.validation.validate_workrooms import stage_inventory, plain
from scripts.validation.validate_scenario import seal_report_before_shutdown

LOAD = {'axial_downward_net_n': 3., 'lateral_stiffness_n_m': 400.,
        'lateral_damping_ns_m': 8., 'tilt_stiffness_nm_rad': 2.,
        'tilt_damping_nms_rad': .1, 'spin_speed_rad_s': .5,
        'spin_velocity_gain_nms': .06, 'spin_integral_gain_nm': .08,
        'spin_integral_cap_nm': .08, 'spin_torque_cap_nm': .12,
        'force_cap_n': 20., 'torque_cap_nm': 1., 'ramp_s': .3}


def make_load_profile(spin_speed_rad_s=.5):
    """Select a declared engineering drive condition without tuning its gains."""
    if spin_speed_rad_s not in (.35, .5):
        raise ValueError('Engineering spin speed must be 0.35 or 0.5 rad/s')
    profile = deepcopy(LOAD)
    profile['spin_speed_rad_s'] = float(spin_speed_rad_s)
    return profile


def diagnostic_environment_class(base, solver_type, output):
    """Change only the scene algorithm before its first native initialization."""
    class DiagnosticEnvironment(base):
        def _initialize(self):
            # SimulationApp has started, so native USD modules are now valid.
            from bench.simulation.isaacsim_backend import scene as scene_module
            from pxr import PhysxSchema, UsdPhysics
            original = scene_module.build_scene
            calls = []
            def build_scene(stage, *args, **kwargs):
                result = original(stage, *args, **kwargs)
                scenes = [p for p in stage.Traverse() if p.IsA(UsdPhysics.Scene)]
                if len(scenes) != 1 or calls:
                    raise RuntimeError('Expected exactly one scene authoring before native reset')
                prim = scenes[0]
                values = lambda: {a.GetName(): a.Get() for a in prim.GetAttributes()}
                before = values(); api = PhysxSchema.PhysxSceneAPI(prim)
                previous = api.GetSolverTypeAttr().Get()
                if previous != self.solver_type:
                    raise RuntimeError('Unexpected parent solver algorithm')
                if solver_type is not None:
                    api.GetSolverTypeAttr().Set(solver_type)
                after = values(); changed = sorted(k for k in set(before)|set(after)
                    if before.get(k) != after.get(k))
                expected = ['physxScene:solverType'] if solver_type not in (None, previous) else []
                if changed != expected or api.GetSolverTypeAttr().Get() != (solver_type or previous):
                    raise RuntimeError('Solver diagnostic changed unexpected scene properties')
                calls.append(True)
                self.diagnostic_solver_record = {'passed': True, 'scene_path': str(prim.GetPath()),
                    'source_solver': previous, 'requested_override': solver_type,
                    'effective_authored_solver': api.GetSolverTypeAttr().Get(),
                    'changed_scene_attributes': changed,
                    'all_other_scene_attributes_unchanged': True,
                    'applied_before_first_world_reset': True,
                    'readback_scope': 'Exact PhysxScene USD algorithm selected before native scene creation; no tensor solver-enum getter is available.'}
                dump(output/'solver_configuration.json', self.diagnostic_solver_record)
                return result
            scene_module.build_scene = build_scene
            try:
                super()._initialize()
                if calls != [True]:
                    raise RuntimeError('Solver authoring hook did not run exactly once')
            finally:
                scene_module.build_scene = original
    return DiagnosticEnvironment


def release_solver_configuration(scene, metadata):
    """Verify the ordinary release-owned selection without authoring an override."""
    expected = scene['isaacsim'].get('solver_type', 'TGS')
    actual = metadata.get('physics_solver', {})
    return {'passed': expected in ('TGS', 'PGS')
            and actual.get('profile_selection') == expected and actual.get('usd_readback') == expected,
        'selection_path': 'ordinary release scene.isaacsim.solver_type via native backend',
        'expected_release_solver': expected, 'metadata_readback': actual,
        'diagnostic_solver_hook_used': False,
        'readback_scope': 'PhysicsScene USD selection before native reset, not a native tensor solver-enum query.'}


def capped(vector, maximum):
    vector = np.asarray(vector, float)
    return vector*min(1., maximum/max(float(np.linalg.norm(vector)), 1e-12))


def tool_load(pose, velocity, com_local, xy_reference, mass, time, dt, integral, *, load_profile=None):
    """Real wrench servo; axial force has no position or screw-lead feedback."""
    profile = LOAD if load_profile is None else load_profile
    rot = quaternion_matrix(np.asarray(pose)[[6, 3, 4, 5]])
    ramp = min(1., max(0., time/profile['ramp_s']))
    tip_velocity = velocity[:3]+np.cross(velocity[3:], -rot@com_local)
    force = np.array([0., 0., (mass*9.81-profile['axial_downward_net_n'])*ramp])
    force[:2] = (-profile['lateral_stiffness_n_m']*(pose[:2]-xy_reference)
                 -profile['lateral_damping_ns_m']*tip_velocity[:2])*ramp
    torque = (profile['tilt_stiffness_nm_rad']*np.cross(rot[:, 2], [0., 0., 1.])
              -profile['tilt_damping_nms_rad']*np.r_[velocity[3:5], 0.])*ramp
    target = -profile['spin_speed_rad_s']*ramp
    error = target-velocity[5]
    proposed = float(np.clip(integral+profile['spin_integral_gain_nm']*error*dt,
                             -profile['spin_integral_cap_nm'], profile['spin_integral_cap_nm']))
    raw = profile['spin_velocity_gain_nms']*error+proposed
    if abs(raw) < profile['spin_torque_cap_nm'] or raw*error < 0:
        integral = proposed
    torque[2] = np.clip(profile['spin_velocity_gain_nms']*error+integral,
                        -profile['spin_torque_cap_nm'], profile['spin_torque_cap_nm'])
    return capped(force, profile['force_cap_n']), capped(torque, profile['torque_cap_nm']), integral, target


class LoadedObserver(ToolObserver):
    def __init__(self, env, output, *, enabled, xy_offset_m, load_profile):
        super().__init__(env, output)
        state = env.capture_state()
        self.motion_anchor = {
            'native_step_index': int(env.world.current_time_step_index),
            'simulation_time_s': float(env.time_s),
            'object_poses_xyzw': {name: state['object_q'][i].tolist()
                                  for name, i in self.indices.items()},
        }
        dump(output/'motion_anchor.json', self.motion_anchor)
        from bench.simulation.isaacsim_backend.env import _np
        import torch
        self.view = env._sim_view.create_rigid_body_view(self.tool_path)
        self.com_local = _np(self.view.get_coms()).reshape(7)[:3]
        self.mass = float(_np(self.view.get_masses()).reshape(-1)[0])
        self.reference = _np(self.view.get_transforms())[0, :2]+[xy_offset_m, 0.]
        self.integral = 0.; self.load_time = 0.; self.enabled = enabled
        self.load_profile = deepcopy(load_profile)
        self.one = torch.tensor([0], dtype=torch.int32, device=env.device)
        self.load_stream = (output/'applied_tool_wrenches.jsonl').open('x')

    def step(self, *args, **kwargs):
        from bench.simulation.isaacsim_backend.env import _np
        import torch
        force, torque, target = np.zeros(3), np.zeros(3), 0.
        pose = _np(self.view.get_transforms())[0]
        com = pose[:3]+quaternion_matrix(pose[[6, 3, 4, 5]])@self.com_local
        if self.enabled and self.phase == 'loaded_turn':
            velocity = _np(self.view.get_velocities())[0]
            force, torque, self.integral, target = tool_load(pose, velocity, self.com_local,
                self.reference, self.mass, self.load_time, self.env.dt, self.integral,
                load_profile=self.load_profile)
            self.load_time += self.env.dt
            tensor = lambda value: torch.as_tensor(np.asarray(value)[None], dtype=torch.float32, device=self.env.device)
            self.view.apply_forces_and_torques_at_position(tensor(force), tensor(torque), tensor(com), self.one, True)
        self.load_stream.write(json.dumps({'phase': self.phase, 'native_step_before': int(self.env.world.current_time_step_index),
            'force_world_n': force.tolist(), 'torque_world_nm': torque.tolist(), 'application_point_world_m': com.tolist(),
            'target_spin_rad_s': target, 'declared_spin_speed_rad_s': self.load_profile['spin_speed_rad_s'],
            'integral_torque_nm': self.integral}, separators=(',', ':'))+'\n')
        return super().step(*args, **kwargs)

    def close(self):
        try:
            return super().close()
        finally:
            self.load_stream.close()


def assess_loaded_lead(rows, paths, dt):
    selected = [r for r in rows if r['phase'] == 'loaded_turn']
    if not selected:
        return {'passed': False, 'reason': 'no_load_phase', 'dynamic_lead_verified': False}
    poses = np.asarray([r['object_poses_xyzw']['bolt'] for r in selected])
    rotations = np.asarray([quaternion_matrix(p[[6, 3, 4, 5]]) for p in poses])
    angle = -np.unwrap(np.arctan2(rotations[:, 1, 0], rotations[:, 0, 0]))
    turn = angle-angle[0]; descent = poses[0, 2]-poses[:, 2]
    # Ignore initial contact seating and measure net motion from first 5 degrees.
    starts = np.flatnonzero(turn >= math.radians(5.))
    start = int(starts[0]) if len(starts) else 0
    measured_turn = turn[start:]-turn[start]
    measured_descent = descent[start:]-descent[start]
    expected = measured_turn*.003/(2*math.pi)
    residual = measured_descent-expected
    tilt = np.arccos(np.clip(rotations[:, 2, 2], -1, 1))
    offset = np.linalg.norm(poses[:, :2]-poses[0, :2], axis=1)
    contacts = []
    for row in selected:
        active = {tuple(sorted((c['actor0'], c['actor1']))) for c in row['raw_contacts']
                  if c['impulse_magnitude_sum_ns'] > 1e-8}
        contacts.append(tuple(sorted((paths['screwdriver'], paths['bolt']))) in active
                        and tuple(sorted((paths['bolt'], paths['fixture']))) in active)
    fraction = sum(contacts)/len(contacts)
    enough = measured_turn[-1] >= math.radians(15.)
    lead = float(np.polyfit(measured_turn, measured_descent, 1)[0]*2*math.pi) if enough else None
    checks = {'at_least_15_degrees_after_seating': bool(enough),
              'lead_slope_within_five_percent': lead is not None and abs(lead/.003-1) <= .05,
              'maximum_lead_residual_below_15um': float(np.max(abs(residual))) <= 15e-6,
              'positive_contact_chain_fraction_above_99_percent': fraction >= .99,
              'tilt_below_point_one_degree': float(np.max(tilt)) <= math.radians(.1),
              'axis_drift_below_50um': float(np.max(offset)) <= 50e-6}
    return {'passed': all(checks.values()), 'dynamic_lead_verified': all(checks.values()), 'checks': checks,
        'bolt_total_turn_degrees': math.degrees(float(turn[-1])),
        'bolt_total_axial_advance_m': float(descent[-1]), 'measured_lead_m_per_revolution': lead,
        'analysis_start_turn_degrees': math.degrees(float(turn[start])),
        'analysis_turn_degrees': math.degrees(float(measured_turn[-1])),
        'maximum_lead_residual_m': float(np.max(abs(residual))), 'final_lead_residual_m': float(residual[-1]),
        'maximum_bolt_tilt_degrees': math.degrees(float(np.max(tilt))),
        'maximum_bolt_axis_drift_m': float(np.max(offset)), 'contact_chain_fraction': fraction,
        'native_sample_count': len(selected), 'physics_dt_s': dt,
        'scope': 'Contact-only response to a bounded external tool wrench; no robot grasp or task score.'}


def run(args):
    root = Path(__file__).resolve().parents[2]; out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    env = observer = None; error = None; checks = {}; samples = 0; geometry = None
    contact_reset = None
    started = datetime.now(timezone.utc).isoformat()
    try:
        load_profile = make_load_profile(args.spin_speed_rad_s)
        from bench.simulation.isaacsim_backend.env import FR3DexterousIsaacEnv
        modules = ('scripts.validation.validate_fastener_scene', 'scripts.validation.mechanical_native_geometry',
                   'scripts.validation.mechanical_fastener_contact_reset',
                   'bench.simulation.isaacsim_backend.env')
        verified = verify_if_snapshot(root, required_modules=modules, entrypoint_file=Path(__file__))
        if verified is None:
            raise ValueError('Complete frozen source required')
        dump(out/'runtime_imports.json', verified)
        uuid = os.environ.get('CUDA_VISIBLE_DEVICES', '')
        if uuid != 'GPU-66d42486-80df-afc7-bf75-e5dd67059f99' or os.environ.get('ISAAC_BENCH_SINGLE_CUDA_DEVICE') != '1':
            raise ValueError('This reviewed diagnostic is pinned to GPU 9B UUID')
        _, scene, task, _ = task_inputs(args.release, 'fastener_assembly')
        initial = deepcopy(task['initial_state'])
        original_scene = deepcopy(scene)
        scene, task, contact_reset = prepare_contact_reset(
            scene, task, getattr(args, 'contact_reset_release', None))
        if args.physics_dt is not None:
            scene['isaacsim']['physics_dt_s'] = args.physics_dt
        if contact_reset is not None:
            dump(out/'contact_reset.json', contact_reset)
            dump(out/'diagnostic_task.json', task)
        dump(out/'scene.json', scene); dump(out/'initial_state.json', initial)
        dump(out/'plan.json', {'schema': 'fastener_scene_load_plan.v1', 'release': str(args.release),
            'source_root': str(root), 'gpu_uuid': uuid, 'driver_override': os.environ.get('ISAAC_ALLOW_UNVALIDATED_DRIVER'),
            'load_enabled': args.load, 'geometry_only': args.geometry_only, 'load_profile': load_profile, 'lateral_tool_reference_offset_m': args.lateral_offset,
            'settle_s': 0. if args.geometry_only else .6,
            'loaded_turn_s': args.turn_duration if args.load else 0.,
            'final_passive_s': 0. if args.geometry_only else args.final_passive_duration,
            'passive_and_penetration_threshold_source': 'frozen task.json evaluation',
            'physics_dt_override_s': args.physics_dt, 'original_isaacsim_options': original_scene['isaacsim'],
            'scene_solver_type_override': args.solver_type,
            'contact_reset': contact_reset,
            'task_qualified': False, 'robot_manipulation_qualified': False, 'model_calls': 0,
            'object_pose_velocity_writes_after_reset': 0, 'external_load_target': 'screwdriver COM only',
            'bolt_actuation': None, 'lead_following_position_target': None})
        environment = (diagnostic_environment_class(FR3DexterousIsaacEnv, args.solver_type, out)
                       if args.solver_type is not None else FR3DexterousIsaacEnv)
        env = environment(scene, device='cuda:0', render_width=320, render_height=240)
        env.reset(seed=0, initial_state=initial)
        metadata = env.metadata(); dump(out/'metadata_initial.json', metadata)
        solver_record = (env.diagnostic_solver_record if args.solver_type is not None
                         else release_solver_configuration(scene, metadata))
        if args.solver_type is None:
            dump(out/'solver_configuration.json', solver_record)
        checks.update(scene_solver_configuration=solver_record,
                      native_task_parameters=native_task_parameters(env), contact_filter_coverage=contact_filter_coverage(env),
                      stage_inventory=stage_inventory(env))
        geometry = collect_task_geometry(env, out)
        dump(out/'native_geometry.json', geometry); checks['native_geometry'] = {'passed': geometry['passed'], 'issues': geometry['issues']}
        if not args.geometry_only:
            fixture = FixtureMonitor(env); env._read_rigid_contacts = fixture.wrap(env._read_rigid_contacts)
            observer = LoadedObserver(env, out, enabled=args.load, xy_offset_m=args.lateral_offset,
                                      load_profile=load_profile)
            phases = [('settle', .6)]+([('loaded_turn', args.turn_duration)] if args.load else [])+[
                ('final_passive', args.final_passive_duration)]
            for phase, duration in phases:
                observer.phase = phase
                for i in range(round(duration/.02)):
                    env.step({'joint_positions_rad': initial['joint_positions_rad'], 'duration_s': .02})
                    if (i+1) % 25 == 0:
                        print(json.dumps({'event': 'fastener_scene_progress', 'phase': phase, 'phase_s': (i+1)*.02,
                                          'native_samples': len(observer.rows)}), flush=True)
            samples = len(observer.rows)
            checks['full_object_motion'] = full_motion_window(
                observer.rows, env.dt, task['evaluation'],
                expected_samples=sum(round(duration/env.dt) for _, duration in phases),
                anchor=observer.motion_anchor)
            paths = {n: env.scene_info['objects'][n]['path'] for n in ('screwdriver', 'bolt')}
            paths['fixture'] = env.scene_info['fixtures']['threaded_fixture']['path']
            passive_paths = dict(paths)
            for config_key, path_key in (('supply_rest_actor', 'bolt_supply_rest'),
                                         ('tool_rest_actor', 'tool_supply_rest')):
                actor_path = task['evaluation'].get(config_key)
                if actor_path:
                    passive_paths[path_key] = actor_path
            supports = passive_support_pairs(scene, task['evaluation'], paths,
                                            contact_reset=contact_reset is not None)
            if ('bolt_table' in supports
                    and supports['bolt_table'][1] not in env.scene_info['support_paths']):
                raise ValueError('Canonical bolt table support missing from native inventory')
            dump(out/'passive_support_pairs.json', supports)
            if args.load:
                checks['loaded_thread_lead'] = assess_loaded_lead(observer.rows, paths, env.dt)
            checks['all_task_contact_penetration'] = task_contact_penetration(observer.rows, paths, env.dt, task['evaluation'])
            checks['passive_settlement'] = passive_phase_window(observer.rows, 'settle', passive_paths, env.dt, task['evaluation'], support_pairs=supports)
            checks['passive_release'] = passive_phase_window(observer.rows, 'final_passive', passive_paths, env.dt, task['evaluation'], support_pairs=supports)
            checks['fixture_every_substep'] = fixture.result(samples)
            checks['robot_noninterference'] = {'passed': not any(r['robot_interference'] for r in observer.rows),
                'interference_samples': sum(bool(r['robot_interference']) for r in observer.rows)}
            exchange = observer.close(); checks['contact_exchange'] = contact_exchange_check(exchange, samples, samples)
            dump(out/'contact_exchange.json', exchange)
        dump(out/'metadata_final.json', env.metadata())
        dump(out/'runtime_imports_final.json', verify_if_snapshot(root, required_modules=modules, entrypoint_file=Path(__file__)))
    except BaseException as exc:
        error = {'type': type(exc).__name__, 'detail': str(exc), 'traceback': traceback.format_exc()}
        print(traceback.format_exc(), flush=True)
    finally:
        if observer is not None and not observer.closed:
            try:
                dump(out/'contact_exchange_on_error.json', observer.close())
            except BaseException as exc:
                checks['observer_close_error'] = {'passed': False, 'error': str(exc)}
    report = {'schema': 'fastener_scene_native_contact_validation.v1', 'started_utc': started,
        'finished_utc': datetime.now(timezone.utc).isoformat(), 'passed': error is None and bool(checks) and all(v['passed'] for v in checks.values()),
        'checks': plain(checks), 'error': error, 'native_samples': samples, 'model_calls': 0,
        'rollout_performed': not args.geometry_only,
        'dynamic_lead_verified': checks.get('loaded_thread_lead', {}).get('passed', False),
        'passive_release_verified': checks.get('passive_release', {}).get('passed', False),
        'reset_scope': 'engaged_contact_diagnostic' if contact_reset is not None else 'release_reset',
        'task_success_claimed': False, 'robot_manipulation_qualified': False,
        'scope': 'Scene geometry, native inertia/COM and isolated bounded tool-load contact response.',
        'artifacts_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in out.iterdir() if p.is_file()}}
    return seal_report_before_shutdown(out, report, env)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--release', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--contact-reset-release', type=Path,
                        help='Explicit pinned native18/native19 engaged reset for scene-only contact diagnostics; never task qualification')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--load', action='store_true')
    mode.add_argument('--geometry-only', action='store_true', help='Only live cooking/inertia/SDF queries after reset; no rollout or dynamic qualification')
    parser.add_argument('--physics-dt', type=float, choices=(.0005, .00025))
    parser.add_argument('--lateral-offset', type=float, choices=(0., -.0002, .0002), default=0.)
    parser.add_argument('--turn-duration', type=float, default=3.)
    parser.add_argument('--spin-speed-rad-s', type=float, choices=(.35, .5), default=.5,
                        help='Declared isolated tool-load target; other gains/caps/ramp and task thresholds remain unchanged')
    parser.add_argument('--final-passive-duration', type=float, default=.8,
                        help='Terminal frozen 0.5s stability window after passive decay; shorter diagnostic records fail that gate')
    parser.add_argument('--solver-type', choices=('TGS', 'PGS'),
                        help='Explicit isolated scene-algorithm sensitivity; original default remains TGS')
    args = parser.parse_args()
    if not np.isfinite(args.turn_duration) or not 1 <= args.turn_duration <= 6 or abs(args.turn_duration/.02-round(args.turn_duration/.02)) > 1e-8:
        raise ValueError('Turn duration must be 1 to 6 seconds on 20ms boundaries')
    if (not np.isfinite(args.final_passive_duration) or not .3 <= args.final_passive_duration <= 2
            or abs(args.final_passive_duration/.02-round(args.final_passive_duration/.02)) > 1e-8):
        raise ValueError('Final passive duration must be 0.3 to 2 seconds on 20ms boundaries')
    if not run(args)['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
