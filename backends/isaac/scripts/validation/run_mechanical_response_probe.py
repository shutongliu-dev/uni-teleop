"""Short privileged scene probes: tool -> part and positioned hand -> tool.

Reset placement is privileged. After reset, parts receive only native contact;
the two tool probes apply bounded wrenches to the independent tool each physics
substep. The hand probe uses ordinary joint drives with unchanged gains.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import traceback

import numpy as np

from bench.geometry import quaternion_matrix
from bench.isaacsim.release import digest, dump, task_inputs
from scripts.maintenance.freeze_project_sources import verify_if_snapshot
from scripts.validation.validate_fastener_scene import make_load_profile, tool_load
from scripts.validation.validate_scenario import seal_report_before_shutdown


def reduce_contacts(rows):
    """Keep directed normal vectors and zero-load touching, aggregate one window."""
    pairs = {}
    for c in rows:
        key = (c['actor0'], c['actor1'])
        p = pairs.setdefault(key, dict(actor0=key[0], actor1=key[1], point_count=0,
            normal_impulse_sum_ns=0., impulse_vector_ns=np.zeros(3),
            min_separation_m=float(c['min_separation_m']), active_times=set(), touching_times=set()))
        p['point_count'] += c['point_count']
        p['normal_impulse_sum_ns'] += c['impulse_magnitude_sum_ns']
        p['impulse_vector_ns'] += np.asarray(c['impulse_vector_ns'], float)
        p['min_separation_m'] = min(p['min_separation_m'], c['min_separation_m'])
        p['touching_times'].add(c['time_s'])
        if c['impulse_magnitude_sum_ns'] > 1.e-8:
            p['active_times'].add(c['time_s'])
    result = []
    for p in pairs.values():
        p['impulse_vector_ns'] = p['impulse_vector_ns'].tolist()
        p['active_time_s'] = sorted(p.pop('active_times'))
        p['active_samples'] = len(p['active_time_s'])
        p['touching_samples'] = len(p.pop('touching_times'))
        result.append(p)
    return result


def build_plan(case, scene, initial):
    if case == 'hand_tool':
        from scripts.validation.mechanical_hand_response_plan import build_hand_response_plan
        return build_hand_response_plan(scene, initial, turn_degrees=6.)
    initial = deepcopy(initial)
    if case == 'gear_tool':
        from scenes.workrooms.mechanical_assembly.tasks.gear_train_assembly.task import contact_probe_reset
        initial['objects'] = deepcopy(contact_probe_reset()['object_overrides'])
        phases = [('settle', .5, 0.), ('forward', 3., math.radians(20.)),
                  ('coast', .5, 0.), ('reverse', 3., -math.radians(20.)), ('final_rest', .5, 0.)]
    else:
        phases = [('settle', .6, 0.), ('loaded_turn', 3., -.35), ('final_rest', .8, 0.)]
    return {'initial_state': initial, 'control_period_s': .02,
            'phases': [dict(name=n, duration_s=t, target_speed_rad_s=w) for n, t, w in phases]}


class SubstepProbe:
    """Apply loads before every native step; retain small actual-load receipts."""

    def __init__(self, env, case, output):
        from bench.simulation.isaacsim_backend.env import _np
        self.env, self.case = env, case
        self.original_step = env.world.step
        self.phase = {'name': 'reset', 'target_speed_rad_s': 0.}
        self.profile = make_load_profile(.35)
        self.integral = self.load_time = 0.
        self.total_substeps = 0
        self.stream = (output/'substeps.jsonl').open('x')
        self.window = []
        self.tool = 'socket_crank' if case == 'gear_tool' else 'screwdriver'
        self.view = env._object_views[self.tool]
        self.com_local = _np(self.view.get_coms()).reshape(7)[:3]
        self.mass = float(_np(self.view.get_masses()).reshape(-1)[0])
        self.reference = _np(self.view.get_transforms())[0, :2].copy()
        env.world.step = self.step

    def step(self, *args, **kwargs):
        from bench.simulation.isaacsim_backend.env import _np
        env = self.env
        force, torque, target, com = np.zeros(3), np.zeros(3), 0., None
        loaded = False
        if self.case == 'fastener_tool' and self.phase['name'] == 'loaded_turn':
            pose = _np(self.view.get_transforms())[0]
            velocity = _np(self.view.get_velocities())[0]
            force, torque, self.integral, target = tool_load(pose, velocity, self.com_local,
                self.reference, self.mass, self.load_time, env.dt, self.integral,
                load_profile=self.profile)
            com = pose[:3] + quaternion_matrix(pose[[6, 3, 4, 5]]) @ self.com_local
            self.load_time += env.dt
            loaded = True
        elif self.case == 'gear_tool' and self.phase['name'] in ('forward', 'reverse'):
            velocity = _np(self.view.get_velocities())[0]
            target = self.phase['target_speed_rad_s']
            torque[2] = np.clip(2. * (target - velocity[5]), -.8, .8)
            loaded = True
        if loaded:
            if not np.isfinite(np.r_[force, torque]).all():
                raise RuntimeError('Nonfinite tool load; not applied')
            tensor = lambda v: env._tensor(np.asarray([v], dtype=np.float32))
            self.view.apply_forces_and_torques_at_position(tensor(force), tensor(torque),
                None if com is None else tensor(com), env._one, True)
        result = self.original_step(*args, **kwargs)
        self.total_substeps += 1
        # Record native maxima as well as 20 ms poses, so a transient is not hidden.
        velocities = {name: _np(v.get_velocities()).reshape(6)
                      for name, v in env._object_views.items()}
        finite = all(np.isfinite(v).all() for v in velocities.values())
        if not finite:
            np.savez(self.stream.name + '.nonfinite.npz', **velocities)
            self.stream.write(json.dumps(dict(native_step=int(env.world.current_time_step_index),
                phase=self.phase['name'], error='nonfinite_object_velocity',
                raw_values_file=Path(self.stream.name).name+'.nonfinite.npz'))+'\n')
            self.stream.flush()
            raise RuntimeError('Nonfinite object velocity; raw values retained')
        row = dict(native_step=int(env.world.current_time_step_index),
            time_s=float(env.world.current_time), phase=self.phase['name'], body=self.tool,
            wrench_applied=loaded, force_world_n=force.tolist(), torque_world_nm=torque.tolist(),
            application_point_world_m=None if com is None else com.tolist(),
            target_speed_rad_s=target, finite_object_velocities=finite,
            maximum_object_linear_speed_m_s=max(float(np.linalg.norm(v[:3])) for v in velocities.values()),
            maximum_object_angular_speed_rad_s=max(float(np.linalg.norm(v[3:])) for v in velocities.values()))
        self.stream.write(json.dumps(row, separators=(',', ':'), allow_nan=False)+'\n')
        self.window.append(row)
        if not finite or row['maximum_object_linear_speed_m_s'] > .5 or row['maximum_object_angular_speed_rad_s'] > 5.:
            self.stream.flush()
            raise RuntimeError('Engineering motion safety bound exceeded; substep retained')
        return result

    def receipt(self):
        rows, self.window = self.window, []
        if not rows:
            return {'native_steps': 0, 'wrench_applied': False}
        torque = np.array([r['torque_world_nm'] for r in rows])
        force = np.array([r['force_world_n'] for r in rows])
        return dict(native_steps=len(rows), wrench_applied=any(r['wrench_applied'] for r in rows),
            target_speed_rad_s=rows[-1]['target_speed_rad_s'],
            applied_torque_world_nm=torque.mean(axis=0).tolist(),
            applied_force_world_n=force.mean(axis=0).tolist(),
            torque_min_world_nm=torque.min(axis=0).tolist(), torque_max_world_nm=torque.max(axis=0).tolist(),
            maximum_native_linear_speed_m_s=max(r['maximum_object_linear_speed_m_s'] for r in rows),
            maximum_native_angular_speed_rad_s=max(r['maximum_object_angular_speed_rad_s'] for r in rows))

    def close(self):
        self.env.world.step = self.original_step
        self.stream.close()


def trace_row(obs, phase, probe):
    objects = {}
    for name, p in obs['objects'].items():
        q = p['quaternion_wxyz']
        objects[name] = dict(pose_xyzw=[*p['position_m'], *q[1:], q[0]],
                            velocity=[*p['linear_velocity_m_s'], *p['angular_velocity_rad_s']])
    return dict(phase=phase, time_s=obs['time_s'], native_step=int(probe.env.world.current_time_step_index),
        objects=objects, joint_positions_rad=obs['joint_positions_rad'],
        joint_targets_rad=obs['robot']['joint_targets_rad'], wrist=obs['robot']['right_hand_flange'],
        drive=probe.receipt(), contacts=reduce_contacts(obs['contacts']['rigid']),
        contact_sampling=obs['contacts']['sampling'], contact_errors=obs['contacts']['errors'],
        metrics=obs['metrics'])


def run(args):
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    env = probe = None
    report = dict(schema='mechanical_privileged_response.v1', case=args.case, passed=False,
        passed_semantics='Execution and evidence complete; physical response is assessed separately.',
        task_success_claimed=False, agent_test=False, model_calls=0,
        object_state_writeback_after_reset=False, controller_gains_modified=False,
        quantitative_contact_force_qualified=False,
        contact_scope='Directed native normal impulses, uncalibrated as reaction force/torque.',
        started_utc=datetime.now(timezone.utc).isoformat())
    try:
        from bench.simulation.isaacsim_backend.env import FR3DexterousIsaacEnv
        root = Path(__file__).resolve().parents[2]
        verified = verify_if_snapshot(root, required_modules=(
            'scripts.validation.run_mechanical_response_probe', 'bench.simulation.isaacsim_backend.env'),
            entrypoint_file=Path(__file__))
        if verified is None:
            raise ValueError('Launch from a complete frozen source snapshot')
        dump(output/'runtime_imports.json', verified)
        if os.environ.get('ISAAC_BENCH_SINGLE_CUDA_DEVICE') != '1' or not os.environ.get('CUDA_VISIBLE_DEVICES', '').startswith('GPU-'):
            raise ValueError('Select one GPU by UUID')
        task_id = 'gear_train_assembly' if args.case == 'gear_tool' else 'fastener_assembly'
        _, scene, task, initial = task_inputs(args.release, task_id)
        plan = build_plan(args.case, scene, initial)
        plan.update(case=args.case, release=str(args.release.resolve()),
            release_manifest_sha256=digest(args.release/'release_manifest.json'),
            source_root=str(root), cuda_visible_devices=os.environ['CUDA_VISIBLE_DEVICES'],
            driver_override=os.environ.get('ISAAC_ALLOW_UNVALIDATED_DRIVER'),
            tool_load_profile=make_load_profile(.35) if args.case == 'fastener_tool' else None,
            gear_load_profile={'gain_nm_s_per_rad': 2., 'torque_cap_nm': .8} if args.case == 'gear_tool' else None,
            object_safety_bounds={'linear_speed_m_s': .5, 'angular_speed_rad_s': 5.})
        dump(output/'scene.json', scene)
        dump(output/'task.json', task)
        dump(output/'initial_state.json', plan['initial_state'])
        dump(output/'plan.json', plan)
        env = FR3DexterousIsaacEnv(scene, device='cuda:0', render_width=960, render_height=720)
        obs = env.reset(initial_state=plan['initial_state'])
        dump(output/'metadata_initial.json', env.metadata())
        from PIL import Image
        camera = 'task' if 'task' in env.camera_ids else env.camera_ids[0]
        Image.fromarray(env.render(camera, width=960, height=720)).save(output/'initial.png')
        probe = SubstepProbe(env, args.case, output)
        q = deepcopy(obs['robot']['joint_targets_rad'])
        samples = 0
        with (output/'response.jsonl').open('x') as stream:
            stream.write(json.dumps(trace_row(obs, 'reset', probe), allow_nan=False)+'\n')
            for phase in plan['phases']:
                probe.phase = phase
                start = deepcopy(q)
                end = phase.get('joint_positions_rad', start)
                count = round(phase['duration_s']/plan['control_period_s'])
                if count < 1 or not math.isclose(count*plan['control_period_s'], phase['duration_s'], abs_tol=1.e-8):
                    raise ValueError('Phase duration must be a positive multiple of the control period')
                for i in range(count):
                    u = (i+1)/count
                    s = 10*u**3 - 15*u**4 + 6*u**5
                    q = {n: start[n]+s*(end[n]-start[n]) for n in start}
                    obs = env.step({'joint_positions_rad': q, 'duration_s': plan['control_period_s']})
                    row = trace_row(obs, phase['name'], probe)
                    stream.write(json.dumps(row, separators=(',', ':'), allow_nan=False)+'\n')
                    samples += 1
                    if not obs['metrics']['state_finite'] or obs['contacts']['errors'] or not obs['contacts']['sampling']['complete']:
                        stream.flush()
                        raise RuntimeError('Nonfinite state or incomplete native contact acquisition')
                stream.flush()
                probe.stream.flush()
                print(json.dumps(dict(event='phase_complete', case=args.case, phase=phase['name'], time_s=obs['time_s'])), flush=True)
                if phase['name'] in ('loaded_turn', 'forward', 'reverse', 'hold_after_turn'):
                    Image.fromarray(env.render(camera, width=960, height=720)).save(output/(phase['name']+'.png'))
        Image.fromarray(env.render(camera, width=960, height=720)).save(output/'final.png')
        dump(output/'metadata_final.json', env.metadata())
        report.update(passed=True, execution_complete=True, response_samples=samples,
            native_substeps=probe.total_substeps, simulated_duration_s=obs['time_s'],
            response_file_sha256=digest(output/'response.jsonl'))
    except Exception as exc:
        report.update(execution_complete=False, error=repr(exc), traceback=traceback.format_exc())
    finally:
        if probe is not None:
            probe.close()
        seal_report_before_shutdown(output, report, env)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', choices=('fastener_tool', 'gear_tool', 'hand_tool'), required=True)
    parser.add_argument('--release', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if not run(args)['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
