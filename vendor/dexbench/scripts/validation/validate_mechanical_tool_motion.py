"""Privileged robot joint-script test of the selected v3 tool; no object forces.

The preinserted reset is a documented simplification. Actual grip/contact and
thread advance are measured separately; merely executing a script is not success.
"""
from __future__ import annotations
import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import traceback
import numpy as np

from bench.isaacsim.release import dump
from bench.geometry import quaternion_matrix
from bench.paths import load_json
from scripts.maintenance.freeze_project_sources import verify_if_snapshot
from scripts.validation.validate_scenario import seal_report_before_shutdown
from scripts.validation.validate_workrooms import (plain, digest, record, check_window,
    capture, stage_inventory, smooth_targets, angle)
from scripts.validation.validate_mechanical_dynamics import (FixtureMonitor,
    native_task_parameters, declared_mount_pairs)
from scripts.validation.mechanical_checks import contact_filter_coverage
from scripts.validation.mechanical_tool_checks import (phase_window, preinserted_window,
    released_phase_window, contact_exchange_check)
from scripts.validation.validate_board_game_pickup import contact_rows
from bench.simulation.isaacsim_backend.contact_sample import SubstepContactExchange


def opposed_handle_contacts(raw, tool_path, pose, pads):
    """Positive pressure on opposite original cylindrical handle sidewalls."""
    rotation = quaternion_matrix(np.asarray(pose)[[6, 3, 4, 5]])
    points = {finger: [] for finger in pads}
    for row in raw:
        pair = (row['actor0'], row['actor1'])
        if tool_path not in pair:
            continue
        other = pair[1] if pair[0] == tool_path else pair[0]
        for finger, paths in pads.items():
            if other not in paths:
                continue
            for impulse, world, separation in zip(row['normal_impulses_ns'], row['points_m'], row['separations_m']):
                local = (np.asarray(world)-pose[:3]) @ rotation
                if (impulse > 1e-8 and separation >= -.002 and .105 <= local[2] <= .207
                        and abs(np.linalg.norm(local[:2])-.022) <= .003):
                    points[finger].append(local.tolist())
    opposed = []
    if points['thumb']:
        thumb = np.mean(points['thumb'], axis=0)[:2]
        for finger in ('index', 'middle', 'ring', 'pinky'):
            if points[finger]:
                other = np.mean(points[finger], axis=0)[:2]
                norm = np.linalg.norm(thumb)*np.linalg.norm(other)
                if norm > 0 and np.dot(thumb, other)/norm < -.3:
                    opposed.append(finger)
    return {'qualified': bool(opposed), 'opposed_fingers': opposed, 'local_points_m': points}


class ToolObserver:
    def __init__(self, env, output):
        self.env, self.original = env, env.world.step
        self.callback = self.step
        self.exchange = SubstepContactExchange(env, self)
        self.stream = (output/'native_frames.jsonl').open('x')
        self.phase, self.rows = 'settle', []
        self.tool_path = env.scene_info['objects']['screwdriver']['path']
        self.robot = set(env.scene_info['robot_link_paths'].values())
        self.hand = {p for n, p in env.scene_info['robot_link_paths'].items() if n.startswith('right_')}
        self.pads = {f: {p for n, p in env.scene_info['robot_link_paths'].items()
                         if n in ('right_'+f+'_elastomer', 'right_'+f+'_DP')}
                     for f in ('thumb', 'index', 'middle', 'ring', 'pinky')}
        self.mount = {tuple(sorted(x)) for x in declared_mount_pairs(env)}
        self.indices = {name: env.object_names.index(name) for name in ('screwdriver', 'bolt')}
        self.closed = False
        env.world.step = self.callback

    def step(self, *args, **kwargs):
        env = self.env
        if self.closed or args or kwargs != {'render': False}:
            raise RuntimeError('Only ordinary native integration is allowed')
        self.exchange.begin_step(env)
        result = self.original(*args, **kwargs)
        sample = self.exchange.acquire(env)
        state = env.capture_state()
        if not all(np.isfinite(v).all() for v in state.values()):
            raise RuntimeError('Nonfinite native tool state')
        try:
            raw = contact_rows(env, sample)
        except BaseException:
            np.savez_compressed(Path(self.stream.name).parent/'rejected_contact_sample.npz',
                **dict(zip(('impulses', 'points', 'normals', 'separations', 'counts', 'starts'), sample.for_env(env))))
            raise
        projected = env._project_rigid_contacts(sample)
        tool = state['object_q'][self.indices['screwdriver']]
        grip = opposed_handle_contacts(raw, self.tool_path, tool, self.pads)
        interference = []
        maximum_penetration = 0.
        for item in raw:
            pair = (item['actor0'], item['actor1'])
            members = self.robot.intersection(pair)
            if not members or item['impulse_magnitude_sum_ns'] <= 1e-8:
                continue
            maximum_penetration = max(maximum_penetration, -min(item['separations_m'], default=0.))
            permitted = (self.tool_path in pair and members <= self.hand)
            if not permitted and tuple(sorted(pair)) not in self.mount:
                interference.append(list(pair))
        row = {'phase': self.phase, 'native_step_index': int(env.world.current_time_step_index),
            'simulation_time_s': float(env._current_step_time), 'grip': grip,
            'joint_positions_rad': {name: float(state['joint_q'][i]) for i, name in enumerate(env.joint_names)},
            'object_poses_xyzw': {k: state['object_q'][i].tolist() for k, i in self.indices.items()},
            'object_velocities': {k: state['object_qd'][i].tolist() for k, i in self.indices.items()},
            'robot_interference': interference, 'maximum_robot_contact_penetration_m': maximum_penetration,
            'raw_contacts': [r for r in raw if self.robot.intersection((r['actor0'], r['actor1']))
                or self.tool_path in (r['actor0'], r['actor1'])
                or self.env.scene_info['objects']['bolt']['path'] in (r['actor0'], r['actor1'])]}
        after = env.capture_state()
        if not all(np.array_equal(state[k], after[k]) for k in state):
            raise RuntimeError('Read-only observer changed physics state')
        self.stream.write(json.dumps(plain(row), allow_nan=False)+'\n')
        self.rows.append(row)
        self.exchange.offer_after_readonly(env, sample, projected)
        return result

    def close(self):
        if not self.closed:
            self.closed = True
            self.env.world.step = self.original
            try:
                return self.exchange.close(self.env, self)
            finally:
                self.stream.close()


def grip_window(rows, dt, duration=.10):
    window, sampling = phase_window(rows, 'hold_grasp', dt, duration)
    good = [r['grip']['qualified'] for r in window]
    longest = current = 0
    for value in good:
        current = 0 if value else current+1
        longest = max(longest, current)
    fraction = sum(good)/len(good) if good else 0.
    return {'passed': sampling['passed'] and fraction >= .9 and longest*dt <= .01,
            'qualified_fraction': fraction, 'maximum_gap_s': longest*dt, 'window_s': duration,
            'sampling': sampling}


def validate_motion_phases(phases, home):
    """Require the measured hold/open/retreat sequence before creating a GPU env."""
    names = [phase['name'] for phase in phases]
    required = ('hold_grasp', 'open', 'retreat', 'final_hold')
    if (len(names) != len(set(names)) or 'settle' in names or not names or names[-1] != 'final_hold'
            or any(name not in names for name in required)
            or not names.index('hold_grasp') < names.index('open') < names.index('retreat') < names.index('final_hold')):
        raise ValueError('Plan needs distinct hold_grasp, open, retreat, final_hold phases in order')
    minimum = {'hold_grasp': .3, 'open': .1, 'retreat': .1, 'final_hold': .1}
    for phase in phases:
        targets, duration = phase['joint_positions_rad'], phase['duration_s']
        if (set(targets) != set(home) or not np.isfinite(list(targets.values())).all()
                or not np.isfinite(duration) or not minimum.get(phase['name'], .02) <= duration <= 10.
                or abs(duration/.02-round(duration/.02)) > 1.e-7):
            raise ValueError('Each phase needs finite complete joint targets and enough 20ms steps for its measured window')
    if any(abs(phases[-1]['joint_positions_rad'][name]-value) > 1.e-12 for name, value in home.items()):
        raise ValueError('Final hold must return to every frozen ready joint target')


def run(args):
    root = Path(__file__).resolve().parents[2]
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    env = observer = None
    checks, frames, records, completed = {}, {}, [], []
    error, native_steps = None, 0
    try:
        from scenes.workrooms.mechanical_assembly.dynamics import preinserted_scene_v3
        from bench.simulation.isaacsim_backend.env import FR3DexterousIsaacEnv
        verification = verify_if_snapshot(root, required_modules=(
            'scripts.validation.validate_mechanical_tool_motion', 'bench.simulation.isaacsim_backend.env'),
            entrypoint_file=Path(__file__))
        if verification is None:
            raise ValueError('Complete frozen source required')
        dump(out/'runtime_imports.json', verification)
        uuid = os.environ.get('CUDA_VISIBLE_DEVICES', '')
        if not uuid.startswith('GPU-') or ',' in uuid or os.environ.get('ISAAC_BENCH_SINGLE_CUDA_DEVICE') != '1':
            raise ValueError('Select one healthy physical GPU UUID')
        if digest(args.motion_plan) != args.motion_plan_sha256:
            raise ValueError('Motion plan hash changed')
        plan = load_json(args.motion_plan)
        dump(out/'motion_plan.json', plan)
        scene, initial, _, ready = preinserted_scene_v3(args.stand_root, args.threaded_root)
        validate_motion_phases(plan['phases'], initial['joint_positions_rad'])
        dump(out/'scene.json', scene)
        dump(out/'initial_state.json', initial)
        dump(out/'plan.json', {'scope': 'Privileged ordinary joint-script diagnostic of preinserted v3 tool; no model or object wrench.',
            'motion_plan_sha256': args.motion_plan_sha256, 'model_calls': 0, 'task_success_claimed': False,
            'gpu_uuid': uuid, 'driver_override': os.environ.get('ISAAC_ALLOW_UNVALIDATED_DRIVER'),
            'read_true_object_state': True, 'object_state_writeback_after_reset': False})
        env = FR3DexterousIsaacEnv(scene, device='cuda:0', render_width=640, render_height=480)
        obs = env.reset(seed=0, initial_state=initial)
        fixture = FixtureMonitor(env)
        env._read_rigid_contacts = fixture.wrap(env._read_rigid_contacts)
        checks.update(stage_inventory=stage_inventory(env), contact_coverage=contact_filter_coverage(env),
                      native_task_parameters=native_task_parameters(env))
        dump(out/'metadata.json', env.metadata())
        if not all(x['passed'] for x in checks.values()):
            raise RuntimeError('Native initialization checks failed')
        records.append(record(env, obs, 0, 'reset'))
        frames['initial'] = capture(env, out, 'initial')
        observer = ToolObserver(env, out)
        home = deepcopy(initial['joint_positions_rad'])
        phases = [{'name': 'settle', 'duration_s': .5, 'joint_positions_rad': home}, *plan['phases']]
        with (out/'physics.jsonl').open('x') as stream:
            stream.write(json.dumps(records[0], allow_nan=False)+'\n')
            for phase in phases:
                name, targets = phase['name'], phase['joint_positions_rad']
                duration = phase['duration_s']
                if set(targets) != set(home) or not .02 <= duration <= 10. or abs(duration/.02-round(duration/.02)) > 1e-7:
                    raise ValueError('Each bounded phase needs all 29 joint targets and a multiple of 20ms')
                observer.phase = name
                offsets = {k: targets[k]-home[k] for k in home}
                count = round(duration/.02)
                print(json.dumps({'event': 'tool_motion_phase', 'phase': name, 'steps': count}), flush=True)
                for i in range(count):
                    command = {'joint_positions_rad': smooth_targets(home, offsets, (i+1)/count), 'duration_s': .02}
                    previous = records[-1]['state']
                    obs = env.step(command)
                    row = record(env, obs, 0, name, command)
                    records.append(row)
                    stream.write(json.dumps(row, allow_nan=False)+'\n')
                    stream.flush()
                    native_steps += int(row['state']['native_physics_step_index'])-int(previous['native_physics_step_index'])
                    check_window(obs, row['state'], previous, env.dt)
                    if (i+1) % 25 == 0 or i+1 == count:
                        print(json.dumps({'event': 'tool_motion_progress', 'phase': name, 'phase_steps': i+1,
                            'native_steps': native_steps, 'simulation_time_s': env.time_s}), flush=True)
                    batch = observer.rows[-round(.02/env.dt):]
                    if any(r['robot_interference'] or r['maximum_robot_contact_penetration_m'] > .002 for r in batch):
                        raise RuntimeError('Script caused unexpected robot contact or excessive penetration')
                home = deepcopy(targets)
                completed.append(name)
                if name == 'settle':
                    paths = {key: env.scene_info['objects'][key]['path'] for key in ('screwdriver', 'bolt')}
                    paths['fixture'] = env.scene_info['fixtures']['threaded_fixture']['path']
                    checks['preinserted_reset_stability'] = preinserted_window(
                        observer.rows, records[0]['observation']['objects'], paths, env.dt)
                    frames['settled'] = capture(env, out, 'settled')
                    if not checks['preinserted_reset_stability']['passed']:
                        raise RuntimeError('Preinserted reset failed passive pose, velocity or contact checks')
                if name == 'hold_grasp':
                    checks['opposed_grasp'] = grip_window(observer.rows, env.dt, .30)
                    frames['grasp'] = capture(env, out, 'grasp')
                    if not checks['opposed_grasp']['passed']:
                        raise RuntimeError('Opposing native finger pressure on the handle was not sustained')
                if name in ('open', 'retreat', 'final_hold'):
                    checks[name+'_endpoint'] = released_phase_window(observer.rows, name,
                        targets, observer.hand, observer.tool_path, env.dt)
                    if not checks[name+'_endpoint']['passed']:
                        raise RuntimeError(name+' did not reach all joint targets or release hand-tool contact')
        checks['fixed_fixture'] = fixture.result(native_steps)
        checks['all_phases_executed'] = {'passed': completed == [p['name'] for p in phases]}
        checks['no_robot_interference'] = {'passed': not any(r['robot_interference'] or r['maximum_robot_contact_penetration_m'] > .002 for r in observer.rows)}
        frames['final'] = capture(env, out, 'final')
    except BaseException as exc:
        error = {'type': type(exc).__name__, 'detail': str(exc), 'traceback': traceback.format_exc()}
    finally:
        if observer is not None:
            try:
                result = observer.close()
                dump(out/'contact_exchange.json', result)
                checks['contact_exchange'] = contact_exchange_check(result, native_steps, len(observer.rows))
                checks['contact_exchange']['hook_restored'] = env.world.step == observer.original
                checks['contact_exchange']['passed'] = bool(checks['contact_exchange']['passed']
                    and checks['contact_exchange']['hook_restored'] and env._contact_exchange is None)
            except BaseException as exc:
                dump(out/'contact_exchange_error.json', {'type': type(exc).__name__, 'detail': str(exc)})
                if error is None:
                    error = {'type': type(exc).__name__, 'detail': str(exc), 'traceback': traceback.format_exc()}
        if env is not None:
            dump(out/'metadata_final.json', env.metadata())
        try:
            dump(out/'runtime_imports_final.json', verify_if_snapshot(root, required_modules=(
                'scripts.validation.validate_mechanical_tool_motion', 'bench.simulation.isaacsim_backend.env'),
                entrypoint_file=Path(__file__)))
        except BaseException as exc:
            if error is None:
                error = {'type': type(exc).__name__, 'detail': str(exc), 'traceback': traceback.format_exc()}
    required = ('preinserted_reset_stability', 'opposed_grasp', 'open_endpoint', 'retreat_endpoint', 'final_hold_endpoint', 'contact_exchange')
    passed = (error is None and all(name in checks for name in required)
        and all(x['passed'] for x in checks.values()) and all(x['passed'] for x in frames.values()))
    report = {'schema': 'mechanical_robot_tool_motion.v2', 'passed': bool(passed), 'error': error,
        'checks': plain(checks), 'rgbd': frames, 'completed_phases': completed,
        'physics_substeps': native_steps, 'model_calls': 0, 'task_success_claimed': False,
        'scope': 'Preinserted passive stability, actual opposing hand-tool contacts, and measured open/retreat/final-hold joint endpoints with contact release only; screw rotation/advance and full L4 success require separate evaluation.',
        'object_external_wrenches': False, 'object_state_writeback_after_reset': False,
        'artifacts_sha256': {str(p.relative_to(out)): digest(p) for p in out.rglob('*') if p.is_file()}}
    return seal_report_before_shutdown(out, report, env)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stand-root', required=True, type=Path)
    parser.add_argument('--threaded-root', required=True, type=Path)
    parser.add_argument('--motion-plan', required=True, type=Path)
    parser.add_argument('--motion-plan-sha256', required=True)
    parser.add_argument('--output', required=True, type=Path)
    if not run(parser.parse_args())['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
