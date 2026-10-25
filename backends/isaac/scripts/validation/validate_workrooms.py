"""Native dynamics checks for the six furnished workrooms, with no model calls.

Run one room per native process from a frozen source tree. This checks diagnostic
cubes and small robot motions, not manipulation tasks or the static room props.
The caller must separately require a zero native process exit status.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import traceback

import numpy as np

from bench.isaacsim.release import dump
from scripts.maintenance.freeze_project_sources import verify_if_snapshot
from scripts.validation.validate_scenario import seal_report_before_shutdown


THRESHOLDS = {
    'control_dt_s': .02, 'settle_steps': 120, 'outward_steps': 50,
    'return_steps': 50, 'post_motion_steps': 50, 'repeat_settle_steps': 120,
    'rest_window_steps': 20, 'minimum_supported_rest_windows': 18,
    'minimum_fall_m': .02, 'minimum_downward_speed_m_s': .05,
    'maximum_rest_speed_m_s': .02, 'maximum_rest_angular_speed_rad_s': .3,
    'maximum_penetration_m': .002, 'maximum_support_gap_m': .005,
    'maximum_rest_drift_m': .002, 'maximum_root_translation_m': .001,
    'maximum_root_angle_rad': .01, 'joint_limit_tolerance_rad': .01,
    'arm_motion_rad': .04, 'finger_motion_rad': .06,
    'minimum_motion_fraction': .5, 'maximum_arm_return_error_rad': .01,
    'maximum_finger_return_error_rad': .02,
    'maximum_reset_position_error_m': 1.e-5,
    'maximum_reset_joint_error_rad': 1.e-5,
    'maximum_reset_angle_error_rad': 1.e-5,
    'maximum_repeated_rest_position_error_m': .002,
    'maximum_repeated_rest_angle_error_rad': .02,
    'minimum_rgb_std': 3., 'minimum_valid_depth_fraction': .05,
    'seed': 0,
}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def plain(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {k: plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    return value


def vector(value, length):
    result = np.asarray(value, dtype=float)
    if result.shape != (length,) or not np.isfinite(result).all():
        raise ValueError('Missing or nonfinite physical vector')
    return result


def rotation(wxyz):
    q = vector(wxyz, 4)
    if not .999 <= np.linalg.norm(q) <= 1.001:
        raise ValueError('Physical quaternion is not normalized')
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


def angle(a, b):
    a, b = vector(a, 4), vector(b, 4)
    rotation(a)
    rotation(b)
    return float(2*np.arccos(np.clip(abs(np.dot(a, b)/(np.linalg.norm(a)*np.linalg.norm(b))), 0., 1.)))


def support_contact(observation, object_path, support_path):
    """Use actual solver actor pairs; unrelated robot/table forces do not count."""
    result = {'impulse_ns': 0., 'min_separation_m': None, 'contact_substeps': 0}
    times = set()
    for row in observation['contacts']['rigid']:
        pair = (row['actor0'], row['actor1'])
        if object_path not in pair:
            continue
        other = pair[1] if pair[0] == object_path else pair[0]
        if other != support_path and not other.startswith(support_path + '/'):
            continue
        impulse, separation = float(row['impulse_magnitude_sum_ns']), float(row['min_separation_m'])
        if not np.isfinite([impulse, separation]).all() or impulse < 0:
            raise ValueError('Invalid native contact evidence')
        if impulse > 0:
            result['impulse_ns'] += impulse
            times.add(float(row['time_s']))
            old = result['min_separation_m']
            result['min_separation_m'] = separation if old is None else min(old, separation)
    result['contact_substeps'] = len(times)
    return result


def assess_objects(initial, rows, diagnostics, paths, thresholds=THRESHOLDS):
    """Recompute gravity/support/settling from recorded native observations."""
    if not rows or len(rows) < thresholds['rest_window_steps']:
        raise ValueError('Insufficient settling observations')
    window = rows[-thresholds['rest_window_steps']:]
    result = {}
    if set(initial['objects']) != set(diagnostics) or set(paths) != set(diagnostics):
        raise ValueError('Diagnostic object inventory mismatch')
    for name, spec in diagnostics.items():
        half = vector(spec['half_extents_m'], 3)
        if np.any(half <= 0):
            raise ValueError('Diagnostic cube extents must be positive')
        support_z = float(spec['support_top_z_m'])
        start = initial['objects'][name]
        p0 = vector(start['position_m'], 3)
        z0 = p0[2] - float(np.abs(rotation(start['quaternion_wxyz'])[2]) @ half)
        gap0 = z0-support_z
        if not np.isfinite(gap0) or gap0 < .04:
            raise ValueError('Diagnostic cube must begin at least 4 cm above its support')
        states = [row['observation']['objects'][name] for row in rows]
        rest = [row['observation']['objects'][name] for row in window]
        positions = np.array([vector(x['position_m'], 3) for x in states])
        rest_positions = np.array([vector(x['position_m'], 3) for x in rest])
        velocities = np.array([vector(x['linear_velocity_m_s'], 3) for x in states])
        speeds = [np.linalg.norm(vector(x['linear_velocity_m_s'], 3)) for x in rest]
        omegas = [np.linalg.norm(vector(x['angular_velocity_rad_s'], 3)) for x in rest]
        bottom_gaps = [float(vector(x['position_m'], 3)[2]
                      - np.abs(rotation(x['quaternion_wxyz'])[2]) @ half-support_z) for x in rest]
        contacts = [support_contact(row['observation'], paths[name], spec['expected_support_path']) for row in rows]
        rest_contacts = contacts[-thresholds['rest_window_steps']:]
        positive_windows = sum(x['impulse_ns'] > 0 for x in rest_contacts)
        separations = [x['min_separation_m'] for x in contacts if x['min_separation_m'] is not None]
        descent = float(p0[2]-positions[:, 2].min())
        min_vz = float(velocities[:, 2].min())
        drift = float(np.linalg.norm(rest_positions-rest_positions[0], axis=1).max())
        checks = {
            'gravity_descent': descent >= min(thresholds['minimum_fall_m'], gap0/2),
            'observed_downward_velocity': min_vz <= -thresholds['minimum_downward_speed_m_s'],
            'positive_expected_support_impulse': sum(x['impulse_ns'] for x in contacts) > 0,
            'persistent_support': positive_windows >= thresholds['minimum_supported_rest_windows'],
            'rest_linear_velocity': max(speeds) <= thresholds['maximum_rest_speed_m_s'],
            'rest_angular_velocity': max(omegas) <= thresholds['maximum_rest_angular_speed_rad_s'],
            'geometry_penetration': min(bottom_gaps) >= -thresholds['maximum_penetration_m'],
            'geometry_support_gap': max(bottom_gaps) <= thresholds['maximum_support_gap_m'],
            'solver_penetration': bool(separations) and min(separations) >= -thresholds['maximum_penetration_m'],
            'rest_drift': drift <= thresholds['maximum_rest_drift_m'],
        }
        result[name] = {'passed': all(checks.values()), 'checks': checks,
            'initial_support_gap_m': gap0, 'maximum_descent_m': descent,
            'minimum_vertical_velocity_m_s': min_vz,
            'maximum_rest_speed_m_s': float(max(speeds)),
            'maximum_rest_angular_speed_rad_s': float(max(omegas)),
            'rest_bottom_gap_range_m': [min(bottom_gaps), max(bottom_gaps)],
            'minimum_solver_separation_m': min(separations) if separations else None,
            'rest_drift_m': drift, 'supported_rest_windows': positive_windows,
            'support_impulse_ns': sum(x['impulse_ns'] for x in contacts),
            'object_actor_path': paths[name], 'expected_support_path': spec['expected_support_path']}
    return {'passed': all(x['passed'] for x in result.values()), 'objects': result}


def motion_spec(action_spec):
    home = action_spec['initial_joint_positions_rad']
    limits = action_spec['joint_limits_rad']
    fingers = [n for n in action_spec['hand_joint_names'] if 'index' in n.lower()]
    if not fingers:
        raise ValueError('No index finger joint in native action spec')
    choices = {}
    for name, magnitude in [('fr3_joint1', THRESHOLDS['arm_motion_rad'])] + [(n, THRESHOLDS['finger_motion_rad']) for n in fingers]:
        low, high = limits[name]
        offset = magnitude if home[name]+magnitude+.01 <= high else -magnitude
        if low+.01 <= home[name]+offset <= high-.01:
            choices[name] = offset
    hand = next((n for n in fingers if n in choices), None)
    if 'fr3_joint1' not in choices or hand is None:
        raise ValueError('No preflight-safe small joint motion')
    return {'arm_joint': 'fr3_joint1', 'finger_joint': hand,
            'offsets_rad': {n: choices[n] for n in ('fr3_joint1', hand)}}


def smooth_targets(home, offsets, fraction):
    if not 0 <= fraction <= 1:
        raise ValueError('Motion fraction outside [0, 1]')
    weight = 10*fraction**3-15*fraction**4+6*fraction**5
    return {name: float(value+weight*offsets.get(name, 0.)) for name, value in home.items()}


def assess_robot(initial, rows, action_spec, motion):
    root0 = vector(initial['root_pose_xyzw'], 7)
    q0 = initial['observation']['joint_positions_rad']
    translation, root_angle, limit_error = 0., 0., 0.
    finite = True
    for row in rows:
        root = vector(row['root_pose_xyzw'], 7)
        translation = max(translation, float(np.linalg.norm(root[:3]-root0[:3])))
        root_angle = max(root_angle, angle(root0[[6, 3, 4, 5]], root[[6, 3, 4, 5]]))
        observation = row['observation']
        finite = finite and observation['metrics']['state_finite'] and not observation['terminated']
        for name, q in observation['joint_positions_rad'].items():
            lo, hi = action_spec['joint_limits_rad'][name]
            if not np.isfinite(q):
                raise ValueError('Nonfinite physical joint')
            limit_error = max(limit_error, lo-q, q-hi)
    checks = {'finite_state': bool(finite),
        'fixed_base_translation': translation <= THRESHOLDS['maximum_root_translation_m'],
        'fixed_base_orientation': root_angle <= THRESHOLDS['maximum_root_angle_rad'],
        'joint_limits': limit_error <= THRESHOLDS['joint_limit_tolerance_rad']}
    tracked = {}
    motion_rows = [r for r in rows if r['phase'] in ('outward', 'return', 'post_motion')]
    if not motion_rows:
        raise ValueError('Missing robot motion evidence')
    for label, tolerance in [('arm', THRESHOLDS['maximum_arm_return_error_rad']),
                             ('finger', THRESHOLDS['maximum_finger_return_error_rad'])]:
        name = motion[label+'_joint']
        offset = motion['offsets_rad'][name]
        excursion = max((r['observation']['joint_positions_rad'][name]-q0[name])*np.sign(offset) for r in motion_rows)
        final_error = abs(motion_rows[-1]['observation']['joint_positions_rad'][name]-action_spec['initial_joint_positions_rad'][name])
        checks[label+'_moved'] = excursion >= abs(offset)*THRESHOLDS['minimum_motion_fraction']
        checks[label+'_returned'] = final_error <= tolerance
        tracked[name] = {'requested_offset_rad': offset, 'actual_directional_excursion_rad': float(excursion),
                         'final_home_error_rad': float(final_error)}
    return {'passed': all(checks.values()), 'checks': checks,
            'maximum_root_translation_m': translation, 'maximum_root_angle_rad': root_angle,
            'maximum_joint_limit_excursion_rad': float(limit_error), 'joint_motion': tracked}


def compare_reset(first, second, first_rest, second_rest):
    positions, angles, rested_positions, rested_angles = [], [], [], []
    for name, old in first['observation']['objects'].items():
        new = second['observation']['objects'][name]
        positions.append(float(np.linalg.norm(vector(old['position_m'], 3)-vector(new['position_m'], 3))))
        angles.append(angle(old['quaternion_wxyz'], new['quaternion_wxyz']))
        a, b = first_rest['observation']['objects'][name], second_rest['observation']['objects'][name]
        rested_positions.append(float(np.linalg.norm(vector(a['position_m'], 3)-vector(b['position_m'], 3))))
        rested_angles.append(angle(a['quaternion_wxyz'], b['quaternion_wxyz']))
    joints = max(abs(q-second['observation']['joint_positions_rad'][n])
                 for n, q in first['observation']['joint_positions_rad'].items())
    a, b = vector(first['root_pose_xyzw'], 7), vector(second['root_pose_xyzw'], 7)
    positions.append(float(np.linalg.norm(a[:3]-b[:3])))
    angles.append(angle(a[[6, 3, 4, 5]], b[[6, 3, 4, 5]]))
    checks = {'reset_position': max(positions) <= THRESHOLDS['maximum_reset_position_error_m'],
              'reset_orientation': max(angles) <= THRESHOLDS['maximum_reset_angle_error_rad'],
              'reset_joints': joints <= THRESHOLDS['maximum_reset_joint_error_rad'],
              'repeated_settled_position': max(rested_positions) <= THRESHOLDS['maximum_repeated_rest_position_error_m'],
              'repeated_settled_orientation': max(rested_angles) <= THRESHOLDS['maximum_repeated_rest_angle_error_rad']}
    return {'passed': all(checks.values()), 'checks': checks,
            'maximum_reset_position_error_m': max(positions), 'maximum_reset_angle_error_rad': max(angles),
            'maximum_reset_joint_error_rad': joints,
            'maximum_repeated_rest_position_error_m': max(rested_positions),
            'maximum_repeated_rest_angle_error_rad': max(rested_angles)}


def check_window(observation, state, previous, dt):
    sample = observation['contacts']['sampling']
    expected = round(.02/dt)
    checks = [not observation['contacts']['errors'], sample['complete'] is True,
        sample['sampled_substeps'] == expected,
        sample['native_end_step_index']-sample['native_start_step_index'] == expected,
        int(state['native_physics_step_index'])-int(previous['native_physics_step_index']) == expected,
        abs(float(state['time_s'])-float(previous['time_s'])-.02) < 1.e-8,
        abs(float(state['native_world_time_s'])-float(previous['native_world_time_s'])-.02) < 1.e-8,
        abs(sample['end_sim_time_s']-sample['start_sim_time_s']-.02) < 1.e-8,
        abs(sample['end_sim_time_s']-observation['sim_time_s']) < 1.e-8]
    if not all(checks):
        raise RuntimeError('Incomplete or inconsistent actual physics/contact sampling window')
    if not all(np.isfinite(x).all() for x in state.values()):
        raise RuntimeError('Nonfinite native state')


def stage_inventory(env):
    from pxr import UsdPhysics
    robot_path = env.scene_info['robot_path']
    articulations, physics, unexpected_dynamic, enabled_environment_joints = [], [], [], []
    expected = {x['path'] for x in env.scene_info['objects'].values()}
    expected_kinematic = set(env.scene_info.get('kinematic_body_paths',
        env.scene_info.get('workbench', {}).get('kinematic_body_paths', [])))
    expected_kinematic.update(x['path'] for x in env.scene_info.get('fixtures', {}).values()
                              if x.get('body_mode') == 'kinematic_fixture')
    found_kinematic = set()
    found = set()
    for prim in env.stage.Traverse():
        path = str(prim.GetPath())
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            articulations.append(path)
        if prim.IsA(UsdPhysics.Scene):
            physics.append(path)
        is_robot = path == robot_path or path.startswith(robot_path+'/')
        if not is_robot and prim.HasAPI(UsdPhysics.RigidBodyAPI):
            body = UsdPhysics.RigidBodyAPI(prim)
            if body.GetRigidBodyEnabledAttr().Get() is not False:
                if path in expected_kinematic and body.GetKinematicEnabledAttr().Get() is True:
                    found_kinematic.add(path)
                elif path in expected and body.GetKinematicEnabledAttr().Get() is not True:
                    found.add(path)
                else:
                    unexpected_dynamic.append(path)
        if not is_robot and prim.IsA(UsdPhysics.Joint):
            if UsdPhysics.Joint(prim).GetJointEnabledAttr().Get() is not False:
                enabled_environment_joints.append(path)
    checks = {'one_articulation': len(articulations) == 1 and env._robot_view.count == 1,
              'robot_29_joints': len(env.joint_names) == 29,
              'one_physics_scene': len(physics) == 1,
              'only_diagnostic_dynamic_objects_outside_robot': found == expected and not unexpected_dynamic,
              'configured_kinematic_supports': found_kinematic == expected_kinematic,
              'environment_joints_disabled': not enabled_environment_joints}
    return {'passed': all(checks.values()), 'checks': checks, 'articulation_roots': articulations,
            'physics_scenes': physics, 'diagnostic_dynamic_paths': sorted(found),
            'kinematic_support_paths': sorted(found_kinematic),
            'unexpected_dynamic_paths': unexpected_dynamic,
            'enabled_environment_joints': enabled_environment_joints}


def record(env, observation, episode, phase, command=None):
    from bench.simulation.isaacsim_backend.env import _np
    return {'episode': episode, 'phase': phase, 'command': deepcopy(command),
            'observation': observation, 'state': plain(env.capture_state()),
            'root_pose_xyzw': _np(env._robot_view.get_root_transforms())[0].tolist()}


def capture(env, output, label):
    from PIL import Image
    result = {}
    for camera in env.camera_ids:
        frame = env.render_rgbd(camera, width=640, height=480)
        rgb, depth = frame['rgb'], frame['depth_m']
        rgb_path, depth_path = output/f'{label}_{camera}.png', output/f'{label}_{camera}_depth.npy'
        Image.fromarray(rgb).save(rgb_path)
        np.save(depth_path, depth, allow_pickle=False)
        stats = {k: v for k, v in frame.items() if k not in ('rgb', 'depth_m')}
        stats.update(rgb_std=float(rgb.std()), valid_depth_fraction=float(np.count_nonzero(depth > 0)/depth.size),
                     rgb_path=str(rgb_path), rgb_sha256=digest(rgb_path),
                     depth_path=str(depth_path), depth_sha256=digest(depth_path),
                     rendering_preserved_physical_state=True)
        stats['passed'] = bool(stats['rgb_std'] >= THRESHOLDS['minimum_rgb_std']
                               and stats['valid_depth_fraction'] >= THRESHOLDS['minimum_valid_depth_fraction'])
        result[camera] = stats
    dump(output/f'{label}_rgbd.json', result)
    return {'passed': bool(result) and all(x['passed'] for x in result.values()), 'cameras': result}


def run(args):
    root = Path(__file__).resolve().parents[2]
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    started = datetime.now(timezone.utc).isoformat()
    plan = {'schema': 'native_workroom_dynamics_plan_v1', 'created_utc': started,
        'family': args.family, 'thresholds': deepcopy(THRESHOLDS), 'model_calls': 0,
        'task_success_claimed': False, 'static_decoration_manipulability_claimed': False,
        'scope': 'Gravity, actual support impulse, settling, small robot joint motion, reset and RGB-D only.',
        'device': args.device, 'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
        'single_cuda_device': os.environ.get('ISAAC_BENCH_SINGLE_CUDA_DEVICE'),
        'driver_override': os.environ.get('ISAAC_ALLOW_UNVALIDATED_DRIVER'),
        'validator_sha256': digest(__file__), 'source_root': str(root)}
    dump(output/'plan.json', plan)
    env, error = None, None
    checks, frames, records = {}, {}, []
    total_steps, native_steps, native_duration = 0, 0, 0.
    try:
        from scenes.workrooms import scene, validate
        from bench.simulation.isaacsim_backend.env import FR3DexterousIsaacEnv
        verification = verify_if_snapshot(root,
            required_modules=('scripts.validation.validate_workrooms', 'scenes.workrooms',
                              'bench.simulation.isaacsim_backend.env'), entrypoint_file=Path(__file__))
        if verification is None:
            raise ValueError('Native validation requires a complete frozen source snapshot')
        dump(output/'runtime_imports.json', verification)
        if args.device != 'cuda:0' or os.environ.get('ISAAC_BENCH_SINGLE_CUDA_DEVICE') != '1':
            raise ValueError('Use single UUID-selected CUDA device with --device cuda:0')
        uuid = os.environ.get('CUDA_VISIBLE_DEVICES', '')
        if not uuid.startswith('GPU-') or ',' in uuid:
            raise ValueError('CUDA_VISIBLE_DEVICES must be one physical GPU UUID')
        query = subprocess.run(['nvidia-smi', '-i', uuid,
            '--query-gpu=uuid,pci.bus_id,name,memory.used,memory.total', '--format=csv,noheader'],
            capture_output=True, text=True, check=True, timeout=20)
        if len(query.stdout.strip().splitlines()) != 1 or not query.stdout.startswith(uuid+','):
            raise RuntimeError('Selected physical GPU UUID did not resolve uniquely')
        dump(output/'gpu_identity.json', {'query': query.args, 'result': query.stdout.strip()})
        spec = scene(args.family, diagnostic=True)
        dump(output/'scene.json', spec)
        # scene() checks pinned hashes without importing pxr. USD composition
        # validation must wait until SimulationApp has loaded its own schemas.
        env = FR3DexterousIsaacEnv(spec, device=args.device, render_width=640, render_height=480)
        observation = env.reset(seed=THRESHOLDS['seed'])
        dump(output/'source_asset_preflight.json', validate(args.family))
        dump(output/'metadata.json', env.metadata())
        checks['stage_inventory'] = stage_inventory(env)
        if not checks['stage_inventory']['passed']:
            raise RuntimeError('Unexpected native stage inventory')
        action_spec = env.action_spec()
        motion = motion_spec(action_spec)
        dump(output/'motion_plan.json', motion)
        diagnostics = spec['workroom_diagnostics']
        paths = {name: row['path'] for name, row in env.scene_info['objects'].items()}
        home = action_spec['initial_joint_positions_rad']
        first = record(env, observation, 0, 'reset')
        records.append(first)
        dump(output/'initial_state.json', first)
        frames['initial'] = capture(env, output, 'initial')
        with (output/'physics.jsonl').open('x') as stream:
            def write(row):
                stream.write(json.dumps(row, allow_nan=False, separators=(',', ':'))+'\n')
                stream.flush()
            write(first)
            def advance(phase, count, episode=0, fraction=None):
                nonlocal total_steps, native_steps, native_duration
                print(json.dumps({'event': 'workroom_phase', 'family': args.family,
                                  'phase': phase, 'episode': episode, 'steps': count}), flush=True)
                batch = []
                for i in range(count):
                    targets = deepcopy(home) if fraction is None else smooth_targets(home, motion['offsets_rad'], fraction(i, count))
                    command = {'joint_positions_rad': targets, 'duration_s': .02}
                    previous = records[-1]['state']
                    observed = env.step(command)
                    row = record(env, observed, episode, phase, command)
                    write(row)
                    records.append(row)
                    batch.append(row)
                    total_steps += 1
                    native_steps += int(row['state']['native_physics_step_index'])-int(previous['native_physics_step_index'])
                    native_duration += float(row['state']['native_world_time_s'])-float(previous['native_world_time_s'])
                    check_window(observed, row['state'], previous, env.dt)
                    if (i+1) % 25 == 0 or i+1 == count:
                        print(json.dumps({'event': 'workroom_progress', 'family': args.family,
                            'phase': phase, 'episode': episode, 'phase_steps': i+1,
                            'total_control_steps': total_steps,
                            'episode_sim_time_s': observed['sim_time_s']}), flush=True)
                return batch
            settled = advance('settle', THRESHOLDS['settle_steps'])
            checks['initial_drop_and_settle'] = assess_objects(first['observation'], settled, diagnostics, paths)
            advance('outward', THRESHOLDS['outward_steps'], fraction=lambda i, n: (i+1)/n)
            advance('return', THRESHOLDS['return_steps'], fraction=lambda i, n: 1-(i+1)/n)
            advance('post_motion', THRESHOLDS['post_motion_steps'])
            checks['robot_motion'] = assess_robot(first, records[1:], action_spec, motion)
            checks['post_motion_support'] = assess_objects(first['observation'], records[1:], diagnostics, paths)
            frames['final'] = capture(env, output, 'final')
            observation = env.reset(seed=THRESHOLDS['seed'])
            second = record(env, observation, 1, 'reset')
            dump(output/'repeated_reset_state.json', second)
            records.append(second)
            write(second)
            repeated = advance('repeat_settle', THRESHOLDS['repeat_settle_steps'], episode=1)
            checks['repeated_drop_and_settle'] = assess_objects(second['observation'], repeated, diagnostics, paths)
            checks['reset_reproducibility'] = compare_reset(first, second, settled[-1], repeated[-1])
            frames['repeat_final'] = capture(env, output, 'repeat_final')
        checks['final_stage_inventory'] = stage_inventory(env)
        final_metadata = env.metadata()
        dump(output/'metadata_final.json', final_metadata)
        support_paths = env.scene_info.get('kinematic_body_paths', [])
        if support_paths:
            guard = final_metadata.get('static_mesh_contact_runtime', {})
            checks['fixed_supports_every_substep'] = {
                'passed': set(guard.get('actor_paths', [])) == set(support_paths)
                          and guard.get('checked_batches', 0) >= native_steps
                          and guard.get('maximum_translation_m', float('inf')) <= 1.e-6
                          and guard.get('maximum_angle_rad', float('inf')) <= 1.e-5,
                **guard}
        dump(output/'runtime_imports_final.json', verify_if_snapshot(root,
             required_modules=('scripts.validation.validate_workrooms', 'scenes.workrooms',
                               'bench.simulation.isaacsim_backend.env'), entrypoint_file=Path(__file__)))
    except BaseException as exc:
        error = {'type': type(exc).__name__, 'detail': str(exc), 'traceback': traceback.format_exc()}
        print(json.dumps({'event': 'workroom_error', 'family': args.family,
                          'type': type(exc).__name__, 'detail': str(exc)}), flush=True)
    passed = bool(error is None and checks and frames and all(x['passed'] for x in checks.values())
                  and all(x['passed'] for x in frames.values()))
    report = {'schema': 'native_workroom_dynamics_check_v1', 'family': args.family,
        'passed': passed, 'model_calls': 0, 'task_success_claimed': False,
        'static_decoration_manipulability_claimed': False,
        'scope': plan['scope'], 'checks': plain(checks), 'rgbd': frames, 'error': error,
        'control_steps': total_steps, 'simulation_duration_s': native_duration,
        'physics_substeps': native_steps,
        'requested_control_duration_s': total_steps*.02,
        'step_count_scope': 'Observed native index/time differences per control action; initialization/reset ticks excluded.',
        'object_state_writeback_after_reset': False, 'plan_sha256': digest(output/'plan.json'),
        'started_utc': started, 'finished_utc': datetime.now(timezone.utc).isoformat(),
        'artifacts_sha256': {str(p.relative_to(output)): digest(p) for p in output.rglob('*') if p.is_file()}}
    return seal_report_before_shutdown(output, report, env)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--family', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    report = run(parser.parse_args())
    if not report['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
