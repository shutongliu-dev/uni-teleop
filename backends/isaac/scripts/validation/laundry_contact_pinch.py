"""Explicit native SHARPA cloth contact canary with optional captured assistance.

Open, approach, close, lift, lower, open and retreat through original joint
drives. By default there are no attachments. The explicitly requested local
assistance requires a measured, stable captured material patch first. Neither
mode changes collision filters. A commanded pinch never implies capture.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
import traceback
import xml.etree.ElementTree as ET

import numpy as np

from bench.geometry import rpy_matrix
from bench.paths import load_json
from scripts.validation.laundry_engineering_motion import minimum_jerk
from scripts.validation.laundry_single_garment_native import (
    DATASETS, PROJECT, NativeArrays, advance, dump, pin, robot_readback)


def hand_collision_vertices(model):
    """Source-mesh bounds for planning only; native cooked contact is separate."""
    import trimesh
    from bench.geometry import transform
    root = ET.parse(model.hand_path).getroot()
    rows = []
    for link in root.findall('link'):
        for collision in link.findall('collision'):
            mesh = collision.find('geometry/mesh')
            if mesh is None:
                raise ValueError('Expected original mesh collision geometry')
            path = (model.hand_path.parent/mesh.get('filename')).resolve()
            vertices = np.asarray(trimesh.load(path, process=False).vertices)
            scale = np.fromstring(mesh.get('scale', '1 1 1'), sep=' ')
            origin = collision.find('origin')
            xyz = np.fromstring(origin.get('xyz', '0 0 0') if origin is not None else '0 0 0', sep=' ')
            rpy = np.fromstring(origin.get('rpy', '0 0 0') if origin is not None else '0 0 0', sep=' ')
            matrix = transform(xyz, rpy_matrix(rpy))
            rows.append({'body_name': link.get('name'), 'source': pin(path),
                'vertices': vertices*scale @ matrix[:3, :3].T+matrix[:3, 3]})
    return rows


def local_hand_points(model, rows, posture):
    poses = model.hand.forward(posture)
    return {row['body_name']: row['vertices'] @ poses[row['body_name']][:3, :3].T
            + poses[row['body_name']][:3, 3] for row in rows}


def pinch_postures(model, side, *, open_distance=.045, close_distance=.010, samples=61):
    """Solve actual thumb/index chains while respecting original joint bounds."""
    from bench.control.hand_presets import PINCH_ANCHORS
    sign = 1. if side == 'right' else -1.
    center = np.asarray([.070, sign*.050, .150])
    posture = dict(PINCH_ANCHORS[side])
    rows = []
    for distance in np.linspace(open_distance, close_distance, samples):
        errors = {}
        for finger, direction in (('thumb', sign), ('index', -sign)):
            point = center+np.asarray([0., direction*distance/2., 0.])
            solved, error = model._finger_inverse(finger, point, posture, tolerance=.0002)
            posture.update(solved)
            errors[finger] = float(error)
        for joint in model.hand.active:
            if not joint.lower <= posture[joint.name] <= joint.upper:
                raise ValueError('Finger IK violated an original joint limit')
        if any(model.hand_posture_prior.violation(finger, posture) for finger in ('thumb', 'index')):
            raise ValueError('Finger IK violated the original posture prior')
        rows.append({'nominal_tip_distance_m': float(distance),
            'hand_posture_rad': dict(posture), 'tip_ik_errors_m': errors})
    return center, rows


def interpolate_posture(rows, amount):
    position = float(np.clip(amount, 0., 1.))*(len(rows)-1)
    low = int(position)
    high = min(low+1, len(rows)-1)
    fraction = position-low
    a, b = rows[low]['hand_posture_rad'], rows[high]['hand_posture_rad']
    return {name: a[name]+fraction*(b[name]-a[name]) for name in a}


def plan(release, side, xy, yaw_deg, pitch_deg, bottom_z, close_distance,
         closure_lift_m=.003, capture_rise_m=0., closure_lift_start=0., open_distance=.045):
    from bench.control.factory import public_robot_geometry, robot_configuration
    if not np.isfinite(closure_lift_m) or not 0. <= closure_lift_m <= .01:
        raise ValueError('Closure rise must be between zero and 10 mm')
    if not np.isfinite(capture_rise_m) or not 0. <= capture_rise_m <= .005:
        raise ValueError('Post-closure seating rise must be between zero and 5 mm')
    if not np.isfinite(closure_lift_start) or not 0. <= closure_lift_start < 1.:
        raise ValueError('Closure rise starts within the open-to-closed fraction [0, 1)')
    release = Path(release).resolve()
    scene, initial = [load_json(release/name) for name in ('scene.json', 'initial_state.json')]
    geometry = public_robot_geometry(robot_configuration(scene))
    model = geometry.sides[side] if hasattr(geometry, 'sides') else geometry
    if not 0. < close_distance < open_distance <= .075:
        raise ValueError('Original-hand pinch distances require 0 < closed < open <= 75 mm')
    center, postures = pinch_postures(model, side, open_distance=open_distance,
                                     close_distance=close_distance)
    meshes = hand_collision_vertices(model)
    rotation = rpy_matrix([np.pi, np.deg2rad(pitch_deg), np.deg2rad(yaw_deg)])
    floors = []
    for row in postures:
        points = local_hand_points(model, meshes, row['hand_posture_rad'])
        floors.append(min(float(((value-center) @ rotation.T)[:, 2].min()) for value in points.values()))
    # One fixed wrist orientation/height throughout closure; do not lift cloth
    # by a hidden changing anchor or change the hand mesh to clear the table.
    tcp = np.asarray([*xy, bottom_z-min(floors)])
    wrist = np.eye(4)
    wrist[:3, :3] = rotation
    wrist[:3, 3] = tcp-rotation @ center
    seed = dict(initial['joint_positions_rad'])
    offset = model.wrist_offset if model.wrist_offset is not None else np.eye(4)
    receipts = []
    for rise in (.10, .06, .03, 0., .05, 0., .10):
        target = wrist.copy(); target[2, 3] += rise
        own = np.linalg.inv(model.arm_base) @ target @ np.linalg.inv(offset)
        result = model.arm.inverse(own, model.arm_tip, seed, iterations=200, multi_starts=12,
            position_tolerance=.0005, rotation_tolerance=.005)
        if not result['converged']:
            raise ValueError('Pinch wrist target has no verified IK: '+json.dumps(result))
        seed.update(result['joint_positions_rad'])
        receipts.append({'rise_m': rise, **result})
    return {'schema': 'laundry_contact_pinch_plan.v1', 'side': side,
        'release': str(release), 'inputs': {name: pin(release/name) for name in
            ('scene.json', 'initial_state.json', 'task.json', 'release_manifest.json')},
        'local_tip_pair_center_m': center.tolist(), 'tcp_world_m': tcp.tolist(),
        'wrist_world_matrix': wrist.tolist(), 'postures': postures,
        'source_mesh_hand_bottom_z_m': (tcp[2]+np.asarray(floors)).tolist(),
        'source_meshes': [{k: v for k, v in row.items() if k != 'vertices'} for row in meshes],
        'arm_ik': receipts, 'source_kinematics': model.asset_sha256,
        'closure_lift_m': float(closure_lift_m),
        'capture_rise_m': float(capture_rise_m),
        'closure_lift_start_fraction': float(closure_lift_start),
        'attachments': False, 'collision_filtering': False, 'cloth_position_velocity_writes': 0,
        'scope': 'Discrete original-hand and wrist targets, not measured capture, contact force or native collision qualification.'}


def native(args):
    from scripts.maintenance.freeze_project_sources import verify_if_snapshot
    source = verify_if_snapshot(PROJECT, required_modules=('scripts.validation.laundry_contact_pinch',),
        entrypoint_file=Path(__file__))
    if source is None:
        raise ValueError('Use a complete frozen source snapshot')
    spec = load_json(args.plan)
    if spec['schema'] != 'laundry_contact_pinch_plan.v1':
        raise ValueError('Expected an explicit contact plan')
    for value in spec['inputs'].values():
        if pin(value['path']) != value:
            raise ValueError('Contact-plan physical input identity changed')
    scene = load_json(Path(spec['inputs']['scene.json']['path']))
    initial = load_json(Path(spec['inputs']['initial_state.json']['path']))
    if scene.get('named_cloth_initialization') != 'author-only' or len(scene['cloths']) != 1:
        raise ValueError('One original authored native cloth is required')
    out = args.output.resolve()
    if not out.is_relative_to(DATASETS) or out.exists():
        raise ValueError('Use a fresh shared experiment directory')
    out.mkdir(parents=True)
    report = {'schema': 'laundry_contact_pinch_native.v1', 'complete': False,
        'source': source, 'source_manifest': pin(PROJECT/'source_snapshot_manifest.json'),
        'plan': pin(args.plan), 'attachments': bool(args.capture_assist), 'collision_filtering': False,
        'capture_assistance_requested': bool(args.capture_assist),
        'cloth_position_velocity_writes': 0, 'task_success_claimed': False,
        'grasp_qualified': False, 'physics_accepted': False,
        'scope': ('Original finger/arm joint-drive canary; optional declared local assistance '
                  'after measured geometric capture is separate from friction-only grasp qualification.')}
    env = capture = writer = log = assist = None
    exit_code = 1
    # Fixed simulation schedule; all changes are original articulation targets.
    phases = [('settle', .5), ('open_hover', 2.), ('descend', 1.5), ('close', 2.)]
    if spec.get('capture_rise_m', 0.):
        phases.append(('capture_raise', 1.))
    phases += [('closed_hold', 1.), ('lift', 1.5), ('lift_hold', .75), ('lower', 1.5),
               ('open', 1.), ('retreat', 1.), ('free', 1.)]
    dt = float(scene['physics']['dt_s'])
    duration = sum(seconds for _, seconds in phases)
    steps = round(duration/dt)
    try:
        from bench.control.factory import public_robot_geometry, robot_configuration
        from bench.simulation.isaacsim_backend.env import FR3DexterousIsaacEnv, _np
        from bench.evaluation.cloth.cloth_robot_collision import collect_colliders
        env = FR3DexterousIsaacEnv(scene); env.reset(initial_state=initial)
        side = spec['side']
        geometry = public_robot_geometry(robot_configuration(scene))
        model = geometry.sides[side] if hasattr(geometry, 'sides') else geometry
        base_wrist = np.asarray(spec['wrist_world_matrix'])
        offset = model.wrist_offset if model.wrist_offset is not None else np.eye(4)
        joint_seed = {n: float(np.clip(v, *env.joint_limits[n]))
                      for n, v in zip(env.joint_names, robot_readback(env)['joint_q'])}
        report.update(joint_names=env.joint_names, body_names=env.body_names,
            native_initialization=env.named_cloth_initialization_receipt,
            actual_control='original native joint drives and original gravity feedforward', phases=phases)
        export = collect_colliders(env); dump(out/'robot_colliders.json', export)
        report['robot_colliders'] = pin(out/'robot_colliders.json')
        view = env._cloth_views[scene['cloths'][0]['id']]
        cloth = lambda: _np(view.get_simulation_nodal_positions())[0]
        if args.capture_assist:
            from scripts.validation.laundry_contact_assist import CapturedPatchAssist
            assist = CapturedPatchAssist(env, scene, side, export, out)
        writer = NativeArrays(out/'states', steps+1)
        env.stage.Export(str(out/'initial.usda'))
        start_step, start_time = env.world.current_time_step_index, env.world.current_time
        env._begin_contact_sampling(); log = (out/'rigid_contacts.jsonl').open('x')
        step, phase_index, pair_count, impulse = 0, -1, 0, 0.
        started = time.monotonic()
        def command(values):
            nonlocal joint_seed
            if any(not env.joint_limits[n][0] <= values[n] <= env.joint_limits[n][1] for n in values):
                raise ValueError('Contact target outside original limits')
            expanded = env._expand_mimic_targets({n: float(values[n]) for n in env.actuated_joint_names})
            env._targets = np.asarray([[expanded[n] for n in env.joint_names]], dtype=np.float32)
            env._robot_view.set_dof_position_targets(env._tensor(env._targets), env._one)
            joint_seed = {**values, **expanded}
        def target(rise, closed, warm=None):
            matrix = base_wrist.copy(); matrix[2, 3] += rise
            ik = model.arm.inverse(np.linalg.inv(model.arm_base) @ matrix @ np.linalg.inv(offset),
                model.arm_tip, joint_seed if warm is None else warm, iterations=120, multi_starts=0,
                position_tolerance=.0005, rotation_tolerance=.005)
            if not ik['converged']:
                raise ValueError('Unverified contact-path IK: '+json.dumps(ik))
            return {**joint_seed, **ik['joint_positions_rad'],
                **interpolate_posture(spec['postures'], closed)}
        if args.media_output:
            from scripts.media.laundry_cloth_frames import LaundryClothFrameCapture
            tcp = np.asarray(spec['tcp_world_m'])
            cameras = {'task': scene['cameras']['task'], 'pinch': {
                'eye': (tcp+np.array([.30, -.34, .26])).tolist(),
                'target': (tcp+[0, 0, .025]).tolist(), 'focal_length_mm': 36.,
                'horizontal_aperture_mm': 36., 'vertical_aperture_mm': 27.,
                'clipping_m': [.01, 12.], 'resolution': [1280, 960],
                'kind': 'world', 'role': 'third_person', 'title_zh': '真实闭指接触近景'}}
            capture = LaundryClothFrameCapture(stage=env.stage, world=env.world, app=env.app,
                cloth_views=env._cloth_views, cloth_specs=scene['cloths'], triangles=env.cloth_triangles,
                cloth_paths={k: v['path'] for k, v in env.scene_info['cloths'].items()},
                output=args.media_output, cameras=cameras)
        def photo(tag):
            before = robot_readback(env)
            capture.capture(tag, camera_ids=('task', 'pinch'))
            if any(not np.array_equal(v, robot_readback(env)[k]) for k, v in before.items()):
                raise RuntimeError('Rendering changed native articulation state')
        def record():
            state = robot_readback(env)
            writer.append({**state, 'cloth_q': cloth(), 'cloth_qd': _np(view.get_simulation_nodal_velocities())[0],
                'control_mask': assist.mask(len(cloth())) if assist else np.zeros(len(cloth()), np.uint8),
                'assist_enabled': np.asarray(bool(assist and assist.active)),
                'assist_anchor_pose_xyzw': assist.anchor_pose.copy() if assist else np.array([0.,0.,0.,0.,0.,0.,1.]),
                'time_s': np.asarray(step*dt), 'native_step': np.asarray(step, np.int64),
                'world_step': np.asarray(env.world.current_time_step_index, np.int64),
                'world_time_s': np.asarray(env.world.current_time),
                'phase_index': np.asarray(phase_index, np.int64), 'joint_target': env._targets[0].copy(),
                'rigid_contact_pair_count': np.asarray(pair_count, np.int64),
                'rigid_contact_impulse_ns': np.asarray(impulse, dtype=np.float64)})
        def tick():
            nonlocal step, pair_count, impulse, started
            if assist:
                assist.update(phases[phase_index][0], step*dt, robot_readback(env),
                              cloth(), env._targets[0])
            env._current_step_time = env.time_s+env.dt; advance(env)
            env._contact_reports.clear(); env._read_rigid_contacts()
            pair_count = len(env._contact_reports)
            impulse = float(sum(x['impulse_magnitude_sum_ns'] for x in env._contact_reports))
            if pair_count: log.write(json.dumps({'native_step': step+1, 'contacts': env._contact_reports})+'\n')
            step += 1; record()
            if capture and int(step*dt*args.media_fps+1e-9) > int((step-1)*dt*args.media_fps+1e-9):
                photo(f'step_{step:06d}')
            if time.monotonic()-started > 20:
                print(json.dumps({'event': 'pinch_progress', 'phase': phases[phase_index][0],
                    'step': step, 'time_s': step*dt}), flush=True); started = time.monotonic()
        record()
        if capture: photo('initial')
        for phase_index, (name, seconds) in enumerate(phases):
            begin = dict(joint_seed)
            hover = target(.10, 0., {**joint_seed, **spec['arm_ik'][0]['joint_positions_rad']}) if name == 'open_hover' else None
            closure_rise = float(spec.get('closure_lift_m', 0.))
            capture_rise = float(spec.get('capture_rise_m', 0.))
            hold_rise = closure_rise+capture_rise
            for local in range(round(seconds/dt)):
                if local % round(.02/dt) == 0:
                    a = minimum_jerk((local+1)*dt, 0., seconds)
                    if name == 'open_hover':
                        command({n: begin[n]+a*(hover[n]-begin[n]) for n in begin})
                    elif name == 'descend': command(target(.10*(1-a), 0.))
                    elif name == 'close':
                        start = float(spec.get('closure_lift_start_fraction', 0.))
                        rise_fraction = a if start == 0. else minimum_jerk(
                            float(np.clip((a-start)/(1-start), 0., 1.)), 0., 1.)
                        command(target(closure_rise*rise_fraction, a))
                    elif name == 'capture_raise': command(target(closure_rise+capture_rise*a, 1.))
                    elif name == 'lift': command(target(hold_rise+.05*a, 1.))
                    elif name == 'lower': command(target(hold_rise+.05*(1-a), 1.))
                    elif name == 'open': command(target(hold_rise, 1-a))
                    elif name == 'retreat': command(target(hold_rise+.10*a, 0.))
                tick()
            print(json.dumps({'event': 'pinch_phase_complete', 'phase': name, 'time_s': step*dt}), flush=True)
        if step != steps or env.world.current_time_step_index-start_step != steps:
            raise RuntimeError('Incomplete native contact sequence')
        if abs(env.world.current_time-start_time-duration) > 4e-6:
            raise RuntimeError('Native time does not match the contact sequence')
        sampling = env._contact_sampling
        if sampling['sampled_substeps'] != steps or sampling['native_end_step_index']-sampling['native_start_step_index'] != steps:
            raise RuntimeError('Incomplete rigid contact acquisition')
        if assist and assist.active:
            raise RuntimeError('Completed contact sequence retained an assistance constraint')
        sampling['complete'] = True
        if capture:
            photo('final'); report['videos'] = {k: capture.encode_video(camera_id=k, fps=args.media_fps)
                                               for k in ('task', 'pinch')}
        env.stage.Export(str(out/'final.usda'))
        report.update(complete=True, native_steps=steps, duration_s=duration,
            native_rigid_contact_sampling=dict(sampling))
        exit_code = 0
    except BaseException as error:
        report['error'] = {'type': type(error).__name__, 'message': str(error), 'traceback': traceback.format_exc()}
        traceback.print_exc()
    finally:
        if assist:
            report['capture_assistance_events'] = assist.events
            report['capture_assistance_active_at_end'] = assist.active
            assist.close()
            report['capture_diagnostics'] = pin(out/'capture_diagnostics.jsonl')
        if log: log.close(); report['rigid_contacts'] = pin(out/'rigid_contacts.jsonl')
        if writer: report['states'] = writer.close()
        if capture: capture.close()
        dump(out/'native_report.json', report)
        print(json.dumps({'event': 'pinch_finished', 'complete': report['complete'], 'output': str(out)}), flush=True)
        if env:
            if exit_code:
                try: raise RuntimeError('Contact canary failed; preserve original evidence')
                finally: env.close()
            env.close()
    return exit_code


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='mode', required=True)
    p = commands.add_parser('plan')
    p.add_argument('--release', type=Path, required=True)
    p.add_argument('--side', choices=('left', 'right'), default='right')
    p.add_argument('--xy', type=float, nargs=2, required=True)
    p.add_argument('--yaw-deg', type=float, default=90.)
    p.add_argument('--pitch-deg', type=float, default=60.)
    p.add_argument('--bottom-z', type=float, default=.902)
    p.add_argument('--close-distance', type=float, default=.010)
    p.add_argument('--open-distance', type=float, default=.045)
    p.add_argument('--closure-lift', type=float, default=.003,
                   help='Wrist rise during closure, in metres; zero isolates tabletop capture.')
    p.add_argument('--capture-rise', type=float, default=0.,
                   help='Slow one-second wrist rise after closure, before the held five-centimetre lift.')
    p.add_argument('--closure-lift-start', type=float, default=0.,
                   help='Fraction of finger closure completed before beginning the simultaneous wrist rise.')
    p.add_argument('--output', type=Path, required=True)
    p = commands.add_parser('native')
    p.add_argument('--plan', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--media-output', type=Path)
    p.add_argument('--media-fps', type=int, default=4)
    p.add_argument('--capture-assist', action='store_true',
                   help='Enable declared wrist-following local assistance only after measured capture prerequisites.')
    args = parser.parse_args()
    if args.mode == 'native': return native(args)
    if args.output.exists(): raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    dump(args.output, plan(args.release, args.side, args.xy, args.yaw_deg, args.pitch_deg,
        args.bottom_z, args.close_distance, args.closure_lift, args.capture_rise,
        args.closure_lift_start, args.open_distance))
    print(str(args.output))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
