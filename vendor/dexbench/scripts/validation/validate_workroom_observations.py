"""Native wrist/third-person visibility under physical motion and room lighting.

Uses privileged instance IDs only for engineering measurements. Robot commands
are joint targets; original object visuals and contact physics are preserved.
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

from bench.isaacsim.release import dump
from bench.paths import load_json
from scripts.maintenance.freeze_project_sources import verify_if_snapshot
from scripts.validation.validate_render_sync import box_depth, erode
from scripts.validation.validate_scenario import digest, seal_report_before_shutdown
from scripts.validation.validate_workrooms import (
    assess_objects, check_window, plain, record, rotation, smooth_targets, stage_inventory)

THRESHOLDS = {'minimum_target_pixels': 16, 'minimum_visible_fraction': .60,
              'maximum_depth_p95_error_m': .008, 'minimum_depth_coverage': .90,
              'minimum_target_luma': 15., 'maximum_target_luma': 245.,
              'minimum_rgb_std': 3., 'minimum_hand_pixels': 16}
PROFILES = ('nominal', 'daylight', 'warm', 'low_light')


def instance_mask(ids, labels, prefix):
    selected = []
    for key, value in labels.items():
        # Native instance_id_segmentation_fast reports prim paths, not classes.
        if not isinstance(value, str):
            raise ValueError('Instance annotator must return prim-path labels')
        if value == prefix or value.startswith(prefix.rstrip('/')+'/'):
            selected.append(int(key))
    return np.isin(ids, selected)


def assess(rgb, depth, calibration, pose, half, target_mask, hand_mask, required):
    expected_depth = box_depth(calibration, pose, half)
    expected = expected_depth > 0
    predicted = int(expected.sum())
    visible = int((target_mask & expected).sum())
    fraction = visible/max(predicted, 1)
    interior = erode(target_mask & expected)
    valid = interior & np.isfinite(depth) & (depth > 0)
    errors = np.abs(depth[valid]-expected_depth[valid])
    p95 = float(np.percentile(errors, 95)) if errors.size else None
    luma = np.asarray(rgb, float) @ np.array([.2126, .7152, .0722])
    median = float(np.median(luma[target_mask])) if target_mask.any() else None
    coverage = float(valid.sum()/max(interior.sum(), 1))
    checks = {'image_has_information': float(luma.std()) >= THRESHOLDS['minimum_rgb_std']}
    if required:
        checks.update(target_resolved=visible >= THRESHOLDS['minimum_target_pixels'],
                      target_not_mostly_occluded=fraction >= THRESHOLDS['minimum_visible_fraction'],
                      target_depth_coverage=coverage >= THRESHOLDS['minimum_depth_coverage'],
                      target_depth_matches_physx=p95 is not None and p95 <= THRESHOLDS['maximum_depth_p95_error_m'],
                      target_exposure=median is not None and THRESHOLDS['minimum_target_luma'] <= median <= THRESHOLDS['maximum_target_luma'],
                      hand_visible=int(hand_mask.sum()) >= THRESHOLDS['minimum_hand_pixels'])
    return {'passed': all(checks.values()), 'checks': checks,
            'target_required': required, 'predicted_target_pixels': predicted,
            'visible_target_pixels': visible, 'target_visible_fraction': fraction,
            'hand_pixels': int(hand_mask.sum()), 'target_depth_coverage': coverage,
            'target_depth_p95_error_m': p95, 'target_median_luma': median,
            'luma_spatial_std': float(luma.std()), 'near_black_fraction': float((luma < 10).mean()),
            'saturated_fraction': float((rgb.min(axis=-1) > 250).mean())}


def run(args):
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[2]
    plan = {'schema': 'workroom_observation_plan.v1', 'family': args.family,
            'created_utc': datetime.now(timezone.utc).isoformat(),
            'thresholds': THRESHOLDS, 'profiles': PROFILES, 'resolution': [960, 720],
            'model_calls': 0, 'task_qualification': False,
            'scope': 'Physical arm motion, native RGB-D and instance visibility; no grasp score',
            'gpu_uuid': os.environ.get('CUDA_VISIBLE_DEVICES'),
            'driver_override': os.environ.get('ISAAC_ALLOW_UNVALIDATED_DRIVER'),
            'validator_sha256': digest(__file__)}
    dump(output/'plan.json', plan)
    env, error, stream = None, None, None
    frames, checks, steps = [], {}, 0
    try:
        from scenes.workrooms import scene
        from scenes.workrooms.observations import expected_camera_world, validation_wrist_poses
        from bench.control.factory import public_robot_geometry, robot_configuration
        from bench.simulation.isaacsim_backend.env import FR3DexterousIsaacEnv
        modules = ('scripts.validation.validate_workroom_observations',
                   'bench.simulation.isaacsim_backend.env', 'scenes.workrooms', 'bench.control.factory')
        verification = verify_if_snapshot(root, required_modules=modules, entrypoint_file=Path(__file__))
        if verification is None:
            raise ValueError('A complete frozen source snapshot is required')
        dump(output/'runtime_imports.json', verification)
        uuid = os.environ.get('CUDA_VISIBLE_DEVICES', '')
        if not uuid.startswith('GPU-') or ',' in uuid or os.environ.get('ISAAC_BENCH_SINGLE_CUDA_DEVICE') != '1':
            raise ValueError('Select one physical GPU UUID in single CUDA mode')
        spec = load_json(args.scene_json) if args.scene_json else scene(args.family)
        if args.scene_json:
            dump(output/'candidate_scene_source.json', {'path': str(args.scene_json.resolve()),
                                                       'sha256': digest(args.scene_json)})
        dump(output/'scene.json', spec)
        env = FR3DexterousIsaacEnv(spec, device=args.device, render_width=960, render_height=720)
        observed = env.reset(seed=0)
        from bench.simulation.assets.workrooms.common import apply_lighting_profile
        import omni.replicator.core as rep
        from PIL import Image
        from pxr import UsdGeom
        checks['stage_inventory'] = stage_inventory(env)
        home = env.action_spec()['initial_joint_positions_rad']
        initial = record(env, observed, 0, 'reset')
        previous = initial['state']
        records = []
        stream = (output/'physics.jsonl').open('x')

        def advance(start, goal, count, phase):
            nonlocal previous, steps
            offsets = {n: goal[n]-start[n] for n in start}
            batch = []
            print(json.dumps({'event': 'motion', 'phase': phase, 'steps': count}), flush=True)
            for i in range(count):
                command = {'joint_positions_rad': smooth_targets(start, offsets, (i+1)/count), 'duration_s': .02}
                obs = env.step(command)
                row = record(env, obs, 0, phase, command)
                check_window(obs, row['state'], previous, env.dt)
                stream.write(json.dumps(row, allow_nan=False, separators=(',', ':'))+'\n')
                previous = row['state']
                records.append(row)
                batch.append(row)
                steps += 1
                if (i+1) % 50 == 0:
                    print(json.dumps({'event': 'motion_progress', 'phase': phase, 'step': i+1}), flush=True)
            stream.flush()
            return batch

        settle = advance(home, home, 120, 'settle')
        paths = {n: row['path'] for n, row in env.scene_info['objects'].items()}
        checks['drop_and_settle'] = assess_objects(initial['observation'], settle, spec['workroom_diagnostics'], paths)
        name = 'cube_table'
        half = spec['workroom_diagnostics'][name]['half_extents_m']

        def capture(pose_name, profile):
            lighting = apply_lighting_profile(env.stage, args.family, profile, light_root='/World/Room/Lights')
            for camera in env.camera_ids:
                before = {k: np.asarray(v).copy() for k, v in env.capture_state().items()}
                frame = env.render_rgbd(camera)
                product, annotators = env._render_products[(camera, 960, 720)]
                if 'instance_id' not in annotators:
                    annotator = rep.AnnotatorRegistry.get_annotator('instance_id_segmentation_fast')
                    annotator.attach([product])
                    annotators['instance_id'] = annotator
                    for _ in range(4):
                        env.render_rgbd(camera)
                frame = env.render_rgbd(camera)
                segmented = annotators['instance_id'].get_data()
                ids = np.asarray(segmented['data']).reshape(720, 960).copy()
                labels = segmented['info']['idToLabels']
                target = instance_mask(ids, labels, paths[name])
                hand = np.zeros(ids.shape, bool)
                for key, value in labels.items():
                    if isinstance(value, str) and '/right_' in value and '/World/Robot' in value:
                        hand |= ids == int(key)
                pose = deepcopy(env.observation()['objects'][name])
                required = pose_name != 'home' and camera in ('overview', 'task', 'wrist')
                stats = assess(frame['rgb'], frame['depth_m'], frame['camera_calibration'], pose, half, target, hand, required)
                after = env.capture_state()
                preserved = all(np.array_equal(v, after[k]) for k, v in before.items())
                stats['checks']['render_preserves_state'] = preserved
                stats['passed'] = all(stats['checks'].values())
                label = f'{pose_name}_{profile}_{camera}'
                Image.fromarray(frame['rgb']).save(output/(label+'.png'))
                Image.fromarray((hand*255).astype(np.uint8)).save(output/(label+'_hand.png'))
                np.save(output/(label+'_depth.npy'), frame['depth_m'], allow_pickle=False)
                np.savez_compressed(output/(label+'_instances.npz'), instance_ids=ids)
                camera_prim = env.stage.GetPrimAtPath(env._camera_prim_path(camera))
                local = UsdGeom.Xformable(camera_prim).GetLocalTransformation()
                native_wrist = deepcopy(env.observation()['robot']['right_hand_flange'])
                if camera == 'wrist':
                    expected = expected_camera_world(env.cameras['wrist'], native_wrist)
                    actual = np.linalg.inv(frame['camera_calibration']['world_to_view'])
                    position_error = float(np.linalg.norm(expected[3, :3]-actual[3, :3]))
                    angle_error = float(np.arccos(np.clip((np.trace(expected[:3, :3] @ actual[:3, :3].T)-1)/2, -1, 1)))
                    stats['camera_native_position_error_m'] = position_error
                    stats['camera_native_angle_error_rad'] = angle_error
                    stats['checks']['camera_follows_native_wrist'] = position_error < .001 and angle_error < .005
                    stats['passed'] = all(stats['checks'].values())
                row = {'label': label, 'pose_name': pose_name, 'profile': profile, 'camera': camera,
                       'statistics': stats, 'object_pose': pose,
                       'physical_state': plain(after), 'lighting': lighting,
                       'camera_local_transform': np.asarray(local).tolist(),
                       'native_wrist_pose': native_wrist,
                       'instance_id_to_prim': labels,
                       'frame_metadata': {k: v for k, v in frame.items() if k not in ('rgb', 'depth_m')}}
                dump(output/(label+'.json'), row)
                frames.append(row)
                print(json.dumps({'event': 'frame', 'label': label, **stats}), flush=True)

        capture('home', 'nominal')
        geometry = public_robot_geometry(robot_configuration(spec))
        poses = validation_wrist_poses(env.observation()['objects'][name]['position_m'])
        current = deepcopy(home)
        motion = []
        for pose_name, wrist in poses.items():
            goal, diagnostic = geometry.solve(current, wrist={'frame_id': 'world', **wrist})
            count = max(80, math.ceil(1.875*max(abs(goal[n]-current[n]) for n in current)/.5/.02))
            motion.append({'name': pose_name, 'wrist': wrist, 'goal': goal, 'ik': diagnostic, 'steps': count})
            dump(output/'motion_plan.json', motion)
            advance(current, goal, count, pose_name)
            advance(goal, goal, 60, pose_name+'_settle')
            actual = env.observation()['robot']['right_hand_flange']
            position_error = float(np.linalg.norm(np.asarray(actual['position_m'])-wrist['position_m']))
            a, b = np.asarray(actual['quaternion_wxyz']), np.asarray(wrist['quaternion_wxyz'])
            angle_error = float(2*np.arccos(np.clip(abs(a @ b)/(np.linalg.norm(a)*np.linalg.norm(b)), 0, 1)))
            checks[pose_name+'_physical_pose'] = {'passed': position_error < .005 and angle_error < .04,
                                                'position_error_m': position_error, 'angle_error_rad': angle_error}
            capture(pose_name, 'nominal')
            if pose_name == 'low':
                for profile in PROFILES[1:]:
                    capture(pose_name, profile)
                apply_lighting_profile(env.stage, args.family, 'nominal', light_root='/World/Room/Lights')
            current = goal
        stream.close()
        checks['support_after_motion'] = assess_objects(initial['observation'], records, spec['workroom_diagnostics'], paths)
        wrist_frames = [f for f in frames if f['camera'] == 'wrist']
        local = np.asarray(wrist_frames[0]['camera_local_transform'])
        views = [np.asarray(f['frame_metadata']['camera_calibration']['world_to_view']) for f in wrist_frames]
        checks['rigid_wrist_mount'] = {'passed': all(np.allclose(f['camera_local_transform'], local, atol=1e-8) for f in wrist_frames)
                                      and max(np.linalg.norm(np.linalg.inv(v)[3, :3]-np.linalg.inv(views[0])[3, :3]) for v in views) > .03}
        for camera in ('overview', 'task', 'room'):
            views = [f['frame_metadata']['camera_calibration']['world_to_view'] for f in frames if f['camera'] == camera]
            checks[camera+'_fixed'] = {'passed': all(np.allclose(v, views[0], atol=1e-6) for v in views)}
        for camera in ('overview', 'task'):
            before = np.asarray(Image.open(output/f'home_nominal_{camera}_hand.png')) > 0
            after = np.asarray(Image.open(output/f'low_nominal_{camera}_hand.png')) > 0
            union = int((before | after).sum())
            iou = float((before & after).sum()/max(union, 1))
            checks[camera+'_hand_image_moves'] = {'passed': before.sum() >= 16 and after.sum() >= 16 and iou < .90,
                                                   'home_low_hand_mask_iou': iou}
        for camera in env.camera_ids:
            baseline_rgb = np.asarray(Image.open(output/f'low_nominal_{camera}.png'), float)
            baseline_depth = np.load(output/f'low_nominal_{camera}_depth.npy')
            for profile in PROFILES[1:]:
                rgb = np.asarray(Image.open(output/f'low_{profile}_{camera}.png'), float)
                depth = np.load(output/f'low_{profile}_{camera}_depth.npy')
                valid = (baseline_depth > 0) & np.isfinite(baseline_depth) & (depth > 0) & np.isfinite(depth)
                delta = float(np.abs(rgb-baseline_rgb).mean())
                depth_error = float(np.percentile(np.abs(depth[valid]-baseline_depth[valid]), 99)) if valid.any() else None
                checks[f'{camera}_{profile}_lighting_only'] = {
                    'passed': delta > .5 and depth_error is not None and depth_error < .001,
                    'mean_rgb_difference': delta, 'depth_difference_p99_m': depth_error}
        checks['all_frames'] = {'passed': all(f['statistics']['passed'] for f in frames), 'count': len(frames)}
        dump(output/'metadata_final.json', env.metadata())
    except BaseException as exc:
        error = {'type': type(exc).__name__, 'detail': str(exc), 'traceback': traceback.format_exc()}
        print(json.dumps({'event': 'error', **error}), flush=True)
    finally:
        if stream is not None:
            stream.close()
    report = {'schema': 'workroom_observations.v1', 'family': args.family,
              'passed': error is None and bool(checks) and all(c['passed'] for c in checks.values()),
              'checks': checks, 'error': error, 'control_steps': steps,
              'frames': [{k: f[k] for k in ('label', 'camera', 'pose_name', 'profile', 'statistics')} for f in frames],
              'model_calls': 0, 'task_qualification': False, 'object_state_writes_after_reset': False,
              'plan_sha256': digest(output/'plan.json'),
              'artifacts_sha256': {str(p.relative_to(output)): digest(p) for p in output.rglob('*') if p.is_file()}}
    return seal_report_before_shutdown(output, plain(report), env)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--family', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--scene-json', type=Path, help='Explicit candidate override, recorded by path and hash')
    report = run(parser.parse_args())
    if not report['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
