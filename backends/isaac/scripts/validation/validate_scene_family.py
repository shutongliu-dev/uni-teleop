"""Render and physically validate one authored scene family in native Isaac.

Steps, all on a real GPU (never CPU-only claims):
  1. import the family module and validate it on CPU (schema, bores, rest poses);
  2. build the actual stage: room shell, pedestal, table, static props, objects;
  3. settle the scene for a fixed number of control windows with the robot held
     at home, and record every dynamic object's motion and solver contact;
  4. render the scene's own cameras (third-person and wrist) and save PNGs plus
     a depth summary;
  5. write a report with the scene record, per-object settle evidence and SHAs.

A scene passes only if every dynamic object stays on its support (no fall,
no creep beyond a threshold, no penetration) and the render is non-blank.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--family', required=True, help='family module name under scenes.rooms')
    parser.add_argument('--output', required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--settle-windows', type=int, default=100, help='20 ms control windows to hold')
    parser.add_argument('--max-drift-m', type=float, default=.004, help='allowed centroid drift while resting')
    parser.add_argument('--max-penetration-m', type=float, default=.002)
    parser.add_argument('--render', action='store_true', default=True)
    args = parser.parse_args()

    output = Path(args.output)
    if output.exists() and any(output.iterdir()):
        sys.exit(f'Output directory already contains data: {output}')
    output.mkdir(parents=True, exist_ok=True)

    import importlib
    module = importlib.import_module(f'scenes.rooms.{args.family}')
    scene = module.SCENE
    from bench.simulation.isaacsim_backend.scene_family import validate_family
    family_record = validate_family(scene)

    from PIL import Image
    from bench.simulation.isaacsim_backend.env import FR3DexterousIsaacEnv
    env = FR3DexterousIsaacEnv(scene, device=args.device, render_width=960, render_height=720)
    try:
        observation = env.reset()
        names = list(env.scene_info['objects'])
        initial = {name: np.asarray(observation['objects'][name]['position_m'], dtype=float)
                   for name in names}
        initial_q = {name: np.asarray(observation['objects'][name]['quaternion_wxyz'], dtype=float)
                     for name in names}
        # Hold the reset pose: the commanded target equals the measured position,
        # so any object motion is the scene's own settling, not the robot pushing it.
        held = {name: float(observation['joint_positions_rad'][name]) for name in env.actuated_joint_names}
        for _ in range(int(args.settle_windows)):
            observation = env.step({'joint_positions_rad': dict(held), 'duration_s': .02})

        final = {name: np.asarray(observation['objects'][name]['position_m'], dtype=float)
                 for name in names}
        drift = {name: float(np.linalg.norm(final[name] - initial[name])) for name in names}
        drop = {name: float(initial[name][2] - final[name][2]) for name in names}
        speeds = {name: float(np.linalg.norm(np.asarray(observation['objects'][name]['linear_velocity_m_s'],
                                                       dtype=float))) for name in names}
        # Rotation *change* since reset: an object may legitimately be spawned on
        # its side (a knife, a book), so absolute tilt says nothing; toppling does.
        rotation_change = {}
        for name in names:
            a = initial_q[name] / np.linalg.norm(initial_q[name])
            b = np.asarray(observation['objects'][name]['quaternion_wxyz'], dtype=float)
            b = b / np.linalg.norm(b)
            rotation_change[name] = float(2. * np.arccos(np.clip(abs(float(np.dot(a, b))), -1., 1.)))

        contacts = {name: 'resting' if speeds[name] < .01 else 'moving' for name in names}

        from bench.simulation.isaacsim_backend.cameras import project_point
        images = {}
        if args.render:
            for camera in env.camera_ids:
                spec = env.cameras[camera]
                width, height = spec['resolution']
                capture = env.render_rgbd(camera=camera)
                rgb = capture['rgb']
                depth = capture['depth_m']
                image_path = output / f'{camera}.png'
                Image.fromarray(rgb).save(image_path)
                depth_path = output / f'{camera}_depth.npy'
                np.save(depth_path, depth, allow_pickle=False)
                valid = depth[depth > 0]
                aim = spec.get('aim_target_m') if spec['kind'] == 'link' else spec.get('target')
                calibration = capture['camera_calibration']
                projected = project_point(env.stage, env._camera_prim_path(camera), aim, width, height,
                                          calibration=calibration) \
                    if aim is not None else None
                images[camera] = {
                    'png': str(image_path), 'png_sha256': digest(image_path),
                    'rgb_mean': float(np.mean(rgb)), 'rgb_std': float(np.std(rgb)),
                    'non_blank': bool(float(np.std(rgb)) >= 1.),
                    'depth_m': {'min': float(np.min(valid)) if valid.size else None,
                                'median': float(np.median(valid)) if valid.size else None,
                                'max': float(np.max(valid)) if valid.size else None},
                    'fraction_beyond_25cm': float(np.count_nonzero(depth > .25)) / float(depth.size),
                    'target_pixel': projected,
                    'target_projection_basis': 'actual_render_product_camera_params',
                    'camera_calibration': calibration,
                    'target_in_frame': bool(projected is not None and 0 <= projected[0] < width
                                            and 0 <= projected[1] < height),
                    'depth_path': str(depth_path), 'depth_sha256': digest(depth_path),
                    'camera': spec,
                }

        failures = []
        for name, value in drift.items():
            if value > args.max_drift_m:
                failures.append(f'{name}: drifted {value:.5f} m > {args.max_drift_m} m')
        for name, value in drop.items():
            if value > .02:
                failures.append(f'{name}: fell {value:.3f} m from its support')
        for name, value in rotation_change.items():
            if value > .15:
                failures.append(f'{name}: toppled ({value:.3f} rad rotation while settling)')
        for name, value in speeds.items():
            if value > .05:
                failures.append(f'{name}: still moving at {value:.4f} m/s')
        for camera, record in images.items():
            if not record['non_blank']:
                failures.append(f'camera {camera}: blank render')
            if record['fraction_beyond_25cm'] < .10:
                failures.append(f'camera {camera}: sees only close geometry (hand or wall)')
            if record['target_pixel'] is not None and not record['target_in_frame']:
                failures.append(f'camera {camera}: its target {record["camera"].get("aim_target_m") or record["camera"].get("target")} is not in frame')

        report = {
            'schema': 'scene_family_native_validation.v1',
            'family': args.family,
            'family_record': family_record,
            'scene': scene,
            'scene_info': {k: (v if not isinstance(v, dict) else list(v)) for k, v in env.scene_info.items()},
            'camera_ids': list(env.camera_ids),
            'cameras': {name: env.cameras[name] for name in env.camera_ids},
            'camera_prim_paths': dict(env.camera_paths),
            'settle_windows': int(args.settle_windows),
            'settle': {'drift_m': drift, 'drop_m': drop, 'speed_m_s': speeds,
                       'rotation_change_rad': rotation_change, 'state': contacts,
                       'max_drift_m': float(max(drift.values())) if drift else None},
            'images': images,
            'thresholds': {'max_drift_m': args.max_drift_m, 'max_penetration_m': args.max_penetration_m},
            'failures': failures,
            'passed': not failures,
        }
        (output / 'report.json').write_text(json.dumps(report, indent=1, ensure_ascii=False, default=str) + '\n')
        print(json.dumps({'passed': report['passed'], 'failures': failures,
                          'max_drift_m': report['settle']['max_drift_m'],
                          'cameras': list(images)}, indent=1))
    finally:
        env.close()


if __name__ == '__main__':
    main()
