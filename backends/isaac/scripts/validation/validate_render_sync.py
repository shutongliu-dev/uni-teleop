"""Check native pixels against a freely falling body's measured PhysX pose.

Run via scripts/isaac_python.sh from a complete, sealed source snapshot. A
visual-only magenta box marks an existing diagnostic body; its mass, colliders
and dynamics are unchanged. Only reset initializes poses; step sends joint
targets. This is a rendering diagnostic, not a task or material qualification.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import traceback

import numpy as np

from bench.isaacsim.release import dump
from scripts.maintenance.freeze_project_sources import verify_if_snapshot
from scripts.validation.validate_scenario import digest, seal_report_before_shutdown
from scripts.validation.validate_workrooms import plain, rotation


HALF_EXTENTS = [.035, .035, .035]
CAPTURE_STEPS = (0, 4, 7, 10, 13, 20)
THRESHOLDS = {'minimum_mask_pixels': 64, 'minimum_silhouette_iou': .80,
              'maximum_depth_p95_error_m': .004, 'minimum_depth_pixels': 40,
              'minimum_depth_coverage': .90, 'minimum_pose_change_m': .015,
              'minimum_rejected_stale_poses': 3, 'minimum_drop_m': .15}


def box_depth(calibration, pose, half_extents=HALF_EXTENTS):
    """Ray/OBB intersection using RTX's actual conformed projection matrix.

Depth parameter is optical z, since each camera-space ray has z=-1. Pixel
centers are (u+.5,v+.5). A zero is a ray that misses the diagnostic box.
"""
    width, height = calibration['resolution']
    projection = np.asarray(calibration['projection'], dtype=float)
    view = np.asarray(calibration['world_to_view'], dtype=float)
    if (not np.isclose(projection[2, 3], -1) or not np.isclose(projection[3, 3], 0)
            or not np.isclose(calibration['meters_per_scene_unit'], 1.)):
        raise ValueError('Probe requires RTX pinhole projection in a meter stage')
    camera_to_world = np.linalg.inv(view)
    rows, columns = np.mgrid[:height, :width]
    x = (2*(columns+.5)/width - 1 + projection[2, 0])/projection[0, 0]
    y = (1 - 2*(rows+.5)/height + projection[2, 1])/projection[1, 1]
    directions = np.stack((x, y, -np.ones_like(x)), axis=-1) @ camera_to_world[:3, :3]
    origin = camera_to_world[3, :3]
    orientation = rotation(pose['quaternion_wxyz'])
    local_origin = (origin-np.asarray(pose['position_m'])) @ orientation
    local_directions = directions @ orientation
    half = np.asarray(half_extents)
    parallel = np.abs(local_directions) < 1e-12
    safe = np.where(parallel, 1., local_directions)
    a, b = (-half-local_origin)/safe, (half-local_origin)/safe
    near = np.where(parallel, -np.inf, np.minimum(a, b)).max(axis=-1)
    far = np.where(parallel, np.inf, np.maximum(a, b)).min(axis=-1)
    outside = np.any(parallel & (np.abs(local_origin) > half), axis=-1)
    valid = (~outside) & (far >= near) & (near > 0)
    return np.where(valid, near, 0.)


def magenta_mask(rgb):
    values = np.asarray(rgb, dtype=float)
    r, g, b = values[..., 0], values[..., 1], values[..., 2]
    return (r > 100) & (b > 100) & (g < .60*np.minimum(r, b))


def erode(mask):
    padded = np.pad(mask, 1, constant_values=False)
    return np.logical_and.reduce([padded[dy:dy+mask.shape[0], dx:dx+mask.shape[1]]
                                  for dy in range(3) for dx in range(3)])


def assess_frame(rgb, depth, calibration, pose):
    expected_depth = box_depth(calibration, pose)
    expected, actual = expected_depth > 0, magenta_mask(rgb)
    union = int(np.count_nonzero(expected | actual))
    intersection = int(np.count_nonzero(expected & actual))
    iou = intersection/union if union else 0.
    interior = erode(expected & actual)
    valid = interior & np.isfinite(depth) & (depth > 0)
    count = int(np.count_nonzero(valid))
    errors = np.abs(depth[valid]-expected_depth[valid])
    p95 = float(np.percentile(errors, 95)) if count else None
    coverage = count/max(int(np.count_nonzero(erode(expected))), 1)
    checks = {'visible_prediction': int(np.count_nonzero(expected)) >= THRESHOLDS['minimum_mask_pixels'],
              'rgb_silhouette': iou >= THRESHOLDS['minimum_silhouette_iou'],
              'depth_pixels': count >= THRESHOLDS['minimum_depth_pixels'],
              'depth_coverage': coverage >= THRESHOLDS['minimum_depth_coverage'],
              'depth_matches_physx_surface': p95 is not None and p95 <= THRESHOLDS['maximum_depth_p95_error_m']}
    return {'passed': all(checks.values()), 'checks': checks, 'silhouette_iou': iou,
            'expected_pixels': int(np.count_nonzero(expected)),
            'magenta_pixels': int(np.count_nonzero(actual)), 'depth_pixels': count,
            'depth_coverage': coverage, 'depth_p95_error_m': p95,
            'depth_max_error_m': float(errors.max()) if count else None}


def author_visual_fiducial(env, object_name):
    """Local diagnostic visuals only; leave all physics schemas untouched."""
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade
    root = env.stage.GetPrimAtPath(env.scene_info['objects'][object_name]['path'])
    hidden = []
    for prim in Usd.PrimRange(root):
        if prim.IsA(UsdGeom.Gprim):
            UsdGeom.Imageable(prim).CreateVisibilityAttr(UsdGeom.Tokens.invisible)
            hidden.append(str(prim.GetPath()))
    marker = UsdGeom.Cube.Define(env.stage, str(root.GetPath())+'/render_sync_fiducial')
    marker.CreateSizeAttr(2.)
    marker.AddScaleOp().Set(Gf.Vec3f(*HALF_EXTENTS))
    material = UsdShade.Material.Define(env.stage, '/World/Looks/render_sync_magenta')
    shader = UsdShade.Shader.Define(env.stage, str(material.GetPath())+'/Shader')
    shader.CreateIdAttr('UsdPreviewSurface')
    shader.CreateInput('diffuseColor', Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0., 0., 0.))
    shader.CreateInput('emissiveColor', Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(1., 0., 1.))
    shader.CreateInput('roughness', Sdf.ValueTypeNames.Float).Set(1.)
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), 'surface')
    UsdShade.MaterialBindingAPI.Apply(marker.GetPrim()).Bind(material)
    return {'prim': str(marker.GetPath()), 'half_extents_m': HALF_EXTENTS,
            'hidden_original_visual_prims': hidden, 'local_position_m': [0., 0., 0.],
            'purpose': 'Visual-only analytic fiducial on actual dynamic body; no collider or mass changes'}


def run(args):
    root = Path(__file__).resolve().parents[2]
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    env, error = None, None
    frames, records, checks = [], [], {}
    plan = {'schema': 'native_render_sync_plan_v1', 'created_utc': datetime.now(timezone.utc).isoformat(),
            'family': args.family, 'capture_control_steps': CAPTURE_STEPS, 'thresholds': THRESHOLDS,
            'model_calls': 0, 'task_qualification': False, 'object_state_writes': 'reset only',
            'source_root': str(root), 'validator_sha256': digest(__file__),
            'device': args.device, 'gpu_uuid': os.environ.get('CUDA_VISIBLE_DEVICES'),
            'driver_override': os.environ.get('ISAAC_ALLOW_UNVALIDATED_DRIVER'),
            'scope': 'Actual native RGB silhouette and RGB-D surface vs PhysX pose; previous-pose negative controls'}
    dump(output/'plan.json', plan)
    modules = ('scripts.validation.validate_render_sync', 'bench.simulation.isaacsim_backend.env',
               'bench.simulation.isaacsim_backend.render_sync', 'scenes.workrooms')
    try:
        from scenes.workrooms import scene
        from bench.simulation.isaacsim_backend.env import FR3DexterousIsaacEnv
        from bench.simulation.isaacsim_backend import render_sync  # included in import verification
        verification = verify_if_snapshot(root, required_modules=modules, entrypoint_file=Path(__file__))
        if verification is None:
            raise ValueError('Native render check requires a complete frozen source snapshot')
        dump(output/'runtime_imports.json', verification)
        uuid = os.environ.get('CUDA_VISIBLE_DEVICES', '')
        if (args.device != 'cuda:0' or not uuid.startswith('GPU-') or ',' in uuid
                or os.environ.get('ISAAC_BENCH_SINGLE_CUDA_DEVICE') != '1'):
            raise ValueError('Use one physical GPU UUID, single CUDA mode and --device cuda:0')
        spec = scene(args.family, diagnostic=True)
        if args.cpu_rigid:
            spec['native_readback_profile'] = 'cpu_rigid_v1'
        name = 'cube_table'
        spec['objects'] = [o for o in spec['objects'] if o['id'] == name]
        spec['cameras'] = {'probe': {'kind': 'world', 'eye': [.8, -1.3, 1.4],
            'target': [.30, -.35, .97], 'focal_length_mm': 32., 'horizontal_aperture_mm': 36.,
            'resolution': [640, 480]}}
        dump(output/'scene.json', spec)
        env = FR3DexterousIsaacEnv(spec, device=args.device, render_width=640, render_height=480)
        observation = env.reset(seed=0)
        dump(output/'fiducial.json', author_visual_fiducial(env, name))
        env.stage.GetRootLayer().Export(str(output/'diagnostic_stage.usda'))
        dump(output/'metadata.json', env.metadata())
        home = env.action_spec()['initial_joint_positions_rad']
        previous = None

        def capture(label):
            nonlocal previous
            from PIL import Image
            pose = deepcopy(env.observation()['objects'][name])
            before_rgbd = {key: np.asarray(value).copy() for key, value in env.capture_state().items()}
            frame = env.render_rgbd('probe', width=640, height=480)
            after_rgbd = {key: np.asarray(value).copy() for key, value in env.capture_state().items()}
            Image.fromarray(frame['rgb']).save(output/f'{label}.png')
            np.save(output/f'{label}_depth.npy', frame['depth_m'], allow_pickle=False)
            predicted = box_depth(frame['camera_calibration'], pose)
            np.save(output/f'{label}_expected_depth.npy', predicted.astype(np.float32), allow_pickle=False)
            Image.fromarray((magenta_mask(frame['rgb'])*255).astype(np.uint8)).save(output/f'{label}_mask.png')
            actual = assess_frame(frame['rgb'], frame['depth_m'], frame['camera_calibration'], pose)
            stale = None
            if previous is not None:
                displacement = float(np.linalg.norm(np.asarray(pose['position_m'])-previous['position_m']))
                if displacement >= THRESHOLDS['minimum_pose_change_m']:
                    stale = {'pose': previous, 'pose_displacement_m': displacement,
                             **assess_frame(frame['rgb'], frame['depth_m'], frame['camera_calibration'], previous)}
            row = {'label': label, 'pose': pose, 'physical_state': plain(env.capture_state()),
                   'current_pose': actual, 'previous_pose_negative_control': stale,
                   'frame_metadata': {k: v for k, v in frame.items() if k not in ('rgb', 'depth_m')}}
            # The legacy RGB entry point must preserve every physical array too.
            raw = env.render('probe', width=640, height=480)
            after_rgb = {key: np.asarray(value).copy() for key, value in env.capture_state().items()}
            state_file = output/f'{label}_render_states.npz'
            np.savez_compressed(state_file, **{
                prefix+'__'+key: value
                for prefix, state in (('before_rgbd', before_rgbd), ('after_rgbd', after_rgbd),
                                      ('after_rgb', after_rgb))
                for key, value in state.items()})
            row['render_state_arrays'] = state_file.name
            row['raw_rgb_current_pose'] = assess_frame(raw, frame['depth_m'], frame['camera_calibration'], pose)
            Image.fromarray(raw).save(output/f'{label}_raw_rgb.png')
            dump(output/f'{label}.json', row)
            frames.append(row)
            previous = pose
            print(json.dumps({'event': 'render_sync_frame', 'label': label,
                              'current': actual, 'stale_rejected': stale is not None and not stale['passed']}), flush=True)

        capture('step_000')
        records.append({'control_step': 0, 'observation': observation})
        for step in range(1, max(CAPTURE_STEPS)+1):
            command = {'joint_positions_rad': deepcopy(home), 'duration_s': .02}
            observation = env.step(command)
            records.append({'control_step': step, 'command': command, 'observation': observation})
            if step in CAPTURE_STEPS:
                capture(f'step_{step:03d}')
        # A reset is the permitted state-write boundary; capture must immediately
        # show the reinitialized high pose, rather than the old resting image.
        env.reset(seed=0)
        capture('reset_again')
        dump(output/'physics.json', records)
        stale = [f['previous_pose_negative_control'] for f in frames if f['previous_pose_negative_control']]
        descent = records[0]['observation']['objects'][name]['position_m'][2]-min(
            r['observation']['objects'][name]['position_m'][2] for r in records)
        checks = {'actual_gravity_motion': descent >= THRESHOLDS['minimum_drop_m'],
                  'all_rgbd_match_current_physx_pose': all(f['current_pose']['passed'] for f in frames),
                  'all_raw_rgb_match_current_physx_pose': all(f['raw_rgb_current_pose']['passed'] for f in frames),
                  'stale_pose_controls_exercised': len(stale) >= THRESHOLDS['minimum_rejected_stale_poses'],
                  'every_stale_pose_rejected': bool(stale) and all(not r['passed'] for r in stale)}
        dump(output/'metadata_final.json', env.metadata())
        dump(output/'runtime_imports_final.json', verify_if_snapshot(root, required_modules=modules, entrypoint_file=Path(__file__)))
    except BaseException as exc:
        error = {'type': type(exc).__name__, 'detail': str(exc), 'traceback': traceback.format_exc()}
        print(json.dumps({'event': 'render_sync_error', **error}), flush=True)
    report = {'schema': 'native_render_sync_check_v1', 'passed': error is None and bool(checks) and all(checks.values()),
              'checks': checks, 'error': error, 'frames': frames, 'model_calls': 0,
              'scope': plan['scope'], 'object_state_writeback_after_reset': False,
              'rendering_preserved_physical_state': bool(frames) and error is None,
              'thresholds': THRESHOLDS, 'plan_sha256': digest(output/'plan.json'),
              'finished_utc': datetime.now(timezone.utc).isoformat(),
              'artifacts_sha256': {str(p.relative_to(output)): digest(p) for p in output.rglob('*') if p.is_file()}}
    return seal_report_before_shutdown(output, report, env)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--family', default='mechanical_assembly')
    parser.add_argument('--cpu-rigid', action='store_true', help='Validate explicit CPU rigid physics with GPU rendering')
    args = parser.parse_args()
    if not run(args)['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
