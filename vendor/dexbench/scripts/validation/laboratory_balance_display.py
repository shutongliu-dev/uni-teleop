"""Native balance readout from independent reset fixtures and real RTX frames.

Run each --case in a fresh process from sealed source using isaac_python.sh.
The only object pose changes are the explicitly recorded reset USD derivation;
the real frozen observer then reads passive GPU dynamics. No readout values are
injected, no robot task score is claimed, and rendering must leave all native
state arrays and the observer unchanged.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import importlib
import json
import os
from pathlib import Path
import traceback

import numpy as np

from bench.paths import OUTPUT_ROOT

KIT = 'bench/apps/workroom.headless.kit'
CAMERA = 'balance_diagnostic'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def derive_fixture(scene, task, output, case, frozen_rules):
    """Copy the task subtree; change declared reset translations before launch."""
    from pxr import Gf, Usd, UsdGeom
    from bench.isaacsim.release import digest, dump
    from scenes.workrooms.laboratory.composition import bind_episode_task
    scene, task = deepcopy(scene), deepcopy(task)
    bundle = scene['native_task_bundle']
    source = bundle['usd_path']
    stage = Usd.Stage.CreateInMemory()
    stage.DefinePrim(bundle['root_prim_path'], 'Xform').GetReferences().AddReference(source, bundle['root_prim_path'])
    overrides = {}
    if case != 'no_cup':
        overrides['weighing_cup'] = [.71, 2.075, .9255]
    if case == 'six_grams':
        for index in range(12):
            overrides[f'pellet_{index:02}'] = [.695+.01*(index//3), 2.075+.01*(index%3-1), .9325]
    for row in task['asset_metadata']['objects']:
        if row['id'] in overrides:
            position = overrides[row['id']]
            prim = stage.GetPrimAtPath(row['body_path'])
            translates = [op for op in UsdGeom.Xformable(prim).GetOrderedXformOps()
                          if op.GetOpType() == UsdGeom.XformOp.TypeTranslate]
            require(len(translates) == 1, 'Expected one task-body translation for reset derivation')
            translates[0].Set(Gf.Vec3d(*position))
            row['initial_position_m'] = position
            row['initial_pose_xyzw'][:3] = position
    path = output/'fixture.usda'
    stage.Flatten().Export(str(path))
    task = bind_episode_task(task, task['episode_seed'], frozen_rules)
    bundle.update(usd_path=str(path), usd_sha256=digest(path))
    bundle['observer']['configuration'] = deepcopy(task)
    dump(output/'fixture_task.json', task)
    dump(output/'fixture_scene.json', scene)
    dump(output/'reset_derivation.json', {'source_usd_path': source,
        'source_usd_sha256': digest(Path(source)), 'derived_usd_sha256': digest(path),
        'object_translation_overrides_m': overrides, 'post_reset_pose_writes': False,
        'collision_mass_material_control_changes': False,
        'scope': 'Independent initial conditions; no canonical reset or robot task qualification.'})
    return scene, task


def run(args):
    from bench.isaacsim.release import _frozen_evaluator_module, digest, dump, task_inputs
    from bench.simulation.isaacsim_backend.env import FR3DexterousIsaacEnv
    from bench.simulation.isaacsim_backend.cameras import project_point
    from scripts.maintenance.freeze_project_sources import verify_if_snapshot
    from scripts.validation.board_game_runtime_smoke import gpu_selection
    from scripts.validation.living_television_visual import actual_cuda_identity, same_physics
    # This fixture dependency is required by the first sealed-source check;
    # derive_fixture() otherwise imports it only after that check has run.
    importlib.import_module('scenes.workrooms.laboratory.composition')
    project = Path(__file__).resolve().parents[2]
    output, media = args.output.resolve(), args.media_output.resolve()
    require(output.is_relative_to(OUTPUT_ROOT), 'Evidence belongs in shared datasets')
    require(media.is_relative_to(OUTPUT_ROOT), 'Media belongs under demo/interface/scenes')
    output.mkdir(parents=True, exist_ok=False)
    media.mkdir(parents=True, exist_ok=False)
    report = {'schema': 'laboratory_balance_native_display.v1', 'passed': False, 'case': args.case,
        'task_success_claimed': False, 'model_calls': 0, 'post_reset_object_pose_writes': 0,
        'scope': __doc__, 'captures': []}
    env = None
    modules = ('scripts.validation.laboratory_balance_display', 'bench.simulation.isaacsim_backend.env',
               'scripts.validation.living_television_visual', 'scenes.workrooms.laboratory.composition')
    try:
        source = verify_if_snapshot(project, required_modules=modules, entrypoint_file=Path(__file__))
        require(source is not None and Path.cwd().resolve() == project, 'Run from the complete frozen source root')
        require(Path(os.environ.get('ISAAC_BENCH_EXPERIENCE', '')).resolve() == project/KIT,
                'Select the frozen headless Kit experience')
        report['source'] = source
        report['source_manifest_sha256'] = digest(project/'source_snapshot_manifest.json')
        report['gpu'] = gpu_selection(args.gpu_uuid)
        report['environment'] = {key: os.environ.get(key) for key in ('CUDA_VISIBLE_DEVICES',
            'ISAAC_BENCH_SINGLE_CUDA_DEVICE', 'ISAAC_ALLOW_UNVALIDATED_DRIVER', 'ISAAC_BENCH_EXPERIENCE')}
        _, scene, task, initial = task_inputs(args.release, 'solid_weighing', seed=0)
        require(task['public_rules']['schema'] == 'laboratory_rules.v2'
                and scene['native_task_bundle'].get('visual_state_bindings'), 'Release lacks the frozen balance display')
        evaluator = _frozen_evaluator_module(args.release.resolve())
        frozen_rules = importlib.import_module(evaluator.__package__+'.rules')
        report['release'] = {'path': str(args.release.resolve()),
            'manifest_sha256': digest(args.release/'release_manifest.json')}
        class BalanceFixtureEnv(FR3DexterousIsaacEnv):
            def _initialize(self):
                # The base constructor has started SimulationApp, so pxr is
                # available. Derive reset poses before any scene/world/reset
                # exists; shared runtime and all post-reset sampling stay intact.
                self.scene_spec, self.fixture_task = derive_fixture(
                    self.scene_spec, task, output, args.case, frozen_rules)
                super()._initialize()

        env = BalanceFixtureEnv(scene, device='cuda:0', render_width=args.width, render_height=args.height)
        observation = env.reset(seed=0, initial_state=initial)
        report['actual_cuda_device'] = actual_cuda_identity(args.gpu_uuid)
        dump(output/'native_metadata.json', env.metadata())
        samples = [observation]
        for _ in range(round(args.duration/.02)):
            observation = env.step({'joint_positions_rad': initial['joint_positions_rad'], 'duration_s': .02})
            require(observation['task_state']['complete'] and not observation['task_state']['errors'],
                    'Real frozen observer rejected the native fixture')
            samples.append(observation)
        with (output/'observations.jsonl').open('w') as file:
            for row in samples:
                file.write(json.dumps(row, allow_nan=False)+'\n')
        state = observation['task_state']
        display = state['detail']['balance_display']
        expected = {'no_cup': None, 'empty': 0., 'six_grams': 6.}[args.case]
        require(display['value_g'] == expected and display['valid'] is (expected is not None),
                'Native mass readout differs from the declared reset fixture: '+str(display))
        require(state['released_stable_s'] == 0., 'Reset fixture must not acquire task-completion credit')
        report['native_readout'] = display
        report['physics_steps'] = round(args.duration/env.dt)
        report['observer_state'] = state
        before = {key: np.array(value, copy=True) for key, value in env.capture_state().items()}
        np.savez_compressed(output/'before_render.npz', **before)
        env.cameras[CAMERA] = {'id': CAMERA, 'kind': 'world',
            'eye': [.71, 1.82, 1.15], 'target': [.71, 2.045, .925],
            'focal_length_mm': 48., 'horizontal_aperture_mm': 36.,
            'clipping_m': [.01, 10.], 'resolution': [args.width, args.height],
            'role': 'operator_balance_display_diagnostic'}
        env.camera_ids.append(CAMERA)
        for _ in range(4):
            frame = env.render_rgbd(camera=CAMERA, width=args.width, height=args.height)
            require(env._task_reader.observe() == state, 'Rendering modified the real observer')
            same_physics(before, env.capture_state())
        receipt = frame['task_visual_state']
        require(receipt['source_native_step'] == state['native_step'], 'Display native tick is stale')
        from pxr import UsdGeom, UsdShade
        cache = UsdGeom.XformCache()
        segment_pixels = []
        for row in receipt['bindings']:
            require(row['value'] is display['segments'][row['state_path'][-1]], 'Digit segment differs from observer')
            prim = env.stage.GetPrimAtPath(row['target_prim_path'])
            material = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()[0]
            require(str(material.GetPath()) == row['material_path'], 'Composed segment material differs from receipt')
            if '/Readout/Top/digit_' in row['target_prim_path']:
                center = list(cache.GetLocalToWorldTransform(prim).ExtractTranslation())
                pixel = project_point(None, None, center, args.width, args.height,
                                      calibration=frame['camera_calibration'])
                require(pixel is not None, 'A displayed digit segment is behind the camera')
                x, y = (int(round(v)) for v in pixel)
                require(1 <= x < args.width-1 and 1 <= y < args.height-1,
                        'A displayed digit segment is outside the RGB-D frame')
                patch = frame['rgb'][y-1:y+2, x-1:x+2].astype(float)
                green_contrast = float(np.mean(patch[:, :, 1]-(patch[:, :, 0]+patch[:, :, 2])/2))
                calibration = frame['camera_calibration']
                units = float(calibration['meters_per_scene_unit'])
                point = np.r_[np.asarray(center)/units, 1.]
                expected_depth = -float((point @ np.asarray(calibration['world_to_view']))[2])*units
                measured_depth = float(frame['depth_m'][y, x])
                require(np.isfinite(measured_depth) and abs(measured_depth-expected_depth) <= .01,
                        'A digit segment is occluded or has stale same-frame depth')
                segment_pixels.append({'id': row['id'], 'lit': row['value'], 'pixel_xy': [x, y],
                    'mean_rgb': patch.mean(axis=(0, 1)).tolist(), 'green_contrast_0_255': green_contrast,
                    'expected_camera_depth_m': expected_depth, 'measured_camera_depth_m': measured_depth})
        require(len(segment_pixels) == 21, 'Expected all three top-screen digits in the RGB-D check')
        lit = [row['green_contrast_0_255'] for row in segment_pixels if row['lit']]
        dark = [row['green_contrast_0_255'] for row in segment_pixels if not row['lit']]
        contrast = float(np.median(lit)-np.median(dark))
        require(contrast >= 12., 'RTX digit pixels do not distinguish lit and dark segments')
        report['pixel_evidence'] = {'passed': True, 'top_digit_segments': segment_pixels,
            'median_lit_minus_dark_green_contrast_0_255': contrast, 'minimum_contrast_0_255': 12.,
            'depth_tolerance_m': .01,
            'scope': 'Same-frame digit-center RGB/depth, engineering visibility check; not OCR or photometric calibration.'}
        from PIL import Image
        rgb = media/(args.case+'.png')
        Image.fromarray(frame['rgb']).save(rgb)
        # Project the real top screen corners from the same-frame RTX matrices.
        corners = [[x, y, .92249] for x in (.654, .766) for y in (2.0137, 2.0287)]
        pixels = [project_point(None, None, point, args.width, args.height,
                               calibration=frame['camera_calibration']) for point in corners]
        require(all(point is not None for point in pixels), 'Top display is behind the camera')
        pixels = np.asarray(pixels)
        lo = np.maximum(np.floor(pixels.min(axis=0))-5, 0).astype(int)
        hi = np.minimum(np.ceil(pixels.max(axis=0))+5, [args.width, args.height]).astype(int)
        require(hi[0]-lo[0] >= 100 and hi[1]-lo[1] >= 12, 'Display has too few pixels to inspect')
        crop = media/(args.case+'_screen.png')
        Image.fromarray(frame['rgb']).crop((*lo, *hi)).save(crop)
        np.save(output/'depth_m.npy', frame['depth_m'])
        capture = {'rgb_path': str(rgb), 'rgb_sha256': digest(rgb), 'crop_path': str(crop),
            'crop_sha256': digest(crop), 'camera_calibration': frame['camera_calibration'],
            'roi_ltrb_px': [int(v) for v in [*lo, *hi]], 'task_visual_state': receipt,
            'render_synchronization': frame['render_synchronization'],
            'physics_unchanged': True, 'observer_unchanged': True,
            'operator_camera_is_agent_observation': False}
        report['captures'].append(capture)
        dump(output/'capture.json', capture)
        report['passed'] = True
    except BaseException as exc:
        report['failure'] = {'type': type(exc).__name__, 'message': str(exc)}
        traceback.print_exc()
    finally:
        try:
            require(verify_if_snapshot(project, required_modules=modules, entrypoint_file=Path(__file__)) is not None,
                    'Missing final source verification')
        except BaseException as exc:
            report['passed'] = False
            report['source_verification_failure'] = str(exc)
        dump(output/'qualification.json', report)
        print(json.dumps({'passed': report['passed'], 'case': args.case, 'output': str(output)}), flush=True)
        if env is not None:
            if report['passed']:
                env.close()
            else:
                try:
                    raise RuntimeError('Balance display diagnostic failed')
                except RuntimeError:
                    env.close()
    return 0 if report['passed'] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--release', required=True, type=Path)
    parser.add_argument('--case', required=True, choices=['no_cup', 'empty', 'six_grams'])
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--media-output', required=True, type=Path)
    parser.add_argument('--gpu-uuid', required=True)
    parser.add_argument('--duration', type=float, default=.6)
    parser.add_argument('--width', type=int, default=1280)
    parser.add_argument('--height', type=int, default=720)
    args = parser.parse_args()
    if not .4 <= args.duration <= 3. or abs(args.duration/.02-round(args.duration/.02)) > 1.e-8:
        parser.error('Duration must be 0.4–3.0 seconds in exact 20 ms steps')
    raise SystemExit(run(args))


if __name__ == '__main__':
    main()
