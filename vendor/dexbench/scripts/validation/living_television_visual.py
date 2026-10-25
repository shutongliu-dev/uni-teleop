"""Frozen-source native RTX diagnostic of the living-room television display.

The canonical reset supplies the real, initially off observer state. Explicit
display-only fixtures then select off/on/off materials at that same native tick.
There are no robot actions, physics steps, post-reset pose writes or task credit.
Run with scripts/isaac_python.sh from a complete frozen source snapshot.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
from itertools import product
import json
import os
from pathlib import Path
import sys
import traceback

import numpy as np

from bench.paths import OUTPUT_ROOT, load_json


SCREEN_PATH = '/World/Room/Geometry/television_screen/Visual'
CAMERA = 'television_diagnostic'
KIT = 'bench/apps/workroom.headless.kit'
RUNTIME = Path(os.environ.get('ISAAC_PATH', '/nonexistent-isaac-runtime'))
MODULES = ('scripts.validation.living_television_visual', 'bench.isaacsim.release',
    'bench.simulation.isaacsim_backend.env', 'bench.simulation.isaacsim_backend.cameras',
    'bench.simulation.isaacsim_backend.task_visual_state',
    'scripts.validation.board_game_runtime_smoke')


def require(value, message):
    if not value:
        raise ValueError(message)


@contextmanager
def display_fixture(reader, bindings, fixture, native_step):
    """Override only the render hook; the real observer and physics stay intact."""
    original = reader.sync_visual_state
    try:
        reader.sync_visual_state = lambda: bindings.sync(fixture, native_step=native_step)
        yield reader.sync_visual_state()
    finally:
        reader.sync_visual_state = original


def same_physics(before, after):
    """Fail on any changed body, velocity, target, particle, native tick or time."""
    require(before.keys() == after.keys(), 'Physical state inventory changed')
    changed = [key for key in before if not np.array_equal(before[key], after[key])]
    require(not changed, 'Display diagnostic changed physical state: '+', '.join(changed))
    return True


def validate_fixture_receipt(actual, expected):
    """Compare selected materials and sampling identity across repeated renders."""
    from copy import deepcopy
    normalized = []
    for receipt in (actual, expected):
        require(isinstance(receipt, dict) and isinstance(receipt.get('bindings'), list),
                'Missing display fixture sync receipt')
        receipt = deepcopy(receipt)
        for row in receipt['bindings']:
            require(type(row.pop('binding_changed', None)) is bool,
                    'Invalid display binding change receipt')
        normalized.append(receipt)
    require(normalized[0] == normalized[1],
            'Captured visual receipt differs from the explicit display fixture')


def screen_roi(corners, calibration, width, height):
    """Locate the composed screen through the captured RTX camera matrices."""
    from bench.simulation.isaacsim_backend.cameras import project_point
    corners = np.asarray(corners, dtype=float)
    require(corners.shape == (8, 3), 'Expected eight composed screen bounding-box corners')
    pixels = [project_point(None, None, point, width, height, calibration=calibration)
              for point in corners]
    require(all(point is not None for point in pixels), 'Television screen is behind the camera')
    pixels = np.asarray(pixels)
    lo, hi = pixels.min(axis=0), pixels.max(axis=0)
    require(bool(np.all(lo >= 0) and np.all(hi <= [width, height])),
            'Television screen is not fully inside the captured frame')
    inset = (hi-lo)*.15
    left, top = np.ceil(lo+inset).astype(int)
    right, bottom = np.floor(hi-inset).astype(int)
    require(right-left >= 64 and bottom-top >= 36, 'Captured screen has too few pixels')
    return [int(left), int(top), int(right), int(bottom)]


def depth_receipt(depth, corners, calibration, roi):
    """Reject an occluding surface or stale depth at the projected screen ROI."""
    units = float(calibration['meters_per_scene_unit'])
    points = np.c_[np.asarray(corners)/units, np.ones(8)]
    expected = -(points @ np.asarray(calibration['world_to_view']))[:, 2]*units
    left, top, right, bottom = roi
    values = np.asarray(depth)[top:bottom, left:right]
    require(values.size > 0, 'Missing screen depth pixels')
    matched = (np.isfinite(values) & (values > 0)
               & (values >= expected.min()-.025) & (values <= expected.max()+.025))
    fraction = float(np.mean(matched))
    return {'passed': fraction >= .98, 'matched_fraction': fraction,
        'required_fraction': .98, 'tolerance_m': .025,
        'expected_camera_z_range_m': [float(expected.min()), float(expected.max())]}


def assess_display(off, on, returned, roi):
    """Engineering image thresholds; neither photometric nor task acceptance."""
    arrays = [np.asarray(value) for value in (off, on, returned)]
    require(all(value.dtype == np.uint8 and value.ndim == 3 and value.shape[2] == 3
                and value.shape == arrays[0].shape for value in arrays),
            'Expected three matching native uint8 RGB images')
    left, top, right, bottom = roi
    height, width = arrays[0].shape[:2]
    require(0 <= left < right <= width and 0 <= top < bottom <= height, 'Invalid display ROI')
    crops = [value[top:bottom, left:right].astype(float) for value in arrays]
    means = [float(value.mean()) for value in crops]
    on_delta = float(np.mean(np.abs(crops[1]-crops[0])))
    return_delta = float(np.mean(np.abs(crops[2]-crops[0])))
    return_limit = max(5., .25*on_delta)
    return {'passed': bool(means[1]-means[0] >= 10. and on_delta >= 10.
                           and return_delta <= return_limit),
        'scope': 'Display-only engineering thresholds in 8-bit RGB; zero task credit.',
        'mean_rgb_0_255': dict(zip(('off', 'on', 'off_return'), means)),
        'on_minus_off_mean': means[1]-means[0], 'on_off_mean_absolute_difference': on_delta,
        'return_off_mean_absolute_difference': return_delta,
        'thresholds': {'minimum_brightness_increase': 10., 'minimum_on_off_difference': 10.,
                       'maximum_return_difference': return_limit}}


def actual_cuda_identity(requested):
    """Read the actual CUDA device, rather than repeat CUDA_VISIBLE_DEVICES."""
    import torch
    require(torch.cuda.device_count() == 1 and torch.cuda.current_device() == 0,
            'Expected exactly one active CUDA device at logical index zero')
    properties = torch.cuda.get_device_properties(0)
    uuid = str(getattr(properties, 'uuid', ''))
    require(uuid.removeprefix('GPU-').lower() == requested.removeprefix('GPU-').lower(),
            'Actual CUDA device UUID is absent or differs from --gpu-uuid: '+uuid)
    return {'logical_index': 0, 'uuid': uuid, 'name': properties.name,
        'total_memory_bytes': properties.total_memory, 'torch_version': str(torch.__version__),
        'scope': 'Actual CUDA/physics device; RTX uses canonical single-CUDA-device selection.'}


def material_receipt(stage, binding, config):
    from pxr import UsdShade
    material, relationship = UsdShade.MaterialBindingAPI(
        stage.GetPrimAtPath(SCREEN_PATH)).ComputeBoundMaterial()
    require(bool(material) and str(material.GetPath()) == binding['material_path'],
            'Composed television material differs from the display fixture receipt')
    shader = UsdShade.Shader.Get(stage, binding['material_path']+'/Surface')
    require(shader.GetIdAttr().Get() == 'UsdPreviewSurface', 'Unexpected television surface shader')
    expected = config['true_material' if binding['value'] else 'false_material']
    actual = {}
    for source, name in (('diffuse_color_rgb', 'diffuseColor'),
                         ('emissive_color_rgb', 'emissiveColor'),
                         ('roughness', 'roughness'), ('metallic', 'metallic')):
        value = shader.GetInput(name).Get()
        actual[name] = list(value) if source.endswith('_rgb') else float(value)
        require(bool(np.allclose(actual[name], expected[source], rtol=0., atol=1.e-6)),
                'Composed television shader differs from frozen material: '+name)
    return {'path': str(material.GetPath()), 'binding_relationship': str(relationship.GetPath()),
            'shader_id': shader.GetIdAttr().Get(), 'inputs': actual}


def run(args):
    from bench.isaacsim.release import digest, dump, task_inputs
    from bench.simulation.isaacsim_backend.env import FR3DexterousIsaacEnv
    from bench.simulation.isaacsim_backend.task_visual_state import BooleanMaterialBindings
    from scripts.maintenance.freeze_project_sources import verify_if_snapshot
    from scripts.validation.board_game_runtime_smoke import gpu_selection

    project = Path(__file__).resolve().parents[2]
    output, media = args.output.resolve(), args.media_output.resolve()
    require(output.is_relative_to(OUTPUT_ROOT), 'Store provenance under shared datasets')
    require(media.is_relative_to(OUTPUT_ROOT), 'Store PNG media under demo/interface/scenes')
    output.mkdir(parents=True, exist_ok=False)
    media.mkdir(parents=True, exist_ok=False)
    env = None
    report = {'schema': 'living_television_display_diagnostic.v1', 'passed': False,
        'started_at': datetime.now(timezone.utc).isoformat(), 'task_id': 'remote_operation',
        'source_snapshot': str(project), 'cwd': str(Path.cwd().resolve()), 'seed': args.seed,
        'scope': 'Explicit display-only Boolean fixtures at one canonical reset tick; native RTX images.',
        'diagnostic_states': ['off', 'on', 'off_return'], 'task_credit': 0,
        'task_success_claimed': False, 'remote_actuation_qualified': False,
        'model_calls': 0, 'robot_actions': 0, 'post_reset_physics_steps': 0,
        'post_reset_object_pose_writes': 0, 'post_reset_physics_writes': 0,
        'operator_camera_is_agent_observation': False, 'captures': []}
    try:
        source = verify_if_snapshot(project, required_modules=MODULES, entrypoint_file=Path(__file__))
        require(source is not None and Path.cwd().resolve() == project,
                'Run from the root of a complete frozen source snapshot')
        manifest = load_json(project/'source_snapshot_manifest.json')
        require(manifest['files_sha256'].get(KIT) == digest(project/KIT)
                and Path(os.environ.get('ISAAC_BENCH_EXPERIENCE', '')).resolve() == project/KIT,
                'Select the headless Kit experience included in this frozen snapshot')
        require(Path(os.environ.get('ISAAC_PATH', '')).resolve() == RUNTIME.resolve()
                and Path(sys.executable).resolve() == (RUNTIME/'kit/python/bin/python3.12').resolve(),
                'Use the pinned shared Isaac Sim 6.1.0 interpreter and runtime')
        report['source_manifest_sha256'] = digest(project/'source_snapshot_manifest.json')
        report['runtime'] = {'root': str(RUNTIME), 'version': (RUNTIME/'VERSION').read_text().strip(),
            'version_sha256': digest(RUNTIME/'VERSION'), 'python': sys.executable,
            'python_sha256': digest(Path(sys.executable)), 'kit_experience': str(project/KIT),
            'kit_experience_sha256': digest(project/KIT)}
        dump(output/'runtime_imports.json', source)
        report['gpu'] = gpu_selection(args.gpu_uuid)
        report['environment'] = {key: os.environ.get(key) for key in (
            'CUDA_VISIBLE_DEVICES', 'ISAAC_BENCH_SINGLE_CUDA_DEVICE', 'ISAAC_ALLOW_UNVALIDATED_DRIVER',
            'ISAAC_BENCH_EXPERIENCE', 'ISAAC_PATH', 'ISAAC_PYTHON_LAUNCHER')}
        release = args.release.resolve()
        entry, scene, task, initial = task_inputs(release, 'remote_operation', seed=args.seed)
        require(task['evaluation']['schema'] == 'living_remote_evaluation.v3',
                'This display diagnostic requires the frozen TV power-cycle task')
        config = scene['native_task_bundle'].get('visual_state_bindings')
        require(isinstance(config, dict) and config.get('schema') == 'boolean-material-bindings-v1'
                and len(config.get('bindings', [])) == 1, 'Missing frozen television visual binding')
        binding = config['bindings'][0]
        require(binding['target_prim_path'] == SCREEN_PATH
                and binding['state_path'] == ['television', 'power_on'], 'Unexpected television binding target/state')
        report['release'] = {'path': str(release), 'manifest_sha256': digest(release/'release_manifest.json'),
            'scene_sha256': digest(release/entry['scene']), 'task_sha256': digest(release/entry['task'])}
        dump(output/'resolved_scene.json', scene)
        dump(output/'resolved_task.json', task)
        dump(output/'canonical_initial_state.json', initial)
        dump(output/'visual_state_bindings.json', config)
        report['input_files_sha256'] = {name: digest(output/name) for name in
            ('resolved_scene.json', 'resolved_task.json', 'canonical_initial_state.json', 'visual_state_bindings.json')}
        print(json.dumps({'event': 'starting_native_display_diagnostic', 'output': str(output)}), flush=True)
        env = FR3DexterousIsaacEnv(scene, device='cuda:0', render_width=args.width, render_height=args.height)
        observation = env.reset(seed=args.seed, initial_state=initial)
        real_state = observation['task_state']
        require(real_state['television']['power_on'] is False and real_state.get('errors') == [],
                'Canonical native observer must report a valid initially off television')
        dump(output/'canonical_observer_state.json', real_state)
        report['actual_cuda_device'] = actual_cuda_identity(args.gpu_uuid)
        app_config = env.app.config
        require(app_config.get('active_cuda_gpus') == [0] and app_config.get('physics_gpu') == 0
                and app_config.get('max_gpu_count') == 1 and app_config.get('multi_gpu') is False
                and app_config.get('renderer') == 'RayTracedLighting',
                'Actual SimulationApp must select one RTX/physics GPU by the verified CUDA index')
        report['actual_app_gpu_selection'] = {key: app_config.get(key) for key in (
            'active_cuda_gpus', 'physics_gpu', 'max_gpu_count', 'multi_gpu', 'renderer')}
        metadata = env.metadata()
        require(metadata['experience']['path'] == str(project/KIT)
                and metadata['experience']['sha256'] == digest(project/KIT)
                and metadata['native_isaacsim_app'] is True, 'Actual native experience differs from frozen selection')
        dump(output/'native_metadata.json', metadata)
        report['native_metadata_sha256'] = digest(output/'native_metadata.json')
        before = {key: np.array(value, copy=True) for key, value in env.capture_state().items()}
        np.savez_compressed(output/'canonical_physics_state.npz', **before)
        tick = int(before['native_physics_step_index'])
        native_time = float(before['native_world_time_s'])
        require(int(real_state['native_step']) == tick and float(real_state['native_time_s']) == native_time,
                'Canonical observer does not correspond to the captured native tick/time')
        from pxr import Usd, UsdGeom
        prim = env.stage.GetPrimAtPath(SCREEN_PATH)
        require(bool(prim), 'Frozen room television screen prim is absent')
        bbox = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
        extent = bbox.ComputeWorldBound(prim).ComputeAlignedRange()
        lo, hi = np.array(extent.GetMin()), np.array(extent.GetMax())
        require(bool(np.isfinite(np.r_[lo, hi]).all() and np.all(hi > lo)), 'Invalid composed screen bounds')
        corners = np.array(list(product(*zip(lo, hi))))
        center = (lo+hi)/2
        eye = [float(lo[0]-1.65), float(center[1]), float(center[2]+.10)]
        camera = {'id': CAMERA, 'kind': 'world', 'eye': eye, 'target': center.tolist(),
            'focal_length_mm': 28., 'horizontal_aperture_mm': 36., 'clipping_m': [.02, 10.],
            'resolution': [args.width, args.height], 'role': 'operator_display_diagnostic'}
        require(not env._cameras_authored and CAMERA not in env.cameras, 'Diagnostic camera must precede rendering')
        env.cameras[CAMERA] = camera
        env.camera_ids.append(CAMERA)
        report['camera'] = camera
        report['screen'] = {'prim_path': SCREEN_PATH, 'world_bounds_m': [lo.tolist(), hi.tolist()],
            'world_corners_m': corners.tolist(), 'facing_direction': [-1, 0, 0]}
        bindings = BooleanMaterialBindings(env.stage, config)
        images, roi, calibration = [], None, None
        for label, power_on in (('off', False), ('on', True), ('off_return', False)):
            same_physics(before, env.capture_state())
            fixture = {'television': {'power_on': power_on}}
            with display_fixture(env._task_reader, bindings, fixture, tick) as receipt:
                require(receipt['source_native_step'] == tick and len(receipt['bindings']) == 1
                        and receipt['bindings'][0]['value'] is power_on, 'Invalid display fixture sync receipt')
                for _ in range(args.render_captures):
                    frame = env.render_rgbd(camera=CAMERA, width=args.width, height=args.height)
                    validate_fixture_receipt(frame.get('task_visual_state'), receipt)
                    material = material_receipt(env.stage, frame['task_visual_state']['bindings'][0], binding)
                    require(env._task_reader.observe() == real_state,
                            'Display fixture changed the canonical task observer')
                    same_physics(before, env.capture_state())
            measured = frame['camera_calibration']
            current_roi = screen_roi(corners, measured, args.width, args.height)
            if roi is None:
                roi, calibration = current_roi, measured
            require(current_roi == roi and measured == calibration, 'Display captures used different camera calibration')
            require(frame['sim_time_s'] == float(before['time_s'])
                    and frame['depth_sim_time_s'] == float(before['time_s']), 'Captured display has a stale simulation time')
            depth = depth_receipt(frame['depth_m'], corners, measured, roi)
            images.append(frame['rgb'])
            from PIL import Image
            rgb_path, crop_path = media/(label+'.png'), media/(label+'_screen.png')
            Image.fromarray(frame['rgb']).save(rgb_path)
            Image.fromarray(frame['rgb']).crop(tuple(roi)).save(crop_path)
            np.save(output/(label+'_depth_m.npy'), frame['depth_m'])
            capture = {'label': label, 'explicit_display_only_fixture': fixture, 'task_credit': 0,
                'native_step': tick, 'native_time_s': native_time, 'sim_time_s': frame['sim_time_s'],
                'fixture_receipt': receipt, 'composed_material': material,
                'frame_task_visual_state': frame['task_visual_state'], 'canonical_observer_unchanged': True,
                'camera_calibration': measured, 'screen_roi_ltrb_px': roi, 'screen_depth': depth,
                'render_synchronization': frame['render_synchronization'],
                'capture_calls_after_material_sync': args.render_captures,
                'rgb': {'path': str(rgb_path), 'sha256': digest(rgb_path)},
                'screen_crop': {'path': str(crop_path), 'sha256': digest(crop_path)},
                'depth': {'path': str(output/(label+'_depth_m.npy')),
                          'sha256': digest(output/(label+'_depth_m.npy')), 'semantics': frame['depth_semantics']},
                'physics_unchanged': True}
            report['captures'].append(capture)
            dump(output/(label+'_capture.json'), capture)
            require(depth['passed'], 'The rendered ROI does not show the expected television surface')
            print(json.dumps({'event': 'display_captured', 'label': label, 'native_step': tick}), flush=True)
        report['image_assessment'] = assess_display(*images, roi)
        require(report['image_assessment']['passed'], 'Native screen images fail off/on/off_return display thresholds')
        report['physics_unchanged'] = same_physics(before, env.capture_state())
        report['canonical_physics_state_sha256'] = digest(output/'canonical_physics_state.npz')
        report['native_step'] = tick
        report['native_time_s'] = native_time
        report['passed'] = True
    except BaseException as exc:
        report['failure'] = {'type': type(exc).__name__, 'message': str(exc)}
        traceback.print_exc()
    finally:
        try:
            final_source = verify_if_snapshot(project, required_modules=MODULES, entrypoint_file=Path(__file__))
            require(final_source is not None, 'Missing final complete frozen-source verification')
            dump(output/'runtime_imports_final.json', final_source)
        except BaseException as exc:
            report['passed'] = False
            report['source_verification_failure'] = str(exc)
        report['finished_at'] = datetime.now(timezone.utc).isoformat()
        report['shutdown_note'] = 'Report flushed before Kit fast shutdown; inspect process exit code separately.'
        dump(output/'display_diagnostic.json', report)
        print(json.dumps({'output': str(output), 'media': str(media), 'passed': report['passed'],
                          'task_credit': 0, 'scope': report['scope']}), flush=True)
        if env is not None:
            if report['passed']:
                env.close()
            else:
                # close() reads the active exception to retain a failing process
                # exit code even when Kit fast shutdown exits before returning.
                try:
                    raise RuntimeError('Television display diagnostic failed; see '+str(output/'display_diagnostic.json'))
                except RuntimeError:
                    env.close()
    return 0 if report['passed'] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--release', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--media-output', type=Path, required=True)
    parser.add_argument('--gpu-uuid', required=True)
    parser.add_argument('--width', type=int, default=1280)
    parser.add_argument('--height', type=int, default=720)
    parser.add_argument('--render-captures', type=int, default=6,
                        help='Canonical RGB-D captures per material, each with native render warmups')
    args = parser.parse_args()
    if not 0 <= args.seed < 2**32 or not (320 <= args.width <= 3840 and 240 <= args.height <= 2160):
        parser.error('Use an unsigned 32-bit seed and resolution 320–3840 by 240–2160')
    if not 2 <= args.render_captures <= 30:
        parser.error('Use 2–30 render captures per diagnostic material')
    raise SystemExit(run(args))


if __name__ == '__main__':
    main()
