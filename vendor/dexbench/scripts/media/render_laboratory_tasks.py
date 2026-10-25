"""Photograph frozen laboratory task releases and their desktop details with RTX.

Extra presentation views do not replace the four public episode cameras. The
timeline stays stopped: these pictures document appearance, not task execution.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import traceback

from bench.isaacsim.release import checked, digest, dump, task_inputs
from bench.paths import load_json, OUTPUT_ROOT
from scenes.workrooms.laboratory.scene import environment_reference
from scripts.maintenance.freeze_project_sources import verify_if_snapshot
from scripts.media.render_task_gallery import capture_image, pose_robot
from scripts.media.render_workroom import now
from scripts.validation.board_game_runtime_smoke import gpu_selection

PROJECT = Path(__file__).resolve().parents[2]


def photograph(args, app, task_id, release, robot_cache):
    import omni.timeline
    import omni.usd
    from pxr import UsdPhysics
    from bench.simulation.isaacsim_backend.robot_asset import import_robot
    from bench.simulation.isaacsim_backend.scene import build_scene

    checked(release['path']+'/release_manifest.json', release['manifest_sha256'])
    _, scene, task, initial = task_inputs(Path(release['path']), task_id, seed=0)
    if scene['workroom'] != environment_reference():
        raise ValueError('Laboratory parent differs from the selected fixed room')
    output = args.output/task_id
    output.mkdir(exist_ok=False)
    robot = scene['robot']
    key = json.dumps(robot, sort_keys=True)
    if key not in robot_cache:
        robot_cache[key] = import_robot(robot['assets_root'], hand_id=robot['hand_id'],
            assembly_version=robot.get('assembly_version'))[0]
    omni.usd.get_context().new_stage()
    stage = omni.usd.get_context().get_stage()
    UsdPhysics.Scene.Define(stage, '/World/PhysicsScene')
    info = build_scene(stage, scene, robot_cache[key])
    record = {'task_id': task_id, 'release': release, 'environment': scene['workroom'],
        'robot_pose': pose_robot(stage, scene, initial, info['robot_link_paths']),
        'physics_steps': 0, 'model_calls': 0, 'images': {},
        'state': 'Canonical authored reset, no settling or object pose changes',
        'scope': 'Static RTX appearance only; no physical operation qualification'}
    cameras = {'task': deepcopy(scene['cameras']['task']),
        'desktop': {'eye': [1.55, 1.15, 2.60], 'target': [.65, .08, .91],
                    'focal_length_mm': 27.},
        'supplies': {'eye': [1.15, 1.04, 1.53], 'target': [.71, .47, .95],
                     'focal_length_mm': 42.},
        'apparatus': {'eye': [1.22, 1.35, 1.60], 'target': [.72, 2.075, .965],
                     'focal_length_mm': 42.}}
    if task_id == 'sample_sorting':
        cameras['apparatus'] = {'eye': [1.15, 1.04, 1.53],
            'target': [.71, .47, .95], 'focal_length_mm': 42.}
    if args.view:
        cameras = {name: cameras[name] for name in args.view}
    timeline = omni.timeline.get_timeline_interface()
    if timeline.is_playing():
        raise RuntimeError('Presentation must not start physics')
    before = timeline.get_current_time()
    for name, camera in cameras.items():
        camera.update(horizontal_aperture_mm=27.*args.width/args.height,
                      vertical_aperture_mm=27., resolution=[args.width, args.height])
        frame_record = {'task_id': task_id}
        capture_image(stage, app, name, camera, output, args, frame_record)
        record['images'].update(frame_record['images'])
        if timeline.is_playing() or timeline.get_current_time() != before:
            raise RuntimeError('Presentation advanced the physical timeline')
    record['status'] = 'rendered'
    dump(output/'provenance.json', record)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--release-map', required=True, type=Path,
        help='JSON task ID -> {path, manifest_sha256}; pins each immutable release')
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--run-output', required=True, type=Path)
    parser.add_argument('--task', action='append')
    parser.add_argument('--view', action='append', choices=['task', 'desktop', 'supplies', 'apparatus'],
        help='Render only selected presentation views; default renders all four')
    parser.add_argument('--width', type=int, default=1920)
    parser.add_argument('--height', type=int, default=1440)
    parser.add_argument('--samples', type=int, default=64)
    parser.add_argument('--render-updates', type=int, default=240)
    args = parser.parse_args()
    if min(args.width, args.height, args.samples, args.render_updates) <= 0:
        parser.error('Image dimensions and sample counts must be positive')
    if not args.output.resolve().is_relative_to(OUTPUT_ROOT):
        parser.error(f'Store presentation images under {OUTPUT_ROOT}')
    if not args.run_output.resolve().is_relative_to(OUTPUT_ROOT):
        parser.error('Store run evidence under shared datasets')
    source = verify_if_snapshot(PROJECT, required_modules=(
        'scenes.workrooms.laboratory.scene', 'scripts.media.render_task_gallery'),
        entrypoint_file=Path(__file__))
    if source is None:
        parser.error('Render from a complete frozen source snapshot')
    releases = load_json(args.release_map)
    if args.task:
        releases = {name: releases[name] for name in args.task}
    if not releases:
        parser.error('Select at least one frozen task release')
    gpu = gpu_selection(os.environ.get('CUDA_VISIBLE_DEVICES', ''))
    args.output.mkdir(parents=True, exist_ok=True)
    args.run_output.mkdir(parents=True, exist_ok=False)
    report = {'schema': 'laboratory_task_detail_renders.v1', 'started_utc': now(),
        'source_snapshot': source, 'gpu': gpu, 'status': 'initializing',
        'release_map_sha256': digest(args.release_map),
        'driver_override': os.environ.get('ISAAC_ALLOW_UNVALIDATED_DRIVER'),
        'physics_steps': 0, 'model_calls': 0, 'tasks': {}, 'errors': {}}
    app = None
    code = 1
    try:
        from isaacsim import SimulationApp
        from bench.simulation.isaacsim_backend.render_sync import startup_arguments, verify_settings
        app = SimulationApp({'headless': True, 'width': 640, 'height': 480,
            'disable_viewport_updates': True, 'multi_gpu': False,
            'active_cuda_gpus': [0], 'max_gpu_count': 1, 'physics_gpu': 0,
            'renderer': 'PathTracing', 'fast_shutdown': True,
            'extra_args': [*startup_arguments(), '--/rtx/verifyDriverVersion/enabled=false']},
            experience=str(PROJECT/'bench/apps/workroom.headless.kit'))
        import carb
        report['render_settings'] = verify_settings(carb.settings.get_settings())
        robot_cache = {}
        for task_id, release in releases.items():
            try:
                report['tasks'][task_id] = photograph(args, app, task_id, release, robot_cache)
            except Exception:
                report['errors'][task_id] = traceback.format_exc()
                traceback.print_exc()
            dump(args.run_output/'report.json', report)
        code = int(bool(report['errors']))
        report['status'] = 'completed' if code == 0 else 'partial_failure'
    except BaseException:
        report['status'] = 'failed'
        report['traceback'] = traceback.format_exc()
        traceback.print_exc()
    finally:
        report['finished_utc'] = now()
        dump(args.run_output/'report.json', report)
        if app is not None:
            app.close(exit_code=code)
    return code


if __name__ == '__main__':
    raise SystemExit(main())
