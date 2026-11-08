"""Photograph catalog-pinned Task10 props with native RTX at the authored reset.

The task and overview views retain their scene framing. Extra macro views are
presentation cameras only; the public episode still has its original five IDs.
No physics steps, object placement changes or task actions are performed.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import os
from pathlib import Path
import traceback

from bench.isaacsim.release import digest, dump, task_inputs
from bench.paths import OUTPUT_ROOT
from scenes.workrooms.catalog import load_task
from scripts.maintenance.freeze_project_sources import verify_if_snapshot
from scripts.media.render_task_gallery import capture_image, pose_robot
from scripts.media.render_workroom import now
from scripts.validation.board_game_runtime_smoke import gpu_selection

PROJECT = Path(__file__).resolve().parents[2]
TASK_ID = 'cooperative_insertion'
VIEWS = ('overview', 'task', 'sleeve', 'peg', 'tray')


def photograph(args, app, record):
    import omni.timeline
    import omni.usd
    from pxr import UsdPhysics
    from bench.simulation.isaacsim_backend.robot_asset import import_robot
    from bench.simulation.isaacsim_backend.scene import build_scene

    loaded = load_task('mechanical_assembly', TASK_ID)
    release = Path(loaded['definition']['source_release']['path'])
    _, scene, task, initial = task_inputs(release, TASK_ID, seed=0)
    if scene['workroom'] != loaded['scene']['workroom']:
        raise ValueError('Frozen parent differs from the task catalog')
    record.update(release=str(release), release_manifest_sha256=digest(release/'release_manifest.json'),
        environment=scene['workroom'], asset=scene['mechanical_cooperative_provenance']['asset_candidate'])
    robot = scene['robot']
    asset = import_robot(robot['assets_root'], hand_id=robot['hand_id'],
        assembly_version=robot['assembly_version'], instances=robot.get('instances'))[0]
    omni.usd.get_context().new_stage()
    stage = omni.usd.get_context().get_stage()
    UsdPhysics.Scene.Define(stage, '/World/PhysicsScene')
    info = build_scene(stage, scene, asset)
    record['robot_pose'] = pose_robot(stage, scene, initial, info['robot_link_paths'])
    record['initial_object_poses'] = {row['id']: row['initial_pose_xyzw']
                                    for row in task['asset_metadata']['objects']}
    cameras = {name: deepcopy(scene['cameras'][name]) for name in ('overview', 'task')}
    cameras.update(
        sleeve={'eye': [.82,-.38,.99], 'target': [.65,-.216,.807], 'focal_length_mm': 50.},
        peg={'eye': [.81,.03,1.02], 'target': [.65,.205,.829], 'focal_length_mm': 55.},
        tray={'eye': [.78,-.31,1.18], 'target': [.49,0.,.78], 'focal_length_mm': 40.})
    timeline = omni.timeline.get_timeline_interface()
    if timeline.is_playing():
        raise RuntimeError('Presentation requires a stopped physics timeline')
    before = timeline.get_current_time()
    for name in args.view or VIEWS:
        camera = cameras[name]
        camera.update(horizontal_aperture_mm=27.*args.width/args.height,
            vertical_aperture_mm=27., resolution=[args.width,args.height])
        frame = {'task_id': TASK_ID}
        capture_image(stage, app, name, camera, args.output, args, frame)
        record['images'].update(frame['images'])
        if timeline.is_playing() or timeline.get_current_time() != before:
            raise RuntimeError('Rendering advanced the physics timeline')
        dump(args.run_output/'report.json', record)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--run-output', type=Path, required=True)
    parser.add_argument('--view', choices=VIEWS, action='append')
    parser.add_argument('--width', type=int, default=1920)
    parser.add_argument('--height', type=int, default=1440)
    parser.add_argument('--samples', type=int, default=64)
    parser.add_argument('--render-updates', type=int, default=240)
    args = parser.parse_args()
    if min(args.width,args.height,args.samples,args.render_updates) <= 0:
        parser.error('Image dimensions and sample counts must be positive')
    if not args.output.resolve().is_relative_to(OUTPUT_ROOT):
        parser.error('Store presentation media under demo/interface')
    if not args.run_output.resolve().is_relative_to(OUTPUT_ROOT):
        parser.error('Store experiment evidence under shared datasets')
    source = verify_if_snapshot(PROJECT,
        required_modules=('scenes.workrooms.catalog','scripts.media.render_task_gallery'),
        entrypoint_file=Path(__file__))
    if source is None:
        parser.error('Render from a complete frozen source snapshot')
    gpu = gpu_selection(os.environ.get('CUDA_VISIBLE_DEVICES',''))
    if args.output.exists() or args.run_output.exists():
        raise FileExistsError('Preserve previous media and experiment records')
    args.output.mkdir(parents=True)
    args.run_output.mkdir(parents=True)
    record = {'schema':'mechanical_cooperative_prop_renders.v1', 'task_id':TASK_ID,
        'started_utc':now(), 'source_snapshot':source, 'gpu':gpu, 'status':'initializing',
        'runtime':'Isaac Sim 6.1.0 native RTX PathTracing', 'physics_steps':0, 'model_calls':0,
        'state':'Canonical authored reset; static robot FK; no object pose edits',
        'scope':'Appearance only. Native scene/support and Agent interface are validated separately.',
        'images':{}}
    app, code = None, 1
    try:
        from isaacsim import SimulationApp
        from bench.simulation.isaacsim_backend.render_sync import startup_arguments, verify_settings
        extra = startup_arguments()
        override = os.environ.get('ISAAC_ALLOW_UNVALIDATED_DRIVER') == '1'
        if override:
            extra.append('--/rtx/verifyDriverVersion/enabled=false')
        record['driver_override'] = override
        app = SimulationApp({'headless':True, 'width':640, 'height':480,
            'disable_viewport_updates':True, 'multi_gpu':False, 'active_cuda_gpus':[0],
            'max_gpu_count':1, 'physics_gpu':0, 'renderer':'PathTracing', 'fast_shutdown':True,
            'extra_args':extra}, experience=str(PROJECT/'bench/apps/workroom.headless.kit'))
        import carb
        record['render_settings'] = verify_settings(carb.settings.get_settings())
        photograph(args, app, record)
        record['source_snapshot_final'] = verify_if_snapshot(PROJECT,
            required_modules=('scripts.media.render_task_gallery',), entrypoint_file=Path(__file__))
        record['status'], code = 'completed', 0
    except BaseException:
        record.update(status='failed', traceback=traceback.format_exc())
        traceback.print_exc()
    finally:
        record['finished_utc'] = now()
        dump(args.run_output/'report.json', record)
        dump(args.output/'provenance.json', record)
        if app is not None:
            app.close(exit_code=code)
    return code


if __name__ == '__main__':
    raise SystemExit(main())
