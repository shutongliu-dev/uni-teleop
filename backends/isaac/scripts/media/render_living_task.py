#!/usr/bin/env python3
"""Build and photograph an actual living-room task candidate using native RTX.

Task objects settle under gravity; robots are the original posed static meshes.
Outputs distinguish reset assets, presentation state and physical observations.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT))

from scripts.media.render_workroom import create_camera, digest, dump, now, validate_camera_geometry
from scripts.maintenance.freeze_project_sources import verify_if_snapshot
from scenes.workrooms.living_room.task_scene import TASK_IDS


def capture(stage, app, cameras, output, args, report):
    import numpy as np
    import omni.usd
    import omni.replicator.core as rep
    import omni.hydratexture
    from carb.eventdispatcher import get_eventdispatcher
    from PIL import Image
    from pxr import Sdf

    report['images'] = {}
    for name in args.cameras:
        definition = cameras[name]
        product = rep.create.render_product(create_camera(stage, name, definition), (args.width, args.height))
        prim = stage.GetPrimAtPath(product.path)
        for key, kind, value in (
            ('samplesPerPixel', Sdf.ValueTypeNames.UInt, args.samples),
            ('samplesPerIteration', Sdf.ValueTypeNames.Int, min(args.samples, 32)),
            ('clampSpp', Sdf.ValueTypeNames.Int, args.samples),
            ('limits:maxBounces', Sdf.ValueTypeNames.UInt, 12),
            ('denoising:enabled', Sdf.ValueTypeNames.Bool, True),
            ('adaptiveSampling:enabled', Sdf.ValueTypeNames.Bool, False)):
            prim.CreateAttribute('omni:rtx:pt:' + key, kind).Set(value)
        rgb_source = rep.AnnotatorRegistry.get_annotator('rgb')
        rgb_source.attach([product])
        info = {}
        def on_frame(event):
            frame = product.hydra_texture.get_frame_info(event['result_handle'])
            info.update({k: frame[k] for k in ('progression', 'frame_number', 'status') if k in frame})
        subscription = get_eventdispatcher().observe_event(observer_name='living_task_' + name,
            event_name=omni.hydratexture.GLOBAL_EVENT_DRAWABLE_CHANGED, on_event=on_frame,
            filter=product.hydra_texture.get_event_key())
        started = time.monotonic()
        for frame in range(args.render_updates):
            app.update()
            pending = int(omni.usd.get_context().get_stage_loading_status()[2])
            if frame % 8 == 0:
                print(json.dumps({'event': 'render', 'task': args.task, 'camera': name,
                    'updates': frame + 1, 'spp': info.get('progression'), 'pending': pending}), flush=True)
            if frame >= 11 and pending == 0 and int(info.get('progression', 0)) >= args.samples:
                break
        rgb = np.asarray(rgb_source.get_data())
        spp = int(info.get('progression', 0))
        if pending or spp < args.samples or rgb.shape not in ((args.height, args.width, 3), (args.height, args.width, 4)):
            raise RuntimeError(f'Incomplete RTX image: {name} spp={spp}, pending={pending}, shape={rgb.shape}')
        rgb = rgb[:, :, :3].astype(np.uint8)
        if rgb.std() < 2:
            raise RuntimeError('Nearly blank RTX image')
        image_path = output / (name + '.png')
        Image.fromarray(rgb).save(image_path)
        report['images'][name] = {'path': str(image_path), 'sha256': digest(image_path),
            'size_px': [args.width, args.height], 'actual_samples_per_pixel': spp,
            'camera': definition, 'native_rtx': True, 'pending_stage_loads': pending,
            'elapsed_s': time.monotonic() - started}
        dump(output / 'provenance.json', report)
        rgb_source.detach([product]); subscription = None; product.destroy()


def enable_usd_readback(settings):
    # Keep the Fabric extension loaded: the Kit experience depends on it, so
    # unloading it also shuts down the experience. Override its USD-write
    # defaults after World setup; the renderer already reads USD transforms.
    expected = {'updateToUsd': True, 'updateParticlesToUsd': True,
        'updateVelocitiesToUsd': True, 'suppressReadback': False, 'outputVelocitiesLocalSpace': False}
    for key, value in expected.items():
        settings.set_bool('/physics/' + key, value)
    actual = {key: settings.get('/physics/' + key) for key in expected}
    if actual != expected:
        raise RuntimeError('USD physics readback settings differ: ' + repr(actual))
    return actual


def body_states(stage, inventory, native_views, world, *, publish_paused=False):
    import omni.physx
    from pxr import Usd, UsdGeom
    # Live observation must not save Fabric to USD: this runtime treats the
    # write as a new dynamic state and can reset native velocities to zero.
    omni.physx.get_physx_simulation_interface().fetch_results()
    rows = {}
    for body in inventory['bodies']:
        view = native_views[body['path']]
        transform = view.get_transforms()[0].tolist()
        velocity = view.get_velocities()[0].tolist()
        if not all(math.isfinite(x) for x in transform + velocity):
            raise RuntimeError('Nonfinite native task state: ' + body['path'])
        rows[body['path']] = {'position_m': transform[:3],
            'orientation_xyzw': transform[3:], 'velocity_m_s': velocity[:3],
            'angular_velocity_rad_s': velocity[3:],
            'source': 'Read-only PhysX tensor view; velocities captured before any export',
            'usd_published': False}
    if not publish_paused:
        return rows
    if world.is_playing():
        raise RuntimeError('Publish USD only after pausing and after all physical measurements')
    import omni.physxfabric
    omni.physx.get_physx_interface().update_transformations(False, True, True, False)
    fabric = omni.physxfabric.get_physx_fabric_interface()
    fabric.force_update(world.get_physics_dt(), world.current_time)
    fabric.save_to_usd()
    cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    for path, row in rows.items():
        matrix = cache.GetLocalToWorldTransform(stage.GetPrimAtPath(path))
        usd_position = list(matrix.ExtractTranslation())
        rotation = matrix.ExtractRotationQuat()
        usd_quaternion = [*rotation.GetImaginary(), rotation.GetReal()]
        transform = row['orientation_xyzw']
        qnorm = sum(x*x for x in transform)**.5
        unorm = sum(x*x for x in usd_quaternion)**.5
        cosine = abs(sum(a*b for a,b in zip(usd_quaternion, transform)) / (qnorm*unorm))
        row.update({'matrix_row_major': [list(v) for v in matrix],
            'usd_published': True, 'usd_position_m': usd_position,
            'usd_native_position_error_m': sum((a-b)**2 for a,b in zip(usd_position,row['position_m']))**.5,
            'usd_native_orientation_error_rad': 2*math.acos(min(1., cosine))})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', choices=TASK_IDS, required=True)
    parser.add_argument('--asset-output', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--run-output', type=Path, required=True)
    parser.add_argument('--presentation', action='store_true')
    parser.add_argument('--tower-profile', choices=('trays_v1',),
                        help='Explicit new tower/tray candidate; omitted preserves historical authors')
    parser.add_argument('--width', type=int, default=1920)
    parser.add_argument('--height', type=int, default=1440)
    parser.add_argument('--samples', type=int, default=128)
    parser.add_argument('--render-updates', type=int, default=160)
    parser.add_argument('--settle-steps', type=int, help='Default: 2880 for liquid, 480 for other scenes')
    parser.add_argument('--dt', type=float, help='Default: 1/960 s for liquid, 1/240 s for other scenes')
    parser.add_argument('--cameras', nargs='+', choices=('room', 'workcell', 'tabletop'), default=['workcell', 'tabletop'])
    args = parser.parse_args()
    if args.tower_profile and args.task not in ('tower_building', 'tower_extraction'):
        parser.error('--tower-profile applies only to wooden-block tasks')
    author_options = {'profile': args.tower_profile} if args.tower_profile else {}
    if args.dt is None:
        args.dt = 1/960 if args.task == 'water_pouring' else 1/240
    if args.settle_steps is None:
        args.settle_steps = 2880 if args.task == 'water_pouring' else 480
    if min(args.width, args.height, args.samples, args.render_updates, args.settle_steps, args.dt) <= 0:
        parser.error('Dimensions, counts and timestep must be positive')
    gpu = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    if not gpu.startswith('GPU-') or ',' in gpu:
        parser.error('Select one physical GPU UUID through CUDA_VISIBLE_DEVICES')
    from scenes.workrooms.living_room import task_scene
    from bench.simulation.isaacsim_backend.render_sync import startup_arguments, verify_settings
    snapshot = verify_if_snapshot(PROJECT, required_modules=('scenes.workrooms.living_room.task_scene',), entrypoint_file=Path(__file__))
    if snapshot is None:
        parser.error('Run from a complete frozen source snapshot')
    for path in (args.asset_output, args.output, args.run_output):
        if path.exists():
            raise FileExistsError('Preserve earlier builds/attempts: ' + str(path))
    for path in (args.asset_output, args.output, args.run_output):
        path.mkdir(parents=True)
    report = {'schema': 'living_room_task_candidate_render.v1', 'task_id': args.task,
        'created_utc': now(), 'status': 'starting', 'source_root': str(PROJECT), 'source_snapshot': snapshot,
        'gpu_uuid': gpu, 'isaac_allow_unvalidated_driver': os.environ.get('ISAAC_ALLOW_UNVALIDATED_DRIVER') == '1',
        'presentation_only': args.presentation, 'model_calls': 0, 'robot_actions': 0,
        'native_manipulation_qualified': False,
        'scope': 'Native task-object settling and RTX presentation with original static robot meshes; no robot strategy or task score'}
    dump(args.output / 'provenance.json', report)
    app = world = None; exit_code = 1
    try:
        from isaacsim import SimulationApp
        extra = startup_arguments()
        if report['isaac_allow_unvalidated_driver']:
            extra.append('--/rtx/verifyDriverVersion/enabled=false')
        app = SimulationApp({'headless': True, 'width': args.width, 'height': args.height,
            'multi_gpu': False, 'active_cuda_gpus': [0], 'max_gpu_count': 1, 'physics_gpu': 0,
            'renderer': 'PathTracing', 'fast_shutdown': True, 'extra_args': extra},
            experience=str(PROJECT / 'bench/apps/workroom.headless.kit'))
        import carb
        import omni.usd
        from isaacsim.core.api import World
        from pxr import PhysxSchema, Usd, UsdGeom
        settings = carb.settings.get_settings()
        for key in ('updateToUsd', 'updateParticlesToUsd', 'updateVelocitiesToUsd'):
            settings.set_bool('/physics/' + key, True)
        report['render_sync'] = verify_settings(settings)
        author = task_scene.task_author(args.task)
        report['source_snapshot_runtime'] = verify_if_snapshot(PROJECT,
            required_modules=(author.__name__, 'scenes.workrooms.living_room.task_scene'), entrypoint_file=Path(__file__))
        # Publish an actual initial-state task layer separately from photographs.
        asset_path = args.asset_output / 'initial.usda'
        asset_stage = Usd.Stage.CreateNew(str(asset_path))
        UsdGeom.Xform.Define(asset_stage, '/World')
        UsdGeom.SetStageMetersPerUnit(asset_stage, 1.)
        UsdGeom.SetStageUpAxis(asset_stage, UsdGeom.Tokens.z)
        initial_metadata = author.author_task(asset_stage, presentation=False, **author_options)
        asset_stage.SetDefaultPrim(asset_stage.GetPrimAtPath('/World'))
        initial_inventory = task_scene.physical_inventory(asset_stage)
        asset_stage.GetRootLayer().Save()
        dump(args.asset_output / 'metadata.json', initial_metadata)
        from scripts.assets.build_card_room import dependency_inventory
        explicit_dependencies = initial_metadata.get('dependencies', [])
        if isinstance(explicit_dependencies, list):
            explicit_dependencies = [entry['path'] if isinstance(entry, dict) else entry
                                     for entry in explicit_dependencies]
        dependencies = dependency_inventory(asset_path, explicit_dependencies)
        dump(args.asset_output / 'manifest.json', {'schema': 'living_task_objects_candidate.v1',
            'task_id': args.task, 'source_root': str(PROJECT), 'source_snapshot': snapshot,
            'initial_usd': str(asset_path), 'initial_usd_sha256': digest(asset_path),
            'metadata_sha256': digest(args.asset_output / 'metadata.json'), 'inventory': initial_inventory,
            'dependencies': dependencies,
            'native_manipulation_qualified': False, 'release_status': 'scene_candidate_not_bench_release'})
        omni.usd.get_context().new_stage()
        stage = omni.usd.get_context().get_stage()
        room = task_scene.add_room_and_robots(stage, args.task, args.run_output)
        if args.tower_profile:
            room['cameras']['tabletop'].update(eye=[1.65, -.95, 1.75],
                target=[.65, 0., .98], focal_length_mm=30.)
        metadata = author.author_task(stage, presentation=args.presentation, **author_options)
        inventory = task_scene.physical_inventory(stage)
        report.update(task_metadata=metadata, task_inventory=inventory, parent_environment=room['parent_environment'],
            robot_installation=room['robots'], initial_asset=str(asset_path), initial_asset_sha256=digest(asset_path),
            task_dependencies=dependencies, room_and_robot_dependencies=room['dependencies'])
        world = World(stage_units_in_meters=1., physics_dt=args.dt, rendering_dt=1/30,
            physics_prim_path='/World/PhysicsScene', backend='numpy')
        context = world.get_physics_context()
        context.enable_gpu_dynamics(True); context.set_broadphase_type('GPU'); context.set_solver_type('TGS')
        report['physics_readback_before_reset'] = enable_usd_readback(settings)
        PhysxSchema.PhysxSceneAPI.Apply(stage.GetPrimAtPath('/World/PhysicsScene')).CreateEnableCCDAttr(False)
        stage.GetRootLayer().Export(str(args.run_output / 'authored_scene.usda'))
        report['authored_scene_sha256'] = digest(args.run_output / 'authored_scene.usda')
        dump(args.run_output / 'task_metadata.json', metadata)
        report['camera_geometry_check'] = validate_camera_geometry(stage, room['cameras'])
        if args.task == 'water_pouring':
            # Particle postprocessing creates the native isosurface lazily.
            app.update(); app.update()
        world.reset()
        report['physics_readback_after_reset'] = enable_usd_readback(settings)
        native_views = {row['path']: world.physics_sim_view.create_rigid_body_view(row['path'])
                        for row in inventory['bodies']}
        if any(view.count != 1 for view in native_views.values()):
            raise RuntimeError('Authored bodies do not match native rigid-body inventory')
        report['initial_body_state'] = body_states(stage, inventory, native_views, world)
        report['physics_time_at_recording_start_s'] = float(world.current_time)
        if args.task == 'water_pouring':
            report['initial_water_state'] = author.read_water_state(stage, metadata)
        with (args.run_output / 'settling.jsonl').open('w') as stream:
            for step in range(args.settle_steps):
                world.step(render=False)
                if step % 24 == 23 or step == args.settle_steps - 1:
                    state = {'step': step + 1, 'time_s': (step + 1)*args.dt,
                             'bodies': body_states(stage, inventory, native_views, world)}
                    stream.write(json.dumps(state) + '\n')
                if step % 120 == 119:
                    print(json.dumps({'event': 'settling', 'task': args.task, 'step': step + 1}), flush=True)
        world.pause()
        report['physics_steps'] = args.settle_steps
        report['physics_time_s'] = float(world.current_time)
        report['explicit_step_duration_s'] = args.settle_steps * args.dt
        report['final_body_state'] = body_states(stage, inventory, native_views, world, publish_paused=True)
        world.render()
        report['live_observation_read_only'] = True
        report['usd_export_after_physics_only'] = True
        if max(row['usd_native_position_error_m'] for row in report['final_body_state'].values()) > 1.e-5:
            raise RuntimeError('Rendered USD positions do not agree with native task-body transforms')
        if max(row['usd_native_orientation_error_rad'] for row in report['final_body_state'].values()) > 1.e-4:
            raise RuntimeError('Rendered USD orientations do not agree with native task-body transforms')
        if args.task == 'water_pouring':
            report['final_water_state'] = author.read_water_state(stage, metadata)
        report['pose_source'] = 'Native PhysX tensor states with independent USD position agreement; not contact-force or task qualification'
        stage.GetRootLayer().Export(str(args.run_output / 'settled_scene.usda'))
        report['settled_scene_sha256'] = digest(args.run_output / 'settled_scene.usda')
        dump(args.run_output / 'settling_report.json', report)
        capture(stage, app, room['cameras'], args.output, args, report)
        report.update(status='rendered', completed_utc=now())
        exit_code = 0
    except BaseException as exc:
        report.update(status='failed', error=repr(exc), traceback=traceback.format_exc())
        traceback.print_exc()
    finally:
        dump(args.output / 'provenance.json', report)
        dump(args.run_output / 'report.json', report)
        if app is not None:
            app.close(exit_code=exit_code)
    return exit_code


if __name__ == '__main__':
    raise SystemExit(main())
