#!/usr/bin/env python3
"""Photograph a pinned static card-room environment with native Isaac RTX.

No articulation, controller or physics scene is created by this preview tool.
The manipulation harness separately owns native task qualification.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time
import traceback

from bench.paths import load_json

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT))

from scripts.media.render_workroom import create_camera, digest, dump, now, validate_camera_geometry
from scripts.maintenance.freeze_project_sources import verify_if_snapshot


def main(*, expected_family='card_room', camera_choices=('room', 'workcell', 'tabletop')):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--asset', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--width', type=int, default=1920)
    parser.add_argument('--height', type=int, default=1440)
    parser.add_argument('--samples', type=int, default=128)
    parser.add_argument('--render-updates', type=int, default=100)
    parser.add_argument('--minimum-render-updates', type=int, default=16,
                        help='Allow initial material/texture updates before accepting the sample count')
    parser.add_argument('--cameras', nargs='+', choices=camera_choices,
                        default=['room', 'workcell', 'tabletop'])
    args = parser.parse_args()
    if min(args.width, args.height, args.samples, args.render_updates, args.minimum_render_updates) <= 0:
        parser.error('Dimensions and render counts must be positive')
    if args.minimum_render_updates > args.render_updates:
        parser.error('Minimum render updates exceed the update budget')
    gpu = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    if not gpu.startswith('GPU-') or ',' in gpu:
        parser.error('CUDA_VISIBLE_DEVICES must select one physical GPU UUID')
    verified = verify_if_snapshot(PROJECT, required_modules=('scripts.media.render_workroom',),
                                  entrypoint_file=Path(__file__))
    if verified is None:
        parser.error('Render from a complete frozen source snapshot')
    manifest = load_json(args.asset / 'manifest.json')
    description = args.asset / 'scene_description.json'
    if digest(description) != manifest['description_sha256']:
        raise ValueError('Environment description hash mismatch')
    scene = load_json(description)
    if scene.get('family') != expected_family:
        raise ValueError('Environment family differs from the selected render entrypoint')
    if digest(manifest['scene']) != manifest['scene_sha256']:
        raise ValueError('Environment USD hash mismatch')
    for path, info in manifest['dependencies'].items():
        if digest(path) != info['sha256']:
            raise ValueError('Environment dependency hash mismatch: ' + path)
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError('Preserve previous attempts; use a new output directory')
    output.mkdir(parents=True)
    override = os.environ.get('ISAAC_ALLOW_UNVALIDATED_DRIVER') == '1'
    report = {'schema': f'scence.{expected_family}_static_render.v1', 'status': 'initializing',
              'created_utc': now(), 'source_snapshot': verified, 'source_root': str(PROJECT),
              'asset_manifest': str(args.asset / 'manifest.json'),
              'asset_manifest_sha256': digest(args.asset / 'manifest.json'),
              'gpu_uuid': gpu, 'isaac_allow_unvalidated_driver': override,
              'runtime': 'Isaac Sim 6.1.0 native RTX PathTracing', 'images': {},
              'minimum_render_updates': args.minimum_render_updates,
              'model_calls': 0, 'physics_steps': 0, 'physics_accepted': False,
              'scope': 'Static room/declared composition presentation only; no task or contact qualification',
              'composition': scene.get('preview_composition')}
    dump(output / 'provenance.json', report)
    app = None; exit_code = 1
    try:
        from isaacsim import SimulationApp
        app = SimulationApp({'headless': True, 'width': args.width, 'height': args.height,
            'multi_gpu': False, 'active_cuda_gpus': [0], 'max_gpu_count': 1,
            'physics_gpu': 0, 'renderer': 'PathTracing', 'fast_shutdown': True,
            'extra_args': ['--/rtx/verifyDriverVersion/enabled=false'] if override else []},
            experience=str(PROJECT / 'bench/apps/workroom.headless.kit'))
        import numpy as np
        import omni.usd
        import omni.replicator.core as rep
        import omni.hydratexture
        from carb.eventdispatcher import get_eventdispatcher
        from PIL import Image
        from pxr import Sdf, UsdGeom
        omni.usd.get_context().new_stage()
        stage = omni.usd.get_context().get_stage()
        root = UsdGeom.Xform.Define(stage, '/World')
        stage.SetDefaultPrim(root.GetPrim())
        root.GetPrim().GetReferences().AddReference(manifest['scene'])
        UsdGeom.SetStageMetersPerUnit(stage, 1.)
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        cameras = {name: scene['cameras'][name] for name in args.cameras}
        report['camera_geometry_check'] = validate_camera_geometry(stage, cameras)
        for name, definition in cameras.items():
            path = create_camera(stage, name, definition)
            product = rep.create.render_product(path, (args.width, args.height))
            rp = stage.GetPrimAtPath(product.path)
            settings = {
                'omni:rtx:pt:samplesPerPixel': (Sdf.ValueTypeNames.UInt, args.samples),
                'omni:rtx:pt:samplesPerIteration': (Sdf.ValueTypeNames.Int, min(args.samples, 32)),
                'omni:rtx:pt:clampSpp': (Sdf.ValueTypeNames.Int, args.samples),
                'omni:rtx:pt:limits:maxBounces': (Sdf.ValueTypeNames.UInt, 8),
                'omni:rtx:pt:denoising:enabled': (Sdf.ValueTypeNames.Bool, True),
                'omni:rtx:pt:adaptiveSampling:enabled': (Sdf.ValueTypeNames.Bool, False),
            }
            for key, (kind, value) in settings.items():
                rp.CreateAttribute(key, kind).Set(value)
            annotator = rep.AnnotatorRegistry.get_annotator('rgb')
            annotator.attach([product])
            info = {}
            def on_frame(event):
                frame = product.hydra_texture.get_frame_info(event['result_handle'])
                info.update({key: frame[key] for key in ('progression', 'frame_number', 'status') if key in frame})
            subscription = get_eventdispatcher().observe_event(
                observer_name='card_room_' + name,
                event_name=omni.hydratexture.GLOBAL_EVENT_DRAWABLE_CHANGED,
                on_event=on_frame, filter=product.hydra_texture.get_event_key())
            start = time.monotonic()
            pending_stage_loads = None
            for frame in range(args.render_updates):
                app.update()
                pending_stage_loads = int(omni.usd.get_context().get_stage_loading_status()[2])
                if frame % 4 == 0:
                    print(json.dumps({'event': 'render_progress', 'camera': name,
                        'update': frame + 1, 'elapsed_s': time.monotonic() - start,
                        'hydra': info}, default=str), flush=True)
                # The first product can reach its sample cap while textured
                # materials still appear black. SPP alone is not readiness.
                if (frame + 1 >= args.minimum_render_updates and pending_stage_loads == 0
                        and int(info.get('progression', 0)) >= args.samples):
                    break
            rgb = np.asarray(annotator.get_data())
            spp = int(info.get('progression', 0))
            if spp < args.samples:
                raise RuntimeError(f'{name} accumulated {spp}/{args.samples} samples')
            if pending_stage_loads != 0:
                raise RuntimeError(f'{name} still has {pending_stage_loads} pending stage loads')
            if rgb.shape not in ((args.height, args.width, 3), (args.height, args.width, 4)):
                raise RuntimeError('Invalid native RGB buffer: ' + str(rgb.shape))
            rgb = rgb[:, :, :3].astype(np.uint8)
            if rgb.std() < 2.:
                raise RuntimeError('Native frame is nearly blank')
            filename = output / (name + '.png')
            Image.fromarray(rgb).save(filename)
            report['images'][name] = {'path': str(filename), 'sha256': digest(filename),
                'size_px': [args.width, args.height], 'actual_samples_per_pixel': spp,
                'sampling_target_reached': True, 'camera': definition,
                'native_rtx': True, 'hydra_frame_info': info,
                'updates_used': frame + 1, 'pending_stage_loads': pending_stage_loads,
                'render_elapsed_s': time.monotonic() - start}
            dump(output / 'provenance.json', report)
            annotator.detach([product]); subscription = None; product.destroy()
        report['status'] = 'rendered'; report['completed_utc'] = now()
        exit_code = 0
    except BaseException as exc:
        report.update(status='failed', error=repr(exc), traceback=traceback.format_exc())
        traceback.print_exc()
    finally:
        dump(output / 'provenance.json', report)
        if app is not None:
            app.close(exit_code=exit_code)
    return exit_code


if __name__ == '__main__':
    raise SystemExit(main())
