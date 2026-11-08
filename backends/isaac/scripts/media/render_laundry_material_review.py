"""Static material comparison of an immutable recorded native cloth endpoint.

This never creates a World or starts physics. It verifies that the exported
render skin equals the archived native final midsurface, changes only explicit
material properties in a session layer, and reuses the static RTX capture tool.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import traceback

import numpy as np

from bench.paths import load_json

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT))

from scripts.maintenance.freeze_project_sources import verify_if_snapshot
from scripts.media.laundry_cloth_frames import media_directory, task_cameras
from scripts.media.render_task_gallery import capture_image
from scripts.media.render_workroom import digest, dump, now


def recorded_task_cameras(native_directory):
    """Use the target from the exact frozen task bound by the native report."""
    path = Path(native_directory) / 'native_report.json'
    native_report = load_json(path)
    entry = native_report['task_input']
    task_path = Path(entry['path'])
    if digest(task_path) != entry['sha256']:
        raise ValueError('Frozen task input differs from the recorded native report')
    task = load_json(task_path)
    center = task['asset_metadata']['target_center_xy_m']
    return task_cameras(center), {'native_report': {'path': str(path), 'sha256': digest(path)},
        'task_input': dict(entry), 'target_center_xy_m': list(center)}


def invariant_digest(stage):
    """Bind every World property except the cloth material and camera overrides."""
    checksum = hashlib.sha256()
    for prim in stage.Traverse():
        path = str(prim.GetPath())
        if not path.startswith('/World') or path.startswith((
                '/World/Looks/cloth_skin_', '/World/Cameras')):
            continue
        checksum.update(repr((path, prim.GetTypeName(), prim.GetAppliedSchemas())).encode())
        for attr in sorted(prim.GetAttributes(), key=lambda a: a.GetName()):
            value = attr.Get()
            checksum.update(repr((attr.GetName(), str(attr.GetTypeName()))).encode())
            array = np.asarray(value)
            if array.dtype.kind in 'biufc':
                checksum.update(repr((array.dtype.str, array.shape)).encode())
                checksum.update(array.tobytes())
            else:
                checksum.update(str(value).encode())
            checksum.update(str(attr.GetConnections()).encode())
        for rel in sorted(prim.GetRelationships(), key=lambda r: r.GetName()):
            checksum.update(repr((rel.GetName(), list(map(str, rel.GetTargets())))).encode())
    return checksum.hexdigest()


def verify_recorded_skin(stage, archive):
    from pxr import UsdGeom
    from bench.simulation.isaacsim_backend.cloth_visuals import closed_skin
    checked = {}
    with np.load(archive, allow_pickle=False) as arrays:
        ids = sorted(key[:-2] for key in arrays.files if key.endswith('_q'))
        if not ids:
            raise ValueError('No recorded cloth positions')
        for name in ids:
            mesh = UsdGeom.Mesh(stage.GetPrimAtPath('/World/ClothVisuals/' + name))
            if not mesh:
                raise ValueError('Recorded endpoint has no visual skin: ' + name)
            points = np.asarray(mesh.GetPointsAttr().Get(), dtype=np.float32)
            final = arrays[name + '_q'][-1]
            tri = arrays[name + '_triangles']
            count = len(final)
            # Source surfaceThickness is a material attribute, while the
            # closed render envelope is independently measurable here.
            separation = np.linalg.norm(points[:count] - points[count:], axis=1)
            thickness = float(np.median(separation))
            expected, faces, _ = closed_skin(final, tri, thickness)
            error = float(np.max(np.abs(points - expected)))
            if error > 2.e-7:
                raise ValueError('Exported skin differs from archived native endpoint')
            if not np.array_equal(np.asarray(mesh.GetFaceVertexIndicesAttr().Get()).reshape(-1, 3), faces):
                raise ValueError('Exported skin topology differs from recorded native topology')
            checked[name] = {'native_vertices': count, 'thickness_m': thickness,
                'max_skin_reconstruction_error_m': error,
                'native_final_q_sha256': hashlib.sha256(final.tobytes()).hexdigest(),
                'native_final_qd_sha256': hashlib.sha256(arrays[name + '_qd'][-1].tobytes()).hexdigest()}
    return checked


def apply_variant(stage, names, variant, albedo):
    from pxr import Gf, Sdf, UsdShade
    changes = []
    for name in names:
        base = '/World/Looks/cloth_skin_' + name
        if variant == 'original':
            continue
        tex = UsdShade.Shader(stage.GetPrimAtPath(base + '/normal'))
        if tex:
            tex.GetOutput('rgb').GetAttr().SetTypeName(Sdf.ValueTypeNames.Float3)
            changes.append({'path': base + '/normal.outputs:rgb', 'type': 'float3'})
        if variant != 'albedo':
            continue
        from PIL import Image
        pixels = np.asarray(Image.open(albedo).convert('RGB'), dtype=np.float64) / 255.
        linear = np.where(pixels <= .04045, pixels / 12.92, ((pixels + .055) / 1.055) ** 2.4)
        mean = linear.mean(axis=(0, 1))
        if np.any(mean <= 1.e-6):
            raise ValueError('Albedo has a zero mean channel; cannot preserve declared dye colors')
        surface = UsdShade.Shader(stage.GetPrimAtPath(base + '/Surface'))
        color = np.asarray(surface.GetInput('diffuseColor').Get(), dtype=float)
        scale = color / mean
        tex = UsdShade.Shader.Define(stage, base + '/review_albedo')
        tex.CreateIdAttr('UsdUVTexture')
        tex.CreateInput('file', Sdf.ValueTypeNames.Asset).Set(str(albedo))
        tex.CreateInput('sourceColorSpace', Sdf.ValueTypeNames.Token).Set('sRGB')
        tex.CreateInput('st', Sdf.ValueTypeNames.Float2).ConnectToSource(
            UsdShade.Shader(stage.GetPrimAtPath(base + '/ST')).ConnectableAPI(), 'result')
        tex.CreateInput('scale', Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(*map(float, scale), 1.))
        for key in ('wrapS', 'wrapT'):
            tex.CreateInput(key, Sdf.ValueTypeNames.Token).Set('repeat')
        tex.CreateOutput('rgb', Sdf.ValueTypeNames.Float3)
        surface.GetInput('diffuseColor').ConnectToSource(tex.ConnectableAPI(), 'rgb')
        changes.append({'path': base + '/Surface.inputs:diffuseColor', 'file': str(albedo),
            'sha256': digest(albedo), 'linear_mean_rgb': mean.tolist(),
            'scale_rgb': scale.tolist(), 'mean_dye_color_rgb': color.tolist(),
            'scope': 'Explicit appearance-only candidate; channel-normalized source albedo preserves blue/ochre dye identity.'})
    return changes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--native-directory', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--media-output', type=Path, required=True)
    parser.add_argument('--variants', nargs='+', choices=('original', 'types', 'albedo'), default=['original', 'types'])
    parser.add_argument('--albedo', type=Path)
    parser.add_argument('--width', type=int, default=1280)
    parser.add_argument('--height', type=int, default=960)
    parser.add_argument('--samples', type=int, default=64)
    parser.add_argument('--render-updates', type=int, default=100)
    parser.add_argument('--warmup-captures', type=int, default=1,
                        help='Preserve full warmup captures before each comparison image')
    args = parser.parse_args()
    if not 0 <= args.warmup_captures <= 2:
        parser.error('Use zero to two explicit bounded warmup captures')
    verified = verify_if_snapshot(PROJECT, required_modules=('scripts.media.render_task_gallery',),
                                  entrypoint_file=Path(__file__))
    if verified is None:
        parser.error('Run from a complete frozen source snapshot')
    gpu = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    if not gpu.startswith('GPU-') or ',' in gpu:
        parser.error('Select one physical GPU UUID')
    if 'albedo' in args.variants and (args.albedo is None or not args.albedo.is_file()):
        parser.error('Albedo variant requires an explicit existing --albedo')
    source = args.native_directory.resolve() / 'final.usda'
    archive = args.native_directory.resolve() / 'native_states.npz'
    cameras, camera_source = recorded_task_cameras(args.native_directory.resolve())
    output, media = args.output.resolve(), media_directory(args.media_output)
    output.mkdir(parents=True, exist_ok=False)
    media.mkdir(parents=True, exist_ok=False)
    report = {'schema': 'laundry-recorded-endpoint-material-review.v1', 'created_utc': now(),
        'source_snapshot': verified, 'status': 'initializing', 'gpu_uuid': gpu,
        'native_final_usd': {'path': str(source), 'sha256': digest(source)},
        'native_archive': {'path': str(archive), 'sha256': digest(archive)},
        'camera_source': camera_source,
        'scope': 'Static RTX presentation of an archived native endpoint; no new native operation or physical qualification.',
        'physics_steps': 0, 'native_state_writes': 0, 'physics_accepted': False,
        'geometry_changed': False, 'frozen_release_changed': False,
        'warmup_captures_per_candidate': args.warmup_captures, 'candidates': []}
    dump(output / 'report.json', report)
    app = None
    exit_code = 1
    try:
        from isaacsim import SimulationApp
        from bench.simulation.isaacsim_backend.render_sync import startup_arguments, verify_settings
        extra = startup_arguments()
        if os.environ.get('ISAAC_ALLOW_UNVALIDATED_DRIVER') == '1':
            extra.append('--/rtx/verifyDriverVersion/enabled=false')
        app = SimulationApp({'headless': True, 'width': 640, 'height': 480,
            'disable_viewport_updates': True, 'multi_gpu': False, 'active_cuda_gpus': [0],
            'max_gpu_count': 1, 'physics_gpu': 0, 'renderer': 'PathTracing',
            'fast_shutdown': True, 'extra_args': extra},
            experience=str(PROJECT / 'bench/apps/workroom.headless.kit'))
        import carb
        import omni.timeline
        import omni.usd
        from pxr import Usd, UsdGeom
        report['render_settings'] = verify_settings(carb.settings.get_settings())
        timeline = omni.timeline.get_timeline_interface()
        if timeline.is_playing():
            raise RuntimeError('Static review requires a stopped timeline')
        original_invariant = invariant_digest(Usd.Stage.Open(str(source)))
        for variant in args.variants:
            # final.usda also contains the old capture's /Render/SDGPipeline
            # graphs. Import its complete World into a fresh stage rather than
            # reactivating those stale render-product/annotator graph handles.
            omni.usd.get_context().new_stage()
            stage = omni.usd.get_context().get_stage()
            root = UsdGeom.Xform.Define(stage, '/World')
            root.GetPrim().GetReferences().AddReference(str(source), '/World')
            stage.SetDefaultPrim(root.GetPrim())
            UsdGeom.SetStageMetersPerUnit(stage, 1.)
            UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
            stage.SetEditTarget(stage.GetSessionLayer())
            check = verify_recorded_skin(stage, archive)
            before = invariant_digest(stage)
            if before != original_invariant:
                raise RuntimeError('Referenced World differs from original physical/geometry properties')
            clock_before = timeline.get_current_time()
            changes = apply_variant(stage, check, variant, args.albedo)
            if invariant_digest(stage) != before:
                raise RuntimeError('Material override changed geometry or physical properties')
            record = {'task_id': args.native_directory.name, 'variant': variant,
                'recorded_skin_check': check, 'overrides': changes, 'invariant_sha256': before}
            camera = {**cameras['thickness'], 'horizontal_aperture_mm': 36.,
                      'vertical_aperture_mm': 36. * args.height / args.width, 'clipping_m': [.01, 8.]}
            # SPP and pending USD loads do not establish texture residency.
            # Keep a separate full product/capture as a visible warmup witness,
            # then compare only the later capture from the same loaded stage.
            record['warmup_images'] = []
            for index in range(args.warmup_captures):
                warmup_dir = media / 'warmup'
                warmup_dir.mkdir(exist_ok=True)
                warmup = {'task_id': args.native_directory.name}
                capture_image(stage, app, f'{variant}_warmup_{index:02d}', camera,
                              warmup_dir, args, warmup)
                if (timeline.is_playing() or timeline.get_current_time() != clock_before
                        or invariant_digest(stage) != before):
                    raise RuntimeError('Warmup changed timeline, geometry or physical properties')
                record['warmup_images'].append(warmup['images'])
            capture_image(stage, app, variant, camera, media, args, record)
            if (timeline.is_playing() or timeline.get_current_time() != clock_before
                    or invariant_digest(stage) != before):
                raise RuntimeError('Static capture changed the timeline, geometry or physical properties')
            stage.GetSessionLayer().Export(str(output / (variant + '_session.usda')))
            record['session_layer_sha256'] = digest(output / (variant + '_session.usda'))
            report['candidates'].append(record)
            dump(output / 'report.json', report)
        if digest(source) != report['native_final_usd']['sha256'] or digest(archive) != report['native_archive']['sha256']:
            raise RuntimeError('Original native evidence changed during material review')
        for bound in (camera_source['native_report'], camera_source['task_input']):
            if digest(Path(bound['path'])) != bound['sha256']:
                raise RuntimeError('Native report or frozen camera target input changed during review')
        report['status'] = 'rendered'
        exit_code = 0
    except BaseException:
        report['status'] = 'failed'
        report['traceback'] = traceback.format_exc()
        traceback.print_exc()
    finally:
        report['completed_utc'] = now()
        dump(output / 'report.json', report)
        if app is not None:
            app.close(exit_code=exit_code)
    return exit_code


if __name__ == '__main__':
    raise SystemExit(main())
