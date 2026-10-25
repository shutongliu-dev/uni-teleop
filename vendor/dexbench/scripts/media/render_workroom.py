#!/usr/bin/env python3
"""Build, passively simulate and photograph an independent Isaac room scene.

There is no benchmark, Agent or task policy in this entrypoint. Physics only
holds the real articulated FR3 in its declared parked pose. The exported USD
retains shared asset references and PhysX schemas, cameras and material inputs.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback

from bench.paths import DATA_ROOT, load_json

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT))

from scripts.maintenance.freeze_project_sources import DEFAULT_EXTRAS, inventory, verify_if_snapshot
from bench.simulation.assets.workrooms import SOURCE_RESOURCES, shared_asset_links

EXTRAS = tuple(dict.fromkeys(DEFAULT_EXTRAS + SOURCE_RESOURCES))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


def dump(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2,
        default=lambda obj: obj.tolist() if hasattr(obj, 'tolist') else str(obj)) + '\n')


def as_numpy(value):
    if hasattr(value, 'detach'):
        value = value.detach().cpu()
    if hasattr(value, 'numpy'):
        value = value.numpy()
    return value.copy()


def source_inventory():
    return inventory(PROJECT, EXTRAS, shared_asset_links(PROJECT))


def robot_source(assets_root):
    root = assets_root / 'robots/assemblies/fr3_sharpa/isaacsim_6_1_v2'
    source = load_json(root / 'SOURCE.json')
    imported = load_json(root / 'USD_IMPORT.json')
    for path, expected in imported['derived_files'].items():
        if digest(path) != expected:
            raise RuntimeError('Shared robot USD dependency changed: ' + path)
    if digest(imported['usd_path']) != imported['usd_sha256']:
        raise RuntimeError('Shared robot root USD does not match its versioned manifest')
    return source, {'source_manifest': str(root / 'SOURCE.json'),
                    'source_manifest_sha256': digest(root / 'SOURCE.json'),
                    'import_manifest': str(root / 'USD_IMPORT.json'),
                    'import_manifest_sha256': digest(root / 'USD_IMPORT.json'),
                    'root_usd': imported['usd_path'],
                    'root_usd_sha256': imported['usd_sha256'],
                    'derived_files': imported['derived_files'],
                    'mount': source['mount'], 'licenses': source['licenses']}


def validate_colliders(stage):
    """Reject reflected/singular primitive collider transforms before PhysX."""
    import numpy as np
    from pxr import Usd, UsdGeom, UsdPhysics
    cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    counts = {'total': 0, 'cube': 0, 'robot': 0}
    invalid = []
    for prim in Usd.PrimRange.Stage(stage, Usd.TraverseInstanceProxies()):
        if not prim.HasAPI(UsdPhysics.CollisionAPI):
            continue
        counts['total'] += 1
        if str(prim.GetPath()).startswith('/World/Robot/'):
            counts['robot'] += 1
        matrix = np.asarray(cache.GetLocalToWorldTransform(prim), dtype=float)
        determinant = float(np.linalg.det(matrix[:3, :3]))
        if not np.isfinite(matrix).all() or determinant <= 0:
            invalid.append({'path': str(prim.GetPath()), 'determinant': determinant})
        if prim.IsA(UsdGeom.Cube):
            counts['cube'] += 1
            size = float(UsdGeom.Cube(prim).GetSizeAttr().Get())
            if not math.isfinite(size) or size <= 0:
                invalid.append({'path': str(prim.GetPath()), 'cube_size': size})
    if invalid:
        raise RuntimeError('Invalid collider dimensions/transforms: ' + repr(invalid[:20]))
    return {**counts, 'invalid': invalid,
            'scope': 'Finite positive transforms and cube sizes; not a contact acceptance test.'}


def author_robot_pose(stage, scene, source):
    """Layer initial joint state and hold drives over the unmodified robot USD."""
    import numpy as np
    from pxr import PhysxSchema, Usd, UsdPhysics
    root_path = scene.get('robot_root', '/World/Robot')
    root = stage.GetPrimAtPath(root_path)
    if not root:
        raise RuntimeError('Builder did not reference the robot: ' + root_path)
    supplied = scene['robot_joint_positions']
    if not isinstance(supplied, dict):
        raise TypeError('robot_joint_positions must map joint names to radians')
    unknown = set(supplied) - set(source['joint_limits'])
    if unknown:
        raise ValueError('Unknown robot joints: ' + repr(sorted(unknown)))
    q = {}
    roots, joints = [], {}
    for prim in Usd.PrimRange(root):
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            roots.append(str(prim.GetPath()))
            physx = PhysxSchema.PhysxArticulationAPI.Apply(prim)
            physx.CreateSolverPositionIterationCountAttr(32)
            physx.CreateSolverVelocityIterationCountAttr(8)
        if not prim.IsA(UsdPhysics.RevoluteJoint):
            continue
        name = prim.GetName()
        limit = source['joint_limits'][name]
        target = float(supplied.get(name, np.clip(0., limit['lower'], limit['upper'])))
        if not math.isfinite(target) or not limit['lower'] - 1.e-6 <= target <= limit['upper'] + 1.e-6:
            raise ValueError(f'{name} target {target} is outside source joint limits')
        q[name] = target
        joints[name] = str(prim.GetPath())
        drive = UsdPhysics.DriveAPI.Apply(prim, 'angular')
        drive.CreateTypeAttr('force')
        drive.CreateStiffnessAttr(3000. if name.startswith('fr3_') else 15.)
        drive.CreateDampingAttr(160. if name.startswith('fr3_') else .8)
        drive.CreateMaxForceAttr(float(limit['effort']))
        drive.CreateTargetPositionAttr(math.degrees(target))
        drive.CreateTargetVelocityAttr(0.)
        state = PhysxSchema.JointStateAPI.Apply(prim, 'angular')
        state.CreatePositionAttr(math.degrees(target))
        state.CreateVelocityAttr(0.)
    if len(roots) != 1 or set(q) != set(source['joint_limits']):
        raise RuntimeError(f'Expected one 29-DOF FR3/SHARPA articulation, got {roots}, {len(q)} joints')
    return roots[0], q, joints


def initialize_and_check(world, scene, articulation_path, joint_q, steps):
    """Reset-only pose writes followed by native stationary hold steps."""
    import numpy as np
    import torch
    world.reset()
    sim_view = world.physics_sim_view
    if sim_view is None:
        raise RuntimeError('Native World has no stage-bound PhysX tensor view')
    sim_view.set_subspace_roots('/')
    robot = sim_view.create_articulation_view(articulation_path)
    if robot.count != 1:
        raise RuntimeError('Native articulation was not created')
    names = list(robot.shared_metatype.dof_names)
    if set(names) != set(joint_q):
        raise RuntimeError('Native robot joint inventory differs from the USD')
    device = 'cuda:0'
    tensor = lambda value: torch.as_tensor(np.asarray(value), device=device).contiguous()
    index = torch.tensor([0], dtype=torch.int32, device=device)
    q = tensor([[joint_q[name] for name in names]]).to(torch.float32)
    base = np.asarray(scene['robot_base_position'], dtype=np.float32)
    quaternion = np.asarray(scene.get('robot_base_quaternion_wxyz', [1., 0., 0., 0.]), dtype=np.float32)
    quaternion /= np.linalg.norm(quaternion)
    root_tf = tensor([[*base, *quaternion[[1, 2, 3, 0]]]]).to(torch.float32)
    robot.set_root_transforms(root_tf, index)
    robot.set_root_velocities(torch.zeros((1, 6), device=device), index)
    robot.set_dof_positions(q, index)
    robot.set_dof_velocities(torch.zeros_like(q), index)
    robot.set_dof_position_targets(q, index)
    robot.set_dof_velocity_targets(torch.zeros_like(q), index)
    mask = tensor([[float(name.startswith('fr3_')) for name in names]]).to(torch.float32)
    q_before = as_numpy(robot.get_dof_positions())
    root_before = as_numpy(robot.get_root_transforms())
    start_step = int(world.current_time_step_index)
    start_time = float(world.current_time)
    max_root_translation_m, max_joint_deviation_rad = 0., 0.
    for _ in range(steps):
        gravity = robot.get_gravity_compensation_forces()
        robot.set_dof_actuation_forces((gravity[:, -len(names):] * mask).contiguous(), index)
        world.step(render=False)
        actual_q = as_numpy(robot.get_dof_positions())
        actual_root = as_numpy(robot.get_root_transforms())
        if not np.isfinite(actual_q).all() or not np.isfinite(actual_root).all():
            raise RuntimeError('Native passive simulation produced nonfinite robot state')
        max_root_translation_m = max(max_root_translation_m,
            float(np.linalg.norm(actual_root[0, :3] - root_before[0, :3])))
        max_joint_deviation_rad = max(max_joint_deviation_rad, float(np.max(np.abs(actual_q - q_before))))
    final_q = as_numpy(robot.get_dof_positions())
    final_qd = as_numpy(robot.get_dof_velocities())
    actual_steps = int(world.current_time_step_index) - start_step
    elapsed = float(world.current_time) - start_time
    passed = (actual_steps == steps and max_root_translation_m < .001
              and max_joint_deviation_rad < .08 and np.isfinite(final_qd).all())
    result = {
        'status': 'passed' if passed else 'failed', 'scope': 'Parked robot passive stability only; no manipulation/task qualification.',
        'physics_dt_s': float(world.get_physics_dt()), 'requested_steps': steps,
        'actual_native_steps': actual_steps, 'native_elapsed_s': elapsed,
        'max_root_translation_m': max_root_translation_m,
        'max_joint_deviation_rad': max_joint_deviation_rad,
        'final_max_joint_velocity_rad_s': float(np.max(np.abs(final_qd))),
        'limits': {'root_translation_m': .001, 'joint_deviation_rad': .08},
        'joint_positions_initial_rad': dict(zip(names, q_before[0].tolist())),
        'joint_positions_final_rad': dict(zip(names, final_q[0].tolist())),
        'arm_gravity_compensation': True, 'hand_gravity_compensation': False,
        'pose_writes_after_initialization': 0,
        'hold_drives': {'arm_kp': 3000., 'arm_kd': 160., 'hand_kp': 15., 'hand_kd': .8,
                        'effort_limits': 'Shared source URDF limits'},
    }
    return robot, result


def create_camera(stage, name, definition):
    from pxr import Gf, UsdGeom
    path = '/World/Cameras/' + name
    camera = UsdGeom.Camera.Define(stage, path)
    eye = Gf.Vec3d(*map(float, definition['eye']))
    target = Gf.Vec3d(*map(float, definition['target']))
    matrix = Gf.Matrix4d().SetLookAt(eye, target, Gf.Vec3d(0., 0., 1.)).GetInverse()
    transform = UsdGeom.Xformable(camera)
    transform.ClearXformOpOrder()
    transform.AddTransformOp().Set(matrix)
    camera.CreateFocalLengthAttr(float(definition.get('focal_length_mm', 24.)))
    camera.CreateHorizontalApertureAttr(36.)
    camera.CreateClippingRangeAttr(Gf.Vec2f(.02, 100.))
    return path


def validate_camera_geometry(stage, cameras):
    """Check viewpoint and sight line against actual room-wall colliders.

    This deliberately does not infer visibility through furniture from loose
    bounding boxes. Native image inspection remains necessary for composition.
    """
    import numpy as np
    from pxr import Gf, Usd, UsdGeom, UsdPhysics
    cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    boxes = []
    for prim in Usd.PrimRange.Stage(stage, Usd.TraverseInstanceProxies()):
        path = str(prim.GetPath())
        parts = path.split('/')
        name = parts[3] if len(parts) > 3 and parts[1:3] == ['World', 'Geometry'] else ''
        if not name.startswith('wall_'):
            continue
        if not prim.IsA(UsdGeom.Cube) or not prim.HasAPI(UsdPhysics.CollisionAPI):
            continue
        matrix = cache.GetLocalToWorldTransform(prim)
        scales = np.linalg.norm(np.asarray(matrix)[:3, :3], axis=1)
        boxes.append((path, matrix.GetInverse(), scales,
                      float(UsdGeom.Cube(prim).GetSizeAttr().Get()) * .5))
    if not boxes:
        raise RuntimeError('Camera validation found no room-wall box colliders')
    records, failures = {}, []
    for name, definition in cameras.items():
        eye = np.asarray(definition['eye'], dtype=float)
        target = np.asarray(definition['target'], dtype=float)
        if (eye.shape != (3,) or target.shape != (3,) or not np.isfinite([eye, target]).all()
                or np.linalg.norm(eye - target) < .02):
            raise ValueError('Invalid camera eye/target: ' + name)
        collisions = []
        min_clearance = math.inf
        for path, inverse, scales, half in boxes:
            start = np.asarray(inverse.Transform(Gf.Vec3d(*eye)), dtype=float)
            end = np.asarray(inverse.Transform(Gf.Vec3d(*target)), dtype=float)
            direction = end - start
            clearance = float(np.linalg.norm(np.maximum(np.abs(start) - half, 0.) * scales))
            min_clearance = min(min_clearance, clearance)
            enter, leave = 0., 1.
            intersects = True
            for axis in range(3):
                if abs(direction[axis]) < 1.e-12:
                    if abs(start[axis]) > half:
                        intersects = False
                        break
                else:
                    low, high = sorted(((-half - start[axis]) / direction[axis],
                                        (half - start[axis]) / direction[axis]))
                    enter, leave = max(enter, low), min(leave, high)
                    if enter > leave:
                        intersects = False
                        break
            if intersects or clearance < .02:
                collisions.append({'path': path, 'eye_clearance_m': clearance,
                    'eye_target_segment_intersects': intersects,
                    'entry_fraction': float(enter) if intersects else None})
        focal = float(definition.get('focal_length_mm', 24.))
        if not math.isfinite(focal) or focal <= 0:
            raise ValueError('Camera focal length must be positive: ' + name)
        records[name] = {'structural_clearance_m': min_clearance,
            'eye_target_distance_m': float(np.linalg.norm(target - eye)),
            'horizontal_fov_deg': math.degrees(2. * math.atan(18. / focal)),
            'structural_collisions': collisions, 'passed': not collisions}
        if collisions:
            failures.append({'camera': name, 'collisions': collisions})
    if failures:
        raise RuntimeError('Camera intersects or views through room structure: ' + repr(failures))
    return {'structural_box_count': len(boxes), 'cameras': records, 'passed': True,
            'scope': 'Actual room-wall box colliders and 2 cm viewpoint clearance; furniture occlusion, ceiling clearance and framing require separate review.'}


def snapshot_link_poses(robot):
    """Read actual PhysX body-frame poses, without changing simulation state."""
    import numpy as np
    paths = robot.link_paths
    if paths and not isinstance(paths[0], str):
        paths = paths[0]
    poses = as_numpy(robot.get_link_transforms())[0]
    if len(paths) != len(poses) or not np.isfinite(poses).all():
        raise RuntimeError('Invalid native link transform inventory')
    return {'schema': 'scence.native_link_poses.v1',
            'source': 'PhysX articulation get_link_transforms, world coordinates',
            'pose_order': 'x y z qx qy qz qw',
            'links': {str(path): pose.tolist() for path, pose in zip(paths, poses)}}


def bake_native_link_poses(stage, snapshot):
    """Write the native parked pose only to the offline exported USD layer.

    Fabric rendering can show a pose that was never authored into USD. Initial
    JointState metadata alone does not pose a scene opened without Play.
    """
    import numpy as np
    from pxr import Gf, Usd, UsdGeom
    targets = {}
    for path, pose in snapshot['links'].items():
        x, y, z, qx, qy, qz, qw = map(float, pose)
        quat = Gf.Quatd(qw, Gf.Vec3d(qx, qy, qz)).GetNormalized()
        matrix = Gf.Matrix4d().SetRotate(quat)
        matrix.SetTranslateOnly(Gf.Vec3d(x, y, z))
        targets[path] = matrix
    cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    for path in sorted(targets, key=lambda p: (p.count('/'), p)):
        prim = stage.GetPrimAtPath(path)
        if not prim:
            raise RuntimeError('Export lacks native articulation link: ' + path)
        parent = cache.GetLocalToWorldTransform(prim.GetParent())
        local = targets[path] * parent.GetInverse()
        xform = UsdGeom.Xformable(prim)
        xform.ClearXformOpOrder()
        xform.AddTransformOp(UsdGeom.XformOp.PrecisionDouble, 'nativePose').Set(local)
        cache.Clear()
    max_error = max(float(np.max(np.abs(np.asarray(cache.GetLocalToWorldTransform(
        stage.GetPrimAtPath(path))) - np.asarray(target)))) for path, target in targets.items())
    if max_error > 1.e-7:
        raise RuntimeError(f'Exported robot world transforms differ from native snapshot: {max_error}')
    return {'source': snapshot['source'], 'body_count': len(targets),
            'max_world_matrix_absolute_error': max_error,
            'purpose': 'Static USD open matches the rendered physical robot pose'}


def capture(stage, world, robot, cameras, output, args, provenance, app):
    import numpy as np
    import omni.replicator.core as rep
    import omni.hydratexture
    from carb.eventdispatcher import get_eventdispatcher
    from PIL import Image
    record = {}
    # Synchronize the last physical pose once, then leave Fabric untouched.
    # World.render() force-updates articulation/Fabric every call, which can
    # invalidate static path-tracing accumulation even without a physics step.
    before_pause_q = as_numpy(robot.get_dof_positions())
    before_pause_time = float(world.current_time)
    world.render()
    world.pause()
    if (float(world.current_time) != before_pause_time
            or not np.array_equal(before_pause_q, as_numpy(robot.get_dof_positions()))):
        raise RuntimeError('Pausing the render scene altered its physical state')
    provenance['render_state_mode'] = 'Paused timeline; app.update after one Fabric synchronization'
    for name, camera in cameras.items():
        path = create_camera(stage, name, camera)
        product = rep.create.render_product(path, (args.width, args.height))
        # Kit 110 stores renderer settings on each USD RenderProduct. Deprecated
        # global carb keys alone do not override the inherited 64 spp product cap.
        from pxr import Sdf
        rp = stage.GetPrimAtPath(product.path)
        product_settings = {}
        if args.renderer == 'PathTracing':
            requested = {
                'omni:rtx:pt:samplesPerPixel': (Sdf.ValueTypeNames.UInt, args.samples),
                'omni:rtx:pt:samplesPerIteration': (Sdf.ValueTypeNames.Int, min(args.samples, 32)),
                'omni:rtx:pt:clampSpp': (Sdf.ValueTypeNames.Int, args.samples),
                'omni:rtx:pt:limits:maxBounces': (Sdf.ValueTypeNames.UInt, 8),
                'omni:rtx:pt:denoising:enabled': (Sdf.ValueTypeNames.Bool, True),
                'omni:rtx:pt:adaptiveSampling:enabled': (Sdf.ValueTypeNames.Bool, False),
            }
            for attr_name, (kind, value) in requested.items():
                attr = rp.CreateAttribute(attr_name, kind)
                attr.Set(value)
                product_settings[attr_name] = attr.Get()
        rgb_annotator = rep.AnnotatorRegistry.get_annotator('rgb')
        rgb_annotator.attach([product])
        frame_info = {}
        def on_frame(event):
            try:
                info = product.hydra_texture.get_frame_info(event['result_handle'])
                frame_info.update({key: info[key] for key in
                    ('progression', 'frame_number', 'subframe_count', 'status', 'fps') if key in info})
            except Exception as exc:
                frame_info['stats_error'] = repr(exc)
        frame_subscription = get_eventdispatcher().observe_event(
            observer_name='scence_' + name, event_name=omni.hydratexture.GLOBAL_EVENT_DRAWABLE_CHANGED,
            on_event=on_frame, filter=product.hydra_texture.get_event_key())
        initial_q = as_numpy(robot.get_dof_positions())
        initial_root = as_numpy(robot.get_root_transforms())
        initial_time = float(world.current_time)
        initial_step = int(world.current_time_step_index)
        render_started = time.monotonic()
        preview_records = []
        # Each synchronous app.update performs Hydra/RTX work on the paused stage.
        # Path tracing sample budget is set explicitly, and temporal convergence is
        # allowed after the render product and material network have been created.
        for frame in range(args.render_updates):
            app.update()
            if frame == 0 or (frame + 1) % 4 == 0 or frame == args.render_updates - 1:
                print(json.dumps({'event': 'render_progress', 'camera': name,
                    'update': frame + 1, 'updates': args.render_updates,
                    'elapsed_s': time.monotonic() - render_started,
                    'hydra_frame_info': frame_info}, default=str), flush=True)
            if (frame + 1) % 8 == 0:
                preview = np.asarray(rgb_annotator.get_data())
                if preview.shape in ((args.height, args.width, 3), (args.height, args.width, 4)):
                    preview_dir = output / '_previews'
                    preview_dir.mkdir(exist_ok=True)
                    preview_path = preview_dir / f'{name}_update{frame + 1:04d}.png'
                    Image.fromarray(preview[:, :, :3].astype(np.uint8)).save(preview_path)
                    preview_records.append({'path': str(preview_path), 'update': frame + 1,
                        'elapsed_s': time.monotonic() - render_started, 'hydra_frame_info': dict(frame_info),
                        'purpose': 'Intermediate native diagnostic; not the final selected image'})
                    print(json.dumps({'event': 'intermediate_image', 'path': str(preview_path)}), flush=True)
            if (args.renderer == 'PathTracing' and frame >= 3
                    and int(frame_info.get('progression', 0)) >= args.samples):
                print(json.dumps({'event': 'sampling_target_reached', 'camera': name,
                    'actual_samples_per_pixel': int(frame_info['progression']),
                    'updates_used': frame + 1, 'elapsed_s': time.monotonic() - render_started}), flush=True)
                break
        rgb = np.asarray(rgb_annotator.get_data())
        if rgb.shape not in ((args.height, args.width, 3), (args.height, args.width, 4)):
            raise RuntimeError(f'Invalid native RGB buffer: {rgb.shape}')
        unchanged = (float(world.current_time) == initial_time
            and int(world.current_time_step_index) == initial_step
            and np.array_equal(initial_q, as_numpy(robot.get_dof_positions()))
            and np.array_equal(initial_root, as_numpy(robot.get_root_transforms())))
        if not unchanged:
            raise RuntimeError('Rendering unexpectedly changed physical state')
        rgb = rgb[:, :, :3].astype(np.uint8)
        if float(rgb.std()) < 2.:
            raise RuntimeError('Native rendering produced a nearly blank frame')
        observed_spp = int(frame_info.get('progression', 0))
        if args.renderer == 'PathTracing' and observed_spp < min(64, args.samples):
            raise RuntimeError(f'PathTracing accumulated only {observed_spp} spp; insufficient diagnostic/final samples')
        image_path = output / (name + '.png')
        Image.fromarray(rgb).save(image_path)
        record[name] = {'path': str(image_path), 'sha256': digest(image_path),
                        'camera': camera, 'native_rtx': True, 'size_px': [args.width, args.height],
                        'physical_state_unchanged_by_render': unchanged,
                        'render_elapsed_s': time.monotonic() - render_started,
                        'hydra_frame_info': dict(frame_info), 'intermediate_previews': preview_records,
                        'render_product_settings': product_settings, 'updates_used': frame + 1,
                        'actual_samples_per_pixel': observed_spp,
                        'sampling_target_reached': (observed_spp >= args.samples if args.renderer == 'PathTracing' else None),
                        'mean_rgb': rgb.mean(axis=(0, 1)).tolist(),
                        'std_rgb': rgb.std(axis=(0, 1)).tolist()}
        provenance['images'] = record
        dump(output / 'provenance.json', provenance)
        rgb_annotator.detach([product])
        frame_subscription = None
        product.destroy()
        print(json.dumps({'event': 'image_saved', 'camera': name, 'path': str(image_path)}), flush=True)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--family', required=True)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--assets-root', type=Path, default=Path(f'{DATA_ROOT}/assets'))
    parser.add_argument('--width', type=int, default=1920)
    parser.add_argument('--height', type=int, default=1440)
    parser.add_argument('--renderer', choices=['PathTracing', 'RayTracedLighting'], default='PathTracing')
    parser.add_argument('--samples', type=int, default=256)
    parser.add_argument('--render-updates', type=int, default=40)
    parser.add_argument('--settle-steps', type=int, default=120)
    parser.add_argument('--cameras', nargs='+')
    parser.add_argument('--export-only', action='store_true')
    args = parser.parse_args()
    if min(args.width, args.height, args.samples, args.render_updates, args.settle_steps) <= 0:
        parser.error('Dimensions, sample counts, updates and settle steps must be positive')
    gpu = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    if not gpu.startswith('GPU-') or ',' in gpu:
        parser.error('CUDA_VISIBLE_DEVICES must select exactly one physical GPU UUID')
    allow_driver = os.environ.get('ISAAC_ALLOW_UNVALIDATED_DRIVER') == '1'
    snapshot_check = verify_if_snapshot(
        PROJECT, required_modules=('bench.simulation.assets.workrooms',), entrypoint_file=Path(__file__))
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        parser.error('Use an empty output directory; previous attempts are preserved')
    output.mkdir(parents=True, exist_ok=True)
    source, robot_record = robot_source(args.assets_root)
    experience = PROJECT / 'bench/apps/workroom.headless.kit'
    provenance = {'schema': 'scence.native_room_preview.v1', 'status': 'initializing',
        'family': args.family, 'created_at_utc': now(), 'project_root': str(PROJECT),
        'source_files_sha256': source_inventory(), 'argv': sys.argv,
        'source_snapshot_runtime': snapshot_check,
        'robot_asset': robot_record, 'runtime': 'Isaac Sim 6.1.0 / PhysX / native RTX',
        'cuda_gpu_uuid': gpu, 'isaac_allow_unvalidated_driver': allow_driver,
        'driver_support_status': ('Explicit unvalidated host-driver compatibility override' if allow_driver else 'Default driver validation'),
        'experience': str(experience), 'renderer': args.renderer,
        'samples_per_pixel_target': args.samples, 'render_updates_per_camera': args.render_updates,
        'model_calls': 0, 'task_qualification': None,
        'scope': 'Scene asset construction, parked robot stability and native presentation rendering.'}
    dump(output / 'provenance.json', provenance)
    app = world = None
    exit_code = 1
    try:
        from isaacsim import SimulationApp
        app = SimulationApp({'headless': True, 'width': args.width, 'height': args.height,
            'multi_gpu': False, 'active_cuda_gpus': [0], 'max_gpu_count': 1,
            'physics_gpu': 0, 'renderer': args.renderer, 'fast_shutdown': True,
            'extra_args': ['--/rtx/verifyDriverVersion/enabled=false'] if allow_driver else []},
            experience=str(experience))
        import carb
        import omni.usd
        from isaacsim.core.api import World
        from pxr import PhysxSchema, Usd, UsdGeom
        from bench.simulation.assets.workrooms.builder import build_scene
        from bench.simulation.assets.workrooms import common, interiors, layouts, props
        provenance['source_snapshot_runtime'] = verify_if_snapshot(
            PROJECT, required_modules=(
                'bench.simulation.assets.workrooms.builder', 'bench.simulation.assets.workrooms.common',
                'bench.simulation.assets.workrooms.interiors', 'bench.simulation.assets.workrooms.layouts',
                'bench.simulation.assets.workrooms.props'), entrypoint_file=Path(__file__))
        settings = carb.settings.get_settings()
        for name in ('updateToUsd', 'updateVelocitiesToUsd', 'updateParticlesToUsd'):
            settings.set_bool('/physics/' + name, True)
        if args.renderer == 'PathTracing':
            settings.set_int('/rtx/pathtracing/spp', min(args.samples, 32))
            settings.set_int('/rtx/pathtracing/totalSpp', args.samples)
            settings.set_int('/rtx/pathtracing/clampSpp', args.samples)
            settings.set_int('/rtx/pathtracing/maxBounces', 8)
            settings.set_bool('/rtx/pathtracing/optixDenoiser/enabled', True)
            settings.set_bool('/rtx/pathtracing/adaptiveSampling/enabled', False)
        provenance['renderer_settings'] = {name: settings.get(name) for name in (
            '/rtx/rendermode', '/rtx/pathtracing/spp', '/rtx/pathtracing/totalSpp',
            '/rtx/pathtracing/clampSpp', '/rtx/pathtracing/maxBounces',
            '/rtx/pathtracing/optixDenoiser/enabled')}
        omni.usd.get_context().new_stage()
        stage = omni.usd.get_context().get_stage()
        world = World(stage_units_in_meters=1., physics_dt=1. / 240., rendering_dt=1. / 30.,
                      backend='torch', device='cuda:0')
        context = world.get_physics_context()
        context.enable_gpu_dynamics(True)
        context.set_broadphase_type('GPU')
        context.set_solver_type('TGS')
        PhysxSchema.PhysxSceneAPI.Apply(stage.GetPrimAtPath(context.prim_path)).CreateEnableCCDAttr(False)
        scene = build_scene(stage, args.family, args.assets_root, output)
        dump(output / 'scene_description.json', scene)
        provenance['scene_description_sha256'] = digest(output / 'scene_description.json')
        provenance['collider_geometry_check'] = validate_colliders(stage)
        articulation_path, joint_q, joint_paths = author_robot_pose(stage, scene, source)
        robot, stability = initialize_and_check(world, scene, articulation_path, joint_q, args.settle_steps)
        provenance['stability'] = stability
        dump(output / 'stability.json', stability)
        if stability['status'] != 'passed':
            raise RuntimeError('Parked native articulation did not pass stability limits; see stability.json')
        cameras = scene['cameras']
        if args.cameras:
            cameras = {name: cameras[name] for name in args.cameras}
        provenance['camera_geometry_check'] = validate_camera_geometry(stage, cameras)
        for name, definition in cameras.items():
            create_camera(stage, name, definition)
        if not args.export_only:
            capture(stage, world, robot, cameras, output, args, provenance, app)
        link_snapshot = snapshot_link_poses(robot)
        snapshot_path = output / 'robot_pose_snapshot.json'
        dump(snapshot_path, link_snapshot)
        provenance['robot_pose_snapshot'] = {'path': str(snapshot_path), 'sha256': digest(snapshot_path)}
        # Export the root layer, preserving reusable asset references. Updating
        # JointState on the live GPU stage invokes forbidden CPU state setters.
        # The initial-state metadata belongs on the exported, offline stage only.
        scene_path = output / 'scene.usda'
        if not stage.GetRootLayer().Export(str(scene_path)):
            raise RuntimeError('USD root layer export failed')
        reopened = Usd.Stage.Open(str(scene_path))
        if not reopened or not reopened.GetPrimAtPath(scene.get('robot_root', '/World/Robot')):
            raise RuntimeError('Exported USD failed to reopen with its robot reference')
        for name, position in stability['joint_positions_final_rad'].items():
            prim = reopened.GetPrimAtPath(joint_paths[name])
            state = PhysxSchema.JointStateAPI.Apply(prim, 'angular')
            state.CreatePositionAttr(math.degrees(position))
            state.CreateVelocityAttr(0.)
        provenance['robot_pose_export'] = bake_native_link_poses(reopened, link_snapshot)
        reopened.GetRootLayer().Save()
        provenance['scene_usd'] = {'path': str(scene_path), 'sha256': digest(scene_path),
                                   'reopened': True, 'meters_per_unit': UsdGeom.GetStageMetersPerUnit(reopened),
                                   'up_axis': str(UsdGeom.GetStageUpAxis(reopened)),
                                   'reference_policy': 'Shared robot and textures remain external, versioned references'}
        provenance['status'] = 'exported' if args.export_only else 'rendered'
        provenance['completed_at_utc'] = now()
        dump(output / 'provenance.json', provenance)
        print(json.dumps({'event': 'scene_complete', 'output': str(output),
                          'status': provenance['status']}), flush=True)
        exit_code = 0
    except BaseException as exc:
        provenance['status'] = 'failed'
        provenance['error'] = repr(exc)
        provenance['traceback'] = traceback.format_exc()
        dump(output / 'provenance.json', provenance)
        traceback.print_exc()
    finally:
        if app is not None:
            app.close(exit_code=exit_code)
    return exit_code


if __name__ == '__main__':
    raise SystemExit(main())
