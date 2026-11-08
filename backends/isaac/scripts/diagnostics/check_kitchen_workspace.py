"""Check kitchen object access with source IK and actual USD collider bounds.

This read-only CPU diagnostic places the original soft-curl middle fingertip
above each source/goal object and its handle when present.  Anonymous USD
overrides own every probe; published assets are never edited.  No grasp,
trajectory, controller, native contact or task-success claim is made.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess

import numpy as np

from bench.paths import load_json

PROJECT = Path(__file__).resolve().parents[2]
NEAR_CLEARANCE_M = .015
APPROACH_CLEARANCE_M = .080
MINIMUM_COLLIDER_GAP_M = .005


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def nearest(first, second, first_names, second_names):
    """Conservative per-shape bounds, never an aggregate robot box."""
    first, second = np.asarray(first), np.asarray(second)
    if (first.ndim != 3 or second.ndim != 3 or first.shape[1:] != (2, 3)
            or second.shape[1:] != (2, 3) or not len(first) or not len(second)
            or not np.isfinite(first).all() or not np.isfinite(second).all()
            or (first[:, 1] < first[:, 0]).any() or (second[:, 1] < second[:, 0]).any()):
        raise ValueError('Finite nonempty ordered Nx2x3 collision bounds required')
    delta = np.maximum(np.maximum(first[:, None, 0]-second[None, :, 1],
                                  second[None, :, 0]-first[:, None, 1]), 0.)
    gaps = np.linalg.norm(delta, axis=2)
    i, j = np.unravel_index(np.argmin(gaps), gaps.shape)
    return {'minimum_aabb_gap_m': float(gaps[i, j]),
            'robot_shape': first_names[i], 'obstacle_shape': second_names[j],
            'clearance_over_5mm': bool(gaps[i, j] > MINIMUM_COLLIDER_GAP_M),
            'overlap_pair_count': int((gaps == 0).sum())}


def probe_rotations(point, base, *, profile='original_v1'):
    """Fixed finite orientation set; every attempted pose remains in receipts.

    Tilting the fingers towards the object tests genuine fingertip access at
    the far edge, instead of incorrectly requiring the wrist over that edge.
    This is a workspace search, not an operational grasp orientation policy.
    """
    from bench.geometry import rpy_matrix, rotvec_matrix
    direction = np.asarray(point)[:2]-np.asarray(base)[:2]
    direction /= max(float(np.linalg.norm(direction)), 1.e-12)
    axis = np.r_[direction[1], -direction[0], 0.]
    if profile not in ('original_v1', 'expanded_v1'):
        raise ValueError('Unknown workspace orientation profile: '+profile)
    original = [(tilt, yaw) for tilt in (0., 30., 55.)
                for yaw in (45., 135., -45., -135.)]
    extra = [(tilt, yaw) for tilt in (0., 15., 30., 45., 55., 70.)
             for yaw in (45., 135., -45., -135., 0., 90., 180., -90.)
             if (tilt, yaw) not in original] if profile == 'expanded_v1' else []
    for tilt, yaw in [*original, *extra]:
        rotation = rotvec_matrix(axis*math.radians(tilt)) @ rpy_matrix([math.pi, 0., math.radians(yaw)])
        yield {'tilt_from_vertical_deg': tilt, 'downward_yaw_deg': yaw}, rotation


def wrist_for_tip(point, tip_local, rotation, clearance):
    point, tip_local, rotation = map(np.asarray, (point, tip_local, rotation))
    if (point.shape != (3,) or tip_local.shape != (3,) or rotation.shape != (3, 3)
            or not all(np.isfinite(v).all() for v in (point, tip_local, rotation))
            or not math.isfinite(clearance) or clearance < 0):
        raise ValueError('Finite point, fingertip, rotation and nonnegative clearance required')
    result = np.eye(4)
    result[:3, :3] = rotation
    result[:3, 3] = point+[0., 0., clearance]-rotation @ tip_local
    return result


def pose_matrix(pose_xyzw):
    from bench.geometry import quaternion_matrix
    pose = np.asarray(pose_xyzw, dtype=float)
    if pose.shape != (7,) or not np.isfinite(pose).all() or not np.isclose(np.linalg.norm(pose[3:]), 1., atol=1.e-7):
        raise ValueError('Finite position plus unit xyzw quaternion required')
    result = np.eye(4)
    result[:3, :3] = quaternion_matrix(pose[[6, 3, 4, 5]])
    result[:3, 3] = pose[:3]
    return result


def composite_volume_poses(metadata, poses):
    """Move each undeformed material component with its parent for static IK.

    This is an anonymous USD rest-shape probe, never a runtime node write or a
    statement that a deformable keeps its rest shape during manipulation.
    """
    originals = {row['id']: row['initial_pose_xyzw'] for row in metadata['objects']}
    result = {}
    for volume in metadata.get('volume_bodies', {}).values():
        parent = volume['parent_rigid_body']
        if parent not in originals or parent not in poses:
            raise ValueError('Volume workspace component lacks its declared rigid parent')
        result[volume['prim_path']] = (pose_matrix(poses[parent]) @ np.linalg.inv(pose_matrix(originals[parent]))
                                       @ pose_matrix(volume['initial_pose_xyzw']))
    return result


def physical_collider_paths(metadata, obj):
    """Logical material objects include every real rigid/volume collider."""
    paths = list(obj['collider_paths'])
    for volume in metadata.get('volume_bodies', {}).values():
        if volume['parent_rigid_body'] == obj['id']:
            paths.append(volume['collision_mesh_path'])
    composite = metadata.get('elastic_composites', {}).get(obj['id'])
    if composite:
        rows = {row['id']: row for row in metadata['objects']}
        if composite['parent_rigid_body'] != obj['id'] or obj['id'] not in composite['member_ids']:
            raise ValueError('Invalid physical composite parent')
        paths = [path for name in composite['member_ids'] for path in rows[name]['collider_paths']]
    if len(set(paths)) != len(paths):
        raise ValueError('Distinct full physical collider inventory required')
    return paths


def checked_candidate(value, task_id, actors):
    """Validate explicit anonymous-USD experiments; never edit a release."""
    if value is None:
        return {'named_features': {}, 'pose_overrides': {}, 'ik_multistarts': 0}
    if (not isinstance(value, dict) or value.get('schema') != 'kitchen_workspace_candidate.v1'
            or value.get('task_id') != task_id or set(value)-{
                'schema', 'task_id', 'named_features', 'pose_overrides', 'ik_multistarts', 'note'}):
        raise ValueError('Workspace candidate schema/task/fields do not match')
    result = {'named_features': {}, 'pose_overrides': {}, 'ik_multistarts': value.get('ik_multistarts', 0)}
    if type(result['ik_multistarts']) is not int or not 0 <= result['ik_multistarts'] <= 4:
        raise ValueError('Workspace candidate ik_multistarts must be an integer from 0 to 4')
    if not isinstance(value.get('named_features', {}), dict) or not isinstance(value.get('pose_overrides', {}), dict):
        raise ValueError('Workspace feature and pose candidate maps must be dictionaries')
    for key, name in value.get('named_features', {}).items():
        parts = key.split(':')
        if (len(parts) != 2 or parts[0] not in actors or parts[1] not in ('source', 'goal')
                or not isinstance(name, str) or not name or '/' in name):
            raise ValueError('Invalid workspace named feature selection')
        result['named_features'][key] = name
    for state, poses in value.get('pose_overrides', {}).items():
        if state not in ('source', 'goal') or not isinstance(poses, dict) or not set(poses) <= set(actors):
            raise ValueError('Invalid workspace candidate object/state')
        for pose in poses.values():
            pose_matrix(pose)
        result['pose_overrides'][state] = dict(poses)
    return result


def run(release, *, object_ids=None, ik_only=False, orientation_profile='original_v1', sample_keys=None,
        candidate=None):
    from pxr import Gf, Usd, UsdGeom, UsdPhysics
    from bench.control.factory import public_robot_geometry, robot_configuration
    from scenes.workrooms.kitchen.release import validate_release

    release = Path(release).resolve()
    manifest = load_json(release/'release_manifest.json')
    validate_release(release, manifest, assets=True)
    scene, task, initial = [load_json(release/name) for name in
                            ('scene.json', 'task.json', 'initial_state.json')]
    metadata = task['asset_metadata']
    probe = checked_candidate(candidate, task['id'], metadata['actors'])
    model = public_robot_geometry(robot_configuration(scene))
    ready = model.checked_joints(initial['joint_positions_rad'], enforce_limits=True)
    tip_local = np.asarray(model.public_geometry(ready)['fingertip_positions_m']['middle'])
    robot_root = Path(scene['robot']['assets_root'])/'robots/assemblies/fr3_sharpa'/scene['robot']['assembly_version']
    imported = load_json(robot_root/'USD_IMPORT.json')
    bindings = {str(release/name): digest(release/name) for name in
                ('release_manifest.json', 'scene.json', 'task.json', 'initial_state.json')}
    bindings.update({imported['usd_path']: imported['usd_sha256'], **imported['derived_files']})
    for path, expected in bindings.items():
        if digest(path) != expected:
            raise ValueError('Changed workspace input: '+path)
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageMetersPerUnit(stage, 1.)
    UsdGeom.SetStageUpAxis(stage, 'Z')
    UsdGeom.Xform.Define(stage, '/World')
    for name, path, root in (
        ('Environment', scene['workroom']['usd_path'], scene['workroom']['source_prim']),
        ('Task', scene['native_task_bundle']['usd_path'], scene['native_task_bundle']['root_prim_path']),
        ('Robot', imported['usd_path'], '/fr3_sharpa')):
        UsdGeom.Xform.Define(stage, '/World/'+name).GetPrim().GetReferences().AddReference(path, root)
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ['default', 'render', 'proxy', 'guide'], False, True)
    transforms = UsdGeom.XformCache(Usd.TimeCode.Default())

    def bound(prim):
        value = cache.ComputeWorldBound(prim).ComputeAlignedRange()
        result = np.asarray([value.GetMin(), value.GetMax()])
        if not np.isfinite(result).all() or (result[1] < result[0]).any():
            raise ValueError('Invalid actual collider bound: '+str(prim.GetPath()))
        return result

    def shapes(root):
        return [prim for prim in Usd.PrimRange(stage.GetPrimAtPath(root), Usd.TraverseInstanceProxies())
                if prim.HasAPI(UsdPhysics.CollisionAPI) and prim.IsA(UsdGeom.Boundable)
                and UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get() is not False]

    robot_shapes = shapes('/World/Robot')
    robot_names = [str(prim.GetPath()) for prim in robot_shapes]
    obstacle_shapes = [*shapes('/World/Environment'), *shapes('/World/Task')]
    obstacle_names = [str(prim.GetPath()) for prim in obstacle_shapes]
    if not robot_shapes or not obstacle_shapes:
        raise ValueError('Missing original robot or environment/task collision shapes')

    def override(prim, matrix):
        parent = transforms.GetLocalToWorldTransform(prim.GetParent())
        xform = UsdGeom.Xformable(prim)
        xform.ClearXformOpOrder()
        xform.AddTransformOp(UsdGeom.XformOp.PrecisionDouble, 'workspaceProbe').Set(
            Gf.Matrix4d(matrix.T.tolist())*parent.GetInverse())
        transforms.Clear()
        np.testing.assert_allclose(np.asarray(transforms.GetLocalToWorldTransform(prim)).T,
                                   matrix, atol=1.e-7, rtol=0.)

    def place_robot(joints):
        joints = model.checked_joints(joints, enforce_limits=True)
        poses = model.arm.forward(joints, model.arm_base)
        poses.update(model.hand.forward(joints, poses[model.arm_tip] @ model.wrist_offset))
        targets = [(prim, poses[prim.GetName()]) for prim in Usd.PrimRange(stage.GetPrimAtPath('/World/Robot'))
                   if prim.IsA(UsdGeom.Xform) and prim.GetName() in poses]
        for prim, matrix in sorted(targets, key=lambda row: str(row[0].GetPath()).count('/')):
            override(prim, matrix)
        cache.Clear()
        return np.asarray([bound(prim) for prim in robot_shapes])

    def task_path(path):
        return '/World/Task'+path.removeprefix(metadata['root_path'])

    def place_objects(poses):
        for row in metadata['objects']:
            pose = poses[row['id']]
            matrix = pose_matrix(pose)
            override(stage.GetPrimAtPath(task_path(row['body_path'])), matrix)
        for path, matrix in composite_volume_poses(metadata, poses).items():
            override(stage.GetPrimAtPath(task_path(path)), matrix)
        cache.Clear()

    def object_colliders(obj):
        paths = physical_collider_paths(metadata, obj)
        result = [stage.GetPrimAtPath(task_path(path)) for path in paths]
        if any(not prim or not prim.HasAPI(UsdPhysics.CollisionAPI)
               or UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get() is False for prim in result):
            raise ValueError('Workspace object requires every declared rigid/volume physical collider')
        return result

    def clearance_check(boxes, obstacles, *, refine):
        broad = nearest(boxes, obstacles, robot_names, obstacle_names)
        if not refine or broad['clearance_over_5mm']:
            return {**broad, 'method': 'per_collider_aabb'}
        from scripts.diagnostics.collider_clearance import certify, usd_geometry
        delta = np.maximum(np.maximum(boxes[:, None, 0]-obstacles[None, :, 1],
                                      obstacles[None, :, 0]-boxes[:, None, 1]), 0.)
        gaps = np.linalg.norm(delta, axis=2)
        robot_geometry, obstacle_geometry, refined = {}, {}, []
        for i, j in np.argwhere(gaps <= MINIMUM_COLLIDER_GAP_M):
            if i not in robot_geometry:
                robot_geometry[i] = usd_geometry(robot_shapes[i], transforms)
            if j not in obstacle_geometry:
                obstacle_geometry[j] = usd_geometry(obstacle_shapes[j], transforms)
            check = certify(robot_geometry[i], obstacle_geometry[j], gap_m=MINIMUM_COLLIDER_GAP_M)
            refined.append({'robot_shape': robot_names[i], 'obstacle_shape': obstacle_names[j], **check})
            if not check['passed']:
                break
        return {**broad, 'method': 'aabb_then_authored_convex_hull_triangle_separation',
                'clearance_over_5mm': all(row['passed'] for row in refined),
                'broadphase_candidate_count': int((gaps <= MINIMUM_COLLIDER_GAP_M).sum()),
                'refinements': refined,
                'scope': 'Conservative authored surface separation, not native cooked SDF distance.'}

    rows = []
    selected = set(object_ids) if object_ids else set(metadata['actors'])
    if not object_ids:
        for parent, composite in metadata.get('elastic_composites', {}).items():
            selected -= set(composite['member_ids'])-{parent}
    if not selected <= set(metadata['actors']):
        raise ValueError('Unknown object selection')
    original = {row['id']: row['initial_pose_xyzw'] for row in metadata['objects']}
    goal = dict(metadata['goals']['body_poses_xyzw'])
    authored_bounds = np.asarray([bound(prim) for prim in obstacle_shapes])
    place_objects(original)
    np.testing.assert_allclose(np.asarray([bound(prim) for prim in obstacle_shapes]),
                               authored_bounds, atol=1.e-7, rtol=0.,
                               err_msg='Reset metadata/xyzw overrides differ from actual authored USD')
    original.update(probe['pose_overrides'].get('source', {}))
    goal.update(probe['pose_overrides'].get('goal', {}))
    object_samples = []
    for state, poses in (('source', original), ('goal', goal)):
        place_objects(poses)
        obstacles = np.asarray([bound(prim) for prim in obstacle_shapes])
        for obj in metadata['objects']:
            if obj['id'] not in selected:
                continue
            colliders = object_colliders(obj)
            feature = [p for p in colliders if any(name in p.GetName().lower() for name in ('handle', 'knob', 'grip'))]
            named = probe['named_features'].get(obj['id']+':'+state)
            if named is not None:
                feature = [p for p in feature if p.GetName() == named]
                if len(feature) != 1:
                    raise ValueError('Named workspace feature must identify one actual handle/knob/grip collider')
            # One independently accessible handle is sufficient for this one-arm
            # workspace check. Report all selected feature provenance explicitly.
            if feature:
                feature = [min(feature, key=lambda p: np.linalg.norm(bound(p).mean(0)[:2]-model.arm_base[:2, 3]))]
                box = bound(feature[0])
                source = str(feature[0].GetPath())
            else:
                boxes = np.asarray([bound(p) for p in colliders])
                box = np.asarray([boxes[:, 0].min(0), boxes[:, 1].max(0)])
                source = task_path(obj['body_path'])+' (union of individual collider bounds)'
                if obj['id'] in metadata.get('composite_objects', {}):
                    source = task_path(obj['body_path'])+' (union of actual rigid and volume physical collider bounds)'
            point = np.r_[box.mean(0)[:2], box[1, 2]]
            if obj['role'] in ('spoon', 'fork', 'knife'):
                # These utensils have one physical mesh, not a separately
                # named handle collider. Read its actual positive-local-Y
                # handle section rather than probing the bowl/tine centre.
                from scripts.diagnostics.collider_clearance import usd_geometry
                world_vertices = np.concatenate([usd_geometry(p, transforms)['vertices'] for p in colliders])
                body = np.asarray(transforms.GetLocalToWorldTransform(stage.GetPrimAtPath(task_path(obj['body_path'])))).T
                local = (world_vertices-body[:3, 3]) @ body[:3, :3]
                y_min, y_max = local[:, 1].min(), local[:, 1].max()
                y_target = y_min+.84*(y_max-y_min)
                region = local[np.abs(local[:, 1]-y_target) <= .08*(y_max-y_min)]
                if not len(region):
                    raise ValueError('Actual utensil handle section is empty')
                grip = np.r_[region[:, :2].mean(0), region[:, 2].max()]
                point = body[:3, :3] @ grip+body[:3, 3]
                source = task_path(obj['body_path'])+' (actual positive-local-Y handle section)'
            elif any(name in obj['role'] for name in ('plate', 'bowl')):
                from scripts.diagnostics.collider_clearance import usd_geometry
                vertices = np.concatenate([usd_geometry(p, transforms)['vertices'] for p in colliders])
                rim = vertices[vertices[:, 2] >= vertices[:, 2].max()-.001]
                index = np.argmin(np.linalg.norm(rim[:, :2]-model.arm_base[:2, 3], axis=1))
                point = rim[index]
                source = task_path(obj['body_path'])+' (actual top rim nearest robot base)'
            for phase, clearance in (('near_object', NEAR_CLEARANCE_M), ('approach_or_lift', APPROACH_CLEARANCE_M)):
                object_samples.append({'object_id': obj['id'], 'state': state, 'phase': phase,
                    'feature': source, 'feature_bounds_m': box.tolist(), 'point_m': point.tolist(),
                    'fingertip_clearance_m': clearance, 'obstacles': obstacles, 'poses': poses})
    if task['id'] == 'surface_wiping' and 'sponge' in selected:
        place_objects(original)
        obstacles = np.asarray([bound(prim) for prim in obstacle_shapes])
        sponge = metadata['objects'][0]
        if 'sponge' in metadata.get('composite_objects', {}):
            from scripts.diagnostics.collider_clearance import usd_geometry
            vertices = np.concatenate([usd_geometry(prim, transforms)['vertices'] for prim in object_colliders(sponge)])
            body = pose_matrix(original['sponge'])
            local = (vertices-body[:3, 3]) @ body[:3, :3]
            bounds = np.asarray([local.min(0), local.max(0)])
            np.testing.assert_allclose(bounds, metadata['composite_objects']['sponge']['rest_local_bounds_m'],
                atol=1.e-7, rtol=0., err_msg='Declared sponge envelope differs from actual rigid/volume rest geometry')
            height = float(bounds[1, 2])
        else:
            height = float(np.asarray(sponge['collision_bounds_m'])[1, 2])
        for patch in metadata['goals']['soil_patches']:
            points = np.asarray([cell['center_world_m'] for cell in metadata['goals']['soil_cells'] if cell['patch'] == patch['id']])
            for name, xy in [('center', points[:, :2].mean(0)),
                             ('min_x', points[np.argmin(points[:, 0]), :2]),
                             ('max_x', points[np.argmax(points[:, 0]), :2]),
                             ('min_y', points[np.argmin(points[:, 1]), :2]),
                             ('max_y', points[np.argmax(points[:, 1]), :2])]:
                point = [*xy.tolist(), metadata['goals']['wiping_surface_z_m']+height]
                object_samples.append({'object_id': 'sponge', 'state': 'wiping', 'phase': patch['id']+'/'+name,
                    'feature': 'nominal sponge top above soil-cell position; sponge is not attached or simulated',
                    'point_m': point, 'fingertip_clearance_m': NEAR_CLEARANCE_M,
                    'obstacles': obstacles, 'poses': original})
    if sample_keys is not None:
        wanted = set(sample_keys)
        keys = {':'.join(row[key] for key in ('object_id', 'state', 'phase')) for row in object_samples}
        if not wanted or not wanted <= keys:
            raise ValueError('Unknown or empty workspace sample selection: '+str(sorted(wanted-keys)))
        object_samples = [row for row in object_samples
                          if ':'.join(row[key] for key in ('object_id', 'state', 'phase')) in wanted]
    arm_seed = {name: ready[name] for name in model.arm.names}
    for sample in object_samples:
        obstacles, poses = sample.pop('obstacles'), sample.pop('poses')
        place_objects(poses)
        attempts, accepted = [], None
        for orientation, rotation in probe_rotations(sample['point_m'], model.arm_base[:3, 3], profile=orientation_profile):
            wrist = wrist_for_tip(sample['point_m'], tip_local, rotation, sample['fingertip_clearance_m'])
            target = np.linalg.inv(model.arm_base) @ wrist @ np.linalg.inv(model.wrist_offset)
            solved = model.arm.inverse(target, model.arm_tip, arm_seed, iterations=180,
                multi_starts=probe['ik_multistarts'], position_tolerance=.0001, rotation_tolerance=.002)
            joints = {**ready, **solved['joint_positions_rad']}
            clearance = None if ik_only else clearance_check(place_robot(joints), obstacles, refine=solved['converged'])
            actual = model.arm.forward(joints, model.arm_base)[model.arm_tip] @ model.wrist_offset
            actual_tip = (actual @ np.r_[tip_local, 1.])[:3]
            expected_tip = np.asarray(sample['point_m'])+[0., 0., sample['fingertip_clearance_m']]
            attempt = {'orientation': orientation, 'solver': solved, 'clearance': clearance,
                'wrist_target_matrix': wrist.tolist(), 'actual_middle_fingertip_m': actual_tip.tolist(),
                'fingertip_error_m': float(np.linalg.norm(actual_tip-expected_tip)),
                'minimum_arm_joint_margin_rad': float(min(min(joints[j.name]-j.lower, j.upper-joints[j.name]) for j in model.arm.active))}
            attempts.append(attempt)
            if solved['converged'] and (ik_only or clearance['clearance_over_5mm']):
                accepted = len(attempts)-1
                break
        rows.append({**sample, 'passed': accepted is not None, 'ik_solved': any(a['solver']['converged'] for a in attempts),
                     'accepted_attempt': accepted, 'attempts': attempts})
    ready_boxes = place_robot(ready)
    place_objects(original)
    result = {'schema': 'kitchen_workspace_endpoint_audit.v1', 'task_id': task['id'], 'release': str(release),
        'passed': all(row['passed'] for row in rows), 'check_mode': 'ik_only' if ik_only else 'ik_and_collision_bounds',
        'sample_count': len(rows), 'ik_solved_count': sum(row['ik_solved'] for row in rows),
        'passed_sample_count': sum(row['passed'] for row in rows), 'rows': rows,
        'input_sha256': bindings, 'original_middle_fingertip_local_m': tip_local.tolist(),
        'robot_collision_shape_count': len(robot_shapes), 'obstacle_collision_shape_count': len(obstacle_shapes),
        'volume_collision_shape_count': sum(prim.IsA(UsdGeom.TetMesh) for prim in obstacle_shapes),
        'composite_object_ids': sorted(metadata.get('composite_objects', {})),
        'ready_clearance': nearest(ready_boxes, np.asarray([bound(p) for p in obstacle_shapes]), robot_names, obstacle_names),
        'parameters': {'near_tip_clearance_m': NEAR_CLEARANCE_M, 'approach_and_lift_tip_clearance_m': APPROACH_CLEARANCE_M,
            'minimum_shape_gap_m': MINIMUM_COLLIDER_GAP_M, 'ik_position_tolerance_m': .0001,
            'ik_rotation_tolerance_rad': .002, 'ik_iterations': 180, 'ik_multistarts': probe['ik_multistarts'],
            'orientation_profile': orientation_profile,
            'orientation_trials_max': 48 if orientation_profile == 'expanded_v1' else 12,
            'every_attempt_seed': ('original_frozen_ready_arm' if probe['ik_multistarts'] == 0 else
                                   'original_frozen_ready_arm_as_primary_seed_plus_declared_solver_multistarts')},
        'selected_sample_keys': sorted(sample_keys) if sample_keys is not None else None,
        'anonymous_usd_candidate': candidate,
        'uses_published_object_poses': not bool(probe['pose_overrides']),
        'reset_pose_override_matches_authored_usd': True,
        'physics_steps': 0, 'controller_changes': False, 'grasp_qualified': False, 'swept_path_qualified': False,
        'exclusions': ['Own robot stand/mount and robot self-collision are not checked.',
            'No held object or payload sweep; near-object access uses the unchanged waiting finger posture.',
            'Source and all-goal arrangements are independent probes, not an ordered manipulation sequence.'],
        'scope': ('Discrete original URDF IK only; no endpoint collision claim. ' if ik_only else
                  'Discrete original URDF IK and conservative per-collider actual-USD clearance to fixed room, task fixtures, context and all task objects. ')+
                 'Failed finite search is not proof of global unreachability; passing is not grasp, path or native task qualification.'}
    for path, expected in bindings.items():
        if digest(path) != expected:
            raise ValueError('Workspace input changed during audit: '+path)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--release', type=Path, action='append', required=True)
    parser.add_argument('--object', dest='object_ids', action='append')
    parser.add_argument('--ik-only', action='store_true', help='Report kinematic feasibility separately; no endpoint collision acceptance')
    parser.add_argument('--orientation-profile', choices=('original_v1', 'expanded_v1'), default='original_v1',
                        help='Explicit finite workspace search; expanded_v1 retains the original 12 attempts first')
    parser.add_argument('--sample', dest='sample_keys', action='append',
                        help='Optional exact object_id:state:phase selection; never changes sample geometry or tolerances')
    parser.add_argument('--candidate', type=Path,
                        help='Explicit anonymous-USD layout/real named-feature/IK multistart experiment; published files stay immutable')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Preserve earlier evidence: '+str(args.output))
    import bench.control.factory
    from scripts.maintenance.freeze_project_sources import verify_if_snapshot
    source = verify_if_snapshot(PROJECT, required_modules=('bench.control.factory',), entrypoint_file=Path(__file__))
    candidate = load_json(args.candidate) if args.candidate else None
    candidate_sha256 = digest(args.candidate) if args.candidate else None
    results = []
    for path in args.release:
        result = run(path, object_ids=args.object_ids, ik_only=args.ik_only,
                     orientation_profile=args.orientation_profile, sample_keys=args.sample_keys, candidate=candidate)
        results.append(result)
        print(json.dumps({'task_id': result['task_id'], 'ik_solved_count': result['ik_solved_count'],
                          'passed_sample_count': result['passed_sample_count'], 'sample_count': result['sample_count']}), flush=True)
    report = {'schema': 'kitchen_workspace_endpoint_collection.v1', 'passed': all(row['passed'] for row in results),
        'source_snapshot': source, 'source_git_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=PROJECT, text=True).strip() if source is None else None,
        'script_sha256': digest(__file__), 'results': results,
        'candidate_file': str(args.candidate.resolve()) if args.candidate else None,
        'candidate_sha256': candidate_sha256}
    if args.candidate and digest(args.candidate) != candidate_sha256:
        raise ValueError('Workspace candidate changed during audit')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps({'passed': report['passed'], 'tasks': [{key: row[key] for key in
        ('task_id', 'passed_sample_count', 'sample_count')} for row in results]}), flush=True)
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
