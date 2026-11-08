"""Read-only native cloth and cooked-robot geometry for ordinary joint episodes.

Geometric hand proximity is explicitly not a measured contact force. Every
native step is checked; no cloth target, attachment or simulation write is used.
"""
from copy import deepcopy

import numpy as np

from bench.convex_geometry import convex_triangle_audit
from bench.evaluation.cloth.cloth_robot_collision import (
    SCHEMA, SUPPORT_SCHEMA, apply_transform, collect_colliders, planes_in_body,
    transform_xyzw,
)


def wrist_frame_bindings(env, active_sides):
    """Bind pure hand frames to native parent bodies, never USD world poses.

    Scene composition already verifies these local Xforms against the pinned
    robot source. Cache their fixed local transforms; each sample below uses
    the same PhysX body_q array as the cooked collision audit.
    """
    from pxr import UsdGeom, UsdPhysics
    frames = env.scene_info.get('robot_frame_paths', {})
    links = env.scene_info.get('robot_link_paths', {})
    if set(frames) & set(links):
        raise ValueError('Nonphysical hand frame aliases a native rigid body')
    parents = {path: name for name, path in links.items()}
    if len(parents) != len(links):
        raise ValueError('Native robot body paths are ambiguous')
    result = {}
    for side in active_sides:
        name = side+'_hand_flange'
        if name in env.body_names:
            if name in frames:
                raise ValueError('Hand frame is both physical and nonphysical')
            continue  # Historical rigid flange keeps its exact native pose.
        if name not in frames:
            raise ValueError('Missing declared native hand frame: '+name)
        prim = env.stage.GetPrimAtPath(frames[name])
        if (not prim or not prim.IsActive() or not prim.IsDefined() or not prim.IsLoaded()
                or not prim.IsA(UsdGeom.Xform) or prim.GetName() != name
                or any(prim.HasAPI(api) for api in
                       (UsdPhysics.RigidBodyAPI, UsdPhysics.MassAPI, UsdPhysics.CollisionAPI))):
            raise ValueError('Hand frame must be a declared pure Xform: '+name)
        parent = prim.GetParent()
        body = parents.get(str(parent.GetPath()))
        xform = UsdGeom.Xformable(prim)
        if (body not in env.body_names or not parent.HasAPI(UsdPhysics.RigidBodyAPI)
                or xform.GetResetXformStack()):
            raise ValueError('Hand frame lacks its native rigid parent: '+name)
        result[side] = {'body_name': body,
                        'body_from_frame': np.asarray(xform.GetLocalTransformation()).T.copy()}
    return result


class CookedRobotGeometry:
    """Cache verified convexes once, transform the actual native body poses."""
    def __init__(self, export, body_names, topology, thickness, clearance_m=.005,
                 penetration_tolerances_m=(.002, .01), active_sides=('left', 'right'),
                 wrist_frames=None):
        if (export.get('schema') != SCHEMA or export.get('status') != 'complete'
                or export.get('issues') or export.get('visual_mesh_fallback') is not False
                or export.get('state_hash_before') != export.get('state_hash_after')
                or not export.get('layer_hashes_before')
                or export['layer_hashes_before'] != export.get('layer_hashes_after')):
            raise ValueError('Complete unchanged native cooked robot collision export required')
        self.body_names, self.topology = list(body_names), topology
        self.thickness, self.clearance = thickness, float(clearance_m)
        self.penetration_tolerances = tuple(float(value) for value in penetration_tolerances_m)
        if any(not np.isfinite(value) or value <= 0 for value in self.penetration_tolerances):
            raise ValueError('Positive finite robot penetration tolerances required')
        self.pieces = []
        if tuple(active_sides) not in (('right',), ('left', 'right')):
            raise ValueError('Explicit official single-right or left/right hands required')
        if tuple(active_sides) == ('right',) and any(name.startswith('left_') for name in self.body_names):
            raise ValueError('Single-arm evidence cannot hide a second robot')
        self.wrist_frames = deepcopy(wrist_frames or {})
        if not set(self.wrist_frames) <= set(active_sides):
            raise ValueError('Wrist frame binding names an inactive hand')
        self.wrists = {}
        for side in active_sides:
            frame = side+'_hand_flange'
            binding = self.wrist_frames.get(side)
            if binding is None:
                self.wrists[side] = self.body_names.index(frame)
                continue
            matrix = np.asarray(binding['body_from_frame'], dtype=float)
            if (frame in self.body_names or matrix.shape != (4, 4) or not np.isfinite(matrix).all()
                    or not np.allclose(matrix[3], [0., 0., 0., 1.], rtol=0., atol=1.e-12)
                    or not np.allclose(matrix[:3, :3].T @ matrix[:3, :3], np.eye(3), rtol=0., atol=1.e-12)
                    or not np.isclose(np.linalg.det(matrix[:3, :3]), 1., rtol=0., atol=1.e-12)):
                raise ValueError('Pure wrist frame requires an unambiguous rigid local transform')
            self.wrists[side] = self.body_names.index(binding['body_name'])
            binding['body_from_frame'] = matrix.copy()
        for row in export['colliders']:
            if row.get('status') != 'verified_cooking_representation' or row['body_name'] not in self.body_names:
                raise ValueError('Unverified native robot collider')
            name, matrix = row['body_name'], row['mesh_to_body_column_matrix']
            side = next((side for side in active_sides if name.startswith(side+'_')), None)
            for piece in row['pieces']:
                if piece.get('geometry_representation') != SUPPORT_SCHEMA:
                    raise ValueError('Cooked support/raw-plane uncertainty bounds required')
                vertices = apply_transform(piece['conservative_outer_vertices'], matrix)
                self.pieces.append({'body': self.body_names.index(name), 'hand': side,
                    'path': row['path'], 'planes': planes_in_body(piece['conservative_outer_planes'], matrix),
                    'inner': planes_in_body(piece['conservative_inner_planes'], matrix),
                    'bounds': np.stack([vertices.min(0), vertices.max(0)])})
        if not self.pieces or not set(active_sides) <= {row['hand'] for row in self.pieces}:
            raise ValueError('Both original hands must have verified native colliders' if len(active_sides) == 2
                             else 'The original right hand must have verified native colliders')

    def check(self, vertices, body_q):
        poses = np.asarray(body_q, dtype=float)
        if poses.shape != (len(self.body_names), 7) or not np.isfinite(poses).all():
            raise ValueError('Complete native robot poses required')
        hands = {side: {'wrist_body_name': self.body_names[index],
            'wrist_position_m': poses[index, :3].tolist(),
            'near_towel': False, 'near_any_cloth': False} for side, index in self.wrists.items()}
        for side, binding in self.wrist_frames.items():
            world_from_frame = transform_xyzw(poses[self.wrists[side]]) @ binding['body_from_frame']
            hands[side].update(wrist_frame_name=side+'_hand_flange',
                               wrist_position_m=world_from_frame[:3, 3].tolist())
        any_near = False
        penetration = 0.
        potential_penetration = False
        uncertain = False
        local = {}
        for row in self.pieces:
            body = row['body']
            inverse = None
            for name, points in vertices.items():
                key = (name, body)
                if key not in local:
                    inverse = np.linalg.inv(transform_xyzw(poses[body])) if inverse is None else inverse
                    local[key] = apply_transform(points, inverse)
                points = local[key]
                margin = self.clearance+self.thickness[name]/2
                lo, hi = row['bounds']
                if np.any(points.max(0) < lo-margin) or np.any(points.min(0) > hi+margin):
                    continue
                expanded = row['planes'].copy()
                expanded[:, 3] -= margin
                near = convex_triangle_audit(points, self.topology[name], expanded,
                    np.stack([lo-margin, hi+margin]), 1.e-7)
                if near['all_triangles_certified_clear_within_tolerance']:
                    continue
                any_near = True
                if row['hand']:
                    hands[row['hand']]['near_any_cloth'] = True
                    hands[row['hand']]['near_towel'] |= name == 'towel'
                # A separate inward test establishes real midsurface penetration;
                # the 5 mm near band never becomes a reported penetration.
                for tolerance in self.penetration_tolerances:
                    # Check both the release and hard-stop erosion separately:
                    # a shallow witness must not hide a smaller deep region.
                    outer = convex_triangle_audit(points, self.topology[name], row['planes'], row['bounds'], tolerance)
                    if not outer['all_triangles_certified_clear_within_tolerance']:
                        inner = convex_triangle_audit(points, self.topology[name], row['inner'], row['bounds'], tolerance)
                        potential_penetration = True
                        uncertain |= bool(set(outer['triangles_penetrating_tolerance']) -
                                          set(inner['triangles_penetrating_tolerance']))
                        penetration = max(penetration, inner['maximum_confirmed_piece_penetration_m'])
        return {'schema': 'laundry_robot_interaction.v1', 'complete': True,
            'method': 'native_cooked_convex_geometry', 'contact_force_measured': False,
            'clearance_m': self.clearance, 'hand_near_cloth': any(row['near_towel'] for row in hands.values()),
            'fully_separated': not any_near, 'hands': hands,
            'robot_cloth_penetration_m': float(penetration),
            'robot_cloth_penetration_checked': True,
            'robot_cloth_geometry_uncertain': bool(uncertain),
            'robot_cloth_near_band_penetration_test_triggered': potential_penetration}


class NativeObserver:
    def __init__(self, env, scene, task):
        from pxr import UsdPhysics, PhysxSchema
        from bench.evaluation.cloth.gpu_intersections import ClothIntersectionAudit
        self.env, self.scene, self.task = env, deepcopy(scene), deepcopy(task)
        if (task.get('evaluation', {}).get('control_mode') != 'robot_joint_targets'
                or scene.get('named_cloth_initialization') != 'author-only'
                or float(env.dt) != task['evaluation']['physics_dt_s']):
            raise ValueError('Laundry robot runtime requires frozen joint mode and authored initialization')
        self.specs = {row['id']: row for row in task['asset_metadata']['cloths']}
        self.paths = {name: row['path'] for name, row in env.scene_info['cloths'].items()}
        if set(self.specs) != set(self.paths) or set(self.specs) != set(env.cloth_names):
            raise ValueError('Native laundry cloth inventory differs from frozen task')
        self.topology = {name: np.asarray(row['triangles'], dtype=np.int32) for name, row in self.specs.items()}
        if any(not np.array_equal(self.topology[name], env.cloth_triangles[name]) for name in self.specs):
            raise ValueError('Native cloth topology differs from frozen task')
        self.thickness = {name: row['physical_thickness_m'] for name, row in self.specs.items()}
        self.measure_crossings = 'crossing_rest_area_fraction_max' in task['evaluation']['thresholds']
        self.oracle = ClothIntersectionAudit(self.topology,
            {name: len(row['rest_vertices_m']) for name, row in self.specs.items()},
            thickness_by_id=self.thickness, device=env.device,
            **({'intersection_pair_capacity': 1024} if self.measure_crossings else {}))
        self.export = collect_colliders(env)
        limits = task['evaluation']['thresholds']
        active_sides = task.get('active_sides', ['left', 'right'])
        self.robot = CookedRobotGeometry(self.export, env.body_names, self.topology, self.thickness,
            penetration_tolerances_m=(limits['max_penetration_m'], limits['hard_penetration_m']),
            active_sides=active_sides, wrist_frames=wrist_frame_bindings(env, active_sides))
        # World owns the PhysicsScene path; the pinned Isaac runtime defaults
        # to /physicsScene, while authored task stages may use another path.
        self.physics_prim = env.stage.GetPrimAtPath(env.world.get_physics_context().prim_path)
        if not self.physics_prim or not self.physics_prim.IsA(UsdPhysics.Scene):
            raise ValueError('Native World physics context does not identify a PhysicsScene')
        self.physics = UsdPhysics.Scene(self.physics_prim)
        self.physx = PhysxSchema.PhysxSceneAPI(self.physics_prim)
        self.cloths = {name: env.stage.GetPrimAtPath(path) for name, path in self.paths.items()}
        self.materials = {name: env.stage.GetPrimAtPath(env.scene_info['cloths'][name]['material_path']) for name in self.paths}
        self.constraint_paths = self._constraint_inventory()
        self.receipt = deepcopy(getattr(env, 'named_cloth_initialization_receipt', None))
        if (not self.receipt or self.receipt.get('mode') != 'author-only'
                or self.receipt.get('position_write_calls') != 0 or self.receipt.get('velocity_write_calls') != 0):
            raise ValueError('Missing native author-only initialization receipt')
        self.native_baseline_step = int(env.world.current_time_step_index)
        self.native_baseline_time = float(env.world.current_time)
        self.previous_step_index = -1
        self.previous_native_time = self.native_baseline_time
        self.static_evidence = self.read_physics()

    def _constraint_inventory(self):
        from pxr import UsdPhysics
        result = {}
        for prim in self.env.stage.Traverse():
            if 'Attachment' in prim.GetTypeName():
                raise ValueError('Ordinary laundry joint episodes do not allow cloth attachments')
            if prim.IsA(UsdPhysics.Joint):
                path = str(prim.GetPath())
                targets = [str(p) for key in ('physics:body0', 'physics:body1')
                           for p in prim.GetRelationship(key).GetTargets()]
                if (not path.startswith('/World/Robot/') or
                        any(not p.startswith('/World/Robot/') for p in targets)):
                    raise ValueError('Unexpected constraint outside the declared robot articulation')
                result[path] = prim
        return result

    def read_physics(self):
        gravity = np.asarray(self.physics.GetGravityDirectionAttr().Get(), dtype=float)*float(self.physics.GetGravityMagnitudeAttr().Get())
        if (not np.allclose(gravity, self.scene['physics']['gravity_m_s2'], atol=1.e-6, rtol=0)
                or gravity[2] >= 0 or self.physx.GetEnableGPUDynamicsAttr().Get() is not True
                or str(self.physx.GetBroadphaseTypeAttr().Get()) != 'GPU'):
            raise ValueError('Native cloth gravity/GPU dynamics differs from frozen physics')
        rows = {}
        for name, prim in self.cloths.items():
            attrs = {key: prim.GetAttribute(key).Get() for key in (
                'omniphysics:deformableBodyEnabled', 'omniphysics:kinematicEnabled',
                'physxDeformableBody:disableGravity', 'physxDeformableBody:selfCollision')}
            if (attrs['omniphysics:deformableBodyEnabled'] is not True
                    or attrs['omniphysics:kinematicEnabled'] is not False
                    or attrs['physxDeformableBody:disableGravity'] is not False
                    or attrs['physxDeformableBody:selfCollision'] is not True):
                raise ValueError('Cloth dynamic/gravity/self-collision configuration changed')
            thickness = self.materials[name].GetAttribute('omniphysics:surfaceThickness').Get()
            if thickness is None or abs(float(thickness)-self.thickness[name]) > 1.e-8:
                raise ValueError('Native material thickness differs from frozen cloth')
            rows[name] = {'attributes': attrs, 'surface_thickness_m': float(thickness),
                         'applied_schemas': list(prim.GetAppliedSchemas())}
        return {'gravity_m_s2': gravity.tolist(), 'gpu_dynamics': True, 'cloths': rows,
            'constraint_paths': sorted(self.constraint_paths), 'hidden_cloth_constraints': 0}

    def frame(self, state, *, step_index, time_s):
        native_time = float(state['native_world_time_s'])
        # Isaac's native callback accumulates a float32 step size. Compare
        # adjacent native ticks, as the contact exchange and replay do, so
        # representational drift never masquerades as a missing sample.
        if (step_index != self.previous_step_index+1
                or int(state['native_physics_step_index']) != self.native_baseline_step+step_index
                or not np.isfinite(native_time)
                or abs(native_time-self.previous_native_time-(self.env.dt if step_index else 0.)) > 1.e-8
                or abs(time_s-step_index*self.env.dt) > 1.e-8):
            raise ValueError('Missing, repeated or mixed-tick native cloth sample')
        self.previous_step_index = step_index
        self.previous_native_time = native_time
        if self.read_physics() != self.static_evidence:
            raise ValueError('Native cloth physical configuration changed')
        vertices = {name: np.asarray(state['cloth_'+name+'_q']) for name in self.specs}
        velocities = {name: np.asarray(state['cloth_'+name+'_qd']) for name in self.specs}
        if any(not np.isfinite(value).all() for value in (*vertices.values(), *velocities.values())):
            raise ValueError('Nonfinite native cloth state')
        geometry = self.oracle.check(vertices)
        crossing = None
        if self.measure_crossings:
            from scenes.workrooms.laundry_room.garment_crossings import crossing_measure
            crossing = crossing_measure(vertices['towel'], self.topology['towel'],
                self.specs['towel']['rest_vertices_m'], geometry.get('self_intersection_pairs', []),
                pair_count=geometry['self_intersections'],
                complete=geometry.get('intersection_pair_inventory_complete', False))
            geometry['self_crossing_measure'] = crossing
        if geometry.get('finite_thickness_compression_checked') is not True:
            raise ValueError('Missing finite-thickness cloth audit')
        interaction = self.robot.check(vertices, np.asarray(state['body_q'])[:len(self.env.body_names)])
        table = self.scene['table']
        center, half = np.asarray(table['position_m']), np.asarray(table['half_extents_m'])
        table_depth = 0.
        for name, points in vertices.items():
            inside = np.all(np.abs(points[:, :2]-center[:2]) <= half[:2], axis=1)
            if inside.any():
                table_depth = max(table_depth, float(np.max(center[2]+half[2]+self.thickness[name]/2-points[inside, 2])))
        penetration = max(0., table_depth, geometry['max_cloth_envelope_penetration_m'], interaction['robot_cloth_penetration_m'])
        geometry_known = not interaction['robot_cloth_geometry_uncertain']
        valid = bool(not geometry['degenerate_triangles'] and geometry_known)
        return {'schema': 'laundry_native_frame.v1', 'complete': True,
            'native_step': step_index, 'time_s': float(time_s), 'control_mode': 'robot_joint_targets',
            'controlled_vertices': {name: [] for name in self.specs},
            'robot_interaction': interaction,
            'cloths': {name: {'enabled': True, 'vertices_m': vertices[name],
                'velocities_m_s': velocities[name], 'triangles': self.topology[name]} for name in self.specs},
            'physics': {'complete': True, 'valid': valid, 'validity_scope': 'native_geometry',
                'gravity_enabled': True, 'hidden_constraints': 0,
                'self_intersections': geometry['self_intersections'],
                'intercloth_intersections': geometry['intercloth_intersections'],
                'max_penetration_m': penetration,
                **({'self_crossing_measure': crossing} if self.measure_crossings else {})},
            'geometry_diagnostics': {**geometry, 'table_envelope_penetration_m': max(0., table_depth)},
            'native_physics_step_index': int(state['native_physics_step_index']),
            'native_world_time_s': float(state['native_world_time_s'])}

    def close(self):
        if set(self._constraint_inventory()) != set(self.constraint_paths) or self.read_physics() != self.static_evidence:
            raise ValueError('Native cloth/constraint configuration changed during episode')
        return {'initialization': self.receipt, 'native_physics': self.static_evidence,
            'cloth_position_write_calls': 0, 'cloth_velocity_write_calls': 0,
            'cloth_attachments': [], 'contact_force_measured': False,
            'scope': 'Every native sample, actual cloth geometry and cooked robot shapes; no robot grasp qualification.'}
