"""Read-only discrete geometry preflight for observed ordinary pressing.

Uses the original USD environment/arm meshes and verified native cooked shapes.
No simulator, attachment, state setter or collision-filter access is performed.
Intentional cloth compression is distinguished from all other sampled postures;
only the latter can be checked against the currently observed static cloth.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
import hashlib
from itertools import combinations

import numpy as np
from scipy.optimize import linprog
from scipy.spatial.transform import Rotation

from bench.evaluation.cloth.cloth_robot_collision import (
    SCHEMA as EXPORT_SCHEMA, SUPPORT_SCHEMA, apply_transform, audit_cloth_robot,
    cooked_piece_geometry, digest_json, planes_in_body,
)
from scripts.diagnostics.check_bimanual_arm_self_collision import (
    ArmSelfCollision, intersection_radius as original_intersection_radius,
)
from scripts.diagnostics.check_bimanual_options import CollisionBounds, gaps
from scenes.workrooms.laundry_room.robot_configurations import stand_geometry

SCHEMA = 'laundry_observed_press_original_geometry_preflight.v1'
DESIGN_PAIRS = frozenset(frozenset((side+'_thumb_MC', side+'_hand_C_MC'))
                         for side in ('left', 'right'))
CHECKED_GEOMETRY = ('environment', 'arm_self', 'dual_arm', 'whole_cooked_hand')


def _array_hash(value, dtype):
    return hashlib.sha256(np.asarray(value, dtype=dtype).tobytes()).hexdigest()


def convex_overlap(first, second):
    """Original convex LP, with explicit separation or same-geometry fallback.

    A numerical solver exception never becomes an implicit zero intersection.
    The fallback uses the same unit-normal outer planes in translated coordinates.
    """
    planes = np.vstack((first['planes'], second['planes']))
    try:
        radius, point = original_intersection_radius(planes)
        return radius, point, None
    except RuntimeError as error:
        receipt = {'original_error': str(error)}
    for a, b, direction in ((first, second, 'first_outside_second_face'),
                             (second, first, 'second_outside_first_face')):
        separation = (a['vertices'] @ b['planes'][:, :3].T + b['planes'][:, 3]).min(0)
        face = int(separation.argmax())
        if separation[face] > 1.e-9:
            receipt.update(resolution='original_face_separation_certificate', direction=direction,
                           face_index=face, world_plane=b['planes'][face].tolist(),
                           minimum_all_vertex_separation_m=float(separation[face]))
            return 0., None, receipt
    origin = (first['bounds'].mean(0) + second['bounds'].mean(0)) * .5
    shifted = planes.copy()
    shifted[:, 3] += shifted[:, :3] @ origin
    result = linprog([0., 0., 0., -1.], A_ub=np.c_[shifted[:, :3], np.ones(len(shifted))],
                     b_ub=-shifted[:, 3], bounds=[(None, None)] * 3 + [(0., None)],
                     method='highs-ipm', options={'primal_feasibility_tolerance': 1.e-9,
                                                 'dual_feasibility_tolerance': 1.e-9})
    receipt.update(resolution='same_planes_translated_interior_point_lp',
                   translated_origin_m=origin.tolist(), status=int(result.status), message=result.message)
    if result.status == 2:
        return 0., None, receipt
    if not result.success:
        raise RuntimeError('Unresolved original cooked convex intersection: '+str(receipt))
    radius = float(result.x[3])
    point = result.x[:3] + origin
    residual = float(np.max(planes[:, :3] @ point + planes[:, 3] + radius))
    if not np.isfinite([radius, residual, *point]).all() or radius < -1.e-9 or residual > 2.e-9:
        raise RuntimeError('Translated convex solution failed original world-plane residual')
    receipt.update(radius_m=radius, world_primal_max_residual_m=residual)
    return radius, point.tolist(), receipt


def design_contact_bounds(references):
    """Only the two original MC/palm pairs, from independent initial states."""
    if set(references) != {'recorded_native_start', 'original_release_initial'}:
        raise ValueError('Both original design-contact references are required')
    bounds = {}
    for state in references.values():
        for hit in state['intersections']:
            pair = frozenset(hit['bodies'])
            radius = float(hit['radius_m'])
            if pair not in DESIGN_PAIRS or not np.isfinite(radius) or not 0. <= radius < .001:
                raise ValueError('Original reference has a non-design or >=1mm intersection')
            bounds[pair] = max(bounds.get(pair, 0.), radius)
    return bounds


def bounded_design_contacts(hits, bounds):
    return all(frozenset(h['bodies']) in bounds and np.isfinite(h['radius_m'])
               and 0. <= h['radius_m'] < .001
               and h['radius_m'] <= bounds[frozenset(h['bodies'])] + 1.e-9 for h in hits)


class _PressPreflight:
    def __init__(self, scene, initial, models, export, cloth_q, triangles, actual_joints, configuration):
        self.models = models
        if set(models) != {'left', 'right'} or configuration not in ('dual_opposed', 'dual_same_side'):
            raise ValueError('Original dual SHARPA/FR3 installation required')
        self.scene, self.initial, self.export = deepcopy(scene), deepcopy(initial), deepcopy(export)
        self.q = np.array(cloth_q, dtype=float, copy=True)
        self.triangles = np.array(triangles, copy=True)
        if (self.q.ndim != 2 or self.q.shape[1] != 3 or not len(self.q)
                or not np.isfinite(self.q).all() or self.triangles.ndim != 2
                or self.triangles.shape[1] != 3 or not len(self.triangles)
                or not np.issubdtype(self.triangles.dtype, np.integer)
                or self.triangles.min() < 0 or self.triangles.max() >= len(self.q)):
            raise ValueError('Finite observed cloth and original integer triangles required')
        self.names = set(n for model in models.values() for n in model.names)
        self.actual = self._joints(actual_joints)
        park = self._joints(initial['joint_positions_rad'])
        self.checker = CollisionBounds(scene, scene['robot'], models, park, stand_geometry(configuration))
        self.arm = ArmSelfCollision(self.checker)
        rows = self.export.get('colliders', [])
        if (self.export.get('schema') != EXPORT_SCHEMA or self.export.get('status') != 'complete'
                or self.export.get('issues') or self.export.get('visual_mesh_fallback') is not False
                or not isinstance(self.export.get('state_hash_before'), str)
                or len(self.export['state_hash_before']) != 64
                or self.export['state_hash_before'] != self.export.get('state_hash_after')
                or not self.export.get('layer_hashes_before')
                or self.export['layer_hashes_before'] != self.export.get('layer_hashes_after')):
            raise ValueError('Complete unchanged-state original native cooked export required')
        source_counts = Counter(body for shapes in self.checker.robot_shapes.values() for body, _ in shapes)
        if (Counter(row.get('body_name') for row in rows) != source_counts
                or len({row.get('path') for row in rows}) != len(rows)):
            raise ValueError('Cooked inventory differs from the original robot collision shapes')
        self.body_names = list(dict.fromkeys(row['body_name'] for row in rows))
        self.local = {side: [] for side in models}
        self.adjacent = set()
        for side, model in models.items():
            self.adjacent.update(frozenset((j.parent, j.child)) for j in model.arm.joints+model.hand.joints)
            self.adjacent.add(frozenset((model.arm_tip, model.hand.root)))
            poses = self.checker.poses(side, park)
            for row in rows:
                if row['body_name'] not in poses:
                    continue
                if row.get('status') != 'verified_cooking_representation' or not row.get('pieces'):
                    raise ValueError('Incomplete cooked collision representation')
                for index, piece in enumerate(row['pieces']):
                    if piece.get('geometry_representation') != SUPPORT_SCHEMA:
                        raise ValueError('Original raw-plane/support-hull cooked bounds required')
                    # Rebuild from original SDK vertices/planes; do not trust a supplied outer cache.
                    geometry = cooked_piece_geometry(piece)
                    matrix = np.asarray(row['mesh_to_body_column_matrix'], float)
                    self.local[side].append({'body': row['body_name'], 'path': row['path'], 'piece': index,
                        'hand': row['body_name'] in model.hand.forward(park),
                        'vertices': apply_transform(geometry['conservative_outer_vertices'], matrix),
                        'planes': planes_in_body(geometry['conservative_outer_planes'], matrix)})
        self.references = {}
        for name, joints in (('recorded_native_start', self.actual), ('original_release_initial', park)):
            world, _ = self._world(joints)
            hits, _, fallbacks = self._internal(world)
            self.references[name] = {'joint_positions_rad': joints, 'intersections': hits,
                                     'numeric_solver_fallbacks': fallbacks}
        self.bounds = design_contact_bounds(self.references)
        self.binding = {'scene_sha256': digest_json(scene), 'initial_sha256': digest_json(initial),
                        'cooked_export_sha256': digest_json(export),
                        'actual_joints_sha256': digest_json(self.actual),
                        'cloth_q_float64_sha256': _array_hash(self.q, '<f8'),
                        'triangles_int64_sha256': _array_hash(self.triangles, '<i8'),
                        'configuration': configuration, 'source_asset_files': deepcopy(self.checker.dependencies)}

    def _joints(self, values):
        if set(values) != self.names or not all(np.isfinite(v) for v in values.values()):
            raise ValueError('Finite complete original robot joint inventory required')
        result = {n: float(v) for n, v in values.items()}
        for model in self.models.values():
            model.checked_joints({n: result[n] for n in model.names}, enforce_limits=True)
            if any(model.hand_posture_prior.violation(f, result)
                   for f in ('thumb', 'index', 'middle', 'ring', 'little')):
                raise ValueError('Original hand posture prior violated')
        return result

    def _world(self, joints):
        world, poses = {}, {}
        for side in self.models:
            own = self.checker.poses(side, joints)
            poses.update(own)
            world[side] = []
            for piece in self.local[side]:
                vertices = apply_transform(piece['vertices'], own[piece['body']])
                world[side].append({**piece, 'vertices': vertices,
                    'planes': planes_in_body(piece['planes'], own[piece['body']]),
                    'bounds': np.asarray([vertices.min(0), vertices.max(0)])})
        return world, poses

    def _internal(self, world):
        hits, fallbacks, tested = [], [], 0
        for side, pieces in world.items():
            for first, second in combinations(pieces, 2):
                if (not (first['hand'] or second['hand']) or first['body'] == second['body']
                        or frozenset((first['body'], second['body'])) in self.adjacent):
                    continue
                if not np.all(np.minimum(first['bounds'][1], second['bounds'][1])
                              - np.maximum(first['bounds'][0], second['bounds'][0]) > 1.e-8):
                    continue
                tested += 1
                radius, point, fallback = convex_overlap(first, second)
                if fallback is not None:
                    fallbacks.append({'side': side, 'bodies': [first['body'], second['body']], **fallback})
                if radius > 1.e-7:
                    hits.append({'bodies': [first['body'], second['body']], 'radius_m': radius,
                                 'point_m': point, 'pieces': [first['piece'], second['piece']]})
        return hits, tested, fallbacks

    def _environment(self, joints, world):
        boxes = {side: self.checker.boxes(side, joints) for side in self.models}
        raw = self.checker.check(boxes)
        other = {key: value for key, value in raw['checks'].items() if not key.endswith('_environment')}
        tables = [i for i, name in enumerate(self.checker.obstacle_names)
                  if name.endswith('/Geometry/worktop/Collider')]
        if len(tables) != 1:
            raise ValueError('One original laundry worktop collider required')
        top = float(self.checker.obstacles[tables[0]][1, 2])
        rows = []
        for side in self.models:
            minimum = min(float(p['vertices'][:, 2].min()) for p in world[side] if p['hand'])
            d = gaps(boxes[side], self.checker.obstacles)
            warnings = []
            for i, j in np.argwhere(d <= .001):
                body = self.checker.robot_shapes[side][i][0]
                hand = any(p['body'] == body and p['hand'] for p in world[side])
                table = int(j) in tables
                if hand and table:
                    continue  # Whole cooked hand floor checked below, never exempted outright.
                warning = {'body': body, 'obstacle': self.checker.obstacle_names[j],
                           'coarse_aabb_gap_m': float(d[i, j]), 'passed': False}
                if table:
                    parts = [p for p in world[side] if p['body'] == body]
                    if parts:
                        gap = min(float(p['vertices'][:, 2].min()) for p in parts) - top
                        warning.update(cooked_vertical_clearance_m=gap, passed=gap > .001)
                warnings.append(warning)
            rows.append({'side': side, 'whole_cooked_hand_min_z_m': minimum,
                         'whole_cooked_hand_table_gap_m': minimum-top,
                         'coarse_warnings': warnings,
                         'passed': minimum >= top and all(r['passed'] for r in warnings)})
        return {'passed': all(r['passed'] for r in rows), 'sides': rows,
                'original_bounds_checks': raw}, other

    def __call__(self, samples):
        if not isinstance(samples, list) or not samples:
            raise ValueError('Complete nonempty press samples required')
        before = digest_json(samples)
        samples = deepcopy(samples)
        rows, static_poses, static_indices = [], [], []
        last_time = -float('inf')
        for index, sample in enumerate(samples):
            if (type(sample.get('intentional_cloth_compression_phase')) is not bool
                    or not isinstance(sample.get('label'), str)
                    or not np.isfinite(sample['time_s']) or sample['time_s'] < last_time):
                raise ValueError('Explicit contact labels and consecutive finite sample times required')
            last_time = sample['time_s']
            joints = self._joints(sample['joint_positions_rad'])
            world, poses = self._world(joints)
            environment, other = self._environment(joints, world)
            arm = self.arm.check(joints)
            hits, tested, fallbacks = self._internal(world)
            internal_passed = bounded_design_contacts(hits, self.bounds)
            static = not sample['intentional_cloth_compression_phase']
            if static:
                static_indices.append(index)
                static_poses.append([np.r_[poses[name][:3, 3], Rotation.from_matrix(poses[name][:3, :3]).as_quat()]
                                     for name in self.body_names])
            rows.append({'sample_index': index, 'time_s': sample['time_s'], 'label': sample['label'],
                         'environment': environment, 'arm_self': arm, 'other_source_checks': other,
                         'cooked_internal': {'passed': internal_passed, 'intersections': hits,
                             'broadphase_candidates': tested, 'numeric_solver_fallbacks': fallbacks},
                         'static_observed_cloth': {'checked': static,
                             'reason': 'intentional_compression_or_contact_withdrawal' if not static else None},
                         'passed': environment['passed'] and arm['clear'] and internal_passed
                                   and all(v['clear'] for v in other.values())})
        static_review = None
        if static_indices:
            static_review = audit_cloth_robot(self.export,
                {'particle_q': np.broadcast_to(self.q, (len(static_indices), *self.q.shape)),
                 'body_q': np.asarray(static_poses)}, self.triangles, self.body_names, tolerance_m=.0001)
            bad = {r['frame'] for key in ('violations', 'representation_boundary_uncertainties')
                   for r in static_review[key]}
            verified = static_review['export_verified'] and not static_review['unresolved_colliders']
            for frame, index in enumerate(static_indices):
                clear = bool(verified and frame not in bad)
                rows[index]['static_observed_cloth'].update(passed=clear, static_audit_frame=frame)
                rows[index]['passed'] = bool(rows[index]['passed'] and clear)
        return {'schema': SCHEMA, 'all_clear': all(row['passed'] for row in rows),
                'sample_count': len(rows), 'samples': rows, 'checked_geometry': list(CHECKED_GEOMETRY),
                'input_binding': self.binding, 'samples_sha256': before,
                'design_contact_reference_states': deepcopy(self.references),
                'design_contact_bounds': [{'bodies': sorted(pair), 'maximum_radius_m': bound}
                                          for pair, bound in self.bounds.items()],
                'excluded_direct_urdf_pairs': [sorted(pair) for pair in sorted(self.adjacent, key=lambda p: sorted(p))],
                'static_cloth_sample_indices': static_indices, 'static_cloth_review': static_review,
                'scope': 'Discrete original-source environment/stands/dual arms/nonadjacent FR3 plus all '
                         'same-side cooked hand/hand and hand/own-arm pairs. Only original direct URDF '
                         'adjacency and wrist mount are excluded. Two sub-mm original MC/palm design '
                         'intersections may not grow beyond independently checked actual/initial states. '
                         'Every non-contact posture checks every observed cloth triangle against every '
                         'cooked robot shape. Intentional compression/initial withdrawal have no static '
                         'cloth exclusion claim; subsequent actual cloth/servo motion needs native audit. '
                         'No CCD, contact force, material response or task success certificate.'}


def make_press_preflight(scene, initial, models, export, cloth_q, triangles, *, actual_joints,
                         configuration='dual_opposed'):
    """Bind fresh original geometry/state; return the indexed whole-path checker."""
    return _PressPreflight(scene, initial, models, export, cloth_q, triangles, actual_joints, configuration)
