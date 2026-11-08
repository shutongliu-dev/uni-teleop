"""Read-only volume boundary clearance against actual cooked robot convexes."""
from copy import deepcopy

import numpy as np

from bench.convex_geometry import convex_triangle_audit
from bench.evaluation.cloth.cloth_robot_collision import (
    SCHEMA, SUPPORT_SCHEMA, apply_transform, planes_in_body, transform_xyzw,
)


def _contains_vertex(points, tetrahedra, vertices):
    """Detect convex enclosure by the FEM solid, including inverted tets."""
    tets = points[tetrahedra]
    lo, hi = tets.min(1), tets.max(1)
    for vertex in vertices:
        candidates = tets[np.all(vertex >= lo - 1.e-9, axis=1) & np.all(vertex <= hi + 1.e-9, axis=1)]
        if not len(candidates):
            continue
        matrix = np.stack([candidates[:, i] - candidates[:, 0] for i in (1, 2, 3)], axis=-1)
        if np.any(np.abs(np.linalg.det(matrix)) < 1.e-18):
            return True  # Singular solid geometry cannot certify separation.
        bary = np.linalg.solve(matrix, np.broadcast_to(vertex, candidates[:, 0].shape)[..., None]
                               - candidates[:, 0, :, None])[..., 0]
        if np.any(np.all(bary >= -1.e-8, axis=1) & (bary.sum(1) <= 1 + 1.e-8)):
            return True
    return False


class VolumeRobotGeometry:
    """All native robot bodies are covered; no hand-side or task-name assumptions."""
    def __init__(self, export, body_names, volume_bodies, clearance_m=.005):
        if (export.get('schema') != SCHEMA or export.get('status') != 'complete'
                or export.get('issues') or export.get('visual_mesh_fallback') is not False
                or not export.get('state_hash_before')
                or export['state_hash_before'] != export.get('state_hash_after')
                or not export.get('layer_hashes_before')
                or export['layer_hashes_before'] != export.get('layer_hashes_after')):
            raise ValueError('Complete unchanged native cooked robot collision export required')
        self.body_names = list(body_names)
        self.info = deepcopy(volume_bodies)
        self.clearance = float(clearance_m)
        if not self.info or not np.isfinite(self.clearance) or self.clearance <= 0:
            raise ValueError('Explicit volume topology and positive clearance required')
        self.pieces = []
        for row in export['colliders']:
            if row.get('status') != 'verified_cooking_representation' or row['body_name'] not in self.body_names:
                raise ValueError('Unverified native robot collider')
            matrix = row['mesh_to_body_column_matrix']
            if not row['pieces']:
                raise ValueError('Empty native cooked collider')
            for piece in row['pieces']:
                if piece.get('geometry_representation') != SUPPORT_SCHEMA:
                    raise ValueError('Cooked support/raw-plane uncertainty bounds required')
                vertices = apply_transform(piece['conservative_outer_vertices'], matrix)
                planes = planes_in_body(piece['conservative_outer_planes'], matrix)
                if not np.isfinite(vertices).all() or not np.isfinite(planes).all():
                    raise ValueError('Nonfinite cooked geometry')
                self.pieces.append({'body': self.body_names.index(row['body_name']), 'path': row['path'],
                    'vertices': vertices, 'planes': planes,
                    'bounds': np.stack([vertices.min(0), vertices.max(0)])})
        if not self.pieces:
            raise ValueError('No verified native robot convexes')

    def check(self, volumes, body_q):
        poses = np.asarray(body_q, dtype=float)
        if (poses.shape != (len(self.body_names), 7) or not np.isfinite(poses).all()
                or not np.allclose(np.linalg.norm(poses[:, 3:], axis=1), 1., atol=1.e-5, rtol=0)):
            raise ValueError('Complete native robot poses required')
        if set(volumes) != set(self.info):
            raise ValueError('Native volume inventory changed')
        points = {name: np.asarray(row['positions_m'], float) for name, row in volumes.items()}
        for name, value in points.items():
            if value.shape != (self.info[name]['node_count'], 3) or not np.isfinite(value).all():
                raise ValueError('Incomplete or nonfinite native volume geometry')
        local, near = {}, []
        for row in self.pieces:
            body = row['body']
            for name, world_points in points.items():
                key = (name, body)
                if key not in local:
                    local[key] = apply_transform(world_points, np.linalg.inv(transform_xyzw(poses[body])))
                value = local[key]
                lo, hi = row['bounds']
                # Expanded AABB encloses the entire clearance band.
                margin = self.clearance + 1.e-7
                if np.any(value.max(0) < lo-margin) or np.any(value.min(0) > hi+margin):
                    continue
                expanded = row['planes'].copy(); expanded[:, 3] -= margin
                audit = convex_triangle_audit(value, np.asarray(self.info[name]['surface_triangles']),
                    expanded, np.stack([lo-margin, hi+margin]), 1.e-8)
                boundary_near = not audit['all_triangles_certified_clear_within_tolerance']
                enclosed = False if boundary_near else _contains_vertex(
                    value, np.asarray(self.info[name]['tetrahedra']), row['vertices'])
                if boundary_near or enclosed:
                    near.append({'volume_body': name, 'robot_collider': row['path'],
                                 'surface_within_clearance': boundary_near, 'solid_enclosure': enclosed})
        return {'schema': 'native_volume_robot_geometry.v1', 'complete': True,
                'method': 'native_cooked_convex_geometry', 'clearance_m': self.clearance,
                'fully_separated': not near, 'contact_force_measured': False,
                'convex_pieces_checked': len(self.pieces), 'near_pairs': near}


class RigidCompositeRobotGeometry:
    """Geometric union of real cooked rigid pieces, with native per-body poses.

    A convex solid is partitioned into geometric tetrahedra only for enclosure
    tests. These are not simulated FEM elements or inferred material states.
    """
    def __init__(self, robot_export, robot_body_names, component_export, component_body_names, clearance_m=.005):
        from scipy.spatial import ConvexHull
        if (component_export.get('schema') != SCHEMA
                or component_export.get('status') != 'complete'
                or component_export.get('issues')
                or component_export.get('visual_mesh_fallback') is not False
                or not component_export.get('state_hash_before')
                or component_export.get('state_hash_before') != component_export.get('state_hash_after')
                or not component_export.get('layer_hashes_before')
                or component_export.get('layer_hashes_before') != component_export.get('layer_hashes_after')):
            raise ValueError('Unchanged native cooked component geometry required')
        expected = set(component_body_names)
        requested = component_export.get('requested_body_paths', {})
        if (not expected or len(expected) != len(component_body_names)
                or expected != set(requested)
                or expected != {r.get('body_name') for r in component_export['colliders']}):
            raise ValueError('Complete declared component collider inventory required')
        self.local = {}
        info = {}
        paths = [r['path'] for r in component_export['colliders']]
        if len(set(paths)) != len(paths):
            raise ValueError('Distinct native component collider paths required')
        for row in component_export['colliders']:
            if row.get('status') != 'verified_cooking_representation':
                raise ValueError('Unverified native component collider')
            if not row['pieces'] or row.get('body_path') != requested[row['body_name']]:
                raise ValueError('Missing or mismatched native component collider')
            for index, piece in enumerate(row['pieces']):
                if piece.get('geometry_representation') != SUPPORT_SCHEMA:
                    raise ValueError('Native component conservative support hull required')
                vertices = apply_transform(piece['conservative_outer_vertices'],
                                           row['mesh_to_body_column_matrix'])
                hull = ConvexHull(vertices)
                points = np.vstack((vertices, vertices.mean(axis=0)))
                triangles = hull.simplices
                tets = np.column_stack((np.full(len(triangles), len(vertices)), triangles))
                name = row['path']+':'+str(index)
                self.local[name] = {'body': row['body_name'], 'points': points}
                info[name] = {'node_count': len(points), 'surface_triangles': triangles.tolist(),
                              'tetrahedra': tets.tolist()}
        if not self.local:
            raise ValueError('Empty component geometry cannot certify release')
        self.audit = VolumeRobotGeometry(robot_export, robot_body_names, info, clearance_m)

    def check(self, objects, body_q):
        poses = np.asarray(body_q, dtype=float)
        if (poses.shape != (len(self.audit.body_names), 7) or not np.isfinite(poses).all()
                or not np.allclose(np.linalg.norm(poses[:, 3:], axis=1), 1., atol=1.e-5, rtol=0)):
            raise ValueError('Complete native robot poses required')
        pieces = {}
        for name, row in self.local.items():
            state = objects[row['body']]
            p, q = state['position_m'], state['quaternion_wxyz']
            transform = transform_xyzw([*p, *q[1:], q[0]])
            pieces[name] = {'positions_m': apply_transform(row['points'], transform)}
        # Broadphase uses the exact original expanded robot-local AABB rotated
        # into a conservative world AABB. Expanding only world bounds by 5 mm
        # would under-cover rotated corners and could change release decisions.
        transforms = np.array([transform_xyzw(pose) for pose in poses])
        inverses = np.linalg.inv(transforms)
        names = list(pieces)
        values = [pieces[name]['positions_m'] for name in names]
        lows = np.array([p.min(0) for p in values])
        highs = np.array([p.max(0) for p in values])
        margin = self.audit.clearance + 1.e-7
        local, near = {}, []
        for row in self.audit.pieces:
            lo, hi = row['bounds']
            center, half = (lo+hi)/2, (hi-lo)/2+margin
            transform = transforms[row['body']]
            world_center = transform[:3, :3]@center+transform[:3, 3]
            world_half = np.abs(transform[:3, :3])@half
            candidates = np.flatnonzero(np.all(highs >= world_center-world_half-1e-12, axis=1)
                                       & np.all(lows <= world_center+world_half+1e-12, axis=1))
            for index in candidates:
                name = names[index]
                key = (name, row['body'])
                if key not in local:
                    local[key] = apply_transform(values[index], inverses[row['body']])
                value = local[key]
                if np.any(value.max(0) < lo-margin) or np.any(value.min(0) > hi+margin):
                    continue
                expanded = row['planes'].copy()
                expanded[:, 3] -= margin
                info = self.audit.info[name]
                audit = convex_triangle_audit(value, np.asarray(info['surface_triangles']),
                    expanded, np.stack([lo-margin, hi+margin]), 1.e-8)
                boundary_near = not audit['all_triangles_certified_clear_within_tolerance']
                enclosed = False if boundary_near else _contains_vertex(
                    value, np.asarray(info['tetrahedra']), row['vertices'])
                if boundary_near or enclosed:
                    near.append({'volume_body': name, 'robot_collider': row['path'],
                                 'surface_within_clearance': boundary_near, 'solid_enclosure': enclosed})
        return {'schema': 'native_rigid_composite_robot_geometry.v1', 'complete': True,
                'method': 'native_cooked_rigid_composite_geometry', 'clearance_m': self.audit.clearance,
                'fully_separated': not near, 'contact_force_measured': False,
                'convex_pieces_checked': len(self.audit.pieces), 'near_pairs': near,
                'component_convex_pieces_checked': len(self.local)}
