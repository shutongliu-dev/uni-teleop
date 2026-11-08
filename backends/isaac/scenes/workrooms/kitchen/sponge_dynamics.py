"""Read-only geometry of a bonded volume sponge, separate from wiping credit.

The native solver owns every node. These calculations never infer an impulse
from overlap, replace a native state, or count undeformed graphics as contact.
The existing kitchen stability window and distances also apply to foam nodes.
"""
from collections import deque

import numpy as np


# Gross numerical failure bounds, not a material calibration or task tolerance.
# Inversion alone misses a positive but effectively collapsed tetrahedron.
NUMERICAL_LIMITS = {'minimum_tetrahedron_volume_ratio': .001,
                    'minimum_edge_length_ratio': .1, 'maximum_edge_length_ratio': 3.}


def tetrahedral_geometry(rest_points, tetrahedra):
    rest = np.asarray(rest_points, dtype=float)
    tets = np.asarray(tetrahedra, dtype=np.int64)
    if (rest.ndim != 2 or rest.shape[1] != 3 or len(rest) < 4
            or not np.isfinite(rest).all() or tets.ndim != 2
            or tets.shape[1] != 4 or len(tets) == 0
            or tets.min() < 0 or tets.max() >= len(rest)):
        raise ValueError('Finite volume vertices and valid tetrahedra required')
    matrix = (rest[tets[:, 1:]] - rest[tets[:, :1]]).transpose(0, 2, 1)
    determinants = np.linalg.det(matrix)
    if np.any(np.abs(determinants) <= 1.e-14):
        raise ValueError('Rest tetrahedron has zero or unresolved volume')
    edges = np.unique(np.sort(np.concatenate(
        [tets[:, [a, b]] for a in range(4) for b in range(a+1, 4)]), axis=1), axis=0)
    lengths = np.linalg.norm(rest[edges[:, 1]] - rest[edges[:, 0]], axis=1)
    return rest, tets, determinants, edges, lengths


class VolumeGeometry:
    """Full-node diagnostics; deformation is measured against the rest solid."""

    def __init__(self, rest_points, tetrahedra):
        self.rest, self.tets, self.det, self.edges, self.lengths = tetrahedral_geometry(
            rest_points, tetrahedra)

    def sample(self, points, velocities):
        points, velocities = np.asarray(points, float), np.asarray(velocities, float)
        if (points.shape != self.rest.shape or velocities.shape != points.shape
                or not np.isfinite(points).all() or not np.isfinite(velocities).all()):
            raise ValueError('Complete finite native foam positions and velocities required')
        matrices = (points[self.tets[:, 1:]] - points[self.tets[:, :1]]).transpose(0, 2, 1)
        ratios = np.linalg.det(matrices) / self.det
        stretch = np.linalg.norm(points[self.edges[:, 1]]-points[self.edges[:, 0]], axis=1) / self.lengths
        return {'node_count': len(points), 'tetrahedron_count': len(self.tets),
                'inverted_tetrahedron_count': int(np.count_nonzero(ratios <= 0.)),
                'minimum_tetrahedron_volume_ratio': float(ratios.min()),
                'maximum_tetrahedron_volume_ratio': float(ratios.max()),
                'total_volume_ratio': float(np.sum(np.abs(self.det)*ratios) / np.abs(self.det).sum()),
                'minimum_edge_length_ratio': float(stretch.min()),
                'maximum_edge_length_ratio': float(stretch.max()),
                'maximum_native_node_speed_m_s': float(np.linalg.norm(velocities, axis=1).max()),
                'bounds_world_m': [points.min(0).tolist(), points.max(0).tolist()]}


class NodeStability:
    """Consecutive native samples with the existing 20 ms/4 mm scene limits."""

    def __init__(self, limits, physics_dt_s):
        self.limits, self.dt = limits, physics_dt_s
        count = round(limits['stability_pose_window_s'] / physics_dt_s)
        if count < 1 or abs(count*physics_dt_s-limits['stability_pose_window_s']) > 1.e-12:
            raise ValueError('Foam stability window must contain whole native steps')
        self.history = deque(maxlen=count+1)
        self.anchor = None
        self.quiet_s = 0.

    def update(self, points, *, fully_separated, baseline=False):
        points = np.array(points, dtype=float, copy=True)
        if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
            raise ValueError('Finite complete foam nodes required for stability')
        if self.history and points.shape != self.history[0].shape:
            raise ValueError('Native foam topology changed within a stability window')
        self.history.append(points)
        speed = None
        if len(self.history) == self.history.maxlen:
            speed = float(np.linalg.norm(points-self.history[0], axis=1).max()
                          / self.limits['stability_pose_window_s'])
        quiet = bool(not baseline and fully_separated and speed is not None
                     and speed <= self.limits['maximum_linear_speed_m_s'])
        drift = None
        if quiet:
            if self.anchor is None:
                self.anchor = points.copy()
            drift = float(np.linalg.norm(points-self.anchor, axis=1).max())
            quiet = drift <= self.limits['maximum_stable_position_range_m']
        if quiet:
            self.quiet_s += self.dt
        else:
            self.quiet_s, self.anchor = 0., None
        return {'quiet': quiet, 'released_quiet_native_s': self.quiet_s,
                'maximum_rolling_node_speed_m_s': speed,
                'maximum_node_drift_from_quiet_anchor_m': drift,
                'fully_separated': bool(fully_separated)}


class BondedSpongeEvidence:
    """Validate the foam and its real material connection to the wiping layer."""

    def __init__(self, metadata, limits, physics_dt_s):
        self.specs = metadata['volume_bodies']
        if set(self.specs) != {'foam'} or self.specs['foam']['parent_rigid_body'] != 'sponge':
            raise ValueError('Wiping requires one declared foam bonded to its sponge layer')
        self.limits = limits
        self.geometry, self.stability = {}, {}
        for name, row in self.specs.items():
            self.geometry[name] = VolumeGeometry(row['rest_points_m'], row['tetrahedra'])
            attachment = row['attachment']
            ids = np.asarray(attachment['node_indices'], dtype=int)
            anchors = np.asarray(attachment['rigid_local_positions_m'], dtype=float)
            if (ids.ndim != 1 or len(ids) < 3 or len(np.unique(ids)) != len(ids)
                    or ids.min() < 0 or ids.max() >= len(row['rest_points_m'])
                    or anchors.shape != (len(ids), 3) or not np.isfinite(anchors).all()):
                raise ValueError('Explicit finite layer-bond nodes and rigid anchors required')
            self.stability[name] = NodeStability(limits, physics_dt_s)

    def sample(self, frame, objects):
        from .native_evidence import rotation
        volumes = frame.get('volume_bodies')
        robot = frame.get('volume_robot_geometry')
        if not isinstance(volumes, dict) and not hasattr(volumes, 'keys'):
            raise ValueError('Missing native volume evidence')
        if set(volumes) != set(self.specs):
            raise ValueError('Native foam inventory differs from the frozen asset')
        if (not robot or robot.get('complete') is not True
                or robot.get('method') != 'native_cooked_convex_geometry'
                or robot.get('clearance_m') != .005
                or type(robot.get('fully_separated')) is not bool):
            raise ValueError('Complete cooked-robot foam release evidence required')
        points_by_actor, diagnostics = {}, {}
        for name, spec in self.specs.items():
            row = volumes[name]
            points = np.asarray(row['positions_m'], dtype=float)
            diagnostics[name] = self.geometry[name].sample(points, row['velocities_m_s'])
            parent = spec['parent_rigid_body']
            obj = objects[parent]
            r = np.asarray(rotation(obj['quaternion_wxyz']))
            bond = spec['attachment']
            anchors = (np.asarray(bond['rigid_local_positions_m']) @ r.T
                       + np.asarray(obj['position_m']))
            error = np.linalg.norm(points[np.asarray(bond['node_indices'])]-anchors, axis=1)
            diagnostics[name]['maximum_material_bond_error_m'] = float(error.max())
            diagnostics[name].update(self.stability[name].update(points,
                fully_separated=robot['fully_separated'], baseline=frame['baseline']))
            points_by_actor.setdefault(parent, []).extend(points.tolist())
        valid = all(row['minimum_tetrahedron_volume_ratio'] >= NUMERICAL_LIMITS['minimum_tetrahedron_volume_ratio']
                    and row['minimum_edge_length_ratio'] >= NUMERICAL_LIMITS['minimum_edge_length_ratio']
                    and row['maximum_edge_length_ratio'] <= NUMERICAL_LIMITS['maximum_edge_length_ratio']
                    and row['maximum_material_bond_error_m'] <= self.limits['maximum_penetration_m']
                    for row in diagnostics.values())
        return points_by_actor, {'schema': 'bonded_sponge_native_geometry.v1',
            'complete': True, 'geometry_valid': valid,
            'fully_separated': robot['fully_separated'],
            'quiet': all(row['quiet'] for row in diagnostics.values()),
            'volumes': diagnostics, 'robot_geometry': dict(robot),
            'numerical_geometry_limits': dict(NUMERICAL_LIMITS),
            'contact_force_measured_on_foam': False}
