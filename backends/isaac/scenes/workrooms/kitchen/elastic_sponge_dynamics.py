"""Read-only evidence for a passive discrete elastic sponge.

Collision vertices follow actual native cell transforms. The center-grid
tetrahedra below measure gross folding/collapse only; they are not FEM elements
and do not supply forces, mass or a continuum material calibration.
"""
from itertools import permutations, product

import numpy as np

from .native_evidence import rotation, world_points
from .sponge_dynamics import NodeStability, VolumeGeometry, NUMERICAL_LIMITS


class ElasticSpongeEvidence:
    def __init__(self, metadata, limits, physics_dt_s):
        self.composites = metadata['elastic_composites']
        if set(self.composites) != {'sponge'}:
            raise ValueError('One complete elastic sponge required')
        self.spec = self.composites['sponge']
        rows = metadata['objects']
        self.objects = rows if isinstance(rows, dict) else {r['id']: r for r in rows}
        members = self.spec['member_ids']
        if (self.spec['parent_rigid_body'] != 'sponge' or len(members) != 28
                or len(set(members)) != len(members) or not set(members) <= self.objects.keys()):
            raise ValueError('All 28 native sponge parts required')
        self.cells = self.spec['cells']
        if (len(self.cells) != 27 or set(r['id'] for r in self.cells) != set(members)-{'sponge'}
                or len(self.spec['links']) != 63):
            raise ValueError('Complete three-dimensional material topology required')
        grid = {tuple(r['grid_index']): i for i, r in enumerate(self.cells)}
        if set(grid) != set(product(range(3), repeat=3)):
            raise ValueError('Unique 3 by 3 by 3 material grid required')
        tets = []
        for origin in product(range(2), repeat=3):
            for order in permutations(range(3)):
                point = np.array(origin)
                tet = [grid[tuple(point)]]
                for axis in order:
                    point[axis] += 1
                    tet.append(grid[tuple(point)])
                tets.append(tet)
        centers = [r['composite_rest_center_m'] for r in self.cells]
        self.center_geometry = VolumeGeometry(centers, tets)
        self.stability = NodeStability(limits, physics_dt_s)
        self.links = self.spec['links']
        self.joint_names = set(metadata['joints'])
        if len(self.joint_names) != len(self.links):
            raise ValueError('Complete native elastic joint inventory required')
        self.rest_lengths = []
        for row in self.links:
            if row['body0'] not in members or row['body1'] not in members:
                raise ValueError('Material link leaves the declared sponge')
            p = [self.objects[row[k]]['composite_rest_center_m'] for k in ('body0', 'body1')]
            self.rest_lengths.append(np.linalg.norm(np.asarray(p[1])-p[0]))
        if not np.isfinite(self.rest_lengths).all() or min(self.rest_lengths) <= 0:
            raise ValueError('Finite nonzero material rest lengths required')

    def sample(self, frame, objects):
        robot = frame.get('elastic_robot_geometry')
        if (not robot or robot.get('complete') is not True
                or robot.get('method') != 'native_cooked_rigid_composite_geometry'
                or robot.get('clearance_m') != .005
                or type(robot.get('fully_separated')) is not bool):
            raise ValueError('Complete cooked-robot elastic release evidence required')
        members = self.spec['member_ids']
        if set(frame.get('joints', {})) != self.joint_names:
            raise ValueError('Missing native elastic joint evidence')
        for link in self.links:
            name = link['prim_path'].rsplit('/', 1)[-1]
            row = frame['joints'][name]
            if (row.get('kind') != 'elastic_d6' or row.get('body0') != link['body0']
                    or row.get('body1') != link['body1'] or row.get('force_measured') is not False):
                raise ValueError('Native elastic joint identity differs')
            delta = np.asarray(row.get('anchor_delta_world_m'), float)
            local = np.asarray(row.get('anchor_delta_joint0_m'), float)
            rotation_matrix = np.asarray(row.get('relative_rotation_matrix'), float)
            gap = row.get('anchor_separation_m')
            if (delta.shape != (3,) or local.shape != (3,) or rotation_matrix.shape != (3, 3)
                    or not np.isfinite(delta).all() or not np.isfinite(local).all()
                    or not np.isfinite(rotation_matrix).all() or type(gap) not in (int, float)
                    or not np.isfinite(gap) or abs(gap-np.linalg.norm(delta)) > 1.e-8
                    or abs(np.linalg.norm(local)-gap) > 1.e-8
                    or not np.allclose(rotation_matrix.T@rotation_matrix, np.eye(3), atol=1.e-6, rtol=0)
                    or abs(np.linalg.det(rotation_matrix)-1) > 1.e-6):
                raise ValueError('Complete finite elastic joint geometry required')
        if not set(members) <= objects.keys():
            raise ValueError('Missing native elastic part')
        points, centers = [], {}
        for name in members:
            row, obj = self.objects[name], objects[name]
            if not row.get('collision_support_points_m'):
                raise ValueError('Full physical cell boundary required')
            points.extend(world_points(obj, row['local_bounds_m'], row['collision_support_points_m']))
            local = (np.asarray(row['composite_rest_center_m'])
                     - row.get('composite_rest_origin_m', row['composite_rest_center_m']))
            centers[name] = np.asarray(obj['position_m']) + np.asarray(rotation(obj['quaternion_wxyz']))@local
        points = np.asarray(points)
        ids = [r['id'] for r in self.cells]
        central = self.center_geometry.sample([centers[n] for n in ids],
                                             np.zeros((len(ids), 3)))
        stretches = np.array([np.linalg.norm(centers[r['body1']]-centers[r['body0']])
                              for r in self.links])/self.rest_lengths
        geometry = {
            'minimum_center_grid_orientation_ratio': central['minimum_tetrahedron_volume_ratio'],
            'inverted_center_grid_simplex_count': central['inverted_tetrahedron_count'],
            'minimum_material_link_length_ratio': float(stretches.min()),
            'maximum_material_link_length_ratio': float(stretches.max()),
            'center_grid_is_geometric_diagnostic_not_fem': True}
        valid = (geometry['minimum_center_grid_orientation_ratio'] >= NUMERICAL_LIMITS['minimum_tetrahedron_volume_ratio']
                 and stretches.min() >= NUMERICAL_LIMITS['minimum_edge_length_ratio']
                 and stretches.max() <= NUMERICAL_LIMITS['maximum_edge_length_ratio'])
        quiet = self.stability.update(points, fully_separated=robot['fully_separated'],
                                      baseline=frame['baseline'])
        return {'sponge': points.tolist()}, {
            'schema': 'elastic_sponge_native_geometry.v1', 'complete': True,
            'member_count': len(members), 'link_count': len(self.links),
            'geometry_valid': bool(valid), 'fully_separated': robot['fully_separated'],
            'quiet': quiet['quiet'], 'boundary_stability': quiet, 'geometry': geometry,
            'robot_geometry': dict(robot), 'numerical_geometry_limits': dict(NUMERICAL_LIMITS),
            'material_force_inferred_from_geometry': False}
