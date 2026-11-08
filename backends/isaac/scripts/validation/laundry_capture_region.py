"""Original material-region checks with bounded contact-triangle support.

Only the explicit towel-stack profile permits a contact triangle's finite
element support vertices to extend beyond the original pointwise region. The
two actual witness points must stay inside that unchanged region on every
sample. No geometry, target, contact, or attachment is created here.
"""
from __future__ import annotations

import copy
import numpy as np


def validate_region_option(options, task_id, *, towel_adapter):
    profile = options.get('capture_region_profile', 'strict_vertices_v1')
    if profile not in ('strict_vertices_v1', 'region_support_v1'):
        raise ValueError('Unknown explicit material capture-region profile')
    if profile == 'region_support_v1' and (not towel_adapter or task_id != 'towel_stacking'):
        raise ValueError('Material support-region profile is restricted to towel stacking')
    return profile


def stack_region_descriptor(pick):
    seeds = pick['seed_vertex_ids']
    point = np.asarray(pick['edge_material_m'], float)
    if len(seeds) != 1 or point.shape != (3,) or not np.isfinite(point).all() or abs(point[0]) > 1e-8:
        raise ValueError('Original material crease-end seed required')
    return {'profile': 'region_support_v1', 'seed_vertex_id': int(seeds[0]),
            'seed_canonical_material_m': point.tolist(), 'radius_m': .030,
            'relative_material_x_half_width_m': .025,
            'coordinate_scope': 'Original rest offsets from the unchanged material seed; never world coordinates.'}


class MaterialCaptureRegion:
    def __init__(self, descriptor, triangles, rest_vertices):
        self.descriptor = copy.deepcopy(descriptor)
        expected = {'profile', 'seed_vertex_id', 'seed_canonical_material_m', 'radius_m',
                    'relative_material_x_half_width_m', 'coordinate_scope'}
        if (not isinstance(descriptor, dict) or set(descriptor) != expected
                or descriptor['profile'] != 'region_support_v1'
                or type(descriptor['seed_vertex_id']) is not int
                or descriptor['radius_m'] != .030
                or descriptor['relative_material_x_half_width_m'] != .025):
            raise ValueError('Fixed original30mm seed/25mm crease material region required')
        self.tri = np.asarray(triangles, np.int64)
        self.rest = np.asarray(rest_vertices, float)
        self.seed = descriptor['seed_vertex_id']
        p = np.asarray(descriptor['seed_canonical_material_m'], float)
        if (self.rest.ndim != 2 or self.rest.shape[1] != 3 or not np.isfinite(self.rest).all()
                or self.tri.ndim != 2 or self.tri.shape[1] != 3 or self.tri.min() < 0
                or self.tri.max() >= len(self.rest) or not 0 <= self.seed < len(self.rest)
                or p.shape != (3,) or not np.isfinite(p).all() or abs(p[0]) > 1e-8):
            raise ValueError('Original finite rest topology and crease seed required')

    def check(self, witness, topology, cloth_q, allowed_material_vertex_ids):
        if allowed_material_vertex_ids is None:
            raise ValueError('Original allowed material IDs must accompany the explicit region')
        q = np.asarray(cloth_q, float)
        if q.shape != self.rest.shape or not np.isfinite(q).all():
            raise ValueError('Complete finite actual cloth coordinates required')
        allowed = set(map(int, allowed_material_vertex_ids))
        if not allowed or min(allowed) < 0 or max(allowed) >= len(self.rest):
            raise ValueError('Original allowed material IDs are invalid')
        support_faces = np.asarray(topology['support_triangle_ids'], int)
        if np.any(support_faces < 0) or np.any(support_faces >= len(self.tri)):
            raise ValueError('Original support topology required')
        ids = np.unique(self.tri[support_faces])
        reasons, evidence = [], []
        for side in ('thumb', 'index'):
            w = witness[side]; face = int(w['triangle_id'])
            if face not in support_faces:
                raise ValueError('Contact face is absent from the original support patch')
            p = np.asarray(w['rest_point_m'], float); world = np.asarray(w['point_m'], float)
            vertices = self.rest[self.tri[face]]
            if p.shape != (3,) or world.shape != (3,) or not np.isfinite([p, world]).all():
                raise ValueError('Finite actual material witnesses required')
            # Recover barycentric weights in the material face and verify the
            # same material point maps to the recorded current world witness.
            uv, _, rank, _ = np.linalg.lstsq((vertices[1:]-vertices[0]).T, p-vertices[0], rcond=None)
            bary = np.r_[1-uv.sum(), uv]
            residual = float(np.linalg.norm(bary@vertices-p))
            world_residual = float(np.linalg.norm(bary@q[self.tri[face]]-world))
            if (rank != 2 or residual > 1e-7 or world_residual > 1e-7 or bary.min() < -1e-7 or bary.max() > 1+1e-7):
                reasons.append('witness_not_on_same_original_material_face')
            delta = p-self.rest[self.seed]
            distance = float(np.linalg.norm(delta)); x = abs(float(delta[0]))
            inside = distance <= .030+1e-9 and x <= .025+1e-9
            if not inside: reasons.append('actual_witness_outside_original_material_region')
            maximum_edge = float(np.linalg.norm(vertices[:, None]-vertices[None, :], axis=2).max())
            evidence.append({'finger': side, 'triangle_id': face, 'barycentric': bary.tolist(),
                'rest_point_m': p.tolist(), 'point_world_m': world.tolist(),
                'distance_to_seed_m': distance, 'absolute_relative_material_x_m': x,
                'inside_original_region': inside, 'face_maximum_rest_edge_m': maximum_edge,
                'rest_barycentric_residual_m': residual, 'world_correspondence_residual_m': world_residual})
        extras = []
        for node in sorted(set(map(int, ids))-allowed):
            memberships = []
            for row in evidence:
                if node in self.tri[row['triangle_id']]:
                    distance = float(np.linalg.norm(self.rest[node]-row['rest_point_m']))
                    if distance <= row['face_maximum_rest_edge_m']+1e-9:
                        memberships.append({'contact_triangle_id': row['triangle_id'],
                            'finger': row['finger'], 'distance_to_witness_m': distance,
                            'face_maximum_rest_edge_m': row['face_maximum_rest_edge_m']})
            if not memberships: reasons.append('outside_support_is_not_a_bounded_contact_face_vertex')
            extras.append({'vertex_id': node, 'distance_to_seed_m': float(np.linalg.norm(self.rest[node]-self.rest[self.seed])),
                'contact_face_bounds': memberships})
        rest_diameter = float(np.linalg.norm(self.rest[ids, None]-self.rest[ids][None], axis=2).max())
        world_diameter = float(np.linalg.norm(q[ids, None]-q[ids][None], axis=2).max())
        if len(ids) > 32 or rest_diameter > .035 or world_diameter > .035:
            reasons.append('complete_support_patch_exceeds_original_locality')
        return {'profile': 'region_support_v1', 'eligible': not reasons,
            'reasons': list(dict.fromkeys(reasons)), 'region': copy.deepcopy(self.descriptor),
            'runtime_rest_seed_m': self.rest[self.seed].tolist(), 'vertex_ids': ids.tolist(),
            'witnesses': evidence, 'additional_contact_support_vertices': extras,
            'material_diameter_m': rest_diameter, 'world_diameter_m': world_diameter,
            'scope': 'Unchanged pointwise material region; only bounded actual contact-face support may extend outside.'}
