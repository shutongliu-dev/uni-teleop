"""A discretized point pivot selected only after complete measured capture.

This changes the holding constraint, never the capture test or the transport
reference. The selected mesh vertex need not itself touch both finger pads.
"""
from __future__ import annotations

import hashlib

import numpy as np

from bench.evaluation.cloth.cloth_robot_collision import (
    apply_transform, planes_in_body, transform_xyzw,
)
from scripts.validation.laundry_material_topology import face_edge_neighbors, local_material_patch
from scripts.validation.laundry_pinch_geometry import _surface


def _on_triangle(point, vertices):
    edges = (vertices[1:]-vertices[0]).T
    weights, _, rank, _ = np.linalg.lstsq(edges, point-vertices[0], rcond=None)
    return (rank == 2 and np.linalg.norm(edges @ weights-(point-vertices[0])) <= 1.e-8
            and np.min(weights) >= -1.e-6 and np.sum(weights) <= 1.+1.e-6)


def select_captured_pivot(result, triangles, rest, cloth_q, bodies, diagnostic,
                          geometry, *, side, time_s, material_reference=None):
    """Return one real shared vertex, with explicit sampling/slot diagnostics.

The full original capture result must already be positive. Rechecking its
identity and topology prevents a partial, stale or remapped patch from being
promoted to a point constraint. No proximity-only fallback is provided.
    """
    if (result.get('capture_candidate') is not True or result.get('reasons')
            or result.get('capture_profile') not in (
                'strict_v1', 'local_assist_v2', 'adjacent_patch_v3', 'angled_adjacent_patch_v4')
            or not np.isfinite(result.get('continuous_duration_s', np.nan))
            or result['continuous_duration_s'] < .1-1.e-9):
        raise ValueError('A complete positive original capture window is required for a pivot')
    q, rest, body = (np.asarray(value, float) for value in (cloth_q, rest, bodies))
    tri = np.asarray(triangles)
    if (q.ndim != 2 or q.shape[1] != 3 or rest.shape != q.shape
            or body.shape != (len(geometry.body_names), 7)
            or tri.ndim != 2 or tri.shape[1] != 3 or not len(tri)
            or not np.issubdtype(tri.dtype, np.integer) or tri.min() < 0 or tri.max() >= len(q)
            or not all(np.isfinite(a).all() for a in (q, rest, body))):
        raise ValueError('Finite actual states and original indexed material are required')
    identity = {'body_q_float64_sha256': hashlib.sha256(body.astype('<f8').tobytes()).hexdigest(),
                'cloth_q_float64_sha256': hashlib.sha256(q.astype('<f8').tobytes()).hexdigest()}
    expected_profile = (result['capture_profile']
                        if result['capture_profile'] == 'angled_adjacent_patch_v4' else None)
    if (diagnostic.get('state_identity') != identity or diagnostic.get('side') != side
            or diagnostic.get('distal_group_severe_penetration') is not False
            or diagnostic.get('contact_force_measured') is not False
            or diagnostic.get('engineering_geometry_profile') != expected_profile):
        raise ValueError('Pivot geometry must match the actual captured state and profile')
    if diagnostic.get('contact_normal_profile','facet_v1')!=result.get('contact_normal_profile','facet_v1'):
        raise ValueError('Pivot and capture must use the same actual contact-normal profile')
    if diagnostic.get('contact_witness_profile','centroid_v1')!=result.get('contact_witness_profile','centroid_v1'):
        raise ValueError('Pivot and capture must use the same actual contact-witness profile')
    evidence = result.get('evidence', {})
    if (not np.isfinite(time_s) or evidence.get('last_time_s') != time_s
            or not np.isfinite(evidence.get('first_time_s', np.nan))
            or time_s-evidence['first_time_s'] < .1-1.e-9):
        raise ValueError('Pivot selection must occur at the final original capture sample')
    faces = evidence.get('triangle_ids', [])
    if (len(faces) != 2 or any(type(i) is not int for i in faces)
            or faces[0] == faces[1] or min(faces) < 0 or max(faces) >= len(tri)):
        raise ValueError('The two original witnessed triangles are required')
    common = np.intersect1d(tri[faces[0]], tri[faces[1]])
    if not len(common):
        raise ValueError('A pivot requires a vertex shared by both actual witnessed triangles')
    topology = local_material_patch(tri, *faces, face_neighbors=face_edge_neighbors(tri))
    if topology is None or evidence.get('material_topology') != topology:
        raise ValueError('Captured material topology differs from the original mesh')
    ids = result.get('vertex_ids', [])
    if (not 3 <= len(ids) <= 32 or any(type(i) is not int for i in ids)
            or len(set(ids)) != len(ids)
            or set(ids) != set(map(int, np.unique(tri[topology['support_triangle_ids']])))):
        raise ValueError('The entire original captured patch must be retained as reference')
    if any(np.linalg.norm(points[:, None]-points[None, :], axis=2).max() > .035
           for points in (q[ids], rest[ids])):
        raise ValueError('Captured pivot patch exceeds the original 35mm locality')
    pair, witness = evidence.get('surface_pair'), evidence.get('witnesses')
    variants = [diagnostic]+diagnostic.get('distal_contact_diagnostics', [])
    if not any(v.get('surface_pair') == pair and not v.get('severe_penetration_pads')
               and v.get('engineering_geometry_profile') == expected_profile
               and v.get('contact_normal_profile','facet_v1') == result.get('contact_normal_profile','facet_v1')
               and v.get('contact_witness_profile','centroid_v1') == result.get('contact_witness_profile','centroid_v1')
               and v.get('multilayer_diagnostic', {}).get('plausible')
               and witness in v['multilayer_diagnostic'].get('paired_witnesses', []) for v in variants):
        raise ValueError('Capture evidence must be one of the fresh actual surface witnesses')
    points = []
    for finger, face in zip(('thumb', 'index'), faces):
        row = witness[finger]
        point = np.asarray(row.get('point_m'), float)
        if (row.get('triangle_id') != face or point.shape != (3,)
                or not np.isfinite(point).all() or not _on_triangle(point, q[tri[face]])):
            raise ValueError('Material witness must lie on its actual original triangle')
        points.append(point)
    midpoint = np.mean(points, axis=0)
    distances = np.linalg.norm(q[common]-midpoint, axis=1)
    selected = int(common[np.lexsort((common, distances))[0]])
    if material_reference is not None:
        from scripts.validation.laundry_pivot_reference import validate_pivot_reference
        reference = validate_pivot_reference(material_reference, rest)
        material_neighbourhood = np.vstack([rest[ids], reference])
        if np.linalg.norm(material_neighbourhood[:, None]-material_neighbourhood[None, :], axis=2).max() > .035:
            raise ValueError('Material pivot reference exceeds the original captured 35mm neighbourhood')
        distances = np.linalg.norm(rest[common]-reference, axis=1)
        eligible = distances <= material_reference['maximum_common_vertex_distance_m']
        if not np.any(eligible):
            raise ValueError('No actual common vertex is near the declared material reference')
        candidates, distances = common[eligible], distances[eligible]
        selected = int(candidates[np.lexsort((candidates, distances))[0]])
    point = q[selected]
    pad_points = np.asarray([witness[f]['surface_point_m'] for f in ('thumb', 'index')], float)
    if pad_points.shape != (2, 3) or not np.isfinite(pad_points).all():
        raise ValueError('Finite actual opposing pad witness points are required')
    axis = pad_points[1]-pad_points[0]
    length = float(np.linalg.norm(axis))
    if length < 1.e-8:
        raise ValueError('Distinct actual pad witness points must define the local slot')
    fraction = float((point-pad_points[0]) @ axis / (length*length))
    source = next((g for g in [geometry]+geometry._variants
                   if dict(zip(('thumb', 'index'), g.surface_pair)) == pair), None)
    if source is None or side not in source.pads:
        raise ValueError('Original cooked surfaces for the actual witnessed pair are required')
    pad_distances = {}
    for finger, pad in source.pads[side].items():
        matrix = transform_xyzw(body[pad['body']])
        distance, nearest, _ = _surface(point, apply_transform(pad['inner'], matrix), pad['faces'],
                                       planes_in_body(pad['inner_planes'], matrix))
        pad_distances[finger] = {'body_name': pad['body_name'],
            'signed_inner_surface_distance_m': distance, 'closest_surface_point_m': nearest.tolist()}
    report = {
        'schema': 'laundry_captured_pivot.v1', 'state_identity': identity,
        'selected_vertex_id': selected, 'common_witness_vertex_candidates': common.tolist(),
        'selection': 'Nearest actual material-witness midpoint; exact distance ties use smallest vertex ID.',
        'trajectory_reference_material_ids': list(ids), 'constraint_material_vertex_ids': [selected],
        'witness_triangle_ids': faces, 'surface_pair': dict(pair),
        'material_topology': topology, 'actual_selected_vertex_m': point.tolist(),
        'actual_witness_midpoint_m': midpoint.tolist(),
        'distance_to_witness_midpoint_m': float(np.linalg.norm(point-midpoint)),
        'distance_to_each_material_witness_m': {f: float(np.linalg.norm(point-p))
                                              for f, p in zip(('thumb', 'index'), points)},
        'pad_witness_axis_lateral_m': float(np.linalg.norm(point-(pad_points[0]+fraction*axis))),
        'pad_witness_segment_fraction': fraction,
        'pad_witness_segment_outside_m': max(0., -fraction, fraction-1.)*length,
        'actual_cooked_pad_distances': pad_distances,
        'selected_vertex_two_finger_contact_proven': False,
        'scope': 'One discretized local holding point after full-patch capture; '
                 'mesh sampling distance is reported, not a point-contact or torque certificate.'}
    if result.get('contact_normal_profile','facet_v1')!='facet_v1':
        report['contact_normal_profile']=result['contact_normal_profile']
    if result.get('contact_witness_profile','centroid_v1')!='centroid_v1':
        report['contact_witness_profile']=result['contact_witness_profile']
    if material_reference is not None:
        report['selection'] = 'Nearest declared original material reference among actual common witness vertices; exact ties use smallest vertex ID.'
        report['material_reference'] = dict(material_reference)
        report['material_reference_within_original_35mm_neighbourhood'] = True
        report['distance_to_material_reference_m'] = float(np.linalg.norm(rest[selected]-reference))
    return [selected],report
