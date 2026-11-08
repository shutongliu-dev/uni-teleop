"""Opt-in material identity for a maximum sleeve-corner point constraint."""
from __future__ import annotations

import numpy as np


PROFILE = 'maximum_sleeve_inset10mm_v1'
SCHEMA = 'laundry_capture_pivot_rest_reference.v1'


def validate_pivot_reference_options(options, task_id, *, garment_adapter=True):
    if 'pivot_material_reference' in options:
        raise ValueError('A material pivot reference must be explicitly scoped to one sleeve phase')
    for index, phase in options.get('phases', {}).items():
        if 'pivot_material_reference' not in phase:
            continue
        if (phase['pivot_material_reference'] != PROFILE or index not in ('0', '1')
                or task_id != 'shirt_folding' or not garment_adapter
                or options.get('attachment_material_mode') != 'captured_pivot_v1'
                or options.get('pickup_profile') != 'shirt_cuff_corners'
                or phase.get('sleeve_cuff_corner') != 'maximum'):
            raise ValueError('Material pivot references require an explicit maximum sleeve corner and point-assist mode')


def bind_pivot_reference(task, edge, phase):
    if 'pivot_material_reference' not in phase:
        return None
    if phase['pivot_material_reference'] != PROFILE:
        raise ValueError('Unknown material pivot reference')
    spec = task['asset_metadata']['cloths'][0]
    rest = np.asarray(spec['rest_vertices_m'], float)
    corner = edge.get('natural_corner_vertex_id')
    feature = edge.get('feature')
    if (type(corner) is not int or feature not in ('left_cuff', 'right_cuff')
            or corner not in spec['boundary_features'][feature]
            or corner not in edge['seed_vertex_ids']):
        raise ValueError('The actual selected original sleeve boundary corner is required')
    point = rest[corner].copy()
    point[0] -= np.sign(point[0]) * .010
    result = {'schema': SCHEMA, 'profile': PROFILE, 'rest_point_m': point.tolist(),
              'natural_corner_vertex_id': corner, 'natural_corner_rest_m': rest[corner].tolist(),
              'maximum_common_vertex_distance_m': .015}
    validate_pivot_reference(result, rest)
    return result


def validate_pivot_reference(reference, rest):
    if reference is None:
        return None
    keys = {'schema', 'profile', 'rest_point_m', 'natural_corner_vertex_id',
            'natural_corner_rest_m', 'maximum_common_vertex_distance_m'}
    if (not isinstance(reference, dict) or set(reference) != keys
            or reference['schema'] != SCHEMA or reference['profile'] != PROFILE
            or type(reference['natural_corner_vertex_id']) is not int
            or type(reference['maximum_common_vertex_distance_m']) not in (int, float)
            or reference['maximum_common_vertex_distance_m'] != .015):
        raise ValueError('A bounded original material pivot reference is required')
    rest = np.asarray(rest, float)
    corner = reference['natural_corner_vertex_id']
    point = np.asarray(reference['rest_point_m'], float)
    source = np.asarray(reference['natural_corner_rest_m'], float)
    if (rest.ndim != 2 or rest.shape[1] != 3 or not 0 <= corner < len(rest)
            or point.shape != (3,) or source.shape != (3,)
            or not np.isfinite(point).all() or not np.isfinite(source).all()
            or not np.array_equal(source, rest[corner]) or abs(source[0]) <= .010):
        raise ValueError('Material pivot reference differs from its original boundary identity')
    expected = source.copy()
    expected[0] -= np.sign(expected[0]) * .010
    if not np.array_equal(point, expected):
        raise ValueError('Material pivot reference must be the declared 10 mm torso inset')
    return point
