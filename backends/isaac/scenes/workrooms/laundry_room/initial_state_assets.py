"""Published garment reset data, independent of experimental run reports.

Export is an authoring operation. Runtime verifies the frozen data and material
contract; it never reopens the experiment that originally supplied the surface.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np

from bench.isaacsim.release import dump
from bench.paths import DATA_ROOT

RESET_PROFILE = 'published_settled_unfolded_v1'
REFERENCE_SCHEMA = 'laundry_unfolded_input.v1'
INPUT_SCHEMA = 'laundry_settled_input.v1'


def mesh_digest(rest, triangles):
    value = {'vertices_m': np.asarray(rest, dtype=float).tolist(),
             'triangles': np.asarray(triangles, dtype=int).tolist()}
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def export_reference(metadata, output):
    """Export only already validated physical inputs; provenance stays external."""
    output = Path(output).resolve()
    if output.exists() or not output.is_relative_to(f'{DATA_ROOT}/assets'):
        raise ValueError('Choose a fresh immutable input file under shared assets')
    if metadata['reset_profile'] != 'physx_native_settled_unfolded_v1':
        raise ValueError('Export requires the existing explicitly settled reset')
    cloth, = metadata['cloths']
    if cloth['initial_vertices_m'] != cloth['unfolded_reference_vertices_m']:
        raise ValueError('Reset must equal the independently authored unfolded surface')
    source = {'schema': INPUT_SCHEMA, 'task_id': metadata['task_id'],
        'mesh_profile': metadata['mesh_profile'],
        'rest_geometry_sha256': mesh_digest(cloth['rest_vertices_m'], cloth['triangles']),
        'material_prior': deepcopy(metadata['material_prior']),
        'physics_options': deepcopy(metadata['physics_options']),
        'source_center_xy_m': deepcopy(metadata['source_center_xy_m']),
        'source_yaw_quarter_turns': metadata['source_yaw_quarter_turns'],
        'source_yaw_degrees': metadata.get('source_yaw_degrees'),
        'initial_velocity_m_s': [0., 0., 0.]}
    record = {'schema': REFERENCE_SCHEMA, 'task_id': metadata['task_id'],
              'vertices_m': deepcopy(cloth['initial_vertices_m']), 'initial_state_input': source}
    dump(output, record)
    return record


def validate_source(source, *, task_id, mesh_profile, rest, triangles, material_prior,
                    physics_options, centre, quarter_turns, yaw_degrees, vertices):
    expected = {'schema': INPUT_SCHEMA, 'task_id': task_id, 'mesh_profile': mesh_profile,
        'rest_geometry_sha256': mesh_digest(rest, triangles),
        'material_prior': material_prior, 'physics_options': physics_options,
        'source_center_xy_m': np.asarray(centre).tolist(),
        'source_yaw_quarter_turns': quarter_turns, 'source_yaw_degrees': yaw_degrees,
        'initial_velocity_m_s': [0., 0., 0.]}
    if source != expected:
        raise ValueError('Published garment reset input differs from geometry/material/physics identity')
    q, rest, triangles = np.asarray(vertices), np.asarray(rest), np.asarray(triangles)
    if q.shape != rest.shape or not np.isfinite(q).all():
        raise ValueError('Published garment reset requires a complete finite surface')
    edges = np.unique(np.sort(np.concatenate((triangles[:, [0, 1]], triangles[:, [1, 2]],
                                              triangles[:, [2, 0]])), axis=1), axis=0)
    ratios = (np.linalg.norm(q[edges[:, 0]]-q[edges[:, 1]], axis=1)
              /np.linalg.norm(rest[edges[:, 0]]-rest[edges[:, 1]], axis=1))
    if (ratios.min() < .8 or ratios.max() > 1.2 or np.ptp(q[:, 2]) > .08
            or q[:, 2].min() < .898 or q[:, 2].max() > .98):
        raise ValueError('Published garment reset is not an intact unfolded surface at the table')
