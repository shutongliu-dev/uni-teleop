"""Reset-only XY translations; material coordinates and all targets stay fixed."""
from copy import deepcopy

import numpy as np

from . import (GENERATOR_VERSION, SHIRT_NEAR_GENERATOR_VERSION, PARAMETERIZED_VERSION,
               checked_generator_version, selection)
from .common import canonical_hash

MAX_TRANSLATION_M = .04
DECLARED_CLEARANCE_M = .002
DIRECTIONS = ((0., 0.), (-1., -1.), (1., -1.), (-1., 1.), (1., 1.),
              (-1., 0.), (1., 0.), (0., -1.), (0., 1.), (.5, -.5))
SHIRT_NEAR_MAX_TRANSLATION_M = .02
SHIRT_NEAR_DIRECTIONS = ((0., 0.), (-1., -1.), (1., -1.), (-1., -.5), (1., -.5),
                         (-1., 0.), (1., 0.), (0., -1.), (0., -.5), (.5, -.5))


def case_plan(task_id, configuration_id, seed, metadata, *, generator_version=GENERATOR_VERSION, attempt=0):
    selection(task_id, seed, configuration_id)
    checked_generator_version(task_id, configuration_id, generator_version)
    if generator_version == PARAMETERIZED_VERSION:
        from .parameterized import make_plan
        return make_plan(task_id, configuration_id, seed, metadata, attempt=attempt)
    if attempt != 0:
        raise ValueError('Historical laundry generators have no candidate attempts')
    near = generator_version == SHIRT_NEAR_GENERATOR_VERSION
    maximum = SHIRT_NEAR_MAX_TRANSLATION_M if near else MAX_TRANSLATION_M
    directions = SHIRT_NEAR_DIRECTIONS if near else DIRECTIONS
    rows = {row['id']: row for row in metadata['cloths']}
    expected = {'towel', 'towel_base'} if task_id == 'towel_stacking' else {'towel'}
    if set(rows) != expected or metadata.get('task_id') != task_id:
        raise ValueError('Original laundry cloth inventory differs')
    vertices = np.asarray(rows['towel']['initial_vertices_m'], dtype=np.float64)
    rectangle = np.asarray(metadata['clear_rectangle_xy_m'], dtype=np.float64)
    if (vertices.ndim != 2 or vertices.shape[1] != 3 or not np.isfinite(vertices).all()
            or rectangle.shape != (4,) or not np.isfinite(rectangle).all()):
        raise ValueError('Finite original cloth points and working rectangle required')
    lo, hi = vertices[:, :2].min(0), vertices[:, :2].max(0)
    lower = lo-rectangle[[0, 2]]-DECLARED_CLEARANCE_M
    upper = rectangle[[1, 3]]-hi-DECLARED_CLEARANCE_M
    if np.any(lower < 0) or np.any(upper < 0):
        raise ValueError('Original movable cloth has insufficient declared XY clearance')
    direction = directions[seed]
    xy = [float(d*min(maximum, lower[i] if d < 0 else upper[i]))
          for i, d in enumerate(direction)]
    # Seed zero is copied, never reserialized through arithmetic.
    if seed == 0:
        xy = [0., 0.]
    return {'schema': 'laundry_seed_plan.v1', 'generator_version': generator_version,
            'task_id': task_id, 'configuration_id': configuration_id, 'seed': seed,
            'translations_xy_m': {name: (xy if name == 'towel' else [0., 0.]) for name in rows},
            'yaw_rad': 0., 'z_translation_m': 0., 'target_coordinates_changed': False,
            'stack_base_position_changed': False,
            'parent_initial_vertices_sha256': canonical_hash({n: r['initial_vertices_m'] for n, r in rows.items()}),
            'appearance_seed': seed,
            'source_bounds_guard': {'rectangle_xy_m': rectangle.tolist(),
                                    'minimum_declared_edge_margin_m': DECLARED_CLEARANCE_M,
                                    'maximum_axis_translation_m': maximum}}


def validate_plan(plan, metadata):
    if not isinstance(plan, dict) or plan != case_plan(
            plan.get('task_id'), plan.get('configuration_id'), plan.get('seed'), metadata,
            generator_version=plan.get('generator_version'), attempt=plan.get('attempt', 0)):
        raise ValueError('Laundry reset plan differs from its finite deterministic recipe')
    return plan


def transformed_metadata(metadata, plan):
    validate_plan(plan, metadata)
    result = deepcopy(metadata)
    if plan['seed'] == 0:
        return result
    if plan['generator_version'] == PARAMETERIZED_VERSION:
        from .parameterized import transform_metadata
        return transform_metadata(result, plan)
    for row in result['cloths']:
        xy = plan['translations_xy_m'][row['id']]
        if xy != [0., 0.]:
            points = np.asarray(row['initial_vertices_m'], dtype=np.float64).copy()
            points[:, :2] += np.asarray(xy)
            row['initial_vertices_m'] = points.tolist()
    if 'source_center_xy_m' in result:
        result['source_center_xy_m'] = (np.asarray(result['source_center_xy_m'], dtype=np.float64)
                                        + plan['translations_xy_m']['towel']).tolist()
    # Existing source-record objects remain historical provenance; exact current
    # initialization is the new rows above and the separate pinned reset asset.
    return result
