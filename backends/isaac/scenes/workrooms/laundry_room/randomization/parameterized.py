"""Reset-time cloth rigid transforms and actual local material parameters."""
from copy import deepcopy
import math

import numpy as np

from . import PARAMETERIZED_VERSION
from .common import canonical_hash

EDGE_MARGIN_M = .002


def _rotation(angle):
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, -s], [s, c]], dtype=np.float64)


def make_plan(task_id, configuration_id, seed, metadata, *, attempt=0):
    from scenes.workrooms.parameterized.spec import sample_parameters
    rows = {row['id']: row for row in metadata['cloths']}
    expected = {'towel', 'towel_base'} if task_id == 'towel_stacking' else {'towel'}
    if set(rows) != expected or metadata.get('task_id') != task_id:
        raise ValueError('Original laundry cloth inventory differs')
    parameters = sample_parameters('laundry_room', task_id, configuration_id, seed, attempt=attempt)
    points = np.asarray(rows['towel']['initial_vertices_m'], dtype=np.float64)
    rectangle = np.asarray(metadata['clear_rectangle_xy_m'], dtype=np.float64)
    if (points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all()
            or rectangle.shape != (4,) or not np.isfinite(rectangle).all()):
        raise ValueError('Finite original cloth points and support rectangle required')
    pivot = (points[:, :2].min(0)+points[:, :2].max(0))/2
    yaw = float(parameters['layout']['yaw_rad'])
    rotated = (points[:, :2]-pivot)@_rotation(yaw).T+pivot
    lower = rectangle[[0, 2]]+EDGE_MARGIN_M-rotated.min(0)
    upper = rectangle[[1, 3]]-EDGE_MARGIN_M-rotated.max(0)
    if np.any(lower > upper):
        raise ValueError('Parameterized cloth yaw cannot fit the original support rectangle; reject this attempt')
    requested = np.asarray(parameters['layout']['translation_xy_m'], dtype=np.float64)
    applied = np.clip(requested, lower, upper) if seed else np.zeros(2)
    if seed and np.linalg.norm(applied) < .04:
        raise ValueError('Support projection leaves less than 4 cm actual translation; reject this attempt')
    return {'schema': 'laundry_seed_plan.v1', 'generator_version': PARAMETERIZED_VERSION,
        'task_id': task_id, 'configuration_id': configuration_id, 'seed': seed, 'attempt': attempt,
        'parameters': parameters, 'rotation_pivot_xy_m': pivot.tolist(),
        'translations_xy_m': {name: applied.tolist() if name == 'towel' else [0., 0.] for name in rows},
        'yaw_rad': yaw, 'z_translation_m': 0., 'target_coordinates_changed': False,
        'stack_base_position_changed': False,
        'parent_initial_vertices_sha256': canonical_hash({name: row['initial_vertices_m'] for name, row in rows.items()}),
        'appearance_seed': seed,
        'source_bounds_guard': {'rectangle_xy_m': rectangle.tolist(), 'minimum_declared_edge_margin_m': EDGE_MARGIN_M,
            'requested_translation_xy_m': requested.tolist(), 'applied_translation_xy_m': applied.tolist(),
            'support_projection_applied': not np.array_equal(requested, applied),
            'minimum_noncanonical_translation_m': .04,
            'actual_translation_norm_m': float(np.linalg.norm(applied)),
            'translation_interval_m': [lower.tolist(), upper.tolist()],
            'robot_dependent_sampling': False}}


def transform_metadata(metadata, plan):
    result = deepcopy(metadata)
    if not plan['seed']:
        return result
    pivot = np.asarray(plan['rotation_pivot_xy_m'])
    rotation = _rotation(plan['yaw_rad'])
    factor = float(plan['parameters']['physics']['density_factor'])
    if factor != plan['parameters']['physics']['mass_factor'] or not np.isfinite(factor) or factor <= 0:
        raise ValueError('Cloth density and mass use the same single positive factor')
    for row in result['cloths']:
        if row['id'] == 'towel':
            points = np.asarray(row['initial_vertices_m'], dtype=np.float64).copy()
            points[:, :2] = (points[:, :2]-pivot)@rotation.T+pivot+plan['translations_xy_m']['towel']
            row['initial_vertices_m'] = points.tolist()
        row['mass_kg'] *= factor
        row['areal_density_kg_m2'] *= factor
    if 'source_center_xy_m' in result:
        result['source_center_xy_m'] = ((np.asarray(result['source_center_xy_m'])-pivot)@rotation.T
                                        +pivot+plan['translations_xy_m']['towel']).tolist()
    return result


def material_metadata(metadata, scene, plan):
    """Bind declared mass/density to the exact source cloth material fields."""
    if not plan['seed']:
        return deepcopy(metadata)
    result = deepcopy(metadata)
    originals = {row['id']: row for row in scene['cloths']}
    factor = plan['parameters']['physics']['friction_factor']
    applied = {}
    for row in result['cloths']:
        original = originals[row['id']]
        mass_factor = plan['parameters']['physics']['density_factor']
        if not math.isclose(row['mass_kg'], original['mass_kg']*mass_factor, rel_tol=1e-12):
            raise ValueError('Cloth metadata and native original mass differ')
        old_mu = float(original.get('friction', .45))
        row['friction'] = old_mu*factor
        applied[row['id']] = {'mass_kg': row['mass_kg'], 'areal_density_kg_m2': row['areal_density_kg_m2'],
            'volume_density_kg_m3': row['mass_kg']/(row['rest_area_m2']*row['physical_thickness_m']),
            'static_friction': row['friction'], 'dynamic_friction': row['friction'],
            'original_mass_kg': original['mass_kg'], 'original_friction': old_mu,
            'mass_factor_applied_once': mass_factor,
            'native_binding': 'scene.cloths[id].mass_kg/friction -> native surface material density/staticFriction/dynamicFriction'}
    result['parameterized_physics'] = {'schema': PARAMETERIZED_VERSION, 'parameters': deepcopy(plan['parameters']),
        'applied': applied, 'native_readback_verified': False, 'material_calibrated': False,
        'material_limitations': ['Static friction is unsupported by PhysX deformables; dynamic friction is used.'],
        'rest_mesh_topology_thickness_unchanged': True, 'solver_controller_unchanged': True}
    if 'material_prior' in result:
        prior = result['material_prior']
        result['parameterized_physics']['parent_material_prior'] = deepcopy(prior)
        prior['mass_kg'] = next(row['mass_kg'] for row in result['cloths'] if row['id'] == 'towel')
        prior['mass_policy'] = 'Original physical thickness/rest area; one explicitly sampled density factor scales mass.'
        prior['parameterized_instance_sha256'] = plan['parameters']['instance_sha256']
        prior['scope'] = 'Uncalibrated local material parameter instance; native dynamics and robot contact require new validation.'
    return result
