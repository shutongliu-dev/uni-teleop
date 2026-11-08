"""CPU rejection checks for native water rendering evidence, not GPU qualification."""
from copy import deepcopy
import hashlib

import numpy as np
import pytest

from scripts.validation.living_water_visual import (
    particle_digest, validate_inventory, validate_native_advance,
    validate_same_tick_surfaces, validate_surfaces,
)


def inventory():
    ids = np.arange(488, dtype=np.int64)
    ids_sha = hashlib.sha256(ids.astype('<i8').tobytes()).hexdigest()
    task = {'evaluation': {'particle_count': 488, 'particle_set_id': 'water',
        'particle_mass_kg': .25 / 488, 'particle_ids_sha256': ids_sha}}
    state = {'task_particle_water_q': np.zeros((488, 3)),
        'task_particle_water_qd': np.zeros((488, 3)), 'task_particle_water_ids': ids,
        'native_physics_step_index': np.asarray(1), 'native_world_time_s': np.asarray(.0005),
        'time_s': np.asarray(0.)}
    observation = {'sim_time_s': 0., 'terminated': False, 'truncated': False,
        'metrics': {'state_finite': True}, 'task_state': {'reset_id': 'first-reset', 'errors': [],
        'water': {'particle_count': 488, 'ids_sha256': ids_sha,
            'counts': {'kettle': 488, 'cup': 0, 'spilled_outside': 0, 'in_flight_or_unresolved': 0}},
        'sampling': {'end_step': 1, 'end_time_s': .0005, 'sampled_substeps': 0}}}
    score = {'success': False, 'progress': 0., 'evidence_status': 'available', 'irrecoverable_reason': None,
        'initial_water_ml': 250., 'kettle_water_ml': 250., 'cup_water_ml': 0.,
        'current_spill_ml': 0., 'unresolved_ml': 0.}
    return state, observation, task, score


def surface(points, *, reused=False):
    return [{'input_sha256': particle_digest(points), 'particle_count': len(points),
        'vertices': 40, 'triangles': 76, 'mesh_path': '/World/ParticleSurfaceVisuals/water',
        'physics_state_writeback': False, 'reused_same_particle_positions': reused}]


def test_inventory_matches_frozen_count_identity_nominal_mass_and_clock():
    values = inventory()
    result = validate_inventory(*values, baseline=True)
    assert result['particle_count'] == 488 and result['declared_total_mass_kg'] == .25
    assert result['input_sha256'] == particle_digest(values[0]['task_particle_water_q'])


@pytest.mark.parametrize('fault', ['ids', 'nan', 'mass', 'count', 'compartment', 'clock', 'success', 'volume', 'reset_credit'])
def test_invalid_or_stale_water_evidence_is_rejected(fault):
    state, obs, task, score = inventory()
    if fault == 'ids':
        state['task_particle_water_ids'][1] = 0
    elif fault == 'nan':
        state['task_particle_water_q'][0, 0] = np.nan
    elif fault == 'mass':
        task['evaluation']['particle_mass_kg'] *= 2
    elif fault == 'count':
        obs['task_state']['water']['particle_count'] -= 1
    elif fault == 'compartment':
        obs['task_state']['water']['counts']['cup'] = 1
    elif fault == 'clock':
        obs['task_state']['sampling']['end_time_s'] += .0005
    elif fault == 'success':
        score['success'] = True
    elif fault == 'volume':
        score['unresolved_ml'] = 1.
    elif fault == 'reset_credit':
        score['progress'] = .25
    with pytest.raises(ValueError):
        validate_inventory(state, obs, task, score, baseline=True)


def test_particle_digest_binds_float32_content_not_memory_layout():
    points = np.arange(30, dtype=np.float64).reshape(10, 3)
    assert particle_digest(points) == hashlib.sha256(points.astype(np.float32).tobytes()).hexdigest()
    assert particle_digest(np.asfortranarray(points)) == particle_digest(points)
    assert particle_digest(points + .01) != particle_digest(points)


def test_surface_must_match_exact_raw_particles_and_repeat_cache_identity():
    points = np.zeros((488, 3), dtype=np.float32)
    first, repeated = surface(points), surface(points, reused=True)
    validate_surfaces(first, points)
    validate_surfaces(repeated, points, expect_cached=True)
    validate_same_tick_surfaces(first, repeated)
    stale = points.copy()
    stale[0, 0] = .001
    with pytest.raises(ValueError, match='stale/different'):
        validate_surfaces(repeated, stale)
    with pytest.raises(ValueError, match='reuse'):
        validate_surfaces(first, points, expect_cached=True)


@pytest.mark.parametrize('field,value', [('vertices', 0), ('triangles', True),
    ('physics_state_writeback', True), ('mesh_path', ''), ('particle_count', 487),
    ('reused_same_particle_positions', None)])
def test_surface_rejects_empty_geometry_writeback_and_missing_cache_receipt(field, value):
    points = np.zeros((488, 3))
    receipts = surface(points)
    receipts[0][field] = value
    with pytest.raises(ValueError):
        validate_surfaces(receipts, points)


def test_equal_input_hash_does_not_hide_mesh_change_on_repeat():
    first = surface(np.zeros((488, 3)))
    repeated = deepcopy(first)
    repeated[0]['triangles'] += 1
    with pytest.raises(ValueError, match='surface identity'):
        validate_same_tick_surfaces(first, repeated)


def test_native_hold_requires_all_clocks_and_exact_substep_count():
    before = {'native_physics_step_index': np.asarray(1), 'native_world_time_s': np.asarray(.0005),
        'time_s': np.asarray(0.)}
    after = {'native_physics_step_index': np.asarray(401), 'native_world_time_s': np.asarray(.2005),
        'time_s': np.asarray(.2)}
    assert validate_native_advance(before, after, dt=.0005, duration=.2) == 400
    for key in after:
        bad = deepcopy(after)
        bad[key] += 1
        with pytest.raises(ValueError, match='native steps'):
            validate_native_advance(before, bad, dt=.0005, duration=.2)


def test_native_clock_accepts_float32_dt_but_rejects_a_missing_substep():
    dt = float(np.float32(.0005))
    before = {'native_physics_step_index': np.asarray(2),
              'native_world_time_s': np.asarray(2 * dt), 'time_s': np.asarray(0.)}
    after = {'native_physics_step_index': np.asarray(602),
             'native_world_time_s': np.asarray(602 * dt), 'time_s': np.asarray(.3)}
    assert validate_native_advance(before, after, dt=.0005, duration=.3) == 600
    after['native_world_time_s'] -= dt
    with pytest.raises(ValueError, match='native steps'):
        validate_native_advance(before, after, dt=.0005, duration=.3)
