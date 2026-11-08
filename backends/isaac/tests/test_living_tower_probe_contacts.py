"""Lossless observation optimization; these CPU fixtures do not run physics."""
from copy import deepcopy
import json

import numpy as np
import pytest

from scenes.workrooms.living_room import native_evidence as original
from scripts.validation.living_tower_interactions import fast_raw_contacts, cached_normalizer, json_checks


def buffers(seed=0, count=19):
    rng = np.random.default_rng(seed)
    impulse = rng.standard_normal((count, 1)).astype(np.float32)
    points = rng.standard_normal((count, 3)).astype(np.float32)
    normals = rng.standard_normal((count, 3)).astype(np.float32)
    separation = rng.standard_normal((count, 1)).astype(np.float32)
    order = rng.permutation(count)
    # Opposite sensor view with reordered points and reversed normal. Retain
    # signed impulses, including negative/zero and subnormal float32 values.
    impulse[:3, 0] = [0., -1.e-30, 1.e-30]
    arrays = [np.concatenate((impulse, impulse[order], np.zeros((1, 1), np.float32))),
        np.concatenate((points, points[order], np.zeros((1, 3), np.float32))),
        np.concatenate((normals, -normals[order], np.zeros((1, 3), np.float32))),
        np.concatenate((separation, separation[order], np.zeros((1, 1), np.float32))),
        np.array([[0, count], [count, 0]], np.int32),
        np.array([[999999, 0], [count, 999999]], np.int32)]
    return arrays, ['/World/A', '/World/B'], [['/World/A', '/World/B']]*2


@pytest.mark.parametrize('seed', range(10))
def test_all_signed_points_and_reordered_reverse_views_match_bit_values(seed):
    args = buffers(seed=seed, count=37)
    a, b = original.raw_contacts(*args), fast_raw_contacts(*args)
    assert a == b and len(a) == 37
    for x, y in zip(a, b):
        assert np.asarray(x['impulse_on_a_n_s'], np.float64).tobytes() == np.asarray(y['impulse_on_a_n_s'], np.float64).tobytes()


@pytest.mark.parametrize('defect', ['mirror', 'nan', 'self', 'negative_start', 'overflow', 'shape'])
def test_rejects_the_same_invalid_evidence_as_frozen_conversion(defect):
    arrays, sensors, filters = buffers()
    if defect == 'mirror': arrays[2][20, 1] += .01
    if defect == 'nan': arrays[1][0, 1] = np.nan
    if defect == 'self': sensors[0] = '/World/B'
    if defect == 'negative_start': arrays[5][0, 1] = -1
    if defect == 'overflow': arrays[5][0, 1] = len(arrays[0])-2
    if defect == 'shape': arrays[4] = arrays[4][:1]
    for converter in (original.raw_contacts, fast_raw_contacts):
        with pytest.raises(ValueError):
            converter(arrays, sensors, filters)


@pytest.mark.parametrize('game', [False, True])
def test_path_cache_preserves_unknown_pair_filtering_and_never_caches_motion(game):
    bodies = {'block_00': '/World/LivingTask/Blocks/block_00'}
    fixtures = {'front_collection_tray_floor': '/World/LivingTask/Trays/front/Floor'}
    convert = cached_normalizer(original, bodies, game=game, contact_actor_paths=fixtures)
    rows = [{'a': '/World/UnknownA', 'b': '/World/UnknownB', 'point_m': [0., 0., 0.],
        'impulse_on_a_n_s': [0., 0., 0.], 'separation_m': 0.}]
    for other in ['/World/Room/Geometry/worktop/Collider', fixtures['front_collection_tray_floor'],
                  '/World/Robot/front/index_proximal']:
        rows.append({**rows[0], 'a': bodies['block_00']+'/Collider', 'b': other})
    for step in range(4):
        changing = deepcopy(rows)
        changing[1]['point_m'][0] = step*.01
        changing[2]['impulse_on_a_n_s'][2] = step*.1
        assert convert(changing) == original.normalized_contacts(changing, bodies, game=game, contact_actor_paths=fixtures)


def test_real_native_position_comparison_exports_as_json_boolean():
    # Exact type boundary of the completed extraction report: JSON endpoint
    # position minus the initial tensor-derived NumPy coordinate.
    final_y = -.32499998807907104
    initial_y = np.asarray([.65, 0., .945], dtype=np.float32)[1]
    native_check = abs(final_y-initial_y) > .3
    assert isinstance(native_check, np.bool_)
    checks = json_checks({'real_continuous_extraction': native_check, 'complete': True})
    assert all(type(value) is bool for value in checks.values())
    assert json.loads(json.dumps({'checks': checks, 'passed': all(checks.values())}, allow_nan=False)) == {
        'checks': {'real_continuous_extraction': True, 'complete': True}, 'passed': True}
    with pytest.raises(TypeError):
        json_checks({'not_a_verdict': np.float32(1.)})
