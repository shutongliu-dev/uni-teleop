import copy

import numpy as np
import pytest

from scripts.validation.laundry_pivot_reference import (
    PROFILE, bind_pivot_reference, validate_pivot_reference, validate_pivot_reference_options)


def options():
    return {'attachment_material_mode': 'captured_pivot_v1', 'pickup_profile': 'shirt_cuff_corners',
            'phases': {'0': {'sleeve_cuff_corner': 'maximum', 'pivot_material_reference': PROFILE}}}


def test_absent_option_is_exact_and_explicit_maximum_phase_is_accepted():
    for task in ('shirt_folding', 'trousers_folding', 'towel_stacking'):
        validate_pivot_reference_options({}, task)
    value = options()
    before = copy.deepcopy(value)
    validate_pivot_reference_options(value, 'shirt_folding')
    assert value == before
    assert bind_pivot_reference({}, {}, {}) is None


@pytest.mark.parametrize('mutation', [
    lambda o: o.update(pivot_material_reference=PROFILE),
    lambda o: o.update(attachment_material_mode='captured_patch'),
    lambda o: o.update(pickup_profile='boundary_midpoint'),
    lambda o: o['phases']['0'].update(sleeve_cuff_corner='minimum'),
    lambda o: o['phases']['0'].update(pivot_material_reference=True),
    lambda o: o['phases'].update({'2': copy.deepcopy(o['phases']['0'])}),
    lambda o: o['phases'].update({False: copy.deepcopy(o['phases']['0'])}),
])
def test_bad_or_unused_later_option_is_rejected_before_startup(mutation):
    o = options()
    mutation(o)
    with pytest.raises(ValueError):validate_pivot_reference_options(o, 'shirt_folding')


def test_wrong_task_or_adapter_does_not_inherit_the_sleeve_policy():
    with pytest.raises(ValueError):validate_pivot_reference_options(options(), 'trousers_folding')
    with pytest.raises(ValueError):validate_pivot_reference_options(options(), 'shirt_folding', garment_adapter=False)


def test_reference_is_an_original_material_inset_and_not_world_coordinates():
    for side in (-1., 1.):
        rest = np.array([[side * .35, .19, .0015], [side * .34, .19, .0015]])
        feature = 'right_cuff' if side > 0 else 'left_cuff'
        task = {'asset_metadata': {'cloths': [{'rest_vertices_m': rest.tolist(),
                'boundary_features': {feature: [0, 1]}}]}}
        edge = {'natural_corner_vertex_id': 0, 'feature': feature, 'seed_vertex_ids': [0]}
        value = bind_pivot_reference(task, edge, {'pivot_material_reference': PROFILE})
        np.testing.assert_allclose(validate_pivot_reference(value, rest), rest[1], atol=1e-16)
        for change in ({'rest_point_m': [3., 2., 1.]}, {'rest_point_m': [float('nan'), .19, .0015]},
                       {'rest_point_m': [[side * .34, .19, .0015]]}, {'natural_corner_vertex_id': True},
                       {'maximum_common_vertex_distance_m': .1}, {'natural_corner_rest_m': [0, 0, 0]},
                       {'unexpected': True}):
            with pytest.raises(ValueError):validate_pivot_reference({**value, **change}, rest)
