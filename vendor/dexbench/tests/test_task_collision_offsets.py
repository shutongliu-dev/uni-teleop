import pytest

from bench.simulation.isaacsim_backend.asset_catalog import collision_offsets, inertia_principal_axes


def test_legacy_stage_defaults_and_new_small_clearance_task_are_distinct():
    assert collision_offsets({}) == {'contact_offset': .001, 'rest_offset': 0.}
    assert collision_offsets({'contact_offset_m': .0001, 'rest_offset_m': 0.}) == {
        'contact_offset': .0001, 'rest_offset': 0.}


@pytest.mark.parametrize('spec', [
    {'contact_offset_m': 0}, {'contact_offset_m': -1}, {'contact_offset_m': float('nan')},
    {'contact_offset_m': True}, {'rest_offset_m': float('inf')},
    {'rest_offset_m': .001}, {'contact_offset_m': .0001, 'rest_offset_m': .0002},
])
def test_invalid_offsets_are_rejected_before_scene_authoring(spec):
    with pytest.raises(ValueError, match='collision offsets'):
        collision_offsets(spec)


def test_asymmetric_asset_mass_axes_are_opt_in_and_preserve_quaternion_order():
    assert inertia_principal_axes({}) == [1., 0., 0., 0.]
    s = 2**-.5
    assert inertia_principal_axes({'principal_axes_xyzw': [0., 0., s, s]}) == [s, 0., 0., s]


@pytest.mark.parametrize('q', [[0, 0, 0, 0], [0, 0, 0, 2], [0, 0, 1],
                             [0, float('nan'), 0, 1], [False, 0, 0, 1]])
def test_invalid_inertia_axes_cannot_silently_become_identity(q):
    with pytest.raises(ValueError, match='principal axes'):
        inertia_principal_axes({'principal_axes_xyzw': q})
