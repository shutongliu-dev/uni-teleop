"""Named-cloth isolation and legacy compatibility; no native physics acceptance."""
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from bench.simulation.isaacsim_backend.env import FR3DexterousIsaacEnv
from bench.simulation.isaacsim_backend.geometry import (
    cloth_grid, cloth_initial_vertices, named_cloth_specs,
)


def spec(name='towel', cells=(3, 2), origin=(0., 0., .905)):
    return {'id': name, 'enabled': True, 'cells_xy': list(cells), 'size_m': [.6, .4],
            'origin_m': list(origin), 'mass_kg': .108, 'friction': .45, 'self_contact': True,
            'isaacsim': {'cloth_thickness_m': .003, 'cloth_contact_offset_m': .004,
                         'cloth_rest_offset_m': .0015}}


class NativeSurfaceView:
    count = 1
    def __init__(self, points):
        self.points = np.asarray(points, dtype=np.float32)[None].copy()
        self.velocities = np.ones_like(self.points)
    def get_simulation_nodal_positions(self):
        return self.points.copy()
    def get_simulation_nodal_velocities(self):
        return self.velocities.copy()
    def set_simulation_nodal_positions(self, value, indices):
        assert np.array_equal(indices, [0])
        self.points = value.copy()
    def set_simulation_nodal_velocities(self, value, indices):
        assert np.array_equal(indices, [0])
        self.velocities = value.copy()


def make_env(specs):
    env = FR3DexterousIsaacEnv.__new__(FR3DexterousIsaacEnv)
    env.scene_spec = {'cloths': deepcopy(specs)}
    infos, views = {}, {}
    for entry in specs:
        name = entry['id']
        vertices, triangles = cloth_grid(entry)
        path = '/World/Cloths/' + name
        infos[name] = {'path': path, 'rest_vertices': vertices, 'triangles': triangles}
        views[path] = NativeSurfaceView(cloth_initial_vertices(entry, vertices))
    env.scene_info = {'cloths': infos}
    env._sim_view = SimpleNamespace(create_surface_deformable_body_view=lambda path: views[path])
    env._tensor = lambda value: np.asarray(value)
    env._one = np.asarray([0], dtype=np.int32)
    env._initialize_named_cloths()
    return env


def test_different_meshes_have_independent_views_and_reset_topologies():
    first, second = spec(), spec('towel_base', (5, 4), (.8, 0, .91))
    env = make_env([first, second])
    assert env.cloth_names == ['towel', 'towel_base']
    assert env._cloth_initials['towel'].shape == (12, 3)
    assert env._cloth_initials['towel_base'].shape == (30, 3)
    original = env._cloth_initials['towel_base'].copy()
    displaced = env._cloth_initials['towel'] + [.1, 0, .2]
    reset = env._validated_named_cloth_reset({'towel': displaced.tolist()})
    env._reset_named_cloths(reset)
    assert np.allclose(env._cloth_views['towel'].points[0], displaced)
    assert np.array_equal(env._cloth_views['towel_base'].points[0], original)
    assert np.array_equal(env._cloth_initials['towel_base'], original)
    assert all(not view.velocities.any() for view in env._cloth_views.values())
    with pytest.raises(ValueError, match='towel_base'):
        env._validated_named_cloth_reset({'towel_base': displaced.tolist()})
    with pytest.raises(ValueError, match='unknown'):
        env._validated_named_cloth_reset({'other': displaced.tolist()})


def test_initial_folded_nodes_do_not_change_flat_rest_topology():
    item = spec()
    rest, triangles = cloth_grid(item)
    initial = cloth_initial_vertices(item, rest)
    initial[:, 2] += .03 * np.sin(initial[:, 0] * 5)
    item['initial_vertices_m'] = initial.tolist()
    before = deepcopy(item)
    assert list(named_cloth_specs({'cloths': [item]})) == ['towel']
    env = make_env([item])
    assert np.array_equal(env._cloth_initials['towel'], initial)
    assert not env.scene_info['cloths']['towel']['rest_vertices'][:, 2].any()
    assert np.array_equal(env.cloth_triangles['towel'], triangles)
    assert item == before


@pytest.mark.parametrize('mutate', [
    lambda item: item.update(id='../bad'),
    lambda item: item.update(cells_xy=[3, 2.5]),
    lambda item: item.update(mass_kg=-.1),
    lambda item: item.update(origin_m=[0, float('nan'), 1]),
    lambda item: item.update(quaternion_wxyz=[0, 0, 0, 0]),
    lambda item: item.update(initial_vertices_m=[[0, 0, 0]]),
    lambda item: item.update(self_contact='yes'),
    lambda item: item['isaacsim'].update(cloth_thickness_m=0),
    lambda item: item['isaacsim'].update(cloth_contact_offset_m=.001),
    lambda item: item['isaacsim'].update(physics_dt_s=.01),
])
def test_malformed_named_specs_rejected_before_gpu_start(mutate):
    item = spec(); mutate(item)
    with pytest.raises(ValueError):
        named_cloth_specs({'cloths': [item]})


def test_identifiers_cannot_conflict_with_other_bodies_or_each_other():
    for scene in ({'cloths': [spec(), spec()]},
                  {'cloths': [spec()], 'objects': [{'id': 'towel'}]},
                  {'cloths': [spec()], 'soft_bodies': [{'id': 'towel'}]}):
        with pytest.raises(ValueError, match='conflict'):
            named_cloth_specs(scene)


def test_surface_native_count_and_node_shape_are_checked_per_id():
    env = make_env([spec()])
    env._cloth_views['towel'].count = 2
    with pytest.raises(RuntimeError, match='towel'):
        env._initialize_named_cloths()
    env = make_env([spec()])
    env._cloth_views['towel'].points = np.zeros((1, 11, 3))
    with pytest.raises(RuntimeError, match='topology'):
        env._initialize_named_cloths()


def capture_env(env, *, legacy=False):
    env._robot_view = SimpleNamespace(
        get_dof_positions=lambda: np.zeros((1, 2)), get_dof_velocities=lambda: np.zeros((1, 2)),
        get_link_transforms=lambda: np.asarray([[[0, 0, 0, 0, 0, 0, 1]]]),
        get_link_velocities=lambda: np.zeros((1, 1, 6)))
    env.object_names = []; env._object_views = {}; env._targets = np.zeros((1, 2))
    env.has_cloth = legacy
    if legacy:
        env._cloth_view = NativeSurfaceView(np.zeros((4, 3)))
    env.time_s = .02
    env.world = SimpleNamespace(current_time=.02, current_time_step_index=40)
    return env.capture_state()


def test_named_capture_does_not_alias_legacy_particle_arrays():
    env = make_env([spec(), spec('towel_base', (5, 4))])
    state = capture_env(env, legacy=True)
    assert state['particle_q'].shape == (4, 3)
    assert state['cloth_towel_q'].shape == (12, 3)
    assert state['cloth_towel_base_q'].shape == (30, 3)
    observation = env._named_cloth_observation(state)
    assert set(observation) == {'towel', 'towel_base'}
    assert len(observation['towel_base']['triangles']) == 40
    assert observation['towel']['vertices_m'] == state['cloth_towel_q'].tolist()
    env._cloth_views['towel'].points[0, 0, 0] += 1
    assert observation['towel']['vertices_m'][0][0] != env._cloth_views['towel'].points[0, 0, 0]


def test_no_collection_preserves_legacy_capture_fields_and_mesh():
    # Legacy specs are intentionally not revalidated or reinterpreted here.
    assert named_cloth_specs({'cloth': {'enabled': True, 'old_field': 'unchanged'}}) == {}
    env = make_env([])
    state = capture_env(env, legacy=True)
    assert not any(key.startswith('cloth_') for key in state)
    assert env._named_cloth_observation(state) == {}
    assert env.has_named_cloths is False
    assert state['particle_q'].shape == (4, 3)
