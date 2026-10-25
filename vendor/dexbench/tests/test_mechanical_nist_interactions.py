"""Engineering-probe reductions remain signed and cannot manufacture support."""
import ast
from copy import deepcopy
import inspect
import numpy as np
import pytest

from scripts.validation import mechanical_nist_interactions as probe
from scripts.validation.mechanical_nist_interactions import (
    components, contact_records, friction_records, geometry_checks, mating_contact_checks, native_clock_check,
    passive_anchor_checks, signed_external_load, terminal_checks,
)


def test_native_probe_uses_one_filter_list_per_sensor_pattern():
    # Exercise the exact production call without importing or starting Isaac.
    # Pinned omni.physics.tensors/api.py requires List[List[str]] for list sensors.
    calls = [node for node in ast.walk(ast.parse(inspect.getsource(probe.run)))
             if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
             and node.func.attr == 'create_rigid_contact_view']
    assert len(calls) == 1
    actors = {'left': '/World/Left', 'right': '/World/Right'}
    filters = [*actors.values(), '/World/Table']

    class SimulationView:
        def create_rigid_contact_view(self, patterns, *, filter_patterns, max_contact_data_count):
            assert patterns == list(actors.values())
            assert filter_patterns == [filters, filters]
            assert filter_patterns[0] is not filter_patterns[1]
            assert max_contact_data_count == probe.CAPACITY
            return 'native-contact-view'

    call = compile(ast.Expression(calls[0]), '<native contact invocation>', 'eval')
    assert eval(call, {'sim': SimulationView(), 'metadata': {'actors': actors},
                       'filters': filters, 'CAPACITY': probe.CAPACITY}) == 'native-contact-view'


def test_native_contact_records_preserve_all_six_buffers_and_signed_normals():
    class ContactView:
        def get_contact_data(self, dt):
            assert dt == 1.  # Keep raw impulse, not force rescaled by dt.
            return (np.array([[2.], [3.]]),
                    np.array([[0., 0., .8], [1., 0., .8]]),
                    np.array([[0., 0., 1.], [0., 0., -1.]]),
                    np.array([[-.001], [.0002]]),
                    np.array([[2]]), np.array([[0]]))

    rows = contact_records(ContactView(), ['/World/Body'], ['/World/Table'])
    assert len(rows) == 1
    assert rows[0]['normal_impulses_ns'] == [2., 3.]
    assert rows[0]['normals'] == [[0., 0., 1.], [0., 0., -1.]]
    assert rows[0]['separations_m'] == [-.001, .0002]


def test_native_friction_keeps_signed_world_vectors_and_pair_buffer_indices():
    class ContactView:
        def get_friction_data(self, dt):
            assert dt == 1.
            return (np.array([[.001, -.002, .003], [-.004, .001, -.005], [.0005, .0002, .006]]),
                    np.array([[0., 0., .8], [1., 0., .8], [2., 0., .8]]),
                    np.array([[2, 1]]), np.array([[0, 2]]))

    rows, matrix = friction_records(ContactView(), ['/World/Body'], ['/World/Left', '/World/Right'])
    np.testing.assert_allclose(matrix, [[[-6., -2., -4.], [1., .4, 12.]]])
    np.testing.assert_allclose(rows[0]['impulse_vector_ns'], [-.003, -.001, -.002])
    assert rows[1]['friction_impulses_ns'] == [[.0005, .0002, .006]]
    assert rows[1]['points_m'] == [[2., 0., .8]]
    assert rows[1]['actor1'] == '/World/Right'


def test_seating_local_point_compares_physical_tip_to_declared_world_point():
    point = np.array([0., 0., .054])
    q = [np.cos(.1), 0., np.sin(.1), 0.]
    origin = point-probe.rotation(q) @ point
    obj = {'position_m': origin.tolist(), 'quaternion_wxyz': q}
    goal = {'id': 'plug', 'actor': 'plug', 'target_position_m': [0., 0., 0.],
        'target_quaternion_wxyz': [1., 0., 0., 0.], 'position_tolerance_m': [.001]*3,
        'orientation_tolerance_rad': .3}
    metadata = {'task_parameters': {'seating_goals': [goal]}}
    assert not geometry_checks(metadata, {'plug': obj})['passed']
    goal['actor_local_point_m'] = point.tolist()
    goal['target_position_m'] = point.tolist()
    assert geometry_checks(metadata, {'plug': obj})['passed']
    obj['position_m'][0] += .002
    assert not geometry_checks(metadata, {'plug': obj})['passed']
    assert goal['target_position_m'] == point.tolist()


def mating_fixture():
    actors = {'belt': '/World/Belt', 'left': '/World/Left', 'right': '/World/Right'}
    state = {'position_m': [0., 0., .8], 'quaternion_wxyz': [1., 0., 0., 0.]}
    metadata = {'actors': actors, 'fixtures': {'table': '/World/Table'},
        'fixture_metadata': {'table': {'initial_pose_xyzw': [0., 0., .78, 0., 0., 0., 1.]}},
        'task_parameters': {'flexible_groups': [{'id': 'belt', 'actor_ids': ['belt'],
            'required_mating_contact_ids': ['left', 'right']}]}}
    rows = [{'time_s': index*.0005, 'objects': dict.fromkeys(actors, deepcopy(state)),
             'raw_contacts': []} for index in range(3)]
    return metadata, rows


def test_plate_supported_belt_requires_both_actual_wheel_pairs():
    metadata, rows = mating_fixture()
    rows[1]['raw_contacts'] = [{'actor0': '/World/Belt', 'actor1': '/World/Table/Shape',
                               'separations_m': [0.]}]
    assert not mating_contact_checks(rows, metadata)['passed']
    rows[1]['raw_contacts'].append({'actor0': '/World/Belt', 'actor1': '/World/Left',
                                    'separations_m': [.0003]})
    assert not mating_contact_checks(rows, metadata)['passed']
    rows[1]['raw_contacts'].append({'actor0': '/World/Right', 'actor1': '/World/Belt',
                                    'separations_m': [.0003]})
    assert mating_contact_checks(rows, metadata)['passed']  # Static sleeping witness.
    rows[2]['objects'] = deepcopy(rows[2]['objects'])
    rows[2]['objects']['belt'] = deepcopy(rows[2]['objects']['belt'])
    rows[2]['objects']['belt']['position_m'][0] += .002
    assert not mating_contact_checks(rows, metadata)['passed']


def test_seat_requires_its_mate_and_distant_manifold_does_not_count():
    metadata, rows = mating_fixture()
    metadata['task_parameters'] = {'seating_goals': [
        {'id': 'seat', 'actor': 'belt', 'contact_ids': ['left']}]}
    rows[1]['raw_contacts'] = [{'actor0': '/World/Belt', 'actor1': '/World/Right',
                               'separations_m': [0.]}]
    assert not mating_contact_checks(rows, metadata)['passed']
    rows[1]['raw_contacts'][0]['actor1'] = '/World/Left'
    rows[1]['raw_contacts'][0]['separations_m'] = [.0006]
    assert not mating_contact_checks(rows, metadata)['passed']
    rows[1]['raw_contacts'][0]['separations_m'] = [.0004]
    assert mating_contact_checks(rows, metadata)['passed']
    rows[2]['objects'] = deepcopy(rows[2]['objects'])
    rows[2]['objects']['left'] = deepcopy(rows[2]['objects']['left'])
    rows[2]['objects']['left']['position_m'][0] += .002
    assert not mating_contact_checks(rows, metadata)['passed']


def test_native_float32_callback_clock_does_not_fail_after_421_steps():
    dt = float(np.float32(probe.PHYSICS_DT))
    baseline = 2*dt
    for tick in (1, 421, 422, 2000, 10000):
        result = native_clock_check(tick=tick, step=2+tick,
            time_s=baseline+tick*dt, baseline_step=2, baseline_time=baseline,
            previous_time=baseline+(tick-1)*dt, configured_dt=probe.PHYSICS_DT)
        assert result['passed']
    assert baseline+422*dt-baseline-422*probe.PHYSICS_DT > 1.e-8


@pytest.mark.parametrize('change', [
    {'step': 4}, {'time_s': .002}, {'configured_dt': .001}, {'previous_time': .0015},
])
def test_native_clock_still_rejects_missing_duplicate_or_changed_steps(change):
    dt = float(np.float32(probe.PHYSICS_DT))
    args = dict(tick=1, step=3, time_s=3*dt, baseline_step=2, baseline_time=2*dt,
                previous_time=2*dt, configured_dt=probe.PHYSICS_DT)
    args.update(change)
    assert not native_clock_check(**args)['passed']


def test_signed_component_support_excludes_internal_links_without_positive_clamping():
    actors = {'a': '/World/A', 'b': '/World/B'}
    filters = ['/World/A', '/World/B', '/World/Table', '/World/Fixture']
    matrix = np.zeros((2, 4, 3))
    matrix[0, 1, 2], matrix[1, 0, 2] = 100., -100.
    matrix[0, 2, 2], matrix[1, 3, 2] = 10., -11.
    assert np.array_equal(signed_external_load(matrix, ['a', 'b'], filters, ['a', 'b'], actors),
                          [0., 0., -1.])
    assert np.array_equal(signed_external_load(matrix, ['b', 'a'], filters, ['a', 'b'], actors),
                          [0., 0., -1.])


def test_support_connectivity_comes_only_from_declared_native_joint_endpoints():
    actors = dict.fromkeys(['a', 'b', 'c', 'd'])
    assert components(actors, {'ab': {'body0': 'a', 'body1': 'b'},
                               'bc': {'body0': 'b', 'body1': 'c'}}) == [('a', 'b', 'c'), ('d',)]


def test_fixed_link_rotation_error_cannot_pass_on_anchor_coincidence_alone():
    row = {'kind': 'fixed', 'anchor_separation_m': 0.,
           'relative_rotation_matrix': [[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]]}
    assert not passive_anchor_checks({'terminal': row})['passed']
    row['kind'] = 'spherical'
    assert passive_anchor_checks({'terminal': row})['passed']
    row['anchor_separation_m'] = .004
    assert not passive_anchor_checks({'terminal': row})['passed']


def test_terminal_probe_requires_complete_window_and_net_upward_support():
    metadata = {'actors': {'a': '/World/A'}, 'task_parameters': {}}
    object_state = {'position_m': [0., 0., .8], 'quaternion_wxyz': [1., 0., 0., 0.],
                    'linear_velocity_m_s': [0., 0., 0.], 'angular_velocity_rad_s': [0., 0., 0.]}
    rows = [{'time_s': .7+tick*.0005, 'objects': {'a': deepcopy(object_state)},
             'component_external_force_n': {'0': [0., 0., 1.]},
             'passive_links': {'passed': True, 'maximum_anchor_error_m': 0.}}
            for tick in range(601)]
    assert terminal_checks(rows, metadata, {}, {'a': .1}, goal=False)['passed']
    assert not terminal_checks(rows[::2], metadata, {}, {'a': .1}, goal=False)['passed']
    for row in rows:
        row['component_external_force_n']['0'][2] = -1.
    result = terminal_checks(rows, metadata, {}, {'a': .1}, goal=False)
    assert not result['passed']
    assert result['components']['0']['mean_signed_external_contact_force_n'][2] == -1.
    for row in rows:
        row['component_normal_force_n'] = {'0': [0., 0., -2.]}
        row['component_friction_force_n'] = {'0': [0., 0., 3.]}
        row['component_external_force_n'] = {'0': [0., 0., -2.]}
        row['component_total_external_force_n'] = {'0': [0., 0., 1.]}
    result = terminal_checks(rows, metadata, {}, {'a': .1}, goal=False)
    assert result['passed']
    assert result['components']['0']['mean_signed_normal_force_n'] == [0., 0., -2.]
    assert result['components']['0']['mean_signed_friction_force_n'] == [0., 0., 3.]
    assert result['components']['0']['mean_signed_external_contact_force_n'] == [0., 0., -2.]
    assert result['components']['0']['mean_signed_total_external_contact_force_n'] == [0., 0., 1.]
