"""Geometric invariants and bounded input; these are not robot grasp tests."""
from copy import deepcopy

import numpy as np
import pytest

from agent.model_client.geometry_math import execute
from bench.geometry import quaternion_matrix


@pytest.mark.parametrize('arguments', [
    {'quaternion_wxyz': [0., 0., 0., 1.000006565]},
    {'rotation_vector_rad': [0., 0., np.pi]},
    {'axis_xyz': [0., 0., 8.], 'angle_rad': np.pi},
])
def test_rotation_forms_produce_the_same_unit_rotation(arguments):
    value = execute('compute_rotation', arguments)
    assert value['defined']
    q = value['result']['quaternion_wxyz']
    assert np.linalg.norm(q) == pytest.approx(1., abs=1.e-14)
    assert quaternion_matrix(q) == pytest.approx(np.diag([-1., -1., 1.]), abs=1.e-14)
    assert not value['ik_checked'] and not value['collision_checked'] and not value['grasp_checked']


def test_pose_delta_distinguishes_parent_from_local_axes_and_preserves_input():
    original = {'frame_id': 'world', 'position_m': [1., 2., 3.],
                'quaternion_wxyz': [2**-.5, 0., 0., 2**-.5]}
    saved = deepcopy(original)
    parent = execute('compute_pose_delta', {'pose': original, 'translation_m': [1., 0., 0.]})
    local = execute('compute_pose_delta', {'pose': original, 'translation_m': [1., 0., 0.], 'coordinates': 'local'})
    assert parent['result']['pose']['position_m'] == pytest.approx([2., 2., 3.])
    assert local['result']['pose']['position_m'] == pytest.approx([1., 3., 3.])
    assert original == saved
    args = {'pose': local['result']['pose'], 'points_m': [[.1, -.2, .3], [0., 0., 0.]]}
    forward = execute('compute_transform_points', args)['result']['points_m']
    inverse = execute('compute_transform_points', {**args, 'points_m': forward, 'inverse': True})
    assert np.array(inverse['result']['points_m']) == pytest.approx(np.array(args['points_m']))


@pytest.mark.parametrize('distance,residual', [(.05, 0.), (.07, .01)])
def test_alignment_reports_pair_mismatch_instead_of_claiming_a_grasp(distance, residual):
    value = execute('compute_frame_alignment', {
        'local_pair_m': [[-.025, 0., 0.], [.025, 0., 0.]],
        'target_pair_m': [[.7, -.1-distance/2, .8], [.7, -.1+distance/2, .8]],
        'local_reference_xyz': [0., 0., 1.], 'target_reference_xyz': [0., 0., 1.]})
    assert value['defined'] and not value['grasp_checked']
    result = value['result']
    assert result['point_residuals_m'] == pytest.approx([residual, residual])
    assert result['pose']['position_m'] == pytest.approx([.7, -.1, .8])
    assert np.array(quaternion_matrix(result['pose']['quaternion_wxyz'])) @ [1., 0., 0.] == pytest.approx([0., 1., 0.])


@pytest.mark.parametrize('name,arguments,reason', [
    ('compute_rotation', {'quaternion_wxyz': [0., 0., 0., 0.]}, 'undefined_zero_vector'),
    ('compute_vector', {'operation': 'normalize', 'a': [0., 0., 0.]}, 'undefined_zero_vector'),
    ('compute_frame_alignment', {'local_pair_m': [[0., 0., 0.], [1., 0., 0.]],
        'target_pair_m': [[0., 0., 0.], [0., 0., 1.]]}, 'reference_parallel_to_pair'),
])
def test_undefined_geometry_is_explicit(name, arguments, reason):
    value = execute(name, arguments)
    assert value == {'defined': False, 'reason': reason,
                     'source': 'explicit_user_geometry_only', 'advances_physics': False}


@pytest.mark.parametrize('op,extra,expected', [
    ('add', {'b': [3., 2., 1.]}, [4., 4., 4.]),
    ('subtract', {'b': [3., 2., 1.]}, [-2., 0., 2.]),
    ('midpoint', {'b': [3., 2., 1.]}, [2., 2., 2.]),
    ('scale', {'scale': 2.}, [2., 4., 6.]),
    ('dot', {'b': [3., 2., 1.]}, 10.),
    ('cross', {'b': [0., 1., 0.]}, [-3., 0., 1.]),
    ('distance', {'b': [1., 2., 3.]}, 0.),
])
def test_explicit_vector_operations(op, extra, expected):
    assert execute('compute_vector', {'operation': op, 'a': [1., 2., 3.], **extra})['result']['value'] == pytest.approx(expected)


@pytest.mark.parametrize('name,arguments', [
    ('compute_python', {'code': 'open("/etc/passwd").read()'}),
    ('compute_vector', {'operation': 'normalize', 'a': [float('inf'), 0., 0.]}),
    ('compute_vector', {'operation': 'scale', 'a': [1., 2., 3.], 'scale': 1.e7}),
    ('compute_rotation', {'quaternion_wxyz': [1., 0., 0., 0.], 'rotation_vector_rad': [0., 0., 0.]}),
    ('compute_vector', {'operation': 'normalize', 'a': [1., 2., 3.], 'path': '/etc/passwd'}),
])
def test_non_numeric_ambiguous_or_unbounded_requests_are_rejected(name, arguments):
    with pytest.raises(ValueError):
        execute(name, arguments)
