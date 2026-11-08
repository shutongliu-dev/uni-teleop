"""Synthetic cooked-solid fixtures; these do not establish native export provenance."""
from copy import deepcopy
import numpy as np
import pytest
from scipy.spatial import ConvexHull

from bench.evaluation.cloth.cloth_robot_collision import SCHEMA, cooked_piece_geometry, transform_xyzw, apply_transform
from bench.evaluation.deformable_robot_geometry import RigidCompositeRobotGeometry


def exported(names, half=(.005, .005, .005), *, robot=False):
    vertices = np.array([[x, y, z] for x in (-half[0], half[0])
                        for y in (-half[1], half[1]) for z in (-half[2], half[2])])
    piece = cooked_piece_geometry({'vertices': vertices.tolist(), 'planes': ConvexHull(vertices).equations.tolist()})
    prefix = '/World/Robot/' if robot else '/World/Task/'
    requested = {name: prefix+name for name in names}
    return {'schema': SCHEMA, 'status': 'complete', 'issues': [], 'visual_mesh_fallback': False,
        'state_hash_before': 'synthetic_hash', 'state_hash_after': 'synthetic_hash',
        'layer_hashes_before': {'synthetic_layer': 'hash'}, 'layer_hashes_after': {'synthetic_layer': 'hash'},
        'requested_body_paths': requested,
        'colliders': [{'status': 'verified_cooking_representation', 'body_name': name,
            'body_path': requested[name], 'path': requested[name]+'/Collider',
            'mesh_to_body_column_matrix': np.eye(4).tolist(), 'pieces': [deepcopy(piece)]} for name in names]}


def pose(xyz=(0., 0., 0.), wxyz=(1., 0., 0., 0.)):
    return {'position_m': list(xyz), 'quaternion_wxyz': list(wxyz)}


def checker(component_export, names):
    return RigidCompositeRobotGeometry(exported(['hand'], half=(.01,)*3, robot=True),
                                       ['hand'], component_export, names)


def test_every_one_of_28_components_can_prevent_false_release():
    names = ['sponge']+['cell_'+str(i) for i in range(27)]
    geometry = checker(exported(names), names)
    objects = {name: pose((1., 0., 0.)) for name in names}
    robot = np.array([[0., 0., 0., 0., 0., 0., 1.]])
    clear = geometry.check(objects, robot)
    assert clear['fully_separated'] and clear['component_convex_pieces_checked'] == 28
    for name in (names[0], names[13], names[-1]):
        objects[name] = pose((.019, 0., 0.))  # 4 mm actual gap, within the 5 mm band.
        report = geometry.check(objects, robot)
        assert report['complete'] and not report['fully_separated']
        assert not report['contact_force_measured']
        assert any(name+'/Collider' in row['volume_body'] for row in report['near_pairs'])
        objects[name] = pose((1., 0., 0.))


def test_mesh_to_body_and_native_wxyz_pose_are_composed_in_metres():
    component = exported(['cell'], half=(.01, .002, .003))
    matrix = np.diag([2., .5, 1., 1.])
    matrix[:3, 3] = [.1, 0., 0.]
    component['colliders'][0]['mesh_to_body_column_matrix'] = matrix.tolist()
    geometry = checker(component, ['cell'])
    objects = {'cell': pose((.2, 0., 0.), (np.sqrt(.5), 0., 0., np.sqrt(.5)))}
    # The collider center becomes (.2,.1,0), elongated along world Y. Applying
    # native local pose or scale twice, or using xyzw as wxyz, misses this pair.
    robot = np.array([[.2, .133, 0., 0., 0., 0., 1.]])
    assert not geometry.check(objects, robot)['fully_separated']
    robot[0, 1] = .15
    assert geometry.check(objects, robot)['fully_separated']


def test_solid_enclosure_is_rejected_without_any_boundary_contact():
    geometry = checker(exported(['large'], half=(1.,)*3), ['large'])
    robot = np.array([[0., 0., 0., 0., 0., 0., 1.]])
    result = geometry.check({'large': pose()}, robot)
    assert not result['fully_separated']
    assert any(row['solid_enclosure'] and not row['surface_within_clearance'] for row in result['near_pairs'])
    # The inverse enclosure direction is caught by component surface triangles.
    small = checker(exported(['small'], half=(.001,)*3), ['small'])
    result = small.check({'small': pose()}, robot)
    assert not result['fully_separated']
    assert any(row['surface_within_clearance'] for row in result['near_pairs'])


@pytest.mark.parametrize('fault', ['lost_collider', 'lost_requested', 'empty_piece', 'duplicate_path',
                                 'mismatched_body_path', 'missing_state_hash', 'missing_layer_hash',
                                 'state_mutated', 'layer_mutated', 'visual_fallback', 'export_issue'])
def test_incomplete_component_export_never_certifies_release(fault):
    component = exported(['a', 'b'])
    if fault == 'lost_collider': component['colliders'].pop()
    elif fault == 'lost_requested': component['requested_body_paths'].pop('b')
    elif fault == 'empty_piece': component['colliders'][1]['pieces'] = []
    elif fault == 'duplicate_path': component['colliders'].append(deepcopy(component['colliders'][0]))
    elif fault == 'mismatched_body_path': component['colliders'][1]['body_path'] = '/World/Other'
    elif fault == 'missing_state_hash':
        component.pop('state_hash_before'); component.pop('state_hash_after')
    elif fault == 'missing_layer_hash':
        component.pop('layer_hashes_before'); component.pop('layer_hashes_after')
    elif fault == 'state_mutated': component['state_hash_after'] = 'different'
    elif fault == 'layer_mutated': component['layer_hashes_after'] = {'other': 'hash'}
    elif fault == 'visual_fallback': component['visual_mesh_fallback'] = True
    elif fault == 'export_issue': component['issues'] = [{'reason': 'missing_native_collider'}]
    with pytest.raises(ValueError):
        checker(component, ['a', 'b'])


def test_all_pieces_of_a_decomposed_member_remain_in_the_union():
    component = exported(['cell'])
    near_piece = deepcopy(component['colliders'][0]['pieces'][0])
    far_piece = deepcopy(near_piece)
    far_piece['conservative_outer_vertices'] = (np.array(far_piece['conservative_outer_vertices'])+[1., 0., 0.]).tolist()
    component['colliders'][0]['pieces'] = [far_piece, near_piece]
    geometry = checker(component, ['cell'])
    report = geometry.check({'cell': pose()}, np.array([[0., 0., 0., 0., 0., 0., 1.]]))
    assert report['component_convex_pieces_checked'] == 2
    assert not report['fully_separated']


@pytest.mark.parametrize('fault', ['missing_member', 'nonfinite', 'nonunit_quaternion', 'missing_robot_pose'])
def test_incomplete_native_poses_fail_closed(fault):
    geometry = checker(exported(['cell']), ['cell'])
    objects = {'cell': pose((1., 0., 0.))}
    robot = np.array([[0., 0., 0., 0., 0., 0., 1.]])
    if fault == 'missing_member': objects.clear()
    elif fault == 'nonfinite': objects['cell']['position_m'][0] = float('nan')
    elif fault == 'nonunit_quaternion': objects['cell']['quaternion_wxyz'][0] = 2.
    elif fault == 'missing_robot_pose': robot = np.empty((0, 7))
    with pytest.raises((ValueError, KeyError)):
        geometry.check(objects, robot)


def test_invalid_robot_export_remains_fail_closed():
    robot = exported(['hand'], robot=True)
    robot['issues'].append({'reason': 'missing_native_collider'})
    with pytest.raises(ValueError):
        RigidCompositeRobotGeometry(robot, ['hand'], exported(['cell']), ['cell'])


@pytest.mark.parametrize('angle', [0., np.pi/4, np.pi/2, .37])
def test_world_broadphase_preserves_original_rotated_clearance_and_enclosure(angle):
    geometry = checker(exported(['a', 'b'], half=(.001,)*3), ['a', 'b'])
    quat = [0., 0., np.sin(angle/2), np.cos(angle/2)]
    robot = np.array([[0., 0., 0., *quat]])
    rotation = transform_xyzw(robot[0])[:3, :3]
    positions = [rotation@np.array([x, y, 0.]) for x in (.0159999, .0160001, .0160003)
                 for y in (0., .01, .016)]
    # A rotated local expanded box projects more than 5 mm onto a world axis.
    # Merely adding 5 mm to an unexpanded world AABB drops this conservative pair.
    positions += [np.array([.0212, 0., 0.]), np.array([0., 0., 0.]), np.ones(3)]
    for position in positions:
        objects = {'a': pose(position, (quat[3], *quat[:3])), 'b': pose((2., 1., 0.))}
        pieces = {}
        for name, row in geometry.local.items():
            state = objects[row['body']]
            q = state['quaternion_wxyz']
            transform = transform_xyzw([*state['position_m'], *q[1:], q[0]])
            pieces[name] = {'positions_m': apply_transform(row['points'], transform)}
        original = geometry.audit.check(pieces, robot)
        fast = geometry.check(objects, robot)
        assert fast['fully_separated'] == original['fully_separated']
        assert fast['near_pairs'] == original['near_pairs']
