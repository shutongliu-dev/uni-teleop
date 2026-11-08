"""Full-part observation contracts, independent of native physical acceptance."""
from copy import deepcopy

import numpy as np
import pytest

from scenes.workrooms.kitchen.elastic_sponge_dynamics import ElasticSpongeEvidence
from scenes.workrooms.kitchen import rules


@pytest.fixture(scope='module')
def metadata():
    from test_sponge_lattice import authored
    _, meta, _, _ = authored()
    return meta


def evidence(metadata):
    model = ElasticSpongeEvidence(metadata, rules.thresholds('surface_wiping'), .0005)
    objects = {r['id']: {'position_m': r['initial_pose_xyzw'][:3],
        'quaternion_wxyz': [r['initial_pose_xyzw'][6], *r['initial_pose_xyzw'][3:6]]}
        for r in metadata['objects']}
    joints = {link['prim_path'].rsplit('/', 1)[-1]: {
        'kind': 'elastic_d6', 'body0': link['body0'], 'body1': link['body1'],
        'anchor_delta_world_m': [0., 0., 0.], 'anchor_delta_joint0_m': [0., 0., 0.],
        'anchor_separation_m': 0., 'relative_rotation_matrix': np.eye(3).tolist(), 'force_measured': False}
        for link in metadata['elastic_composites']['sponge']['links']}
    frame = {'baseline': True, 'joints': joints,
             'elastic_robot_geometry': {'complete': True, 'fully_separated': True,
                'method': 'native_cooked_rigid_composite_geometry', 'clearance_m': .005}}
    return model, objects, frame


def test_full_boundary_and_all_parts_at_rest(metadata):
    model, objects, frame = evidence(metadata)
    points, result = model.sample(frame, objects)
    assert result['geometry_valid'] and not result['quiet']
    assert result['member_count'] == 28 and result['link_count'] == 63
    assert len(points['sponge']) == sum(len(r['collision_support_points_m']) for r in metadata['objects'])
    for _ in range(850):
        frame['baseline'] = False
        _, result = model.sample(frame, objects)
    assert result['quiet']
    assert result['boundary_stability']['released_quiet_native_s'] >= .4


def test_workspace_uses_all_material_collision_surfaces(metadata):
    from scripts.diagnostics.check_kitchen_workspace import physical_collider_paths
    parent = next(r for r in metadata['objects'] if r['id'] == 'sponge')
    paths = physical_collider_paths(metadata, parent)
    assert len(paths) == 28
    assert set(paths) == {p for r in metadata['objects'] for p in r['collider_paths']}


def test_one_moving_top_cell_cannot_hide_behind_quiet_pad(metadata):
    model, objects, frame = evidence(metadata)
    moving = 'sponge_cell_2_2_2'
    for tick in range(60):
        frame['baseline'] = tick == 0
        objects[moving]['position_m'] = np.asarray(objects[moving]['position_m']) + [.00005, 0., 0.]
        _, result = model.sample(frame, objects)
    assert not result['quiet']
    assert result['boundary_stability']['maximum_rolling_node_speed_m_s'] == pytest.approx(.1)


def test_missing_joint_or_part_or_robot_geometry_fails_closed(metadata):
    model, objects, frame = evidence(metadata)
    frame['joints'].pop(next(iter(frame['joints'])))
    with pytest.raises(ValueError, match='joint evidence'):
        model.sample(frame, objects)
    model, objects, frame = evidence(metadata)
    objects.pop('sponge_cell_2_2_2')
    with pytest.raises(ValueError, match='Missing native elastic part'):
        model.sample(frame, objects)
    model, objects, frame = evidence(metadata)
    frame['elastic_robot_geometry']['complete'] = False
    with pytest.raises(ValueError, match='cooked-robot'):
        model.sample(frame, objects)
    model, objects, frame = evidence(metadata)
    frame['joints'][next(iter(frame['joints']))]['relative_rotation_matrix'][0][0] = float('nan')
    with pytest.raises(ValueError, match='finite elastic joint geometry'):
        model.sample(frame, objects)


def test_separated_cell_and_center_grid_inversion_are_invalid(metadata):
    model, objects, frame = evidence(metadata)
    objects['sponge_cell_2_2_2']['position_m'] = np.asarray(objects['sponge_cell_2_2_2']['position_m']) + [0, 0, .3]
    _, result = model.sample(frame, objects)
    assert not result['geometry_valid']
    assert result['geometry']['maximum_material_link_length_ratio'] > 3
    model, objects, frame = evidence(metadata)
    a, b = 'sponge_cell_1_1_1', 'sponge_cell_1_1_2'
    objects[a]['position_m'], objects[b]['position_m'] = objects[b]['position_m'], objects[a]['position_m']
    _, result = model.sample(frame, objects)
    assert not result['geometry_valid']
    assert result['geometry']['inverted_center_grid_simplex_count'] > 0


def test_robot_nearness_resets_whole_sponge_stability(metadata):
    model, objects, frame = evidence(metadata)
    for tick in range(850):
        frame['baseline'] = tick == 0
        _, result = model.sample(frame, objects)
    assert result['quiet']
    frame['elastic_robot_geometry']['fully_separated'] = False
    _, result = model.sample(frame, objects)
    assert not result['quiet'] and result['boundary_stability']['released_quiet_native_s'] == 0


def test_elastic_public_rules_are_explicit_and_keep_old_thresholds(metadata):
    result = rules.public_rules('surface_wiping', metadata=metadata)
    assert result['schema'] == 'kitchen_rules.v6'
    assert '/World/' not in str(result) and 'vertex_member_indices' not in str(result)
    declared = deepcopy(result['referee']['thresholds'])
    assert declared.pop('minimum_robot_elastic_clearance_m') == .005
    assert declared == rules.thresholds('surface_wiping')
    with pytest.raises(ValueError, match='only for surface wiping'):
        rules.public_rules('table_clearing', metadata=metadata)


def test_full_observer_accepts_composite_and_checks_union_containment(metadata):
    from scenes.workrooms.kitchen.composition import bind_episode_task
    from scenes.workrooms.kitchen.evaluation import NativeTaskObserver
    task = bind_episode_task({'id': 'surface_wiping', 'schema_version': 'kitchen-manipulation-task-v1',
                              'asset_metadata': metadata}, 0, rules)
    observer = NativeTaskObserver(task)
    _, objects, frame = evidence(metadata)
    for obj in objects.values():
        obj.update(linear_velocity_m_s=[0., 0., 0.], angular_velocity_rad_s=[0., 0., 0.])
    frame.update(objects=objects, native_step=0, time_s=0., native_time_s=0.,
                 physics_dt_s=.0005, contacts=[])
    observer.on_frame(frame)
    assert observer.on_observe()['errors'] == []
    points = observer._points('sponge', objects)
    assert len(points) > len(metadata['objects'][0]['collision_support_points_m'])
    assert np.ptp(np.asarray(points)[:, 2]) > .04
