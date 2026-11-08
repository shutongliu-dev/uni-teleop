"""Frozen identity compatibility and same-agent two-arm installation checks."""
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from scenes.workrooms.mechanical_assembly import cooperative_robot_layout as layout
from scenes.workrooms.mechanical_assembly import robot_layout as source


def test_default_reproduces_historical_opposed_identity():
    from scenes.workrooms.living_room.composition import _robots
    robot, initial, ready, _ = _robots('tower_extraction')
    assert layout.robot_inputs() == (robot, initial, ready)
    assert layout.robot_inputs(layout.OPPOSED_LAYOUT_ID) == (robot, initial, ready)


def test_same_side_preserves_robot_parameters_and_every_source_joint():
    original = source.robot_configuration()
    original_copy = deepcopy(original)
    source_initial, _ = source.ready_configuration(original)
    robot, initial, ready = layout.robot_inputs(layout.SAME_SIDE_LAYOUT_ID)
    assert robot['hand_id'] == 'fr3_sharpa_instances'
    assert set(robot['instances']) == {'front', 'rear'}
    assert len(initial['joint_positions_rad']) == 58
    for side, one in layout.robot_configurations().items():
        assert {key: value for key, value in one.items()
                if key not in ('base_position_m', 'base_quaternion_wxyz')} == {
            key: value for key, value in original.items()
            if key not in ('base_position_m', 'base_quaternion_wxyz')}
        for name, value in source_initial['joint_positions_rad'].items():
            target = 'fr3_'+side+'_'+name[4:] if name.startswith('fr3_') else side+'_'+name[6:]
            assert initial['joint_positions_rad'][target] == value
        assert ready[side]['source_joint_positions_preserved'] is True
        np.testing.assert_allclose(ready[side]['wrist_fk_matrix'][0][3], .23, atol=1e-4)
    assert original == original_copy


@pytest.mark.parametrize('layout_id', [layout.SAME_SIDE_LAYOUT_ID, layout.MIRRORED_LAYOUT_ID])
def test_stands_are_separate_continuous_and_reject_unknown_layout(layout_id):
    stands = layout.stand_geometry(layout_id)
    for row in stands.values():
        assert row['foot_to_table_edge_m'] >= .05
        assert row['stand_to_other_stand_foot_gap_m'] >= .19
        boxes = row['boxes']
        bottoms = [box['position_m'][2]-box['size_m'][2]/2 for box in boxes]
        tops = [box['position_m'][2]+box['size_m'][2]/2 for box in boxes]
        np.testing.assert_allclose(bottoms[0], 0., atol=1e-12)
        np.testing.assert_allclose(bottoms[1:], tops[:-1], atol=1e-12)
        np.testing.assert_allclose(tops[-1], .78, atol=1e-12)
    for call in (layout.placements, layout.robot_configurations,
                 layout.robot_inputs, layout.stand_geometry):
        with pytest.raises(ValueError, match='Unknown cooperative robot layout'):
            call('unregistered_layout')


def test_official_left_right_hands_have_mirrored_fk_and_inward_thumbs():
    """Compare source FK rather than treating namespace names as chirality."""
    from bench.control.factory import public_robot_geometry, robot_configuration
    from bench.geometry import quaternion_matrix
    from bench.simulation.isaacsim_backend.robot_layout import world_base_poses

    original = source.robot_configuration()
    source_initial, _ = source.ready_configuration(original)
    robot, initial, ready = layout.robot_inputs(layout.MIRRORED_LAYOUT_ID)
    assert robot['hand_id'] == 'dual_fr3_sharpa_w01'
    assert robot['assembly_version'] == 'isaacsim_6_1_v2'
    assert 'instances' not in robot and 'arm_home_rad' not in robot
    for key in ('arm_kp', 'arm_kd', 'hand_kp', 'hand_kd',
                'mount_position_m', 'mount_quaternion_wxyz', 'assets_root'):
        assert robot[key] == original[key]
    assert world_base_poses(robot) == {
        'left': {'position_m': [-.1, .3, .78], 'quaternion_wxyz': [1., 0., 0., 0.]},
        'right': {'position_m': [-.1, -.3, .78], 'quaternion_wxyz': [1., 0., 0., 0.]}}
    geometry = public_robot_geometry(robot_configuration({'robot': robot}))
    assert geometry.frame_ids == ['left_hand_flange', 'right_hand_flange']
    assert len(geometry.names) == len(initial['joint_positions_rad']) == 58
    assert set(geometry.names) == set(initial['joint_positions_rad'])
    points = {}
    for side, model in geometry.sides.items():
        q = {name: initial['joint_positions_rad'][name] for name in model.names}
        for name, value in source_initial['joint_positions_rad'].items():
            if name.startswith('fr3_'):
                expected = value*(-1 if side == 'left' and int(name[-1]) in (1, 3, 5, 7) else 1)
                target = 'fr3_'+side+'_'+name[4:]
            else:
                expected, target = value, side+'_'+name[6:]
            assert q[target] == expected
        assert model.hand_posture_prior.side == side
        assert any(Path(path).name == f'{side}_sharpa_wave_with_flange.urdf'
                   for path in model.asset_sha256)
        poses = model.arm.forward(q, model.arm_base)
        poses.update(model.hand.forward(q, poses[model.arm_tip] @ model.wrist_offset))
        points[side] = {name: poses[side+'_'+name][:3, 3] for name in (
            'hand_flange', 'hand_C_MC', 'thumb_fingertip', 'index_fingertip',
            'middle_fingertip', 'ring_fingertip', 'pinky_fingertip')}
        thumb_offset = points[side]['thumb_fingertip'][1]-points[side]['hand_C_MC'][1]
        assert thumb_offset*(-1 if side == 'left' else 1) > .09
        assert ready[side]['source_hand_chirality'] == side
        assert ready[side]['source_finger_joint_values_preserved'] is True
    for name in points['left']:
        np.testing.assert_allclose(points['left'][name]*[1, -1, 1], points['right'][name], atol=2e-6, rtol=0.)
    hands = geometry.public_hand_geometries(initial['joint_positions_rad'])
    mirror = np.diag([1., -1., 1.])
    np.testing.assert_allclose(mirror @ quaternion_matrix(hands['right']['wrist']['quaternion_wxyz']) @ mirror,
                               quaternion_matrix(hands['left']['wrist']['quaternion_wxyz']), atol=1e-9, rtol=0.)
    with pytest.raises(ValueError, match='one bimanual configuration'):
        layout.robot_configurations(layout.MIRRORED_LAYOUT_ID)
