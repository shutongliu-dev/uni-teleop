"""Laundry installations preserve official hands, source joints and controls."""
import numpy as np
import pytest

from scenes.workrooms.laundry_room import robot_layout as layout


def test_relocation_preserves_official_task10_robot_and_all_58_joints():
    from scenes.workrooms.mechanical_assembly.cooperative_robot_layout import (
        MIRRORED_LAYOUT_ID, robot_inputs as original)
    from bench.simulation.isaacsim_backend.robot_layout import world_base_poses
    from bench.control.factory import public_robot_geometry, robot_configuration

    source, source_initial, _ = original(MIRRORED_LAYOUT_ID)
    robot, initial, ready = layout.robot_inputs()
    assert initial == source_initial
    assert len(initial['joint_positions_rad']) == 58
    changed = {key for key in source if source[key] != robot[key]}
    assert changed == {'base_position_m', 'arm_base_poses'}
    assert robot['hand_id'] == 'dual_fr3_sharpa_w01'
    assert robot['assembly_version'] == 'isaacsim_6_1_v2'
    worlds = world_base_poses(robot)
    model = public_robot_geometry(robot_configuration({'robot': robot}))
    assert set(model.sides) == {'left', 'right'}
    for side in layout.SIDES:
        expected = layout.placements()[side]
        np.testing.assert_allclose(worlds[side]['position_m'], expected['base_position_m'], atol=1e-12)
        np.testing.assert_allclose(worlds[side]['quaternion_wxyz'], expected['base_quaternion_wxyz'], atol=1e-12)
        assert model.sides[side].hand_posture_prior.side == side
        assert ready[side]['native_qualified'] is False


def test_near_cloth_mapping_uses_actual_original_middle_fingertip():
    from bench.control.factory import public_robot_geometry, robot_configuration
    robot, initial, _ = layout.robot_inputs()
    geometry = public_robot_geometry(robot_configuration({'robot': robot}))
    for side, model in geometry.sides.items():
        q = {name: initial['joint_positions_rad'][name] for name in model.names}
        tip = model.public_geometry(q)['fingertip_positions_m']['middle']
        point = [.44, layout.placements()[side]['assigned_cloth_edge_y_m'], .903]
        wrist = layout.hand_probe_transform(side, point)
        actual = (wrist @ [*tip, 1.])[:3]
        np.testing.assert_allclose(actual, np.asarray(point)+[0., 0., .015], atol=1e-10)
    for side, point, clearance in [('front', [0, 0, 0], .01), ('left', [0, 0], .01),
                                   ('right', [0, 0, 0], float('nan'))]:
        with pytest.raises(ValueError):
            layout.hand_probe_transform(side, point, fingertip_clearance_m=clearance)


def test_stands_reach_robot_mount_from_room_floor_without_table_overlap():
    for side, row in layout.stand_geometry().items():
        boxes = row['boxes']
        bottoms = [box['position_m'][2]-box['size_m'][2]/2 for box in boxes]
        tops = [box['position_m'][2]+box['size_m'][2]/2 for box in boxes]
        np.testing.assert_allclose(bottoms[0], 0., atol=1e-12)
        np.testing.assert_allclose(bottoms[1:], tops[:-1], atol=1e-12)
        np.testing.assert_allclose(tops[-1], layout.placements()[side]['base_position_m'][2], atol=1e-12)
        for box in boxes:
            low = np.array(box['position_m'])-np.array(box['size_m'])/2
            high = np.array(box['position_m'])+np.array(box['size_m'])/2
            assert (low[1] > layout.TABLE_BOUNDS_XY_M[1][1] if side == 'left' else
                    high[1] < layout.TABLE_BOUNDS_XY_M[0][1])
