"""Real-asset CPU geometry checks; no claim of native manipulation acceptance."""
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from bench.paths import DATA_ROOT
from scenes.workrooms.mechanical_assembly.robot_layout import (
    geometry_model, link_world_transforms, ready_configuration, robot_configuration,
    stand_geometry,
)
from scripts.diagnostics.probe_mechanical_layout import probe


@pytest.fixture(scope='module')
def report():
    if not Path(f'{DATA_ROOT}/assets/robots/franka/fr3/fr3_no_hand.absolute.urdf').exists():
        pytest.skip('Pinned original robot assets unavailable')
    return probe()


def test_ready_and_all_endpoint_probes_pass_on_original_robot(report):
    assert report['passed']
    assert report['ready']['minimum_arm_joint_margin_rad'] > .30
    assert report['ready']['position_error_m'] < .0001
    assert report['ready']['rotation_error_rad'] < .002
    assert len(report['endpoint_probes']) == 36
    assert all(row['converged'] for row in report['endpoint_probes'])
    assert all(row['minimum_arm_joint_margin_rad'] > .30 for row in report['endpoint_probes'])
    meshes = report['ready_source_mesh_checks']
    assert meshes['source_collision_mesh_count'] >= 35
    assert meshes['minimum_hand_nonadjacent_arm_gap_lower_bound_m'] > .04
    assert meshes['minimum_table_gap_lower_bound_m'] > .025
    assert meshes['minimum_moving_link_stand_gap_lower_bound_m'] > .025
    assert not report['native_runtime_started']
    assert not report['native_qualified'] and not report['manipulation_qualified']


def test_preserves_rigid_v12_robot_parameters(report):
    from bench.isaacsim.release import task_inputs
    from scenes.workrooms.registry import BASE_RELEASE
    _, original, _, _ = task_inputs(BASE_RELEASE, '01_cube_push')
    allowed = {'base_position_m', 'base_quaternion_wxyz', 'arm_home_rad'}
    robot = report['robot']
    for key, value in original['robot'].items():
        if key not in allowed:
            assert robot[key] == value, key
    assert robot['hand_id'] == 'sharpa_w01_right'
    assert robot['assembly_version'] == 'isaacsim_6_1_v2'
    assert robot['base_position_m'] == [-.23, 0., .78]


def test_stand_is_continuous_floor_to_mount_and_set_back(report):
    stand = stand_geometry()
    boxes = stand['boxes']
    bottoms = [p['position_m'][2]-p['size_m'][2]/2 for p in boxes]
    tops = [p['position_m'][2]+p['size_m'][2]/2 for p in boxes]
    assert bottoms[0] == pytest.approx(0.)
    assert bottoms[1:] == pytest.approx(tops[:-1])
    assert tops[-1] == pytest.approx(report['robot']['base_position_m'][2])
    assert all(min(p['size_m']) > 0 for p in boxes)
    assert stand['owner'] == 'task'
    assert stand['base_center_to_table_edge_m'] == pytest.approx(.38)
    assert stand['foot_to_table_edge_m'] == pytest.approx(.20)


def test_fk_export_agrees_with_public_wrist_and_is_pure(report):
    robot, initial = deepcopy(report['robot']), deepcopy(report['initial_state'])
    before = deepcopy((robot, initial))
    poses = link_world_transforms(robot, initial)
    assert (robot, initial) == before
    assert 'fr3_link0' in poses and 'right_hand_flange' in poses
    assert np.linalg.det(poses['right_hand_flange'][:3, :3]) == pytest.approx(1.)
    assert poses['right_hand_flange'][:3, 3] == pytest.approx(
        report['ready']['wrist_fk']['position_m'], abs=1e-12)


def test_rejects_unreviewed_base_or_hand(report):
    robot = deepcopy(report['robot'])
    robot['base_position_m'][0] += .01
    with pytest.raises(ValueError, match='setback base'):
        ready_configuration(robot)
    robot = deepcopy(report['robot'])
    robot['assembly_version'] = 'isaacsim_6_1_v1'
    with pytest.raises(ValueError, match='single-arm v12'):
        geometry_model(robot)
