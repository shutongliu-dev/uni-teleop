"""Original-asset CPU kitchen placement and obstruction checks, without physics."""
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from bench.paths import DATA_ROOT
from scenes.workrooms.kitchen import robot_layout as kitchen
from scenes.workrooms.mechanical_assembly import robot_layout as source


@pytest.fixture(scope='module')
def placement():
    if not Path(f'{DATA_ROOT}/assets/robots/franka/fr3/fr3_no_hand.absolute.urdf').is_file():
        pytest.skip('Pinned original robot assets unavailable')
    robot = source.robot_configuration()
    initial, _ = source.ready_configuration(robot)
    moved = kitchen.robot_configuration(robot)
    ready, design = kitchen.ready_configuration(moved, initial)
    return robot, initial, moved, ready, design


def test_original_joint_targets_and_robot_configuration_are_preserved(placement):
    robot, initial, moved, ready, design = placement
    assert initial == ready
    assert {key: value for key, value in robot.items() if not key.startswith('base_')} == {
        key: value for key, value in moved.items() if not key.startswith('base_')}
    assert moved['base_position_m'] == [.65, 0., .78]
    assert design['minimum_arm_joint_margin_rad'] > .30
    assert design['source_joint_positions_preserved'] is True
    assert design['native_qualified'] is False
    assert design['manipulation_qualified'] is False
    assert np.asarray(design['wrist_fk_matrix'])[:3, 3] == pytest.approx([.65, .33, 1.4], abs=1.e-4)


def test_all_original_link_frames_are_rigidly_relocated_without_scaling(placement):
    robot, initial, moved, ready, _ = placement
    before = deepcopy((robot, initial, moved, ready))
    original = source.link_world_transforms(robot, initial)
    actual = kitchen.link_world_transforms(moved, ready)
    transform = kitchen.placement_transform()
    assert original.keys() == actual.keys()
    assert len(actual) >= 43
    assert np.linalg.det(transform[:3, :3]) == pytest.approx(1.)
    for name in original:
        assert actual[name] == pytest.approx(transform @ original[name], abs=1.e-9), name
    assert (robot, initial, moved, ready) == before


def test_stand_continuity_and_clearance_between_counter_and_bar():
    stand = kitchen.stand_geometry()
    boxes = stand['boxes']
    low = [np.asarray(box['position_m']) - np.asarray(box['size_m']) / 2 for box in boxes]
    high = [np.asarray(box['position_m']) + np.asarray(box['size_m']) / 2 for box in boxes]
    assert low[0] == pytest.approx([.45, -.18, 0.])
    assert high[0] == pytest.approx([.85, .18, .05])
    assert [item[2] for item in low[1:]] == pytest.approx([item[2] for item in high[:-1]])
    assert high[-1][2] == pytest.approx(.78)
    assert stand['foot_to_preparation_edge_m'] == pytest.approx(.39)
    assert stand['foot_to_bar_edge_m'] == pytest.approx(.37)


@pytest.fixture(scope='module')
def preview(tmp_path_factory, placement):
    pytest.importorskip('pxr.Usd')
    from pxr import Usd, UsdGeom, UsdPhysics
    from bench.simulation.assets.workrooms.builder import Builder
    from scenes.workrooms.kitchen.preview import add_robot_preview, audit_environment
    stage = Usd.Stage.CreateInMemory()
    builder = Builder(stage, f'{DATA_ROOT}/assets', tmp_path_factory.mktemp('kitchen_preview'))
    builder.box('room_floor', (.5, -.525, -.06), (5.2, 3.95, .12), 'floor')
    builder.box('worktop', (.45, -.95, .8775), (2.8, .8, .045), 'oak')
    builder.box('kitchen_counter_middle', (.65, .94, .88), (1.0, .74, .04), 'white')
    builder.box('kitchen_cooktop', (1.65, .94, .905), (.5, .5, .01), 'black')
    scenery = UsdGeom.Xform.Define(stage, '/World/Geometry/disabled_scenery_actor').GetPrim()
    UsdPhysics.RigidBodyAPI.Apply(scenery).CreateRigidBodyEnabledAttr(False)
    UsdGeom.Xform.Define(stage, '/World/Robot')
    environment = {'family': 'kitchen', 'layout': {'room': {'bounds': [-2.1, 3.1, -2.5, 1.45]}},
        'support_colliders': {'floor': ['/World/Geometry/room_floor/Collider'],
            'bar': '/World/Geometry/worktop/Collider',
            'preparation': '/World/Geometry/kitchen_counter_middle/Collider',
            'cooking': '/World/Geometry/kitchen_cooktop/Collider'}}
    audit = audit_environment(stage, environment)
    return stage, builder, environment, audit, add_robot_preview(stage, builder, environment=environment)


def test_static_preview_checks_all_original_bodies_and_actual_geometry(preview):
    from pxr import Usd, UsdPhysics
    stage, _, _, parent, composition = preview
    assert parent['status'] == 'passed'
    assert composition['source_rigid_body_count'] == 43
    assert composition['posed_link_count'] >= 43
    assert composition['physics_steps'] == 0
    audit = composition['clearance_audit']
    assert audit['status'] == 'passed'
    assert audit['foot_to_preparation_edge_m'] == pytest.approx(.39, abs=1.e-7)
    assert audit['foot_to_bar_edge_m'] == pytest.approx(.37, abs=1.e-7)
    assert audit['robot_nearest_environment_shape']['gap_m'] > .005
    assert audit['stand_nearest_environment_shape']['gap_m'] > .005
    assert audit['robot_shape_count'] >= 43
    assert not composition['native_contact_qualified']
    for prim in Usd.PrimRange(stage.GetPrimAtPath('/World/TaskRobot')):
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            assert UsdPhysics.RigidBodyAPI(prim).GetRigidBodyEnabledAttr().Get() is False
    assert not any(prim.IsA(UsdPhysics.Scene) for prim in stage.Traverse())


def test_actual_obstruction_in_robot_corridor_is_rejected(preview):
    from scenes.workrooms.kitchen.preview import clearance_audit
    stage, builder, environment, _, _ = preview
    builder.box('unexpected_obstruction', (.65, 0., .40), (.3, .3, .7), 'white')
    with pytest.raises(ValueError, match='shape clearance'):
        clearance_audit(stage, environment)
