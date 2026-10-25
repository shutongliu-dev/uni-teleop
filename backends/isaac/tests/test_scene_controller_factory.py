"""Real-scene CPU controller composition; no native or model qualification."""
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from bench.control.bimanual import BimanualFingertipPolicy
from bench.control.factory import configured_policy, public_robot_geometry, robot_configuration
from bench.control.fingertip import ClassicalFingertipPolicy, ExecutorLimits
from bench.geometry import quaternion_matrix, transform
from scenes.workrooms import registry
from scenes.workrooms.catalog import list_tasks, load_task


def direct_policy(robot):
    """Independent old call shape: preserve explicitly selected scene parameters."""
    kwargs = {'hand_id': robot['hand_id'], 'assembly_version': robot['assembly_version']}
    if robot['hand_id'] == 'dual_fr3_sharpa_w01':
        cls = BimanualFingertipPolicy
        kwargs['arm_base_poses'] = robot['arm_base_poses']
    else:
        cls = ClassicalFingertipPolicy
    return cls(robot['assets_root'], robot['base_position_m'],
               robot.get('base_quaternion_wxyz', [1., 0., 0., 0.]), **kwargs)


@pytest.mark.parametrize('family', registry.FAMILIES)
def test_six_diagnostic_room_scenes_use_the_same_pinned_robot_factory(family):
    scene = registry.scene(family, diagnostic=False)
    before = deepcopy(scene)
    config = robot_configuration(scene)
    policy = configured_policy(config)
    geometry = public_robot_geometry(config)
    old = direct_policy(scene['robot'])
    assert policy.descriptor == old.descriptor
    assert policy.provenance == old.provenance
    assert config['assembly_version'] == 'isaacsim_6_1_v2'
    assert len(geometry.names) == 29
    np.testing.assert_allclose(geometry.arm_base,
        transform(scene['robot']['base_position_m'], quaternion_matrix(scene['robot'].get('base_quaternion_wxyz', [1., 0., 0., 0.]))), atol=1e-12)
    q = {joint.name: (joint.lower + joint.upper) / 2 for joint in geometry.joints}
    assert geometry.public_finger_geometry(q) == policy.geometry.public_finger_geometry(q)
    assert scene == before


@pytest.fixture(scope='module', params=('tic_tac_toe_3x3', 'connect_four_5x5', 'hex7x7'))
def card_task(request):
    return load_task('card_room', request.param)


def test_three_frozen_card_tasks_keep_controller_descriptors_and_ready_fk(card_task):
    from scenes.board_games.tabletop import geometry_for
    from scenes.workrooms.card_room.robot_layout import ready_configuration

    before = deepcopy(card_task)
    scene, initial = card_task['scene'], card_task['initial_state']
    config = robot_configuration(scene)
    policy, old = configured_policy(config), direct_policy(scene['robot'])
    geometry = public_robot_geometry(config)
    assert len(geometry.names) == 58
    assert policy.descriptor == old.descriptor
    assert policy.provenance == old.provenance
    q = initial['joint_positions_rad']
    assert geometry.public_hand_geometries(q) == old.geometry.public_hand_geometries(q)
    for side in ('left', 'right'):
        np.testing.assert_allclose(geometry.sides[side].arm_base, old.geometry.sides[side].arm_base, atol=0, rtol=0)
    regenerated, _, _ = ready_configuration(scene['robot'], geometry_for(card_task['task']['task_id']))
    assert set(regenerated['joint_positions_rad']) == set(q)
    np.testing.assert_allclose([regenerated['joint_positions_rad'][name] for name in q], list(q.values()), atol=1e-12, rtol=0)
    assert card_task == before


def test_mechanical_setback_layout_keeps_its_ready_pose_and_single_arm_configuration():
    from scenes.workrooms.mechanical_assembly import robot_layout

    robot = robot_layout.robot_configuration()
    before = deepcopy(robot)
    initial, _ = robot_layout.ready_configuration(robot)
    configuration = robot_configuration({'robot': robot})
    policy = configured_policy(configuration)
    geometry = robot_layout.geometry_model(robot)
    old = direct_policy(robot)
    assert policy.descriptor == old.descriptor
    assert policy.provenance == old.provenance
    assert len(geometry.names) == 29
    assert configuration['base_position_m'] == [-.23, 0., .78]
    assert configuration['assembly_version'] == 'isaacsim_6_1_v2'
    q = initial['joint_positions_rad']
    assert geometry.public_finger_geometry(q) == old.geometry.public_finger_geometry(q)
    np.testing.assert_allclose(geometry.public_geometry(q)['wrist']['position_m'], [.10, 0., 1.40], atol=1e-4)
    assert robot == before


@pytest.mark.parametrize('bimanual', [False, True])
def test_changed_scene_bases_rotate_fk_and_repin_new_executors_without_mutating_old_ones(bimanual):
    if bimanual:
        scene = load_task('card_room', 'tic_tac_toe_3x3')['scene']
    else:
        scene = registry.scene('study', diagnostic=False)
    configuration = robot_configuration(scene)
    first = configured_policy(configuration)
    captured = deepcopy(first.robot_configuration)
    previous = first.geometry.sides['left'] if bimanual else first.geometry
    old_base = previous.arm_base.copy()
    q = {joint.name: (joint.lower + joint.upper) / 2 for joint in previous.joints}
    old_point = np.asarray(previous.public_geometry(q)['wrist']['position_m'])
    changed = configuration['arm_base_poses']['left'] if bimanual else configuration
    position_key = 'position_m' if bimanual else 'base_position_m'
    rotation_key = 'quaternion_wxyz' if bimanual else 'base_quaternion_wxyz'
    changed[position_key][0] += .12
    changed[rotation_key] = [np.sqrt(.5), 0., 0., np.sqrt(.5)]
    second = configured_policy(configuration)
    current = second.geometry.sides['left'] if bimanual else second.geometry
    delta = current.arm_base @ np.linalg.inv(old_base)
    expected = delta[:3, :3] @ old_point + delta[:3, 3]
    np.testing.assert_allclose(current.public_geometry(q)['wrist']['position_m'], expected, atol=1e-12)
    assert first.descriptor.implementation_sha256 != second.descriptor.implementation_sha256
    assert first.robot_configuration == captured
    np.testing.assert_array_equal(previous.arm_base, old_base)
    if bimanual:
        np.testing.assert_array_equal(first.geometry.sides['right'].arm_base, second.geometry.sides['right'].arm_base)
    repeated = configured_policy(captured)
    assert repeated is not first and repeated.geometry is not first.geometry
    assert repeated.descriptor == first.descriptor


@pytest.mark.parametrize('invalid', ['unknown_hand', 'unknown_assembly', 'missing_side', 'nonunit_side_rotation', 'single_arm_override'])
def test_factory_preserves_strict_hand_assembly_and_bimanual_layout_validation(invalid):
    scene = deepcopy(load_task('card_room', 'tic_tac_toe_3x3')['scene'])
    robot = scene['robot']
    if invalid == 'unknown_hand':
        robot['hand_id'] = 'unknown_robot'
    elif invalid == 'unknown_assembly':
        robot['assembly_version'] = 'unregistered_assembly'
    elif invalid == 'missing_side':
        del robot['arm_base_poses']['left']
    elif invalid == 'nonunit_side_rotation':
        robot['arm_base_poses']['left']['quaternion_wxyz'] = [2., 0., 0., 0.]
    else:
        robot['hand_id'] = 'sharpa_w01_right'
    with pytest.raises(ValueError):
        configured_policy(robot_configuration(scene))


def test_factory_source_is_pinned_by_service_and_complete_snapshot_inventories():
    from bench.runtime.service import frontend_sources
    from scripts.maintenance.freeze_project_sources import inventory, suite_inventory

    relative = 'bench/control/factory.py'
    sources = inventory(Path(__file__).resolve().parents[1], ())
    assert frontend_sources()[relative] == sources[relative]
    assert suite_inventory(sources)[relative] == sources[relative]


@pytest.mark.parametrize('family,task_id', [
    (task['family'], task['task_id']) for task in list_tasks()
])
def test_all_current_task_releases_bind_reset_and_cameras_to_shared_factory(family, task_id):
    """Exercise executable task releases, beyond the six diagnostic room layouts."""
    from bench.isaacsim.release import digest
    from bench.simulation.isaacsim_backend.cameras import resolve_cameras

    loaded = load_task(family, task_id)
    scene, initial = loaded['scene'], loaded['initial_state']
    configuration = robot_configuration(scene)
    policy = configured_policy(configuration)
    geometry = policy.geometry
    positions = initial['joint_positions_rad']
    assert set(positions) == set(geometry.names)
    assert geometry.robot_id == scene['robot']['hand_id']
    assert policy.descriptor.supported_robot_ids == [geometry.robot_id]
    assert policy.descriptor.control_period_s == .02
    assert policy.limits == ExecutorLimits()
    sides = geometry.sides if hasattr(geometry, 'sides') else {'single': geometry}
    assert len(geometry.names) == 29 * len(sides)
    all_links = set()
    for name, side in sides.items():
        own = {joint: positions[joint] for joint in side.names}
        side.checked_joints(own, enforce_limits=True)
        all_links.update(side.arm.forward(own))
        all_links.update(side.hand.forward(own))
        if hasattr(policy, 'sides'):
            assert policy.sides[name].limits == ExecutorLimits()
    cameras = resolve_cameras(scene)
    wrist_cameras = [row for row in cameras.values() if row['kind'] == 'link']
    assert len(cameras) == 3 + len(sides)
    assert len(wrist_cameras) == len(sides)
    for camera in wrist_cameras:
        assert camera['mount_link'] in all_links
    # Building an executor cannot mutate its caller's scene or the frozen release.
    assert policy.robot_configuration == configuration
    reference = loaded['definition']['source_release']
    assert digest(Path(reference['path'])/'release_manifest.json') == reference['manifest_sha256']
