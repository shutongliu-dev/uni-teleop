import json
from pathlib import Path

import pytest

from bench.paths import load_json
from scenes.workrooms.card_room.game_task import BASE_RELEASE, make_scene, room_reference
from bench.control.bimanual import BimanualGeometry


@pytest.fixture(scope='module', params=['tic_tac_toe_3x3', 'connect_four_5x5', 'hex7x7'])
def prepared(request):
    return make_scene(request.param)


def test_scene_retains_contact_model_and_two_original_articulations(prepared):
    scene, initial = prepared
    assert scene['physics']['dt_s'] == scene['isaacsim']['physics_dt_s']
    original = load_json(BASE_RELEASE/'physics_profile.json')['isaacsim_options']
    for key in ('robot_pad_compliance', 'robot_contact_materials', 'hand_actuator_model',
                'robot_self_collisions', 'physics_dt_s'):
        assert scene['isaacsim'][key] == original[key]
    assert scene['isaacsim']['robot_base_constraint'] == 'native_fixed_base_v1'
    robot = scene['robot']
    geometry = BimanualGeometry(robot['assets_root'], robot['base_position_m'],
        assembly_version=robot['assembly_version'], arm_base_poses=robot['arm_base_poses'])
    assert set(initial['joint_positions_rad']) == set(geometry.names)
    assert len(geometry.names) == 58
    for name, value in initial['joint_positions_rad'].items():
        joint = geometry.by_name[name]
        assert joint.lower-1e-8 <= value <= joint.upper+1e-8


def test_scene_preserves_all_tokens_and_rigid_wrist_mounts(prepared):
    from bench.simulation.isaacsim_backend.cameras import resolve_cameras
    scene, _ = prepared
    assert len(resolve_cameras(scene)) == 5
    from scenes.board_games.tabletop import BoardGameMatch, BoardGameGeometry
    geometry = BoardGameGeometry(**scene['board_game_task']['geometry'])
    assert {row['id'] for row in scene['objects']} == BoardGameMatch(geometry).inventory
    assert len(scene['objects']) == geometry.size**2
    assert {row['id'] for row in scene['fixtures']} == {'board', 'stand_left', 'stand_right', *geometry.reserve_stations()}
    for side in ('left', 'right'):
        camera = scene['cameras']['wrist_'+side]
        assert camera['mount_link'] == side+'_hand_flange'
        assert camera['kind'] == 'link' and camera['aim'] is False
    assert scene['cameras']['task']['eye'][2] > 2.


def test_explicit_ready_state_has_no_token_pose_playback(prepared):
    scene, initial = prepared
    assert scene['board_game_task']['initial_state'] == initial
    assert set(initial) == {'joint_positions_rad'}
    assert len(initial['joint_positions_rad']) == 58
    for side, home in scene['board_game_task']['home_design'].items():
        assert home['wrist_probe_points_world']
        assert scene['cameras']['wrist_'+side]['aim'] is False


def test_game_uses_parent_scene_identity_and_rejects_cross_scene_selection(prepared):
    from scenes.workrooms.card_room.scene import environment_reference
    assert prepared[0]['workroom'] == environment_reference()
    other_room = {**environment_reference(), 'family': 'kitchen'}
    with pytest.raises(ValueError, match='belong to the card_room'):
        make_scene(room=other_room)


def test_alternate_environment_requires_both_content_pins(tmp_path):
    with pytest.raises(ValueError, match='both expected SHA256'):
        room_reference(tmp_path, usd_sha256='0'*64)
