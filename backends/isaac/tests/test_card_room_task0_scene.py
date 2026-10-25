"""Task0 composition and immutable release; CPU checks, not grasp acceptance."""
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from bench.control.factory import public_robot_geometry, robot_configuration
from bench.paths import DATA_ROOT, load_json
from scenes.workrooms.card_room.scene import environment_reference
from scenes.workrooms.card_room.tasks.task0.task import ASSET_ROOT, build_task_inputs


@pytest.fixture(scope='module')
def built():
    if not (ASSET_ROOT/'manifest.json').exists():
        pytest.skip('Shared versioned board assets are not installed')
    return build_task_inputs(seed=0)


def test_single_robot_and_only_three_unmodified_black_pieces(built):
    scene = built['scene']
    assert scene['workroom'] == environment_reference()
    assert scene['robot']['hand_id'] == 'sharpa_w01_right'
    assert scene['robot']['assembly_version'] == 'isaacsim_6_1_v2'
    assert scene['robot']['base_position_m'] == [.65, -.6, .9]
    assert 'arm_base_poses' not in scene['robot']
    assert 'board_game_task' not in scene
    assert 'robot_pad_compliance' not in scene['isaacsim']
    assert scene['isaacsim']['robot_base_constraint'] == 'native_fixed_base_v1'
    assert [row['id'] for row in scene['objects']] == ['black_00', 'black_01', 'black_02']
    assert {row['id'] for row in scene['fixtures']} == {'board', 'stand_left', 'reserve_black_west'}
    for row in scene['objects']:
        assert row['asset_id'] == 'token_black' and Path(row['asset_root']) == ASSET_ROOT
        assert row['position_m'][0] == pytest.approx(.14)
        assert row['position_m'][2] == pytest.approx(.807)
        assert 'scale' not in row
    assert {row['position_m'][1] for row in scene['objects']} == {-.38, -.52, -.66}


def test_full_single_arm_reset_and_fixed_four_camera_configuration(built):
    scene, initial = built['scene'], built['initial_state']
    model = public_robot_geometry(robot_configuration(scene))
    assert set(initial) == {'joint_positions_rad'}
    assert len(initial['joint_positions_rad']) == 29
    assert set(initial['joint_positions_rad']) == set(model.names)
    model.checked_joints(initial['joint_positions_rad'], enforce_limits=True)
    wrist = model.public_geometry(initial['joint_positions_rad'])['wrist']
    assert np.linalg.norm(np.array(wrist['position_m'])-[.65, -.4, 1.42]) < .0001
    assert set(scene['cameras']) == {'overview', 'task', 'room', 'wrist'}
    camera = scene['cameras']['wrist']
    assert camera['mount_link'] == 'right_hand_flange' and camera['aim'] is False
    assert scene['card_room_placement']['reserve_supports'] == {
        name: {'fixture_id': 'reserve_black_west', 'surface_z_m': .8}
        for name in ('black_00', 'black_01', 'black_02')}
    budgets = built['task']['recommended_budgets']
    assert budgets['sim_time_s'] <= 120 and budgets['wall_time_s'] <= 1800 and budgets['max_actions'] <= 6000


def test_episode_seed_changes_goals_without_changing_scene_or_reset(built):
    from scenes.workrooms.card_room.tasks.task0.rules import bind_episode_targets
    original = deepcopy(built)
    targets = set()
    for seed in range(12):
        task = bind_episode_targets(built['task'], seed)
        cells = task['evaluation']['target_cells']
        assert len(cells) == 3 and len({tuple(c) for c in cells}) == 3
        assert all(0 <= q < 3 and 0 <= r < 3 for q, r in cells)
        assert task['public_goal']['target_cells'] == cells
        assert task['initial_state'] == built['initial_state']
        targets.add(tuple(map(tuple, cells)))
        assert bind_episode_targets(built['task'], seed) == task
    assert len(targets) > 1
    assert built == original


def test_release_creation_refuses_existing_path_or_project_storage(tmp_path):
    from scenes.workrooms.card_room.tasks.task0.release import prepare_task_release
    with pytest.raises(ValueError, match='shared datasets'):
        prepare_task_release(tmp_path/'release')


def test_release_freezes_actual_controller_urdfs_and_source_licenses(built):
    import json
    from scenes.workrooms.card_room.tasks.task0.release import _asset_inventory
    robot = built['scene']['robot']
    path = Path(robot['assets_root'])/'robots/assemblies/fr3_sharpa'/robot['assembly_version']/'SOURCE.json'
    source = load_json(path)
    inventory = _asset_inventory(built['scene'])
    for name, digest in {**source['sources'], **source['mesh_inputs'], **source['license_sha256'],
                         source['urdf_path']: source['urdf_sha256']}.items():
        assert inventory[name]['sha256'] == digest


def test_explicit_collision_candidate_preserves_task_reset_controller_and_rules(built):
    from scenes.workrooms.card_room.tasks.task0.release import _asset_inventory
    root = Path(f'{DATA_ROOT}/assets/task_parts/board_games/tic_tac_toe_3x3/task0_union_sdf512_candidate03_20261018')
    if not (root/'manifest.json').exists():
        pytest.skip('Versioned candidate assets unavailable; no native acceptance is inferred')
    selection = {'asset_root': str(root), 'asset_id': 'token_black_union_sdf512',
                 'manifest_sha256': 'cb2248884ac24e9483024e84ef4119aa838599bb8fc0c2527c32bd5ecf5b5f95'}
    candidate = build_task_inputs(seed=0, token_collision_asset=selection)
    assert candidate['initial_state'] == built['initial_state']
    assert candidate['task'] == built['task']
    scene = deepcopy(candidate['scene'])
    provenance = scene['card_room_placement_provenance'].pop('token_collision_asset')
    assert provenance['selection'] == selection
    for before, after in zip(built['scene']['objects'], scene['objects']):
        assert after['id'] == before['id'] and after['position_m'] == before['position_m']
        assert after['quaternion_wxyz'] == before['quaternion_wxyz']
        assert after['usd_physics_policy'] == 'preserve_authored_v1'
    scene['objects'] = deepcopy(built['scene']['objects'])
    assert scene == built['scene']
    inventory = _asset_inventory(candidate['scene'])
    assert inventory[str(root/'manifest.json')]['sha256'] == selection['manifest_sha256']
    assert str(root/'token_black.usda') in inventory


def test_release_inventory_rejects_manifest_that_omits_actual_usd_hash(built, tmp_path):
    import json
    from scenes.workrooms.card_room.tasks.task0.release import _asset_inventory
    root = Path(f'{DATA_ROOT}/assets/task_parts/board_games/tic_tac_toe_3x3/task0_union_sdf512_candidate03_20261018')
    if not (root/'manifest.json').exists():
        pytest.skip('Versioned candidate assets unavailable')
    catalog = load_json(root/'manifest.json')
    asset_id = 'token_black_union_sdf512'
    asset = catalog['assets'][asset_id]
    del asset['source_dependencies_sha256'][asset['usd_path']]
    (tmp_path/'manifest.json').write_text(json.dumps(catalog))
    scene = deepcopy(built['scene'])
    for spec in scene['objects']:
        spec.update(asset_root=str(tmp_path), asset_id=asset_id)
    with pytest.raises(ValueError, match='USD reference.*dependency hash'):
        _asset_inventory(scene)


def test_registered_task0_uses_canonical_frozen_single_arm_release():
    from bench.isaacsim.release import task_inputs, load_evaluator
    from scenes.workrooms.catalog import load_task
    loaded = load_task('card_room', 'task0')
    release = loaded['definition']['source_release']['path']
    _, scene, task, initial = task_inputs(release, 'task0', seed=0)
    assert (scene, task, initial) == (loaded['scene'], loaded['task'], loaded['initial_state'])
    assert loaded['definition']['difficulty'] == 0
    assert loaded['definition']['robot_requirements']['articulations'] == 1
    assert not loaded['definition']['qualification']['manipulation']
    assert load_evaluator(release)(task).inspect_conditions()['success'] is False
