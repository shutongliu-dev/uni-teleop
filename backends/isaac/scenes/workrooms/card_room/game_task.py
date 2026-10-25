"""Prepare a versioned black/white wooden board-game task in the fixed card room; no simulation or model calls."""
from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path

from bench.paths import DATA_ROOT, load_json
from scenes.board_games.tabletop import geometry_for
from scenes.board_games.tabletop.release import prepare_task_release
from bench.isaacsim.release import task_inputs
from scenes.workrooms.registry import PHYSICS_OVERRIDES
from scenes.workrooms.observations import camera_specs
from scenes.workrooms.card_room.robot_layout import ready_configuration, arm_base_poses
from bench.simulation.isaacsim_backend.workroom import SCHEMA, checked, digest, verify_reference

BASE_RELEASE = Path(f'{DATA_ROOT}/datasets/fr3_sharpa_bench/isaacsim/releases/simple_manipulation/v13_pad_compliance_r2/bimanual')
BASE_SHA256 = 'be1799c4c3e95c25d468f1622a2a5929b3b3e5c6fb839acd67403c972b1f74eb'
ASSET_ROOT = Path(f'{DATA_ROOT}/assets/task_parts/board_games')


def default_assets(game_id):
    geometry_for(game_id)
    return ASSET_ROOT/game_id/('v4' if game_id == 'hex7x7' else 'v3')


def verify_game_assets(asset_root, geometry):
    """Reject an empty, modified or wrong-level library before release creation."""
    from bench.simulation.assets.task_parts import load_task_catalog, load_task_asset
    catalog = load_task_catalog(asset_root)
    expected = json.loads(json.dumps(geometry.specification()))
    if catalog.get('task') != geometry.game_id or catalog.get('geometry') != expected:
        raise ValueError('Task asset library geometry differs from the selected game')
    ids = {'board', 'token_black', 'token_white', 'stand_left', 'stand_right', *geometry.reserve_stations()}
    if set(catalog['assets']) != ids:
        raise ValueError('Board game asset library requires board, tokens, stands and four reserve stations')
    for asset_id in sorted(ids):
        load_task_asset(asset_id, asset_root, verify_hashes=True)
    return {'manifest_path': catalog['manifest_path'], 'manifest_sha256': catalog['manifest_sha256'],
            'game_id': geometry.game_id, 'asset_ids': sorted(ids)}


def room_reference(room_root=None, *, usd_sha256=None, description_sha256=None):
    """Use the parent scene, or an explicitly pinned development environment."""
    if room_root is None:
        from scenes.workrooms.card_room.scene import environment_reference
        return environment_reference()
    if not usd_sha256 or not description_sha256:
        raise ValueError('Alternate room requires both expected SHA256 values')
    root = Path(room_root).resolve()
    return {'schema': SCHEMA, 'family': 'card_room', 'version': root.name,
            'usd_path': str(root/'scene.usda'), 'usd_sha256': usd_sha256,
            'source_prim': '/World', 'dependency_manifest_path': str(root/'scene_description.json'),
            'dependency_manifest_sha256': description_sha256}


def make_scene(game_id='tic_tac_toe_3x3', asset_root=None, *, room=None):
    checked(BASE_RELEASE/'release_manifest.json', BASE_SHA256)
    entry, result, _, _ = task_inputs(BASE_RELEASE, '05_bimanual_box_lift')
    result = deepcopy(result)
    g = geometry_for(game_id)
    reference = deepcopy(room) if room is not None else room_reference()
    if reference.get('family') != 'card_room':
        raise ValueError('Board games belong to the card_room scene')
    source_hashes = verify_reference(reference)
    result['isaacsim'].update(deepcopy(PHYSICS_OVERRIDES))
    result['isaacsim']['workroom_lighting'] = {'schema': 'workroom-lighting-v1', 'family': 'card_room', 'profile': 'nominal'}
    for key in ('room', 'scene_variant', 'static_props', 'surfaces', 'robot_pedestal', 'soft_bodies'):
        result.pop(key, None)
    result['description'] = f'Two independent opposing FR3+SHARPA agents play physical {game_id}, level {g.level}, in the fixed heritage card room.'
    result.update(objects=[], fixtures=[], goal_markers=[], cloth={'enabled': False},
                  workbench={'enabled': False}, workcell_details={'enabled': False}, workroom=reference)
    result['physics'] = {'dt_s': .0005, 'control_dt_s': .02, 'gravity_m_s2': [0, 0, -9.81],
                         'backend': 'native_isaacsim_physx_gpu'}
    result['table'] = {'position_m': [.65, 0, .76], 'half_extents_m': [.5, .5, .02], 'friction': .4}
    robot = result['robot']
    robot.update(base_position_m=[0., 0., 0.], base_quaternion_wxyz=[1., 0., 0., 0.])
    robot['arm_base_poses'] = arm_base_poses(g)
    initial, homes, wrist_cameras = ready_configuration(robot, g)
    asset_root = Path(asset_root or default_assets(game_id)).resolve()
    asset_identity = verify_game_assets(asset_root, g)
    for color, count in g.inventory_counts.items():
        result['objects'].extend({'id': f'{color}_{i:02d}', 'asset_id': 'token_'+color,
            'asset_root': str(asset_root), 'position_m': list(g.reserve_position(color, i)),
            'quaternion_wxyz': [1, 0, 0, 0], 'friction': .65} for i in range(count))
    result['fixtures'] = [{'id': 'board', 'asset_id': 'board', 'asset_root': str(asset_root),
                            'position_m': [*g.center_xy_m, g.table_top_z_m], 'friction': .4}] + [
        {'id': 'stand_'+side, 'asset_id': 'stand_'+side, 'asset_root': str(asset_root),
         'position_m': [0, 0, 0], 'friction': .4} for side in ('left', 'right')] + [
        {'id': name, 'asset_id': name, 'asset_root': str(asset_root), 'position_m': [0, 0, 0],
         'friction': .4} for name in g.reserve_stations()]
    cameras = camera_specs(load_json(Path(reference['dependency_manifest_path'])))
    cameras['task'].update(eye=[.65, -.02, 2.30], target=[.65, 0., .79], focal_length_mm=23.)
    wrist = cameras.pop('wrist')
    for side in ('left', 'right'):
        cameras['wrist_'+side] = {**deepcopy(wrist), **wrist_cameras['wrist_'+side],
            'title_zh': ('黑方' if side == 'left' else '白方')+'腕部相机'}
    result['cameras'] = cameras
    result['board_game_task'] = {'schema': 'board_game_native_task.v1', 'geometry': asdict(g), 'initial_state': initial,
        'players': [{'player_id': 'player_0', 'color': 'black', 'robot_id': 'left'},
                    {'player_id': 'player_1', 'color': 'white', 'robot_id': 'right'}],
        'home_design': homes, 'board_fixture_id': 'board',
        'reserve_supports': {f'{color}_{index:02d}': {'fixture_id': g.reserve_fixture_id(color, index),
            'surface_z_m': g.reserve_surface_z_m} for color, count in g.inventory_counts.items() for index in range(count)},
        'scope': 'Simulation-only design prior; physical move requires native contact settlement. No score at preparation.'}
    result['board_game_provenance'] = {'base_release': str(BASE_RELEASE), 'base_manifest_sha256': BASE_SHA256,
        'base_task_id': entry['id'], 'base_physics_profile_sha256': digest(BASE_RELEASE/'physics_profile.json'),
        'source_file_hashes': source_hashes, 'pad_model': 'unchanged v13 force_spring_v1 20000/40',
        'asset_root': str(asset_root), 'asset_identity': asset_identity, 'preparation_status': 'unvalidated',
        'physics_overrides': deepcopy(PHYSICS_OVERRIDES),
        'virtual_wrist_mount': 'Fixed local extrinsics; simulation design assumption, not hardware calibration.'}
    return result, initial


def main(game_id=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--asset-root', type=Path)
    parser.add_argument('--game', choices=('tic_tac_toe_3x3', 'connect_four_5x5', 'hex7x7'), default=game_id or 'tic_tac_toe_3x3')
    parser.add_argument('--room-root', type=Path)
    parser.add_argument('--room-usd-sha256')
    parser.add_argument('--room-description-sha256')
    args = parser.parse_args()
    room_args = (args.room_root, args.room_usd_sha256, args.room_description_sha256)
    if any(value is not None for value in room_args) and not all(room_args):
        parser.error('An alternate room requires its root, USD SHA256 and description SHA256')
    room = room_reference(args.room_root, usd_sha256=args.room_usd_sha256,
                          description_sha256=args.room_description_sha256) if args.room_root else None
    if game_id is not None and args.game != game_id:
        parser.error('This task entrypoint cannot prepare a different game')
    asset_root = args.asset_root or default_assets(args.game)
    scene, initial = make_scene(args.game, asset_root, room=room)
    files = [p for p in asset_root.rglob('*') if p.is_file()]
    files.extend(Path(p) for p in scene['board_game_provenance']['source_file_hashes'])
    result = prepare_task_release(args.output, scene=scene, geometry=geometry_for(args.game), initial_state=initial,
                                 asset_files=tuple(files), protocol={'default': 'rgbd', 'privileged_engineering': 'explicit_separate_mode'})
    print(json.dumps({'output': str(args.output), 'status': result['status'], 'pieces': len(scene['objects'])}))


if __name__ == '__main__':
    main()
