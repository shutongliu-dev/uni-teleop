"""Native preflight gates with explicit CPU doubles; no physics qualification."""
from dataclasses import asdict
import json
from pathlib import Path
import shutil

import pytest

from bench.paths import load_json
from scenes.board_games import tabletop
from scenes.board_games.tabletop import BoardGameMatch, GAME_IDS, geometry_for, task_specification
from scripts.validation.validate_board_game import (ReservePreflightObserver, load_release,
    physical_checks, preflight_camera_ids, table_contact_paths, reserve_support_spec)
from scripts.validation.validate_scenario import digest
from test_tabletop_native_observer import FakeEnv, contact, ROBOT

TABLE = '/World/Room/Geometry/worktop/Collider'
CAMERAS = ['overview', 'task', 'room', 'wrist_left', 'wrist_right']


def reserve_env(game_id):
    match = BoardGameMatch(geometry_for(game_id))
    env = FakeEnv(match)
    env.scene_info['workbench'] = {'kind': 'referenced_workroom', 'supports': {'table': TABLE}}
    for paths in env._contact_filter_paths:
        paths.append(TABLE)
    declarations = {}
    for name in match.inventory:
        color, index = name.rsplit('_', 1)
        fixture = match.geometry.reserve_fixture_id(color, int(index))
        path = '/World/Fixtures/'+fixture
        env.scene_info['fixtures'][fixture] = {'contact_filter_paths': [path]}
        declarations[name] = {'fixture_id': fixture, 'surface_z_m': match.geometry.reserve_surface_z_m}
        for paths in env._contact_filter_paths:
            if path not in paths:
                paths.append(path)
    env.scene_spec = {'board_game_task': {'reserve_supports': declarations}}
    env.next_rows = [contact(name=name, other='/World/Fixtures/'+declarations[name]['fixture_id']) for name in match.inventory]
    initial = {'objects': {name: {'position_m': match.geometry.reserve_position(name.split('_')[0], int(name[-2:]))}
                           for name in match.inventory}}
    return env, match, initial


@pytest.mark.parametrize('game_id', GAME_IDS)
def test_all_inventory_tokens_require_full_native_support_dwell(game_id):
    env, match, initial = reserve_env(game_id)
    observer = ReservePreflightObserver(env, match, initial).attach()
    for _ in range(31):
        env.advance()
    preflight = physical_checks(observer, match, expected_samples=31, phase='ready_pose_hold')
    assert preflight['reserve_support']['passed']
    assert preflight['reserve_support']['token_count'] == match.geometry.size**2
    assert observer.samples == env.acquisitions == 31
    assert not observer.interference
    env.next_rows = [row for row in env.next_rows if row['actor0'] != '/World/Objects/black_00']
    env.advance()
    report = observer.preflight_summary()
    assert not report['passed']
    assert report['tokens']['black_00']['stable_supported_dwell_s'] == 0.
    assert report['tokens']['white_00']['passed']
    post = physical_checks(observer, match, expected_samples=32, phase='missing_support')
    assert not post['reserve_support']['passed'] and post['native_observer']['passed']
    assert preflight['reserve_support']['passed']  # Prior result is an independent snapshot.
    observer.close()


@pytest.mark.parametrize('game_id', GAME_IDS)
def test_robot_token_interference_is_not_qualified_as_reserve_stability(game_id):
    env, match, initial = reserve_env(game_id)
    observer = ReservePreflightObserver(env, match, initial).attach()
    for _ in range(31):
        env.advance()
    env.next_rows.append(contact('black_00', ROBOT))
    env.advance()
    report = physical_checks(observer, match, expected_samples=32, phase='robot_contact')
    assert not report['reserve_support']['passed']
    assert not report['robot_interference']['passed']
    assert report['robot_interference']['pairs'][0]['kind'] == 'robot_token'
    assert report['robot_interference']['pairs'][0]['positive_substeps'] == 1
    observer.close()


@pytest.mark.parametrize('game_id', GAME_IDS)
def test_missing_separation_cannot_pass_support_gate(game_id):
    env, match, initial = reserve_env(game_id)
    observer = ReservePreflightObserver(env, match, initial).attach()
    del env.next_rows[0]['min_separation_m']
    for _ in range(31):
        env.advance()
    report = physical_checks(observer, match, expected_samples=31, phase='missing_separation')
    assert not report['reserve_support']['passed']
    assert not report['native_observer']['passed']
    assert report['native_observer']['unsupported_samples'] == 31
    observer.close()


def test_original_table_actor_mapping_and_missing_identity():
    actor = '/World/Room/Geometry/worktop'
    assert table_contact_paths({'workbench': {'supports': {'table': TABLE}},
                                'collider_actor_paths': {TABLE: actor}}) == (actor,)
    with pytest.raises(ValueError, match='original table'):
        table_contact_paths({'workbench': {'kind': 'referenced_workroom'}})


def fake_release(tmp_path, game_id='tic_tac_toe_3x3', *, mutate_scene=None, mutate_task=None,
                 mutate_manifest=None):
    geometry = geometry_for(game_id)
    match = BoardGameMatch(geometry)
    initial = {'joint_positions_rad': {'q': 0.}}
    scene = {'objects': [{'id': name, 'asset_id': 'token_'+name.split('_')[0],
                          'position_m': list(geometry.reserve_position(name.split('_')[0], int(name[-2:]))) }
                         for name in sorted(match.inventory)],
             'fixtures': [{'id': 'board', 'asset_id': 'board',
                           'position_m': [*geometry.center_xy_m, geometry.table_top_z_m]},
                          {'id': 'stand_left', 'asset_id': 'stand_left', 'position_m': [0, 0, 0]},
                          {'id': 'stand_right', 'asset_id': 'stand_right', 'position_m': [0, 0, 0]}],
             'robot': {'hand_id': 'dual_fr3_sharpa_w01'},
             'board_game_task': {'schema': 'board_game_native_task.v1', 'initial_state': initial,
                 'geometry': asdict(geometry), 'board_fixture_id': 'board',
                 'players': [{'player_id': 'player_0', 'color': 'black', 'robot_id': 'left'},
                             {'player_id': 'player_1', 'color': 'white', 'robot_id': 'right'}]}}
    scene['fixtures'].extend({'id': name, 'asset_id': name, 'position_m': [0, 0, 0]} for name in geometry.reserve_stations())
    scene['board_game_task']['reserve_supports'] = {name: {'fixture_id': geometry.reserve_fixture_id(name.split('_')[0], int(name[-2:])),
        'surface_z_m': geometry.reserve_surface_z_m} for name in match.inventory}
    task = task_specification(geometry)
    if mutate_scene:
        mutate_scene(scene)
    if mutate_task:
        mutate_task(task)
    for name, data in [('scene.json', scene), ('task.json', task), ('initial_state.json', initial),
                       ('rules.json', task['public_rules'])]:
        (tmp_path/name).write_text(json.dumps(data))
    frozen = tmp_path/'evaluator/tabletop'
    frozen.mkdir(parents=True)
    for path in Path(tabletop.__file__).parent.glob('*.py'):
        shutil.copy2(path, frozen/path.name)
    manifest = {'schema': 'robot_board_game_task_release.v1', 'task_id': game_id,
                'public_rules': 'rules.json',
                'frozen_evaluator': 'evaluator/tabletop', 'asset_files_sha256': {},
                'files_sha256': {str(path.relative_to(tmp_path)): digest(path)
                                 for path in tmp_path.rglob('*') if path.is_file()}}
    if mutate_manifest:
        mutate_manifest(manifest)
    (tmp_path/'release_manifest.json').write_text(json.dumps(manifest))
    return manifest


@pytest.mark.parametrize('game_id', GAME_IDS)
def test_loader_accepts_consistent_frozen_game_identity(tmp_path, game_id):
    fake_release(tmp_path, game_id)
    scene, task, initial, geometry, thresholds, source = load_release(tmp_path)
    assert len(scene['objects']) == geometry.size**2
    assert task['task_id'] == game_id
    assert initial == scene['board_game_task']['initial_state']
    assert source['actual_evaluator_verification']['all_import_paths_verified']


def test_native_loader_rejects_rehashed_public_rule_divergence(tmp_path):
    fake_release(tmp_path)
    rules_path = tmp_path/'rules.json'
    rules = load_json(rules_path)
    rules['allow_object_teleportation'] = True
    rules_path.write_text(json.dumps(rules))
    manifest_path = tmp_path/'release_manifest.json'
    manifest = load_json(manifest_path)
    manifest['files_sha256']['rules.json'] = digest(rules_path)
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        load_release(tmp_path)


@pytest.mark.parametrize('target,mutation', [
    ('manifest', lambda data: data.update(schema='robot_hex_task_release.v1')),
    ('manifest', lambda data: data.update(task_id='hex7x7')),
    ('task', lambda data: data.update(schema='robot_hex_task.v1')),
    ('task', lambda data: data.update(task_id='hex7x7')),
    ('task', lambda data: data['geometry'].update(pitch_m=.125)),
    ('task', lambda data: data['rules'].update(turn_order='white_first_then_alternating')),
    ('scene', lambda data: data['board_game_task'].update(schema='hex_native_task.v1')),
    ('scene', lambda data: data['board_game_task']['geometry'].update(board_thickness_m=.02)),
    ('scene', lambda data: data['board_game_task']['players'][0].update(color='white')),
    ('scene', lambda data: data['board_game_task']['players'][1].update(robot_id='left')),
    ('scene', lambda data: data['board_game_task'].update(board_fixture_id='other_board')),
    ('scene', lambda data: data['fixtures'][0].update(id='other_board')),
    ('scene', lambda data: data['fixtures'][0].update(asset_id='stand_left')),
    ('scene', lambda data: data['fixtures'][0].update(position_m=[.9, 0., .78])),
    ('scene', lambda data: data['fixtures'][0].update(quaternion_wxyz=[0., 1., 0., 0.])),
    ('scene', lambda data: data['fixtures'].append(dict(data['fixtures'][0]))),
    ('scene', lambda data: data['robot'].update(hand_id='sharpa_w01_right')),
    ('scene', lambda data: data['objects'][0].update(id='white_99')),
    ('scene', lambda data: data['objects'][0].update(asset_id='token_white')),
])
def test_loader_rejects_internally_inconsistent_but_rehashed_release(tmp_path, target, mutation):
    fake_release(tmp_path, **{'mutate_'+target: mutation})
    with pytest.raises(ValueError):
        load_release(tmp_path)


def test_loader_requires_all_three_hashed_inputs(tmp_path):
    fake_release(tmp_path, mutate_manifest=lambda manifest: manifest['files_sha256'].pop('initial_state.json'))
    with pytest.raises(ValueError, match='explicit initial_state'):
        load_release(tmp_path)


def test_loader_rejects_changed_frozen_evaluator(tmp_path):
    fake_release(tmp_path)
    (tmp_path/'evaluator/tabletop/settlement.py').write_text('# changed\n')
    with pytest.raises(ValueError):
        load_release(tmp_path)


@pytest.mark.parametrize('requested', [None, ['room'], ['task'], ['wrist_left', 'wrist_left']])
def test_camera_subselection_cannot_bypass_any_of_five_required_views(requested):
    selected = preflight_camera_ids(requested, CAMERAS)
    assert len(selected) == len(set(selected)) == 5
    assert set(selected) == set(CAMERAS)


@pytest.mark.parametrize('missing', CAMERAS)
def test_missing_required_view_rejected_before_native_render(missing):
    with pytest.raises(ValueError, match=missing):
        preflight_camera_ids(['room'], [name for name in CAMERAS if name != missing])


def test_wrong_station_contact_does_not_replace_assigned_support():
    env, match, initial = reserve_env('hex7x7')
    observer = ReservePreflightObserver(env, match, initial).attach()
    for row in env.next_rows:
        if row['actor0'].endswith('/black_00'):
            row['actor1'] = '/World/Fixtures/reserve_black_east'
    for _ in range(31):
        env.advance()
    report = observer.preflight_summary()
    assert not report['passed']
    assert not report['tokens']['black_00']['checks']['supported_by_declared_reserve_surface']
    assert report['tokens']['white_00']['passed']
    observer.close()


@pytest.mark.parametrize('fault', ['missing_token', 'wrong_height', 'missing_fixture', 'foreign_actor', 'missing_filter'])
def test_invalid_or_unobserved_reserve_support_is_rejected(fault):
    env, match, initial = reserve_env('tic_tac_toe_3x3')
    support = env.scene_spec['board_game_task']['reserve_supports']
    fixture = support['black_00']['fixture_id']
    if fault == 'missing_token':
        del support['black_00']
    elif fault == 'wrong_height':
        support['black_00']['surface_z_m'] = float('nan')
    elif fault == 'missing_fixture':
        del env.scene_info['fixtures'][fixture]
    elif fault == 'foreign_actor':
        env.scene_info['fixtures'][fixture]['contact_filter_paths'] = [TABLE]
    else:
        for paths in env._contact_filter_paths:
            paths.remove('/World/Fixtures/'+fixture)
    with pytest.raises((ValueError, RuntimeError)):
        ReservePreflightObserver(env, match, initial)


@pytest.mark.parametrize('mutation', [
    lambda s: s['board_game_task']['reserve_supports'].pop('black_00'),
    lambda s: s['board_game_task']['reserve_supports']['black_00'].update(surface_z_m=.78),
    lambda s: s['objects'][0].update(position_m=[.65, -.45, .807]),
    lambda s: next(f for f in s['fixtures'] if f['id']=='reserve_black_west').update(position_m=[0, 0, .01]),
])
def test_release_rejects_rehashed_reserve_station_mismatch(tmp_path, mutation):
    fake_release(tmp_path, mutate_scene=mutation)
    with pytest.raises(ValueError):
        load_release(tmp_path)
