"""Admission/readback boundaries; synthetic evidence is never native qualification."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from bench.paths import DATA_ROOT, load_json
from bench.runtime.card_game_worker import CardGameWorker
from scenes.board_games.expansion_native_observer import aggregate_contacts, robot_ownership, ExpansionNativeObserver
from scenes.board_games.expansion_release import checked_asset, sha256
from bench.simulation.isaacsim_backend.native_task_bundle import rigid_contact_sampling_spec


DECK = '/World/CardGame/Fixtures/Deck'
FLOOR = '/World/Room/Geometry/floor'
A, B = ('/World/CardGame/Actors/'+name for name in ('a', 'b'))
HAND, ARM = '/World/Robot/left_index_DP', '/World/Robot/fr3_left_link7'
RIGHT = '/World/Robot/right_index_DP'


def test_sleep_sampling_is_explicit_and_legacy_default_unchanged():
    assert rigid_contact_sampling_spec(None) is None
    value = {'schema': 'native-rigid-body-contact-sampling.v1', 'sleep_threshold': 0., 'stabilization_threshold': 0.}
    assert rigid_contact_sampling_spec(value) == value
    for invalid in ({}, {**value, 'sleep_threshold': False}, {**value, 'stabilization_threshold': .01},
                    {**value, 'solver_iterations': 99}):
        with pytest.raises(ValueError):
            rigid_contact_sampling_spec(invalid)


def owners():
    return robot_ownership({'left_index_DP': HAND, 'fr3_left_link7': ARM, 'right_index_DP': RIGHT})


def contact(a, b, z=.1, *, t=.0005):
    return {'actor0': a, 'actor1': b, 'time_s': t, 'impulse_vector_ns': [0., 0., z],
            'impulse_magnitude_sum_ns': abs(z), 'min_separation_m': -.0001}


def aggregate(rows, **kwargs):
    return aggregate_contacts(rows, now=.0005, paths={A: 'a', B: 'b'}, deck_paths=[DECK],
                              owners=owners(), static_paths=[FLOOR], **kwargs)


def test_floor_is_never_deck_and_unknown_is_never_static():
    support, _, _, _, _, other = aggregate([contact(A, FLOOR), contact(B, '/World/Unknown')])
    assert support == {'a': 0., 'b': 0.}
    assert other == {'a': .1, 'b': 0.}


def test_signed_action_reaction_and_stacking_support():
    support, _, _, stack, _, _ = aggregate([contact(DECK, A, -.1), contact(B, A, .03)])
    assert support == {'a': .1, 'b': 0.}
    assert stack == {'a': {'b': 0.}, 'b': {'a': .03}}
    assert aggregate([contact(A, DECK, -.1)])[0]['a'] == 0.


def test_arm_contact_is_owned_but_never_hand_evidence():
    _, all_robot, nonhand, _, _, _ = aggregate([contact(A, HAND), contact(B, ARM)])
    assert all_robot['a']['player_0'] == all_robot['b']['player_0'] == .1
    assert nonhand['a']['player_0'] == 0. and nonhand['b']['player_0'] == .1


@pytest.mark.parametrize('rows', [
    [contact(A, DECK), contact(DECK, A, -.1)],
    [contact(A, DECK, t=0.)],
    [contact(A, DECK, float('nan'))],
    [contact(A, '/World/Robot/undeclared')],
])
def test_bad_native_contacts_rejected(rows):
    with pytest.raises(RuntimeError):
        aggregate(rows)


def test_robot_ownership_rejects_unowned_actual_link():
    with pytest.raises(ValueError):
        robot_ownership({'left_index_DP': HAND, 'right_index_DP': RIGHT, 'unowned': '/World/Robot/unowned'})


def worker():
    result = CardGameWorker.__new__(CardGameWorker)
    result.match = SimpleNamespace(turn_index=4, current_player='player_0', terminal=False)
    result._owners = {'player_0': frozenset(['left_joint']), 'player_1': frozenset(['right_joint'])}
    result._drive_targets = {'left_joint': 0., 'right_joint': .2}
    result.env = SimpleNamespace(_validate_action=lambda command: None)
    return result


def test_epoch_and_other_robot_target_admission():
    w = worker()
    command = {'joint_positions_rad': {'left_joint': .1, 'right_joint': .2}}
    assert w.authorize_step(command, {'player_id': 'player_0', 'turn_index': 4}) == 4
    with pytest.raises(ValueError):
        w.authorize_step(command, {'player_id': 'player_0', 'turn_index': 3})
    with pytest.raises(ValueError):
        w.authorize_step({'joint_positions_rad': {'left_joint': .1, 'right_joint': .3}},
                         {'player_id': 'player_0', 'turn_index': 4})


def test_complete_paginated_event_delta_consumed_only_on_reply():
    w = worker()
    w._sent_event_cursor = -1
    calls = []
    def event(sequence):
        return {'sequence': sequence, 'type': 'operation_committed', 'task_id': 'polyomino_territory',
                'player_id': 'player_0', 'turn_index': sequence+1, 'native_substep_index': sequence+1,
                'simulation_time_s': .0005*(sequence+1), 'details': {'operation': 'pass', 'piece_ids': []}}
    def events_since(*, after, limit):
        calls.append(after)
        if after < 1:
            return {'events': [event(after+1)], 'next_cursor': after+1, 'has_more': after < 0}
        return {'events': [], 'next_cursor': 1, 'has_more': False}
    w.match.events_since = events_since
    assert w._event_delta() == [event(0), event(1)]
    assert w._event_delta() == []
    assert calls == [-1, 0, 1]


def test_close_detects_final_unobserved_step():
    observer = ExpansionNativeObserver.__new__(ExpansionNativeObserver)
    observer.env = SimpleNamespace(_reset_count=1, _rigid_contacts=object(), _contact_sensor_paths=[A],
        _contact_filter_paths=[[DECK]], dt=.0005, world=SimpleNamespace(current_time_step_index=2,
        current_time=.001), time_s=.001)
    observer.epoch, observer.dt = 1, .0005
    observer.last_step, observer.last_time, observer.last_native_time = 1, .0005, .0005
    observer._view = SimpleNamespace(prim_paths=[A])
    observer._paths = (A,)
    observer._mapping = ((A,), ((DECK,),))
    observer._contact_view = observer.env._rigid_contacts
    observer._closed = observer._attached = observer._failed = False
    observer.samples = observer.unsupported_samples = 0
    observer._stream, observer.frame_jsonl = None, None
    observer.deck_paths, observer.owners = (DECK,), owners()
    report = observer.close()
    assert report['failed'] and report['final_context_verified'] is False


def test_asset_metadata_change_cannot_be_admitted_by_current_file_digest(tmp_path):
    source = Path(f'{DATA_ROOT}/assets/task_parts/board_games/polyomino_territory/scene_v2')
    if not source.is_dir():
        pytest.skip('Shared immutable candidate assets unavailable')
    for name in ('initial.usda', 'metadata.json', 'manifest.json'):
        (tmp_path/name).write_bytes((source/name).read_bytes())
    manifest = load_json(tmp_path/'manifest.json')
    manifest.update(initial_usd=str(tmp_path/'initial.usda'), metadata=str(tmp_path/'metadata.json'))
    (tmp_path/'manifest.json').write_text(json.dumps(manifest))
    checked_asset(tmp_path)
    metadata = load_json(tmp_path/'metadata.json')
    metadata['masses_kg']['black_mono'] *= 2
    (tmp_path/'metadata.json').write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match='missing or changed'):
        checked_asset(tmp_path)


def test_asset_parent_room_mismatch_rejected_even_with_rehashed_metadata(tmp_path):
    source = Path(f'{DATA_ROOT}/assets/task_parts/board_games/polyomino_territory/scene_v2')
    if not source.is_dir():
        pytest.skip('Shared immutable candidate assets unavailable')
    for name in ('initial.usda', 'metadata.json', 'manifest.json'):
        (tmp_path/name).write_bytes((source/name).read_bytes())
    metadata = load_json(tmp_path/'metadata.json')
    metadata['parent_environment']['version'] = 'another_room'
    (tmp_path/'metadata.json').write_text(json.dumps(metadata))
    manifest = load_json(tmp_path/'manifest.json')
    manifest.update(initial_usd=str(tmp_path/'initial.usda'), metadata=str(tmp_path/'metadata.json'),
                    metadata_sha256=sha256(tmp_path/'metadata.json'))
    manifest['files_sha256']['metadata.json'] = manifest['metadata_sha256']
    (tmp_path/'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='identity differs'):
        checked_asset(tmp_path)


@pytest.mark.parametrize('omit_static_filter', [False, True])
def test_real_referee_worker_receives_existing_single_acquisition_and_intent_stays_pending(omit_static_filter):
    from scenes.workrooms.card_room.expansion_referee import public_rule_contract
    from scenes.board_games.expansion_release import canonical_hash
    metadata, _ = checked_asset(f'{DATA_ROOT}/assets/task_parts/board_games/polyomino_territory/scene_v2')
    identities = list(metadata['actors'])
    body_paths = [metadata['actors'][name] for name in identities]
    coms = np.array([[*metadata['mass_properties'][name]['center_of_mass_m'], 0., 0., 0., 1.] for name in identities])
    view = SimpleNamespace(prim_paths=body_paths, count=len(identities), get_coms=lambda: coms,
        get_masses=lambda: np.array([metadata['masses_kg'][name] for name in identities]),
        get_inertias=lambda: np.array([metadata['mass_properties'][name]['inertia_tensor_kg_m2'] for name in identities]),
        get_transforms=lambda: np.array([[*metadata['initial_positions_m'][name], 0., 0., 0., 1.] for name in identities]),
        get_velocities=lambda: np.zeros((len(identities), 6)))
    filters = body_paths+[DECK, HAND, ARM, RIGHT]+([] if omit_static_filter else [FLOOR])
    env = SimpleNamespace(_reset_count=1, dt=.0005, time_s=0., _targets=np.zeros((1, 2)),
        actuated_joint_names=['left_joint', 'right_joint'], joint_names=['left_joint', 'right_joint'],
        world=SimpleNamespace(current_time_step_index=0, current_time=0.),
        _sim_view=SimpleNamespace(create_rigid_body_view=lambda pattern: view),
        scene_info={'objects': {name: {'path': path} for name, path in metadata['actors'].items()},
                    'robot_link_paths': {'left_index_DP': HAND, 'fr3_left_link7': ARM, 'right_index_DP': RIGHT},
                    'support_paths': [FLOOR], 'fixtures': {'deck': {'collision_paths': [DECK]}},
                    'native_task_bundle': {}},
        _contact_sensor_paths=body_paths, _contact_filter_paths=[filters]*len(identities),
        _rigid_contacts=object(), _contact_errors=[], _contact_reports=[],
        _contact_sampling={'sampled_substeps': 0})
    acquisitions = []
    def acquire():
        acquisitions.append(1)
        env._contact_reports.extend(contact(path, DECK, metadata['masses_kg'][name]*9.81*.0005)
                                    for name, path in metadata['actors'].items())
        env._contact_sampling.update(sampled_substeps=1, native_end_step_index=1, end_sim_time_s=.0005)
    env._read_rigid_contacts = acquire
    contract = public_rule_contract('polyomino_territory')
    task = {'schema': 'card_game_expansion_task.v1', 'task_id': 'polyomino_territory',
            'public_rules': contract, 'public_rules_sha256': canonical_hash(contract),
            'public_rules_file': 'rules.json', 'asset_metadata': metadata,
            'capabilities': {'robot_episodes': True}, 'physics_accepted': False, 'task_policy_success_claimed': False}
    scene = {'board_game_task': {'schema': 'card_game_expansion_native_task.v1', 'task_id': task['task_id'],
             'players': [{'player_id': 'player_0', 'color': 'black', 'robot_id': 'left'},
                         {'player_id': 'player_1', 'color': 'white', 'robot_id': 'right'}]}}
    if omit_static_filter:
        with pytest.raises(RuntimeError, match='known-static contact coverage'):
            CardGameWorker(env, scene, task)
        return
    w = CardGameWorker(env, scene, task)
    actor = {'player_id': 'player_0', 'turn_index': 0}
    intent = {'type': 'place_piece', 'piece_id': 'black_mono', 'anchor': [0, 0], 'rotation_deg': 0}
    assert 'board_game_error' in w.request(intent, actor=actor)
    env.world.current_time_step_index = 1
    env.time_s = env.world.current_time = .0005
    env._read_rigid_contacts()
    assert acquisitions == [1] and w.observer.samples == 1
    reply = w.request(intent, actor=actor)
    assert 'board_game_error' not in reply
    assert reply['board_game']['pending_move']['type'] == 'place_piece'
    assert reply['board_game']['turn_index'] == 0 and reply['card_game_events'] == []
    assert reply['score']['success'] is False and reply['score']['end_reason'] is None
    assert w.close(reason='cancelled')['final_context_verified'] is True
