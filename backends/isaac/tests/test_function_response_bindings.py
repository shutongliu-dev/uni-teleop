"""Response identity and frozen-contract checks with in-memory HTTP replies."""
from contextlib import contextmanager
from copy import deepcopy
import hashlib
import json

import httpx
import pytest

from agent.model_client.bench_http import BenchHTTPError
from agent.model_client.function_http import FunctionHTTPClient
from bench.protocol.schemas import EpisodeResult, EpisodeSpec


def episode_spec(task_id):
    return EpisodeSpec(
        episode_id='ep1', agent_id='review', agent_version='v1',
        agent_config_sha256='0' * 64, task_id=task_id,
        task_release_sha256='1' * 64, evaluator_sha256='2' * 64,
        robot_id='review_robot', robot_asset_sha256='3' * 64,
        physics_profile_sha256='4' * 64, seed=7, track='direct_joints',
        instruction='Use the frozen task.',
    ).model_dump(mode='json')


def player(kind):
    binding = ({'player_id': 'player_0', 'color': 'black', 'hand': 'left', 'robot_id': 'left',
                'joint_names': ['left_joint']} if kind == 'board_game' else
               {'player_id': 'alpha', 'hand': 'front', 'robot_id': 'front',
                'joint_names': ['front_joint']})
    return {'binding': binding, 'session_token': 'test_player_token_' + 'a' * 32}


@contextmanager
def client_for(reply, *, kind='episode', spec=None):
    def handler(request):
        value = reply(request) if callable(reply) else reply
        return httpx.Response(200, json=value)
    with httpx.Client(transport=httpx.MockTransport(handler)) as http:
        yield FunctionHTTPClient('http://bench.test', 'ep1', client=http, spec=spec,
                                 session_kind=kind, player=None if kind == 'episode' else player(kind))


def game_result(kind, spec):
    episode = EpisodeResult(spec=spec, outcome='cancelled',
                            elapsed_sim_s=1., elapsed_wall_s=1.).model_dump(mode='json')
    state = ({'task_id': spec['task_id'], 'turn_index': 0, 'commit_count': 0,
              'current_player': None, 'terminal': True, 'winner': None, 'draw': False,
              'pending_move': None, 'status': {
                  'outcome': 'aborted', 'end_reason': 'cancelled', 'recovery_required': False,
                  'evidence_unavailable': False, 'reason': 'cancelled', 'retryable': False}}
             if kind == 'board_game' else
             {'game_id': spec['task_id'], 'status': 'interrupted', 'turn_index': 0,
              'current_player_id': None, 'players': ['alpha', 'beta'], 'winner_player_id': None,
              'loser_player_id': None, 'terminal_reason': 'cancelled', 'violation_reason': None,
              'sim_time_s': 1., 'public_rules': {'intents': ['select']}, 'public_progress': {}})
    return {kind: state, 'episode': episode}


@pytest.mark.parametrize('kind,task', [
    ('board_game', 'tic_tac_toe_3x3'), ('turn_game', 'tower_extraction'),
])
@pytest.mark.parametrize('field,value', [
    ('task_release_sha256', 'f' * 64), ('evaluator_sha256', 'f' * 64),
    ('physics_profile_sha256', 'f' * 64), ('scene_sha256', 'f' * 64),
    ('robot_asset_sha256', 'f' * 64), ('runtime_code_sha256', 'f' * 64),
    ('sensor_config_sha256', 'f' * 64), ('agent_config_sha256', 'f' * 64),
    ('robot_id', 'other_robot'), ('seed', 999), ('budgets', {'max_actions': 4}),
])
def test_game_result_rejects_changed_frozen_spec(kind, task, field, value):
    spec = episode_spec(task)
    reply = game_result(kind, spec)
    reply['episode']['spec'][field] = value
    with client_for(reply, kind=kind, spec=spec) as client:
        with pytest.raises(BenchHTTPError) as failure:
            client.game_result()
        assert failure.value.category == 'invalid_response'


@pytest.mark.parametrize('kind,task', [
    ('board_game', 'tic_tac_toe_3x3'), ('turn_game', 'tower_extraction'),
])
@pytest.mark.parametrize('bound', [False, True])
def test_matching_game_result_keeps_standalone_client_compatibility(kind, task, bound):
    spec = episode_spec(task)
    reply = game_result(kind, spec)
    with client_for(reply, kind=kind, spec=spec if bound else None) as client:
        assert client.game_result()['episode']['spec'] == spec


def board_public():
    value = {
        'board_game': {'task_id': 'tic_tac_toe_3x3', 'turn_index': 0, 'commit_count': 0,
                      'current_player': 'player_0', 'terminal': False, 'winner': None,
                      'draw': False, 'pending_move': None,
                      'status': {'outcome': 'ongoing', 'end_reason': None, 'recovery_required': False,
                                 'evidence_unavailable': False, 'reason': 'ready', 'retryable': True}},
        'public_rules': {'task_id': 'tic_tac_toe_3x3', 'revision': 'frozen_v1',
                         'rules': {'stable_s': .3},
                         'geometry': {'game_id': 'tic_tac_toe_3x3', 'pitch_m': .14}},
        'observation': {'episode_id': 'ep1', 'sequence': 2, 'sim_time_s': .4,
                        'instruction': 'Use the frozen task.', 'profile': 'vision_proprio',
                        'proprioception': {'joint_positions_rad': {'left_joint': 0.},
                                           'joint_velocities_rad_s': {'left_joint': 0.}},
                        'cameras': []},
        'player': player('board_game')['binding'],
    }
    update_board_hash(value)
    return value


def update_board_hash(value):
    rules = value['public_rules']
    value['rules'], value['geometry'] = deepcopy(rules['rules']), deepcopy(rules['geometry'])
    value['public_rules_sha256'] = hashlib.sha256(json.dumps(
        rules, sort_keys=True, separators=(',', ':'), ensure_ascii=True, allow_nan=False).encode()).hexdigest()


@pytest.mark.parametrize('initial_binding', ['creation', 'first_state'])
@pytest.mark.parametrize('endpoint', ['state', 'intent'])
@pytest.mark.parametrize('change', ['rules', 'geometry', 'revision'])
def test_board_contract_drift_is_rejected_even_with_a_matching_new_hash(initial_binding, endpoint, change):
    initial = board_public()
    reply = deepcopy(initial)
    with client_for(reply, kind='board_game', spec=episode_spec('tic_tac_toe_3x3')) as client:
        if initial_binding == 'creation':
            # Creation wrappers also contain host-only credentials; only public fields are bound.
            initial['players'] = [player('board_game'), {'session_token': 'private_peer_token_' + 'b' * 32}]
            client.bind_board_state(initial)
            assert 'private_peer_token_' not in repr(client.__dict__)
        else:
            client.state()
        if change == 'rules':
            reply['public_rules']['rules']['stable_s'] = 0.
        elif change == 'geometry':
            reply['public_rules']['geometry']['pitch_m'] = .2
        else:
            reply['public_rules']['revision'] = 'replacement_v2'
        update_board_hash(reply)
        reply['observation']['sequence'] += 1
        with pytest.raises(BenchHTTPError) as failure:
            (client.state() if endpoint == 'state' else client.intent(
                {'type': 'placement_intent', 'piece_id': 'black_00', 'q': 0, 'r': 0}))
        assert failure.value.category == 'invalid_response'
        # A rejected response cannot replace the previously accepted contract.
        reply.clear()
        reply.update(board_public())
        assert client.state()['public_rules'] == board_public()['public_rules']


def test_board_binding_copies_creation_rules_and_allows_normal_turn_progress():
    initial = board_public()
    reply = deepcopy(initial)
    with client_for(reply, kind='board_game') as client:
        client.bind_board_state(initial)
        initial['public_rules']['rules']['stable_s'] = 0.
        reply['board_game'].update(turn_index=1, commit_count=1, current_player='player_1')
        reply['observation']['sequence'] += 1
        assert client.state()['board_game']['current_player'] == 'player_1'


def ik_response(request, *, solution_found=True):
    hand = ({'solution_found': True, 'joint_positions_rad': {'right_index_MP': .1},
             'verified_geometry': {'wrist': {'frame_id': 'world', 'position_m': [0., 0., 0.],
                                             'quaternion_wxyz': [1., 0., 0., 0.]},
                                   'fingertip_frame_id': 'right_hand_flange', 'fingertip_positions_m': {}},
             'residuals': {'fingertip_errors_m': {}, 'wrist_position_error_m': 0.,
                           'wrist_rotation_error_rad': 0.}} if solution_found else
            {'solution_found': False, 'diagnostic': {'code': 'ik_solution_unverified'}})
    return {
        'ok': True, 'request_id': request['request_id'], 'function': 'query_ik',
        'context': {'episode_id': 'ep1', 'observation_sequence': 99, 'observation_sim_time_s': 0.,
                    'episode_finished': False, 'request_recorded': True,
                    'budget': {'limits': {}, 'action_attempts': 0, 'elapsed_sim_s': 0.,
                               'elapsed_wall_s': 1., 'remaining_wall_s': 100.}},
        'result': {'protocol': 'bench_ik_query_v1', 'observation_sequence': 99, 'sim_time_s': 0.,
                   'solution_found': solution_found, 'hands': {'right': hand},
                   'advances_physics': False, 'mutates_controller': False, 'collision_checked': False,
                   'trajectory_checked': False, 'scope': 'Read-only geometric query'},
    }


@pytest.mark.parametrize('solution_found', [False, True])
@pytest.mark.parametrize('change', ['result_sequence', 'context_sequence', 'result_time', 'context_time', None])
def test_ik_response_is_bound_to_requested_snapshot_and_context(solution_found, change):
    payload = {'request_id': 'review_ik', 'function': 'query_ik',
               'arguments': {'observation_sequence': 99,
                             'geometry': {'hand_posture_rad': {'right_index_MP': .1}}}}
    reply = ik_response(payload, solution_found=solution_found)
    if change is not None:
        owner, field = change.split('_')
        reply[owner]['observation_sequence' if field == 'sequence' else
                     'observation_sim_time_s' if owner == 'context' else 'sim_time_s'] = 1
    with client_for(reply) as client:
        if change is None:
            assert client.call(payload)['result']['solution_found'] is solution_found
        else:
            with pytest.raises(BenchHTTPError) as failure:
                client.call(payload)
            assert failure.value.category == 'invalid_response'


@pytest.mark.parametrize('offset,accepted', [(5.e-10, True), (1.e-9, True), (2.e-9, False)])
def test_ik_clock_uses_the_existing_nanosecond_tolerance(offset, accepted):
    payload = {'request_id': 'review_ik', 'function': 'query_ik',
               'arguments': {'observation_sequence': 99,
                             'geometry': {'hand_posture_rad': {'right_index_MP': .1}}}}
    reply = ik_response(payload)
    reply['result']['sim_time_s'] += offset
    with client_for(reply) as client:
        if accepted:
            assert client.call(payload)['result']['sim_time_s'] == offset
        else:
            with pytest.raises(BenchHTTPError) as failure:
                client.call(payload)
            assert failure.value.category == 'invalid_response'
