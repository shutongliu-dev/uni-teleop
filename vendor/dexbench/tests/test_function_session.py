"""Public-only session doubles; no model calls or native simulation."""
from copy import deepcopy

import pytest

from bench.protocol.functions import FUNCTION_INPUTS, FunctionFailure
from agent.model_client.bench_http import BenchHTTPError
from agent.model_client.function_session import FunctionSession
from agent.model_client.learning.robot_api import ActionRecoveryError, RobotAPI


def player():
    return {'player_id': 'player_0', 'color': 'black', 'hand': 'left',
            'robot_id': 'left', 'joint_names': ['left_arm', 'left_finger']}


class Rejected(BenchHTTPError):
    def __init__(self, name, code='invalid_arguments'):
        super().__init__('Public function rejected', category='function_rejected', status_code=422)
        self.definite_no_action = code == 'invalid_arguments'
        self.response = FunctionFailure(function=name, request_id='agent_request_test', context=None,
            error={'code': code, 'message': 'Correct the arguments', 'http_status': 422}).model_dump(mode='json')


class Client:
    episode_id = 'ep1'
    base_url = 'http://bench.test'

    def __init__(self, *, board=False):
        self.binding = player() if board else None
        self.deadline = None
        self.obs = {'episode_id': self.episode_id, 'sequence': 0, 'sim_time_s': 0.,
            'instruction': 'Place pieces', 'profile': 'vision_proprio',
            'proprioception': {'joint_positions_rad': {'left_arm': 0., 'left_finger': 0.},
                              'joint_velocities_rad_s': {'left_arm': 0., 'left_finger': 0.}},
            'cameras': [], 'last_action_status': 'idle'}
        self.public = {'observation': self.obs}
        if board:
            self.public.update(player=self.binding,
                board_game={'current_player': 'player_0', 'terminal': False, 'turn_index': 0},
                rules={'settlement_dwell_s': .3}, geometry={'cell_spacing_m': .14})
        self.calls, self.posts, self.intents, self.queries, self.cancels = [], [], [], [], []
        self.receipts = {}
        self.auto_finish = True
        self.reject = self.lose_post = self.no_status = self.handoff = False
        self.unavailable = set()

    def state(self):
        return deepcopy(self.public)

    def catalog(self):
        return {'functions': [{'name': name, 'available': name not in self.unavailable}
                              for name in FUNCTION_INPUTS],
                'robot_capabilities': {'arm_joints': ['left_arm'], 'hand_joints': ['left_finger']}}

    def call(self, payload):
        self.calls.append(deepcopy(payload))
        if payload['function'] == 'get_robot':
            result = {'hands': {'left': {'fingers': {'index': {'joint_names': ['left_finger']}}}},
                      'joint_limits': {'left_finger': {'lower_rad': 0., 'upper_rad': 1.}},
                      'ik_query': {'available': True}, 'hand_presets': {'available': True}}
        else:
            result = {'function': payload['function'], 'arguments': payload['arguments']}
        return {'ok': True, 'function': payload['function'], 'request_id': payload['request_id'],
                'result': result}

    def submit(self, episode, command):
        assert episode == self.episode_id
        self.posts.append(deepcopy(command))
        if self.reject:
            raise Rejected(command['function'], self.reject if isinstance(self.reject, str) else 'invalid_arguments')
        receipt = {'action_id': command['action_id'], 'status': 'accepted',
                   'sim_time_s': self.obs['sim_time_s']}
        self.receipts[command['action_id']] = receipt
        if self.lose_post:
            raise BenchHTTPError('Response lost', category='transport_timeout')
        return deepcopy(receipt)

    def get_action(self, episode, action_id):
        self.queries.append(action_id)
        if self.no_status or action_id not in self.receipts:
            raise BenchHTTPError('Unknown action', status_code=404)
        if self.auto_finish and self.receipts[action_id]['status'] == 'accepted':
            status = 'cancelled' if self.handoff else 'completed'
            self.obs.update(sequence=self.obs['sequence'] + 1, sim_time_s=self.obs['sim_time_s'] + .02,
                            last_action_id=action_id, last_action_status=status)
            self.receipts[action_id] = {'action_id': action_id, 'status': status,
                                       'sim_time_s': self.obs['sim_time_s']}
            if self.handoff:
                self.receipts[action_id]['termination_reason'] = 'board_turn_changed'
                self.public['board_game'].update(current_player='player_1', turn_index=1)
        return deepcopy(self.receipts[action_id])

    def get_observation(self, episode):
        return deepcopy(self.obs)

    def cancel_action(self, episode, action_id):
        self.cancels.append(action_id)
        return {**self.receipts[action_id], 'cancellation_requested': True}

    def intent(self, payload):
        self.intents.append(deepcopy(payload))
        return deepcopy(self.public['board_game'])


def session(tmp_path, client=None, **kwargs):
    client = client or Client()
    host = FunctionSession(client, output=tmp_path / 'run', spec={'task_id': 'task0'},
        trial_id='trial1', max_actions=kwargs.pop('max_actions', 10),
        max_calls=kwargs.pop('max_calls', 30), **kwargs)
    host.initialize()
    return host, client


def decision(name='hold', **arguments):
    return {'function': name, 'arguments': arguments or {'duration_s': .2}}


def test_catalog_exposes_complete_public_hand_schemas_and_guide(tmp_path):
    host, _ = session(tmp_path)
    value = host.context()
    tools = {row['function']: row['arguments_schema'] for row in value['tools']}
    assert set(tools) == set(FUNCTION_INPUTS)
    assert 'hand_posture_rad' in tools['set_hand_targets']['properties']
    assert 'fingertip_positions_m' in tools['set_hand_targets']['properties']
    assert 'hand_posture_rad' in tools['move_geometry']['$defs']['GeometryObjective']['properties']
    assert 'hand_posture_rad' in tools['query_ik']['$defs']['GeometryObjective']['properties']
    for schema in tools.values():
        assert not {'action_id', 'observation_sequence'} & schema['properties'].keys()
        assert not {'action_id', 'observation_sequence'} & set(schema['required'])
    assert value['robot']['hands']['left']['fingers']['index']['joint_names'] == ['left_finger']
    assert 'joint_limits' in value['robot'] and value['robot']['hand_presets']['available']


@pytest.mark.parametrize(('name', 'arguments'), [
    ('project_pixel', {'camera_id': 'overview', 'u': 4, 'v': 7}),
    ('hand_preset', {'preset': 'pinch_prepose', 'side': 'left', 'amount': .6}),
    ('query_ik', {'hand_geometries': {'left': {'hand_posture_rad': {'left_finger': .1},
                                           'fingertip_frame_id': 'left_hand_flange'}}}),
    ('transform_points', {'source_frame': 'left_hand_flange', 'target_frame': 'world',
                          'points_m': [[0., 0., .1]]}),
])
def test_all_public_queries_bind_sequence_and_use_player_request_prefix(tmp_path, name, arguments):
    host, client = session(tmp_path, Client(board=True))
    assert host.execute({'function': name, 'arguments': arguments})['ok']
    payload = client.calls[-1]
    assert payload['function'] == name and payload['request_id'].startswith('player_0_request_')
    if 'observation_sequence' in FUNCTION_INPUTS[name].model_fields:
        assert payload['arguments']['observation_sequence'] == 0
    assert client.posts == [] and host.actions == 0 and host.calls == 1


@pytest.mark.parametrize('arguments', [
    {'hand_posture_rad': {'left_finger': .2}},
    {'fingertip_positions_m': {'index': [0., 0., .12]}},
    {'hand_posture_rad': {'left_finger': .2}, 'fingertip_positions_m': {'index': [0., 0., .12]}},
])
def test_hand_targets_execute_through_durable_ledger_and_match_observation(tmp_path, arguments):
    host, client = session(tmp_path, Client(board=True))
    result = host.execute(decision('set_hand_targets', hand='left', timeout_sim_s=.5, **arguments))
    assert result['ok'] and result['result']['status'] == 'completed'
    assert host.pending is None and host.observation['sequence'] == 1
    assert host.observation['last_action_id'] == client.posts[0]['action_id']
    assert client.posts[0]['action_id'].startswith('player_0_action_')
    assert host.calls == host.actions == len(client.posts) == 1
    assert host.action_api.journal.read()[-1]['kind'] == 'action_reconciled'


@pytest.mark.parametrize(('name', 'arguments'), [
    ('set_hand_targets', {'hand': 'right', 'timeout_sim_s': .5, 'hand_posture_rad': {'right_finger': .1}}),
    ('set_hand_targets', {'hand': 'left', 'timeout_sim_s': .5, 'hand_posture_rad': {'right_finger': .1}}),
    ('move_joints', {'duration_s': .2, 'arm_joint_positions_rad': {'right_arm': .1}, 'hand_joint_positions_rad': {'right_finger': .1}}),
    ('move_geometry', {'timeout_sim_s': .2, 'geometry': {'hand_posture_rad': {'left_finger': .1}}}),
    ('query_ik', {'hand_geometries': {'right': {'hand_posture_rad': {'right_finger': .1}}}}),
    ('hand_preset', {'preset': 'open', 'side': 'right'}),
])
def test_cross_side_control_and_queries_reject_before_transport(tmp_path, name, arguments):
    host, client = session(tmp_path, Client(board=True))
    result = host.execute({'function': name, 'arguments': arguments})
    assert result['error']['code'] == 'permission_denied'
    assert client.posts == [] and len(client.calls) == 1


@pytest.mark.parametrize('key', ['observation_sequence', 'action_id', 'request_id', 'episode_id', 'player_id', 'session_token'])
def test_model_cannot_inject_transport_identity_or_snapshot(tmp_path, key):
    host, client = session(tmp_path)
    result = host.execute(decision(**{key: 'injected', 'duration_s': .2}))
    assert result['error']['code'] == 'invalid_arguments' and not client.posts


def test_stale_decision_is_rejected_then_corrected_on_current_sequence(tmp_path):
    host, client = session(tmp_path)
    client.obs.update(sequence=1, sim_time_s=.02)
    rejected = host.execute(decision())
    assert rejected['error']['code'] == 'stale_observation' and not client.posts
    assert host.observation['sequence'] == 1
    assert host.execute(decision())['ok']
    assert client.posts[0]['observation_sequence'] == 1 and host.actions == 2


def test_definite_function_rejection_allows_corrected_new_action_and_recovers_ledger(tmp_path):
    host, client = session(tmp_path)
    client.reject = True
    rejected = host.execute(decision())
    assert not rejected['ok'] and rejected['error']['code'] == 'invalid_arguments'
    assert host.pending is None
    assert host.action_api.journal.read()[-1]['kind'] == 'action_rejected'
    client.reject = False
    assert host.execute(decision())['ok']
    assert len(client.posts) == 2 and client.posts[0]['action_id'] != client.posts[1]['action_id']
    recovered = RobotAPI(client, host.output / 'host/action_journal', episode_id='ep1',
                        trial_id='trial1', binding=host.action_api.binding['contract'])
    assert recovered.pending is None


@pytest.mark.parametrize('failure', ['timeout', 'infrastructure_error', 'action_conflict'])
def test_uncertain_post_never_unlocks_or_resubmits_even_on_404(tmp_path, failure):
    host, client = session(tmp_path)
    client.no_status = True
    if failure == 'timeout':
        client.lose_post = True
    else:
        client.reject = failure
    rejected = host.execute(decision())
    assert not rejected['ok'] and host.pending is not None
    action_id = host.pending['action_id']
    assert host.execute(decision())['error']['code'] == 'action_pending'
    assert len(client.posts) == 1 and client.queries == [action_id]
    assert not any(row['kind'] == 'action_rejected' for row in host.action_api.journal.read())


def test_unknown_post_recovers_original_id_and_allows_next_decision(tmp_path):
    host, client = session(tmp_path)
    client.lose_post = True
    result = host.execute(decision())
    assert result['ok'] and host.pending is None
    assert len(client.posts) == 1 and client.queries == [client.posts[0]['action_id']]


def test_pending_poll_limit_is_finite_and_cancellation_ack_does_not_unlock(tmp_path, monkeypatch):
    from agent.model_client.learning import robot_api
    monkeypatch.setattr(robot_api.time, 'sleep', lambda _: None)
    host, client = session(tmp_path, max_status_queries=2)
    client.auto_finish = False
    result = host.execute(decision())
    assert result['error']['error_category'] == 'request_budget_exhausted'
    action_id = host.pending['action_id']
    assert client.queries == [action_id, action_id]
    result = host.execute({'function': 'cancel_action', 'arguments': {}})
    assert not result['ok'] and host.pending is not None
    assert client.cancels == [action_id] and len(client.posts) == 1
    client.auto_finish = True
    assert host.execute({'function': 'get_action', 'arguments': {}})['ok']
    assert host.pending is None and len(client.posts) == 1


def test_turn_handoff_cancelled_receipt_is_legitimate_and_reconciled(tmp_path):
    host, client = session(tmp_path, Client(board=True))
    client.handoff = True
    result = host.execute(decision())
    assert result['ok'] and result['result']['status'] == 'cancelled'
    assert result['result']['termination_reason'] == 'board_turn_changed'
    assert host.pending is None and host.state['board_game']['current_player'] == 'player_1'
    assert host.execute(decision())['error']['code'] == 'permission_denied'
    assert len(client.posts) == 1


def test_intents_require_current_player_owned_piece_and_never_claim_placement(tmp_path):
    host, client = session(tmp_path, Client(board=True))
    bad = host.execute({'intent': {'type': 'placement_intent', 'piece_id': 'white_00', 'q': 0, 'r': 0}})
    assert bad['error']['code'] == 'permission_denied' and client.intents == []
    result = host.execute({'intent': {'type': 'placement_intent', 'piece_id': 'black_00', 'q': 0, 'r': 0}})
    assert result['ok'] and result['result']['turn_index'] == 0
    assert host.execute({'intent': {'type': 'cancel_placement'}})['ok']
    assert client.posts == [] and host.actions == 0


def test_task0_is_ordinary_session_with_no_board_tools(tmp_path):
    host, client = session(tmp_path)
    assert all('intent' not in item for item in host.context()['tools'])
    result = host.execute({'intent': {'type': 'cancel_placement'}})
    assert result['error']['code'] == 'unsupported_function' and client.intents == []
    assert host.execute(decision())['ok']


def test_unavailable_function_is_not_advertised_or_called(tmp_path):
    client = Client()
    client.unavailable.add('move_joints')
    host, client = session(tmp_path, client)
    assert 'move_joints' not in {row['function'] for row in host.context()['tools']}
    assert host.execute(decision('move_joints'))['error']['code'] == 'unsupported_function'
    assert client.posts == []


def test_budget_and_trial_reopen_do_not_reset_attempts(tmp_path):
    host, client = session(tmp_path, max_actions=1, max_calls=2)
    assert host.execute(decision(duration_s=-1))['error']['code'] == 'invalid_arguments'
    assert host.actions == 1 and not client.posts
    assert host.execute(decision())['error']['code'] == 'budget_exhausted'
    assert host.execute({'function': 'observe', 'arguments': {}})['error']['code'] == 'budget_exhausted'
    assert host.calls == 2
    with pytest.raises(ActionRecoveryError, match='cannot reopen'):
        session(tmp_path, client)


def test_deadline_propagates_and_preserves_pending_if_submit_crosses_it(tmp_path, monkeypatch):
    from agent.model_client import function_session
    now = [1.]
    monkeypatch.setattr(function_session.time, 'monotonic', lambda: now[0])
    host, client = session(tmp_path, deadline=2.)
    original = client.submit
    def late(episode, command):
        assert client.deadline == 2.
        value = original(episode, command)
        now[0] = 3.
        return value
    client.submit = late
    result = host.execute(decision())
    assert result['error']['error_category'] == 'request_budget_exhausted'
    assert host.pending is not None and len(client.posts) == 1 and client.deadline is None


@pytest.mark.parametrize('malformed', [None, [], {'function': [], 'arguments': {}},
    {'function': 'hold', 'arguments': {'duration_s': .2}, 'player_id': 'player_1'}])
def test_malformed_decisions_remain_structured_and_do_not_reach_transport(tmp_path, malformed):
    host, client = session(tmp_path)
    result = host.execute(malformed)
    assert not result['ok'] and not client.posts and len(client.calls) == 1


def test_unmatched_terminal_observation_blocks_following_motion(tmp_path):
    host, client = session(tmp_path)
    original = client.get_observation
    def wrong(episode):
        value = original(episode)
        value['last_action_id'] = 'unrelated_action'
        return value
    client.get_observation = wrong
    result = host.execute(decision())
    assert not result['ok'] and host.pending is not None
    assert host.execute(decision())['error']['code'] == 'action_pending'
    client.get_observation = original
    assert host.execute({'function': 'get_action', 'arguments': {}})['ok']
    assert host.pending is None and len(client.posts) == 1


def test_infrastructure_failure_cannot_be_marked_definite_by_transport_callback(tmp_path):
    host, client = session(tmp_path)
    def wrongly_classified(episode, command):
        error = Rejected(command['function'], 'infrastructure_error')
        error.definite_no_action = True
        raise error
    client.submit = wrongly_classified
    result = host.execute(decision())
    assert result['error']['error_category'] == 'action_recovery_error'
    assert host.pending is not None
    assert not any(row['kind'] == 'action_rejected' for row in host.action_api.journal.read())


def test_player_schemas_expose_only_owned_geometry_and_correct_frame_default(tmp_path):
    host, _ = session(tmp_path, Client(board=True))
    tools = {row['function']: row['arguments_schema'] for row in host.context()['tools'] if 'function' in row}
    for name, selector in [('move_wrist', 'hand'), ('set_hand_targets', 'hand'), ('hand_preset', 'side')]:
        assert tools[name]['properties'][selector]['enum'] == ['left']
    for name in ('move_geometry', 'query_ik'):
        schema = tools[name]
        assert 'geometry' not in schema['properties']
        assert 'hand_geometries' in schema['required']
        geometry_map = schema['properties']['hand_geometries']
        assert geometry_map['required'] == ['left']
        assert set(geometry_map['properties']) == {'left'}
        assert geometry_map['additionalProperties'] is False
        frame = schema['$defs']['GeometryObjective']['properties']['fingertip_frame_id']
        assert frame['default'] == 'left_hand_flange' and frame['enum'] == ['left_hand_flange']
    # Projection must not mutate the canonical schemas or another player's tools.
    canonical = FUNCTION_INPUTS['query_ik'].model_json_schema()
    assert canonical['$defs']['GeometryObjective']['properties']['fingertip_frame_id']['default'] == 'right_hand_flange'


@pytest.mark.parametrize('name', ['move_geometry', 'query_ik'])
def test_left_geometry_omitted_frame_is_bound_to_owned_hand(tmp_path, name):
    host, client = session(tmp_path, Client(board=True))
    arguments = {'hand_geometries': {'left': {'fingertip_positions_m': {'index': [0., 0., .1]}}}}
    if name == 'move_geometry':
        arguments['timeout_sim_s'] = .5
    assert host.execute({'function': name, 'arguments': arguments})['ok']
    sent = client.posts[-1] if name == 'move_geometry' else client.calls[-1]
    assert sent['arguments']['hand_geometries']['left']['fingertip_frame_id'] == 'left_hand_flange'
    assert 'fingertip_frame_id' not in arguments['hand_geometries']['left']


@pytest.mark.parametrize('name', ['move_geometry', 'query_ik'])
def test_explicit_other_hand_frame_is_rejected_without_rewriting_or_call(tmp_path, name):
    host, client = session(tmp_path, Client(board=True))
    arguments = {'hand_geometries': {'left': {'hand_posture_rad': {'left_finger': .1},
                                           'fingertip_frame_id': 'right_hand_flange'}}}
    if name == 'move_geometry':
        arguments['timeout_sim_s'] = .5
    result = host.execute({'function': name, 'arguments': arguments})
    assert result['error']['code'] == 'permission_denied'
    assert client.posts == [] and len(client.calls) == 1
    assert arguments['hand_geometries']['left']['fingertip_frame_id'] == 'right_hand_flange'
