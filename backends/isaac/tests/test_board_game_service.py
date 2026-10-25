"""HTTP/player lifecycle checks using frozen tasks and explicit CPU evidence.

The worker double sends synthetic complete frames into the frozen referee. It
does not simulate contacts and these tests do not qualify native manipulation.
"""
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import fields
import json
from pathlib import Path
from types import SimpleNamespace

from fastapi.testclient import TestClient
import pytest

from bench.control.factory import public_robot_geometry, robot_configuration
from bench.isaacsim.release import task_inputs
from bench.protocol.board_games import BoardGameState, BoardGameResult
from bench.runtime.board_game_worker import BoardGameWorker
from bench.runtime.controllers import default_registry
from bench.runtime.server import create_app
from bench.runtime.service import AgentService, ServiceError
from scenes.board_games.tabletop import (ActuatorOwnership, BoardGameGeometry, BoardGameMatch,
                                        BoardGamePlayerProtocol, SettlementThresholds)
from scenes.board_games.tabletop.settlement import NativeFrame, PieceEvidence
from scenes.workrooms.catalog import task_definition
from test_agent_service import FakeWorker, PRIVATE_MARKER, join_action


class FakeBoardWorker(FakeWorker):
    """Real robot metadata, frozen referee and synthetic frames; no simulator."""

    def __init__(self, *, release, task_id, hand_id, **kwargs):
        assert hand_id == 'dual_fr3_sharpa_w01'
        super().__init__(hand_id='sharpa_w01_right', acknowledged_targets=True, depth=True, **kwargs)
        _, scene, task, initial = task_inputs(release, task_id)
        model = public_robot_geometry(robot_configuration(scene))
        self.arm = list(model.arm_names)
        self.active = list(model.names)
        self.hand = [name for name in self.active if name not in self.arm]
        self.raw['joint_positions_rad'] = deepcopy(initial['joint_positions_rad'])
        self.raw['joint_velocities_rad_s'] = dict.fromkeys(self.active, 0.)
        self.raw['robot'] = {'joint_targets_rad': deepcopy(initial['joint_positions_rad'])}
        self.initial['metadata'].update(scene_spec=scene, action_spec={
            'joint_names': self.active, 'arm_joint_names': self.arm, 'hand_joint_names': self.hand,
            'joint_limits_rad': {name: [float(side.lower[i]), float(side.upper[i])]
                                 for side in model.sides.values() for i, name in enumerate(side.names)}})
        geometry = BoardGameGeometry(**{f.name: task['geometry'][f.name] for f in fields(BoardGameGeometry)})
        thresholds = SettlementThresholds(**{f.name: task['settlement'][f.name] for f in fields(SettlementThresholds)})
        judge = BoardGameWorker.__new__(BoardGameWorker)
        judge.match = BoardGameMatch(geometry, thresholds=thresholds)
        ownership = tuple(ActuatorOwnership(f'player_{i}', side, tuple(model.sides[side].names))
                          for i, side in enumerate(('left', 'right')))
        judge.protocol = BoardGamePlayerProtocol(judge.match, ownership)
        judge._owners = {row.player_id: row for row in ownership}
        judge._drive_targets = deepcopy(initial['joint_positions_rad'])
        judge.env = SimpleNamespace(_validate_action=self._validate_command)
        judge.observer = SimpleNamespace(samples=0, unsupported_samples=0)
        judge.output = None
        self.judge = judge
        self.settled_pending = False
        self.orphan_piece = None
        self.fail_step = False
        self.fail_end = False
        self.fail_intent = False
        self.broken = False
        self.inactive_tracking_error = 0.
        self.initial.update(observation=deepcopy(self.raw), score=judge.score(), board_game=judge.public_state())

    def _validate_command(self, command):
        assert set(command) == {'joint_positions_rad', 'duration_s'}
        assert set(command['joint_positions_rad']) == set(self.active)
        assert command['duration_s'] == .02

    def _synthetic_frames(self, duration):
        """Only test code supplies evidence; public requests cannot call this."""
        match, geometry = self.judge.match, self.judge.match.geometry
        for _ in range(round(duration / .0005)):
            self.judge.observer.samples += 1
            index = self.judge.observer.samples
            pieces = {}
            for piece_id in match.inventory:
                cell = match.committed.get(piece_id)
                if match.pending and match.pending.piece_id == piece_id and self.settled_pending:
                    cell = (match.pending.q, match.pending.r)
                if piece_id == self.orphan_piece and cell is None:
                    cell = (2, 2)
                color, number = piece_id.split('_')
                position = geometry.cell_center(*cell) if cell is not None else geometry.reserve_position(color, int(number))
                pieces[piece_id] = PieceEvidence(tuple(position), (1., 0., 0., 0.), (0., 0., 0.),
                    (0., 0., 0.), .001 if cell is not None else 0., 0., 0.)
            match.observe(NativeFrame(index, index * .0005, .0005, pieces, True))

    def request(self, operation, **payload):
        if operation == 'board_game':
            self.calls.append((operation, deepcopy(payload)))
            if self.fail_intent and payload['command'].get('type') == 'placement_intent':
                self.broken = True
                raise RuntimeError('Synthetic intent transport failure')
            if self.fail_end and payload['command'].get('type') == 'end_episode':
                self.broken = True
                raise RuntimeError('Synthetic transport failure')
            return self.judge.request(payload['command'], actor=payload.get('actor'))
        if operation == 'step':
            if self.fail_step:
                self.calls.append((operation, deepcopy(payload)))
                self.broken = True
                raise RuntimeError('Synthetic transport failure')
            admitted = self.judge.authorize_step(payload['command'], payload.get('actor'))
            reply = super().request(operation, **payload)
            self.raw['robot']['joint_targets_rad'] = deepcopy(payload['command']['joint_positions_rad'])
            self.raw['joint_positions_rad']['fr3_right_joint1'] += self.inactive_tracking_error
            self._synthetic_frames(payload['command']['duration_s'])
            reply.update(observation=deepcopy(self.raw), score=self.judge.score(),
                         **self.judge.acknowledge_step(payload['command'], admitted))
            return reply
        return super().request(operation, **payload)


@pytest.fixture
def game(tmp_path):
    @contextmanager
    def start(task_id='tic_tac_toe_3x3', *, track='direct_joints', budgets=None):
        root = Path(task_definition('card_room', task_id)['source_release']['path'])
        workers = []
        def factory(**kwargs):
            worker = FakeBoardWorker(**kwargs)
            workers.append(worker)
            return worker
        api = AgentService(tmp_path / task_id, release=root, registry=default_registry(root), worker_factory=factory)
        request = {'task_id': task_id, 'agent_id': 'cpu_test', 'agent_version': 'v1',
                   'agent_config_sha256': 'b' * 64, 'robot_id': 'dual_fr3_sharpa_w01', 'track': track}
        if track == 'direct_joints':
            request.update(policy_id=None, policy_version=None)
        if budgets is not None:
            request['budgets'] = budgets
        with TestClient(create_app(service=api)) as client:
            legacy_request = {**request, 'track': 'direct_joints', 'policy_id': None, 'policy_version': None}
            assert client.post('/episodes', json=legacy_request).status_code == 403
            assert not workers
            response = client.post('/board-games', json=request)
            assert response.status_code == 201, response.text
            created = response.json()
            session = api.sessions[created['episode_id']]
            tokens = {row['binding']['player_id']: row['session_token'] for row in created['players']}
            yield SimpleNamespace(client=client, api=api, session=session, worker=workers[-1],
                                  created=created, tokens=tokens, request=request)
    return start


def endpoint(g, player='player_0'):
    return f'/board-games/{g.session.spec.episode_id}/players/{player}'


def headers(g, player='player_0'):
    return {'Authorization': 'Bearer ' + g.tokens[player]}


def call(g, function, arguments=None, *, player='player_0', status=None, request_id='test/call'):
    response = g.client.post(endpoint(g, player) + '/call', headers=headers(g, player), json={
        'function': function, 'arguments': arguments or {}, 'request_id': request_id})
    if status is not None:
        assert response.status_code == status, response.text
    assert PRIVATE_MARKER not in response.text
    return response


def state(g, player='player_0'):
    response = g.client.get(endpoint(g, player) + '/state', headers=headers(g, player))
    assert response.status_code == 200, response.text
    value = response.json()
    BoardGameState.model_validate(value['board_game'])
    assert PRIVATE_MARKER not in response.text
    assert not {'commit_log', 'logical_board', 'committed', 'objects'} & value['board_game'].keys()
    return value


def intent(g, piece='black_00', q=0, r=0, *, player='player_0', status=200):
    response = g.client.post(endpoint(g, player) + '/intent', headers=headers(g, player), json={
        'type': 'placement_intent', 'piece_id': piece, 'q': q, 'r': r})
    assert response.status_code == status, response.text
    return response


def hold(g, action_id, duration=.02, *, player='player_0'):
    payload = {'action_id': action_id, 'observation_sequence': g.session.sequence, 'duration_s': duration}
    response = call(g, 'hold', payload, player=player)
    assert response.status_code in (200, 202), response.text
    assert response.json()['ok'], response.text
    join_action(g.session)
    return call(g, 'get_action', {'action_id': action_id}, player=player, status=200).json(), payload


def steps(g):
    return [payload for operation, payload in g.worker.calls if operation == 'step']


@pytest.mark.parametrize('task_id', ['tic_tac_toe_3x3', 'connect_four_5x5', 'hex7x7'])
def test_real_game_discovery_binds_existing_authenticated_routes(game, task_id):
    with game(task_id, track='system0') as g:
        response = g.client.get('/interfaces', params={'task_id': task_id})
        assert response.status_code == 200
        catalog = response.json()
        assert g.client.get('/interfaces').json() == catalog
        assert g.client.get('/interfaces', params={'task_id': 'unknown'}).status_code == 422
        common = catalog['shared_contract']
        call_route = 'POST /board-games/{episode_id}/players/{player_id}/call'
        assert common['action_endpoint'] == common['function_call_endpoint'] == call_route
        assert common['robot_guide_endpoint'] == common['ik_query_endpoint'] == call_route
        assert common['robot_guide_function'] == 'get_robot'
        assert common['ik_query_function'] == 'query_ik'
        assert common['authentication']['scheme'] == 'bearer'
        routes = {(method, route.path) for route in g.client.app.routes
                  for method in getattr(route, 'methods', ())}
        advertised = {tuple(value.split(' ', 1)) for key, value in common.items()
                      if key.endswith('_endpoint')}
        assert advertised <= routes
        for method, template in advertised:
            if '/players/' not in template:
                continue
            path = template.format(episode_id=g.created['episode_id'], player_id='player_0')
            kwargs = {'json': {}} if method == 'POST' else {}
            assert g.client.request(method, path, **kwargs).status_code == 403
        rows = {row['id']: row for row in catalog['interfaces']}
        assert rows['public_functions']['available'] is True
        assert 'agent.model_client.function_runner' in rows['public_functions']['client_command']
        assert all(not row['available'] and row['unavailable_reason']
                   for name, row in rows.items() if name != 'public_functions')
        assert all(not row.get('supported_policy_ids')
                   for name, row in rows.items() if name != 'public_functions')
        assert rows['fingertip_action']['backend_policy_ids'] == ['classical_fingertip_bimanual']
        call(g, common['robot_guide_function'], status=200)
        observation = call(g, 'observe', status=200).json()['result']
        geometry = observation['proprioception']['hand_geometries']['left']
        reply = call(g, common['ik_query_function'], {
            'observation_sequence': observation['sequence'],
            'hand_geometries': {'left': {'wrist': geometry['wrist'],
                                        'fingertip_frame_id': 'left_hand_flange'}}}, status=200)
        assert reply.json()['result']['solution_found']
        ordinary = '/episodes/' + g.created['episode_id']
        assert g.client.get(ordinary + '/functions').status_code == 403
        assert g.client.get(ordinary + '/robot').status_code == 403
        payload = {'request_id': 'ordinary-bypass', 'function': 'get_robot', 'arguments': {}}
        assert g.client.post(ordinary + '/call', json=payload).status_code == 403
        assert not steps(g)


@pytest.mark.parametrize('task_id,count', [('tic_tac_toe_3x3', 9), ('connect_four_5x5', 25), ('hex7x7', 49)])
def test_current_game_creation_publishes_owned_functions_without_private_state(game, task_id, count):
    with game(task_id) as g:
        assert len(g.worker.judge.match.inventory) == count
        rows = g.created['players']
        assert rows[0]['session_token'] != rows[1]['session_token']
        owned = [set(row['binding']['joint_names']) for row in rows]
        assert [len(names) for names in owned] == [29, 29]
        assert not owned[0] & owned[1]
        assert owned[0] | owned[1] == set(g.worker.active)
        assert 'fr3_left_joint1' in owned[0] and 'fr3_right_joint1' in owned[1]
        for player in ('player_0', 'player_1'):
            public = state(g, player)
            assert public['board_game']['turn_index'] == 0
            assert public['board_game']['current_player'] == 'player_0'
            assert public['public_rules']['revision'] == 'recoverable_physical_moves_v2'
            response = g.client.get(endpoint(g, player) + '/functions', headers=headers(g, player))
            assert response.status_code == 200
            catalog = response.json()
            assert catalog['reference'] is None
            assert '/board-games/' in catalog['functions']['call_endpoint']
            assert {'get_robot', 'hand_preset', 'query_ik', 'move_wrist', 'set_hand_targets'} <= {
                row['name'] for row in catalog['functions']['functions']}
            assert call(g, 'get_robot', player=player, status=200).json()['ok']
        assert not steps(g)
        for path in g.session.directory.rglob('*'):
            if path.is_file() and path.suffix in ('.json', '.jsonl'):
                assert all(token not in path.read_text() for token in g.tokens.values())


def test_player_credentials_block_legacy_routes_and_cross_player_access(game):
    with game() as g:
        old = '/episodes/' + g.session.spec.episode_id
        for route in ('/functions', '/observation', '/robot', '/result', '/actions/unknown'):
            assert g.client.get(old + route).status_code == 403
        for route, payload in [('/call', {'function': 'hold', 'arguments': {}}),
                               ('/actions', {}), ('/actions/unknown/cancel', {}), ('/finish', {})]:
            assert g.client.post(old + route, json=payload).status_code == 403
        for player in ('player_0', 'player_1'):
            target = endpoint(g, player)
            assert g.client.get(target + '/state').status_code == 403
            other = 'player_1' if player == 'player_0' else 'player_0'
            assert g.client.get(target + '/state', headers=headers(g, other)).status_code == 403
            assert g.client.get(target + '/state', headers={'Authorization': b'Bearer \xff'}).status_code == 403
            assert g.client.get(target + '/state', params={'token': g.tokens[player]}).status_code == 403
            assert g.client.post(target + '/call', json={'function': 'hold', 'arguments': {}}).status_code == 403
        camera = g.created['observation']['cameras'][0]
        for uri in (camera['rgb_uri'], camera['depth_uri']):
            assert g.client.get(uri).status_code == 403
            assert g.client.get(uri, headers=headers(g)).status_code == 200
        with pytest.raises(ServiceError):
            g.session.submit({'kind': 'direct_joints', 'action_id': 'bypass'})
        assert g.session.clock.action_attempts == 0 and not steps(g)


def test_failed_intent_transport_seals_aborted_result_without_inventing_a_move(game):
    with game() as g:
        g.worker.fail_intent = True
        intent(g, status=503)
        response = g.client.get(endpoint(g) + '/result', headers=headers(g))
        assert response.status_code == 200, response.text
        result = BoardGameResult.model_validate(response.json())
        assert result.board_game.status.outcome == 'aborted'
        assert result.board_game.status.end_reason == 'runtime_error'
        assert result.board_game.status.evidence_unavailable is True
        assert result.board_game.turn_index == 0 and result.board_game.pending_move is None
        assert result.episode.outcome == 'infrastructure_error'
        assert g.session.clock.action_attempts == 0 and not steps(g)


def test_cross_side_inactive_and_malformed_requests_cannot_spend_other_players_actions(game):
    with game() as g:
        base = {'action_id': 'owned', 'observation_sequence': 0, 'duration_s': .1,
                'arm_joint_positions_rad': {'fr3_right_joint1': 0.},
                'hand_joint_positions_rad': {'right_index_DIP': .42}}
        assert not call(g, 'move_joints', base, status=403).json()['ok']
        wrong_hand = {'action_id': 'wrong_hand', 'observation_sequence': 0, 'hand': 'right',
                      'hand_posture_rad': {'right_index_DIP': .4}, 'timeout_sim_s': .1}
        assert not call(g, 'set_hand_targets', wrong_hand, status=403).json()['ok']
        assert not call(g, 'hold', {'action_id': 'inactive'}, player='player_1', status=403).json()['ok']
        intent(g, 'white_00', player='player_1', status=403)
        assert g.session.clock.action_attempts == 0 and not steps(g)
        malformed = call(g, 'hold', {'action_id': 'bad'}, status=422).json()
        assert not malformed['ok'] and malformed['error']['code'] == 'invalid_arguments'
        assert g.session.clock.action_attempts == 1


def test_owned_direct_action_has_one_id_one_ledger_and_exact_retry_after_ack(game):
    with game() as g:
        original = deepcopy(g.worker.raw['robot']['joint_targets_rad'])
        args = {'action_id': 'move_once', 'observation_sequence': 0, 'duration_s': .1,
                'arm_joint_positions_rad': {'fr3_left_joint1': original['fr3_left_joint1'] + .001},
                'hand_joint_positions_rad': {'left_index_DIP': original['left_index_DIP']}}
        response = call(g, 'move_joints', args)
        assert response.status_code in (200, 202), response.text
        join_action(g.session)
        result = call(g, 'get_action', {'action_id': 'move_once'}, status=200).json()['result']
        assert result['status'] == 'completed'
        commands = deepcopy(steps(g))
        assert len(commands) == 5
        for row in commands:
            assert row['actor'] == {'player_id': 'player_0', 'turn_index': 0}
            assert all(row['command']['joint_positions_rad'][name] == value for name, value in original.items()
                       if name.startswith(('fr3_right_', 'right_')))
        replay = call(g, 'move_joints', args, status=200, request_id='retry/new-id').json()
        assert replay['result'] == result
        assert g.session.clock.action_attempts == 1 and steps(g) == commands
        changed = deepcopy(args)
        changed['duration_s'] = .2
        call(g, 'move_joints', changed, status=409)
        call(g, 'get_action', {'action_id': 'move_once'}, player='player_1', status=403)
        call(g, 'cancel_action', {'action_id': 'move_once'}, player='player_1', status=403)
        assert g.session.clock.action_attempts == 2 and steps(g) == commands


def test_system0_core_functions_share_one_actor_and_preserve_inactive_drive_targets(game):
    with game(track='system0') as g:
        guide = call(g, 'get_robot', status=200).json()['result']
        assert guide['ik_query']['available'] is True and set(guide['hands']) == {'left', 'right'}
        preset = call(g, 'hand_preset', {'preset': 'pinch_prepose', 'side': 'left', 'amount': .4}, status=200).json()['result']
        assert len(preset['hand_posture_rad']) == 22 and preset['validation']['advances_physics'] is False
        public = state(g)['observation']
        wrist = public['proprioception']['hand_geometries']['left']['wrist']
        query = call(g, 'query_ik', {'observation_sequence': g.session.sequence,
            'hand_geometries': {'left': {'wrist': wrist, 'fingertip_frame_id': 'left_hand_flange'}}}, status=200).json()['result']
        assert query['solution_found'] and not query['advances_physics']
        assert not steps(g) and g.session.clock.action_attempts == 0
        right_targets = {name: value for name, value in g.worker.raw['robot']['joint_targets_rad'].items()
                         if name.startswith(('fr3_right_', 'right_'))}
        g.worker.inactive_tracking_error = .005
        wrist = deepcopy(wrist)
        wrist['position_m'][0] += .003
        arguments = {'action_id': 'wrist_sdk', 'observation_sequence': g.session.sequence,
                     'hand': 'left', 'wrist': wrist, 'timeout_sim_s': 1.,
                     'position_tolerance_m': .0005, 'orientation_tolerance_rad': .01}
        response = call(g, 'move_wrist', arguments)
        assert response.status_code in (200, 202), response.text
        join_action(g.session)
        result = call(g, 'get_action', {'action_id': 'wrist_sdk'}, status=200).json()['result']
        assert result['status'] == 'completed', result
        assert steps(g)
        snapshot = state(g)['observation']
        posture = {name: value for name, value in snapshot['proprioception']['joint_positions_rad'].items()
                   if name.startswith('left_')}
        posture['left_thumb_CMC_AA'] += .03
        arguments = {'action_id': 'hand_sdk', 'observation_sequence': g.session.sequence,
                     'hand': 'left', 'hand_posture_rad': posture, 'timeout_sim_s': 1.}
        response = call(g, 'set_hand_targets', arguments)
        assert response.status_code in (200, 202), response.text
        join_action(g.session)
        result = call(g, 'get_action', {'action_id': 'hand_sdk'}, status=200).json()['result']
        assert result['status'] == 'completed', result
        assert g.session.clock.action_attempts == 2
        for row in steps(g):
            assert row['actor'] == {'player_id': 'player_0', 'turn_index': 0}
            assert {name: row['command']['joint_positions_rad'][name] for name in right_targets} == right_targets


def test_intent_cancel_reselection_and_waiting_never_move_a_piece_or_consume_turn(game):
    with game() as g:
        first = intent(g).json()['board_game']['pending_move']
        assert intent(g).json()['board_game']['pending_move'] == first
        revised = intent(g, 'black_01', 1, 1).json()['board_game']['pending_move']
        assert revised['attempt_index'] > first['attempt_index']
        intent(g, 'white_00', status=422)
        intent(g, 'black_99', status=422)
        intent(g, q=3, status=422)
        assert state(g)['board_game']['pending_move'] == revised
        assert not steps(g) and g.session.clock.action_attempts == 0
        cancelled = g.client.post(endpoint(g) + '/intent', headers=headers(g), json={'type': 'cancel_placement'})
        assert cancelled.status_code == 200
        assert cancelled.json()['board_game']['pending_move'] is None
        intent(g)
        hold(g, 'wait_for_real_evidence', .4)
        value = state(g)['board_game']
        assert value['turn_index'] == 0 and value['winner'] is None and not value['terminal']
        assert value['status']['retryable'] is True


def test_native_referee_receipt_stops_former_action_and_only_next_player_can_act(game):
    with game() as g:
        intent(g)
        g.worker.settled_pending = True
        result, args = hold(g, 'black_release', 1.)
        assert result['result']['status'] == 'cancelled'
        assert result['result']['termination_reason'] == 'board_turn_changed'
        assert result['result']['elapsed_sim_s'] == pytest.approx(.32)
        value = state(g)['board_game']
        assert value['turn_index'] == 1 and value['current_player'] == 'player_1'
        assert value['pending_move'] is None and value['commit_count'] == 1
        before = deepcopy(steps(g))
        replay = call(g, 'hold', args, status=200).json()
        assert replay['result'] == result['result'] and steps(g) == before
        call(g, 'hold', {'action_id': 'old_turn', 'observation_sequence': g.session.sequence, 'duration_s': .02}, status=403)
        assert g.session.clock.action_attempts == 1
        hold(g, 'white_hold', .02, player='player_1')
        assert steps(g)[-1]['actor'] == {'player_id': 'player_1', 'turn_index': 1}


def test_unrequested_piece_blocks_commit_with_recoverable_public_reason(game):
    with game() as g:
        intent(g)
        cancelled = g.client.post(endpoint(g) + '/intent', headers=headers(g), json={'type': 'cancel_placement'})
        assert cancelled.status_code == 200
        g.worker.orphan_piece = 'black_00'
        hold(g, 'recoverable_pause', .04)
        value = state(g)['board_game']
        assert value['status']['reason'] == 'unrequested_piece_on_board'
        assert value['status']['recovery_required'] and value['status']['retryable']
        assert not value['terminal'] and value['turn_index'] == 0
        assert 'black_00' not in json.dumps(value['status'])
        intent(g, 'black_00', 2, 2)
        g.worker.settled_pending = True
        hold(g, 'recover_by_reselection', .4)
        assert state(g)['board_game']['turn_index'] == 1


@pytest.mark.parametrize('ending,expected,reason', [('finish', 'aborted', 'cancelled'),
                                                  ('budget', 'truncated', 'time_limit'),
                                                  ('action_budget', 'truncated', 'step_limit'),
                                                  ('worker_error', 'aborted', 'runtime_error')])
def test_non_game_endings_do_not_award_opponent_a_win(game, ending, expected, reason):
    options = ({'budgets': {'sim_time_s': .04}} if ending == 'budget' else
               {'budgets': {'max_actions': 1}} if ending == 'action_budget' else {})
    with game(**options) as g:
        intent(g)
        if ending == 'finish':
            call(g, 'finish_episode', status=200)
        elif ending == 'budget':
            hold(g, 'exhaust', .1)
        elif ending == 'action_budget':
            hold(g, 'only_action', .02)
            call(g, 'hold', {'action_id': 'exhaust', 'observation_sequence': g.session.sequence,
                            'duration_s': .02}, status=409)
        else:
            g.worker.fail_step = True
            hold(g, 'transport_failure', .1)
        response = g.client.get(endpoint(g) + '/result', headers=headers(g))
        assert response.status_code == 200, response.text
        result = BoardGameResult.model_validate(response.json())
        assert result.board_game.status.outcome == expected
        assert result.board_game.status.end_reason == reason
        assert result.board_game.winner is None and not result.board_game.draw
        assert result.board_game.commit_count == 0
        assert result.episode.outcome != 'success'


@pytest.mark.parametrize('finalize_failure', [False, True])
def test_verified_win_is_separate_from_execution_outcome_and_sealed_reads(game, finalize_failure):
    with game() as g:
        g.worker.settled_pending = True
        for number, (piece, q, r, player) in enumerate([
                ('black_00', 0, 0, 'player_0'), ('white_00', 0, 1, 'player_1'),
                ('black_01', 1, 0, 'player_0'), ('white_01', 1, 1, 'player_1'),
                ('black_02', 2, 0, 'player_0')]):
            intent(g, piece, q, r, player=player)
            if number == 4 and finalize_failure:
                g.worker.camera_escape = True
            hold(g, f'cpu_evidence_{number}', 1., player=player)
        result = g.client.get(endpoint(g) + '/result', headers=headers(g))
        assert result.status_code == 200, result.text
        parsed = BoardGameResult.model_validate(result.json())
        assert parsed.board_game.status.outcome == 'win'
        assert parsed.board_game.winner == 'player_0' and not parsed.board_game.draw
        assert parsed.board_game.commit_count == 5
        assert parsed.episode.outcome == ('infrastructure_error' if finalize_failure else 'success')
        before = {str(path.relative_to(g.session.directory)): path.read_bytes()
                  for path in g.session.directory.rglob('*') if path.is_file()}
        for player in ('player_0', 'player_1'):
            assert state(g, player)['board_game']['winner'] == 'player_0'
            final = g.client.get(endpoint(g, player) + '/result', headers=headers(g, player))
            assert final.json() == result.json()
        after = {str(path.relative_to(g.session.directory)): path.read_bytes()
                 for path in g.session.directory.rglob('*') if path.is_file()}
        assert before == after


def test_full_non_winning_board_is_a_draw_and_never_an_opponent_win(game):
    with game() as g:
        g.worker.settled_pending = True
        cells = [(0, 0), (1, 0), (2, 0), (1, 1), (0, 1), (2, 1), (1, 2), (0, 2), (2, 2)]
        for number, (q, r) in enumerate(cells):
            player = f'player_{number % 2}'
            color = 'black' if number % 2 == 0 else 'white'
            intent(g, f'{color}_{number // 2:02d}', q, r, player=player)
            hold(g, f'cpu_draw_{number}', 1., player=player)
        response = g.client.get(endpoint(g) + '/result', headers=headers(g))
        assert response.status_code == 200, response.text
        result = BoardGameResult.model_validate(response.json())
        assert result.board_game.status.outcome == 'draw'
        assert result.board_game.winner is None and result.board_game.draw
        assert result.board_game.commit_count == 9 and result.episode.outcome == 'success'
