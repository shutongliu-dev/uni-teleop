"""Adversarial private receipts and public retries; CPU-only protocol evidence."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from bench.protocol.board_games import BoardGameResult
from bench.runtime.controllers import default_registry
from bench.runtime.service import AgentService, CreateBoardGameRequest
from scenes.workrooms.catalog import task_definition
from test_board_game_service import (FakeBoardWorker, call, endpoint, game, headers,
                                     hold, intent, state, steps)


def final_result(g):
    response = g.client.get(endpoint(g) + '/result', headers=headers(g))
    assert response.status_code == 200, response.text
    result = BoardGameResult.model_validate(response.json())
    assert result.episode.outcome == 'infrastructure_error'
    assert result.board_game.status.outcome == 'aborted'
    assert result.board_game.winner is None and not result.board_game.draw
    return result


@pytest.mark.parametrize('damage', [{'success': True}, {'success': 0},
                                   {'outcome': 'win'}, {'turn_index': 1}])
def test_initial_score_disagreement_never_creates_a_player_session(tmp_path, damage):
    root = Path(task_definition('card_room', 'tic_tac_toe_3x3')['source_release']['path'])
    workers = []
    def factory(**kwargs):
        worker = FakeBoardWorker(**kwargs)
        worker.initial['score'].update(damage)
        workers.append(worker)
        return worker
    api = AgentService(tmp_path / 'episodes', release=root,
        registry=default_registry(root), worker_factory=factory)
    request = CreateBoardGameRequest(task_id='tic_tac_toe_3x3', agent_id='receipt_test',
        agent_version='v1', agent_config_sha256='a' * 64, track='direct_joints',
        policy_id=None, policy_version=None)
    with pytest.raises(RuntimeError, match='score disagrees'):
        api.create(request, board_game=True)
    assert api.sessions == {}
    assert len(workers) == 1 and workers[0].closed and workers[0].closed_force
    assert not [row for row in workers[0].calls if row[0] == 'step']


@pytest.mark.parametrize('operation', ['step', 'intent'])
@pytest.mark.parametrize('damage', [{'success': True}, {'winner': 'player_0'},
                                   {'outcome': 'win'}])
def test_score_disagreement_during_execution_is_an_infrastructure_abort(game, monkeypatch, operation, damage):
    with game() as g:
        original = g.worker.request
        def corrupted(op, **payload):
            reply = original(op, **payload)
            if (op == 'step' and operation == 'step') or (op == 'board_game' and operation == 'intent'
                    and payload['command']['type'] == 'placement_intent'):
                reply['score'].update(damage)
            return reply
        monkeypatch.setattr(g.worker, 'request', corrupted)
        if operation == 'step':
            hold(g, 'bad_score')
        else:
            intent(g, status=503)
        result = final_result(g)
        assert result.board_game.turn_index == 0
        assert g.session.done.is_set() and g.session.recorder.closed


@pytest.mark.parametrize('damage', ['missing_state', 'invalid_state', 'score_disagreement'])
def test_bad_step_receipt_preserves_last_acknowledged_sequence_time_and_clock(game, monkeypatch, damage):
    with game() as g:
        previous_sequence = g.session.sequence
        previous_time = g.session.raw['sim_time_s']
        previous_q = deepcopy(g.session.raw['joint_positions_rad'])
        original = g.worker.request
        def corrupted(op, **payload):
            reply = original(op, **payload)
            if op == 'step':
                if damage == 'missing_state':
                    del reply['board_game']
                elif damage == 'invalid_state':
                    reply['board_game']['turn_index'] = 'invalid'
                else:
                    reply['score']['success'] = True
            return reply
        monkeypatch.setattr(g.worker, 'request', corrupted)
        hold(g, 'bad_ack')
        result = final_result(g)
        assert len(steps(g)) == 1 and g.worker.raw['sim_time_s'] > previous_time
        assert g.session.sequence == g.session.snapshot.sequence == previous_sequence
        assert g.session.raw['sim_time_s'] == g.session.snapshot.sim_time_s == previous_time
        assert g.session.raw['joint_positions_rad'] == previous_q
        assert g.session.clock.sim_time_s == result.episode.elapsed_sim_s == previous_time
        assert result.board_game.turn_index == 0
        # The subsequent end RPC can still return valid native referee
        # evidence; it cannot turn this failed execution receipt into validity.
        assert result.episode.physics_valid is False


def test_score_consistent_but_spontaneous_no_step_win_is_rejected(game, monkeypatch):
    with game() as g:
        intent(g)
        g.worker.settled_pending = True
        hold(g, 'first_move', 1.)
        assert state(g)['board_game']['turn_index'] == 1
        before = deepcopy(steps(g))
        original = g.worker.request
        def corrupted(op, **payload):
            reply = original(op, **payload)
            if op == 'board_game' and payload['command']['type'] == 'placement_intent':
                board = reply['board_game']
                board.update(terminal=True, current_player=None, winner='player_0', pending_move=None)
                board['status'].update(outcome='win', end_reason=None, reason='committed', retryable=False)
                reply['score'].update(success=True, outcome='win', winner='player_0',
                                      end_reason=None, reason='committed')
            return reply
        monkeypatch.setattr(g.worker, 'request', corrupted)
        intent(g, piece='white_00', q=1, r=1, player='player_1', status=503)
        result = final_result(g)
        assert result.board_game.turn_index == 1 and steps(g) == before


def test_former_player_can_only_replay_exact_call_without_spending_next_players_budget(game):
    with game() as g:
        intent(g)
        g.worker.settled_pending = True
        result, original = hold(g, 'black_action', 1.)
        before, attempts = deepcopy(steps(g)), g.session.clock.action_attempts
        assert attempts == 1 and state(g)['board_game']['current_player'] == 'player_1'
        replay = call(g, 'hold', original, status=200, request_id='new-correlation-id')
        assert replay.json()['result'] == result['result']
        for arguments in ({'action_id': original['action_id']},
                          {**original, 'duration_s': .2},
                          {**original, 'duration_s': 'bad'},
                          {**original, 'observation_sequence': g.session.sequence}):
            response = call(g, 'hold', arguments, status=403)
            assert response.json()['error']['code'] == 'permission_denied'
            assert g.session.clock.action_attempts == attempts and steps(g) == before
        call(g, 'move_joints', {'action_id': original['action_id']}, status=403)
        assert g.session.clock.action_attempts == attempts and steps(g) == before
        hold(g, 'white_action', .02, player='player_1')
        assert g.session.clock.action_attempts == attempts + 1


def test_current_owned_changed_id_conflict_uses_one_common_attempt_and_no_second_motion(game):
    with game() as g:
        targets = g.worker.raw['robot']['joint_targets_rad']
        args = dict(action_id='own_direct', observation_sequence=0, duration_s=.1,
            arm_joint_positions_rad={'fr3_left_joint1': targets['fr3_left_joint1']},
            hand_joint_positions_rad={'left_index_DIP': targets['left_index_DIP']})
        response = call(g, 'move_joints', args)
        assert response.status_code in (200, 202)
        from test_agent_service import join_action
        join_action(g.session)
        before = deepcopy(steps(g))
        assert g.session.clock.action_attempts == 1
        changed = {**args, 'duration_s': .2}
        response = call(g, 'move_joints', changed, status=409)
        assert response.json()['error']['code'] == 'action_conflict'
        assert response.json()['context']['budget']['action_attempts'] == 2
        assert g.session.clock.action_attempts == 2 and steps(g) == before
        assert set(g.session.requests) == {'own_direct'}
        attempts = [json.loads(row) for row in (g.session.directory / 'private/action_attempts.jsonl').read_text().splitlines()]
        assert len(attempts) == 2 and attempts[1]['payload']['function'] == 'move_joints'
        call(g, 'move_joints', args, status=200)
        assert g.session.clock.action_attempts == 2 and steps(g) == before
