"""Budget-late native adjudication cannot become a public multiplayer score.

Board tests use the frozen referee with synthetic contact evidence; turn tests
use an explicit CPU receipt double. Neither is native manipulation evidence.
"""
from copy import deepcopy
import json

import pytest

from bench.paths import load_json
from bench.protocol.board_games import BoardGameResult
from bench.protocol.turn_games import TurnGameResult
from pydantic import ValidationError
from test_agent_service import ManualTime, frozen_inputs, join_action
from test_board_game_service import game as board_game, hold, intent
from test_turn_game_service import game as turn_game, call as turn_call
from test_function_response_bindings import episode_spec, game_result


@pytest.mark.parametrize('kind', ['board', 'turn'])
@pytest.mark.parametrize('terminal', ['win', 'draw'])
@pytest.mark.parametrize('offset', [1., 0., -.1],
                         ids=['late_ack', 'at_wall_deadline', 'on_time_then_slow_archive'])
def test_terminal_ack_respects_budget_without_losing_native_evidence(
        request, kind, terminal, offset):
    late = offset >= 0.
    factory = request.getfixturevalue(kind+'_game')
    with factory(track='direct_joints') as g:
        clock = ManualTime()
        g.session.clock._monotonic = clock
        g.session.clock._start = 0.
        horizon = g.session.spec.budgets.wall_time_s
        captured = []
        eligible_states = []
        original = g.worker.request

        def native_reply(operation, **payload):
            reply = original(operation, **payload)
            if operation == 'step':
                if kind == 'turn':
                    g.worker.state.update(protocol='turn_game_state_v1', status='completed', current_player_id=None,
                        terminal_reason='collapse' if terminal == 'win' else 'no_legal_moves',
                        winner_player_id='beta' if terminal == 'win' else None,
                        loser_player_id='alpha' if terminal == 'win' else None)
                    g.worker.sync_score()
                    reply.update(turn_game=deepcopy(g.worker.state), score=deepcopy(g.worker.score))
                finished = (reply['board_game']['terminal'] if kind == 'board'
                            else reply['turn_game']['status'] == 'completed')
                if finished:
                    captured.append(deepcopy(reply))
                    eligible_states.append(g.session.player_session.state.model_dump(mode='json'))
                    clock.value = horizon+offset
            elif operation == 'cameras' and captured and not late:
                # Final recording may be slow, but the accepted terminal ACK
                # stopped the active episode clock before this operation.
                clock.value = horizon+100.
            return reply

        g.worker.request = native_reply
        if kind == 'board':
            g.worker.settled_pending = True
            cells = ([(0, 0), (0, 1), (1, 0), (1, 1), (2, 0)] if terminal == 'win' else
                     [(0, 0), (1, 0), (2, 0), (1, 1), (0, 1), (2, 1), (1, 2), (0, 2), (2, 2)])
            for number, (q, r) in enumerate(cells):
                player = f'player_{number % 2}'
                color = 'black' if number % 2 == 0 else 'white'
                intent(g, f'{color}_{number // 2:02d}', q, r, player=player)
                previous = g.session.player_session.state.model_dump(mode='json')
                hold(g, f'placement_{number}', 1., player=player)
            player, last_action = 'player_0', f'placement_{len(cells)-1}'
        else:
            previous = g.session.player_session.state.model_dump(mode='json')
            response = turn_call(g, 'hold', {'action_id':'last', 'observation_sequence':0,
                                             'duration_s':.04})
            assert response.status_code in (200, 202), response.text
            join_action(g.session)
            player, last_action = 'alpha', 'last'
        assert len(captured) == 1
        raw = captured[0]
        result = g.session.player_session.result(player)
        model = BoardGameResult if kind == 'board' else TurnGameResult
        model.model_validate(result)
        public = result[kind+'_game']
        assert result['episode']['outcome'] == ('wall_timeout' if late else 'success')
        assert result['episode']['physics_valid'] is True
        assert g.session.raw == raw['observation']
        assert g.session.player_session.last_targets == g.worker.raw['robot']['joint_targets_rad']
        assert g.session.sequence == sum(op == 'step' for op, _ in g.worker.calls)
        assert g.worker.closed and g.session.done.is_set()
        assert g.session.life.active_action_id is None
        assert g.session.status(last_action)['status'] == ('failed' if late else 'completed')
        assert g.session.status(last_action)['termination_reason'] == (
            'episode_terminated' if late else 'task_terminated')
        evidence = load_json(g.session.directory/'private/evaluator_evidence.json')
        assert evidence == raw['score'] and evidence['success'] is True
        if late:
            assert public['turn_index'] == previous['turn_index']
            if kind == 'board':
                assert public['status']['outcome'] == 'truncated'
                assert public['status']['end_reason'] == 'time_limit'
                assert public['winner'] is None and public['draw'] is False
                assert public['commit_count'] == previous['commit_count']
            else:
                assert public['status'] == 'interrupted'
                assert public['terminal_reason'] == 'time_limit'
                assert public['winner_player_id'] is None and public['loser_player_id'] is None
                assert public['sim_time_s'] == raw['observation']['sim_time_s']
            receipt = load_json(g.session.directory/'private/late_game_receipt.json')
            assert receipt['native_state'] == raw[kind+'_game']
            assert receipt['native_score'] == raw['score']
            assert receipt['last_budget_state'] == eligible_states[0]
        else:
            assert public == raw[kind+'_game']
            assert result['episode']['elapsed_wall_s'] == pytest.approx(horizon-.1)
            assert not (g.session.directory/'private/late_game_receipt.json').exists()
        clock.value += 100.
        assert g.session.finish() == result['episode']
        assert g.session.player_session.result(player) == result


@pytest.mark.parametrize('kind', ['board_game', 'turn_game'])
@pytest.mark.parametrize('terminal', ['win', 'draw'])
@pytest.mark.parametrize('outcome', ['wall_timeout', 'sim_timeout', 'action_budget', 'model_budget'])
def test_public_result_contract_rejects_budget_with_competitive_outcome(kind, terminal, outcome):
    task = 'tic_tac_toe_3x3' if kind == 'board_game' else 'tower_extraction'
    value = game_result(kind, episode_spec(task))
    value['episode']['outcome'] = outcome
    state = value[kind]
    if kind == 'board_game':
        state.update(turn_index=5 if terminal == 'win' else 9,
                     commit_count=5 if terminal == 'win' else 9,
                     winner='player_0' if terminal == 'win' else None, draw=terminal == 'draw')
        state['status'].update(outcome=terminal, end_reason=None, reason=terminal)
        model = BoardGameResult
    else:
        state.update(status='completed', winner_player_id='alpha' if terminal == 'win' else None,
                     loser_player_id='beta' if terminal == 'win' else None, terminal_reason=terminal)
        model = TurnGameResult
    with pytest.raises(ValidationError, match='budget-terminated'):
        model.model_validate(value)
