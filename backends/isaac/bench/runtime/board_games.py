"""Authenticated player facade over one trusted episode and action ledger.

The native worker owns the referee. This facade only admits owned commands and
projects its public receipt; it never observes object poses or implements play.
"""
from __future__ import annotations

from copy import deepcopy
import secrets

from bench.protocol import BudgetExceeded
from bench.protocol.board_games import (BoardGameState, BoardGameResult, PlayerBinding,
                                       PlayerSession, INTENT_ADAPTER)
from .player_sessions import PlayerActionSession


def _error(status, message, code=None):
    from .service import ServiceError
    return ServiceError(status, message, code=code)


class BoardGameSession(PlayerActionSession):
    def __init__(self, session, initial, scene, task):
        self.session = session
        self.state = BoardGameState.model_validate(initial)
        self._check_score(self.state, session.score)
        if self.state.task_id != task['task_id'] or self.state.turn_index != 0 or self.state.terminal:
            raise ValueError('Invalid initial board referee receipt')
        self.bindings = {}
        for row in scene['board_game_task']['players']:
            side = row['robot_id']
            names = [n for n in session.capabilities.joint_limits
                     if n.startswith(side + '_') or n.startswith('fr3_' + side + '_')]
            binding = PlayerBinding(**row, hand=side, joint_names=names)
            self.bindings[binding.player_id] = binding
        if (set(self.bindings) != {'player_0', 'player_1'} or
                set(self.bindings['player_0'].joint_names) & set(self.bindings['player_1'].joint_names) or
                set().union(*(set(b.joint_names) for b in self.bindings.values())) != set(session.capabilities.joint_limits)):
            raise ValueError('Board player ownership must partition the native joint inventory')
        self._tokens = {player: secrets.token_urlsafe(32) for player in self.bindings}
        self.actions = {}
        self.prepared = {}
        self.call_identity = {}
        self._admitting_player = None
        self.last_targets = session._acknowledged_drive_targets()
        if self.last_targets is None:
            raise ValueError('Board runtime requires acknowledged drive targets')
        self.static = {key: deepcopy(task[key]) for key in
                       ('rules', 'geometry', 'public_rules', 'public_rules_sha256')}
        self.commit_budget_state()

    def credentials(self):
        return [PlayerSession(binding=b, session_token=self._tokens[p]).model_dump(mode='json')
                for p, b in self.bindings.items()]

    def public(self, player_id):
        with self.session.lock:
            observation = self.session.observe()
            return {'board_game': self.state.model_dump(mode='json'),
                    'player': self.bindings[player_id].model_dump(mode='json'),
                    **deepcopy(self.static), 'observation': observation}

    def catalog(self, player_id):
        with self.session.lock:
            value = self.session.functions.catalog().model_dump(mode='json')
            value['call_endpoint'] = 'POST /board-games/{episode_id}/players/{player_id}/call'
            return {'player': self.bindings[player_id].model_dump(mode='json'), 'functions': value,
                    'intent_endpoint': 'POST /board-games/{episode_id}/players/{player_id}/intent',
                    'state_endpoint': 'GET /board-games/{episode_id}/players/{player_id}/state',
                    'result_endpoint': 'GET /board-games/{episode_id}/players/{player_id}/result',
                    'direct_joint_scope': 'Submit only owned joints; the bench fills all omitted joints from acknowledged drive targets.',
                    'turn_handoff': 'Native substep adjudication; stop the former action at the enclosing 20 ms ACK. No subsequent former-player command is admitted.',
                    'reference': None}

    def acknowledge(self, reply, command=None):
        state = BoardGameState.model_validate(reply['board_game'])
        self._check_score(state, reply['score'])
        if state.task_id != self.state.task_id or state.turn_index not in (self.state.turn_index, self.state.turn_index + 1):
            raise RuntimeError('Invalid board referee progression')
        changed = state.turn_index != self.state.turn_index
        if self.state.terminal and state != self.state:
            raise RuntimeError('An accepted terminal referee state is immutable')
        if command is None and changed:
            raise RuntimeError('Only a native robot step can commit a board move')
        if state.status.outcome in ('win', 'draw') and not self.state.terminal and not changed:
            raise RuntimeError('A game result requires a newly committed native move')
        if command is not None:
            if reply.get('board_game_turn_changed') is not changed:
                raise RuntimeError('Board ACK lacks a consistent turn transition receipt')
            self.last_targets = dict(command.joint_positions_rad)
        self.state = state
        self.session.score = deepcopy(reply['score'])
        return changed

    @staticmethod
    def _check_score(state, score):
        expected = {'success': state.status.outcome in ('win', 'draw'),
                    'outcome': state.status.outcome, 'winner': state.winner,
                    'draw': state.draw, 'turn_index': state.turn_index,
                    'end_reason': state.status.end_reason, 'reason': state.status.reason}
        if (not isinstance(score, dict) or type(score.get('success')) is not bool
                or any(key not in score or score[key] != value for key, value in expected.items())):
            raise RuntimeError('Private evaluator score disagrees with its public board receipt')

    def intent(self, player_id, payload):
        from .service import ServiceError
        with self.session.lock:
            self._current(player_id)
            try:
                command = INTENT_ADAPTER.validate_python(payload).model_dump(mode='json')
            except (ValueError, TypeError) as exc:
                raise _error(422, 'Placement intent violates the advertised schema') from exc
            if self.session.life.active_action_id is not None or (self.session.thread and self.session.thread.is_alive()):
                raise _error(409, 'Wait for the current action ACK before changing placement intent')
            try:
                self.session.clock.check()
                if hasattr(self.session.worker, 'set_deadline'):
                    remaining = self.session.clock.remaining_wall_s
                    self.session.worker.set_deadline(None if remaining is None else max(.001, remaining))
                reply = self.session.worker.request('board_game', command=command,
                    actor={'player_id': player_id, 'turn_index': self.state.turn_index})
                self.acknowledge(reply)
                self._record({'player_id': player_id, 'intent': command, 'board_game': self.state.model_dump(mode='json'),
                              'error': reply.get('board_game_error')})
                if error := reply.get('board_game_error'):
                    raise _error(422, error['message'], error['code'])
                self.session.clock.check()
                self.commit_budget_state()
            except BudgetExceeded as exc:
                self.session._finish(exc.outcome)
                raise _error(409, 'Episode budget exhausted', 'budget_exhausted') from exc
            except ServiceError:
                raise
            except Exception as exc:
                self.session._finish('infrastructure_error', physics_valid=False)
                raise _error(503, 'Native board referee failed; inspect operator diagnostics') from exc
            return self.public(player_id)

    def end(self, outcome):
        reason = ('time_limit' if outcome in ('sim_timeout', 'wall_timeout') else
                  'step_limit' if outcome in ('action_budget', 'model_budget') else
                  'cancelled' if outcome == 'cancelled' else 'runtime_error')
        if self.state.terminal:
            if self.state.status.outcome in ('win', 'draw'):
                value = self.late_terminal_state(outcome)
                if value is not None:
                    value.update(terminal=True, current_player=None, winner=None, draw=False,
                                 pending_move=None, settlement_dwell_s=0.)
                    value['status'].update(outcome='truncated', end_reason='time_limit',
                        retryable=False, recovery_required=False, evidence_unavailable=False, reason='time_limit')
                    self.state = BoardGameState.model_validate(value)
            return
        try:
            if getattr(self.session.worker, 'broken', False):
                raise RuntimeError('Native worker transport failed')
            if hasattr(self.session.worker, 'set_deadline'):
                self.session.worker.set_deadline(None if self.session.spec.budgets.wall_time_s is None else 30)
            reply = self.session.worker.request('board_game', command={'type': 'end_episode', 'reason': reason})
            self.acknowledge(reply)
            self.session.score = reply['score']
        except Exception:
            # No native evidence is invented if transport was lost. The last
            # acknowledged referee state remains the sole accepted move history.
            state = self.state.model_dump(mode='json')
            state.update(terminal=True, current_player=None, settlement_dwell_s=0.)
            state['status'].update(outcome='aborted', end_reason='runtime_error', retryable=False,
                                   evidence_unavailable=True, reason='runtime_error')
            self.state = BoardGameState.model_validate(state)
            raise

    def result(self, player_id):
        with self.session.lock:
            if self.session.result is None:
                raise _error(409, 'Board game is still running', 'result_pending')
            return BoardGameResult(board_game=self.state, episode=self.session.result).model_dump(mode='json')
