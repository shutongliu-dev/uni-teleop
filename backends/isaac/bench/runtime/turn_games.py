"""Player facade for scene-owned native turn referees."""
from __future__ import annotations

from copy import deepcopy
import math
import secrets

from bench.protocol import BudgetExceeded
from bench.protocol.turn_games import (PlayerBinding, PlayerSession, TurnGameState,
                                       TurnGameResult, TurnIntent)
from .player_sessions import PlayerActionSession, _error


class TurnGameSession(PlayerActionSession):
    kind = 'turn_game'

    def __init__(self, session, initial, scene, task):
        self.session = session
        self.state = TurnGameState.model_validate(initial)
        self._check_score(self.state, session.score)
        if self.state.game_id != task['task_id'] or self.state.turn_index != 0 or self.state.terminal:
            raise ValueError('Invalid initial turn referee receipt')
        if not self._same_clock(self.state.sim_time_s, session.raw.get('sim_time_s')):
            raise ValueError('Initial turn referee clock differs from the robot snapshot')
        self.bindings = {}
        for row in scene['turn_game_task']['players']:
            side = row['robot_id']
            names = [n for n in session.capabilities.joint_limits
                     if n.startswith(side + '_') or n.startswith('fr3_' + side + '_')]
            binding = PlayerBinding(**row, hand=side, joint_names=names)
            if binding.player_id in self.bindings:
                raise ValueError('Duplicate player binding')
            self.bindings[binding.player_id] = binding
        owned = [n for binding in self.bindings.values() for n in binding.joint_names]
        if (set(self.bindings) != set(self.state.players) or len(owned) != len(set(owned))
                or set(owned) != set(session.capabilities.joint_limits)):
            raise ValueError('Player ownership must partition the native joint inventory')
        self._tokens = {p: secrets.token_urlsafe(32) for p in self.bindings}
        self.actions, self.prepared, self.call_identity = {}, {}, {}
        self._admitting_player = None
        self.last_targets = session._acknowledged_drive_targets()
        if self.last_targets is None:
            raise ValueError('Turn runtime requires acknowledged drive targets')
        self.static = {'public_rules': deepcopy(self.state.public_rules)}
        self.commit_budget_state()

    def credentials(self):
        return [PlayerSession(binding=b, session_token=self._tokens[p]).model_dump(mode='json')
                for p, b in self.bindings.items()]

    def public(self, player_id):
        with self.session.lock:
            return {'turn_game': self.state.model_dump(mode='json'),
                    'player': self.bindings[player_id].model_dump(mode='json'),
                    'observation': self.session.observe()}

    def catalog(self, player_id):
        with self.session.lock:
            value = self.session.functions.catalog().model_dump(mode='json')
            base = '/turn-games/{episode_id}/players/{player_id}'
            value['call_endpoint'] = 'POST ' + base + '/call'
            return {'player': self.bindings[player_id].model_dump(mode='json'), 'functions': value,
                    'intent_endpoint': 'POST ' + base + '/intent',
                    'state_endpoint': 'GET ' + base + '/state',
                    'result_endpoint': 'GET ' + base + '/result',
                    'direct_joint_scope': 'Only owned joints; omitted joints retain acknowledged drive targets.',
                    'turn_handoff': 'Native adjudication stops the former action at its enclosing 20 ms ACK.',
                    'reference': None}

    @staticmethod
    def _same_clock(referee_time, robot_time):
        return (type(robot_time) in (int, float) and math.isfinite(robot_time)
                and abs(referee_time - robot_time) <= 1.e-9)

    @staticmethod
    def _check_score(state, score):
        expected = {'status': state.status, 'turn_index': state.turn_index,
                    'winner_player_id': state.winner_player_id, 'loser_player_id': state.loser_player_id}
        if (not isinstance(score, dict) or type(score.get('success')) is not bool
                or (score['success'] and state.status != 'completed')
                or any(k not in score or score[k] != v for k, v in expected.items())):
            raise RuntimeError('Private evaluator score disagrees with public turn receipt')

    def acknowledge(self, reply, command=None):
        state = TurnGameState.model_validate(reply['turn_game'])
        self._check_score(state, reply['score'])
        old = self.state
        changed = state.turn_index != old.turn_index
        if (state.game_id != old.game_id or state.players != old.players
                or state.public_rules != old.public_rules or state.sim_time_s < old.sim_time_s
                or state.turn_index not in (old.turn_index, old.turn_index + 1)
                or (old.terminal and state != old)
                or (not changed and not state.terminal and state.current_player != old.current_player)
                or (changed and not state.terminal and state.current_player == old.current_player)):
            raise RuntimeError('Invalid native turn progression')
        if command is None and (changed or (state.status == 'completed' and not old.terminal)):
            raise RuntimeError('Only native robot steps may complete a turn or game')
        expected_time = (reply['observation']['sim_time_s'] if command is not None else old.sim_time_s)
        if not self._same_clock(state.sim_time_s, expected_time):
            raise RuntimeError('Turn referee clock differs from its acknowledged robot snapshot')
        if command is not None:
            if reply.get('turn_game_turn_changed') is not changed:
                raise RuntimeError('Turn ACK lacks a consistent transition receipt')
            self.last_targets = dict(command.joint_positions_rad)
        self.state = state
        self.session.score = deepcopy(reply['score'])
        return changed or (state.terminal and not old.terminal)

    def intent(self, player_id, payload):
        from .service import ServiceError
        with self.session.lock:
            self._current(player_id)
            try:
                command = TurnIntent.model_validate(payload).model_dump(mode='json')
            except (ValueError, TypeError) as exc:
                raise _error(422, 'Intent violates the advertised schema') from exc
            if self.session.life.active_action_id is not None or (self.session.thread and self.session.thread.is_alive()):
                raise _error(409, 'Wait for the current action ACK before changing intent')
            try:
                self.session.clock.check()
                if hasattr(self.session.worker, 'set_deadline'):
                    remaining = self.session.clock.remaining_wall_s
                    self.session.worker.set_deadline(None if remaining is None else max(.001, remaining))
                reply = self.session.worker.request('turn_game', command=command,
                    actor={'player_id': player_id, 'turn_index': self.state.turn_index})
                self.acknowledge(reply)
                self._record({'player_id': player_id, 'intent': command,
                              'turn_game': self.state.model_dump(mode='json'), 'error': reply.get('turn_game_error')})
                if error := reply.get('turn_game_error'):
                    # Scene error codes belong to its referee diagnostics; the
                    # shared HTTP error vocabulary stays task independent.
                    raise _error(422, error['message'], 'invalid_arguments')
                self.session.clock.check()
                self.commit_budget_state()
            except BudgetExceeded as exc:
                self.session._finish(exc.outcome)
                raise _error(409, 'Episode budget exhausted', 'budget_exhausted') from exc
            except ServiceError:
                raise
            except Exception as exc:
                self.session._finish('infrastructure_error', physics_valid=False)
                raise _error(503, 'Native turn referee failed; inspect operator diagnostics') from exc
            return self.public(player_id)

    def end(self, outcome):
        if self.state.terminal:
            if self.state.status == 'completed':
                value = self.late_terminal_state(outcome)
                if value is not None:
                    value.update(status='interrupted', current_player_id=None, winner_player_id=None,
                        loser_player_id=None, terminal_reason='time_limit', violation_reason=None,
                        sim_time_s=self.session.raw['sim_time_s'])
                    self.state = TurnGameState.model_validate(value)
            return
        reason = ('time_limit' if outcome in ('sim_timeout', 'wall_timeout') else
                  'step_limit' if outcome in ('action_budget', 'model_budget') else
                  'cancelled' if outcome == 'cancelled' else 'runtime_error')
        try:
            if getattr(self.session.worker, 'broken', False):
                raise RuntimeError('Native worker transport failed')
            if hasattr(self.session.worker, 'set_deadline'):
                self.session.worker.set_deadline(None if self.session.spec.budgets.wall_time_s is None else 30)
            reply = self.session.worker.request('turn_game', command={'type': 'end_episode', 'reason': reason})
            self.acknowledge(reply)
        except Exception:
            value = self.state.model_dump(mode='json')
            value.update(status='interrupted', current_player_id=None, winner_player_id=None,
                         loser_player_id=None, terminal_reason='runtime_error')
            self.state = TurnGameState.model_validate(value)
            raise

    def result(self, player_id):
        with self.session.lock:
            if self.session.result is None:
                raise _error(409, 'Turn game is still running', 'result_pending')
            return TurnGameResult(turn_game=self.state, episode=self.session.result).model_dump(mode='json')
