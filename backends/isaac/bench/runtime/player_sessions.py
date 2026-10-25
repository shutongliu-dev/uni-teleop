"""Shared player authentication, joint ownership and action-ledger admission.

Task referees own turn transitions. This facade only admits robot commands.
"""
from __future__ import annotations

import json
import secrets

from bench.protocol.functions import ACTION_FUNCTIONS, CALL_ADAPTER
from .functions import FunctionCallError, public_failure


def _error(status, message, code=None):
    from .service import ServiceError
    return ServiceError(status, message, code=code)


class PlayerActionSession:
    kind = 'board_game'

    def commit_budget_state(self):
        """Keep the last adjudication admitted before the episode horizon."""
        self.last_budget_state = self.state.model_copy(deep=True)

    def late_terminal_state(self, outcome):
        """Archive a late native result before projecting a budget interruption.

        Robot ACKs and the raw evaluator score remain intact. A final native
        receipt outside the horizon cannot replace the last eligible game state.
        Callers only use this for a newly completed competitive result.
        """
        if (outcome not in ('wall_timeout', 'sim_timeout')
                or self.last_budget_state.terminal):
            return None
        previous = self.last_budget_state.model_dump(mode='json')
        evidence = {'outcome': outcome, 'native_state': self.state.model_dump(mode='json'),
                    'native_score': self.session.score, 'last_budget_state': previous,
                    'acknowledged_sim_time_s': self.session.raw['sim_time_s'],
                    'elapsed_wall_s': self.session.clock.wall_time_s,
                    'budgets': self.session.spec.budgets.model_dump(mode='json')}
        (self.session.directory/'private/late_game_receipt.json').write_text(
            json.dumps(evidence, sort_keys=True, allow_nan=False)+'\n')
        return previous

    def authenticate(self, player_id, token):
        expected = self._tokens.get(player_id)
        if (not isinstance(token, str) or not token.isascii() or expected is None
                or not secrets.compare_digest(expected, token)):
            raise _error(403, 'A valid player session token is required')
        return self

    def authenticate_media(self, token):
        if (not isinstance(token, str) or not token.isascii()
                or not any(secrets.compare_digest(t, token) for t in self._tokens.values())):
            raise _error(403, 'A valid player session token is required')

    def _current(self, player_id):
        if self.state.terminal or self.session.result is not None:
            raise _error(409, 'The game has ended', 'episode_finished')
        if self.state.current_player != player_id:
            raise _error(403, 'Only the current player may act')

    def _owned_action(self, player_id, action_id):
        if self.actions.get(action_id, {}).get('player_id') != player_id:
            raise _error(403, 'This action belongs to another player or does not exist')

    def _record(self, event):
        if self.session.result is None:
            with (self.session.directory / f'private/{self.kind}_calls.jsonl').open('a') as stream:
                stream.write(json.dumps(event, allow_nan=False) + '\n')

    def _prepare(self, player_id, call):
        name, args = call.function, call.arguments.model_dump(mode='json')
        action_id = args.get('action_id')
        binding = self.bindings[player_id]
        if name in ACTION_FUNCTIONS:
            if action_id in self.actions:
                self._owned_action(player_id, action_id)
            else:
                self._current(player_id)
            if name in ('move_wrist', 'set_hand_targets') and args['hand'] != binding.hand:
                raise _error(403, 'Motion may target only the player-owned hand')
            if name == 'move_geometry':
                if args.get('geometry') is not None or set(args.get('hand_geometries') or {}) != {binding.hand}:
                    raise _error(403, 'Motion may target only the player-owned hand')
                objectives = args['hand_geometries'].values()
            else:
                objectives = [args] if name == 'set_hand_targets' else []
            for objective in objectives:
                if not set(objective.get('hand_posture_rad') or {}) <= set(binding.joint_names):
                    raise _error(403, 'Hand posture includes unowned joints')
            if name == 'move_joints':
                targets = args['arm_joint_positions_rad'] | args['hand_joint_positions_rad']
                if not targets or not set(targets) <= set(binding.joint_names):
                    raise _error(403, 'Direct motion accepts only player-owned joints')
                if (not set(args['arm_joint_positions_rad']) <= set(self.session.capabilities.arm_joints) or
                        not set(args['hand_joint_positions_rad']) <= set(self.session.capabilities.hand_joints)):
                    raise _error(422, 'Arm and hand joint categories must match the robot')
                key = json.dumps({'function': name, 'arguments': args}, sort_keys=True)
                if action_id in self.prepared:
                    original, expanded = self.prepared[action_id]
                    if key != original:
                        raise _error(409, 'action_id was already used for different content', 'action_conflict')
                else:
                    full = self.last_targets | targets
                    expanded = {**args,
                        'arm_joint_positions_rad': {n: full[n] for n in self.session.capabilities.arm_joints},
                        'hand_joint_positions_rad': {n: full[n] for n in self.session.capabilities.hand_joints}}
                return {**call.model_dump(mode='json'), 'arguments': expanded}, (action_id, key, expanded)
        if name in ('cancel_action', 'get_action'):
            self._owned_action(player_id, action_id)
        if name == 'finish_episode':
            self._current(player_id)
        return call, None

    def invoke(self, player_id, payload):
        """Authenticated caller only; bad motion still uses the common budget ledger."""
        from .service import ServiceError
        raw = payload.model_dump(mode='json') if hasattr(payload, 'model_dump') else payload
        raw = raw if isinstance(raw, dict) else {}
        with self.session.lock:
            parsed = None
            try:
                # Reject the inactive player before malformed calls can spend the
                # current player's shared action allocation.
                if isinstance(raw.get('function'), str) and raw['function'] in ACTION_FUNCTIONS:
                    args = raw.get('arguments')
                    action_id = args.get('action_id') if isinstance(args, dict) else None
                    if isinstance(action_id, str) and action_id in self.actions:
                        self._owned_action(player_id, action_id)
                        if self.state.current_player != player_id:
                            try:
                                retry = CALL_ADAPTER.validate_python(payload)
                                identity = json.dumps({'function': retry.function,
                                    'arguments': retry.arguments.model_dump(mode='json')}, sort_keys=True)
                            except (ValueError, TypeError):
                                identity = None
                            if identity is None or self.call_identity.get(action_id) != identity:
                                raise _error(403, 'A former player may only replay its unchanged accepted call')
                    else:
                        self._current(player_id)
                try:
                    parsed = CALL_ADAPTER.validate_python(payload)
                except (ValueError, TypeError):
                    # A malformed cancel/finish must not mutate state; the common
                    # dispatcher returns its typed schema failure.
                    prepared, cache = payload, None
                else:
                    prepared, cache = self._prepare(player_id, parsed)
                self._admitting_player = player_id
                response = self.session.functions.invoke(prepared)
                if parsed is not None and parsed.function in ACTION_FUNCTIONS:
                    action_id = parsed.arguments.action_id
                    if action_id in self.actions and action_id not in self.call_identity:
                        self.call_identity[action_id] = json.dumps({'function': parsed.function,
                            'arguments': parsed.arguments.model_dump(mode='json')}, sort_keys=True)
                if cache is not None and cache[0] in self.actions:
                    self.prepared[cache[0]] = cache[1:]
                return response
            except ServiceError as exc:
                if (exc.status != 403 and isinstance(raw.get('function'), str)
                        and raw['function'] in ACTION_FUNCTIONS and self.session.result is None
                        and self.state.current_player == player_id):
                    # Owned, invalid motion uses precisely the existing attempt
                    # budget and function log; permission rejection is separate.
                    functions = self.session.functions
                    functions._submitted = False
                    recorded = functions._record({'phase': 'request', 'request': raw,
                                                   'valid': parsed is not None})
                    return functions._failed(FunctionCallError(exc.code, str(exc), exc.status),
                        raw['function'], raw.get('request_id'), recorded, raw)
                response = public_failure(code=exc.code, message=str(exc), status=exc.status,
                    function=raw.get('function'), request_id=raw.get('request_id'))
                self._record({'player_id': player_id, 'denied': response.model_dump(mode='json')})
                return response
            finally:
                self._admitting_player = None

    def admit(self, payload):
        """Backstop in EpisodeSession.submit, including direct Python misuse."""
        player_id = self._admitting_player
        if player_id not in self.bindings:
            raise _error(403, 'Player motion requires a player session')
        action_id = payload.get('action_id') if isinstance(payload, dict) else None
        if isinstance(action_id, str) and action_id in self.actions:
            self._owned_action(player_id, action_id)
        else:
            self._current(player_id)

    def accepted(self, action_id):
        self.actions[action_id] = {'player_id': self._admitting_player, 'turn_index': self.state.turn_index}
        self._record({'action_id': action_id, 'actor': self.actions[action_id]})

    def command_actor(self, action_id, command):
        actor = self.actions[action_id]
        self._current(actor['player_id'])
        if actor['turn_index'] != self.state.turn_index:
            raise _error(403, 'The action belongs to a previous turn')
        owned = set(self.bindings[actor['player_id']].joint_names)
        for name, value in command.joint_positions_rad.items():
            if name not in owned and abs(value - self.last_targets[name]) > 1e-7:
                raise RuntimeError('Controller attempted to change an inactive player drive target')
            if name not in owned:
                command.joint_positions_rad[name] = self.last_targets[name]
        return dict(actor)

