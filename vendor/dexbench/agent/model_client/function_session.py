"""Model-facing public functions with host-owned identities and action recovery.

This layer has no simulator access and executes no model-provided Python. A
player session owns one action ledger; the scheduler must reconcile that ledger
before allowing the other player to act.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import math
from pathlib import Path
import time
from uuid import uuid4

from bench.protocol.board_games import INTENT_ADAPTER, PlayerBinding
from bench.protocol.functions import ACTION_FUNCTIONS, FUNCTION_INPUTS, FunctionFailure
from bench.protocol.turn_games import PlayerBinding as TurnPlayerBinding, TurnIntent

from .actions import public_observation
from .bench_http import BenchHTTPError, _token
from .capture import save
from .learning.journal import canonical_json, checked_directory
from .learning.robot_api import ActionPendingError, ActionRecoveryError, RobotAPI
from .time_limits import minimum_limit


_HOST_FIELDS = frozenset(('action_id', 'observation_sequence', 'request_id', 'episode_id',
                          'player_id', 'session_token', 'protocol'))
_TOWER_RESET_GUIDANCE = (
    'The tower is reset_pending, before playable turns. Read-only queries do not advance simulation. '
    'Only the current player may explicitly call hold with mode="acknowledged_drive" to let the '
    'bench check stability while keeping both robots at their acknowledged drive targets. '
    'Default/reference hold, other motion and all turn intents are unavailable during reset. '
    'Query/cancel an owned action or finish_episode remain available. After each hold inspect '
    'state.turn_game.public_progress.status; wait for await_intent before selecting a block or moving. '
    'The bench may interrupt initialization; no hold or intent guarantees readiness or a winner.')


def _turn_intent_tools(task_id):
    """Agent-owned adapters describe known public intent payloads, never physics."""
    schema = TurnIntent.model_json_schema()
    examples = [{'intent': {'type': 'turn_intent', 'name': 'advertised intent', 'arguments': {}}}]
    scope = ('Use only intents advertised in turn_game.public_rules. Intents do not advance physics '
             'or complete a turn; native referee settlement controls handoff.')
    if task_id == 'tower_extraction':
        examples = [
            {'intent': {'type': 'turn_intent', 'name': 'select',
                        'arguments': {'block_id': 'ID from state.turn_game.public_progress.legal_blocks'}}},
            {'intent': {'type': 'turn_intent', 'name': 'cancel', 'arguments': {}}},
        ]
        schema['properties']['name']['enum'] = ['select', 'cancel', 'push', 'pull', 'regrasp', 'release', 'wait']
        schema['properties']['arguments'] = {'type': 'object', 'properties': {
            'block_id': {'type': 'string', 'description': 'Block ID from state.turn_game.public_progress.legal_blocks'}},
            'additionalProperties': False}
        schema['allOf'] = [{'if': {'properties': {'name': {'const': 'select'}}},
                            'then': {'properties': {'arguments': {'required': ['block_id']}},
                                     'required': ['arguments']}}]
        schema['examples'] = [row['intent'] for row in examples]
        scope = ('Select a block ID from state.turn_game.public_progress.legal_blocks before robot contact. '
                 'Inspect selected_block and selection_locked there: first physical hand contact locks the selection; '
                 'cancel/reselection is allowed only before that lock. All intents, including push/pull/regrasp/release/wait, '
                 'are declarations without physical motion. Execute public robot functions to extract and place the block '
                 'in your collection zone; never restack it. Native stable release decides handoff.')
    return {'intent': schema, 'scope': scope}, examples


class FunctionDecisionError(ValueError):
    def __init__(self, code, message):
        self.code = code
        super().__init__(message)


def _positive_count(value, name):
    if type(value) is not int or value <= 0:
        raise ValueError(f'{name} must be a positive integer')
    return value


def _submission_rejection(error):
    if isinstance(error, BenchHTTPError) and getattr(error, 'definite_no_action', False) is True:
        return deepcopy(error.response)
    return None


class FunctionSession:
    """One model context, public identity and bounded stream of function calls."""

    def __init__(self, client, *, output: Path, spec: dict, trial_id: str,
                 max_actions: int, max_calls: int, deadline: float | None = None,
                 max_status_queries: int = 1200):
        self.client = client
        self.episode_id = _token(client.episode_id)
        self.trial_id = _token(trial_id)
        self.spec = deepcopy(spec)
        self.max_actions = _positive_count(max_actions, 'max_actions')
        self.max_calls = _positive_count(max_calls, 'max_calls')
        if deadline is not None and (type(deadline) not in (int, float) or not math.isfinite(deadline)):
            raise ValueError('deadline must be a finite monotonic timestamp or None')
        self.deadline = deadline
        binding = getattr(client, 'binding', None)
        if hasattr(binding, 'model_dump'):
            binding = binding.model_dump(mode='json')
        self.session_kind = getattr(client, 'session_kind', 'board_game' if binding else 'episode')
        if (self.session_kind not in ('episode', 'board_game', 'turn_game')
                or (self.session_kind == 'episode') != (binding is None)):
            raise ValueError('Invalid function session kind')
        binding_model = TurnPlayerBinding if self.session_kind == 'turn_game' else PlayerBinding
        self.binding = None if binding is None else binding_model.model_validate(binding).model_dump(mode='json')
        self.prefix = self.binding['player_id'] if self.binding else 'agent'
        self.output = checked_directory(output, create=True)
        self.calls = self.actions = 0
        self.state = self.observation = self.catalog = self.robot = None
        self.last_feedback = None
        self._available = set()
        self._latest_action_id = None
        host = checked_directory(self.output / 'host', create=True)
        ledger = host / 'action_journal'
        try:
            ledger.mkdir(mode=0o700)
        except FileExistsError:
            raise ActionRecoveryError('FunctionSession cannot reopen a trial; recover its original RobotAPI ledger') from None
        self.action_api = RobotAPI(client, ledger, episode_id=self.episode_id,
            trial_id=self.trial_id, request=self.http, max_status_queries=max_status_queries,
            submission_rejection=_submission_rejection,
            binding={'interface': 'public_functions_v1', 'player': self.binding,
                     'spec_sha256': hashlib.sha256(canonical_json(self.spec)).hexdigest(),
                     'max_actions': self.max_actions, 'max_calls': self.max_calls})

    @property
    def pending(self):
        return self.action_api.pending

    def remaining(self):
        if self.deadline is None:
            return None
        remaining = self.deadline - time.monotonic()
        if remaining <= 0:
            raise BenchHTTPError('Function session wall budget expired', category='request_budget_exhausted')
        return remaining

    def http(self, function, *args, **kwargs):
        self.remaining()
        previous = getattr(self.client, 'deadline', None)
        self.client.deadline = minimum_limit(previous, self.deadline)
        try:
            return function(*args, **kwargs)
        finally:
            self.client.deadline = previous

    def refresh(self):
        value = self.http(self.client.state)
        parsed = public_observation(value['observation'])
        if (parsed.episode_id != self.episode_id or
                (self.observation is not None and parsed.sequence < self.observation['sequence'])):
            raise BenchHTTPError('Public state has wrong episode or regressing sequence', category='invalid_response')
        if self.binding is not None and value.get('player') != self.binding:
            raise BenchHTTPError('Public state changed the player binding', category='invalid_response')
        if self.session_kind == 'turn_game' and value.get('turn_game', {}).get('game_id') != self.spec.get('task_id'):
            raise BenchHTTPError('Public turn state belongs to another task', category='invalid_response')
        self.state = deepcopy(value)
        self.observation = self.state['observation']
        return self.state

    def _payload(self, function, arguments):
        return {'request_id': f'{self.prefix}_request_{uuid4().hex}',
                'function': function, 'arguments': arguments}

    def initialize(self):
        self.catalog = deepcopy(self.http(self.client.catalog))
        self._available = {entry['name'] for entry in self.catalog['functions'] if entry['available']}
        if 'get_robot' not in self._available:
            raise BenchHTTPError('Function catalog does not expose get_robot', category='invalid_response')
        self.refresh()
        response = self.http(self.client.call, self._payload('get_robot', {}))
        if not response['ok']:
            self.last_feedback = response
            raise BenchHTTPError('Public robot guide unavailable', category='invalid_response')
        self.robot = deepcopy(response['result'])
        save(self.output / 'catalog.json', self.catalog)
        save(self.output / 'robot.json', self.robot)
        return self.context()

    def _tools(self):
        tools = []
        resetting = self._tower_reset_pending()
        for name, model in FUNCTION_INPUTS.items():
            if name not in self._available:
                continue
            if resetting and name in ACTION_FUNCTIONS and name != 'hold':
                continue
            schema = deepcopy(model.model_json_schema())
            schema['properties'] = {key: value for key, value in schema.get('properties', {}).items()
                                    if key not in _HOST_FIELDS}
            schema['required'] = [key for key in schema.get('required', []) if key not in _HOST_FIELDS]
            if self.binding is not None:
                self._player_schema(name, schema)
            if resetting and name == 'hold':
                schema['properties']['mode'].update(enum=['acknowledged_drive'],
                    description='During reset_pending, explicitly preserve the stationary acknowledged drive targets.')
                schema['properties']['mode'].pop('default', None)
                schema['required'].append('mode')
            tool = {'function': name, 'arguments_schema': schema}
            if resetting and name == 'hold':
                tool['scope'] = _TOWER_RESET_GUIDANCE
            if name in ('get_action', 'cancel_action'):
                tool['scope'] = 'The host selects this session\'s latest owned action; no action ID is accepted.'
            tools.append(tool)
        if self.binding is not None and not resetting:
            if self.session_kind == 'turn_game':
                tools.append(_turn_intent_tools(self.spec.get('task_id'))[0])
            else:
                tools.append({'intent': INTENT_ADAPTER.json_schema(),
                              'scope': 'Reserve or cancel a placement; physical release and referee settlement are still required.'})
        return tools

    def _player_schema(self, name, schema):
        """Project authority without hiding the full robot guide or hand targets."""
        side = self.binding['hand']
        properties = schema['properties']
        for selector in ('hand', 'side'):
            if selector in properties:
                properties[selector]['enum'] = [side]
        if name not in ('move_geometry', 'query_ik'):
            return
        properties.pop('geometry', None)
        properties['hand_geometries'] = {
            'type': 'object', 'properties': {side: {'$ref': '#/$defs/GeometryObjective'}},
            'additionalProperties': False, 'required': [side],
            'description': 'Target only the player-owned hand; an omitted fingertip frame uses that hand flange.'}
        schema['required'].append('hand_geometries')
        frame = schema['$defs']['GeometryObjective']['properties']['fingertip_frame_id']
        frame.update(enum=[f'{side}_hand_flange'], default=f'{side}_hand_flange')

    def context(self):
        if self.robot is None:
            raise RuntimeError('Initialize the function session before requesting model context')
        self.refresh()
        examples = []
        resetting = self._tower_reset_pending()
        if resetting:
            examples = [{'function': 'hold', 'arguments': {'duration_s': .2, 'mode': 'acknowledged_drive'}}]
        elif self.session_kind == 'turn_game':
            examples = _turn_intent_tools(self.spec.get('task_id'))[1]
        elif self.binding:
            examples = [{'intent': {'type': 'placement_intent', 'piece_id': 'owned piece', 'q': 0, 'r': 0}},
                        {'intent': {'type': 'cancel_placement'}}]
        return deepcopy({'spec': self.spec, 'state': self.state, 'robot': self.robot,
            'catalog': self.catalog, 'tools': self._tools(), 'last_feedback': self.last_feedback,
            'decision_schema': {'one_of': [{'function': 'advertised name', 'arguments': {}},
                *examples]},
            'host_contract': {'identities': 'Host assigns request/action IDs and binds the displayed observation sequence.',
                'action_recovery': 'An unresolved action blocks new motion and turn handoff; query or cancel only that owned action.',
                'robot_queries': 'Use query_ik and hand_preset through these functions; player commands use only the owned robot instance.',
                **({'reset_phase': _TOWER_RESET_GUIDANCE} if resetting else {}),
                **({'turn_handoff': 'Only the current player may act; multiple actions may be needed in one turn. The bench native referee decides handoff. A turn_changed or board_turn_changed cancellation confirms handoff, not completion of the requested motion.'} if self.binding else {}),
                'execution': 'A successful function response is not evidence of grasp, placement or task success.'},
            'budgets': {'calls': self.calls, 'max_calls': self.max_calls,
                        'actions': self.actions, 'max_actions': self.max_actions},
            'pending_action': self._pending_summary()})

    def _pending_summary(self):
        pending = self.pending
        return None if pending is None else {'action_id': pending['action_id'],
                                             'receipt': pending['receipt'],
                                             'cancellation_requested': pending['cancellation_requested']}

    def _owned(self, name, arguments):
        if self.binding is None:
            return
        side, joints = self.binding['hand'], set(self.binding['joint_names'])
        if name in ACTION_FUNCTIONS or name == 'finish_episode':
            if not self._is_current_player():
                raise FunctionDecisionError('permission_denied', 'Only the current player may issue motion or finish the game')
        if (self._tower_reset_pending() and name in ACTION_FUNCTIONS
                and (name != 'hold' or arguments.get('mode') != 'acknowledged_drive')):
            raise FunctionDecisionError('permission_denied',
                'Tower reset_pending permits only explicit hold(mode="acknowledged_drive"); wait for await_intent before other motion')
        if name in ('move_wrist', 'set_hand_targets') and arguments['hand'] != side:
            raise FunctionDecisionError('permission_denied', 'Targets must use the player-owned hand')
        if name == 'hand_preset' and arguments['side'] != side:
            raise FunctionDecisionError('permission_denied', 'Preset must select the player-owned hand')
        if name in ('move_geometry', 'query_ik'):
            if arguments.get('geometry') is not None or set(arguments.get('hand_geometries') or {}) != {side}:
                raise FunctionDecisionError('permission_denied', 'Use hand_geometries with only the player-owned hand')
            objectives = arguments['hand_geometries'].values()
        else:
            objectives = [arguments] if name == 'set_hand_targets' else []
        for objective in objectives:
            if objective.get('fingertip_frame_id', f'{side}_hand_flange') != f'{side}_hand_flange':
                raise FunctionDecisionError('permission_denied', 'Finger targets must use the player-owned hand frame')
            if not set(objective.get('hand_posture_rad') or {}) <= joints:
                raise FunctionDecisionError('permission_denied', 'Hand posture includes unowned joints')
        if name == 'move_joints':
            targets = arguments['arm_joint_positions_rad'] | arguments['hand_joint_positions_rad']
            if not targets or not set(targets) <= joints:
                raise FunctionDecisionError('permission_denied', 'Joint motion accepts only owned joints')

    def _arguments(self, name, supplied):
        if not isinstance(supplied, dict) or _HOST_FIELDS & supplied.keys():
            raise FunctionDecisionError('invalid_arguments', 'Arguments must omit host-owned identifiers and observation sequence')
        arguments = deepcopy(supplied)
        if self.binding is not None and name in ('move_geometry', 'query_ik'):
            side = self.binding['hand']
            geometries = arguments.get('hand_geometries')
            if isinstance(geometries, dict) and isinstance(geometries.get(side), dict):
                objective = geometries[side]
                frame = objective.setdefault('fingertip_frame_id', f'{side}_hand_flange')
                if frame != f'{side}_hand_flange':
                    raise FunctionDecisionError('permission_denied', 'Finger targets must use the player-owned hand frame')
        fields = FUNCTION_INPUTS[name].model_fields
        if 'observation_sequence' in fields:
            arguments['observation_sequence'] = self.observation['sequence']
        if name in ACTION_FUNCTIONS:
            arguments['action_id'] = f'{self.prefix}_action_{uuid4().hex}'
        elif 'action_id' in fields:
            pending = self.pending
            action_id = pending['action_id'] if pending else self._latest_action_id
            if action_id is None:
                raise FunctionDecisionError('not_found', 'This session has no owned action to inspect or cancel')
            arguments['action_id'] = action_id
        try:
            arguments = FUNCTION_INPUTS[name].model_validate(arguments).model_dump(mode='json', exclude_none=True)
        except (TypeError, ValueError):
            raise FunctionDecisionError('invalid_arguments', 'Arguments do not satisfy the advertised function schema') from None
        self._owned(name, arguments)
        return arguments

    def _wait(self):
        settled = self.action_api.wait(timeout_s=self.remaining())
        self.observation = settled['observation']
        self.refresh()
        return settled

    def _action(self, name, arguments):
        self.action_api.assert_idle()
        self._latest_action_id = arguments['action_id']
        command = {'function': name, 'arguments': arguments,
                   'action_id': arguments['action_id'],
                   'observation_sequence': arguments['observation_sequence']}
        try:
            self.action_api.submit(command, self.observation)
        except BenchHTTPError as original:
            if self.pending is None:
                raise
            try:
                settled = self._wait()
            except BenchHTTPError:
                raise original from None
        else:
            settled = self._wait()
        return {'ok': True, 'function': name, 'result': settled['receipt'],
                'observation_sequence': settled['observation']['sequence']}

    def _intent(self, value):
        if self.binding is None:
            raise FunctionDecisionError('unsupported_function', 'Ordinary episodes have no player-game intent')
        self.action_api.assert_idle()
        if self._tower_reset_pending():
            raise FunctionDecisionError('permission_denied',
                'Tower reset_pending has no turn intents; explicitly hold(mode="acknowledged_drive") and wait for await_intent')
        try:
            intent = (TurnIntent.model_validate(value) if self.session_kind == 'turn_game'
                      else INTENT_ADAPTER.validate_python(value)).model_dump(mode='json')
        except (TypeError, ValueError):
            raise FunctionDecisionError('invalid_arguments', 'Intent does not satisfy the advertised game schema') from None
        if not self._is_current_player():
            raise FunctionDecisionError('permission_denied', 'Only the current player may change game intent')
        if intent['type'] == 'placement_intent' and not intent['piece_id'].startswith(self.binding['color'] + '_'):
            raise FunctionDecisionError('permission_denied', 'Placement must select an owned piece')
        result = self.http(self.client.intent, intent)
        self.refresh()
        return {'ok': True, 'intent': intent['type'], 'result': result}

    def _is_current_player(self):
        if self.session_kind == 'turn_game':
            game = self.state.get('turn_game', {})
            return game.get('status') == 'active' and game.get('current_player_id') == self.binding['player_id']
        game = self.state.get('board_game', {})
        return not game.get('terminal') and game.get('current_player') == self.binding['player_id']

    def _tower_reset_pending(self):
        if self.session_kind != 'turn_game' or self.spec.get('task_id') != 'tower_extraction':
            return False
        game = (self.state or {}).get('turn_game', {})
        return game.get('status') == 'active' and game.get('public_progress', {}).get('status') == 'reset_pending'

    @staticmethod
    def _error_feedback(error, name):
        response = getattr(error, 'response', None)
        if isinstance(error, BenchHTTPError) and response is not None:
            try:
                return FunctionFailure.model_validate(response).model_dump(mode='json')
            except (TypeError, ValueError):
                pass
        if isinstance(error, FunctionDecisionError):
            detail = {'code': error.code, 'message': str(error)}
        elif isinstance(error, ActionPendingError):
            detail = {'code': 'action_pending', 'message': 'Resolve the owned action before issuing another command'}
        elif isinstance(error, ActionRecoveryError):
            detail = {'code': 'infrastructure_error', 'error_category': 'action_recovery_error'}
        elif isinstance(error, BenchHTTPError):
            detail = {'code': 'infrastructure_error', **error.public_metadata()}
        else:
            detail = {'code': 'invalid_arguments', 'message': 'Invalid function decision'}
        return {'ok': False, 'function': name, 'error': detail}

    def execute(self, decision: dict):
        """Execute one decision; failures stay structured and never replay POST."""
        name = decision.get('function') if isinstance(decision, dict) else None
        if not isinstance(name, str):
            name = None
        recorded_call = None
        try:
            if self.robot is None:
                raise RuntimeError('Initialize the session before executing decisions')
            if self.calls >= self.max_calls:
                raise FunctionDecisionError('budget_exhausted', 'Client function-call budget exhausted')
            self.calls += 1
            recorded_call = self.calls
            if name in ACTION_FUNCTIONS:
                if self.actions >= self.max_actions:
                    raise FunctionDecisionError('budget_exhausted', 'Client action-attempt budget exhausted')
                self.actions += 1
            if not isinstance(decision, dict):
                raise FunctionDecisionError('invalid_arguments', 'Decision must be a JSON object')
            expected = self.observation['sequence']
            self.refresh()
            if set(decision) == {'intent'}:
                feedback = self._intent(decision['intent'])
            else:
                if set(decision) != {'function', 'arguments'} or name not in self._available:
                    raise FunctionDecisionError('unsupported_function', 'Choose one advertised function and its arguments')
                if 'observation_sequence' in FUNCTION_INPUTS[name].model_fields and expected != self.observation['sequence']:
                    raise FunctionDecisionError('stale_observation', 'Observation changed after the decision; inspect the new public state')
                arguments = self._arguments(name, decision['arguments'])
                if name in ACTION_FUNCTIONS:
                    feedback = self._action(name, arguments)
                elif name in ('get_action', 'cancel_action') and self.pending is not None:
                    if name == 'cancel_action':
                        self.action_api.cancel()
                    settled = self._wait()
                    feedback = {'ok': True, 'function': name, 'result': settled['receipt']}
                else:
                    if name == 'finish_episode':
                        self.action_api.assert_idle()
                    feedback = self.http(self.client.call, self._payload(name, arguments))
                    self.refresh()
                    if name == 'get_robot' and feedback['ok']:
                        self.robot = deepcopy(feedback['result'])
        except (FunctionDecisionError, ActionRecoveryError, BenchHTTPError) as error:
            feedback = self._error_feedback(error, name)
            # A known rejection can be corrected using a newly sampled state;
            # infrastructure failure remains explicit even if refresh succeeds.
            try:
                self.refresh()
            except BenchHTTPError as refresh_error:
                feedback['refresh_error'] = refresh_error.public_metadata()
        feedback['pending_action'] = self._pending_summary()
        self.last_feedback = deepcopy(feedback)
        if recorded_call is not None:
            save(self.output / 'calls' / f'{recorded_call:06d}.json', feedback)
        return deepcopy(feedback)
