"""Episode-bound public functions and authenticated player-game transport.

Only public protocol models are imported. Player tokens stay on the host and
are attached to individual requests, including verified RGB-D downloads.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import re
import time
from typing import Literal
from uuid import uuid4

import httpx
from pydantic import Field

from bench.protocol.board_games import (BoardGameResult, BoardGameState,
                                       INTENT_ADAPTER, PlayerBinding, PlayerSession)
from bench.protocol.functions import (ACTION_FUNCTIONS, CALL_ADAPTER,
                                      RESPONSE_ADAPTER, FunctionCatalog)
from bench.protocol.schemas import (AgentObservation, Contract, EpisodeSpec,
                                    Hash, Name, RobotCapabilities)
from bench.protocol.task_referee import PublicTaskRequirements
from bench.protocol.turn_games import (PlayerBinding as TurnPlayerBinding,
                                       PlayerSession as TurnPlayerSession,
                                       TurnGameResult, TurnGameState, TurnIntent)
from .actions import ActionValidationError, public_observation
from .bench_http import BenchHTTPClient, BenchHTTPError, _token
from .time_limits import minimum_limit


_NO_ACTION_CODES = frozenset({
    'invalid_arguments', 'unsupported_function', 'unsupported_capability',
    'stale_observation', 'episode_finished', 'budget_exhausted', 'permission_denied',
})
_UNBOUND_CODES = frozenset({'permission_denied', 'not_found', 'infrastructure_error'})


def _invalid():
    return BenchHTTPError('public function response violates its contract', category='invalid_response')


def _validated(adapter, value):
    try:
        parser = getattr(adapter, 'validate_python', None) or adapter.model_validate
        return parser(value)
    except (ValueError, TypeError):
        raise _invalid() from None


def _public_observation(value, episode_id):
    # FunctionResponse serialization includes the optional privileged fields
    # as null. Remove only these empty schema defaults before the existing
    # Agent boundary; any actual privileged value still fails closed.
    if isinstance(value, dict):
        value = {key: item for key, item in value.items() if not (
            key in ('privileged_object_poses', 'privileged_cloth_vertices_m') and item is None)}
    try:
        parsed = public_observation(value)
    except (ActionValidationError, ValueError, TypeError):
        raise _invalid() from None
    if parsed.episode_id != episode_id:
        raise _invalid()
    return parsed.model_dump(mode='json', exclude_none=True)


class FunctionRejected(BenchHTTPError):
    """Typed rejection, distinct from an ambiguous transport/response failure."""

    def __init__(self, response):
        self.response = deepcopy(response)
        code = self.response['error']['code']
        # Never expose the service's free-text error, even in exception attributes.
        self.response['error']['message'] = 'Public function rejected: ' + code
        self.definite_no_action = code in _NO_ACTION_CODES
        super().__init__('Public function rejected: ' + code,
                         status_code=self.response['error']['http_status'], category='function_rejected')


class _BoardPublic(Contract):
    board_game: BoardGameState
    rules: dict
    geometry: dict
    public_rules: dict
    public_rules_sha256: Hash
    observation: AgentObservation


class _BoardState(_BoardPublic):
    player: PlayerBinding


class _BoardCreated(_BoardPublic):
    episode_id: Name
    spec: EpisodeSpec
    capabilities: RobotCapabilities
    players: list[PlayerSession] = Field(min_length=2, max_length=2, repr=False)
    reference: None = None


class _BoardCatalog(Contract):
    player: PlayerBinding
    functions: FunctionCatalog
    intent_endpoint: Literal['POST /board-games/{episode_id}/players/{player_id}/intent']
    state_endpoint: Literal['GET /board-games/{episode_id}/players/{player_id}/state']
    result_endpoint: Literal['GET /board-games/{episode_id}/players/{player_id}/result']
    direct_joint_scope: str
    turn_handoff: str
    reference: None = None


class _TurnPublic(Contract):
    turn_game: TurnGameState
    observation: AgentObservation


class _TurnState(_TurnPublic):
    player: TurnPlayerBinding


class _TurnCreated(_TurnPublic):
    episode_id: Name
    spec: EpisodeSpec
    capabilities: RobotCapabilities
    players: list[TurnPlayerSession] = Field(min_length=2, max_length=8, repr=False)
    public_rules: dict
    reference: None = None


class _TurnCatalog(Contract):
    player: TurnPlayerBinding
    functions: FunctionCatalog
    intent_endpoint: Literal['POST /turn-games/{episode_id}/players/{player_id}/intent']
    state_endpoint: Literal['GET /turn-games/{episode_id}/players/{player_id}/state']
    result_endpoint: Literal['GET /turn-games/{episode_id}/players/{player_id}/result']
    direct_joint_scope: str
    turn_handoff: str
    reference: None = None


def _validate_turn_public(parsed, episode_id):
    value = parsed.model_dump(mode='json')
    value['observation'] = _public_observation(value['observation'], episode_id)
    if abs(parsed.turn_game.sim_time_s - parsed.observation.sim_time_s) > 1.e-9:
        raise _invalid()
    return value


def _validate_turn_progress(state, old):
    if (len(state['players']) == 2
            and (state['winner_player_id'] is None) != (state['loser_player_id'] is None)):
        raise _invalid()
    if old is not None and (state['game_id'] != old['game_id']
            or state['players'] != old['players'] or state['public_rules'] != old['public_rules']
            or state['turn_index'] < old['turn_index'] or state['sim_time_s'] < old['sim_time_s']
            or (old['status'] != 'active' and state != old)
            or (state['status'] == 'active' and state['turn_index'] == old['turn_index']
                and state['current_player_id'] != old['current_player_id'])):
        raise _invalid()


def _validate_board_public(parsed, episode_id):
    value = parsed.model_dump(mode='json')
    value['observation'] = _public_observation(value['observation'], episode_id)
    try:
        digest = hashlib.sha256(json.dumps(value['public_rules'], sort_keys=True,
            separators=(',', ':'), ensure_ascii=True, allow_nan=False).encode()).hexdigest()
    except (ValueError, TypeError):
        raise _invalid() from None
    if (digest != value['public_rules_sha256']
            or value['public_rules'].get('task_id') != parsed.board_game.task_id
            or value['geometry'].get('game_id') != parsed.board_game.task_id
            or value['public_rules'].get('rules') != value['rules']
            or value['public_rules'].get('geometry') != value['geometry']):
        raise _invalid()
    return value


def create_board_game(bench: BenchHTTPClient, payload: dict, *, on_created=None) -> dict:
    """Create once; credentials in this return value must never be archived.

    The caller archives a separate public projection without ``players`` and
    passes each in-memory player session to its own FunctionHTTPClient.
    An optional trusted-host callback receives the raw response to bind cleanup
    transports before full public-state validation. It must validate identity
    and credentials itself and must not archive the raw response.
    """
    if not isinstance(payload, dict) or payload.get('task_id') not in (
            'tic_tac_toe_3x3', 'connect_four_5x5', 'hex7x7'):
        raise BenchHTTPError('board-game creation requires a supported game task', category='invalid_request')
    raw = bench._json('POST', '/board-games', payload=payload)
    if on_created is not None:
        on_created(raw)
    parsed = _validated(_BoardCreated, raw)
    episode_id = _token(parsed.episode_id)
    value = _validate_board_public(parsed, episode_id)
    bindings = [row.binding for row in parsed.players]
    joint_sets = [set(row.joint_names) for row in bindings]
    if (parsed.spec.episode_id != episode_id or parsed.spec.task_id != payload['task_id']
            or parsed.board_game.task_id != payload['task_id']
            or parsed.spec.observation_profile != 'vision_proprio'
            or parsed.board_game.terminal or parsed.board_game.turn_index != 0
            or {row.player_id for row in bindings} != {'player_0', 'player_1'}
            or parsed.players[0].session_token == parsed.players[1].session_token
            or joint_sets[0] & joint_sets[1]
            or joint_sets[0] | joint_sets[1] != set(parsed.capabilities.joint_limits)):
        raise _invalid()
    for key in ('agent_id', 'agent_version', 'agent_config_sha256', 'robot_id', 'track',
                'scene_variant', 'seed', 'observation_profile'):
        if key in payload and getattr(parsed.spec, key) != payload[key]:
            raise _invalid()
    return value


def create_turn_game(bench: BenchHTTPClient, payload: dict, *, on_created=None) -> dict:
    """Create one native turn game; keep the returned credentials host-only.

    As with ``create_board_game``, a trusted callback can bind cleanup before
    full validation. Public rules and observations are the only task feedback.
    """
    if not isinstance(payload, dict) or not isinstance(payload.get('task_id'), str):
        raise BenchHTTPError('turn-game creation requires a task ID', category='invalid_request')
    _token(payload['task_id'])
    raw = bench._json('POST', '/turn-games', payload=payload)
    if on_created is not None:
        on_created(raw)
    parsed = _validated(_TurnCreated, raw)
    episode_id = _token(parsed.episode_id)
    value = _validate_turn_public(parsed, episode_id)
    bindings = [row.binding for row in parsed.players]
    player_ids = [row.player_id for row in bindings]
    try:
        for player_id in player_ids:
            _token(player_id)
    except BenchHTTPError:
        raise _invalid() from None
    joints = [joint for row in bindings for joint in row.joint_names]
    if (parsed.spec.episode_id != episode_id or parsed.spec.task_id != payload['task_id']
            or parsed.turn_game.game_id != payload['task_id']
            or parsed.spec.observation_profile != 'vision_proprio'
            or parsed.turn_game.terminal or parsed.turn_game.turn_index != 0
            or len(set(player_ids)) != len(player_ids)
            or set(player_ids) != set(parsed.turn_game.players)
            or len({row.robot_id for row in bindings}) != len(bindings)
            or len({row.session_token for row in parsed.players}) != len(bindings)
            or len(set(joints)) != len(joints)
            or set(joints) != set(parsed.capabilities.joint_limits)
            or value['public_rules'] != value['turn_game']['public_rules']):
        raise _invalid()
    for key in ('agent_id', 'agent_version', 'agent_config_sha256', 'robot_id', 'track',
                'scene_variant', 'seed', 'observation_profile'):
        if key in payload and getattr(parsed.spec, key) != payload[key]:
            raise _invalid()
    return value


class FunctionHTTPClient(BenchHTTPClient):
    """One ordinary episode or one authenticated player; never a host credential pool."""

    def __init__(self, base_url, episode_id, *, player=None, client=None,
                 timeout_s=30., get_retries=2, session_kind=None, spec=None):
        episode_id = _token(episode_id)
        session_kind = session_kind or ('episode' if player is None else 'board_game')
        if (session_kind not in ('episode', 'board_game', 'turn_game')
                or (session_kind == 'episode') != (player is None)):
            raise BenchHTTPError('invalid player session kind', category='invalid_request')
        try:
            model = TurnPlayerSession if session_kind == 'turn_game' else PlayerSession
            session = None if player is None else model.model_validate(player)
        except (ValueError, TypeError):
            raise BenchHTTPError('invalid player session', category='invalid_request') from None
        if session is not None:
            _token(session.binding.player_id)
        super().__init__(base_url, client=client, timeout_s=timeout_s, get_retries=get_retries)
        self.episode_id = episode_id
        self.session_kind = session_kind
        self._binding = None if session is None else session.binding.model_dump(mode='json')
        self._session_token = None if session is None else session.session_token
        self._task_id = None
        self._episode_spec = None
        self._public_rules_sha256 = None
        self._scoring_contract = None
        self._board_rules_snapshot = None
        self._turn_state_snapshot = None
        game_route = 'turn-games' if session_kind == 'turn_game' else 'board-games'
        self._endpoint = (f'/episodes/{episode_id}' if session is None else
            f'/{game_route}/{episode_id}/players/{session.binding.player_id}')
        if spec is not None:
            self.bind_episode(spec)

    @property
    def binding(self):
        return deepcopy(self._binding)

    def bind_episode(self, spec, *, task=None):
        parsed = _validated(EpisodeSpec, spec)
        value = parsed.model_dump(mode='json')
        if (parsed.episode_id != self.episode_id
                or (self._episode_spec is not None and value != self._episode_spec)
                or (self._task_id is not None and parsed.task_id != self._task_id)):
            raise _invalid()
        rules_hash = None
        scoring_contract = None
        if task is not None and 'public_rules_sha256' in task:
            requirements = _validated(PublicTaskRequirements, task)
            rules_hash = requirements.public_rules_sha256
            scoring_contract = (requirements.scoring_contract.model_dump(mode='json', by_alias=True)
                                if requirements.scoring_contract is not None else None)
            if (requirements.task_id != parsed.task_id
                    or (self._public_rules_sha256 is not None
                        and (rules_hash != self._public_rules_sha256 or scoring_contract != self._scoring_contract))):
                raise _invalid()
        self._episode_spec = value
        self._task_id = parsed.task_id
        if rules_hash is not None:
            self._public_rules_sha256 = rules_hash
            self._scoring_contract = deepcopy(scoring_contract)

    def _check_scoring(self, scoring):
        if self._public_rules_sha256 is None:
            return  # Standalone callers may not have bound a public task.
        if self._scoring_contract is None:
            if scoring is not None:
                raise _invalid()
        elif scoring is None or not scoring.matches(self._scoring_contract):
            raise _invalid()

    def bind_board_state(self, state):
        """Bind creation rules without retaining host-only player credentials."""
        if self.session_kind != 'board_game':
            raise BenchHTTPError('board state requires a board-game client', category='invalid_request')
        if not isinstance(state, dict):
            raise _invalid()
        public = {key: state[key] for key in _BoardPublic.model_fields if key in state}
        self._bind_board_public(_validated(_BoardPublic, public))

    def _bind_board_public(self, parsed):
        value = _validate_board_public(parsed, self.episode_id)
        rules = {key: value[key] for key in ('public_rules', 'public_rules_sha256')}
        if ((self._task_id is not None and parsed.board_game.task_id != self._task_id)
                or (self._board_rules_snapshot is not None and rules != self._board_rules_snapshot)):
            raise _invalid()
        self._task_id = parsed.board_game.task_id
        self._board_rules_snapshot = deepcopy(rules)
        return value

    def bind_turn_state(self, state):
        """Bind validated creation/state evidence before exposing player context."""
        if self.session_kind != 'turn_game':
            raise BenchHTTPError('turn state requires a turn-game client', category='invalid_request')
        parsed = _validated(TurnGameState, state)
        if (self._binding['player_id'] not in parsed.players
                or (self._task_id is not None and parsed.game_id != self._task_id)):
            raise _invalid()
        value = parsed.model_dump(mode='json')
        _validate_turn_progress(value, self._turn_state_snapshot)
        self._task_id = parsed.game_id
        self._turn_state_snapshot = deepcopy(value)

    def _check_episode(self, episode_id):
        if episode_id != self.episode_id:
            raise BenchHTTPError('request belongs to another episode', category='invalid_request')

    def _request_headers(self, path):
        if self._session_token is None:
            return {}
        media = re.fullmatch(rf'/episodes/{re.escape(self.episode_id)}/(?:images|depth)/'
                             r'(?:0|[1-9][0-9]*)/[A-Za-z0-9_-][A-Za-z0-9_.-]{0,159}', path)
        if path not in {self._endpoint + '/' + suffix for suffix in
                        ('functions', 'state', 'intent', 'call', 'result')} and media is None:
            raise BenchHTTPError('request is outside player endpoints', category='invalid_request')
        return {'Authorization': 'Bearer ' + self._session_token}

    def _reply(self, method, path, *, payload=None, deadline=None):
        """Preserve typed non-2xx bodies; never retry a POST or follow redirects."""
        local = None if self.timeout_s is None else time.monotonic() + self.timeout_s
        deadline = minimum_limit(deadline if deadline is not None else local, self.deadline)
        attempts = self.get_retries + 1 if method == 'GET' else 1
        for attempt in range(attempts):
            remaining = minimum_limit(self.timeout_s, None if deadline is None else deadline-time.monotonic())
            if remaining is not None and remaining <= 0:
                raise BenchHTTPError('public request deadline expired', category='request_budget_exhausted')
            try:
                reply = self.client.request(method, self.base_url + path, json=payload,
                    headers=self._request_headers(path), timeout=remaining, follow_redirects=False)
            except httpx.TransportError as error:
                if attempt + 1 < attempts:
                    continue
                raise BenchHTTPError('public bench transport failed', category=(
                    'transport_timeout' if isinstance(error, httpx.TimeoutException) else 'transport_error')) from None
            if reply.status_code in (502, 503, 504) and attempt + 1 < attempts:
                continue
            try:
                value = reply.json()
            except (ValueError, UnicodeError):
                raise _invalid() from None
            if not isinstance(value, dict):
                raise _invalid()
            return reply.status_code, value
        raise AssertionError('unreachable request loop')

    def _failure(self, status, value):
        response = _validated(RESPONSE_ADAPTER, value)
        if response.ok or response.error.http_status != status:
            raise _invalid()
        if response.context is not None and response.context.episode_id != self.episode_id:
            raise _invalid()
        value = response.model_dump(mode='json')
        value['error']['message'] = 'Public function rejected: ' + response.error.code
        return value

    def _get(self, suffix):
        status, value = self._reply('GET', self._endpoint + suffix)
        if status != 200:
            if value.get('ok') is False:
                raise FunctionRejected(self._failure(status, value))
            raise BenchHTTPError('public bench request failed', status_code=status)
        return value

    def catalog(self):
        value = self._get('/functions')
        if self._binding is not None:
            wrapper = _validated(_TurnCatalog if self.session_kind == 'turn_game' else _BoardCatalog, value)
            if wrapper.player.model_dump(mode='json') != self._binding:
                raise _invalid()
            parsed = wrapper.functions
        else:
            parsed = _validated(FunctionCatalog, value)
        route = 'turn-games' if self.session_kind == 'turn_game' else 'board-games'
        expected = ('POST /episodes/{episode_id}/call' if self._binding is None else
                    f'POST /{route}/{{episode_id}}/players/{{player_id}}/call')
        names = [row.name for row in parsed.functions]
        if parsed.call_endpoint != expected or len(names) != len(set(names)):
            raise _invalid()
        return parsed.model_dump(mode='json')

    def call(self, payload, *, deadline=None):
        try:
            request = CALL_ADAPTER.validate_python(payload)
        except (ValueError, TypeError):
            raise BenchHTTPError('public function request violates its contract', category='invalid_request') from None
        status, value = self._reply('POST', self._endpoint + '/call',
                                    payload=request.model_dump(mode='json'), deadline=deadline)
        response = _validated(RESPONSE_ADAPTER, value)
        if not response.ok:
            value = self._failure(status, value)
            # Authentication/missing-session failures precede request parsing.
            unbound = (response.request_id is None and response.function is None
                       and response.context is None and response.error.code in _UNBOUND_CODES)
            if not unbound and (response.request_id != request.request_id or response.function != request.function):
                raise _invalid()
            return value
        if (response.request_id != request.request_id or response.function != request.function
                or response.context is None or response.context.episode_id != self.episode_id):
            raise _invalid()
        expected = (202 if response.function in ACTION_FUNCTIONS
                    and response.result.status in ('accepted', 'running') else 200)
        if status != expected:
            raise _invalid()
        value = response.model_dump(mode='json')
        if response.function == 'observe':
            value['result'] = _public_observation(value['result'], self.episode_id)
            if (response.result.sequence != response.context.observation_sequence
                    or response.result.sim_time_s != response.context.observation_sim_time_s):
                raise _invalid()
        if response.function in ACTION_FUNCTIONS | {'get_action', 'cancel_action'}:
            if response.result.action_id != request.arguments.action_id:
                raise _invalid()
        if response.function in ('get_result', 'finish_episode'):
            if (response.result.spec.episode_id != self.episode_id
                    or (self._episode_spec is not None
                        and response.result.spec.model_dump(mode='json') != self._episode_spec)):
                raise _invalid()
            self._check_scoring(response.result.scoring)
        if response.function == 'get_task':
            scoring_contract = (response.result.scoring_contract.model_dump(mode='json', by_alias=True)
                                if response.result.scoring_contract is not None else None)
            if ((self._task_id is not None and response.result.task_id != self._task_id)
                    or (self._public_rules_sha256 is not None
                        and (response.result.public_rules_sha256 != self._public_rules_sha256
                             or scoring_contract != self._scoring_contract))):
                raise _invalid()
            if response.result.episode.terminal:
                self._check_scoring(response.result.scoring)
        if response.function == 'query_ik':
            if (response.result.observation_sequence != request.arguments.observation_sequence
                    or response.context.observation_sequence != request.arguments.observation_sequence
                    or abs(response.result.sim_time_s - response.context.observation_sim_time_s) > 1.e-9):
                raise _invalid()
        if response.function in ('read_camera', 'project_pixel', 'transform_points'):
            if response.result.observation_sequence != request.arguments.observation_sequence:
                raise _invalid()
            if response.function in ('read_camera', 'project_pixel'):
                camera_id = (response.result.camera.camera_id if response.function == 'read_camera'
                             else response.result.camera_id)
                if camera_id != request.arguments.camera_id:
                    raise _invalid()
        return value

    def _invoke(self, function, arguments=None, *, deadline=None):
        prefix = self._binding['player_id'] if self._binding else 'agent'
        response = self.call({'request_id': f'{prefix}_request_{uuid4().hex}', 'function': function,
                              'arguments': arguments or {}}, deadline=deadline)
        if not response['ok']:
            raise FunctionRejected(response)
        return response['result']

    def _board_state(self, value):
        parsed = _validated(_BoardState, value)
        if parsed.player.model_dump(mode='json') != self._binding:
            raise _invalid()
        return self._bind_board_public(parsed)

    def _turn_state(self, value):
        parsed = _validated(_TurnState, value)
        if (parsed.player.model_dump(mode='json') != self._binding
                or self._binding['player_id'] not in parsed.turn_game.players):
            raise _invalid()
        value = _validate_turn_public(parsed, self.episode_id)
        self.bind_turn_state(value['turn_game'])
        return value

    def state(self):
        if self._binding is None:
            return {'observation': self._invoke('observe')}
        if self.session_kind == 'turn_game':
            return self._turn_state(self._get('/state'))
        return self._board_state(self._get('/state'))

    def intent(self, payload):
        if self._binding is None:
            raise BenchHTTPError('game intent requires a player session', category='invalid_request')
        try:
            request = (TurnIntent.model_validate(payload) if self.session_kind == 'turn_game'
                       else INTENT_ADAPTER.validate_python(payload))
        except (ValueError, TypeError):
            raise BenchHTTPError('game intent violates its contract', category='invalid_request') from None
        status, value = self._reply('POST', self._endpoint + '/intent', payload=request.model_dump(mode='json'))
        if status != 200:
            raise FunctionRejected(self._failure(status, value))
        return self._turn_state(value) if self.session_kind == 'turn_game' else self._board_state(value)

    def game_result(self):
        if self._binding is None:
            raise BenchHTTPError('game result requires a player session', category='invalid_request')
        status, value = self._reply('GET', self._endpoint + '/result')
        if status != 200:
            failure = self._failure(status, value)
            if status == 409 and failure['error']['code'] == 'result_pending':
                return None
            raise FunctionRejected(failure)
        parsed = _validated(TurnGameResult if self.session_kind == 'turn_game' else BoardGameResult, value)
        task_id = parsed.turn_game.game_id if self.session_kind == 'turn_game' else parsed.board_game.task_id
        if (parsed.episode.spec.episode_id != self.episode_id
                or (self._task_id is not None and task_id != self._task_id)
                or (self._episode_spec is not None
                    and parsed.episode.spec.model_dump(mode='json') != self._episode_spec)):
            raise _invalid()
        if self.session_kind == 'turn_game' and self._binding['player_id'] not in parsed.turn_game.players:
            raise _invalid()
        if self.session_kind == 'turn_game':
            self.bind_turn_state(parsed.turn_game.model_dump(mode='json'))
        return parsed.model_dump(mode='json')

    def get_observation(self, episode_id):
        self._check_episode(episode_id)
        return self.state()['observation']

    def get_action(self, episode_id, action_id, *, deadline=None):
        self._check_episode(episode_id)
        return self._invoke('get_action', {'action_id': action_id}, deadline=deadline)

    def cancel_action(self, episode_id, action_id):
        self._check_episode(episode_id)
        return self._invoke('cancel_action', {'action_id': action_id})

    def submit(self, episode_id, command):
        self._check_episode(episode_id)
        if (not isinstance(command, dict) or set(command) != {
                'action_id', 'observation_sequence', 'function', 'arguments'}
                or command.get('function') not in ACTION_FUNCTIONS
                or not isinstance(command.get('arguments'), dict)):
            raise BenchHTTPError('invalid function action command', category='invalid_request')
        arguments = deepcopy(command['arguments'])
        for key in ('action_id', 'observation_sequence'):
            if key in arguments and arguments[key] != command[key]:
                raise BenchHTTPError('function action identities disagree', category='invalid_request')
            arguments[key] = command[key]
        return self._invoke(command['function'], arguments)

    def get_result(self, episode_id):
        self._check_episode(episode_id)
        try:
            return self._invoke('get_result')
        except FunctionRejected as error:
            if error.response['error']['code'] == 'result_pending':
                return None
            raise

    def finish(self, episode_id):
        self._check_episode(episode_id)
        return self._invoke('finish_episode')

    def download_cameras(self, episode_id, observation):
        self._check_episode(episode_id)
        return super().download_cameras(episode_id, observation)
