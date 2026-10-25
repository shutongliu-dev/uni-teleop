"""Independent model contexts for the explicitly labelled Hex protocol smoke.

This is a small caller-owned bridge, not another server or simulator controller.
It accepts published snapshots and a prevalidated bounded joint command. Model
output can reserve a legal move and select that exact command, never write poses.
No credentials, native state, opponent history or raw model prose are recorded.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import math
from typing import Mapping
from uuid import uuid4

from scenes.board_games.hex import HexPlayerProtocol, IllegalMove, SMOKE_MODE, validate_cell
from agent.model_client.transport import ModelGatewayError


class HexBridgeError(ValueError):
    """Stable local category; never echo provider bodies or rejected model text."""

    def __init__(self, category: str, *, gateway_error: ModelGatewayError | None = None):
        super().__init__(category)
        self.category = category
        # Use the transport's fixed-vocabulary/numeric projection, never its
        # message, request ID, exception chain, response body or headers.
        self._gateway_error_metadata = (None if gateway_error is None
                                        else ModelGatewayError.public_metadata(gateway_error))

    def public_metadata(self) -> dict:
        result = {'category': self.category}
        if self._gateway_error_metadata is not None:
            result['gateway_error'] = deepcopy(self._gateway_error_metadata)
        return result


def _json(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def _hash(value) -> str:
    return hashlib.sha256(_json(value).encode()).hexdigest()


def _number(value) -> bool:
    return type(value) in (int, float) and math.isfinite(value)


def parse_hex_decision(text: str) -> dict:
    """Exactly one JSON object, without prose, markdown, duplicate keys or NaN."""
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError('duplicate')
            result[key] = value
        return result

    def constant(_):
        raise ValueError('nonfinite')

    if not isinstance(text, str) or not text or len(text) > 32768:
        raise HexBridgeError('invalid_model_json')
    try:
        value = json.loads(text, object_pairs_hook=pairs, parse_constant=constant)
    except (ValueError, RecursionError):
        raise HexBridgeError('invalid_model_json') from None
    if not isinstance(value, dict) or set(value) != {'placement_intent', 'joint_targets'}:
        raise HexBridgeError('invalid_decision_schema')
    return value


def _public_calibration(value):
    """Only optical calibration fields; no arbitrary nested native payload."""
    if not isinstance(value, Mapping):
        raise HexBridgeError('invalid_public_calibration')
    allowed = {'intrinsics_3x3', 'world_from_camera', 'fx', 'fy', 'cx', 'cy',
               'width', 'height', 'near_m', 'far_m', 'depth_scale_m'}
    if set(value)-allowed:
        raise HexBridgeError('invalid_public_calibration')
    result = {}
    for key, item in value.items():
        if key == 'world_from_camera':
            if (not isinstance(item, Mapping)
                    or set(item) != {'frame_id', 'position_m', 'quaternion_wxyz'}
                    or item['frame_id'] != 'world'):
                raise HexBridgeError('invalid_public_calibration')
            for field, size in (('position_m', 3), ('quaternion_wxyz', 4)):
                if (not isinstance(item[field], (tuple, list)) or len(item[field]) != size
                        or not all(_number(number) for number in item[field])):
                    raise HexBridgeError('invalid_public_calibration')
            result[key] = deepcopy(dict(item))
        elif key == 'intrinsics_3x3':
            if not isinstance(item, (tuple, list)) or len(item) != 9 or not all(_number(v) for v in item):
                raise HexBridgeError('invalid_public_calibration')
            result[key] = list(item)
        elif not _number(item):
            raise HexBridgeError('invalid_public_calibration')
        else:
            result[key] = item
    return result


def _public_snapshot(snapshot):
    required = {'snapshot_id', 'simulation_time_s', 'rgb', 'depth', 'calibration'}
    if not isinstance(snapshot, Mapping) or not required.issubset(snapshot):
        raise HexBridgeError('invalid_public_snapshot')
    if (not isinstance(snapshot['snapshot_id'], str) or not snapshot['snapshot_id']
            or len(snapshot['snapshot_id']) > 256 or not _number(snapshot['simulation_time_s'])
            or snapshot['simulation_time_s'] < 0):
        raise HexBridgeError('invalid_public_snapshot')
    # These are published media references, not fields holding private arrays.
    if any(not isinstance(snapshot[key], str) or len(snapshot[key]) > 4096 for key in ('rgb', 'depth')):
        raise HexBridgeError('invalid_public_media_reference')
    return {**{key: snapshot[key] for key in required-{'calibration'}},
            'calibration': _public_calibration(snapshot['calibration'])}


@dataclass(frozen=True)
class PreparedHexTurn:
    proposal_id: str
    player_id: str
    readiness: bool
    snapshot_id: str
    turn_index: int
    state_fingerprint: str
    piece_id: str
    q: int
    r: int
    robot_id: str
    joint_positions: tuple[tuple[str, float], ...]
    duration_s: float
    response_metadata_json: str

    def decision(self) -> dict:
        return {'placement_intent': {'type': 'placement_intent', 'piece_id': self.piece_id,
                                     'q': self.q, 'r': self.r},
                'joint_targets': {'type': 'joint_targets', 'robot_id': self.robot_id,
                                  'joint_positions': dict(self.joint_positions), 'duration_s': self.duration_s}}

    def evidence(self) -> dict:
        return {'proposal_id': self.proposal_id, 'player_id': self.player_id,
                'snapshot_id': self.snapshot_id, 'turn_index': self.turn_index,
                'status': ('readiness_proposal_unexecuted_requires_current_turn_and_fresh_request'
                           if self.readiness else 'validated_proposal_unexecuted'),
                'decision': self.decision(), 'model_response': json.loads(self.response_metadata_json),
                'logical_move_committed': False, 'joint_command_dispatched': False}


class HexAgentBridge:
    """Two isolated histories over stateless Responses clients, no fallback.

    `joint_command` is supplied by the caller after its own native validation.
    It may be a tiny joint step or hold. Exact template matching limits the smoke
    to one advertised command; it is not evidence of model manipulation skill.
    """

    def __init__(self, protocol: HexPlayerProtocol, clients: Mapping[str, object], *,
                 model: str = 'gpt-6-astra', reasoning_effort: str = 'max',
                 max_output_tokens: int = 8192):
        if set(clients) != set(protocol.match.rules.players):
            raise ValueError('Exactly one client binding per authenticated player is required')
        if model != 'gpt-6-astra' or reasoning_effort != 'max':
            raise ValueError('This explicit smoke profile requires gpt-6-astra with reasoning max')
        if type(max_output_tokens) is not int or not 4096 <= max_output_tokens <= 16384:
            raise ValueError('max_output_tokens must be between 4096 and 16384')
        self.protocol, self._clients = protocol, dict(clients)
        self.model, self.reasoning_effort = model, reasoning_effort
        self.max_output_tokens = max_output_tokens
        self._history = {player: [] for player in clients}
        self._completed_responses = {player: [] for player in clients}
        self._context_ids = {player: 'hex-context-'+uuid4().hex for player in clients}
        self._owners = {owner.player_id: owner for owner in protocol.ownership}
        self._proposals: dict[str, PreparedHexTurn] = {}
        self._snapshots = {}
        self._latest_snapshot_id = None
        self.calls = {player: 0 for player in clients}

    def history(self, player_id: str) -> list[dict]:
        """Caller may archive one player's public history, never feed it to the other."""
        if player_id not in self._history:
            raise HexBridgeError('unknown_player')
        return deepcopy(self._history[player_id])

    def _state_fingerprint(self):
        match = self.protocol.match
        return _hash({'turn_index': match.rules.turn_index, 'current_player': match.rules.current_player,
                      'terminal': match.terminal, 'board': sorted((q, r, c) for (q, r), c in match.rules.board.items()),
                      'pending': None if match.pending is None else
                      [match.pending.player_id, match.pending.piece_id, match.pending.q, match.pending.r]})

    def _joint_template(self, player_id, command):
        owner = self._owners[player_id]
        if (not isinstance(command, Mapping)
                or set(command) != {'type', 'robot_id', 'joint_positions', 'duration_s'}
                or command['type'] != 'joint_targets' or command['robot_id'] != owner.robot_id):
            raise HexBridgeError('invalid_owned_joint_template')
        joints = command['joint_positions']
        if (not isinstance(joints, Mapping) or not joints or not set(joints).issubset(owner.joint_names)
                or not all(_number(value) for value in joints.values())
                or not _number(command['duration_s']) or not 0 < command['duration_s'] <= 1.):
            raise HexBridgeError('invalid_owned_joint_template')
        return {'type': 'joint_targets', 'robot_id': owner.robot_id,
                'joint_positions': {name: float(value) for name, value in joints.items()},
                'duration_s': float(command['duration_s'])}

    def _intent(self, player_id, intent):
        match = self.protocol.match
        if not isinstance(intent, Mapping) or set(intent) != {'type', 'piece_id', 'q', 'r'} or intent['type'] != 'placement_intent':
            raise HexBridgeError('invalid_placement_intent')
        try:
            q, r = validate_cell(intent['q'], intent['r'])
        except ValueError:
            raise HexBridgeError('invalid_placement_intent') from None
        piece = intent['piece_id']
        if not isinstance(piece, str) or piece not in match.inventory:
            raise HexBridgeError('unknown_piece_identity')
        if not piece.startswith(match.rules.color(player_id)+'_'):
            raise HexBridgeError('wrong_piece_owner')
        if piece in match.committed:
            raise HexBridgeError('already_committed')
        if (q, r) in match.rules.board:
            raise HexBridgeError('occupied_cell')
        if match.terminal:
            raise HexBridgeError('match_terminal')
        if match.pending is not None and (q, r) == (match.pending.q, match.pending.r):
            raise HexBridgeError('cell_reserved_by_pending_move')
        return {'type': 'placement_intent', 'piece_id': piece, 'q': q, 'r': r}

    def _response_metadata(self, response):
        if (response.model != self.model or response.requested_model != self.model
                or response.reasoning_effort != self.reasoning_effort or response.protocol != 'responses'
                or response.finish_reason != 'completed' or response.attempts != 1):
            raise HexBridgeError('unexpected_model_response_contract')
        usage = {}
        for field in ('input_tokens', 'output_tokens', 'total_tokens', 'prompt_tokens', 'completion_tokens'):
            value = response.usage.get(field)
            if _number(value) and value >= 0:
                usage[field] = value
        for field, detail in (('input_tokens_details', 'cached_tokens'), ('output_tokens_details', 'reasoning_tokens')):
            value = response.usage.get(field, {})
            if isinstance(value, Mapping) and _number(value.get(detail)) and value[detail] >= 0:
                usage[field] = {detail: value[detail]}
        if not _number(response.duration_s) or response.duration_s < 0:
            raise HexBridgeError('unexpected_model_response_contract')
        return {'model': response.model, 'requested_model': self.model,
                'reasoning_effort': self.reasoning_effort, 'reasoning_effort_verified_by_server': False,
                'usage': usage, 'duration_s': response.duration_s, 'attempts': 1,
                'protocol': 'responses', 'finish_reason': 'completed',
                'max_output_tokens': self.max_output_tokens}

    def propose(self, player_id: str, published_snapshot: Mapping, *, joint_command: Mapping,
                images: tuple[bytes, ...] = (), readiness: bool = False) -> PreparedHexTurn:
        if player_id not in self._clients or type(readiness) is not bool:
            raise HexBridgeError('unknown_player_or_mode')
        match = self.protocol.match
        if match.terminal:
            raise HexBridgeError('match_terminal')
        if not readiness:
            try:
                match.require_actor(player_id)
            except IllegalMove:
                raise HexBridgeError('player_not_current') from None
            if match.pending is not None:
                raise HexBridgeError('move_already_pending')
        snapshot = _public_snapshot(published_snapshot)
        if (not isinstance(images, (tuple, list)) or len(images) > 2
                or any(not isinstance(raw, bytes) or not raw.startswith(b'\x89PNG\r\n\x1a\n') for raw in images)):
            raise HexBridgeError('invalid_public_png_images')
        attached_images = []
        if images:
            attached_images.append({'index': 0, 'kind': 'rgb', 'published_reference': snapshot['rgb']})
        if len(images) == 2:
            scale = snapshot['calibration'].get('depth_scale_m')
            if not _number(scale) or scale <= 0:
                raise HexBridgeError('depth_visualization_scale_required')
            attached_images.append({'index': 1, 'kind': 'depth_visualization',
                                    'encoding': 'uint8_grayscale', 'meters_per_nonzero_code': scale,
                                    'zero_code': 'invalid', 'saturation_depth_m': 255*scale,
                                    'published_metric_depth_reference': snapshot['depth']})
        image_hashes = [hashlib.sha256(raw).hexdigest() for raw in images]
        snapshot_hash = _hash({'snapshot': snapshot, 'image_sha256': image_hashes})
        previous = self._snapshots.get(snapshot['snapshot_id'])
        if previous is not None and previous != snapshot_hash:
            raise HexBridgeError('snapshot_content_changed')
        self._snapshots[snapshot['snapshot_id']] = snapshot_hash
        self._latest_snapshot_id = snapshot['snapshot_id']
        command = self._joint_template(player_id, joint_command)
        observation = self.protocol.observation(player_id, snapshot)
        color = match.rules.color(player_id)
        # This static catalog includes every own identity, including used ones.
        # It exposes neither current availability nor ground-truth token poses.
        own_piece_ids = sorted(piece for piece in match.inventory if piece.startswith(color+'_'))
        state_before = self._state_fingerprint()
        request = {'schema': 'hex_model_smoke_request.v1', 'context_id': self._context_ids[player_id],
                   'mode': 'readiness_only_unexecuted' if readiness else 'current_turn_proposal',
                   'public_observation': observation, 'snapshot_image_sha256': image_hashes,
                   'attached_images': attached_images,
                   'allowed_joint_command': command,
                   'own_piece_ids': own_piece_ids,
                   'piece_id_format': color+'_NN with exactly two digits NN=00..24',
                   'piece_catalog_scope': 'static identity catalog; includes used tokens; no poses or availability',
                   'response_schema': {'placement_intent': {'type': 'placement_intent', 'piece_id': '<exact own_piece_ids string for an unused token>',
                                                            'q': '<integer 0..6>', 'r': '<integer 0..6>'},
                                       'joint_targets': command},
                   'limitations': 'Protocol smoke with caller-bounded joint command; no model manipulation score.'}
        instructions = (
            'You are one independent robot Hex player. Use only this player context and its public observation. '
            'Attached image 0 is RGB. When image 1 is attached, it is a depth visualization with the '
            'quantization, invalid value and clipping range specified in request.attached_images. '
            'Return exactly one JSON object with keys placement_intent and joint_targets, and no prose or markdown. '
            'Choose one legal unoccupied Hex cell and one unused token of your color. '
            'For piece_id, copy an exact string from request.own_piece_ids, using the color prefix, '
            'underscore and two zero-padded digits; for example red_00 or blue_00 according to your color. '
            'The catalog is static and includes already-used tokens; use your public observation/history to avoid reuse. '
            'Copy allowed_joint_command exactly as joint_targets. Never request object poses, another robot, '
            'extra fields, or a logical commit. The intent only reserves a move; native physical settlement commits it. '
            'For readiness_only_unexecuted, your proposal will not be executed; a fresh request on your actual turn is required. '
            'This bounded protocol smoke does not establish a physical placement or a completed game.')
        prompt = _json({'own_player_history': self._history[player_id], 'request': request})
        self.calls[player_id] += 1
        try:
            response = self._clients[player_id].complete(
                model=self.model, reasoning_effort=self.reasoning_effort, instructions=instructions,
                prompt=prompt, images=tuple(images), max_output_tokens=self.max_output_tokens)
        except ModelGatewayError as exc:
            raise HexBridgeError('model_request_failed_no_fallback', gateway_error=exc) from None
        except Exception:
            raise HexBridgeError('model_request_failed_no_fallback') from None
        metadata = self._response_metadata(response)
        completion = {'request_number': self.calls[player_id], 'model_response': metadata,
                      'decision_validation': 'pending', 'rejection_category': None}
        self._completed_responses[player_id].append(completion)
        try:
            if self._state_fingerprint() != state_before:
                raise HexBridgeError('match_changed_during_model_request')
            decision = parse_hex_decision(response.text)
            intent = self._intent(player_id, decision['placement_intent'])
            returned_command = self._joint_template(player_id, decision['joint_targets'])
            if returned_command != command:
                raise HexBridgeError('model_changed_bounded_joint_command')
        except HexBridgeError as exc:
            completion.update(decision_validation='rejected', rejection_category=exc.category)
            raise
        completion['decision_validation'] = 'accepted'
        proposal = PreparedHexTurn(uuid4().hex, player_id, readiness, snapshot['snapshot_id'],
                                   match.rules.turn_index, state_before, intent['piece_id'], intent['q'], intent['r'],
                                   command['robot_id'], tuple(sorted(command['joint_positions'].items())),
                                   command['duration_s'], _json(metadata))
        self._proposals[proposal.proposal_id] = proposal
        self._history[player_id].append({'request': request, 'validated_response': proposal.decision(),
                                         'proposal_id': proposal.proposal_id, 'status': proposal.evidence()['status']})
        return proposal

    def execute(self, proposal: PreparedHexTurn, *, execute_joint_targets,
                published_snapshot: Mapping) -> dict:
        """Admit a current-turn proposal once, then invoke one caller-owned command.

        Caller supplies the latest published decision-boundary snapshot and keeps
        physics paused during model calls. Callback returns are deliberately not
        copied: they may contain private native state. No rollback after dispatch.
        """
        if (not isinstance(proposal, PreparedHexTurn)
                or self._proposals.get(proposal.proposal_id) is not proposal):
            raise HexBridgeError('unknown_or_consumed_proposal')
        if proposal.readiness:
            raise HexBridgeError('readiness_proposal_cannot_execute')
        snapshot = _public_snapshot(published_snapshot)
        if (snapshot['snapshot_id'] != proposal.snapshot_id or self._latest_snapshot_id != proposal.snapshot_id
                or self._state_fingerprint() != proposal.state_fingerprint):
            raise HexBridgeError('stale_turn_or_snapshot')
        history_entry = next(item for item in self._history[proposal.player_id]
                             if item['proposal_id'] == proposal.proposal_id)
        stored_request = history_entry['request']
        if snapshot != stored_request['public_observation']['snapshot']:
            raise HexBridgeError('snapshot_content_changed')
        # Validate both components again before any reservation is admitted.
        decision = proposal.decision()
        self._intent(proposal.player_id, decision['placement_intent'])
        self._joint_template(proposal.player_id, decision['joint_targets'])
        self.protocol.match.require_actor(proposal.player_id)
        del self._proposals[proposal.proposal_id]
        committed_before = self.protocol.match.rules.turn_index
        self.protocol.admit_action(proposal.player_id, decision['placement_intent'])
        command = self.protocol.admit_action(proposal.player_id, decision['joint_targets'])
        try:
            execute_joint_targets(deepcopy(command))
        except Exception:
            history_entry['execution_status'] = 'callback_failed_pending_move_retained'
            raise HexBridgeError('joint_execution_failed_no_replay') from None
        result = {**proposal.evidence(), 'status': 'owned_joint_callback_returned',
                  'joint_command_dispatched': True,
                  'logical_move_committed': self.protocol.match.rules.turn_index > committed_before,
                  'pending_move_remains': self.protocol.match.pending is not None,
                  'native_execution_evidence': 'attach_separate_native_observer_report',
                  'model_manipulation_score_claimed': False}
        history_entry['execution_status'] = result['status']
        return result

    def run_turn(self, published_snapshot: Mapping, *, joint_command: Mapping,
                 execute_joint_targets, images: tuple[bytes, ...] = ()) -> dict:
        player = self.protocol.match.rules.current_player
        proposal = self.propose(player, published_snapshot, joint_command=joint_command, images=images)
        return self.execute(proposal, execute_joint_targets=execute_joint_targets,
                            published_snapshot=published_snapshot)

    def summary(self) -> dict:
        return {'schema': 'hex_two_agent_smoke_bridge.v1', 'model': self.model,
                'reasoning_effort': self.reasoning_effort, 'max_output_tokens': self.max_output_tokens,
                'contexts': [{'player_id': player, 'context_id': self._context_ids[player],
                              'model_calls': self.calls[player], 'validated_proposals': len(self._history[player]),
                              'completed_responses': deepcopy(self._completed_responses[player])}
                             for player in self.protocol.match.rules.players],
                'cross_player_history_shared': False, 'model_fallback': False,
                'raw_model_responses_recorded': False, 'private_native_state_in_model_inputs': False,
                'observation_mode': self.protocol.observation_mode,
                'exact_logical_board_state_in_model_inputs': self.protocol.observation_mode == SMOKE_MODE,
                'model_manipulation_score_claimed': False}
