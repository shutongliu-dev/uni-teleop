"""Equal-information player projection and disjoint actuator admission."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
import math
from typing import Mapping

from .settlement import BoardGameMatch
from .rules import contract_sha256

RGBD_MODE = 'rgbd'
SMOKE_MODE = 'protocol_smoke_board_state'


@dataclass(frozen=True)
class ActuatorOwnership:
    player_id: str
    robot_id: str
    joint_names: tuple[str, ...]

    def __post_init__(self):
        if (not self.player_id or not self.robot_id or not self.joint_names
                or len(set(self.joint_names)) != len(self.joint_names)
                or any(not isinstance(name, str) or not name for name in self.joint_names)):
            raise ValueError('Ownership requires player, articulation and unique joint names')


class BoardGamePlayerProtocol:
    """Runtime supplies identical published RGB-D snapshots to both identities.

    RGB-D/calibration references must already be public snapshot references. This
    class accepts no private native payload. It forwards only a fixed allowlist.
    Request identity must come from the authenticated session, not a model field.
    """

    def __init__(self, match: BoardGameMatch, ownership: tuple[ActuatorOwnership, ActuatorOwnership],
                 *, observation_mode: str = RGBD_MODE):
        if observation_mode not in (RGBD_MODE, SMOKE_MODE):
            raise ValueError('Unknown observation protocol')
        if ({owner.player_id for owner in ownership} != set(match.rules.players)
                or len(ownership) != 2 or len({owner.robot_id for owner in ownership}) != 2):
            raise ValueError('Two player IDs must own two distinct articulations')
        # This bench uses globally side-prefixed joints, including in composites.
        if set(ownership[0].joint_names) & set(ownership[1].joint_names):
            raise ValueError('Actuator ownership overlaps')
        self.match, self.ownership, self.observation_mode = match, ownership, observation_mode
        self._owners = {owner.player_id: owner for owner in ownership}

    def observation(self, player_id: str, published_snapshot: Mapping) -> dict:
        if player_id not in self._owners:
            raise ValueError('Unknown player session')
        required = ('snapshot_id', 'simulation_time_s', 'rgb', 'depth', 'calibration')
        if any(key not in published_snapshot for key in required):
            raise ValueError('A published synchronized RGB-D snapshot is required')
        own = self._owners[player_id]
        match = self.match
        contract = match.rule_contract
        public = {
            'schema': 'robot_board_game_player_observation.v1', 'observation_mode': self.observation_mode,
            'player_id': player_id, 'color': match.rules.color(player_id),
            'owned_actuators': asdict(own), 'turn_index': match.rules.turn_index,
            'current_player': None if match.terminal else match.rules.current_player,
            'terminal': match.terminal, 'winner': match.rules.winner, 'draw': match.rules.draw,
            'invalid_reason': 'physical_board_integrity_violation' if match.invalid_reason else None,
            'pending_move': asdict(match.pending) if match.pending else None,
            'rules': match.rules.specification(), 'geometry': match.geometry.specification(),
            'public_rules': contract, 'public_rules_sha256': contract_sha256(contract),
            'status': match.public_status(),
            'snapshot': deepcopy({key: published_snapshot[key] for key in required}),
            'physical_commit_required': True,
        }
        if self.observation_mode == SMOKE_MODE:
            public['privilege_label'] = 'logical_board_and_legal_cells_for_protocol_smoke_only_not_model_score'
            public['board'] = [{'q': q, 'r': r, 'color': color}
                               for (q, r), color in sorted(match.rules.board.items())]
            public['legal_cells'] = [list(cell) for cell in match.rules.legal_moves()]
        return public

    def admit_action(self, authenticated_player_id: str, action: Mapping) -> dict:
        """Return validated own-joint command or reserve an intent; never move a body."""
        self.match.require_actor(authenticated_player_id)
        if not isinstance(action, Mapping):
            raise ValueError('Action must be an object')
        owner = self._owners[authenticated_player_id]
        kind = action.get('type')
        if kind == 'cancel_placement':
            if set(action) != {'type'}:
                raise ValueError('cancel_placement has exactly the type field')
            self.match.cancel_move(authenticated_player_id)
            return {'type': kind, 'pending_move': None, 'physical_step': False}
        if kind == 'placement_intent':
            if set(action) != {'type', 'piece_id', 'q', 'r'}:
                raise ValueError('Placement intents have exactly type,piece_id,q,r')
            intent = self.match.begin_move(authenticated_player_id, action['piece_id'], action['q'], action['r'])
            return {'type': kind, 'pending_move': asdict(intent), 'physical_step': False}
        if kind != 'joint_targets' or set(action) != {'type', 'robot_id', 'joint_positions', 'duration_s'}:
            raise ValueError('Only placement_intent, cancel_placement and owned robot joint_targets are permitted')
        if action['robot_id'] != owner.robot_id:
            raise ValueError('Another player owns this articulation')
        targets = action['joint_positions']
        if (not isinstance(targets, Mapping) or not targets
                or not set(targets).issubset(owner.joint_names)
                or any(type(value) not in (int, float) or not math.isfinite(value) for value in targets.values())):
            raise ValueError('Joint targets must be finite named values owned by this player')
        duration = action['duration_s']
        if type(duration) not in (int, float) or not math.isfinite(duration) or not 0 < duration <= 1.:
            raise ValueError('Joint-target duration must be in (0,1] seconds')
        return {'type': kind, 'player_id': authenticated_player_id, 'robot_id': owner.robot_id,
                'joint_positions': dict(targets), 'duration_s': float(duration), 'physical_step': True}

    def specification(self) -> dict:
        return {'schema': 'robot_board_game_player_protocol.v1', 'observation_mode': self.observation_mode,
                'ownership': [asdict(owner) for owner in self.ownership],
                'public_inputs': 'same_published_synchronized_RGBD_snapshot_and_static_rules_for_both_players',
                'identity': 'two_independent_sessions_no_opponent_reasoning_or_model_transcripts',
                'step_actions': 'owned_robot_joint_targets_only',
                'placement_intent': 'reservation_only_never_a_physical_action_or_logical_commit',
                'cancel_placement': 'clear_pending_intent_only_no_physical_state_change',
                'revised_intent': 'same_turn_replacement_and_fresh_stable_dwell',
                'public_rules_sha256': contract_sha256(self.match.rule_contract),
                'invalid_action': 'rejected_without_changing_match_state_or_assigning_a_loss',
                'state_initialization': 'reset_only', 'object_pose_step_actions': False,
                'idle_player': 'holds_last_command_no_new_agent_targets',
                'runtime_obligations': ['enforce_native_joint_limits', 'sample_all_native_substeps',
                                        'no_welds_no_object_pose_playback', 'bind_authenticated_player_identity']}
