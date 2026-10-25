"""Private native evidence gate. This module never writes a physical pose.

Evidence must come from each consecutive native integration substep, not from
model-produced JSON or the logical board. A complete frame includes every token.
Scene/runtime binding and real contact qualification remain separate obligations.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Mapping

from .geometry import BoardGameGeometry, geometry_for
from .rules import BoardGameRules, IllegalMove, public_rule_contract


@dataclass(frozen=True)
class SettlementThresholds:
    dwell_s: float = .300
    maximum_linear_speed_m_s: float = .015
    maximum_angular_speed_rad_s: float = .100
    maximum_tilt_rad: float = math.radians(5)
    bottom_z_tolerance_m: float = .003
    edge_clearance_m: float = .001
    minimum_support_impulse_ns: float = 1.e-8
    maximum_robot_contact_impulse_ns: float = 1.e-8
    maximum_penetration_m: float = .001
    maximum_sample_gap_s: float = .020

    def __post_init__(self):
        if any(not math.isfinite(value) or value <= 0 for value in asdict(self).values()):
            raise ValueError('Settlement thresholds must be finite and positive')


@dataclass(frozen=True)
class PieceEvidence:
    position_m: tuple[float, float, float]
    quaternion_wxyz: tuple[float, float, float, float]
    linear_velocity_m_s: tuple[float, float, float]
    angular_velocity_rad_s: tuple[float, float, float]
    board_support_impulse_ns: float
    all_robot_contact_impulse_ns: float
    # None is unavailable evidence, never an inferred zero penetration.
    maximum_penetration_m: float | None = None


@dataclass(frozen=True)
class NativeFrame:
    substep_index: int
    simulation_time_s: float
    dt_s: float
    pieces: Mapping[str, PieceEvidence]
    contacts_complete: bool
    source: str = 'native_solver_substeps'


@dataclass(frozen=True)
class PlacementIntent:
    player_id: str
    piece_id: str
    q: int
    r: int
    turn_index: int
    attempt_index: int = 0


@dataclass(frozen=True)
class SettlementResult:
    accepted: bool
    reason: str
    dwell_s: float
    committed: bool = False
    invalidated: bool = False
    recovery_required: bool = False
    evidence_unavailable: bool = False


class PhysicalSettlementVerifier:
    def __init__(self, geometry: BoardGameGeometry, thresholds: SettlementThresholds | None = None):
        self.geometry = geometry
        self.thresholds = thresholds or SettlementThresholds()
        self._last_index: int | None = None
        self._last_time: float | None = None
        self._dwell_s = 0.
        self._stable_started_at_s: float | None = None
        self._intent: PlacementIntent | None = None

    def reset_dwell(self):
        self._dwell_s = 0.
        self._stable_started_at_s = None

    def _reject(self, reason: str, *, recovery_required=False, evidence_unavailable=False) -> SettlementResult:
        self.reset_dwell()
        return SettlementResult(False, reason, 0., recovery_required=recovery_required,
                                evidence_unavailable=evidence_unavailable)

    @staticmethod
    def _valid_piece(piece: PieceEvidence) -> bool:
        vectors = ((piece.position_m, 3), (piece.quaternion_wxyz, 4),
                   (piece.linear_velocity_m_s, 3), (piece.angular_velocity_rad_s, 3))
        if any(len(value) != size or not all(math.isfinite(v) for v in value)
               for value, size in vectors):
            return False
        scalars = (piece.board_support_impulse_ns, piece.all_robot_contact_impulse_ns,
                   piece.maximum_penetration_m)
        return (all(isinstance(value, (int, float)) and math.isfinite(value) and value >= 0 for value in scalars)
                and abs(sum(v*v for v in piece.quaternion_wxyz)-1) <= 1.e-3)

    def position_rejection(self, piece: PieceEvidence, q: int, r: int) -> str | None:
        geometry, limits = self.geometry, self.thresholds
        if not self._valid_piece(piece):
            return 'nonfinite_or_invalid_piece_evidence'
        w, x, y, z = piece.quaternion_wxyz
        # Quaternions may have native normalization roundoff; normalize once.
        norm = w*w+x*x+y*y+z*z
        cosine = max(-1., min(1., 1-2*(x*x+y*y)/norm))
        tilt = math.acos(cosine)
        if tilt > limits.maximum_tilt_rad:
            return 'piece_not_upright'
        # Bound the complete tilted base/handle by a circle about base origin.
        radius = max(geometry.piece_radius_m + geometry.piece_base_height_m/2*math.sin(tilt),
                     geometry.handle_radius_m + (geometry.piece_height_m
                                                 -geometry.piece_base_height_m/2)*math.sin(tilt))
        if geometry.footprint_clearance_m(q, r, piece.position_m[:2], radius) < limits.edge_clearance_m:
            return 'piece_footprint_outside_cell'
        bottom_z = (piece.position_m[2] - geometry.piece_radius_m*math.sin(tilt)
                    - geometry.piece_base_height_m/2*math.cos(tilt))
        if abs(bottom_z-geometry.board_top_z_m) > limits.bottom_z_tolerance_m:
            return 'piece_not_at_board_height'
        if piece.maximum_penetration_m > limits.maximum_penetration_m:
            return 'excessive_penetration'
        return None

    def stability_rejection(self, piece: PieceEvidence, q: int, r: int) -> str | None:
        reason = self.position_rejection(piece, q, r)
        if reason:
            return reason
        limits = self.thresholds
        if piece.board_support_impulse_ns <= limits.minimum_support_impulse_ns:
            return 'native_board_support_required'
        if piece.all_robot_contact_impulse_ns > limits.maximum_robot_contact_impulse_ns:
            return 'robot_must_release_piece'
        if math.sqrt(sum(v*v for v in piece.linear_velocity_m_s)) > limits.maximum_linear_speed_m_s:
            return 'piece_still_translating'
        if math.sqrt(sum(v*v for v in piece.angular_velocity_rad_s)) > limits.maximum_angular_speed_rad_s:
            return 'piece_still_rotating'
        return None

    def observe(self, intent: PlacementIntent | None, frame: NativeFrame,
                committed: Mapping[str, tuple[int, int]], inventory: frozenset[str]) -> SettlementResult:
        if intent != self._intent:
            self._intent, self._dwell_s = intent, 0.
            self._stable_started_at_s = None
        if (frame.source != 'native_solver_substeps' or frame.contacts_complete is not True
                or type(frame.substep_index) is not int or frame.substep_index < 0
                or not math.isfinite(frame.simulation_time_s) or frame.simulation_time_s < 0
                or not math.isfinite(frame.dt_s) or not 0 < frame.dt_s <= self.thresholds.maximum_sample_gap_s
                or set(frame.pieces) != inventory):
            return self._reject('incomplete_or_invalid_native_frame', evidence_unavailable=True)
        if any(not self._valid_piece(piece) for piece in frame.pieces.values()):
            return self._reject('nonfinite_or_invalid_piece_evidence', evidence_unavailable=True)
        if self._last_index is not None:
            elapsed = frame.simulation_time_s-self._last_time
            if frame.substep_index <= self._last_index or elapsed <= 0:
                return self._reject('duplicate_or_out_of_order_frame', evidence_unavailable=True)
            consecutive = (frame.substep_index == self._last_index+1
                           and abs(elapsed-frame.dt_s) <= max(1.e-9, frame.dt_s*1.e-5))
            if not consecutive:
                self._last_index, self._last_time = frame.substep_index, frame.simulation_time_s
                return self._reject('missing_native_substeps', evidence_unavailable=True)
        self._last_index, self._last_time = frame.substep_index, frame.simulation_time_s
        for piece_id, cell in committed.items():
            reason = self.stability_rejection(frame.pieces[piece_id], *cell)
            if reason:
                return self._reject('committed_piece_needs_recovery:'+piece_id+':'+reason, recovery_required=True)
        for piece_id, piece in frame.pieces.items():
            if (piece_id not in committed and (intent is None or piece_id != intent.piece_id)
                    and piece.board_support_impulse_ns > self.thresholds.minimum_support_impulse_ns):
                return self._reject('unrequested_piece_on_board:'+piece_id, recovery_required=True)
        if intent is None:
            return self._reject('no_pending_move')
        piece = frame.pieces[intent.piece_id]
        reason = self.stability_rejection(piece, intent.q, intent.r)
        if reason:
            return self._reject(reason)
        # The first position sample establishes the beginning of a dwell;
        # never credit the unobserved interval immediately before that sample.
        if self._stable_started_at_s is None:
            self._stable_started_at_s = frame.simulation_time_s
        self._dwell_s = frame.simulation_time_s-self._stable_started_at_s
        elapsed_dwell = self._dwell_s
        accepted = elapsed_dwell+1.e-12 >= self.thresholds.dwell_s
        return SettlementResult(accepted, 'settled' if accepted else 'settling', elapsed_dwell)


class BoardGameMatch:
    """Authoritative match: a placement intent never changes logical occupancy."""

    def __init__(self, geometry: BoardGameGeometry | None = None, *,
                 thresholds: SettlementThresholds | None = None,
                 players: tuple[str, str] = ('player_0', 'player_1')):
        self.geometry = geometry or geometry_for('tic_tac_toe_3x3')
        self.rules = BoardGameRules(self.geometry, players)
        self.verifier = PhysicalSettlementVerifier(self.geometry, thresholds)
        self.pending: PlacementIntent | None = None
        self._committed: dict[str, tuple[int, int]] = {}
        self.inventory = frozenset(f'{color}_{i:02d}' for color, count in self.geometry.inventory_counts.items()
                                   for i in range(count))
        self.invalid_reason: str | None = None
        self.end_reason: str | None = None
        self.last_result: SettlementResult | None = None
        self._attempt_index = 0
        self.commit_log: list[dict] = []

    @property
    def terminal(self) -> bool:
        return self.rules.terminal or self.end_reason is not None

    @property
    def rule_contract(self) -> dict:
        return public_rule_contract(self.geometry, asdict(self.verifier.thresholds))

    @property
    def outcome(self) -> str:
        if self.rules.winner is not None:
            return 'win'
        if self.rules.draw:
            return 'draw'
        if self.end_reason in ('time_limit', 'step_limit'):
            return 'truncated'
        return 'aborted' if self.end_reason is not None else 'ongoing'

    def public_status(self) -> dict:
        """Coarse referee feedback; no exact native poses or logical occupancy."""
        result = self.last_result
        return {'outcome': self.outcome, 'end_reason': self.end_reason,
                'recovery_required': bool(result and result.recovery_required),
                'evidence_unavailable': bool(result and result.evidence_unavailable),
                'reason': result.reason.split(':', 1)[0] if result else 'awaiting_native_evidence',
                'retryable': not self.terminal}

    def end_episode(self, reason: str):
        """Runner-only interruption, never a player action or an opponent win."""
        if reason not in ('time_limit', 'step_limit', 'runtime_error', 'cancelled'):
            raise ValueError('Unknown episode interruption reason')
        if self.terminal:
            raise IllegalMove('Terminal result is immutable')
        self.end_reason = reason
        self.verifier.reset_dwell()

    @property
    def committed(self) -> dict[str, tuple[int, int]]:
        return dict(self._committed)

    def require_actor(self, player_id: str):
        if self.terminal:
            raise IllegalMove('Terminal match accepts no further actions')
        if player_id != self.rules.current_player:
            raise IllegalMove('Only the current player may act')

    def begin_move(self, player_id: str, piece_id: str, q: int, r: int) -> PlacementIntent:
        self.require_actor(player_id)
        self.rules.validate_move(player_id, q, r)
        if (not isinstance(piece_id, str) or piece_id not in self.inventory
                or not piece_id.startswith(self.rules.color(player_id)+'_')):
            raise IllegalMove('Piece does not belong to the player')
        if piece_id in self._committed:
            raise IllegalMove('A committed piece cannot be reused')
        if self.pending and (self.pending.player_id, self.pending.piece_id, self.pending.q, self.pending.r) == (player_id, piece_id, q, r):
            return self.pending
        self._attempt_index += 1
        self.pending = PlacementIntent(player_id, piece_id, q, r, self.rules.turn_index, self._attempt_index)
        self.verifier.reset_dwell()
        self.last_result = None
        return self.pending

    def cancel_move(self, player_id: str):
        self.require_actor(player_id)
        self.pending = None
        self.verifier.reset_dwell()
        self.last_result = None

    def observe(self, frame: NativeFrame) -> SettlementResult:
        if self.terminal:
            return SettlementResult(False, 'match_terminal', 0.)
        result = self.verifier.observe(self.pending, frame, self._committed, self.inventory)
        self.last_result = result
        if not result.accepted:
            return result
        intent = self.pending
        self.rules.play(intent.player_id, intent.q, intent.r)
        self._committed[intent.piece_id] = (intent.q, intent.r)
        self.commit_log.append({**asdict(intent), 'native_substep_index': frame.substep_index,
                                'simulation_time_s': frame.simulation_time_s,
                                'verified_dwell_s': result.dwell_s})
        self.pending = None
        self.last_result = SettlementResult(True, 'committed', result.dwell_s, committed=True)
        return self.last_result
