"""Private native evidence gate. This module never writes a physical pose.

Evidence must come from each consecutive native integration substep, not from
model-produced JSON or the logical board. A complete frame includes every token.
Scene/runtime binding and real contact qualification remain separate obligations.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Mapping

from .geometry import HexGeometry
from .rules import HexRules, IllegalMove


@dataclass(frozen=True)
class SettlementThresholds:
    dwell_s: float = .300
    maximum_linear_speed_m_s: float = .015
    maximum_angular_speed_rad_s: float = .100
    maximum_tilt_rad: float = math.radians(5)
    bottom_z_tolerance_m: float = .003
    edge_clearance_m: float = .002
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


@dataclass(frozen=True)
class SettlementResult:
    accepted: bool
    reason: str
    dwell_s: float
    committed: bool = False
    invalidated: bool = False


class PhysicalSettlementVerifier:
    def __init__(self, geometry: HexGeometry, thresholds: SettlementThresholds | None = None):
        self.geometry = geometry
        self.thresholds = thresholds or SettlementThresholds()
        self._last_index: int | None = None
        self._last_time: float | None = None
        self._dwell_s = 0.
        self._stable_started_at_s: float | None = None
        self._intent: PlacementIntent | None = None

    def _reject(self, reason: str, *, invalidated=False) -> SettlementResult:
        self._dwell_s = 0.
        self._stable_started_at_s = None
        return SettlementResult(False, reason, 0., invalidated=invalidated)

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
            return self._reject('incomplete_or_invalid_native_frame')
        if any(not self._valid_piece(piece) for piece in frame.pieces.values()):
            return self._reject('nonfinite_or_invalid_piece_evidence')
        if self._last_index is not None:
            elapsed = frame.simulation_time_s-self._last_time
            if frame.substep_index <= self._last_index or elapsed <= 0:
                return self._reject('duplicate_or_out_of_order_frame')
            consecutive = (frame.substep_index == self._last_index+1
                           and abs(elapsed-frame.dt_s) <= max(1.e-9, frame.dt_s*1.e-5))
            if not consecutive:
                self._last_index, self._last_time = frame.substep_index, frame.simulation_time_s
                return self._reject('missing_native_substeps')
        self._last_index, self._last_time = frame.substep_index, frame.simulation_time_s
        for piece_id, cell in committed.items():
            reason = self.position_rejection(frame.pieces[piece_id], *cell)
            if reason:
                return self._reject('committed_piece_displaced:'+piece_id+':'+reason, invalidated=True)
        for piece_id, piece in frame.pieces.items():
            if (piece_id not in committed and (intent is None or piece_id != intent.piece_id)
                    and piece.board_support_impulse_ns > self.thresholds.minimum_support_impulse_ns):
                return self._reject('unrequested_piece_on_board:'+piece_id, invalidated=True)
        if intent is None:
            return self._reject('no_pending_move')
        piece = frame.pieces[intent.piece_id]
        reason = self.position_rejection(piece, intent.q, intent.r)
        if reason:
            return self._reject(reason)
        if piece.board_support_impulse_ns <= self.thresholds.minimum_support_impulse_ns:
            return self._reject('native_board_support_required')
        if piece.all_robot_contact_impulse_ns > self.thresholds.maximum_robot_contact_impulse_ns:
            return self._reject('robot_must_release_piece')
        if math.sqrt(sum(v*v for v in piece.linear_velocity_m_s)) > self.thresholds.maximum_linear_speed_m_s:
            return self._reject('piece_still_translating')
        if math.sqrt(sum(v*v for v in piece.angular_velocity_rad_s)) > self.thresholds.maximum_angular_speed_rad_s:
            return self._reject('piece_still_rotating')
        # The first position sample establishes the beginning of a dwell;
        # never credit the unobserved interval immediately before that sample.
        if self._stable_started_at_s is None:
            self._stable_started_at_s = frame.simulation_time_s
        self._dwell_s = frame.simulation_time_s-self._stable_started_at_s
        elapsed_dwell = self._dwell_s
        accepted = elapsed_dwell+1.e-12 >= self.thresholds.dwell_s
        return SettlementResult(accepted, 'settled' if accepted else 'settling', elapsed_dwell)


class HexMatch:
    """Authoritative match: a placement intent never changes logical occupancy."""

    def __init__(self, geometry: HexGeometry | None = None, *,
                 thresholds: SettlementThresholds | None = None,
                 players: tuple[str, str] = ('player_0', 'player_1')):
        self.geometry = geometry or HexGeometry()
        self.rules = HexRules(players)
        self.verifier = PhysicalSettlementVerifier(self.geometry, thresholds)
        self.pending: PlacementIntent | None = None
        self._committed: dict[str, tuple[int, int]] = {}
        self.inventory = frozenset(f'{color}_{i:02d}' for color in ('red', 'blue') for i in range(25))
        self.invalid_reason: str | None = None
        self.commit_log: list[dict] = []

    @property
    def terminal(self) -> bool:
        return self.rules.winner is not None or self.invalid_reason is not None

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
        if self.pending is not None:
            raise IllegalMove('A move is already pending physical settlement')
        if (not isinstance(piece_id, str) or piece_id not in self.inventory
                or not piece_id.startswith(self.rules.color(player_id)+'_')):
            raise IllegalMove('Piece does not belong to the player')
        if piece_id in self._committed:
            raise IllegalMove('A committed piece cannot be reused')
        self.pending = PlacementIntent(player_id, piece_id, q, r, self.rules.turn_index)
        return self.pending

    def observe(self, frame: NativeFrame) -> SettlementResult:
        if self.terminal:
            return SettlementResult(False, 'match_terminal', 0.)
        result = self.verifier.observe(self.pending, frame, self._committed, self.inventory)
        if result.invalidated:
            self.invalid_reason = result.reason
        if not result.accepted:
            return result
        intent = self.pending
        self.rules.play(intent.player_id, intent.q, intent.r)
        self._committed[intent.piece_id] = (intent.q, intent.r)
        self.commit_log.append({**asdict(intent), 'native_substep_index': frame.substep_index,
                                'simulation_time_s': frame.simulation_time_s,
                                'verified_dwell_s': result.dwell_s})
        self.pending = None
        return SettlementResult(True, 'committed', result.dwell_s, committed=True)
