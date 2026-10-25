"""Public tabletop referee contracts, separate from robot execution outcomes.

Identity is supplied by authenticated transport, never by a placement request.
Native evidence and logical occupancy deliberately have no public schema here.
"""
from __future__ import annotations

from typing import Annotated, Literal

from pydantic import ConfigDict, Field, TypeAdapter, model_validator

from .schemas import Contract, EpisodeResult, Name, Nonnegative

PROTOCOL = 'bench_board_game_state_v1'
VERSION = '1.0.0'
PlayerId = Literal['player_0', 'player_1']
Hand = Literal['left', 'right']
GameId = Literal['tic_tac_toe_3x3', 'connect_four_5x5', 'hex7x7']
GameOutcome = Literal['ongoing', 'win', 'draw', 'truncated', 'aborted']
GameEndReason = Literal['time_limit', 'step_limit', 'runtime_error', 'cancelled']
PieceId = Annotated[str, Field(pattern=r'^(black|white)_[0-9]{2}$')]
CellCoordinate = Annotated[int, Field(strict=True, ge=0, le=6)]


class BoardGameContract(Contract):
    model_config = ConfigDict(strict=True, revalidate_instances='always')


class PlayerBinding(BoardGameContract):
    player_id: PlayerId
    color: Literal['black', 'white']
    hand: Hand
    robot_id: Hand
    joint_names: list[Name] = Field(min_length=1, max_length=256)

    @model_validator(mode='after')
    def consistent_binding(self):
        expected = ('black', 'left') if self.player_id == 'player_0' else ('white', 'right')
        if (self.color, self.hand) != expected or self.robot_id != self.hand:
            raise ValueError('Player color and articulation must match the frozen card-room binding')
        if len(set(self.joint_names)) != len(self.joint_names):
            raise ValueError('Owned joint names must be unique')
        return self


class PlayerSession(BoardGameContract):
    binding: PlayerBinding
    session_token: str = Field(min_length=32, max_length=256, pattern=r'^[A-Za-z0-9_-]+$', repr=False)


class PlacementIntentRequest(BoardGameContract):
    type: Literal['placement_intent'] = 'placement_intent'
    piece_id: PieceId
    q: CellCoordinate
    r: CellCoordinate


class CancelPlacementRequest(BoardGameContract):
    type: Literal['cancel_placement'] = 'cancel_placement'


BoardGameIntent = Annotated[PlacementIntentRequest | CancelPlacementRequest, Field(discriminator='type')]
INTENT_ADAPTER = TypeAdapter(BoardGameIntent)


class PendingPlacement(BoardGameContract):
    player_id: PlayerId
    piece_id: PieceId
    q: CellCoordinate
    r: CellCoordinate
    turn_index: int = Field(strict=True, ge=0)
    attempt_index: int = Field(strict=True, ge=0)

    @model_validator(mode='after')
    def owned_piece(self):
        color = 'black' if self.player_id == 'player_0' else 'white'
        if not self.piece_id.startswith(color + '_'):
            raise ValueError('Pending placement piece must belong to its player')
        return self


class RefereeStatus(BoardGameContract):
    outcome: GameOutcome
    end_reason: GameEndReason | None
    recovery_required: bool
    evidence_unavailable: bool
    reason: str = Field(min_length=1, max_length=160, pattern=r'^[a-z][a-z0-9_]*$')
    retryable: bool

    @model_validator(mode='after')
    def consistent_outcome(self):
        if self.outcome == 'truncated':
            valid_reason = self.end_reason in ('time_limit', 'step_limit')
        elif self.outcome == 'aborted':
            valid_reason = self.end_reason in ('runtime_error', 'cancelled')
        else:
            valid_reason = self.end_reason is None
        if not valid_reason:
            raise ValueError('Game outcome and interruption reason disagree')
        if self.retryable != (self.outcome == 'ongoing'):
            raise ValueError('Only an ongoing match is retryable')
        return self


class BoardGameState(BoardGameContract):
    protocol: Literal['bench_board_game_state_v1'] = PROTOCOL
    task_id: GameId
    turn_index: int = Field(strict=True, ge=0, le=49)
    current_player: PlayerId | None
    terminal: bool
    winner: PlayerId | None
    draw: bool
    pending_move: PendingPlacement | None
    status: RefereeStatus
    settlement_dwell_s: Nonnegative = 0.
    commit_count: int = Field(default=0, strict=True, ge=0, le=49)

    @model_validator(mode='after')
    def consistent_match(self):
        outcome = self.status.outcome
        size = {'tic_tac_toe_3x3': 3, 'connect_four_5x5': 5, 'hex7x7': 7}[self.task_id]
        if self.turn_index > size * size or self.commit_count != self.turn_index:
            raise ValueError('Turn index must equal the verified commit count within task inventory')
        if self.terminal != (outcome != 'ongoing'):
            raise ValueError('Terminal flag and game outcome disagree')
        expected_player = None if self.terminal else ('player_0' if self.turn_index % 2 == 0 else 'player_1')
        if self.current_player != expected_player:
            raise ValueError('Current player must match the verified turn index')
        if (self.winner is not None) != (outcome == 'win') or self.draw != (outcome == 'draw'):
            raise ValueError('Only verified wins have a winner and only draws set the draw flag')
        if outcome == 'draw' and (self.task_id == 'hex7x7' or self.turn_index != size * size):
            raise ValueError('A draw requires a full non-Hex board')
        if outcome == 'win' and self.winner != ('player_0' if self.turn_index % 2 else 'player_1'):
            raise ValueError('The most recently committed player must be the winner')
        if self.pending_move is not None:
            pending = self.pending_move
            actor = 'player_0' if self.turn_index % 2 == 0 else 'player_1'
            if outcome in ('win', 'draw') or pending.turn_index != self.turn_index or pending.player_id != actor:
                raise ValueError('Pending placement must belong to the uncommitted current turn')
            if pending.q >= size or pending.r >= size:
                raise ValueError('Pending cell is outside the task board')
        return self


class BoardGameResult(BoardGameContract):
    board_game: BoardGameState
    episode: EpisodeResult

    @model_validator(mode='after')
    def completed_match(self):
        if not self.board_game.terminal:
            raise ValueError('A final board-game result requires a terminal match')
        if self.episode.spec.task_id != self.board_game.task_id:
            raise ValueError('Game and robot episode task identities disagree')
        if (self.episode.outcome in ('wall_timeout', 'sim_timeout', 'action_budget', 'model_budget')
                and self.board_game.status.outcome in ('win', 'draw')):
            raise ValueError('A budget-terminated episode cannot certify a win or draw')
        # An already accepted win/draw survives a later recording/worker fault.
        # Keep the execution outcome separate instead of converting it to a loss.
        return self


def board_game_schemas() -> dict[str, dict]:
    return {'intent': INTENT_ADAPTER.json_schema(), **{
        model.__name__: model.model_json_schema() for model in
        (PlayerBinding, PlayerSession, BoardGameState, BoardGameResult)}}
