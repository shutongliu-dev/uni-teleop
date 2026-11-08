"""Public contracts for the four physical card-room expansion games.

Intent is a declaration, never a pose command or physical evidence. Operation
epochs deliberately differ from player rounds: a first reveal hands control
back to the same player under a fresh epoch. Legacy tabletop schemas are not
broadened by this protocol.
"""
from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field, JsonValue, TypeAdapter, model_validator

from .board_games import BoardGameContract, PlayerBinding, PlayerId, PlayerSession
from .schemas import EpisodeResult, Name, Nonnegative

PROTOCOL = 'bench_card_game_state_v1'
VERSION = '1.0.0'
GameId = Literal['polyomino_territory', 'barrier_race', 'memory_pairs', 'disk_curling']
PieceId = Annotated[str, Field(pattern=r'^(black|white)_[a-z][A-Za-z0-9_]{0,40}$')]
CardId = Annotated[str, Field(pattern=r'^card_(0[0-9]|1[01])$')]
ObjectId = Annotated[str, Field(pattern=r'^(card_(0[0-9]|1[01])|(black|white)_[a-z][A-Za-z0-9_]{0,40})$')]
Coordinate = Annotated[int, Field(strict=True, ge=0, le=5)]
Cell = Annotated[list[Coordinate], Field(min_length=2, max_length=2)]
Operation = Literal['place_piece', 'pass', 'move_pawn', 'place_barrier',
                    'reveal_card', 'collect_pair', 'flip_back', 'shoot_disk']
EndReason = Literal['time_limit', 'step_limit', 'runtime_error', 'cancelled',
                    'missing_physical_evidence', 'physical_failure',
                    'shot_rule_violation', 'rule_violation']


class PlacePiece(BoardGameContract):
    type: Literal['place_piece'] = 'place_piece'
    piece_id: PieceId
    anchor: Cell
    rotation_deg: Annotated[int, Field(strict=True)] = 0

    @model_validator(mode='after')
    def quarter_turn(self):
        if self.rotation_deg not in (0, 90, 180, 270):
            raise ValueError('Only quarter-turn rotations are legal')
        return self


class PassTurn(BoardGameContract):
    type: Literal['pass'] = 'pass'


class MovePawn(BoardGameContract):
    type: Literal['move_pawn'] = 'move_pawn'
    destination: Cell

    @model_validator(mode='after')
    def inside_board(self):
        if any(value > 4 for value in self.destination):
            raise ValueError('Pawn destination must be on the 5x5 board')
        return self


class PlaceBarrier(BoardGameContract):
    type: Literal['place_barrier'] = 'place_barrier'
    piece_id: PieceId
    anchor: Cell
    orientation: Literal['horizontal', 'vertical']

    @model_validator(mode='after')
    def inside_board(self):
        if any(value > 3 for value in self.anchor):
            raise ValueError('Barrier anchors are 0..3 on each axis')
        return self


class RevealCard(BoardGameContract):
    type: Literal['reveal_card'] = 'reveal_card'
    card_id: CardId


class CollectPair(BoardGameContract):
    type: Literal['collect_pair'] = 'collect_pair'


class FlipBack(BoardGameContract):
    type: Literal['flip_back'] = 'flip_back'


class ShootDisk(BoardGameContract):
    type: Literal['shoot_disk'] = 'shoot_disk'
    piece_id: PieceId


class CancelIntent(BoardGameContract):
    type: Literal['cancel_intent'] = 'cancel_intent'


CardGameIntent = Annotated[PlacePiece | PassTurn | MovePawn | PlaceBarrier | RevealCard |
                           CollectPair | FlipBack | ShootDisk | CancelIntent,
                           Field(discriminator='type')]
INTENT_ADAPTER = TypeAdapter(CardGameIntent)
TASK_OPERATIONS = {
    'polyomino_territory': {'place_piece', 'pass'},
    'barrier_race': {'move_pawn', 'place_barrier'},
    'memory_pairs': {'reveal_card', 'collect_pair', 'flip_back'},
    'disk_curling': {'shoot_disk'},
}


class PendingMove(BoardGameContract):
    player_id: PlayerId
    type: Operation
    piece_ids: list[ObjectId] = Field(max_length=2)
    parameters: dict[str, JsonValue] = Field(default_factory=dict, max_length=3)
    turn_index: int = Field(strict=True, ge=0)
    attempt_index: int = Field(strict=True, ge=0)

    @model_validator(mode='after')
    def public_intent_only(self):
        count = 0 if self.type == 'pass' else 2 if self.type in ('collect_pair', 'flip_back') else 1
        if len(self.piece_ids) != count or len(set(self.piece_ids)) != count:
            raise ValueError('Pending operation must identify exactly its own objects')
        command = {'type': self.type, **self.parameters}
        if self.type in ('place_piece', 'place_barrier', 'shoot_disk'):
            command['piece_id'] = self.piece_ids[0]
        elif self.type == 'reveal_card':
            command['card_id'] = self.piece_ids[0]
        value = INTENT_ADAPTER.validate_python(command)
        allowed = set(value.model_dump()) - {'type', 'piece_id', 'card_id'}
        if set(self.parameters) != allowed:
            raise ValueError('Pending parameters must contain exactly the declared public parameters')
        if self.type in ('reveal_card', 'collect_pair', 'flip_back'):
            if any(not identity.startswith('card_') for identity in self.piece_ids):
                raise ValueError('Memory operations refer only to cards')
        elif self.piece_ids:
            color = 'black' if self.player_id == 'player_0' else 'white'
            if any(not identity.startswith(color + '_') for identity in self.piece_ids):
                raise ValueError('Pending operation must use own objects')
            if self.type == 'move_pawn' and self.piece_ids != [color + '_pawn']:
                raise ValueError('Pawn movement must identify the owned pawn')
        return self


class CardRefereeStatus(BoardGameContract):
    outcome: Literal['ongoing', 'win', 'draw', 'truncated', 'aborted']
    end_reason: EndReason | None
    recovery_required: bool
    evidence_unavailable: bool
    reason: Annotated[str, Field(min_length=1, max_length=160, pattern=r'^[a-z][a-z0-9_]*$')]
    retryable: bool

    @model_validator(mode='after')
    def consistent_outcome(self):
        if self.outcome == 'truncated':
            valid = self.end_reason in ('time_limit', 'step_limit')
        elif self.outcome == 'aborted':
            valid = self.end_reason is not None and self.end_reason not in ('time_limit', 'step_limit')
        else:
            valid = self.end_reason is None
        if not valid or self.retryable != (self.outcome == 'ongoing'):
            raise ValueError('Outcome, interruption and retry permission disagree')
        return self


class CardGameState(BoardGameContract):
    protocol: Literal['bench_card_game_state_v1'] = PROTOCOL
    task_id: GameId
    turn_index: int = Field(strict=True, ge=0)
    round_index: int = Field(strict=True, ge=0)
    current_player: PlayerId | None
    terminal: bool
    winner: PlayerId | None
    draw: bool
    pending_move: PendingMove | None
    status: CardRefereeStatus
    settlement_dwell_s: Nonnegative = 0.
    commit_count: int = Field(strict=True, ge=0)
    phase: Name
    event_cursor: int = Field(strict=True, ge=-1)

    @model_validator(mode='after')
    def consistent_match(self):
        outcome = self.status.outcome
        if self.commit_count != self.turn_index or self.round_index > self.turn_index:
            raise ValueError('Operation epoch must equal native commit count and contain all completed rounds')
        if self.terminal != (outcome != 'ongoing') or self.terminal != (self.current_player is None):
            raise ValueError('Only an active game has a current player')
        if (self.winner is not None) != (outcome == 'win') or self.draw != (outcome == 'draw'):
            raise ValueError('Incomplete games cannot invent winners or draws')
        if self.pending_move is not None:
            pending = self.pending_move
            if (pending.turn_index != self.turn_index or outcome in ('win', 'draw')
                    or (not self.terminal and pending.player_id != self.current_player)
                    or pending.type not in TASK_OPERATIONS[self.task_id]):
                raise ValueError('Pending operation does not belong to this game and epoch')
        return self


class CardRevealedDetails(BoardGameContract):
    card_id: CardId
    symbol: Annotated[str, Field(pattern=r'^pair_[0-5]$')]
    expected: bool


class OperationDetails(BoardGameContract):
    operation: Operation
    piece_ids: list[ObjectId] = Field(max_length=2)


class _Event(BoardGameContract):
    sequence: int = Field(strict=True, ge=0)
    task_id: GameId
    player_id: PlayerId
    turn_index: int = Field(strict=True, ge=0)
    native_substep_index: int = Field(strict=True, ge=0)
    simulation_time_s: Nonnegative


class CardRevealedEvent(_Event):
    type: Literal['card_revealed'] = 'card_revealed'
    details: CardRevealedDetails

    @model_validator(mode='after')
    def memory_only(self):
        if self.task_id != 'memory_pairs':
            raise ValueError('Only memory has reveal events')
        return self


class OperationCommittedEvent(_Event):
    type: Literal['operation_committed'] = 'operation_committed'
    details: OperationDetails

    @model_validator(mode='after')
    def game_operation(self):
        if self.details.operation not in TASK_OPERATIONS[self.task_id]:
            raise ValueError('Operation belongs to a different game')
        return self


CardGameEvent = Annotated[CardRevealedEvent | OperationCommittedEvent, Field(discriminator='type')]
EVENT_ADAPTER = TypeAdapter(CardGameEvent)


class CardGameEventPage(BoardGameContract):
    events: list[CardGameEvent] = Field(max_length=256)
    next_cursor: int = Field(strict=True, ge=-1)
    has_more: bool

    @model_validator(mode='after')
    def continuous_page(self):
        if self.events:
            sequence = [event.sequence for event in self.events]
            if sequence != list(range(sequence[0], sequence[0] + len(sequence))) or self.next_cursor != sequence[-1]:
                raise ValueError('Public event pages must be consecutive and end at their cursor')
        elif self.has_more:
            raise ValueError('An empty page cannot claim more events')
        return self


class CardGameResult(BoardGameContract):
    board_game: CardGameState
    episode: EpisodeResult

    @model_validator(mode='after')
    def terminal_result(self):
        if not self.board_game.terminal or self.board_game.task_id != self.episode.spec.task_id:
            raise ValueError('Result requires the matching terminal game')
        if (self.episode.outcome in ('wall_timeout', 'sim_timeout', 'action_budget', 'model_budget')
                and self.board_game.status.outcome in ('win', 'draw')):
            raise ValueError('A budget interruption cannot certify a completed game')
        return self


def card_game_schemas():
    return {'intent': INTENT_ADAPTER.json_schema(), **{
        model.__name__: model.model_json_schema() for model in
        (PlayerBinding, PlayerSession, CardGameState, CardGameResult, CardGameEventPage)}}
