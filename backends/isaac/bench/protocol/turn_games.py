"""Authenticated turn-taking contracts independent of task props and rules."""
from __future__ import annotations

import json
from typing import Literal

from pydantic import ConfigDict, Field, JsonValue, model_validator

from .schemas import Contract, EpisodeResult, HandNamespace, Name, Nonnegative


class TurnContract(Contract):
    model_config = ConfigDict(strict=True, revalidate_instances='always')


class PlayerBinding(TurnContract):
    player_id: Name
    robot_id: HandNamespace
    hand: HandNamespace
    joint_names: list[Name] = Field(min_length=1, max_length=256)

    @model_validator(mode='after')
    def owned_instance(self):
        if self.robot_id != self.hand or len(set(self.joint_names)) != len(self.joint_names):
            raise ValueError('Player binding must identify one instance with unique joints')
        return self


class PlayerSession(TurnContract):
    binding: PlayerBinding
    session_token: str = Field(min_length=32, max_length=256, pattern=r'^[A-Za-z0-9_-]+$', repr=False)


class TurnIntent(TurnContract):
    type: Literal['turn_intent'] = 'turn_intent'
    name: Name
    arguments: dict[str, JsonValue] = Field(default_factory=dict, max_length=32)

    @model_validator(mode='after')
    def bounded_request(self):
        if len(json.dumps(self.arguments, allow_nan=False).encode()) > 4096:
            raise ValueError('Intent arguments exceed the public size limit')
        return self


class TurnGameState(TurnContract):
    protocol: Literal['turn_game_state_v1'] = 'turn_game_state_v1'
    game_id: Name
    status: Literal['active', 'completed', 'interrupted']
    turn_index: int = Field(strict=True, ge=0)
    current_player_id: Name | None
    players: list[Name] = Field(min_length=2, max_length=8)
    winner_player_id: Name | None
    loser_player_id: Name | None
    terminal_reason: Name | None
    violation_reason: Name | None
    sim_time_s: Nonnegative
    public_rules: dict[str, JsonValue] = Field(default_factory=dict)
    public_progress: dict[str, JsonValue] = Field(default_factory=dict)

    @property
    def terminal(self):
        return self.status != 'active'

    @property
    def current_player(self):
        return self.current_player_id

    @model_validator(mode='after')
    def consistent_turn(self):
        if len(set(self.players)) != len(self.players):
            raise ValueError('Player identities must be unique')
        if self.terminal != (self.current_player_id is None):
            raise ValueError('An active game requires a current player')
        for player in (self.current_player_id, self.winner_player_id, self.loser_player_id):
            if player is not None and player not in self.players:
                raise ValueError('Unknown player identity')
        if self.status != 'completed' and (self.winner_player_id is not None or self.loser_player_id is not None):
            raise ValueError('Only a completed native game may assign a winner or loser')
        if self.winner_player_id is not None and self.winner_player_id == self.loser_player_id:
            raise ValueError('Winner and loser must differ')
        if self.terminal != (self.terminal_reason is not None):
            raise ValueError('Terminal status requires a reason')
        if len(json.dumps(self.model_dump(mode='json'), allow_nan=False).encode()) > 32768:
            raise ValueError('Public referee receipt is too large')
        return self


class TurnGameResult(TurnContract):
    turn_game: TurnGameState
    episode: EpisodeResult

    @model_validator(mode='after')
    def completed_episode(self):
        if not self.turn_game.terminal or self.turn_game.game_id != self.episode.spec.task_id:
            raise ValueError('A result requires the matching terminal referee receipt')
        if (self.episode.outcome in ('wall_timeout', 'sim_timeout', 'action_budget', 'model_budget')
                and self.turn_game.status == 'completed'):
            raise ValueError('A budget-terminated episode cannot certify a completed game')
        return self


def turn_game_schemas():
    return {model.__name__: model.model_json_schema() for model in
            (PlayerBinding, PlayerSession, TurnIntent, TurnGameState, TurnGameResult)}
