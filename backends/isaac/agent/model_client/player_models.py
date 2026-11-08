"""Strict, private per-player model selection; never load or publish API keys here."""
from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path

from bench.paths import load_json


PLAYERS = ('player_0', 'player_1')
FIELDS = frozenset({'provider', 'model', 'reasoning_effort', 'credentials'})


@dataclass(frozen=True)
class PlayerModelBinding:
    provider: str
    model: str
    reasoning_effort: str
    credentials: Path = field(repr=False)

    def public_metadata(self):
        """Only the explicit experimental conditions, without private paths."""
        return {'provider': self.provider, 'model': self.model,
                'reasoning_effort': self.reasoning_effort}


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('Player model JSON must not contain duplicate keys')
        result[key] = value
    return result


def load_player_models(path):
    """Require two complete bindings, with relative profiles beside the real JSON.

    Provider/model/effort support belongs to the configured transport. Do not
    infer a highest setting, normalize model IDs, inherit absent fields, or
    silently substitute another provider. Error text never includes file paths
    or untrusted JSON values, which may accidentally contain secret material.
    """
    try:
        config_path = Path(path).expanduser().resolve()
        value = load_json(config_path, object_pairs_hook=_unique_object)
    except (OSError, ValueError, TypeError, RuntimeError):
        raise ValueError('Could not read strict player model JSON') from None
    if not isinstance(value, dict) or set(value) != set(PLAYERS):
        raise ValueError('Player models must bind exactly player_0 and player_1')
    result = {}
    for player in PLAYERS:
        row = value[player]
        if not isinstance(row, dict) or set(row) != FIELDS:
            raise ValueError('Each player requires exactly provider, model, reasoning_effort and credentials')
        if any(not isinstance(item, str) or not item.strip() for item in row.values()):
            raise ValueError('Player model binding fields must be nonempty strings')
        for key in ('provider', 'model', 'reasoning_effort'):
            item = row[key]
            if item != item.strip() or any(ord(char) < 32 for char in item):
                raise ValueError('Model identifiers and reasoning effort must be explicit unpadded strings')
        try:
            profile = Path(row['credentials']).expanduser()
            if not profile.is_absolute():
                profile = config_path.parent / profile
            profile = profile.resolve()
        except (OSError, ValueError, RuntimeError):
            raise ValueError('Could not resolve player credential profile') from None
        result[player] = PlayerModelBinding(row['provider'], row['model'], row['reasoning_effort'], profile)
    return result
