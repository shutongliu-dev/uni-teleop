"""Authenticated expansion facade with lossless, shared public event history.

Only native receipts can append events. Paging and observing never advance
physics; both players can read the same history after the worker has closed.
The facade has no access to hidden card identities or object state.
"""
from __future__ import annotations

from copy import deepcopy
import json

from bench.protocol.card_games import (PROTOCOL, CardGameState, CardGameResult,
    CardGameEventPage, EVENT_ADAPTER, INTENT_ADAPTER)
from .board_games import BoardGameSession, _error


class CardGameSession(BoardGameSession):
    state_model = CardGameState
    result_model = CardGameResult
    intent_adapter = INTENT_ADAPTER
    static_fields = ('public_rules', 'public_rules_sha256')
    invalid_intent_message = 'Game intent violates the advertised card-game schema'

    def __init__(self, session, initial, scene, task):
        self._public_events = []
        super().__init__(session, initial, scene, task)
        if self.state.event_cursor != -1 or self.state.round_index != 0:
            raise ValueError('An initial card game cannot inherit events or completed rounds')

    def catalog(self, player_id):
        value = super().catalog(player_id)
        value.update(events_endpoint='GET /board-games/{episode_id}/players/{player_id}/events',
            intent_schema_endpoint='GET /board-games/schema?protocol=' + PROTOCOL,
            game_protocol=PROTOCOL,
            turn_handoff='Every native-confirmed operation advances the authorization epoch, even when the same player continues. Stop at the enclosing 20 ms ACK; fetch state and shared events before the next operation.')
        return value

    def acknowledge(self, reply, command=None):
        state = self.state_model.model_validate(reply['board_game'])
        raw = reply.get('card_game_events')
        if not isinstance(raw, list) or len(raw) > 256:
            raise RuntimeError('Native ACK must include a bounded complete public event delta')
        events = [EVENT_ADAPTER.validate_python(event) for event in raw]
        expected = list(range(self.state.event_cursor + 1, state.event_cursor + 1))
        if (state.event_cursor < self.state.event_cursor
                or [event.sequence for event in events] != expected
                or len(self._public_events) != self.state.event_cursor + 1):
            raise RuntimeError('Native event stream has a gap, duplicate or reversed cursor')
        if command is None and events:
            raise RuntimeError('Only native physics can produce public game events')
        if any(event.task_id != state.task_id or event.turn_index > state.turn_index for event in events):
            raise RuntimeError('Public event belongs to a different game or future operation')
        committed = [event for event in events if event.type == 'operation_committed']
        advanced = state.turn_index - self.state.turn_index
        if len(committed) != advanced:
            raise RuntimeError('Every operation epoch must have exactly one native commit event')
        if committed:
            event, pending = committed[0], self.state.pending_move
            if (pending is None or event.turn_index != state.turn_index
                    or event.player_id != pending.player_id or event.details.operation != pending.type
                    or event.details.piece_ids != pending.piece_ids):
                raise RuntimeError('Commit event does not match the admitted pending operation')
        if any(event.player_id != self.state.current_player for event in events):
            raise RuntimeError('Event was attributed to a different player')
        if any(event.type == 'card_revealed' and not event.details.expected for event in events):
            if state.status.outcome != 'aborted' or state.status.end_reason != 'rule_violation':
                raise RuntimeError('Unexpected revealed information must stop the game')
        if (not state.terminal and state.turn_index == self.state.turn_index
                and state.current_player != self.state.current_player):
            raise RuntimeError('Player ownership changed without a native operation commit')
        previous = self._public_events[-1] if self._public_events else None
        for event in events:
            value = event.model_dump(mode='json')
            if previous is not None and any(value[key] < previous[key] for key in
                    ('native_substep_index', 'simulation_time_s', 'turn_index')):
                raise RuntimeError('Public events moved backward in native time')
            previous = value
        changed = super().acknowledge(reply, command)
        self._public_events.extend(event.model_dump(mode='json') for event in events)
        return changed

    def end(self, outcome):
        super().end(outcome)
        boundary = self.state.event_cursor + 1
        if len(self._public_events) > boundary:
            # A late success may be superseded by the earlier budget receipt.
            # Preserve its actual native events privately, then keep the public
            # cache exactly aligned with the accepted state, including on reads
            # after the worker closes.
            path = self.session.directory/'private/late_card_game_events.json'
            path.write_text(json.dumps({'accepted_event_cursor': self.state.event_cursor,
                'outcome': outcome, 'late_native_events': self._public_events[boundary:]},
                sort_keys=True, allow_nan=False)+'\n')
            del self._public_events[boundary:]

    def events(self, player_id, after=-1, limit=256):
        with self.session.lock:
            if player_id not in self.bindings:
                raise _error(403, 'Unknown player')
            if (type(after) is not int or not -1 <= after <= self.state.event_cursor
                    or type(limit) is not int or not 1 <= limit <= 256):
                raise _error(422, 'Event cursor must be within this game history and limit must be 1..256')
            stop = min(after + 1 + limit, self.state.event_cursor + 1)
            values = deepcopy(self._public_events[after + 1:stop])
            page = CardGameEventPage(events=values, next_cursor=stop - 1,
                                     has_more=stop <= self.state.event_cursor)
            return page.model_dump(mode='json')

    def execution_termination(self):
        """Stop at the first terminal ACK without turning a fault into a loss.

        Episode outcomes describe execution; the original native game receipt
        remains aborted/truncated with no winner for every interruption.
        """
        if not self.state.terminal:
            return None
        if self.state.status.outcome in ('win', 'draw'):
            return 'success', True
        return {
            'time_limit': ('sim_timeout', True),
            'step_limit': ('action_budget', True),
            'runtime_error': ('infrastructure_error', False),
            'cancelled': ('cancelled', True),
            'physical_failure': ('safety_stop', False),
            'missing_physical_evidence': ('evaluator_invalid', False),
            'shot_rule_violation': ('task_failure', True),
            'rule_violation': ('task_failure', True),
        }[self.state.status.end_reason]
