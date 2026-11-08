"""Trusted four-game worker; requests are intents, native frames are authority."""
from copy import deepcopy
import json
from pathlib import Path

import numpy as np

from scenes.board_games.expansion_native_observer import ExpansionNativeObserver
from bench.protocol.card_games import CardGameState, CardGameEventPage


PLAYERS = [{'player_id': 'player_0', 'color': 'black', 'robot_id': 'left'},
           {'player_id': 'player_1', 'color': 'white', 'robot_id': 'right'}]


class CardGameWorker:
    def __init__(self, env, scene, task, *, output=None):
        from scenes.board_games.expansion_release import verify_task
        from scenes.workrooms.card_room.expansion_referee import ExpansionMatch
        verify_task(task)
        binding = scene.get('board_game_task', {})
        if (binding.get('schema') != 'card_game_expansion_native_task.v1'
                or binding.get('task_id') != task['task_id'] or binding.get('players') != PLAYERS):
            raise ValueError('Frozen expansion task/player ownership binding is required')
        self.env, self.task = env, task
        self.output = None if output is None else Path(output)
        if self.output is not None:
            self.output.mkdir(parents=True, exist_ok=True)
        self.match = ExpansionMatch(task['task_id'], asset_metadata=task['asset_metadata'],
                                    private_pairs=task['asset_metadata'].get('private_reset_pairs'))
        names = tuple(env.actuated_joint_names)
        self._owners = {row['player_id']: frozenset(name for name in names
            if name.startswith((row['robot_id']+'_', 'fr3_'+row['robot_id']+'_'))) for row in PLAYERS}
        if (not all(self._owners.values()) or set.union(*(set(v) for v in self._owners.values())) != set(names)
                or set.intersection(*(set(v) for v in self._owners.values()))):
            raise ValueError('Every native actuator must have exactly one player owner')
        self._drive_targets = {name: float(env._targets[0, env.joint_names.index(name)]) for name in names}
        self.observer = ExpansionNativeObserver(env, self.match, task['asset_metadata'],
            frame_jsonl=None if self.output is None else self.output/'native_frames.jsonl').attach()
        self.closed = False
        self._sent_event_cursor = -1

    def public_state(self):
        return CardGameState.model_validate(self.match.public_state()).model_dump(mode='json')

    def score(self):
        state = self.public_state()
        return {**state, 'schema': 'card_game_expansion_worker_score.v1',
                'success': state['status']['outcome'] in ('win', 'draw'),
                'outcome': state['status']['outcome'], 'reason': state['status']['reason'],
                'end_reason': state['status']['end_reason'],
                'native_observer_samples': self.observer.samples,
                'native_observer_unsupported_samples': self.observer.unsupported_samples}

    def _require_actor(self, actor):
        if (not isinstance(actor, dict) or set(actor) != {'player_id', 'turn_index'}
                or not isinstance(actor['player_id'], str) or actor['player_id'] not in self._owners
                or type(actor['turn_index']) is not int
                or actor['turn_index'] != self.match.turn_index
                or actor['player_id'] != self.match.current_player or self.match.terminal):
            raise ValueError('Command lacks current authenticated player and authorization epoch')
        return actor['player_id']

    def authorize_step(self, command, actor):
        player = self._require_actor(actor)
        self.env._validate_action(command)
        for name, value in command['joint_positions_rad'].items():
            if name not in self._owners[player] and np.float32(value) != np.float32(self._drive_targets[name]):
                raise ValueError('A player command changed the other robot drive target')
        return self.match.turn_index

    def _event_delta(self):
        events = []
        while True:
            page = CardGameEventPage.model_validate(self.match.events_since(
                after=self._sent_event_cursor, limit=256)).model_dump(mode='json')
            new = page['events']
            if new and new[0]['sequence'] != self._sent_event_cursor+1:
                raise RuntimeError('Native public event history skipped an acknowledged cursor')
            events.extend(deepcopy(new))
            cursor = page['next_cursor']
            if cursor < self._sent_event_cursor or (page['has_more'] and cursor == self._sent_event_cursor):
                raise RuntimeError('Native public event cursor failed to advance')
            self._sent_event_cursor = cursor
            if not page['has_more']:
                return events

    def events_for_reply(self):
        return self._event_delta()

    def acknowledge_step(self, command, admitted_turn):
        self._drive_targets = dict(command['joint_positions_rad'])
        return {'board_game': self.public_state(),
                'board_game_turn_changed': self.match.turn_index != admitted_turn,
                'card_game_events': self._event_delta()}

    def request(self, command, *, actor=None):
        try:
            if not isinstance(command, dict):
                raise ValueError('A game command must be an object')
            kind = command.get('type')
            events = None
            if kind == 'end_episode':
                if set(command) != {'type', 'reason'} or actor is not None:
                    raise ValueError('Only the trusted host may interrupt an episode')
                self.match.interrupt(command['reason'])
            elif kind == 'events_since':
                if set(command) != {'type', 'after', 'limit'} or actor is not None:
                    raise ValueError('Only the authenticated host may query public event history')
                events = self.match.events_since(after=command['after'], limit=command['limit'])
            else:
                player = self._require_actor(actor)
                if kind == 'cancel_intent':
                    if set(command) != {'type'}:
                        raise ValueError('Cancel intent has no extra fields')
                    self.match.cancel_intent(player)
                else:
                    self.match.admit_intent(player, command)
            result = {'ok': True, 'board_game': self.public_state(), 'score': self.score()}
            if events is not None:
                result['events'] = events
        except (ValueError, TypeError, KeyError) as exc:
            result = {'ok': True, 'board_game': self.public_state(), 'score': self.score(),
                      'board_game_error': {'code': 'invalid_arguments', 'message': str(exc)}}
        result['card_game_events'] = self._event_delta()
        if self.output is not None:
            with (self.output/'requests.jsonl').open('a') as stream:
                stream.write(json.dumps({'command': command, 'actor': actor, 'reply': result}, allow_nan=False)+'\n')
        return result

    def close(self, *, reason='runtime_error'):
        if self.closed:
            return self.observer.summary()
        if not self.match.terminal:
            self.match.interrupt(reason)
        summary = self.observer.close()
        self.closed = True
        if self.output is not None:
            report = {'schema': 'card_game_expansion_worker_evidence.v1', 'observer': summary,
                      'score': self.score(), 'public_state': self.public_state(),
                      'robot_target_overrides': False, 'object_state_writes': False,
                      'handoff': 'next_20ms_control_boundary; every_confirmed_phase_advances_authorization_epoch'}
            (self.output/'evidence.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
        if summary['failed'] or summary['final_context_verified'] is not True:
            raise RuntimeError('Expansion native observer failed final evidence verification')
        return summary
