"""Trusted board-game state machine inside the native worker.

Only the frozen native observer can commit a move. Player requests declare an
intent; they cannot provide evidence, logical occupancy, or object transforms.
Each admitted robot command lasts the existing 20 ms control period. A commit
may occur inside it, but no subsequent command is admitted for the old turn.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, fields
import json
from pathlib import Path

import numpy as np

from scenes.board_games.tabletop import (
    ActuatorOwnership, BoardGameGeometry, BoardGameMatch, BoardGameNativeObserver,
    BoardGamePlayerProtocol, SettlementThresholds,
)
from scenes.board_games.tabletop.release import verify_public_rule_contract


class BoardGameWorker:
    """Private native owner; its public_state is an explicit safe projection."""

    def __init__(self, env, scene, task, *, output=None):
        verify_public_rule_contract(task)
        binding = scene.get('board_game_task', {})
        if binding.get('schema') != 'board_game_native_task.v1':
            raise ValueError('A frozen board-game native task binding is required')
        geometry = BoardGameGeometry(**{item.name: task['geometry'][item.name]
                                      for item in fields(BoardGameGeometry)})
        thresholds = SettlementThresholds(**{item.name: task['settlement'][item.name]
                                             for item in fields(SettlementThresholds)})
        self.match = BoardGameMatch(geometry, thresholds=thresholds)
        self.env = env
        self.output = None if output is None else Path(output)
        if self.output is not None:
            self.output.mkdir(parents=True, exist_ok=True)
        players = binding.get('players')
        if not isinstance(players, list) or len(players) != 2:
            raise ValueError('A board-game task requires two frozen player bindings')
        expected = [dict(player_id=player, color=color, robot_id=side)
                    for player, color, side in (('player_0', 'black', 'left'),
                                                ('player_1', 'white', 'right'))]
        if players != expected:
            raise ValueError('Board-game players must retain frozen black/left and white/right ownership')
        names = tuple(env.actuated_joint_names)
        ownership = tuple(ActuatorOwnership(row['player_id'], row['robot_id'],
                           tuple(name for name in names if name.startswith(
                               (row['robot_id']+'_', 'fr3_'+row['robot_id']+'_'))))
                          for row in players)
        if set(names) != {name for owner in ownership for name in owner.joint_names}:
            raise ValueError('Every native actuator must have exactly one player owner')
        self.protocol = BoardGamePlayerProtocol(self.match, ownership)
        self._owners = {owner.player_id: owner for owner in ownership}
        # Cache commanded references, not measured q. The inactive articulation
        # keeps its last native target even when contact creates tracking error.
        self._drive_targets = {name: float(env._targets[0, env.joint_names.index(name)]) for name in names}
        self.observer = BoardGameNativeObserver(env, self.match,
            board_fixture_id=binding['board_fixture_id'],
            frame_jsonl=None if self.output is None else self.output/'native_frames.jsonl').attach()
        self.closed = False

    def public_state(self):
        match = self.match
        return {'protocol': 'bench_board_game_state_v1', 'task_id': match.geometry.game_id,
                'turn_index': match.rules.turn_index,
                'current_player': None if match.terminal else match.rules.current_player,
                'terminal': match.terminal, 'winner': match.rules.winner, 'draw': match.rules.draw,
                'pending_move': asdict(match.pending) if match.pending else None,
                'status': match.public_status(),
                'settlement_dwell_s': float(match.last_result.dwell_s) if match.last_result else 0.,
                'commit_count': len(match.commit_log)}

    def score(self):
        """Private evaluator result; game completion does not name the agent winner."""
        match = self.match
        return {'schema': 'board_game_worker_score.v1',
                'success': match.outcome in ('win', 'draw'), 'reason': match.public_status()['reason'],
                'outcome': match.outcome, 'winner': match.rules.winner, 'draw': match.rules.draw,
                'end_reason': match.end_reason, 'turn_index': match.rules.turn_index,
                'commit_log': deepcopy(match.commit_log),
                'native_observer_samples': self.observer.samples,
                'native_observer_unsupported_samples': self.observer.unsupported_samples}

    def _require_actor(self, actor):
        if (not isinstance(actor, dict) or set(actor) != {'player_id', 'turn_index'}
                or not isinstance(actor['player_id'], str)
                or actor['player_id'] not in self._owners
                or type(actor['turn_index']) is not int):
            raise ValueError('A trusted player and turn binding is required')
        self.match.require_actor(actor['player_id'])
        if actor['turn_index'] != self.match.rules.turn_index:
            raise ValueError('The command belongs to an earlier turn')
        return self._owners[actor['player_id']]

    def authorize_step(self, command, actor):
        owner = self._require_actor(actor)
        self.env._validate_action(command)
        owned = frozenset(owner.joint_names)
        for name, value in command['joint_positions_rad'].items():
            # Isaac stores position targets as float32. Equality at that actual
            # precision permits serialization roundoff, not a different drive.
            if name not in owned and np.float32(value) != np.float32(self._drive_targets[name]):
                raise ValueError('A player command changed the other robot drive target')
        return self.match.rules.turn_index

    def acknowledge_step(self, command, admitted_turn):
        self._drive_targets = dict(command['joint_positions_rad'])
        return {'board_game': self.public_state(),
                'board_game_turn_changed': self.match.rules.turn_index != admitted_turn}

    def request(self, command, *, actor=None):
        """Handle only trusted intent/end IPC. Invalid requests preserve the match."""
        try:
            if not isinstance(command, dict):
                raise ValueError('A board-game command must be an object')
            kind = command.get('type')
            if kind == 'end_episode':
                if set(command) != {'type', 'reason'} or actor is not None:
                    raise ValueError('Only the trusted runtime may end a board-game episode')
                if not self.match.terminal:
                    self.match.end_episode(command['reason'])
                elif command['reason'] not in ('time_limit', 'step_limit', 'runtime_error', 'cancelled'):
                    raise ValueError('Unknown episode interruption reason')
            else:
                owner = self._require_actor(actor)
                if kind not in ('placement_intent', 'cancel_placement'):
                    raise ValueError('Only placement_intent and cancel_placement are player commands')
                self.protocol.admit_action(owner.player_id, command)
            result = {'ok': True, 'board_game': self.public_state(), 'score': self.score()}
        except (ValueError, TypeError, KeyError) as exc:
            result = {'ok': True, 'board_game': self.public_state(), 'score': self.score(),
                      'board_game_error': {'code': 'invalid_arguments', 'message': str(exc)}}
        if self.output is not None:
            with (self.output/'requests.jsonl').open('a') as stream:
                stream.write(json.dumps({'command': command, 'actor': actor, 'reply': result},
                                        allow_nan=False)+'\n')
        return result

    def close(self, *, reason='runtime_error'):
        if self.closed:
            return self.observer.summary()
        if not self.match.terminal:
            self.match.end_episode(reason)
        summary = self.observer.close()
        self.closed = True
        if self.output is not None:
            report = {'schema': 'board_game_worker_evidence.v1', 'observer': summary,
                      'score': self.score(), 'public_state': self.public_state(),
                      'turn_handoff': 'stop_at_next_20ms_control_boundary_no_new_old_turn_commands',
                      'robot_target_overrides': False, 'object_state_writes': False}
            (self.output/'evidence.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
        if summary['failed'] or summary['final_context_verified'] is not True:
            raise RuntimeError('Board-game native observer failed final evidence verification')
        return summary
