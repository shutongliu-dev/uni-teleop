"""Ordinary-episode native runtime with a separately locked learner side."""
from copy import deepcopy
import json
from pathlib import Path

import numpy as np

from scenes.board_games.expansion_native_observer import ExpansionNativeObserver
from scenes.board_games.expansion_release import canonical_hash, sha256
from .referee import PracticeReferee


class PracticeObserver(ExpansionNativeObserver):
    def _observe(self, rows, sampled_before):
        # Same already acquired native rows, needed for horizontal disk/disk
        # contact; upward support alone cannot establish an impact.
        self.match.contact_rows = rows
        return super()._observe(rows, sampled_before)


class PracticeRuntime:
    def __init__(self, env, scene, task, output, *, evaluator_factory=None):
        self.env, self.task = env, task
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.evaluator = PracticeReferee(task)
        self.side = task['practice_case']['public']['active_side']
        if scene.get('card_practice_task') != {'schema': 'card_rsi_practice_native.v1',
                'task_id': task['id'], 'active_side': self.side,
                'private_case_sha256': task['practice_case']['private_case_sha256']}:
            raise ValueError('Practice native scene/case/side binding mismatch')
        self.names = tuple(env.actuated_joint_names)
        self.owned = frozenset(n for n in self.names if n.startswith((self.side+'_', 'fr3_'+self.side+'_')))
        if len(self.owned) != 29 or len(self.names) != 58:
            raise ValueError('Practice retains both original 29-DOF robots')
        self.targets = {n: float(env._targets[0, env.joint_names.index(n)]) for n in self.names}
        self.observer = PracticeObserver(env, self.evaluator, task['asset_metadata'],
            frame_jsonl=self.output/'native_frames.jsonl').attach()
        self.start_native_step = self.observer.last_step
        self.start_simulation_time_s = self.observer.last_time
        self.closed = False

    def public_state(self):
        return None

    def score(self):
        return deepcopy(self.evaluator.score())

    def authorize_step(self, command, actor=None):
        if actor is not None or self.closed or self.evaluator.terminal:
            raise ValueError('Practice requires an active ordinary single-learner episode')
        self.env._validate_action(command)
        if any(np.float32(v) != np.float32(self.targets[n]) for n, v in command['joint_positions_rad'].items() if n not in self.owned):
            raise ValueError('Practice command changed the inactive original robot drive')
        return None

    def acknowledge_step(self, command, admitted_turn):
        self.targets = dict(command['joint_positions_rad'])
        return {}

    def request(self, command, actor=None):
        raise ValueError('Practice uses ordinary public robot actions only')

    def close(self, reason='runtime_error'):
        if self.closed:
            return self.observer.summary()
        summary = self.observer.close()
        self.closed = True
        if not self.evaluator.terminal:
            self.evaluator.interrupt('missing_physical_evidence' if summary['failed'] else reason)
        from scripts.maintenance.freeze_project_sources import verify_if_snapshot
        verification = verify_if_snapshot(Path(__file__).resolve().parents[4], required_modules=(
            'scenes.workrooms.card_room.practice.runtime',
            'scenes.workrooms.card_room.practice.referee',
            'scenes.workrooms.card_room.practice.cases',
            'scenes.board_games.expansion_native_observer',
            'scenes.workrooms.card_room.expansion_referee'), entrypoint_file=Path(__file__))
        verification_path = self.output/'runtime_imports_final.json'
        verification_path.write_text(json.dumps(verification or {'verified': False,
            'runtime_imports_verified': False, 'reason': 'source_not_frozen'}, indent=2, allow_nan=False)+'\n')
        (self.output/'evidence.json').write_text(json.dumps({
            'schema': 'card_rsi_practice_native_evidence.v1', 'observer': summary,
            'episode_id': self.output.parents[1].name, 'task_id': self.task['id'],
            'active_side': self.side, 'case_id': self.evaluator.public['case_id'],
            'split': self.evaluator.public['split'],
            'native_start_step': self.start_native_step,
            'native_start_time_s': self.start_simulation_time_s,
            'physics_dt_s': self.observer.dt,
            'native_frames_sha256': sha256(self.output/'native_frames.jsonl'),
            'observer_inputs_sha256': sha256(self.output/'observer_inputs.json'),
            'runtime_imports_final_sha256': sha256(verification_path),
            'release_manifest_sha256': self.task['runtime_release_manifest_sha256'],
            'parent_release_sha256': self.evaluator.public['parent_release_sha256'],
            'public_rules_sha256': self.evaluator.public['public_rules_sha256'],
            'case_fingerprint': self.evaluator.case['case_fingerprint'],
            'evaluator_sources_sha256': canonical_hash(self.task['sources_sha256']),
            'score': self.score(), 'confirmations': self.evaluator.confirmations,
            'case_sha256': self.evaluator.public['case_sha256'],
            'private_case_sha256': self.evaluator.case['private_case_sha256'],
            'inactive_robot_drive_locked': True, 'object_state_writes': False,
            'robot_target_overrides': False, 'formal_game_result': False}, indent=2, allow_nan=False)+'\n')
        if summary['failed'] or summary['final_context_verified'] is not True:
            raise RuntimeError('Practice native observer failed final evidence verification')
        return summary


def build_runtime(env, scene, task, output, *, evaluator_factory):
    return PracticeRuntime(env, scene, task, output, evaluator_factory=evaluator_factory)
