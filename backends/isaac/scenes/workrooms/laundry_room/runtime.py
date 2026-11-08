"""Ordinary robot-joint lifecycle with a frozen, read-only laundry referee."""
from copy import deepcopy
import importlib
import io
import json
from pathlib import Path
import zipfile

import numpy as np

from bench.simulation.isaacsim_backend.contact_sample import SubstepContactExchange
from .rules import TASK_SCHEMA


def require_robot_episode(task):
    if task.get('schema_version') == TASK_SCHEMA and task.get('capabilities', {}).get('robot_episodes') is not True:
        raise ValueError('Laundry task-only release supports task_inputs and engineering cloth probes; ordinary /episodes requires a separately validated robot installation. Direct cloth-point commands are not robot joint actions.')


def _json(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


class LaundryTaskRuntime:
    def __init__(self, env, scene, task, output, *, evaluator_factory):
        require_robot_episode(task)
        if task['evaluation'].get('control_mode') != 'robot_joint_targets':
            raise ValueError('Laundry robot episodes require an explicit frozen joint-control contract')
        module = (importlib.import_module(evaluator_factory.__module__.rsplit('.', 1)[0]+'.native_observer')
                  if evaluator_factory.__module__.startswith('_isaac_frozen_evaluator_') else
                  importlib.import_module('scenes.workrooms.laundry_room.native_observer'))
        self.env, self.task = env, deepcopy(task)
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.observer = module.NativeObserver(env, scene, task)
        self.evaluator = evaluator_factory(task)
        self.steps = self.actions = 0
        self.action_id = None
        self.action_ids = set()
        self.closed = False
        self.original = env.world.step
        self.exchange = None
        self.states = []
        self.chunk_index = 0
        self.archive = zipfile.ZipFile(self.output/'laundry_native_evidence.zip', 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=1)
        self.trace = (self.output/'laundry_native_frames.jsonl').open('x')
        try:
            self.archive.writestr('topology.json', json.dumps(self.observer.topology, default=_json))
            self.archive.writestr('robot_colliders.json', json.dumps(self.observer.export, allow_nan=False))
            state = env.capture_state()
            frame = self.observer.frame(state, step_index=0, time_s=0.)
            self.evaluator.observe(frame)
            self._record(state, frame)
            if self.score().get('referee', {}).get('state') == 'evaluator_invalid':
                raise RuntimeError('Laundry referee rejected the native reset baseline')
            self.exchange = SubstepContactExchange(env, self)
            env.world.step = self._step
        except BaseException:
            self.archive.close()
            self.trace.close()
            raise

    def _record(self, state, frame):
        self.states.append({**{key: np.asarray(value).copy() for key, value in state.items()},
            # env.time_s is advanced after world.step returns. Preserve its raw
            # capture and also record this observer's native-aligned clock.
            'laundry_observed_time_s': np.asarray(frame['time_s']),
            'laundry_native_step': np.asarray(frame['native_step'])})
        row = {key: value for key, value in frame.items() if key != 'cloths'}
        row['score'] = self.score()
        self.trace.write(json.dumps(row, default=_json, separators=(',', ':'), allow_nan=False)+'\n')
        if len(self.states) >= 128:
            self._flush_states()

    def _flush_states(self):
        if not self.states:
            return
        stream = io.BytesIO()
        np.savez(stream, **{key: np.stack([row[key] for row in self.states]) for key in self.states[0]})
        self.archive.writestr(f'states/{self.chunk_index:06d}.npz', stream.getvalue())
        self.chunk_index += 1
        self.states.clear()
        self.trace.flush()

    def _invalidate(self, reason):
        # The scorer's public API validates a deliberately incomplete frame and
        # latches unknown evidence; no mutable flags or invented success input.
        self.evaluator.observe({'schema': 'laundry_native_frame.v1', 'complete': False,
                                'runtime_evidence_error': reason})

    def _step(self, *args, **kwargs):
        if args or kwargs != {'render': False}:
            raise RuntimeError('Laundry observer only observes existing native physics steps')
        env = self.env
        try:
            self.exchange.begin_step(env)
            result = self.original(*args, **kwargs)
            sample = self.exchange.acquire(env)
            before = env.capture_state()
            self.steps += 1
            frame = self.observer.frame(before, step_index=self.steps, time_s=float(env._current_step_time))
            projected = env._project_rigid_contacts(sample)
            after = env.capture_state()
            if set(before) != set(after) or any(not np.array_equal(before[key], after[key]) for key in before):
                raise RuntimeError('Laundry observer changed native physical state')
            self.exchange.offer_after_readonly(env, sample, projected)
            self.evaluator.observe(frame)
            self._record(before, frame)
            return result
        except BaseException:
            self.exchange.invalidate()
            self._invalidate('native_observer_failure')
            raise

    def score(self):
        return deepcopy(self.evaluator.report())

    def public_state(self):
        return None

    def _terminal(self):
        score = self.score()
        return score.get('terminal') is True or score.get('referee', {}).get('terminal') is True

    def begin_action(self, action_id):
        if not isinstance(action_id, str) or not action_id.strip():
            raise ValueError('Laundry native commands require a public action identity')
        if self.closed or self._terminal():
            raise RuntimeError('Laundry episode is already terminal')
        if action_id != self.action_id:
            if action_id in self.action_ids:
                raise ValueError('A previous laundry action cannot resume after a newer action')
            self.action_ids.add(action_id)
            self.action_id = action_id
            self.actions += 1

    def authorize_step(self, command, actor=None):
        if actor is not None:
            raise ValueError('Laundry bimanual tasks have one Agent and no players or turns')
        if self.action_id is None or self.closed or self._terminal():
            raise ValueError('Laundry requires an active ordinary episode action')
        return None

    def acknowledge_step(self, command, admitted_turn):
        return {}

    def request(self, command, actor=None):
        raise ValueError('Laundry tasks use ordinary episode joint actions')

    def close(self, reason='cancelled'):
        if self.closed:
            return None
        self.closed = True
        self.env.world.step = self.original
        receipt = {'reason': reason, 'native_steps': self.steps, 'robot_actions': self.actions,
                   'complete': False, 'public_cloth_state_write_interface': False}
        try:
            receipt['contact_exchange'] = self.exchange.close(self.env, self) if self.exchange else None
            receipt['native_observer'] = self.observer.close()
            receipt['complete'] = bool(receipt['contact_exchange'] and
                                      receipt['contact_exchange']['all_samples_consumed_once'])
            if not receipt['complete'] or reason == 'runtime_error':
                self._invalidate('incomplete_native_evidence')
            return receipt
        except BaseException:
            self._invalidate('native_close_verification_failed')
            raise
        finally:
            self._flush_states()
            self.trace.close()
            self.archive.close()
            receipt['final_score'] = self.score()
            (self.output/'laundry_sampling_receipt.json').write_text(
                json.dumps(receipt, indent=2, allow_nan=False)+'\n')


def build_runtime(env, scene, task, output, *, evaluator_factory):
    return LaundryTaskRuntime(env, scene, task, output, evaluator_factory=evaluator_factory)
