"""Study referee lifecycle on existing native substeps and public joint actions."""
from copy import deepcopy
import importlib
import json
from pathlib import Path
import uuid
import zipfile

import numpy as np

from bench.simulation.isaacsim_backend.contact_sample import SubstepContactExchange
from .native_observer import NativeObserver, raw_contacts


class StudyTaskRuntime:
    def __init__(self, env, scene, task, output, *, evaluator_factory):
        self.env, self.task = env, deepcopy(task)
        # A canonical release also freezes the evidence conversion, so a later
        # mutable scene edit cannot silently change an old release's scorer input.
        if evaluator_factory.__module__.startswith('_isaac_frozen_evaluator_'):
            observer_module = importlib.import_module(
                evaluator_factory.__module__.rsplit('.', 1)[0] + '.native_observer')
            self.observer = observer_module.NativeObserver(env, task)
            self.raw_contacts = observer_module.raw_contacts
        else:
            self.observer = NativeObserver(env, task)
            self.raw_contacts = raw_contacts
        self.evaluator = evaluator_factory(task, trusted_normalized_input=True)
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.steps = self.actions = 0
        self.action_id = None
        self.action_ids = set()
        self.closed = False
        self.episode_id = uuid.uuid4().hex
        self.original = env.world.step
        self.exchange = None
        self.archive = zipfile.ZipFile(self.output / 'study_native_evidence.zip', 'x',
            compression=zipfile.ZIP_DEFLATED, compresslevel=1)
        try:
            frame = self._frame(env.capture_state(), [], 0.)
            self.evaluator.update(frame)
            self._record(frame)
            if self.score().get('referee', {}).get('state') == 'evaluator_invalid':
                raise RuntimeError('Study referee rejected the native reset baseline')
            self.exchange = SubstepContactExchange(env, self)
            env.world.step = self._step
        except BaseException:
            self.archive.close()
            raise

    def _frame(self, state, rows, time_s):
        return self.observer.frame(state, rows, episode_id=self.episode_id,
            step_index=self.steps, time_s=time_s, actions=self.actions)

    def _record(self, frame):
        self.archive.writestr(f'{self.steps:09d}.json',
            json.dumps(frame, ensure_ascii=False, allow_nan=False))

    def _step(self, *args, **kwargs):
        if args or kwargs != {'render': False}:
            raise RuntimeError('Study observer only observes existing native physics steps')
        env = self.env
        try:
            self.exchange.begin_step(env)
            result = self.original(*args, **kwargs)
            sample = self.exchange.acquire(env)
            before = env.capture_state()
            rows = self.raw_contacts(sample.arrays, env._contact_sensor_paths, env._contact_filter_paths)
            self.steps += 1
            frame = self._frame(before, rows, float(env._current_step_time))
            projected = env._project_rigid_contacts(sample)
            after = env.capture_state()
            if set(before) != set(after) or any(not np.array_equal(before[k], after[k]) for k in before):
                raise RuntimeError('Study observer changed native state')
            self.exchange.offer_after_readonly(env, sample, projected)
            self.evaluator.update(frame)
            self._record(frame)
            return result
        except BaseException:
            self.exchange.invalidate()
            self.evaluator.finish('evidence_missing')
            raise

    def score(self):
        return deepcopy(self.evaluator.inspect_conditions())

    def public_state(self):
        # Ordinary /episodes exposes the generic public task/referee contract.
        return None

    def begin_action(self, action_id):
        if not isinstance(action_id, str) or not action_id.strip():
            raise ValueError('Study native commands require a public action identity')
        if self.closed or self.score().get('terminal') is True:
            raise RuntimeError('Study episode is already terminal')
        if action_id != self.action_id:
            if action_id in self.action_ids:
                raise ValueError('A previous study action cannot resume after a newer action')
            maximum = self.task['evaluation']['configuration'].get('budget', {}).get('max_robot_actions')
            if maximum is not None and self.actions >= maximum:
                raise RuntimeError('Study robot action budget is exhausted')
            self.action_ids.add(action_id)
            self.action_id = action_id
            self.actions += 1

    def authorize_step(self, command, actor=None):
        if self.action_id is None:
            raise ValueError('Study native commands require a public action identity')
        if self.closed or self.score().get('terminal') is True:
            raise RuntimeError('Study episode is already terminal')
        return None

    def acknowledge_step(self, command, admitted_turn):
        return {}

    def request(self, command, actor=None):
        raise ValueError('Study tasks use ordinary episode joint actions')

    def close(self, reason='budget'):
        if self.closed:
            return None
        self.closed = True
        self.env.world.step = self.original
        receipt = None
        try:
            if self.exchange is not None:
                receipt = self.exchange.close(self.env, self)
            self.evaluator.finish('evidence_missing' if receipt is not None and
                not receipt['all_samples_consumed_once'] else
                {'budget': 'budget_exhausted', 'cancelled': 'cancelled', 'timeout': 'budget_exhausted'}
                .get(reason, 'evidence_missing'))
        finally:
            self.archive.close()
        (self.output / 'study_sampling_receipt.json').write_text(
            json.dumps(receipt, indent=2, allow_nan=False) + '\n')
        return receipt


def build_runtime(env, scene, task, output, *, evaluator_factory):
    return StudyTaskRuntime(env, scene, task, output, evaluator_factory=evaluator_factory)
