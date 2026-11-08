"""CPU deadline counterexamples and real Bench consumption, not native physics."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from bench.paths import DATA_ROOT, load_json
from bench.protocol import Budgets, PolicyRegistry
from bench.runtime import service
from bench.runtime.worker import public_task_projection
from scenes.workrooms.catalog import task_definition
from scenes.workrooms.laboratory.evaluation import SequenceEvaluator
from scenes.workrooms.laboratory.rules import TASK_IDS
from test_agent_service import FakeWorker, request, action, join_action


class SummaryTrace:
    """Complete consecutive CPU summaries; no simulated object trajectory."""
    def __init__(self, task, factory=SequenceEvaluator):
        self.evaluator = factory(task)
        self.index = -1
        self.result = self.send()

    def send(self, *, goal=False, actions=1, exhausted=False, **changes):
        self.index += 1
        t = self.index*.02
        e = self.evaluator
        state = {'schema': 'laboratory_native_state.v1', 'configuration_sha256': e.digest,
            'task_id': e.c['task_id'], 'time_s': t, 'native_time_s': t,
            'native_step': self.index*40, 'baseline': self.index == 0,
            'complete': True, 'errors': [], 'events': [],
            'phase': 'settling' if goal else 'working',
            'reason': 'reset_baseline_not_scored' if self.index == 0 else
                'waiting_for_release_and_stability' if goal else 'task_in_progress',
            'released_stable_s': e.c['thresholds']['stable_s'] if goal else 0.,
            'terminal_failure': None, 'safety_stop': None, 'detail': {}}
        state.update(changes)
        self.result = e.update({'time_s': t, 'task_state': state,
                               'action_count': actions, 'budget_exhausted': exhausted})
        return self.result

    def through(self, time_s):
        while self.index < round(time_s/.02):
            self.send()
        return self.result


@pytest.fixture(params=TASK_IDS)
def task(request):
    reference = task_definition('laboratory', request.param)['source_release']
    return load_json(Path(reference['path'])/'task.json')


def assert_incomplete(result):
    assert result['status'] == 'incomplete' and result['terminal'] is True
    assert result['reason'] == 'budget_exhausted' and not result['success']
    assert not result['physical_failure'] and result['irrecoverable_reason'] is None
    assert result['referee']['state'] == 'ongoing' and not result['referee']['terminal']
    assert result['referee']['reason_codes'] == ['incomplete_budget']


@pytest.mark.parametrize('first_expired_goal', [False, True])
def test_time_horizon_precedes_success_and_latches(task, first_expired_goal):
    trace = SummaryTrace(task)
    assert trace.result['reason'] == 'reset_baseline_not_scored'
    trace.through(trace.evaluator.c['thresholds']['max_time_s'])
    assert trace.result['status'] == 'ongoing'  # The documented upper bound is inclusive.
    stopped = trace.send(goal=first_expired_goal)
    assert_incomplete(stopped)
    assert trace.send(goal=True) == stopped
    assert trace.evaluator.update({}) == stopped  # Late/malformed ACK cannot rewrite termination.


@pytest.mark.parametrize('trigger', ['actions', 'explicit'])
def test_action_or_explicit_budget_precedes_goal_and_latches(task, trigger):
    trace = SummaryTrace(task)
    trace.through(.3)
    options = {'actions': trace.evaluator.c['thresholds']['max_actions']+1} if trigger == 'actions' else {'exhausted': True}
    stopped = trace.send(goal=True, **options)
    assert_incomplete(stopped)
    assert trace.send(goal=True) == stopped
    trace.evaluator.reset()
    assert not trace.evaluator.incomplete and trace.evaluator.previous_time is None


def test_exact_time_and_action_boundaries_allow_earned_goal(task):
    trace = SummaryTrace(task)
    trace.through(trace.evaluator.c['thresholds']['max_time_s']-.02)
    result = trace.send(goal=True, actions=trace.evaluator.c['thresholds']['max_actions'])
    assert result['success'] and result['referee']['state'] == 'success'
    assert trace.send(exhausted=True) == result  # Already earned success remains immutable.


@pytest.mark.parametrize('physical', ['object_fell', 'physics_safety_stop'])
def test_real_physical_failure_and_safety_keep_their_existing_priority(task, physical):
    trace = SummaryTrace(task)
    trace.through(.3)
    changes = {'terminal_failure': physical} if physical == 'object_fell' else {'safety_stop': physical}
    result = trace.send(goal=True, exhausted=True, **changes)
    assert result['referee']['state'] == ('task_failure' if physical == 'object_fell' else 'safety_stop')
    assert result['referee']['terminal'] and not result['success']


@pytest.mark.parametrize('trigger', ['time', 'actions', 'explicit'])
def test_actual_episode_consumer_stops_frozen_budget_without_physical_failure(tmp_path, monkeypatch, trigger):
    root = Path(task_definition('laboratory', 'tube_clamp_transfer')['source_release']['path'])
    task = load_json(root/'task.json')
    trace = SummaryTrace(task)
    baseline = deepcopy(trace.result)
    trace.through(180. if trigger == 'time' else .3)
    options = {'actions': 9001} if trigger == 'actions' else {'exhausted': True} if trigger == 'explicit' else {}
    stopped = trace.send(goal=True, **options)
    ack_time = trace.index*.02
    assert_incomplete(stopped)
    workers = []

    def factory(**kwargs):
        def step(worker):
            worker.raw['sim_time_s'] = ack_time
            worker.score = deepcopy(stopped)
        worker = FakeWorker(on_step=step, **kwargs)
        worker.score = deepcopy(baseline)
        worker.initial['score'] = deepcopy(baseline)
        worker.initial['task'] = public_task_projection(task)
        workers.append(worker)
        return worker

    # Current release inputs were audited separately; avoid rehashing assets for
    # a pure scorer-to-service receipt test. The service and public consumer are real.
    manifest = load_json(root/'release_manifest.json')
    scene = load_json(root/'scene.json')
    initial = load_json(root/'initial_state.json')
    monkeypatch.setattr(service, 'validate', lambda _: deepcopy(manifest))
    monkeypatch.setattr(service, 'task_inputs', lambda *a, **k:
                        (deepcopy(manifest['tasks'][0]), deepcopy(scene), deepcopy(task), deepcopy(initial)))
    api = service.AgentService(tmp_path/'episodes', release=root,
                               registry=PolicyRegistry(), worker_factory=factory)
    ready = api.create(request(task_id=task['id'], budgets=Budgets(sim_time_s=None, wall_time_s=600.)))
    session = api.get(ready['episode_id'])
    session.submit(action(session))
    join_action(session)
    assert session.result.outcome == 'sim_timeout'
    assert session.result.physics_valid
    assert session.result.completion_sim_s is None
    assert session.sequence == 1 and session.raw['sim_time_s'] == ack_time
    public = session.task_requirements()
    assert public['episode'] == {'terminal': True, 'outcome': 'sim_timeout'}
    assert public['referee']['state'] == 'ongoing' and not public['referee']['terminal']
    assert workers[0].closed


def test_budget_derivation_replaces_only_the_reviewed_class():
    from scenes.workrooms.laboratory.budget_release import repaired_evaluator_source, _class_span
    # Keep a historical immutable source for the bounded migration regression.
    root = Path(f'{DATA_ROOT}/datasets/fr3_sharpa_bench/isaacsim/releases/laboratory/sample_sorting/v3')
    original = (root/'evaluator/sequence_evaluator.py').read_text()
    patched = repaired_evaluator_source(original)
    before, _, after = _class_span(original)
    patched_before, _, patched_after = _class_span(patched)
    assert (before, after) == (patched_before, patched_after)
    with pytest.raises(ValueError, match='Unreviewed'):
        repaired_evaluator_source(original.replace("return self._result('incomplete_budget')",
                                                   "return self._result('success')"))
    with pytest.raises(ValueError, match='Unreviewed'):
        repaired_evaluator_source(patched)
