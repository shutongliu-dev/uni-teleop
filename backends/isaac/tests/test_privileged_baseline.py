"""Real client/HTTP/classical flow, with a CPU worker double; no Isaac claim."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from bench.paths import load_json
from bench.protocol import PolicyRegistry
from bench.control.fingertip import ClassicalFingertipPolicy
from bench.isaacsim.release import task_inputs
from scenes.manipulation import DEFAULT_TASK_RELEASE
from bench.runtime.controllers import robot_configuration
from bench.runtime.service import AgentService
from scripts.validation import run_privileged_baseline as harness
from test_agent_service import FakeWorker, PRIVATE_MARKER
from test_model_geometry_actions import public_geometry_inputs


class HoldStrategy:
    def __init__(self, scene, task):
        self.count = 0

    def next_decision(self, observation, private, previous_feedback):
        assert observation['profile'] == 'vision_proprio'
        assert not any(name.startswith('privileged') for name in observation)
        assert private['sequence'] == observation['sequence']
        assert private['sim_time_s'] == observation['sim_time_s']
        assert PRIVATE_MARKER in private['objects']
        assert set(private) == {'sequence', 'sim_time_s', 'objects', 'cloth'}
        assert previous_feedback is None if self.count == 0 else previous_feedback['status'] == 'completed'
        # Deliberate mutation demonstrates this is not a live state handle.
        private['objects'][PRIVATE_MARKER]['position_m'][0] = 999.
        self.count += 1
        return {'action': 'hold', 'duration_s': .04}

    def diagnostic(self):
        return {'decisions': self.count}


@pytest.fixture
def factory(tmp_path, monkeypatch):
    monkeypatch.setattr(harness, 'verify_if_snapshot', lambda *a, **kw: {'verified': True, 'test_double': True})
    _, scene, _, _ = task_inputs(DEFAULT_TASK_RELEASE, '01_cube_push')
    services, workers = [], []

    def make(directory, *, release, device, success=False):
        policy = ClassicalFingertipPolicy(**robot_configuration(scene))
        policy.robot_configuration = robot_configuration(scene)
        geometry = policy.geometry
        registry = PolicyRegistry()
        registry.register(policy)

        def worker_factory(**kwargs):
            worker = FakeWorker(**kwargs, depth=True)
            worker.arm = list(geometry.arm.names)
            worker.hand = list(geometry.hand.names)
            worker.active = worker.arm + worker.hand
            q = {joint.name: float(np.clip(0., joint.lower, joint.upper)) for joint in geometry.joints}
            q.update(dict(zip(worker.arm, scene['robot']['arm_home_rad'])))
            worker.raw['joint_positions_rad'] = q
            worker.raw['joint_velocities_rad_s'] = dict.fromkeys(q, 0.)
            worker.initial['metadata']['action_spec'] = {
                'joint_names': worker.active, 'arm_joint_names': worker.arm,
                'hand_joint_names': worker.hand,
                'joint_limits_rad': {j.name: [j.lower, j.upper] for j in geometry.joints}}
            worker.initial['observation'] = deepcopy(worker.raw)
            if success:
                worker.on_step = lambda w: w.score.update(success=True)
            workers.append(worker)
            return worker

        service = AgentService(directory, release=release, device=device,
                               registry=registry, worker_factory=worker_factory)
        services.append(service)
        return service

    return make, services, workers


def args(tmp_path, count=2):
    return SimpleNamespace(output=tmp_path / 'baseline', release=DEFAULT_TASK_RELEASE,
                           task='01_cube_push', device='cuda:0', max_actions=count, seed=0)


def test_holds_use_original_action_pipeline_and_isolated_private_state(tmp_path, factory, monkeypatch):
    make, services, workers = factory
    original_http, original_interaction = harness.BenchHTTPClient, harness.InteractionSession
    deadlines = {}
    def http(*args, **kwargs):
        deadlines['http'] = kwargs['timeout_s']
        return original_http(*args, **kwargs)
    def interaction(*args, **kwargs):
        deadlines['interaction'] = kwargs['deadline']
        return original_interaction(*args, **kwargs)
    monkeypatch.setattr(harness, 'BenchHTTPClient', http)
    monkeypatch.setattr(harness, 'InteractionSession', interaction)
    report = harness.run(args(tmp_path), service_factory=make, strategy_factory=HoldStrategy)
    assert report['error'] is None
    assert report['model_calls'] == 0
    assert report['completed_actions'] == report['submitted_actions'] == report['primitive_attempts'] == 2
    assert report['result']['outcome'] == 'cancelled'
    assert report['episode_artifacts_verified']
    assert not report['accepted'] and not report['physics_valid'] and not report['native_isaac']
    assert workers[0].closed
    assert workers[0].raw['objects'][PRIVATE_MARKER]['position_m'][0] == .123
    assert services[0].get(report['result']['spec']['episode_id']).policy.descriptor.policy_id == 'classical_fingertip'
    folder = tmp_path / 'baseline/decisions/0001'
    assert load_json(folder / 'action.json')['skill'] == 'hold'
    assert load_json(folder / 'execution.json')['status'] == 'completed'
    assert (folder / 'before/task.png').is_file()
    assert (folder / 'before/task_depth.npy').is_file()
    assert (tmp_path / 'baseline/private/strategy_inputs/0001.json').is_file()
    trajectory = load_json(tmp_path / 'baseline/trajectory.json')
    assert trajectory['model_calls'] == 0 and len(trajectory['decisions']) == 2
    assert trajectory['decisions'][1]['action']['action'] == 'hold'
    assert trajectory['decisions'][1]['feedback']['status'] == 'completed'
    assert trajectory['decisions'][1]['after']['sequence'] > trajectory['decisions'][1]['before']['sequence']
    assert report['trajectory_complete'] and trajectory['integrity']['complete']
    assert deadlines == {'http': None, 'interaction': None}
    spec = report['result']['spec']
    assert spec['budgets']['wall_time_s'] is None and spec['budgets']['max_actions'] == 2
    assert spec['budgets']['max_action_sim_s'] == 5.
    assert spec['agent_id'] == 'privileged_scripted_manipulation'


def test_invalid_script_action_rejects_before_native_and_still_seals(tmp_path, factory):
    make, _, workers = factory

    class InvalidStrategy(HoldStrategy):
        def next_decision(self, *unused):
            return {'action': 'move_geometry', 'duration_s': .4, 'wrist_delta_m': [9., 0., 0.]}

    report = harness.run(args(tmp_path), service_factory=make, strategy_factory=InvalidStrategy)
    assert report['error']['type'] == 'ActionValidationError'
    assert report['primitive_attempts'] == 1 and report['submitted_actions'] == 0
    assert workers[0].closed
    assert not any(op == 'step' for op, _ in workers[0].calls)
    assert report['result']['outcome'] == 'cancelled'
    assert report['episode_artifacts_verified']
    assert (tmp_path / 'baseline/decisions/0000/rejection.json').is_file()
    trajectory = load_json(tmp_path / 'baseline/trajectory.json')
    assert len(trajectory['decisions']) == 1
    assert trajectory['decisions'][0]['after'] is None
    assert 'rejection.json' in trajectory['decisions'][0]['artifacts']


def test_strategy_error_preserves_prior_execution_and_cleans_worker(tmp_path, factory):
    make, _, workers = factory

    class BrokenStrategy(HoldStrategy):
        def next_decision(self, *arguments):
            if self.count:
                raise RuntimeError('deliberate scripted fault')
            return super().next_decision(*arguments)

    report = harness.run(args(tmp_path), service_factory=make, strategy_factory=BrokenStrategy)
    assert report['error'] == {'type': 'RuntimeError', 'detail': 'deliberate scripted fault'}
    assert report['completed_actions'] == 1
    assert workers[0].closed and not report['cleanup_errors']
    assert report['episode_artifacts_verified']
    assert load_json(tmp_path / 'baseline/report.json') == report


def test_fake_task_success_does_not_become_native_acceptance(tmp_path, factory):
    make, _, _ = factory
    report = harness.run(args(tmp_path), service_factory=lambda *a, **kw: make(*a, **kw, success=True),
                         strategy_factory=HoldStrategy)
    assert report['result']['outcome'] == 'success'
    assert report['task_success']
    assert not report['accepted'] and not report['physics_valid']
    assert report['physics_audit'] is None


def test_oracle_rejects_stale_public_decision(tmp_path, factory):
    make, services, _ = factory
    report = harness.run(args(tmp_path, count=1), service_factory=make, strategy_factory=HoldStrategy)
    session = services[0].get(report['result']['spec']['episode_id'])
    observation = session.observe()
    different_episode = deepcopy(observation)
    different_episode['episode_id'] = 'another_episode'
    with pytest.raises(RuntimeError, match='different episodes'):
        harness.privileged_state(session, different_episode)
    observation['sequence'] -= 1
    with pytest.raises(RuntimeError, match='not synchronized'):
        harness.privileged_state(session, observation)


def test_editable_root_is_not_allowed_to_start_native(tmp_path, monkeypatch):
    monkeypatch.setattr(harness, 'verify_if_snapshot', lambda *a, **kw: None)
    with pytest.raises(ValueError, match='complete frozen source snapshot'):
        harness.run(args(tmp_path), strategy_factory=HoldStrategy)
    assert not (tmp_path / 'baseline').exists()


def test_broken_diagnostic_cannot_prevent_worker_seal_or_failure_report(tmp_path, factory):
    make, _, workers = factory

    class BrokenDiagnostic(HoldStrategy):
        def diagnostic(self):
            raise ValueError('deliberate diagnostic fault')

    report = harness.run(args(tmp_path), service_factory=make, strategy_factory=BrokenDiagnostic)
    assert report['error']['type'] == 'ValueError'
    assert report['cleanup_errors'][0]['stage'] == 'strategy_diagnostic'
    assert workers[0].closed
    assert report['episode_artifacts_verified'] and not report['accepted']
    assert (tmp_path / 'baseline/report.json').is_file()


@pytest.fixture
def recorded_trajectory(tmp_path, public_geometry_inputs):
    output = tmp_path / 'recorded'
    base_observation, capabilities, spec = deepcopy(public_geometry_inputs)
    spec['episode_id'] = 'episode'
    config = {'release_manifest_sha256': 'a'*64, 'strategy_implementation': {'version': 'test'}}
    harness.save(output / 'config.json', config)
    ready = {'episode_id': 'episode', 'spec': spec, 'capabilities': capabilities}
    previous_feedback = None
    for index in range(2):
        folder = output / 'decisions' / f'{index:04d}'
        action_id = f'action_{index}'
        decision = {'action': 'hold', 'duration_s': .04}
        before = deepcopy(base_observation)
        before.update(episode_id='episode', sequence=index*2, sim_time_s=index*.04)
        after = {'episode_id': 'episode', 'sequence': (index+1)*2, 'sim_time_s': (index+1)*.04,
                 'last_action_id': action_id, 'last_action_status': 'completed'}
        feedback = {'action_id': action_id, 'status': 'completed', 'sim_time_s': after['sim_time_s'], 'elapsed_sim_s': .04}
        command = harness.build_geometry_action(decision, before, capabilities, spec,
            action_id=action_id, expected_sequence=before['sequence'], actions_used=index)
        for name, value in {'decision.json': decision, 'before/observation.json': before,
                'after_observation.json': after, 'execution.json': feedback,
                'execution_source.json': feedback, 'accepted.json': {'action_id': action_id, 'status': 'accepted'},
                'action.json': command,
                'primitive_request.json': {'decision': decision, 'expected_sequence': before['sequence']},
                'strategy_diagnostic.json': {}}.items():
            harness.save(folder / name, value)
        harness.save(output / 'private/strategy_inputs' / f'{index:04d}.json', {
            'observation_sequence': before['sequence'], 'previous_feedback': previous_feedback,
            'privileged_state': {'sequence': before['sequence'], 'sim_time_s': before['sim_time_s']}})
        previous_feedback = feedback
    return output, ready, config, {'primitive_attempts': 2, 'submitted_actions': 2, 'completed_actions': 2}


def test_complete_trajectory_is_verified(recorded_trajectory):
    output, ready, config, counts = recorded_trajectory
    result = harness.package_trajectory(output, ready, config, expected_counts=counts)
    assert result['integrity']['complete'] and not result['integrity']['issues']


@pytest.mark.parametrize('mutation', ['missing_execution', 'missing_last_decision', 'sequence_gap', 'action_id', 'time', 'malformed_json'])
def test_incomplete_trajectory_keeps_records_and_cannot_claim_complete(recorded_trajectory, mutation):
    import shutil
    output, ready, config, counts = recorded_trajectory
    folder = output / 'decisions/0001'
    if mutation == 'missing_execution':
        (folder / 'execution.json').unlink()
    elif mutation == 'missing_last_decision':
        shutil.rmtree(folder)
    elif mutation == 'malformed_json':
        (folder / 'decision.json').write_text('{broken')
    else:
        name = 'before/observation.json' if mutation == 'sequence_gap' else 'execution.json'
        value = load_json(folder / name)
        value[{'sequence_gap': 'sequence', 'action_id': 'action_id', 'time': 'sim_time_s'}[mutation]] = {
            'sequence_gap': 999, 'action_id': 'unrelated_action', 'time': 999.}[mutation]
        harness.save(folder / name, value)
    result = harness.package_trajectory(output, ready, config, expected_counts=counts)
    assert not result['integrity']['complete'] and result['integrity']['issues']
    assert (output / 'trajectory.json').is_file()


def test_episode_ledger_must_cover_every_recorded_file(tmp_path):
    episode = tmp_path / 'episode'
    harness.save(episode / 'one.json', {'one': 1})
    harness.save(episode / 'two.json', {'two': 2})
    full = {name: harness.digest(episode / name) for name in ('one.json', 'two.json')}
    harness.save(episode / 'artifact_hashes.json', full)
    assert harness.check_episode_artifacts(episode)
    harness.save(episode / 'artifact_hashes.json', {'one.json': full['one.json']})
    assert not harness.check_episode_artifacts(episode)
