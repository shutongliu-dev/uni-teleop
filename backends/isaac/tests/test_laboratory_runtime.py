"""Real frozen laboratory contracts over HTTP, with a CPU worker double.

The worker supplies no physical feasibility or robot manipulation evidence.
Native smoke and object probes remain separately source-bound experiments.
"""
from copy import deepcopy
import json
from pathlib import Path

from fastapi.testclient import TestClient
import pytest

from bench.control.factory import public_robot_geometry, robot_configuration
from bench.isaacsim.release import digest, load_evaluator, task_inputs, validate
from bench.protocol import PolicyRegistry
from bench.protocol.task_referee import PublicTaskRequirements, project_task_rules
from bench.runtime import service
from bench.runtime.server import create_app
from bench.runtime.worker import public_task_projection
from bench.simulation.isaacsim_backend.native_task_bundle import load_observer
from scenes.workrooms.catalog import load_task, task_definition
from scenes.workrooms.laboratory.rules import TASK_IDS
from test_agent_service import FakeWorker, PRIVATE_MARKER, join_action, publish_targets, request
from test_bench_function_http import post


@pytest.fixture(scope='module', params=TASK_IDS)
def laboratory(request):
    task_id = request.param
    definition = task_definition('laboratory', task_id)
    root = Path(definition['source_release']['path'])
    if not (root/'release_manifest.json').is_file():
        pytest.skip('Requires the published immutable laboratory release')
    assert digest(root/'release_manifest.json') == definition['source_release']['manifest_sha256']
    manifest = validate(root)
    inputs = task_inputs(root, task_id, seed=71)
    return task_id, root, manifest, inputs


def test_five_frozen_releases_bind_seed_rules_native_reader_and_catalog(laboratory):
    task_id, root, manifest, (_, scene, task, initial) = laboratory
    assert manifest['status'] == 'prepared_unvalidated'
    assert manifest['physics_accepted'] is False
    assert manifest['task_policy_success_claimed'] is False
    assert task['episode_seed'] == 71
    assert scene['robot']['hand_id'] == 'sharpa_w01_right'
    assert len(initial['joint_positions_rad']) == 29
    assert set(scene['cameras']) == {'overview', 'task', 'room', 'wrist'}
    bundle = scene['native_task_bundle']
    assert bundle['observer']['configuration'] == task
    assert bundle['rigid_bodies'] == task['asset_metadata']['actors']
    observer = load_observer(bundle['observer'])
    evaluator = load_evaluator(root)(task)
    assert observer.digest == evaluator.digest
    assert observer.__class__.__module__.startswith('_native_task_observer_')
    assert evaluator.__class__.__module__.startswith('_isaac_frozen_evaluator_')
    assert public_task_projection(task) == project_task_rules(task)
    assert evaluator.inspect_conditions()['success'] is False
    assert ('reduced_dynamics' in bundle) == (task_id == 'pipette_transfer')
    if task_id == 'pipette_transfer':
        provider = bundle['reduced_dynamics']
        assert provider['configuration'] == task
        assert provider['files'] == bundle['observer']['files']
        assert Path(provider['path']).is_relative_to(root)
        assert digest(provider['path']) == provider['sha256']
        assert bundle['volume_visual_bindings'] == task['asset_metadata']['volume_visual_bindings']
    loaded = load_task('laboratory', task_id)
    _, default_scene, default_task, default_initial = task_inputs(root, task_id)
    # The catalog retains release-relative Python paths; the worker resolves
    # those same frozen paths for import, without reauthoring the scene.
    for hook in ('observer', 'reduced_dynamics'):
        if hook in loaded['scene']['native_task_bundle']:
            row = loaded['scene']['native_task_bundle'][hook]
            row['path'] = str((root/row['path']).resolve())
    assert (loaded['scene'], loaded['task'], loaded['initial_state']) == (
        default_scene, default_task, default_initial)
    with pytest.raises(ValueError, match='exact task and original reset'):
        task_inputs(root, task_id, phase='at_goal')
    assert digest(root/'release_manifest.json') == task_definition(
        'laboratory', task_id)['source_release']['manifest_sha256']


def laboratory_service(tmp_path, laboratory, *, mutate_initial=None, on_step=None):
    task_id, root, _, (_, scene, task, initial) = laboratory
    workers = []
    geometry = public_robot_geometry(robot_configuration(scene))

    def factory(**kwargs):
        assert kwargs['task_id'] == task_id and kwargs['seed'] == 71
        worker = FakeWorker(**kwargs, depth=True, acknowledged_targets=True,
            on_step=on_step or publish_targets)
        worker.arm = list(geometry.arm.names)
        worker.hand = list(geometry.hand.names)
        worker.active = list(geometry.names)
        worker.raw['joint_positions_rad'] = deepcopy(initial['joint_positions_rad'])
        worker.raw['joint_velocities_rad_s'] = dict.fromkeys(worker.active, 0.)
        worker.raw['robot'] = {'joint_targets_rad': deepcopy(initial['joint_positions_rad'])}
        worker.initial['observation'] = deepcopy(worker.raw)
        worker.initial['metadata'].update(scene_spec=deepcopy(scene), action_spec={
            'joint_names': worker.active, 'arm_joint_names': worker.arm,
            'hand_joint_names': worker.hand,
            'joint_limits_rad': {name: [float(geometry.lower[i]), float(geometry.upper[i])]
                                 for i, name in enumerate(geometry.names)}})
        worker.score.update(load_evaluator(root)(task).inspect_conditions())
        worker.score['private_liquid_ledger'] = PRIVATE_MARKER
        worker.initial['score'] = deepcopy(worker.score)
        worker.initial['task'] = public_task_projection(task)
        if mutate_initial is not None:
            mutate_initial(worker.initial)
        workers.append(worker)
        return worker

    return service.AgentService(tmp_path/'episodes', release=root,
        registry=PolicyRegistry(), worker_factory=factory), workers


def test_five_laboratory_tasks_use_ordinary_http_rules_hold_and_sealed_result(tmp_path, laboratory):
    task_id, _, _, (_, _, task, initial) = laboratory
    api, workers = laboratory_service(tmp_path, laboratory)
    with TestClient(create_app(service=api)) as client:
        listing = client.get('/tasks')
        assert listing.status_code == 200
        assert listing.json() == [project_task_rules(task)]
        assert PRIVATE_MARKER not in listing.text
        created = client.post('/episodes', json=request(task_id=task_id, seed=71).model_dump(mode='json'))
        assert created.status_code == 201, created.text
        ready = created.json()
        assert PRIVATE_MARKER not in created.text
        public = PublicTaskRequirements.model_validate(ready['task'])
        assert public.create_endpoint == 'POST /episodes'
        assert public.referee.state == 'ongoing'
        assert public.episode.terminal is False
        session = api.get(ready['episode_id'])
        endpoint = '/episodes/'+ready['episode_id']
        catalog = client.get(endpoint+'/functions').json()
        assert 'get_task' in {row['name'] for row in catalog['functions']}
        read = post(client, endpoint, 'get_task', {}, 'lab/task')
        assert read.result.task_id == task_id
        assert read.context.budget.action_attempts == 0
        assert not workers[-1].calls or all(op != 'step' for op, _ in workers[-1].calls)
        accepted = post(client, endpoint, 'hold', {
            'action_id': 'lab_hold', 'observation_sequence': session.sequence,
            'duration_s': .04}, 'lab/hold', expected_status=202)
        assert accepted.result.status == 'accepted'
        join_action(session)
        status = post(client, endpoint, 'get_action', {'action_id': 'lab_hold'}, 'lab/action')
        assert status.result.status == 'completed'
        assert status.result.elapsed_sim_s == pytest.approx(.04)
        assert status.context.budget.action_attempts == 1
        executed = [payload for op, payload in workers[-1].calls if op == 'step']
        assert len(executed) == 2
        assert {payload['action_id'] for payload in executed} == {'lab_hold'}
        assert all(set(payload['command']) == {'joint_positions_rad', 'duration_s'} for payload in executed)
        assert all(payload['command']['joint_positions_rad'] == initial['joint_positions_rad']
                   for payload in executed)
        public = post(client, endpoint, 'get_task', {}, 'lab/after_hold').result
        assert public.referee.state == 'ongoing' and not public.episode.terminal
        finished = post(client, endpoint, 'finish_episode', {}, 'lab/finish')
        assert finished.result.outcome == 'cancelled'
        assert finished.result.action_attempts == 1
        assert finished.result.completion_sim_s is None
        assert workers[-1].closed
        assert post(client, endpoint, 'get_result', {}, 'lab/result').result == finished.result
        ended = post(client, endpoint, 'get_task', {}, 'lab/ended').result
        assert ended.referee.state == 'ongoing'
        assert ended.episode.terminal and ended.episode.outcome == 'cancelled'
        private = (session.directory/'private/evaluator_evidence.json').read_text()
        assert PRIVATE_MARKER in private  # The canary actually reached private storage.
        public_records = json.dumps([ready, read.model_dump(mode='json'),
            finished.model_dump(mode='json'), ended.model_dump(mode='json')])
        assert PRIVATE_MARKER not in public_records
        assert '/World/LaboratoryTask' not in public_records
        assert 'private_liquid_ledger' not in public_records


@pytest.mark.parametrize('change', [
    lambda initial: initial.pop('task'),
    lambda initial: initial['task']['public_rules']['thresholds'].update(stable_s=.01),
    lambda initial: initial['task'].update(private_objects=PRIVATE_MARKER),
])
def test_laboratory_rejects_worker_task_identity_drift_before_publication(
        tmp_path, laboratory, change):
    api, workers = laboratory_service(tmp_path, laboratory, mutate_initial=change)
    with pytest.raises(service.ServiceError, match='Native worker episode targets differ'):
        api.create(request(task_id=laboratory[0], seed=71))
    assert len(workers) == 1 and workers[0].closed
    assert not api.sessions
    assert all(op != 'step' for op, _ in workers[0].calls)


@pytest.mark.parametrize('change', [
    {'robot_id': 'dual_fr3_sharpa_w01'}, {'scene_variant': 'enriched_v2'},
])
def test_laboratory_rejects_noncanonical_robot_or_scene_before_worker(tmp_path, laboratory, change):
    api, workers = laboratory_service(tmp_path, laboratory)
    with pytest.raises(service.ServiceError, match='frozen single-arm'):
        api.create(request(task_id=laboratory[0], seed=71, **change))
    assert workers == []


def test_worker_omits_task_for_legacy_contract_without_public_rules():
    assert public_task_projection({'id': 'legacy'}) is None
