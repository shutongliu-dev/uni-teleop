"""Current ordinary workroom releases share one public execution boundary.

Actual immutable task contracts are loaded, but the native worker is a CPU
double. These are integration checks, not scene or manipulation qualification.
"""
from copy import deepcopy
import json
from pathlib import Path

from fastapi.testclient import TestClient
import pytest

from bench.control.factory import public_robot_geometry, robot_configuration
from bench.isaacsim.release import digest, task_inputs
from bench.paths import load_json
from bench.protocol import PolicyRegistry
from bench.runtime import service
from bench.runtime.server import create_app
from bench.runtime.worker import public_task_projection
from scenes.workrooms.catalog import list_tasks
from test_agent_service import FakeWorker, PRIVATE_MARKER, join_action, publish_targets, request
from test_bench_function_http import artifact_hashes, commands, post
from test_scene_function_runner import reset_score


ORDINARY = tuple(row for row in list_tasks()
                 if row['task_id'] not in ('tic_tac_toe_3x3', 'connect_four_5x5',
                                           'hex7x7', 'tower_extraction'))
NIST = tuple(row for row in ORDINARY if row['task_id'].startswith('nist_'))


@pytest.fixture(params=ORDINARY, ids=lambda row: row['family']+'/'+row['task_id'])
def workroom_service(request, tmp_path):
    definition = request.param
    root = Path(definition['source_release']['path'])
    if not (root/'release_manifest.json').is_file():
        pytest.skip('Requires the current immutable workroom release')
    assert digest(root/'release_manifest.json') == definition['source_release']['manifest_sha256']
    workers = []

    def factory(**kwargs):
        _, scene, task, initial = task_inputs(root, kwargs['task_id'], seed=kwargs['seed'])
        geometry = public_robot_geometry(robot_configuration(scene))
        worker = FakeWorker(**kwargs, depth=True, acknowledged_targets=True, on_step=publish_targets)
        worker.arm, worker.hand = list(geometry.arm.names), list(geometry.hand.names)
        worker.active = list(geometry.names)
        worker.raw['joint_positions_rad'] = deepcopy(initial['joint_positions_rad'])
        worker.raw['joint_velocities_rad_s'] = dict.fromkeys(worker.active, 0.)
        worker.raw['robot'] = {'joint_targets_rad': deepcopy(initial['joint_positions_rad'])}
        worker.raw['task_state'] = {'private_native_geometry': PRIVATE_MARKER}
        worker.raw['native_task_bundle'] = {'private_contacts': PRIVATE_MARKER}
        worker.score.update(reset_score(root, task))
        worker.score['private_inventory'] = PRIVATE_MARKER
        worker.initial.update(observation=deepcopy(worker.raw), score=deepcopy(worker.score),
                              task=public_task_projection(task))
        worker.initial['metadata'].update(scene_spec=scene, action_spec={
            'joint_names': worker.active, 'arm_joint_names': worker.arm,
            'hand_joint_names': worker.hand,
            'joint_limits_rad': {name: [float(geometry.lower[i]), float(geometry.upper[i])]
                                 for i, name in enumerate(geometry.names)}})
        workers.append(worker)
        return worker

    api = service.AgentService(tmp_path/'episodes', release=root, registry=PolicyRegistry(),
                               worker_factory=factory)
    yield api, workers, definition
    for session in api.sessions.values():
        if session.result is None:
            session.finish()


def test_all_current_ordinary_tasks_bind_worker_contract_replay_and_cancel(workroom_service):
    api, workers, definition = workroom_service
    # Legacy scenarios pin initialization; episode-bound tasks (including all
    # NIST adaptations) must also work with an explicitly requested nonzero seed.
    manifest = load_json(api.release/'release_manifest.json')
    frozen_scene = load_json(api.release/manifest['tasks'][0]['scene'])
    seed = frozen_scene.get('scenario', {}).get('seed', 71)
    with TestClient(create_app(service=api)) as client:
        listing = client.get('/tasks')
        assert listing.status_code == 200
        assert [row['task_id'] for row in listing.json()] == [definition['task_id']]
        created = client.post('/episodes', json=request(
            task_id=definition['task_id'], seed=seed).model_dump(mode='json'))
        assert created.status_code == 201, created.text
        ready = created.json()
        assert ready['spec']['task_release_sha256'] == definition['source_release']['manifest_sha256']
        assert ready['spec']['seed'] == seed
        session = api.get(ready['episode_id'])
        endpoint = '/episodes/'+ready['episode_id']
        assert ready['task']['create_endpoint'] == 'POST /episodes'
        assert session.public_task == workers[0].initial['task']
        assert len(session.capabilities.arm_joints) == 7
        assert len(session.capabilities.hand_joints) == 22
        if session.referee_contract is not None:
            public_task = post(client, endpoint, 'get_task', {}, 'audit/task').result.model_dump(mode='json')
        else:
            public_task = client.get(endpoint+'/task').json()
        assert public_task['task_id'] == definition['task_id']
        original = ready['observation']['proprioception']['commanded_joint_positions_rad']
        target = dict(original)
        target['fr3_joint7'] += .01
        movement = {
            'action_id': 'shared_move', 'observation_sequence': session.sequence,
            'duration_s': .12,
            'arm_joint_positions_rad': {name: target[name] for name in session.capabilities.arm_joints},
            'hand_joint_positions_rad': {name: target[name] for name in session.capabilities.hand_joints},
        }
        post(client, endpoint, 'move_joints', movement, 'audit/move', expected_status=202)
        join_action(session)
        moved = post(client, endpoint, 'get_action', {'action_id': 'shared_move'}, 'audit/moved')
        assert moved.result.status == 'completed'
        assert moved.result.elapsed_sim_s == pytest.approx(.12)
        executed = commands(workers[0])
        assert len(executed) == 6 and executed[-1]['joint_positions_rad'] == target
        assert session.snapshot.proprioception.commanded_joint_positions_rad == target
        args = {'action_id': 'shared_hold', 'observation_sequence': session.sequence,
                'duration_s': .04}
        post(client, endpoint, 'hold', args, 'audit/hold', expected_status=202)
        join_action(session)
        completed = post(client, endpoint, 'get_action', {'action_id': 'shared_hold'}, 'audit/status')
        assert completed.result.status == 'completed'
        steps = sum(op == 'step' for op, _ in workers[0].calls)
        assert steps == 8
        # A direct hold retains the last acknowledged command for every joint.
        assert all(row['joint_positions_rad'] == target for row in commands(workers[0])[-2:])
        replay = post(client, endpoint, 'hold', args, 'audit/replay')
        assert replay.result == completed.result
        assert sum(op == 'step' for op, _ in workers[0].calls) == steps
        assert replay.context.budget.action_attempts == 2
        moved_replay = post(client, endpoint, 'move_joints', movement, 'audit/move-replay')
        # Status carries the current episode clock, while the original action's
        # elapsed time, outcome and contact receipt remain unchanged.
        assert moved_replay.result.model_dump(exclude={'sim_time_s'}) == moved.result.model_dump(
            exclude={'sim_time_s'})
        assert moved_replay.result.sim_time_s == pytest.approx(.16)
        assert sum(op == 'step' for op, _ in workers[0].calls) == steps
        assert moved_replay.context.budget.action_attempts == 2
        finished = post(client, endpoint, 'finish_episode', {}, 'audit/finish')
        assert finished.result.outcome == 'cancelled' and workers[0].closed
        assert finished.result.spec.model_dump(mode='json') == ready['spec']
        assert finished.result.action_attempts == 2
        assert finished.result.elapsed_sim_s == pytest.approx(.16)
        sealed = artifact_hashes(session.directory)
        sealed_calls = deepcopy(workers[0].calls)
        for function in ('get_result', 'finish_episode', 'get_result'):
            result = post(client, endpoint, function, {}, 'audit/sealed/'+function)
            assert result.result == finished.result
            assert result.context.request_recorded is False
        assert artifact_hashes(session.directory) == sealed
        assert workers[0].calls == sealed_calls
        if definition['task_id'] == 'tower_building':
            assert finished.result.scoring.score == 0.
            assert finished.result.scoring.score_unit == 'm'
        else:
            assert finished.result.scoring is None
        public = json.dumps([listing.json(), ready, public_task,
                             moved.model_dump(mode='json'), completed.model_dump(mode='json'),
                             finished.model_dump(mode='json')])
        assert PRIVATE_MARKER not in public
        assert 'private_inventory' not in public
        assert 'private_native_geometry' not in public and 'private_contacts' not in public
        assert str(api.release) not in public
        assert PRIVATE_MARKER in (session.directory/'private/evaluator_evidence.json').read_text()


@pytest.mark.parametrize('changes', [
    {'robot_id': 'allegro_v5_right'}, {'scene_variant': 'enriched_v2'},
])
def test_all_current_ordinary_tasks_reject_frozen_robot_or_scene_changes(workroom_service, changes):
    api, workers, definition = workroom_service
    with pytest.raises(service.ServiceError, match='frozen single-arm'):
        api.create(request(task_id=definition['task_id'], seed=71, **changes))
    assert workers == [] and api.sessions == {}


@pytest.mark.parametrize('workroom_service', NIST, indirect=True,
                         ids=lambda row: row['task_id'])
@pytest.mark.parametrize('projection', ['missing', 'different'])
def test_nist_worker_contract_must_match_before_observations_are_published(
        workroom_service, projection):
    api, workers, definition = workroom_service
    factory = api.worker_factory

    def mismatched_factory(**kwargs):
        worker = factory(**kwargs)
        if projection == 'missing':
            del worker.initial['task']
        else:
            worker.initial['task']['public_rules_sha256'] = '0'*64
        return worker

    api.worker_factory = mismatched_factory
    with TestClient(create_app(service=api)) as client:
        response = client.post('/episodes', json=request(
            task_id=definition['task_id'], seed=71).model_dump(mode='json'))
        assert response.status_code == 503
        assert response.json() == {'detail': 'Native worker episode targets differ from the public task'}
        assert PRIVATE_MARKER not in response.text
    assert api.sessions == {}
    assert len(workers) == 1 and workers[0].closed and workers[0].closed_force
    assert workers[0].calls == []
