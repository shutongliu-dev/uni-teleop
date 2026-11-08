"""Run the native smoke's motion plan through real HTTP/System0 with a CPU worker."""
import pytest
from fastapi.testclient import TestClient

from bench.runtime.server import create_app
from bench.isaacsim.release import task_inputs
from bench.runtime.worker import public_task_projection
from scenes.workrooms.catalog import task_definition
from scripts.validation.public_http_smoke import geometry_smoke
from scripts.validation.public_http_smoke import agent_readonly_probe
from test_agent_service import join_action, request
from test_bench_function_http import post
from test_scene_function_runner import mechanical_service


@pytest.mark.parametrize('family,task_id,expected_hands', [
    ('laboratory', 'solid_weighing', {'right'}),
    ('mechanical_assembly', 'cooperative_insertion', {'left', 'right'}),
])
@pytest.mark.parametrize('hold_duration_s', [.12, .5])
@pytest.mark.parametrize('profile', ['original', 'l_mount'])
def test_geometry_smoke_uses_public_system0_actions(tmp_path, family, task_id, expected_hands, hold_duration_s, profile):
    # Both physical robot profiles publish the task; the mount changes the hand frames.
    release = task_definition(family, task_id, robot_profile=profile)['source_release']['path']
    api, workers, _ = mechanical_service(tmp_path, task_id, family, release=release)
    robot_id = task_inputs(release, task_id)[1]['robot']['hand_id']
    with TestClient(create_app(service=api)) as http:
        policy_id = 'classical_fingertip_bimanual' if len(expected_hands) > 1 else 'classical_fingertip'
        controller, = [row for row in http.get('/policies', params={'task_id': task_id}).json()
                       if row['policy_id'] == policy_id]
        payload = request(task_id=task_id, robot_id=robot_id, track='system0',
            policy_id=policy_id, policy_version=controller['version']).model_dump(mode='json')
        response = http.post('/episodes', json=payload)
        assert response.status_code == 201, response.text
        created = response.json()
        session = api.get(created['episode_id'])
        endpoint = '/episodes/' + created['episode_id']
        guide = post(http, endpoint, 'get_robot', {}, 'guide').result.model_dump(mode='json')

        action_times = {}
        def action(name, function, arguments):
            parameters = dict(arguments, action_id=name, observation_sequence=session.sequence)
            post(http, endpoint, function, parameters, name+'/submit', expected_status=202)
            join_action(session)
            status = post(http, endpoint, 'get_action', {'action_id': name}, name+'/status')
            assert status.result.status == 'completed', status
            observed = post(http, endpoint, 'observe', {}, name+'/observe').result.model_dump(mode='json')
            post(http, endpoint, function, parameters, name+'/replay')
            repeated = post(http, endpoint, 'observe', {}, name+'/repeat').result.model_dump(mode='json')
            assert repeated == observed
            action_times[name] = observed['sim_time_s']
            return observed

        options = {} if hold_duration_s == .12 else {'hold_duration_s': hold_duration_s}
        final, motion = geometry_smoke(action, created['observation'], guide, **options)
        assert set(motion['hands']) == expected_hands
        assert final['sim_time_s'] > 0
        returned = next(value for name, value in action_times.items() if name.startswith('return-'))
        assert final['sim_time_s']-returned == pytest.approx(hold_duration_s)
        result = post(http, endpoint, 'finish_episode', {}, 'finish').result
        assert result.outcome == 'cancelled' and result.action_attempts == 3
    assert workers[0].closed


def test_agent_probe_accepts_seeded_placement_without_structured_referee(tmp_path):
    api, workers, _ = mechanical_service(tmp_path, 'task0', 'card_room')
    factory = api.worker_factory

    def seeded_worker(**kwargs):
        worker = factory(**kwargs)
        task = task_inputs(kwargs['release'], kwargs['task_id'], seed=kwargs['seed'])[2]
        worker.initial['task'] = public_task_projection(task)
        return worker

    api.worker_factory = seeded_worker
    with TestClient(create_app(service=api)) as http:
        controller, = [row for row in http.get('/policies', params={'task_id': 'task0'}).json()
                       if row['policy_id'] == 'classical_fingertip']
        payload = request(task_id='task0', seed=71, track='system0',
            policy_id=controller['policy_id'], policy_version=controller['version']).model_dump(mode='json')
        response = http.post('/episodes', json=payload)
        assert response.status_code == 201, response.text
        created = response.json()
        receipt, calls = agent_readonly_probe(http, 'http://testserver', family='card_room',
            task_id='task0', created=created, phase='created',
            expected_observation=created['observation'], create_request=payload, controller=controller)
        assert receipt['task_requirements_source'] == 'validated_episode_ready'
        assert receipt['agent_ready_contract_verified'] and receipt['public_rules_sha256'] is None
        assert [row['request']['function'] for row in calls] == ['observe', 'observe']
        assert created['task']['episode_seed'] == 71
        assert len(created['task']['goal']['target_cells']) == 3
        api.get(created['episode_id']).finish()
    assert workers[0].closed and all(operation != 'step' for operation, _ in workers[0].calls)
