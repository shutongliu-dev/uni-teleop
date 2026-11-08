"""Admission is checked before unavailable robots or native workers are used."""
from copy import deepcopy
import json
import sys

import pytest
from fastapi.testclient import TestClient

from bench.runtime import service
from bench.runtime.server import create_app
from bench.runtime.task_capabilities import TaskCapabilityError, require_robot_episode_capability
from test_agent_service import frozen_inputs, request


ENTRYPOINT = 'scripts.validation.laundry_cloth_interactions'
PRIVATE = 'private_evaluator_canary_892718'


@pytest.mark.parametrize('task', [{}, {'capabilities': {}},
    {'capabilities': {'robot_episodes': True}},
    {'capabilities': {'engineering_cloth_probe': True}}])
def test_legacy_and_robot_capable_tasks_keep_admission(task):
    before = deepcopy(task)
    assert require_robot_episode_capability(task) is None
    assert task == before


@pytest.mark.parametrize('entrypoint', [None, '../' + PRIVATE,
    'scripts.validation.probe; ' + PRIVATE, {'private': PRIVATE}])
def test_denial_does_not_echo_private_or_executable_text(entrypoint):
    task = {'capabilities': {'robot_episodes': False, 'engineering_entrypoint': entrypoint,
                            'private_reason': PRIVATE}, 'evaluation': {'private': PRIVATE}}
    with pytest.raises(TaskCapabilityError) as error:
        require_robot_episode_capability(task)
    assert '/episodes' in str(error.value)
    assert 'engineering' in str(error.value)
    assert PRIVATE not in str(error.value)


def test_http_rejects_engineering_only_task_before_robot_or_worker(tmp_path, monkeypatch, frozen_inputs):
    manifest, inputs = deepcopy(frozen_inputs)
    entry, scene, task, initial = inputs
    scene['robot'] = {'enabled': False}
    task['capabilities'] = {'robot_episodes': False, 'task_inputs': True,
        'engineering_cloth_probe': True, 'engineering_entrypoint': ENTRYPOINT,
        'private_reason': PRIVATE}
    task['evaluation'] = {'private': PRIVATE}
    monkeypatch.setattr(service, 'validate', lambda release: manifest)
    monkeypatch.setattr(service, 'task_inputs', lambda *args, **kwargs: (entry, scene, task, initial))
    calls = []

    def forbidden_worker(**kwargs):
        calls.append(kwargs)
        raise AssertionError('Native worker must not be launched for engineering-only tasks')

    api = service.AgentService(tmp_path / 'runner', worker_factory=forbidden_worker)
    with TestClient(create_app(service=api), raise_server_exceptions=False) as client:
        response = client.post('/episodes', json=request().model_dump(mode='json'))
    assert response.status_code == 403, response.text
    assert response.json() == {'detail':
        f'This task does not support ordinary robot /episodes. Use the engineering script: python -m {ENTRYPOINT}.'}
    assert PRIVATE not in response.text
    assert calls == []
    assert api.sessions == {}


def test_direct_worker_launch_is_rejected_before_native_environment(tmp_path, monkeypatch):
    from bench.isaacsim import release
    from bench.runtime import worker
    from bench.simulation.isaacsim_backend import env, robot_registry

    task = {'capabilities': {'robot_episodes': False, 'engineering_entrypoint': ENTRYPOINT}}
    monkeypatch.setattr(release, 'task_inputs', lambda *args, **kwargs: ({}, {}, task, {}))
    forbidden_calls = []

    def forbidden_setup(*args, **kwargs):
        forbidden_calls.append(True)
        raise AssertionError('Robot/native setup must follow capability admission')

    monkeypatch.setattr(env, 'FR3DexterousIsaacEnv', forbidden_setup)
    monkeypatch.setattr(robot_registry, 'scene_with_hand', forbidden_setup)
    monkeypatch.setattr(worker, 'verify_if_snapshot', lambda *args, **kwargs: None)

    class Socket:
        def __init__(self, *args):
            pass
        def bind(self, *args):
            pass
        def listen(self, *args):
            pass
        def accept(self):
            return self, None
        def close(self):
            pass

    monkeypatch.setattr(worker.socket, 'socket', Socket)
    replies = []
    monkeypatch.setattr(worker, 'send', lambda connection, reply: replies.append(reply))
    config = tmp_path / 'config.json'
    config.write_text(json.dumps({'output': str(tmp_path / 'native'), 'release': 'unused',
        'task_id': 'engineering_fixture', 'seed': 0, 'hand_id': 'unused', 'device': 'cuda:0'}))
    monkeypatch.setattr(sys, 'argv', ['worker', '--config', str(config), '--socket', 'unused'])
    with pytest.raises(TaskCapabilityError, match=ENTRYPOINT):
        worker.main()
    assert forbidden_calls == []
    assert len(replies) == 1 and replies[0]['ok'] is False
