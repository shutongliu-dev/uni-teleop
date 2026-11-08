"""Public placement intent from frozen tasks; CPU transport, no native claim."""
from copy import deepcopy
import json
from pathlib import Path

from fastapi.testclient import TestClient
import httpx
import pytest

from agent.model_client.function_http import FunctionHTTPClient
from agent.model_client.function_runner import task_requirements_context
from agent.model_client.function_session import FunctionSession
from bench.isaacsim.release import checked
from bench.paths import load_json
from bench.protocol.task_referee import PublicTaskRequirements, project_task_rules
from bench.runtime import service
from bench.runtime.server import create_app
from scenes.workrooms.laundry_room.catalog_definition import ROBOT_RELEASES
from scenes.workrooms.laundry_room.rules import ALL_TASK_IDS
from test_agent_service import PRIVATE_MARKER, frozen_inputs, runner, session_for
from test_task_referee_contract import declared_task, install_contract


def frozen_task(task_id):
    pointer = ROBOT_RELEASES[task_id]
    root = Path(pointer['path'])
    checked(root/'release_manifest.json', pointer['manifest_sha256'])
    manifest = load_json(root/'release_manifest.json')
    checked(root/'task.json', manifest['files']['task.json']['sha256'])
    return load_json(root/'task.json')


@pytest.mark.parametrize('task_id', tuple(ROBOT_RELEASES))
def test_frozen_laundry_goal_is_public_without_rewriting_its_rules(task_id):
    task = frozen_task(task_id)
    before = deepcopy(task)
    task['public_goal']['private_vertices'] = PRIVATE_MARKER
    public = project_task_rules(task)
    assert public['goal'] == before['public_goal']
    assert public['public_rules'] == before['public_rules']['referee']
    assert public['public_rules_sha256'] == before['public_referee_contract_sha256']
    assert task['public_rules'] == before['public_rules']
    assert PRIVATE_MARKER not in json.dumps(public)
    assert task_requirements_context(public)['goal'] == before['public_goal']
    public['goal']['target_center_xy_m'][0] += 1
    assert task['public_goal']['target_center_xy_m'] == before['public_goal']['target_center_xy_m']


@pytest.mark.parametrize('change', [
    {'target_center_xy_m': [float('nan'), 0.]},
    {'target_center_xy_m': [True, 0.]},
    {'target_center_xy_m': [0., 0., 0.]},
    {'frame_id': 'private_material_frame'},
    {'operation': 'other_task'},
    {'description': 'different instruction'},
])
def test_malformed_or_mismatched_static_goal_is_rejected(change):
    task = frozen_task('towel_flattening')
    task['public_goal'].update(change)
    with pytest.raises(ValueError):
        project_task_rules(task)


def test_unrelated_referee_task_projection_is_unchanged():
    task = declared_task()
    before = project_task_rules(task)
    task['public_goal'] = {'private_state': PRIVATE_MARKER}
    assert project_task_rules(task) == before
    assert 'goal' not in before


def install_goal(monkeypatch, inputs):
    install_contract(monkeypatch, inputs)
    provider = service.task_inputs
    def task_inputs(*args):
        values = provider(*args)
        task = values[2]
        task['schema_version'] = 'laundry-cloth-task-v1'
        task['public_goal'] = {'operation': task['id'], 'frame_id': 'world_m_z_up',
            'description': task['public_rules']['referee']['instruction_zh'],
            'target_center_xy_m': [.75, -.55], 'private_vertices': PRIVATE_MARKER}
        return values
    monkeypatch.setattr(service, 'task_inputs', task_inputs)


@pytest.mark.parametrize('tamper', [None, 'changed', 'removed'])
def test_get_task_and_model_context_preserve_bound_static_goal(
        runner, frozen_inputs, monkeypatch, tmp_path, tamper):
    install_goal(monkeypatch, frozen_inputs)
    make, _, workers = runner
    api = make()
    session, ready = session_for(api)
    expected = ready['task']['goal']
    assert expected['target_center_xy_m'] == [.75, -.55]
    with TestClient(create_app(service=api)) as transport:
        assert transport.get(f'/episodes/{session.spec.episode_id}/task').json()['goal'] == expected
        original = transport.request
        def altered(method, url, **kwargs):
            reply = original(method, url, **kwargs)
            if tamper and (kwargs.get('json') or {}).get('function') == 'get_task':
                data = reply.json()
                if tamper == 'changed':
                    data['result']['goal']['target_center_xy_m'][0] += .1
                else:
                    data['result'].pop('goal')
                return httpx.Response(reply.status_code, json=data)
            return reply
        monkeypatch.setattr(transport, 'request', altered)
        client = FunctionHTTPClient('http://testserver', session.spec.episode_id,
                                    client=transport, spec=ready['spec'])
        client.bind_episode(ready['spec'], task=ready['task'])
        host = FunctionSession(client, output=tmp_path/'agent', spec=ready['spec'],
            trial_id='static_goal', max_actions=2, max_calls=5)
        host.initialize()
        feedback = host.execute({'function': 'get_task', 'arguments': {}})
        assert feedback['ok'] is (tamper is None)
        if tamper is None:
            typed = PublicTaskRequirements.model_validate(feedback['result'])
            assert typed.goal.model_dump(mode='json') == expected
            assert task_requirements_context(feedback['result'])['goal'] == expected
            assert host.context()['last_feedback']['result']['goal'] == expected
            assert PRIVATE_MARKER not in json.dumps(feedback)
        else:
            assert feedback['error']['error_category'] == 'invalid_response'
        assert host.actions == session.clock.action_attempts == 0
        assert not any(op == 'step' for op, _ in workers[-1].calls)
