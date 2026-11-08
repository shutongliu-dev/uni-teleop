"""Engineering task discovery publishes typed static rules, never private cloth state."""
from copy import deepcopy
import json
from types import SimpleNamespace

from fastapi.testclient import TestClient
import pytest

from bench.protocol import PolicyRegistry
from bench.protocol.task_referee import contract_sha256, project_task_rules, validated_public_contract
from bench.runtime.server import create_app
from bench.runtime.task_capabilities import project_engineering_task
from scenes.workrooms.laundry_room.rules import TASK_IDS, PRACTICAL_PROFILE, public_rules, bind_task_rules

PRIVATE = 'private_cloth_vertices_canary_47284'


def laundry_task(task_id):
    rules = public_rules(task_id, profile=PRACTICAL_PROFILE)
    return bind_task_rules({'id': task_id, 'title_zh': '毛巾工程任务',
        'steps': [{'instruction_zh': rules['instruction_zh']}], 'public_rules': rules,
        'capabilities': {'robot_episodes': False, 'engineering_cloth_probe': True,
            'engineering_entrypoint': 'scripts.validation.laundry_cloth_interactions'},
        'asset_metadata': {'initial_vertices_m': PRIVATE},
        'evaluation': {'private_vertices': PRIVATE},
        'initial_state': {'cloths_vertices_m': PRIVATE}})


def http_fixture(tmp_path, monkeypatch, task, *, entry_id=None):
    (tmp_path/'task.json').write_text(json.dumps(task))
    manifest = {'tasks': [{'id': entry_id or task['id'], 'task': 'task.json'}]}
    from bench.isaacsim import release
    monkeypatch.setattr(release, 'validate', lambda *args, **kwargs: deepcopy(manifest))
    manager = SimpleNamespace(release=tmp_path, registry=PolicyRegistry(), shutdown=lambda: None)
    return TestClient(create_app(service=manager), raise_server_exceptions=False)


@pytest.mark.parametrize('task_id', TASK_IDS)
def test_http_laundry_rules_are_visible_without_robot_execution_or_private_evidence(tmp_path, monkeypatch, task_id):
    task = laundry_task(task_id)
    with http_fixture(tmp_path, monkeypatch, task) as client:
        response = client.get('/tasks')
    assert response.status_code == 200
    published, = response.json()
    assert published['task_kind'] == 'engineering'
    assert published['runnable'] is False and published['robot_episodes'] is False
    assert published['create_endpoint'] is None
    assert published['public_rules'] == task['public_rules']['referee']
    assert published['public_rules_sha256'] == contract_sha256(published['public_rules'])
    assert published['public_rules']['thresholds']['speed_node_percentile'] == 95.
    assert PRIVATE not in response.text
    assert 'initial_vertices_m' not in response.text and 'cloths_vertices_m' not in response.text
    assert 'evaluation' not in published and 'asset_metadata' not in published
    assert project_task_rules(task) is None
    published['public_rules']['success_conditions'].clear()
    assert task['public_rules']['referee']['success_conditions']


@pytest.mark.parametrize('damage', ('rules_hash', 'contract_hash', 'rules_path', 'task_id',
                                    'contract_private_field', 'nested_private_field', 'manifest_id'))
def test_invalid_engineering_rule_contract_is_not_published(tmp_path, monkeypatch, damage):
    task = laundry_task('towel_flattening')
    entry_id = None
    if damage == 'rules_hash':
        task['public_rules_sha256'] = '0'*64
    elif damage == 'contract_hash':
        task['public_referee_contract_sha256'] = '0'*64
    elif damage == 'rules_path':
        task['public_rules_file'] = '../rules.json'
    elif damage == 'task_id':
        task['id'] = 'wrong_identity'
    elif damage == 'manifest_id':
        entry_id = 'wrong_manifest_identity'
    else:
        contract = task['public_rules']['referee']
        if damage == 'contract_private_field':
            contract['private_vertices'] = PRIVATE
        else:
            contract['object_properties'][0]['private_vertices'] = PRIVATE
        # Correct hashes cannot make an undeclared private field public.
        task = bind_task_rules(task)
    with http_fixture(tmp_path, monkeypatch, task, entry_id=entry_id) as client:
        response = client.get('/tasks')
    assert response.status_code == 503
    assert response.json() == {'detail': 'Native service failed; operator diagnostics are private'}
    assert PRIVATE not in response.text and 'public_rules' not in response.json()
    # Engineering tasks still cannot be projected as ordinary episodes,
    # regardless of whether their optional discovery contract is malformed.
    assert project_task_rules(task) is None


def test_shared_contract_validation_does_not_change_ordinary_episode_projection():
    task = laundry_task('towel_flattening')
    task.pop('capabilities')
    contract = validated_public_contract(task)
    assert project_task_rules(task) == {'task_id': task['id'], 'task_kind': 'manipulation',
        'create_endpoint': 'POST /episodes', 'instruction': contract['instruction_zh'],
        'public_rules': contract, 'public_rules_sha256': contract_sha256(contract)}
    contract['success_conditions'].clear()
    assert task['public_rules']['referee']['success_conditions']


def test_no_declared_rules_preserves_the_original_engineering_response():
    task = {'id': 'generic', 'title_zh': '工程任务', 'steps': [{'instruction_zh': '整理'}],
            'capabilities': {'robot_episodes': False}}
    assert validated_public_contract(task) is None
    assert project_engineering_task(task, 'generic') == {
        'task_id': 'generic', 'task_kind': 'engineering', 'instruction': '工程任务。整理',
        'capabilities': {'robot_episodes': False}, 'robot_episodes': False,
        'engineering_entrypoint': None, 'create_endpoint': None, 'runnable': False,
        'availability_scope': 'Engineering task inputs only; ordinary robot execution is unavailable.',
        'unavailable_reason': 'This task does not support ordinary robot /episodes. Use the engineering validation entrypoint documented for this task.'}
    assert project_task_rules({'id': 'old', 'public_rules': {'goal': 'old'}}) is None
