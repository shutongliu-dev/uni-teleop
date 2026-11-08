"""Dispatch tests use CPU contract doubles; they establish no native behavior."""
from copy import deepcopy
import json
import sys
import types

import pytest

from bench.isaacsim import release
from bench.runtime import reset_cases
from bench.runtime import robot_configurations as bindings
from bench.runtime.worker import public_task_projection
from test_task_referee_contract import declared_task
from test_agent_service import frozen_inputs, runner, request


@pytest.fixture
def laundry_modules(monkeypatch):
    prefix = 'scenes.workrooms.laundry_room.randomization'
    package = types.ModuleType(prefix); package.__path__ = []
    modules = {name: types.ModuleType(prefix+'.'+name) for name in ('release', 'public')}
    monkeypatch.setitem(sys.modules, prefix, package)
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, prefix+'.'+name, module)
    return modules


def test_release_validation_and_case_selection_delegate_exact_identity(tmp_path, laundry_modules):
    manifest = {'schema': 'laundry_randomized_release.v1', 'configuration_id': 'dual_opposed'}
    (tmp_path/'release_manifest.json').write_text(json.dumps(manifest))
    calls = []
    def validate(root, value, *, assets):
        calls.append(('validate', root, deepcopy(value), assets))
        return value
    def inputs(root, task_id, value, *, seed, phase):
        calls.append(('inputs', root, task_id, deepcopy(value), seed, phase))
        return ({'id': task_id}, {'private_scene': True}, {'private_task': True}, {'joints': {}})
    laundry_modules['release'].validate_release = validate
    laundry_modules['release'].task_inputs = inputs
    assert release.validate(tmp_path, assets=False) == manifest
    assert calls == [('validate', tmp_path.resolve(), manifest, False)]
    for seed in (0, 9):
        expected = release.task_inputs(tmp_path, 'towel_stacking', seed=seed)
        assert expected[0]['id'] == 'towel_stacking' and len(expected) == 4
        assert calls[-2:] == [('validate', tmp_path.resolve(), manifest, True),
            ('inputs', tmp_path, 'towel_stacking', manifest, seed, 'nominal')]
    calls.clear()
    with pytest.raises(ValueError, match='practice'):
        release.task_inputs(tmp_path, 'towel_stacking', practice={'active_side': 'left'})
    assert all(row[0] == 'validate' for row in calls)


def test_worker_uses_separate_laundry_public_projection(laundry_modules):
    task = declared_task('towel_folding')
    task['reset_case'] = {'schema': 'laundry_reset_case.v1', 'private_nodes': [[1, 2, 3]]}
    calls = []
    def project(value, projection):
        calls.append(deepcopy(value))
        return {**projection, 'reset_case': {'fixture_public_identity': True}}
    laundry_modules['public'].project_reset_case = project
    result = public_task_projection(task)
    assert calls == [task]
    assert result['reset_case'] == {'fixture_public_identity': True}
    assert 'private_nodes' not in str(result)
    # The worker must still publish the original frozen referee, not fabricate
    # successful cloth state or use the fixture's hidden nodes in public fields.
    assert result['public_rules'] == task['public_rules']['referee']
    assert reset_cases.project_reset_case(task, {'task_id': task['id']})['reset_case'] == result['reset_case']


@pytest.mark.parametrize('configuration', ('single_arm', 'dual_same_side', 'dual_opposed'))
@pytest.mark.parametrize('fault', (None, 'installation', 'task', 'robot', 'parent_pin'))
def test_configuration_binding_preserves_exact_frozen_installation(tmp_path, monkeypatch, configuration, fault):
    robot = {'hand_id': 'sharpa_w01_right' if configuration == 'single_arm' else 'dual_fr3_sharpa_w01',
             'assembly_version': 'isaacsim_6_1_v2'}
    scene = {'robot': deepcopy(robot)}
    manifest = {'schema': 'laundry_randomized_release.v1', 'configuration_id': configuration,
                'tasks': [{'id': 'towel_folding', 'scene': 'scene.json'}]}
    binding = {'task_id': 'towel_folding', 'releases': {configuration:
        {'path': str(tmp_path), 'manifest_sha256': 'a'*64}},
        'robot_configurations': {configuration: {'robot_requirements': robot}}}
    if fault == 'installation':
        manifest['configuration_id'] = 'dual_opposed' if configuration != 'dual_opposed' else 'single_arm'
    elif fault == 'task':
        manifest['tasks'][0]['id'] = 'shirt_folding'
    elif fault == 'robot':
        scene['robot']['assembly_version'] = 'changed'
    (tmp_path/'scene.json').write_text(json.dumps(scene))
    monkeypatch.setattr(bindings, 'validate', lambda *a, **kw: manifest)
    monkeypatch.setattr(bindings, 'digest', lambda *a: ('b' if fault == 'parent_pin' else 'a')*64)
    if fault:
        with pytest.raises(ValueError):
            bindings.checked_configuration_release(binding, configuration)
    else:
        assert bindings.checked_configuration_release(binding, configuration) == tmp_path


@pytest.mark.parametrize('seed', (10, 71, 2**32-1))
@pytest.mark.parametrize('schema', ('laundry_randomized_release.v1',
                                  'living_room_randomized_release.v1',
                                  'workroom_randomized_release.v1'))
def test_http_out_of_pool_creation_is_configuration_error_before_case_or_worker(
        runner, frozen_inputs, monkeypatch, seed, schema):
    from fastapi.testclient import TestClient
    from bench.runtime import service
    from bench.runtime.server import create_app
    manifest = deepcopy(frozen_inputs[0])
    manifest.update(schema=schema, seed_values=list(range(10)))
    monkeypatch.setattr(service, 'validate', lambda *a, **kw: manifest)
    def no_case(*a, **kw):
        raise AssertionError('An unsupported seed reached the private case loader')
    monkeypatch.setattr(service, 'task_inputs', no_case)
    make, _, workers = runner
    api = make()
    with TestClient(create_app(service=api)) as http:
        response = http.post('/episodes', json=request(seed=seed).model_dump(mode='json'))
    assert response.status_code == 422
    assert '0 through 9' in response.json()['detail']
    assert not workers and not api.sessions
