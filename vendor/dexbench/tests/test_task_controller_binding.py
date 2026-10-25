"""Task-bound controller discovery and CPU lifecycle checks; no native physics."""
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi.testclient import TestClient

from bench.paths import DATA_ROOT
from bench.protocol import PolicyRegistry
from bench.runtime import controllers, service
from bench.runtime.controllers import default_registry, robot_configuration
from bench.runtime.server import create_app
from bench.runtime.service import ServiceError
from test_agent_service import Teacher, frozen_inputs, request, runner


def _teacher(digest='c', configuration=None):
    policy = Teacher()
    policy.descriptor = policy.descriptor.model_copy(update={
        'implementation_sha256': digest*64, 'observation_profile': 'vision_proprio'})
    if configuration is not None:
        policy.robot_configuration = deepcopy(configuration)
    return policy


def test_task_factories_pin_descriptors_and_reject_unknown_selectors_without_calling():
    registry = PolicyRegistry()
    calls = []

    def factory():
        calls.append(True)
        return _teacher()

    descriptor = _teacher().descriptor
    registry.register_task_factory('task_a', descriptor, factory)
    descriptor.skills.append('UNREGISTERED')
    assert registry.descriptors()[0].skills == ['test_skill']
    for policy_id, version, task_id in (
            ('module.Class', 'v1', 'task_a'), ('test_teacher', 'v1', 'task_b'),
            ('test_teacher', 'v1', None)):
        with pytest.raises(ValueError, match='not registered'):
            registry.resolve(policy_id, version, task_id=task_id)
    assert not calls
    first = registry.resolve('test_teacher', 'v1', task_id='task_a')
    second = registry.resolve('test_teacher', 'v1', task_id='task_a')
    assert first is not second and len(calls) == 2
    with pytest.raises(ValueError, match='already registered'):
        registry.register_task_factory('task_a', first.descriptor, factory)
    with pytest.raises(ValueError, match='already registered'):
        registry.register(first)
    with pytest.raises(TypeError, match='callable'):
        registry.register_task_factory('task_b', first.descriptor, 'private.module:Factory')


@pytest.mark.parametrize('returned', [None, 'different_descriptor'])
def test_factory_output_must_match_the_pinned_policy(returned):
    registry = PolicyRegistry()
    registry.register_task_factory('task', _teacher().descriptor,
                                   lambda: None if returned is None else _teacher('d'))
    with pytest.raises(ValueError, match='invalid policy|descriptor changed'):
        registry.resolve('test_teacher', 'v1', task_id='task')


@pytest.fixture
def dual_layouts(monkeypatch):
    robot = {'assets_root': f'{DATA_ROOT}/assets', 'hand_id': 'dual_fr3_sharpa_w01',
             'base_position_m': [0., 0., 0.], 'base_quaternion_wxyz': [1., 0., 0., 0.],
             'arm_base_poses': {
                 'left': {'position_m': [.65, -.60, .90], 'quaternion_wxyz': [1., 0., 0., 0.]},
                 'right': {'position_m': [.65, .60, .90], 'quaternion_wxyz': [1., 0., 0., 0.]}}}
    scenes = {'task_a': {'robot': robot}, 'task_b': {'robot': deepcopy(robot)}}
    scenes['task_b']['robot']['arm_base_poses']['left']['position_m'] = [.45, -.70, .85]
    scenes['task_b']['robot']['arm_base_poses']['right']['position_m'] = [.75, .80, .95]
    manifest = {'tasks': [{'id': name} for name in scenes]}
    monkeypatch.setattr(controllers, 'validate', lambda *args, **kwargs: deepcopy(manifest))
    monkeypatch.setattr(controllers, 'task_inputs', lambda root, task: (
        {'id': task}, deepcopy(scenes[task]), {}, {}))
    return scenes, manifest


def test_default_factory_binds_each_tasks_actual_left_and_right_base_fk(dual_layouts):
    scenes, _ = dual_layouts
    registry = default_registry('operator_release')
    policies = {}
    for task_id, scene in scenes.items():
        descriptor, = registry.descriptors(task_id=task_id)
        policy = registry.resolve(descriptor.policy_id, descriptor.version, task_id=task_id)
        assert policy.robot_configuration == robot_configuration(scene)
        assert policy.descriptor == descriptor
        for side, pose in scene['robot']['arm_base_poses'].items():
            np.testing.assert_allclose(policy.geometry.sides[side].arm_base[:3, 3], pose['position_m'])
        policies[task_id] = policy
    first, second = policies.values()
    assert first.descriptor.policy_id == second.descriptor.policy_id
    assert first.descriptor.version == second.descriptor.version
    assert first.descriptor.implementation_sha256 != second.descriptor.implementation_sha256
    q = {joint.name: (joint.lower+joint.upper)/2 for joint in first.geometry.joints}
    for side in ('left', 'right'):
        before = first.geometry.public_hand_geometries(q)[side]['wrist']['position_m']
        after = second.geometry.public_hand_geometries(q)[side]['wrist']['position_m']
        expected = (np.array(scenes['task_b']['robot']['arm_base_poses'][side]['position_m'])
                    - scenes['task_a']['robot']['arm_base_poses'][side]['position_m'])
        np.testing.assert_allclose(np.array(after)-before, expected, atol=1e-12)
    repeated = registry.resolve(first.descriptor.policy_id, first.descriptor.version, task_id='task_a')
    assert repeated is not first and repeated.sides['left'] is not first.sides['left']
    assert repeated.descriptor == first.descriptor
    with pytest.raises(ValueError, match='Task ID is required'):
        registry.descriptors()


def test_default_factory_keeps_single_layout_discovery_and_constructor_hash(monkeypatch):
    from bench.control.fingertip import ClassicalFingertipPolicy
    scene = {'robot': {'assets_root': f'{DATA_ROOT}/assets',
                       'base_position_m': [0., 0., .77], 'base_quaternion_wxyz': [1., 0., 0., 0.]}}
    manifest = {'tasks': [{'id': 'task_a'}, {'id': 'task_b'}]}
    monkeypatch.setattr(controllers, 'validate', lambda *args, **kwargs: manifest)
    monkeypatch.setattr(controllers, 'task_inputs', lambda root, task: ({'id': task}, scene, {}, {}))
    registry = default_registry('operator_release')
    descriptor, = registry.descriptors()
    assert descriptor == ClassicalFingertipPolicy(**robot_configuration(scene)).descriptor
    assert registry.descriptors(task_id='task_a') == registry.descriptors(task_id='task_b')
    # Changing the caller's source object cannot mutate either captured factory.
    scene['robot']['base_position_m'][0] = 9.
    policy = registry.resolve(descriptor.policy_id, descriptor.version, task_id='task_a')
    assert policy.robot_configuration['base_position_m'] == [0., 0., .77]


def test_http_discovery_requires_task_only_when_configuration_hashes_differ(monkeypatch):
    from bench.isaacsim import release
    registry = PolicyRegistry()
    for task_id, digest in (('task_a', 'c'), ('task_b', 'd')):
        registry.register_task_factory(task_id, _teacher(digest).descriptor,
                                       lambda digest=digest: _teacher(digest))
    monkeypatch.setattr(release, 'validate', lambda *args, **kwargs: {
        'tasks': [{'id': 'task_a'}, {'id': 'task_b'}]})
    manager = SimpleNamespace(registry=registry, release='operator_release', shutdown=lambda: None)
    with TestClient(create_app(service=manager)) as client:
        for endpoint in ('policies', 'interfaces'):
            assert client.get('/'+endpoint).status_code == 422
            assert client.get('/'+endpoint, params={'task_id': 'unknown'}).status_code == 422
            assert client.get('/'+endpoint, params={'task_id': 'task_b'}).status_code == 200
        rows = client.get('/policies', params={'task_id': 'task_b'}).json()
        assert len(rows) == 1 and rows[0]['implementation_sha256'] == 'd'*64


def test_service_selects_verified_task_before_policy_and_keeps_episode_binding(runner, frozen_inputs, monkeypatch):
    make, _, workers = runner
    manifest, original = frozen_inputs
    task_ids = [entry['id'] for entry in manifest['tasks'][:2]]
    inputs = {}
    registry = PolicyRegistry()
    constructed = []
    for index, task_id in enumerate(task_ids):
        payload = deepcopy(original)
        payload[0]['id'] = task_id
        payload[1]['robot']['base_position_m'] = [float(index)*.2, 0., .77]
        inputs[task_id] = payload
        configuration = robot_configuration(payload[1])

        def factory(configuration=deepcopy(configuration), task_id=task_id):
            constructed.append(task_id)
            return _teacher(configuration=configuration)

        registry.register_task_factory(task_id, _teacher().descriptor, factory)
    monkeypatch.setattr(service, 'task_inputs', lambda root, task_id: deepcopy(inputs[task_id]))
    api = make(registry=registry)
    with pytest.raises(ServiceError, match='Unknown task ID'):
        api.create(request(task_id='unknown', track='system0', policy_id='test_teacher', policy_version='v1'))
    assert not constructed and not workers
    sessions = []
    for task_id in reversed(task_ids):
        reply = api.create(request(task_id=task_id, track='system0',
                                   policy_id='test_teacher', policy_version='v1'))
        session = api.get(reply['episode_id'])
        sessions.append(session)
        assert session.policy.robot_configuration == robot_configuration(inputs[task_id][1])
        assert reply['spec']['task_id'] == task_id
        assert reply['spec']['system0'] == session.policy.descriptor.model_dump(mode='json')
        session.finish()
    assert constructed == list(reversed(task_ids))
    assert sessions[0].policy is not sessions[1].policy
    # The service must reject changed scene configuration even if an operator
    # factory itself continues to return its original, correctly hashed policy.
    inputs[task_ids[0]][1]['robot']['base_position_m'][0] += .1
    previous_workers = len(workers)
    with pytest.raises(ServiceError, match='configuration differs'):
        api.create(request(task_id=task_ids[0], track='system0',
                           policy_id='test_teacher', policy_version='v1'))
    assert len(workers) == previous_workers
