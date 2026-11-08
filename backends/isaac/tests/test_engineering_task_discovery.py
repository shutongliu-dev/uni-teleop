"""Engineering opt-outs remain explicit from discovery through Agent startup."""
from copy import deepcopy
import json
from types import SimpleNamespace

from fastapi.testclient import TestClient
import pytest

from agent.model_client import scene_tasks
from bench.protocol import PolicyRegistry
from bench.runtime import controllers
from bench.runtime.catalog import interface_catalog
from bench.runtime.free_track import advertise_free_track
from bench.runtime.server import create_app
from bench.runtime.task_capabilities import engineering_capabilities


ENTRYPOINT = 'scripts.validation.generic_engineering_probe'
PRIVATE = 'private_vertex_control_canary_73981'
CAPABILITIES = {'robot_episodes': False, 'task_inputs': True,
                'engineering_entrypoint': ENTRYPOINT, 'private_control': PRIVATE}


@pytest.fixture
def discovery(tmp_path, monkeypatch):
    tasks = {
        'engineering_fixture': {'id': 'engineering_fixture', 'title_zh': 'Engineering fixture',
            'steps': [{'instruction_zh': 'Public instruction'}], 'capabilities': CAPABILITIES,
            'asset_metadata': {'vertices_m': PRIVATE}, 'evaluation': {'controlled_vertices': PRIVATE}},
        'legacy_fixture': {'id': 'legacy_fixture', 'title_zh': 'Legacy fixture',
            'steps': [{'instruction_zh': 'Public instruction'}]},
    }
    for name, task in tasks.items():
        (tmp_path/(name+'.json')).write_text(json.dumps(task))
    manifest = {'tasks': [{'id': name, 'task': name+'.json'} for name in tasks]}
    from bench.isaacsim import release
    monkeypatch.setattr(release, 'validate', lambda *args, **kwargs: deepcopy(manifest))
    manager = SimpleNamespace(release=tmp_path, registry=PolicyRegistry(), shutdown=lambda: None)
    return manager, tasks


def test_http_discovery_is_explicit_and_drops_private_engineering_evidence(discovery):
    manager, _ = discovery
    with TestClient(create_app(service=manager)) as client:
        tasks = client.get('/tasks').json()
        selected = client.get('/interfaces?task_id=engineering_fixture').json()
        assert client.get('/policies?task_id=engineering_fixture').json() == []
        legacy = client.get('/interfaces?task_id=legacy_fixture').json()
        mixed = client.get('/interfaces').json()
    task, old = tasks
    assert task['task_kind'] == 'engineering' and task['runnable'] is False
    assert task['capabilities'] == {'robot_episodes': False, 'task_inputs': True,
                                    'engineering_entrypoint': ENTRYPOINT}
    assert task['engineering_entrypoint'] == ENTRYPOINT
    assert task['robot_episodes'] is False and task['create_endpoint'] is None
    assert ENTRYPOINT in task['unavailable_reason']
    assert all(not row['available'] and not row['backend_available'] for row in selected['interfaces'])
    assert all(row['client_command'] is None for row in selected['interfaces'])
    assert selected['robot_episodes'] is False and selected['engineering_entrypoint'] == ENTRYPOINT
    assert selected['shared_contract']['action_endpoint'] is None
    assert selected['shared_contract']['function_call_endpoint'] is None
    assert selected['geometry_policies'] == []
    assert PRIVATE not in json.dumps([tasks, selected, mixed])
    # Historical task and interface responses stay byte-equivalent as JSON values.
    assert old == {'task_id': 'legacy_fixture', 'instruction': 'Legacy fixture。Public instruction'}
    assert legacy == advertise_free_track(interface_catalog(), ())
    assert mixed['task_capabilities'] == {'engineering_fixture': task['capabilities']}
    assert mixed['interfaces'] == legacy['interfaces']


def test_default_registry_does_not_construct_a_robot_for_an_opt_out(monkeypatch):
    monkeypatch.setattr(controllers, 'validate', lambda *args, **kwargs:
                        {'tasks': [{'id': 'engineering_fixture'}]})
    monkeypatch.setattr(controllers, 'task_inputs', lambda *args, **kwargs:
                        ({}, {'robot': {'enabled': False}}, {'capabilities': CAPABILITIES}, {}))
    def forbidden(*args, **kwargs):
        raise AssertionError('Robot construction must not precede the explicit task capability')
    monkeypatch.setattr(controllers, 'robot_configuration', forbidden)
    monkeypatch.setattr(controllers, '_configured_policy', forbidden)
    assert controllers.default_registry('fixture').descriptors() == []


@pytest.mark.parametrize('entrypoint', [None, '../'+PRIVATE, 'scripts.validation.probe;'+PRIVATE])
def test_public_opt_out_never_echoes_untrusted_entrypoint(entrypoint):
    task = {'capabilities': {**CAPABILITIES, 'engineering_entrypoint': entrypoint}}
    public = engineering_capabilities(task)
    assert public == {'robot_episodes': False, 'task_inputs': True}
    error = scene_tasks.TaskSetupError('robot_episodes_unavailable', capabilities=public)
    assert PRIVATE not in str(error) and PRIVATE not in json.dumps(error.public_metadata())


def test_selector_rejects_generic_opt_out_and_service_only_opt_out(monkeypatch):
    definition = {'family': 'laboratory', 'task_id': 'engineering_fixture', 'status': 'implemented',
        'task_number': 1, 'capabilities': CAPABILITIES, 'source_release': {'manifest_sha256': 'a'*64},
        'rules': {'contract': {}}, 'robot_requirements': {}}
    monkeypatch.setattr(scene_tasks, 'list_tasks', lambda *args, **kwargs: [deepcopy(definition)])
    selection = scene_tasks.select_task('task1', 'laboratory')
    metadata = selection.metadata()
    assert selection.session_kind == 'engineering' and selection.players == ()
    assert not selection.adapter_available and metadata['runnable'] is False
    assert metadata['practice_rsi_allowed'] is False
    assert metadata['capabilities']['engineering_entrypoint'] == ENTRYPOINT
    assert PRIVATE not in json.dumps(metadata)
    with pytest.raises(scene_tasks.TaskSetupError, match=ENTRYPOINT) as caught:
        selection.require_runnable()
    assert caught.value.code == 'robot_episodes_unavailable'
    assert ENTRYPOINT in caught.value.public_metadata()['message']
    # Unknown-to-catalog task IDs must still honor a connected service opt-out.
    remote = scene_tasks.TaskSelection(family=None, task_id='external_engineering')
    with pytest.raises(scene_tasks.TaskSetupError, match=ENTRYPOINT):
        scene_tasks.published_task(remote, [{'task_id': remote.task_id, 'capabilities': CAPABILITIES}])


@pytest.mark.parametrize('name,family', [('task1', 'laundry_room'), ('towel_flattening', None)])
def test_current_flattening_aliases_use_explicit_robot_release(name, family):
    selection = scene_tasks.select_task(name, family)
    assert selection.metadata()['runnable'] is True
    assert selection.session_kind == 'single_agent'
    selection.require_runnable()


def test_function_runner_rejects_before_http_or_model_setup(tmp_path, monkeypatch):
    from agent.model_client import function_runner
    def forbidden(*args, **kwargs):
        raise AssertionError('Engineering task must fail before HTTP/model startup')
    bench = SimpleNamespace(list_tasks=forbidden, list_policies=forbidden, create=forbidden)
    monkeypatch.setattr(function_runner, 'configured_model', forbidden)
    # Historical engineering releases retain their opt-out after the current
    # scene catalog selects a separately versioned robot installation.
    historical_capabilities = {**CAPABILITIES,
        'engineering_entrypoint': 'scripts.validation.laundry_cloth_interactions'}
    selection = scene_tasks.TaskSelection(family='laundry_room', task_id='towel_flattening',
        session_kind='engineering', capabilities=historical_capabilities)
    monkeypatch.setattr(function_runner, 'select_task', lambda *a, **kw: selection)
    report = function_runner.run(function_runner.RunConfig(bench_url='http://unused',
        task='task1', family='laundry_room', output=tmp_path/'agent'), bench=bench)
    assert report['episode_id'] is None and report['model_requests'] == {}
    assert report['run_status'] == 'configuration_error'
    assert report['task_setup_error']['code'] == 'robot_episodes_unavailable'
    assert 'scripts.validation.laundry_cloth_interactions' in report['task_setup_error']['message']
    assert report['task_selection']['session_kind'] == 'engineering'


def test_legacy_selector_metadata_does_not_gain_capability_fields():
    selection = scene_tasks.select_task('task1', 'kitchen')
    selection.require_runnable()
    assert 'capabilities' not in selection.metadata()
    assert selection.metadata()['runnable'] is True and selection.session_kind == 'single_agent'


@pytest.mark.parametrize('fingertip', [False, True])
def test_legacy_runner_stops_before_controller_discovery_or_episode_creation(tmp_path, monkeypatch, fingertip):
    from agent.model_client import runner
    from test_online_model_runner import setup_loop
    args, bench, model = setup_loop(monkeypatch, tmp_path, [{'stop': True}], fingertip=fingertip)
    bench.list_tasks = lambda: [{'task_id': args.task, 'instruction': 'Public instruction',
                               'capabilities': CAPABILITIES}]
    report = runner.run(args)
    assert bench.discovery_requests == [] and bench.create_requests == []
    assert model.requests == [] and bench.commands == []
    assert report['episode_id'] is None and report['run_status'] == 'configuration_error'
    assert report['run_error']['code'] == 'robot_episodes_unavailable'
    assert ENTRYPOINT in report['run_error']['message']
    assert PRIVATE not in json.dumps(report)
