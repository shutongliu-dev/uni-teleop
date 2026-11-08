"""Configuration admission and model choice use real HTTP and frozen CPU doubles.

Native geometry/IK and physical bimanual execution are separately qualified by
scene-owned checks. These tests never claim a symbolic option is a new layout.
"""
from copy import deepcopy
from dataclasses import replace
import json
import os
from pathlib import Path

from fastapi.testclient import TestClient
import pytest

from agent.model_client.bench_http import BenchHTTPClient, BenchHTTPError
from agent.model_client.configuration_choice import choose_configuration
from agent.model_client.function_runner import RunConfig, run
from agent.model_client.scene_tasks import select_configuration, select_task, TaskSetupError
from agent.model_client.transport import ModelGatewayError
from bench.paths import load_json, DATA_ROOT
from bench.protocol.robot_configurations import checked_public_configurations, project_robot_configurations
from bench.runtime import robot_configurations as bindings
from bench.runtime.server import create_app
from bench.runtime.service import CreateEpisodeRequest
from bench.isaacsim.release import digest, validate, task_inputs
from test_function_runner import ScriptedModel
from test_scene_function_runner import mechanical_service
from test_agent_service import PRIVATE_MARKER


TASK = 'parts_sorting'


def option(reference, identifier='single_arm'):
    dual = identifier != 'single_arm'
    return {'source_release': deepcopy(reference), 'label_zh': identifier,
            'description_zh': '原单右手' if not dual else '单 Agent 官方左右手协作',
            'robot_requirements': {'hand_id': 'dual_fr3_sharpa_w01' if dual else 'sharpa_w01_right',
                                   'assembly_version': 'isaacsim_6_1_v2',
                                   'articulations': 2 if dual else 1,
                                   **({'active_sides': ['left', 'right']} if dual else {})}}


@pytest.fixture
def configured_service(tmp_path, monkeypatch):
    # option() describes the original-mount robot (the dexbench catalog default).
    release = select_task(TASK, family='mechanical_assembly', robot_profile='original').source_release
    api, workers, verified = mechanical_service(tmp_path, TASK, release=release['path'])
    original = select_task(TASK, family='mechanical_assembly', robot_profile='original')
    options = {'single_arm': option(release)}
    public = project_robot_configurations(options)
    api._configuration_bindings = {TASK: {
        'task_id': TASK, 'default_configuration_id': 'single_arm',
        'scene_description_zh': '机械装配室，工作台上的零件分类任务。',
        'robot_configurations': public,
        'releases': {key: row['source_release'] for key, row in options.items()},
    }}
    selected = replace(original, robot_configurations=options)
    monkeypatch.setattr('agent.model_client.function_runner.select_task', lambda *a, **kw: selected)
    return api, workers, selected


def test_public_configuration_projection_exposes_identity_without_paths():
    options = {key: option({'path': '/private/release/'+key,
                           'manifest_sha256': str(n)*64}, key)
               for n, key in enumerate(('single_arm', 'dual_same_side', 'dual_opposed'), 1)}
    public = project_robot_configurations(options)
    assert checked_public_configurations(public) == public
    assert '/private/' not in json.dumps(public)
    assert public['dual_opposed']['robot_requirements']['active_sides'] == ['left', 'right']
    bad = deepcopy(public); bad['dual_opposed']['robot_requirements']['active_sides'] = ['right', 'right']
    with pytest.raises(ValueError): checked_public_configurations(bad)
    bad = deepcopy(public); bad['single_arm']['configuration_id'] = 'dual_opposed'
    with pytest.raises(ValueError): checked_public_configurations(bad)


def test_catalog_selection_pins_all_options_and_rejects_tampered_public_metadata():
    selected = select_task(TASK, family='mechanical_assembly')
    options = {key: option({'path': '/private/'+key, 'manifest_sha256': str(n)*64}, key)
               for n, key in enumerate(('single_arm', 'dual_same_side', 'dual_opposed'), 1)}
    selected = replace(selected, robot_configurations=options)
    published = {'robot_configurations': project_robot_configurations(options)}
    for key in options:
        result = select_configuration(selected, published, key)
        assert result.configuration_id == key
        assert result.source_release == options[key]['source_release']
        assert result.robot_requirements == options[key]['robot_requirements']
        result.verify_release({'configuration_id': key, 'task_release_sha256': str(list(options).index(key)+1)*64})
        with pytest.raises(TaskSetupError): result.verify_release({'task_release_sha256': result.source_release['manifest_sha256']})
    tampered = deepcopy(published)
    tampered['robot_configurations']['dual_same_side']['task_release_sha256'] = 'f'*64
    with pytest.raises(TaskSetupError, match='differ from the pinned catalog'):
        select_configuration(selected, tampered, 'dual_same_side')
    with pytest.raises(TaskSetupError):
        select_configuration(select_task('task1', family='card_room'), published, 'dual_same_side')


def test_actual_http_default_and_selected_contract_keep_private_paths_out(configured_service):
    api, workers, selected = configured_service
    with TestClient(create_app(service=api)) as http:
        client = BenchHTTPClient('http://testserver', client=http)
        default, = client.list_tasks()
        configured, = client.list_tasks(task_id=TASK, configuration_id='single_arm')
        assert configured['configuration_id'] == 'single_arm'
        assert configured['task_release_sha256'] == selected.source_release['manifest_sha256']
        assert configured['public_rules'] == default['public_rules']
        assert client.list_policies(task_id=TASK, configuration_id='single_arm') == client.list_policies(task_id=TASK)
        assert client.list_interfaces(task_id=TASK, configuration_id='single_arm') == client.list_interfaces(task_id=TASK)
        assert DATA_ROOT not in json.dumps(default['robot_configurations'])
        for endpoint in ('tasks', 'policies', 'interfaces'):
            assert http.get('/'+endpoint, params={'configuration_id':'single_arm'}).status_code == 422
            assert http.get('/'+endpoint, params={'task_id':TASK,'configuration_id':'dual_opposed'}).status_code == 422
    assert workers == []


@pytest.mark.parametrize('explicit', [False, True])
def test_agent_selects_before_create_and_binds_actual_selected_public_contract(tmp_path, configured_service, explicit):
    api, workers, selected = configured_service
    decisions = ([{'configuration_id':'single_arm','reason':'使用原手完成零件分类'}] if not explicit else [])
    decisions += [{'function':'get_task','arguments':{}},{'stop':True}]
    model = ScriptedModel(decisions)
    with TestClient(create_app(service=api)) as http:
        report = run(RunConfig('http://testserver', TASK, tmp_path/'run', family='mechanical_assembly',
                              configuration='single_arm' if explicit else 'auto', max_decisions=3),
                     bench=BenchHTTPClient('http://testserver', client=http), models={'single':model})
    assert report['error_type'] is None, report
    assert report['pre_episode_model_calls'] == (0 if explicit else 1)
    assert report['decisions']['single'] == len(decisions)
    request = load_json(tmp_path/'run/create_request.json')
    ready = load_json(tmp_path/'run/episode_ready.json')
    assert request['configuration_id'] == ready['spec']['configuration_id'] == 'single_arm'
    assert ready['spec']['task_release_sha256'] == selected.source_release['manifest_sha256']
    assert ready['spec']['robot_id'] == 'sharpa_w01_right'
    assert len(workers) == 1 and workers[0].closed
    assert not any(operation == 'step' for operation, _ in workers[0].calls)
    assert PRIVATE_MARKER not in json.dumps(model.turns)
    if not explicit:
        context = json.loads(model.turns[0][0]['content'][0]['text'])
        assert context['scene_description_zh'] and context['robot_configurations']
        assert 'spec' not in context  # The model chooses before native creation.
        assert report['configuration_choice']['metadata']['attempts'] == 1


@pytest.mark.parametrize('decision', [
    {'configuration_id':'dual_opposed','reason':'not offered'},
    {'configuration_id':'single_arm'},
    {'configuration_id':'single_arm','reason':'ok','path':'/private/release'},
    'not JSON',
])
def test_invalid_model_choice_is_recorded_and_never_falls_back_or_starts_worker(tmp_path, configured_service, decision):
    api, workers, _ = configured_service
    model = ScriptedModel([decision])
    with TestClient(create_app(service=api)) as http:
        report = run(RunConfig('http://testserver', TASK, tmp_path/'run', family='mechanical_assembly'),
                     bench=BenchHTTPClient('http://testserver', client=http), models={'single':model})
    assert report['task_setup_error']['code'] == 'configuration_choice_invalid'
    assert report['pre_episode_model_calls'] == 1
    assert report['episode_id'] is None and workers == [] and api.sessions == {}
    assert (tmp_path/'run/configuration_choice.json').is_file()
    assert not (tmp_path/'run/create_request.json').exists()


def test_explicit_unavailable_configuration_rejected_before_worker(configured_service):
    api, workers, _ = configured_service
    request = CreateEpisodeRequest(task_id=TASK, configuration_id='dual_opposed',
        agent_id='test', agent_version='1', agent_config_sha256='0'*64)
    with TestClient(create_app(service=api)) as http:
        response = http.post('/episodes', json=request.model_dump(mode='json'))
        assert response.status_code == 422
    assert workers == []


def test_model_request_failure_preserves_choice_request_and_safe_attempt_metadata(tmp_path, configured_service):
    api, workers, _ = configured_service
    model = ScriptedModel([ModelGatewayError('private-provider-body', category='transport_timeout', attempts=2)])
    with TestClient(create_app(service=api)) as http:
        report = run(RunConfig('http://testserver', TASK, tmp_path/'run', family='mechanical_assembly'),
                     bench=BenchHTTPClient('http://testserver', client=http), models={'single':model})
    receipt = load_json(tmp_path/'run/configuration_choice.json')
    assert receipt['status'] == 'request_failed' and receipt['messages']
    assert receipt['metadata']['attempts'] == 2
    assert report['pre_episode_model_requests'] == [receipt['metadata']]
    assert report['pre_episode_model_calls'] == 1 and report['episode_id'] is None
    assert report['configuration_choice'] == receipt
    assert 'private-provider-body' not in json.dumps(report)
    assert not workers and not (tmp_path/'run/create_request.json').exists()


def test_current_catalog_pins_are_required_for_discovery(tmp_path, monkeypatch):
    root = tmp_path/'release'; root.mkdir(); (root/'release_manifest.json').write_text('{}')
    digest = bindings.digest(root/'release_manifest.json')
    definition = {'task_id':TASK, 'title':'零件分类', 'source_release':{'path':str(root),'manifest_sha256':digest},
                  'default_configuration_id':'single_arm'}
    definition['robot_configurations']={'single_arm':option(definition['source_release'])}
    monkeypatch.setattr('scenes.workrooms.catalog.list_tasks', lambda **kw:[definition])
    monkeypatch.setattr(bindings, 'validate', lambda *a, **kw:{'tasks':[{'id':TASK}]})
    assert TASK in bindings.discover_configuration_bindings(root)
    definition['robot_configurations']['single_arm']['source_release']['manifest_sha256']='f'*64
    assert bindings.discover_configuration_bindings(root)=={}


def test_configuration_query_encoder_requires_task_identity():
    with pytest.raises(BenchHTTPError):
        BenchHTTPClient('http://localhost').list_policies(configuration_id='dual_opposed')


def test_selected_task_projection_replaces_default_rules_before_creation(tmp_path, configured_service):
    api, workers, _ = configured_service
    model = ScriptedModel([{'configuration_id':'single_arm','reason':'选定已发布配置'}, {'stop':True}])
    with TestClient(create_app(service=api)) as http:
        class DifferentDefaultProjection(BenchHTTPClient):
            def list_tasks(self, task_id=None, configuration_id=None, *, seed=None):
                rows = super().list_tasks(task_id, configuration_id, seed=seed)
                if configuration_id is None:
                    # A different default projection must not be reused after
                    # a configuration-specific /tasks request has completed.
                    rows[0]['instruction'] = 'Different default installation instruction'
                    rows[0]['public_rules_sha256'] = 'f'*64
                return rows
        client = DifferentDefaultProjection('http://testserver', client=http)
        report = run(RunConfig('http://testserver', TASK, tmp_path/'run', family='mechanical_assembly', max_decisions=2),
                     bench=client, models={'single':model})
    assert report['error_type'] is None, report
    assert report['ready_validated'] and len(workers)==1
    ready=load_json(tmp_path/'run/episode_ready.json')
    assert ready['task']['public_rules_sha256'] != 'f'*64


def test_changed_release_pin_is_rejected_without_native_start(configured_service):
    api, workers, _ = configured_service
    api._configuration_bindings[TASK]['releases']['single_arm']['manifest_sha256']='f'*64
    request = CreateEpisodeRequest(task_id=TASK, configuration_id='single_arm',
        agent_id='test', agent_version='1', agent_config_sha256='0'*64)
    with TestClient(create_app(service=api)) as http:
        assert http.post('/episodes', json=request.model_dump(mode='json')).status_code==422
    assert not workers and not api.sessions


def test_smoke_precreation_uses_actual_agent_configuration_selection(configured_service, monkeypatch):
    from scripts.validation.public_http_smoke import selected_agent_task
    api, workers, selected = configured_service
    monkeypatch.setattr('agent.model_client.scene_tasks.select_task', lambda *a, **kw: selected)
    with TestClient(create_app(service=api)) as http:
        selection, published = selected_agent_task(BenchHTTPClient('http://testserver', client=http),
            family='mechanical_assembly', task_id=TASK, configuration_id='single_arm')
    assert selection.configuration_id == published['configuration_id'] == 'single_arm'
    assert selection.source_release['manifest_sha256'] == published['task_release_sha256']
    assert not workers and not api.sessions


@pytest.mark.parametrize('requested,manifest', [
    ('dual_opposed', {'schema':'workroom_robot_configuration_release.v1','configuration_id':'dual_same_side'}),
    ('single_arm', {'schema':'workroom_robot_configuration_release.v1','configuration_id':'dual_same_side'}),
    ('dual_same_side', {'schema':'original_single_release.v1'}),
])
def test_configuration_selector_cannot_rename_a_valid_release(tmp_path, monkeypatch, requested, manifest):
    root = tmp_path/'release'; root.mkdir(); (root/'release_manifest.json').write_text('{}')
    reference = {'path':str(root), 'manifest_sha256':bindings.digest(root/'release_manifest.json')}
    binding = {'task_id':TASK, 'releases':{requested:reference}}
    monkeypatch.setattr(bindings, 'validate', lambda *a, **kw:manifest)
    with pytest.raises(ValueError, match='another robot installation'):
        bindings.checked_configuration_release(binding, requested)


def test_public_layout_preserves_static_bases_and_handedness_without_private_fields():
    config = option({'path':'/private/release', 'manifest_sha256':'a'*64}, 'dual_same_side')
    config['layout'] = {'robot_layout_id':'mechanical_same_side_v1', 'reference_surface':'装配工作台前沿',
        'frame':'world', 'units':'metres', 'control_scope':'one_agent_58_dof',
        'private_reset_path':'/private/reset',
        'placements': {side: {'base_position_m':[x,-.65,.72], 'base_quaternion_wxyz':[1.,0.,0.,0.],
            'base_yaw_rad':0., 'hand_chirality':side, 'private_asset_path':'/private/robot'}
            for side,x in [('left',-.35),('right',.35)]}}
    public = project_robot_configurations({'dual_same_side':config})
    model = ScriptedModel([{'configuration_id':'dual_same_side','reason':'两手同侧覆盖台前零件'}])
    chosen, receipt = choose_configuration(model, {'robot_configurations':public}, max_output_tokens=256, timeout_s=5.)
    context = json.loads(receipt['messages'][0]['content'][0]['text'])
    layout = context['robot_configurations']['dual_same_side']['layout']
    assert chosen == 'dual_same_side' and layout['frame'] == 'world'
    assert layout['placements']['left']['base_position_m'] == [-.35,-.65,.72]
    assert layout['placements']['right']['hand_chirality'] == 'right'
    assert '/private/' not in json.dumps(context)
    bad = deepcopy(public); bad['dual_same_side']['layout']['placements']['left']['hand_chirality']='right'
    with pytest.raises(ValueError): checked_public_configurations(bad)
    bad = deepcopy(public); bad['dual_same_side']['layout']['placements'].pop('left')
    with pytest.raises(ValueError): checked_public_configurations(bad)


@pytest.mark.parametrize('robot_profile', ['original', 'l_mount'])
@pytest.mark.parametrize('configuration_id', ['dual_same_side', 'dual_opposed'])
def test_real_dual_release_model_choice_pins_robot_rules_and_controller(tmp_path, monkeypatch, configuration_id, robot_profile):
    """Use a published wrapper or an explicitly supplied frozen probe; no GPU."""
    from scenes.workrooms.catalog import task_definition
    from scenes.workrooms.parameterized.pairs import PROFILES
    definition = task_definition('mechanical_assembly', TASK, robot_profile=robot_profile)
    probe = os.environ.get('BENCH_TEST_ROBOT_CONFIGURATION_RELEASE')
    if probe is not None and robot_profile != 'original':
        pytest.skip('An explicit configuration probe is an original-mount release')
    path = probe or definition.get('robot_configurations', {}).get(configuration_id, {}).get('source_release', {}).get('path')
    if path is None:
        pytest.skip('No published dual release; set BENCH_TEST_ROBOT_CONFIGURATION_RELEASE to a frozen probe')
    release = Path(path)
    manifest = validate(release, assets=False)
    # Published options wrap one frozen configuration release in its frozen
    # seed cases and, for the L-mount profile, the robot-only revision.
    wrappers = []
    while manifest['schema'] != 'workroom_robot_configuration_release.v1':
        wrappers.append(manifest['schema'])
        manifest = load_json(Path(manifest['base_release']['path'])/'release_manifest.json')
    assert wrappers == ([] if probe else {
        'original': ['workroom_randomized_release.v1'],
        'l_mount': ['workroom_robot_revision_release.v1', 'workroom_randomized_release.v1']}[robot_profile])
    _, scene, _, _ = task_inputs(release, TASK)
    identifier = scene['robot_configuration']['id']
    assert identifier in ('dual_same_side', 'dual_opposed')
    assert scene['robot']['assembly_version'] == PROFILES[robot_profile]
    if identifier != configuration_id:
        pytest.skip('The explicit probe covers only '+identifier)
    api, workers, _ = mechanical_service(tmp_path, TASK, release=definition['source_release']['path'])
    options = deepcopy(definition.get('robot_configurations')) if not probe else None
    options = options or {'single_arm': option(definition['source_release']),
               identifier: option({'path':str(release), 'manifest_sha256':digest(release/'release_manifest.json')}, identifier)}
    selected = replace(select_task(TASK, family='mechanical_assembly', robot_profile=robot_profile),
                       robot_configurations=options)
    api._configuration_bindings = {TASK: {'task_id':TASK, 'default_configuration_id':'single_arm',
        'scene_description_zh':'装配台，单 Agent 官方左右手', 'robot_configurations':project_robot_configurations(options),
        'releases':{key: row['source_release'] for key,row in options.items()}}}
    monkeypatch.setattr('agent.model_client.function_runner.select_task', lambda *a, **kw:selected)
    monkeypatch.setattr('agent.model_client.scene_tasks.select_task', lambda *a, **kw:selected)
    model = ScriptedModel([{'configuration_id':identifier,'reason':'使用两手覆盖装配台'},
                           {'function':'get_robot','arguments':{}}, {'stop':True}])
    with TestClient(create_app(service=api)) as http:
        report = run(RunConfig('http://testserver', TASK, tmp_path/'run', family='mechanical_assembly', max_decisions=3),
                     bench=BenchHTTPClient('http://testserver', client=http), models={'single':model})
    assert report['error_type'] is None, report
    assert report['ready_validated'] and report['decisions']['single'] == 3
    ready = load_json(tmp_path/'run/episode_ready.json')
    assert ready['spec']['configuration_id'] == identifier
    assert ready['spec']['robot_id'] == 'dual_fr3_sharpa_w01'
    assert ready['spec']['task_release_sha256'] == options[identifier]['source_release']['manifest_sha256']
    assert ready['spec']['system0']['policy_id'] == 'classical_fingertip_bimanual'
    assert len(ready['capabilities']['arm_joints']) == 14
    assert len(ready['capabilities']['hand_joints']) == 44
    assert len(workers) == 1 and workers[0].closed
    assert all(operation != 'step' for operation, _ in workers[0].calls)
