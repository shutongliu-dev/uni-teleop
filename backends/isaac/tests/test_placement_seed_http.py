"""Real published placement releases through Agent/HTTP, CPU doubles only.

Run after all final catalog pins are installed. Missing/wrong releases FAIL;
these tests never skip an unpublished installation. Optional environment:
PLACEMENT_CATALOG_PINS selects an explicit final catalog-pins JSON (read only),
PLACEMENT_HTTP_EVIDENCE persists each case's HTTP/Agent records in a fresh tree.
Use pytest from the desired frozen/current source; this file does not change
sys.path or replace production release validation. No simulator or model API.
"""
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path

from fastapi.testclient import TestClient
import pytest

from agent.model_client.bench_http import BenchHTTPClient
from agent.model_client.function_runner import RunConfig, run
from agent.model_client.scene_tasks import select_task
from bench.control.factory import public_robot_geometry, robot_configuration
from bench.isaacsim.release import digest, task_inputs
from bench.paths import load_json
from bench.runtime.controllers import default_registry
from bench.runtime.server import create_app
from bench.runtime.service import AgentService
from bench.runtime.worker import public_task_projection
from bench.simulation.isaacsim_backend.robot_registry import get_robot_spec
from scenes.workrooms.catalog import task_definition
from scenes.workrooms.initial_randomization import PLACEMENT_TASK_FAMILIES, configurations
from scenes.workrooms.initial_randomization import catalog as seeded_catalog
from test_agent_service import FakeWorker, PRIVATE_MARKER, publish_targets
from test_function_runner import ScriptedModel
from test_scene_function_runner import reset_score

INSTALLATIONS = tuple((family, task, configuration)
    for task, family in PLACEMENT_TASK_FAMILIES.items() for configuration in configurations(task))
CASES = (*[(family,task,configuration,5) for family,task,configuration in INSTALLATIONS],
         ('mechanical_assembly','cooperative_insertion','dual_same_side',None))


def _json(path):
    return load_json(Path(path))


def _write(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n')


@pytest.fixture
def placement_catalog(monkeypatch):
    override = os.environ.get('PLACEMENT_CATALOG_PINS')
    if override:
        path = Path(override)
        assert path.is_absolute() and path.is_file(), 'Use an existing absolute final catalog-pins JSON'
        monkeypatch.setattr(seeded_catalog, 'PINS', path)
    return seeded_catalog.PINS


def placement_service(directory, release, task_id):
    """Production task/evaluator/robot contracts; only the worker is synthetic."""
    workers = []
    def factory(**kwargs):
        assert kwargs['task_id'] == task_id
        _, scene, task, initial = task_inputs(kwargs['release'], task_id, seed=kwargs['seed'])
        geometry = public_robot_geometry(robot_configuration(scene))
        models = list(geometry.sides.values()) if hasattr(geometry,'sides') else [geometry]
        arm = [name for model in models for name in model.arm.names]
        hand = [name for model in models for name in model.hand.names]
        class PlacementWorker(FakeWorker):
            hand_counts = {**FakeWorker.hand_counts, kwargs['hand_id']:len(hand)}
        worker = PlacementWorker(**kwargs, depth=True, acknowledged_targets=True, on_step=publish_targets)
        worker.arm,worker.hand,worker.active = arm,hand,list(geometry.names)
        worker.raw['joint_positions_rad'] = deepcopy(initial['joint_positions_rad'])
        worker.raw['joint_velocities_rad_s'] = dict.fromkeys(worker.active,0.)
        worker.raw['robot'] = {'joint_targets_rad':deepcopy(initial['joint_positions_rad'])}
        worker.raw['task_state'] = {'private_native_geometry':PRIVATE_MARKER}
        worker.raw['native_task_bundle'] = {'private_contacts':PRIVATE_MARKER}
        worker.score.update(reset_score(kwargs['release'],task))
        worker.score['private_inventory'] = PRIVATE_MARKER
        worker.initial.update(observation=deepcopy(worker.raw),score=deepcopy(worker.score),
                              task=public_task_projection(task))
        worker.initial['metadata'].update(scene_spec=scene,hand_spec=get_robot_spec(scene['robot']),action_spec={
            'joint_names':worker.active,'arm_joint_names':arm,'hand_joint_names':hand,
            'joint_limits_rad':{joint.name:[float(joint.lower),float(joint.upper)] for joint in geometry.joints}})
        worker.selected_inputs = (deepcopy(scene),deepcopy(task),deepcopy(initial))
        workers.append(worker)
        return worker
    api = AgentService(directory/'episodes',release=release,registry=default_registry(release),worker_factory=factory)
    return api,workers


def _assert_public(value):
    text = json.dumps(value)
    assert PRIVATE_MARKER not in text
    forbidden = {'actual_body_poses_m','initial_pose','interaction_targets','fixed_support_paths',
                 'asset_reference','private_native_geometry','private_inventory','private_contacts'}
    def walk(obj):
        if isinstance(obj,dict):
            assert not forbidden.intersection(obj)
            for child in obj.values(): walk(child)
        elif isinstance(obj,list):
            for child in obj: walk(child)
    walk(value)


def test_exact_new_task_and_installation_scope():
    assert len(INSTALLATIONS)==49 and len(set(INSTALLATIONS))==49
    assert sum(family=='study' for family,_,_ in INSTALLATIONS)==9
    assert sum(family=='mechanical_assembly' for family,_,_ in INSTALLATIONS)==25
    assert sum(family=='kitchen' for family,_,_ in INSTALLATIONS)==3
    assert sum(family=='laboratory' for family,_,_ in INSTALLATIONS)==12
    assert [c for _,task,c in INSTALLATIONS if task=='cooperative_insertion']==['dual_same_side']
    assert not {'parts_sorting','stationery_sorting','book_stacking','book_shelving'}.intersection(
        task for _,task,_ in INSTALLATIONS)


@pytest.mark.parametrize('robot_profile',('original','l_mount'))
@pytest.mark.parametrize('family,task_id,configuration,seed',CASES,
    ids=[f'{family}/{task}/{configuration}/'+('default0' if seed is None else f'seed{seed}')
         for family,task,configuration,seed in CASES])
def test_published_placement_agent_http_lifecycle(tmp_path,placement_catalog,
                                                family,task_id,configuration,seed,robot_profile):
    fixed = task_id=='cooperative_insertion' or family=='kitchen'
    effective = 0 if seed is None else seed
    original_selection = select_task(task_id,family,robot_profile=robot_profile)
    assert original_selection.configuration_id is None
    definition = task_definition(family,task_id,None if fixed else configuration,robot_profile=robot_profile)
    release = Path(definition['source_release']['path'])
    assert release.is_absolute() and (release/'release_manifest.json').is_file(), 'Final release is missing; not a passing skip'
    manifest = _json(release/'release_manifest.json')
    placement_reference = definition['source_release']
    if robot_profile=='l_mount':
        # The L-mount task is a robot-only revision of the paired original placement wrapper.
        placement_reference = task_definition(family,task_id,None if fixed else configuration,
                                              robot_profile='original')['source_release']
        assert manifest['schema']=='workroom_robot_revision_release.v1'
        assert manifest['base_release']=={**placement_reference,'schema':'workroom_randomized_release.v1'}
    placement = Path(placement_reference['path']); placement_manifest = _json(placement/'release_manifest.json')
    assert placement_manifest['schema']=='workroom_randomized_release.v1', 'Final placement wrapper has not been published'
    for value in (manifest,placement_manifest):
        assert value['configuration_id']==configuration and value['task_id']==task_id
    assert digest(release/'release_manifest.json')==definition['source_release']['manifest_sha256']
    catalog_before = digest(placement_catalog)
    _,scene,task,initial = task_inputs(release,task_id,seed=effective)
    placed_scene,placed_task,placed_initial = ((scene,task,initial) if placement==release
                                               else task_inputs(placement,task_id,seed=effective)[1:])
    parent_root = Path(placement_manifest['base_release']['path'])
    assert digest(parent_root/'release_manifest.json')==placement_manifest['base_release']['manifest_sha256']
    parent_manifest = _json(parent_root/'release_manifest.json'); parent_entry = parent_manifest['tasks'][0]
    parent_scene,parent_initial = (_json(parent_root/parent_entry[name]) for name in ('scene','initial_state'))
    assert placed_scene['robot']==parent_scene['robot'] and placed_initial==parent_initial
    assert (placement==release)==(robot_profile=='original')
    if robot_profile=='l_mount':
        # Same seed, same physical world: only the robot, its ready reset and
        # the derived case identity differ from the paired original case.
        revision = _json(release/'robot_revision.json')
        assert scene['robot']==revision['robot'] and scene['robot']!=placed_scene['robot']
        assert initial['joint_positions_rad']==revision['joint_positions_rad']
        assert {k:v for k,v in initial.items() if k!='joint_positions_rad'}=={
            k:v for k,v in placed_initial.items() if k!='joint_positions_rad'}
        assert task['reset_case']['robot_revision']['parent_case_sha256']==placed_task['reset_case']['case_sha256']
        assert {k for k in set(task)|set(placed_task) if task.get(k)!=placed_task.get(k)}<={'initial_state','reset_case'}
        for key in set(scene)|set(placed_scene):
            if key not in ('robot','workroom_reset_case','native_task_bundle','task_layers'):
                assert scene.get(key)==placed_scene.get(key),key
    geometry = public_robot_geometry(robot_configuration(scene))
    expected_sides = set(geometry.sides) if hasattr(geometry,'sides') else {'right'}
    assert expected_sides==({'right'} if configuration=='single_arm' else {'left','right'})
    assert scene.get('robot_configuration')==parent_scene.get('robot_configuration')
    assert scene.get('workroom')==parent_scene.get('workroom')
    assert task['reset_case']['configuration_id']==configuration
    destination = os.environ.get('PLACEMENT_HTTP_EVIDENCE')
    directory = (Path(destination)/robot_profile/family/task_id/configuration/('default0' if seed is None else f'seed{seed}')
                 if destination else tmp_path/'case')
    assert not directory.exists(), 'Evidence directory must be fresh; preserve earlier failures'
    directory.mkdir(parents=True)
    _write(directory/'input_identity.json',{
        'schema':'placement_http_cpu_inputs.v1','family':family,'task_id':task_id,'configuration_id':configuration,
        'robot_profile':robot_profile,'requested_seed':seed,'effective_seed':effective,'catalog_pins':str(placement_catalog),
        'catalog_pins_sha256':catalog_before,'release':str(release),
        'release_manifest_sha256':digest(release/'release_manifest.json'),
        'parent_release':placement_manifest['base_release'],'case_sha256':task['reset_case']['case_sha256'],
        'case_fingerprint':task['reset_case']['case_fingerprint'],
        'scope':'Real published contracts; synthetic CPU worker/model; no native or manipulation qualification'})
    api,workers = placement_service(directory,release,task_id)
    decisions = [{'function':'get_task','arguments':{}},
                 {'function':'hold','arguments':{'duration_s':.04}},
                 {'function':'get_task','arguments':{}},{'stop':True}]
    if not fixed:
        decisions.insert(0,{'configuration_id':configuration,'reason':'CPU fixture selects this advertised original installation.'})
    class BeforeEpisodeChoice(ScriptedModel):
        def generate(self,prompt,turns,**kwargs):
            if not self.turns and not fixed: assert not workers and not api.sessions
            return super().generate(prompt,turns,**kwargs)
    model = BeforeEpisodeChoice(decisions)
    with TestClient(create_app(service=api)) as http:
        before = http.get('/tasks',params={'task_id':task_id})
        assert before.status_code==200,before.text
        public = before.json()[0]
        if fixed:
            assert not public.get('robot_configurations') and not definition.get('robot_configurations')
            assert public['configuration_id']==configuration
            assert public['reset_case']['configuration_id']==configuration
            for unsupported in ('single_arm','dual_opposed','dual_same_side'):
                assert http.get('/tasks',params={'task_id':task_id,'configuration_id':unsupported}).status_code==422
        else:
            assert set(public['robot_configurations'])=={'single_arm','dual_same_side','dual_opposed'}
        for invalid in ('10','-1','true','1.0','01'):
            response=http.get('/tasks',params={'task_id':task_id,'seed':invalid})
            assert response.status_code==422
        assert not workers and not api.sessions
        bench=BenchHTTPClient('http://testserver',client=http)
        report=run(RunConfig('http://testserver',task_id,directory/'run',family=family,seed=seed,
            robot_profile=robot_profile,configuration='auto',max_decisions=7,max_status_queries=100000,sim_budget=None),
            bench=bench,models={'single':model})
        assert report['error_type'] is None and report['task_setup_error'] is None,report
        assert report['run_status']=='completed' and report['termination']=='model_requested_stop'
        assert not report['board_game'] and not report['turn_game']
        ready=_json(directory/'run/episode_ready.json'); request=_json(directory/'run/create_request.json')
        discovery=_json(directory/'run/episode_case_discovery.json'); expected=discovery['reset_case']
        assert expected==ready['task']['reset_case']
        assert expected['seed']==effective and expected['configuration_id']==configuration
        assert expected['case_sha256']==task['reset_case']['case_sha256']
        assert 'case_fingerprint' not in expected  # Private case-content binding is not public geometry.
        assert request['seed']==ready['spec']['seed']==effective
        assert request['robot_id']==ready['spec']['robot_id']==scene['robot']['hand_id']
        assert ready['spec']['task_release_sha256']==definition['source_release']['manifest_sha256']
        if fixed:
            assert request.get('configuration_id') is None and ready['spec'].get('configuration_id') is None
            assert report['task_selection'].get('configuration_id') is None
            assert report['pre_episode_model_calls']==0 and report['configuration_choice'] is None
        else:
            assert request['configuration_id']==ready['spec']['configuration_id']==configuration
            assert report['task_selection']['configuration_id']==configuration
            assert report['pre_episode_model_calls']==1
            assert report['configuration_choice']['selected']['configuration_id']==configuration
        response=http.get(f"/episodes/{report['episode_id']}/task")
        assert response.status_code==200 and response.json()['reset_case']==expected
        assert response.json()['episode']['terminal'] is True
        for invalid in (10,-1,True,1.0):
            rejected=http.post('/episodes',json={**request,'seed':invalid})
            assert rejected.status_code==422,rejected.text
        assert len(workers)==1  # Rejected seeds never create another worker.
        _assert_public([before.json(),ready,discovery,response.json(),model.turns])
    assert len(workers)==1 and workers[0].closed
    assert workers[0].selected_inputs==(scene,task,initial)
    assert set(workers[0].active)==set(initial['joint_positions_rad'])
    assert set(ready['observation']['proprioception']['joint_positions_rad'])==set(initial['joint_positions_rad'])
    assert report['physical_attempts']=={'single':1}
    assert sum(operation=='step' for operation,_ in workers[0].calls)==2
    assert not any(operation in ('board_game','turn_game') for operation,_ in workers[0].calls)
    assert report['result']['outcome']=='cancelled' and report['result']['elapsed_sim_s']==pytest.approx(.04)
    assert report['result']['action_attempts']==1 and report['result']['spec']==ready['spec']
    contexts=[json.loads(turns[-1]['content'][0]['text']) for turns in model.turns[(0 if fixed else 1):]]
    assert len(contexts)==4 and all(context['task']['reset_case']==expected for context in contexts)
    for index in (1,3):
        assert contexts[index]['last_feedback']['function']=='get_task'
        assert contexts[index]['last_feedback']['ok'] is True
        assert contexts[index]['last_feedback']['result']['reset_case']==expected
    assert contexts[2]['last_feedback']['function']=='hold' and contexts[2]['last_feedback']['ok'] is True
    assert contexts[2]['last_feedback']['result']['status']=='completed'
    assert PRIVATE_MARKER in json.dumps(workers[0].raw)
    assert digest(placement_catalog)==catalog_before
    assert digest(release/'release_manifest.json')==definition['source_release']['manifest_sha256']
    _write(directory/'cpu_review.json',{'schema':'placement_http_cpu_review.v1','passed':True,
        'family':family,'task_id':task_id,'configuration_id':configuration,'robot_profile':robot_profile,'seed':effective,
        'default_seed':seed is None,'case_sha256':expected['case_sha256'],'case_fingerprint':task['reset_case']['case_fingerprint'],
        'release_manifest_sha256':ready['spec']['task_release_sha256'],
        'worker_count':len(workers),'worker_closed':True,'synthetic_steps':2,'synthetic_elapsed_sim_s':.04,
        'scripted_model_calls':len(model.turns),'pre_episode_configuration_model_calls':int(not fixed),
        'active_sides':sorted(expected_sides),'agent_result_outcome':report['result']['outcome'],
        'artifact_sha256':{str(p.relative_to(directory)):hashlib.sha256(p.read_bytes()).hexdigest()
                           for p in (directory/'input_identity.json',directory/'run/run_report.json',
                                     directory/'run/episode_ready.json',directory/'run/episode_case_discovery.json')},
        'native_qualified':False,'manipulation_qualified':False,'external_model_calls':0})
