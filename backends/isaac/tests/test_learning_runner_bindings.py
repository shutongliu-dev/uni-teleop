"""Real HTTP learner routing with CPU doubles, not model/native qualification."""
from copy import deepcopy
import json
from pathlib import Path

from fastapi.testclient import TestClient
import pytest

from agent.model_client.bench_http import BenchHTTPClient,BenchHTTPError
from agent.model_client.function_http import FunctionHTTPClient
from agent.model_client.function_runner import RunConfig,run
from agent.model_client.learning.bindings import SkillBinding,load_skill_bindings
from agent.model_client.learning.journal import canonical_json
from agent.model_client.learning.skills import SkillLibrary
from agent.model_client.runner import canonical_hash
from bench.paths import load_json
from bench.runtime.controllers import default_registry
from bench.runtime.server import create_app
from bench.runtime.service import AgentService
from scenes.workrooms.catalog import task_definition
from test_board_game_service import FakeBoardWorker
from test_function_runner import ScriptedModel


SOURCE='''def run(api,args,state):
    for i in range(args["count"]):
        reply = api.call("hold",{"duration_s":0.02})
    return reply
'''


def candidate(library,source,transaction='version1'):
    revision=library.write_author('skills/example.py',source,expected_revision=library.author_revision())
    revision=library.write_author('catalog.json',canonical_json({'skills':[{
        'skill_id':'example','module':'skills/example.py','entrypoint':'run',
        'description':'CPU fixture for learner-written source routing','tags':[],
        'input_schema':{'type':'object'},'output_schema':{}}]}),expected_revision=revision)
    event=library.publish(revision,transaction_id=transaction,reason='CPU model-authored source fixture')
    return {'candidate_event_id':event['event_id'],'bundle_sha256':event['payload']['bundle_hash']}


def bindings(tmp_path,*,phase='practice',source=SOURCE):
    definition=task_definition('card_room','tic_tac_toe_3x3')
    libraries={key:SkillLibrary.create(key,learning_root=tmp_path/'libraries',contract={
        'dependencies':{},'compatibility':{'task_ids':['tic_tac_toe_3x3'],
        'task_releases':{'tic_tac_toe_3x3':definition['source_release']['manifest_sha256']}}})
        for key in ('player_0','player_1')}
    versions={key:candidate(library,source) for key,library in libraries.items()}
    bound={key:SkillBinding(library,phase,**(versions[key] if phase!='practice' else {}))
           for key,library in libraries.items()}
    return bound,versions


def run_http(tmp_path,models,bound,**config):
    release=Path(task_definition('card_room','tic_tac_toe_3x3')['source_release']['path'])
    workers=[]
    def factory(**kwargs):
        worker=FakeBoardWorker(**kwargs);workers.append(worker);return worker
    service=AgentService(tmp_path/'episodes',release=release,registry=default_registry(release),worker_factory=factory)
    output=tmp_path/'run'
    with TestClient(create_app(service=service)) as http:
        report=run(RunConfig('http://testserver','task1',output,family='card_room',
            max_decisions=6,**config),bench=BenchHTTPClient('http://testserver',client=http),
            models=models,learning_bindings=bound)
    return report,output,workers


@pytest.mark.parametrize('budget', ['calls','actions'])
def test_board_intent_tools_and_actual_nested_skill_calls_obey_host_budgets(tmp_path,budget):
    bound,versions=bindings(tmp_path)
    decision={'function':'skill_execute','arguments':{'skill_id':'example',
        'candidate_event_id':versions['player_0']['candidate_event_id'],'args':{'count':3},'state':{}}}
    models={'player_0':ScriptedModel([decision,{'stop':True}]),'player_1':ScriptedModel([])}
    options={'max_function_calls':3,'max_actions':10} if budget=='calls' else {'max_function_calls':10,'max_actions':2}
    report,output,workers=run_http(tmp_path,models,bound,**options)
    assert report['error_type'] is None and report['finalization_error'] is None,report
    assert report['function_calls']['player_0']==(3 if budget=='calls' else 4)
    assert report['physical_attempts']['player_0']==2
    assert report['physical_attempts']['player_1']==0 and not models['player_1'].turns
    context=json.loads(models['player_0'].turns[0][-1]['content'][0]['text'])
    assert any('intent' in tool for tool in context['tools'])  # Intent is not a function-keyed entry.
    assert {'skill_write','skill_read','skill_publish','skill_execute'} <= {tool.get('function') for tool in context['tools']}
    assert 'def run(api, args, state)' in context['skill_runtime']['language_spec']['entrypoint']
    assert 'api.call("observe", {})' in context['skill_runtime']['language_spec']['example']
    execution=load_json(output/'player_0/decisions/0000/execution.json')
    assert execution['ok'] and execution['result']['status']=='completed'
    calls=execution['result']['calls']
    assert len(calls)==3 and calls[-1]['error']['code']=='budget_exhausted'
    assert [row['primitive_status'] for row in calls[:2]]==['completed','completed']
    assert len(list((output/'player_0/calls').glob('*.json')))==report['function_calls']['player_0']
    settings=load_json(output/'agent_config.json')
    request=load_json(output/'create_request.json')
    assert canonical_hash(settings)==request['agent_config_sha256']
    assert all(str(binding.library.path) not in json.dumps(context) for binding in bound.values())


@pytest.mark.parametrize('phase',['validation','evaluation'])
def test_frozen_runner_reads_and_executes_exact_pin_and_rejects_write(tmp_path,phase):
    original='def run(api,args,state):\n    return api.call("observe",{})'
    bound,versions=bindings(tmp_path,phase=phase,source=original)
    # Practice author content may exist, but frozen reads cannot use its later version.
    for key,binding in bound.items(): candidate(binding.library,'def run(api,args,state):\n    return 999','later')
    revisions={key:binding.library.author_revision() for key,binding in bound.items()}
    models={'player_0':ScriptedModel([
        {'function':'skill_read','arguments':{'skill_id':'example'}},
        {'function':'skill_write','arguments':{'skill_id':'example','source':'def run(api,args,state):\n    return 123',
            'description':'forbidden frozen revision','expected_revision':revisions['player_0']}},
        {'function':'skill_execute','arguments':{'skill_id':'example','args':{},'state':{}}}, {'stop':True}]),
        'player_1':ScriptedModel([])}
    report,output,_=run_http(tmp_path,models,bound,max_function_calls=12)
    assert report['error_type'] is None,report
    context=json.loads(models['player_0'].turns[0][-1]['content'][0]['text'])
    names={row.get('function') for row in context['tools']}
    assert 'skill_write' not in names and 'skill_publish' not in names
    assert context['skill_runtime']['phase']==phase
    read=load_json(output/'player_0/decisions/0000/execution.json')
    assert read['result']['source']==original and read['result']['bundle_hash']==versions['player_0']['bundle_sha256']
    rejected=load_json(output/'player_0/decisions/0001/execution.json')
    assert rejected['ok'] is False and rejected['error']['code'] in ('unsupported_function','phase_read_only')
    execution=load_json(output/'player_0/decisions/0002/execution.json')
    assert execution['result']['bundle_hash']==versions['player_0']['bundle_sha256']
    assert execution['result']['last_feedback']['function']=='observe'
    assert all(binding.library.author_revision()==revisions[key] for key,binding in bound.items())
    assert report['function_calls']['player_0']==4  # read + rejected write + execute + inner observe


def test_internal_public_transport_failure_is_not_hidden_by_outer_skill_ok(tmp_path,monkeypatch):
    source='def run(api,args,state):\n    return api.call("get_robot",{})'
    bound,versions=bindings(tmp_path,source=source)
    decision={'function':'skill_execute','arguments':{'skill_id':'example',
        'candidate_event_id':versions['player_0']['candidate_event_id'],'args':{},'state':{}}}
    models={'player_0':ScriptedModel([decision,{'stop':True}]),'player_1':ScriptedModel([])}
    original=FunctionHTTPClient.call
    counts={}
    def fail(client,payload,**kwargs):
        if payload['function']=='get_robot':
            owner=client.binding['player_id'];counts[owner]=counts.get(owner,0)+1
            if owner=='player_0' and counts[owner]==2:
                raise BenchHTTPError('PRIVATE_TRANSPORT_CANARY',category='transport_timeout')
        return original(client,payload,**kwargs)
    monkeypatch.setattr(FunctionHTTPClient,'call',fail)
    report,output,_=run_http(tmp_path,models,bound,max_function_calls=12)
    assert report['run_status']=='infra_error' and report['termination']=='run_error',report
    assert report['run_error']['error_category']=='transport_timeout'
    assert report['decisions']['player_0']==1 and len(models['player_0'].turns)==1
    assert not report['action_recovery_pending'] and report['physical_attempts']['player_0']==0
    saved=load_json(output/'player_0/decisions/0000/execution.json')
    assert saved['ok'] is True and saved['result']['stop_reason']=='public_transport_failure'
    assert saved['result']['last_feedback']['error']['error_category']=='transport_timeout'
    assert 'PRIVATE_TRANSPORT_CANARY' not in json.dumps(report)+json.dumps(saved)


def test_skill_wall_budget_is_not_runner_infrastructure_failure(tmp_path,monkeypatch):
    from types import SimpleNamespace
    from agent.model_client import function_session
    from agent.model_client.learning import runtime as runtime_module
    source='def run(api,args,state):\n    return api.call("get_robot",{})'
    bound,versions=bindings(tmp_path,source=source)
    decision={'function':'skill_execute','arguments':{'skill_id':'example',
        'candidate_event_id':versions['player_0']['candidate_event_id'],'args':{},'state':{}}}
    models={'player_0':ScriptedModel([decision,{'stop':True}]),'player_1':ScriptedModel([])}
    clock=[runtime_module.time.monotonic()]
    for module in (runtime_module,function_session):
        monkeypatch.setattr(module,'time',SimpleNamespace(monotonic=lambda:clock[0]))
    original=FunctionHTTPClient.call
    counts={}
    def expire(client,payload,**kwargs):
        if payload['function']=='get_robot':
            owner=client.binding['player_id'];counts[owner]=counts.get(owner,0)+1
            if owner=='player_0' and counts[owner]==2:
                clock[0]=client.deadline+1
                raise BenchHTTPError('PRIVATE_TIMEOUT_CANARY',category='request_budget_exhausted')
        return original(client,payload,**kwargs)
    monkeypatch.setattr(FunctionHTTPClient,'call',expire)
    report,output,_=run_http(tmp_path,models,bound,max_function_calls=12,wall_budget=None)
    assert report['error_type'] is None and report['termination']=='model_requested_stop',report
    assert report['decisions']['player_0']==2 and report['physical_attempts']['player_0']==0
    assert not report['action_recovery_pending']
    saved=load_json(output/'player_0/decisions/0000/execution.json')
    assert saved['ok'] and saved['result']['stop_reason']=='skill_wall_deadline'
    assert saved['result']['last_feedback']['error']['error_category']=='request_budget_exhausted'
    assert 'PRIVATE_TIMEOUT_CANARY' not in json.dumps(report)+json.dumps(saved)


@pytest.mark.parametrize('kind',['partial','extra','shared'])
def test_file_binding_identities_fail_closed(tmp_path,kind):
    bound,_=bindings(tmp_path)
    entries={key:{'library':str(value.library.path),'phase':'practice'} for key,value in bound.items()}
    if kind=='partial': entries.pop('player_1')
    if kind=='extra': entries['player_2']=deepcopy(entries['player_0'])
    if kind=='shared': entries['player_1']=deepcopy(entries['player_0'])
    path=tmp_path/'binding.json';path.write_text(json.dumps({'protocol':'agent_skill_bindings.v1','bindings':entries}))
    with pytest.raises(ValueError): load_skill_bindings(path)
