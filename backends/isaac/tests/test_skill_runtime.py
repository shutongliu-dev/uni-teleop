"""Model-authored revision/execute/freeze tests; CPU public-session doubles only."""
from copy import deepcopy
from dataclasses import replace
import hashlib
import json
from pathlib import Path

import pytest

from agent.model_client.learning.journal import canonical_json
from agent.model_client.learning.skills import SkillLibrary
from agent.model_client.learning.runtime import SkillRuntime, SkillRuntimeLimits
from agent.model_client.learning.skill_language import SkillLanguageLimits
from test_function_session import Client, session


SIMPLE='''def run(api,args,state):
    first = api.call("observe",{})
    for i in range(args["repeats"]):
        reply = api.call("hold",{"duration_s":args["duration"]})
        if not reply["ok"]:
            return {"recovery_needed":True,"feedback":reply}
    return {"feedback":reply,"state":state}
'''


def make(tmp_path,*,phase='practice',library=None,candidate=None,limits=None,client=None):
    library=library or SkillLibrary.create('learner',learning_root=tmp_path/'libraries',
        contract={'dependencies':{},'compatibility':{'task_id':'public_fixture'}})
    host,client=session(tmp_path/'session',client or Client(board=True))
    runtime=SkillRuntime(library,session=host,phase=phase,journal_path=tmp_path/'runtime_journal',
        limits=limits,frozen_candidate_event_id=candidate)
    return runtime,host,client


def publish(runtime,source=SIMPLE,skill_id='adapt',transaction_id='v1'):
    revision=runtime.execute_tool('skill_status',{})['result']['author_revision']
    written=runtime.execute_tool('skill_write',{'skill_id':skill_id,'source':source,
        'description':'Model-authored feedback-dependent program','expected_revision':revision})
    assert written['ok'],written
    published=runtime.execute_tool('skill_publish',{'expected_revision':written['result']['author_revision'],
        'transaction_id':transaction_id,'reason':'Self-authored candidate'})
    assert published['ok'],published
    return published['result']


def execute(runtime,candidate,*,skill_id='adapt',args=None,**extra):
    return runtime.execute_tool('skill_execute',{'skill_id':skill_id,'candidate_event_id':candidate['candidate_event_id'],
        'args':args or {'repeats':2,'duration':.1},'state':{'from_public_reasoning':True},**extra})


def test_empty_library_model_writes_executes_feedback_revision_and_exact_trace(tmp_path):
    runtime,host,client=make(tmp_path)
    assert runtime.execute_tool('skill_status',{})['result']['catalog']=={'skills':[]}
    first=publish(runtime)
    result=execute(runtime,first)
    assert result['ok'] and result['result']['status']=='completed',result
    run=result['result']
    assert len(client.posts)==2 and host.actions==2 and host.calls==3
    assert len(run['calls'])==3 and run['last_feedback']['result']['status']=='completed'
    for row in run['calls']:
        record=runtime.execute_tool('skill_trace',{'execution_id':run['execution_id'],'call_index':row['call_index']})
        assert record['ok'] and record['result']['record_sha256']==row['record_sha256']
        assert hashlib.sha256(canonical_json(record['result']['record'])).hexdigest()==row['record_sha256']
    # A real rewritten source changes execution, rather than selecting a built-in script.
    second=publish(runtime,'def run(api,args,state):\n    return api.call("hold",{"duration_s":0.3})',transaction_id='v2')
    assert first['bundle_hash']!=second['bundle_hash']
    assert execute(runtime,second)['result']['status']=='completed'
    assert client.posts[-1]['arguments']['duration_s']==.3
    assert runtime.library.read_bundle_file(first['candidate_event_id'],'skills/adapt.py').decode()==SIMPLE
    assert len([e for e in host.action_api.journal.read() if e['kind']=='action_prepared'])==3
    assert len([e for e in runtime.journal.read() if e['kind']=='skill_api_completed'])==4


@pytest.mark.parametrize('phase',['validation','evaluation'])
def test_frozen_candidate_cannot_write_publish_override_or_read_author(tmp_path,phase):
    practice,_,_=make(tmp_path/'practice')
    pinned=publish(practice)
    other=publish(practice,'def run(api,args,state):\n    return 999',transaction_id='v2')
    runtime,host,client=make(tmp_path/phase,phase=phase,library=practice.library,candidate=pinned['candidate_event_id'])
    before=practice.library.author_revision()
    assert 'skill_write' not in {row['function'] for row in runtime.tools()}
    write=runtime.execute_tool('skill_write',{'skill_id':'adapt','source':SIMPLE,'description':'attempt','expected_revision':before})
    assert write['error']['code']=='phase_read_only'
    assert runtime.execute_tool('skill_publish',{'expected_revision':before,'transaction_id':'bad','reason':'attempt'})['error']['code']=='phase_read_only'
    assert execute(runtime,other)['error']['code']=='frozen_candidate_mismatch'
    assert execute(runtime,pinned,source='def run(api,args,state):\n    return 999')['error']['code']=='invalid_arguments'
    read=runtime.execute_tool('skill_read',{'skill_id':'adapt'})
    assert read['result']['source']==SIMPLE and read['result']['bundle_hash']==pinned['bundle_hash']
    assert execute(runtime,pinned)['result']['status']=='completed' and len(client.posts)==2
    assert practice.library.author_revision()==before


@pytest.mark.parametrize('handoff',[False,True])
def test_any_confirmed_epoch_change_stops_before_the_next_api_even_same_player(tmp_path,handoff):
    runtime,host,client=make(tmp_path)
    pinned=publish(runtime)
    original=host.execute
    def change(decision):
        reply=original(decision)
        if decision.get('function')=='hold':
            client.public['board_game']['turn_index']+=1
            if handoff: client.public['board_game']['current_player']='player_1'
            host.refresh()
        return reply
    host.execute=change
    result=execute(runtime,pinned)['result']
    assert result['status']=='stopped' and result['stop_reason']=='authorization_epoch_changed'
    assert len(client.posts)==1 and len(result['calls'])==2
    assert result['last_feedback']['result']['status']=='completed'


def test_public_rejections_can_drive_model_written_recovery_and_do_not_become_success(tmp_path):
    runtime,host,client=make(tmp_path)
    source='''def run(api,args,state):
    reply = api.call("hold",{"duration_s":-1})
    if not reply["ok"]:
        return api.call("hold",{"duration_s":0.2})
    return reply
'''
    pinned=publish(runtime,source)
    result=execute(runtime,pinned)['result']
    assert result['status']=='completed' and result['calls'][0]['ok'] is False
    assert result['calls'][0]['error']['code']=='invalid_arguments'
    assert result['calls'][1]['primitive_status']=='completed' and len(client.posts)==1
    assert host.actions==2  # Both attempted public action calls remain counted.


def test_source_private_access_rejected_before_library_or_action_changes(tmp_path):
    runtime,host,client=make(tmp_path)
    revision=runtime.library.author_revision()
    reply=runtime.execute_tool('skill_write',{'skill_id':'unsafe','description':'not permitted',
        'expected_revision':revision,'source':'def run(api,args,state):\n    return open("PRIVATE_HOST_CANARY")'})
    assert reply['error']['code']=='skill_source_rejected'
    assert 'PRIVATE_HOST_CANARY' not in json.dumps(reply)
    assert runtime.library.author_revision()==revision and not client.posts


def test_reserved_callback_names_cannot_call_runtime_tools_or_private_api(tmp_path):
    runtime,host,client=make(tmp_path)
    for index,name in enumerate(('skill_write','read_decision','set_object_pose','get_private_state')):
        source=f'def run(api,args,state):\n    return api.call("{name}",{{}})'
        pinned=publish(runtime,source,transaction_id=f'v{index}')
        reply=execute(runtime,pinned)['result']
        assert reply['status']=='stopped' and reply['stop_reason']=='unavailable_public_function'
    assert not client.posts and host.calls==0


def test_skill_cumulative_budget_cannot_reset_by_reopening_runtime(tmp_path):
    limits=SkillRuntimeLimits(max_api_calls=1)
    runtime,host,client=make(tmp_path,limits=limits)
    pinned=publish(runtime)
    result=execute(runtime,pinned)['result']
    assert result['stop_reason']=='runtime_api_budget' and len(result['calls'])==1 and not client.posts
    assert execute(runtime,pinned)['result']['stop_reason']=='runtime_api_budget'
    with pytest.raises(ValueError,match='cannot reopen'):
        SkillRuntime(runtime.library,session=host,phase='practice',journal_path=tmp_path/'runtime_journal')


def test_trace_is_own_runtime_and_integrity_checked(tmp_path):
    runtime,_,_=make(tmp_path/'one'); pinned=publish(runtime)
    report=execute(runtime,pinned)['result']
    other,_,_=make(tmp_path/'two')
    args={'execution_id':report['execution_id'],'call_index':0}
    assert other.execute_tool('skill_trace',args)['error']['code']=='trace_unavailable'
    path=runtime.records/report['execution_id']/'call_000000.json'
    path.write_text('{"tampered":true}')
    assert runtime.execute_tool('skill_trace',args)['error']['code']=='trace_integrity'
    assert runtime.execute_tool('skill_trace',{**args,'path':'/private'})['error']['code']=='invalid_arguments'


def test_published_transaction_retry_retains_original_candidate_after_later_edit(tmp_path):
    runtime,_,_=make(tmp_path)
    first=publish(runtime)
    publish(runtime,'def run(api,args,state):\n    return 2',transaction_id='v2')
    retry=runtime.execute_tool('skill_publish',{'expected_revision':first['author_revision'],
        'transaction_id':'v1','reason':'Self-authored candidate'})
    assert retry['ok'] and retry['result']==first
    assert len([e for e in runtime.library.journal.read() if e['kind']=='candidate_committed'])==2


def test_maximum_library_transaction_id_remains_executable(tmp_path):
    runtime,_,_=make(tmp_path)
    pinned=publish(runtime,'def run(api,args,state):\n    return 1',transaction_id='t'*128)
    assert len(pinned['candidate_event_id'])>128
    assert execute(runtime,pinned)['result']['return_value']==1


def test_language_value_limit_cannot_erase_complete_actual_api_feedback(tmp_path):
    runtime,host,client=make(tmp_path,limits=SkillRuntimeLimits(language=replace(SkillLanguageLimits(),max_string_length=256)))
    pinned=publish(runtime,'def run(api,args,state):\n    return api.call("observe",{})')
    original=host.execute
    def bulky(decision):
        feedback=original(decision)
        feedback['result']['complete_public_text']='x'*512
        return feedback
    host.execute=bulky
    result=execute(runtime,pinned)['result']
    assert result['status']=='budget_exceeded' and result['error']['code']=='string_limit'
    assert result['last_feedback']['result']['complete_public_text']=='x'*512
    trace=runtime.execute_tool('skill_trace',{'execution_id':result['execution_id'],'call_index':0})
    assert trace['result']['record']['feedback']==result['last_feedback']


def test_episode_and_skill_deadlines_tightened_for_actual_calls_and_restored(tmp_path):
    runtime,host,client=make(tmp_path,limits=SkillRuntimeLimits(language=replace(SkillLanguageLimits(),max_runtime_s=10)))
    pinned=publish(runtime,'def run(api,args,state):\n    return api.call("observe",{})')
    seen=[]; original=host.execute
    def check(decision):
        seen.append(host.deadline)
        return original(decision)
    host.execute=check
    assert host.deadline is None
    assert execute(runtime,pinned)['result']['status']=='completed'
    assert len(seen)==1 and seen[0] is not None and host.deadline is None and client.deadline is None


def test_ordinary_episode_terminal_receipt_stops_before_next_action(tmp_path):
    runtime,host,client=make(tmp_path,client=Client(board=False))
    pinned=publish(runtime)
    original=host.execute
    def terminal(decision):
        reply=original(decision)
        if decision.get('function')=='hold': reply['result']['episode_finished']=True
        return reply
    host.execute=terminal
    result=execute(runtime,pinned)['result']
    assert result['stop_reason']=='episode_terminal' and len(client.posts)==1
    assert result['last_feedback']['result']['episode_finished'] is True


def test_wall_expiry_during_action_preserves_pending_and_never_claims_physics_stopped(tmp_path,monkeypatch):
    from types import SimpleNamespace
    from agent.model_client import function_session
    from agent.model_client.bench_http import BenchHTTPError
    runtime,host,client=make(tmp_path)
    pinned=publish(runtime)
    clock=[function_session.time.monotonic()]
    monkeypatch.setattr(function_session,'time',SimpleNamespace(monotonic=lambda:clock[0]))
    def expiry(episode,action_id):
        clock[0]=host.deadline+1
        raise BenchHTTPError('PRIVATE_TIMEOUT_CANARY',category='request_budget_exhausted')
    client.get_action=expiry
    result=execute(runtime,pinned)['result']
    assert result['stop_reason']=='action_pending' and result['physical_action_pending'] is True
    assert len(client.posts)==1 and host.pending is not None and host.deadline is None
    assert 'PRIVATE_TIMEOUT_CANARY' not in json.dumps(result)


@pytest.mark.parametrize(('episode_after','category','expires','expected'),[
    (None,'request_budget_exhausted',True,'skill_wall_deadline'),
    (None,'transport_timeout',True,'skill_wall_deadline'),
    (5.,'request_budget_exhausted',True,'episode_wall_deadline'),
    (10.,'transport_timeout',True,'episode_wall_deadline'),
    (20.,'transport_timeout',True,'skill_wall_deadline'),
    (None,'transport_timeout',False,'public_transport_failure'),
    (None,'request_budget_exhausted',False,'public_transport_failure'),
    (None,'invalid_response',True,'public_transport_failure'),
    (None,'transport_error',True,'public_transport_failure'),
])
def test_deadline_owner_is_explicit_and_only_expired_timeouts_are_budgets(
        tmp_path,monkeypatch,episode_after,category,expires,expected):
    from types import SimpleNamespace
    from agent.model_client import function_session
    from agent.model_client.learning import runtime as runtime_module
    from agent.model_client.bench_http import BenchHTTPError
    runtime,host,client=make(tmp_path,limits=SkillRuntimeLimits(
        language=replace(SkillLanguageLimits(),max_runtime_s=10)))
    pinned=publish(runtime,'def run(api,args,state):\n    return api.call("get_robot",{})')
    clock=[runtime_module.time.monotonic()]
    previous=None if episode_after is None else clock[0]+episode_after
    host.deadline=previous
    for module in (runtime_module,function_session):
        monkeypatch.setattr(module,'time',SimpleNamespace(monotonic=lambda:clock[0]))
    def failure(payload):
        clock[0]=host.deadline+(1 if expires else -1)
        raise BenchHTTPError('PRIVATE_TIMEOUT_CANARY',category=category)
    client.call=failure
    result=execute(runtime,pinned)['result']
    assert result['status']=='stopped' and result['stop_reason']==expected
    assert result['error']['error_category']==category
    assert result['last_feedback']['error']['error_category']==category
    assert len(result['calls'])==1 and host.calls==1 and not client.posts
    assert not result['physical_action_pending'] and host.deadline==previous and client.deadline is None
    assert 'PRIVATE_TIMEOUT_CANARY' not in json.dumps(result)


def test_expired_preflight_retains_previous_completed_feedback_and_new_error(tmp_path,monkeypatch):
    from types import SimpleNamespace
    from agent.model_client import function_session
    from agent.model_client.learning import runtime as runtime_module
    runtime,host,client=make(tmp_path)
    pinned=publish(runtime)
    clock=[runtime_module.time.monotonic()]
    for module in (runtime_module,function_session):
        monkeypatch.setattr(module,'time',SimpleNamespace(monotonic=lambda:clock[0]))
    original=host.execute
    def expire_after_completion(decision):
        reply=original(decision)
        clock[0]=host.deadline+1
        return reply
    host.execute=expire_after_completion
    result=execute(runtime,pinned)['result']
    assert result['stop_reason']=='skill_wall_deadline'
    assert result['last_feedback']['ok'] and result['last_feedback']['function']=='observe'
    assert result['error']['error_category']=='request_budget_exhausted'
    assert len(result['calls'])==1 and not client.posts and host.deadline is None


def test_expired_initial_refresh_is_budget_without_entering_source(tmp_path,monkeypatch):
    from types import SimpleNamespace
    from agent.model_client import function_session
    from agent.model_client.learning import runtime as runtime_module
    runtime,host,client=make(tmp_path)
    pinned=publish(runtime)
    clock=[runtime_module.time.monotonic()]
    host.deadline=clock[0]-1
    for module in (runtime_module,function_session):
        monkeypatch.setattr(module,'time',SimpleNamespace(monotonic=lambda:clock[0]))
    result=execute(runtime,pinned)['result']
    assert result['stop_reason']=='episode_wall_deadline' and not result['language_entered']
    assert result['error']['error_category']=='request_budget_exhausted'
    assert result['calls']==[] and result['last_feedback'] is None and not client.posts


def test_pending_action_stops_program_without_resubmission(tmp_path):
    runtime,host,client=make(tmp_path)
    pinned=publish(runtime)
    client.lose_post=True;client.no_status=True
    host.max_status_queries=1
    result=execute(runtime,pinned)['result']
    assert result['status']=='stopped' and result['stop_reason']=='action_pending'
    assert result['physical_action_pending'] is True
    assert len(client.posts)==1 and host.pending is not None
    assert execute(runtime,pinned)['result']['stop_reason']=='action_pending' and len(client.posts)==1


def test_enabled_geometry_math_is_the_same_public_numeric_callback(tmp_path):
    from agent.model_client.geometry_math import FUNCTION_INPUTS,execute as compute
    runtime,host,client=make(tmp_path)
    host.local_functions=FUNCTION_INPUTS;host.local_execute=compute
    pinned=publish(runtime,'def run(api,args,state):\n    return api.call("compute_rotation",{"quaternion_wxyz":[2,0,0,0]})')
    before=len(client.calls)
    result=execute(runtime,pinned)['result']
    assert result['status']=='completed' and result['last_feedback']['ok']
    assert result['return_value']['result']['result']['quaternion_wxyz']==[1.,0.,0.,0.]
    assert len(client.calls)==before and not client.posts and host.calls==1


def test_real_http_memory_reveal_stops_skill_before_second_call_and_keeps_public_event(tmp_path):
    from fastapi.testclient import TestClient
    from agent.model_client.card_games import CARD_PROTOCOL
    from agent.model_client.function_http import FunctionHTTPClient
    from agent.model_client.function_session import FunctionSession
    from bench.runtime.controllers import default_registry
    from bench.runtime.server import create_app
    from bench.runtime.service import AgentService
    from scenes.workrooms.catalog import task_definition
    from test_card_game_http_lifecycle import SyntheticCardWorker
    release=Path(task_definition('card_room','memory_pairs')['source_release']['path'])
    workers=[]
    def factory(**kwargs):
        worker=SyntheticCardWorker(**kwargs);workers.append(worker);return worker
    service=AgentService(tmp_path/'episodes',release=release,registry=default_registry(release),worker_factory=factory)
    with TestClient(create_app(service=service)) as http:
        ready=http.post('/board-games',json={'task_id':'memory_pairs','agent_id':'runtime_cpu','agent_version':'v1',
            'agent_config_sha256':'b'*64,'robot_id':'dual_fr3_sharpa_w01','track':'system0'}).json()
        client=FunctionHTTPClient('http://testserver',ready['episode_id'],player=ready['players'][0],
            client=http,game_protocol=CARD_PROTOCOL,spec=ready['spec'])
        client.bind_board_state(ready)
        host=FunctionSession(client,output=tmp_path/'agent',spec=ready['spec'],trial_id='player_0',max_actions=10,max_calls=30)
        host.initialize()
        baseline=host.execute({'function':'hold','arguments':{'duration_s':.4,'mode':'acknowledged_drive'}})
        assert baseline['ok'],baseline
        workers[0].mode='execute_pending'
        library=SkillLibrary.create('http_learner',learning_root=tmp_path/'library',contract={'dependencies':{},'compatibility':{}})
        runtime=SkillRuntime(library,session=host,phase='practice',journal_path=tmp_path/'journal')
        source='''def run(api,args,state):
    reserved = api.intent({"type":"reveal_card","card_id":args["card"]})
    if not reserved["ok"]:
        return reserved
    revealed = api.call("hold",{"duration_s":0.5})
    forbidden_later = api.call("hold",{"duration_s":0.1})
    return forbidden_later
'''
        pinned=publish(runtime,source)
        result=execute(runtime,pinned,args={'card':'card_00'})['result']
        assert result['status']=='stopped' and result['stop_reason']=='authorization_epoch_changed',result
        assert len(result['calls'])==2 and result['last_feedback']['result']['termination_reason']=='board_turn_changed'
        assert host.state['board_game']['current_player']=='player_0' and host.state['board_game']['turn_index']==1
        assert len([row for row in host.action_api.journal.read() if row['kind']=='action_prepared'])==2
        assert client.events()['events'][0]['type']=='card_revealed'
        assert result['physical_action_pending'] is False
        # The authenticated peer receives the same public reveal, not skill source or trace.
        peer=FunctionHTTPClient('http://testserver',ready['episode_id'],player=ready['players'][1],
            client=http,game_protocol=CARD_PROTOCOL,spec=ready['spec'])
        peer.bind_board_state(ready)
        assert peer.events()['events']==client.events()['events']
        assert 'forbidden_later' not in json.dumps(peer.events())
        host.execute({'function':'finish_episode','arguments':{}})
