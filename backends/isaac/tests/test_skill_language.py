"""Capability/fuel tests for interpreted skills; no Python or native execution."""
from dataclasses import replace
import json
import pytest

from agent.model_client.learning.skill_language import (SkillLanguageLimits, SkillStop,
    SkillLanguageError, execute_source, validate_source)


def execute(source, *, args=None, state=None, limits=None, invoke=None):
    calls=[]
    def callback(name,value):
        calls.append((name,value))
        return invoke(name,value) if invoke else {'ok':True,'result':{'attempt':len(calls)}}
    result=execute_source(source,args or {},state or {},callback,limits=limits)
    return result,calls


def test_model_authored_helpers_math_branches_and_feedback_loop_are_executed():
    source='''def norm(v):
    total = 0
    for item in v:
        total += item * item
    return sqrt(total)

def run(api,args,state):
    distances = []
    i = 0
    while i < len(args["points"]):
        distance = norm(args["points"][i])
        distances = distances + [distance]
        reply = api.call("query_motion", {"requested_duration_s": distance})
        if reply["result"]["attempt"] == 2:
            break
        i += 1
    state["norms"] = distances
    return {"state":state,"last":reply,"angle":atan2(1,1)}
'''
    state={'unchanged':True};args={'points':[[3,4],[5,12],[8,15]]}
    result,calls=execute(source,args=args,state=state)
    assert result['status']=='completed' and len(calls)==2
    assert [c[1][1]['requested_duration_s'] for c in calls]==[5.,13.]
    assert result['return_value']['state']=={'unchanged':True,'norms':[5.,13.]}
    assert state=={'unchanged':True} and args['points'][0]==[3,4]


@pytest.mark.parametrize('body',[
    'import os', 'from os import environ', 'return open("private")',
    'return __import__("os")', 'return eval("1")', 'return exec("pass")',
    'return api.__dict__', 'return api.call.__self__', 'return args.__class__',
    'return (1).__class__', 'return getattr(api,"call")', 'return globals()',
    'return locals()', 'return api', 'return api.library', 'return api.state()',
    'return [x for x in [1,2]]', 'return lambda x: x', 'return f"{args}"',
    'yield 1', 'raise Exception()', 'with open("x") as f:\n        pass',
    'try:\n        pass\n    except:\n        pass', 'return api.call("observe", **args)',
])
def test_forbidden_syntax_is_rejected_before_even_earlier_api_effect(body):
    source='def run(api,args,state):\n    api.call("hold",{"duration_s":0.1})\n    '+body+'\n'
    result,calls=execute(source)
    assert result['status']=='rejected' and calls==[], result


@pytest.mark.parametrize('source',[
    'api.call("hold",{})\ndef run(api,args,state):\n    return 1',
    '@decorate\ndef run(api,args,state):\n    return 1',
    'def run(api,args,state={}):\n    return 1',
    'def run(api,args,state):\n    return 1\ndef run(api,args,state):\n    return 2',
    'def run(api,args,state):\n    def helper():\n        return 1\n    return helper()',
    'async def run(api,args,state):\n    return 1',
])
def test_module_and_signature_restrictions_are_atomic(source):
    result,calls=execute(source)
    assert result['status']=='rejected' and not calls


def test_dynamic_dictionary_keys_and_list_updates_remain_json_only():
    result,calls=execute('''def run(api,args,state):
    positions = {}
    for i in range(len(args["names"])):
        positions[args["names"][i]] = args["values"][i]
    rows = [[1,2],[3,4]]
    rows[1][0] = 9
    return {"positions":positions,"rows":rows,"fallback":get(state,"missing",0),"names":keys(positions)}
''',args={'names':['joint1','joint2'],'values':[.1,.2]})
    assert result['status']=='completed' and not calls
    assert result['return_value']=={'positions':{'joint1':.1,'joint2':.2},'rows':[[1,2],[9,4]],'fallback':0,'names':['joint1','joint2']}


@pytest.mark.parametrize('source,code',[
    ('def run(api,args,state):\n    while True:\n        pass','loop_iteration_limit'),
    ('def recurse(x):\n    return recurse(x)\ndef run(api,args,state):\n    return recurse(1)','function_depth_limit'),
    ('def run(api,args,state):\n    for i in range(1000000):\n        pass','loop_iteration_limit'),
    ('def run(api,args,state):\n    return 1 / 0','division_by_zero'),
    ('def run(api,args,state):\n    return sqrt(-1)','invalid_calculation'),
    ('def run(api,args,state):\n    return {"x":1,"x":2}','duplicate_dictionary_key'),
    ('def run(api,args,state):\n    return args["__class__"]','invalid_dictionary_lookup'),
])
def test_limits_and_arithmetic_faults_are_explicit_and_finite(source,code):
    result,calls=execute(source,limits=replace(SkillLanguageLimits(),max_loop_iterations=4,max_function_depth=3))
    assert result['error']['code']==code and not calls,result


def test_fuel_and_tool_limits_preserve_actual_prior_feedback():
    source='def run(api,args,state):\n    for i in range(8):\n        result = api.call("observe",{})\n    return result'
    result,calls=execute(source,limits=replace(SkillLanguageLimits(),max_tool_calls=2))
    assert result['status']=='budget_exceeded' and len(calls)==2
    assert result['last_feedback']['result']['attempt']==2
    result,calls=execute(source,limits=replace(SkillLanguageLimits(),max_fuel=2))
    assert result['status']=='budget_exceeded' and not calls


def test_host_epoch_stop_retains_actual_complete_feedback_and_no_later_calls():
    feedback={'ok':True,'result':{'status':'cancelled','termination_reason':'turn_changed','contact_history':{'public':'kept'}}}
    def stop(name,args): raise SkillStop('authorization_epoch_changed',feedback)
    result,calls=execute('def run(api,args,state):\n    api.call("hold",{})\n    api.call("hold",{})',invoke=stop)
    assert result['status']=='stopped' and result['last_feedback']==feedback and len(calls)==1


def test_no_host_exception_text_or_non_json_object_hooks_are_exposed():
    class HostObject:
        def __iter__(self): raise AssertionError('must not run')
        def __repr__(self): return 'PRIVATE_HOST_CANARY'
    result,_=execute('def run(api,args,state):\n    return api.call("observe",{})',invoke=lambda *_:HostObject())
    assert result['error']['code']=='non_json_value' and 'PRIVATE_HOST_CANARY' not in json.dumps(result)
    def broken(*_): raise RuntimeError('PRIVATE_HOST_CANARY')
    result,_=execute('def run(api,args,state):\n    return api.call("observe",{})',invoke=broken)
    assert result['error']['code']=='runtime_internal_error' and 'PRIVATE_HOST_CANARY' not in json.dumps(result)


def test_input_source_and_value_limits_precede_calls():
    with pytest.raises(SkillLanguageError): validate_source('x'*100,replace(SkillLanguageLimits(),max_source_bytes=32))
    result,calls=execute('def run(api,args,state):\n    return api.call("observe",{})',args={'x':float('nan')})
    assert result['error']['code']=='nonfinite_number' and not calls
    result,calls=execute('def run(api,args,state):\n    x=[1,2]\n    return x+x',limits=replace(SkillLanguageLimits(),max_container_items=3))
    assert result['error']['code']=='concatenation_limit' and not calls


def test_temporary_copies_and_aggregate_json_size_are_bounded_before_growth():
    source='def run(api,args,state):\n    for i in range(100):\n        copy = args["payload"]\n    return copy'
    result,calls=execute(source,args={'payload':['x'*100]*4},
        limits=replace(SkillLanguageLimits(),max_total_value_copy_bytes=3000))
    assert result['error']['code']=='total_value_copy_bytes_limit' and not calls
    result,calls=execute('def run(api,args,state):\n    return api.call("observe",{})',
        args={'payload':['x'*100]*100},limits=replace(SkillLanguageLimits(),max_value_bytes=1000))
    assert result['error']['code']=='value_bytes_limit' and not calls
