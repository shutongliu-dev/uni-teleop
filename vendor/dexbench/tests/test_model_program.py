"""Generated programs execute a closed, bounded language, never Python code."""
from dataclasses import replace
import hashlib
import json

import pytest

from agent.model_client.program import ProgramLimits, ProgramStop, execute_program


class PublicTools:
    def __init__(self):
        self.calls, self.sequence = [], 0

    def __call__(self, tool, arguments):
        self.calls.append((tool, arguments))
        if tool == 'observe':
            return {'sequence': self.sequence, 'tactile': {'status': 'available', 'load_n': self.sequence * .1}}
        if tool == 'move_geometry':
            self.sequence += 1
            return {'status': 'completed', 'sequence': self.sequence}
        if tool == 'point_at':
            return {'tool':'point_at', 'episode_id':'fixture_episode',
                    'sequence':self.sequence, 'sim_time_s':self.sequence*.02,
                    'camera_id':arguments['camera_id'], 'camera_revision':self.sequence,
                    'u':arguments['u'], 'v':arguments['v'],
                    'pixel_center_uv':[arguments['u']+.5, arguments['v']+.5],
                    'depth_m':.3, 'depth_semantics':'camera_z', 'depth_unit':'m',
                    'depth_sha256':'a'*64, 'world_position_m':[.1, .2, .3],
                    'frame_id':'world',
                    'scope':'visible RGB-D pixel; no object identity or grasp inference'}
        return {'status': 'cancelled', 'reason': arguments['reason']}


def run(source, *, invoke=None, **changes):
    tools = invoke or PublicTools()
    return execute_program(source, tools, limits=replace(ProgramLimits(), **changes)), tools


MOVE = 'move_geometry(wrist_delta_m=[0, 0, 0.005], duration_s=1.0)'


def test_wrist_precision_reaches_same_tool_parser_through_code_and_free_envelope():
    from agent.model_client.tools import parse_tool_call
    from agent.model_client.free_interface import parse_free_decision
    source = 'move_geometry(wrist_delta_m=[0,0,-0.0001], duration_s=1.0, position_tolerance_m=0.000025, orientation_tolerance_rad=0.005)'
    envelope = parse_free_decision({'interface': 'fingertip_code', 'payload': source})
    recorded = []
    original = PublicTools()
    def invoke(name, arguments):
        recorded.append(parse_tool_call({'tool': name, 'arguments': arguments}))
        return original(name, arguments)
    result, _ = run(envelope['payload'], invoke=invoke)
    assert result['status'] == 'completed'
    assert recorded[0]['arguments']['position_tolerance_m'] == .000025
    assert recorded[0]['arguments']['orientation_tolerance_rad'] == .005


@pytest.mark.parametrize('value', ['True', '0', '-1'])
def test_code_wrist_precision_uses_original_action_validation(value):
    from agent.model_client.tools import parse_tool_call
    result, _ = run('move_geometry(wrist_delta_m=[0,0,-0.0001],duration_s=1,position_tolerance_m='+value+')',
                    invoke=lambda name, arguments: parse_tool_call({'tool': name, 'arguments': arguments}))
    assert result['status'] != 'completed'


def test_program_branches_on_fresh_public_data_between_subactions():
    result, tools = run('''o = observe()
for i in range(5):
    if o["tactile"]["status"] == "available" and o["tactile"]["load_n"] >= 0.2:
        break
    r = move_geometry(wrist_delta_m=[0, 0, -0.005], duration_s=1.0)
    if r["status"] != "completed":
        break
    o = observe()
stop(reason="Public load threshold reached")''')
    assert result['status'] == 'stopped'
    assert result['stop_reason'] == 'Public load threshold reached'
    assert result['move_calls'] == 2 and result['observe_calls'] == 3
    assert [row['result']['sequence'] for row in result['trace'] if row['tool'] == 'observe'] == [0, 1, 2]
    assert [name for name, _ in tools.calls] == ['observe', 'move_geometry', 'observe', 'move_geometry', 'observe', 'stop']
    assert all(row['status'] in ('completed', 'stopped') for row in result['trace'])


def test_failed_action_is_json_feedback_not_interpreter_or_infrastructure_failure():
    calls=[]
    def invoke(name,args):
        calls.append(name)
        return {'feedback':{'status':'failed'},'observation':{'sequence':0}}
    result, _ = run('''for i in range(3):
    state = move_geometry(wrist_delta_m=[0,0,0.005],duration_s=1)
    if state["feedback"]["status"] != "completed":
        break''',invoke=invoke)
    assert result['status']=='completed' and calls==['move_geometry']
    assert result['trace'][0]['result']['feedback']['status']=='failed'


def test_character_and_utf8_byte_limits_both_precede_parser(monkeypatch):
    import agent.model_client.program as module
    monkeypatch.setattr(module.ast,'parse',lambda *_args,**_kwargs:pytest.fail('Oversized source reached parser'))
    for source in ('x'*9,'界'*3):
        result, tools=run(source,max_source_bytes=8)
        assert result['status']=='budget_exceeded' and result['error']['code']=='source_bytes_limit'
        assert tools.calls==[]


def test_point_query_and_arithmetic_are_bounded_json_tool_arguments():
    result, tools = run('''p = point_at("task", u=320, v=240)
z = p["world_position_m"][2]
delta = min([abs(z / 100), 0.01])
move_geometry(wrist_delta_m=[0, 0, delta], fingertip_delta_m={"thumb":[0.001,0,0]}, duration_s=1.0)''')
    assert result['status'] == 'completed' and result['point_queries'] == result['move_calls'] == 1
    assert tools.calls[0] == ('point_at', {'camera_id':'task', 'u':320, 'v':240})
    assert tools.calls[1][1]['wrist_delta_m'] == [0, 0, .003]


def test_range_negative_step_continue_and_pure_functions():
    result, tools = run('''a = [2, 3, 4]
for i in range(4, 0, -1):
    if i % 2 != 0:
        continue
    if len(a) == 3 and max(a) == 4 and min(5, 6) == 5:
        point_at("overview", i, abs(-2))''')
    assert result['status'] == 'completed' and result['loop_iterations'] == 4
    assert [args['u'] for name,args in tools.calls] == [4,2]


def test_dictionary_membership_and_short_circuit_do_not_read_missing_values():
    result, tools = run('''o = {"status":"not_sampled"}
if "load_n" in o and o["load_n"] > 0:
    stop()
if not ("load_n" in o) or o["load_n"] > 0:
    observe()''')
    assert result['status'] == 'completed' and result['observe_calls'] == 1


@pytest.mark.parametrize('bad', [
    'import os', 'from os import system', 'x = observe.__class__', 'x = (1).__class__',
    'x = getattr(1, "__class__")', 'open("/tmp/private")', 'exec("x=1")', 'eval("1")',
    'x = __import__("os")', 'while True:\n    observe()', 'def f():\n    observe()',
    'class A:\n    x=1', 'x = lambda: 1', 'x = [i for i in range(3)]',
    'x = {i: i for i in range(3)}', 'x = (i for i in range(3))', 'x = {1,2}',
    'try:\n    observe()\nexcept:\n    observe()', 'with observe():\n    stop()',
    'x = f"secret{1}"', 'x = (y := 1)', 'x = 2 ** 100', 'x = 1 << 100',
    'x = [1][0:1]', 'x = {"__class__":1}', 'x = {"a":1,"a":2}',
    'x = {**{"a":1}}', 'x = (1, 2)', 'x = b"bytes"', 'x = ...',
    'x = observe()["__class__"]', '_hidden = 1', 'x__y = 1', 'observe = 1',
    'x = observe', 'x = [observe()]', 'x = abs(observe())', 'if observe():\n    stop()',
    'move_geometry(wrist_delta_m=observe(), duration_s=1)',
    'observe(x=1)', 'point_at("task", 1, u=2, v=3)', 'stop(reason="x", reason="y")',
    'move_geometry(**{"duration_s":1})', 'move_geometry(*[1])',
    'move_geometry(duration_s=1)', 'move_geometry(wrist_delta_m=[0,0,0])',
    'move_geometry(wrist_delta_m=[0,0,0],duration_s=1,object_pose=[0,0,0])',
    'a, b = [1,2]', 'x = y = 1', 'a[0] = 1', 'x: int = 1', 'x += 1',
    'del x', 'assert True', 'global x', 'raise Exception()', 'break', 'continue',
    'for i in [1,2]:\n    observe()', 'for i in range(x):\n    observe()',
    'for i in range(True):\n    observe()', 'for i in range(2):\n    observe()\nelse:\n    stop()',
])
def test_full_prevalidation_rejects_unsafe_suffix_before_any_tool_effect(bad):
    tools = PublicTools()
    result = execute_program('observe()\n' + bad, tools)
    assert result['status'] == 'rejected', result
    assert result['tool_calls'] == 0 and tools.calls == [] and result['trace'] == []


def test_unreachable_unsafe_branch_is_still_rejected_before_execution():
    result, tools = run('observe()\nif False:\n    import os')
    assert result['status'] == 'rejected' and tools.calls == []


@pytest.mark.parametrize('source', [
    '```\nobserve()\n```', '```py\nobserve()\n```',
    '```python\nobserve()\n```\n```python\nstop()\n```',
    'Here is code:\n```python\nobserve()\n```', '```python\nobserve()\n```\ntrailing',
    '', '   ', '# only comment', 'if', '\ud800', None,
])
def test_empty_invalid_or_ambiguous_source_has_no_effect(source):
    result, tools = run(source)
    assert result['status'] == 'rejected' and tools.calls == []


def test_single_python_fence_and_hashes_preserve_original_and_program():
    source = '```python\nobserve()\n```'
    result, _ = run(source)
    assert result['status'] == 'completed'
    assert result['source_sha256'] == hashlib.sha256(source.encode()).hexdigest()
    assert result['program_sha256'] == hashlib.sha256(b'observe()').hexdigest()


@pytest.mark.parametrize('source,changes,code', [
    ('observe()', {'max_source_bytes':3}, 'source_bytes_limit'),
    ('observe()', {'max_ast_nodes':2}, 'ast_structure_limit'),
    ('x = [[[1]]]', {'max_ast_depth':3}, 'ast_structure_limit'),
    ('x = [[[1]]]', {'max_value_depth':1}, 'value_structure_limit'),
    ('x = [1,2,3]', {'max_container_items':2}, 'container_limit'),
    ('x = "abcdef"', {'max_string_length':3}, 'string_limit'),
    ('x = 100', {'max_number_abs':99}, 'number_limit'),
    ('x = 0x'+'f'*1000, {}, 'number_limit'),
    ('for i in range(17):\n    observe()', {}, 'loop_iteration_limit'),
    ('for i in range(2):\n    for j in range(2):\n        observe()', {'max_loop_depth':1}, 'loop_depth_limit'),
])
def test_pre_execution_structural_limits(source, changes, code):
    result, tools = run(source, **changes)
    assert result['status'] == 'budget_exceeded' and result['error']['code'] == code
    assert tools.calls == []


@pytest.mark.parametrize('source,code', [
    ('x = 1 / 0', 'division_by_zero'), ('x = 1 // 0', 'division_by_zero'),
    ('x = 1 % 0', 'division_by_zero'), ('x = [1] * 1000000000', 'numeric_operand_required'),
    ('x = "x" * 1000000000', 'numeric_operand_required'), ('x = True + 1', 'numeric_operand_required'),
    ('x = missing', 'undefined_variable'), ('x = {}["missing"]', 'invalid_dictionary_lookup'),
    ('x = [1][10]', 'invalid_sequence_lookup'), ('x = [1][True]', 'invalid_sequence_lookup'),
    ('x = 1[0]', 'value_is_not_indexable'), ('x = 1 < "a"', 'invalid_comparison_operands'),
    ('x = 1 in "a"', 'invalid_membership_operands'), ('x = len(1)', 'len_requires_container'),
    ('x = min([])', 'empty_min_max'), ('x = abs(True)', 'numeric_operand_required'),
    ('point_at("task", -1, 0)', 'invalid_point_arguments'), ('point_at("task", True, 0)', 'invalid_point_arguments'),
    ('stop(reason=42)', 'invalid_stop_reason'),
])
def test_runtime_errors_stop_without_calling_unrelated_tools(source, code):
    result, tools = run(source + '\nobserve()')
    assert result['status'] == 'runtime_error' and result['error']['code'] == code
    assert tools.calls == []


@pytest.mark.parametrize('source,changes,code', [
    ('x = 10 * 11', {'max_number_abs':100}, 'number_limit'),
    ('x=1\ny=2', {'max_variables':1}, 'variable_limit'),
    ('x="abcd"\ny="abcd"', {'max_live_value_bytes':8}, 'live_value_bytes_limit'),
    ('x=[1,2,3]', {'max_value_nodes':3}, 'value_structure_limit'),
    ('x="abcd"', {'max_value_bytes':5}, 'value_bytes_limit'),
    ('for i in range(10):\n    x=i+1', {'max_fuel':8}, 'fuel_limit'),
    ('for i in range(3):\n    for j in range(3):\n        x=j', {'max_total_loop_iterations':6}, 'total_loop_iteration_limit'),
])
def test_runtime_cost_limits_are_explicit(source, changes, code):
    result, _ = run(source, **changes)
    assert result['status'] == 'budget_exceeded' and result['error']['code'] == code


@pytest.mark.parametrize('tool_source,limit,count_key', [
    ('observe()', 'max_observe_calls', 'observe_calls'),
    (MOVE, 'max_move_calls', 'move_calls'),
    ('point_at("task",0,0)', 'max_point_queries', 'point_queries'),
    ('observe()', 'max_tool_calls', 'tool_calls'),
])
def test_tool_budget_stops_before_excess_call_and_keeps_prior_effects(tool_source, limit, count_key):
    result, tools = run('\n'.join([tool_source]*3), **{limit:2})
    assert result['status'] == 'budget_exceeded' and result[count_key] == 2
    assert len(tools.calls) == len(result['trace']) == 2


def test_trace_reservation_prevents_unrecordable_tool_effect():
    result, tools = run('observe()', max_trace_bytes=10)
    assert result['status'] == 'budget_exceeded' and tools.calls == []


def test_tool_result_limit_is_explicit_after_an_effect_without_retry():
    calls=[]
    def invoke(name, args):
        calls.append(name)
        return {'large':'x'*100}
    result, _ = run('observe()\nobserve()', invoke=invoke, max_string_length=20)
    assert result['status'] == 'budget_exceeded' and calls == ['observe']
    assert result['trace'][0]['status'] == 'error' and 'result' not in result['trace'][0]


@pytest.mark.parametrize('bad', [float('nan'), float('inf'), (1,2), b'secret', object(), {1:'non-string-key'}, {'__class__':'private'}])
def test_non_json_tool_results_never_become_program_values(bad):
    result, _ = run('o=observe()\nstop()', invoke=lambda *_:bad)
    assert result['status'] == 'tool_error' and result['tool_calls'] == 1 and result['last_result'] is None


def test_cyclic_tool_result_is_rejected_without_recursing_forever():
    value={};value['self']=value
    result, _ = run('observe()', invoke=lambda *_:value)
    assert result['status'] == 'tool_error' and result['error']['code'] == 'cyclic_value'


def test_python_object_hooks_are_never_used_for_callback_values():
    class HostObject:
        def __deepcopy__(self, memo):
            pytest.fail('Host deepcopy hook executed')
        def __iter__(self):
            pytest.fail('Host iterator hook executed')
    result, _ = run('o=observe()', invoke=lambda *_:HostObject())
    assert result['status'] == 'tool_error'


def test_values_and_trace_are_copies_not_callback_or_variable_aliases():
    returned={'x':[1]};received=[]
    def invoke(name,args):
        received.append(args)
        if name=='observe': return returned
        args['wrist_delta_m'][0]=99
        return {'ok':True}
    result, _ = run('o=observe()\nx=o["x"][0]\nmove_geometry(wrist_delta_m=[x,0,0],duration_s=1)',invoke=invoke)
    returned['x'][0]=999
    assert result['status']=='completed'
    assert result['trace'][0]['result']=={'x':[1]}
    assert result['trace'][1]['arguments']['wrist_delta_m']==[1,0,0]


def test_callback_exceptions_do_not_leak_private_error_strings_or_retry():
    calls=[]
    def invoke(name,args):
        calls.append(name)
        raise RuntimeError('token_secret /datasets/private/object_pose')
    result, _ = run('observe()\n'+MOVE, invoke=invoke)
    assert result['status']=='tool_error' and calls==['observe']
    assert 'token_secret' not in json.dumps(result) and '/datasets' not in json.dumps(result)


@pytest.mark.parametrize('stop_tool', [True,False])
def test_stop_or_callback_terminal_halts_all_following_statements(stop_tool):
    calls=[]
    def invoke(name,args):
        calls.append(name)
        if not stop_tool: raise ProgramStop('episode_finished', {'outcome':'success'})
        return {'status':'cancelled'}
    source='stop(reason="agent_done")' if stop_tool else MOVE
    result, _ = run(source+'\nobserve()',invoke=invoke)
    assert result['status']=='stopped' and len(calls)==1
    assert result['trace'][0]['status']=='stopped'
    assert result['stop_reason']==('agent_done' if stop_tool else 'episode_finished')


def test_invalid_callback_stop_reason_is_not_silently_treated_as_completion():
    def invoke(*_): raise ProgramStop(None, {'ok':True})
    result, _ = run('observe()\nobserve()',invoke=invoke)
    assert result['status']=='tool_error' and result['tool_calls']==1


def test_callback_wall_timeout_detected_after_return_never_runs_next_tool(monkeypatch):
    import agent.model_client.program as module
    now=[0.]
    monkeypatch.setattr(module.time,'monotonic',lambda:now[0])
    calls=[]
    def invoke(name,args):
        calls.append(name);now[0]=2.;return {'status':'completed'}
    result, _ = run(MOVE+'\nobserve()', invoke=invoke,max_runtime_s=1.)
    assert result['status']=='budget_exceeded' and result['error']['code']=='program_wall_limit'
    assert calls==['move_geometry']


def test_fresh_execution_has_no_variables_or_counts_from_previous_program():
    first, _ = run('x=1\nobserve()')
    second, tools = run('point_at("task",x,0)')
    assert first['status']=='completed'
    assert second['status']=='runtime_error' and second['tool_calls']==0 and tools.calls==[]


@pytest.mark.parametrize('field,value', [('max_fuel',0),('max_fuel',True),('max_number_abs',float('inf')),
                                       ('max_runtime_s',float('nan')),('max_runtime_s',-1)])
def test_limit_configuration_is_strict(field,value):
    with pytest.raises(ValueError): replace(ProgramLimits(),**{field:value})
