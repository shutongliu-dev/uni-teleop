"""No active time cap is an explicit null, with count/fuel limits retained."""
from argparse import ArgumentTypeError
import hashlib
from io import BytesIO
import json
from types import SimpleNamespace

import httpx
import numpy as np
from PIL import Image
import pytest

from agent.model_client import runner, suite
from agent.model_client.bench_http import BenchHTTPClient, BenchHTTPError
from agent.model_client.interaction import InteractionSession
from agent.model_client.program import ProgramLimits, execute_program
from agent.model_client.time_limits import optional_seconds
from agent.model_client.transport import ModelClient
from bench.paths import load_json
from test_model_transport import completion, config, generate
from test_model_suite import environment, fake_runner
from test_online_model_runner import setup_loop
from test_multimode_runner import loop


@pytest.mark.parametrize('text', ['none', 'null', 'disabled', 'NONE'])
def test_cli_explicit_disabled_values(text):
    assert optional_seconds(text) is None


@pytest.mark.parametrize('text', ['0', '-1', 'nan', 'inf', 'true'])
def test_cli_does_not_use_fake_unlimited_numbers(text):
    with pytest.raises(ArgumentTypeError):
        optional_seconds(text)


def test_model_without_timeout_disables_all_httpx_timeouts_and_keeps_max(monkeypatch):
    clock = [0.]
    monkeypatch.setattr('agent.model_client.transport.time.monotonic', lambda: clock[0])
    def handler(request):
        assert all(value is None for value in request.extensions['timeout'].values())
        assert json.loads(request.content)['reasoning_effort'] == 'max'
        clock[0] = 5000.
        return httpx.Response(200, json=completion())
    with ModelClient(config(), 'gpt-6-astra', timeout_s=None, max_retries=0,
                     transport=httpx.MockTransport(handler)) as client:
        response = generate(client)
        assert client.public_metadata()['timeout_s'] is None
    assert response.duration_s == 5000. and response.attempts == 1


def test_optional_model_timeout_still_obeys_enabled_per_call_limit():
    def handler(request):
        assert 0 < request.extensions['timeout']['read'] <= 3.
        return httpx.Response(200, json=completion())
    with ModelClient(config(), 'gpt-6-astra', timeout_s=None,
                     transport=httpx.MockTransport(handler)) as client:
        assert generate(client, timeout_s=3.).attempts == 1


def test_unlimited_model_http_failure_is_reported_without_retry():
    calls = []
    def handler(request):
        calls.append(request)
        return httpx.Response(524, json={'error':'upstream timeout'})
    from agent.model_client.transport import ModelGatewayError
    with ModelClient(config(), 'gpt-6-astra', timeout_s=None, max_retries=0,
                     transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(ModelGatewayError) as caught:
            generate(client)
    assert caught.value.public_metadata()['http_status'] == 524 and len(calls) == 1


def test_http_polling_with_no_deadline_survives_arbitrary_elapsed_time(monkeypatch):
    clock, calls = [0.], []
    monkeypatch.setattr('agent.model_client.bench_http.time.monotonic', lambda: clock[0])
    monkeypatch.setattr('agent.model_client.bench_http.time.sleep', lambda seconds: None)
    def handler(request):
        calls.append(request)
        assert all(value is None for value in request.extensions['timeout'].values())
        clock[0] += 10000.
        return httpx.Response(200, json={'action_id':'a1',
            'status':'running' if len(calls) < 3 else 'completed'})
    client = BenchHTTPClient('http://bench.test', timeout_s=None,
                            client=httpx.Client(transport=httpx.MockTransport(handler)))
    assert client.wait_action('ep1', 'a1', timeout_s=None)['status'] == 'completed'
    assert len(calls) == 3 and client._remaining_timeout() is None


def test_no_http_cap_does_not_disable_explicit_host_deadline(monkeypatch):
    monkeypatch.setattr('agent.model_client.bench_http.time.monotonic', lambda: 10.)
    client = BenchHTTPClient('http://bench.test', timeout_s=None,
        client=httpx.Client(transport=httpx.MockTransport(lambda request: pytest.fail('expired'))))
    client.deadline = 9.
    with pytest.raises(BenchHTTPError, match='deadline'):
        client.list_tasks()
    with pytest.raises(BenchHTTPError, match='deadline'):
        client._remaining_timeout()


def test_rgb_and_depth_streams_do_not_restore_a_default_timeout():
    rgb_stream, depth_stream = BytesIO(), BytesIO()
    Image.new('RGB', (8, 6)).save(rgb_stream, format='PNG')
    np.save(depth_stream, np.ones((6, 8), dtype=np.float32), allow_pickle=False)
    png, npy = rgb_stream.getvalue(), depth_stream.getvalue()
    def handler(request):
        assert all(value is None for value in request.extensions['timeout'].values())
        is_depth = request.url.path == '/depth'
        return httpx.Response(200, content=npy if is_depth else png,
            headers={'content-type':'application/octet-stream' if is_depth else 'image/png'})
    client = BenchHTTPClient('http://bench.test', timeout_s=None,
                            client=httpx.Client(transport=httpx.MockTransport(handler)))
    assert client._png('/rgb', expected_sha=hashlib.sha256(png).hexdigest(), width=8, height=6) == png
    raw, depth = client._depth_npy('/depth', expected_sha=hashlib.sha256(npy).hexdigest(), width=8, height=6)
    assert raw == npy and np.all(depth == 1.)


def test_unlimited_program_waits_but_keeps_move_and_ast_fuel_limits(monkeypatch):
    clock, calls = [0.], []
    monkeypatch.setattr('agent.model_client.program.time.monotonic', lambda: clock[0])
    def invoke(name, arguments):
        calls.append(name)
        clock[0] += 5000.
        return {'status':'completed'}
    code = '\n'.join(['move_geometry(wrist_delta_m=[0,0,0.01],duration_s=1.0)']*4)
    result = execute_program(code, invoke, limits=ProgramLimits(max_runtime_s=None,
        max_move_calls=3, max_tool_calls=8))
    assert result['status'] == 'budget_exceeded' and result['move_calls'] == 3
    assert calls == ['move_geometry']*3
    result = execute_program('x=1\nx=2\nx=3', invoke,
                             limits=ProgramLimits(max_runtime_s=None, max_fuel=2))
    assert result['error']['code'] == 'fuel_limit'


def test_interaction_optional_deadlines_preserve_outer_host_constraint(monkeypatch):
    monkeypatch.setattr('agent.model_client.interaction.time.monotonic', lambda: 10.)
    bench = SimpleNamespace(deadline=None)
    session = InteractionSession(bench, 'ep', {}, {}, {}, None, control_mode='fingertip',
        visual_input='rgbd', tactile_input='on', limits=None, deadline=None, max_actions=48)
    assert session.remaining() is None
    assert session.http(lambda: bench.deadline) is None
    bench.deadline = 20.
    assert session.http(lambda: bench.deadline) == 20.
    session.program_deadline = 15.
    assert session.remaining() == 5.
    assert session.http(lambda: bench.deadline) == 15.
    assert bench.deadline == 20.


@pytest.mark.parametrize('control', ['joints', 'fingertip', 'tools', 'code'])
def test_full_online_loop_null_limits_keeps_count_budget(monkeypatch, tmp_path, control):
    if control in ('tools', 'code'):
        response = ('move_geometry(wrist_delta_m=[0,0,0.01],duration_s=3.0)' if control == 'code'
                    else {'tool':'move_geometry','arguments':{'wrist_delta_m':[0,0,.01], 'duration_s':3.}})
        args, bench, model = loop(monkeypatch, tmp_path, [response], mode=control)
    else:
        response = ({'action':'move_geometry','wrist_delta_m':[0,0,.01], 'duration_s':3.}
                    if control == 'fingertip' else
                    {'action':'joint_delta','delta_rad':{'arm':.01},'duration_s':.4})
        args, bench, model = setup_loop(monkeypatch, tmp_path, [response], fingertip=control=='fingertip')
    for name in ('sim_budget','wall_budget','model_timeout','http_timeout','program_timeout'):
        setattr(args, name, None)
    create = bench.create
    def create_with_null_budgets(request):
        ready = create(request)
        ready['spec']['budgets'].update(request['budgets'])
        return ready
    bench.create = create_with_null_budgets
    report = runner.run(args)
    assert report['error_type'] is None
    assert report['termination'] == 'client_decision_budget'
    assert len(bench.commands) == len(model.requests) == 1
    assert model.requests[0][2]['timeout_s'] is None
    settings = load_json(args.output/'agent_config.json')
    assert settings['schema'] == 'online_visual_agent.v11'
    assert settings['budgets']['sim_time_s'] is settings['budgets']['wall_time_s'] is None
    assert settings['program_limits']['max_runtime_s'] is None


@pytest.mark.parametrize('outcome', ['success', 'cancelled'])
def test_null_matrix_full_reporting_preserves_unknown_penalty(environment, outcome):
    _, _, make = environment
    path, plan = make(**{name:None for name in
        ('sim_budget','wall_budget','model_timeout','http_timeout','program_timeout')})
    execute, calls = fake_runner(environment, outcome=outcome, elapsed_sim=500., elapsed_wall=5000.)
    report = suite.run_plan(path, run_one=execute)
    assert len(calls) == 4 and report['all_slots_valid']
    assert all(args.program_timeout is None for args in calls)
    for row in report['slots']:
        assert row['task_success'] == (outcome == 'success')
        assert row['elapsed_wall_s'] == 5000.
        assert row['completion_wall_s'] == (5000. if outcome == 'success' else None)
        assert row['diagnostic_penalized_wall_cost_s'] is None
        assert row['diagnostic_penalized_sim_cost_s'] is None
    for group in report['cohorts']:
        assert group['actual_elapsed_wall_s'] == 5000.
        assert group['diagnostic_penalized_wall_cost_s'] is None
        assert group['diagnostic_penalized_sim_cost_s'] is None
        assert group['diagnostic_success_per_penalized_wall_second'] is None
        assert group['diagnostic_success_per_penalized_sim_second'] is None
    assert 'Infinity' not in json.dumps(report, allow_nan=False)


def test_null_matrix_missing_slots_never_zero_penalty(environment):
    _, _, make = environment
    path, _ = make(sim_budget=None, wall_budget=None)
    report = suite.report_plan(path)
    assert all(row['diagnostic_penalized_sim_cost_s'] is None for row in report['slots'])
    assert all(row['diagnostic_penalized_wall_cost_s'] is None for row in report['cohorts'])


@pytest.mark.parametrize('field', ['sim_budget','wall_budget','model_timeout','http_timeout','program_timeout'])
def test_matrix_requires_real_null_not_text(field):
    with pytest.raises(ValueError):
        suite.Matrix.model_validate({field:'null'})


@pytest.mark.parametrize('field, replacement', [
    ('sim_time_s', 120.), ('wall_time_s', 1800.), ('max_actions', 2),
    ('max_model_calls', 2), ('max_action_sim_s', .5), ('max_model_calls', True)])
@pytest.mark.parametrize('missing', [False, True])
def test_ready_budget_mismatch_stops_before_first_model_request(monkeypatch, tmp_path, field, replacement, missing):
    args, bench, model = setup_loop(monkeypatch, tmp_path, [{'action':'stop'}])
    args.sim_budget = args.wall_budget = args.model_timeout = args.http_timeout = args.program_timeout = None
    create = bench.create
    def wrong_budget(request):
        ready = create(request)
        if missing:
            ready['spec']['budgets'].pop(field)
        else:
            ready['spec']['budgets'][field] = replacement
        return ready
    bench.create = wrong_budget
    report = runner.run(args)
    assert report['error_type'] == 'ValueError'
    assert not model.requests and not bench.commands
    assert report['model_call_attempts'] == 0
    assert bench.finished and bench.closed and model.closed


@pytest.mark.parametrize('dimension', ['sim_budget', 'wall_budget'])
def test_single_disabled_dimension_keeps_other_penalty(environment, dimension):
    _, _, make = environment
    path, _ = make(**{dimension:None})
    execute, _ = fake_runner(environment, outcome='success')
    report = suite.run_plan(path, run_one=execute)
    disabled = 'sim' if dimension == 'sim_budget' else 'wall'
    enabled = 'wall' if disabled == 'sim' else 'sim'
    for row in report['cohorts']:
        assert row[f'diagnostic_penalized_{disabled}_cost_s'] is None
        assert row[f'diagnostic_success_per_penalized_{disabled}_second'] is None
        assert row[f'diagnostic_penalized_{enabled}_cost_s'] > 0
        assert row[f'diagnostic_success_per_penalized_{enabled}_second'] > 0
