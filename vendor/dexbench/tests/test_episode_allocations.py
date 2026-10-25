"""Published mechanical budgets must pass both Agent and service admission."""
from fastapi.testclient import TestClient
import pytest

from agent.model_client.bench_http import BenchHTTPClient
from agent.model_client.function_runner import RunConfig, run
from bench.protocol import Budgets
from bench.runtime.server import create_app
from bench.runtime.service import ServiceError
from test_agent_service import runner, frozen_inputs, session_for
from test_function_runner import ScriptedModel
from test_scene_function_runner import mechanical_service


@pytest.mark.parametrize('task_id', ['shaft_spacer_assembly', 'fastener_assembly'])
def test_frozen_recommended_budgets_reach_real_public_episode(tmp_path, task_id):
    api, workers, verified = mechanical_service(tmp_path, task_id)
    budgets = verified['task']['recommended_budgets']
    with TestClient(create_app(service=api)) as http:
        report = run(RunConfig('http://testserver', task_id, tmp_path / 'run',
            family='mechanical_assembly', seed=0, max_decisions=1,
            sim_budget=budgets['sim_time_s'], wall_budget=budgets['wall_time_s'],
            max_actions=budgets['max_actions']),
            bench=BenchHTTPClient('http://testserver', client=http),
            models={'single': ScriptedModel([{'stop': True}])})
    assert report['run_status'] == 'completed', report
    assert report['task_setup_error'] is None
    actual = report['result']['spec']['budgets']
    assert all(actual[key] == value for key, value in budgets.items())
    assert len(workers) == 1 and workers[0].closed


@pytest.mark.parametrize('field,value', [
    ('sim_time_s', 150.001), ('wall_time_s', 2400.001), ('max_actions', 7001)])
def test_over_allocation_rejected_before_native_or_agent_output(runner, tmp_path, field, value):
    make, _, workers = runner
    with pytest.raises(ServiceError) as rejected:
        session_for(make(), budgets=Budgets(**{field: value}))
    assert rejected.value.status == 422
    assert not workers
    option = {'sim_time_s': 'sim_budget', 'wall_time_s': 'wall_budget',
              'max_actions': 'max_actions'}[field]
    output = tmp_path / 'not_created'
    with pytest.raises(ValueError):
        run(RunConfig('http://unused', 'fastener_assembly', output, **{option: value}))
    assert not output.exists()
