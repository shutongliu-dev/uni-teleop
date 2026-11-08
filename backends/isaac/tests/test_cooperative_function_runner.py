"""Same-agent two-robot contracts through the actual HTTP/Agent stack, CPU double."""
from copy import deepcopy
import json

from fastapi.testclient import TestClient

from agent.model_client.bench_http import BenchHTTPClient
from agent.model_client.function_runner import RunConfig, run
from agent.model_client.scene_tasks import select_task
from bench.control.factory import public_robot_geometry, robot_configuration
from bench.paths import load_json
from bench.runtime.server import create_app
from test_agent_service import PRIVATE_MARKER
from test_function_runner import ScriptedModel
from test_scene_function_runner import mechanical_service


def test_one_agent_can_target_both_frozen_instances_in_one_ordinary_action(tmp_path):
    api, workers, verified = mechanical_service(tmp_path, 'cooperative_insertion')
    geometry = public_robot_geometry(robot_configuration(verified['scene']))
    q = verified['initial_state']['joint_positions_rad']
    public = geometry.public_hand_geometries(q)
    targets = {name: {'wrist': value['wrist'], 'fingertip_frame_id': name+'_hand_flange'}
               for name, value in public.items()}
    model = ScriptedModel([
        {'function': 'get_task', 'arguments': {}},
        {'function': 'query_ik', 'arguments': {'hand_geometries': targets}},
        {'function': 'move_geometry', 'arguments': {'hand_geometries': targets, 'timeout_sim_s': .4}},
        {'function': 'hold', 'arguments': {'duration_s': .04}}, {'stop': True},
    ])
    with TestClient(create_app(service=api)) as http:
        bench = BenchHTTPClient('http://testserver', client=http)
        report = run(RunConfig('http://testserver', 'task10', tmp_path/'run',
            family='mechanical_assembly', sim_budget=None, max_decisions=6,
            max_status_queries=100000), bench=bench, models={'single': model})
        assert report['error_type'] is None, report
        create = load_json(tmp_path/'run/create_request.json')
        for route in ('/board-games', '/turn-games'):
            assert http.post(route, json=create).status_code == 403
        wrong = deepcopy(create)
        wrong['robot_id'] = 'sharpa_w01_right'
        assert http.post('/episodes', json=wrong).status_code == 422
        assert len(workers) == 1
    assert create['robot_id'] == 'dual_fr3_sharpa_w01'
    assert create['policy_id'] == 'classical_fingertip_bimanual'
    assert create['budgets']['sim_time_s'] is None
    assert report['task_selection']['players'] == ['single']
    assert report['task_selection']['session_kind'] == 'single_agent'
    assert report['board_game'] is False and report['turn_game'] is False
    assert report['physical_attempts'] == {'single': 2}
    assert report['result']['outcome'] == 'cancelled'
    assert PRIVATE_MARKER not in json.dumps(model.turns)
    context = json.loads(model.turns[0][-1]['content'][0]['text'])
    assert set(context['robot']['hands']) == {'left', 'right'}
    for index in (2, 3):
        context = json.loads(model.turns[index][-1]['content'][0]['text'])
        assert context['last_feedback']['ok'] is True
    assert workers[0].closed


def test_cooperative_aliases_keep_single_context_and_catalog_pins():
    expected = select_task('cooperative_insertion', family='mechanical_assembly')
    assert expected == select_task('task10', family='mechanical_assembly')
    assert expected == select_task('L10', family='mechanical_assembly')
    assert expected.robot_requirements['hand_id'] == 'dual_fr3_sharpa_w01'
    assert expected.robot_requirements['articulations'] == 2
    assert expected.players == ('single',) and expected.practice_rsi_allowed
