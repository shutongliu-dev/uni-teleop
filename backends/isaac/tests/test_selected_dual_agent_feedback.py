"""Selected two-hand installations preserve commands and different public goals.

Frozen tasks and the full Agent/HTTP path are real; native physics alone uses a
CPU double. This checks transport and feedback, not manipulation qualification.
"""
from copy import deepcopy
import json

from fastapi.testclient import TestClient
import pytest

from agent.model_client.bench_http import BenchHTTPClient
from agent.model_client.function_runner import RunConfig, run
from bench.control.factory import public_robot_geometry, robot_configuration
from bench.isaacsim.release import task_inputs
from bench.paths import load_json
from bench.runtime.server import create_app
from bench.runtime.worker import public_task_projection
from scenes.runtime import terminal_scoring_contract
from scenes.workrooms.catalog import task_definition
from test_agent_service import PRIVATE_MARKER
from test_bench_function_http import commands
from test_function_runner import ScriptedModel
from test_scene_function_runner import mechanical_service


@pytest.mark.parametrize('family,task_id,configuration', [
    ('card_room', 'task0', 'dual_same_side'),
    ('laboratory', 'solid_weighing', 'dual_opposed'),
    ('living_room', 'tower_building', 'dual_same_side'),
    ('living_room', 'tower_building', 'dual_opposed'),
])
def test_selected_dual_action_keeps_both_hands_and_returns_acknowledged_targets(
        tmp_path, family, task_id, configuration):
    definition = task_definition(family, task_id)
    selected = definition['robot_configurations'][configuration]['source_release']
    _, scene, _, initial = task_inputs(selected['path'], task_id, seed=7)
    geometry = public_robot_geometry(robot_configuration(scene))
    positions = deepcopy(initial['joint_positions_rad'])
    targets = {side: {'wrist': deepcopy(value['wrist']), 'fingertip_frame_id': side + '_hand_flange',
                      'position_tolerance_m': .001}
               for side, value in geometry.public_hand_geometries(positions).items()}
    for target in targets.values():
        target['wrist']['position_m'][2] += .005
    api, workers, _ = mechanical_service(tmp_path, task_id, family)
    original_factory = api.worker_factory

    def factory(**kwargs):
        worker = original_factory(**kwargs)
        _, _, task, _ = task_inputs(kwargs['release'], kwargs['task_id'], seed=kwargs['seed'])
        # Include seeded task0's dedicated public projection; the generic scene
        # fixture otherwise covers only referee-style public task contracts.
        worker.initial['task'] = public_task_projection(task)
        return worker

    api.worker_factory = factory
    model = ScriptedModel([
        {'configuration_id': configuration, 'reason': '选择公开的双手安装'},
        {'function': 'get_robot' if task_id == 'task0' else 'get_task', 'arguments': {}},
        {'function': 'move_geometry', 'arguments': {
            'hand_geometries': targets, 'timeout_sim_s': 1.2}},
        {'function': 'hold', 'arguments': {'duration_s': .04}}, {'stop': True}])
    with TestClient(create_app(service=api)) as http:
        report = run(RunConfig('http://testserver', task_id, tmp_path / 'run', family=family,
                              seed=7, max_decisions=5, max_status_queries=100000),
                     bench=BenchHTTPClient('http://testserver', client=http), models={'single': model})
    assert report['error_type'] is None, report
    assert report['ready_validated'] and report['decisions'] == {'single': 5}
    assert report['physical_attempts'] == {'single': 2}
    assert report['result']['spec']['configuration_id'] == configuration
    assert report['result']['spec']['task_release_sha256'] == selected['manifest_sha256']
    assert len(workers) == 1 and workers[0].closed
    executed = commands(workers[0])
    assert len(executed) > 2  # Both-hand geometry motion plus two hold windows.
    final_targets = executed[-1]['joint_positions_rad']
    assert len(final_targets) == 58
    for hand in geometry.sides.values():
        assert max(abs(final_targets[name] - positions[name]) for name in hand.arm.names) > 1.e-6
    movement = load_json(tmp_path / 'run/single/decisions/0002/execution.json')
    assert movement['ok'] and movement['result']['status'] == 'completed'
    context = json.loads(model.turns[-1][-1]['content'][0]['text'])
    assert set(context['robot']['hands']) == {'left', 'right'}
    assert context['state']['observation']['proprioception']['commanded_joint_positions_rad'] == final_targets
    assert context['last_feedback']['ok'] is True
    assert PRIVATE_MARKER not in json.dumps(model.turns)
    if task_id == 'task0':
        assert context['task']['episode_seed'] == 7
        assert len(context['task']['goal']['target_cells']) == 3
    if task_id == 'tower_building':
        _, _, base_task, _ = task_inputs(definition['source_release']['path'], task_id, seed=7)
        expected = terminal_scoring_contract(base_task)
        assert context['task']['scoring_contract'] == expected
        readback = load_json(tmp_path / 'run/single/decisions/0001/execution.json')
        assert readback['result']['scoring_contract'] == expected
        assert report['task_score'] == 0. and report['task_score_valid']
        assert report['result']['scoring']['score_unit'] == expected['score_unit']
        assert report['task_success'] is None and report['task_outcome'] == 'unknown'
