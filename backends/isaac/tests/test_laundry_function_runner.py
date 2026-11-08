"""Laundry uses the shared one-Agent left/right robot and function contracts.

These CPU transport checks use the actual frozen releases and public clients.
The native worker alone is replaced; they do not establish cloth grasp success.
"""
from copy import deepcopy
import json

from fastapi.testclient import TestClient
import pytest

from agent.model_client.bench_http import BenchHTTPClient
from agent.model_client.function_runner import RunConfig, learning_policy, run
from agent.model_client.scene_tasks import TaskSetupError, list_scene_tasks, select_task
from bench.control.factory import public_robot_geometry, robot_configuration
from bench.paths import load_json
from bench.runtime.server import create_app
from scenes.workrooms.parameterized.pairs import PROFILES
from test_agent_service import PRIVATE_MARKER
from test_function_runner import ScriptedModel
from test_scene_function_runner import mechanical_service


TASKS = ((1, 'towel_flattening'), (2, 'towel_folding'), (3, 'towel_stacking'),
         (4, 'shirt_folding'), (5, 'trousers_folding'))


@pytest.mark.parametrize('robot_profile', [None, 'original', 'l_mount'])
@pytest.mark.parametrize('number,canonical', TASKS)
def test_laundry_robot_aliases_bind_one_agent_to_both_official_hands(number, canonical, robot_profile):
    selected = select_task(f'task{number}', family='laundry_room', robot_profile=robot_profile)
    assert selected == select_task(canonical, family='laundry_room', robot_profile=robot_profile)
    selected.require_runnable()
    assert selected.session_kind == 'single_agent' and selected.players == ('single',)
    assert selected.pin_release and selected.robot_requirements['hand_id'] == 'dual_fr3_sharpa_w01'
    assert selected.robot_requirements['articulations'] == 2
    assert set(selected.robot_requirements['active_sides']) == {'left', 'right'}
    # dexbench: the catalog default is the original robot; the L-mount robot stays selectable.
    assert selected.robot_requirements['assembly_version'] == PROFILES[robot_profile or 'original']
    # Task2/3 are family-local numbers, not the frozen-learning card games.
    policy = learning_policy(f'task{number}', family='laundry_room')
    assert policy['practice_rsi_allowed'] and not policy['frozen_skills_required']
    assert select_task(canonical).family == 'laundry_room'
    assert not select_task(canonical).pin_release


def test_all_five_tasks_are_one_agent_dual_robot_tasks():
    rows = {row['task_id']: row for row in list_scene_tasks('laundry_room')}
    assert set(rows) == {name for _, name in TASKS}
    assert all(rows[name]['runnable'] and rows[name]['session_kind'] == 'single_agent'
               for _, name in TASKS)


@pytest.mark.parametrize('number,canonical', TASKS)
def test_laundry_ordinary_agent_reads_rules_and_targets_both_hands(tmp_path, number, canonical):
    service, workers, verified = mechanical_service(tmp_path, canonical, family='laundry_room')
    geometry = public_robot_geometry(robot_configuration(verified['scene']))
    positions = verified['initial_state']['joint_positions_rad']
    assert len(positions) == 58 and len(geometry.names) == 58
    hands = geometry.public_hand_geometries(positions)
    assert set(hands) == {'left', 'right'}
    targets = {side: {'wrist': value['wrist'], 'fingertip_frame_id': side+'_hand_flange'}
               for side, value in hands.items()}
    model = ScriptedModel([
        {'function': 'get_task', 'arguments': {}},
        {'function': 'get_hand_presets', 'arguments': {}},
        {'function': 'query_ik', 'arguments': {'hand_geometries': targets}},
        {'function': 'move_geometry', 'arguments': {'hand_geometries': targets, 'timeout_sim_s': .4}},
        {'function': 'hold', 'arguments': {'duration_s': .04}},
        {'stop': True},
    ])
    with TestClient(create_app(service=service)) as transport:
        client = BenchHTTPClient('http://testserver', client=transport)
        task, = client.list_tasks()
        assert task['task_kind'] == 'manipulation' and task['create_endpoint'] == 'POST /episodes'
        assert task['public_rules'] == verified['task']['public_rules']['referee']
        assert task['public_rules_sha256'] == verified['task']['public_referee_contract_sha256']
        result = run(RunConfig('http://testserver', f'task{number}', tmp_path/'run',
            family='laundry_room', configuration='dual_opposed', sim_budget=None, max_decisions=7,
            max_status_queries=100000), bench=client, models={'single': model})
        assert result['error_type'] is None, result
        create = load_json(tmp_path/'run/create_request.json')
        wrong_robot = deepcopy(create)
        wrong_robot['robot_id'] = 'sharpa_w01_right'
        assert transport.post('/episodes', json=wrong_robot).status_code == 422
        assert transport.post('/board-games', json=create).status_code == 403
        assert transport.post('/turn-games', json=create).status_code == 403
        assert len(workers) == 1
    assert create['robot_id'] == 'dual_fr3_sharpa_w01'
    assert create['policy_id'] == 'classical_fingertip_bimanual'
    assert create['budgets']['sim_time_s'] is None
    assert result['task_selection']['players'] == ['single']
    assert result['task_selection']['session_kind'] == 'single_agent'
    assert result['board_game'] is False and result['turn_game'] is False
    assert result['physical_attempts'] == {'single': 2}
    assert result['result']['outcome'] == 'cancelled'
    assert PRIVATE_MARKER not in json.dumps(model.turns)
    first_context = json.loads(model.turns[0][-1]['content'][0]['text'])
    assert set(first_context['robot']['hands']) == {'left', 'right'}
    for turn in model.turns[1:]:
        context = json.loads(turn[-1]['content'][0]['text'])
        assert context['last_feedback']['ok'] is True
    assert workers[0].closed
