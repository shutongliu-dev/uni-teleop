"""Actual auto-selection runner/HTTP path for the three mechanism tasks.

Models and native workers are CPU doubles; task releases, Agent selection,
controller descriptors, public function schemas and HTTP service are real.
These checks establish interface behavior, not physical manipulation success.
"""
from copy import deepcopy
import json
from pathlib import Path
import sys

from fastapi.testclient import TestClient
import pytest

from agent.model_client.bench_http import BenchHTTPClient
from agent.model_client.function_runner import RunConfig, run
from agent.model_client.scene_tasks import select_task
from bench.isaacsim.release import digest, load_evaluator, task_inputs
from bench.paths import load_json
from bench.runtime.server import create_app
from test_agent_service import PRIVATE_MARKER
from test_function_runner import ScriptedModel
from test_scene_function_runner import mechanical_service
import test_scene_function_runner as scene_runner_tests


TASKS = (
    ('living_room', 'task5', 'storage_box'),
    ('study', 'task6', 'reading_stand'),
    ('laboratory', 'task6', 'tube_clamp_transfer'),
)
CONFIGURATIONS = ('single_arm', 'dual_same_side', 'dual_opposed')


@pytest.fixture(autouse=True)
def storage_worker_reset(monkeypatch):
    """Supply a declared CPU baseline, never physical evidence or task credit.

    Storage's scorer deliberately has no result before its first observation;
    the older common SceneWorker fixture assumes inspect_conditions is a dict.
    Only this test double receives a closed, stationary reset frame derived
    from the frozen contract. Native workers acquire their own actual readback.
    """
    original = scene_runner_tests.reset_score

    def reset_score(release, task):
        if task['id'] != 'storage_box':
            return original(release, task)
        evaluator_type = load_evaluator(release)
        observer_type = sys.modules[evaluator_type.__module__].NativeTaskObserver
        observer = observer_type(task)
        frame = {'time_s': 0., 'native_time_s': 0., 'native_step': 0,
                 'physics_dt_s': .0005, 'baseline': True, 'contacts': [],
                 'objects': {name: {'position_m': position, 'quaternion_wxyz': [1., 0., 0., 0.],
                                    'linear_velocity_m_s': [0., 0., 0.],
                                    'angular_velocity_rad_s': [0., 0., 0.]}
                             for name, position in task['evaluation']['initial_positions_m'].items()}}
        observer.on_frame(frame)
        state = observer.on_observe()
        assert state['complete'] and not state['errors']
        score = evaluator_type(task).update({'time_s': 0., 'task_state': state})
        assert not score['success'] and not score['terminal']
        assert score['reason'] == 'reset_baseline_not_scored'
        return score

    monkeypatch.setattr(scene_runner_tests, 'reset_score', reset_score)


@pytest.mark.parametrize('family,alias,task_id', TASKS)
@pytest.mark.parametrize('configuration_id', CONFIGURATIONS)
def test_auto_runner_selects_and_keeps_mechanism_configuration(
        tmp_path, family, alias, task_id, configuration_id):
    api, workers, _ = mechanical_service(tmp_path, task_id, family)
    selection = select_task(alias, family)
    assert selection == select_task(task_id, family)
    assert selection.session_kind == 'single_agent' and selection.players == ('single',)
    options = selection.robot_configurations
    reference = options[configuration_id]['source_release']
    assert digest(Path(reference['path'])/'release_manifest.json') == reference['manifest_sha256']
    expected_task = task_inputs(Path(reference['path']), task_id, seed=0)[2]
    dual = configuration_id != 'single_arm'
    expected_sides = {'left', 'right'} if dual else {'right'}
    events = []
    alternate_read = []
    output = tmp_path/'run'
    decisions = [
        {'configuration_id': configuration_id, 'reason': '按公开场景与任务选择这套安装。'},
        {'function': 'get_task', 'arguments': {}},
        {'function': 'hold', 'arguments': {'duration_s': .04}},
        {'stop': True},
    ]
    model = ScriptedModel(decisions)

    class RecordingBench(BenchHTTPClient):
        def create(self, request):
            events.append(('create', request['configuration_id']))
            assert len(model.turns) == 1 and not workers
            assert request['configuration_id'] == configuration_id
            return super().create(request)

    with TestClient(create_app(service=api)) as http:
        bench = RecordingBench('http://testserver', client=http)
        original_generate = model.generate

        def generate(prompt, turns, **kwargs):
            context = json.loads(turns[-1]['content'][0]['text'])
            if not model.turns:
                events.append(('model_choose', configuration_id))
                assert not workers and not api.sessions
                assert context['task_id'] == task_id
                assert set(context['robot_configurations']) == set(CONFIGURATIONS)
                assert context['scene_description_zh']
                assert 'spec' not in context
                for identifier, option in context['robot_configurations'].items():
                    assert option['task_release_sha256'] == options[identifier]['source_release']['manifest_sha256']
                    assert option['robot_requirements']['articulations'] == (1 if identifier == 'single_arm' else 2)
                    placements = option['layout']['placements']
                    assert set(placements) == ({'right'} if identifier == 'single_arm' else {'left', 'right'})
                    assert all(value['hand_chirality'] == side for side, value in placements.items())
            else:
                events.append(('model_control', configuration_id))
                assert len(workers) == len(api.sessions) == 1
                assert context['spec']['configuration_id'] == configuration_id
                assert context['spec']['task_release_sha256'] == reference['manifest_sha256']
                guide = context['robot']
                assert set(guide['hands']) == expected_sides
                assert len(guide['arm_joints']) == (14 if dual else 7)
                assert len(guide['hand_joints']) == (44 if dual else 22)
                measured = context['state']['observation']['proprioception']['joint_positions_rad']
                assert set(measured) == set(guide['arm_joints']) | set(guide['hand_joints'])
                if dual:
                    for side in expected_sides:
                        assert sum(name.startswith(side+'_') for name in guide['hand_joints']) == 22
                        assert sum(name.startswith('fr3_'+side+'_') for name in guide['arm_joints']) == 7
                assert context['task']['public_rules'] == expected_task['public_rules']['referee']
                assert context['task']['public_rules_sha256'] == expected_task['public_referee_contract_sha256']
                assert not {'referee', 'scoring', 'episode'} & context['task'].keys()
                assert not {'player', 'board_game', 'turn_game'} & context['state'].keys()
                tools = {row['function'] for row in context['tools']}
                assert {'get_task', 'hold', 'move_wrist', 'set_hand_targets'} <= tools
                if not alternate_read:
                    # Discovery of another installation after creation must not
                    # rewrite the active session, its controller or release.
                    session = api.sessions[context['spec']['episode_id']]
                    before = session.spec.model_dump(mode='json')
                    other = 'single_arm' if dual else 'dual_same_side'
                    other_task, = bench.list_tasks(task_id=task_id, configuration_id=other)
                    bench.list_policies(task_id=task_id, configuration_id=other)
                    bench.list_interfaces(task_id=task_id, configuration_id=other)
                    assert other_task['configuration_id'] == other
                    assert other_task['task_release_sha256'] != reference['manifest_sha256']
                    assert session.spec.model_dump(mode='json') == before
                    assert len(workers) == 1
                    alternate_read.append(other)
                if len(model.turns) == 2:
                    assert context['last_feedback']['function'] == 'get_task'
                    assert context['last_feedback']['ok']
                    assert context['last_feedback']['result']['public_rules'] == expected_task['public_rules']['referee']
            return original_generate(prompt, turns, **kwargs)

        model.generate = generate
        report = run(RunConfig('http://testserver', alias, output, family=family,
                              configuration='auto', seed=0, max_decisions=5,
                              max_actions=4, max_status_queries=100000),
                     bench=bench, models={'single': model})

    assert report['error_type'] is None, report
    assert report['task_setup_error'] is None and report['ready_validated']
    assert not report['board_game'] and not report['turn_game']
    assert report['termination'] == 'model_requested_stop'
    assert report['pre_episode_model_calls'] == 1
    assert report['decisions'] == {'single': 4}
    assert report['function_calls'] == {'single': 2}
    assert report['physical_attempts'] == {'single': 1}
    assert events[:2] == [('model_choose', configuration_id), ('create', configuration_id)]
    assert len(alternate_read) == 1
    request = load_json(output/'create_request.json')
    ready = load_json(output/'episode_ready.json')
    assert request['configuration_id'] == ready['spec']['configuration_id'] == configuration_id
    assert request['robot_id'] == ('dual_fr3_sharpa_w01' if dual else 'sharpa_w01_right')
    assert ready['spec']['task_release_sha256'] == reference['manifest_sha256']
    assert ready['spec']['budgets'] == request['budgets']
    assert request['budgets']['sim_time_s'] == 120.
    assert request['budgets']['max_model_calls'] == 5 and request['budgets']['max_actions'] == 4
    assert len(ready['capabilities']['arm_joints']) + len(ready['capabilities']['hand_joints']) == (58 if dual else 29)
    assert report['result']['spec']['configuration_id'] == configuration_id
    assert report['result']['spec']['task_release_sha256'] == reference['manifest_sha256']
    assert len(workers) == 1 and workers[0].closed
    assert any(operation == 'step' for operation, _ in workers[0].calls)
    assert PRIVATE_MARKER not in json.dumps(model.turns)
    assert report['result']['outcome'] != 'success'


@pytest.mark.parametrize('failure', ['invalid_choice', 'stale_public_pin'])
def test_mechanism_auto_runner_rejects_bad_choice_or_pin_before_worker(tmp_path, failure):
    api, workers, _ = mechanical_service(tmp_path, 'storage_box', 'living_room')
    model = ScriptedModel([{'configuration_id': 'not_published', 'reason': '不可用的构型。'}])

    class StaleDiscoveryBench(BenchHTTPClient):
        def list_tasks(self, task_id=None, configuration_id=None):
            rows = deepcopy(super().list_tasks(task_id, configuration_id))
            if failure == 'stale_public_pin':
                rows[0]['robot_configurations']['dual_opposed']['task_release_sha256'] = 'f'*64
            return rows

    with TestClient(create_app(service=api)) as http:
        report = run(RunConfig('http://testserver', 'task5', tmp_path/'run',
                              family='living_room', configuration='auto'),
                     bench=StaleDiscoveryBench('http://testserver', client=http),
                     models={'single': model})
    assert report['task_setup_error']['code'] == (
        'configuration_choice_invalid' if failure == 'invalid_choice' else 'configuration_contract_mismatch')
    assert len(model.turns) == (1 if failure == 'invalid_choice' else 0)
    assert report['episode_id'] is None and not workers and not api.sessions
    assert not (tmp_path/'run/create_request.json').exists()
