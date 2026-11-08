"""Public discovery/creation bindings through HTTP, with CPU worker/model doubles."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

from fastapi.testclient import TestClient
import pytest

from agent.model_client.bench_http import BenchHTTPClient, BenchHTTPError
from agent.model_client.function_runner import RunConfig, run, select_task_controller
from agent.model_client.scene_tasks import select_task
from bench.paths import load_json
from bench.protocol.schemas import PolicyDescriptor
from bench.runtime.controllers import default_registry
from bench.runtime.server import create_app
from bench.runtime.service import AgentService
from scenes.workrooms.catalog import task_definition
from test_board_game_service import FakeBoardWorker
from test_function_runner import ScriptedModel
from test_scene_function_runner import mechanical_service


@pytest.mark.parametrize('field', ['rules', 'geometry'])
def test_board_discovery_drift_is_rejected_before_model_or_motion(tmp_path, monkeypatch, field):
    task_id = 'tic_tac_toe_3x3'
    release = Path(task_definition('card_room', task_id)['source_release']['path'])
    workers = []

    def factory(**kwargs):
        worker = FakeBoardWorker(**kwargs)
        workers.append(worker)
        return worker

    api = AgentService(tmp_path / 'episodes', release=release,
                       registry=default_registry(release), worker_factory=factory)
    models = {key: ScriptedModel([{'stop': True}]) for key in ('player_0', 'player_1')}
    with TestClient(create_app(service=api)) as http:
        bench = BenchHTTPClient('http://testserver', client=http)
        list_tasks = bench.list_tasks

        def stale_discovery():
            tasks = deepcopy(list_tasks())
            task = tasks[0]
            # Internally self-consistent stale discovery, unchanged real service.
            task[field]['win_length' if field == 'rules' else 'pitch_m'] += 1
            task['public_rules'][field] = deepcopy(task[field])
            task['public_rules_sha256'] = hashlib.sha256(json.dumps(
                task['public_rules'], sort_keys=True, separators=(',', ':'),
                ensure_ascii=True, allow_nan=False).encode()).hexdigest()
            return tasks

        monkeypatch.setattr(bench, 'list_tasks', stale_discovery)
        report = run(RunConfig('http://testserver', 'task1', tmp_path / 'run',
                              family='card_room', max_decisions=1), bench=bench, models=models)
    assert not report['ready_validated'], report
    assert report['run_status'] == 'infra_error'
    assert report['run_error']['error_category'] == 'invalid_response'
    assert not any(model.turns for model in models.values())
    assert report['decisions'] == report['physical_attempts'] == {}
    assert not report['result_valid_for_scoring']
    assert report['evaluation_player_outcomes'] == {'player_0': 'unknown', 'player_1': 'unknown'}
    assert len(workers) == 1 and workers[0].closed
    assert not any(op == 'step' for op, _ in workers[0].calls)
    assert report['finalization_error'] is None


def historical_cooperative_release(version):
    """Historical releases sit beside the canonical one that both profiles wrap."""
    roots = set()
    for profile in ('original', 'l_mount'):
        root = Path(task_definition('mechanical_assembly', 'cooperative_insertion',
                                    robot_profile=profile)['source_release']['path'])
        manifest = load_json(root / 'release_manifest.json')
        # Seed cases and the L-mount revision wrap the canonical family release.
        while manifest['schema'] in ('workroom_robot_revision_release.v1', 'workroom_randomized_release.v1'):
            root = Path(manifest['base_release']['path'])
            manifest = load_json(root / 'release_manifest.json')
        assert manifest['schema'] == 'mechanical_cooperative_task_release.v1'
        roots.add(root)
    current, = roots
    return current.parent / version


@pytest.mark.parametrize('version', ['v1', 'v3'])
def test_bare_cooperative_id_uses_the_published_historical_robot(tmp_path, version):
    task_id = 'cooperative_insertion'
    release = historical_cooperative_release(version)
    api, workers, _ = mechanical_service(tmp_path, task_id, release=release)
    model = ScriptedModel([{'function': 'get_task', 'arguments': {}}, {'stop': True}])
    with TestClient(create_app(service=api)) as http:
        bench = BenchHTTPClient('http://testserver', client=http)
        report = run(RunConfig('http://testserver', task_id, tmp_path / 'run',
                              sim_budget=None, max_decisions=2), bench=bench, models={'single': model})
    assert report['error_type'] is None, report
    assert report['ready_validated'] and not report['board_game'] and not report['turn_game']
    assert report['task_selection']['release_binding'] == 'service_selected'
    request = load_json(tmp_path / 'run/create_request.json')
    ready = load_json(tmp_path / 'run/episode_ready.json')
    settings = load_json(tmp_path / 'run/agent_config.json')
    assert request['robot_id'] == 'fr3_sharpa_instances'
    assert request['policy_id'] == 'classical_fingertip_instances'
    assert settings['controller'] == ready['spec']['system0'] == ready['capabilities']['system0']
    assert ready['spec']['task_release_sha256'] == hashlib.sha256(
        (release / 'release_manifest.json').read_bytes()).hexdigest()
    context = json.loads(model.turns[0][-1]['content'][0]['text'])
    assert set(context['robot']['hands']) == {'front', 'rear'}
    assert len(context['state']['observation']['proprioception']['joint_positions_rad']) == 58
    assert len(workers) == 1 and workers[0].closed
    assert report['physical_attempts'] == {'single': 0}


def test_explicit_family_still_rejects_a_historical_release_with_explicit_robot(tmp_path):
    task_id = 'cooperative_insertion'
    api, workers, _ = mechanical_service(tmp_path, task_id, release=historical_cooperative_release('v3'))
    model = ScriptedModel([{'stop': True}])
    with TestClient(create_app(service=api)) as http:
        report = run(RunConfig('http://testserver', 'task10', tmp_path / 'run',
                              family='mechanical_assembly', robot='fr3_sharpa_instances'),
                     bench=BenchHTTPClient('http://testserver', client=http), models={'single': model})
    assert report['task_setup_error']['code'] == 'release_mismatch'
    assert not report['ready_validated'] and not model.turns
    assert not report['result_valid_for_scoring'] and report['physical_attempts'] == {}
    assert len(workers) == 1 and workers[0].closed
    assert not any(op == 'step' for op, _ in workers[0].calls)


def descriptor(robot, policy_id='classical_fingertip_instances'):
    return PolicyDescriptor(policy_id=policy_id, version='test', implementation_sha256='a' * 64,
                            trained=False, camera_observation_mode='decision_only',
                            skills=['move_geometry', 'hold'], supported_robot_ids=[robot],
                            geometry_inputs=['wrist_pose', 'fingertip_positions']).model_dump(mode='json')


@pytest.mark.parametrize('case', ['multiple_policies', 'multiple_robots', 'no_robots'])
def test_implicit_historical_robot_requires_unique_public_identity(case):
    selected = select_task('cooperative_insertion')
    policies = [descriptor('fr3_sharpa_instances')]
    if case == 'multiple_policies':
        policies.append(descriptor('dual_fr3_sharpa_w01', 'classical_fingertip_bimanual'))
    else:
        policies[0]['supported_robot_ids'] = (
            ['fr3_sharpa_instances', 'dual_fr3_sharpa_w01'] if case == 'multiple_robots' else [])
    with pytest.raises((ValueError, BenchHTTPError)):
        select_task_controller(policies, selected)


@pytest.mark.parametrize('name,family,robot,accepted', [
    ('cooperative_insertion', None, None, True),
    ('task10', 'mechanical_assembly', None, False),
    ('cooperative_insertion', None, 'dual_fr3_sharpa_w01', False),
    ('unregistered_service_task', None, None, False),
    ('unregistered_service_task', None, 'fr3_sharpa_instances', True),
])
def test_robot_inference_is_limited_to_known_unpinned_ordinary_tasks(name, family, robot, accepted):
    selected = select_task(name, family)
    policies = [descriptor('fr3_sharpa_instances')]
    if not accepted:
        with pytest.raises(ValueError):
            select_task_controller(policies, selected, robot=robot)
    else:
        actual, controller = select_task_controller(policies, selected, robot=robot)
        assert actual == 'fr3_sharpa_instances'
        assert controller == PolicyDescriptor.model_validate(policies[0])


def test_explicit_robot_disambiguates_public_controllers_without_rewriting_their_identity():
    policies = [descriptor('fr3_sharpa_instances'),
                descriptor('dual_fr3_sharpa_w01', 'classical_fingertip_bimanual')]
    actual, controller = select_task_controller(policies, select_task('cooperative_insertion'),
                                               robot='dual_fr3_sharpa_w01')
    assert actual == 'dual_fr3_sharpa_w01'
    assert controller == PolicyDescriptor.model_validate(policies[1])
