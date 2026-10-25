"""Canonical CPU/API contracts only; these do not qualify native manipulation."""
from copy import deepcopy
import json
import shutil

from fastapi.testclient import TestClient
import pytest

from bench.isaacsim.release import digest, dump, load_evaluator, task_inputs, validate
from bench.paths import load_json
from bench.protocol import PolicyRegistry
from bench.runtime.placement_tasks import project_placement_task
from bench.runtime.server import create_app
from bench.runtime.service import AgentService, ServiceError
from test_agent_service import FakeWorker, PRIVATE_MARKER, join_action, publish_targets, request


@pytest.fixture(scope='module')
def placement_release(tmp_path_factory):
    from scenes.workrooms.card_room.tasks.task0 import release
    parent = tmp_path_factory.mktemp('card_room_placement')
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(release, 'DATASET_ROOT', parent)
        root = parent/'candidate'
        release.prepare_task_release(root)
    return root


def test_episode_seed_selects_goals_from_frozen_rules_without_moving_reset(placement_release):
    _, scene0, task0, initial0 = task_inputs(placement_release, 'task0', seed=0)
    _, scene7, task7, initial7 = task_inputs(placement_release, 'task0', seed=7)
    assert scene0 == scene7 and initial0 == initial7
    assert task0['episode_seed'] == 0 and task7['episode_seed'] == 7
    assert task0['public_goal']['target_cells'] != task7['public_goal']['target_cells']
    assert task_inputs(placement_release, 'task0', seed=7)[2] == task7
    assert len({tuple(cell) for cell in task7['public_goal']['target_cells']}) == 3
    assert load_evaluator(placement_release)(task7).task == task7
    with pytest.raises(ValueError, match='reset-at-goal'):
        task_inputs(placement_release, 'task0', 'qualification_probe')
    for seed in (-1, 2**32, True, '7'):
        with pytest.raises(ValueError):
            task_inputs(placement_release, 'task0', seed=seed)


def test_public_placement_projection_has_no_private_geometry_or_seed_zero_goal(placement_release):
    task = task_inputs(placement_release, 'task0', seed=7)[2]
    task['public_goal']['native_object_state'] = PRIVATE_MARKER
    task['public_goal']['geometry'] = {'private': PRIVATE_MARKER}
    task['evaluation']['private'] = PRIVATE_MARKER
    task['public_rules']['success']['native_state'] = PRIVATE_MARKER
    task['public_rules']['success']['thresholds']['private'] = PRIVATE_MARKER
    task['public_rules']['recovery']['private'] = PRIVATE_MARKER
    row = project_placement_task(task)
    assert PRIVATE_MARKER not in json.dumps(row)
    assert row['episode_seed'] == 7
    assert row['goal']['target_cells'] == task['evaluation']['target_cells']
    assert row['public_rules']['success']['thresholds']['dwell_s'] == .3
    assert row['public_rules']['recovery']['automatic_forfeit'] is False
    catalog = project_placement_task(task, episode=False)
    assert 'episode_seed' not in catalog
    assert 'target_cells' not in catalog['goal'] and 'target_positions_m' not in catalog['goal']
    assert len(catalog['goal']['candidate_cells']) == 9
    row['goal']['target_cells'][0][0] = 99
    assert project_placement_task(task)['goal']['target_cells'] == task['evaluation']['target_cells']


@pytest.mark.parametrize('damage', ['robot', 'extra_piece', 'reset_pose', 'extra_reset', 'rules', 'profile', 'claim'])
def test_even_repinned_release_contract_damage_is_rejected(placement_release, tmp_path, damage):
    root = tmp_path/'release'
    shutil.copytree(placement_release, root)
    manifest = load_json(root/'release_manifest.json')
    name = 'scene.json'
    value = load_json(root/name)
    if damage == 'robot': value['robot']['hand_id'] = 'dual_fr3_sharpa_w01'
    elif damage == 'extra_piece': value['objects'].append(deepcopy(value['objects'][0]))
    elif damage == 'reset_pose': value['objects'][0]['position_m'][0] += .1
    elif damage == 'extra_reset':
        name = 'task.json'
        value = load_json(root/name)
        value['initial_state']['objects'] = {}
    elif damage == 'rules': name, value = 'rules.json', {}
    elif damage == 'profile':
        name = 'physics_profile.json'
        value = load_json(root/name)
        value['isaacsim_options']['physics_dt_s'] = .001
    else: manifest['physics_accepted'] = True
    dump(root/name, value)
    manifest['files'][name]['sha256'] = digest(root/name)
    if name == 'physics_profile.json': manifest['physics_profile_sha256'] = digest(root/name)
    dump(root/'release_manifest.json', manifest)
    with pytest.raises((ValueError, KeyError)):
        validate(root, assets=False)


def make_service(tmp_path, release, *, corrupt_worker=False):
    workers = []
    def factory(**kwargs):
        worker = FakeWorker(**kwargs, acknowledged_targets=True, on_step=publish_targets)
        task = task_inputs(kwargs['release'], kwargs['task_id'], seed=kwargs['seed'])[2]
        worker.initial['task'] = project_placement_task(task)
        if corrupt_worker:
            worker.initial['task']['episode_seed'] += 1
        workers.append(worker)
        return worker
    return AgentService(tmp_path, release=release, registry=PolicyRegistry(), worker_factory=factory), workers


def test_ordinary_episode_publishes_seed_bound_goals_and_uses_shared_functions(placement_release, tmp_path):
    api, workers = make_service(tmp_path/'episodes', placement_release)
    with TestClient(create_app(service=api)) as client:
        listed = client.get('/tasks').json()
        assert listed[0]['create_endpoint'] == 'POST /episodes'
        assert 'target_cells' not in listed[0]['goal']
        response = client.post('/episodes', json=request(task_id='task0', seed=7).model_dump(mode='json'))
        assert response.status_code == 201, response.text
        reply = response.json()
        assert reply['task']['episode_seed'] == 7
        endpoint = f"/episodes/{reply['episode_id']}"
        assert client.get(endpoint+'/task').json() == reply['task']
        assert PRIVATE_MARKER not in json.dumps(reply)
        session = api.get(reply['episode_id'])
        result = session.functions.invoke({'function':'hold', 'request_id':'placement/hold',
            'arguments':{'action_id':'hold_1','observation_sequence':0,'duration_s':.04}})
        assert result.ok
        join_action(session)
        assert session.clock.action_attempts == 1
        assert len([op for op, _ in workers[0].calls if op == 'step']) == 2
        assert client.get(endpoint+'/task').json() == reply['task']


@pytest.mark.parametrize('changes', [{'robot_id':'dual_fr3_sharpa_w01'}, {'scene_variant':'enriched_v2'}])
def test_fixed_single_robot_and_scene_are_rejected_before_worker(placement_release, tmp_path, changes):
    api, workers = make_service(tmp_path/'episodes', placement_release)
    with pytest.raises(ServiceError, match='frozen single-arm'):
        api.create(request(task_id='task0', **changes))
    assert not workers


def test_worker_target_mismatch_fails_setup_without_public_episode(placement_release, tmp_path):
    api, workers = make_service(tmp_path/'episodes', placement_release, corrupt_worker=True)
    with pytest.raises(ServiceError, match='episode targets differ'):
        api.create(request(task_id='task0', seed=7))
    assert not api.sessions and workers[0].closed_force


@pytest.mark.parametrize('explicit', [False, True])
def test_omitting_seed_chooses_random_seed_but_explicit_zero_is_reproducible(
        placement_release, tmp_path, monkeypatch, explicit):
    from bench.runtime import service
    sampled = []
    def randbelow(limit):
        sampled.append(limit)
        return 8191
    monkeypatch.setattr(service.secrets, 'randbelow', randbelow)
    api, _ = make_service(tmp_path/'episodes', placement_release)
    chosen = request(task_id='task0', **({'seed':0} if explicit else {}))
    assert ('seed' in chosen.model_fields_set) is explicit
    try:
        reply = api.create(chosen)
        expected = 0 if explicit else 8191
        assert reply['spec']['seed'] == reply['task']['episode_seed'] == expected
        assert sampled == ([] if explicit else [2**32])
        assert chosen.seed == 0  # The caller's request remains untouched.
    finally:
        api.shutdown()
