"""NIST task publication and interface contracts; CPU evidence only.

Published-release tests use the actual immutable releases when selected by the
catalog. Worker doubles and synthetic layout fixtures do not qualify physics.
"""
from copy import deepcopy
import json
from pathlib import Path
import shutil

from fastapi.testclient import TestClient
import pytest

from agent.model_client.scene_tasks import TaskSetupError, select_task
from bench.control.factory import public_robot_geometry, robot_configuration
from bench.isaacsim.release import digest, load_evaluator, task_inputs, validate
from bench.paths import load_json
from bench.protocol import PolicyRegistry
from bench.protocol.task_referee import project_task_rules
from bench.runtime import service
from bench.runtime.reset_cases import project_reset_case
from bench.runtime.server import create_app
from bench.runtime.worker import public_task_projection
from bench.simulation.isaacsim_backend.l_mount import public_mount
from bench.simulation.isaacsim_backend.native_task_bundle import load_observer
from scenes.workrooms.catalog import load_task, task_definition
from scenes.workrooms.mechanical_assembly.nist_composition import TASK_IDS, TASK_SCHEMA, PHYSICS_DT
from scenes.workrooms.mechanical_assembly.nist_release import initial_state
from scenes.workrooms.mechanical_assembly.task_scene import empty_task_scene
from test_agent_service import FakeWorker, PRIVATE_MARKER, join_action, publish_targets, request
from test_bench_function_http import post


@pytest.fixture(scope='module')
def layout_contract():
    """A minimal task envelope over the actual pinned room, stand and robot."""
    scene, initial, _ = empty_task_scene()
    task_id = TASK_IDS[0]
    scene.pop('scenario', None)
    scene['physics'] = {'dt_s': PHYSICS_DT, 'control_dt_s': .02}
    scene['isaacsim']['physics_dt_s'] = PHYSICS_DT
    candidate = {'usd_path': '/unused/nist/task.usda', 'usd_sha256': 'a'*64,
                 'root_prim_path': '/World/NistTask'}
    task = {'id': task_id, 'task_id': task_id, 'schema_version': TASK_SCHEMA,
            'initial_state': initial, 'evaluation': {'physics_dt_s': PHYSICS_DT},
            'asset_metadata': {'actors': {'plug': '/World/NistTask/plug'}, 'joints': {}}}
    scene['mechanical_nist_provenance'] = {'asset_candidate': candidate}
    scene['native_task_bundle'] = {
        **candidate, 'rigid_bodies': task['asset_metadata']['actors'],
        'joints': {}, 'particle_sets': {},
        'observer': {'class_name': 'NativeTaskObserver', 'configuration': deepcopy(task)}}
    return scene, task, initial


def test_nist_reset_contract_preserves_the_original_29_joint_layout(layout_contract):
    scene, task, initial = deepcopy(layout_contract)
    assert initial_state(scene, task) == initial
    assert len(initial['joint_positions_rad']) == 29
    assert scene['robot']['base_position_m'] == [-.23, 0., .78]
    assert scene['robot']['hand_id'] == 'sharpa_w01_right'
    # The tolerated cross-runtime IK roundoff does not allow a new ready pose.
    scene['robot']['arm_home_rad'][0] += 4.e-16
    assert initial_state(scene, task) == initial


@pytest.mark.parametrize('change,reason', [
    (lambda s, t: s['workroom'].update(version='different_room'), 'parent identity'),
    (lambda s, t: s['robot']['base_position_m'].__setitem__(0, -.10), 'robot differs'),
    (lambda s, t: s['robot']['arm_home_rad'].__setitem__(0, .1), 'robot differs'),
    (lambda s, t: t['initial_state']['joint_positions_rad'].update(fr3_joint1=.1), 'ready joints'),
    (lambda s, t: s['fixtures'][0]['position_m'].__setitem__(0, .1), 'original stand'),
    (lambda s, t: s['objects'].append({'id': 'untracked_body'}), 'original stand'),
    (lambda s, t: s['cloth'].update(enabled=True), 'original stand'),
    (lambda s, t: s.update(scenario={'task_id': TASK_IDS[1]}), 'original stand'),
    (lambda s, t: t.update(engineering_probe=True), 'original stand'),
    (lambda s, t: s['isaacsim'].update(physics_dt_s=.001), 'clocks differ'),
    (lambda s, t: s['physics'].update(control_dt_s=.04), 'clocks differ'),
    (lambda s, t: t['evaluation'].update(physics_dt_s=.001), 'clocks differ'),
    (lambda s, t: s['native_task_bundle']['observer'].update(class_name='OtherObserver'), 'referee input'),
    (lambda s, t: s['native_task_bundle'].update(rigid_bodies={}), 'referee input'),
    (lambda s, t: s['native_task_bundle'].update(joints={'extra': '/World/NistTask/extra'}), 'referee input'),
    (lambda s, t: s['native_task_bundle'].update(usd_sha256='b'*64), 'referee input'),
    (lambda s, t: s['native_task_bundle'].update(reduced_dynamics={'class_name': 'ExtraModel'}), 'native passive'),
])
def test_nist_reset_rejects_setup_and_sampling_drift(layout_contract, change, reason):
    scene, task, _ = deepcopy(layout_contract)
    change(scene, task)
    with pytest.raises(ValueError, match=reason):
        initial_state(scene, task)


@pytest.mark.parametrize('number,task_id', tuple(enumerate(TASK_IDS, 6)))
def test_nist_numbers_are_independent_single_agent_tasks_with_explicit_availability(number, task_id):
    definition = task_definition('mechanical_assembly', task_id)
    for alias in (task_id, f'task{number}', f'L{number}'):
        selected = select_task(alias, family='mechanical_assembly')
        assert (selected.task_id, selected.task_number) == (task_id, number)
        assert selected.session_kind == 'single_agent'
        assert selected.players == ('single',)
        assert selected.practice_rsi_allowed and selected.pin_release
        assert selected.source_release == definition['source_release']
        if definition['status'] == 'planned':
            with pytest.raises(TaskSetupError, match='planned'):
                selected.require_runnable()
        else:
            selected.require_runnable()
            assert selected.metadata()['runnable']
    # Task10 was subsequently added as independent bimanual cooperation;
    # it is still not the omitted ATB5 fastening duplicate.
    cooperative = select_task('task10', family='mechanical_assembly')
    assert cooperative.task_id == 'cooperative_insertion'
    assert cooperative.task_id not in TASK_IDS


# Both physical robot profiles publish frozen reset cases 0..9 over one NIST
# task release; nonzero cases check that the episode seed is really bound.
SEED, OTHER_SEED = 7, 3
PROFILES = ('original', 'l_mount')
WRAPPERS = ('workroom_robot_revision_release.v1', 'workroom_randomized_release.v1')


def family_release(root):
    """The unrandomized NIST release under the seed-case and robot-revision wrappers."""
    manifest = load_json(root/'release_manifest.json')
    while manifest['schema'] in WRAPPERS:
        root = Path(manifest['base_release']['path'])
        manifest = load_json(root/'release_manifest.json')
    assert manifest['schema'] == 'mechanical_nist_task_release.v1'
    return root


def hook_root(root, manifest):
    """A robot revision reuses its parent's frozen evaluator files byte-for-byte."""
    if manifest['schema'] == 'workroom_robot_revision_release.v1':
        return Path(manifest['base_release']['path'])
    return root


@pytest.fixture(scope='module', params=[(task_id, profile) for task_id in TASK_IDS for profile in PROFILES],
                ids=lambda value: '-'.join(value))
def published_nist(request):
    task_id, profile = request.param
    definition = task_definition('mechanical_assembly', task_id, robot_profile=profile)
    if definition['status'] != 'implemented':
        pytest.skip('NIST candidate has no catalog-selected immutable release yet')
    root = Path(definition['source_release']['path'])
    assert (root/'release_manifest.json').is_file(), 'Published release must exist'
    assert digest(root/'release_manifest.json') == definition['source_release']['manifest_sha256']
    return task_id, root, validate(root), task_inputs(root, task_id, seed=SEED), profile


@pytest.fixture(scope='module', params=TASK_IDS)
def nist_task_release(request):
    """The one NIST task release that both robot profiles' wrappers bind."""
    roots = set()
    for profile in PROFILES:
        definition = task_definition('mechanical_assembly', request.param, robot_profile=profile)
        if definition['status'] != 'implemented':
            pytest.skip('NIST candidate has no catalog-selected immutable release yet')
        roots.add(family_release(Path(definition['source_release']['path'])))
    root, = roots
    return root


def test_nist_frozen_inputs_bind_rules_observer_assets_and_catalog(published_nist):
    task_id, root, manifest, (_, scene, task, initial), profile = published_nist
    assert manifest['status'] == 'prepared_unvalidated'
    assert manifest['physics_accepted'] is False
    assert manifest['task_policy_success_claimed'] is False
    assert task['episode_seed'] == SEED == task['reset_case']['seed']
    assert scene['objects'] == [] and len(scene['fixtures']) == 1
    assert len(initial['joint_positions_rad']) == 29
    assert set(scene['cameras']) == {'overview', 'task', 'room', 'wrist'}
    bundle = scene['native_task_bundle']
    assert bundle['observer']['configuration'] == task
    assert bundle['rigid_bodies'] == task['asset_metadata']['actors']
    assert bundle['joints'] == task['asset_metadata'].get('joints', {})
    assert 'reduced_dynamics' not in bundle
    observer = load_observer(bundle['observer'])
    evaluator = load_evaluator(root)(task)
    assert observer.__class__.__module__.startswith('_native_task_observer_')
    assert evaluator.__class__.__module__.startswith('_isaac_frozen_evaluator_')
    assert evaluator.inspect_conditions()['success'] is False
    # The worker publishes the referee rules plus the selected public reset case.
    assert public_task_projection(task) == project_reset_case(task, project_task_rules(task))
    assert project_task_rules(task)['create_endpoint'] == 'POST /episodes'
    _, other_scene, other_task, _ = task_inputs(root, task_id, seed=OTHER_SEED)
    assert other_task['episode_seed'] == OTHER_SEED and task['episode_seed'] == SEED
    assert other_scene['native_task_bundle']['observer']['configuration'] == other_task
    load_evaluator(root)(other_task)
    loaded = load_task('mechanical_assembly', task_id, robot_profile=profile)
    _, default_scene, default_task, default_initial = task_inputs(root, task_id)
    loaded['scene']['native_task_bundle']['observer']['path'] = str(
        hook_root(root, manifest)/loaded['scene']['native_task_bundle']['observer']['path'])
    assert (loaded['scene'], loaded['task'], loaded['initial_state']) == (
        default_scene, default_task, default_initial)
    with pytest.raises(ValueError, match='frozen ordinary initial cases|exact nominal task'):
        task_inputs(root, task_id, phase='at_goal')
    with pytest.raises(ValueError, match='frozen seed 0..9'):
        task_inputs(root, task_id, seed=29)
    other = 'l_mount' if profile == 'original' else 'original'
    other_root = Path(task_definition('mechanical_assembly', task_id, robot_profile=other)['source_release']['path'])
    assert family_release(other_root) == family_release(root)


def rewrite_frozen_json(root, name, value, manifest):
    """Deliberately rehash a fixture so semantic validation, not SHA drift, runs."""
    (root/name).write_text(json.dumps(value))
    sha = digest(root/name)
    manifest['files'][name]['sha256'] = sha
    manifest['files_sha256'][name] = sha
    for key in ('scene', 'task', 'initial_state'):
        if name == key+'.json':
            manifest['tasks'][0][key+'_sha256'] = sha
    if name == 'physics_profile.json':
        manifest['physics_profile_sha256'] = sha


@pytest.mark.parametrize('mutation,reason', [
    ('clock', 'clocks differ'), ('stand', 'original stand'),
    ('observer', 'referee input'), ('catalog', 'asset catalog'),
])
def test_nist_rehashed_release_still_rejects_semantic_drift(tmp_path, nist_task_release, mutation, reason):
    original = nist_task_release
    root = tmp_path/'release'
    shutil.copytree(original, root)
    manifest = load_json(root/'release_manifest.json')
    scene = load_json(root/'scene.json')
    if mutation == 'clock':
        scene['isaacsim']['physics_dt_s'] = .001
        profile = load_json(root/'physics_profile.json')
        profile['isaacsim_options'] = deepcopy(scene['isaacsim'])
        rewrite_frozen_json(root, 'physics_profile.json', profile, manifest)
    elif mutation == 'stand':
        scene['fixtures'][0]['position_m'][0] += .05
    elif mutation == 'observer':
        scene['native_task_bundle']['observer']['class_name'] = 'SequenceEvaluator'
    else:
        manifest['asset_catalog']['sha256'] = '0'*64
    rewrite_frozen_json(root, 'scene.json', scene, manifest)
    (root/'release_manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match=reason):
        validate(root, assets=False)


def nist_service(tmp_path, published_nist):
    task_id, root, _, (_, scene, task, initial), _ = published_nist
    workers = []
    geometry = public_robot_geometry(robot_configuration(scene))

    def factory(**kwargs):
        assert kwargs['task_id'] == task_id and kwargs['seed'] == SEED
        worker = FakeWorker(**kwargs, depth=True, acknowledged_targets=True, on_step=publish_targets)
        worker.arm, worker.hand, worker.active = list(geometry.arm.names), list(geometry.hand.names), list(geometry.names)
        worker.raw['joint_positions_rad'] = deepcopy(initial['joint_positions_rad'])
        worker.raw['joint_velocities_rad_s'] = dict.fromkeys(worker.active, 0.)
        worker.raw['robot'] = {'joint_targets_rad': deepcopy(initial['joint_positions_rad'])}
        worker.initial['observation'] = deepcopy(worker.raw)
        worker.initial['metadata'].update(scene_spec=deepcopy(scene), action_spec={
            'joint_names': worker.active, 'arm_joint_names': worker.arm, 'hand_joint_names': worker.hand,
            'joint_limits_rad': {name: [float(geometry.lower[i]), float(geometry.upper[i])]
                                 for i, name in enumerate(geometry.names)}})
        worker.score.update(load_evaluator(root)(task).inspect_conditions())
        worker.score['private_chain_geometry'] = PRIVATE_MARKER
        worker.initial['score'] = deepcopy(worker.score)
        worker.initial['task'] = public_task_projection(task)
        workers.append(worker)
        return worker

    return service.AgentService(tmp_path/'episodes', release=root,
        registry=PolicyRegistry(), worker_factory=factory), workers


@pytest.mark.parametrize('change', [
    {'robot_id': 'dual_fr3_sharpa_w01'}, {'scene_variant': 'enriched_v2'},
])
def test_nist_setup_rejected_before_worker_for_changed_robot_or_scene(tmp_path, published_nist, change):
    api, workers = nist_service(tmp_path, published_nist)
    with pytest.raises(service.ServiceError, match='requires its frozen robot and original scene'):
        api.create(request(task_id=published_nist[0], seed=SEED, **change))
    assert workers == [] and not api.sessions


def test_nist_public_http_hold_finish_and_private_evidence_isolation(tmp_path, published_nist):
    task_id, root, manifest, (_, scene, task, initial), profile = published_nist
    api, workers = nist_service(tmp_path, published_nist)
    with TestClient(create_app(service=api)) as client:
        listing = client.get('/tasks')
        assert listing.status_code == 200 and len(listing.json()) == 1
        discovery = listing.json()[0]
        # Discovery publishes the frozen seed-0 case of the selected installation.
        _, _, discovered, _ = task_inputs(root, task_id)
        expected = project_reset_case(discovered, project_task_rules(discovered))
        mount = public_mount(scene['robot'])
        assert (mount is None) == (profile == 'original')
        if mount is not None:
            expected['robot_mount'] = mount
        expected.update(configuration_id=manifest['configuration_id'],
                        task_release_sha256=digest(root/'release_manifest.json'))
        assert {key: value for key, value in discovery.items()
                if key not in {'robot_configurations', 'default_configuration_id',
                               'scene_description_zh'}} == expected
        assert set(discovery['robot_configurations']) == {
            'single_arm', 'dual_same_side', 'dual_opposed'}
        created = client.post('/episodes', json=request(task_id=task_id, seed=SEED).model_dump(mode='json'))
        assert created.status_code == 201, created.text
        ready = created.json()
        select_task(task_id, family='mechanical_assembly', robot_profile=profile).verify_release(ready['spec'])
        session = api.get(ready['episode_id'])
        endpoint = '/episodes/'+ready['episode_id']
        public = post(client, endpoint, 'get_task', {}, 'nist/task')
        assert public.result.task_id == task_id and not public.result.episode.terminal
        accepted = post(client, endpoint, 'hold', {
            'action_id': 'nist_hold', 'observation_sequence': session.sequence,
            'duration_s': .04}, 'nist/hold', expected_status=202)
        assert accepted.result.status == 'accepted'
        join_action(session)
        status = post(client, endpoint, 'get_action', {'action_id': 'nist_hold'}, 'nist/status')
        assert status.result.status == 'completed'
        commands = [value for operation, value in workers[-1].calls if operation == 'step']
        assert len(commands) == 2
        assert all(value['command']['joint_positions_rad'] == initial['joint_positions_rad'] for value in commands)
        finished = post(client, endpoint, 'finish_episode', {}, 'nist/finish')
        assert finished.result.outcome == 'cancelled'
        assert finished.result.action_attempts == 1
        assert finished.result.completion_sim_s is None
        assert workers[-1].closed
        assert post(client, endpoint, 'get_result', {}, 'nist/result').result == finished.result
        assert PRIVATE_MARKER in (session.directory/'private/evaluator_evidence.json').read_text()
        exposed = json.dumps([ready, listing.json(), public.model_dump(mode='json'),
                              finished.model_dump(mode='json')])
        assert PRIVATE_MARKER not in exposed
        assert '/World/NistTask' not in exposed
        assert 'private_chain_geometry' not in exposed
