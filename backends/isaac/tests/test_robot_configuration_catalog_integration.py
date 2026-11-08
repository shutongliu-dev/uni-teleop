"""Real frozen-release selection, with CPU workers only for episode lifecycle.

Set BIMANUAL_CONFIGURATION_INDEX to a completed preparation index (or colon-
separated complete family indexes) to exercise their real pins before reviewed
rows are integrated into the source catalog.
No symbolic stand-ins or changed task releases are accepted by this fixture.
"""
from copy import deepcopy
import json
import os
from pathlib import Path

from fastapi.testclient import TestClient
import pytest

from agent.model_client.bench_http import BenchHTTPClient
from agent.model_client.function_runner import RunConfig, run
from agent.model_client.scene_tasks import select_task
from bench.isaacsim.release import digest
from bench.paths import DATA_ROOT, load_json
from bench.protocol.robot_configurations import project_robot_configurations
from bench.runtime.robot_configurations import checked_configuration_release, discover_configuration_bindings
from bench.runtime.server import create_app
from bench.runtime.service import AgentService
from scenes.workrooms import robot_configuration_catalog
from scenes.workrooms.catalog import list_tasks, task_definition
from scenes.workrooms.robot_configurations import CONFIGURATION_IDS, ELIGIBLE_TASKS
from test_function_runner import ScriptedModel
from test_scene_function_runner import mechanical_service


EXPECTED = {(family, task) for family, tasks in ELIGIBLE_TASKS.items() for task in tasks}
PROBE = Path(f'{DATA_ROOT}/datasets/fr3_sharpa_bench/development/'
             'bimanual_options_20261023/probe_releases/parts_sorting/dual_same_side_v2')


def probe_binding(configuration_id):
    if not (PROBE/'release_manifest.json').is_file():
        pytest.skip('The explicit frozen derivative probe is unavailable')
    reference = {'path': str(PROBE), 'manifest_sha256': digest(PROBE/'release_manifest.json')}
    return {'task_id': 'parts_sorting', 'releases': {configuration_id: reference},
        'robot_configurations': {configuration_id: {'robot_requirements': {
            'hand_id': 'dual_fr3_sharpa_w01', 'assembly_version': 'isaacsim_6_1_v2',
            'articulations': 2, 'active_sides': ['left', 'right']}}}}


def test_real_probe_accepts_its_actual_configuration_identity():
    assert checked_configuration_release(probe_binding('dual_same_side'), 'dual_same_side') == PROBE


def test_real_probe_cannot_be_mislabeled_as_opposed_configuration():
    # All hashes and robot identities are real and valid; only the advertised
    # side arrangement is wrong. This must fail before any native startup.
    with pytest.raises(ValueError, match='configuration|Configuration|installation'):
        checked_configuration_release(probe_binding('dual_opposed'), 'dual_opposed')


@pytest.fixture
def published_catalog(monkeypatch):
    candidate_path = os.environ.get('BIMANUAL_CONFIGURATION_INDEX')
    if candidate_path:
        rows = {}
        variant_count = 0
        for path in candidate_path.split(os.pathsep):
            index = load_json(Path(path))
            assert index['schema'] == 'workroom_robot_configuration_candidates.v1'
            assert index['complete'] is True and index['variants'] == 2*len(index['tasks'])
            for row in index['tasks']:
                pair = row['family'], row['task_id']
                assert pair not in rows, 'Two preparation indexes define the same task'
                rows[pair] = row
            variant_count += index['variants']
        assert set(rows) == EXPECTED and variant_count == 2*len(EXPECTED)
        monkeypatch.setattr(robot_configuration_catalog, 'RELEASES', rows)
    if not robot_configuration_catalog.RELEASES:
        pytest.skip('The completed configuration catalog/index has not been published')
    assert set(robot_configuration_catalog.RELEASES) == EXPECTED
    return robot_configuration_catalog.RELEASES


@pytest.mark.parametrize('family,task_id', sorted(EXPECTED))
def test_every_real_task_offers_three_exact_distinct_release_pins(published_catalog, family, task_id):
    definition = task_definition(family, task_id)
    options = definition['robot_configurations']
    assert tuple(options) == CONFIGURATION_IDS
    assert options['single_arm']['source_release'] == definition['source_release']
    assert len({row['source_release']['manifest_sha256'] for row in options.values()}) == 3
    public = project_robot_configurations(options)
    assert DATA_ROOT not in json.dumps(public)
    for configuration_id in CONFIGURATION_IDS:
        layout = public[configuration_id]['layout']
        assert layout['placements'] == options[configuration_id]['layout']['placements']
        assert set(layout['placements']) == ({'right'} if configuration_id == 'single_arm' else {'left', 'right'})
        assert layout['frame'] == 'world' and layout['units'] == 'metres'
        reference = options[configuration_id]['source_release']
        binding = discover_configuration_bindings(reference['path'])[task_id]
        assert binding['default_configuration_id'] == configuration_id
        assert binding['robot_configurations'] == public
        assert checked_configuration_release(binding, configuration_id) == Path(reference['path'])
        selected = task_definition(family, task_id, configuration_id)
        assert selected['source_release'] == reference
        assert selected['configuration_id'] == configuration_id


def test_excluded_rooms_and_existing_dual_tasks_have_no_new_options(published_catalog):
    from scenes.workrooms.laundry_room.rules import CURRENT_TASK_IDS as LAUNDRY_TASKS
    for definition in list_tasks(include_planned=True):
        pair = definition['family'], definition['task_id']
        if pair[0] == 'laundry_room' and pair[1] in LAUNDRY_TASKS:
            # Laundry now owns its separately frozen deformable releases; it
            # does not enter the rigid-task derivative registry above.
            assert set(definition['robot_configurations']) == set(CONFIGURATION_IDS)
            assert definition['default_configuration_id'] == 'dual_opposed'
            continue
        if pair not in EXPECTED:
            assert 'robot_configurations' not in definition, pair
            selected = select_task(definition['task_id'], family=definition['family'])
            assert selected.robot_configurations is None
            assert selected.configuration_id is None
    for family, task_id, kind in [
        ('card_room', 'tic_tac_toe_3x3', 'board_game'),
        ('living_room', 'tower_extraction', 'two_player_adversarial'),
        ('mechanical_assembly', 'cooperative_insertion', 'single_agent')]:
        assert select_task(task_id, family=family).session_kind == kind


@pytest.mark.parametrize('robot_profile', ['original', 'l_mount'])
@pytest.mark.parametrize('configuration_id', ['dual_same_side', 'dual_opposed'])
def test_agent_selects_real_dual_release_and_session_keeps_it_frozen(
        published_catalog, tmp_path, configuration_id, robot_profile):
    family, task_id = 'mechanical_assembly', 'parts_sorting'
    definition = task_definition(family, task_id, robot_profile=robot_profile)
    selected = definition['robot_configurations'][configuration_id]
    api, workers, _ = mechanical_service(tmp_path, task_id, family, release=definition['source_release']['path'])
    model = ScriptedModel([
        {'configuration_id': configuration_id, 'reason': '根据桌面布局选择双手协作'},
        {'function': 'get_task', 'arguments': {}}, {'stop': True}])
    with TestClient(create_app(service=api)) as http:
        client = BenchHTTPClient('http://testserver', client=http)
        # Mechanical L1 publishes frozen seed cases 0..9 for each installation
        # and robot profile; seed 0 selects the first frozen case.
        report = run(RunConfig('http://testserver', task_id, tmp_path/'run', family=family,
                              max_decisions=3, seed=0, robot_profile=robot_profile),
                     bench=client, models={'single': model})
        assert report['error_type'] is None, report
        assert report['ready_validated'] is True
        ready = load_json(tmp_path/'run/episode_ready.json')
        spec, capabilities = ready['spec'], ready['capabilities']
        assert spec['configuration_id'] == configuration_id
        assert spec['task_release_sha256'] == selected['source_release']['manifest_sha256']
        assert spec['robot_id'] == 'dual_fr3_sharpa_w01'
        assert len(capabilities['arm_joints']) + len(capabilities['hand_joints']) == 58
        assert set(ready['observation']['proprioception']['hand_geometries']) == {'left', 'right'}
        assert len(workers) == 1 and workers[0].closed
        assert not any(operation == 'step' for operation, _ in workers[0].calls)
        # Further discovery of another installation never mutates an existing
        # session's spec, controller or retained physical installation.
        original_spec = deepcopy(api.get(report['episode_id']).spec.model_dump(mode='json'))
        alternate = 'dual_opposed' if configuration_id == 'dual_same_side' else 'dual_same_side'
        alternate_task, = client.list_tasks(task_id=task_id, configuration_id=alternate)
        assert alternate_task['task_release_sha256'] != spec['task_release_sha256']
        assert api.get(report['episode_id']).spec.model_dump(mode='json') == original_spec
        assert api.release == Path(definition['source_release']['path'])


def test_service_on_dual_release_still_offers_exact_original_default(published_catalog, tmp_path):
    definition = task_definition('mechanical_assembly', 'parts_sorting')
    dual = definition['robot_configurations']['dual_opposed']['source_release']
    api = AgentService(tmp_path/'episodes', release=dual['path'], worker_factory=lambda **kwargs: None)
    with TestClient(create_app(service=api)) as http:
        default, = http.get('/tasks').json()
        assert default['default_configuration_id'] == 'dual_opposed'
        single = http.get('/tasks', params={'task_id': 'parts_sorting', 'configuration_id': 'single_arm'})
        assert single.status_code == 200
        assert single.json()[0]['task_release_sha256'] == definition['source_release']['manifest_sha256']
        assert api.release == Path(dual['path'])
