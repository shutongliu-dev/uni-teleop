"""Actual cloth releases exercise ordinary configuration selection over HTTP."""
from copy import deepcopy
import json

import pytest

from bench.isaacsim.release import digest

from bench.paths import load_json, DATA_ROOT
from scenes.workrooms.laundry_room import assets, release, robot_configurations as layouts
from scenes.workrooms.laundry_room.configuration_index import configuration_row, material_identity
from test_laundry_robot_configurations import STANDS


@pytest.fixture
def installation_releases(tmp_path, monkeypatch):
    from scripts.maintenance import freeze_project_sources
    from scenes.workrooms.laundry_room.randomization import catalog as seeded_catalog
    from scenes.workrooms.parameterized import catalog as parameterized
    from scenes.workrooms.robot_revision import catalog as revised_catalog
    empty = tmp_path/'seed_pins.json'
    empty.write_text(json.dumps({'schema': 'laundry_seed_catalog.v1', 'tasks': {}}))
    monkeypatch.setattr(seeded_catalog, 'PINS', empty)
    # The robot revision and paired parameterized overlays pin the published
    # seed releases as parents; these fixture installations replace them.
    revision = tmp_path/'revision_pins.json'
    revision.write_text(json.dumps({'schema': 'robot_revision_catalog.v1', 'tasks': {}}))
    monkeypatch.setattr(revised_catalog, 'PINS', revision)
    neutral = tmp_path/'parameterized_pins.json'
    neutral.write_text(json.dumps({'schema': parameterized.SCHEMA, 'tasks': {}}))
    monkeypatch.setattr(parameterized, 'PINS', neutral)
    monkeypatch.setattr(assets, 'ASSETS', tmp_path)
    monkeypatch.setattr(release, 'DATASETS', tmp_path)
    monkeypatch.setattr(freeze_project_sources, 'verify_if_snapshot', lambda *a, **k: {'fixture_only': True})
    monkeypatch.setattr(release, 'asset_inventory', lambda scene: {
        path: {'sha256': sha} for path, sha in scene['laundry_task_provenance']['asset_candidate']['files_sha256'].items()})
    asset = tmp_path/'asset'
    assets.prepare_assets('towel_folding', asset, material_profile='towel_4mm_density_v1')
    paths = {}
    for identifier in layouts.CONFIGURATION_IDS:
        paths[identifier] = tmp_path/identifier
        release.prepare_task_release('towel_folding', paths[identifier],
            asset_manifest=asset/'manifest.json', configuration_id=identifier,
            stand_manifest=STANDS.parent/'contact_options_v2'/identifier/'manifest.json')
    return paths


def test_configuration_index_preserves_material_identity_and_never_inherits_qualification(installation_releases):
    row = configuration_row('towel_folding', installation_releases)
    assert row['base_release'] == row['configurations']['dual_opposed']['source_release']
    assert len({r['source_release']['manifest_sha256'] for r in row['configurations'].values()}) == 3
    assert all(r['qualification']['native_preflight'] is False for r in row['configurations'].values())
    assert all(r['qualification']['robot_manipulation'] is False for r in row['configurations'].values())
    assert all(r['objects']['cloths'][0]['physical_thickness_m'] == .004 for r in row['configurations'].values())
    with pytest.raises(ValueError, match='identity differs'):
        configuration_row('towel_folding', dict(installation_releases,
            single_arm=installation_releases['dual_opposed']))
    path = installation_releases['single_arm']
    scene = load_json(path/'scene.json')
    task = load_json(path/'task.json')
    identity = material_identity(scene, task)
    moved = deepcopy(task)
    moved['asset_metadata']['cloths'][0]['initial_vertices_m'][0][0] += .01
    assert material_identity(scene, moved) == identity
    heavier = deepcopy(task)
    heavier['asset_metadata']['cloths'][0]['mass_kg'] *= 2
    assert material_identity(scene, heavier) != identity
    changed = deepcopy(scene); changed['isaacsim']['cloth_thickness_m'] *= 2
    assert material_identity(changed, task) != identity


def test_all_three_laundry_configurations_run_through_ordinary_agent_http(installation_releases, tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    from agent.model_client.bench_http import BenchHTTPClient
    from agent.model_client.function_runner import RunConfig, run
    from bench.runtime.server import create_app
    from scenes.workrooms.laundry_room import catalog_definition, robot_configuration_catalog
    from scenes.workrooms.laundry_room.tasks.towel_folding import definition
    from test_scene_function_runner import mechanical_service
    from test_function_runner import ScriptedModel
    row = configuration_row('towel_folding', installation_releases)
    monkeypatch.setitem(catalog_definition.RELEASES, 'towel_folding', row['base_release'])
    monkeypatch.setattr(robot_configuration_catalog, 'RELEASES', {'towel_folding': row})
    monkeypatch.setattr(definition, 'TASK', catalog_definition.task_definition('towel_folding'))
    for identifier in layouts.CONFIGURATION_IDS:
        root = tmp_path/('http_'+identifier)
        api, workers, _ = mechanical_service(root, 'towel_folding', 'laundry_room')
        model = ScriptedModel([{'configuration_id': identifier, 'reason': 'Choose reviewed fixture installation'},
            {'function': 'get_task', 'arguments': {}},
            {'function': 'hold', 'arguments': {'duration_s': .04}}, {'stop': True}])
        with TestClient(create_app(service=api)) as http:
            client = BenchHTTPClient('http://testserver', client=http)
            published, = client.list_tasks(task_id='towel_folding', configuration_id=identifier)
            assert published['configuration_id'] == identifier
            assert DATA_ROOT not in json.dumps(published['robot_configurations'])
            report = run(RunConfig('http://testserver', 'task2', root/'run',
                family='laundry_room', configuration='auto', max_decisions=4),
                bench=client, models={'single': model})
        assert report['error_type'] is None, report
        ready = load_json(root/'run/episode_ready.json')
        assert ready['spec']['configuration_id'] == identifier
        assert ready['spec']['task_release_sha256'] == row['configurations'][identifier]['source_release']['manifest_sha256']
        assert len(workers) == 1 and workers[0].closed
        assert len(workers[0].active) == (29 if identifier == 'single_arm' else 58)
        assert any(operation == 'step' for operation, _ in workers[0].calls)
        assert report['result']['outcome'] != 'success'



def test_current_five_task_installations_bind_rules_material_and_unvalidated_preparation():
    from pathlib import Path
    from scenes.workrooms.catalog import task_definition
    from bench.protocol.robot_configurations import project_robot_configurations
    from scenes.workrooms.laundry_room.catalog_definition import HISTORICAL_ROBOT_RELEASES
    from scenes.workrooms.laundry_room.rules import CURRENT_TASK_IDS
    for number, task_id in enumerate(CURRENT_TASK_IDS, 1):
        default = task_definition('laundry_room', task_id)
        assert default['task_number'] == number
        assert default['default_configuration_id'] == 'dual_opposed'
        assert default['source_release'] != HISTORICAL_ROBOT_RELEASES[task_id]
        public = json.dumps(project_robot_configurations(default['robot_configurations']))
        assert DATA_ROOT not in public and 'native_interface_evidence' not in public
        material = None
        for identifier in layouts.CONFIGURATION_IDS:
            definition = task_definition('laundry_room', task_id, identifier)
            root = Path(definition['source_release']['path'])
            scene, task = [load_json(root/name) for name in ('scene.json', 'task.json')]
            assert digest(root/'release_manifest.json') == definition['source_release']['manifest_sha256']
            assert task['public_rules'] == definition['rules']['contract']
            assert task['robot_configuration_id'] == identifier
            assert definition['layout']['robot_installation'] == task['workspace_contract']['robot_layout_id']
            assert definition['objects']['cloths'][0]['physical_thickness_m'] == task['asset_metadata']['physical_thickness_m']
            qualification = definition['qualification']
            assert qualification['status'] == 'prepared_unvalidated'
            # The paired parameterized overlay uses its own qualification record;
            # every claim it carries, and both physics/manipulation, stay False.
            assert qualification['robot_manipulation'] is False and qualification['physics_accepted'] is False
            for key in ('native_preflight', 'agent_bench_interface', 'workspace_ik', 'workspace_clearance',
                        'inherited_qualification'):
                assert qualification.get(key, False) is False
            assert 'native_interface_evidence' not in qualification and 'native_runtime_source' not in qualification
            assert scene['isaacsim']['workroom_surface_materials']['surfaces']['table']['dynamic_friction'] == .45
            assert all(cloth.get('visual_profile') for cloth in scene['cloths'])
            if task_id == 'towel_folding':
                assert task['public_rules']['rule_profile'] == 'laundry_practical_folding_task.v3'
            if task_id == 'towel_stacking':
                assert task['asset_metadata']['reset_profile'] == 'towel_isometric_u_fold_v1'
                assert task['public_rules']['rule_profile'] == 'laundry_practical_stacking_task.v3'
            current = material_identity(scene, task)
            assert material is None or current == material
            material = current
    assert task_definition('laundry_room', 'shirt_stacking')['source_release'] == HISTORICAL_ROBOT_RELEASES['shirt_stacking']

def test_catalog_private_thresholds_match_each_selected_frozen_rule_profile():
    from pathlib import Path
    from scenes.workrooms.catalog import task_definition
    from scenes.workrooms.laundry_room.rules import CURRENT_TASK_IDS, thresholds
    for task_id in CURRENT_TASK_IDS:
        for identifier in layouts.CONFIGURATION_IDS:
            definition = task_definition('laundry_room', task_id, identifier)
            task = load_json(Path(definition['source_release']['path'])/'task.json')
            selected = task['public_rules']['rule_profile']
            assert definition['rules']['contract']['rule_profile'] == selected
            assert definition['evaluator']['thresholds'] == thresholds(task_id, selected)
