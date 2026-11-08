"""Actual public discovery and Agent selection for frozen fabric candidates, no native worker."""
from copy import deepcopy
import json
from pathlib import Path

from fastapi.testclient import TestClient
import pytest

from agent.model_client.bench_http import BenchHTTPClient
from agent.model_client.scene_tasks import published_task, select_configuration, select_task, TaskSetupError
from bench.isaacsim.release import digest
from bench.paths import load_json
from bench.protocol.task_referee import project_task_rules
from scenes.runtime import terminal_scoring_contract
from bench.runtime.controllers import default_registry
from bench.runtime.server import create_app
from bench.runtime.service import AgentService
from scenes.workrooms.catalog import task_definition
from scenes.workrooms.laundry_room.robot_configuration_catalog import RELEASES
from scenes.workrooms.laundry_room.robot_configurations import CONFIGURATION_IDS
from scenes.workrooms.laundry_room.rules import CURRENT_TASK_IDS


def forbidden_worker(**kwargs):
    raise AssertionError('Discovery and pre-episode selection must not start native physics')


@pytest.mark.parametrize('task_id', CURRENT_TASK_IDS)
@pytest.mark.parametrize('configuration_id', CONFIGURATION_IDS)
def test_current_fabric_selection_uses_exact_public_manifest_rules_and_material(task_id, configuration_id, tmp_path):
    definition = task_definition('laundry_room', task_id)
    row = {'configurations': definition['robot_configurations']}
    assert definition['default_configuration_id'] == 'dual_opposed'
    assert definition['source_release'] == row['configurations']['dual_opposed']['source_release']
    release = Path(definition['source_release']['path'])
    api = AgentService(tmp_path/'episodes', release=release, registry=default_registry(release),
                       worker_factory=forbidden_worker)
    with TestClient(create_app(service=api)) as http:
        bench = BenchHTTPClient('http://testserver', client=http)
        initial = select_task('task'+str(definition['task_number']), family='laundry_room')
        initial.require_runnable()
        public, game = published_task(initial, bench.list_tasks())
        assert not game and set(public['robot_configurations']) == set(CONFIGURATION_IDS)
        selected = select_configuration(initial, public, configuration_id)
        expected = row['configurations'][configuration_id]
        assert selected.source_release == expected['source_release']
        source = Path(selected.source_release['path'])
        manifest = load_json(source/'release_manifest.json')
        # The default L-mount profile revises only the robot of the original
        # profile's frozen seed release, which binds the original installation.
        assert manifest['schema'] == 'workroom_robot_revision_release.v1'
        parent = task_definition('laundry_room', task_id, robot_profile='original')[
            'robot_configurations'][configuration_id]['source_release']
        assert manifest['base_release'] == {**parent, 'schema': 'laundry_randomized_release.v1'}
        assert digest(Path(parent['path'])/'release_manifest.json') == parent['manifest_sha256']
        parent_manifest = load_json(Path(parent['path'])/'release_manifest.json')
        assert parent_manifest['schema'] == 'laundry_randomized_release.v1'
        original = RELEASES[task_id]['configurations'][configuration_id]
        assert parent_manifest['base_release'] == {**original['source_release'], 'schema': original['release_schema']}
        frozen_task = load_json(source/'task.json')
        frozen_scene = load_json(source/'scene.json')
        assert not any(key in value for value in (manifest, parent_manifest)
                       for key in ('fabric_visual_derivation', 'table_material_derivation'))
        assert digest(source/'release_manifest.json') == selected.source_release['manifest_sha256']
        assert manifest['configuration_id'] == parent_manifest['configuration_id'] == configuration_id
        chosen, game = published_task(selected, bench.list_tasks(task_id=task_id, configuration_id=configuration_id))
        assert not game
        selected.verify_release(chosen)
        projected = project_task_rules(frozen_task, scoring_contract=terminal_scoring_contract(frozen_task))
        assert all(chosen[key] == value for key,value in projected.items())
        assert frozen_task['public_rules'] == expected['rules']['contract']
        assert chosen['public_rules']['object_properties'] == frozen_task['public_rules']['referee']['object_properties']
        assert chosen['public_rules']['object_properties']  # Nominal material facts reach the model.
        assert expected['objects']['cloths'] == frozen_task['public_rules']['material_contract']['cloths']
        assert selected.robot_requirements['articulations'] == (1 if configuration_id=='single_arm' else 2)
        assert selected.robot_requirements['active_sides'] == (['right'] if configuration_id=='single_arm' else ['left','right'])
        policies = bench.list_policies(task_id=task_id, configuration_id=configuration_id)
        assert len(policies) == 1
        assert policies[0]['policy_id'] == ('classical_fingertip' if configuration_id=='single_arm' else 'classical_fingertip_bimanual')
        assert policies[0]['supported_robot_ids'] == [frozen_scene['robot']['hand_id']]
        assert bench.list_interfaces(task_id=task_id, configuration_id=configuration_id)
        assert all(cloth.get('visual_profile') for cloth in frozen_scene['cloths'])
        assert expected['qualification']['status'] == 'prepared_unvalidated'
        assert all(value is False for key,value in expected['qualification'].items() if key not in ('status','scope'))
        tampered = deepcopy(public)
        tampered['robot_configurations'][configuration_id]['task_release_sha256'] = 'f'*64
        with pytest.raises(TaskSetupError): select_configuration(initial,tampered,configuration_id)
    assert not api.sessions



@pytest.mark.parametrize('robot_profile', ['original', 'l_mount'])
def test_current_catalog_contains_input_contracts_without_development_report_bindings(robot_profile):
    for task_id in CURRENT_TASK_IDS:
        row = {'configurations': task_definition('laundry_room', task_id, robot_profile=robot_profile)['robot_configurations']}
        assert set(row['configurations']) == set(CONFIGURATION_IDS)
        # The paired parameterized overlay publishes its immutable pair records
        # and L-mount releases from the parameterized-seed dataset, pinned by
        # hash; no other value may bind development evidence.
        pinned = ('source_release', 'physical_instance_pair')
        assert '/development/' not in json.dumps({key: {name: value for name, value in option.items() if name not in pinned}
                                                  for key, option in row['configurations'].items()})
        for option in row['configurations'].values():
            assert digest(option['physical_instance_pair']['path']) == option['physical_instance_pair']['sha256']
            assert 'input_identity_evidence' not in option
            assert option['qualification']['status'] == 'prepared_unvalidated'
            assert all(value is False for key, value in option['qualification'].items()
                       if key not in ('status', 'scope'))
            root = Path(option['source_release']['path'])
            manifest = load_json(root/'release_manifest.json')
            assert digest(root/'release_manifest.json') == option['source_release']['manifest_sha256']
            assert not manifest['physics_accepted'] and not manifest['task_policy_success_claimed']
            assert not any('report' in Path(path).name or 'qualification' in Path(path).name
                           for path in manifest['asset_files_sha256'])
