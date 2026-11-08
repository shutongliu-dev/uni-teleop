"""Cheap selector guards: immutable parent metadata, never inherited qualification."""
from copy import deepcopy

import pytest

from bench.protocol.robot_configurations import project_robot_configurations
from scenes.workrooms.catalog import task_definition
from scenes.workrooms.laundry_room.robot_configuration_catalog import RELEASES
from scenes.workrooms.laundry_room.randomization import SCHEMA, TASK_IDS, VERSION
from scenes.workrooms.laundry_room.randomization import catalog, common


@pytest.fixture
def pins(tmp_path, monkeypatch):
    # These minimal manifest/task fixtures exercise only the deliberately cheap
    # catalog guard. They are not published or passed as complete releases.
    path = tmp_path/'pins.json'
    common.write_json(path, {'schema': 'laundry_seed_catalog.v1', 'tasks': {}})
    monkeypatch.setattr(catalog, 'PINS', path)
    # The robot revision and paired parameterized overlays pin the published
    # seed releases as parents; neutralize them to read the original menus.
    from scenes.workrooms.parameterized import catalog as parameterized
    from scenes.workrooms.robot_revision import catalog as revised_catalog
    common.write_json(tmp_path/'revision_pins.json', {'schema': 'robot_revision_catalog.v1', 'tasks': {}})
    monkeypatch.setattr(revised_catalog, 'PINS', tmp_path/'revision_pins.json')
    common.write_json(tmp_path/'parameterized_pins.json', {'schema': parameterized.SCHEMA, 'tasks': {}})
    monkeypatch.setattr(parameterized, 'PINS', tmp_path/'parameterized_pins.json')
    definitions = {task: task_definition('laundry_room', task) for task in TASK_IDS}
    rows = deepcopy(RELEASES)
    for task in TASK_IDS:
        for configuration, option in rows[task]['configurations'].items():
            old = deepcopy(option)
            folder = tmp_path/task/configuration
            folder.mkdir(parents=True)
            common.write_json(folder/'task.json', {'public_rules': old['rules']['contract']})
            common.write_json(folder/'release_manifest.json', {
                'schema': SCHEMA, 'schema_version': VERSION, 'task_id': task,
                'configuration_id': configuration, 'seed_values': list(range(10)),
                'base_release': {**old['source_release'], 'schema': old['release_schema']},
                'files_sha256': {'task.json': common.file_hash(folder/'task.json')}})
            option.update(source_release={'path': str(folder),
                'manifest_sha256': common.file_hash(folder/'release_manifest.json')},
                release_schema=SCHEMA, release_schema_version=VERSION,
                qualification=deepcopy(catalog.QUALIFICATION))
        rows[task]['base_release'] = deepcopy(rows[task]['configurations']['dual_opposed']['source_release'])
    common.write_json(path, {'schema': 'laundry_seed_catalog.v1', 'tasks': rows})
    return path, rows, definitions


def test_all_five_pins_preserve_public_installation_labels_and_physical_metadata(pins):
    _, _, definitions = pins
    for task in TASK_IDS:
        bound = catalog.bind_definition(definitions[task])
        public = project_robot_configurations(bound['robot_configurations'])
        assert set(public) == {'single_arm', 'dual_same_side', 'dual_opposed'}
        assert bound['default_configuration_id'] == 'dual_opposed'
        assert bound['layout'] == definitions[task]['layout']
        for configuration, row in public.items():
            original = definitions[task]['robot_configurations'][configuration]
            assert row['label_zh'] == original['label_zh']
            assert row['description_zh'] == original['description_zh']
            for key in ('robot_requirements', 'layout', 'objects'):
                assert bound['robot_configurations'][configuration][key] == original[key]
            assert bound['robot_configurations'][configuration]['qualification'] == catalog.QUALIFICATION


@pytest.mark.parametrize('change', ['default', 'qualification', 'robot', 'layout', 'objects',
                                  'rules', 'injected_description', 'parent_path', 'parent_hash'])
def test_selector_rejects_wrong_parent_robot_or_inherited_validation(pins, change):
    path, rows, definitions = pins
    row = rows['towel_folding']; option = row['configurations']['single_arm']
    if change == 'default':
        row['default_configuration_id'] = 'single_arm'; row['base_release'] = option['source_release']
    elif change == 'qualification':
        option['qualification']['physics_accepted'] = True
    elif change == 'robot':
        option['robot_requirements']['articulations'] = 2
    elif change == 'layout':
        option['layout']['robot_layout_id'] = 'wrong_installation'
    elif change == 'objects':
        option['objects']['unexpected'] = {'id': 'extra'}
    elif change == 'rules':
        option['rules']['contract']['unexpected'] = True
    elif change == 'injected_description':
        option['label_zh'] = 'unreviewed label'
    else:
        from pathlib import Path
        manifest_path = Path(option['source_release']['path'])/'release_manifest.json'
        manifest = common.read_json(manifest_path)
        manifest['base_release']['path' if change == 'parent_path' else 'manifest_sha256'] = (
            '/another/parent' if change == 'parent_path' else '0'*64)
        common.write_json(manifest_path, manifest)
        option['source_release']['manifest_sha256'] = common.file_hash(manifest_path)
    common.write_json(path, {'schema': 'laundry_seed_catalog.v1', 'tasks': rows})
    with pytest.raises(ValueError):
        catalog.bind_definition(definitions['towel_folding'])
