"""Catalog publication never transfers old evidence to changed physical inputs."""
from copy import deepcopy

import pytest

from scenes.workrooms.living_room.randomization import catalog
from scenes.workrooms.robot_configurations import attach_robot_configurations


def fixture_row():
    options = {}
    for configuration in catalog.CONFIGURATIONS:
        options[configuration] = {
            'source_release': {'path': '/inputs/' + configuration, 'manifest_sha256': 'a' * 64},
            'release_schema': catalog.SCHEMA, 'release_schema_version': catalog.VERSION,
            'robot_requirements': {'active_sides': ['right'] if configuration == 'single_arm' else ['left', 'right']},
            'qualification': {'status': 'prepared_unvalidated', 'manipulation': False},
            'objects': {'asset_bindings': 'per-seed physical inputs'},
            'rules': {'contract': {'static': True}}, 'layout': {'configuration_id': configuration},
        }
    return {'base_release': deepcopy(options['single_arm']['source_release']),
            'configurations': options, 'single_arm_layout': {'configuration_id': 'single_arm'}}


def original(task_id='remote_operation'):
    return {'family': 'living_room', 'task_id': task_id, 'source_release': {'path': '/historical'},
            'qualification': {'public_http_smoke': True, 'evidence': '/old/experiment.json'},
            'layout': {'parent_scene': 'same_room'}}


def test_changed_geometry_uses_new_installations_without_inheriting_evidence(monkeypatch):
    row = fixture_row()
    monkeypatch.setattr(catalog, 'RELEASES', {'remote_operation': row})
    old = original()
    selected = catalog.bind_definition(old)
    published = attach_robot_configurations(selected)
    assert old == original()
    assert 'evidence' not in str(published)
    assert 'old/experiment' not in str(published)
    assert published['layout']['parent_scene'] == 'same_room'
    for configuration, option in published['robot_configurations'].items():
        assert option['source_release'] == row['configurations'][configuration]['source_release']
        assert option['qualification']['manipulation'] is False
    published['objects']['asset_bindings'] = 'mutation'
    assert row['configurations']['single_arm']['objects']['asset_bindings'] != 'mutation'


def test_both_tower_tasks_keep_their_original_definition(monkeypatch):
    monkeypatch.setattr(catalog, 'RELEASES', {'remote_operation': fixture_row()})
    for task_id in ('tower_building', 'tower_extraction'):
        task = original(task_id)
        assert catalog.bind_definition(task) is task
    monkeypatch.setattr(catalog, 'RELEASES', {'tower_building': fixture_row()})
    with pytest.raises(ValueError, match='three ordinary'):
        catalog.bind_definition(original('tower_building'))


def test_incomplete_or_wrong_default_installations_fail_closed(monkeypatch):
    row = fixture_row()
    monkeypatch.setattr(catalog, 'RELEASES', {'remote_operation': row})
    row['base_release'] = {'path': '/another'}
    with pytest.raises(ValueError, match='default'):
        catalog.bind_definition(original())
    row['configurations'].pop('dual_opposed')
    with pytest.raises(ValueError, match='three frozen'):
        catalog.bind_definition(original())


def test_real_single_arm_discovery_distinguishes_hand_chirality_from_installation_slot():
    from agent.model_client.scene_tasks import select_task
    from scenes.workrooms.catalog import task_definition

    for task_id in catalog.TASK_IDS:
        definition = task_definition('living_room', task_id)
        public = select_task(task_id, family='living_room').metadata()['robot_configurations']
        single = public['single_arm']
        assert single['robot_requirements']['active_sides'] == ['right']
        assert single['layout']['placements']['right']['base_position_m'] == definition['robot_requirements']['base_positions_m']['front']
