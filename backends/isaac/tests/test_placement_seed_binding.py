"""Fixed cooperative installation and legacy publication identity boundaries."""
from copy import deepcopy

import pytest

from scenes.workrooms.initial_randomization import configurations, selection
from scenes.workrooms.initial_randomization import catalog
from scenes.workrooms.initial_randomization.release import _parent_configuration
from scenes.workrooms.initial_randomization import physical_inputs


def test_fixed_dual_task_keeps_original_robot_and_no_new_configuration_menu(monkeypatch):
    from scenes.workrooms.mechanical_assembly.tasks.cooperative_insertion.definition import TASK
    original = deepcopy(TASK)
    release = {'path': '/new/immutable/cooperative', 'manifest_sha256': 'a' * 64}
    row = {'family': original['family'], 'parent_default_release': original['source_release'],
           'parents': {'dual_same_side': original['source_release']},
           'releases': {'dual_same_side': release}}
    monkeypatch.setattr(catalog, 'read_json', lambda path: {'cooperative_insertion': row})
    selected = catalog.bind_definition(original)
    assert original == TASK
    assert selected['source_release'] == release
    assert selected['robot_requirements'] == original['robot_requirements']
    assert selected['rules'] == original['rules']
    assert not selected.get('robot_configurations')
    assert configurations('cooperative_insertion') == ('dual_same_side',)


@pytest.mark.parametrize('configuration', ['single_arm', 'dual_opposed'])
def test_cooperative_seed_selection_rejects_unpublished_installations(configuration):
    with pytest.raises(ValueError, match='installation'):
        selection('cooperative_insertion', 2, configuration)


def test_cooperative_parent_identity_cannot_relabel_a_single_robot():
    scene = {'robot': {'hand_id': 'sharpa_w01_right'}}
    with pytest.raises(ValueError, match='fixed dual robot'):
        _parent_configuration(scene, 'cooperative_insertion')
    assert _parent_configuration(scene, 'peg_insertion') == 'single_arm'


def test_new_scope_does_not_rebind_unpublished_or_existing_tasks(monkeypatch):
    from scenes.workrooms.study.tasks.book_retrieval.definition import TASK
    original = deepcopy(TASK)
    monkeypatch.setattr(catalog, 'read_json', lambda path: {})
    assert catalog.bind_definition(original) == TASK


@pytest.mark.parametrize('root,body,expected', [
    ('/original_bolt', '/original_bolt/bolt', '/World/Objects/bolt/original_bolt/bolt'),
    ('/scaled_root', '/scaled_root', '/World/Objects/bolt/scaled_root'),
])
def test_anchored_body_identity_preserves_source_ancestor(root, body, expected, monkeypatch):
    asset = {'placement_form': 'anchored', 'usd_default_prim': root, 'source_rigid_body_prim': body}
    monkeypatch.setattr(physical_inputs, 'read_json', lambda path: {'assets': {'source': asset}})
    scene = {'objects': [{'id': 'bolt', 'asset_id': 'source', 'asset_root': '/frozen/catalog'}]}
    assert physical_inputs.actor_paths(scene) == {'bolt': expected}
    physical_inputs.check_actor_inventory(scene, scene, {'bolt': {'prim_path': expected}})
    with pytest.raises(ValueError, match='actual scene declarations'):
        physical_inputs.check_actor_inventory(scene, scene, {'bolt': {'prim_path': '/World/Objects/bolt'}})


def test_anchored_body_cannot_escape_source_root(monkeypatch):
    asset = {'placement_form': 'anchored', 'usd_default_prim': '/source', 'source_rigid_body_prim': '/other/body'}
    monkeypatch.setattr(physical_inputs, 'read_json', lambda path: {'assets': {'source': asset}})
    with pytest.raises(ValueError, match='escapes'):
        physical_inputs.object_actor_path({'id': 'bolt', 'asset_id': 'source', 'asset_root': '/frozen/catalog'})


def test_actual_fastener_uses_original_child_actor_without_changing_source():
    from scenes.workrooms.catalog import task_definition
    from scenes.workrooms.initial_randomization.common import read_json
    from pathlib import Path
    definition = task_definition('mechanical_assembly', 'fastener_assembly')
    scene = read_json(Path(definition['source_release']['path'])/'scene.json')
    assert physical_inputs.actor_paths(scene) == {
        'screwdriver': '/World/Objects/screwdriver',
        'bolt': '/World/Objects/bolt/sharpa_threaded_bolt/allen_bolt'}
