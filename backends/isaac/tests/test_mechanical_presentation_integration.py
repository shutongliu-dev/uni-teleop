"""Current-parent and failure checks for the restored mechanical appearance path."""
from copy import deepcopy
from pathlib import Path

import pytest

from bench.isaacsim.release import digest
from bench.paths import load_json
from scenes.workrooms.mechanical_assembly import presentation_release as release
from scenes.workrooms.mechanical_assembly.table_distractors import validate_distractors


@pytest.fixture(scope='module')
def current_parents():
    return release.parent_releases()


@pytest.fixture(scope='module')
def loose_composition(current_parents):
    root = Path(current_parents['fastener_assembly']['path'])
    scene, task = (load_json(root/name) for name in ('scene.json', 'task.json'))
    changed, updated, receipt = release.presentation_inputs(scene, task)
    return scene, task, changed, updated, receipt


def test_default_parents_use_updated_task_candidates(current_parents):
    assert Path(current_parents['shaft_spacer_assembly']['path']).name == 'shaft_spacer_candidate_v2'
    assert Path(current_parents['fastener_assembly']['path']).name == 'loose_bolt_candidate_v10_pgs'
    assert current_parents['gear_train_assembly']['manifest_sha256'] == '62bd44368d8d42a36885d29272d7ef0dae2582acf84da2f8faee6d6fe37c96f5'
    altered = deepcopy(current_parents)
    altered['fastener_assembly']['manifest_sha256'] = '0'*64
    with pytest.raises(ValueError, match='Frozen input changed'):
        release.parent_releases(parent_map=altered)


@pytest.mark.parametrize('task_id', ['shaft_spacer_assembly', 'fastener_assembly', 'gear_train_assembly'])
def test_current_parent_supports_reset_and_scoring_are_preserved(current_parents, task_id):
    root = Path(current_parents[task_id]['path'])
    scene, task = (load_json(root/name) for name in ('scene.json', 'task.json'))
    changed, updated, receipt = release.presentation_inputs(scene, task)
    assert changed['fixtures'] == scene['fixtures']
    assert changed['objects'][:len(scene['objects'])] == scene['objects']
    assert updated['evaluation'] == task['evaluation']
    assert updated.get('reset_conditions') == task.get('reset_conditions')
    assert receipt['supply_reset']['removed_fixture_ids'] == []
    assert len(validate_distractors(changed, updated)) == 5
    release.check_parent_physics(scene, task, changed, updated, receipt)


@pytest.mark.parametrize('mutation', ['physics', 'object', 'fixture', 'task'])
def test_visual_revision_cannot_change_parent_physics_or_rules(loose_composition, mutation):
    scene, task, changed, updated, receipt = deepcopy(loose_composition)
    if mutation == 'physics':
        changed['physics']['gravity_m_s2'] = [0., 0., 0.]
    elif mutation == 'object':
        changed['objects'][0]['usd_physics_material_fallback']['static_friction'] *= 2
    elif mutation == 'fixture':
        changed['fixtures'][1]['position_m'][0] += .01
    else:
        updated['evaluation']['unexpected_override'] = True
    with pytest.raises(ValueError, match='parent physics'):
        release.check_parent_physics(scene, task, changed, updated, receipt)


@pytest.mark.parametrize('mutation', ['missing', 'scored', 'roles', 'asset', 'extra'])
def test_unscored_inventory_requires_matching_task_and_scene(loose_composition, mutation):
    _, _, scene, task, _ = deepcopy(loose_composition)
    if mutation == 'missing':
        del task['distractors']
    elif mutation == 'scored':
        task['distractors']['scored'] = True
    elif mutation == 'roles':
        task['distractors']['roles']['table_short_bolt'] = 'wrong_asset'
    elif mutation == 'asset':
        scene['objects'][-1]['asset_id'] = 'wrong_asset'
    else:
        scene['objects'].append(dict(scene['objects'][-1], id='undeclared_object'))
    with pytest.raises(ValueError, match='Distractor'):
        validate_distractors(scene, task)


def test_texture_replay_rejects_changed_and_missing_source_hashes(tmp_path, monkeypatch):
    from scenes.workrooms.mechanical_assembly import presentation_materials
    texture = tmp_path/'map.png'
    texture.write_bytes(b'original texture')
    monkeypatch.setattr(presentation_materials, 'required_texture_files', lambda: [texture])
    source = {'dependencies_sha256': {str(texture): digest(texture)}}
    assert release.checked_texture_dependencies(source) == source['dependencies_sha256']
    texture.write_bytes(b'modified texture')
    with pytest.raises(ValueError, match='Frozen input changed'):
        release.checked_texture_dependencies(source)
    with pytest.raises(ValueError, match='original texture'):
        release.checked_texture_dependencies({})


def test_loose_release_inventory_includes_clutter_license_and_presentation(loose_composition, tmp_path):
    from scenes.workrooms.mechanical_assembly.tasks.fastener_assembly.release import _asset_inventory
    _, _, scene, task, _ = deepcopy(loose_composition)
    layer = tmp_path/'visual.usda'
    layer.write_text('#usda 1.0\ndef Xform "Asset" {}\n')
    scene['objects'][0]['presentation'] = {
        'schema': 'asset_presentation.v1', 'usd_path': str(layer), 'sha256': digest(layer),
        'prim_path': '/Asset', 'dependencies_sha256': {},
    }
    inventory = _asset_inventory(scene)
    clutter = Path(task['distractors']['source_asset_root'])
    for path in [clutter/'SOURCE.json', clutter/'LICENSE.txt', layer]:
        assert inventory[str(path)]['sha256'] == digest(path)


def test_gallery_preserves_task_supply_rests(loose_composition, tmp_path):
    from bench.isaacsim.release import dump
    from scripts.media.publish_mechanical_presentation import release_receipt
    _, _, scene, task, audit = deepcopy(loose_composition)
    for name, value in [('scene.json', scene), ('task.json', task), ('presentation_audit.json', audit)]:
        dump(tmp_path/name, value)
    manifest = {'files': {name: {'sha256': digest(tmp_path/name)}
                         for name in ('scene.json', 'task.json', 'presentation_audit.json')}}
    dump(tmp_path/'release_manifest.json', manifest)
    receipt = release_receipt('fastener_assembly', {
        'release': str(tmp_path), 'release_manifest_sha256': digest(tmp_path/'release_manifest.json'),
    })
    assert receipt['display_supply_pads'] == []
    assert {'bolt_supply_rest', 'tool_supply_rest'} <= {row['id'] for row in scene['fixtures']}
