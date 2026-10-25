"""Exact board boundary and explicit task binding; no native qualification."""
from copy import deepcopy
import json
import pytest

from bench.isaacsim.release import digest
from bench.paths import load_json
from scripts.assets.prepare_card_room_board_mesh import box_surface, geometry_review, prepare, SOURCE
from scenes.workrooms.card_room.tasks.task0.assets import bind_board_collision_asset


@pytest.mark.parametrize('cell', [.005, .01])
def test_box_grid_preserves_closed_surface_without_bevels(cell):
    vertices, faces = box_surface(cell)
    result = geometry_review(vertices, faces)
    assert result['all_triangles_on_original_box_faces']
    assert result['bounds_m'] == [[-.23, -.23, 0.], [.23, .23, .012]]
    assert result['volume_m3'] == pytest.approx(.46*.46*.012, abs=1e-15)
    moved = vertices.copy()
    moved[0, 2] += .0001
    with pytest.raises(ValueError, match='complete closed box'):
        geometry_review(moved, faces)


@pytest.fixture
def candidate(tmp_path):
    from scenes.workrooms.card_room.tasks.task0.task import build_task_inputs
    if not (SOURCE/'manifest.json').exists():
        pytest.skip('Original shared board asset unavailable')
    original_digest = digest(SOURCE/'manifest.json')
    prepared = prepare(tmp_path/'candidate')
    selection = {key: prepared[key] for key in ('asset_root', 'asset_id', 'manifest_sha256')}
    assert digest(SOURCE/'manifest.json') == original_digest
    return build_task_inputs(seed=7), selection


def test_candidate_only_changes_board_asset_binding(candidate):
    from scenes.workrooms.card_room.tasks.task0.task import build_task_inputs
    from scenes.workrooms.card_room.tasks.task0.release import _asset_inventory
    original, selection = candidate
    updated = build_task_inputs(seed=7, board_collision_asset=selection)
    assert updated['task'] == original['task']
    assert updated['initial_state'] == original['initial_state']
    scene = deepcopy(updated['scene'])
    provenance = scene['card_room_placement_provenance'].pop('board_collision_asset')
    assert provenance['selection'] == selection
    assert provenance['bindings'][0]['contact_actor_path_unchanged'] == '/World/Fixtures/board/collision_000'
    scene['fixtures'][0] = original['scene']['fixtures'][0]
    assert scene == original['scene']
    inventory = _asset_inventory(updated['scene'])
    assert selection['asset_root']+'/collision_surface.obj' in inventory
    assert str(SOURCE/'LICENSE.txt') in inventory


@pytest.mark.parametrize('field,value', [
    ('dynamic', True), ('mass_kg', .1), ('bounds_max_m', [.23, .23, .013]),
    ('contact_offset_m', .002), ('rest_offset_m', -.001),
    ('visual_parts', []), ('material_prior', {}), ('source_dependencies_sha256', {})])
def test_binding_rejects_changed_physical_or_visual_contract(candidate, field, value):
    from pathlib import Path
    original, selection = candidate
    manifest = Path(selection['asset_root'])/'manifest.json'
    data = load_json(manifest)
    data['assets'][selection['asset_id']][field] = value
    manifest.write_text(json.dumps(data))
    selection = {**selection, 'manifest_sha256': digest(manifest)}
    before = deepcopy(original['scene'])
    with pytest.raises(ValueError):
        bind_board_collision_asset(original['scene'], selection)
    assert original['scene'] == before


def test_board_binding_requires_fixed_actor_observability(candidate):
    original, selection = candidate
    original['scene']['isaacsim'].pop('static_mesh_contact_mode')
    with pytest.raises(ValueError, match='observability'):
        bind_board_collision_asset(original['scene'], selection)
