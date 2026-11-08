"""Task-owned folding friction is not a shared material or controller change."""
from copy import deepcopy

from scenes.workrooms.laundry_room import garment_folding_assets as assets
from scenes.workrooms.laundry_room.composition import build_task_inputs


def test_explicit_single_garment_friction_reaches_native_scene_only(monkeypatch):
    baseline = build_task_inputs('shirt_folding')
    metadata = deepcopy(baseline['task']['asset_metadata'])
    metadata['cloths'][0]['friction'] = .25
    monkeypatch.setattr(assets, 'candidate_metadata', lambda *args, **kwargs: metadata)
    candidate = build_task_inputs('shirt_folding')
    actual = deepcopy(candidate['scene']['cloths'][0])
    assert actual.pop('friction') == .25
    original = deepcopy(baseline['scene']['cloths'][0])
    assert original.pop('friction') == .45
    assert actual == original
    assert candidate['scene']['isaacsim'] == baseline['scene']['isaacsim']
    assert build_task_inputs('towel_folding')['scene']['cloths'][0]['friction'] == .45
    assert all(row['friction'] == .45 for row in build_task_inputs('shirt_stacking')['scene']['cloths'])
