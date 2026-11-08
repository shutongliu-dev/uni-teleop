"""Obsolete SDFs must be excluded without changing replacement or body physics."""
from types import SimpleNamespace
from copy import deepcopy
import pytest
from pxr import Usd
from scenes.workrooms.kitchen.assets import author_task
from scenes.workrooms.kitchen.utensil_convex import deactivate_replaced_colliders
from scripts.validation.kitchen_asset_identity import _property_graph


@pytest.mark.parametrize('task', ['utensil_sorting', 'table_clearing'])
def test_deactivating_legacy_shapes_preserves_all_original_layer_properties(task):
    stage = Usd.Stage.CreateInMemory()
    metadata = author_task(stage, task, prop_profile='household_v1',
                          utensil_contact_profile='floor_grid_2mm_fullconvex_v5')
    before = _property_graph(stage, metadata['actors'].values())
    original_meta = deepcopy(metadata)
    deactivate_replaced_colliders(SimpleNamespace(stage=stage, meta=metadata))
    after = _property_graph(stage, metadata['actors'].values())
    changed = {row['deactivated_legacy_collision_prim'] for row in metadata['objects']
               if 'deactivated_legacy_collision_prim' in row}
    assert changed
    assert before.keys() == after.keys()
    for path, old in before.items():
        if path in changed:
            old['metadata']['active'] = False
            assert not stage.GetPrimAtPath(path).IsActive()
            assert stage.GetPrimAtPath(path).GetAttribute('physics:collisionEnabled').Get() is False
        assert old == after[path], path
    for row in metadata['objects']:
        row.pop('deactivated_legacy_collision_prim', None)
    assert metadata == original_meta
    with pytest.raises(ValueError, match='Expected disabled original'):
        deactivate_replaced_colliders(SimpleNamespace(stage=stage, meta=metadata))
