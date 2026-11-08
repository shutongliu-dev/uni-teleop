"""The resolution comparison must not change actual spoon or support solids."""
from copy import deepcopy
from pxr import Usd
from scenes.workrooms.kitchen.assets import author_task
from scripts.validation.kitchen_asset_identity import _property_graph


def test_spoon_sdf_resolution_changes_only_two_sampling_attributes():
    stages = [Usd.Stage.CreateInMemory(), Usd.Stage.CreateInMemory()]
    metas = [author_task(stage, 'utensil_sorting', prop_profile='household_v1',
        utensil_contact_profile=profile) for stage, profile in zip(stages,
        ('floor_grid_2mm_convex_v2', 'floor_grid_2mm_convex_sdf768_v4'))]
    before, after = deepcopy(metas)
    before.pop('utensil_contact_profile'); after.pop('utensil_contact_profile')
    changed = set()
    for row in after['objects']:
        if row['role'] == 'spoon':
            receipt = row.pop('sdf_sampling_candidate')
            assert receipt['resolution'] == 768 and receipt['previous_resolution'] == 384
            assert receipt['nominal_grid_spacing_after_m']*2 == receipt['nominal_grid_spacing_before_m']
            changed.add(receipt['collider_path'])
    assert before == after and len(changed) == 2
    graphs = [_property_graph(stage, metas[0]['actors'].values()) for stage in stages]
    assert graphs[0].keys() == graphs[1].keys()
    differences = []
    for path, old in graphs[0].items():
        for key in old:
            if old[key] != graphs[1][path][key]:
                assert key == 'attributes' and path in changed
                for name, value in old[key].items():
                    if value != graphs[1][path][key][name]:
                        assert name == 'physxSDFMeshCollision:sdfResolution'
                        differences.append(path)
    assert set(differences) == changed
