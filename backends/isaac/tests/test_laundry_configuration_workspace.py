"""World extrema cannot replace all four material corners of a folded towel."""
import pytest

from scripts.validation.laundry_configuration_workspace import targets
from scenes.workrooms.laundry_room.assets import candidate_metadata


@pytest.mark.parametrize('task', ['towel_flattening', 'towel_folding', 'towel_stacking'])
def test_towel_probe_keeps_every_material_corner_even_when_world_extrema_coincide(task):
    options = {'material_profile':'towel_4mm_density_v1'}
    if task == 'towel_flattening':
        options['reset_profile'] = 'towel_partial_edge_fold_v1'
        options['cells_xy'] = (60, 40)
    meta = candidate_metadata(task, **options)
    rows = targets({'asset_metadata':meta})
    for cloth in meta['cloths']:
        nx, ny = cloth['cells_xy']
        expected = {0, nx, ny*(nx+1), (ny+1)*(nx+1)-1}
        actual = {int(row['label'].split('/')[-1]) for row in rows if row['label'].startswith(cloth['id']+'/')}
        assert expected <= actual
