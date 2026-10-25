"""CPU catalog integrity tests; native contact qualification has separate records."""
import hashlib
import json
from pathlib import Path

import pytest

from bench.paths import DATA_ROOT
from bench.simulation.isaacsim_backend.asset_catalog import load_task_asset, load_task_catalog


def small_catalog(tmp_path):
    usd = tmp_path / 'source.usda'
    usd.write_text('#usda 1.0\n')
    mesh = tmp_path / 'hull.obj'
    mesh.write_text('v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n')
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    asset = {'backend': 'usd_reference', 'usd_path': str(usd),
             'source_dependencies_sha256': {str(usd): sha(usd)},
             'collision_parts': [{'geometry': 'convex_mesh', 'mesh_path': 'hull.obj', 'sha256': sha(mesh)},
                                 {'geometry': 'usd_convexDecomposition', 'usd_prim_path': '/Object/Concave'}]}
    (tmp_path / 'manifest.json').write_text(json.dumps({'schema': 'collaborator_native_usd_catalog.v1', 'assets': {'sample': asset}}))
    return usd, mesh


def test_usd_catalog_loads_without_pxr_and_keeps_concavity(tmp_path):
    small_catalog(tmp_path)
    asset = load_task_asset('sample', tmp_path, verify_hashes=True)
    assert asset['backend'] == 'usd_reference'
    assert asset['collision_parts'][0]['mesh_path'] == str(tmp_path / 'hull.obj')
    assert asset['collision_parts'][1]['geometry'] == 'usd_convexDecomposition'
    assert 'mesh_path' not in asset  # Rendering references original USD, no lossy conversion.
    assert load_task_catalog(tmp_path)['schema'] == 'collaborator_native_usd_catalog.v1'


@pytest.mark.parametrize('changed', ['source', 'hull'])
def test_asset_dependency_drift_is_rejected(tmp_path, changed):
    source, hull = small_catalog(tmp_path)
    (source if changed == 'source' else hull).write_text('changed')
    with pytest.raises(ValueError, match='dependency hash mismatch'):
        load_task_asset('sample', tmp_path, verify_hashes=True)


def test_catalog_result_is_independent(tmp_path):
    small_catalog(tmp_path)
    first = load_task_asset('sample', tmp_path)
    first['collision_parts'].clear()
    assert len(load_task_asset('sample', tmp_path)['collision_parts']) == 2


def test_shared_priority_assets_are_hash_verified():
    root = Path(f'{DATA_ROOT}/assets/objects/hand_system0/bench_catalog/v1')
    if not (root / 'manifest.json').exists():
        pytest.skip('Shared collaborator catalog is not installed')
    for suffix in ('cube_small', 'blue_cup', 'bench_spoon', 'peg', 'peg_hole', 'rubber_mallet', 'apple'):
        asset = load_task_asset('collab_' + suffix, root, verify_hashes=True)
        assert asset['mass_kg'] > 0
        assert all(b > a for a, b in zip(asset['bounds_min_m'], asset['bounds_max_m']))
        assert asset['collision_parts']
        assert Path(asset['usd_path']).is_file()
    cube = load_task_asset('collab_cube_small', root)
    assert cube['dimensions_m'] == pytest.approx([.032, .04, .06])
