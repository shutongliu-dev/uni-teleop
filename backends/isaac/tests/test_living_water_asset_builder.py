"""CPU candidate construction; mocked snapshot receipts do not qualify physics."""
import json
from pathlib import Path

import numpy as np
import pytest
from pxr import Usd, UsdGeom, UsdPhysics

from bench.paths import load_json
from scenes.workrooms.living_room.tasks.water_pouring import build_assets as builder


@pytest.fixture
def output_root(tmp_path, monkeypatch):
    root = tmp_path/'shared_assets/water_pouring'
    monkeypatch.setattr(builder, 'ASSET_ROOT', root)
    return root


@pytest.fixture
def authored_output(output_root, tmp_path, monkeypatch):
    # Only the pre-existing freeze guard is replaced. USD authoring, export,
    # dependency resolution, hashes and metadata below use the real code.
    source = tmp_path/'snapshot'
    source.mkdir()
    (source/'source_snapshot_manifest.json').write_text('{"fixture": "CPU guard double"}\n')
    monkeypatch.setattr(builder, 'PROJECT', source)
    monkeypatch.setattr(builder, 'verify_if_snapshot', lambda *a, **kw:
        {'verified': True, 'snapshot_root': str(source), 'native_validation_performed': False})
    return output_root/'candidate'


@pytest.mark.parametrize('spacing,sleep,count', [(.003, None, 9259), (.008, 0., 488), (.01, .125, 250)])
def test_build_keeps_fixed_mass_and_records_explicit_resolution_and_sleep(authored_output, spacing, sleep, count):
    manifest = builder.build_assets(authored_output, particle_spacing_m=spacing, vessel_sleep_threshold=sleep)
    metadata = load_json(authored_output/'metadata.json')
    stage = Usd.Stage.Open(manifest['initial_usd'])
    water = metadata['water']
    points = stage.GetPrimAtPath(water['particles_path'])
    xyz = np.asarray(UsdGeom.Points(points).GetPointsAttr().Get())
    assert len(xyz) == water['particle_count'] == count
    assert len(np.unique(xyz, axis=0)) == count
    assert list(UsdGeom.Points(points).GetIdsAttr().Get()) == list(range(count))
    assert UsdPhysics.MassAPI(points).GetMassAttr().Get() == pytest.approx(.25)
    assert water['particle_mass_kg']*count == pytest.approx(.25)
    assert water['particle_spacing_m'] == spacing and water['initial_volume_ml'] == 250.
    for name, mass in [('kettle', .32), ('cup', .105)]:
        prim = stage.GetPrimAtPath(metadata['containers'][name]['path'])
        assert UsdPhysics.MassAPI(prim).GetMassAttr().Get() == pytest.approx(mass)
        attr = prim.GetAttribute('physxRigidBody:sleepThreshold')
        if sleep is None:
            assert not attr.HasAuthoredValueOpinion()
        else:
            assert attr.Get() == pytest.approx(sleep)
    assert not stage.GetPrimAtPath('/World/PhysicsScene')
    assert stage.GetDefaultPrim().GetPath() == '/World'
    assert json.loads(stage.GetPrimAtPath('/World/LivingTask').GetCustomDataByKey('candidateMetadata')) == metadata
    assert metadata['asset_build'] == manifest['build_parameters']
    assert manifest['build_parameters']['vessel_sleep_threshold'] == sleep
    assert manifest['schema'] == 'living_task_objects_candidate.v1'
    assert manifest['release_status'] == 'scene_candidate_not_bench_release'
    assert manifest['native_manipulation_qualified'] is False
    assert manifest['initial_usd_sha256'] == builder.digest(authored_output/'initial.usda')
    assert manifest['metadata_sha256'] == builder.digest(authored_output/'metadata.json')
    assert manifest['source_snapshot_manifest_sha256'] == builder.digest(manifest['source_snapshot_manifest_path'])
    assert all(Path(path).is_absolute() and builder.digest(path) == row['sha256']
               for path, row in manifest['dependencies'].items())


@pytest.mark.parametrize('kwargs', [
    {'particle_spacing_m': .0029}, {'particle_spacing_m': .0101},
    {'particle_spacing_m': float('nan')}, {'particle_spacing_m': True},
    {'vessel_sleep_threshold': -1.}, {'vessel_sleep_threshold': float('inf')},
    {'vessel_sleep_threshold': True},
])
def test_invalid_build_parameters_leave_no_output(authored_output, kwargs):
    with pytest.raises(ValueError):
        builder.build_assets(authored_output, **kwargs)
    assert not authored_output.exists()


def test_fresh_output_and_shared_root_boundary(authored_output):
    builder.build_assets(authored_output)
    original = (authored_output/'manifest.json').read_bytes()
    with pytest.raises(FileExistsError):
        builder.build_assets(authored_output, particle_spacing_m=.008)
    assert (authored_output/'manifest.json').read_bytes() == original
    for path in (builder.ASSET_ROOT, builder.ASSET_ROOT.parent/'outside'):
        with pytest.raises(ValueError, match='shared water_pouring asset root'):
            builder.build_assets(path)
    dangling = builder.ASSET_ROOT/'dangling'
    dangling.symlink_to(builder.ASSET_ROOT/'missing')
    with pytest.raises(FileExistsError):
        builder.build_assets(dangling)
    escaping = builder.ASSET_ROOT/'escape'
    escaping.symlink_to(builder.ASSET_ROOT.parent, target_is_directory=True)
    with pytest.raises(ValueError, match='shared water_pouring asset root'):
        builder.build_assets(escaping/'new')


def test_editable_source_is_rejected_before_output(output_root):
    with pytest.raises(ValueError, match='complete frozen source snapshot'):
        builder.build_assets(output_root/'unfrozen')
    assert not output_root.exists()
