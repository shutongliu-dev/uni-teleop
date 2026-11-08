"""Actual-parent CPU checks; no native/contact or manipulation qualification."""
from copy import deepcopy
import json
from pathlib import Path
import shutil

import numpy as np
import pytest
from pxr import Usd, UsdGeom, UsdPhysics, UsdShade

from bench.paths import DATA_ROOT, load_json
from scenes.workrooms.parameterized.spec import sample_parameters
from scenes.workrooms.living_room import seed_variants as living
from scenes.workrooms.living_room.randomization import cases as living_cases
from scenes.workrooms.living_room.randomization import parameterized as living_parameters
from scenes.workrooms.laundry_room.randomization import (
    appearance, author, cases, common, release, PARAMETERIZED_VERSION)
from scenes.workrooms.laundry_room.robot_configuration_catalog import RELEASES


def cloth_parent(task='towel_folding', configuration='single_arm'):
    return Path(RELEASES[task]['configurations'][configuration]['source_release']['path'])


def cloth_plan(task, configuration, seed, metadata):
    for attempt in range(100):
        try:
            return cases.case_plan(task, configuration, seed, metadata,
                generator_version=PARAMETERIZED_VERSION, attempt=attempt)
        except ValueError as error:
            if 'reject this attempt' not in str(error):
                raise
    raise AssertionError('No support-valid candidate in bounded test search')


@pytest.mark.parametrize('task,version', [('remote_operation','v6'), ('storage_box','v2'), ('water_pouring','v2')])
def test_living_actual_mass_inertia_and_collider_material_binding(tmp_path, task, version):
    root = Path(f'{DATA_ROOT}/assets/task_parts/living_room')/task/version
    original = Usd.Stage.Open(str(root/'initial.usda'))
    metadata = load_json(root/'metadata.json')
    before = original.GetRootLayer().ExportToString()
    plan = living_cases.plan_case(task, metadata, 1, generator_version=PARAMETERIZED_VERSION)
    geometry_only = Usd.Stage.Open(original.Flatten()); geometry_metadata = deepcopy(metadata)
    living.apply_geometry(geometry_only, geometry_metadata, task, plan['geometry_variant'])
    artifact = living_parameters.appearance_assets(tmp_path/'appearance', plan['parameters'])
    stage, current, _, visual, _ = living.derive_stage(original, metadata, plan, appearance_artifact=artifact)
    factor = plan['parameters']['physics']['density_factor']
    for name, path in living.actors(metadata, task).items():
        old, new = [UsdPhysics.MassAPI(s.GetPrimAtPath(path)) for s in (geometry_only, stage)]
        assert new.GetMassAttr().Get() == pytest.approx(old.GetMassAttr().Get()*factor, rel=1e-6)
        if old.GetDiagonalInertiaAttr().HasAuthoredValue():
            np.testing.assert_allclose(new.GetDiagonalInertiaAttr().Get(),
                np.asarray(old.GetDiagonalInertiaAttr().Get())*factor, rtol=2e-6, atol=1e-12)
        assert current['mass_properties'][name]['mass_kg'] == new.GetMassAttr().Get()
        assert next(r for r in current['static_body_geometry'] if r['object_id']==name)['nominal_mass_kg'] == new.GetMassAttr().Get()
    report = current['parameterized_physics']
    for path, expected in report['collider_material_bindings'].items():
        material, _ = UsdShade.MaterialBindingAPI(stage.GetPrimAtPath(path)).ComputeBoundMaterial(materialPurpose='physics')
        assert str(material.GetPath()) == expected
        row = next(value for value in report['materials'].values() if value['path']==expected)
        api = UsdPhysics.MaterialAPI(material.GetPrim())
        assert api.GetDynamicFrictionAttr().Get() == pytest.approx(row['baseline']['dynamic_friction']*plan['parameters']['physics']['friction_factor'])
    assert visual['meshes']
    assert original.GetRootLayer().ExportToString() == before
    if task == 'water_pouring':
        assert current['water'] == metadata['water']
        for s in (original, stage):
            material, _ = UsdShade.MaterialBindingAPI(s.GetPrimAtPath('/World/LivingTask/Cup/EtchedMark_0')).ComputeBoundMaterial()
            assert str(material.GetPath()) == '/World/LivingTask/Materials/WarmWhite'
        original_shader, _, _ = UsdShade.Material(original.GetPrimAtPath('/World/LivingTask/Materials/WarmWhite')).ComputeSurfaceSource()
        current_shader, _, _ = UsdShade.Material(stage.GetPrimAtPath('/World/LivingTask/Materials/WarmWhite')).ComputeSurfaceSource()
        assert original_shader.GetInput('diffuseColor').Get() == current_shader.GetInput('diffuseColor').Get()
        assert not current_shader.GetInput('diffuseColor').GetAttr().GetConnections()


@pytest.mark.parametrize('task,version', [('remote_operation','v6'), ('storage_box','v2'), ('water_pouring','v2')])
def test_living_canonical_physics_is_exact(task, version):
    root = Path(f'{DATA_ROOT}/assets/task_parts/living_room')/task/version
    original = Usd.Stage.Open(str(root/'initial.usda'))
    metadata = load_json(root/'metadata.json')
    plan = living_cases.plan_case(task, metadata, 0, generator_version=PARAMETERIZED_VERSION)
    stage, _, _, visual, identity = living.derive_stage(original, metadata, plan)
    assert living.physics_inventory(stage) == living.physics_inventory(original)
    assert not identity['changed_prim_paths'] and visual['id'] == 'canonical'


@pytest.mark.parametrize('task', list(RELEASES))
@pytest.mark.parametrize('configuration', ['single_arm','dual_same_side','dual_opposed'])
def test_cloth_large_actual_transform_and_physics_preserve_material_rest_geometry(task, configuration):
    root = cloth_parent(task, configuration)
    scene, task_input = [load_json(root/(name+'.json')) for name in ('scene','task')]
    metadata = task_input['asset_metadata']; contents = set()
    from scenes.workrooms.laundry_room.randomization.parameterized import material_metadata
    for seed in range(10):
        plan = cloth_plan(task, configuration, seed, metadata)
        transformed = cases.transformed_metadata(metadata, plan)
        current = material_metadata(transformed, scene, plan)
        if seed == 0:
            assert current == metadata
            continue
        delta = np.array(plan['translations_xy_m']['towel'])
        assert np.linalg.norm(delta) >= .04
        assert abs(plan['yaw_rad']) >= np.deg2rad(15)
        for old, new in zip(metadata['cloths'], current['cloths'], strict=True):
            for key in ('rest_vertices_m','triangles','cells_xy','size_m','rest_area_m2','physical_thickness_m'):
                assert new.get(key) == old.get(key)
            p, q = np.asarray(old['initial_vertices_m']), np.asarray(new['initial_vertices_m'])
            assert np.array_equal(p[:,2],q[:,2])
            # A rigid reset preserves actual edge lengths, including initial folds.
            np.testing.assert_allclose(np.linalg.norm(np.diff(p,axis=0),axis=1),np.linalg.norm(np.diff(q,axis=0),axis=1),atol=3e-16)
            if old['id']=='towel_base':
                assert old['initial_vertices_m']==new['initial_vertices_m']
            assert new['mass_kg']==pytest.approx(old['mass_kg']*plan['parameters']['physics']['density_factor'])
        assert current['target_center_xy_m']==metadata['target_center_xy_m']
        contents.add(common.canonical_hash([row['initial_vertices_m'] for row in current['cloths']]))
    assert len(contents)==9


@pytest.fixture
def frozen_author(tmp_path, monkeypatch):
    from scenes.workrooms.laundry_room import fabric_visual_release
    assets = tmp_path/'assets'
    monkeypatch.setattr(common,'ASSET_ROOT',assets)
    monkeypatch.setattr(fabric_visual_release,'ASSETS',assets)
    snapshot = tmp_path/'source'; snapshot.mkdir()
    hashes = {}
    for name in common.PARAMETERIZED_SOURCE_FILES:
        destination = snapshot/name; destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(common.SOURCE_ROOT/name,destination); hashes[name]=common.file_hash(destination)
    common.write_json(snapshot/'source_snapshot_manifest.json',{'schema':'complete_benchmark_source_snapshot_v1',
        'files_sha256':hashes,'test_fixture_only':True})
    provenance = {'source_snapshot':{'snapshot_root':str(snapshot),'runtime_imports_verified':True,'test_fixture_only':True},
        'source_manifest_sha256':common.file_hash(snapshot/'source_snapshot_manifest.json'),'author_sources_sha256':hashes}
    monkeypatch.setattr(common,'source_provenance',lambda **kwargs:deepcopy(provenance))
    return assets


def test_laundry_asset_roundtrip_binds_real_mass_friction_and_private_parameters(frozen_author):
    from scenes.workrooms.laundry_room.randomization.public import project_reset_case
    root=cloth_parent();scene,task=[common.read_json(root/(n+'.json'))for n in ('scene','task')]
    plan=cloth_plan('towel_folding','single_arm',1,task['asset_metadata'])
    pin=appearance.prepare_profile(scene['cloths'][0]['visual_profile'],1,frozen_author/'appearance',parameters=plan['parameters'])
    asset=author.prepare_asset(root,1,frozen_author/'case',visual_profiles={'towel':pin},
        generator_version=PARAMETERIZED_VERSION,attempt=plan['attempt'])
    new_scene,new_task,initial,case,sources,_=release.compose_case(root,asset,1)
    metadata=new_task['asset_metadata'];row=metadata['cloths'][0];spec=new_scene['cloths'][0]
    assert spec['mass_kg']==row['mass_kg']
    assert spec['friction']==row['friction']==pytest.approx(scene['cloths'][0]['friction']*plan['parameters']['physics']['friction_factor'])
    assert metadata==new_task['evaluation']['metadata']
    assert new_task['initial_state']==initial
    assert case['parameterized_physics']==common.read_json(asset)['parameterized_physics']
    assert new_task['public_goal']==task['public_goal']
    assert new_task['evaluation']['thresholds']==task['evaluation']['thresholds']
    for key in ('robot','isaacsim','physics','robot_configuration','workroom'):
        assert new_scene[key]==scene[key]
    assert release.PARAMETERIZED_DESCRIPTION in sources['rules.py']
    projected=project_reset_case(new_task,{'task_id':'towel_folding'})
    assert projected['reset_case']['variation']=='xy_yaw_appearance_and_local_material_parameters'
    assert all(word not in json.dumps(projected) for word in ('friction_factor','initial_vertices','parameterized_physics','translations_xy_m'))
    # A profile is reusable across configuration and layout-attempt streams.
    other=sample_parameters('laundry_room','towel_folding','dual_opposed',1,attempt=2)
    appearance.validate_profile(pin,scene['cloths'][0]['visual_profile'],1,parameters=other)
    tampered=common.read_json(asset)
    tampered['parameterized_physics']['applied']['applied']['towel']['mass_kg']*=1.1
    common.write_json(asset,tampered)
    with pytest.raises(ValueError,match='material recipe'):
        author.load_asset(root,asset,1)


def test_cloth_print_uses_physical_pattern_scale_and_preserves_registration(tmp_path):
    from PIL import Image
    from scenes.workrooms.laundry_room.randomization.parameterized_appearance import render_atlas
    source=tmp_path/'original.png';pixels=np.full((96,192,3),[40,110,180],dtype=np.uint8);pixels[40:44]=0
    Image.fromarray(pixels).save(source)
    parent={'uv_frame':{'origin_xy_m':[0,0],'size_xy_m':[.6,.4],'atlas':'two_sided'}}
    hashes=set()
    for seed in range(1,10):
        parameters=sample_parameters('laundry_room','towel_folding','single_arm',seed)
        result=render_atlas(parent,source,parameters)
        assert np.array_equal(result['albedo'][40:44],pixels[40:44])
        assert (result['albedo']<=pixels).all()
        assert result['roughness'].shape==pixels.shape[:2]
        hashes.add(common.canonical_hash(result['albedo'].tolist()))
    assert len(hashes)==9
