"""Real catalog CPU derivation, never a native/contact qualification."""
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest
from pxr import Usd, UsdGeom, UsdPhysics, UsdShade

from bench.paths import load_json
from scenes.workrooms.parameterized.mechanical_authoring import TASKS, author_catalog, validate_catalog
from scenes.workrooms.parameterized.spec import sample_parameters
from scenes.workrooms.initial_randomization.release import adapter
from bench.simulation.isaacsim_backend.asset_catalog import load_task_asset
from scenes.workrooms.mechanical_assembly.placement_authoring import _catalog_stage
from scenes.workrooms.parameterized.workroom import _diff


def admission_data(before, after):
    return {'scene_changes':_diff(before['scene'],after['scene']),
            'task_changes':_diff(before['task'],after['task']),
            **{key:deepcopy(after[key]) for key in ('actual_body_poses_m','static_geometry',
                'interaction_targets','fixed_support_paths')}}


def test_mechanical_author_source_is_required_for_freeze_and_admission():
    from scenes.workrooms.parameterized.workroom import _author_files
    path='scenes/workrooms/parameterized/mechanical_authoring.py'
    assert path in set(_author_files('mechanical_assembly'))
    assert path not in set(_author_files('kitchen'))


def canonical(task_id):
    source=Path(__file__).resolve().parents[1]
    pins=load_json(source/'scenes/workrooms/initial_randomization/catalog_pins.json')
    root=Path(pins[task_id]['releases']['single_arm']['path'])
    manifest=load_json(root/'release_manifest.json')
    task=load_json(root/manifest['cases']['0']['task'])
    return adapter(task_id).compose_case(manifest['base_release']['path'],
        task['reset_case']['asset_reference']['path'],'single_arm')


@pytest.mark.parametrize('task_id',TASKS)
def test_canonical_is_exact_and_does_not_create_assets(tmp_path,task_id):
    values=canonical(task_id);original=deepcopy(values)
    result=author_catalog(values,sample_parameters('mechanical_assembly',task_id,'single_arm',0),tmp_path/'new')
    assert {k:result[k] for k in values}==values==original
    assert not (tmp_path/'new').exists()
    assert validate_catalog(values,sample_parameters('mechanical_assembly',task_id,'single_arm',0),
        result['applied'],admission_data(values,result))['canonical']


@pytest.mark.parametrize('task_id',TASKS)
def test_real_catalog_mass_friction_layout_and_appearance(tmp_path,task_id):
    from bench.simulation.isaacsim_backend.presentation import apply_presentation
    from scenes.workrooms.mechanical_assembly.placement_variants import catalog_body_path
    values=canonical(task_id);before=deepcopy(values)
    parameters=sample_parameters('mechanical_assembly',task_id,'single_arm',1)
    current=author_catalog(values,parameters,tmp_path/'new')
    assert values==before
    assert current['task']['public_goal']==values['task']['public_goal']
    assert {k:v for k,v in current['task']['evaluation'].items() if k!='initial_objects'}=={
        k:v for k,v in values['task']['evaluation'].items() if k!='initial_objects'}
    for key in ('robot','isaacsim','physics','workroom'):
        assert current['scene'][key]==values['scene'][key]
    for old,new in zip(values['scene']['objects'],current['scene']['objects'],strict=True):
        name=old['id'];a=load_task_asset(old['asset_id'],old['asset_root'],verify_hashes=True)
        b=load_task_asset(new['asset_id'],new['asset_root'],verify_hashes=True)
        assert a.get('backend')==b.get('backend')
        assert catalog_body_path(old)==catalog_body_path(new)==current['actual_body_poses_m'][name]['prim_path']
        assert b['mass_kg']==pytest.approx(a['mass_kg']*parameters['physics']['mass_factor'],rel=2e-6)
        if a['inertia_diag_kg_m2'] is not None:
            np.testing.assert_allclose(b['inertia_diag_kg_m2'],np.asarray(a['inertia_diag_kg_m2'])*parameters['physics']['mass_factor'],rtol=2e-6)
        else:
            assert b['inertia_diag_kg_m2'] is None
            assert current['applied']['body_physics'][name]['inertia_mode']=='native_geometry_computation'
        assert next(v for v in current['static_geometry']['bodies'] if v['object_id']==name)['nominal_mass_kg']==b['mass_kg']
        old_stage,_,_,old_points=_catalog_stage(old)
        stage,root,_,points=_catalog_stage(new)
        np.testing.assert_array_equal(points,old_points)
        assert apply_presentation(stage,root,new)['physical_inputs_unchanged']
        assert current['applied']['appearance']['objects'][name]['meshes']
        if b.get('backend')=='usd_reference':
            source=Usd.Stage.Open(b['usd_path']);api=UsdPhysics.MassAPI(source.GetPrimAtPath(b['source_rigid_body_prim']))
            assert api.GetMassAttr().Get()==b['mass_kg']
            original_source=Usd.Stage.Open(a['usd_path'])
            for prim in original_source.Traverse():
                if prim.HasAPI(UsdPhysics.CollisionAPI):
                    assert UsdPhysics.CollisionAPI(source.GetPrimAtPath(prim.GetPath())).GetCollisionEnabledAttr().Get()==UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get()
            mu=current['applied']['material_physics']['objects'][name]
            assert all('/ParameterizedPhysics/' not in key for key in mu['materials'])
            for collider,material in mu['collider_material_bindings'].items():
                resolved,_=UsdShade.MaterialBindingAPI(source.GetPrimAtPath(collider)).ComputeBoundMaterial('physics')
                assert str(resolved.GetPath())==material
        else:
            assert new['friction']==old.get('friction',.65)*parameters['physics']['friction_factor']
        assert current['task']['evaluation']['initial_objects'][name]=={k:new[k] for k in ('position_m','quaternion_wxyz')}
    # All source bodies and their original local supports move coherently.
    for group in current['applied']['layout']:
        before_positions=np.array([v['position_m'] for v in group['before'].values()])
        after_positions=np.array([v['position_m'] for v in group['after'].values()])
        np.testing.assert_allclose(np.linalg.norm(before_positions[:,None]-before_positions[None,:],axis=-1),
            np.linalg.norm(after_positions[:,None]-after_positions[None,:],axis=-1),atol=1e-14)
        assert np.linalg.norm(group['translation_xy_m'])>=.04
    moving={n for g in current['applied']['layout'] for n in g['fixture_ids']}
    for old,new in zip(values['scene']['fixtures'],current['scene']['fixtures'],strict=True):
        if old['id'] not in moving: assert old==new
    assert current['interaction_targets']==values['interaction_targets']
    data=admission_data(values,current)
    assert validate_catalog(values,parameters,current['applied'],data)['validated']
    invalid=deepcopy(current['applied']);invalid['body_physics'][values['scene']['objects'][0]['id']]['mass_kg']*=parameters['physics']['mass_factor']
    with pytest.raises(ValueError,match='mass'):
        validate_catalog(values,parameters,invalid,data)
    invalid=deepcopy(data);task=deepcopy(current['task']);task['evaluation']['unapproved_threshold']=1.
    invalid['task_changes']=_diff(values['task'],task)
    with pytest.raises(ValueError,match='unsupported edit'):
        validate_catalog(values,parameters,current['applied'],invalid)
    invalid=deepcopy(data);invalid['actual_body_poses_m'][values['scene']['objects'][0]['id']]['position_m'][0]+=.001
    with pytest.raises(ValueError,match='projection'):
        validate_catalog(values,parameters,current['applied'],invalid)
    if task_id=='fastener_assembly':
        import subprocess,sys
        payload=tmp_path/'admission.json';payload.write_text(json.dumps([values,parameters,current['applied'],data]))
        code=('import json,sys; from scenes.workrooms.parameterized.mechanical_authoring import validate_catalog; '
              'validate_catalog(*json.load(open(sys.argv[1]))); '
              'assert not any(n=="pxr" or n.startswith("pxr.") for n in sys.modules)')
        subprocess.run([sys.executable,'-B','-c',code,str(payload)],check=True,cwd=Path(__file__).resolve().parents[1])
