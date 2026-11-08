"""CPU checks on the actual pinned parent assets; no simulator or policy."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys
import types

import numpy as np
import pytest
from pxr import Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

from bench.paths import load_json
from scenes.workrooms.laboratory import seed_variants as lab
from scenes.workrooms.mechanical_assembly import seed_variants as mech
from scenes.workrooms.robot_configuration_catalog import RELEASES
from scenes.workrooms.seed_geometry import canonical_hash,collision_bounds,file_hash


def read(p):return load_json(Path(p))


def parent(family,task,configuration='single_arm'):
    row=RELEASES[family,task]
    return Path((row['base_release'] if configuration=='single_arm' else row['configurations'][configuration]['source_release'])['path'])


def subtree(stage,path):
    layer=Sdf.Layer.CreateAnonymous();Sdf.CopySpec(stage.Flatten(),path,layer,'/Root')
    return layer.ExportToString()


def frozen_evaluator(source,tmp_path):
    name='_cpu_seed_eval_'+str(abs(hash(str(tmp_path))))
    directory=tmp_path/'evaluator';directory.mkdir()
    for file,text in source.items():(directory/file).write_text(text)
    package=types.ModuleType(name);package.__path__=[str(directory)];sys.modules[name]=package
    spec=importlib.util.spec_from_file_location(name+'.sequence_evaluator',directory/'sequence_evaluator.py')
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize('task',lab.TASKS)
@pytest.mark.parametrize('seed',range(10))
def test_actual_lab_seed_geometry_and_three_configuration_contracts(tmp_path,task,seed):
    base=parent('laboratory',task);original=read(base/'task.json');manifest_path=lab.prepare_asset(base,seed,tmp_path/'asset')
    manifest=read(manifest_path);metadata=read(manifest['metadata']);stage=Usd.Stage.Open(manifest['initial_usd'])
    factor=1. if seed==0 else (.96 if seed%2 else 1.04)
    assert set(manifest['actual_body_poses_m'])==set(metadata['actors'])
    assert set(r['object_id'] for r in manifest['static_geometry']['bodies'])==set(metadata['actors'])
    before=Usd.Stage.Open(Usd.Stage.Open(read(base/'scene.json')['native_task_bundle']['usd_path']).Flatten())
    assert subtree(stage,metadata['root_path']+'/DesktopDressing')==subtree(before,metadata['root_path']+'/DesktopDressing')
    if seed==0:
        assert metadata==original['asset_metadata']
        assert manifest['initial_usd_sha256']==read(base/'scene.json')['native_task_bundle']['usd_sha256']
    for row in metadata['objects']:
        source=next(x for x in original['asset_metadata']['objects'] if x['id']==row['id'])
        assert row['mass_kg']==pytest.approx(source['mass_kg']*factor**2,rel=1e-12)
        assert UsdPhysics.MassAPI(stage.GetPrimAtPath(row['body_path'])).GetMassAttr().Get()==pytest.approx(row['mass_kg'],rel=1e-6)
        assert np.allclose(collision_bounds(stage,row['body_path']),np.array(source['local_bounds_m'])*[factor,factor,1.],atol=3e-9)
        assert set(row['collider_paths'])==set(source['collider_paths'])
    if task=='vial_racking':
        p=metadata['task_parameters'];assert p['radial_clearance_m']==pytest.approx(.0025*factor)
        assert p['source_rack']['exposed_vial_above_guide_m']==.0653
        assert p['insert_depth_min_m']==.045
        for source,target in zip(p['source_holes'],p['holes']):
            assert source['radius_m']==target['radius_m']==pytest.approx(.012*factor)
            assert source['initial_pose_xyzw'][:2]==pytest.approx(source['center_m'][:2])
    if seed:
        labels=[x for x in metadata['visual_detail']['labels'] if 'Identity' in x['name']]
        for label in labels:
            mass=next(r['mass_kg'] for r in metadata['objects'] if r['id']==label['body_id'])
            assert format(mass*1000,'.6g')+' g' in label['text']
        assert manifest['visual_variant']['visual_mesh_paths']
        reference=Usd.Stage.CreateInMemory();root=stage.GetDefaultPrim().GetPath() if stage.GetDefaultPrim() else Sdf.Path(metadata['root_path'])
        reference.DefinePrim('/Referenced','Xform').GetReferences().AddReference(manifest['initial_usd'],metadata['root_path'])
        for path in manifest['visual_variant']['visual_mesh_paths']:
            p=reference.GetPrimAtPath(path.replace(metadata['root_path'],'/Referenced',1))
            material,_=UsdShade.MaterialBindingAPI(p).ComputeBoundMaterial()
            assert material and material.ComputeSurfaceSource()[0]
            # Every original collision still has its original physics material relationship.
            assert list(stage.GetPrimAtPath(path).GetRelationship('material:binding:physics').GetTargets())==list(before.GetPrimAtPath(path).GetRelationship('material:binding:physics').GetTargets())
    for config in ('single_arm','dual_same_side','dual_opposed'):
        p=parent('laboratory',task,config);oldscene=read(p/'scene.json');oldtask=read(p/'task.json')
        result=lab.compose_case(p,manifest_path,config)
        for key in ('robot','robots','robot_layout','workroom','robot_control','simulation','physics'):
            assert result['scene'].get(key)==oldscene.get(key)
        assert result['task']['initial_state']==oldtask['initial_state']
        assert result['task'].get('active_sides')==oldtask.get('active_sides')
        assert result['task']['evaluation']['thresholds']==oldtask['evaluation']['thresholds']
        for key in ('native_evidence.py',):assert result['evaluator_sources'][key]==(p/'evaluator'/key).read_text()
        evaluator=frozen_evaluator(result['evaluator_sources'],_directory(tmp_path/config))
        evaluator.configuration(result['task'])
        evaluator.SequenceEvaluator(result['task'])


def _directory(path):path.mkdir();return path


@pytest.mark.parametrize('seed',range(10))
def test_actual_mechanical_all_seeds_keep_sdf_bore_and_three_configs(tmp_path,seed):
    base=parent('mechanical_assembly','parts_sorting');original=read(base/'task.json');oldscene=read(base/'scene.json')
    manifest_path=mech.prepare_asset(base,seed,tmp_path/'asset');manifest=read(manifest_path)
    comparison=read(manifest['metadata'])['source_geometry_comparison']
    assert .00059 < comparison['maximum_sdf_vertex_to_nearest_legacy_vertex_m'] < .00061
    assert comparison['maximum_envelope_coordinate_difference_m'] < 2e-9
    factor=1. if seed==0 else (.96 if seed%2 else 1.04)
    for config in ('single_arm','dual_same_side','dual_opposed'):
        p=parent('mechanical_assembly','parts_sorting',config);old=read(p/'scene.json')
        result=mech.compose_case(p,manifest_path,config);scene=result['scene'];task=result['task']
        assert {k:v for k,v in scene.items() if k not in ('objects','fixtures','scenario')}=={k:v for k,v in old.items() if k not in ('objects','fixtures','scenario')}
        assert [r for r in scene['fixtures'] if not r['id'].startswith('tray_')]==[r for r in old['fixtures'] if not r['id'].startswith('tray_')]
        for source in (p/'evaluator').rglob('*.py'):
            assert result['evaluator_sources'][source.name]==source.read_text()
        for name in mech.NAMES:
            spec=next(r for r in scene['objects'] if r['id']==name)
            asset=read(Path(spec['asset_root'])/'manifest.json')['assets'][spec['asset_id']]
            initial=next(r for r in oldscene['objects'] if r['id']==name)
            oldasset=read(Path(initial['asset_root'])/'manifest.json')['assets'][initial['asset_id']]
            assert asset['mass_kg']==pytest.approx(oldasset['mass_kg']*factor**2,rel=1e-12)
            if name!='ring' or seed==0:
                assert np.allclose(task['evaluation']['geometry_vertices_local_m'][name],np.asarray(original['evaluation']['geometry_vertices_local_m'][name])*[factor,factor,1.])
            assert task['goals'][name]['inner_half_extents_xy_m']==pytest.approx([.065*factor]*2)
            assert spec.get('usd_physics_policy')==initial.get('usd_physics_policy')
            if name=='ring':
                assert asset['geometry_design']['bore_radius_m']==pytest.approx(.0175*factor)
                stage=Usd.Stage.Open(asset['usd_path']);collider=stage.GetPrimAtPath('/Ring/collision')
                assert UsdPhysics.MeshCollisionAPI(collider).GetApproximationAttr().Get()=='sdf'
                assert collider.GetAttribute('physxSDFMeshCollision:sdfResolution').Get()==1024
                points=np.asarray(UsdGeom.Mesh(collider).GetPointsAttr().Get())
                assert np.linalg.norm(points[:,:2],axis=1).min()==pytest.approx(.0175*factor,abs=3e-9)
                if seed:assert manifest['visual_variant']['visual_mesh_paths']
            else:
                assert len(asset['collision_parts'])==len(oldasset['collision_parts'])
                assert spec['collision_mesh_coordinate_scale']==1000.
                for part in [asset,*asset['visual_parts'],*asset['collision_parts']]:
                    assert file_hash(Path(spec['asset_root'])/part['mesh_path'])==part['sha256']
            if seed==0:
                assert spec==initial
        assert task['evaluation']['initial_objects']=={s['id']:{k:s[k] for k in ('position_m','quaternion_wxyz')} for s in scene['objects']}


def test_tampering_and_overwrite_rejected(tmp_path):
    p=parent('laboratory','sample_sorting');m=lab.prepare_asset(p,1,tmp_path/'asset')
    with pytest.raises(FileExistsError):lab.prepare_asset(p,1,tmp_path/'asset')
    metadata=Path(read(m)['metadata']);metadata.write_text(metadata.read_text()+' ')
    with pytest.raises(ValueError,match='identity differs'):lab.compose_case(p,m,'single_arm')


def test_nominal_mass_opt_in_and_unknown_inventory_fail_closed(tmp_path):
    p=parent('laboratory','sample_sorting');m=lab.prepare_asset(p,1,tmp_path/'asset');c=lab.compose_case(p,m,'single_arm')
    module=frozen_evaluator(c['evaluator_sources'],tmp_path)
    wrong=deepcopy(c['task']);wrong['evaluation']['metadata']['geometry_variant']['schema']='unknown'
    with pytest.raises(ValueError,match='nominal object masses differ'):module.configuration(wrong)
    wrong=deepcopy(c['task']);wrong['evaluation']['metadata']['objects'].pop()
    with pytest.raises(ValueError):module.configuration(wrong)


def test_full_tensor_and_com_match_original_affine_prior(tmp_path):
    from pxr import Gf
    from scenes.workrooms.seed_geometry import bake_body_scale
    task=read(parent('laboratory','sample_sorting')/'task.json')
    source=read(parent('laboratory','sample_sorting')/'scene.json')['native_task_bundle']['usd_path']
    s=Usd.Stage.Open(Usd.Stage.Open(source).Flatten());path=task['asset_metadata']['actors']['sample_01']
    api=UsdPhysics.MassAPI(s.GetPrimAtPath(path))
    def tensor():
        r=np.asarray(Gf.Matrix3d(Gf.Quatd(api.GetPrincipalAxesAttr().Get()))).T
        return r@np.diag(api.GetDiagonalInertiaAttr().Get())@r.T
    old=tensor();mass=api.GetMassAttr().Get();com=np.array(api.GetCenterOfMassAttr().Get())
    a=np.diag([1.04,1.04,1.]);cov=np.eye(3)*np.trace(old)/2-old;newcov=np.linalg.det(a)*a@cov@a.T
    expected=np.eye(3)*np.trace(newcov)-newcov
    bake_body_scale(s,path,[1.04,1.04,1.])
    assert np.linalg.norm(tensor()-expected)/np.linalg.norm(expected)<3e-7  # float32 principal-axis quaternion
    assert np.allclose(api.GetCenterOfMassAttr().Get(),a@com,atol=2e-9)
    assert api.GetMassAttr().Get()==pytest.approx(mass*np.linalg.det(a),rel=1e-7)


def test_mechanical_actual_mesh_vertices_agree_with_frozen_evaluator(tmp_path):
    from bench.simulation.assets.task_parts import load_obj_mesh
    p=parent('mechanical_assembly','parts_sorting');path=mech.prepare_asset(p,2,tmp_path/'assets')
    result=mech.compose_case(p,path,'single_arm')
    for spec in result['scene']['objects']:
        root=Path(spec['asset_root']);asset=read(root/'manifest.json')['assets'][spec['asset_id']]
        frozen=np.asarray(result['task']['evaluation']['geometry_vertices_local_m'][spec['id']])
        if spec['id']=='ring':
            stage=Usd.Stage.Open(asset['usd_path'])
            points=np.asarray(UsdGeom.Mesh(stage.GetPrimAtPath('/Ring/collision')).GetPointsAttr().Get())
        else:
            groups=[]
            for part in asset['collision_parts']:
                points,_=load_obj_mesh(root/part['mesh_path'])
                groups.append(points*np.asarray(part['scale'])+np.asarray(part['position_m']))
            points=np.concatenate(groups)
        # Original OBJ and actual USD points use float32; compare actual surface vertices without index ordering assumptions.
        assert max(np.min(np.linalg.norm(frozen-p,axis=1)) for p in points)<1e-8
