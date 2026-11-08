"""Actual frozen laboratory assets; CPU authoring is not native qualification."""
from copy import deepcopy
from functools import lru_cache
import importlib
import json
from pathlib import Path
import subprocess
import sys
import shutil

import numpy as np
from pxr import Gf, Usd, UsdGeom, UsdPhysics, UsdShade
import pytest

from bench.paths import DATA_ROOT, load_json
from scenes.workrooms.laboratory import placement_variants as author

PARENTS = {task:Path(f'{DATA_ROOT}/datasets/fr3_sharpa_bench/isaacsim/releases/laboratory')/task/'budget_v1'
           for task in author.TASKS}


@lru_cache
def parent(task):
    if not PARENTS[task].is_dir(): pytest.skip('Original shared laboratory assets unavailable')
    _,values = author._parent(PARENTS[task])
    return Usd.Stage.Open(values['scene']['native_task_bundle']['usd_path']),values


@pytest.mark.parametrize('task',author.TASKS)
@pytest.mark.parametrize('seed',range(10))
def test_actual_original_stage_only_coherent_world_x(task,seed):
    original,values = parent(task); md = values['task']['asset_metadata']
    before = author._stage_values(original)
    stage,actual,record = author.derive_stage(original,md,seed)
    assert author._stage_values(original)==before
    assert actual==author.shifted_metadata(md,record['plan'])
    moved = {p for row in record['plan']['groups'].values() for p in row['prim_paths']}
    dx = author.OFFSETS_X_M[seed]
    for name,path in {**md['actors'],**md['fixtures']}.items():
        a=author.pose(original.GetPrimAtPath(path)); b=author.pose(stage.GetPrimAtPath(path))
        np.testing.assert_allclose(np.array(b['position_m'])-a['position_m'],[dx,0,0] if path in moved else [0,0,0],atol=1e-12)
        assert b['quaternion_wxyz']==a['quaternion_wxyz']
        np.testing.assert_allclose(author.collision_bounds(original,path),author.collision_bounds(stage,path),atol=1e-12)
    assert author.mass_inventory(original,md['actors'])==record['mass_properties']
    assert set(record['actual_body_poses_m'])==set(md['actors'])
    assert {r['object_id'] for r in record['static_geometry']['bodies']}==set(md['actors'])
    assert record['fixed_support_paths']
    for path in record['fixed_support_paths']:
        assert any(path.startswith(p+'/') for p in md['fixtures'].values())
        assert stage.GetPrimAtPath(path).HasAPI(UsdPhysics.CollisionAPI)
    if seed==0:
        assert actual==md and author._stage_values(stage)==before
    for row in record['interaction_targets']:
        assert stage.GetPrimAtPath(row['prim_path'])
        assert row['reference_state']=='initial'
        assert row['type'] in ('actual_surface','goal')
        assert np.linalg.norm(row['approach_direction_local'])==pytest.approx(1.)
        if row['type']=='actual_surface':
            assert row['source_collider_path'].startswith(row['prim_path']+'/')
            author._surface_witness(stage,row['prim_path'],row['source_collider_path'][len(row['prim_path'])+1:],row['local_position_m'])


@pytest.mark.parametrize('task',author.TASKS)
def test_metadata_world_references_only_and_destinations_fixed(task):
    _,v=parent(task); md=v['task']['asset_metadata']
    plans=[author.plan_case(task,i,md) for i in range(10)]
    assert len({tuple(x['translation_xy_m']) for x in plans})==10
    assert max(abs(x['translation_xy_m'][0]) for x in plans)==.010
    assert all(x['translation_xy_m'][1]==0 and x['yaw_rad']==0 for x in plans)
    result=author.shifted_metadata(md,plans[8]);p=result['task_parameters'];old=md['task_parameters'];dx=.010
    if task=='cap_placement':
        assert p['holder_center_m']==old['holder_center_m'] and p['cap_tray_target']==old['cap_tray_target']
        assert p['bottle_home']['center_m'][0]==old['bottle_home']['center_m'][0]+dx
        assert p['thread']['closed_cap_relative_position_m']==old['thread']['closed_cap_relative_position_m']
    elif task=='solid_weighing':
        assert p['platform']==old['platform'] and p['weighing_cup_target_position_m']==old['weighing_cup_target_position_m']
        assert p['source_tray']['initial_position_m'][0]==old['source_tray']['initial_position_m'][0]+dx
        assert p['scoop']['initial_position_m'][0]==old['scoop']['initial_position_m'][0]+dx
        assert p['cup']['initial_position_m'][0]==old['cup']['initial_position_m'][0]+dx
        assert p['pellet_mass_kg']==old['pellet_mass_kg'] and p['target_mass_kg']==old['target_mass_kg']
    elif task=='pipette_transfer':
        assert p['hydraulics']==old['hydraulics']
        assert result['volume_visual_bindings']==md['volume_visual_bindings']
        assert result['packet_visual_bindings']==md['packet_visual_bindings']
        assert p['tool_return']['initial_pose_xyzw'][0]==old['tool_return']['initial_pose_xyzw'][0]+dx
    else:
        assert p['receiver_position_m']==old['receiver_position_m'] and p['spring']==old['spring']
        assert p['tube_grip_center_in_frame_m']==old['tube_grip_center_in_frame_m']
        assert p['tool_return_position_m'][0]==old['tool_return_position_m'][0]+dx
    assert result['desktop_dressing']==md['desktop_dressing'] if 'desktop_dressing' in md else True


@pytest.mark.parametrize('seed',[-1,10,True,1.,None,'1'])
def test_seed_rejection(seed):
    with pytest.raises(ValueError):author.plan_case('cap_placement',seed,{})


def test_prior_lab_tasks_remain_separate():
    from scenes.workrooms.laboratory import seed_variants
    assert set(author.TASKS).isdisjoint(seed_variants.TASKS)
    with pytest.raises(ValueError):author.plan_case('sample_sorting',0,{})


@pytest.mark.parametrize('change',['mass','mesh','joint','fixture','yaw'])
def test_protected_physics_guard_rejects_changes(change):
    original,v=parent('tube_clamp_transfer');stage,md,receipt=author.derive_stage(original,v['task']['asset_metadata'],1)
    body=stage.GetPrimAtPath(md['actors']['clamp_frame'])
    if change=='mass':UsdPhysics.MassAPI(body).GetMassAttr().Set(.12)
    elif change=='mesh':UsdGeom.Mesh(stage.GetPrimAtPath(str(body.GetPath())+'/Shape_000')).GetPointsAttr().Set([Gf.Vec3f(0,0,0)])
    elif change=='joint':UsdPhysics.PrismaticJoint(stage.GetPrimAtPath(md['joints']['jaw_slide'])).GetLowerLimitAttr().Set(-.021)
    elif change=='fixture':stage.GetPrimAtPath(md['fixtures']['receiver_holder']).GetAttribute('xformOp:translate').Set(Gf.Vec3d(.93,.57,.9))
    else:body.GetAttribute('xformOp:orient').Set(Gf.Quatf(.9,Gf.Vec3f(0,0,.1)))
    with pytest.raises(ValueError):author.assert_translation_only(original,stage,receipt['plan'])


def test_world_joint_and_wrong_surface_rejected():
    stage,v=parent('pipette_transfer');stage=Usd.Stage.Open(stage.Flatten());md=v['task']['asset_metadata']
    UsdPhysics.Joint(stage.GetPrimAtPath(md['joints']['plunger'])).GetBody0Rel().ClearTargets(True)
    with pytest.raises(ValueError,match='joint endpoint'):author.derive_stage(stage,md,1)
    with pytest.raises(ValueError,match='not on'):
        author._surface_witness(stage,md['actors']['pipette'],'Barrel',[0.,0.,.055])


@pytest.mark.parametrize('task',author.TASKS)
def test_texture_keeps_geometry_physics_and_original_materials(tmp_path,task):
    original,v=parent(task);stage,md,_=author.derive_stage(original,v['task']['asset_metadata'],1)
    before=author._stage_values(stage)
    phys={str(p.GetPath()):str(UsdShade.MaterialBindingAPI(p).ComputeBoundMaterial(materialPurpose='physics')[0].GetPath())
          for p in stage.Traverse() if p.HasAPI(UsdPhysics.CollisionAPI)}
    appearance=author._appearance(stage,md,1,tmp_path)
    author.assert_finish_only(before,stage,appearance)
    assert appearance['changed_meshes']
    for p,bound in phys.items():
        assert str(UsdShade.MaterialBindingAPI(stage.GetPrimAtPath(p)).ComputeBoundMaterial(materialPurpose='physics')[0].GetPath())==bound
    for source,target in appearance['material_clones'].items():
        assert not stage.GetPrimAtPath(target).HasAPI(UsdPhysics.MaterialAPI)
        old=UsdShade.Material(stage.GetPrimAtPath(source)).ComputeSurfaceSource()[0]
        new=UsdShade.Material(stage.GetPrimAtPath(target)).ComputeSurfaceSource()[0]
        assert str(new.GetPath()).startswith(target+'/')
        for i in old.GetInputs():
            if i.GetBaseName()!='roughness':assert str(i.Get())==str(new.GetInput(i.GetBaseName()).Get())
        assert new.GetInput('roughness').HasConnectedSource()
    assert author.file_hash(tmp_path/'surface_roughness.png') in appearance['dependencies_sha256'].values()
    # The body geometry and the physical-purpose material relation cannot hide behind a visual change.
    prim=stage.GetPrimAtPath(appearance['changed_meshes'][0]);UsdGeom.Mesh(prim).GetPointsAttr().Set([Gf.Vec3f(0,0,0)])
    with pytest.raises(ValueError,match='protected'):author.assert_finish_only(before,stage,appearance)


def test_pipette_visual_clone_cannot_replace_original_physics_fallback(tmp_path):
    original,v=parent('pipette_transfer');stage,md,_=author.derive_stage(original,v['task']['asset_metadata'],2)
    before=author._stage_values(stage);appearance=author._appearance(stage,md,2,tmp_path)
    assert appearance['explicit_preserved_physics_bindings']
    path,source=next(iter(appearance['explicit_preserved_physics_bindings'].items()))
    stage.GetPrimAtPath(path).GetRelationship('material:binding:physics').SetTargets([appearance['material_clones'][source]])
    with pytest.raises(ValueError,match='physics material'):author.assert_finish_only(before,stage,appearance)


def _frozen_sources(tmp_path,task):
    folder=tmp_path/('frozen_'+task);folder.mkdir();(folder/'__init__.py').write_text('')
    sources=author.evaluator_sources(PARENTS[task],task)
    for name,text in sources.items():
        p=folder/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(text)
        assert text==(PARENTS[task]/'evaluator'/name).read_text()
    return folder.name,sources


@pytest.mark.parametrize('task',author.TASKS)
def test_prepare_compose_and_original_evaluator_all_seed_metadata(tmp_path,monkeypatch,task):
    import scripts.maintenance.freeze_project_sources as frozen
    # Explicit CPU-only source receipt fixture; never publishes to shared assets.
    snapshot=tmp_path/'source_fixture';snapshot.mkdir()
    project=Path(author.__file__).resolve().parents[3]
    for name in author.SOURCE_FILES:
        target=snapshot/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(project/name,target)
    author.write_json(snapshot/'source_snapshot_manifest.json',{'schema':'complete_benchmark_source_snapshot_v1',
        'snapshot_root':str(snapshot),'files_sha256':{n:author.file_hash(snapshot/n)for n in author.SOURCE_FILES}})
    monkeypatch.setattr(frozen,'verify_if_snapshot',lambda *a,**k:{'verified':True,'test_fixture_only':True,'snapshot_root':str(snapshot)})
    monkeypatch.setattr(author,'ASSET_ROOT',tmp_path)
    package,sources=_frozen_sources(tmp_path,task);monkeypatch.syspath_prepend(str(tmp_path))
    evaluator=importlib.import_module(package+'.sequence_evaluator')
    original,values=parent(task);md=values['task']['asset_metadata'];old=values['scene']
    for seed in range(10):
        case=deepcopy(values['task']);case['episode_seed']=seed
        case['asset_metadata']=author.shifted_metadata(md,author.plan_case(task,seed,md))
        case['evaluation']['metadata']=deepcopy(case['asset_metadata']);case['evaluation']['episode_seed']=seed
        evaluator.NativeTaskObserver(case);evaluator.SequenceEvaluator(case)
    for seed in (0,8):
        path=author.prepare_asset(PARENTS[task],seed,tmp_path/('seed_'+str(seed)))
        manifest=load_json(path);out=author.compose_case(PARENTS[task],path,'single_arm')
        assert out['evaluator_sources']==sources
        assert out['initial_state']==values['initial_state']
        assert out['task']['public_rules']==values['task']['public_rules']
        assert out['task']['public_goal']==values['task']['public_goal']
        assert out['task']['evaluation']['thresholds']==values['task']['evaluation']['thresholds']
        for k in ('robot','physics','workroom','cameras','reduced_liquid_payload'):
            assert out['scene'].get(k)==old.get(k)
        assert out['scene']['native_task_bundle']['observer']['configuration']==out['task']
        if task=='pipette_transfer':
            assert out['scene']['native_task_bundle']['reduced_dynamics']['configuration']==out['task']
        if seed==0:
            assert Path(manifest['initial_usd']).read_bytes()==Path(old['native_task_bundle']['usd_path']).read_bytes()
            assert out['task']['evaluation']==values['task']['evaluation']
        with pytest.raises(ValueError,match='configuration'):
            author.compose_case(PARENTS[task],path,'dual_same_side')
        for config in ('dual_same_side','dual_opposed'):
            root=Path(f'{DATA_ROOT}/datasets/fr3_sharpa_bench/isaacsim/releases/robot_configurations_20261026/laboratory_budget_v1')/task/config/'v1'
            _,dual=author._parent(root)
            composed=author.compose_case(root,path,config)
            assert composed['initial_state']==dual['initial_state']
            assert composed['scene']['robot']==dual['scene']['robot']
            assert composed['scene']['robot_configuration']==dual['scene']['robot_configuration']
            assert composed['task']['public_rules']==dual['task']['public_rules']
            assert composed['task']['evaluation']['thresholds']==dual['task']['evaluation']['thresholds']
            assert composed['evaluator_sources']=={str(p.relative_to(root/'evaluator')):p.read_text()for p in (root/'evaluator').rglob('*.py')}
            target=tmp_path/('frozen_'+task+'_'+config+'_'+str(seed));target.mkdir();(target/'__init__.py').write_text('')
            for name,text in composed['evaluator_sources'].items():
                f=target/name;f.parent.mkdir(parents=True,exist_ok=True);f.write_text(text)
            module=importlib.import_module(target.name+'.sequence_evaluator')
            module.NativeTaskObserver(composed['task']);module.SequenceEvaluator(composed['task'])
        for field in ('translation_xy_m','yaw_rad'):
            bad=deepcopy(manifest);bad['plan'][field]=[0.,.01] if field=='translation_xy_m' else .1
            altered=tmp_path/('bad_'+field+'.json');author.write_json(altered,bad)
            with pytest.raises(ValueError,match='plan'):author.compose_case(PARENTS[task],altered,'single_arm')
        bad=deepcopy(manifest);bad['author_source']['source_files_sha256'].pop(next(iter(bad['author_source']['source_files_sha256'])))
        altered=tmp_path/'bad_author.json';author.write_json(altered,bad)
        with pytest.raises(ValueError,match='source identity'):author.compose_case(PARENTS[task],altered,'single_arm')
    (snapshot/author.SOURCE_FILES[-1]).write_text('# changed author helper\n')
    with pytest.raises(ValueError,match='input differs'):author.compose_case(PARENTS[task],path,'single_arm')


def test_import_is_prekit_safe():
    script="import sys; from scenes.workrooms.laboratory import placement_variants; assert not any(x == 'pxr' or x.startswith('pxr.') for x in sys.modules)"
    done=subprocess.run([sys.executable,'-B','-c',script],capture_output=True,text=True)
    assert done.returncode==0,done.stderr
