"""Original mechanical apparatus remains immutable under finite XY/texture seeds."""
from copy import deepcopy
import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from bench.paths import DATA_ROOT, load_json
from scenes.workrooms.mechanical_assembly import placement_variants as v
from scenes.workrooms.mechanical_assembly import placement_authoring as a
from scenes.workrooms.initial_randomization.release import _compose
from scenes.workrooms.seed_geometry import file_hash,write_json
from scripts.assets.prepare_workroom_randomization import original_task_definition


def parents(task_id):
    task=original_task_definition('mechanical_assembly',task_id)
    options=task.get('robot_configurations') or {'dual_same_side':task}
    return {key:Path(row['source_release']['path']) for key,row in options.items()}


@pytest.fixture(scope='module')
def originals():
    return {task_id:{cfg:(root,*v._inputs(root)) for cfg,root in parents(task_id).items()}
            for task_id in v.TASK_IDS}


@pytest.fixture(scope='module')
def authored(tmp_path_factory,originals):
    out=tmp_path_factory.mktemp('mechanical_xy_real_assets');result={}
    actual_source_root=Path(v.__file__).resolve().parents[3]
    source_root=out/'test_only_source'
    paths=('scenes/workrooms/mechanical_assembly/placement_variants.py',
        'scenes/workrooms/mechanical_assembly/placement_authoring.py','scenes/workrooms/seed_geometry.py',
        'bench/simulation/isaacsim_backend/asset_catalog.py','bench/simulation/isaacsim_backend/presentation.py')
    for name in paths:
        (source_root/name).parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(actual_source_root/name,source_root/name)
    hashes={name:file_hash(source_root/name)for name in paths}
    snapshot=source_root/'source_snapshot_manifest.json'
    write_json(snapshot,{'schema':'complete_benchmark_source_snapshot_v1','snapshot_root':str(source_root),'files_sha256':hashes})
    proof={'snapshot_root':str(source_root),'manifest_path':str(snapshot),'manifest_sha256':file_hash(snapshot),
        'source_files_sha256':{str(source_root/name):sha for name,sha in hashes.items()}}
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(a,'_author_source_identity',lambda:deepcopy(proof))
        for task_id,configs in originals.items():
            parent=next(iter(configs.values()))[0]
            for seed in (0,1):
                result[task_id,seed]=v.prepare_asset(parent,seed,out/task_id/str(seed),generator_version=v.VERSION)
    return result


@pytest.mark.parametrize('task_id',v.TASK_IDS)
def test_ten_unique_xy_cases_preserve_connected_groups(originals,task_id):
    _,scene,task,_=next(iter(originals[task_id].values()))
    plans=[v.plan_case(scene,task,seed)for seed in range(10)]
    assert len({tuple(p['translation_xy_m'])for p in plans})==10
    assert plans[0]['translation_xy_m']==[0.,0.]
    assert all(max(map(abs,p['translation_xy_m']))<=.01 for p in plans)
    assert all(p['groups']==plans[0]['groups']for p in plans)
    if task_id=='nist_belt_drive_assembly':
        assert plans[0]['fixed_actor_ids']==['drive_pulley']
        assert len(plans[0]['groups'][0]['actor_ids'])==64
    if task_id in ('nist_cable_routing','nist_wire_harness'):
        declared=task['asset_metadata']['task_parameters']['flexible_groups']
        assert [x['actor_ids']for x in plans[0]['groups']]==[x['actor_ids']for x in declared]


@pytest.mark.parametrize('bad',[True,-1,10,1.0,None])
def test_seed_rejected(originals,bad):
    _,scene,task,_=next(iter(originals[v.TASK_IDS[0]].values()))
    with pytest.raises(ValueError):v.plan_case(scene,task,bad)


@pytest.mark.parametrize('task_id',v.TASK_IDS)
def test_real_author_roundtrip_seedzero_and_all_original_installations(originals,authored,task_id):
    for cfg,(root,scene,task,initial) in originals[task_id].items():
        zero=v.compose_case(root,authored[task_id,0],cfg)
        assert zero['scene']==scene and zero['task']==task and zero['initial_state']==initial
        for seed in (0,1):
            child,childtask,joints,case,sources,deps=_compose(root,authored[task_id,seed],task_id,seed,cfg)
            assert joints==initial
            for key in ('robot','physics','isaacsim','robot_configuration','workroom','cameras','controller'):
                assert child.get(key)==scene.get(key)
            assert all(text==(root/'evaluator'/name).read_text()for name,text in sources.items())
            from bench.isaacsim.release import load_evaluator
            load_evaluator(root)(childtask)
            assert case['actual_body_poses_m']==load_json(authored[task_id,seed])['actual_body_poses_m']
            assert all(file_hash(path)==sha for path,sha in deps.items())
            before=task.get('evaluation',{}).get('thresholds')
            assert childtask.get('evaluation',{}).get('thresholds')==before
            if 'native_task_bundle'in scene:
                assert childtask['asset_metadata']['task_parameters']==task['asset_metadata']['task_parameters']
                assert child['native_task_bundle']['joints']==scene['native_task_bundle']['joints']
            else:
                for old,new in zip(scene['objects'],child['objects']):
                    assert {k:x for k,x in old.items()if k!='position_m'}=={k:x for k,x in new.items()if k not in ('position_m','presentation')}
    data=load_json(Path(load_json(authored[task_id,1])['metadata']))
    assert data['protected_physics_before_sha256']==data['protected_physics_after_sha256']
    assert all(row['only_albedo_changed']for row in data['appearance'])
    zero_files=list(authored[task_id,0].parent.rglob('*'))
    assert not any(p.suffix in ('.usd','.usda','.png')for p in zero_files)


def test_fastener_actual_anchored_body_and_scaled_geometry(originals,authored):
    root,scene,task,_=next(iter(originals['fastener_assembly'].values()))
    result=v.compose_case(root,authored['fastener_assembly',1],'single_arm')
    assert result['actual_body_poses_m']['bolt']['prim_path']=='/World/Objects/bolt/sharpa_threaded_bolt/allen_bolt'
    bolt=next(s for s in scene['objects']if s['id']=='bolt')
    stage,wrapper,asset,points=a._catalog_stage(bolt)
    from pxr import UsdGeom
    matrix=UsdGeom.XformCache().GetLocalToWorldTransform(stage.GetPrimAtPath(v.catalog_body_path(bolt)))
    assert matrix.ExtractTranslation().GetLength()==0.
    assert float(np.linalg.norm(np.array(matrix)[0,:3]))==pytest.approx(1.5)
    bounds=next(x for x in result['static_geometry']['bodies']if x['object_id']=='bolt')['local_bounds_m']
    np.testing.assert_allclose(bounds,[points.min(0),points.max(0)],atol=0,rtol=0)
    feature=next(x for x in result['interaction_targets']if x['label']=='bolt_pickup')
    assert feature['prim_path']==wrapper
    assert feature['approach_direction_local']==pytest.approx([0,0,-1])
    assert feature['local_position_m'][2]==pytest.approx(points[:,2].min())


def test_manifest_and_parent_configuration_tampering_rejected(authored,originals,tmp_path):
    parent=next(iter(originals['peg_insertion'].values()))[0]
    path=authored['peg_insertion',1]
    with pytest.raises(ValueError,match='installation'):v.compose_case(parent,path,'dual_opposed')
    original=load_json(path);metadata=load_json(Path(original['metadata']))
    for key in ('position_m','quaternion_wxyz'):
        altered=deepcopy(metadata);altered['actual_body_poses_m']['peg_0'][key][0]+=.02
        target=tmp_path/(key+'.json');write_json(target,altered)
        manifest=deepcopy(original);manifest['metadata']=str(target);manifest['metadata_sha256']=file_hash(target)
        manifest['dependencies_sha256'][str(target)]=file_hash(target)
        manifest['actual_body_poses_m']=altered['actual_body_poses_m']
        candidate=tmp_path/(key+'_manifest.json');write_json(candidate,manifest)
        with pytest.raises(ValueError,match='XY-only'):v.compose_case(parent,candidate,'single_arm')
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(a,'_author_source_identity',lambda:{})
        with pytest.raises(FileExistsError):v.prepare_asset(parent,1,path.parent)


def test_mutable_source_cannot_publish(originals,tmp_path):
    parent=next(iter(originals['peg_insertion'].values()))[0]
    out=tmp_path/'forbidden_asset'
    with pytest.raises(ValueError,match='complete frozen source'):v.prepare_asset(parent,1,out)
    assert not out.exists()


def test_texture_only_albedo_and_joint_guard(tmp_path):
    from pxr import Usd,UsdGeom,UsdShade,UsdPhysics,Sdf,Gf
    stage=Usd.Stage.CreateInMemory();body=UsdGeom.Xform.Define(stage,'/Task/Body').GetPrim()
    UsdPhysics.RigidBodyAPI.Apply(body)
    mesh=UsdGeom.Mesh.Define(stage,'/Task/Body/visual');mesh.CreatePointsAttr([(0,0,0),(1,0,0),(0,1,0)])
    mesh.CreateFaceVertexCountsAttr([3]);mesh.CreateFaceVertexIndicesAttr([0,1,2])
    material=UsdShade.Material.Define(stage,'/Task/Body/Looks/metal')
    shader=UsdShade.Shader.Define(stage,'/Task/Body/Looks/metal/Surface');shader.CreateIdAttr('UsdPreviewSurface')
    shader.CreateInput('diffuseColor',Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(.8,.1,.1))
    shader.CreateInput('roughness',Sdf.ValueTypeNames.Float).Set(.23)
    shader.CreateInput('metallic',Sdf.ValueTypeNames.Float).Set(.71)
    shader.CreateInput('ior',Sdf.ValueTypeNames.Float).Set(1.6)
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(),'surface')
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)
    texture=a.finish_texture(tmp_path,2);before=a._physics(stage,['/Task'],set())
    a._surface_texture(stage,['/Task/Body'],texture,2,material_root='/Task/SeedLooks')
    actual=UsdShade.MaterialBindingAPI(mesh.GetPrim()).ComputeBoundMaterial()[0].ComputeSurfaceSource()[0]
    assert actual.GetInput('diffuseColor').HasConnectedSource()
    for field in ('roughness','metallic','ior'):assert actual.GetInput(field).Get()==shader.GetInput(field).Get()
    assert not actual.GetInput('opacity')
    assert a._physics(stage,['/Task'],set())==before
    joint=UsdPhysics.SphericalJoint.Define(stage,'/Task/joint');joint.CreateBody0Rel().SetTargets(['/Task/Body'])
    before=a._physics(stage,['/Task'],set());joint.CreateLocalPos0Attr(Gf.Vec3f(.1,0,0))
    assert a._physics(stage,['/Task'],set())!=before
    with pytest.raises(ValueError,match='fixed/world joint'):a._assert_internal_joints(stage,['/Task'],{'/Task/Body'})


def _published_belt_asset(seed):
    return Path(f'{DATA_ROOT}/assets/task_parts/seed_randomization/v1/mechanical_assembly/'
        'nist_belt_drive_assembly/publication01_chains')/f'seed_{seed:02d}'/'seed_manifest.json'


def test_belt_v2_default_and_actual_v1_recomposition(originals):
    for ident,configs in originals.items():
        parent,scene,task,_=next(iter(configs.values()))
        default=v.plan_case(scene,task,1)
        assert default['generator_version']==(v.BELT_SAFE_VERSION if ident=='nist_belt_drive_assembly'else v.VERSION)
        if ident!='nist_belt_drive_assembly':
            assert default==v.plan_case(scene,task,1,generator_version=v.VERSION)
    # publication01_chains was authored from the superseded usability_v3 base.
    belt=original_task_definition('mechanical_assembly','nist_belt_drive_assembly')
    parent=Path(belt['previous_source_release']['path']);scene,task,_=v._inputs(parent)
    for seed in range(10):
        path=_published_belt_asset(seed);manifest=load_json(path)
        data=load_json(Path(manifest['metadata']))
        assert data['plan']==v.plan_case(scene,task,seed,generator_version=v.VERSION)
        result=v.compose_case(parent,path,'single_arm')
        assert result['actual_body_poses_m']==manifest['actual_body_poses_m']
        assert result['scene']['native_task_bundle']['joints']==scene['native_task_bundle']['joints']


def test_belt_version_is_task_specific(originals):
    for ident,configs in originals.items():
        _,scene,task,_=next(iter(configs.values()))
        with pytest.raises(ValueError,match='generator version'):
            v.plan_case(scene,task,1,generator_version='unknown')
        if ident!='nist_belt_drive_assembly':
            with pytest.raises(ValueError,match='generator version'):
                v.plan_case(scene,task,1,generator_version=v.BELT_SAFE_VERSION)


def _belt_stage(scene,plan):
    from pxr import Usd,UsdGeom
    bundle=scene['native_task_bundle'];root=bundle['root_prim_path']
    stage=Usd.Stage.CreateInMemory()
    UsdGeom.Xform.Define(stage,root).GetPrim().GetReferences().AddReference(bundle['usd_path'],root)
    moving={bundle['rigid_bodies'][name]for group in plan['groups']for name in group['actor_ids']}
    before=a._physics(stage,[root],moving)
    for path in moving:
        attr=stage.GetPrimAtPath(path).GetAttribute('xformOp:translate');p=attr.Get()
        attr.Set(type(p)(p[0]+plan['translation_xy_m'][0],p[1]+plan['translation_xy_m'][1],p[2]))
    assert a._physics(stage,[root],moving)==before
    a._assert_internal_joints(stage,[root],moving)
    return stage,root,moving


def test_all_nine_belt_v2_actual_source_solids_and_support(originals):
    _,scene,task,_=originals['nist_belt_drive_assembly']['single_arm']
    plans=[v.plan_case(scene,task,seed)for seed in range(10)]
    assert len({tuple(p['translation_xy_m'])for p in plans})==10
    assert plans[0]['translation_xy_m']==[0.,0.]
    assert {tuple(p['translation_xy_m'])for p in plans[1:]}=={
        (x,y)for x in (-.008,-.006,-.004)for y in (-.01,-.005,0.)}
    for plan in plans[1:]:
        stage,root,moving=_belt_stage(scene,plan)
        receipt=a._belt_initial_clearance(stage,root,moving)
        assert receipt['moving_collider_count']==66 and receipt['fixed_collider_count']==8
        assert min(receipt['workplate_xy_margin_m'])>.003
        assert receipt['minimum_original_support_gap_m']>0
        assert all(row['common_inscribed_sphere_radius_m']<-.0017 for row in receipt['narrow_phase'])
        assert plan['fixed_actor_ids']==['drive_pulley']


def test_unsafe_original_belt_translation_fails_new_source_guard(originals):
    _,scene,task,_=originals['nist_belt_drive_assembly']['single_arm']
    legacy=v.plan_case(scene,task,4,generator_version=v.VERSION)
    stage,root,moving=_belt_stage(scene,legacy)
    with pytest.raises(ValueError,match='intersects a fixed collider'):
        a._belt_initial_clearance(stage,root,moving)


def test_belt_v2_seed_zero_and_real_author_roundtrip(originals,tmp_path,monkeypatch):
    source_root=tmp_path/'test_only_source'
    actual_source_root=Path(v.__file__).resolve().parents[3]
    paths=('scenes/workrooms/mechanical_assembly/placement_variants.py',
        'scenes/workrooms/mechanical_assembly/placement_authoring.py','scenes/workrooms/seed_geometry.py',
        'bench/simulation/isaacsim_backend/asset_catalog.py','bench/simulation/isaacsim_backend/presentation.py')
    for name in paths:
        (source_root/name).parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(actual_source_root/name,source_root/name)
    hashes={name:file_hash(source_root/name)for name in paths}
    snapshot=source_root/'source_snapshot_manifest.json'
    write_json(snapshot,{'schema':'complete_benchmark_source_snapshot_v1','snapshot_root':str(source_root),'files_sha256':hashes})
    proof={'snapshot_root':str(source_root),'manifest_path':str(snapshot),'manifest_sha256':file_hash(snapshot),
        'source_files_sha256':{str(source_root/name):sha for name,sha in hashes.items()}}
    monkeypatch.setattr(a,'_author_source_identity',lambda:deepcopy(proof))
    parent,scene,task,initial=originals['nist_belt_drive_assembly']['single_arm']
    zero=v.prepare_asset(parent,0,tmp_path/'zero')
    result=v.compose_case(parent,zero,'single_arm')
    assert (result['scene'],result['task'],result['initial_state'])==(scene,task,initial)
    assert not any(p.suffix in ('.usd','.usda','.png')for p in zero.parent.rglob('*'))
    one=v.prepare_asset(parent,9,tmp_path/'nine')
    manifest=load_json(one);data=load_json(Path(manifest['metadata']))
    assert manifest['generator_version']==v.BELT_SAFE_VERSION
    assert data['source_initial_clearance']['moving_collider_count']==66
    for cfg,(parent,scene,task,initial)in originals['nist_belt_drive_assembly'].items():
        result=v.compose_case(parent,one,cfg)
        assert result['initial_state']==initial
        assert result['scene']['native_task_bundle']['joints']==scene['native_task_bundle']['joints']
        assert result['task']['asset_metadata']['task_parameters']==task['asset_metadata']['task_parameters']
        for name,text in result['evaluator_sources'].items():assert text==(parent/'evaluator'/name).read_text()
        for name,row in result['actual_body_poses_m'].items():
            old=next(x for x in task['asset_metadata']['objects']if x['id']==name)
            assert row['position_m'][2]==old['initial_position_m'][2]
            if name=='drive_pulley':assert row['position_m']==old['initial_position_m']
    # A locally rehashed version cannot reinterpret a v2 plan as the old recipe.
    bad=deepcopy(manifest);bad['generator_version']=v.VERSION
    candidate=tmp_path/'wrong_version.json';write_json(candidate,bad)
    with pytest.raises(ValueError,match='Frozen supply translation'):
        v.compose_case(originals['nist_belt_drive_assembly']['single_arm'][0],candidate,'single_arm')


def test_only_the_audio_plug_seat_changes_its_goal_approach(authored):
    for (task_id, seed), path in authored.items():
        for target in load_json(path)['interaction_targets']:
            if target['type'] != 'goal':
                continue
            # The audio pin is inserted along the socket bore (world +Y), from its own plug side.
            expected = [0., -1., 0.] if (task_id, target['label']) == ('nist_cable_routing', 'cable_end_placement') else [0., 0., 1.]
            assert target['approach_direction_local'] == pytest.approx(expected, abs=1e-12), (task_id, seed, target['label'])
