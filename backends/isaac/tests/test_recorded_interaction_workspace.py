"""The offline state replay must preserve native sources and rotate local targets."""
import hashlib
from itertools import product
import json
from pathlib import Path

import numpy as np
import pytest
from pxr import Gf, Usd, UsdGeom, UsdPhysics

from bench.paths import DATA_ROOT, load_json
from scripts.diagnostics.check_bimanual_options import apply_recorded_body_poses
from scripts.diagnostics.check_interaction_workspace import (
    actual_single_stand, free_surface_approach, interaction_targets, target_point,
    separating_axis_gap, validate_recorded_stages,
)


def body(stage,path):
    prim=UsdGeom.Xform.Define(stage,path).GetPrim()
    UsdPhysics.RigidBodyAPI.Apply(prim)
    return prim


def test_recorded_world_pose_respects_parent_and_preserves_source(tmp_path):
    path=tmp_path/'source.usda'
    source=Usd.Stage.CreateNew(str(path))
    parent=UsdGeom.Xform.Define(source,'/World')
    parent.AddTranslateOp().Set(Gf.Vec3d(2.,1.,0.))
    body(source,'/World/Body')
    source.GetRootLayer().Save()
    before=hashlib.sha256(path.read_bytes()).hexdigest()
    offline=Usd.Stage.CreateInMemory()
    offline.GetRootLayer().subLayerPaths.append(str(path))
    q=[np.sqrt(.5),0.,0.,np.sqrt(.5)]
    apply_recorded_body_poses(offline,[{'prim_path':'/World/Body','position_m':[.3,.4,.8],'quaternion_wxyz':q}])
    matrix=np.asarray(UsdGeom.XformCache().GetLocalToWorldTransform(offline.GetPrimAtPath('/World/Body'))).T
    assert matrix[:3,3]==pytest.approx([.3,.4,.8])
    assert matrix[:3,0]==pytest.approx([0.,1.,0.],abs=1e-10)
    assert hashlib.sha256(path.read_bytes()).hexdigest()==before
    assert source.GetPrimAtPath('/World/Body').GetAttribute('xformOpOrder').Get() is None


def test_recorded_pose_cannot_mutate_persistent_stage(tmp_path):
    stage=Usd.Stage.CreateNew(str(tmp_path/'persistent.usda'))
    with pytest.raises(ValueError,match='anonymous offline'):
        apply_recorded_body_poses(stage,[])


@pytest.mark.parametrize('change',[
    {'quaternion_wxyz':[0.,0.,0.,0.]},
    {'position_m':[0.,float('nan'),0.]},
    {'prim_path':'/World/Missing'},
])
def test_rejects_missing_or_invalid_recorded_body(change):
    stage=Usd.Stage.CreateInMemory();body(stage,'/World/Body')
    record={'prim_path':'/World/Body','position_m':[0.,0.,0.],'quaternion_wxyz':[1.,0.,0.,0.]}
    with pytest.raises(ValueError): apply_recorded_body_poses(stage,[{**record,**change}])


def test_targets_follow_measured_body_instead_of_initial_reference():
    target={'label':'handle','reference_position_m':[100.,100.,100.],
            'body_id':'panel','local_position_m':[.2,0.,0.],
            'approach_direction_world':[0.,1.,0.],'approach_distance_m':.06}
    pose={'id':'panel','position_m':[.3,.4,.8],'quaternion_wxyz':[np.sqrt(.5),0.,0.,np.sqrt(.5)]}
    reference,approach=target_point(target,[pose])
    assert reference==pytest.approx([.3,.6,.8])
    assert approach==pytest.approx([.3,.66,.8])


def test_ambiguous_authored_targets_fail_closed():
    task={'evaluation':{'interaction_targets':[{'label':'first'}]},
          'asset_metadata':{'interaction_targets':[{'label':'second'}]}}
    with pytest.raises(ValueError,match='unambiguous'): interaction_targets(task)


def test_oriented_bounds_remove_diagonal_aabb_false_hit_but_keep_real_hit():
    from scipy.spatial.transform import Rotation
    panel=np.array(list(product((-.5,.5),(-.01,.01),(-.2,.2))))
    panel=Rotation.from_euler('z',45,degrees=True).apply(panel)
    small=np.array(list(product((-.02,.02),repeat=3)))
    outside=small+[-.3,.3,0.]
    assert np.all(np.minimum(panel.max(0),outside.max(0))>np.maximum(panel.min(0),outside.min(0)))
    assert separating_axis_gap(panel,outside)>.38
    assert separating_axis_gap(panel,small+[-.2,-.2,0.])<0.


def test_selected_storage_stand_uses_actual_front_mount_not_old_preview():
    path=Path(f'{DATA_ROOT}/datasets/fr3_sharpa_bench/isaacsim/releases/living_room/storage_box/v3/scene.json')
    if not path.exists(): pytest.skip('Shared frozen storage-box release unavailable')
    scene=load_json(path);stand=actual_single_stand(scene)
    mount=next(box for box in stand['boxes'] if box['name']=='robot_stand_mount')
    assert mount['position_m'][:2]==pytest.approx(scene['robot']['base_position_m'][:2])
    assert mount['position_m'][1]==pytest.approx(-.76)
    assert mount['position_m'][2]+mount['size_m'][2]/2==pytest.approx(.78)


def test_stage_pose_must_equal_its_original_raw_frame(tmp_path):
    raw={'time_s':.4,'objects':{'panel':{'position_m':[.3,0.,.8],'quaternion_wxyz':[1.,0.,0.,0.]}}}
    path=tmp_path/'raw.jsonl';path.write_text(json.dumps(raw)+'\n')
    pose={'id':'panel','prim_path':'/World/Panel',**raw['objects']['panel']}
    saved={'native_trace_files':[{'path':str(path)}],'workspace_stages':[
        {'trace_frame_index':0,'sim_time_s':.4,'body_poses':[pose]}]}
    scene={'native_task_bundle':{'rigid_bodies':{'panel':'/World/Panel'}}}
    assert validate_recorded_stages(saved,scene,{})['passed']
    pose['position_m']=[.4,0.,.8]
    with pytest.raises(ValueError,match='differs from native frame'):
        validate_recorded_stages(saved,scene,{})


def test_fixed_fixture_target_uses_actual_usd_world_pose():
    stage=Usd.Stage.CreateInMemory();prim=UsdGeom.Xform.Define(stage,'/World/Socket')
    prim.AddTranslateOp().Set(Gf.Vec3d(.92,.57,.9))
    target={'label':'receiver','reference_position_m':[.92,.57,.966],
            'body_id':'socket','local_position_m':[0.,0.,.066], 'approach_direction':[0.,0.,-1.]}
    point,approach=target_point(target,[],stage=stage,static_paths={'socket':'/World/Socket'})
    assert point==pytest.approx([.92,.57,.966])
    assert approach==pytest.approx([.92,.57,1.011])


def test_scaled_recorded_body_is_rejected_before_transform_write():
    stage=Usd.Stage.CreateInMemory();prim=body(stage,'/World/Body')
    UsdGeom.Xformable(prim).AddScaleOp().Set(Gf.Vec3f(2.))
    before=stage.GetRootLayer().ExportToString()
    with pytest.raises(ValueError,match='unit-scale'):
        apply_recorded_body_poses(stage,[{'prim_path':'/World/Body','position_m':[0.,0.,0.],
                                          'quaternion_wxyz':[1.,0.,0.,0.]}])
    assert before==stage.GetRootLayer().ExportToString()


def test_center_goal_is_preserved_while_free_approach_clears_the_tube():
    from types import SimpleNamespace
    tube=np.array(list(product((-.015,.015),(-.015,.015),(0.,.12))))
    checker=SimpleNamespace(obstacle_names=['/Task/Tube'],obstacle_corners=[tube])
    reference=np.array([0.,0.,.06]);authored=reference+[0.,0.,.045]
    approach,description=free_surface_approach(checker,reference,authored)
    assert approach==pytest.approx([0.,0.,.165])
    assert description['enclosing_surface_distance_m']==pytest.approx(.06)
    assert reference==pytest.approx([0.,0.,.06])


def test_free_approach_does_not_jump_over_an_unrelated_cover():
    from types import SimpleNamespace
    target=np.array(list(product((-.01,.01),repeat=3)))
    cover=np.array(list(product((-.1,.1),(-.1,.1),(.025,.035))))
    checker=SimpleNamespace(obstacle_names=['/Task/Target','/Task/Cover'],obstacle_corners=[target,cover])
    point,description=free_surface_approach(checker,np.zeros(3),np.array([0.,0.,.045]))
    assert point==pytest.approx([0.,0.,.055])
    assert description['surface_shapes']==['/Task/Target']
    assert description['blocked'] and description['forward_non_target_shapes']==['/Task/Cover']


def test_body_cap_counts_as_its_own_surface_instead_of_another_object():
    from types import SimpleNamespace
    trunk=np.array(list(product((-.01,.01),(-.01,.01),(0.,.11))))
    cap=np.array(list(product((-.012,.012),(-.012,.012),(.11,.12))))
    checker=SimpleNamespace(obstacle_names=['/Task/Tube/body','/Task/Tube/cap'],obstacle_corners=[trunk,cap])
    point,description=free_surface_approach(checker,np.array([0.,0.,.06]),np.array([0.,0.,.105]),owner_paths=['/Task/Tube'])
    assert point==pytest.approx([0.,0.,.165])
    assert not description['blocked']
