import math
import numpy as np
import pytest
from pxr import Usd, UsdGeom, UsdPhysics
from scenes.workrooms.laboratory.tasks.pipette_transfer.assets import author_task, PISTON_AREA_M2, STROKE_M
from scenes.workrooms.laboratory.tasks.pipette_transfer.liquid import PistonVolumeModel, ReducedLiquidDynamics


def fixture():
    stage=Usd.Stage.CreateInMemory();metadata=author_task(stage)
    p=metadata['task_parameters']['hydraulics']
    poses={o['id']:o['initial_pose_xyzw'] for o in metadata['objects'] if o['id'] in p['containers']}
    return stage,metadata,p,poses


def test_asset_has_real_prismatic_bodies_and_open_shells():
    stage,m,p,_=fixture()
    assert set(m['actors'])=={'source','receiver_a','receiver_b','pipette','plunger'}
    assert not m['particle_sets']
    fixture_meta=m['fixture_metadata']['pipette_cradle']
    assert fixture_meta['initial_pose_xyzw']==[0.,0.,0.,0.,0.,0.,1.]
    assert all(stage.GetPrimAtPath(path).HasAPI(UsdPhysics.CollisionAPI) for path in fixture_meta['collider_paths'])
    joint=UsdPhysics.PrismaticJoint(stage.GetPrimAtPath(m['joints']['plunger']))
    assert joint.GetBody0Rel().GetTargets()==[m['actors']['pipette']]
    assert joint.GetBody1Rel().GetTargets()==[m['actors']['plunger']]
    assert joint.GetLowerLimitAttr().Get()==0
    assert joint.GetUpperLimitAttr().Get()==pytest.approx(STROKE_M)
    assert UsdPhysics.DriveAPI(joint.GetPrim(),'linear').GetStiffnessAttr().Get()==0
    assert p['initial_total_ml']==80
    assert PISTON_AREA_M2*STROKE_M*1e6>24
    for o in m['objects']:
        assert stage.GetPrimAtPath(o['body_path']).HasAPI(UsdPhysics.RigidBodyAPI)
        lo,hi=o['local_bounds_m']
        assert all(a<b for a,b in zip(lo,hi))
    for name,c in p['containers'].items():
        assert not stage.GetPrimAtPath(c['initial_surface_path']).HasAPI(UsdPhysics.CollisionAPI)
        mesh=UsdGeom.Mesh(stage.GetPrimAtPath(c['path']+'/HollowVessel'))
        pts=np.array(mesh.GetPointsAttr().Get())
        assert np.any(np.isclose(np.linalg.norm(pts[:,:2],axis=1),c['radius_m']))


def run_stroke(model,poses,tip,q0,q1,steps=100):
    for q in np.linspace(q0,q1,steps+1)[1:]:
        model.step(.005,piston_q_m=q,tool_pose_xyzw=[*tip,0,0,0,1],container_poses=poses)


def test_native_geometry_dependent_aspiration_dispense_and_stop():
    _,m,p,poses=fixture();model=PistonVolumeModel(p)
    q20=.020e-3/PISTON_AREA_M2
    source_tip=np.array(poses['source'][:3])+[0,0,.010]
    run_stroke(model,poses,source_tip,0,q20)
    assert model.state()['tool_ml']==pytest.approx(20)
    assert model.state()['source_ml']==pytest.approx(60)
    for name,q0,q1 in [('receiver_a',q20,q20/2),('receiver_b',q20/2,0)]:
        tip=np.array(poses[name][:3])+[0,0,.085]
        run_stroke(model,poses,tip,q0,q1)
        run_stroke(model,poses,tip,q1,q1)
        assert model.state()['receivers_ml'][name]==pytest.approx(10)
    assert model.state()['cumulative_spill_ml']==0
    assert model.state()['in_flight_ml']==0
    assert model.state()['inventory_closed']
    assert model.state()['outlet_flow_ml_s']==0


def test_air_pull_and_missing_receiver_do_not_manufacture_success():
    _,_,p,poses=fixture();model=PistonVolumeModel(p)
    q10=.010e-3/PISTON_AREA_M2
    run_stroke(model,poses,[.7,1.3,1.1],0,q10)
    assert model.state()['tool_ml']==0
    run_stroke(model,poses,[.7,1.3,1.1],q10,0)
    run_stroke(model,poses,np.array(poses['source'][:3])+[0,0,.010],0,q10)
    run_stroke(model,poses,[.7,1.3,1.1],q10,0)
    run_stroke(model,poses,[.7,1.3,1.1],0,0)
    assert model.state()['cumulative_spill_ml']==pytest.approx(10)
    assert model.state()['receivers_ml']=={'receiver_a':0,'receiver_b':0}
    assert model.state()['inventory_closed']


def test_tip_above_surface_cannot_aspirate_and_tipped_vessel_spills():
    _,_,p,poses=fixture();model=PistonVolumeModel(p)
    run_stroke(model,poses,np.array(poses['source'][:3])+[0,0,.07],0,.03)
    assert model.state()['tool_ml']==0
    poses['source']=[*poses['source'][:3],math.sin(math.pi/4),0,0,math.cos(math.pi/4)]
    run_stroke(model,poses,[.7,1.3,1.1],.03,.03,steps=1)
    assert model.state()['source_ml']==0
    assert model.state()['cumulative_spill_ml']==80
    assert model.state()['inventory_closed']


def test_component_requires_native_baseline_and_continuous_frames():
    _,m,_,_=fixture();provider=ReducedLiquidDynamics({'asset_metadata':m})
    objects={o['id']:{'position_m':o['initial_position_m'],'quaternion_wxyz':[1,0,0,0],
        'linear_velocity_m_s':[0,0,0],'angular_velocity_rad_s':[0,0,0]} for o in m['objects']}
    frame={'objects':objects,'native_step':0,'native_time_s':0.,'physics_dt_s':.0005,'baseline':False}
    with pytest.raises(ValueError,match='baseline'):provider.on_frame(frame)
    frame['baseline']=True;provider.on_frame(frame)
    frame.update(baseline=False,native_step=2,native_time_s=.001)
    with pytest.raises(ValueError,match='consecutive'):provider.on_frame(frame)
    frame.update(native_step=1,native_time_s=.0005);provider.on_frame(frame)
    assert provider.on_observe()['inventory_closed']
    assert provider.on_observe()['step_count']==1


def test_task_liquid_visual_bindings_show_actual_volume_without_physics_mutation():
    from bench.simulation.isaacsim_backend.volume_visual_state import CylinderVolumeBindings
    stage,m,p,poses=fixture();model=PistonVolumeModel(p)
    bindings=CylinderVolumeBindings(stage,m['volume_visual_bindings'])
    physical={str(prim.GetPath()):{a.GetName():a.Get() for a in prim.GetAttributes()}
        for prim in stage.Traverse() if prim.HasAPI(UsdPhysics.RigidBodyAPI) or prim.HasAPI(UsdPhysics.CollisionAPI)}
    q10=10e-6/PISTON_AREA_M2
    run_stroke(model,poses,np.array(poses['source'][:3])+[0,0,.010],0,q10)
    receipt=bindings.sync({'detail':{'water_balance':model.state()}},native_step=100)
    by_id={x['id']:x for x in receipt['bindings']}
    assert by_id['liquid_tool']['volume_ml']==pytest.approx(10.)
    assert by_id['liquid_tool']['visibility']=='inherited'
    assert by_id['liquid_receiver_a']['visibility']=='invisible'
    assert by_id['liquid_source']['volume_ml']==pytest.approx(70.)
    run_stroke(model,poses,np.array(poses['receiver_a'][:3])+[0,0,.085],q10,0)
    run_stroke(model,poses,np.array(poses['receiver_a'][:3])+[0,0,.085],0,0)
    receipt=bindings.sync({'detail':{'water_balance':model.state()}},native_step=300)
    by_id={x['id']:x for x in receipt['bindings']}
    assert by_id['liquid_receiver_a']['volume_ml']==pytest.approx(10.)
    assert by_id['liquid_tool']['visibility']=='invisible'
    for path,attrs in physical.items():
        assert {a.GetName():a.Get() for a in stage.GetPrimAtPath(path).GetAttributes()}==attrs
