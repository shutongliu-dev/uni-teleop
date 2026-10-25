"""Physical authoring contracts for externally sourced NIST candidates."""
from pathlib import Path
import numpy as np
import pytest
from pxr import Usd, UsdPhysics
from scenes.workrooms.mechanical_assembly.nist_assets import SOURCE_ROOT,TASK_IDS,author_task,structural_audit

pytestmark=pytest.mark.skipif(not (SOURCE_ROOT/'extracted/atb4_stl').is_dir(),reason='NIST source snapshot not available')

@pytest.mark.parametrize('task_id',TASK_IDS)
def test_passive_topology_and_goal_anchors(task_id):
    stage=Usd.Stage.CreateInMemory();meta=author_task(stage,task_id)
    audit=structural_audit(stage,meta)
    assert audit['passed'],audit['errors']
    assert audit['maximum_goal_anchor_gap_m']<1e-6
    assert meta['source']['scale_to_m']==.001
    assert meta['source']['external_assets']
    assert not meta['physics_accepted']
    for obj in meta['objects']:
        prim=stage.GetPrimAtPath(obj['body_path'])
        assert prim.HasAPI(UsdPhysics.RigidBodyAPI)
        assert not UsdPhysics.RigidBodyAPI(prim).GetKinematicEnabledAttr().Get()
        assert min(obj['diagonal_inertia_kg_m2'])>0
        for path in obj['collider_paths']:
            assert UsdPhysics.CollisionAPI(stage.GetPrimAtPath(path)).GetCollisionEnabledAttr().Get()
    for path in meta['joints'].values():
        joint=UsdPhysics.Joint(stage.GetPrimAtPath(path))
        assert joint.GetPrim().GetTypeName() in {'PhysicsSphericalJoint','PhysicsFixedJoint'}
        assert not joint.GetCollisionEnabledAttr().Get()
        assert not any('DriveAPI' in s for s in joint.GetPrim().GetAppliedSchemas())


def test_cable_gate_excludes_top_of_closed_tube():
    stage=Usd.Stage.CreateInMemory();meta=author_task(stage,'nist_cable_routing')
    gate=meta['task_parameters']['flexible_groups'][0]['route_gates'][0]
    # Original minimum measured bore10.6389mm, cable radius1.75mm;
    # tolerance0.6111mm admits normal clearance but excludes wall-top routing.
    assert gate['radius_m']==.0095
    assert gate['radius_m']<.013208+.00175
    assert gate['radius_m']<.0106389


def test_cable_seating_measures_actual_pin_not_tail_origin():
    from scenes.workrooms.mechanical_assembly.nist_assets import _r
    stage=Usd.Stage.CreateInMemory();meta=author_task(stage,'nist_cable_routing')
    goal=meta['task_parameters']['seating_goals'][0]
    pose=meta['diagnostic_goal_poses'][goal['actor']]
    mating_point=np.asarray(pose[:3])+_r(pose[3:])@goal['actor_local_point_m']
    socket=np.asarray(meta['fixture_metadata']['audio_socket']['initial_position_m'])
    assert np.allclose(mating_point,socket,atol=1e-9)
    assert np.allclose(goal['target_position_m'],socket,atol=1e-9)
    assert np.linalg.norm(mating_point-pose[:3])==pytest.approx(.054)
    assert goal['contact_ids']==['audio_socket']
    group=next(g for g in meta['task_parameters']['flexible_groups']
               if g['id']==goal['support_via_flexible_group'])
    assert goal['actor'] in group['actor_ids']
    assert group['minimum_support_fraction']>0


def test_belt_goal_preserves_exact_link_lengths():
    stage=Usd.Stage.CreateInMemory();meta=author_task(stage,'nist_belt_drive_assembly')
    objs={o['id']:o for o in meta['objects']}
    group=meta['task_parameters']['flexible_groups'][0]
    assert len(group['actor_ids'])==64
    assert abs(sum(objs[n]['link_length_m'] for n in group['actor_ids'])-.675)<1e-10
    assert {'drive_pulley','timing_pulley','tensioner_idler'}=={x['actor'] for x in meta['task_parameters']['seating_goals']}
    assert structural_audit(stage,meta)['maximum_goal_anchor_gap_m']<1e-6

@pytest.mark.parametrize('task_id',['nist_belt_drive_assembly','nist_cable_routing','nist_wire_harness'])
def test_native_primitive_bounds_preserve_intended_link_dimensions(task_id):
    from pxr import UsdGeom,Gf
    stage=Usd.Stage.CreateInMemory();meta=author_task(stage,task_id)
    objects=[o for o in meta['objects'] if 'analytic_geometry' in o]
    assert objects
    for obj in objects:
        prim=stage.GetPrimAtPath(obj['collider_paths'][0])
        assert prim.GetTypeName() in {'Capsule','Cube'}
        lo,hi=UsdGeom.Boundable.ComputeExtentFromPlugins(UsdGeom.Boundable(prim),Usd.TimeCode.Default())
        points=np.asarray([[x,y,z] for x in [lo[0],hi[0]] for y in [lo[1],hi[1]] for z in [lo[2],hi[2]]])
        transform=UsdGeom.Xformable(prim).GetLocalTransformation()
        points=np.asarray([transform.Transform(Gf.Vec3d(*p)) for p in points])
        assert np.allclose([points.min(0),points.max(0)],obj['local_bounds_m'],atol=2e-9)


def test_belt_mating_requires_shaft_and_both_pulleys():
    stage=Usd.Stage.CreateInMemory();meta=author_task(stage,'nist_belt_drive_assembly')
    parameters=meta['task_parameters']
    for goal in parameters['seating_goals']:
        assert goal['contact_ids']==[goal['actor']+'_shaft']
        assert 'workplate' not in goal['contact_ids']
    assert set(parameters['flexible_groups'][0]['required_mating_contact_ids'])=={'drive_pulley','timing_pulley'}
    # Entire supply ring clears the fixed idler post shoulder in the XY plane.
    ids=parameters['flexible_groups'][0]['actor_ids']
    objects={o['id']:o for o in meta['objects']}
    center=np.asarray(meta['fixture_metadata']['tensioner_idler_shaft']['initial_position_m'][:2])
    points=np.asarray([objects[n]['initial_position_m'][:2] for n in ids])
    assert np.linalg.norm(points-center,axis=1).min()>.0067+.00125+.001


def test_harness_terminal_exit_does_not_block_attached_capsule():
    from pxr import UsdGeom,Gf
    from scenes.workrooms.mechanical_assembly.nist_assets import _r
    stage=Usd.Stage.CreateInMemory();meta=author_task(stage,'nist_wire_harness')
    objects={o['id']:o for o in meta['objects']}
    for color in ['red','blue']:
        for side,index,direction in [('a',0,1),('b',29,-1)]:
            fixture=meta['fixture_metadata'][f'{color}_end_{side}_terminal']
            assert fixture['cable_exit']['direction_local']==[0,direction,0]
            assert fixture['cable_exit']['width_m']==.006
            assert len(fixture['collider_paths'])==6
            link=f'{color}_link_{index:03}'
            pose=meta['diagnostic_goal_poses'][link]
            geometry=objects[link]['analytic_geometry']
            axis=_r(pose[3:])[:,0]*geometry['cylinder_length_m']/2
            endpoints=np.asarray([pose[:3]-axis,pose[:3]+axis])
            for path in fixture['collider_paths']:
                prim=stage.GetPrimAtPath(path)
                assert prim.IsA(UsdGeom.Cube)
                local=UsdGeom.Xformable(prim).GetLocalTransformation()
                corners=np.asarray([local.Transform(Gf.Vec3d(x,y,z))
                    for x in [-.5,.5] for y in [-.5,.5] for z in [-.5,.5]])
                corners+=fixture['initial_position_m']
                gap=np.maximum(np.maximum(corners.min(0)-endpoints.max(0),
                                           endpoints.min(0)-corners.max(0)),0)
                # Axis-aligned cable segment vs exact terminal Cube. The
                # 0.3 mm reset floor gap remains; no wall intersects the wire.
                assert np.linalg.norm(gap)-geometry['radius_m']>.00029
