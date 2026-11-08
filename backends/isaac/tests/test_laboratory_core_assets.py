"""Structural contracts and usable clearances for the core laboratory tasks."""
import importlib
import math
import numpy as np
import pytest
import trimesh
from pxr import Gf, Usd, UsdGeom, UsdPhysics

TASKS=('sample_sorting','vial_racking','cap_placement')

@pytest.mark.parametrize('task_id',TASKS)
@pytest.mark.parametrize('presentation',[False,True])
def test_inventory_is_dynamic_and_has_no_hidden_robot_or_joint(task_id,presentation):
    stage=Usd.Stage.CreateInMemory()
    meta=importlib.import_module('scenes.workrooms.laboratory.tasks.'+task_id+'.assets').author_task(stage,presentation=presentation)
    assert meta['task_id']==task_id
    assert len(meta['objects'])=={'sample_sorting':4,'vial_racking':3,'cap_placement':2}[task_id]
    assert meta['joints']=={} and meta['particle_sets']=={}
    for entry in meta['objects']:
        prim=stage.GetPrimAtPath(entry['body_path'])
        assert prim.HasAPI(UsdPhysics.RigidBodyAPI)
        assert not UsdPhysics.RigidBodyAPI(prim).GetKinematicEnabledAttr().Get()
        assert entry['mass_kg']>0
        assert np.all(np.asarray(entry['diagonal_inertia_kg_m2'])>0)
        assert len(entry['initial_pose_xyzw'])==7
        assert len(entry['local_bounds_m'])==2
    assert not any(p.IsA(UsdPhysics.Scene) or p.IsA(UsdPhysics.Joint) for p in stage.Traverse())


def test_rack_clearance_and_target_do_not_have_a_solid_top():
    stage=Usd.Stage.CreateInMemory()
    from scenes.workrooms.laboratory.tasks.vial_racking.assets import author_task
    meta=author_task(stage)
    for hole in meta['task_parameters']['holes']:
        assert hole['radius_m']>.0095+.002
        assert hole['rim_z_m']-hole['bottom_z_m']>=.055
    source=meta['task_parameters']['source_holes']
    target=meta['task_parameters']['holes']
    assert len(source)==len(target)==len(meta['objects'])==3
    assert {h['fixture'] for h in source}=={'supply_cradle'}
    assert {h['fixture'] for h in target}=={'rack'}
    assert meta['fixtures']['supply_cradle']!=meta['fixtures']['rack']
    assert max(h['center_m'][1] for h in source)<min(h['center_m'][1] for h in target)-1.
    occupied=set()
    for obj in meta['objects']:
        assert obj['nominal_support']=='supply_cradle'
        assert obj['orientation_xyzw']==pytest.approx([0,0,0,1])
        assert obj['mass_kg']==pytest.approx(.020)
        assert obj['local_bounds_m'][1][2]-obj['local_bounds_m'][0][2]==pytest.approx(.090)
        index,hole=min(enumerate(source),key=lambda pair:
            np.linalg.norm(np.asarray(obj['initial_pose_xyzw'][:2])-pair[1]['center_m'][:2]))
        assert index not in occupied
        occupied.add(index)
        assert obj['initial_pose_xyzw'][:2]==pytest.approx(hole['center_m'][:2])
        assert obj['initial_pose_xyzw'][2]==pytest.approx(hole['bottom_z_m'],abs=.001)
        assert hole['radius_m']>.0095+.002
        assert .02<hole['rim_z_m']-hole['bottom_z_m']<.03
        assert obj['initial_pose_xyzw'][2]+.090-hole['rim_z_m']>.05


def test_source_and_target_rack_collider_openings_are_real_and_have_bottom_support():
    from scenes.workrooms.laboratory.tasks.vial_racking.assets import author_task
    stage=Usd.Stage.CreateInMemory()
    meta=author_task(stage)
    cache=UsdGeom.XformCache()
    # Each authored rack collider is a convex piece. Test its actual half-spaces
    # rather than accepting the metadata's nominal hole radius as an opening.
    for holes in (meta['task_parameters']['source_holes'],meta['task_parameters']['holes']):
        fixture=meta['fixture_metadata'][holes[0]['fixture']]
        planes=[]
        for path in fixture['collider_paths']:
            mesh=UsdGeom.Mesh(stage.GetPrimAtPath(path))
            transform=cache.GetLocalToWorldTransform(mesh.GetPrim())
            vertices=np.asarray([transform.Transform(Gf.Vec3d(*point)) for point in mesh.GetPointsAttr().Get()])
            contact=trimesh.Trimesh(vertices,np.asarray(mesh.GetFaceVertexIndicesAttr().Get()).reshape(-1,3),process=False)
            planes.append((contact.face_normals,np.einsum('ij,ij->i',contact.face_normals,contact.triangles_center)))
        def occupied(point):
            return any(np.all(normals@point<=offsets+1.e-7) for normals,offsets in planes)
        for hole in holes:
            x,y=hole['center_m'][:2]
            guide_z=hole['rim_z_m']-.003
            assert not occupied(np.array([x,y,guide_z]))
            for angle in np.linspace(0,2*math.pi,8,endpoint=False):
                assert not occupied(np.array([x+.0095*math.cos(angle),y+.0095*math.sin(angle),guide_z]))
            assert occupied(np.array([x+.017*math.cos(math.pi/24),y+.017*math.sin(math.pi/24),guide_z]))
            assert occupied(np.array([x,y,hole['bottom_z_m']-.003]))


def test_cap_has_contact_thread_and_two_independent_bodies():
    stage=Usd.Stage.CreateInMemory()
    from scenes.workrooms.laboratory.tasks.cap_placement.assets import author_task
    meta=author_task(stage)
    assert set(meta['actors'])=={'sample_bottle','cap'}
    thread=meta['task_parameters']['thread']
    assert thread['open_cap_z_m']-thread['closed_cap_z_m']>.004
    assert len(meta['objects'][0]['collider_paths'])==11
    assert meta['objects'][0]['collision_approximation']=='sdf'
    for path in meta['objects'][0]['collider_paths']:
        authored=stage.GetPrimAtPath(path).GetMetadata('apiSchemas').GetAppliedItems()
        assert 'PhysxSDFMeshCollisionAPI' in authored
        assert 'PhysxCollisionAPI' in authored
    assert meta['task_parameters']['holder_inner_half_extents_m'][0]>.022
