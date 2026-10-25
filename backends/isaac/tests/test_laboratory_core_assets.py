"""Structural contracts and usable clearances for the core laboratory tasks."""
import importlib
import math
import numpy as np
import pytest
from pxr import Usd, UsdPhysics

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
    assert all(abs(o['orientation_xyzw'][0]-math.sqrt(.5))<1e-8 for o in meta['objects'])


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
