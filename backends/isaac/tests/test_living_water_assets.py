"""CPU authoring checks, never a liquid dynamics qualification."""
from copy import deepcopy
import json

import numpy as np
import pytest
import trimesh
from pxr import Gf, Usd, UsdGeom, UsdPhysics

from scenes.workrooms.living_room.tasks.water_pouring import assets


def make():
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageMetersPerUnit(stage, 1.)
    UsdGeom.SetStageUpAxis(stage, 'Z')
    return stage, assets.author_task(stage)


@pytest.mark.parametrize('outer,inner', [(assets.KETTLE_OUTER, assets.KETTLE_INNER),
                                      (assets.CUP_OUTER, assets.CUP_INNER)])
def test_shell_is_closed_positive_volume_material_with_actual_inner_faces(outer, inner):
    points, faces = assets.vessel_mesh(outer, inner)
    mesh = trimesh.Trimesh(vertices=points, faces=faces, process=False)
    assert mesh.is_watertight and mesh.is_winding_consistent and mesh.volume > 0
    assert mesh.euler_number == 2
    # A solid cylinder would have far more material and no open top cavity.
    assert mesh.volume < np.pi * max(row[1] for row in outer)**2 * outer[-1][0] * .35


def test_author_is_local_preserves_parent_and_uses_real_fluid_schemas():
    stage, meta = make()
    assert stage.GetPrimAtPath('/World/PhysicsScene').IsValid() is False
    assert all(str(p.GetPath()).startswith('/World/LivingTask') or str(p.GetPath()) == '/World'
               for p in stage.Traverse())
    assert json.loads(json.dumps(meta))['water']['particle_count'] == 9259
    assert meta['native_accepted'] is False
    for role in ('kettle', 'cup'):
        body = stage.GetPrimAtPath(meta['containers'][role]['path'])
        assert UsdPhysics.RigidBodyAPI(body).GetRigidBodyEnabledAttr().Get()
        assert not UsdPhysics.RigidBodyAPI(body).GetKinematicEnabledAttr().Get()
        shell = stage.GetPrimAtPath(str(body.GetPath())+'/HollowShell')
        assert UsdPhysics.MeshCollisionAPI(shell).GetApproximationAttr().Get() == 'sdf'
    water = stage.GetPrimAtPath(meta['water']['particles_path'])
    assert water.GetAttribute('physxParticle:fluid').Get() is True
    assert UsdPhysics.MassAPI(water).GetMassAttr().Get() == pytest.approx(.25)
    assert meta['containers']['kettle']['dry_mass_kg'] == pytest.approx(.32)
    with pytest.raises(ValueError, match='already exists'):
        assets.author_task(stage)


def test_initial_particles_are_inside_true_cavity_with_positive_wall_floor_clearance():
    stage, meta = make()
    c = meta['containers']['kettle']
    points = np.asarray(UsdGeom.Points(stage.GetPrimAtPath(meta['water']['particles_path'])).GetPointsAttr().Get())
    inverse = UsdGeom.XformCache().GetLocalToWorldTransform(stage.GetPrimAtPath(c['path'])).GetInverse()
    local = np.array([inverse.Transform(Gf.Vec3d(*p.astype(float))) for p in points])
    assert assets.cavity_contains(local, c['inner_profile_z_radius_spout_m']).all()
    assert local[:, 2].min() - .006 > .0029
    assert np.linalg.norm(local[:, :2], axis=1).max() < .042
    assert len(set(map(tuple, points.tolist()))) == 9259
    assert meta['water']['particle_mass_kg'] * len(points) == pytest.approx(.25)


def test_readback_refuses_smoothed_points_and_detects_missing_inventory():
    stage, meta = make()
    system = stage.GetPrimAtPath(meta['water']['system_path'])
    smoothing = system.GetAttribute('physxParticleSmoothing:particleSmoothingEnabled')
    smoothing.Set(True)
    with pytest.raises(RuntimeError, match='no render fallback'):
        assets.read_water_state(stage, meta)
    smoothing.Set(False)
    # Initial authored state is not evidence of native progress.
    p = stage.GetPrimAtPath(meta['water']['particles_path'])
    state = assets.read_water_state(stage, meta)
    assert state['native_progress_verified_here'] is False
    assert state['counts'] == {'kettle': 9259, 'cup': 0, 'outside_vessels': 0}
    assert sum(state['mass_kg'].values()) == pytest.approx(.25)
    UsdGeom.Points(p).GetIdsAttr().Set([0])
    with pytest.raises(RuntimeError, match='inventory'):
        assets.read_water_state(stage, meta)


def test_presentation_preserves_particle_inventory_and_dynamics():
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageMetersPerUnit(stage, 1.)
    meta = assets.author_task(stage, '/World/Presentation', presentation=True)
    assert meta['water']['particle_count'] == 9259
    assert UsdPhysics.RigidBodyAPI(stage.GetPrimAtPath(meta['containers']['kettle']['path'])).GetRigidBodyEnabledAttr().Get()


def test_export_reopens_without_losing_body_or_water_source_state(tmp_path):
    stage, meta = make()
    path = tmp_path/'candidate.usda'
    assert stage.Export(str(path))
    reopened = Usd.Stage.Open(str(path))
    assert len(UsdGeom.Points(reopened.GetPrimAtPath(meta['water']['particles_path'])).GetPointsAttr().Get()) == 9259
    assert reopened.GetPrimAtPath(meta['water']['system_path']).GetTypeName() == 'PhysxParticleSystem'
    assert reopened.GetPrimAtPath(meta['water']['system_path']).GetRelationship('simulationOwner').GetTargets()[0] == '/World/PhysicsScene'


@pytest.mark.parametrize('mass', [.125, .25, .375])
def test_water_amount_changes_inventory_without_changing_dry_body_mass(mass):
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageMetersPerUnit(stage, 1.)
    meta = assets.author_task(stage, water_mass_kg=mass)
    assert meta['water']['particle_count'] == round(mass/(1000*.003**3))
    assert meta['water']['particle_count']*meta['water']['particle_mass_kg'] == pytest.approx(mass)
    assert meta['water']['initial_volume_ml'] == pytest.approx(mass*1000)
    assert UsdPhysics.MassAPI(stage.GetPrimAtPath(meta['containers']['kettle']['path'])).GetMassAttr().Get() == pytest.approx(.32)
    state = assets.read_water_state(stage, meta)
    assert state['counts']['kettle'] == meta['water']['particle_count']


@pytest.mark.parametrize('mass', [0., -1., float('nan'), float('inf'), True, .6])
def test_bad_or_overfilled_particle_inventory_is_rejected(mass):
    with pytest.raises(ValueError):
        assets.initial_particles([.52,-.055,.7861], 0., water_mass_kg=mass)


def test_native_world_pose_classification_does_not_need_live_usd_write():
    stage, meta = make()
    poses = {}
    cache = UsdGeom.XformCache()
    for name, info in meta['containers'].items():
        matrix = cache.GetLocalToWorldTransform(stage.GetPrimAtPath(info['path']))
        q = matrix.ExtractRotationQuat()
        poses[name] = list(matrix.ExtractTranslation()) + list(q.GetImaginary()) + [q.GetReal()]
    baseline = assets.read_water_state(stage, meta, container_pose_xyzw=poses)
    assert baseline['counts']['kettle'] == 9259
    poses['kettle'][0] += 1.
    moved = assets.read_water_state(stage, meta, container_pose_xyzw=poses)
    assert moved['counts']['outside_vessels'] == 9259
    assert assets.read_water_state(stage, meta)['counts']['kettle'] == 9259
    assert moved['container_pose_source'] == 'native_world_pose_argument'
    with pytest.raises(RuntimeError, match='inventory'):
        assets.read_water_state(stage, meta, container_pose_xyzw={})
    poses['kettle'][6] = 2.
    with pytest.raises(RuntimeError, match='Invalid native'):
        assets.read_water_state(stage, meta, container_pose_xyzw=poses)


def test_tray_rim_metadata_matches_actual_authored_wall_geometry():
    stage, meta = make()
    tray = meta['tray']
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
    for name in ['Front', 'Rear', 'Left', 'Right']:
        bounds = cache.ComputeWorldBound(stage.GetPrimAtPath(tray['path']+'/'+name)).ComputeAlignedRange()
        assert bounds.GetMax()[2] == pytest.approx(tray['rim_top_z_m'])
        size = bounds.GetSize()
        assert min(size[0], size[1]) == pytest.approx(tray['wall_thickness_m'])
