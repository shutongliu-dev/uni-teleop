"""The presentation adapter must never alter the physical particle ledger."""
import hashlib
from types import SimpleNamespace

import numpy as np
import pytest
from pxr import Sdf, Usd, UsdGeom, UsdPhysics, UsdShade, Vt

from bench.simulation.isaacsim_backend.particle_surfaces import ParticleSurfaceRenderer


def stage_fixture():
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.Xform.Define(stage, '/World')
    system = stage.DefinePrim('/World/WaterSystem', 'PhysxParticleSystem')
    for name, kind, value in [
        ('physxParticleIsosurface:isosurfaceEnabled', Sdf.ValueTypeNames.Bool, True),
        ('physxParticleSmoothing:particleSmoothingEnabled', Sdf.ValueTypeNames.Bool, False),
        ('fluidRestOffset', Sdf.ValueTypeNames.Float, .004),
    ]:
        system.CreateAttribute(name, kind).Set(value)
    material = UsdShade.Material.Define(stage, '/World/WaterMaterial')
    UsdShade.MaterialBindingAPI.Apply(system).Bind(material)
    particles = UsdGeom.Points.Define(stage, '/World/WaterParticles')
    particles.CreatePointsAttr([(0., 0., .1), (.008, 0., .1)])
    particles.CreateVelocitiesAttr([(0., 0., -.1), (0., 0., -.2)])
    particles.CreateIdsAttr([15, 89])
    particles.GetPrim().CreateAttribute('physxParticle:fluid', Sdf.ValueTypeNames.Bool).Set(True)
    particles.GetPrim().CreateRelationship('physxParticle:particleSystem').SetTargets([system.GetPath()])
    UsdPhysics.MassAPI.Apply(particles.GetPrim()).CreateMassAttr(.001)
    UsdGeom.Mesh.Define(stage, '/World/WaterSystem/Isosurface')
    return stage, particles


@pytest.fixture
def extractor(monkeypatch):
    from bench.simulation.isaacsim_backend import particle_surface_mesh
    calls = []
    def reconstruct(points, spacing_m, *, device):
        calls.append((points.copy(), spacing_m, device))
        vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32) * .004 + points[0]
        return SimpleNamespace(vertices=vertices,
            triangles=np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]]),
            normals=np.ones_like(vertices) / np.sqrt(3.), metadata={'fixture': True})
    monkeypatch.setattr(particle_surface_mesh, 'reconstruct_particle_surface', reconstruct)
    return calls


def test_surface_changes_only_session_visuals_and_hides_both_old_drawables(extractor):
    stage, particles = stage_fixture()
    original = stage.GetRootLayer().ExportToString()
    ledger = {name: particles.GetPrim().GetAttribute(name).Get() for name in
              ['points', 'velocities', 'ids', 'physics:mass']}
    renderer = ParticleSurfaceRenderer(stage)
    report, = renderer.sync()
    assert stage.GetRootLayer().ExportToString() == original
    assert all(particles.GetPrim().GetAttribute(name).Get() == value for name, value in ledger.items())
    assert particles.ComputeVisibility() == 'invisible'
    assert UsdGeom.Imageable(stage.GetPrimAtPath('/World/WaterSystem/Isosurface')).ComputeVisibility() == 'invisible'
    mesh = stage.GetPrimAtPath(report['mesh_path'])
    assert mesh.IsA(UsdGeom.Mesh) and not mesh.HasAPI(UsdPhysics.CollisionAPI)
    assert not mesh.HasAPI(UsdPhysics.RigidBodyAPI)
    assert report['particle_count'] == 2 and report['physics_state_writeback'] is False
    assert report['input_sha256'] == hashlib.sha256(np.asarray(ledger['points'], dtype=np.float32).tobytes()).hexdigest()
    assert len(extractor) == 1


def test_surface_reuses_identical_points_and_refreshes_after_new_tick_or_reset(extractor):
    stage, particles = stage_fixture()
    renderer = ParticleSurfaceRenderer(stage)
    first, = renderer.sync()
    second, = renderer.sync()
    assert second['reused_same_particle_positions'] is True and len(extractor) == 1
    moved = np.asarray(particles.GetPointsAttr().Get(), dtype=np.float32) + [0., 0., .01]
    particles.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(moved.astype(np.float32)))
    third, = renderer.sync()
    assert third['input_sha256'] != first['input_sha256'] and len(extractor) == 2
    assert third['reused_same_particle_positions'] is False
    # Native postprocessing can recreate a visible mesh during World.reset.
    with Usd.EditContext(stage, stage.GetSessionLayer()):
        UsdGeom.Imageable(stage.GetPrimAtPath('/World/WaterSystem/Isosurface')).MakeVisible()
    renderer.hide_native_surfaces()
    assert UsdGeom.Imageable(stage.GetPrimAtPath('/World/WaterSystem/Isosurface')).ComputeVisibility() == 'invisible'


def test_failed_reconstruction_does_not_hide_physical_input(monkeypatch):
    from bench.simulation.isaacsim_backend import particle_surface_mesh
    stage, particles = stage_fixture()
    def fail(*args, **kwargs):
        raise RuntimeError('No surface')
    monkeypatch.setattr(particle_surface_mesh, 'reconstruct_particle_surface', fail)
    with pytest.raises(RuntimeError, match='No surface'):
        ParticleSurfaceRenderer(stage).sync()
    assert particles.ComputeVisibility() == 'inherited'


def test_rebuild_after_session_layer_is_cleared(extractor):
    stage, particles = stage_fixture()
    renderer = ParticleSurfaceRenderer(stage)
    first, = renderer.sync()
    stage.GetSessionLayer().Clear()
    assert not stage.GetPrimAtPath(first['mesh_path'])
    second, = renderer.sync()
    assert not second['reused_same_particle_positions'] and len(extractor) == 2
    assert stage.GetPrimAtPath(second['mesh_path'])
    assert particles.ComputeVisibility() == 'invisible'


def test_disabled_or_absent_surface_keeps_ordinary_particles_unchanged():
    stage, particles = stage_fixture()
    stage.GetPrimAtPath('/World/WaterSystem').GetAttribute('physxParticleIsosurface:isosurfaceEnabled').Set(False)
    original = stage.GetRootLayer().ExportToString()
    renderer = ParticleSurfaceRenderer(stage)
    assert not renderer and renderer.sync() == []
    assert particles.ComputeVisibility() == 'inherited'
    assert stage.GetRootLayer().ExportToString() == original
