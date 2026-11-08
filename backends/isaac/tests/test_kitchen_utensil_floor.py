"""The contact grid changes floor topology, never support shape or materials."""
import numpy as np
import pytest
from pxr import Usd, UsdGeom
import trimesh

from scenes.workrooms.kitchen.assets import author_task


@pytest.mark.parametrize('presentation', [False, True])
@pytest.mark.parametrize('task_id, profile, spacing', [
    ('utensil_sorting', 'legacy', .020),
    ('utensil_sorting', 'floor_grid_2mm_v1', .002),
    ('table_clearing', 'floor_grid_2mm_v1', .002),
])
def test_household_floor_grid_preserves_the_solid_and_visuals(monkeypatch, presentation,
                                                            task_id, profile, spacing):
    from scenes.workrooms.kitchen import utensil_floor
    baseline = Usd.Stage.CreateInMemory()
    with monkeypatch.context() as patch:
        patch.setattr(utensil_floor, 'refine_sorting_tray_floor', lambda author, **kwargs: None)
        before = author_task(baseline, task_id, prop_profile='household_v1',
                             presentation=presentation, utensil_contact_profile=profile)
    refined = Usd.Stage.CreateInMemory()
    after = author_task(refined, task_id, prop_profile='household_v1',
                        presentation=presentation, utensil_contact_profile=profile)
    assert before['objects'] == after['objects']
    assert before['goals'] == after['goals']
    row = after['fixture_metadata']['cutlery_tray']
    receipt = row['contact_tessellation']
    path = receipt['collider_path']
    assert receipt['grid_spacing_m'] == spacing
    assert receipt['maximum_interior_triangle_edge_m'] < spacing*1.45
    assert receipt['original_boundary_edge_count'] == 48
    assert receipt['new_floor_triangle_count'] > receipt['old_floor_triangle_count']
    shapes = []
    for stage in (baseline, refined):
        usd = UsdGeom.Mesh(stage.GetPrimAtPath(path))
        shapes.append(trimesh.Trimesh(np.asarray(usd.GetPointsAttr().Get()),
            np.asarray(usd.GetFaceVertexIndicesAttr().Get()).reshape(-1, 3), process=False))
    old, new = shapes
    assert new.is_watertight and new.is_winding_consistent
    np.testing.assert_array_equal(new.vertices[:len(old.vertices)], old.vertices)
    np.testing.assert_array_equal(new.bounds, old.bounds)
    assert new.volume == pytest.approx(old.volume, rel=1e-12, abs=1e-15)
    np.testing.assert_allclose(new.center_mass, old.center_mass, atol=1e-12, rtol=0.)
    nonfloor = old.faces[~((np.ptp(old.triangles[:, :, 2], axis=1) < 1e-10)
        & (np.abs(old.triangles[:, 0, 2]-.005) < 1e-8)
        & (old.face_normals[:, 2] > 1.-1e-10))]
    np.testing.assert_array_equal(new.faces[:len(nonfloor)], nonfloor)
    allowed = {'points', 'faceVertexCounts', 'faceVertexIndices', 'normals'}
    differences = []
    for prim in refined.Traverse():
        previous = baseline.GetPrimAtPath(prim.GetPath())
        assert previous
        for attr in prim.GetAttributes():
            if repr(attr.Get()) != repr(previous.GetAttribute(attr.GetName()).Get()):
                differences.append(str(attr.GetPath()))
                assert str(prim.GetPath()) == path
                assert attr.GetName() in allowed
        sdf = prim.GetAttribute('physxSDFMeshCollision:sdfResolution')
        if sdf and sdf.HasValue():
            assert sdf.Get() == 384
    assert len(differences) == 4


def test_legacy_floor_keeps_its_original_contact_topology():
    stage = Usd.Stage.CreateInMemory()
    meta = author_task(stage, 'utensil_sorting')
    assert 'contact_tessellation' not in meta['fixture_metadata']['cutlery_tray']


@pytest.mark.parametrize('task_id, fixture', [('utensil_sorting', 'source_pad'),
                                            ('table_clearing', 'source_mat')])
@pytest.mark.parametrize('profile, spacing', [('floor_grid_2mm_convex_v2', .002),
                                            ('source_grid_1mm_convex_v3', .001)],
                         ids=['source_2mm', 'source_1mm'])
def test_source_contact_grid_keeps_triangle_collision_and_exact_solid(task_id, fixture, profile, spacing):
    stages = [Usd.Stage.CreateInMemory(), Usd.Stage.CreateInMemory()]
    before, after = [author_task(stage, task_id, prop_profile='household_v1',
        utensil_contact_profile=selected) for stage, selected in zip(stages,
        ('floor_grid_2mm_convex_v1', profile))]
    assert before['objects'] == after['objects'] and before['goals'] == after['goals']
    receipt = after['fixture_metadata'][fixture].pop('contact_tessellation')
    assert before['fixture_metadata'] == after['fixture_metadata']
    assert receipt['grid_spacing_m'] == spacing
    assert receipt['original_boundary_edge_count'] == 48
    path = receipt['collider_path']
    meshes = []
    for stage in stages:
        usd = UsdGeom.Mesh(stage.GetPrimAtPath(path))
        assert usd.GetPrim().GetAttribute('physics:approximation').Get() == 'none'
        meshes.append(trimesh.Trimesh(np.asarray(usd.GetPointsAttr().Get()),
            np.asarray(usd.GetFaceVertexIndicesAttr().Get()).reshape(-1, 3), process=False))
    old, new = meshes
    np.testing.assert_array_equal(new.vertices[:len(old.vertices)], old.vertices)
    np.testing.assert_array_equal(new.bounds, old.bounds)
    assert new.volume == pytest.approx(old.volume, rel=1e-12, abs=1e-15)
    np.testing.assert_allclose(new.center_mass, old.center_mass, rtol=0., atol=1e-12)
    np.testing.assert_allclose(new.moment_inertia, old.moment_inertia, rtol=1e-10, atol=1e-16)
    differences = []
    for prim in stages[1].Traverse():
        previous = stages[0].GetPrimAtPath(prim.GetPath())
        assert previous
        for attr in prim.GetAttributes():
            if repr(attr.Get()) != repr(previous.GetAttribute(attr.GetName()).Get()):
                differences.append(str(attr.GetPath()))
                assert str(prim.GetPath()) == path
                assert attr.GetName() in {'points', 'faceVertexCounts', 'faceVertexIndices', 'normals'}
    assert len(differences) == 4
