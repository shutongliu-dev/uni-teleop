"""Physical topology and bond contract checks; no native success is inferred."""
import numpy as np
import pytest
import trimesh
from pxr import Usd, UsdGeom, UsdPhysics

from scenes.workrooms.kitchen.assets import author_task
from scenes.workrooms.kitchen.sponge_deformable import (
    foam_topology, topology_quality, _signed_tet_volumes, TOTAL_MASS_KG, FOAM_MASS_KG, SCOURING_MASS_KG,
    PROFILE, COARSE_PROFILE, PROFILES)
from scripts.assets.build_kitchen_tasks import audit_task


def authored(profile='bonded_fem_v1'):
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageMetersPerUnit(stage, 1.)
    UsdGeom.SetStageUpAxis(stage, 'Z')
    metadata = author_task(stage, 'surface_wiping', prop_profile='household_v1',
                          sponge_physics_profile=profile)
    return stage, metadata


@pytest.mark.parametrize('profile', PROFILES)
def test_tets_are_positive_conforming_connected_and_closed(profile):
    points, tets, surface, bonded = foam_topology(profile)
    volumes = _signed_tet_volumes(points.astype(float), tets)
    assert (volumes*6).min() > 4e-9  # PhysX initialization rejects det <= 1e-9
    assert len(np.unique(tets)) == len(points)
    assert len(np.unique(np.sort(tets, axis=1), axis=0)) == len(tets)
    shell = trimesh.Trimesh(points, surface, process=True)
    assert shell.is_watertight and shell.is_winding_consistent
    assert shell.volume == pytest.approx(volumes.sum(), rel=1e-6)
    assert len(shell.split(only_watertight=False)) == 1
    np.testing.assert_allclose(points[bonded, 2], .004, atol=1e-9)
    assert set(bonded) == set(np.flatnonzero(np.isclose(points[:, 2], .004)))
    np.testing.assert_allclose(points.min(0), [-.06, -.039, .004], atol=1e-8)
    np.testing.assert_allclose(points.max(0), [.06, .039, .045], atol=1e-8)
    quality = topology_quality(points, tets)
    assert quality['maximum_rest_matrix_condition_number'] < 50
    assert quality['minimum_tetrahedron_mean_ratio'] > .1
    assert quality['hypothetical_mass_ratio_if_inverse_mass_clamped_at_50'] < 1.001


@pytest.mark.parametrize('profile', PROFILES)
def test_composite_mass_bond_and_visual_owner_are_exact(profile):
    stage, meta = authored(profile)
    old_stage, old = authored('rigid')
    v = meta['volume_bodies']['foam']
    row = meta['objects'][0]
    assert row['mass_kg'] == SCOURING_MASS_KG
    assert v['total_mass_kg'] == FOAM_MASS_KG
    assert row['mass_kg']+v['total_mass_kg'] == TOTAL_MASS_KG
    assert UsdPhysics.MassAPI(stage.GetPrimAtPath(row['body_path'])).GetMassAttr().Get() == pytest.approx(SCOURING_MASS_KG)
    assert meta['composite_objects']['sponge']['release_requires_all_components'] is True
    bond = stage.GetPrimAtPath(v['attachment']['prim_path'])
    assert bond.GetTypeName() == 'OmniPhysicsVtxXformAttachment'
    assert list(map(str, bond.GetRelationship('omniphysics:src0').GetTargets())) == [v['simulation_mesh_path']]
    assert list(map(str, bond.GetRelationship('omniphysics:src1').GetTargets())) == [row['body_path']]
    assert list(bond.GetAttribute('omniphysics:vtxIndicesSrc0').Get()) == v['attachment']['node_indices']
    np.testing.assert_array_equal(bond.GetAttribute('omniphysics:localPositionsSrc1').Get(),
                                  v['attachment']['rigid_local_positions_m'])
    root = stage.GetPrimAtPath(v['prim_path'])
    assert root.GetAttribute('omniphysics:deformableBodyEnabled').Get() is True
    assert root.GetAttribute('omniphysics:kinematicEnabled').Get() is False
    for p in Usd.PrimRange(root):
        assert not any('kinematicTarget' in a.GetName() for a in p.GetAttributes())
    skin = stage.GetPrimAtPath(v['visual_bindings'][0]['prim_path'])
    old_skin = old_stage.GetPrimAtPath(row['body_path']+'/Visual/finish_porous_foam')
    assert not stage.GetPrimAtPath(str(old_skin.GetPath()))
    for name in ('points', 'faceVertexIndices', 'primvars:displayColor'):
        np.testing.assert_array_equal(skin.GetAttribute(name).Get(), old_skin.GetAttribute(name).Get())
    np.testing.assert_array_equal(skin.GetAttribute('deformablePose:default:omniphysics:points').Get(),
                                  skin.GetAttribute('points').Get())
    assert meta['goals'] == old['goals']
    assert v['physics_accepted'] is False
    audit = audit_task(stage, 'surface_wiping', meta)
    assert audit['dynamic_mass_kg'] == pytest.approx(TOTAL_MASS_KG)
    assert audit['volume_body_count'] == 1


@pytest.mark.parametrize('task,prop,profile', [
    ('surface_wiping', 'legacy', 'bonded_fem_v1'),
    ('breakfast_serving', 'household_v1', 'bonded_fem_v1'),
    ('surface_wiping', 'legacy', COARSE_PROFILE),
    ('breakfast_serving', 'household_v1', COARSE_PROFILE),
    ('surface_wiping', 'household_v1', 'unknown'),
])
def test_profile_is_explicit_and_task_scoped(task, prop, profile):
    with pytest.raises(ValueError):
        author_task(Usd.Stage.CreateInMemory(), task, prop_profile=prop, sponge_physics_profile=profile)


def test_coarse_discretization_retains_boundary_volume_and_mass_distribution():
    from scipy.spatial import ConvexHull
    original, original_tets, original_surface, _ = foam_topology()
    coarse, coarse_tets, coarse_surface, bonded = foam_topology(COARSE_PROFILE)
    assert (len(coarse), len(coarse_tets), len(bonded)) == (1350, 6048, 150)
    np.testing.assert_array_equal(np.unique(original[:, 2]), np.unique(coarse[:, 2]))
    for height in np.unique(original[:, 2]):
        outlines = []
        for points in (original, coarse):
            layer = points[points[:, 2] == height, :2]
            vertices = layer[ConvexHull(layer).vertices]
            outlines.append(vertices[np.lexsort((vertices[:, 1], vertices[:, 0]))])
        np.testing.assert_array_equal(*outlines)
    a = trimesh.Trimesh(original, original_surface, process=True).mass_properties
    b = trimesh.Trimesh(coarse, coarse_surface, process=True).mass_properties
    assert a.volume == pytest.approx(b.volume, rel=1e-7)
    np.testing.assert_allclose(a.center_mass, b.center_mass, atol=1e-9, rtol=0)
    np.testing.assert_allclose(a.inertia/a.mass, b.inertia/b.mass, atol=1e-10, rtol=1e-7)
    assert _signed_tet_volumes(coarse.astype(float), coarse_tets).sum() == pytest.approx(a.volume, rel=1e-7)


def test_coarse_profile_preserves_material_pad_and_detailed_skin():
    baseline_stage, baseline = authored(PROFILE)
    stage, meta = authored(COARSE_PROFILE)
    assert baseline['objects'] == meta['objects']
    assert baseline['composite_objects'] == meta['composite_objects']
    first, second = (x['volume_bodies']['foam'] for x in (baseline, meta))
    for key in ('total_mass_kg', 'initial_pose_xyzw', 'physical_boundary_change', 'limitations'):
        assert first[key] == second[key]
    assert set(first['material']) == set(second['material'])
    for key in first['material']:
        if key == 'density_kg_m3':
            assert first['material'][key] == pytest.approx(second['material'][key], rel=1e-7)
        else:
            assert first['material'][key] == second['material'][key]
    for prim_path in (first['prim_path'], first['visual_bindings'][0]['prim_path'],
                      baseline['actors']['sponge']+'/Visual/finish_green_scouring_layer'):
        old, new = (s.GetPrimAtPath(prim_path) for s in (baseline_stage, stage))
        assert old.GetAppliedSchemas() == new.GetAppliedSchemas()
        assert {a.GetName(): a.Get() for a in old.GetAttributes()} == {a.GetName(): a.Get() for a in new.GetAttributes()}
    comparison = second['discretization_comparison']
    assert comparison['candidate_only'] and not comparison['native_validation_complete']
    assert comparison['physical_parameters_unchanged']
    assert 'discretization_comparison' not in first
