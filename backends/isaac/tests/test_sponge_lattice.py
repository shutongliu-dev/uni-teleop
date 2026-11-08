"""Independent physical author invariants; no native qualification implied."""
import copy
import numpy as np
import pytest
import trimesh
from scipy.spatial.transform import Rotation
from pxr import Usd, UsdGeom, UsdPhysics

from scenes.workrooms.kitchen.assets import author_task
from scenes.workrooms.kitchen.sponge_deformable import foam_topology, FOAM_MASS_KG, TOTAL_MASS_KG
from scenes.workrooms.kitchen.sponge_lattice import lattice_geometry, author_lattice_sponge


def authored():
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageMetersPerUnit(stage, 1.)
    UsdGeom.SetStageUpAxis(stage, 'Z')
    meta = author_task(stage, 'surface_wiping', prop_profile='household_v1')
    original = copy.deepcopy(meta)
    record = author_lattice_sponge(stage, meta)
    return stage, meta, record, original


def test_partition_preserves_volume_com_inertia_and_original_taper():
    cells = lattice_geometry()
    assert len(cells) == 27
    assert all(cell.is_watertight and cell.is_convex and cell.volume > 0 for cell in cells.values())
    points, _, faces, _ = foam_topology()
    baseline = trimesh.Trimesh(points, faces, process=True)
    whole = trimesh.util.concatenate(list(cells.values()))
    assert whole.volume == pytest.approx(baseline.volume, rel=1e-7)
    np.testing.assert_allclose(whole.center_mass, baseline.center_mass, atol=1e-9)
    np.testing.assert_allclose(whole.moment_inertia/whole.volume,
                               baseline.moment_inertia/baseline.volume, rtol=1e-7, atol=1e-11)
    np.testing.assert_allclose(whole.bounds, baseline.bounds, atol=1e-8)


def test_mass_and_native_inventory_are_complete_and_goals_transform_all_cells():
    stage, meta, record, original = authored()
    assert (record['native_body_count'], record['logical_object_count'], record['joint_count']) == (28, 1, 63)
    assert sum(row['mass_kg'] for row in meta['objects']) == pytest.approx(TOTAL_MASS_KG, abs=1e-15)
    assert sum(row['mass_kg'] for row in record['cells']) == pytest.approx(FOAM_MASS_KG, abs=1e-15)
    native = {str(p.GetPath()): UsdPhysics.MassAPI(p).GetMassAttr().Get()
              for p in stage.Traverse() if p.HasAPI(UsdPhysics.RigidBodyAPI)}
    assert set(native) == set(meta['actors'].values())
    assert sum(native.values()) == pytest.approx(TOTAL_MASS_KG, rel=1e-7)
    for row in meta['objects']:
        assert np.all(np.array(row['diagonal_inertia_kg_m2']) > 0)
        assert UsdPhysics.RigidBodyAPI(stage.GetPrimAtPath(row['body_path'])).GetKinematicEnabledAttr().Get() is False
    assert set(meta['goals']['body_poses_xyzw']) == set(meta['actors'])
    assert meta['goals']['body_poses_xyzw']['sponge'] == original['goals']['body_poses_xyzw']['sponge']
    assert meta['public_goal'] == original['public_goal']


def test_springs_have_coincident_rest_anchors_and_finite_physical_coefficients():
    stage, meta, record, _ = authored()
    rows = {x['id']: x for x in meta['objects']}
    assert set(meta['joints'].values()) == {x['prim_path'] for x in record['links']}
    for link in record['links']:
        joint = UsdPhysics.Joint(stage.GetPrimAtPath(link['prim_path']))
        anchors, frames = [], []
        for i in range(2):
            row = rows[link['body'+str(i)]]
            pose = row['initial_pose_xyzw']
            rotation = Rotation.from_quat(pose[3:])
            local = np.array(getattr(joint, 'GetLocalPos'+str(i)+'Attr')().Get())
            anchors.append(np.array(pose[:3])+rotation.apply(local))
            quat = getattr(joint, 'GetLocalRot'+str(i)+'Attr')().Get()
            frames.append(rotation*Rotation.from_quat([*quat.GetImaginary(), quat.GetReal()]))
        np.testing.assert_allclose(*anchors, atol=1e-8)
        np.testing.assert_allclose(frames[0].as_matrix(), frames[1].as_matrix(), atol=1e-8)
        assert joint.GetExcludeFromArticulationAttr().Get() is True
        assert joint.GetCollisionEnabledAttr().Get() is False
        assert not any(name.startswith('PhysicsLimitAPI') for name in joint.GetPrim().GetAppliedSchemas())
        for axis, properties in link['drives'].items():
            api = UsdPhysics.DriveAPI(joint.GetPrim(), axis)
            assert api.GetTypeAttr().Get() == 'force'
            assert api.GetTargetPositionAttr().Get() == api.GetTargetVelocityAttr().Get() == 0
            assert api.GetStiffnessAttr().Get() == pytest.approx(properties['usd_stiffness'], rel=1e-7)
            assert all(np.isfinite(v) and v > 0 for v in properties.values())
        assert link['drives']['rotY'] == link['drives']['rotZ']


def test_filter_is_only_internal_and_skin_binding_reconstructs_rest_exactly():
    stage, meta, record, _ = authored()
    paths = set(meta['actors'].values())
    for path in paths:
        targets = set(map(str, UsdPhysics.FilteredPairsAPI(stage.GetPrimAtPath(path)).GetFilteredPairsRel().GetTargets()))
        assert targets == paths-{path}
    binding = record['visual_skin_bindings'][0]
    import hashlib
    from bench.simulation.isaacsim_backend.elastic_visual_skin import ElasticVisualSkin
    points = np.asarray(UsdGeom.Mesh(stage.GetPrimAtPath(binding['prim_path'])).GetPointsAttr().Get(), float)
    assert binding['rest_points_sha256'] == hashlib.sha256(points.tobytes()).hexdigest()
    assert 'elastic_lattices' not in meta and 'rest_points_m' not in binding
    renderer = ElasticVisualSkin(stage, meta['elastic_composites'])
    indices, weights = renderer.bindings[0]['indices'], renderer.bindings[0]['weights']
    centers = np.array([row['composite_rest_center_m'] for row in record['cells']])
    local = points[:, None]-centers[indices]
    reconstructed = ((local+centers[indices])*weights[:, :, None]).sum(1)
    np.testing.assert_allclose(reconstructed, points, atol=1e-15)
    np.testing.assert_allclose(weights.sum(1), 1, atol=1e-15)
    assert not binding['runtime_deformation_verified']
    assert UsdGeom.Imageable(stage.GetPrimAtPath(binding['prim_path'])).GetVisibilityAttr().Get() == 'invisible'
    assert not record['physics_accepted']


def test_default_is_unchanged_and_double_application_is_rejected():
    stage, meta, _, _ = authored()
    with pytest.raises(ValueError, match='fresh rigid'):
        author_lattice_sponge(stage, meta)
    ordinary = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageMetersPerUnit(ordinary, 1.)
    legacy = author_task(ordinary, 'surface_wiping', prop_profile='household_v1')
    assert len(legacy['objects']) == 1
    assert not any(p.IsA(UsdPhysics.Joint) for p in ordinary.Traverse())
