"""Original-corner compound geometry, cavity and authoring preservation checks."""
from types import SimpleNamespace

import numpy as np
import pytest
from pxr import Usd, UsdGeom, UsdPhysics, UsdShade
from scipy.spatial import ConvexHull
import trimesh

from scenes.workrooms.kitchen.assets import author_task
from scenes.workrooms.kitchen.utensil_spoon_convex import (
    apply_spoon_convex_colliders, partition_spoon_mesh,
)


def _author(task):
    stage = Usd.Stage.CreateInMemory()
    meta = author_task(stage, task, prop_profile='household_v1')
    return SimpleNamespace(stage=stage, meta=meta,
        materials={'steel': UsdShade.Material.Get(stage, '/World/KitchenTask/Materials/steel')},
        physical_material=UsdShade.Material.Get(stage, '/World/KitchenTask/Materials/Contact'))


def _mesh(stage, path):
    mesh = UsdGeom.Mesh(stage.GetPrimAtPath(path))
    return trimesh.Trimesh(np.asarray(mesh.GetPointsAttr().Get()),
        np.asarray(mesh.GetFaceVertexIndicesAttr().Get()).reshape(-1, 3), process=False)


def _parameters(row):
    return dict(rest_transform=row.get('natural_rest', {}).get('source_to_rest_transform'),
                scale_xyz=row['dimension_profile']['linear_scale_xyz'])


def _snapshot(stage):
    return {str(prim.GetPath()): {
        'schemas': list(prim.GetAppliedSchemas()),
        'attributes': {a.GetName(): str(a.Get()) for a in prim.GetAttributes()},
        'relationships': {r.GetName(): list(map(str, r.GetTargets())) for r in prim.GetRelationships()}}
        for prim in stage.Traverse()}


@pytest.mark.parametrize('task', ['utensil_sorting', 'table_clearing'])
def test_compound_preserves_every_existing_property_except_old_collision_enable(task):
    author = _author(task)
    before = _snapshot(author.stage)
    spoons = [row for row in author.meta['objects'] if row['role'] == 'spoon']
    originals = {row['id']: _mesh(author.stage, row['collider_paths'][0]) for row in spoons}
    old_paths = {row['collider_paths'][0] for row in spoons}
    apply_spoon_convex_colliders(author)
    after = _snapshot(author.stage)
    for path, old in before.items():
        if path in old_paths:
            old['attributes']['physics:collisionEnabled'] = 'False'
        assert after[path] == old, path
    for row in spoons:
        receipt = row['spoon_convex_collision_partition']
        assert row['collision_count'] == receipt['part_count'] == 220
        assert receipt['maximum_vertex_count'] <= 16
        assert receipt['native_physics_qualified'] is False
        reference = receipt['finite_precision_reference']
        assert reference['original_boundary_triangles'] == 1008
        assert reference['reference_tetrahedra'] == 1224
        assert reference['bowl_convex_union_to_original_solid_distance_bound_m'] < 2e-8
        points = set(map(tuple, originals[row['id']].vertices))
        for path in row['collider_paths']:
            prim = author.stage.GetPrimAtPath(path)
            assert not prim.HasAPI(UsdPhysics.RigidBodyAPI)
            assert UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr().Get() == 'convexHull'
            assert prim.GetAttribute('physxCollision:contactOffset').Get() == pytest.approx(.0002)
            assert set(map(tuple, _mesh(author.stage, path).vertices)) <= points
        # The central bowl remains open above its original 3 mm floor.
        transform = np.asarray(row.get('natural_rest', {}).get('source_to_rest_transform', np.eye(4)))
        cavity = trimesh.transform_points(np.array([[0., -.053, z] for z in np.linspace(.0031, .020, 35)]), transform)
        cavity *= row['dimension_profile']['linear_scale_xyz']
        for path in row['collider_paths']:
            hull = ConvexHull(_mesh(author.stage, path).vertices)
            assert not np.any(np.all(cavity@hull.equations[:, :3].T+hull.equations[:, 3] <= 1e-10, axis=1))
    with pytest.raises(ValueError, match='original enabled|already been authored'):
        apply_spoon_convex_colliders(author)


def test_reordered_actual_vertices_preserve_partition_and_reject_changed_surface():
    author = _author('utensil_sorting')
    row = next(row for row in author.meta['objects'] if row['role'] == 'spoon')
    original = _mesh(author.stage, row['collider_paths'][0])
    order = np.random.default_rng(73).permutation(len(original.vertices))
    inverse = np.empty(len(order), dtype=int)
    inverse[order] = np.arange(len(order))
    reordered = trimesh.Trimesh(original.vertices[order], inverse[original.faces], process=False)
    parts, receipt = partition_spoon_mesh(reordered, **_parameters(row))
    assert len(parts) == 220
    assert abs(receipt['relative_volume_difference']) < 1e-6
    assert receipt['maximum_COM_difference_m'] < 1e-8
    assert receipt['relative_unit_density_inertia_difference'] < 1e-6
    changed = reordered.copy()
    changed.vertices[0, 0] += 1e-6
    with pytest.raises(ValueError, match='recipe|topology'):
        partition_spoon_mesh(changed, **_parameters(row))
    missing_face = trimesh.Trimesh(reordered.vertices, reordered.faces[1:], process=False)
    with pytest.raises(ValueError, match='topology'):
        partition_spoon_mesh(missing_face, **_parameters(row))


def test_bad_transform_is_not_silently_fitted_to_a_different_body():
    author = _author('table_clearing')
    row = next(row for row in author.meta['objects'] if row['role'] == 'spoon')
    mesh = _mesh(author.stage, row['collider_paths'][0])
    with pytest.raises(ValueError, match='Invalid spoon'):
        partition_spoon_mesh(mesh, scale_xyz=[1.15, -1.15, 1.15])
    transform = np.eye(4)
    transform[0, 3] = .001
    with pytest.raises(ValueError, match='recipe'):
        partition_spoon_mesh(mesh, rest_transform=transform, scale_xyz=[1.15]*3)
