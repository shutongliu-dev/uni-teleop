"""Stationery layout geometry checks; these do not qualify robot manipulation."""
from itertools import combinations

import numpy as np
import pytest

from scenes.workrooms.study import task_scene
from scenes.workrooms.study.rules import _reset_bounds


def _separated(first, second):
    return np.any(first[1] <= second[0]) or np.any(second[1] <= first[0])


@pytest.fixture
def stationery_scene(tmp_path):
    from pxr import Usd

    config = task_scene.configuration('stationery_sorting')
    stage = Usd.Stage.CreateInMemory()
    builder = task_scene._builder(stage)
    assets = {
        ident: task_scene._stationery(stage, builder, ident, obj, tmp_path,
                                     task_scene.DEFAULT_ASSETS_ROOT)
        for ident, obj in config['objects'].items()
    }
    task_scene._fixtures(builder, 'stationery_sorting', config, tmp_path,
                         task_scene.DEFAULT_ASSETS_ROOT)
    task_scene._physics(stage, stage.GetPrimAtPath(task_scene.TASK_ROOT + '/Fixtures'))
    return stage, config, assets


def test_stationery_supply_has_a_clear_lane_beside_the_nearby_organizer(stationery_scene):
    from pxr import Usd, UsdGeom, UsdPhysics

    stage, config, _ = stationery_scene
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ['default', 'render'])
    rails = []
    for prim in Usd.PrimRange(stage.GetPrimAtPath(task_scene.TASK_ROOT + '/Fixtures')):
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            bounds = cache.ComputeWorldBound(prim).ComputeAlignedRange()
            rails.append(np.array([bounds.GetMin(), bounds.GetMax()]))
    assert len(rails) == 6
    near_rail_x = min(bounds[0, 0] for bounds in rails)
    supply = [np.array(_reset_bounds(obj)) for obj in config['objects'].values()]
    table = np.array(config['supports']['table']['bounds_m'])
    for bounds in supply:
        assert near_rail_x - bounds[1, 0] >= .05
        assert bounds[0, 0] - table[0, 0] >= .05
        assert np.all(bounds[0, :2] >= table[0, :2])
        assert np.all(bounds[1, :2] <= table[1, :2])
        assert bounds[0, 2] == pytest.approx(table[1, 2])
        assert all(_separated(bounds, rail) for rail in rails)
    assert all(_separated(a, b) for a, b in combinations(supply, 2))

    for ident, target in config['targets'].items():
        bounds = np.array(target['bounds_m'])
        # Keep the goal center near the robot instead of the far desktop edge.
        assert .45 <= bounds[:, 0].mean() <= .55
        assert all(_separated(bounds, rail) for rail in rails)
        assert all(_separated(bounds, source) for source in supply)
        label = stage.GetPrimAtPath(task_scene.TASK_ROOT + '/Fixtures/' + ident + '_label')
        printed = np.array(UsdGeom.Mesh(label).GetPointsAttr().Get())
        assert np.all(printed.min(axis=0) >= bounds[0] - 1.e-7)
        assert np.all(printed.max(axis=0) <= bounds[1] + 1.e-7)
        # Check all actual render/collision geometry, not just the nominal center.
        source = np.array(_reset_bounds(config['objects'][ident]))
        body = stage.GetPrimAtPath(task_scene.OBJECT_ROOT + '/' + ident)
        actual = cache.ComputeWorldBound(body).ComputeAlignedRange()
        actual = np.array([actual.GetMin(), actual.GetMax()])
        assert np.all(actual[0] >= source[0] - 2.e-7)
        assert np.all(actual[1] <= source[1] + 2.e-7)
        # Retain the source yaw and table height when moving to the slot center.
        displacement = np.r_[bounds[:, :2].mean(axis=0) - source[:, :2].mean(axis=0), 0.]
        for placed in (source + displacement, actual + displacement):
            assert np.all(placed[0] >= bounds[0] - 2.e-7)
            assert np.all(placed[1] <= bounds[1] + 2.e-7)
            assert all(_separated(placed, rail) for rail in rails)


def test_pencil_uses_visible_hexagonal_collision_and_surface_print(stationery_scene):
    from pxr import Usd, UsdGeom, UsdPhysics

    stage, config, assets = stationery_scene
    root = stage.GetPrimAtPath(task_scene.OBJECT_ROOT + '/pencil')
    colliders = [prim for prim in Usd.PrimRange(root)
                 if prim.HasAPI(UsdPhysics.CollisionAPI)]
    assert {prim.GetName() for prim in colliders} == {'LacquerShaft', 'CedarTip'}
    for prim in colliders:
        assert prim.IsA(UsdGeom.Mesh)
        assert UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr().Get() == 'convexHull'
        assert UsdGeom.Imageable(prim).ComputeVisibility() != UsdGeom.Tokens.invisible
        assert UsdGeom.Imageable(prim).ComputePurpose() in ('default', 'render')

    shaft = UsdGeom.Mesh(stage.GetPrimAtPath(str(root.GetPath()) + '/Geometry/LacquerShaft'))
    points = np.array(shaft.GetPointsAttr().Get())
    assert points.shape == (12, 3)
    for x in np.unique(points[:, 0]):
        ring = points[points[:, 0] == x, 1:]
        assert ring.shape == (6, 2)
        # Six equal radial vertices: a real 15 mm hexagon, not a bounding-box proxy.
        np.testing.assert_allclose(np.linalg.norm(ring, axis=1) * 2., .015, atol=1.e-9)
        angles = np.sort(np.arctan2(ring[:, 1], ring[:, 0]))
        np.testing.assert_allclose(np.diff(np.r_[angles, angles[0] + 2 * np.pi]),
                                   np.pi / 3, atol=1.e-7)
    physical_points = np.concatenate([np.array(UsdGeom.Mesh(p).GetPointsAttr().Get())
                                      for p in colliders])
    assert np.ptp(physical_points[:, 0]) == pytest.approx(.180)
    declared = np.array(config['objects']['pencil']['local_bounds_m'])
    np.testing.assert_allclose(physical_points.min(axis=0), declared[0], atol=1.e-9)
    np.testing.assert_allclose(physical_points.max(axis=0)[:2], declared[1, :2], atol=1.e-9)

    printed = stage.GetPrimAtPath(str(root.GetPath()) + '/Geometry/Print')
    assert not printed.HasAPI(UsdPhysics.CollisionAPI)
    ink = np.array(UsdGeom.Mesh(printed).GetPointsAttr().Get())
    top_face = points[np.isclose(points[:, 2], points[:, 2].max(), atol=1.e-9)]
    assert len(top_face) == 4
    assert np.all(ink[:, :2] >= top_face[:, :2].min(axis=0))
    assert np.all(ink[:, :2] <= top_face[:, :2].max(axis=0))
    np.testing.assert_allclose(ink[:, 2] - top_face[0, 2], .000002, atol=1.e-9)
    assert np.all(ink.min(axis=0) >= declared[0] - 1.e-9)
    assert np.all(ink.max(axis=0) <= declared[1] + 1.e-9)

    # USD mass and its declared envelope approximation must use the enlarged body.
    mass_api = UsdPhysics.MassAPI(root)
    mass = mass_api.GetMassAttr().Get()
    assert mass == pytest.approx(assets['pencil']['mass_kg'])
    assert mass == pytest.approx(.0081 * (.015 / .011) ** 2)
    dimensions = declared[1] - declared[0]
    expected_inertia = mass / 12. * (np.sum(dimensions ** 2) - dimensions ** 2)
    np.testing.assert_allclose(mass_api.GetDiagonalInertiaAttr().Get(), expected_inertia,
                               rtol=2.e-7, atol=1.e-12)
    np.testing.assert_allclose(assets['pencil']['diagonal_inertia_kg_m2'], expected_inertia,
                               rtol=2.e-7, atol=1.e-12)


def test_pencil_contact_material_override_does_not_leak(stationery_scene):
    from pxr import Usd, UsdPhysics, UsdShade

    stage, _, assets = stationery_scene
    material_paths = {}
    roots = {ident: task_scene.OBJECT_ROOT + '/' + ident for ident in assets}
    roots['fixtures'] = task_scene.TASK_ROOT + '/Fixtures'
    for ident, path in roots.items():
        expected = (.70, .55) if ident == 'pencil' else (.50, .40)
        colliders = [p for p in Usd.PrimRange(stage.GetPrimAtPath(path))
                     if p.HasAPI(UsdPhysics.CollisionAPI)]
        assert colliders
        material_paths[ident] = set()
        for collider in colliders:
            material, _ = UsdShade.MaterialBindingAPI(collider).ComputeBoundMaterial(
                materialPurpose='physics')
            assert material
            material_paths[ident].add(str(material.GetPath()))
            api = UsdPhysics.MaterialAPI(material.GetPrim())
            assert (api.GetStaticFrictionAttr().Get(), api.GetDynamicFrictionAttr().Get()) == pytest.approx(expected)
            assert api.GetRestitutionAttr().Get() == pytest.approx(.02)
            assert collider.GetAttribute('physxCollision:contactOffset').Get() == pytest.approx(.001)
            assert collider.GetAttribute('physxCollision:restOffset').Get() == 0.
        if ident in assets:
            physics = assets[ident]['physics']
            assert (physics['static_friction'], physics['dynamic_friction']) == pytest.approx(expected)
    assert len(material_paths['pencil']) == 1
    assert all(material_paths['pencil'].isdisjoint(material_paths[ident])
               for ident in ('marker', 'eraser', 'fixtures'))
