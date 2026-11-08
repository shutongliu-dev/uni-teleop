"""CPU contracts for a coarse, passive weighing scene; not native acceptance."""
import json

import numpy as np
import pytest
from pxr import Gf, Usd, UsdGeom, UsdPhysics, UsdShade

from scenes.workrooms.laboratory.tasks.solid_weighing.assets import author_task


def _world_points(stage, path):
    prim = stage.GetPrimAtPath(path)
    transform = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    return np.array([transform.Transform(Gf.Vec3d(*point))
                     for point in UsdGeom.Mesh(prim).GetPointsAttr().Get()])


def _collision_bounds(stage, record):
    points = np.concatenate([_world_points(stage, path) for path in record['collider_paths']])
    return np.array([points.min(0), points.max(0)])


def test_inventory_is_passive_mass_conserving_and_has_no_instrument_actuator():
    stage = Usd.Stage.CreateInMemory()
    meta = author_task(stage)
    assert len(meta["actors"]) == 26
    assert set(meta["fixtures"]) == {"source_tray", "balance_platform", "scoop_rest"}
    assert meta["joints"] == meta["particle_sets"] == {}
    assert sum(o["mass_kg"] for o in meta["objects"] if o["role"] == "model_granule") == pytest.approx(.012)
    assert len(json.dumps(meta, allow_nan=False)) > 1000
    for item in meta["objects"]:
        prim = stage.GetPrimAtPath(item["body_path"])
        body = UsdPhysics.RigidBodyAPI(prim)
        assert body.GetRigidBodyEnabledAttr().Get()
        assert not body.GetKinematicEnabledAttr().Get()
        mass = UsdPhysics.MassAPI(prim)
        assert mass.GetMassAttr().Get() == pytest.approx(item["mass_kg"])
        assert np.all(np.asarray(mass.GetDiagonalInertiaAttr().Get()) > 0)
        assert len(item["initial_pose_xyzw"]) == 7
    assert not any(p.IsA(UsdPhysics.Scene) or p.IsA(UsdPhysics.Joint) for p in stage.Traverse())
    assert meta["physics_accepted"] is False


def test_granules_are_analytic_spheres_and_reset_has_positive_pair_clearance():
    stage = Usd.Stage.CreateInMemory()
    meta = author_task(stage)
    grains = [item for item in meta["objects"] if item["role"] == "model_granule"]
    points = np.asarray([item["initial_position_m"] for item in grains])
    distances = np.linalg.norm(points[:, None]-points[None, :], axis=-1)
    distances += np.eye(24)
    assert distances.min() >= .0099
    for item in grains:
        sphere = UsdGeom.Sphere(stage.GetPrimAtPath(item["collider_paths"][0]))
        assert sphere.GetRadiusAttr().Get() == pytest.approx(.004)
        assert item["initial_position_m"][2]-.004 > .903


def test_open_cup_and_scoop_have_no_hull_covering_their_cavities():
    stage = Usd.Stage.CreateInMemory()
    meta = author_task(stage)
    objects = {item["id"]: item for item in meta["objects"]}
    for name, point in (("weighing_cup", [0., 0., .018]), ("scoop", [0., 0., .010])):
        for path in objects[name]["collider_paths"]:
            points = np.asarray(UsdGeom.Mesh(stage.GetPrimAtPath(path)).GetPointsAttr().Get())
            # Each convex contact part stays outside an interior test point.
            assert not np.all((point >= points.min(0)) & (point <= points.max(0)))
    scoop = objects["scoop"]
    assert scoop["mouth_axis_local"] == [1., 0., 0.]
    assert scoop["interior_bounds_m"][1][1]-scoop["interior_bounds_m"][0][1] > .028
    params = meta["task_parameters"]
    assert params["tolerance_mass_kg"]/params["target_mass_kg"] == pytest.approx(.1)
    assert params["target_mass_kg"]/params["pellet_mass_kg"] == pytest.approx(12.)


def test_presentation_is_separate_reset_and_keeps_all_particles_dynamic():
    stage = Usd.Stage.CreateInMemory()
    meta = author_task(stage, presentation=True)
    objects = {item["id"]: item for item in meta["objects"]}
    assert meta["presentation"] is True
    assert objects["weighing_cup"]["initial_position_m"] == [.71, 2.075, .9255]
    assert sum(objects[f"pellet_{i:02}"]["mass_kg"] for i in range(12)) == pytest.approx(.006)
    assert all(UsdPhysics.RigidBodyAPI(stage.GetPrimAtPath(path)).GetRigidBodyEnabledAttr().Get()
               for path in meta["actors"].values())


def test_probe_counts_whole_granules_not_centres_at_the_cup_rim():
    from scripts.validation.laboratory_solid_interactions import classify
    stage = Usd.Stage.CreateInMemory()
    meta = author_task(stage, presentation=True)
    names = list(meta["actors"])
    objects = {item["id"]: item for item in meta["objects"]}
    poses = np.array([objects[name]["initial_pose_xyzw"] for name in names])
    state = classify(poses, names, meta)
    assert len(state["cup"]) == 12
    # A center just inside the wall still represents a partly protruding grain.
    poses[names.index("pellet_00"), :3] = [.743, 2.075, .94]
    state = classify(poses, names, meta)
    assert "pellet_00" not in state["cup"]


def test_scoop_handle_overhangs_real_near_edge_with_separate_cup_lane():
    from scenes.workrooms.laboratory.robot_layout import workspace_contract

    stage = Usd.Stage.CreateInMemory()
    meta = author_task(stage)
    objects = {item['id']: item for item in meta['objects']}
    scoop, cup = (_collision_bounds(stage, objects[name]) for name in ('scoop', 'weighing_cup'))
    source = _collision_bounds(stage, meta['fixture_metadata']['source_tray'])
    edge = workspace_contract()['surfaces']['island']['near_edge_y_m']
    handle = _world_points(stage, objects['scoop']['path']+'/Collision/handle')
    assert handle[:, 1].max()-edge == pytest.approx(.05, abs=1e-7)
    assert handle[:, 1].max() == pytest.approx(scoop[1, 1], abs=1e-7)
    assert scoop[0, 0]-cup[1, 0] >= .13
    assert cup[0, 0]-source[1, 0] >= .06
    assert objects['scoop']['initial_pose_xyzw'][5] < 0  # -90 degrees around Z
    # The source lane and original receptacle geometry/quantities stay fixed.
    assert meta['fixture_metadata']['source_tray']['initial_position_m'] == [.61, .51, .90]
    assert objects['weighing_cup']['interior_bounds_m'] == [[-.0345, -.0295, .003], [.0345, .0295, .035]]
    assert objects['scoop']['interior_bounds_m'] == [[-.027, -.019, .002], [.025, .019, .019]]


def test_two_visible_support_levels_bracket_the_free_scoop_center_of_mass():
    from itertools import product
    from scipy.spatial import ConvexHull
    from scenes.workrooms.laboratory.robot_layout import workspace_contract

    stage = Usd.Stage.CreateInMemory()
    meta = author_task(stage)
    scoop = next(item for item in meta['objects'] if item['id'] == 'scoop')
    rest = meta['fixture_metadata']['scoop_rest']
    table = np.array(workspace_contract()['surfaces']['island']['surface_bounds_xy_m'])
    rest_bounds = _collision_bounds(stage, rest)
    assert np.all(rest_bounds[0, :2] >= table[0])
    assert np.all(rest_bounds[1, :2] <= table[1]-.0099)
    assert rest_bounds[0, 2] == pytest.approx(.90, abs=1e-7)
    assert not stage.GetPrimAtPath(rest['path']).HasAPI(UsdPhysics.RigidBodyAPI)
    patches = []
    for tool_name, support_name, top_z in [('bowl_floor', 'pad', .906),
                                          ('handle', 'handle_bridge', .913)]:
        tool = _world_points(stage, scoop['path']+'/Collision/'+tool_name)
        support_path = rest['path']+'/Collision/'+support_name
        support = _world_points(stage, support_path)
        assert UsdGeom.Imageable(stage.GetPrimAtPath(support_path)).ComputeVisibility() != 'invisible'
        assert support[:, 2].max() == pytest.approx(top_z, abs=1e-7)
        assert tool[:, 2].min()-support[:, 2].max() == pytest.approx(.0005, abs=1e-7)
        lo = np.maximum(tool[:, :2].min(0), support[:, :2].min(0))
        hi = np.minimum(tool[:, :2].max(0), support[:, :2].max(0))
        assert np.all(hi-lo > .01)
        patches.extend(product(*zip(lo, hi)))
    prim = stage.GetPrimAtPath(scoop['path'])
    com_local = UsdPhysics.MassAPI(prim).GetCenterOfMassAttr().Get()
    com = np.array(UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(
        Usd.TimeCode.Default()).Transform(Gf.Vec3d(*com_local)))
    hull = ConvexHull(patches)
    signed_distances = hull.equations[:, :2] @ com[:2]+hull.equations[:, 2]
    assert signed_distances.max() < -.008
    # A single bowl-floor patch is insufficient for this existing tool COM.
    bowl = _world_points(stage, scoop['path']+'/Collision/bowl_floor')
    assert com[1] > bowl[:, 1].max()+.007
    assert not any(p.IsA(UsdPhysics.Joint) for p in stage.Traverse())
    assert not UsdPhysics.RigidBodyAPI(prim).GetKinematicEnabledAttr().Get()


def test_rotated_whole_tool_fits_return_zone_with_replacement_margin():
    from itertools import product

    stage = Usd.Stage.CreateInMemory()
    meta = author_task(stage)
    scoop = next(item for item in meta['objects'] if item['id'] == 'scoop')
    actual = _collision_bounds(stage, scoop)
    target = np.array(meta['task_parameters']['scoop_return_bounds_m'])
    # Includes the deliberately overhanging handle; the target is not the tray
    # footprint. Real tray contact is still a separate referee requirement.
    assert np.all(actual[0, :2]-target[0, :2] >= .0149)
    assert np.all(target[1, :2]-actual[1, :2] >= .0149)
    for x, y in product((-.01, 0., .01), repeat=2):
        translated = actual+np.array([x, y, 0.])
        assert np.all(translated[0] >= target[0])
        assert np.all(translated[1] <= target[1])


def test_layout_change_preserves_apparatus_mass_and_dry_contact_prior():
    stage = Usd.Stage.CreateInMemory()
    meta = author_task(stage)
    objects = {item['id']: item for item in meta['objects']}
    assert objects['scoop']['mass_kg'] == .026
    assert objects['weighing_cup']['mass_kg'] == .028
    for name in ('scoop', 'weighing_cup'):
        for path in objects[name]['collider_paths']:
            prim = stage.GetPrimAtPath(path)
            material, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial('physics')
            api = UsdPhysics.MaterialAPI(material.GetPrim())
            assert [api.GetStaticFrictionAttr().Get(), api.GetDynamicFrictionAttr().Get(),
                    api.GetRestitutionAttr().Get()] == pytest.approx([.35, .30, .01])
            assert prim.GetAttribute('physxCollision:contactOffset').Get() == pytest.approx(.0002)
            assert prim.GetAttribute('physxCollision:restOffset').Get() == 0.
