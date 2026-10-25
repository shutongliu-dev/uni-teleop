"""CPU contracts for a coarse, passive weighing scene; not native acceptance."""
import json

import numpy as np
import pytest
from pxr import Usd, UsdGeom, UsdPhysics

from scenes.workrooms.laboratory.tasks.solid_weighing.assets import author_task


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
