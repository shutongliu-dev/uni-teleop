"""Static background supplies must not change task physics or obstruct resets."""
from copy import deepcopy
import importlib

import numpy as np
import pytest
from pxr import Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

from scenes.workrooms.laboratory.desktop_dressing import author_desktop_dressing

ROOT = "/World/LaboratoryTask/DesktopDressing"
TASKS = ("sample_sorting", "vial_racking", "cap_placement", "solid_weighing", "pipette_transfer")


@pytest.fixture(scope="module")
def desktop():
    stage = Usd.Stage.CreateInMemory()
    return stage, author_desktop_dressing(stage)


def _overlap(a, b, tolerance=1e-6):
    a, b = np.asarray(a), np.asarray(b)
    return bool(np.all(np.minimum(a[1], b[1])-np.maximum(a[0], b[0]) > tolerance))


def _bounds(stage, path):
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render"], useExtentsHint=False)
    bound = cache.ComputeWorldBound(stage.GetPrimAtPath(path)).ComputeAlignedRange()
    return [list(bound.GetMin()), list(bound.GetMax())]


def test_static_supply_inventory_has_real_collision_but_no_task_or_robot_state(desktop):
    stage, meta = desktop
    assert meta["fixed_group_count"] >= 15
    assert meta["visible_asset_count"] >= 40
    recorded = {p for item in meta["items"] for p in item["collider_paths"]}
    actual = {str(p.GetPath()) for p in stage.Traverse() if p.HasAPI(UsdPhysics.CollisionAPI)}
    assert recorded == actual
    assert len(recorded) == meta["collider_count"]
    assert all(path.startswith(ROOT+"/") for path in recorded)
    for path in recorded:
        prim = stage.GetPrimAtPath(path)
        material, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial(materialPurpose="physics")
        assert str(material.GetPath()) == ROOT+"/Materials/DryContact"
        assert prim.GetAttribute("physxCollision:contactOffset").Get() == pytest.approx(.0002)
        assert prim.GetAttribute("physxCollision:restOffset").Get() == 0.
    material = UsdPhysics.MaterialAPI(stage.GetPrimAtPath(ROOT+"/Materials/DryContact"))
    assert material.GetStaticFrictionAttr().Get() == pytest.approx(.30)
    assert material.GetDynamicFrictionAttr().Get() == pytest.approx(.25)
    assert material.GetRestitutionAttr().Get() == pytest.approx(.01)
    for prim in stage.Traverse():
        assert not prim.HasAPI(UsdPhysics.RigidBodyAPI)
        assert not prim.HasAPI(UsdPhysics.MassAPI)
        assert not prim.HasAPI(UsdPhysics.ArticulationRootAPI)
        assert not prim.IsA(UsdPhysics.Scene)
        assert not prim.IsA(UsdPhysics.Joint)
    for item in meta["items"]:
        assert item["collider_paths"]
        assert item["dynamic"] is item["task_object"] is item["scoring_support"] is False
        assert item["world_bounds_m"][0][2] >= .9-1e-6


def test_groups_stay_separate_with_clear_working_and_robot_areas(desktop):
    _, meta = desktop
    for index, item in enumerate(meta["items"]):
        low, high = np.asarray(item["world_bounds_m"])
        zone_low, zone_high = np.asarray(meta["zones_world_bounds_m"][item["zone"]])
        assert np.all(low >= zone_low-1e-6)
        assert np.all(high <= zone_high+1e-6)
        for bounds in meta["keep_clear_world_bounds_m"].values():
            assert not _overlap(item["world_bounds_m"], bounds)
        for other in meta["items"][index+1:]:
            assert not _overlap(item["world_bounds_m"], other["world_bounds_m"]), (item["id"], other["id"])


@pytest.mark.parametrize("task_id", TASKS)
@pytest.mark.parametrize("presentation", [False, True])
def test_decoration_does_not_mutate_task_or_intersect_its_objects(desktop, task_id, presentation):
    dressing_stage, dressing = desktop
    stage = Usd.Stage.CreateInMemory()
    author = importlib.import_module("scenes.workrooms.laboratory.tasks."+task_id+".assets").author_task
    metadata = author(stage, presentation=presentation)
    baseline = stage.GetRootLayer().ExportToString()
    original_metadata = deepcopy(metadata)
    assert Sdf.CopySpec(dressing_stage.GetRootLayer(), ROOT, stage.GetRootLayer(), ROOT)
    assert metadata == original_metadata
    for path in (*metadata["actors"].values(), *metadata["fixtures"].values()):
        bounds = _bounds(stage, path)
        for item in dressing["items"]:
            assert not _overlap(bounds, item["world_bounds_m"]), (task_id, path, item["id"])
    stage.RemovePrim(ROOT)
    # Includes original collision, mass/inertia, materials and reset transforms.
    assert stage.GetRootLayer().ExportToString() == baseline


def test_hollow_label_core_has_no_invisible_solid_plug(desktop):
    stage, meta = desktop
    item = next(row for row in meta["items"] if row["id"] == "label_roll")
    segments = [path for path in item["collider_paths"] if "roll_segment_" in path]
    assert len(segments) >= 24
    for path in segments:
        mesh = UsdGeom.Mesh(stage.GetPrimAtPath(path))
        vertices = np.asarray(mesh.GetPointsAttr().Get())
        assert np.min(np.linalg.norm(vertices[:, :2], axis=1)) >= .0199
        assert UsdPhysics.MeshCollisionAPI(mesh).GetApproximationAttr().Get() == "convexHull"


def test_image_only_diagnostics_explicitly_disable_all_collision():
    stage = Usd.Stage.CreateInMemory()
    meta = author_desktop_dressing(stage, collision=False)
    assert meta["collision_enabled"] is False
    assert meta["collider_count"] == 0
    assert not any(prim.HasAPI(UsdPhysics.CollisionAPI) for prim in stage.Traverse())
    assert all(not item["collider_paths"] for item in meta["items"])


def test_cannot_author_background_outside_task_or_overwrite_existing(desktop):
    stage, _ = desktop
    with pytest.raises(ValueError, match="fresh task version"):
        author_desktop_dressing(stage)
    with pytest.raises(ValueError, match="task subtree"):
        author_desktop_dressing(Usd.Stage.CreateInMemory(), root_path="/World/Geometry/Extras")
