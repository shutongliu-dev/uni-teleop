"""Appearance must not alter the apparatus' physical or scoring contract."""
import copy
import importlib
import math

import numpy as np
import pytest
from pxr import Usd, UsdGeom, UsdPhysics

from scenes.workrooms.laboratory.visual_details import add_task_visual_details

TASKS = ("sample_sorting", "vial_racking", "cap_placement", "solid_weighing", "pipette_transfer")


def physical_signature(stage):
    result = {}
    for prim in stage.Traverse():
        if not any(prim.HasAPI(api) for api in (
                UsdPhysics.CollisionAPI, UsdPhysics.RigidBodyAPI, UsdPhysics.MaterialAPI)) and not prim.IsA(UsdPhysics.Joint):
            continue
        # Default shading/visibility can change. The separate physics-purpose
        # material relationship and all geometry/dynamic properties must match.
        attrs = {a.GetName(): repr(a.Get()) for a in prim.GetAttributes()
                 if a.GetName() not in ("visibility", "primvars:displayColor", "primvars:displayOpacity")}
        rels = {r.GetName(): [str(p) for p in r.GetTargets()] for r in prim.GetRelationships()
                if r.GetName() != "material:binding"}
        result[str(prim.GetPath())] = (str(prim.GetMetadata("apiSchemas")), attrs, rels)
    return result


@pytest.mark.parametrize("task_id", TASKS)
@pytest.mark.parametrize("presentation", (False, True))
def test_visual_details_preserve_physical_stage_and_existing_metadata(task_id, presentation, monkeypatch):
    author = importlib.import_module(f"scenes.workrooms.laboratory.tasks.{task_id}.assets")
    # Capture the original author output immediately before the cosmetic layer.
    monkeypatch.setattr(author, "add_task_visual_details", lambda stage, metadata: metadata)
    stage = Usd.Stage.CreateInMemory()
    metadata = author.author_task(stage, presentation=presentation)
    before_metadata = copy.deepcopy(metadata)
    before_physics = physical_signature(stage)
    enriched = add_task_visual_details(stage, metadata)
    assert physical_signature(stage) == before_physics
    assert {k: v for k, v in enriched.items() if k != "visual_detail"} == before_metadata
    details = enriched["visual_detail"]
    assert details["visual_prim_paths"] and details["labels"]
    for path in details["visual_prim_paths"]:
        prim = stage.GetPrimAtPath(path)
        assert prim and not prim.HasAPI(UsdPhysics.CollisionAPI)
        assert not prim.HasAPI(UsdPhysics.RigidBodyAPI)
        assert not prim.HasAPI(UsdPhysics.MassAPI)
        assert np.all(np.isfinite(UsdGeom.Mesh(prim).GetPointsAttr().Get()))


@pytest.mark.parametrize("task_id", TASKS)
def test_manipulable_apparatus_labels_use_the_authored_mass(task_id):
    module = importlib.import_module(f"scenes.workrooms.laboratory.tasks.{task_id}.assets")
    stage = Usd.Stage.CreateInMemory()
    metadata = module.author_task(stage)
    labels = metadata["visual_detail"]["labels"]
    for row in metadata["objects"]:
        if row["id"].startswith("pellet_"):
            continue
        expected = format(row["mass_kg"]*1000, "g")+" g"
        texts = [label["text"] for label in labels if label["body_id"] == row["id"]]
        assert any(expected in text for text in texts), (row["id"], expected, texts)


def test_vessel_graduations_match_actual_container_cross_section_and_keep_apertures_open():
    from scenes.workrooms.laboratory.tasks.pipette_transfer.assets import author_task
    stage = Usd.Stage.CreateInMemory()
    metadata = author_task(stage)
    containers = metadata["task_parameters"]["hydraulics"]["containers"]
    for name, container in containers.items():
        root = metadata["actors"][name]+"/VisualDetails"
        ticks = [prim for prim in stage.GetPrimAtPath(root).GetChildren()
                 if prim.GetName().startswith("VolumeTick_")]
        assert ticks
        for prim in ticks:
            ml = float(prim.GetName().split("_")[-1])
            points = np.asarray(UsdGeom.Mesh(prim).GetPointsAttr().Get())
            height = .5*(points[:, 2].min()+points[:, 2].max())
            assert (height-container["floor_z_m"])*math.pi*container["radius_m"]**2*1.e6 == pytest.approx(ml, abs=1.e-4)
        rim = np.asarray(UsdGeom.Mesh(stage.GetPrimAtPath(root+"/RimHighlight")).GetPointsAttr().Get())
        assert np.linalg.norm(rim[:, :2], axis=1).min() > container["radius_m"]
