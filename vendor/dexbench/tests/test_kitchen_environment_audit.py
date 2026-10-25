"""A disabled kitchen support must fail the CPU structural publication audit."""

import pytest

from bench.paths import load_json

pytest.importorskip('pxr.Usd')
from pxr import Gf, Usd, UsdGeom, UsdPhysics

from scenes.workrooms.kitchen.scene import ENVIRONMENT_ROOT
from scripts.assets.build_card_room import digest
from scripts.assets.build_kitchen import audit_environment


def test_disabled_support_is_rejected_without_modifying_published_asset():
    source = ENVIRONMENT_ROOT / 'scene.usda'
    if not source.is_file():
        pytest.skip('Pinned kitchen environment is unavailable')
    before = digest(source)
    description = load_json(ENVIRONMENT_ROOT / 'scene_description.json')
    stage = Usd.Stage.Open(str(source))
    stage.SetEditTarget(stage.GetSessionLayer())
    support = UsdPhysics.CollisionAPI(stage.GetPrimAtPath('/World/Geometry/worktop/Collider'))
    assert support.GetCollisionEnabledAttr().Get() is True
    support.GetCollisionEnabledAttr().Set(False)
    with pytest.raises(ValueError, match='Missing or disabled declared support collider'):
        audit_environment(stage, description)
    stage.GetSessionLayer().Clear()
    assert support.GetCollisionEnabledAttr().Get() is True
    assert digest(source) == before


def test_tabletop_clutter_in_reserved_task_area_is_rejected():
    source = ENVIRONMENT_ROOT / 'scene.usda'
    if not source.is_file():
        pytest.skip('Pinned kitchen environment is unavailable')
    before = digest(source)
    description = load_json(ENVIRONMENT_ROOT / 'scene_description.json')
    stage = Usd.Stage.Open(str(source))
    stage.SetEditTarget(stage.GetSessionLayer())
    cube = UsdGeom.Cube.Define(stage, '/World/Geometry/unwanted_task_clutter')
    cube.CreateSizeAttr(.12)
    UsdGeom.Xformable(cube).AddTranslateOp().Set(Gf.Vec3d(.55, -.90, .96))
    with pytest.raises(ValueError, match='Reserved bar task rectangle is obstructed'):
        audit_environment(stage, description)
    stage.GetSessionLayer().Clear()
    assert digest(source) == before
