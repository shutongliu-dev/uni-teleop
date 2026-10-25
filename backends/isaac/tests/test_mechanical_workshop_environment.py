"""The new workshop must preserve empty task space and immutable ownership."""
from pathlib import Path

import numpy as np
import pytest

from bench.paths import DATA_ROOT

pytest.importorskip('pxr.Usd')
from pxr import Usd, UsdGeom

from scripts.assets.build_mechanical_workshop import audit_environment
from bench.simulation.assets.workrooms.mechanical_workshop import build_mechanical_workshop
from bench.simulation.assets.workrooms.common import apply_lighting_profile

ASSETS = Path(f'{DATA_ROOT}/assets')


@pytest.fixture(scope='module')
def room(tmp_path_factory):
    stage = Usd.Stage.CreateInMemory()
    scene = build_mechanical_workshop(stage, ASSETS, tmp_path_factory.mktemp('mechanical_workshop'))
    return stage, scene


def test_empty_work_surface_and_robot_floor_space(room):
    stage, scene = room
    report = audit_environment(stage)
    assert report['status'] == 'passed'
    assert scene['objects'] == [] and scene['task_independent']
    assert not stage.GetPrimAtPath('/World/Geometry/robot_column')
    assert not stage.GetPrimAtPath('/World/Geometry/robot_anchor_plate')


def test_intruding_decoration_is_rejected(room):
    stage, _ = room
    layer = stage.GetSessionLayer()
    with Usd.EditContext(stage, layer):
        obstruction = UsdGeom.Cube.Define(stage, '/World/Geometry/test_blocking_fixture')
        obstruction.CreateSizeAttr(.1)
        UsdGeom.Xformable(obstruction).AddTranslateOp().Set((.40, 0., .9))
        try:
            with pytest.raises(ValueError, match='intrudes'):
                audit_environment(stage)
        finally:
            stage.RemovePrim('/World/Geometry/test_blocking_fixture')


def test_furniture_groups_remain_outside_workcell_and_inside_walls(room):
    stage, scene = room
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ['default', 'render'])
    for group in scene['furnishings']['groups']:
        boxes = [cache.ComputeWorldBound(stage.GetPrimAtPath(path)).ComputeAlignedRange()
                 for path in group['geometry_paths']]
        low = np.min([extent.GetMin() for extent in boxes], axis=0)
        high = np.max([extent.GetMax() for extent in boxes], axis=0)
        np.testing.assert_allclose(group['bounds_m'], [low, high], atol=1.e-7)
        assert (low >= [-1.45, -1.60, -1.e-6]).all(), group['name']
        # Wall fixture anchor studs may enter the 140 mm rear wall; floor
        # furniture and supported decor must stay within the clear room.
        rear_limit = 2.19 if group['role'] == 'wall_fixture' else 2.05
        assert (high <= [2.65, rear_limit, 2.80]).all(), group['name']


def test_lighting_restores_published_baseline(room):
    stage, scene = room
    apply_lighting_profile(stage, 'mechanical_assembly', 'low_light')
    restored = apply_lighting_profile(stage, 'mechanical_assembly', 'nominal')
    assert restored['lights'] == scene['lighting']['lights']
