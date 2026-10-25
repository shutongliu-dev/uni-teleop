"""Stationery layout geometry checks; these do not qualify robot manipulation."""
from itertools import combinations

import numpy as np
import pytest

from scenes.workrooms.study import task_scene
from scenes.workrooms.study.rules import _reset_bounds


def _separated(first, second):
    return np.any(first[1] <= second[0]) or np.any(second[1] <= first[0])


def test_stationery_supply_has_a_clear_lane_beside_the_nearby_organizer(tmp_path):
    from pxr import Usd, UsdGeom, UsdPhysics

    config = task_scene.configuration('stationery_sorting')
    stage = Usd.Stage.CreateInMemory()
    builder = task_scene._builder(stage)
    task_scene._fixtures(builder, 'stationery_sorting', config, tmp_path,
                         task_scene.DEFAULT_ASSETS_ROOT)
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ['default', 'render'])
    rails = []
    for prim in stage.Traverse():
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
        # Each unchanged object still fits entirely in its compartment.
        size = np.diff(np.array(config['objects'][ident]['local_bounds_m']), axis=0)[0]
        assert np.all(size <= bounds[1] - bounds[0])
