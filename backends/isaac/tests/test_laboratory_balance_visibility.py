"""Readability/fixture geometry guards; native occlusion remains a separate check."""
import numpy as np
import pytest
from pxr import Gf, Usd, UsdGeom, UsdShade

from scenes.workrooms.laboratory.tasks.solid_weighing.assets import author_task
from scenes.workrooms.laboratory.weight_display import balance_platform_parts, weighing_camera_overrides
from scripts.validation.laboratory_balance_display import display_surface_points, primary_display


def test_housing_keeps_original_pan_and_open_vertical_load_corridor():
    parts = dict(balance_platform_parts())
    assert parts['base'].bounds == pytest.approx(np.array([[-.0675, -.0625, 0.], [.0675, .0625, .022]]))
    assert parts['pan'].bounds == pytest.approx(np.array([[-.050, -.045, .022], [.050, .045, .025]]))
    for name, mesh in parts.items():
        assert mesh.is_watertight and mesh.is_convex
        if name in ('base', 'pan'):
            continue
        lo, hi = mesh.bounds
        overlaps_pan_xy = np.all(hi[:2] > [-.05, -.045]) and np.all(lo[:2] < [.05, .045])
        assert not overlaps_pan_xy or hi[2] <= .022
        assert lo[2] >= -1.e-10
    bounds = np.array([mesh.bounds for mesh in parts.values()])
    assert bounds[:, 0, 0].min() >= -.115 and bounds[:, 1, 0].max() <= .115
    assert bounds[:, 0, 1].min()+2.075 >= 1.99


def test_static_display_is_unknown_and_primary_surface_tracks_composed_transform():
    stage = Usd.Stage.CreateInMemory()
    meta = author_task(stage)
    panel = primary_display({'asset_metadata': meta})
    assert panel['digit_height_m'] >= .024
    for row in meta['visual_state_bindings']['bindings']:
        prim = stage.GetPrimAtPath(row['target_prim_path'])
        material = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()[0]
        expected = 'on' if row['id'].endswith('_g') else 'off'
        assert material.GetPath().name == expected
    before = np.array(display_surface_points(stage, panel['screen_prim_path']))
    fixture = UsdGeom.Xformable(stage.GetPrimAtPath(meta['fixtures']['balance_platform']))
    translate = next(op for op in fixture.GetOrderedXformOps() if op.GetOpType() == UsdGeom.XformOp.TypeTranslate)
    translate.Set(translate.Get()+Gf.Vec3d(.20, -.10, .05))
    after = np.array(display_surface_points(stage, panel['screen_prim_path']))
    assert after-before == pytest.approx(np.tile([.20, -.10, .05], (4, 1)))


def test_published_weighing_view_has_resolvable_digits_before_native_occlusion_check():
    stage = Usd.Stage.CreateInMemory()
    meta = author_task(stage)
    panel = primary_display({'asset_metadata': meta})
    camera = weighing_camera_overrides()
    eye = np.array(camera['eye']); forward = np.array(camera['target'])-eye
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, [0., 0., 1.]); right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    def project(point):
        delta = np.asarray(point)-eye
        depth = delta @ forward
        assert depth > 0
        return np.array([480., 360.])+camera['focal_length_mm']/36.*960/depth*np.array([delta@right, -delta@up])
    corners = np.array([project(p) for p in display_surface_points(stage, panel['screen_prim_path'])])
    assert np.all(corners >= [0, 0]) and np.all(corners < [960, 720])
    assert np.ptp(corners[:, 0]) >= 100.
    cache = UsdGeom.XformCache()
    for digit in range(3):
        endpoints = [project(cache.GetLocalToWorldTransform(stage.GetPrimAtPath(
            panel['panel_prim_path']+f'/digit_{digit}_{segment}')).ExtractTranslation()) for segment in ('a', 'd')]
        assert np.linalg.norm(endpoints[1]-endpoints[0]) >= 16.
