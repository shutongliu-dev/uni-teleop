"""Spoon support geometry, independent of controller or native success gates."""
import math

import numpy as np
import pytest
from pxr import Usd, UsdGeom
from scipy.spatial import ConvexHull
import trimesh

from scenes.workrooms.kitchen.assets import _utensil, author_task
from scenes.workrooms.kitchen.utensil_rest import spoon_rest_transform


def support_margin(vertices, center):
    vertices = np.asarray(vertices)
    support = vertices[vertices[:, 2] <= vertices[:, 2].min()+1e-8]
    polygon = ConvexHull(support[:, :2])
    margin = -(polygon.equations[:, :2]@np.asarray(center)[:2]
               +polygon.equations[:, 2]).max()
    return support, margin


def test_spoon_rest_uses_actual_bowl_and_handle_triangle():
    original, original_details = _utensil('spoon')
    assert support_margin(original.vertices, original.center_mass)[1] < -.04
    transform, receipt = spoon_rest_transform(original)
    rested, details = _utensil('spoon', resting=True)
    support, margin = support_margin(rested.vertices, rested.center_mass)
    assert len(support) == 3
    assert margin > .0015
    assert math.degrees(receipt['rotation_angle_rad']) == pytest.approx(.9144599266)
    anchors = np.asarray(receipt['support_vertices_before_transform_m'])
    np.testing.assert_allclose(np.sort(anchors[:, 1]), [-.0443, .081, .081])
    np.testing.assert_allclose(np.sort(anchors[:, 2]), [0., .002, .002])
    np.testing.assert_allclose(rested.vertices,
        trimesh.transform_points(original.vertices, transform), atol=1e-12)
    assert rested.volume == pytest.approx(original.volume, rel=1e-12)
    np.testing.assert_allclose(np.linalg.eigvalsh(rested.moment_inertia),
                               np.linalg.eigvalsh(original.moment_inertia), rtol=1e-12)
    for (_, a, _), (_, b, _) in zip(original_details, details):
        np.testing.assert_allclose(b.vertices,
            trimesh.transform_points(a.vertices, transform), atol=1e-12)


@pytest.mark.parametrize('presentation', [False, True])
def test_sorting_spoons_author_real_stable_support_in_both_resets(presentation):
    stage = Usd.Stage.CreateInMemory()
    meta = author_task(stage, 'utensil_sorting', presentation=presentation,
                       prop_profile='household_v1')
    for item in meta['objects']:
        if item['category'] != 'spoon':
            assert 'natural_rest' not in item
            continue
        vertices = np.asarray(UsdGeom.Mesh(
            stage.GetPrimAtPath(item['collider_paths'][0])).GetPointsAttr().Get())
        support, margin = support_margin(vertices, item['center_of_mass_m'])
        assert len(support) == 3
        assert margin > .0017
        assert abs(vertices[:, 2].min()) < 1e-8
        assert item['mass_kg'] == pytest.approx(.032*1.15**3)
        rotation = np.asarray(item['natural_rest']['source_to_rest_transform'])[:3, :3]
        np.testing.assert_allclose(item['handle_axis_local'], rotation@[0., 1., 0.])
        np.testing.assert_allclose(item['food_contact_axis_local'], rotation@[0., -1., 0.])


def test_legacy_sorting_spoons_keep_the_original_recipe():
    stage = Usd.Stage.CreateInMemory()
    meta = author_task(stage, 'utensil_sorting')
    original, _ = _utensil('spoon')
    for item in meta['objects']:
        if item['category'] != 'spoon':
            continue
        vertices = np.asarray(UsdGeom.Mesh(
            stage.GetPrimAtPath(item['collider_paths'][0])).GetPointsAttr().Get())
        np.testing.assert_allclose(vertices, original.vertices, atol=1e-8)
        assert 'natural_rest' not in item


def test_existing_table_spoon_shape_stays_unchanged():
    stage = Usd.Stage.CreateInMemory()
    meta = author_task(stage, 'table_clearing', prop_profile='household_v1')
    item = next(item for item in meta['objects'] if item['id'] == 'spoon')
    original, _ = _utensil('spoon')
    vertices = np.asarray(UsdGeom.Mesh(
        stage.GetPrimAtPath(item['collider_paths'][0])).GetPointsAttr().Get())
    np.testing.assert_allclose(vertices, original.vertices*1.15, atol=1e-8)
    assert 'natural_rest' not in item
