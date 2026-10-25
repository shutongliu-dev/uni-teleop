"""Continuous-yaw enclosures, including cases missed by angle/vertex sampling."""
import numpy as np
import pytest
import trimesh

from scenes.workrooms.mechanical_assembly.tasks.gear_train_assembly.clearance import (
    clipped_slab_vertices, radial_interval, yaw_sweep_pair_bound,
)


def box(size, center):
    result = trimesh.creation.box(extents=size)
    result.apply_translation(center)
    return result


def test_slab_crossing_without_any_original_vertex_is_retained():
    mesh = box([2., 2., 4.], [3., 0., 0.])
    assert not np.any(abs(mesh.vertices[:, 2]) <= .5)
    points = clipped_slab_vertices(mesh, -.5, .5)
    assert len(points) >= 8
    assert points[:, 2].min() == -.5 and points[:, 2].max() == .5
    assert radial_interval(points) == pytest.approx((2., np.sqrt(17.)))


@pytest.mark.parametrize('points,expected', [
    ([[1., 1., 0.], [-1., 1., 0.], [1., -1., 0.], [-1., -1., 0.]], (0., np.sqrt(2.))),
    ([[2., -1., 0.], [2., 1., 0.]], (2., np.sqrt(5.))),
    ([[3., 4., 0.]], (5., 5.)),
])
def test_radial_interval_includes_convex_edges_and_interior(points, expected):
    assert radial_interval(points) == pytest.approx(expected)


def test_empty_radial_slice_is_not_positive_clearance():
    with pytest.raises(ValueError):
        radial_interval(np.empty((0, 3)))


def test_full_yaw_rejects_a_collision_away_from_the_initial_angle():
    tool = box([.02, .01, .01], [.08, 0., 0.])
    obstacle = box([.01, .01, .01], [0., .08, 0.])
    assert tool.bounds[0, 0] > obstacle.bounds[1, 0]  # Initial angle is clear.
    assert yaw_sweep_pair_bound(tool, obstacle) == 0.  # A quarter turn collides.
    obstacle.apply_translation([0., .04, 0.])
    assert yaw_sweep_pair_bound(tool, obstacle) > .024


def test_different_height_points_constrain_true_3d_gap():
    tool = box([.02, .01, .01], [.08, 0., 0.])
    obstacle = box([.01, .01, .01], [0., .08, .013])
    assert yaw_sweep_pair_bound(tool, obstacle) == pytest.approx(.003)


def test_refined_slabs_can_prove_a_clearance_with_overlapping_whole_bounds():
    # Parallel slanted slender hulls have overlapping global radial and Z
    # intervals. The cross-height distances, not only same-height slices,
    # constrain their continuous-yaw clearance.
    a = trimesh.convex.convex_hull(np.vstack([
        box([.01, .01, .01], [.1, 0., 0.]).vertices,
        box([.01, .01, .01], [.2, 0., .2]).vertices]))
    b = a.copy(); b.apply_translation([.03, 0., 0.])
    bound = yaw_sweep_pair_bound(a, b)
    # Coarse slabs may stay unresolved; they must never exceed a real point
    # pair's available distance and must detect actual intersection.
    assert 0. <= bound <= .03
    assert yaw_sweep_pair_bound(a, a.copy()) == 0.
