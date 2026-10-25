"""Regressions for cloth faces crossing a support despite clear vertices."""
import numpy as np
import pytest

from bench.evaluation.cloth.audit_cloth_surface import (
    cylinder_sdf,
    grid_triangles,
    summarize_surface,
    topology_report,
)


CENTER = np.array([.53, .10, .85])
RADIUS = .08


def support_sdf(points):
    return cylinder_sdf(points, CENTER, np.array([1., 0., 0.]), RADIUS, .30)


@pytest.mark.parametrize("subdivisions", [1, 2, 8])
def test_face_interior_penetration_is_detected_when_vertices_and_edges_are_clear(subdivisions):
    # An equilateral cross-section encloses the cylinder: circumradius 20 cm,
    # inradius 10 cm, cylinder radius 8 cm. Every edge is outside the cylinder,
    # while the face centroid is on its axis. Vertex/edge-only checks miss it.
    vertices = CENTER + np.array([
        [0., .20, 0.],
        [0., -.10, np.sqrt(3) * .10],
        [0., -.10, -np.sqrt(3) * .10],
    ])
    midpoints = (vertices + np.roll(vertices, -1, axis=0)) / 2
    assert np.all(support_sdf(vertices) > 0)
    assert np.all(support_sdf(midpoints) > 0)

    report = summarize_surface(vertices, np.array([[0, 1, 2]]), support_sdf, subdivisions)

    assert report["minimum_vertex_sdf_m"] == pytest.approx(.12)
    assert report["minimum_sampled_surface_sdf_m"] == pytest.approx(-RADIUS)
    assert report["triangle_count_with_sampled_penetration"] == 1
    assert report["triangle_count_with_certified_nonpenetration"] == 0
    assert report["triangle_count_unresolved_by_sampling_bound"] == 0
    assert not report["all_triangles_certified_outside"]
    assert report["worst_triangle_id"] == 0
    np.testing.assert_allclose(report["worst_sample_position_m"], CENTER, atol=1e-12)
    np.testing.assert_allclose(report["worst_sample_barycentric"], [1/3] * 3, atol=1e-12)


@pytest.mark.parametrize("subdivisions", [2, 8])
def test_small_safe_face_has_positive_conservative_clearance_certificate(subdivisions):
    # The whole face has y-center_y = 12 cm, so its exact minimum clearance
    # from the 8 cm cylinder is 4 cm, attained on the z=center_z edge.
    vertices = CENTER + np.array([
        [-.01, .12, 0.], [.01, .12, 0.], [0., .12, .01],
    ])

    report = summarize_surface(vertices, np.array([[0, 1, 2]]), support_sdf, subdivisions)

    assert report["minimum_sampled_surface_sdf_m"] == pytest.approx(.04)
    assert 0 < report["minimum_certified_surface_sdf_lower_bound_m"] <= .04
    assert report["triangle_count_with_sampled_penetration"] == 0
    assert report["triangle_count_with_certified_nonpenetration"] == 1
    assert report["triangle_count_unresolved_by_sampling_bound"] == 0
    assert report["all_triangles_certified_outside"]


def test_positive_vertices_and_samples_do_not_certify_an_unresolved_face():
    # This face really is safe (all points have y >= radius + 1 mm), but the
    # coarse sample covering bound is too loose to prove it. Keep "unknown"
    # separate from both penetration and certified nonpenetration.
    vertices = CENTER + np.array([
        [-.01, .081, 0.], [.01, .081, 0.], [0., .081, .01],
    ])

    report = summarize_surface(vertices, np.array([[0, 1, 2]]), support_sdf, subdivisions=1)

    assert report["minimum_vertex_sdf_m"] > 0
    assert report["minimum_sampled_surface_sdf_m"] > 0
    assert report["minimum_certified_surface_sdf_lower_bound_m"] < 0
    assert report["triangle_count_with_sampled_penetration"] == 0
    assert report["triangle_count_with_certified_nonpenetration"] == 0
    assert report["triangle_count_unresolved_by_sampling_bound"] == 1
    assert not report["all_triangles_certified_outside"]


def test_rectangular_cloth_grid_is_one_connected_surface_with_only_outer_boundary():
    report, edges = topology_report(grid_triangles((2, 1)), vertex_count=6)

    assert report["connected_component_sizes"] == [6]
    assert report["vertices"] == 6
    assert report["triangles"] == 4
    assert report["unique_edges"] == len(edges) == 9
    assert report["boundary_edges"] == 6
    assert report["nonmanifold_edges"] == 0
    assert report["vertices"] - report["unique_edges"] + report["triangles"] == 1


def test_disconnected_faces_and_an_isolated_particle_are_reported_as_separate_components():
    report, edges = topology_report(np.array([[0, 1, 2], [3, 4, 5]]), vertex_count=7)

    assert sorted(report["connected_component_sizes"]) == [1, 3, 3]
    assert report["unique_edges"] == len(edges) == 6
    assert report["boundary_edges"] == 6
    assert report["nonmanifold_edges"] == 0
