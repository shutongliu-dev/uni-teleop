import numpy as np

from bench.convex_geometry import convex_triangle_audit


def audit_box(vertices, half=(1., 1., 1.), center=(0., 0., 0.)):
    half = np.asarray(half)
    normals = np.concatenate([np.eye(3), -np.eye(3)])
    planes = np.column_stack([normals, -np.tile(half, 2)])
    return convex_triangle_audit(np.asarray(vertices) - center, np.array([[0, 1, 2]]),
                                 planes, np.stack([-half, half]))


def test_full_triangle_crossing_is_detected_even_when_every_vertex_is_outside():
    vertices = np.array([[-2., 0., 0.], [2., 0., 0.], [0., 2., 0.]])
    assert (np.max(np.abs(vertices), axis=1) > 1).all()
    result = audit_box(vertices)
    assert result["triangles_penetrating_tolerance"] == [0]
    assert result["maximum_confirmed_piece_penetration_m"] > .4


def test_thin_sharp_edge_crossing_is_detected_between_triangle_samples():
    vertices = [[-1., -.1, 0.], [1., -.1, 0.], [0., .9, 0.]]
    result = audit_box(vertices, half=(.0001, .0001, .02), center=(.123, .137, 0.))
    assert not result["all_triangles_certified_clear_within_tolerance"]
    assert result["maximum_confirmed_piece_penetration_m"] > 1e-6


def test_nonpenetrating_touch_and_compound_gap_are_preserved():
    tangent = [[-2., 0., 1.], [2., 0., 1.], [0., 2., 1.]]
    assert audit_box(tangent)["all_triangles_certified_clear_within_tolerance"]
    # Two separate convex collision pieces leave an open channel. A global
    # convex hull would seal it and incorrectly report this cloth as inside.
    gap_triangle = [[-.05, -.05, 0.], [.05, -.05, 0.], [0., .05, 0.]]
    for x in [-.2, .2]:
        assert audit_box(gap_triangle, half=(.1, .2, .2), center=(x, 0, 0))[
            "all_triangles_certified_clear_within_tolerance"]

