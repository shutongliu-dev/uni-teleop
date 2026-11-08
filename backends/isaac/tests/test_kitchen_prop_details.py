"""Geometric checks for a visual finish that must preserve contact surfaces."""
import numpy as np
import pytest
import trimesh

pytest.importorskip('pxr')

from scenes.workrooms.kitchen.prop_details import (
    SPONGE_MAX_INSET_M, sponge_visual_geometry,
)


def test_sponge_finish_is_closed_combined_skin_with_flat_bottom_and_green_band():
    parts = sponge_visual_geometry()
    assert len(parts) == 2
    assert [material for _, _, material in parts] == ['crumb', 'leaf']
    combined = trimesh.util.concatenate([mesh for _, mesh, _ in parts])
    combined.merge_vertices()
    assert combined.is_watertight
    assert combined.is_winding_consistent
    assert combined.volume > 0
    assert len(combined.faces) < 160_000
    assert np.allclose(combined.bounds, [[-.050, -.0325, 0], [.050, .0325, .045]], atol=.00015)
    green = parts[1][1]
    assert green.bounds[0, 2] == 0
    assert green.bounds[1, 2] <= .004 + 1e-12
    bottom = np.asarray(green.vertices)[np.asarray(green.vertices)[:, 2] == 0]
    assert len(bottom) > 20
    assert np.ptp(bottom[:, 0]) > .09
    assert np.ptp(bottom[:, 1]) > .055


def test_sponge_finish_has_recesses_on_four_sides_and_top_without_outward_pores():
    foam = sponge_visual_geometry()[0][1]
    vertices = np.asarray(foam.vertices)
    # Test local planar patches away from rounded edges and material seams.
    for axis, extent, other_extent in ((0, .050, .025), (1, .0325, .040)):
        for sign in (-1, 1):
            points = vertices[(vertices[:, axis] * sign > extent - .0002)
                              & (np.abs(vertices[:, 1-axis]) < other_extent)
                              & (vertices[:, 2] > .006) & (vertices[:, 2] < .020)]
            recess = extent - sign * points[:, axis]
            assert len(points) > 20
            assert np.min(recess) >= -1e-12
            assert np.count_nonzero(recess > .000025) > 5
            # The original rounded-loop polygon has small oblique side facets;
            # an axis-plane gap includes that pre-existing chord offset.
            assert np.max(recess) <= .0002
    points = vertices[(np.abs(vertices[:, 0]) < .025)
                      & (np.abs(vertices[:, 1]) < .013) & (vertices[:, 2] > .0448)]
    assert np.min(.045 - points[:, 2]) >= -1e-12
    assert np.count_nonzero(.045 - points[:, 2] > .000025) > 20
    assert np.max(.045 - points[:, 2]) <= SPONGE_MAX_INSET_M + 1e-12
    for _, mesh, _ in sponge_visual_geometry():
        assert mesh.metadata['visual_only'] is True
        assert mesh.metadata['collision'] is False
        assert mesh.metadata['max_local_inset_m'] * 1.2 <= .0002


def test_sponge_colour_resolves_recesses_and_uses_distinct_nonwoven_finish():
    foam, green = [mesh for _, mesh, _ in sponge_visual_geometry()]
    for mesh in (foam, green):
        rgb = np.asarray(mesh.visual.vertex_colors)[:, :3]
        assert rgb.shape == (len(mesh.vertices), 3)
        assert np.ptp(rgb[:, 1]) > 15
        assert np.min(rgb) > 0
        assert np.max(rgb) < 255
    vertices = np.asarray(foam.vertices)
    # Compare actual depressions and undented neighbouring planar skin on the
    # same side, independently of the colour-generation metadata.
    side = ((vertices[:, 1] < -.0323) & (np.abs(vertices[:, 0]) < .035)
            & (vertices[:, 2] > .006) & (vertices[:, 2] < .020))
    depth = .0325 + vertices[:, 1]
    dark = np.asarray(foam.visual.vertex_colors)[side & (depth > .00007), :3]
    light = np.asarray(foam.visual.vertex_colors)[side & (depth < .000012), :3]
    assert len(dark) > 10 and len(light) > 10
    assert dark.mean() < light.mean() * .85
