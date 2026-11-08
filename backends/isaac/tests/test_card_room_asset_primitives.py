"""Independent mass and support checks for the shared tabletop primitives."""
from dataclasses import replace

import numpy as np
import pytest
import trimesh

from scenes.board_games.tabletop.geometry import geometry_for
from scenes.workrooms.card_room.asset_primitives import stand, token_mass_properties
from scenes.workrooms.card_room.game_assets import token


@pytest.mark.parametrize('geometry', [
    geometry_for('hex7x7'),
    replace(geometry_for('hex7x7'), handle_radius_m=.010, handle_height_m=.060, piece_mass_kg=.035),
    replace(geometry_for('hex7x7'), piece_base_height_m=.020, handle_height_m=.025, piece_mass_kg=.120),
])
def test_analytic_mass_properties_match_closed_collision_volume_integration(geometry):
    """Integrate the actual collider records, independently of mass authoring.

    Cosmetic roundovers must not silently change the analytic physical inertia.
    The independent triangle approximation here is made from the collision
    shapes, rather than treating appearance meshes as collision geometry.
    """
    asset = token('black', geometry)
    cylinders = []
    for _, _, part in asset.collision:
        mesh = trimesh.creation.cylinder(radius=part['radius_m'], height=2*part['half_height_m'], sections=256)
        mesh.apply_translation(part['position_m'])
        cylinders.append(mesh)
    body = trimesh.util.concatenate(cylinders)
    assert body.is_watertight
    body.density = geometry.piece_mass_kg / body.volume
    analytic = token_mass_properties(geometry)
    np.testing.assert_allclose(analytic['center_of_mass_m'], body.center_mass, atol=1.e-12)
    # Triangle integration must remain close to the two true cylinders, not a
    # bounding-box tensor or a tensor calculated from the rounded appearance.
    np.testing.assert_allclose(np.diag(analytic['inertia_diag_kg_m2']),
                               body.moment_inertia, rtol=1.e-3, atol=1.e-15)
    assert analytic['center_of_mass_m'][2] > 0
    inertia = np.asarray(analytic['inertia_diag_kg_m2'])
    assert (inertia > 0).all()
    assert all(inertia[i] < sum(inertia) - inertia[i] for i in range(3))


def test_mass_and_length_scaling_preserve_inertia_dimensions():
    g = geometry_for('hex7x7'); baseline = token_mass_properties(g)
    heavier = token_mass_properties(replace(g, piece_mass_kg=2 * g.piece_mass_kg))
    np.testing.assert_allclose(heavier['center_of_mass_m'], baseline['center_of_mass_m'])
    np.testing.assert_allclose(heavier['inertia_diag_kg_m2'],
                               2 * np.array(baseline['inertia_diag_kg_m2']))
    scale = .8
    scaled = token_mass_properties(replace(g, piece_radius_m=g.piece_radius_m * scale,
        piece_base_height_m=g.piece_base_height_m * scale, handle_radius_m=g.handle_radius_m * scale,
        handle_height_m=g.handle_height_m * scale))
    np.testing.assert_allclose(scaled['center_of_mass_m'], scale * np.array(baseline['center_of_mass_m']))
    np.testing.assert_allclose(scaled['inertia_diag_kg_m2'],
                               scale**2 * np.array(baseline['inertia_diag_kg_m2']))


@pytest.mark.parametrize('sign', [-1, 1])
def test_pedestal_collisions_are_exactly_the_original_four_boxes(sign):
    fixture = stand(sign)
    expected = {
        'foot': ([.65, sign*.74, .025], [.20, .19, .025]),
        'column': ([.65, sign*.74, .455], [.095, .095, .455]),
        'cantilever': ([.65, sign*.625, .915], [.11, .205, .015]),
        'mount_plate': ([.65, sign*.52, .9375], [.11, .10, .0125]),
    }
    assert len(fixture.collision) == len(expected)
    for label, mesh, part in fixture.collision:
        assert mesh is None
        assert part['geometry'] == 'box'
        np.testing.assert_allclose(part['position_m'], expected[label][0], atol=1.e-15)
        np.testing.assert_allclose(part['half_extents_m'], expected[label][1], atol=1.e-15)
