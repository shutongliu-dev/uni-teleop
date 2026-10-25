"""Physical fixture contracts; no robot grasp or native physics acceptance."""
import numpy as np
import pytest

from scenes.workrooms.card_room.game_assets import stand
from scenes.workrooms.card_room.reserve_assets import reserve_tray


def positions(wing, sign, count):
    xs = (.04, .14, .24) if wing == 'left' else (1.06, 1.16, 1.26)
    points = [(x, sign * y, .807) for y in (.36, .46, .56, .66) for x in xs]
    if count == 13:
        points.append((.14 if wing == 'left' else 1.16, sign * .76, .807))
    return points[:count]


def bounds(part):
    center, half = np.asarray(part['position_m']), np.asarray(part['half_extents_m'])
    return np.array([center - half, center + half])


def overlap(a, b):
    return (np.minimum(a[1], b[1]) - np.maximum(a[0], b[0]) > 1.e-9).all()


@pytest.mark.parametrize('wing,sign,count', [
    ('left', -1, 13), ('right', -1, 12), ('left', 1, 12), ('right', 1, 12),
    ('left', -1, 3), ('right', -1, 2), ('left', 1, 2), ('right', 1, 2),
    ('left', -1, 7), ('right', -1, 6), ('left', 1, 6), ('right', 1, 6),
])
def test_real_support_geometry_preserves_token_margin_and_table(wing, sign, count):
    points = positions(wing, sign, count)
    asset = reserve_tray('reserve_' + wing, points)
    assert asset.dynamic is False and asset.mass_kg == 0.
    assert asset.friction == .4
    assert all(mesh is None and part['geometry'] == 'box' for _, mesh, part in asset.collision)
    pieces = {label: part for label, _, part in asset.collision}
    plate = bounds(pieces['reserve_platform'])
    xy = np.array(points)[:, :2]
    np.testing.assert_allclose(plate[:, :2], [xy.min(0) - .04, xy.max(0) + .04], atol=1.e-12)
    np.testing.assert_allclose(plate[:, 2], [.78, .80], atol=1.e-12)
    assert (xy - .028 >= plate[0, :2] + .012 - 1.e-12).all()
    assert (xy + .028 <= plate[1, :2] - .012 + 1.e-12).all()
    assert asset.data['required_token_base_center_z_m'] == pytest.approx(.807)
    table = np.array([[.15, -.5, .74], [1.15, .5, .78]])
    for part in pieces.values():
        assert not overlap(bounds(part), table)
    legs = [bounds(part) for label, part in pieces.items() if label.startswith('outboard_leg_')]
    assert len(legs) == 2
    for leg in legs:
        assert leg[0, 2] == pytest.approx(0.)
        assert leg[1, 2] == pytest.approx(plate[0, 2])
        assert (leg[0, :2] >= plate[0, :2]).all()
        assert (leg[1, :2] <= plate[1, :2]).all()
        # Whole posts clear the table footprint at every height, including legs
        # and aprons that sit lower than the selected worktop collider.
        assert leg[1, 0] <= .142 or leg[0, 0] >= 1.158 or leg[1, 1] <= -.508 or leg[0, 1] >= .508


@pytest.mark.parametrize('wing,sign,count', [
    ('left', -1, 13), ('right', -1, 12), ('left', 1, 12), ('right', 1, 12),
])
def test_corner_trays_clear_both_actual_robot_support_assets(wing, sign, count):
    asset = reserve_tray('corner_tray', positions(wing, sign, count))
    for side in (-1, 1):
        support = stand(side)
        for tray_label, _, tray_part in asset.collision:
            for stand_label, _, stand_part in support.collision:
                assert not overlap(bounds(tray_part), bounds(stand_part)), (tray_label, stand_label)


def test_sparse_group_can_rest_on_table_or_use_front_edge_trestle():
    table_only = reserve_tray('small_plate', [(.24, -.36, .807)])
    assert table_only.data['outboard_leg_count'] == 0
    assert len(table_only.collision) == 1
    outside_y = reserve_tray('outer_plate', [(.24, -.76, .807)])
    assert outside_y.data['support_edge'] == {'axis': 'y', 'side': -1}
    assert outside_y.data['outboard_leg_count'] == 2
    for label, _, part in outside_y.collision:
        if label.startswith('outboard_'):
            assert bounds(part)[1, 1] < -.50


def test_visible_surface_has_no_raised_grasp_obstacles_and_recipe_is_stateless():
    points = positions('left', -1, 13)
    first = reserve_tray('reserve', points)
    again = reserve_tray('reserve', reversed(points))
    assert first.collision == again.collision
    for name, mesh, _ in first.visual:
        assert np.isfinite(mesh.vertices).all(), name
        assert (mesh.area_faces > 1.e-18).all(), name
        if name in ('walnut_grain', 'reserve_position_rings'):
            assert mesh.bounds[0, 2] >= .80
            assert mesh.bounds[1, 2] <= .80006 + 1.e-12
            assert (mesh.face_normals[:, 2] > .999).all()
        assert mesh.bounds[0, 2] >= -1.e-12
        assert mesh.bounds[1, 2] <= .80006 + 1.e-12
    assert sum(len(mesh.faces) for _, mesh, _ in first.visual) < 7000
    assert first.data['appearance']['raised_rims_or_slots'] is False
    assert all(roughness >= .6 and metallic == 0. for roughness, metallic in first.visual_finish.values())


@pytest.mark.parametrize('points,height', [([], .8), ([[.1]], .8),
    ([[.1, np.nan]], .8), ([[.1, -.4]], .79), ([[.1, -.4]], float('inf')),
    ([[.24, -.36]], .90)])
def test_invalid_or_floating_support_is_rejected(points, height):
    with pytest.raises(ValueError):
        reserve_tray('invalid', points, surface_z=height)
