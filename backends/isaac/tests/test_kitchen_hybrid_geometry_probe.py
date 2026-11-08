"""Pure geometry/layout guards for read-only native hybrid measurements."""
import numpy as np
import pytest
import trimesh
from scripts.validation.kitchen_hybrid_geometry_probe import exact_mesh_distances, evaluate_layout, make_samples, calibrate_layout, read_native_sdf_fields


def test_exact_triangle_distance_sign_and_boundary():
    box = trimesh.creation.box(extents=[2., 2., 2.])
    points = np.array([[0, 0, 0], [3, 0, 0], [1, 0, 0], [.2, .3, .4]])
    signed, closest, triangle, winding = exact_mesh_distances(box, points)
    np.testing.assert_allclose(signed, [-1., 2., 0., -.6], atol=1e-14)
    assert np.isclose(abs(winding[0]), 1) and np.isclose(winding[1], 0)
    assert closest.shape == points.shape and triangle.shape == (4,)
    broken = trimesh.Trimesh(box.vertices, box.faces[:-1], process=False)
    with pytest.raises(ValueError, match='Closed'):
        exact_mesh_distances(broken, points)


def plane_samples():
    centers = np.array([[-.004, 0, 0], [.004, 0, 0]])
    points = list(centers); pairs = []
    for p in centers:
        row = []
        for axis in range(3):
            row.append([len(points), len(points)+1]); d = np.eye(3)[axis]*.00005
            points.extend((p+d, p-d))
        pairs.append(row)
    boundary = len(points); points.extend(([0, 0, 0], [.00005, 0, 0]))
    points = np.asarray(points)
    desc = {'witness_count': 2, 'boundary_first': boundary, 'requested_pitch_m': .0001,
            'finite_difference_pairs': pairs, 'expected_signs': [-1, 1]}
    normals = np.tile([1., 0., 0.], (len(points), 1))
    return points, desc, normals


def test_documented_layout_and_finite_difference_agree():
    points, desc, normals = plane_samples()
    raw = np.c_[points[:, 0], normals]
    result = evaluate_layout(raw, points, points[:, 0], normals, desc, distance_channel=0, gradient_channels=[1, 2, 3])
    assert result['passed'] and result['smooth_witnesses'] == 2
    raw[:, 1:] = 0
    result = evaluate_layout(raw, points, points[:, 0], normals, desc, distance_channel=0, gradient_channels=[1, 2, 3])
    assert not result['passed'] and not result['finite_difference_gradient_passed']


def test_reversed_sdk_layout_is_not_silently_accepted():
    points, desc, normals = plane_samples(); raw = np.c_[normals, points[:, 0]]
    documented = evaluate_layout(raw, points, points[:, 0], normals, desc, distance_channel=0, gradient_channels=[1, 2, 3])
    alternative = evaluate_layout(raw, points, points[:, 0], normals, desc, distance_channel=3, gradient_channels=[0, 1, 2])
    assert not documented['passed'] and alternative['passed']
    original = raw.copy()
    calibrated = calibrate_layout(raw, points, points[:, 0], normals, desc)
    assert calibrated['passed'] and calibrated['passing_layout_count'] == 1
    assert calibrated['selected_distance_channel'] == 3 and calibrated['matches_documentation'] is False
    assert not calibrated['candidate_layouts'][0]['passed']
    assert np.array_equal(original, raw) and not calibrated['scale_fitted']
    missing = calibrate_layout(np.zeros_like(raw), points, points[:, 0], normals, desc)
    assert not missing['passed'] and missing['selected_distance_channel'] is None
    raw[0, 0] = np.nan
    with pytest.raises(ValueError, match='finite native'):
        evaluate_layout(raw, points, points[:, 0], normals, desc, distance_channel=0, gradient_channels=[1, 2, 3])


def test_bowl_witnesses_really_separate_material_cavity_and_boundary():
    from scenes.workrooms.kitchen import geometry as g
    profile = np.array([(0, 0), (.006, 0), (.012, .002), (.0175, .007), (.018, .009),
                        (.0165, .010), (.014, .006), (.008, .0035), (0, .003)])
    bowl = g.lathe(profile, 72, (1., 1.45)); bowl.apply_translation((0, -.053, 0)); bowl.apply_scale(1.15)
    row = {'dimension_profile': {'linear_scale_xyz': [1.15]*3},
           'spoon_split_bowl_collision_partition': {'bowl_sdf_resolution': 768}}
    points, desc = make_samples(row, bowl)
    exact, _, _, _ = exact_mesh_distances(bowl, points)
    n = desc['witness_count']
    assert n == 53 and len(points)-desc['boundary_first'] == 72
    assert np.array_equal(np.sign(exact[:n]), desc['expected_signs'])
    assert np.min(abs(exact[:n])) > 2*desc['requested_pitch_m']
    assert np.max(abs(exact[desc['boundary_first']:])) <= .51*desc['requested_pitch_m']


def test_native_sdf_uses_exact_string_views_and_preserves_shape_order():
    from types import SimpleNamespace
    calls = []
    points = np.arange(24, dtype=np.float32).reshape(2, 4, 3)
    def create(path, count):
        assert isinstance(path, str)
        calls.append((path, count))
        return SimpleNamespace(count=1, object_paths=[path], max_num_points=count,
            check=lambda: True,
            get_sdf_and_gradients=lambda p: np.concatenate((p, p[:, :, :1]), axis=2))
    sim = SimpleNamespace(create_sdf_shape_view=create)
    result = read_native_sdf_fields(sim, ['/World/bowl_1', '/World/bowl_0'], points, np.asarray, np.asarray)
    assert calls == [('/World/bowl_1', 4), ('/World/bowl_0', 4)]
    np.testing.assert_array_equal(result[:, :, :3], points)


def test_native_sdf_rejects_wrong_shape_identity_and_missing_samples():
    from types import SimpleNamespace
    points = np.ones((1, 3, 3), np.float32)
    view = SimpleNamespace(count=1, object_paths=['/World/wrong'], max_num_points=3, check=lambda: True)
    sim = SimpleNamespace(create_sdf_shape_view=lambda path, count: view)
    with pytest.raises(ValueError, match='inventory differs'):
        read_native_sdf_fields(sim, ['/World/bowl'], points, np.asarray, np.asarray)
    view.object_paths = ['/World/bowl']
    view.get_sdf_and_gradients = lambda p: np.zeros((1, 2, 4))
    with pytest.raises(ValueError, match='Complete finite'):
        read_native_sdf_fields(sim, ['/World/bowl'], points, np.asarray, np.asarray)
