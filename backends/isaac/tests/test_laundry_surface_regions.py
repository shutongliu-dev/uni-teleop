"""Finite edge regions preserve real surface directions and all capture gates."""
from copy import deepcopy

import numpy as np
import pytest
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation

from bench.evaluation.cloth.cloth_robot_collision import apply_transform, planes_in_body
from scripts.validation.laundry_edge_regions import convex_edges, world_edges, clipped_edge_region
from scripts.validation.laundry_pinch_geometry import PinchGeometry, _surface, _area
from scripts.validation.laundry_capture_gate import CaptureGate
from scripts.validation.laundry_contact_closure_stop import ContactClosureStop
from scripts.validation.laundry_capture_settings import phase_capture_settings, DEFAULTS
from test_laundry_contact_normals import sample
from test_laundry_facet_regions import sewn_fixture


def edge_fixture():
    vertices = np.array([[x, y, z] for x in [-.01, .01]
                         for y in [-.01, .01] for z in [-.01, .01]])
    hull = ConvexHull(vertices)
    pad = {'inner': vertices, 'faces': hull.simplices, 'inner_planes': hull.equations}
    edges = convex_edges(vertices, hull.simplices, hull.equations)
    assert len(edges) == 12  # Internal face triangulation diagonals are omitted.
    edge = next(e for e in edges if np.all(e['endpoints'][:, :2] == [.01, .01]))
    triangle = np.array([[.011, .011, -.004], [.012, .011, .004], [.011, .012, .004]])
    options = dict(axis=np.array([1., 1., 0.])/np.sqrt(2), orientation=1,
        alo=-.1, bhi=.1, thickness=.0012, band=.0023, penetration=.0005,
        minimum_area=5.e-7, surface_query=_surface, polygon_area=_area)
    return pad, edge, triangle, options


def test_finite_edge_has_exact_nearest_points_and_true_distance_gradients():
    pad, edge, triangle, options = edge_fixture()
    result = clipped_edge_region(triangle, edge, pad, **options)
    assert result and result['clipped_area_m2'] >= 5.e-7
    assert result['closest_feature'] == 'exterior_convex_edge'
    for point, check in zip(result['clipped_polygon_m'], result['finite_region_vertex_checks']):
        assert check['distance_m'] <= options['band']+1.e-8
        assert check['nearest_residual_m'] < 1.e-7
        normal = np.array(point)-[.01, .01, point[2]]
        assert np.linalg.norm(normal) <= options['band']+1.e-8
    point, surface = np.array(result['point_m']), np.array(result['surface_point_m'])
    np.testing.assert_allclose(result['normal_world'], (point-surface)/np.linalg.norm(point-surface), atol=1.e-12)


@pytest.mark.parametrize('change', ['beyond_segment', 'beyond_band', 'tiny', 'wrong_direction', 'outside_slot'])
def test_edge_does_not_supply_false_area_or_override_original_direction_slot(change):
    pad, edge, triangle, options = edge_fixture()
    if change == 'beyond_segment': triangle += [0, 0, .1]
    elif change == 'beyond_band': triangle += [.01, .01, 0]
    elif change == 'tiny': triangle = triangle[:1]+(triangle-triangle[:1])*1.e-4
    elif change == 'wrong_direction': options['orientation'] = -1
    else: options['bhi'] = 0.
    assert clipped_edge_region(triangle, edge, pad, **options) is None


def test_cached_edges_follow_actual_rigid_body_without_rebuilding_geometry():
    pad, edge, triangle, options = edge_fixture()
    expected = clipped_edge_region(triangle, edge, pad, **options)
    matrix = np.eye(4)
    matrix[:3, :3] = Rotation.from_rotvec([.3, -.4, .2]).as_matrix()
    matrix[:3, 3] = [.6, -.8, .9]
    moved = {'inner': apply_transform(pad['inner'], matrix), 'faces': pad['faces'],
             'inner_planes': planes_in_body(pad['inner_planes'], matrix)}
    axis = matrix[:3, :3] @ options['axis']
    shift = float(matrix[:3, 3] @ axis)
    actual = clipped_edge_region(apply_transform(triangle, matrix), world_edges([edge], matrix)[0], moved,
        **{**options, 'axis': axis, 'alo': options['alo']+shift, 'bhi': options['bhi']+shift})
    assert actual['clipped_area_m2'] == pytest.approx(expected['clipped_area_m2'])
    np.testing.assert_allclose(actual['normal_world'], matrix[:3, :3] @ expected['normal_world'], atol=1.e-12)


def opposing_edge_fixture():
    from test_laundry_pinch_geometry import fixture
    export, names, poses, _, tri = fixture()
    for row, limits in zip(export['colliders'], [(-.010, -.002), (.002, .010)]):
        vertices = np.array([[x, y, z] for x in limits for y in limits for z in [-.008, .008]])
        planes = ConvexHull(vertices).equations.tolist()
        row['pieces'][0].update(vertices=vertices.tolist(), conservative_outer_vertices=vertices.tolist(),
            conservative_outer_planes=planes, conservative_inner_planes=planes)
    q = np.array([[-.001, -.001, -.004], [.001, .001, -.004],
                  [.001, .001, .004], [-.001, -.001, .004]])
    return export, names, poses, q, tri


def test_opposing_edge_regions_find_same_local_material_without_arbitrary_facet_normal():
    export, names, poses, q, tri = opposing_edge_fixture()
    geometry = PinchGeometry(export, names)
    options = dict(contact_normal_profile='distance_gradient_v1')
    original = geometry.diagnose('right', poses, q, tri, q, .0012,
        contact_witness_profile='facet_regions_v1', **options)
    assert not original['multilayer_diagnostic']['plausible']
    result = geometry.diagnose('right', poses, q, tri, q, .0012,
        contact_witness_profile='surface_regions_v2', **options)
    pairs = result['multilayer_diagnostic']['paired_witnesses']
    assert result['multilayer_diagnostic']['plausible'] and pairs
    for pair in pairs:
        assert pair['opposing_normal_dot'] <= -.5
        assert pair['local_topology_connected']
        assert all(pair[f]['closest_feature'] == 'exterior_convex_edge' for f in ('thumb', 'index'))
    distant_rest = q*20
    assert not geometry.diagnose('right', poses, q, tri, distant_rest, .0012,
        contact_witness_profile='surface_regions_v2', **options)['multilayer_diagnostic']['plausible']


def test_old_default_and_facet_records_stay_exact_after_new_edge_caches():
    export, names, poses, q, tri = sewn_fixture()
    geometry = PinchGeometry(export, names)
    args = ('right', poses, q, tri, q, .0012)
    baseline = geometry.diagnose(*args)
    face = geometry.diagnose(*args, contact_normal_profile='distance_gradient_v1', contact_witness_profile='facet_regions_v1')
    surface = geometry.diagnose(*args, contact_normal_profile='distance_gradient_v1', contact_witness_profile='surface_regions_v2')
    for old, new in zip([face]+face['distal_contact_diagnostics'], [surface]+surface['distal_contact_diagnostics']):
        for finger in ('thumb', 'index'):
            a = old['multilayer_diagnostic']['per_finger_witnesses'][finger]
            assert new['multilayer_diagnostic']['per_finger_witnesses'][finger][:len(a)] == a
    assert geometry.diagnose(*args) == baseline
    assert geometry.diagnose(*args, contact_normal_profile='distance_gradient_v1', contact_witness_profile='facet_regions_v1') == face


def test_explicit_surface_profile_requires_matching_gate_stop_and_same_thresholds(sample):
    options = dict(contact_normal_profile='distance_gradient_v1', contact_witness_profile='surface_regions_v2')
    assert phase_capture_settings({}) == DEFAULTS
    assert phase_capture_settings(options) == {**DEFAULTS, **options}
    assert phase_capture_settings({'phases': {'1': {'capture_settings': options}}}, 1) == {**DEFAULTS, **options}
    with pytest.raises(ValueError, match='distance-gradient'):
        phase_capture_settings({'contact_witness_profile': 'surface_regions_v2'})
    diagnostic = deepcopy(sample['diag'])
    diagnostic.update(contact_normal_profile='distance_gradient_v1', contact_witness_profile='facet_regions_v1')
    gate = CaptureGate(*sample['args'], **options)
    stop = ContactClosureStop(*sample['args'], {'right_thumb_IP': (-1, 1), 'right_index_DIP': (-1, 1)}, **options)
    assert gate.thresholds == CaptureGate(*sample['args']).thresholds
    for t in np.arange(0, .101, .02):
        g = gate.observe(t, diagnostic, sample['body'], [.5, .5], [0, 0], [.51, .51], .06, cloth_q=sample['q'])
        s = stop.observe(t, diagnostic, sample['body'], [.5, .5], [.51, .51], .06, cloth_q=sample['q'])
    assert not g['capture_candidate'] and not s['latched']
    assert 'contact_witness_profile_mismatch' in g['reasons'] and 'contact_witness_profile_mismatch' in s['reasons']
