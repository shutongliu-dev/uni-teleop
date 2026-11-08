"""Face-interior footprints are measured regions, not chosen corner normals."""
from copy import deepcopy

import numpy as np
import pytest
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation

from scripts.validation.laundry_facet_regions import coplanar_facets, world_facets, clipped_facet_region
from scripts.validation.laundry_pinch_geometry import PinchGeometry, _surface, _area
from scripts.validation.laundry_capture_gate import CaptureGate
from scripts.validation.laundry_contact_closure_stop import ContactClosureStop
from scripts.validation.laundry_capture_settings import DEFAULTS, phase_capture_settings
from test_laundry_contact_normals import sample
from test_laundry_contact_stop_integration import setup


def cube_pad():
    vertices = np.array([[x, y, z] for x in (-.01, .01) for y in (-.01, .01) for z in (-.01, .01)])
    hull = ConvexHull(vertices)
    pad = {'inner': vertices, 'faces': hull.simplices, 'inner_planes': hull.equations}
    facets = coplanar_facets(vertices, hull.simplices, hull.equations)
    assert len(facets) == 6
    return pad, next(f for f in facets if np.allclose(f['plane'][:3], [1, 0, 0]))


def clip(triangle, face, pad):
    return clipped_facet_region(triangle, face, pad, band=.0023, penetration=.0005,
        minimum_area=5.e-7, surface_query=_surface, polygon_area=_area)


def test_clipped_subregion_finds_face_contact_lost_by_whole_triangle_mean():
    pad, face = cube_pad()
    triangle = np.array([[.011, .008, -.004], [.011, .014, -.004], [.011, .014, .004]])
    original = _surface(triangle.mean(0), pad['inner'], pad['faces'], pad['inner_planes'],
                        contact_normal_profile='distance_gradient_v1')
    assert original[2][0] < .5  # The whole mean is outside the face projection.
    region = clip(triangle, face, pad)
    assert region is not None and region['clipped_area_m2'] >= 5.e-7
    assert region['normal_agreement'] == pytest.approx(1.)
    assert region['signed_inner_distance_m'] == pytest.approx(.001)
    np.testing.assert_allclose(region['normal_world'], [1, 0, 0], atol=1.e-12)
    assert np.asarray(region['clipped_polygon_m'])[:, 1].max() <= .01+1.e-12


@pytest.mark.parametrize('triangle', [
    [[.011, .012, -.004], [.011, .014, -.004], [.011, .014, .004]],  # Outside projection.
    [[.013, -.004, -.004], [.013, .004, -.004], [.013, 0, .004]],  # Outside original band.
    [[.009, -.004, -.004], [.009, .004, -.004], [.009, 0, .004]],  # Too far inside pad.
    [[.010, -.004, -.004], [.010, .004, -.004], [.010, 0, .004]],  # Undefined distance gradient.
    [[.011, 0, 0], [.011, .0001, 0], [.011, 0, .0001]],  # Below original finite area.
])
def test_prism_distance_penetration_ambiguity_and_area_cannot_supply_false_witness(triangle):
    pad, face = cube_pad()
    assert clip(np.array(triangle), face, pad) is None


def test_wrong_face_cannot_replace_a_closer_real_boundary():
    pad, face = cube_pad()
    # This lies inside the +X projection prism, but is closer to +Y internally.
    triangle = np.array([[.00955, .0099, -.004], [.00995, .0099, -.004], [.00955, .0099, .004]])
    assert _area(triangle) > 5.e-7
    assert clip(triangle, face, pad) is None


def test_cached_facets_and_regions_follow_rigid_body_transform():
    from bench.evaluation.cloth.cloth_robot_collision import apply_transform, planes_in_body
    pad, face = cube_pad()
    triangle = np.array([[.011, .008, -.004], [.011, .014, -.004], [.011, .014, .004]])
    baseline = clip(triangle, face, pad)
    matrix = np.eye(4)
    matrix[:3, :3] = Rotation.from_rotvec([.3, -.4, .2]).as_matrix()
    matrix[:3, 3] = [.6, -.8, .9]
    moved = {'inner': apply_transform(pad['inner'], matrix), 'faces': pad['faces'],
             'inner_planes': planes_in_body(pad['inner_planes'], matrix)}
    actual = clip(apply_transform(triangle, matrix), world_facets([face], matrix)[0], moved)
    assert actual['clipped_area_m2'] == pytest.approx(baseline['clipped_area_m2'])
    np.testing.assert_allclose(actual['point_m'], apply_transform(np.array([baseline['point_m']]), matrix)[0], atol=1.e-12)
    np.testing.assert_allclose(actual['normal_world'], matrix[:3, :3] @ baseline['normal_world'], atol=1.e-12)


def sewn_fixture():
    from test_laundry_pinch_geometry import fixture
    export, names, poses, q, tri = fixture()
    copies = deepcopy(export['colliders'])
    for row in copies:
        row['body_name'] = row['body_name'].replace('elastomer', 'DP')
        row['path'] = row['path'].replace('elastomer', 'DP')
        names.append(row['body_name'])
    export['colliders'].extend(copies)
    poses[0, 0] -= .002
    poses[1, 0] += .002
    poses = np.vstack([poses, poses.copy()])
    q = np.vstack([q+[-.002, 0, 0], q+[.002, 0, 0],
                   [[-.002, .05, -.004], [-.002, .05, .004], [.002, .05, -.004], [.002, .05, .004]]])
    tri = np.vstack([tri, tri+4, [[1, 8, 9], [1, 9, 2], [5, 10, 11], [5, 11, 6], [8, 10, 11], [8, 11, 9]]])
    return export, names, poses, q, tri


def test_explicit_profile_propagates_to_every_surface_and_default_remains_exact_after_cache():
    export, names, poses, q, tri = sewn_fixture()
    geometry = PinchGeometry(export, names)
    baseline = geometry.diagnose('right', poses, q, tri, q, .0012)
    result = geometry.diagnose('right', poses, q, tri, q, .0012,
        contact_normal_profile='distance_gradient_v1', contact_witness_profile='facet_regions_v1')
    for variant in [result]+result['distal_contact_diagnostics']:
        assert variant['contact_witness_profile'] == 'facet_regions_v1'
        assert variant['contact_normal_profile'] == 'distance_gradient_v1'
        assert variant['multilayer_diagnostic']['plausible']
        for pair in variant['multilayer_diagnostic']['paired_witnesses']:
            assert pair['opposing_normal_dot'] <= -.5
            assert all(pair[f]['normal_agreement'] >= 1.-1.e-6 for f in ('thumb', 'index'))
    assert geometry.diagnose('right', poses, q, tri, q, .0012) == baseline
    assert geometry.diagnose('right', poses, q, tri, q, .0012, contact_witness_profile='centroid_v1') == baseline
    for kwargs in ({'contact_witness_profile': 'facet_regions_v1'}, {'contact_witness_profile': 'arbitrary_face'}):
        with pytest.raises(ValueError):
            geometry.diagnose('right', poses, q, tri, q, .0012, **kwargs)


def test_gate_stop_require_matching_sampling_marker_and_keep_all_thresholds(sample):
    for requested, reported in [('facet_regions_v1', 'centroid_v1'), ('centroid_v1', 'facet_regions_v1')]:
        diagnostic = deepcopy(sample['diag'])
        diagnostic.update(contact_normal_profile='distance_gradient_v1', contact_witness_profile=reported)
        options = dict(contact_normal_profile='distance_gradient_v1', contact_witness_profile=requested)
        gate = CaptureGate(*sample['args'], **options)
        stop = ContactClosureStop(*sample['args'], {'right_thumb_IP': (-1, 1), 'right_index_DIP': (-1, 1)}, **options)
        assert gate.thresholds == CaptureGate(*sample['args']).thresholds
        for t in np.arange(0, .101, .02):
            g = gate.observe(t, diagnostic, sample['body'], [.5, .5], [0, 0], [.51, .51], .06, cloth_q=sample['q'])
            s = stop.observe(t, diagnostic, sample['body'], [.5, .5], [.51, .51], .06, cloth_q=sample['q'])
        assert not g['capture_candidate'] and not s['latched']
        assert 'contact_witness_profile_mismatch' in g['reasons'] and 'contact_witness_profile_mismatch' in s['reasons']
        assert not gate.history


def test_settings_validate_coupled_profile_before_start_and_preserve_defaults():
    options = {'contact_witness_profile': 'facet_regions_v1', 'contact_normal_profile': 'distance_gradient_v1'}
    assert phase_capture_settings({}) == DEFAULTS
    assert phase_capture_settings(options) == {**DEFAULTS, **options}
    assert phase_capture_settings({'phases': {'1': {'capture_settings': options}}}, 1) == {**DEFAULTS, **options}
    with pytest.raises(ValueError, match='distance-gradient'):
        phase_capture_settings({'contact_witness_profile': 'facet_regions_v1'})
    with pytest.raises(ValueError):
        phase_capture_settings({'contact_witness_profile': 'arbitrary_face'})


def test_assist_propagates_sampling_to_geometry_gate_and_actual_stop(setup):
    options = dict(contact_closure_stop=True, contact_closure_target_policy='hold_command_v2',
        contact_normal_profile='distance_gradient_v1', contact_witness_profile='facet_regions_v1')
    assist = setup.make(**options)
    assert assist.geometry_options == {k: options[k] for k in ('contact_normal_profile', 'contact_witness_profile')}
    assert assist.gate.contact_witness_profile == assist.closure_stop.contact_witness_profile == 'facet_regions_v1'
    original = assist.geometry.diagnose
    calls = []
    def measured(*args, **kwargs):
        calls.append(kwargs)
        result = original(*args)
        result.update(kwargs)
        return result
    assist.geometry.diagnose = measured
    assert assist.observe_closure('close', 0., setup.state, setup.q, setup.env._targets[0])
    assert calls == [assist.geometry_options]
    assert assist.closure_stop.receipt['contact_witness_profile'] == 'facet_regions_v1'
    assert assist.prim is None and not assist.active
