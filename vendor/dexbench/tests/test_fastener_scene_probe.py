import math
import numpy as np
import pytest

from scripts.validation.mechanical_native_geometry import (inertia_checks, sdf_distance_column,
    pitch_from_profiles, zero_surface_check, source_surface_errors)
from scripts.validation.validate_fastener_scene import tool_load, LOAD, assess_loaded_lead, make_load_profile


def test_inertia_uses_com_body_frame_without_second_rotation():
    theta = .4
    r = np.array([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])
    inertia = r@np.diag([2e-5, 3e-5, 4e-5])@r.T
    checked = inertia_checks(.2, inertia, [0, 0, .04, 0, 0, np.sin(theta/2), np.cos(theta/2)])
    assert checked['passed']
    assert np.allclose(checked['principal_moments_kg_m2'], [2e-5, 3e-5, 4e-5])


@pytest.mark.parametrize('mass,inertia,com', [
    (0, np.eye(3), [0, 0, 0, 0, 0, 0, 1]),
    (1, np.diag([-1, 2, 2]), [0, 0, 0, 0, 0, 0, 1]),
    (1, np.diag([1, 1, 3]), [0, 0, 0, 0, 0, 0, 1]),
    (1, np.eye(3), [0, 0, 0, 0, 0, 0, 2]),
])
def test_inertia_rejects_invalid_body(mass, inertia, com):
    assert not inertia_checks(mass, inertia, com)['passed']


@pytest.mark.parametrize('column', [0, 3])
def test_sdf_convention_detected_from_witnesses(column):
    values = np.array([[-.005, 1, 0, 0], [.005, 1, 0, 0], [.005, 0, 1, 0]])
    if column == 3:
        values = values[:, [1, 2, 3, 0]]
    assert sdf_distance_column(values, [-1, 1, 1]) == column


def test_sdf_missing_inside_witness_is_not_pass():
    with pytest.raises(ValueError):
        sdf_distance_column(np.ones((3, 4)), [-1, 1, 1])


@pytest.mark.parametrize('pitch,passed', [(0.002, True), (.0023, False)])
def test_native_sdf_pitch_resolves_scale_once(pitch, passed):
    z = np.linspace(.005, .023, 721)
    values = np.array([.0004*np.cos(2*np.pi*z/pitch+theta) for theta in np.linspace(0, 6, 24)])
    assert pitch_from_profiles(z, values, 1.5)['passed'] is passed


@pytest.mark.parametrize('pitch,passed', [(0.002, True), (.0023, False)])
def test_zero_surface_checks_all_pairs(pitch, passed):
    z = np.linspace(.005, .023, 721)
    radius = np.array([.007+.0004*np.cos(2*np.pi*z/pitch+theta) for theta in np.linspace(0, 6, 24)])
    assert zero_surface_check(z, radius, 1.5)['passed'] is passed


def test_zero_surface_local_defect_is_not_filtered():
    z = np.linspace(.005, .023, 721)
    radius = np.array([.007+.0004*np.cos(2*np.pi*z/.002+theta) for theta in np.linspace(0, 6, 24)])
    radius[12, 320] += .0002
    assert not zero_surface_check(z, radius, 1.5)['passed']


def test_exact_source_triangle_distance_inside_and_outside():
    import trimesh
    mesh = trimesh.creation.box(extents=[2, 2, 2])
    assert np.allclose(source_surface_errors(mesh, [[0, 0, 0], [0, 0, 1], [0, 0, 1.1], [2, 2, 2]]),
                       [1., 0., .1, np.sqrt(3)])


@pytest.mark.parametrize('size', [1., .001, .000001])
def test_source_distance_small_triangle_interior_projection_is_scale_invariant(size):
    import trimesh
    vertices = np.array([[0, 0, 0], [size, 0, 0], [0, size, 0]])
    mesh = trimesh.Trimesh(vertices=vertices, faces=[[0, 1, 2]], process=False)
    # Analytical face-interior and edge projections; no triangle-library oracle.
    points = np.array([[.3, .3, .01], [.8, .8, .01]])*size
    assert np.allclose(source_surface_errors(mesh, points),
                       [size*.01, size*np.sqrt(.3**2+.3**2+.01**2)], rtol=1e-10, atol=size*1e-12)


def test_load_is_bounded_and_has_no_axial_position_tracking():
    pose = np.array([.4, .2, .9, 0., 0., 0., 1.])
    velocity = np.array([10., -10., 5., 4., -4., 10.])
    first = tool_load(pose, velocity, [0, 0, .11], [.3, .3], .3039, 1, .0005, 0)
    pose[2] += .15
    second = tool_load(pose, velocity, [0, 0, .11], [.3, .3], .3039, 1, .0005, 0)
    assert np.array_equal(first[0], second[0])
    assert np.linalg.norm(first[0]) <= LOAD['force_cap_n']+1e-12
    assert np.linalg.norm(first[1]) <= LOAD['torque_cap_nm']+1e-12


def test_declared_load_changes_only_target_and_never_mutates_default():
    original = dict(LOAD)
    profile = make_load_profile(.35)
    assert {k for k in LOAD if LOAD[k] != profile[k]} == {'spin_speed_rad_s'}
    profile['spin_speed_rad_s'] = .5
    assert profile == original and LOAD == original
    profile['spin_torque_cap_nm'] = 99.
    assert make_load_profile() == original and LOAD == original


@pytest.mark.parametrize('speed,target,integral,torque_z', [
    (.5, -.25, -.00001, -.01501), (.35, -.175, -.000007, -.010507)])
def test_declared_target_retains_original_ramp_pi_and_axial_load(speed, target, integral, torque_z):
    pose = np.array([.4, .2, .9, 0., 0., 0., 1.])
    args = (pose, np.zeros(6), [0, 0, .11], [.4, .2], .3039, .15, .0005, 0.)
    force, torque, actual_integral, actual_target = tool_load(*args, load_profile=make_load_profile(speed))
    assert np.allclose(force, [0., 0., -.0093705], rtol=0, atol=1e-12)
    assert np.allclose(torque, [0., 0., torque_z], rtol=0, atol=1e-12)
    assert actual_integral == pytest.approx(integral, abs=1e-15)
    assert actual_target == pytest.approx(target, abs=1e-15)
    # Calling the alternate condition cannot change a subsequent default call.
    default = tool_load(*args)
    assert default[3] == -.25 and default[2] == pytest.approx(-.00001)
    assert default[1][2] == pytest.approx(-.01501)
    assert LOAD['spin_speed_rad_s'] == .5


@pytest.mark.parametrize('speed', [0., -.35, .4, 1., float('nan'), float('inf')])
def test_undeclared_engineering_spin_speed_rejected(speed):
    with pytest.raises(ValueError, match='0.35 or 0.5'):
        make_load_profile(speed)


@pytest.mark.parametrize('option,expected', [([], .5), (['--spin-speed-rad-s', '.35'], .35)])
def test_spin_speed_cli_reaches_run_without_native_start(monkeypatch, option, expected):
    import sys
    import scripts.validation.validate_fastener_scene as probe
    calls = []
    monkeypatch.setattr(sys, 'argv', ['probe', '--release', '/unused', '--output', '/unused', *option])
    monkeypatch.setattr(probe, 'run', lambda args: calls.append(args.spin_speed_rad_s) or {'passed': True})
    probe.main()
    assert calls == [expected]


def lead_rows(scale=1., contacts=True):
    rows = []
    for theta in np.linspace(0, math.radians(40), 401):
        rows.append({'phase': 'loaded_turn', 'object_poses_xyzw': {'bolt':
            [0, 0, .9-scale*theta*.003/(2*np.pi), 0, 0, -np.sin(theta/2), np.cos(theta/2)]},
            'raw_contacts': ([{'actor0': 'tool', 'actor1': 'bolt', 'impulse_magnitude_sum_ns': .001},
                              {'actor0': 'bolt', 'actor1': 'fixture', 'impulse_magnitude_sum_ns': .001}]
                             if contacts else [])})
    return rows


@pytest.mark.parametrize('scale,contacts,passed', [(1., True, True), (.7, True, False), (1., False, False)])
def test_lead_requires_measured_advance_and_contact_chain(scale, contacts, passed):
    result = assess_loaded_lead(lead_rows(scale, contacts), {'screwdriver': 'tool', 'bolt': 'bolt', 'fixture': 'fixture'}, .0005)
    assert result['passed'] is passed


@pytest.mark.parametrize('expected,selected,readback,passed', [
    ('PGS', 'PGS', 'PGS', True), ('TGS', 'TGS', 'TGS', True),
    ('PGS', 'TGS', 'PGS', False), ('PGS', 'PGS', 'TGS', False),
    ('PGS', None, None, False)])
def test_release_owned_solver_requires_configuration_and_stage_readback(expected, selected, readback, passed):
    from scripts.validation.validate_fastener_scene import release_solver_configuration
    result = release_solver_configuration({'isaacsim': {'solver_type': expected}},
        {'physics_solver': {'profile_selection': selected, 'usd_readback': readback}})
    assert result['passed'] is passed
    assert result['diagnostic_solver_hook_used'] is False
