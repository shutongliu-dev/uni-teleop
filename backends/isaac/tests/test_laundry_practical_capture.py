"""Bounded loaded settling tests; synthetic witnesses do not qualify a grasp."""
from copy import deepcopy
import hashlib
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from scripts.validation.laundry_capture_gate import CaptureGate
from scripts.validation.laundry_capture_settings import phase_capture_settings
from scripts.validation.laundry_contact_assist import CapturedPatchAssist
from scripts.validation.laundry_contact_stability import PROFILE, digest
from scripts.validation.laundry_pinch_geometry import PinchGeometry
from test_laundry_capture_gate import CaptureGateTests
from test_laundry_pinch_geometry import fixture as geometry_fixture


@pytest.fixture
def setup():
    base = CaptureGateTests()
    base.setUp()
    export, names, poses, _, _ = geometry_fixture()
    shells = deepcopy(export['colliders'])
    for row in shells:
        row['body_name'] = row['body_name'].replace('elastomer', 'DP')
        row['path'] = row['path'].replace('elastomer', 'DP')
        names.append(row['body_name'])
    export['colliders'].extend(shells)
    base.names.extend(names)
    base.body = np.vstack([base.body, [[0, 0, .9, 0, 0, 0, 1]]*4])
    geometry = PinchGeometry(export, base.names)
    base.diag.update(cooked_export_sha256=geometry.export_sha256,
        contact_normal_profile='distance_gradient_v1', contact_witness_profile='surface_regions_v2',
        contact_body_names={'thumb': 'right_thumb_elastomer', 'index': 'right_index_DP'})
    base.reference = geometry.contact_motion_reference('right')
    base.geometry, base.export = geometry, export
    return base


def gate(base, **kwargs):
    return CaptureGate('right', base.tri, base.rest, base.names, base.joints,
                       profile=PROFILE, contact_motion_reference=base.reference,
                       contact_normal_profile='distance_gradient_v1',
                       contact_witness_profile='surface_regions_v2', **kwargs)


def test_loaded_convergence_keeps_full_dwell_and_reports_old_static_failure(setup):
    current = gate(setup)
    original = CaptureGate('right', setup.tri, setup.rest, setup.names, setup.joints,
                           profile='adjacent_patch_v3', contact_normal_profile='distance_gradient_v1',
                           contact_witness_profile='surface_regions_v2')
    for i, t in enumerate(np.arange(0, .101, .02)):
        body = setup.body.copy()
        body[setup.names.index('right_index_DP'), 1] += .001*i/5
        kw = dict(body=body, joint=[.5+.018*i/5, .5], speed=[.64, 0.], target=[.53, .51])
        result = setup.observe(current, float(t), **kw)
        old = setup.observe(original, float(t), **kw)
        if i < 5:
            assert not result['capture_candidate']
    assert result['capture_candidate'] and not old['capture_candidate']
    assert old['reasons'] == ['fingers_not_stopped']
    assert result['schema'] == 'laundry_capture_candidate.v6'
    assert not result['evidence']['legacy_joint_static_conditions_passed']
    assert result['evidence']['closed_finger_commands_held_bitexact']
    assert result['evidence']['original_cooked_surface_motion'][
        'maximum_contact_surface_hand_span_m'] == pytest.approx(.001)
    assert result['evidence']['maximum_finger_speed_rad_s'] == .64
    assert not result['friction_grasp_qualified'] and not result['contact_force_measured']


@pytest.mark.parametrize('failure', [
    'changing_command', 'surface_translation', 'surface_rotation', 'opposite_excursions',
    'angular_range', 'material_drift', 'target_error', 'same_facing', 'open_hand',
    'far_material', 'deep', 'missing_witness',
])
def test_practical_profile_rejects_motion_and_original_geometric_negatives(setup, failure):
    if failure == 'far_material':
        setup.rest *= 10
    current = gate(setup)
    for i, t in enumerate(np.arange(0, .101, .02)):
        body = setup.body.copy()
        diag = deepcopy(setup.diag)
        kw = dict(body=body, diag=diag)
        index = setup.names.index('right_index_DP')
        if failure == 'changing_command': kw['target'] = [.51+1.e-6*i, .51]
        if failure == 'surface_translation': body[index, 1] += .0021*i/5
        if failure == 'surface_rotation':
            body[index, 3:] = Rotation.from_euler('z', .4*i/5).as_quat()
        if failure == 'opposite_excursions':
            body[index, 1] += 0. if i == 0 else .0011*(-1 if i < 3 else 1)
        if failure == 'angular_range': kw['joint'] = [.5+np.deg2rad(2.1)*i/5, .5]
        if failure == 'material_drift': kw['cloth'] = setup.q+[0, .0021*i/5, 0]
        if failure == 'target_error': kw['target'] = [.541, .51]
        if failure == 'same_facing':
            diag['multilayer_diagnostic']['paired_witnesses'][0]['opposing_normal_dot'] = .36
        if failure == 'open_hand': kw['open_gap'] = .012
        if failure == 'deep': diag['distal_group_severe_penetration'] = True
        if failure == 'missing_witness': diag['multilayer_diagnostic']['plausible'] = False
        result = setup.observe(current, float(t), **kw)
    assert not result['capture_candidate'], failure


def test_whole_original_cooked_surface_moves_even_if_tip_anchor_does_not(setup):
    current = gate(setup)
    original_tips = setup.body[1:3].copy()
    for i, t in enumerate(np.arange(0, .101, .02)):
        body = setup.body.copy()
        body[setup.names.index('right_index_DP'), 1] += .003*i/5
        result = setup.observe(current, float(t), body=body)
        np.testing.assert_array_equal(body[1:3], original_tips)
    assert 'contact_surfaces_moving_in_actual_hand_frame' in result['reasons']


def test_reference_is_complete_source_bound_and_does_not_change_old_diagnostic(setup):
    before = deepcopy(setup.geometry.pads)
    reference = setup.geometry.contact_motion_reference('right')
    assert set(reference['surfaces']) == {
        f'right_{finger}_{part}' for finger in ('thumb', 'index') for part in ('DP', 'elastomer')}
    assert reference['body_names_sha256'] == digest(setup.names)
    for side in before:
        for finger in before[side]:
            np.testing.assert_array_equal(before[side][finger]['outer'], setup.geometry.pads[side][finger]['outer'])
    for change in ('missing', 'wrong_source', 'degenerate'):
        changed = deepcopy(reference)
        if change == 'missing': changed['surfaces'].pop('right_index_DP')
        if change == 'wrong_source': changed['body_names_sha256'] = 'wrong'
        if change == 'degenerate': changed['surfaces']['right_index_DP'] = [[0, 0, 0]]*4
        with pytest.raises(ValueError):
            CaptureGate('right', setup.tri, setup.rest, setup.names, setup.joints,
                        profile=PROFILE, contact_motion_reference=changed,
                        contact_normal_profile='distance_gradient_v1',
                        contact_witness_profile='surface_regions_v2')
    current = gate(setup)
    diag = deepcopy(setup.diag)
    diag['cooked_export_sha256'] = '0'*64
    with pytest.raises(ValueError, match='reference differ'):
        setup.observe(current, 0., diag=diag)


def settings():
    return dict(capture_profile=PROFILE, contact_closure_stop=True,
                contact_closure_target_policy='hold_command_v2',
                contact_closure_geometry_profile='adjacent_patch_v3',
                contact_normal_profile='distance_gradient_v1', contact_witness_profile='surface_regions_v2')


def test_profile_requires_actual_stop_original_geometry_and_unchanged_command():
    assert phase_capture_settings(settings()) == settings()
    for change in ({'contact_closure_stop': False, 'contact_closure_target_policy': 'reduce_preload_v1'},
                   {'contact_closure_target_policy': 'reduce_preload_v1'},
                   {'contact_closure_geometry_profile': 'angled_adjacent_patch_v4'},
                   {'attachment_material_mode': 'captured_pivot_v1'}):
        with pytest.raises(ValueError): phase_capture_settings({**settings(), **change})


@pytest.mark.parametrize('latched', [False, True])
def test_real_assist_wiring_requires_real_latch_and_full_stable_window(setup, tmp_path, latched):
    env = SimpleNamespace(body_names=setup.names, joint_names=setup.joints,
                          joint_limits={name: (0., 2.) for name in setup.joints},
                          actuated_joint_names=setup.joints)
    scene = {'cloths': [{'id': 'towel', 'rest_vertices_m': setup.rest.tolist(),
                        'triangles': setup.tri.tolist(), 'isaacsim': {'cloth_thickness_m': .004}}]}
    assist = CapturedPatchAssist(env, scene, 'right', setup.export, tmp_path, **settings())
    assert assist.gate.contact_motion.export_sha256 == assist.geometry.export_sha256
    assist.open_tip_gap_m = .045
    if latched:
        assist.closure_stop.receipt = {'finger_joint_targets_rad': {name: .51 for name in setup.joints}}
    assist.geometry.diagnose = lambda *a, **k: {
        **deepcopy(setup.diag), 'state_identity': {
            'body_q_float64_sha256': hashlib.sha256(setup.body.astype('<f8').tobytes()).hexdigest(),
            'cloth_q_float64_sha256': hashlib.sha256(setup.q.astype('<f8').tobytes()).hexdigest()}}
    enabled = []
    assist._enable = lambda *args: enabled.append(args)
    state = {'robot_body_q': setup.body, 'joint_q': np.array([.5, .5]),
             'joint_qd': np.zeros(2)}
    try:
        for t in np.arange(0, .101, .02):
            assist.update('closed_hold', float(t), state, setup.q, np.array([.51, .51]))
        assert bool(enabled) == latched
        if not latched:
            assert not assist.gate.history
        else:
            assert len(enabled) == 1
            assert enabled[0][0] == pytest.approx(.1)
            assert enabled[0][1]['capture_candidate']
    finally:
        assist.diagnostics.close()
        assist.closure_diagnostics.close()
