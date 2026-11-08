"""Provisional aid has an explicit mask before confirmation, never transport credit."""
from copy import deepcopy
import hashlib
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.validation.laundry_contact_assist import CapturedPatchAssist, capture_constraint_vertices
from scripts.validation.laundry_contact_engagement import PROFILE, ContactEngagement
from scripts.validation.laundry_capture_settings import phase_capture_settings
from test_laundry_practical_capture import setup, settings, gate


@pytest.fixture
def assisted(setup, tmp_path, monkeypatch):
    b = setup
    env = SimpleNamespace(body_names=b.names, joint_names=b.joints,
        joint_limits={n: (0., 2.) for n in b.joints}, actuated_joint_names=b.joints,
        stage=object(), scene_info={'cloths': {'towel': {'path': '/World/towel'}}})
    scene = {'cloths': [{'id': 'towel', 'rest_vertices_m': b.rest.tolist(),
                        'triangles': b.tri.tolist(), 'isaacsim': {'cloth_thickness_m': .004}}]}
    a = CapturedPatchAssist(env, scene, 'right', b.export, tmp_path,
        **{**settings(), 'capture_profile': PROFILE}, allowed_material_vertex_ids=range(len(b.q)))
    a.open_tip_gap_m = .045
    a.closure_stop.receipt = {'finger_joint_targets_rad': {n: .51 for n in b.joints}}
    state = {'robot_body_q': b.body, 'joint_q': np.array([.5, .5]), 'joint_qd': np.zeros(2)}
    diagnostic = deepcopy(b.diag)
    def diagnose(*args, **kwargs):
        return {**deepcopy(diagnostic), 'state_identity': {
            'body_q_float64_sha256': hashlib.sha256(np.asarray(args[1], dtype='<f8').tobytes()).hexdigest(),
            'cloth_q_float64_sha256': hashlib.sha256(np.asarray(args[2], dtype='<f8').tobytes()).hexdigest()}}
    monkeypatch.setattr(a.geometry, 'diagnose', diagnose)
    attrs = {}
    class Attribute:
        def __init__(self, value): self.value = value
        def Get(self): return self.value
        def Set(self, value): self.value = value
    def attach(stage, path, cloth, anchor, ids, offsets):
        attrs.update({'omniphysics:attachmentEnabled': Attribute(True),
                      'omniphysics:vtxIndicesSrc0': Attribute(list(ids))})
        return SimpleNamespace(GetAttribute=lambda name: attrs[name])
    op = SimpleNamespace(Set=lambda value: None)
    xform = SimpleNamespace(AddTranslateOp=lambda _: op, AddOrientOp=lambda _: op)
    monkeypatch.setitem(sys.modules, 'pxr', SimpleNamespace(UsdGeom=SimpleNamespace(
        Xform=SimpleNamespace(Define=lambda *_: xform), XformOp=SimpleNamespace(PrecisionDouble=0))))
    physx = SimpleNamespace(get_physx_simulation_interface=lambda: SimpleNamespace(flush_changes=lambda: None))
    monkeypatch.setitem(sys.modules, 'omni', SimpleNamespace(physx=physx))
    monkeypatch.setitem(sys.modules, 'omni.physx', physx)
    monkeypatch.setattr('scripts.validation.laundry_cloth_interactions.attachment', attach)
    monkeypatch.setattr(a, '_pose', lambda value: setattr(a, 'anchor_pose', np.asarray(value).copy()))
    value = SimpleNamespace(a=a, b=b, state=state, diagnostic=diagnostic, attrs=attrs,
                            targets=np.array([.51, .51]))
    yield value
    a.close()


def tick(s, t, phase='closed_hold'):
    s.a.update(phase, t, s.state, s.b.q, s.targets)


def engage(s):
    tick(s, 0.)
    assert not s.a.active
    tick(s, .02)
    assert s.a.active and not s.a.transport_ready


def test_explicit_configuration_requires_original_v3_true_contact_and_held_stop():
    valid = {**settings(), 'capture_profile': PROFILE}
    assert phase_capture_settings(valid) == valid
    for change in ({'contact_closure_stop': False}, {'contact_closure_target_policy': 'reduce_preload_v1'},
                   {'contact_closure_geometry_profile': 'angled_adjacent_patch_v4'},
                   {'contact_normal_profile': 'facet_v1'}, {'contact_witness_profile': 'centroid_v1'},
                   {'attachment_material_mode': 'shared_edge'}, {'attachment_material_mode': 'captured_pivot_v1'}):
        with pytest.raises(ValueError): phase_capture_settings({**valid, **change})


def test_two_real_samples_enable_only_provisional_mask_then_fresh_full_window(assisted):
    s = assisted
    original = deepcopy(s.state), s.b.q.copy(), s.targets.copy()
    engage(s)
    event = s.a.events[0]
    assert event['event'] == 'enable'
    assert not event['captured_gate_evidence']['capture_candidate']
    assert event['contact_engagement']['engagement_candidate']
    assert not event['contact_engagement']['pre_assistance_100ms_passed']
    assert not s.a.gate.history
    np.testing.assert_array_equal(s.a.mask(len(s.b.q)), np.ones(len(s.b.q)))
    with pytest.raises(RuntimeError, match='transport reference'): s.a.trajectory_reference_ids()
    for t in [.04, .06, .08, .10, .12]:
        tick(s, t)
        assert not s.a.transport_ready
    tick(s, .14)
    assert s.a.transport_ready and s.a.engagement.confirmed
    assert s.a.events[-1]['event'] == 'engagement_confirmed'
    confirmed = s.a.events[-1]['contact_engagement']['full_v6_gate']
    assert confirmed['capture_candidate'] and confirmed['evidence']['first_time_s'] == .04
    assert s.a.trajectory_reference_ids() == s.a.ids
    tick(s, .15, 'open')
    assert not s.a.active and not s.a.transport_ready
    assert not s.attrs['omniphysics:attachmentEnabled'].Get()
    np.testing.assert_array_equal(s.a.mask(len(s.b.q)), np.zeros(len(s.b.q)))
    for n in s.state: np.testing.assert_array_equal(s.state[n], original[0][n])
    np.testing.assert_array_equal(s.b.q, original[1])
    np.testing.assert_array_equal(s.targets, original[2])


@pytest.mark.parametrize('failure', ['deep', 'changed_targets', 'transport', 'open', 'timeout'])
def test_provisional_failure_disables_native_flag_and_mask_before_error(assisted, failure):
    s = assisted
    engage(s)
    phase = 'closed_hold'
    if failure == 'deep': s.diagnostic['distal_group_severe_penetration'] = True
    if failure == 'changed_targets': s.targets[0] += 1.e-6
    if failure == 'transport': phase = 'fold'
    if failure == 'open': phase = 'open'
    if failure == 'timeout':
        s.diagnostic['multilayer_diagnostic']['plausible'] = False
        for t in np.arange(.04, .52, .02): tick(s, float(t))
    with pytest.raises(RuntimeError, match='Provisional contact assistance failed'):
        tick(s, .52 if failure == 'timeout' else .04, phase)
    assert not s.a.active and s.a.engagement.stage == 'failed'
    assert not s.attrs['omniphysics:attachmentEnabled'].Get()
    np.testing.assert_array_equal(s.a.mask(len(s.b.q)), np.zeros(len(s.b.q)))
    assert s.a.events[-1]['event'] == 'engagement_failed'


def test_every_native_tick_moves_only_anchor_and_cannot_change_held_targets(assisted):
    s = assisted
    engage(s)
    s.state['robot_body_q'] = s.state['robot_body_q'].copy()
    s.state['robot_body_q'][s.a.wrist_index, :3] += [.0001, 0, 0]
    tick(s, .0205)
    np.testing.assert_array_equal(s.a.anchor_pose, s.state['robot_body_q'][s.a.wrist_index])
    s.targets[1] += 1.e-7
    with pytest.raises(RuntimeError, match='issued_finger_targets_changed'):
        tick(s, .021)
    assert not s.a.active


@pytest.mark.parametrize('failure', ['single_sample', 'gap', 'unlatched', 'same_facing', 'open_hand',
                                    'far_material', 'deep', 'outside_allowed', 'moving_target'])
def test_original_negatives_never_engage(assisted, failure):
    s = assisted
    if failure == 'unlatched': s.a.closure_stop.receipt = None
    if failure == 'same_facing':
        s.diagnostic['multilayer_diagnostic']['paired_witnesses'][0]['opposing_normal_dot'] = .4
    if failure == 'open_hand': s.a.open_tip_gap_m = .012
    if failure == 'far_material': s.a.gate.rest *= 10
    if failure == 'deep': s.diagnostic['distal_group_severe_penetration'] = True
    if failure == 'outside_allowed': s.a.allowed_ids = {0}
    tick(s, 0.)
    if failure == 'moving_target': s.targets[0] += .0001
    if failure != 'single_sample': tick(s, .06 if failure == 'gap' else .02)
    assert not s.a.active and not s.a.events


def test_different_material_and_pre_assistance_history_cannot_confirm(setup):
    b = setup; g = gate(b); p = ContactEngagement()
    for t in [0., .02]:
        r = b.observe(g, t)
        action, receipt = p.observe(t, r, latched=True)
    assert action == 'engage'
    assert capture_constraint_vertices(receipt, b.tri, 'captured_patch') == r['vertex_ids']
    with pytest.raises(ValueError): capture_constraint_vertices(receipt, b.tri, 'shared_edge')
    for t in [.04, .06, .08, .1]: r = b.observe(g, t)
    assert r['capture_candidate']  # Deliberately retained pre-assistance history is not accepted.
    action, _ = p.observe(.1, r, latched=True)
    assert action == 'wait' and not p.confirmed
    changed = deepcopy(r); changed['vertex_ids'] = [20, 21, 22]
    action, _ = p.observe(.12, changed, latched=True)
    assert action == 'fail' and not p.confirmed


def test_fresh_geometry_exception_disables_existing_provisional_constraint(assisted, monkeypatch):
    s = assisted
    engage(s)
    def invalid(*args, **kwargs): raise ValueError('Invalid fresh state')
    monkeypatch.setattr(s.a.geometry, 'diagnose', invalid)
    with pytest.raises(ValueError, match='Invalid fresh state'): tick(s, .04)
    assert not s.a.active and not s.attrs['omniphysics:attachmentEnabled'].Get()
    assert s.a.engagement.stage == 'failed'


def test_complete_old_capture_cannot_be_relabeled_as_provisional_contact(setup):
    b = setup; g = gate(b)
    for t in np.arange(0., .101, .02): r = b.observe(g, float(t))
    assert r['capture_candidate']
    p = ContactEngagement()
    action, _ = p.observe(.1, r, latched=True)
    assert action == 'wait' and p.stage == 'waiting_contact'
