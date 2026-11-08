"""Timing changes original commands; measured contact/stability still gates assist."""
from copy import deepcopy
from unittest.mock import patch

import numpy as np
import pytest

from scripts.validation.laundry_closure_timing import (
    bind_closure_timing, closure_fractions, closure_hand_posture)
from scripts.validation.laundry_contact_pinch import interpolate_posture
from scripts.validation.laundry_engineering_motion import minimum_jerk
from scripts.validation.laundry_capture_options import validate_hand_options
from scripts.validation.laundry_closure_feed import closure_displacement, ClosureFeedProgress
from scripts.validation.laundry_captured_garment_fold import command_joint_targets, opening_capture_targets
from test_laundry_contact_stop_integration import setup
from test_laundry_corner_tangent import fixture as towel_fixture


PROFILE = 'thumb_early_index_late_v1'


def plan(side='right'):
    return {'postures': [{'hand_posture_rad': {
        side+'_thumb_IP': .48+.04*u,
        side+'_index_DIP': .48+.04*u,
        side+'_middle_PIP': .5+.1*u}} for u in np.linspace(0., 1., 61)],
        'closure_timing_profile': PROFILE, 'closure_lift_m': 0.,
        'closure_feed': {'schema': 'laundry_observed_edge_feed.v1',
                         'translation_world_m': [-.009, 0., 0.]}}


def test_original_and_explicit_synchronous_plans_keep_commands_exact():
    current = plan(); current.pop('closure_timing_profile')
    before = deepcopy(current)
    assert bind_closure_timing(current, {}) is current and current == before
    for amount in np.linspace(0., 1., 101):
        assert closure_hand_posture(current, amount) == interpolate_posture(current['postures'], amount)
    bind_closure_timing(current, {'closure_timing_profile': 'synchronous_v1'})
    for amount in np.linspace(0., 1., 101):
        assert closure_hand_posture(current, amount) == interpolate_posture(current['postures'], amount)


@pytest.mark.parametrize('side', ['right', 'left'])
def test_true_two_second_clock_and_endpoints_without_retiming_wrist_or_other_fingers(side):
    current = plan(side); before = deepcopy(current)
    for elapsed in np.linspace(0., 2., 101):
        amount = minimum_jerk(elapsed, 0., 2.)
        fractions = closure_fractions(amount, PROFILE)
        assert fractions['thumb'] == pytest.approx(minimum_jerk(elapsed, 0., 1.4), abs=1.e-11)
        assert fractions['index'] == pytest.approx(minimum_jerk(elapsed, .6, 2.), abs=1.e-11)
        hand = closure_hand_posture(current, amount)
        assert hand[side+'_thumb_IP'] == pytest.approx(.48+.04*fractions['thumb'])
        assert hand[side+'_index_DIP'] == pytest.approx(.48+.04*fractions['index'])
        assert hand[side+'_middle_PIP'] == interpolate_posture(current['postures'], amount)[side+'_middle_PIP']
        np.testing.assert_array_equal(closure_displacement(current, amount), [-.009*amount, 0., 0.])
    at_1p4 = closure_fractions(minimum_jerk(1.4, 0., 2.), PROFILE)
    assert at_1p4['thumb'] == pytest.approx(1.)
    assert 0. < at_1p4['index'] < 1.  # Third minimum_jerk parameter is end time, not duration.
    for amount in (0., 1.):
        assert closure_hand_posture(current, amount) == interpolate_posture(current['postures'], amount)
    assert current == before


@pytest.mark.parametrize('value', [None, False, 1, [], {}, 'faster', float('nan')])
def test_unknown_timing_is_rejected_for_later_unselected_hands_before_planning(value):
    options = {'phases': {'9': {'hands': {'left': {'closure_timing_profile': value}}}}}
    for towel in (False, True):
        with pytest.raises(ValueError, match='timing'):
            validate_hand_options(options, towel=towel)


def test_towel_binds_only_explicit_hand_profile_without_changing_old_geometry():
    from scripts.validation.laundry_captured_towel_tasks import prepare_phase
    task, scene, rest = towel_fixture('towel_folding'); q = rest+[.8, -.55, .902]
    with patch('scripts.validation.laundry_captured_towel_tasks.pinch_plan', side_effect=lambda *a, **k: {}):
        original = prepare_phase(None, task, scene, q, 0, {})
        changed = prepare_phase(None, task, scene, q, 0,
            {'phases': {'0': {'hands': {'right': {'closure_timing_profile': PROFILE}}}}})
    assert changed['edges'][1] == original['edges'][1]
    assert changed['edges'][0]['contact_plan'].pop('closure_timing_profile') == PROFILE
    assert changed == original


def test_garment_binds_same_explicit_profile_and_does_not_modify_other_contact_fields(monkeypatch):
    import scripts.validation.laundry_captured_garment_fold as garment
    monkeypatch.setattr(garment, 'phase_edges', lambda task: [[{'feature': 'edge'}]])
    pick = {'edge_world_m': [.8, -.7, .902], 'inward_world_xy': [-1., 0.],
            'pinch_tcp_xy_m': [.794, -.7]}
    monkeypatch.setattr(garment, 'edge_pick', lambda *args: deepcopy(pick))
    monkeypatch.setattr(garment, 'pinch_plan', lambda *args: {})
    task = {'id': 'shirt_folding', 'asset_metadata': {
        'cloths': [{'physical_thickness_m': .0016}],
        'design': {'fold_sequence_axis_hinge_sign_radius_m': [None]}}}
    scene = {'table': {'position_m': [0., 0., .8], 'half_extents_m': [1., 1., .1]}}
    original = garment.prepare_phase(None, task, scene, np.zeros((4, 3)), 0, {})
    changed = garment.prepare_phase(None, task, scene, np.zeros((4, 3)), 0,
        {'phases': {'0': {'hands': {'right': {'closure_timing_profile': PROFILE}}}}})
    assert changed['edges'][0]['contact_plan'].pop('closure_timing_profile') == PROFILE
    assert changed == original


def test_actual_submitted_mixed_targets_freeze_then_open_from_that_same_command(setup, monkeypatch):
    s = setup; current = plan(); before_state = deepcopy(s.state); before_q = s.q.copy()
    assist = s.make(contact_closure_stop=True, contact_closure_target_policy='hold_command_v2')
    values = dict(zip(s.env.joint_names, map(float, s.env._targets[0])))
    amount = minimum_jerk(1.0, 0., 2.)
    values.update(closure_hand_posture(current, amount))
    command_joint_targets(s.env, values, [assist], 'close')
    submitted = s.env._targets[0].copy()
    assert submitted[0] != submitted[1]
    assert assist.observe_closure('close', 7., s.state, s.q, submitted)
    assert not assist.active and assist.prim is None
    feed = ClosureFeedProgress(); edges = [{'side': 'right', 'contact_plan': current}]
    # A native stop freezes the last already-submitted wrist feed, not the
    # independently advanced thumb or index fraction.
    feed.last['right'] = amount
    for phase in ('close', 'capture_raise', 'closed_hold', 'fold'):
        request = {**values, **closure_hand_posture(current, .98)}
        command_joint_targets(s.env, request, [assist], phase)
        np.testing.assert_array_equal(s.env._targets[0, :2], submitted[:2])
        assert feed.fractions(edges, [assist], .98, 7.1)['right'] == amount
    enabled = []
    monkeypatch.setattr(assist, '_enable', lambda time_s, result, body, q: enabled.append(time_s))
    for t in (7., 7.02, 7.04, 7.06, 7.08):
        assist.update('closed_hold', t, s.state, s.q, s.env._targets[0]); assert not enabled
    assist.update('closed_hold', 7.1, s.state, s.q, s.env._targets[0]); assert enabled == [7.1]
    start = dict(zip(s.env.joint_names, map(float, s.env._targets[0])))
    phase = {'edges': edges}
    preview = assist.opening_postures(current['postures'], start)
    events = []; assist.active = True
    def release(time_s, reason):
        if assist.active: events.append('release'); assist.active = False
    monkeypatch.setattr(assist, 'release', release)
    def write(values, index):
        assert not assist.active; events.append('command')
    s.env._robot_view.set_dof_position_targets = write
    for index, u in enumerate(np.linspace(0., 1., 61)):
        request = opening_capture_targets([assist], phase, start, values, u, 10.+u)
        command_joint_targets(s.env, request, [assist], 'open')
        assert {n: request[n] for n in preview[index]['hand_posture_rad']} == preview[index]['hand_posture_rad']
    assert events[0] == 'release' and events.count('release') == 1
    for key in before_state: np.testing.assert_array_equal(s.state[key], before_state[key])
    np.testing.assert_array_equal(s.q, before_q)


def test_timed_commands_without_actual_contact_cannot_stop_or_create_assistance(setup):
    s = setup; s.mode['value'] = 'missing'; current = plan()
    assist = s.make(contact_closure_stop=True, contact_closure_target_policy='hold_command_v2')
    values = dict(zip(s.env.joint_names, map(float, s.env._targets[0])))
    for t in np.arange(0., 2.001, .02):
        hand = closure_hand_posture(current, minimum_jerk(float(t), 0., 2.))
        command_joint_targets(s.env, {**values, **hand}, [assist], 'close')
        assert not assist.observe_closure('close', float(t), s.state, s.q, s.env._targets[0])
        assist.update('closed_hold', float(t), s.state, s.q, s.env._targets[0])
    assert not assist.closure_stop.latched and not assist.active and not assist.gate.history
