"""Pinned-side FK and whole-domain checks, never native grasp qualification."""
from copy import deepcopy
from dataclasses import replace
import hashlib
from pathlib import Path
import pickle
from types import SimpleNamespace

import numpy as np
import pytest

from bench.control.factory import public_robot_geometry
from bench.control.hand_posture import HandPosturePrior
from bench.control.hand_presets import (
    FINGERS, PINCH_ANCHORS, PINCH_EVIDENCE_SHA256,
    generate_hand_preset, hand_preset_catalog,
)
from bench.geometry import quaternion_matrix
from bench.paths import DATA_ROOT, load_json
from bench.protocol.hand_presets import HandPresetRequest, HandPresetResult, HandPresetValidation
from bench.protocol.schemas import GeometryObjective

ASSETS = Path(f'{DATA_ROOT}/assets')


@pytest.fixture(scope='module')
def dual():
    return public_robot_geometry({'assets_root': str(ASSETS), 'base_position_m': [0., 0., 0.],
        'hand_id': 'dual_fr3_sharpa_w01', 'assembly_version': 'isaacsim_6_1_v2'})


def preset(geometry, side='right', amount=1., name='pinch_prepose'):
    return generate_hand_preset(geometry, {'side': side, 'preset': name, 'amount': amount})


@pytest.mark.parametrize('amount', [-1e-9, 1.000000001, float('nan'), float('inf'), True, False, '0.5', None])
def test_amount_rejects_invalid_or_coerced_values(amount):
    with pytest.raises(ValueError):
        HandPresetRequest(preset='pinch_prepose', side='left', amount=amount)


@pytest.mark.parametrize('update', [{'side': 'both'}, {'preset': 'envelop_prepose'},
                                   {'preset': 'grasp'}, {'object_id': 'target'}, {'force_n': 3.}])
def test_requests_do_not_invent_unreviewed_or_object_specific_presets(update):
    with pytest.raises(ValueError):
        HandPresetRequest.model_validate({'preset': 'open', 'side': 'right', **update})


def test_catalog_distinguishes_single_dual_and_unavailable_geometry(dual):
    catalog = hand_preset_catalog(dual)
    assert catalog.available and catalog.sides == ['left', 'right']
    assert [entry.preset for entry in catalog.presets] == ['open', 'pinch_prepose']
    assert 'envelop_prepose' in catalog.unavailable_presets
    single = public_robot_geometry({'assets_root': str(ASSETS), 'base_position_m': [0., 0., .77],
                                   'assembly_version': 'isaacsim_6_1_v2'})
    assert hand_preset_catalog(single).sides == ['right']
    with pytest.raises(ValueError, match='not present'):
        preset(single, side='left')
    for unavailable in (None, SimpleNamespace(embodiment=SimpleNamespace(sharpa_joint_semantics=False))):
        catalog = hand_preset_catalog(unavailable)
        assert not catalog.available and catalog.reason and not catalog.presets
        with pytest.raises(ValueError, match='not present'):
            preset(unavailable)


@pytest.mark.parametrize('side', ['left', 'right'])
def test_original_anchor_parameters_match_the_hashed_engineering_record(side):
    # This optional artifact check adds provenance, never a runtime dependency.
    path = Path(f'{DATA_ROOT}/datasets/fr3_sharpa_bench/development/'
                'card_room_reserve_access_20261017/pickup02/inputs/grasp_plan.json')
    if not path.exists():
        pytest.skip('Historical engineering source record is not installed')
    assert hashlib.sha256(path.read_bytes()).hexdigest() == PINCH_EVIDENCE_SHA256
    source = load_json(path)['sides'][side]
    assert PINCH_ANCHORS[side] == source['open_hand_posture_rad']
    assert PINCH_ANCHORS[side] != source['closed_hand_posture_rad']


def tip_speed_bounds(hand, q_slope, tips):
    """Robot-independent bound on |dp_tip/damount| for a revolute chain.

    Each joint's lever arm is at most the sum of downstream fixed translations.
    Rotation preserves length, so this holds throughout the whole interval.
    """
    by_child = {j.child: j for j in hand.joints}
    result = {}
    for finger, tip in tips.items():
        radius = 0.
        speed = 0.
        while tip in by_child:
            joint = by_child[tip]
            assert joint.kind in ('fixed', 'revolute')
            speed += abs(q_slope.get(joint.name, 0.)) * radius
            radius += float(np.linalg.norm(joint.origin[:3, 3]))
            tip = joint.parent
        result[finger] = speed
    return result


@pytest.mark.parametrize('side', ['left', 'right'])
def test_entire_amount_domain_limits_coupling_and_fk_separation(dual, side):
    model = dual.sides[side]
    start, end = preset(dual, side, 0.), preset(dual, side, 1.)
    assert all(q == 0. for q in start.hand_posture_rad.values())
    assert start.hand_posture_rad == preset(dual, side, .73, 'open').hand_posture_rad
    # Every requested joint is affine in amount. Its interval extrema, and the
    # extrema of each linear DIP-PIP inequality, are exactly at these endpoints.
    for joint in model.hand.active:
        endpoints = [start.hand_posture_rad[joint.name], end.hand_posture_rad[joint.name]]
        assert joint.lower <= min(endpoints) <= max(endpoints) <= joint.upper
    for finger, (pip, dip) in model.hand_posture_prior.pairs.items():
        assert max(row.hand_posture_rad[dip] - row.hand_posture_rad[pip]
                   for row in (start, end)) <= .5

    speed = tip_speed_bounds(model.hand, end.hand_posture_rad, model.fingertip_links)
    samples = 257
    bounds = []
    before = pickle.dumps(dual)
    for amount in np.linspace(0., 1., samples):
        row = preset(dual, side, float(amount))
        q = row.hand_posture_rad
        assert set(q) == set(model.hand.names)
        assert all(q[name] == amount * end.hand_posture_rad[name] for name in q)
        fk = model.hand.forward(q)
        for finger, tip in row.fingertip_poses.items():
            T = fk[model.fingertip_links[finger]]
            np.testing.assert_allclose(tip.position_m, T[:3, 3], atol=1e-12)
            np.testing.assert_allclose(quaternion_matrix(tip.quaternion_wxyz), T[:3, :3], atol=1e-12)
            assert np.isfinite(T).all() and np.linalg.det(T[:3, :3]) == pytest.approx(1., abs=1e-12)
        for i, finger in enumerate(FINGERS):
            for other in FINGERS[:i]:
                distance = np.linalg.norm(np.asarray(row.fingertip_poses[finger].position_m)
                                          - row.fingertip_poses[other].position_m)
                # Distance is Lipschitz with the sum of both tip speed bounds.
                # Any amount lies <= half a sample spacing from a grid point.
                bounds.append(distance - (speed[finger] + speed[other]) / (2 * (samples - 1)))
    assert min(bounds) >= row.validation.all_amount_virtual_tip_separation_lower_bound_m
    assert pickle.dumps(dual) == before
    # Finite rigid URDF transforms also prove FK exists over bounded q; this
    # distance certificate concerns virtual origins, NOT collision geometry.
    assert not row.validation.collision_checked and not row.validation.native_grasp_qualified


# Independent recorded FK checkpoint values, rather than left-name substitution.
GOLDEN_PINCH = {
    'left': {'thumb': [.06998965374508873, -.08199530218269518, .14998617813819626],
             'index': [.0699968818853063, -.01800335287952856, .15000072902793066]},
    'right': {'thumb': [.06999049551193727, .08199519508819239, .14998575781588985],
              'index': [.06999657183929424, .018003116344385228, .15000098735303755]},
}


def test_each_actual_side_matches_fk_checkpoints_and_reflected_orientation(dual):
    for side in ('left', 'right'):
        row = preset(dual, side)
        for finger, expected in GOLDEN_PINCH[side].items():
            np.testing.assert_allclose(row.fingertip_poses[finger].position_m, expected, atol=1e-12, rtol=0)
        assert .06398 < row.thumb_index_distance_m < .06401
    mirror = np.diag([1., -1., 1.])
    for amount in np.linspace(0., 1., 33):
        left, right = (preset(dual, side, float(amount)) for side in ('left', 'right'))
        for finger in FINGERS:
            a, b = left.fingertip_poses[finger], right.fingertip_poses[finger]
            # Source URDFs round nominal pi/2 rotations to 4 decimals; they are
            # independently evaluated, approximately reflected, not renamed.
            np.testing.assert_allclose(a.position_m, mirror @ b.position_m, atol=2e-6, rtol=0)
            np.testing.assert_allclose(quaternion_matrix(a.quaternion_wxyz),
                mirror @ quaternion_matrix(b.quaternion_wxyz) @ mirror, atol=4e-5, rtol=0)


@pytest.mark.parametrize('side', ['left', 'right'])
def test_targets_are_accepted_by_ordinary_hand_posture_objective_without_arm_changes(dual, side):
    model = dual.sides[side]
    row = preset(dual, side, .75)
    objective = GeometryObjective(hand_posture_rad=row.hand_posture_rad, fingertip_frame_id=row.frame_id)
    measured = {j.name: (j.lower + j.upper) / 2 for j in model.joints}
    goal, _ = model.solve(measured, hand_posture_rad=objective.hand_posture_rad)
    assert {n: goal[n] for n in model.hand.names} == row.hand_posture_rad
    assert {n: goal[n] for n in model.arm.names} == {n: measured[n] for n in model.arm.names}
    assert HandPresetResult.model_validate_json(row.model_dump_json()) == row


@pytest.mark.parametrize('tamper', ['axis', 'origin', 'frame', 'prior', 'limit', 'missing_asset'])
def test_changed_axes_origins_frame_or_coupling_fail_closed(dual, tamper, tmp_path):
    model = deepcopy(dual.sides['left'])
    if tamper == 'axis':
        model.hand.active[0].axis[:] *= -1
    elif tamper == 'origin':
        model.hand.active[0].origin[0, 3] += .01
    elif tamper == 'frame':
        model.frame_id = 'right_hand_flange'
    elif tamper == 'limit':
        model.hand.active[0] = replace(model.hand.active[0], upper=3.)
    elif tamper == 'missing_asset':
        model.hand_path = tmp_path/'not_installed.urdf'
    else:
        model.hand_posture_prior = HandPosturePrior(maximum_dip_minus_pip_rad=.6, side='left')
    assert not hand_preset_catalog(model).available
    with pytest.raises(ValueError):
        preset(model, 'left')


def test_public_function_revalidates_constructed_request_instances(dual):
    invalid = HandPresetRequest.model_construct(preset='unreviewed', side='right', amount=.5)
    with pytest.raises(ValueError):
        generate_hand_preset(dual, invalid)


def test_anchor_changes_cannot_silently_reuse_frozen_version(dual, monkeypatch):
    monkeypatch.setitem(PINCH_ANCHORS['left'], 'left_thumb_IP', .3)
    with pytest.raises(ValueError, match='parameters changed'):
        preset(dual, 'left')


@pytest.mark.parametrize('value', [True, 0, 'false'])
def test_result_cannot_claim_native_or_collision_qualification(value):
    with pytest.raises(ValueError):
        HandPresetValidation(native_grasp_qualified=value)
    with pytest.raises(ValueError):
        HandPresetValidation(collision_checked=value)
