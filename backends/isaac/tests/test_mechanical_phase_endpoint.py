"""Contact can limit the matching digit without excusing arm arrival errors."""
from copy import deepcopy

import pytest

from scripts.validation.mechanical_phase_endpoint import assess_phase_endpoint


OBJECT = '/World/Objects/bolt'
INDEX = '/World/Robot/right_index_DP'
THUMB = '/World/Robot/right_thumb_DP'
DT = .0005


def _probe(*, prefix=OBJECT, actor=INDEX, object_actor=OBJECT, impulse=.001):
    targets = {'fr3_joint2': 1., 'right_index_PIP': 1., 'right_thumb_IP': 1.}
    phase = {'id': 'pickup_close', 'joint_positions_rad': targets}
    if prefix is not None:
        phase['hand_contact_actor_prefix'] = prefix
    actual = {**targets, 'right_index_PIP': .6}
    rows = [{'phase': phase['id'], 'native_step': 801+index,
             'time_s': (801+index)*DT,
             'raw_contacts': [{'actor0': actor, 'actor1': object_actor,
                               'impulse_magnitude_sum_ns': impulse}]}
            for index in range(200)]
    return phase, actual, rows


def _assess(phase, actual, rows, **kwargs):
    return assess_phase_endpoint(phase, actual, rows, DT, native_step=1000,
                                 robot_actor_paths=(INDEX, THUMB), **kwargs)


def test_sustained_matching_digit_contact_waives_only_its_blocked_target():
    phase, actual, rows = _probe()
    before = deepcopy((phase, actual, rows))
    result = _assess(phase, actual, rows)
    assert result['passed'] is True
    assert result['raw_max_error_rad'] == pytest.approx(.4)
    assert result['unwaived_max_error_rad'] == 0.
    assert result['waived_joint_names'] == ['right_index_PIP']
    assert result['digit_receipts']['index']['positive_contact_substeps_per_block'] == [40]*5
    assert result['digit_receipts']['thumb']['passed'] is False
    assert result['native_contact_window']['complete'] is True
    assert result['task_success_credit'] == 0
    assert (phase, actual, rows) == before


def test_arm_error_is_never_waived_by_hand_contact():
    phase, actual, rows = _probe()
    actual['fr3_joint2'] += .03
    result = _assess(phase, actual, rows)
    assert result['passed'] is False
    assert result['unwaived_max_error_rad'] == pytest.approx(.03)
    assert result['waived_joint_names'] == ['right_index_PIP']


def test_noncontacting_digit_error_is_not_waived():
    phase, actual, rows = _probe()
    actual['right_thumb_IP'] -= .3
    result = _assess(phase, actual, rows)
    assert result['passed'] is False
    assert result['unwaived_max_error_rad'] == pytest.approx(.3)
    assert result['waived_joint_names'] == ['right_index_PIP']


@pytest.mark.parametrize('object_actor', [
    '/World/Objects/screwdriver', '/World/Objects/bolt_other',
    '/World/Fixtures/bolt_supply_rest', THUMB,
])
def test_other_objects_fixtures_and_self_contact_cannot_waive(object_actor):
    phase, actual, rows = _probe(object_actor=object_actor)
    result = _assess(phase, actual, rows)
    assert result['passed'] is False
    assert result['waived_joint_names'] == []


@pytest.mark.parametrize('actor', [THUMB, '/World/Fixtures/right_index_DP',
                                   '/World/Robot/right_index_unregistered'])
def test_wrong_digit_or_noninventory_actor_cannot_waive(actor):
    phase, actual, rows = _probe(actor=actor)
    result = _assess(phase, actual, rows)
    assert result['passed'] is False
    assert result['waived_joint_names'] == []


def test_transient_contact_must_not_replace_sustained_contact():
    phase, actual, rows = _probe()
    for row in rows[40:80]:
        row['raw_contacts'] = []
    result = _assess(phase, actual, rows)
    assert result['passed'] is False
    assert result['digit_receipts']['index']['positive_contact_substeps_per_block'] == [40, 0, 40, 40, 40]


def test_one_positive_substep_in_each_complete_control_block_is_sufficient():
    phase, actual, rows = _probe()
    for index, row in enumerate(rows):
        if index % 40:
            row['raw_contacts'] = []
        else:
            row['raw_contacts'][0]['actor0'], row['raw_contacts'][0]['actor1'] = OBJECT, INDEX
    result = _assess(phase, actual, rows)
    assert result['passed'] is True
    assert result['digit_receipts']['index']['positive_contact_substeps_per_block'] == [1]*5


@pytest.mark.parametrize('damage', [
    'missing_prefix', 'zero_impulse', 'missing_raw_contacts', 'missing_impulse',
    'short_window', 'missing_last_row', 'native_gap', 'clock_gap', 'other_phase',
])
def test_incomplete_or_absent_contact_evidence_keeps_all_joint_gate(damage):
    phase, actual, rows = _probe()
    if damage == 'missing_prefix':
        phase.pop('hand_contact_actor_prefix')
    elif damage == 'zero_impulse':
        for row in rows:
            row['raw_contacts'][0]['impulse_magnitude_sum_ns'] = 0.
    elif damage == 'missing_raw_contacts':
        rows[10].pop('raw_contacts')
    elif damage == 'missing_impulse':
        rows[10]['raw_contacts'][0].pop('impulse_magnitude_sum_ns')
    elif damage == 'short_window':
        rows.pop(0)
    elif damage == 'missing_last_row':
        rows.insert(0, {**deepcopy(rows[0]), 'native_step': 800, 'time_s': .4})
        rows.pop()
    elif damage == 'native_gap':
        rows[10]['native_step'] += 1
    elif damage == 'clock_gap':
        rows[10]['time_s'] += .001
    elif damage == 'other_phase':
        rows[10]['phase'] = 'previous_close'
    result = _assess(phase, actual, rows)
    assert result['passed'] is False
    assert result['waived_joint_names'] == []
    assert result['unwaived_max_error_rad'] == pytest.approx(.4)


def test_arrived_hand_and_arm_do_not_require_contact_evidence():
    phase, _, _ = _probe(prefix=None)
    result = _assess(phase, dict(phase['joint_positions_rad']), [])
    assert result['passed'] is True
    assert result['waived_joint_names'] == []
    assert result['native_contact_window']['complete'] is False
