"""Task4 referee load paths from synthetic evidence, not native manipulation."""
from copy import deepcopy

import pytest

from scenes.workrooms.laboratory.evaluation import SequenceEvaluator
from scenes.workrooms.laboratory.native_evidence import rotation
from test_laboratory_review_contracts import setup, step
from test_laboratory_weight_feedback import balance_fixture


def test_stacked_scoop_load_needs_real_support_and_loaded_motion():
    _, meta, frame, observer = setup('solid_weighing')
    x, y, z = frame['objects']['scoop']['position_m']
    for name, height in [('pellet_00', .006), ('pellet_01', .014)]:
        frame['objects'][name]['position_m'] = [x, y, z+height]
    pairs = [('pellet_00', 'scoop'), ('pellet_01', 'pellet_00'), ('scoop', 'robot_hand')]
    result = step(frame, observer, meta, pairs)
    assert not result['detail']['scooped_ids']
    for name in ('scoop', 'pellet_00', 'pellet_01'):
        frame['objects'][name]['position_m'][0] += .019
    result = step(frame, observer, meta, pairs)
    assert not result['detail']['scooped_ids']
    for name in ('scoop', 'pellet_00', 'pellet_01'):
        frame['objects'][name]['position_m'][0] += .002
    result = step(frame, observer, meta, pairs)
    assert result['detail']['scooped_ids'] == ['pellet_00', 'pellet_01']
    result = step(frame, observer, meta, pairs+[('pellet_01', 'robot_hand')])
    assert result['detail']['scooped_ids'] == ['pellet_00']
    # Losing hand contact does not restore the old transport credit.
    result = step(frame, observer, meta, pairs)
    assert result['detail']['scooped_ids'] == ['pellet_00']


def test_stacked_cup_readout_keeps_sleeping_support_but_rejects_separation():
    _, meta, frame, observer, pairs = balance_fixture(12)
    frame['objects']['pellet_01']['position_m'] = frame['objects']['pellet_00']['position_m'].copy()
    frame['objects']['pellet_01']['position_m'][2] += .008
    pairs.remove(('pellet_01', 'weighing_cup'))
    pairs.append(('pellet_01', 'pellet_00'))
    for _ in range(700):
        result = step(frame, observer, meta, pairs)
    assert result['detail']['balance_display']['value_g'] == pytest.approx(6.)
    # Both levels have actual prior contact witnesses; sleeping contacts need
    # not be re-emitted each substep. No pose or material was invented.
    pairs.remove(('pellet_01', 'pellet_00'))
    pairs.remove(('pellet_00', 'weighing_cup'))
    for _ in range(120):
        result = step(frame, observer, meta, pairs)
    assert result['detail']['balance_display']['value_g'] == pytest.approx(6.)
    frame['objects']['pellet_01']['position_m'][2] += .003
    result = step(frame, observer, meta, pairs)
    assert not result['detail']['cup_grains_supported']
    assert result['detail']['balance_display']['text'] == '---'


@pytest.mark.parametrize('fault', ['void', 'cycle', 'outside_member', 'hand'])
def test_cup_load_has_no_credit_without_an_allowed_rooted_path(fault):
    _, meta, frame, observer, pairs = balance_fixture(12)
    names = ['pellet_00', 'pellet_01', 'pellet_02']
    pairs = [pair for pair in pairs if pair[0] not in names]
    if fault == 'cycle':
        pairs += list(zip(names, names[1:]+names[:1]))
    elif fault == 'outside_member':
        # A grain outside the cup is not an in-cup load-path shortcut.
        pairs += [(name, 'pellet_12') for name in names]
        pairs.append(('pellet_12', 'weighing_cup'))
    elif fault == 'hand':
        pairs += [('pellet_00', 'weighing_cup'), ('pellet_01', 'pellet_00'),
                  ('pellet_02', 'pellet_00'), ('pellet_00', 'robot_hand')]
    for _ in range(700):
        result = step(frame, observer, meta, pairs)
    assert not result['detail']['cup_grains_supported']
    assert result['detail']['balance_display']['text'] == '---'
    assert result['released_stable_s'] == 0.


@pytest.mark.parametrize('fault', ['void', 'cycle', 'hand'])
def test_scoop_cannot_transport_an_unsupported_or_hand_held_stack(fault):
    _, meta, frame, observer = setup('solid_weighing')
    x, y, z = frame['objects']['scoop']['position_m']
    names = ['pellet_00', 'pellet_01', 'pellet_02']
    for i, name in enumerate(names):
        frame['objects'][name]['position_m'] = [x+.01*(i-1), y, z+.006]
    pairs = []
    if fault == 'cycle':
        pairs = list(zip(names, names[1:]+names[:1]))
    elif fault == 'hand':
        pairs = [('pellet_00', 'scoop'), ('pellet_01', 'pellet_00'),
                 ('pellet_02', 'pellet_00'), ('pellet_00', 'robot_hand')]
    step(frame, observer, meta, pairs)
    for name in ['scoop', *names]:
        frame['objects'][name]['position_m'][0] += .03
    result = step(frame, observer, meta, pairs)
    assert not result['detail']['scooped_ids']


def completed_goal_fixture(count=12, spill_count=0):
    task, meta, frame, observer = setup('solid_weighing')
    evaluator = SequenceEvaluator(task)
    assert not evaluator.update({'time_s': 0., 'task_state': observer.on_observe()})['success']
    names = meta['task_parameters']['pellet_ids'][:count]
    scoop_reset = deepcopy(frame['objects']['scoop'])
    x, y, z = scoop_reset['position_m']
    scoop_rotation = rotation(scoop_reset['quaternion_wxyz'])
    for i, name in enumerate(names):
        local_position = [-.020+.010*(i//3), .010*(i%3-1), .006]
        frame['objects'][name]['position_m'] = (scoop_rotation @ local_position + [x, y, z]).tolist()
    carrying = [(name, 'scoop') for name in names]+[('scoop', 'robot_hand')]
    step(frame, observer, meta, carrying)
    for name in ['scoop', *names]:
        frame['objects'][name]['position_m'][0] += .03
    assert set(step(frame, observer, meta, carrying)['detail']['scooped_ids']) == set(names)
    frame['objects']['scoop'] = scoop_reset
    frame['objects']['weighing_cup']['position_m'] = [.71, 2.075, .9255]
    for i, name in enumerate(names):
        frame['objects'][name]['position_m'] = [.695+.01*(i//3), 2.075+.01*(i%3-1), .9325]
    pairs = [('weighing_cup', 'balance_platform'), ('scoop', 'scoop_rest')]
    pairs += [(name, 'weighing_cup') for name in names]
    remaining = meta['task_parameters']['pellet_ids'][count:]
    # Returning a supported spoon with leftover material is explicitly legal.
    leftover = remaining[0]
    frame['objects'][leftover]['position_m'] = [x, y, z+.006]
    pairs.append((leftover, 'scoop'))
    spilled = remaining[-spill_count:] if spill_count else []
    for i, name in enumerate(spilled):
        frame['objects'][name]['position_m'] = [.71, .78, .904+.008*i]
        pairs.append((name, spilled[i-1] if i else 'table'))
    pairs += [(name, 'source_tray') for name in remaining if name != leftover and name not in spilled]
    return meta, frame, observer, evaluator, pairs, spilled


@pytest.mark.parametrize('spill_count', [2, 3])
def test_external_stacked_spills_are_accounted_and_keep_the_two_grain_budget(spill_count):
    meta, frame, observer, evaluator, pairs, spilled = completed_goal_fixture(spill_count=spill_count)
    # Spill classification first needs its original 100 ms quiet interval;
    # only then can the unchanged 300 ms completion window begin.
    for _ in range(1000):
        result = step(frame, observer, meta, pairs)
        if frame['native_step'] % 40 == 0:
            score = evaluator.update({'time_s': frame['time_s'], 'task_state': result})
    assert not result['detail']['inventory_ids']['pending']
    assert set(result['detail']['inventory_ids']['spill']) == set(spilled)
    assert result['detail']['cumulative_spill_mass_g'] == pytest.approx(.5*spill_count)
    if spill_count == 2:
        assert score['success'] and not score['physical_failure']
    else:
        assert score['status'] == 'task_failure' and score['reason'] == 'spill_limit_exceeded'
    # Recovering the particles never refunds historical spills.
    for name in spilled:
        spec = next(row for row in meta['objects'] if row['id'] == name)
        frame['objects'][name]['position_m'] = spec['initial_position_m'].copy()
    pairs = [pair for pair in pairs if pair[0] not in spilled]+[(name, 'source_tray') for name in spilled]
    recovered = step(frame, observer, meta, pairs)
    assert recovered['detail']['cumulative_spill_mass_g'] == pytest.approx(.5*spill_count)


@pytest.mark.parametrize('count, accepted', [(10, False), (11, True), (13, True), (14, False)])
def test_mass_tolerance_and_returned_spoon_with_leftovers_are_unchanged(count, accepted):
    meta, frame, observer, evaluator, pairs, _ = completed_goal_fixture(count=count)
    for _ in range(720):
        result = step(frame, observer, meta, pairs)
        if frame['native_step'] % 40 == 0:
            score = evaluator.update({'time_s': frame['time_s'], 'task_state': result})
    assert score['success'] is accepted
    assert result['detail']['balance_display']['value_g'] == pytest.approx(.5*count)
    assert len(result['detail']['inventory_ids']['scoop']) == 1
