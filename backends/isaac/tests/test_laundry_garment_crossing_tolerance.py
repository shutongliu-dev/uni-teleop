"""Dimension-based cloth contact tolerances; no native policy success claim."""
from copy import deepcopy

import numpy as np
import pytest

from bench.protocol.task_referee import PublicContract
from scenes.workrooms.laundry_room.garment_crossings import crossing_measure
from scenes.workrooms.laundry_room.rules import (
    GARMENT_FOLDING_LEGACY_PROFILE, public_rules, thresholds, bind_task_rules, digest,
)
from test_laundry_single_garment_folding import folding_task, operate, finish
from test_laundry_robot_rules import robot_frame


def small_measure():
    # Sizes resemble the measured shirt wrinkle; synthetic rule evidence only.
    return {'schema': 'laundry_garment_crossing_measure.v1', 'complete': True,
        'pair_count': 11, 'affected_triangle_count': 13,
        'maximum_pair_length_m': .0087, 'total_intersection_length_m': .0522,
        'maximum_component_diameter_m': .0105, 'affected_rest_area_fraction': .00116,
        'coplanar_overlap_area_sum_m2': 0.}


def crossing_frame(frame, measure):
    frame['physics']['self_intersections'] = measure['pair_count']
    frame['physics']['self_crossing_measure'] = deepcopy(measure)


def test_small_dimensioned_crossings_can_release_but_raw_counts_remain():
    task, goal = folding_task()
    result = finish(task, operate(task, goal), goal,
                    mutate=lambda frame: crossing_frame(frame, small_measure()))
    assert result['success'], result
    assert result['detail']['self_intersection_pairs'] == 11
    assert result['detail']['crossing_size_acceptable']
    assert result['strict_diagnostics']['violating_frames'] > 0
    PublicContract.model_validate(task['public_rules']['referee'])


@pytest.mark.parametrize('changes', [
    {'pair_count': 1, 'maximum_pair_length_m': .02, 'total_intersection_length_m': .02},
    {'pair_count': 20, 'total_intersection_length_m': .16},
    {'maximum_component_diameter_m': .04},
    {'affected_rest_area_fraction': .004},
    {'coplanar_overlap_area_sum_m2': .000002},
])
def test_long_distributed_spreading_or_coplanar_defects_cannot_pass(changes):
    task, goal = folding_task()
    evaluator = operate(task, goal)
    measure = {**small_measure(), **changes}
    result = finish(task, evaluator, goal, mutate=lambda frame: crossing_frame(frame, measure))
    assert not result['success'] and result['reason'] == 'contact_recovery_required', result
    # A genuinely recovered final half-second can pass without reset.
    for step in range(254, 505):
        result = evaluator.observe(robot_frame(task, step, goal))
    assert result['success'], result


def test_large_material_crossing_stops_and_incomplete_sizes_remain_unknown():
    task, goal = folding_task()
    for change, expected in (({'affected_rest_area_fraction': .03}, 'safety_stop'),
                             ({'complete': False}, 'unknown')):
        evaluator = operate(task, goal)
        frame = robot_frame(task, 3, goal)
        crossing_frame(frame, {**small_measure(), **change})
        result = evaluator.observe(frame)
        assert result['status'] == expected and not result['success']


def test_zero_pair_count_does_not_hide_malformed_or_truncated_size_evidence():
    task, goal = folding_task()
    for measure in ('not-an-audit', {'complete': False, 'pair_count': 0}):
        evaluator = operate(task, goal)
        frame = robot_frame(task, 3, goal)
        frame['physics']['self_crossing_measure'] = measure
        result = evaluator.observe(frame)
        assert result['status'] == 'unknown' and not result['success']


def test_v2_allows_reasonable_table_placement_without_relaxing_shape():
    task, goal = folding_task()
    goal['towel'][:, 0] += .10
    assert finish(task, operate(task, goal), goal)['success']
    old = deepcopy(task)
    old['public_rules'] = public_rules(old['id'], profile=GARMENT_FOLDING_LEGACY_PROFILE,
                                      control_mode=old['evaluation']['control_mode'])
    old['evaluation'].update(rule_profile=GARMENT_FOLDING_LEGACY_PROFILE,
        thresholds=thresholds(old['id'], GARMENT_FOLDING_LEGACY_PROFILE),
        public_rules_sha256=digest(old['public_rules']))
    old = bind_task_rules(old)
    assert not finish(old, operate(old, goal), goal)['success']
    assert task['evaluation']['thresholds']['garment_shape_rms_m'] == old['evaluation']['thresholds']['garment_shape_rms_m']


@pytest.mark.parametrize('task_id,mode,expected', [
    ('shirt_folding', 'engineering_local_grasp', '7cb97594a6802ca82f6233a75099535946c7c3fddcf0e093cd489a0cb2eeb843'),
    ('shirt_folding', 'robot_joint_targets', 'eca7a7fe5dddbc06c2f0e3b9af415d78a9cf54baf4491b7ec172ab4e0e8eed4a'),
    ('trousers_folding', 'engineering_local_grasp', 'b9a27a9078c166dc50b514bdcd12b9ec8bb558cbc5899756100d0093078f6460'),
    ('trousers_folding', 'robot_joint_targets', '8eda685464aef21ed916beaac848f92fd3b07fc558ad42804df22bf89d312ed5'),
])
def test_original_folding_v1_public_contract_is_unchanged(task_id, mode, expected):
    assert digest(public_rules(task_id, profile=GARMENT_FOLDING_LEGACY_PROFILE, control_mode=mode)) == expected


def test_exact_crossing_length_coplanar_area_and_structural_neighbors():
    q = np.array([[-.01, -.01, 0], [.01, -.01, 0], [0, .01, 0],
                  [0, -.005, -.005], [0, -.005, .005], [0, .005, 0]])
    tri = np.arange(6).reshape(2, 3)
    result = crossing_measure(q, tri, q, [[0, 1]], pair_count=1)
    assert result['complete']
    assert result['maximum_pair_length_m'] == pytest.approx(.01)
    assert result['coplanar_overlap_area_sum_m2'] == 0
    assert result['affected_rest_area_fraction'] == 1.
    # Shared material boundaries do not excuse positive coplanar overlap.
    q = np.array([[0., 0, 0], [1., 0, 0], [0., 1, 0], [.5, .5, 0]])
    result = crossing_measure(q, [[0, 1, 2], [0, 1, 3]], q, [[0, 1]], pair_count=1)
    assert result['complete'] and result['structural_neighbor_pair_count'] == 1
    assert result['coplanar_overlap_area_sum_m2'] == pytest.approx(.25)
    assert not crossing_measure(q, [[0, 1, 2], [0, 1, 3]], q, [[0, 1]], pair_count=2)['complete']
