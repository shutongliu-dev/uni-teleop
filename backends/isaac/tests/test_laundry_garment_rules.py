"""Synthetic referee fixtures, never native robot grasp qualification."""
from copy import deepcopy

import numpy as np
import pytest

from scenes.workrooms.laundry_room.evaluation import SequenceEvaluator
from scenes.workrooms.laundry_room.rules import (
    GARMENT_PROFILE, GARMENT_LEGACY_PROFILE, GARMENT_TASK_IDS, PRACTICAL_PROFILE, LEGACY_PROFILE,
    ROBOT_CONTROL, TASK_SCHEMA, bind_task_rules, default_profile, digest,
    public_rules, thresholds)
from test_laundry_robot_rules import robot_frame


def garment_task(task_id, profile=GARMENT_PROFILE):
    from scenes.workrooms.laundry_room.garments import candidate_metadata
    metadata = candidate_metadata(task_id)
    rules = public_rules(task_id, profile=profile, control_mode=ROBOT_CONTROL)
    return bind_task_rules({
        'schema_version': TASK_SCHEMA, 'id': task_id, 'task_id': task_id,
        'episode_seed': 0, 'asset_metadata': metadata, 'public_rules': rules,
        'evaluation': {'task_id': task_id, 'metadata': deepcopy(metadata),
            'physics_dt_s': .002, 'rule_profile': profile,
            'control_mode': ROBOT_CONTROL, 'public_rules_sha256': digest(rules),
            'thresholds': thresholds(task_id, profile)}})


def garment_goal(task):
    """Translate the authored folded references into an actually supported stack."""
    meta = task['asset_metadata']
    goal = {}
    for spec in meta['cloths']:
        v = np.array(spec['folded_reference_vertices_m'])
        v[:, :2] += np.asarray(meta['target_center_xy_m'])-(v[:, :2].min(axis=0)+v[:, :2].max(axis=0))/2
        v[:, 2] += meta['table_top_z_m']+spec['physical_thickness_m']/2-v[:, 2].min()
        goal[spec['id']] = v
    top, base = goal['towel'], goal['towel_base']
    spec = next(s for s in meta['cloths'] if s['id'] == 'towel')
    # Reference fixtures have level top support faces; this is a placement,
    # not a simulation or a claim that a hand can achieve it.
    top[:, 2] += base[:, 2].max()-top[:, 2].min()+spec['physical_thickness_m']
    return goal


def operated(task, goal):
    scorer = SequenceEvaluator(task)
    scorer.observe(robot_frame(task, 0))
    scorer.observe(robot_frame(task, 1, near=True))
    score = scorer.observe(robot_frame(task, 2, goal, near=True, x=.015))
    assert score['status'] != 'unknown', score
    return scorer


def released(task, scorer, goal, *, first=3, last=253, **kwargs):
    for step in range(first, last+1):
        score = scorer.observe(robot_frame(task, step, goal, **kwargs))
    return score


@pytest.mark.parametrize('task_id', GARMENT_TASK_IDS)
def test_garment_public_contract_has_its_own_profile_and_labels(task_id):
    rules = public_rules(task_id, control_mode=ROBOT_CONTROL)
    assert rules['rule_profile'] == default_profile(task_id) == GARMENT_PROFILE
    assert rules['referee']['thresholds'] == thresholds(task_id)
    assert '毛巾' not in str(rules)
    label = '上衣' if task_id == 'shirt_stacking' else '长裤'
    assert all(label in row['label'] for row in rules['referee']['object_properties'])
    for profile in (PRACTICAL_PROFILE, LEGACY_PROFILE):
        with pytest.raises(ValueError):
            public_rules(task_id, profile=profile, control_mode=ROBOT_CONTROL)


def test_support_uses_actual_triangles_and_the_highest_layer():
    # Two real triangles leave an uncovered gap even though vertices are near.
    vertices = np.array([[0., 0., 1.], [.04, 0., 1.04], [0., .04, 1.],
                         [0., 0., 1.1], [.04, 0., 1.14], [0., .04, 1.1]])
    heights = SequenceEvaluator._surface_top_at(
        np.array([[.01, .01], [.03, .03], [.041, 0.]]), vertices,
        np.array([[0, 1, 2], [3, 4, 5]]))
    assert heights[0] == pytest.approx(1.11)
    assert np.isneginf(heights[1:]).all()


@pytest.mark.parametrize('task_id', GARMENT_TASK_IDS)
def test_folded_garments_allow_small_placement_error_wrinkles_and_outlier_jitter(task_id):
    task = garment_task(task_id)
    goal = garment_goal(task)
    # Whole-garment yaw is free; modest translation and wrinkles remain valid.
    angle = .31
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    center = np.asarray(task['asset_metadata']['target_center_xy_m'])
    for v in goal.values():
        v[:, :2] = (v[:, :2]-center)@rotation.T+center
        v[:, 0] += .035
        v[:, 2] += .0015*(1+np.sin(v[:, 1]*9.))
    scorer = operated(task, goal)
    for step in range(3, 254):
        jittered = {name: v.copy() for name, v in goal.items()}
        jittered['towel'][0, 0] += .0001*(-1)**step
        frame = robot_frame(task, step, jittered)
        frame['cloths']['towel']['velocities_m_s'][0, 0] = .2
        result = scorer.observe(frame)
        if step == 252:
            assert not result['success']
            assert result['released_stable_s'] == pytest.approx(.498)
    assert result['success'], result
    assert not result['robot_manipulation_qualified']
    assert not result['physics_accepted']
    assert result['detail']['geometry']['cloth_support_fraction'] >= .85
    assert result['strict_diagnostics']['max_node_speed_m_s'] == pytest.approx(.2)
    assert not scorer.observe(robot_frame(task, 254, goal, near=True))['success']
    assert released(task, scorer, goal, first=255, last=505)['success']


@pytest.mark.parametrize('task_id', GARMENT_TASK_IDS)
@pytest.mark.parametrize('failure', ['unfolded', 'collapsed', 'stretched', 'floating',
    'misplaced', 'bottom_floating', 'intersections', 'held', 'passive'])
def test_garment_counterexamples_cannot_succeed(task_id, failure):
    task = garment_task(task_id)
    goal = garment_goal(task)
    if failure == 'unfolded':
        goal['towel'] = np.asarray(task['asset_metadata']['cloths'][0]['rest_vertices_m'])+[*task['asset_metadata']['target_center_xy_m'], .95]
    elif failure == 'collapsed':
        goal['towel'][:, 2] = goal['towel'][:, 2].mean()
    elif failure == 'stretched':
        v = goal['towel']
        v[:, :2] = (v[:, :2]-v[:, :2].mean(axis=0))*1.7+v[:, :2].mean(axis=0)
    elif failure == 'floating':
        goal['towel'][:, 2] += .06
    elif failure == 'misplaced':
        goal['towel'][:, 0] += .12
    elif failure == 'bottom_floating':
        for v in goal.values():
            v[:, 2] += .06
    scorer = SequenceEvaluator(task) if failure == 'passive' else operated(task, goal)
    if failure == 'passive':
        scorer.observe(robot_frame(task, 0))
        result = released(task, scorer, goal, first=1)
    elif failure == 'intersections':
        for step in range(3, 254):
            frame = robot_frame(task, step, goal)
            frame['physics']['intercloth_intersections'] = 1
            result = scorer.observe(frame)
    else:
        result = released(task, scorer, goal, separated=False if failure == 'held' else True)
    assert not result['success'], result


@pytest.mark.parametrize('task_id', GARMENT_TASK_IDS)
def test_garment_geometry_and_topology_evidence_are_frozen(task_id):
    task = garment_task(task_id)
    for field in ('folded_layer_pairs', 'support_bottom_vertex_indices', 'folded_reference_vertices_m'):
        malformed = deepcopy(task)
        del malformed['asset_metadata']['cloths'][0][field]
        malformed['evaluation']['metadata'] = deepcopy(malformed['asset_metadata'])
        with pytest.raises((KeyError, ValueError)):
            SequenceEvaluator(malformed)
    frame = robot_frame(task, 0)
    frame['cloths']['towel']['triangles'] = frame['cloths']['towel']['triangles'][:-1]
    assert SequenceEvaluator(task).observe(frame)['status'] == 'unknown'


@pytest.mark.parametrize('task_id', GARMENT_TASK_IDS)
def test_passive_garment_motion_cannot_gain_retroactive_hand_credit(task_id):
    task = garment_task(task_id)
    goal = garment_goal(task)
    scorer = SequenceEvaluator(task)
    scorer.observe(robot_frame(task, 0))
    scorer.observe(robot_frame(task, 1, goal))
    scorer.observe(robot_frame(task, 2, goal, near=True))
    scorer.observe(robot_frame(task, 3, goal, near=True, x=.015))
    result = released(task, scorer, goal, first=4, last=254)
    assert not result['success']
    assert not result['detail']['operation_displacement_seen']


@pytest.mark.parametrize('task_id', GARMENT_TASK_IDS)
@pytest.mark.parametrize('vertical_scale, accepted', [(.10, False), (.20, False), (.40, True)])
def test_garment_layer_compression_blocks_a_pancake_but_allows_settling(task_id, vertical_scale, accepted):
    # Immutable v1 retains the original reset-gap criterion.
    task = garment_task(task_id, GARMENT_LEGACY_PROFILE)
    goal = garment_goal(task)
    for v in goal.values():
        v[:, 2] = v[:, 2].min()+(v[:, 2]-v[:, 2].min())*vertical_scale
    thickness = task['asset_metadata']['cloths'][0]['physical_thickness_m']
    goal['towel'][:, 2] += goal['towel_base'][:, 2].max()+thickness-goal['towel'][:, 2].min()
    scorer = operated(task, goal)
    result = released(task, scorer, goal)
    assert result['success'] is accepted, result
    fraction = result['detail']['geometry']['towel']['ordered_layer_fraction']
    assert (fraction >= .85) is accepted


@pytest.mark.parametrize('task_id', GARMENT_TASK_IDS)
def test_public_nominal_masses_exactly_match_the_frozen_garment_metadata(task_id):
    task = garment_task(task_id)
    declared = {row['id']: row['nominal_mass_kg']
        for row in task['public_rules']['referee']['object_properties']}
    actual = {row['id']: row['mass_kg'] for row in task['asset_metadata']['cloths']}
    assert declared == actual
