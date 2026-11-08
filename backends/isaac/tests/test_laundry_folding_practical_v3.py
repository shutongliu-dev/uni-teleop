"""Versioned tolerance for small free-corner curls; CPU contract fixtures only."""
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from bench.paths import DATA_ROOT
from scenes.workrooms.laundry_room.composition import build_task_inputs
from scenes.workrooms.laundry_room.evaluation import SequenceEvaluator
from scenes.workrooms.laundry_room.rules import (
    FOLDING_PRACTICAL_PROFILE, PRACTICAL_PROFILE, ROBOT_CONTROL, thresholds,
)
from test_laundry_practical_rules import goal_vertices, sample, start, finish


def task(profile=FOLDING_PRACTICAL_PROFILE, **kwargs):
    return build_task_inputs('towel_folding', rule_profile=profile,
        physics_options={'physics_dt_s': .002}, **kwargs)['task']


def fold_geometry(spec, vertices):
    evaluator = SequenceEvaluator(spec)
    return evaluator._fold('towel', vertices,
        np.asarray(spec['asset_metadata']['target_center_xy_m']), table_supported=True)


def test_opt_in_only_task2_and_old_default_unchanged():
    assert build_task_inputs('towel_folding')['task']['evaluation']['rule_profile'] == PRACTICAL_PROFILE
    spec = task()
    assert spec['public_rules']['schema'] == 'laundry_rules.v6'
    old = thresholds('towel_folding', PRACTICAL_PROFILE)
    assert {k: v for k, v in spec['evaluation']['thresholds'].items() if old.get(k) != v} == {
        'folded_pair_max_m': .08, 'folded_pair_area_percentile': 95.,
        'folded_pair_percentile_m': .06}
    for identifier in ('towel_flattening', 'towel_stacking', 'shirt_folding', 'trousers_folding'):
        with pytest.raises(ValueError):
            build_task_inputs(identifier, rule_profile=FOLDING_PRACTICAL_PROFILE)


def test_small_free_corner_is_not_a_whole_fold_failure():
    spec = task()
    q = goal_vertices(spec)['towel']
    mesh = SequenceEvaluator(spec).meshes['towel']
    # Fold the material corner inward; keep its layer, all outer extents and
    # the rest of the material correspondence. This isolates shape scoring,
    # not physical validity of a synthetic deformation.
    corner = mesh['upper'][np.argmax(mesh['rest'][mesh['upper'], 1])]
    q[corner, 1] -= .07
    passed, detail = fold_geometry(spec, q)
    assert passed, detail
    assert detail['pair_max_m'] == pytest.approx(.07)
    assert detail['pair_percentile_m'] < .01
    assert not fold_geometry(task(PRACTICAL_PROFILE), q)[0]
    q[corner, 1] -= .011
    assert not fold_geometry(spec, q)[0]  # Absolute bound still protects outliers.


def test_area_percentile_rejects_a_misaligned_material_strip():
    spec = task()
    q = goal_vertices(spec)['towel']
    mesh = SequenceEvaluator(spec).meshes['towel']
    upper = mesh['upper']
    # Move an interior strip inward so neither bounding box nor overall RMS
    # conceals the area whose half-fold correspondence is wrong.
    rest = mesh['rest'][upper]
    selected = upper[(rest[:, 1] > -.025) & (rest[:, 1] < .025)]
    assert len(selected) > .05*len(upper)
    q[selected, 1] += .07
    passed, detail = fold_geometry(spec, q)
    assert detail['pair_rms_m'] < .03
    assert detail['pair_max_m'] < .08
    assert detail['pair_percentile_m'] == pytest.approx(.07)
    assert not passed


@pytest.mark.parametrize('above', (False, True))
def test_five_percent_boundary_uses_rest_material_area(above):
    spec = task()
    q = goal_vertices(spec)['towel']
    mesh = SequenceEvaluator(spec).meshes['towel']
    weights = mesh['weights'][mesh['lower']]+mesh['weights'][mesh['upper']]
    order = np.argsort(np.abs(mesh['rest'][mesh['upper'], 1]), kind='stable')
    first_above = np.flatnonzero(np.cumsum(weights[order])/weights.sum() > .05)[0]+1
    selected = order[:first_above if above else first_above-1]
    fraction = weights[selected].sum()/weights.sum()
    assert (.05 < fraction < .06) if above else (.04 < fraction < .05)
    q[mesh['upper'][selected], 1] += .065
    passed, detail = fold_geometry(spec, q)
    assert detail['pair_rms_m'] < .03 and detail['pair_max_m'] < .08
    assert passed is (not above)


@pytest.mark.parametrize('kind', ('held', 'unfolded', 'crossing', 'hidden_constraint', 'no_operation', 'missing_frame'))
def test_practical_corner_rule_keeps_task_evidence_guards(kind):
    spec = task()
    goal = goal_vertices(spec)
    scorer = start(spec, goal) if kind != 'no_operation' else SequenceEvaluator(spec)
    if kind == 'no_operation':
        scorer.observe(sample(spec, 0, goal))
        scorer.observe(sample(spec, 1, goal))
    if kind == 'unfolded':
        goal['towel'] = np.asarray(spec['asset_metadata']['cloths'][0]['initial_vertices_m'])
    physics = ({'self_intersections': 1, 'valid': False, 'validity_scope': 'strict_geometry'}
               if kind == 'crossing' else {'hidden_constraints': 1} if kind == 'hidden_constraint' else None)
    result = finish(spec, scorer, goal, first=3 if kind == 'missing_frame' else 2,
        controlled=kind == 'held', physics=physics)
    assert not result['success'], result


def test_normal_fold_still_scores_after_fresh_release():
    spec = task()
    goal = goal_vertices(spec)
    result = finish(spec, start(spec, goal), goal)
    assert result['success'], result
    assert not result['robot_manipulation_qualified']


@pytest.mark.parametrize('configuration', ('single_arm', 'dual_same_side', 'dual_opposed'))
def test_three_robot_installations_bind_identical_public_rule(configuration):
    stands = Path(f'{DATA_ROOT}/assets/task_assets/laundry_room/robot_installation/contact_options_v2')
    spec = task(configuration_id=configuration, stand_manifest=stands/configuration/'manifest.json')
    evaluator = SequenceEvaluator(spec)
    assert evaluator.control_mode == ROBOT_CONTROL
    assert spec['public_rules']['referee']['thresholds'] == thresholds('towel_folding', FOLDING_PRACTICAL_PROFILE)
    changed = deepcopy(spec)
    changed['evaluation']['thresholds']['folded_pair_max_m'] = .09
    with pytest.raises(ValueError):
        SequenceEvaluator(changed)
