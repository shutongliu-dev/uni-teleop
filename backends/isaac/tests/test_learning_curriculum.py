from copy import deepcopy

import pytest

from agent.model_client.learning.assessment import CURRICULUM_STRATA, content_sha256
from agent.model_client.learning.curriculum import StudyBudget, bind_continuation, plan_from_cases, study_condition_sha256


def inputs():
    parent = 'polyomino_territory'
    cases = {}
    for stratum in CURRICULUM_STRATA[parent]:
        for side in ('left', 'right'):
            for variant in range(6):
                split = 'train' if variant < 4 else 'validation'
                key = f'{split}/{side}/{stratum}_{variant:02d}'
                cases[key] = {'public': {'gym_id': parent + '_practice', 'stratum': stratum,
                    'active_side': side, 'split': split, 'case_id': f'{stratum}_{variant:02d}',
                    'case_sha256': content_sha256(key), 'public_rules_sha256': 'a' * 64},
                    'case_fingerprint': content_sha256(['actual', key]),
                    'private_case_sha256': content_sha256(['private', key])}
    return {'task_id': parent + '_practice', 'parent_release': {'manifest_sha256': 'b' * 64}}, cases


def prepare(full=False, **changes):
    manifest, cases = inputs()
    return plan_from_cases(manifest, cases, parent_task_id='polyomino_territory',
        model={'model': 'test'}, source_manifest_sha256='c' * 64,
        gym_release_sha256='d' * 64, budget=StudyBudget(**changes), full_curriculum=full)


def test_smoke_both_hands_disjoint_actual_cases_and_exact_file_hash():
    plan = prepare()
    assert plan.gym_release_sha256 == 'd' * 64
    assert plan.validation.scope == 'integration_smoke'
    assert {s.active_side for s in plan.validation.slots} == {'left', 'right'}
    assert len(plan.validation.slots) == 2
    assert not set(plan.validation.practice_case_fingerprints) & {s.case_fingerprint for s in plan.validation.slots}
    assert [s.selection['active_side'] for s in plan.practice_slots] == ['left', 'right', 'left', 'right']


def test_full_curriculum_requires_all_four_strata_both_hands():
    plan = prepare(True, practice_episodes=8)
    assert plan.validation.scope == 'full_curriculum'
    assert len(plan.validation.slots) == 8
    assert len({s.case['public']['stratum'] for s in plan.practice_slots}) == 4


def test_duplicate_content_cannot_pass_by_changing_split_name():
    manifest, cases = inputs()
    cases['validation/left/mono_translate_04']['case_fingerprint'] = cases['train/left/mono_translate_00']['case_fingerprint']
    with pytest.raises(ValueError, match='same actual case content'):
        plan_from_cases(manifest, cases, parent_task_id='polyomino_territory', model={},
            source_manifest_sha256='c' * 64, gym_release_sha256='d' * 64, budget=StudyBudget())


def test_budget_participates_in_condition_and_no_empty_practice_batches():
    assert prepare().validation.condition_sha256 != prepare(max_actions_per_episode=65).validation.condition_sha256
    with pytest.raises(ValueError, match='at least one practice'):
        StudyBudget(practice_episodes=1, validation_batches=2)


def test_continuation_changes_condition_without_relabelling_original_plan():
    original = prepare()
    continued = bind_continuation(original, 'e' * 64)
    assert original.continuation_sha256 is None
    assert continued.continuation_sha256 == 'e' * 64
    assert continued.validation.condition_sha256 != original.validation.condition_sha256
    assert continued.practice_slots == original.practice_slots
    assert continued.validation.slots == original.validation.slots
    assert continued.budget == original.budget
    assert study_condition_sha256(original) == original.validation.condition_sha256
    assert study_condition_sha256(continued) == continued.validation.condition_sha256
    with pytest.raises(ValueError):
        bind_continuation(original, 'invalid')
    with pytest.raises(ValueError, match='already binds'):
        bind_continuation(continued, 'f' * 64)


def test_match_time_budget_is_independent_and_part_of_condition():
    base = prepare()
    changed = prepare(match_sim_seconds=140., match_wall_seconds=600.)
    assert base.budget.sim_seconds_per_episode == changed.budget.sim_seconds_per_episode == 45.
    assert base.budget.wall_seconds_per_episode is changed.budget.wall_seconds_per_episode is None
    assert base.budget.match_sim_seconds == 120. and base.budget.match_wall_seconds is None
    assert changed.validation.condition_sha256 != base.validation.condition_sha256
    assert StudyBudget(match_sim_seconds=None).match_sim_seconds is None
    for name, value in [('match_sim_seconds', 0.), ('match_sim_seconds', 151.),
                        ('match_wall_seconds', 0.), ('match_wall_seconds', 2401.)]:
        with pytest.raises(ValueError):
            StudyBudget(**{name: value})
