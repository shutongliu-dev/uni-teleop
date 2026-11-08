"""Host-only qualification boundaries: native evidence is never synthesized here."""
import json

import pytest
from pydantic import ValidationError

from agent.model_client.learning.assessment import (
    CURRICULUM_STRATA, ClosedLoopAssessment, Coverage, Effort, EvaluationPlan,
    FreezeManifest, FrozenMatchReceipt, PublishedCandidate, RevisionReceipt, TrialRecord, ValidationSlot,
    assess_validation, audit_closed_loop, check_frozen_execution, content_sha256,
    public_validation_feedback, select_budget_fallback, select_candidate, sum_effort, validate_freeze,
)


def h(value):
    return content_sha256(value)


def replace(model, **values):
    return type(model).model_validate(model.model_dump() | values)


def plan_for(task='polyomino_territory', *, smoke=False):
    slots = [ValidationSlot(slot_id=f'{side}_{i}', active_side=side, stratum=stratum,
        case_id=f'{stratum}_04', case_sha256=h(['public', side, stratum]),
        case_fingerprint=h(['actual', side, stratum]), private_case_sha256=h(['private', side, stratum]))
        for side in ('left', 'right') for i, stratum in enumerate(CURRICULUM_STRATA[task])]
    if smoke:
        slots = slots[:1]
    return EvaluationPlan(plan_id='registered_01', gym_id=task+'_practice', parent_task_id=task,
        parent_release_sha256=h('release'), public_rules_sha256=h('rules'), condition_sha256=h('condition'),
        source_manifest_sha256=h('source'), scope='integration_smoke' if smoke else 'full_curriculum',
        slots=slots, required_coverage=[Coverage(active_side=s.active_side, stratum=s.stratum) for s in slots],
        practice_case_fingerprints=[h('practice_case')])


def trial(plan, slot, *, candidate='candidate_b', bundle=None, **changes):
    data = dict(schema='card_rsi_trial_record.v1', phase='validation',
        **slot.model_dump(), gym_id=plan.gym_id, parent_task_id=plan.parent_task_id,
        parent_release_sha256=plan.parent_release_sha256, public_rules_sha256=plan.public_rules_sha256,
        condition_sha256=plan.condition_sha256, source_manifest_sha256=plan.source_manifest_sha256,
        candidate_event_id=candidate, bundle_hash=bundle or h(candidate), attempt_id=slot.slot_id+'_attempt0',
        attempt_index=0, receipt_sequence=20+plan.slots.index(slot), episode_id=slot.slot_id+'_episode',
        execution_status='completed', outcome='success', physics_valid=True, result_valid=True,
        evidence_source='native_solver_substeps', native_substeps=100, native_evidence_sha256=h('native'+slot.slot_id),
        result_sha256=h('result'+slot.slot_id), public_feedback_sha256=h('feedback'+slot.slot_id),
        skill_invocations=1, skill_language_entries=1, executed_bundle_hashes=[bundle or h(candidate)], temporary_skill_changes=False,
        fresh_context=True, author_history_visible=False, rule_violations=0,
        cost=Effort(model_requests=1, model_logical_calls=1, physical_attempts=1,
                    accepted_actions=1, completed_actions=1, native_sim_s=.5))
    data.update(changes)
    return TrialRecord.model_validate(data)


def assess(plan, rows, candidate='candidate_b'):
    return assess_validation(plan, rows, candidate_event_id=candidate, bundle_hash=h(candidate),
                             condition_sha256=plan.condition_sha256)


def frozen(plan, assessment):
    return FreezeManifest(learner_id='learner', learning_run_id='run', gym_id=plan.gym_id,
        parent_task_id=plan.parent_task_id, parent_release_sha256=plan.parent_release_sha256,
        public_rules_sha256=plan.public_rules_sha256, condition_sha256=plan.condition_sha256,
        source_manifest_sha256=plan.source_manifest_sha256, candidate_event_id=assessment.candidate_event_id,
        bundle_hash=assessment.bundle_hash, selection_event_id='selection_01',
        validation_plan_sha256=content_sha256(plan), validation_assessment_sha256=content_sha256(assessment),
        selection_mode='validated_rank', validation_complete=assessment.all_slots_resolved,
        immutable_validation_complete=assessment.immutable_execution_complete, qualification_reasons=assessment.reasons,
        validation_event_ids=['validation_01'], qualified_for_declared_curriculum=assessment.qualified_for_declared_curriculum,
        freeze_sequence=40, development_cost=assessment.cost, development_ledger_sha256=h('ledger'))


@pytest.mark.parametrize('task', list(CURRICULUM_STRATA))
def test_full_curriculum_is_only_declared_practice_qualification(task):
    plan = plan_for(task)
    result = assess(plan, [trial(plan, s) for s in plan.slots])
    assert result.qualified_for_declared_curriculum
    assert result.successes == 8 and result.failures == result.unknown == 0
    assert result.coverage_successes == {s.key(): 1 for s in plan.slots}
    assert result.formal_match_qualified is False
    assert validate_freeze(frozen(plan, result), plan, result)


def test_smoke_cannot_qualify_or_claim_full_curriculum():
    plan = plan_for(smoke=True)
    result = assess(plan, [trial(plan, plan.slots[0])])
    assert result.successes == 1 and not result.qualified_for_declared_curriculum
    assert select_candidate([result]) is None
    assert select_candidate([result], allow_unqualified=True) == result
    with pytest.raises(ValidationError, match='four registered strata'):
        replace(plan, scope='full_curriculum')


def test_case_renaming_cannot_disguise_practice_overlap_or_duplicate_content():
    plan = plan_for()
    with pytest.raises(ValidationError, match='same actual case'):
        replace(plan, practice_case_fingerprints=[plan.slots[0].case_fingerprint])
    renamed = replace(plan.slots[1], case_id='totally_new_seed', case_fingerprint=plan.slots[0].case_fingerprint)
    with pytest.raises(ValidationError, match='Duplicate planned'):
        replace(plan, slots=[plan.slots[0], renamed, *plan.slots[2:]])


def test_arbitrary_easy_labels_cannot_replace_registered_curriculum():
    plan = plan_for()
    slots = [replace(s, stratum='easy'+s.stratum) for s in plan.slots]
    with pytest.raises(ValidationError, match='four registered strata'):
        replace(plan, slots=slots, required_coverage=[Coverage(active_side=s.active_side, stratum=s.stratum) for s in slots])


@pytest.mark.parametrize('field', ['case_sha256', 'case_fingerprint', 'private_case_sha256',
    'parent_release_sha256', 'public_rules_sha256', 'condition_sha256', 'source_manifest_sha256', 'bundle_hash'])
def test_reject_trial_identity_substitution(field):
    plan = plan_for()
    row = trial(plan, plan.slots[0])
    with pytest.raises(ValueError):
        assess(plan, [replace(row, **{field: h('substitution')})])


def test_first_valid_failure_survives_later_success_and_all_costs_count():
    plan = plan_for()
    rows = [trial(plan, s) for s in plan.slots]
    rows[0] = replace(rows[0], outcome='failure')
    retry = replace(rows[0], attempt_id='retry', attempt_index=1, receipt_sequence=30, outcome='success')
    result = assess(plan, rows+[retry])
    assert (result.successes, result.failures, result.unknown) == (7, 1, 0)
    assert result.slots[0].selected_attempt_id == rows[0].attempt_id
    assert not result.qualified_for_declared_curriculum
    assert result.cost.totals['model_requests'] == result.cost.attempts == 9


def test_unresolved_infrastructure_attempt_is_not_failure_or_free_retry():
    plan = plan_for()
    rows = [trial(plan, s) for s in plan.slots]
    infra = replace(rows[0], execution_status='infra_error', outcome='unknown', result_valid=False,
                    physics_valid=None, native_substeps=0, native_evidence_sha256=None, result_sha256=None)
    result = assess(plan, [infra, *rows[1:]])
    assert (result.successes, result.failures, result.unknown) == (7, 0, 1)
    assert not result.qualified_for_declared_curriculum
    retry = replace(rows[0], attempt_id='retry', attempt_index=1, receipt_sequence=30)
    resolved = assess(plan, [infra, retry, *rows[1:]])
    assert resolved.qualified_for_declared_curriculum and resolved.cost.attempts == 9
    assert resolved.slots[0].selected_attempt_id == 'retry'


@pytest.mark.parametrize('changes', [dict(evidence_source='synthetic'), dict(native_substeps=0),
    dict(physics_valid=None), dict(native_evidence_sha256=None), dict(result_valid=False)])
def test_unknown_or_synthetic_receipts_never_become_native_pass(changes):
    plan = plan_for()
    rows = [trial(plan, s) for s in plan.slots]
    rows[0] = replace(rows[0], **changes)
    result = assess(plan, rows)
    assert result.unknown == 1 and not result.qualified_for_declared_curriculum


@pytest.mark.parametrize('changes', [dict(temporary_skill_changes=True), dict(fresh_context=False),
    dict(author_history_visible=True), dict(skill_invocations=0, skill_language_entries=0, executed_bundle_hashes=[]),
    dict(executed_bundle_hashes=[h('another_bundle')])])
def test_true_task_success_does_not_qualify_unexecuted_or_modified_skill(changes):
    plan = plan_for()
    rows = [trial(plan, s) for s in plan.slots]
    rows[0] = replace(rows[0], **changes)
    retry = trial(plan, plan.slots[0], attempt_id='clean_retry', attempt_index=1, receipt_sequence=30)
    result = assess(plan, rows+[retry])
    assert result.successes == 8 and not result.immutable_execution_complete
    assert not result.qualified_for_declared_curriculum
    assert result.slots[0].selected_attempt_id == rows[0].attempt_id


def test_no_cherry_picked_attempt_gaps_duplicates_or_reversed_receipts():
    plan = plan_for()
    row = trial(plan, plan.slots[0])
    for rows in ([replace(row, attempt_index=1)], [row, row],
                 [row, replace(row, attempt_id='retry', attempt_index=1, receipt_sequence=19)]):
        with pytest.raises(ValueError):
            assess(plan, rows)


def test_all_violations_and_nullable_usage_are_preserved():
    plan = plan_for()
    rows = [trial(plan, s) for s in plan.slots]
    rows[0] = replace(rows[0], rule_violations=1)
    result = assess(plan, rows)
    assert result.rule_violations == 1 and not result.qualified_for_declared_curriculum
    unknown = result.cost.provider_usage['output_tokens']
    assert not unknown.complete and unknown.unknown_attempts == 8
    totals = sum_effort([Effort(output_tokens=10, reasoning_tokens=8), Effort(output_tokens=None)])
    assert totals.provider_usage['output_tokens'].known_total == 10
    assert not totals.provider_usage['output_tokens'].complete
    assert 'total_tokens' not in totals.totals


def test_public_feedback_has_no_private_case_ids_or_hashes_and_waits_for_batch():
    plan = plan_for()
    result = assess(plan, [trial(plan, s) for s in plan.slots])
    with pytest.raises(ValueError, match='entire batch'):
        public_validation_feedback(result, batch_closed=False)
    encoded = json.dumps(public_validation_feedback(result, batch_closed=True))
    for slot in plan.slots:
        assert slot.case_fingerprint not in encoded and slot.private_case_sha256 not in encoded
        assert slot.case_id not in encoded


def test_json_roundtrip_and_strict_types():
    plan = plan_for()
    rows = [trial(plan, s) for s in plan.slots]
    assert assess(plan, [r.model_dump() for r in rows]) == assess(plan, rows)
    assert EvaluationPlan.model_validate_json(plan.model_dump_json()) == plan
    with pytest.raises(ValidationError):
        Effort(model_requests=True)
    with pytest.raises(ValidationError):
        replace(rows[0], fresh_context=1)
    with pytest.raises(ValidationError):
        replace(rows[0], leaked_private_extra='not_allowed')


def test_selection_is_fixed_and_not_biased_towards_partial_or_unknown_cost():
    plan = plan_for()
    a = assess(plan, [trial(plan, s, candidate='candidate_a') for s in plan.slots], 'candidate_a')
    b = assess(plan, [trial(plan, s) for s in plan.slots])
    assert select_candidate([b, a]).candidate_event_id == 'candidate_a'
    incomplete = assess(plan, [trial(plan, plan.slots[0], candidate='candidate_c')], 'candidate_c')
    assert select_candidate([incomplete, a, b]).candidate_event_id == 'candidate_a'
    with pytest.raises(ValueError):
        select_candidate([a, a])


def test_freeze_pins_source_and_does_not_allow_strategy_rewrite():
    plan = plan_for()
    result = assess(plan, [trial(plan, s) for s in plan.slots])
    freeze = frozen(plan, result)
    kwargs = dict(bundle_hash=freeze.bundle_hash, condition_sha256=freeze.condition_sha256,
        source_manifest_sha256=freeze.source_manifest_sha256, temporary_skill_changes=False)
    assert check_frozen_execution(freeze, **kwargs)
    for field, value in [('bundle_hash', h('x')), ('condition_sha256', h('x')),
                         ('source_manifest_sha256', h('x')), ('temporary_skill_changes', True)]:
        with pytest.raises(ValueError):
            check_frozen_execution(freeze, **(kwargs | {field: value}))
    with pytest.raises(ValueError):
        validate_freeze(replace(freeze, source_manifest_sha256=h('different')), plan, result)
    with pytest.raises(ValidationError):
        replace(freeze, formal_match_qualified=True)


def closed_loop_inputs():
    plan = plan_for()
    old = trial(plan, plan.slots[0], candidate='candidate_a', phase='practice', receipt_sequence=2,
        case_fingerprint=h('practice_case'), case_id='practice_00')
    revision = RevisionReceipt(receipt_sequence=10, author='model', previous_bundle_hash=h('candidate_a'),
        candidate_event_id='candidate_b', bundle_hash=h('candidate_b'), changed_executable_content=True,
        model_request_sha256=h('request'), model_response_sha256=h('response'), actual_model_requests=1,
        practice_attempt_ids=[old.attempt_id], public_feedback_sha256=[old.public_feedback_sha256])
    rows = [trial(plan, s) for s in plan.slots]
    assessment = assess(plan, rows)
    freeze = frozen(plan, assessment)
    match = FrozenMatchReceipt(receipt_sequence=50, episode_id='formal_match_01', parent_task_id=plan.parent_task_id,
        parent_release_sha256=plan.parent_release_sha256, bundle_hash=freeze.bundle_hash,
        condition_sha256=freeze.condition_sha256, source_manifest_sha256=freeze.source_manifest_sha256,
        result_sha256=h('match_result'), outcome='incomplete', physics_valid=True,
        evidence_source='native_solver_substeps', native_substeps=100, native_evidence_sha256=h('match_native'),
        skill_invocations=1, skill_language_entries=1, executed_bundle_hashes=[freeze.bundle_hash], temporary_skill_changes=False,
        fresh_context=True, author_history_visible=False)
    return dict(plan=plan, practice=[old], revision=revision, validation=rows, assessment=assessment, freeze=freeze, match=match)


def test_real_closed_loop_is_distinct_from_improvement_and_match_completion():
    data = closed_loop_inputs()
    result = audit_closed_loop(**data)
    assert result.closed_loop_executed and not result.performance_improvement_established
    assert not result.formal_match_completed
    result = audit_closed_loop(**(data | {'match': replace(data['match'], outcome='loss')}))
    assert result.closed_loop_executed and result.formal_match_completed
    assert not result.performance_improvement_established
    with pytest.raises(ValidationError):
        replace(result, performance_improvement_established=True)


@pytest.mark.parametrize('changes', [dict(author='human'), dict(author='scripted'), dict(actual_model_requests=0),
    dict(changed_executable_content=False), dict(previous_bundle_hash=h('candidate_b')),
    dict(public_feedback_sha256=[h('unrelated_feedback')]), dict(receipt_sequence=1)])
def test_fake_revision_does_not_establish_autonomous_rsi(changes):
    data = closed_loop_inputs()
    data['revision'] = replace(data['revision'], **changes)
    result = audit_closed_loop(**data)
    assert not result.autonomous_revision_verified and not result.closed_loop_executed


@pytest.mark.parametrize('changes', [dict(evidence_source='synthetic'), dict(native_substeps=0),
    dict(skill_invocations=0, skill_language_entries=0, executed_bundle_hashes=[]), dict(temporary_skill_changes=True),
    dict(author_history_visible=True), dict(receipt_sequence=39)])
def test_no_formal_native_skill_execution_no_closed_loop(changes):
    data = closed_loop_inputs()
    data['match'] = replace(data['match'], **changes)
    result = audit_closed_loop(**data)
    assert not result.frozen_match_executed and not result.closed_loop_executed


def test_validation_must_follow_revision_and_match_exact_receipts():
    data = closed_loop_inputs()
    data['validation'][0] = replace(data['validation'][0], receipt_sequence=9)
    result = audit_closed_loop(**data)
    assert not result.independent_validation_verified and not result.closed_loop_executed


def test_practice_may_use_learning_history_but_must_not_touch_validation_content():
    data = closed_loop_inputs()
    data['practice'][0] = replace(data['practice'][0], author_history_visible=True, fresh_context=False)
    assert audit_closed_loop(**data).autonomous_revision_verified
    data['practice'].append(trial(data['plan'], data['plan'].slots[1], candidate='candidate_a',
        phase='practice', receipt_sequence=3))
    assert not audit_closed_loop(**data).autonomous_revision_verified


def test_host_validation_rechecks_mutated_model_containers():
    plan = plan_for()
    plan.slots.append(plan.slots[0])
    with pytest.raises(ValidationError):
        assess(plan, [])


def test_explicit_budget_fallback_preserves_unknown_and_does_not_block_frozen_execution():
    plan = plan_for()
    result = assess(plan, [trial(plan, plan.slots[0])])
    assert not result.all_slots_resolved and not result.qualified_for_declared_curriculum
    assert select_candidate([result], allow_unqualified=True) is None
    publication = PublishedCandidate(candidate_event_id=result.candidate_event_id, bundle_hash=result.bundle_hash,
        publication_sequence=1, is_empty=False, author='model')
    assert select_budget_fallback(plan, [publication], budget_exhausted=True) == publication
    freeze = replace(frozen(plan, result), selection_mode='budget_fallback',
        qualification_reasons=result.reasons+['declared_budget_fallback'])
    assert validate_freeze(freeze, plan, result)
    assert not freeze.validation_complete and not freeze.immutable_validation_complete
    assert not freeze.qualified_for_declared_curriculum
    assert check_frozen_execution(freeze, bundle_hash=freeze.bundle_hash,
        condition_sha256=freeze.condition_sha256, source_manifest_sha256=freeze.source_manifest_sha256,
        temporary_skill_changes=False)
    with pytest.raises(ValueError):
        validate_freeze(replace(freeze, qualified_for_declared_curriculum=True), plan, result)
    with pytest.raises(ValueError):
        validate_freeze(replace(freeze, qualification_reasons=['declared_budget_fallback']), plan, result)


def test_budget_fallback_is_preregistered_not_best_retry_or_implicit_empty():
    plan = plan_for()
    empty = PublishedCandidate(candidate_event_id='empty', bundle_hash=h('empty'), publication_sequence=0,
        is_empty=True, author='host_empty_initialization')
    a = PublishedCandidate(candidate_event_id='a', bundle_hash=h('a'), publication_sequence=1, is_empty=False, author='model')
    b = PublishedCandidate(candidate_event_id='b', bundle_hash=h('b'), publication_sequence=2, is_empty=False, author='model')
    assert select_budget_fallback(plan, [empty, b, a], budget_exhausted=True) == b
    assert select_budget_fallback(plan, [], budget_exhausted=True) is None
    with pytest.raises(ValueError, match='budget closure'):
        select_budget_fallback(plan, [a], budget_exhausted=False)
    empty_plan = replace(plan, budget_fallback='empty')
    assert select_budget_fallback(empty_plan, [b, empty, a], budget_exhausted=True) == empty
    assert select_budget_fallback(empty_plan, [a, b], budget_exhausted=True) is None
    with pytest.raises(ValueError, match='Duplicate publication'):
        select_budget_fallback(plan, [a, a], budget_exhausted=True)
    with pytest.raises(ValidationError, match='cannot introduce a strategy'):
        replace(a, author='host_empty_initialization')


def test_unassessed_budget_freeze_is_honest_and_does_not_manufacture_validation():
    plan = plan_for()
    result = assess(plan, [])
    freeze = replace(frozen(plan, result), selection_mode='budget_fallback', validation_assessment_sha256=None,
        validation_event_ids=[], qualification_reasons=['no_validation_assessment', 'declared_budget_fallback'])
    assert validate_freeze(freeze, plan)
    assert not freeze.qualified_for_declared_curriculum and not freeze.validation_complete
    with pytest.raises(ValueError, match='cannot invent'):
        validate_freeze(replace(freeze, validation_complete=True), plan)
    with pytest.raises(ValueError):
        validate_freeze(replace(freeze, selection_mode='validated_rank'), plan)
