"""Host-only RSI assessment contracts and deterministic, pure reductions.

These objects are NOT model context. Case fingerprints and private-case hashes
stay with the trusted scheduler. Bench adapters must verify the actual result,
native receipts, source imports and executed bundle before constructing a
TrialRecord: validating JSON cannot establish that those events happened.
Persistence remains SkillLibrary/Journal's responsibility; no simulator, model,
filesystem or network access is performed here.
"""
from __future__ import annotations

import hashlib
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictInt, model_validator

from .journal import canonical_json

Hash = Annotated[str, Field(pattern=r'^[0-9a-f]{64}$')]
Name = Annotated[str, Field(min_length=1, max_length=160, pattern=r'^[A-Za-z0-9][A-Za-z0-9_.:-]*$')]
Count = Annotated[StrictInt, Field(ge=0)]
Amount = Annotated[float, Field(ge=0., allow_inf_nan=False)]
Rate = Annotated[float, Field(ge=0., le=1., allow_inf_nan=False)]
Side = Literal['left', 'right']
Task = Literal['polyomino_territory', 'barrier_race', 'memory_pairs', 'disk_curling']
Outcome = Literal['success', 'failure', 'unknown']
CURRICULUM_STRATA = {
    'polyomino_territory': ('mono_translate', 'domino_rotate', 'concave_rotate', 'multi_piece_adjacent'),
    'barrier_race': ('pawn_step', 'wall_place', 'wall_rotate', 'pawn_obstacle'),
    'memory_pairs': ('flip', 'restore', 'collect_pair', 'flip_restore_collect'),
    'disk_curling': ('launch', 'release_line', 'target_score', 'obstacle_contact'),
}


class HostContract(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True, validate_assignment=True, revalidate_instances='always')


def content_sha256(value):
    if isinstance(value, BaseModel):
        value = value.model_dump(mode='json')
    return hashlib.sha256(canonical_json(value)).hexdigest()


class Coverage(HostContract):
    active_side: Side
    stratum: Name

    def key(self):
        return self.active_side + ':' + self.stratum


class ValidationSlot(Coverage):
    slot_id: Name
    case_id: Name
    case_sha256: Hash
    case_fingerprint: Hash
    private_case_sha256: Hash


class EvaluationPlan(HostContract):
    schema: Literal['card_rsi_evaluation_plan.v1'] = 'card_rsi_evaluation_plan.v1'
    plan_id: Name
    gym_id: Name
    parent_task_id: Task
    parent_release_sha256: Hash
    public_rules_sha256: Hash
    condition_sha256: Hash
    source_manifest_sha256: Hash
    scope: Literal['full_curriculum', 'integration_smoke']
    slots: list[ValidationSlot] = Field(min_length=1, max_length=4096)
    required_coverage: list[Coverage] = Field(min_length=1, max_length=128)
    practice_case_fingerprints: list[Hash] = Field(default_factory=list, max_length=100000)
    minimum_success_rate: Rate = 1.
    minimum_successes_per_coverage: Count = 1
    maximum_rule_violations: Count = 0
    selection_policy: Literal['success_rate_violations_requests_actions_sim_v1'] = 'success_rate_violations_requests_actions_sim_v1'
    budget_fallback: Literal['last_published', 'empty'] = 'last_published'

    @model_validator(mode='after')
    def coherent_plan(self):
        if self.gym_id != self.parent_task_id+'_practice':
            raise ValueError('Gym must bind its corresponding formal task')
        for values in ([s.slot_id for s in self.slots], [s.case_fingerprint for s in self.slots],
                       [c.key() for c in self.required_coverage], self.practice_case_fingerprints):
            if len(values) != len(set(values)):
                raise ValueError('Duplicate planned slot, case content, coverage or practice identity')
        if set(self.practice_case_fingerprints) & {s.case_fingerprint for s in self.slots}:
            raise ValueError('Practice and validation contain the same actual case content')
        required = {c.key() for c in self.required_coverage}
        if {s.key() for s in self.slots} != required:
            raise ValueError('Slots must cover exactly the declared curriculum strata and sides')
        if self.minimum_successes_per_coverage < 1:
            raise ValueError('Every declared coverage cell needs an actual success')
        if any(sum(s.key() == key for s in self.slots) < self.minimum_successes_per_coverage for key in required):
            raise ValueError('Coverage success requirement exceeds its planned slots')
        if self.scope == 'full_curriculum':
            left = {c.stratum for c in self.required_coverage if c.active_side == 'left'}
            right = {c.stratum for c in self.required_coverage if c.active_side == 'right'}
            if left != right or left != set(CURRICULUM_STRATA[self.parent_task_id]) or len(self.slots) < 8:
                raise ValueError('Full curriculum requires the four registered strata on both real hands')
        return self


class Effort(HostContract):
    """Actual consumption, including failures; unknown provider usage stays null.

    reasoning_tokens can be a subset of output_tokens. They are never added
    together here. Reservations and maximum action windows are not actual sim.
    """
    model_requests: Count = 0
    model_logical_calls: Count = 0
    input_tokens: Count | None = None
    output_tokens: Count | None = None
    reasoning_tokens: Count | None = None
    cached_input_tokens: Count | None = None
    physical_attempts: Count = 0
    accepted_actions: Count = 0
    completed_actions: Count = 0
    read_only_calls: Count = 0
    reset_attempts: Count = 0
    native_sim_s: Amount = 0.
    wall_s: Amount = 0.
    cpu_s: Amount = 0.
    provider_cost_usd: Amount | None = None

    @model_validator(mode='after')
    def counters(self):
        if not self.completed_actions <= self.accepted_actions <= self.physical_attempts:
            raise ValueError('Completed/accepted physical actions exceed actual attempts')
        return self


class NullableTotal(HostContract):
    known_total: Amount
    unknown_attempts: Count
    complete: StrictBool


class EffortSummary(HostContract):
    attempts: Count
    totals: dict[str, Amount]
    provider_usage: dict[str, NullableTotal]


def sum_effort(values: list[Effort]) -> EffortSummary:
    nullable = ('input_tokens', 'output_tokens', 'reasoning_tokens', 'cached_input_tokens', 'provider_cost_usd')
    totals = {name: float(sum(getattr(v, name) for v in values))
              for name in Effort.model_fields if name not in nullable}
    provider = {}
    for name in nullable:
        items = [getattr(v, name) for v in values]
        missing = sum(v is None for v in items)
        provider[name] = NullableTotal(known_total=float(sum(v for v in items if v is not None)),
                                      unknown_attempts=missing, complete=missing == 0)
    return EffortSummary(attempts=len(values), totals=totals, provider_usage=provider)


class TrialRecord(ValidationSlot):
    schema: Literal['card_rsi_trial_record.v1'] = 'card_rsi_trial_record.v1'
    phase: Literal['practice', 'validation', 'heldout']
    gym_id: Name
    parent_task_id: Task
    parent_release_sha256: Hash
    public_rules_sha256: Hash
    condition_sha256: Hash
    candidate_event_id: Name
    bundle_hash: Hash
    attempt_id: Name
    attempt_index: Count
    receipt_sequence: Count
    episode_id: Name | None = None
    execution_status: Literal['completed', 'agent_budget', 'agent_error', 'infra_error', 'invalid_reset', 'protocol_invalid', 'unknown']
    outcome: Outcome
    physics_valid: StrictBool | None
    result_valid: StrictBool
    evidence_source: Literal['native_solver_substeps', 'synthetic', 'unverified']
    native_substeps: Count = 0
    native_evidence_sha256: Hash | None = None
    source_manifest_sha256: Hash
    result_sha256: Hash | None = None
    public_feedback_sha256: Hash | None = None
    skill_invocations: Count = 0
    skill_language_entries: Count = 0
    executed_bundle_hashes: list[Hash] = Field(default_factory=list, max_length=32)
    temporary_skill_changes: StrictBool
    fresh_context: StrictBool
    author_history_visible: StrictBool
    rule_violations: Count = 0
    cost: Effort
    verification_issues: list[str] = Field(default_factory=list)
    artifact_sha256: dict[str, Hash] = Field(default_factory=dict)

    @model_validator(mode='after')
    def binding_consistency(self):
        if self.gym_id != self.parent_task_id+'_practice':
            raise ValueError('Trial gym/formal task mismatch')
        if len(self.executed_bundle_hashes) != len(set(self.executed_bundle_hashes)):
            raise ValueError('Executed bundle identities must be unique')
        if bool(self.skill_invocations) != bool(self.executed_bundle_hashes):
            raise ValueError('Skill invocation and executed-bundle receipt disagree')
        if self.skill_language_entries > self.skill_invocations:
            raise ValueError('Language entries exceed recorded invocation attempts')
        if self.result_valid and (self.outcome == 'unknown' or self.result_sha256 is None or self.episode_id is None):
            raise ValueError('Valid outcome needs an authoritative result and episode')
        return self

    def native_result(self):
        # An authoritative terminal result survives later reporting/network
        # errors. Execution status alone neither invalidates it nor proves it.
        return bool(self.result_valid and self.physics_valid is True
            and self.evidence_source == 'native_solver_substeps'
            and self.native_substeps > 0 and self.native_evidence_sha256
            and self.result_sha256 and self.episode_id and self.outcome != 'unknown')

    def immutable_skill_execution(self):
        return (self.skill_language_entries > 0 and self.executed_bundle_hashes == [self.bundle_hash]
                and not self.temporary_skill_changes and self.fresh_context and not self.author_history_visible)


class SlotAssessment(HostContract):
    slot_id: Name
    active_side: Side
    stratum: Name
    selected_attempt_id: Name | None
    outcome: Outcome
    immutable_skill_execution: StrictBool
    rule_violations: Count
    attempted: StrictBool


class ValidationAssessment(HostContract):
    schema: Literal['card_rsi_validation_assessment.v1'] = 'card_rsi_validation_assessment.v1'
    plan_sha256: Hash
    scope: Literal['full_curriculum', 'integration_smoke']
    candidate_event_id: Name
    bundle_hash: Hash
    condition_sha256: Hash
    slots: list[SlotAssessment]
    successes: Count
    failures: Count
    unknown: Count
    success_rate: Rate | None
    coverage_successes: dict[str, Count]
    all_slots_attempted: StrictBool
    all_slots_resolved: StrictBool
    immutable_execution_complete: StrictBool
    rule_violations: Count
    qualified_for_declared_curriculum: StrictBool
    formal_match_qualified: Literal[False] = False
    reasons: list[str]
    cost: EffortSummary
    trial_receipt_sha256: list[Hash]


def assess_validation(plan: EvaluationPlan, records: list[TrialRecord], *,
                      candidate_event_id: str, bundle_hash: str,
                      condition_sha256: str) -> ValidationAssessment:
    """Use the first authoritative native result of every planned slot.

    A bad/failing first valid result is not replaced by a better retry. A
    temporary skill edit also remains attached to that first result; a later
    unmodified retry cannot launder it into an independently verified bundle.
    All attempts, including unresolved and later retries, contribute cost.
    """
    plan = EvaluationPlan.model_validate(plan)
    if condition_sha256 != plan.condition_sha256:
        raise ValueError('Validation condition differs from its preregistered plan')
    slots = {s.slot_id: s for s in plan.slots}
    grouped = {key: [] for key in slots}
    seen = set()
    rows = []
    for raw in records:
        row = TrialRecord.model_validate(raw)
        if row.attempt_id in seen:
            raise ValueError('Duplicate attempt receipt')
        seen.add(row.attempt_id)
        if row.slot_id not in slots or row.phase != 'validation':
            raise ValueError('Unexpected validation slot or phase')
        slot = slots[row.slot_id]
        for key in ValidationSlot.model_fields:
            if getattr(row, key) != getattr(slot, key):
                raise ValueError('Trial differs from its fixed case or coverage binding')
        for key in ('gym_id', 'parent_task_id', 'parent_release_sha256', 'public_rules_sha256', 'condition_sha256', 'source_manifest_sha256'):
            if getattr(row, key) != getattr(plan, key):
                raise ValueError('Trial changes the preregistered condition')
        if (row.candidate_event_id, row.bundle_hash, row.condition_sha256) != (
                candidate_event_id, bundle_hash, condition_sha256):
            raise ValueError('A validation batch cannot mix candidate identities')
        grouped[row.slot_id].append(row)
        rows.append(row)
    results = []
    coverage = {c.key(): 0 for c in plan.required_coverage}
    violations = 0
    for slot in plan.slots:
        attempts = sorted(grouped[slot.slot_id], key=lambda r: r.attempt_index)
        if [r.attempt_index for r in attempts] != list(range(len(attempts))):
            raise ValueError('Attempt indices must start at zero, without duplicates or omitted predecessors')
        if any(b.receipt_sequence <= a.receipt_sequence for a, b in zip(attempts, attempts[1:])):
            raise ValueError('Attempt receipt order contradicts attempt indices')
        selected = next((r for r in attempts if r.native_result()), None)
        violations += sum(r.rule_violations for r in attempts)
        outcome = selected.outcome if selected else 'unknown'
        immutable = selected.immutable_skill_execution() if selected else False
        if outcome == 'success' and immutable:
            coverage[slot.key()] += 1
        results.append(SlotAssessment(slot_id=slot.slot_id, active_side=slot.active_side, stratum=slot.stratum,
            selected_attempt_id=selected.attempt_id if selected else None, outcome=outcome,
            immutable_skill_execution=immutable, rule_violations=selected.rule_violations if selected else 0,
            attempted=bool(attempts)))
    successes = sum(r.outcome == 'success' for r in results)
    failures = sum(r.outcome == 'failure' for r in results)
    unknown = len(results)-successes-failures
    rate = successes/(successes+failures) if successes+failures else None
    immutable = all(r.immutable_skill_execution for r in results)
    reasons = []
    if plan.scope != 'full_curriculum': reasons.append('integration_smoke_is_not_curriculum_qualification')
    if unknown: reasons.append('unresolved_slots')
    if not immutable: reasons.append('missing_independent_unchanged_skill_execution')
    if any(n < plan.minimum_successes_per_coverage for n in coverage.values()): reasons.append('missing_successful_side_stratum_coverage')
    if rate is None or rate < plan.minimum_success_rate: reasons.append('success_rate_below_declared_gate')
    if violations > plan.maximum_rule_violations: reasons.append('rule_violations_exceed_declared_gate')
    return ValidationAssessment(plan_sha256=content_sha256(plan), scope=plan.scope,
        candidate_event_id=candidate_event_id, bundle_hash=bundle_hash, condition_sha256=condition_sha256,
        slots=results, successes=successes, failures=failures, unknown=unknown, success_rate=rate,
        coverage_successes=coverage, all_slots_attempted=all(r.attempted for r in results),
        all_slots_resolved=unknown == 0, immutable_execution_complete=immutable, rule_violations=violations,
        qualified_for_declared_curriculum=not reasons, reasons=reasons,
        cost=sum_effort([r.cost for r in rows]), trial_receipt_sha256=[content_sha256(r) for r in rows])


def select_candidate(assessments: list[ValidationAssessment], *, allow_unqualified=False):
    """Choose a fully resolved, unchanged execution under one fixed plan.

    Optional unqualified selection can freeze a declared best available library;
    it never changes the assessment into a qualification. No partial batch is
    ranked against a complete batch, and missing costs are never treated as zero
    in the ordering: fixed, fully counted resource proxies are used instead.
    """
    if not assessments:
        return None
    identities = {(a.plan_sha256, a.condition_sha256) for a in assessments}
    if len(identities) != 1 or len({a.candidate_event_id for a in assessments}) != len(assessments):
        raise ValueError('Selection must compare distinct candidates under one preregistered plan')
    eligible = [a for a in assessments if a.all_slots_resolved and a.immutable_execution_complete
                and a.qualified_for_declared_curriculum]
    if not eligible and allow_unqualified:
        eligible = [a for a in assessments if a.all_slots_resolved and a.immutable_execution_complete]
    if not eligible:
        return None
    def key(a):
        costs = a.cost.totals
        return (-a.success_rate, a.rule_violations, costs['model_requests'], costs['physical_attempts'],
                costs['native_sim_s'], a.candidate_event_id)
    return min(eligible, key=key)


class PublishedCandidate(HostContract):
    """Complete immutable publication verified by the existing SkillLibrary."""
    candidate_event_id: Name
    bundle_hash: Hash
    publication_sequence: Count
    is_empty: StrictBool
    author: Literal['model', 'inherited_model', 'host_empty_initialization']

    @model_validator(mode='after')
    def empty_initialization(self):
        if self.author == 'host_empty_initialization' and not self.is_empty:
            raise ValueError('Host initialization cannot introduce a strategy')
        return self


def select_budget_fallback(plan: EvaluationPlan, published: list[PublishedCandidate], *, budget_exhausted: bool):
    """Budget closure may freeze an unqualified bundle, without changing its score.

    No publication means no implicit empty library. The empty policy requires
    an explicitly published host-created empty baseline. Publication validity
    and the real exhausted budget are checked by the trusted scheduler.
    """
    plan = EvaluationPlan.model_validate(plan)
    published = [PublishedCandidate.model_validate(p) for p in published]
    if budget_exhausted is not True:
        raise ValueError('Declared budget fallback is only available after budget closure')
    for values in ([p.candidate_event_id for p in published], [p.publication_sequence for p in published]):
        if len(values) != len(set(values)):
            raise ValueError('Duplicate publication identity or sequence')
    if plan.budget_fallback == 'empty':
        candidates = [p for p in published if p.is_empty and p.author == 'host_empty_initialization']
        if len(candidates) > 1:
            raise ValueError('Empty fallback must bind one fixed initialization')
        return candidates[0] if candidates else None
    return max(published, key=lambda p: p.publication_sequence) if published else None


class FreezeManifest(HostContract):
    schema: Literal['card_rsi_freeze_manifest.v1'] = 'card_rsi_freeze_manifest.v1'
    learner_id: Name
    learning_run_id: Name
    gym_id: Name
    parent_task_id: Task
    parent_release_sha256: Hash
    public_rules_sha256: Hash
    condition_sha256: Hash
    source_manifest_sha256: Hash
    candidate_event_id: Name
    bundle_hash: Hash
    selection_event_id: Name
    validation_plan_sha256: Hash
    selection_mode: Literal['validated_rank', 'budget_fallback']
    validation_assessment_sha256: Hash | None
    validation_event_ids: list[Name]
    validation_complete: StrictBool
    immutable_validation_complete: StrictBool
    qualification_reasons: list[str]
    qualified_for_declared_curriculum: StrictBool
    scope: Literal['declared_practice_skills_only'] = 'declared_practice_skills_only'
    formal_match_qualified: Literal[False] = False
    freeze_sequence: Count
    development_cost: EffortSummary
    development_ledger_sha256: Hash


def validate_freeze(freeze: FreezeManifest, plan: EvaluationPlan, assessment: ValidationAssessment | None = None):
    """Freeze integrity is required; practice qualification is not task admission.

    The host also verifies the selection event against select_candidate or the
    preregistered select_budget_fallback and its immutable publication ledger.
    """
    freeze = FreezeManifest.model_validate(freeze)
    plan = EvaluationPlan.model_validate(plan)
    for key in ('gym_id', 'parent_task_id', 'parent_release_sha256', 'public_rules_sha256', 'condition_sha256', 'source_manifest_sha256'):
        if getattr(freeze, key) != getattr(plan, key):
            raise ValueError('Frozen identity differs from its validation plan')
    if freeze.validation_plan_sha256 != content_sha256(plan):
        raise ValueError('Freeze changes its preregistered plan')
    if len(freeze.validation_event_ids) != len(set(freeze.validation_event_ids)):
        raise ValueError('Duplicate validation event identity')
    if assessment is None:
        if (freeze.validation_assessment_sha256 is not None or freeze.validation_event_ids
            or freeze.validation_complete or freeze.immutable_validation_complete
            or freeze.qualified_for_declared_curriculum or freeze.selection_mode != 'budget_fallback'
            or 'no_validation_assessment' not in freeze.qualification_reasons):
            raise ValueError('Unassessed freeze cannot invent validation or qualification')
    else:
        assessment = ValidationAssessment.model_validate(assessment)
        if (assessment.plan_sha256 != content_sha256(plan)
            or freeze.validation_assessment_sha256 != content_sha256(assessment)
            or freeze.candidate_event_id != assessment.candidate_event_id or freeze.bundle_hash != assessment.bundle_hash
            or freeze.validation_complete != assessment.all_slots_resolved
            or freeze.immutable_validation_complete != assessment.immutable_execution_complete
            or not set(assessment.reasons).issubset(freeze.qualification_reasons)):
            raise ValueError('Freeze does not preserve its candidate assessment')
        if freeze.selection_mode == 'validated_rank' and (
            not freeze.validation_event_ids or not assessment.all_slots_resolved
            or not assessment.immutable_execution_complete
            or freeze.qualified_for_declared_curriculum != assessment.qualified_for_declared_curriculum):
            raise ValueError('Validated selection needs its complete unchanged candidate assessment')
    if freeze.selection_mode == 'budget_fallback' and (
        freeze.qualified_for_declared_curriculum or 'declared_budget_fallback' not in freeze.qualification_reasons):
        raise ValueError('Budget fallback must retain an explicit unqualified label')
    return True


def check_frozen_execution(freeze: FreezeManifest, *, bundle_hash: str, condition_sha256: str,
                           source_manifest_sha256: str, temporary_skill_changes: bool):
    if (bundle_hash != freeze.bundle_hash or condition_sha256 != freeze.condition_sha256
        or source_manifest_sha256 != freeze.source_manifest_sha256 or temporary_skill_changes is not False):
        raise ValueError('Frozen execution changed code, condition, source or strategy')
    return True


def public_validation_feedback(assessment: ValidationAssessment, *, batch_closed: bool):
    if batch_closed is not True:
        raise ValueError('Validation feedback is withheld until the entire batch closes')
    return {'schema': 'card_rsi_validation_feedback.v1',
        'candidate_event_id': assessment.candidate_event_id, 'bundle_hash': assessment.bundle_hash,
        'successes': assessment.successes, 'failures': assessment.failures, 'unknown': assessment.unknown,
        'coverage_successes': dict(assessment.coverage_successes),
        'qualified_for_declared_curriculum': assessment.qualified_for_declared_curriculum,
        'reasons': list(assessment.reasons), 'scope': 'declared_practice_skills_only',
        'formal_match_qualified': False}


class RevisionReceipt(HostContract):
    """Host-verified authorship and feedback edges, not model self-reports."""
    schema: Literal['card_rsi_revision_receipt.v1'] = 'card_rsi_revision_receipt.v1'
    receipt_sequence: Count
    author: Literal['model', 'human', 'scripted', 'unknown']
    previous_bundle_hash: Hash
    candidate_event_id: Name
    bundle_hash: Hash
    changed_executable_content: StrictBool
    model_request_sha256: Hash
    model_response_sha256: Hash
    actual_model_requests: Count
    practice_attempt_ids: list[Name] = Field(min_length=1)
    public_feedback_sha256: list[Hash] = Field(min_length=1)

    @model_validator(mode='after')
    def unique_edges(self):
        if len(set(self.practice_attempt_ids)) != len(self.practice_attempt_ids):
            raise ValueError('Duplicate practice evidence edge')
        if len(set(self.public_feedback_sha256)) != len(self.public_feedback_sha256):
            raise ValueError('Duplicate feedback evidence edge')
        return self


class FrozenMatchReceipt(HostContract):
    schema: Literal['card_rsi_frozen_match_receipt.v1'] = 'card_rsi_frozen_match_receipt.v1'
    receipt_sequence: Count
    episode_id: Name
    parent_task_id: Task
    parent_release_sha256: Hash
    bundle_hash: Hash
    condition_sha256: Hash
    source_manifest_sha256: Hash
    result_sha256: Hash
    outcome: Literal['win', 'loss', 'draw', 'incomplete', 'unknown']
    physics_valid: StrictBool | None
    evidence_source: Literal['native_solver_substeps', 'synthetic', 'unverified']
    native_substeps: Count
    native_evidence_sha256: Hash | None
    skill_invocations: Count
    skill_language_entries: Count = 0
    executed_bundle_hashes: list[Hash]
    temporary_skill_changes: StrictBool
    fresh_context: StrictBool
    author_history_visible: StrictBool

    @model_validator(mode='after')
    def execution_counts(self):
        if self.skill_language_entries > self.skill_invocations:
            raise ValueError('Language entries cannot exceed invocation attempts')
        return self


class ClosedLoopAssessment(HostContract):
    schema: Literal['card_rsi_closed_loop_assessment.v1'] = 'card_rsi_closed_loop_assessment.v1'
    autonomous_revision_verified: StrictBool
    independent_validation_verified: StrictBool
    frozen_match_executed: StrictBool
    formal_match_completed: StrictBool
    closed_loop_executed: StrictBool
    performance_improvement_established: Literal[False] = False
    reasons: list[str]


def audit_closed_loop(*, plan: EvaluationPlan, practice: list[TrialRecord], revision: RevisionReceipt,
                      validation: list[TrialRecord], assessment: ValidationAssessment,
                      freeze: FreezeManifest, match: FrozenMatchReceipt) -> ClosedLoopAssessment:
    """Check host-verified evidence edges; never infer improvement from a revision.

    The trust boundary is the scheduler's artifact/native/API verifier, not this
    pure reducer. A changed file or a nonzero counter cannot prove its own
    authorship. This checks recorded independent stages and temporal order after
    that verification; a failed/incomplete physical match is still a real stage.
    """
    plan = EvaluationPlan.model_validate(plan)
    revision = RevisionReceipt.model_validate(revision)
    freeze = FreezeManifest.model_validate(freeze)
    match = FrozenMatchReceipt.model_validate(match)
    practice = [TrialRecord.model_validate(r) for r in practice]
    validation = [TrialRecord.model_validate(r) for r in validation]
    assessment = ValidationAssessment.model_validate(assessment)
    reasons = []
    indexed = {r.attempt_id: r for r in practice}
    if len(indexed) != len(practice):
        raise ValueError('Duplicate practice attempt receipt')
    linked = [indexed.get(key) for key in revision.practice_attempt_ids]
    practice_partition_ok = all(r.phase == 'practice' and r.case_fingerprint in plan.practice_case_fingerprints
                               for r in practice)
    practice_ok = practice_partition_ok and all(r is not None and r.phase == 'practice' and r.native_result()
        and r.skill_language_entries > 0 and r.executed_bundle_hashes == [revision.previous_bundle_hash]
        and not r.temporary_skill_changes and r.bundle_hash == revision.previous_bundle_hash
        and r.receipt_sequence < revision.receipt_sequence
        and r.gym_id == plan.gym_id and r.parent_release_sha256 == plan.parent_release_sha256
        and r.public_rules_sha256 == plan.public_rules_sha256 and r.condition_sha256 == plan.condition_sha256
        and r.source_manifest_sha256 == plan.source_manifest_sha256
        and r.case_fingerprint in plan.practice_case_fingerprints for r in linked)
    feedback = {r.public_feedback_sha256 for r in linked if r is not None}
    revision_ok = bool(practice_ok and None not in feedback and feedback == set(revision.public_feedback_sha256)
        and revision.author == 'model' and revision.actual_model_requests > 0
        and revision.changed_executable_content and revision.previous_bundle_hash != revision.bundle_hash)
    if not revision_ok:
        reasons.append('missing_verified_model_revision_from_native_practice_feedback')
    recomputed = assess_validation(plan, validation, candidate_event_id=revision.candidate_event_id,
        bundle_hash=revision.bundle_hash, condition_sha256=plan.condition_sha256)
    validation_ok = (content_sha256(recomputed) == content_sha256(assessment)
        and assessment.all_slots_resolved and assessment.immutable_execution_complete
        and all(r.receipt_sequence > revision.receipt_sequence for r in validation))
    if not validation_ok:
        reasons.append('missing_independent_post_revision_validation')
    try:
        validate_freeze(freeze, plan, assessment)
        check_frozen_execution(freeze, bundle_hash=match.bundle_hash, condition_sha256=match.condition_sha256,
            source_manifest_sha256=match.source_manifest_sha256, temporary_skill_changes=match.temporary_skill_changes)
        freeze_ok = (all(r.receipt_sequence < freeze.freeze_sequence for r in validation)
            and match.receipt_sequence > freeze.freeze_sequence)
    except ValueError:
        freeze_ok = False
    match_ok = bool(freeze_ok and match.parent_task_id == plan.parent_task_id
        and match.parent_release_sha256 == plan.parent_release_sha256
        and match.evidence_source == 'native_solver_substeps' and match.physics_valid is True
        and match.native_substeps > 0 and match.native_evidence_sha256
        and match.skill_language_entries > 0 and match.executed_bundle_hashes == [freeze.bundle_hash]
        and match.fresh_context and not match.author_history_visible)
    if not match_ok:
        reasons.append('missing_unchanged_frozen_skill_execution_in_native_formal_match')
    return ClosedLoopAssessment(autonomous_revision_verified=revision_ok,
        independent_validation_verified=validation_ok, frozen_match_executed=match_ok,
        formal_match_completed=match_ok and match.outcome in ('win', 'loss', 'draw'),
        closed_loop_executed=revision_ok and validation_ok and match_ok, reasons=reasons)
