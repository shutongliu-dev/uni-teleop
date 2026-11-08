"""Host preparation of fixed practice/validation identities and resource limits."""
from __future__ import annotations

from pathlib import Path
import hashlib
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .assessment import (CURRICULUM_STRATA, Coverage, EvaluationPlan,
                         ValidationSlot, content_sha256)


class StudyBudget(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)
    practice_episodes: int = Field(default=4, ge=1, le=256)
    validation_batches: int = Field(default=1, ge=1, le=16)
    max_decisions_per_episode: int = Field(default=32, ge=1, le=256)
    max_actions_per_episode: int = Field(default=64, ge=1, le=7000)
    max_function_calls_per_episode: int = Field(default=2048, ge=1, le=100000)
    sim_seconds_per_episode: float | None = Field(default=45., gt=0, le=150)
    wall_seconds_per_episode: float | None = Field(default=None, gt=0, le=2400)
    max_action_sim_seconds: float = Field(default=20., gt=0)
    max_output_tokens: int | None = Field(default=None, ge=1, le=32768)
    match_max_decisions_per_player: int = Field(default=128, ge=1, le=256)
    match_max_actions: int = Field(default=256, ge=1, le=7000)
    match_commits_per_player: int | None = Field(default=1, ge=1)
    match_sim_seconds: float | None = Field(default=120., gt=0, le=150)
    match_wall_seconds: float | None = Field(default=None, gt=0, le=2400)

    @model_validator(mode='after')
    def batches_fit(self):
        if self.validation_batches > self.practice_episodes:
            raise ValueError('Every validation batch must follow at least one practice episode')
        return self


class PracticeSlot(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)
    slot_id: str
    selection: dict
    case: dict  # Host-only: contains private case fingerprints, never model input.


class StudyPlan(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)
    protocol: Literal['card_rsi_study_plan.v1'] = 'card_rsi_study_plan.v1'
    parent_task_id: str
    gym_id: str
    gym_release_sha256: str
    parent_release_sha256: str
    source_manifest_sha256: str
    continuation_sha256: str | None = Field(default=None, pattern=r'^[0-9a-f]{64}$')
    model: dict
    budget: StudyBudget
    practice_slots: list[PracticeSlot]
    validation: EvaluationPlan
    practice_budget_fallback: Literal['last_published'] = 'last_published'
    operator_manipulation_recipe: Literal[False] = False
    native_validation_claimed: Literal[False] = False

    @model_validator(mode='after')
    def counts(self):
        if len(self.practice_slots) != self.budget.practice_episodes:
            raise ValueError('Practice slots must match the declared episode budget')
        return self


def bind_continuation(plan: StudyPlan, checkpoint_sha256: str) -> StudyPlan:
    """Bind inherited experience without relabelling its original conditions."""
    if plan.continuation_sha256 is not None:
        raise ValueError('Study plan already binds a continuation')
    value = plan.model_dump(mode='json')
    value['continuation_sha256'] = checkpoint_sha256
    value['validation']['condition_sha256'] = content_sha256({
        'base_condition_sha256': plan.validation.condition_sha256,
        'continuation_sha256': checkpoint_sha256})
    return StudyPlan.model_validate(value)


def study_condition_sha256(plan: StudyPlan) -> str:
    """Recompute frozen conditions from plan inputs, including inherited state."""
    base = _base_condition(plan.model, plan.budget, plan.source_manifest_sha256,
                           plan.gym_release_sha256, plan.parent_release_sha256)
    return (base if plan.continuation_sha256 is None else content_sha256({
        'base_condition_sha256': base, 'continuation_sha256': plan.continuation_sha256}))


def _base_condition(model, budget, source_manifest_sha256, gym_release_sha256, parent_sha):
    return content_sha256({'model': model, 'budget': budget.model_dump(mode='json'),
        'source_manifest_sha256': source_manifest_sha256, 'gym_release': gym_release_sha256,
        'parent_release_sha256': parent_sha, 'visual_input': 'rgbd', 'history_mode': 'current_with_recall',
        'geometry_math': True, 'skill_language': 'agent_json_skill_ast.v1',
        'operator_manipulation_recipe': False})


def prepare_study_plan(gym_release: Path, *, parent_task_id, model,
                       source_manifest_sha256, budget=None, full_curriculum=False):
    """Read frozen inputs as the host; public goals still come from Bench at run time."""
    from bench.isaacsim.release import load_practice_inputs
    loaded = load_practice_inputs(Path(gym_release))
    return plan_from_cases(loaded['manifest'], loaded['cases'],
        gym_release_sha256=hashlib.sha256((Path(gym_release) / 'release_manifest.json').read_bytes()).hexdigest(),
        parent_task_id=parent_task_id, model=model, source_manifest_sha256=source_manifest_sha256,
        budget=budget or StudyBudget(), full_curriculum=full_curriculum)


def plan_from_cases(manifest, cases, *, parent_task_id, model, source_manifest_sha256,
                    gym_release_sha256, budget: StudyBudget, full_curriculum=False):
    """Pure plan construction, with deterministic side/stratum coverage ordering."""
    if parent_task_id not in CURRICULUM_STRATA:
        raise ValueError('Unsupported practice family')
    gym_id = parent_task_id + '_practice'
    if manifest['task_id'] != gym_id:
        raise ValueError('Practice release belongs to a different task')
    strata = CURRICULUM_STRATA[parent_task_id]
    selected_strata = strata if full_curriculum else strata[:1]
    train = []
    validation = []
    for key, case in cases.items():
        public = case['public']
        if public['gym_id'] != gym_id or public['stratum'] not in selected_strata:
            continue
        (train if public['split'] == 'train' else validation).append(case)
    def order(case):
        public = case['public']
        variant = int(public['case_id'].rsplit('_', 1)[1])
        return (variant, selected_strata.index(public['stratum']), public['active_side'])
    train.sort(key=order)
    validation.sort(key=order)
    if not train or not validation:
        raise ValueError('Frozen task has no complete practice and validation cases')
    practice = []
    for index in range(budget.practice_episodes):
        case = train[index % len(train)]
        public = case['public']
        practice.append(PracticeSlot(slot_id=f'practice_{index:04d}',
            selection={key: public[key] for key in ('active_side', 'case_id', 'split')}, case=case))
    slots, covered = [], set()
    for case in validation:
        public = case['public']
        coverage = (public['active_side'], public['stratum'])
        if coverage in covered:
            continue
        covered.add(coverage)
        slots.append(ValidationSlot(slot_id=f"{coverage[0]}_{coverage[1]}",
            **{key: public[key] for key in ('active_side', 'stratum', 'case_id', 'case_sha256')},
            case_fingerprint=case['case_fingerprint'], private_case_sha256=case['private_case_sha256']))
    expected = {(side, stratum) for side in ('left', 'right') for stratum in selected_strata}
    if covered != expected:
        raise ValueError('Validation cases do not cover both original hands')
    parent_sha = manifest['parent_release']['manifest_sha256']
    rule_hashes = {case['public']['public_rules_sha256'] for case in train + validation}
    if len(rule_hashes) != 1:
        raise ValueError('Case rules differ within a frozen study')
    condition = _base_condition(model, budget, source_manifest_sha256, gym_release_sha256, parent_sha)
    evaluation = EvaluationPlan(plan_id='validation_01', gym_id=gym_id, parent_task_id=parent_task_id,
        parent_release_sha256=parent_sha, public_rules_sha256=next(iter(rule_hashes)),
        condition_sha256=condition, source_manifest_sha256=source_manifest_sha256,
        scope='full_curriculum' if full_curriculum else 'integration_smoke', slots=slots,
        required_coverage=[Coverage(active_side=s.active_side, stratum=s.stratum) for s in slots],
        practice_case_fingerprints=sorted({slot.case['case_fingerprint'] for slot in practice}))
    return StudyPlan(parent_task_id=parent_task_id, gym_id=gym_id,
        gym_release_sha256=gym_release_sha256, parent_release_sha256=parent_sha,
        source_manifest_sha256=source_manifest_sha256, model=model, budget=budget,
        practice_slots=practice, validation=evaluation)
