"""Success and time metrics with explicit timeout/failure denominators."""
from __future__ import annotations

import hashlib
import json
import math
import statistics
from collections import Counter, defaultdict

from .schemas import EpisodeResult, EvaluationPlan
from .task_scoring import SCORABLE_OUTCOMES

# Outcomes that say nothing about the agent: excluded from scores, reported apart.
UNSCORED_OUTCOMES = ("infrastructure_error", "evaluator_invalid")


def comparison_key(result: EpisodeResult) -> str:
    """Never pool robots, physics, policy, observation or timing regimes."""
    s = result.spec
    fields = s.model_dump(mode="json", exclude={"episode_id", "task_id", "seed", "instruction", "scene_sha256"})
    if result.scoring is not None:
        fields["scoring_contract"] = result.scoring.model_dump(
            mode="json", include={"mode", "objective", "score_unit", "metric_units"})
    return hashlib.sha256(json.dumps(fields, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _wilson(successes: int, count: int) -> list[float] | None:
    if not count:
        return None
    z = 1.959963984540054
    p = successes / count
    d = 1 + z * z / count
    center = (p + z * z / (2 * count)) / d
    radius = z * math.sqrt((p * (1-p) + z*z / (4*count)) / count) / d
    return [max(0.0, center-radius), min(1.0, center+radius)]


def _summary(rows: list[EpisodeResult]) -> dict:
    if rows and rows[0].scoring is not None:
        return _continuous_summary(rows)
    # An evaluator that latched invalid evidence cannot score the episode, so it
    # is accounted like an infrastructure failure, never as an agent failure.
    infrastructure = [r for r in rows if r.outcome in UNSCORED_OUTCOMES]
    scored = [r for r in rows if r.outcome not in UNSCORED_OUTCOMES]
    success = [r for r in scored if r.outcome == "success"]
    # Failing quickly cannot increase success/time: failures consume the full
    # preregistered budget in the denominator. Actual elapsed time is kept below.
    # A disabled horizon has no finite failure penalty. Do not replace it with
    # elapsed time or infinity and make unlimited runs appear more efficient.
    sim_cost = (sum(r.completion_sim_s if r.outcome == "success" else r.spec.budgets.sim_time_s for r in scored)
                if all(r.spec.budgets.sim_time_s is not None for r in rows) else None)
    wall_cost = (sum(r.completion_wall_s if r.outcome == "success" else r.spec.budgets.wall_time_s for r in scored)
                 if all(r.spec.budgets.wall_time_s is not None for r in rows) else None)
    n = len(scored)
    return dict(
        attempted_episodes=len(rows), scored_episodes=n, successes=len(success),
        infrastructure_errors=len(infrastructure),
        # Infrastructure failures require a declared rerun; the report is
        # descriptive and not leaderboard eligible until every planned slot runs.
        leaderboard_eligible=(not infrastructure and all(
            r.physics_valid and all(getattr(r.spec, field) is not None for field in
                ('runtime_code_sha256', 'sensor_config_sha256', 'scene_sha256',
                 'execution_environment_sha256')) for r in rows)),
        success_rate=len(success)/n if n else None,
        success_rate_wilson95=_wilson(len(success), n),
        outcome_counts=dict(Counter(r.outcome for r in rows)),
        mean_completion_sim_s_success_only=statistics.mean(r.completion_sim_s for r in success) if success else None,
        median_completion_sim_s_success_only=statistics.median(r.completion_sim_s for r in success) if success else None,
        mean_completion_wall_s_success_only=statistics.mean(r.completion_wall_s for r in success) if success else None,
        median_completion_wall_s_success_only=statistics.median(r.completion_wall_s for r in success) if success else None,
        penalized_sim_time_s=sim_cost, penalized_wall_time_s=wall_cost,
        success_per_sim_second=len(success)/sim_cost if sim_cost is not None and sim_cost > 0 else None,
        success_per_wall_second=len(success)/wall_cost if wall_cost is not None and wall_cost > 0 else None,
        actual_elapsed_sim_s=sum(r.elapsed_sim_s for r in rows),
        actual_elapsed_wall_s=sum(r.elapsed_wall_s for r in rows),
        total_model_latency_s=sum(r.model_latency_s for r in rows) if all(r.model_accounting == "server_observed" for r in rows) else None,
        total_model_calls=sum(r.model_calls for r in rows) if all(r.model_accounting == "server_observed" for r in rows) else None,
        model_accounting_counts=dict(Counter(r.model_accounting for r in rows)),
        model_call_budget_verified=bool(rows) and all(r.model_accounting == "server_observed" for r in rows),
        total_action_attempts=sum(r.action_attempts for r in rows),
        invalid_physics_episodes=sum(not r.physics_valid for r in rows))


def _continuous_summary(rows: list[EpisodeResult]) -> dict:
    """Keep earned numeric scores separate from binary success denominators."""
    declaration = rows[0].scoring
    if any(r.scoring is None or any(getattr(r.scoring, key) != getattr(declaration, key)
           for key in ('mode', 'objective', 'score_unit', 'metric_units')) for r in rows):
        raise ValueError("Cannot pool different scoring modes, units or objectives")
    valid = [r for r in rows if r.scoring.score_valid and r.physics_valid
             and r.outcome in SCORABLE_OUTCOMES]
    values = [r.scoring.score for r in valid]
    return dict(
        scoring_mode="continuous", objective=declaration.objective, score_unit=declaration.score_unit,
        attempted_episodes=len(rows), scored_episodes=len(valid),
        invalid_score_episodes=len(rows)-len(valid),
        score_mean=statistics.mean(values) if values else None,
        score_median=statistics.median(values) if values else None,
        score_best=(max(values) if declaration.objective == "maximize" else min(values)) if values else None,
        metric_units=declaration.metric_units,
        metric_means={name: statistics.mean(r.scoring.metrics[name] for r in valid)
                      if valid else None for name in declaration.metric_units},
        successes=None, success_rate=None, success_rate_wilson95=None,
        infrastructure_errors=sum(r.outcome in UNSCORED_OUTCOMES for r in rows),
        leaderboard_eligible=(len(valid) == len(rows) and all(
            all(getattr(r.spec, field) is not None for field in
                ('runtime_code_sha256', 'sensor_config_sha256', 'scene_sha256',
                 'execution_environment_sha256')) for r in rows)),
        outcome_counts=dict(Counter(r.outcome for r in rows)),
        mean_completion_sim_s_success_only=None, median_completion_sim_s_success_only=None,
        mean_completion_wall_s_success_only=None, median_completion_wall_s_success_only=None,
        penalized_sim_time_s=None, penalized_wall_time_s=None,
        success_per_sim_second=None, success_per_wall_second=None,
        actual_elapsed_sim_s=sum(r.elapsed_sim_s for r in rows),
        actual_elapsed_wall_s=sum(r.elapsed_wall_s for r in rows),
        total_model_latency_s=sum(r.model_latency_s for r in rows)
            if all(r.model_accounting == "server_observed" for r in rows) else None,
        total_model_calls=sum(r.model_calls for r in rows)
            if all(r.model_accounting == "server_observed" for r in rows) else None,
        model_accounting_counts=dict(Counter(r.model_accounting for r in rows)),
        model_call_budget_verified=bool(rows) and all(r.model_accounting == "server_observed" for r in rows),
        total_action_attempts=sum(r.action_attempts for r in rows),
        invalid_physics_episodes=sum(not r.physics_valid for r in rows))


def aggregate_results(results: list[EpisodeResult | dict], *, expected_episode_ids: list[str] | None = None,
                      plan: EvaluationPlan | None = None) -> dict:
    rows = [r if isinstance(r, EpisodeResult) else EpisodeResult.model_validate(r) for r in results]
    ids = [r.spec.episode_id for r in rows]
    if plan is not None:
        expected_specs = {s.episode_id: s for s in plan.episodes}
        if expected_episode_ids is not None and set(expected_episode_ids) != set(expected_specs):
            raise ValueError("expected IDs disagree with the evaluation plan")
        expected_episode_ids = list(expected_specs)
        for row in rows:
            if row.spec.episode_id not in expected_specs or row.spec != expected_specs[row.spec.episode_id]:
                raise ValueError("episode spec does not match its preregistered plan slot")
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate episode IDs cannot be counted twice")
    slots = [(comparison_key(r), r.spec.task_id, r.spec.seed) for r in rows]
    if len(slots) != len(set(slots)):
        raise ValueError("duplicate task/seed within a comparison cohort; explicitly preregister repeats as distinct seeds")
    if expected_episode_ids is not None:
        if len(expected_episode_ids) != len(set(expected_episode_ids)):
            raise ValueError("expected episode IDs must be unique")
        extra = sorted(set(ids) - set(expected_episode_ids))
        if extra:
            raise ValueError(f"unplanned episode IDs: {extra}")
        missing = sorted(set(expected_episode_ids) - set(ids))
    else:
        missing = []
    groups = defaultdict(list)
    for row in rows:
        groups[comparison_key(row)].append(row)
    reports = []
    for key, group in sorted(groups.items()):
        tasks = defaultdict(list)
        for row in group:
            tasks[row.spec.task_id].append(row)
        task_reports = {name: _summary(values) for name, values in sorted(tasks.items())}
        summary = _summary(group)
        summary["leaderboard_eligible"] &= plan is not None and not missing
        for task_report in task_reports.values():
            task_report["leaderboard_eligible"] &= summary["leaderboard_eligible"]
        rates = [r["success_rate"] for r in task_reports.values() if r["success_rate"] is not None]
        reports.append(dict(comparison_key=key,
            fixed_configuration=group[0].spec.model_dump(mode="json", exclude={"episode_id", "task_id", "seed", "instruction", "scene_sha256"}),
            summary=summary, task_macro_success_rate=statistics.mean(rates) if rates else None,
            per_task=task_reports, episode_ids=[r.spec.episode_id for r in group],
            episode_slots=[dict(episode_id=r.spec.episode_id, task_id=r.spec.task_id,
                                seed=r.spec.seed, scene_sha256=r.spec.scene_sha256) for r in group]))
    return dict(protocol="agent_bench_metrics_v1", plan_id=plan.plan_id if plan else None,
                plan_pins_verified=plan is not None,
                planned_episode_count=len(expected_episode_ids) if expected_episode_ids is not None else None,
                missing_episode_ids=missing, complete=expected_episode_ids is not None and not missing,
                leaderboard_eligible=bool(reports) and all(r["summary"]["leaderboard_eligible"] for r in reports),
                cohorts=reports,
                definitions={"success": "Hidden frozen evaluator first verifies task success while physics is valid and within both clocks.",
                  "continuous_scoring": "Declared terminal numeric scores use separate cohorts by mode, objective, score unit and metric units; valid scores contribute to score_mean/median/best, while binary success rates and success/time efficiency are null. Invalid evidence and physical/finalization faults remain unscored.",
                  "completion_time": "First verified success time; unsuccessful episodes have null completion times.",
                  "success_per_second": "sum(success) / sum(success completion time, full preregistered horizon for each failure); sim and wall reported separately. A disabled (null) horizon makes that dimension's penalized cost and efficiency null; actual elapsed time remains recorded.",
                  "infrastructure_error": "Reported separately, excluded from descriptive score, makes cohort ineligible until its planned slot is validly rerun.",
                  "evaluator_invalid": "The frozen evaluator latched invalid evidence (e.g. incomplete contact sampling); accounted exactly like infrastructure_error. Object motion beyond the frozen speed bound is instead a scored safety_stop.",
                  "wall_clock": "Monotonic from ready/reset initial observation through terminal decision, including agent/model/network, policy and simulation wait; startup/warmup excluded and separately recorded.",
                  "execution_environment": "A pinned static hardware/software/resource-allocation declaration separates cohorts; its hash does not prove exclusive resources or absence of contention. Shared-resource wall timings are descriptive diagnostics.",
                  "ranking": "Report success rate with uncertainty first; report penalized wall and simulation efficiency alongside. Never mix observation/control/robot/policy cohorts."})
