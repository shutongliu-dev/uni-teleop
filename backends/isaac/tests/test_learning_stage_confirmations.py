"""Native-shaped CPU parser fixtures; no robot or physical success is claimed."""
from copy import deepcopy
import hashlib
import json

import pytest

from agent.model_client.learning.assessment import ValidationSlot
from bench.paths import load_json
from test_learning_receipts import fixture, collect, h, save, scene_hash


def seal_frames(f, rows):
    path = f.episode/'private/task_runtime/native_frames.jsonl'
    path.write_text(''.join(json.dumps(row)+'\n' for row in rows))
    evidence_path = f.episode/'private/task_runtime/evidence.json'
    evidence = load_json(evidence_path)
    evidence['native_frames_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
    save(evidence_path, evidence)


def completed_stages(f, monkeypatch, count=1):
    """600 genuine-sized settling intervals, then 33 latched success frames.

    This mirrors source03's final ACK shape, while retaining the fixture's
    explicit stubbed native provenance (not a physical validation claim).
    """
    task = load_json(f.episode/'private/task.json')
    case, public = task['practice_case'], task['practice_case']['public']
    public['goal']['stages'] *= count
    public['case_sha256'] = scene_hash({k: v for k, v in public.items() if k != 'case_sha256'})
    case['case_fingerprint'] = scene_hash(dict(parent_release_sha256=h('parent'), reset=case['reset'],
        active_side='left', goal=public['goal'], private_pairs=None))
    case['private_case_sha256'] = scene_hash({k: v for k, v in case.items() if k != 'private_case_sha256'})
    f.slot = ValidationSlot.model_validate(f.slot.model_dump() | dict(case_sha256=public['case_sha256'],
        case_fingerprint=case['case_fingerprint'], private_case_sha256=case['private_case_sha256']))
    f.plan.slots = [f.slot]
    save(f.episode/'private/task.json', task)
    monkeypatch.setattr('bench.isaacsim.release.task_inputs', lambda *a, **kw: ({}, {}, deepcopy(task), {}))
    rows, confirmations = [], []
    def add(reason, stage, success=False):
        row = deepcopy(f.frames[0])
        row['frame'].update(substep_index=len(rows)+3, simulation_time_s=(len(rows)+1)*.0005)
        row['result'] = dict(reason=reason, committed=False, practice_stage_index=stage, practice_success=success)
        rows.append(row)
        return row['frame']
    add('baseline', 0)
    for stage in range(count):
        start = (len(rows)+1)*.0005
        for _ in range(600):
            add('settling', stage)
        frame = add('success' if stage+1 == count else 'stage_confirmed', stage+1, stage+1 == count)
        confirmations.append(dict(stage_index=stage, native_substep=frame['substep_index'],
            simulation_time_s=frame['simulation_time_s'], stable_start_s=start))
    for _ in range(32):
        add('success', count, True)
    score = dict(success=True, reason='success',
        practice_progress=dict(stage_index=count, stage_count=count, confirmed_stages=count))
    evidence = deepcopy(f.evidence)
    evidence.update(score=score, confirmations=confirmations, case_sha256=public['case_sha256'],
        case_fingerprint=case['case_fingerprint'], private_case_sha256=case['private_case_sha256'])
    evidence['observer'].update(samples=len(rows), last_native_step=rows[-1]['frame']['substep_index'])
    save(f.episode/'private/task_runtime/evidence.json', evidence)
    seal_frames(f, rows)
    evaluator_sha = save(f.episode/'private/evaluator_evidence.json', score)
    save(f.episode/'private/final_score.json', score)
    result = f.report['result'] | dict(outcome='success', elapsed_sim_s=rows[-1]['frame']['simulation_time_s'],
        evaluator_evidence_sha256=evaluator_sha)
    save(f.episode/'agent/result.json', result)
    save(f.agent/'run_report.json', f.report | dict(result=result, task_outcome='success'))
    return rows


@pytest.mark.parametrize('stage_count', [1, 2])
def test_only_stage_transitions_confirm_despite_repeated_terminal_success(fixture, monkeypatch, stage_count):
    rows = completed_stages(fixture, monkeypatch, stage_count)
    assert sum(row['result']['reason'] == 'success' for row in rows) == 33
    record = collect(fixture)
    assert record.native_result() and record.outcome == 'success', record.verification_issues
    assert record.native_substeps == len(rows) and record.verification_issues == []


@pytest.mark.parametrize('kind,reason', [
    ('jump', 'practice_stage_jump_or_backtrack'),
    ('backtrack', 'practice_stage_jump_or_backtrack'),
    ('repeated_nonterminal', 'practice_confirmation_without_transition'),
    ('no_transition', 'practice_confirmation_without_transition'),
    ('premature_success', 'practice_stage_transition_reason_mismatch'),
    ('terminal_changed', 'practice_terminal_success_changed'),
    ('short_settling', 'practice_stage_without_stability_window'),
    ('drift', 'practice_stability_drift_mismatch'),
    ('terminal_contacts_missing', 'non_native_or_incomplete_frame'),
    ('terminal_clock_gap', 'native_substep_or_clock_gap'),
])
def test_stage_and_terminal_corruption_remains_unverified(fixture, monkeypatch, kind, reason):
    rows = completed_stages(fixture, monkeypatch, 2)
    first = next(i for i, row in enumerate(rows) if row['result']['reason'] == 'stage_confirmed')
    if kind == 'jump': rows[0]['result']['practice_stage_index'] = 2
    elif kind == 'backtrack': rows[first+1]['result']['practice_stage_index'] = 0
    elif kind == 'repeated_nonterminal': rows[first+1]['result']['reason'] = 'stage_confirmed'
    elif kind == 'no_transition': rows[5]['result']['reason'] = 'stage_confirmed'
    elif kind == 'premature_success': rows[first]['result'].update(reason='success', practice_success=True)
    elif kind == 'terminal_changed': rows[-1]['result']['practice_success'] = False
    elif kind == 'short_settling': rows[1]['result']['reason'] = 'release_support_required'
    elif kind == 'drift': rows[9]['frame']['objects']['black_mono']['position_m'][0] += .004
    elif kind == 'terminal_contacts_missing': rows[-1]['frame']['contacts_complete'] = False
    elif kind == 'terminal_clock_gap': rows[-1]['frame']['simulation_time_s'] += .001
    seal_frames(fixture, rows)
    record = collect(fixture)
    assert record.outcome == 'unknown' and record.evidence_source == 'unverified'
    assert record.verification_issues == ['native:'+reason]


@pytest.mark.parametrize('field,delta', [('stage_index', 1), ('native_substep', 1),
    ('simulation_time_s', .0005), ('stable_start_s', .0005)])
def test_confirmation_must_match_transition_and_exact_stability_start(fixture, monkeypatch, field, delta):
    completed_stages(fixture, monkeypatch)
    path = fixture.episode/'private/task_runtime/evidence.json'
    evidence = load_json(path)
    evidence['confirmations'][0][field] += delta
    save(path, evidence)
    record = collect(fixture)
    assert record.verification_issues == ['native:stage_confirmation_mismatch']
    assert not record.native_result()


def test_duplicate_stored_confirmation_is_not_silently_deduplicated(fixture, monkeypatch):
    completed_stages(fixture, monkeypatch)
    path = fixture.episode/'private/task_runtime/evidence.json'
    evidence = load_json(path)
    evidence['confirmations'].append(deepcopy(evidence['confirmations'][0]))
    save(path, evidence)
    assert collect(fixture).verification_issues == ['native:stage_confirmation_mismatch']


def test_terminal_success_still_requires_original_stream_hash(fixture, monkeypatch):
    rows = completed_stages(fixture, monkeypatch)
    rows[-1]['raw_contacts'].append({'altered': 'terminal frame is not skipped'})
    (fixture.episode/'private/task_runtime/native_frames.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in rows))
    assert collect(fixture).verification_issues == ['native:native_stream_hash_mismatch']
