"""Audit boundaries for lineage, budget credit and historical consumption."""
from copy import deepcopy
import json
import sys
from types import SimpleNamespace

import pytest

from agent.model_client.learning import audit_study as audit
from agent.model_client.learning.assessment import Effort, TrialRecord, content_sha256, sum_effort
from agent.model_client.learning.curriculum import bind_continuation, StudyPlan, study_condition_sha256
from agent.model_client.learning.receipts import _Reader, ReceiptError
from bench.paths import load_json
from test_learning_curriculum import prepare
from test_learning_study_audit import fixture_study, invoke, save


def lineage(tmp_path, monkeypatch):
    manifest = {'schema': 'synthetic_checkpoint', 'parent': 'fixture-parent'}
    checkpoint = content_sha256(manifest)
    plan = bind_continuation(prepare(practice_episodes=2), checkpoint)
    prior = Effort(model_requests=5, model_logical_calls=4, native_sim_s=2., input_tokens=None)
    current = Effort(model_requests=1, model_logical_calls=1)
    cont = SimpleNamespace(manifest=manifest, checkpoint_sha256=checkpoint,
        prior_efforts=lambda player: [prior] if player == 'player_0' else [],
        completed_slots=lambda player: ('practice_0000',) if player == 'player_0' else ())
    calls = []
    def load(path, supplied_plan):
        calls.append((path, supplied_plan))
        return cont
    def verify(c, player, path):
        assert c is cont
        calls.append(('verify_seed', player, path))
        return {'learner': player, 'checkpoint_sha256': checkpoint}
    monkeypatch.setitem(sys.modules, 'agent.model_client.learning.continuation',
        SimpleNamespace(load_continuation=load, verify_seed_library=verify))
    save(tmp_path/'continuation_request.json', {'schema': 'synthetic_request'})
    save(tmp_path/'continuation.json', manifest)
    seed = {'learner': 'player_0', 'checkpoint_sha256': checkpoint}
    save(tmp_path/'libraries/player_0/host/continuation.json', seed)
    rows = []
    def event(kind, value): rows.append(dict(sequence=len(rows), kind=kind, value=value))
    event('library_continued', dict(learner='player_0', continuation_sha256=checkpoint,
                                   seed_receipt_sha256=content_sha256(seed)))
    event('practice_slot_inherited', dict(learner='player_0', attempt_id='practice_0000',
        selection=plan.practice_slots[0].selection, continuation_sha256=checkpoint))
    event('trial_start', dict(learner='player_0', attempt_id='practice_0001'))
    event('trial_end', dict(learner='player_0', attempt_id='practice_0001', cost=current.model_dump(mode='json')))
    summary = dict(continuation_sha256=checkpoint, inherited_qualification=False,
        inherited_cost=sum_effort([prior]).model_dump(mode='json'),
        cumulative_cost=sum_effort([prior, current]).model_dump(mode='json'))
    return SimpleNamespace(root=tmp_path, plan=plan, cont=cont, rows=rows, summary=summary,
                           calls=calls, reader=_Reader(tmp_path, 'study', {}))


def test_lineage_reloads_parent_and_checks_seed_without_claiming_qualification(tmp_path, monkeypatch):
    f = lineage(tmp_path, monkeypatch); report = {}
    result = audit._continuation(f.reader, f.plan, f.summary, f.rows, report)
    audit._continuation_library(f.reader, 'player_0', f.rows, result, report)
    assert result is f.cont and len(f.calls) == 2
    assert f.calls[0][1] is f.plan
    assert report['continuation']['inherited_qualification'] is False
    assert report['continuation']['libraries']['player_0']['verified'] is True
    assert f.summary['cumulative_cost']['totals']['model_requests'] == 6
    assert f.summary['cumulative_cost']['provider_usage']['input_tokens']['complete'] is False
    assert 'closed_loop_verified' not in report


@pytest.mark.parametrize('mutation', ['manifest', 'summary_identity', 'qualified', 'parent_cost',
    'cumulative_cost', 'missing_credit', 'duplicate_credit', 'wrong_learner', 'wrong_selection',
    'wrong_credit_identity', 'reexecuted_slot', 'condition', 'seed_learner', 'duplicate_seed', 'credit_before_seed'])
def test_continuation_cannot_hide_history_or_mint_budget_credit(tmp_path, monkeypatch, mutation):
    f = lineage(tmp_path, monkeypatch)
    if mutation == 'manifest': save(tmp_path/'continuation.json', {'schema': 'other'})
    elif mutation == 'summary_identity': f.summary['continuation_sha256'] = '0'*64
    elif mutation == 'qualified': f.summary['inherited_qualification'] = True
    elif mutation == 'parent_cost': f.summary['inherited_cost'] = sum_effort([]).model_dump(mode='json')
    elif mutation == 'cumulative_cost': f.summary['cumulative_cost'] = f.summary['inherited_cost']
    elif mutation == 'missing_credit': f.rows.pop(1)
    elif mutation == 'duplicate_credit': f.rows.insert(2, deepcopy(f.rows[1]))
    elif mutation == 'wrong_learner': f.rows[1]['value']['learner'] = 'foreign'
    elif mutation == 'wrong_selection': f.rows[1]['value']['selection'] = {}
    elif mutation == 'wrong_credit_identity': f.rows[1]['value']['continuation_sha256'] = '0'*64
    elif mutation == 'reexecuted_slot': f.rows[2]['value']['attempt_id'] = 'practice_0000'
    elif mutation == 'seed_learner': f.rows[0]['value']['learner'] = 'foreign'
    elif mutation == 'duplicate_seed': f.rows.append(deepcopy(f.rows[0]))
    elif mutation == 'credit_before_seed': f.rows[1]['sequence'] = -1
    else: f.plan.validation.condition_sha256 = '0'*64
    with pytest.raises(ReceiptError): audit._continuation(f.reader, f.plan, f.summary, f.rows, {})


def test_parent_reverification_failure_is_not_replaced_by_stored_manifest(tmp_path, monkeypatch):
    f = lineage(tmp_path, monkeypatch)
    def changed(*args): raise ValueError('parent_artifact_changed')
    monkeypatch.setattr(sys.modules['agent.model_client.learning.continuation'], 'load_continuation', changed)
    with pytest.raises(ValueError, match='parent_artifact_changed'):
        audit._continuation(f.reader, f.plan, f.summary, f.rows, {})


def test_seed_verification_and_ledger_binding_are_both_required(tmp_path, monkeypatch):
    f = lineage(tmp_path, monkeypatch); report = {}
    audit._continuation(f.reader, f.plan, f.summary, f.rows, report)
    f.rows[0]['value']['seed_receipt_sha256'] = '0'*64
    with pytest.raises(ReceiptError, match='continuation_seed_ledger_changed'):
        audit._continuation_library(f.reader, 'player_0', f.rows, f.cont, report)
    def changed(*args): raise ValueError('seed_origin_changed')
    monkeypatch.setattr(sys.modules['agent.model_client.learning.continuation'], 'verify_seed_library', changed)
    with pytest.raises(ValueError, match='seed_origin_changed'):
        audit._continuation_library(f.reader, 'player_0', f.rows, f.cont, report)


def test_legacy_plan_cannot_gain_undeclared_inherited_credits(tmp_path):
    reader = _Reader(tmp_path, 'study', {})
    assert audit._continuation(reader, prepare(), {}, [], {}) is None
    with pytest.raises(ReceiptError, match='undeclared_study_continuation'):
        audit._continuation(reader, prepare(), {}, [dict(kind='practice_slot_inherited')], {})


def test_frozen_development_cost_includes_only_this_learners_prior_effort(tmp_path, monkeypatch):
    cfg = fixture_study(tmp_path, monkeypatch)
    reader = _Reader(cfg.output, 'study', {})
    plan = StudyPlan.model_validate(reader.json('study_plan.json'))
    rows = audit._ledger(reader)
    player = 'player_0'
    records = [TrialRecord.model_validate(reader.json(str(p.relative_to(cfg.output))))
               for p in sorted((cfg.output/player).glob('*/host/trial_record.json'))]
    prior = Effort(model_requests=5, input_tokens=None)
    path = cfg.output/player/'freeze_manifest.json'
    freeze = load_json(path)
    costs = [Effort.model_validate(r['value']['cost']) for r in rows[:freeze['freeze_sequence']]
             if r['kind'] == 'trial_end' and r['value']['learner'] == player]
    freeze['development_cost'] = sum_effort([prior, *costs]).model_dump(mode='json')
    save(path, freeze)
    checked, _, _ = audit._freeze(reader, player, plan, rows, records, prior_effort=[prior])
    assert checked.development_cost.totals['model_requests'] == 5
    with pytest.raises(ReceiptError, match='freeze_cost_mismatch'):
        audit._freeze(reader, player, plan, rows, records)


def test_failed_lineage_recheck_cannot_leave_accounting_complete_true(tmp_path, monkeypatch):
    cfg = fixture_study(tmp_path, monkeypatch)
    reader = _Reader(cfg.output, 'study', {})
    manifest = {'schema': 'synthetic_invalid_parent'}
    plan = bind_continuation(StudyPlan.model_validate(reader.json('study_plan.json')), content_sha256(manifest))
    save(cfg.output/'study_plan.json', plan.model_dump(mode='json'))
    save(cfg.output/'continuation_request.json', {'schema': 'synthetic_request'})
    save(cfg.output/'continuation.json', manifest)
    def changed(*args): raise ValueError('parent_artifact_changed')
    monkeypatch.setitem(sys.modules, 'agent.model_client.learning.continuation',
                        SimpleNamespace(load_continuation=changed))
    result = invoke(cfg)
    assert result['cost_accounting_complete'] is False
    assert result['continuation']['verified'] is False
    assert result['closed_loop_verified'] is False and result['formal_match_completed'] is False


@pytest.mark.parametrize('mutation', [None, 'later_author_edit', 'parent_report', 'seed_archive', 'double_cost'])
def test_real_checkpoint_loader_and_seed_verifier_feed_the_audit(tmp_path, monkeypatch, mutation):
    from agent.model_client.learning import continuation as c
    from agent.model_client.learning.skills import SkillLibrary
    from test_learning_continuation import parent_fixture
    parent = parent_fixture(tmp_path, monkeypatch)
    # The source/model edits made by this synthetic parent fixture need a
    # freshly computed base condition, as prepare_study_plan supplies in runs.
    parent.plan.validation.condition_sha256 = study_condition_sha256(parent.plan)
    cont = c.load_continuation(parent.request, parent.plan)
    plan = bind_continuation(parent.plan, cont.checkpoint_sha256)
    study = tmp_path/'child-study'
    save(study/'continuation_request.json', load_json(parent.request))
    save(study/'continuation.json', cont.manifest)
    rows = []; libraries = {}
    for player in ('player_0', 'player_1'):
        libraries[player] = SkillLibrary.create(player, learning_root=study/'libraries', contract={
            'dependencies': {}, 'compatibility': parent.library.contract['compatibility'],
            'source_manifest_sha256': plan.source_manifest_sha256, 'model': plan.model,
            'condition_sha256': plan.validation.condition_sha256})
        seeded = c.seed_library(cont, player, libraries[player])
        rows.append(dict(sequence=len(rows), kind='library_continued', value=dict(learner=player,
            continuation_sha256=cont.checkpoint_sha256, seed_receipt_sha256=content_sha256(seeded))))
        for slot in cont.completed_slots(player):
            selection = next(s.selection for s in plan.practice_slots if s.slot_id == slot)
            rows.append(dict(sequence=len(rows), kind='practice_slot_inherited', value=dict(learner=player,
                attempt_id=slot, selection=selection, continuation_sha256=cont.checkpoint_sha256)))
    prior = [e for player in ('player_0', 'player_1') for e in cont.prior_efforts(player)]
    summary = dict(continuation_sha256=cont.checkpoint_sha256, inherited_qualification=False,
        inherited_cost=sum_effort(prior).model_dump(mode='json'), cumulative_cost=sum_effort(prior).model_dump(mode='json'))
    if mutation == 'parent_report':
        save(parent.study/'study_report.json', parent.summary | {'status': 'forged_status'})
    elif mutation == 'seed_archive':
        archived = libraries['player_0'].host/'continuation_author/notes/unpublished.md'
        archived.chmod(0o600)
        archived.write_text('forged seed')
    elif mutation == 'double_cost':
        summary['cumulative_cost'] = sum_effort(prior+prior).model_dump(mode='json')
    elif mutation == 'later_author_edit':
        lib = libraries['player_0']
        lib.write_author('notes/unpublished.md', 'New learner draft', expected_revision=lib.author_revision())
    reader = _Reader(study, 'study', {}); report = {}
    def verify():
        checked = audit._continuation(reader, plan, summary, rows, report)
        for player in ('player_0', 'player_1'):
            audit._continuation_library(reader, player, rows, checked, report)
    if mutation in ('parent_report', 'seed_archive', 'double_cost'):
        with pytest.raises((ValueError, RuntimeError)): verify()
        assert report.get('continuation', {}).get('verified') is not True
    else:
        verify()
        assert report['continuation']['verified'] is True
        assert report['continuation']['inherited_qualification'] is False
        assert libraries['player_0'].selected() is None
