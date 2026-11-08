"""CPU audit orchestration and fail-closed fixtures; never native qualification."""
import hashlib
import json

import pytest

from agent.model_client.learning import audit_study as audit, runner, receipts
from agent.model_client.learning.assessment import Effort, FrozenMatchReceipt, RevisionReceipt, content_sha256
from agent.model_client.learning.curriculum import plan_from_cases
from agent.model_client.learning.journal import read_json
from agent.model_client.player_models import load_player_models
from bench.protocol.schemas import Budgets
from test_learning_curriculum import inputs
from test_learning_study import config, scripted_publication, unknown_record


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def fixture_study(tmp_path, monkeypatch, *, claim_native=False, scripted_revision=False, publication=True):
    cfg = config(tmp_path)
    cfg.gym_episode_root.mkdir(parents=True)
    cfg.formal_episode_root.mkdir(parents=True)
    manifest, cases = inputs()
    monkeypatch.setattr(runner, 'prepare_study_plan', lambda _release, **kw: plan_from_cases(
        manifest, cases, gym_release_sha256='d'*64, **kw))
    sequence = {}
    def episode(config, *, learning_bindings, **kwargs):
        config.output.mkdir(parents=True)
        single = 'single' in learning_bindings
        players = ('single',) if single else ('player_0', 'player_1')
        model = runner.public_model_condition(load_player_models(cfg.player_models)[cfg.model_slot])
        model.pop('provider')  # Real gateway metadata omits it.
        settings = dict(budgets=Budgets(sim_time_s=config.sim_budget, wall_time_s=config.wall_budget,
            max_actions=config.max_actions, max_model_calls=config.max_decisions*len(players),
            max_action_sim_s=config.max_action_sim_s).model_dump(mode='json'),
            max_decisions_per_player=config.max_decisions,
            max_function_calls_per_player=config.max_function_calls, history_mode=config.history_mode,
            local_geometry_math={'enabled': True}, visual_input='rgbd', models={p: model for p in players})
        if not single:
            settings['validation_target'] = dict(commits_per_player=config.stop_after_commits_per_player,
                source='verified_public_operation_committed_events', pass_counts=False, game_completion_claimed=False)
        save(config.output/'agent_config.json', settings)
        if single and learning_bindings['single'].phase == 'practice' and publication:
            library = learning_bindings['single'].library
            sequence[library.path] = sequence.get(library.path, 0)+1
            scripted_publication(library, sequence[library.path])
        report = {'episode_id': config.output.name, 'result': {'physics_valid': True, 'elapsed_sim_s': 0.},
                  'run_status': 'completed', 'result_valid_for_scoring': False}
        save(config.output/'run_report.json', report)
        return report
    def trial(plan, slot, **kwargs):
        row = unknown_record(plan, slot, **kwargs)
        row.public_feedback_sha256 = content_sha256({'outcome': 'unknown'})
        if claim_native:
            # Deliberately forged host flags: only the injected-reader boundary
            # can accept this fixture, and the top-level native claim stays false.
            row = row.model_copy(update=dict(outcome='failure', physics_valid=True, result_valid=True,
                evidence_source='native_solver_substeps', native_substeps=2, native_evidence_sha256='f'*64,
                result_sha256='e'*64, skill_invocations=1,
                skill_language_entries=1, executed_bundle_hashes=[row.bundle_hash]))
        return row
    def revision(**kw):
        if not scripted_revision:
            return None
        events = audit._Reader(kw['library_root'], 'fixture', {}).journal('host/journal')
        bundle = next(e['payload']['bundle_hash'] for e in events if e['event_id'] == kw['candidate_event_id'])
        old = kw['practice_records'][0]
        return RevisionReceipt(receipt_sequence=kw['receipt_sequence'], author='model',
            previous_bundle_hash=old.bundle_hash, candidate_event_id=kw['candidate_event_id'], bundle_hash=bundle,
            changed_executable_content=True, model_request_sha256='a'*64, model_response_sha256='b'*64,
            actual_model_requests=1, practice_attempt_ids=[old.attempt_id],
            public_feedback_sha256=[old.public_feedback_sha256])
    monkeypatch.setattr(receipts, 'collect_revision_receipt', revision)
    runner.run_study(cfg, source_verification={'manifest_sha256': 'c'*64}, episode_runner=episode,
        receipt_collector=trial, effort_collector=lambda *a, **k: Effort(),
        match_effort_collector=lambda *a, **k: Effort(),
        feedback_projector=lambda *a: {'outcome': 'unknown'})
    monkeypatch.setattr(audit, '_source_record', lambda *a, **k: {'snapshot_root': '/fixture/source02'})
    return cfg


def invoke(cfg, *, trial=None, revision=None, match=None, job=None, effort=None, match_effort=None):
    return audit.audit_study(cfg.output, practice_episodes=cfg.gym_episode_root,
        formal_episodes=cfg.formal_episode_root, auditor_identity={'verified': False, 'scope': 'CPU fixture'},
        study_job=job, trial_collector=trial or (lambda *a, **k: read_json(k['agent_output']/'host/trial_record.json')),
        revision_collector=revision or (lambda **kw: None), match_collector=match or (lambda *a, **k: None),
        effort_collector=effort or (lambda *a, **k: Effort(**k)),
        match_effort_collector=match_effort or (lambda *a, **k: Effort(**k)))


def test_rechecks_all_trial_records_instead_of_archived_native_flags(tmp_path, monkeypatch):
    cfg = fixture_study(tmp_path, monkeypatch, claim_native=True)
    calls = []
    def reject_claim(plan, slot, **kw):
        calls.append((kw['phase'], kw['episode_root']))
        return unknown_record(plan, slot, **kw)
    report = invoke(cfg, trial=reject_claim)
    assert len(calls) == 8 and {phase for phase, _ in calls} == {'practice', 'validation'}
    assert all(path.parent == cfg.gym_episode_root for _, path in calls)
    assert not report['closed_loop_verified'] and not report['formal_match_completed']
    assert any('rechecked_trial_differs_from_archive' in issue
               for p in report['players'].values() for issue in p['issues'])
    assert report['runtime_source']['snapshot_root'] == '/fixture/source02'
    assert report['auditor_source']['scope'] == 'CPU fixture'


def test_recomputes_revision_wrappers_and_calls_formal_reader(tmp_path, monkeypatch):
    cfg = fixture_study(tmp_path, monkeypatch, claim_native=True)
    revisions, matches = [], []
    def revision(**kwargs):
        revisions.append(kwargs)
        kwargs['diagnostics'].update(verified=False, reason='synthetic_fixture')
        return None
    def match(*args, **kwargs):
        matches.append(kwargs)
        return None
    report = invoke(cfg, revision=revision, match=match)
    assert len(revisions) == 2 and len(matches) == 1
    assert all(len(row['practice_records']) == 1 for row in revisions)
    assert matches[0]['episode_root'] == cfg.formal_episode_root/'match'
    assert report['injected_test_collectors'] and not report['closed_loop_verified']


@pytest.mark.parametrize('diagnostic_mismatch', [False, True])
def test_both_reducers_run_but_injected_positive_flags_never_become_native_proof(tmp_path, monkeypatch, diagnostic_mismatch):
    cfg = fixture_study(tmp_path, monkeypatch, claim_native=True, scripted_revision=True)
    def revision(**kw):
        path = kw['agent_output']/'host'/('revision_'+kw['previous_candidate_event_id']+'.json')
        return RevisionReceipt.model_validate(read_json(path)['receipt'])
    def match(freezes, **kw):
        kw['diagnostics']['match_effort'] = Effort(model_requests=int(diagnostic_mismatch)).model_dump(mode='json') | {'wall_s': None, 'cpu_s': None}
        return {player: FrozenMatchReceipt(receipt_sequence=kw['receipt_sequence'], episode_id='match',
            parent_task_id=f.parent_task_id, parent_release_sha256=f.parent_release_sha256, bundle_hash=f.bundle_hash,
            condition_sha256=f.condition_sha256, source_manifest_sha256=f.source_manifest_sha256,
            result_sha256='a'*64, outcome='incomplete', physics_valid=True, evidence_source='native_solver_substeps',
            native_substeps=2, native_evidence_sha256='b'*64, skill_invocations=1, skill_language_entries=1,
            executed_bundle_hashes=[f.bundle_hash], temporary_skill_changes=False, fresh_context=True,
            author_history_visible=False) for player, f in freezes.items()}
    report = invoke(cfg, revision=revision, match=match)
    if diagnostic_mismatch:
        assert report['cost_accounting_complete'] is False
        assert 'formal_native_audit_effort_differs_from_summary' in report['issues']
    else:
        assert all(row['closed_loop']['closed_loop_executed'] for row in report['players'].values()), report
    assert report['injected_test_collectors'] is True and report['closed_loop_verified'] is False
    assert report['formal_match_completed'] is False


def test_frozen_global_development_prefix_cannot_be_rewritten(tmp_path, monkeypatch):
    cfg = fixture_study(tmp_path, monkeypatch)
    path = cfg.output/'ledger/000000.json'
    row = read_json(path)
    row['value']['extra_unrecorded_field'] = True
    save(path, row)
    report = invoke(cfg)
    assert all('frozen_development_ledger_mismatch' in row['issues'] for row in report['players'].values())
    assert not report['closed_loop_verified']


@pytest.mark.parametrize('change', ['budget', 'model', 'incomplete_cost', 'missing_revision'])
def test_changed_contracts_or_incomplete_evidence_do_not_qualify(tmp_path, monkeypatch, change):
    cfg = fixture_study(tmp_path, monkeypatch, claim_native=change == 'missing_revision')
    if change in ('budget', 'model'):
        path = cfg.output/'match/agent_config.json'
        row = read_json(path)
        if change == 'budget': row['budgets']['sim_time_s'] = 45.
        else: row['models']['player_0']['endpoint'] = 'https://wrong.invalid/v1/chat/completions'
        save(path, row)
    elif change == 'incomplete_cost':
        path = cfg.output/'study_report.json'
        row = read_json(path)
        row.update(cost_accounting_complete=False, unaccounted_attempts=[{'attempt_id': 'match'}])
        save(path, row)
    else:
        next((cfg.output/'player_0').glob('*/host/revision_*.json')).unlink()
    report = invoke(cfg)
    assert not report['closed_loop_verified']
    if change == 'budget': assert 'run_conditions_differ_from_plan' in report['issues']
    if change == 'model': assert 'run_model_condition_changed' in report['issues']
    if change == 'incomplete_cost': assert not report['cost_accounting_complete']
    if change == 'missing_revision': assert any('FileNotFoundError' in e for e in report['players']['player_0']['issues'])


def test_job_nonzero_with_report_keeps_failed_scope(tmp_path, monkeypatch):
    cfg = fixture_study(tmp_path, monkeypatch)
    job = tmp_path/'job'
    save(job/'completion.json', {'state': 'command_exited', 'exit_code': 3})
    report = invoke(cfg, job=job)
    assert report['study_status'] == 'stages_executed' and report['status'] == 'study_failed_exit'
    assert report['job_completion']['exit_code'] == 3 and not report['closed_loop_verified']


def test_wait_finishes_when_job_exits_without_study_report(tmp_path):
    study, job = tmp_path/'study', tmp_path/'job'
    study.mkdir(); job.mkdir()
    sleeps = []
    def sleep(seconds):
        sleeps.append(seconds)
        save(job/'completion.json', {'state': 'command_exited', 'exit_code': 1})
    result = audit.await_study_report(study, job, sleep=sleep)
    assert sleeps == [30] and not result['report_available']
    assert result['reason'] == 'study_job_finished_without_report'


def test_wait_returns_stopped_report_without_polling_forever(tmp_path):
    study, job = tmp_path/'study', tmp_path/'job'
    job.mkdir()
    save(study/'study_report.json', {'status': 'stopped_error'})
    save(job/'completion.json', {'state': 'command_exited', 'exit_code': 1})
    result = audit.await_study_report(study, job, sleep=lambda _: pytest.fail('must not wait'))
    assert result['report_available']


def test_wait_requires_supervisor_completion_after_report_has_been_written(tmp_path):
    study, job = tmp_path/'study', tmp_path/'job'
    job.mkdir()
    save(study/'study_report.json', {'status': 'stages_executed'})
    sleeps = []
    def sleep(seconds):
        sleeps.append(seconds)
        save(job/'completion.json', {'state': 'command_exited', 'exit_code': 0})
    result = audit.await_study_report(study, job, sleep=sleep)
    assert sleeps == [30]
    assert result['report_available'] and result['job_completion']['exit_code'] == 0


def test_ledger_rejects_gaps(tmp_path):
    save(tmp_path/'ledger/000001.json', {'sequence': 1, 'kind': 'trial_start', 'value': {}})
    with pytest.raises(ValueError, match='study_ledger_gap'):
        audit._ledger(audit._Reader(tmp_path, 'study', {}))


def test_missing_cost_event_cannot_be_repaired_by_summary_flag(tmp_path, monkeypatch):
    cfg = fixture_study(tmp_path, monkeypatch)
    rows = audit._ledger(audit._Reader(cfg.output, 'study', {}))
    summary = read_json(cfg.output/'study_report.json')
    assert audit._accounting_complete(summary, rows)
    index = next(i for i, row in enumerate(rows) if row['kind'] == 'trial_end')
    rows.pop(index)
    assert not audit._accounting_complete(summary, rows)
    assert summary['cost_accounting_complete'] is True  # Original record untouched.


def test_cli_records_completed_job_missing_report_without_waiting_or_execution(tmp_path, monkeypatch):
    paths = {key: tmp_path/key for key in ('study', 'practice', 'formal', 'job', 'output')}
    for key, path in paths.items():
        if key != 'output': path.mkdir()
    save(paths['job']/'completion.json', {'state': 'command_exited', 'exit_code': 1})
    monkeypatch.setattr(audit, 'auditor_source_identity', lambda: {'verified': False, 'scope': 'CPU fixture'})
    monkeypatch.setattr('sys.argv', ['audit_study', '--study', str(paths['study']),
        '--practice-episodes', str(paths['practice']), '--formal-episodes', str(paths['formal']),
        '--output', str(paths['output']), '--wait-for-study-report', '--study-job', str(paths['job'])])
    audit.main()
    report = read_json(paths['output']/'audit_report.json')
    assert report['status'] == 'study_failed_exit' and not report['closed_loop_verified']
    assert report['wait_result']['reason'] == 'study_job_finished_without_report'
    assert report['runtime_source'] is None and report['auditor_source']['scope'] == 'CPU fixture'


@pytest.mark.parametrize('field,value', [('model_requests', 2), ('physical_attempts', 3),
    ('input_tokens', 100), ('native_sim_s', 4.5)])
def test_consistent_ledger_and_summary_cannot_hide_actual_trial_effort(tmp_path, monkeypatch, field, value):
    cfg = fixture_study(tmp_path, monkeypatch)
    report = invoke(cfg, effort=lambda *a, **k: Effort(**k, **{field: value}))
    assert report['reported_cost_accounting_complete'] is True
    assert report['cost_accounting_complete'] is False
    assert len(report['effort_audit']['trials']) == 8
    assert all(row['issue'] == 'actual_trial_effort_differs_from_ledger' for row in report['effort_audit']['trials'])
    assert not report['closed_loop_verified']


def test_no_publication_still_recounts_every_trial(tmp_path, monkeypatch):
    cfg = fixture_study(tmp_path, monkeypatch, publication=False)
    called = []
    def effort(folder, **kwargs):
        called.append(folder)
        return Effort(model_requests=1, **kwargs)
    report = invoke(cfg, effort=effort)
    assert len(called) == 2 and not list(cfg.output.glob('player_0/*/host/trial_record.json'))
    assert report['cost_accounting_complete'] is False and report['status'] == 'study_stopped'
    assert all(row['actual']['model_requests'] == 1 for row in report['effort_audit']['trials'])


def test_actual_match_effort_is_checked_even_if_native_qualification_is_missing(tmp_path, monkeypatch):
    cfg = fixture_study(tmp_path, monkeypatch)
    report = invoke(cfg, match_effort=lambda *a, **k: Effort(model_requests=2, **k))
    assert report['effort_audit']['match']['issue'] == 'actual_match_effort_differs_from_summary'
    assert report['cost_accounting_complete'] is False and not report['closed_loop_verified']


def test_native_trial_recheck_cost_must_also_match_the_ledger(tmp_path, monkeypatch):
    cfg = fixture_study(tmp_path, monkeypatch)
    def trial(*args, **kwargs):
        record = read_json(kwargs['agent_output']/'host/trial_record.json')
        record['cost']['model_requests'] = 7
        return record
    report = invoke(cfg, trial=trial)
    assert report['cost_accounting_complete'] is False
    assert any('trial_record_cost_differs_from_ledger' in issue
               for player in report['players'].values() for issue in player['issues'])


def test_real_receipt_counter_catches_zero_ledger_without_a_published_skill(tmp_path):
    folder = tmp_path/'player_0/practice_0000'
    metadata = {'attempts': 1, 'usage': {'input_tokens': 10, 'output_tokens': 2}}
    save(folder/'single/decisions/0000/model_response.json', {'metadata': metadata})
    save(folder/'run_report.json', {'episode_id': None, 'model_requests': {'single': [metadata]},
        'pre_episode_model_requests': [], 'physical_attempts': {}, 'function_calls': {}, 'result': None})
    rows = [{'sequence': 0, 'kind': 'trial_start', 'value': {'learner': 'player_0', 'attempt_id': 'practice_0000'}},
            {'sequence': 1, 'kind': 'trial_end', 'value': {'learner': 'player_0', 'attempt_id': 'practice_0000',
                'cost': Effort(wall_s=2., cpu_s=.2).model_dump(mode='json')}}]
    result = audit._recheck_effort(audit._Reader(tmp_path, 'study', {}), {}, rows,
        trial_collector=receipts.collect_effort, match_collector=receipts.collect_match_effort)
    assert not result['complete']
    actual = result['trials'][0]['actual']
    assert actual['model_requests'] == actual['model_logical_calls'] == 1
    assert actual['input_tokens'] == 10 and actual['output_tokens'] == 2
    assert actual['wall_s'] == 2. and actual['cpu_s'] == .2
    assert 'inherited host measurements' in result['clock_scope']


@pytest.mark.parametrize('match_started', [False, True])
def test_missing_formal_root_preserves_completed_learner_diagnostics(tmp_path, monkeypatch, match_started):
    cfg = fixture_study(tmp_path, monkeypatch)
    cfg.formal_episode_root.rmdir()
    if not match_started:
        rows = audit._ledger(audit._Reader(cfg.output, 'study', {}))
        first_match = next(row['sequence'] for row in rows if row['kind'] == 'frozen_match_start')
        for row in rows[first_match:]:
            (cfg.output/'ledger'/f"{row['sequence']:06d}.json").unlink()
        summary = read_json(cfg.output/'study_report.json')
        summary['status'] = 'stopped_error'
        summary.pop('match_cost')
        summary['cost'] = audit.sum_effort([
            Effort.model_validate(row['value']['cost']) for row in rows[:first_match] if row['kind'] == 'trial_end']).model_dump(mode='json')
        save(cfg.output/'study_report.json', summary)
    report = invoke(cfg)
    assert not cfg.formal_episode_root.exists()
    assert all(len(p['trials']) == 4 and p['development_ledger_verified']
               for p in report['players'].values())
    assert report['cost_accounting_complete'] is True
    assert not report['closed_loop_verified'] and not report['formal_match_completed']
    if match_started:
        assert 'FileNotFoundError' in report['issues']
    else:
        assert report['status'] == 'study_stopped'
        assert 'FileNotFoundError' not in report['issues']
        assert 'missing_or_duplicate_frozen_match_end' in report['issues']


def test_corrected_recheck_retains_original_classification_hash_and_freeze(tmp_path, monkeypatch):
    cfg = fixture_study(tmp_path, monkeypatch)
    paths = sorted(cfg.output.glob('player_*/*/host/trial_record.json'))
    freezes = sorted(cfg.output.glob('player_*/freeze_manifest.json'))
    originals = {p: p.read_bytes() for p in paths+freezes}
    def corrected(*args, **kwargs):
        value = read_json(kwargs['agent_output']/'host/trial_record.json')
        value.update(outcome='success', result_valid=True, physics_valid=True,
            evidence_source='native_solver_substeps', native_substeps=2, native_evidence_sha256='f'*64,
            public_feedback_sha256='a'*64, result_sha256='e'*64)
        return value
    report = invoke(cfg, trial=corrected)
    for player, detail in report['players'].items():
        archived_freeze = detail['archived_freeze']
        assert archived_freeze['development_ledger_verified'] is True
        assert archived_freeze['manifest'] == read_json(cfg.output/player/'freeze_manifest.json')
        for row in detail['trials']:
            correction = row['corrected_recheck']
            old = cfg.output/player/row['attempt_id']/'host/trial_record.json'
            assert correction['archived_file_sha256'] == hashlib.sha256(originals[old]).hexdigest()
            assert correction['archived_classification']['outcome'] == 'unknown'
            assert correction['rechecked_classification']['outcome'] == 'success'
            assert 'outcome' in correction['changed_fields']
            assert correction['accepted_for_original_qualification'] is False
            assert row['issue'] == 'rechecked_trial_differs_from_archive'
    assert all(p.read_bytes() == data for p, data in originals.items())
    assert not report['closed_loop_verified'] and not report['formal_match_completed']


def test_corrected_recheck_does_not_bypass_original_ledger_binding(tmp_path, monkeypatch):
    cfg = fixture_study(tmp_path, monkeypatch)
    path = cfg.output/'player_0/practice_0000/host/trial_record.json'
    value = read_json(path)
    value['public_feedback_sha256'] = '0'*64
    save(path, value)
    report = invoke(cfg)
    row = report['players']['player_0']['trials'][0]
    assert row['issue'] == 'trial_ledger_receipt_mismatch'
    assert 'corrected_recheck' not in row
    assert not report['closed_loop_verified']


@pytest.mark.parametrize('change', ['result_missing', 'finalization_timeout', 'elapsed_mismatch'])
def test_native_cost_requires_terminal_response_not_zero_counter_agreement(tmp_path, monkeypatch, change):
    cfg = fixture_study(tmp_path, monkeypatch)
    path = cfg.output/'player_0/practice_0000/run_report.json'
    value = read_json(path)
    if change == 'result_missing':
        value['result'] = None
    elif change == 'finalization_timeout':
        value['finalization_error'] = {'error_category': 'transport_timeout'}
    else:
        value['result']['elapsed_sim_s'] = 44.74
    save(path, value)
    report = invoke(cfg)
    row = report['effort_audit']['trials'][0]
    assert report['reported_cost_accounting_complete'] is True
    assert row['recorded']['native_sim_s'] == row['actual']['native_sim_s'] == 0.
    assert row['native_cost_verified'] is False and row['verified'] is False
    assert row['issue'] == ('actual_native_effort_differs_from_terminal_result' if change == 'elapsed_mismatch'
                            else 'native_effort_unavailable_without_terminal_result')
    assert report['cost_accounting_complete'] is False and not report['closed_loop_verified']
