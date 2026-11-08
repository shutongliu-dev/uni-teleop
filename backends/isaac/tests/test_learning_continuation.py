"""CPU continuation evidence/migration fixtures; no model or native execution."""
from copy import deepcopy
import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest

from agent.model_client.learning import continuation as c
from agent.model_client.learning.assessment import Effort, ValidationSlot, content_sha256, sum_effort
from agent.model_client.learning.curriculum import bind_continuation
from agent.model_client.learning.journal import atomic_write_json, read_json
from agent.model_client.learning.skills import SkillLibrary
from test_learning_curriculum import prepare
from test_learning_study import scripted_publication, unknown_record


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(path, value)


def parent_fixture(tmp_path, monkeypatch, *, completed=1):
    """Exported for continuation/auditor integration tests; source verifier injected."""
    root=tmp_path/'parent'; study=root/'study'; episodes=root/'episodes/gym'; job=root/'study_job'
    old=prepare(); new=old.model_copy(deep=True)
    new.source_manifest_sha256='e'*64;new.validation.source_manifest_sha256='e'*64
    new.model={**old.model, 'retry_policy':'gateway_transient_backoff_v1', 'max_retries':None}
    new.validation.condition_sha256=c.study_condition_sha256(new)
    snapshot=tmp_path/'old-source';snapshot.mkdir()
    save(snapshot/'source_snapshot_manifest.json',{'fixture':'CPU, no native source qualification'})
    (snapshot/'agent/model_client/learning').mkdir(parents=True)
    (snapshot/'agent/model_client/learning/audit_study.py').write_text('# CPU auditor fixture\n')
    # Keep the plan source digest a real manifest digest even though its complete
    # source verifier is explicitly mocked for this synthetic unit fixture.
    old.source_manifest_sha256=c._ref(snapshot/'source_snapshot_manifest.json')['sha256']
    old.validation.source_manifest_sha256=old.source_manifest_sha256
    old.validation.condition_sha256=c.study_condition_sha256(old)
    contract={'dependencies':{},'compatibility':{'task_ids':[old.gym_id,old.parent_task_id],
        'task_releases':{old.gym_id:old.gym_release_sha256,old.parent_task_id:old.parent_release_sha256}},
        'condition_sha256':old.validation.condition_sha256,'source_manifest_sha256':old.source_manifest_sha256,'model':old.model}
    library=SkillLibrary.create('player_0',learning_root=study/'libraries',contract=contract)
    candidate=scripted_publication(library,1)
    library.write_author('notes/unpublished.md','Own unfinished model draft\n',expected_revision=library.author_revision())
    ledger=[];feedback=[];costs=[];audit_trials=[];effort_trials=[];reports={};terminal={}
    for i in range(completed+1):
        slot=old.practice_slots[i];attempt=slot.slot_id;eid='episode'+str(i);folder=study/'player_0'/attempt
        failed=i==completed;cost=Effort(model_requests=i+1,model_logical_calls=i+1,
            physical_attempts=1,accepted_actions=1,completed_actions=1,native_sim_s=float(i+1))
        costs.append(cost)
        result={'outcome':'cancelled' if failed else 'success','physics_valid':True,'elapsed_sim_s':cost.native_sim_s}
        report={'episode_id':eid,'result':result,'finalization_error':None,'action_recovery_pending':False,
            'run_error':{'error_type':'ModelGatewayError','error_category':'http_status','http_status':429} if failed else None,
            'run_status':'infra_error' if failed else 'completed','task_outcome':'unknown' if failed else 'success'}
        reports[attempt]=report;terminal[eid]={'finished':True,'worker_exit_code':0}
        save(folder/'run_report.json',report);save(episodes/eid/'agent/result.json',result)
        save(episodes/eid/'private/physics_audit.json',{'physics_valid':True})
        ledger.append({'sequence':len(ledger),'kind':'trial_start','value':{'learner':'player_0','attempt_id':attempt,
            'phase':'practice','selection':slot.selection,'candidate_event_id':None}})
        ledger.append({'sequence':len(ledger),'kind':'trial_end','value':{'learner':'player_0','attempt_id':attempt,
            'episode_id':eid,'cost':cost.model_dump(mode='json')}})
        auditrow={'attempt_id':attempt,'phase':'practice','native_rechecked':False}
        if not failed:
            public=slot.case['public'];case=ValidationSlot(slot_id=slot.slot_id,
                **{k:public[k] for k in ('active_side','stratum','case_id','case_sha256')},
                case_fingerprint=slot.case['case_fingerprint'],private_case_sha256=slot.case['private_case_sha256'])
            record=unknown_record(old.validation,case,candidate_event_id=candidate['event_id'],
                bundle_hash=candidate['payload']['bundle_hash'],attempt_id=attempt,attempt_index=0,
                receipt_sequence=len(ledger),phase='practice')
            record=record.model_copy(update={'episode_id':eid,'outcome':'success','result_valid':True,
                'physics_valid':True,'evidence_source':'native_solver_substeps','native_substeps':2,
                'native_evidence_sha256':'1'*64,'result_sha256':'2'*64,'cost':cost})
            payload={'outcome':'success','attempt_id':attempt}
            save(folder/'host/trial_record.json',record.model_dump(mode='json'))
            save(folder/'host/public_learning_feedback.json',payload)
            feedback.append({'assessment':payload,'recent_public_executions':[]})
            ledger.append({'sequence':len(ledger),'kind':'practice_feedback','value':{'learner':'player_0','attempt_id':attempt,
                'public_feedback_sha256':content_sha256(payload),'publications':[{'candidate_event_id':candidate['event_id']}]}})
            auditrow.update(native_rechecked=True,record=record.model_dump(mode='json'))
        else:auditrow['issue']='missing_or_duplicate_practice_feedback'
        audit_trials.append(auditrow)
        effort_trials.append({'learner':'player_0','attempt_id':attempt,'verified':True,'native_cost_verified':True,
            'actual':cost.model_dump(mode='json'),'recorded':cost.model_dump(mode='json')})
    for row in ledger:save(study/'ledger'/f'{row["sequence"]:06d}.json',row)
    summary={'schema':'card_rsi_study_report.v1','status':'stopped_error','cost_accounting_complete':True,
        'unaccounted_attempts':[],'cost':sum_effort(costs).model_dump(mode='json'),
        'learners':{'player_0':{'public_feedback':feedback}}}
    completion={'state':'command_exited','exit_code':1,'completed_at_utc':'2026-10-29T00:00:00Z','elapsed_wall_s':4.,'session_id':99999999}
    save(study/'study_plan.json',old.model_dump(mode='json'));save(study/'study_report.json',summary)
    save(study/'source_verification.json',{'snapshot_root':str(snapshot)})
    save(job/'completion.json',completion)
    closure={'complete':True,'servers':{}}
    for kind,items in (('gym',terminal),('formal',{})):
        save(root/(kind+'_server_job')/'completion.json',{**completion,'exit_code':0})
        shutdown=root/(kind+'_server')/'server_shutdown.json';save(shutdown,{'episodes':items})
        closure['servers'][kind]={'complete':True,'shutdown':c._ref(shutdown),'episodes':items}
    save(root/'closure01.json',closure)
    audit={'schema':'card_rsi_study_audit.v1','status':'study_failed_exit','injected_test_collectors':False,
        'cost_accounting_complete':True,'unaccounted_attempts':[],'job_completion':{k:completion[k] for k in
            ('state','exit_code','completed_at_utc','elapsed_wall_s')},
        'runtime_source':{'verified':True,'manifest_sha256':old.source_manifest_sha256,'snapshot_root':str(snapshot)},
        'auditor_source':{'verified':True,'snapshot_root':str(snapshot),'manifest_sha256':old.source_manifest_sha256,
            'module_sha256':c._ref(snapshot/'agent/model_client/learning/audit_study.py')['sha256']},
        'artifact_sha256':{'study/'+name:c._ref(study/name)['sha256'] for name in
            ('study_plan.json','study_report.json','source_verification.json')},
        'issues':['both_verified_freezes_required'],
        'effort_audit':{'complete':True,'trials':effort_trials},'players':{
            'player_0':{'trials':audit_trials,'issues':[f'practice_{completed:04d}:missing_or_duplicate_practice_feedback','FileNotFoundError']},
            'player_1':{'trials':[],'issues':['FileNotFoundError']}}}
    audit_path=root/'audit01/audit_report.json';save(audit_path,audit)
    request=tmp_path/'request.json';save(request,{'schema':'card_rsi_continuation_request.v1','study':str(study),
        'study_job':str(job),'audit_report':str(audit_path),'practice_episodes':str(episodes)})
    import scripts.maintenance.background_job as jobs
    import scripts.maintenance.freeze_project_sources as freeze
    monkeypatch.setattr(jobs,'status',lambda p:{**read_json(Path(p)/'completion.json'),
        'live_session_members':[{'pid':123}] if (Path(p)/'live').exists() else []})
    monkeypatch.setattr(freeze,'verify',lambda p:{'fixture':True})
    monkeypatch.setattr(c,'_source_record',lambda *a,**k:{'snapshot_root':str(snapshot)})
    costmap={f'practice_{i:04d}':v for i,v in enumerate(costs)}
    monkeypatch.setattr(c,'collect_effort',lambda folder,**kw:costmap[Path(folder).name])
    return SimpleNamespace(root=root,study=study,request=request,plan=new,old_plan=old,audit_path=audit_path,
        job=job,library=library,episodes=episodes,summary=summary,costs=costs,reports=reports,candidate=candidate)


def new_library(fixture,continuation,player='player_0'):
    plan=bind_continuation(fixture.plan,continuation.checkpoint_sha256)
    return SkillLibrary.create(player,learning_root=fixture.root.parent/'new-libraries',contract={
        'dependencies':{},'compatibility':fixture.library.contract['compatibility'],
        'model':plan.model,'source_manifest_sha256':plan.source_manifest_sha256,
        'condition_sha256':plan.validation.condition_sha256})


def test_prefix_cost_and_bound_plan_have_stable_checkpoint(tmp_path,monkeypatch):
    f=parent_fixture(tmp_path,monkeypatch,completed=2);cont=c.load_continuation(f.request,f.plan)
    assert cont.completed_slots('player_0')==('practice_0000','practice_0001')
    assert cont.completed_slots('player_1')==() and len(cont.prior_efforts('player_0'))==3
    assert sum(e.native_sim_s for e in cont.prior_efforts('player_0'))==6
    assert cont.manifest['interrupted_attempt']['slot_id']=='practice_0002'
    bound=bind_continuation(f.plan,cont.checkpoint_sha256)
    assert c.load_continuation(f.request,bound).manifest==cont.manifest
    feedback=cont.public_feedback('player_0');feedback.clear()
    assert len(cont.public_feedback('player_0'))==2


def test_import_and_unpublished_draft_preserve_origin_without_qualification(tmp_path,monkeypatch):
    f=parent_fixture(tmp_path,monkeypatch);cont=c.load_continuation(f.request,f.plan)
    old_bytes=c._tree_inventory(f.library.path);lib=new_library(f,cont)
    receipt=c.seed_library(cont,'player_0',lib)
    assert c._tree_inventory(f.library.path)==old_bytes
    assert lib.read_author('notes/unpublished.md')==f.library.read_author('notes/unpublished.md')
    assert receipt['imported_candidates'][0]['candidate_event_id']==f.candidate['event_id']
    imported=lib._candidate(f.candidate['event_id'])
    origin=read_json(lib.host/'origins'/(imported['payload']['import']['origin_sha256']+'.json'))
    assert origin['experiment']==f.library.experiment and origin['journal']==f.library.journal.read()
    assert not receipt['inherited_qualification'] and not any(e['kind']=='candidate_selected' for e in lib.journal.read())
    lib.write_author('notes/unpublished.md','New learner edit',expected_revision=lib.author_revision())
    assert c.verify_seed_library(cont,'player_0',lib)==receipt
    with pytest.raises(c.ContinuationError,match='not_empty'):c.seed_library(cont,'player_0',lib)


def test_unstarted_second_learner_remains_empty_but_has_seed_receipt(tmp_path,monkeypatch):
    f=parent_fixture(tmp_path,monkeypatch);cont=c.load_continuation(f.request,f.plan);lib=new_library(f,cont,'player_1')
    receipt=c.seed_library(cont,'player_1',lib)
    assert receipt['parent_library'] is None and receipt['imported_candidates']==[]
    assert read_json(lib.author/'catalog.json')=={'skills':[]}
    assert c.verify_seed_library(cont,'player_1',lib)==receipt


@pytest.mark.parametrize('change,reason',[
    (lambda f:setattr(f.plan.budget,'practice_episodes',3),'budget'),
    (lambda f:f.plan.model.update(model='other'),'routing'),
    (lambda f:setattr(f.plan,'gym_release_sha256','9'*64),'task_or_cases'),
    (lambda f:f.plan.practice_slots[0].case.update(reset={'changed':True}),'task_or_cases'),
    (lambda f:setattr(f.plan.validation.slots[0],'case_fingerprint','8'*64),'validation_cases'),
])
def test_incompatible_conditions_rejected(tmp_path,monkeypatch,change,reason):
    f=parent_fixture(tmp_path,monkeypatch);change(f)
    with pytest.raises(ValueError):c.load_continuation(f.request,f.plan)


@pytest.mark.parametrize('mutation,reason',[
    (lambda r:r['run_error'].update(http_status=503),'only_final_sealed_429'),
    (lambda r:r.update(action_recovery_pending=True),'not_closed'),
    (lambda r:r.update(finalization_error={'error_type':'Timeout'}),'not_closed'),
    (lambda r:r['result'].update(physics_valid=False),'not_closed'),
])
def test_only_explicit_closed_429_is_retryable(tmp_path,monkeypatch,mutation,reason):
    f=parent_fixture(tmp_path,monkeypatch);path=f.study/'player_0/practice_0001/run_report.json';r=read_json(path);mutation(r);save(path,r)
    with pytest.raises(c.ContinuationError,match=reason):c.load_continuation(f.request,f.plan)


def test_live_parent_and_missing_worker_closure_rejected(tmp_path,monkeypatch):
    f=parent_fixture(tmp_path,monkeypatch);(f.job/'live').touch()
    with pytest.raises(c.ContinuationError,match='process_not_closed'):c.load_continuation(f.request,f.plan)
    (f.job/'live').unlink();path=f.root/'closure01.json';x=read_json(path);x['complete']=False;save(path,x)
    with pytest.raises(c.ContinuationError,match='cleanup'):c.load_continuation(f.request,f.plan)


def test_parent_artifact_and_source_hash_change_fail_closed(tmp_path,monkeypatch):
    f=parent_fixture(tmp_path,monkeypatch);path=f.study/'study_report.json';x=read_json(path);x['cost']['totals']['model_requests']+=1;save(path,x)
    with pytest.raises(c.ContinuationError,match='artifact_changed'):c.load_continuation(f.request,f.plan)


def test_existing_validation_and_unresolved_publication_are_not_migrated(tmp_path,monkeypatch):
    f=parent_fixture(tmp_path,monkeypatch)
    f.library.journal.append('validation_recorded',{})
    with pytest.raises(c.ContinuationError,match='validation_or_selection'):c.load_continuation(f.request,f.plan)


def test_migration_rechecks_parent_and_fixed_author_archive(tmp_path,monkeypatch):
    f=parent_fixture(tmp_path,monkeypatch);cont=c.load_continuation(f.request,f.plan);lib=new_library(f,cont)
    f.library.write_author('notes/later.md','Do not silently import',expected_revision=f.library.author_revision())
    with pytest.raises(c.ContinuationError,match='changed_since_checkpoint'):c.seed_library(cont,'player_0',lib)


def test_seed_receipt_does_not_allow_forged_initial_draft(tmp_path,monkeypatch):
    f=parent_fixture(tmp_path,monkeypatch);cont=c.load_continuation(f.request,f.plan);lib=new_library(f,cont)
    c.seed_library(cont,'player_0',lib)
    path=lib.host/'continuation_author/notes/unpublished.md';path.chmod(0o600);path.write_text('forged')
    with pytest.raises(c.ContinuationError,match='initial_author_changed'):c.verify_seed_library(cont,'player_0',lib)


def test_paths_cannot_be_relative_or_aliased(tmp_path,monkeypatch):
    f=parent_fixture(tmp_path,monkeypatch);x=read_json(f.request);x['study']='relative-study';save(f.request,x)
    with pytest.raises(c.ContinuationError,match='absolute'):c.load_continuation(f.request,f.plan)


def test_unrelated_audit_failures_are_not_hidden_by_429(tmp_path,monkeypatch):
    f=parent_fixture(tmp_path,monkeypatch);value=read_json(f.audit_path)
    value['issues'].append('native_source_mismatch');save(f.audit_path,value)
    with pytest.raises(c.ContinuationError,match='unexpected_issue'):c.load_continuation(f.request,f.plan)


def test_seed_requires_new_bound_condition_and_release_contract(tmp_path,monkeypatch):
    f=parent_fixture(tmp_path,monkeypatch);cont=c.load_continuation(f.request,f.plan)
    contract={**f.library.contract,'source_manifest_sha256':f.plan.source_manifest_sha256,'model':f.plan.model}
    lib=SkillLibrary.create('player_0',learning_root=tmp_path/'wrong-condition',contract=contract)
    with pytest.raises(c.ContinuationError,match='condition_mismatch'):c.seed_library(cont,'player_0',lib)
    plan=bind_continuation(f.plan,cont.checkpoint_sha256)
    contract={**contract,'condition_sha256':plan.validation.condition_sha256,
              'compatibility':{'task_ids':['other'], 'task_releases':{}}}
    lib=SkillLibrary.create('player_0',learning_root=tmp_path/'wrong-release',contract=contract)
    with pytest.raises(c.ContinuationError,match='condition_mismatch'):c.seed_library(cont,'player_0',lib)
