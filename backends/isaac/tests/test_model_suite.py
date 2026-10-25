"""Development suite accounting and pin checks; no model or native simulator."""
from copy import deepcopy
from pathlib import Path
import threading

import pytest
from pydantic import ValidationError

from agent.model_client import suite
from agent.model_client.runner import CLIENT_SOURCE_DEPENDENCIES


@pytest.fixture
def environment(tmp_path, monkeypatch):
    project=tmp_path/'project'
    source=project/'agent/model_client/runner.py'
    source.parent.mkdir(parents=True)
    source.write_text('# fixed fake runner source\n')
    (project/'scripts').mkdir()
    (project/'scripts/isaac_python.sh').write_text('# fixed fake native launcher\n')
    for name in CLIENT_SOURCE_DEPENDENCIES:
        dependency = project/name
        dependency.parent.mkdir(parents=True, exist_ok=True)
        dependency.write_text('# fixed fake client dependency\n')
    monkeypatch.setattr(suite,'PROJECT',project)
    release=tmp_path/'release'
    (release/'evaluator').mkdir(parents=True)
    evaluator=release/'evaluator/sequence_evaluator.py'
    evaluator.write_text('# frozen evaluator\n')
    suite.save(release/'release_manifest.json',{'tasks':[{'id':'01_cube_push'}],
        'files':{'evaluator/sequence_evaluator.py':{'sha256':suite.digest(evaluator)}}})
    policy={'policy_id':'classical_fingertip','version':'1.0.1','trained':False,
        'implementation_sha256':'a'*64,'weights_sha256':None,'observation_profile':'vision_proprio',
        'skills':['move_geometry','hold'],'control_period_s':.02,'geometry_inputs':['wrist_pose','fingertip_positions'],
        'force_objectives':False,'camera_observation_mode':'decision_only','supported_robot_ids':['sharpa_w01_right']}
    inventory={'tasks':[{'task_id':'01_cube_push','instruction':'Push cube'}],'policy':policy}
    monkeypatch.setattr(suite,'service_inventory',lambda matrix:deepcopy(inventory))
    config={'task_release':str(release)}
    def plan(**changes):
        value={**config,**changes}
        directory=tmp_path/f'run_{len(list(tmp_path.glob("run_*")))}'
        result=suite.create_plan(value,directory)
        return directory/'plan.json',result
    return project,inventory,plan


def fake_runner(environment, *, outcome='cancelled', runtime_hash='b'*64, wrong_policy=False, error_type=None,
                metadata=None, elapsed_sim=.4, elapsed_wall=2., legacy_retry_metadata=False,
                terminal_scoring=None):
    project,inventory,_=environment
    calls=[]
    def run(args):
        calls.append(args)
        args.output.mkdir(parents=True)
        policy = (deepcopy(inventory['policies_by_task'][args.task] if 'policies_by_task' in inventory
                           else inventory['policy']) if args.control_mode in ('fingertip', 'hybrid') else None)
        if wrong_policy and policy:
            policy['version']='changed'
        settings={key:getattr(args,key)for key in ('model','reasoning_effort','control_mode','interaction_mode',
            'visual_input','tactile_input','max_decisions','max_actions','max_tool_calls_per_decision','max_program_actions',
            'max_output_tokens')}
        settings.update(model_timeout_s=args.model_timeout, http_timeout_s=args.http_timeout,
                        program_limits={'max_runtime_s':args.program_timeout}, controller=deepcopy(policy))
        if args.control_mode == 'hybrid':
            settings['interface_inventory'] = deepcopy(inventory['free_interfaces_by_task'][args.task]
                if 'free_interfaces_by_task' in inventory else inventory['free_interface'])
        retry_policy = args.model_retry_policy
        legacy_micu = retry_policy == 'bounded_same_endpoint_model_request_v1'
        retries = args.model_retries if args.model_retries is not None else (
            0 if args.provider=='gateway' else 1 if legacy_micu else None)
        settings.update(provider=args.provider, model_retries=retries, model_retry_policy=retry_policy,
                        gateway={'protocol':'responses' if args.provider=='micu' else 'chat/completions',
                                 'endpoint':'https://api.micuapi.ai/v1/responses' if args.provider=='micu'
                                 else 'http://127.0.0.1:8788/v1/chat/completions'})
        if args.provider == 'micu':
            settings['gateway'].update(retry_policy=retry_policy, max_retries=retries)
            if not legacy_micu:
                settings['gateway'].update(retry_delay_s=10, timeout_scope='per_attempt')
        if legacy_retry_metadata:
            settings.pop('model_retry_policy')
            settings['gateway'].pop('retry_policy', None)
        source=project/'agent/model_client/runner.py'
        copied=args.output/'client_source/runner.py';copied.parent.mkdir()
        copied.write_bytes(source.read_bytes())
        settings['client_sources_sha256']={'runner.py':suite.digest(source)}
        for name in CLIENT_SOURCE_DEPENDENCIES:
            copied = args.output/'client_source/dependencies'/name
            copied.parent.mkdir(parents=True, exist_ok=True)
            copied.write_bytes((project/name).read_bytes())
            settings['client_sources_sha256']['dependencies/'+name] = suite.digest(copied)
        suite.save(args.output/'agent_config.json',settings)
        episode_id='fixture_'+args.output.name
        spec={**args.expected_spec_pins, 'episode_id':episode_id,
            'task_id':args.task,'robot_id':args.robot,'seed':args.seed,
            'scene_variant':'original_v1','observation_profile':'vision_proprio','execution_mode':'pause_physics_during_agent',
            'track':('free_tool_agent' if args.control_mode=='hybrid' else
                     'system0' if args.control_mode=='fingertip' else 'direct_joints'),'system0':policy,
            'scene_sha256':'c'*64,'agent_config_sha256':suite.canonical_hash(settings),
            'budgets':{'max_actions':args.max_actions,'max_model_calls':args.max_decisions,
                       'sim_time_s':args.sim_budget,'wall_time_s':args.wall_budget}}
        spec.update({name:runtime_hash for name in suite.RUNTIME_PINS})
        suite.save(args.output/'episode_ready.json',{'episode_id':episode_id,'spec':spec})
        directory=args.output/'decisions/0000/tools/0000';directory.mkdir(parents=True)
        suite.save(directory/'primitive_request.json',{'attempt':1})
        suite.save(directory/'execution.json',{'status':'failed'})
        report={'episode_id':episode_id,
                'result':{'spec':spec,'outcome':outcome,'physics_valid':True,'action_attempts':1,
                         'elapsed_sim_s':elapsed_sim,'elapsed_wall_s':elapsed_wall,'model_accounting':'unavailable',
                         'completion_sim_s':elapsed_sim if outcome=='success' else None,
                         'completion_wall_s':elapsed_wall if outcome=='success' else None},
                'termination':'client_decision_budget','error_type':error_type,
                'model_call_attempts':args.max_decisions,'submitted_actions':1,'completed_actions':0,
                'model_accounting':'client_observed_not_server_verified'}
        if metadata is not None:
            report['model_response_metadata']=metadata
        if terminal_scoring is not None:
            report['result']['scoring'] = deepcopy(terminal_scoring)
        suite.save(args.output/'run_report.json',report)
        suite.save(args.output/'artifact_hashes.json',{str(p.relative_to(args.output)):suite.digest(p)
            for p in sorted(args.output.rglob('*'))if p.is_file()})
        return report
    return run,calls


def continuous_contract():
    return {'schema': 'terminal_task_scoring_contract.v1', 'mode': 'continuous',
            'objective': 'maximize', 'score_unit': 'm', 'metric_units': {}}


def continuous_result(score=0., **changes):
    return {**continuous_contract(), 'schema': 'terminal_task_scoring.v1',
            'score': score, 'score_valid': True, 'metrics': {}, **changes}


@pytest.mark.parametrize('score', [0., .125])
def test_continuous_suite_preserves_score_without_binary_penalty(environment, score):
    _, inventory, make = environment
    inventory['tasks'][0]['scoring_contract'] = continuous_contract()
    path, _ = make(modes=['joints_action'])
    run, _ = fake_runner(environment, terminal_scoring=continuous_result(score))
    report = suite.run_plan(path, run_one=run)
    row = report['slots'][0]
    assert report['all_slots_valid']
    assert row['valid_recorded_episode'] and row['task_score_valid'] and row['result_valid_for_scoring']
    assert row['task_score'] == score and row['task_score_unit'] == 'm'
    assert row['task_outcome'] == 'unknown' and row['task_success'] is None
    assert row['diagnostic_penalized_sim_cost_s'] is None
    assert row['diagnostic_penalized_wall_cost_s'] is None
    cohort = report['cohorts'][0]
    assert cohort['valid_recorded_episodes'] == cohort['valid_continuous_episodes'] == 1
    assert cohort['valid_binary_episodes'] == cohort['planned_binary_slots'] == 0
    assert cohort['success_rate_valid_episodes'] is None
    assert cohort['success_fraction_of_planned_slots'] is None
    assert cohort['diagnostic_task_macro_success_rate'] is None
    assert cohort['continuous_scores'] == [{
        'task_id': '01_cube_push', 'score_unit': 'm', 'objective': 'maximize',
        'valid_episodes': 1, 'mean': score, 'min': score, 'max': score}]


@pytest.mark.parametrize('changes', [
    {'terminal_scoring': continuous_result(.125, score_valid=False)},
    {'terminal_scoring': continuous_result(.125), 'error_type': 'ModelGatewayError'},
    {'terminal_scoring': continuous_result(.125), 'outcome': 'infrastructure_error'},
])
def test_continuous_suite_keeps_invalid_score_only_for_audit(environment, changes):
    _, inventory, make = environment
    inventory['tasks'][0]['scoring_contract'] = continuous_contract()
    path, _ = make(modes=['joints_action'])
    run, _ = fake_runner(environment, **changes)
    report = suite.run_plan(path, run_one=run)
    row = report['slots'][0]
    assert row['task_score'] == .125
    assert not row['valid_recorded_episode'] and not row['task_score_valid']
    assert not row['result_valid_for_scoring'] and row['task_success'] is None
    assert report['cohorts'][0]['continuous_scores'] == []


@pytest.mark.parametrize('declared,scoring', [
    (True, None),
    (True, continuous_result(.125, score_unit='cm')),
    (True, continuous_result(.125, objective='minimize')),
    (True, continuous_result(.125, metric_units={'height': 'm'}, metrics={'height': .125})),
    (True, continuous_result(True)),
    (False, continuous_result(.125)),
])
def test_suite_terminal_score_must_match_frozen_contract(environment, declared, scoring):
    _, inventory, make = environment
    if declared:
        inventory['tasks'][0]['scoring_contract'] = continuous_contract()
    path, _ = make(modes=['joints_action'])
    run, _ = fake_runner(environment, terminal_scoring=scoring)
    report = suite.run_plan(path, run_one=run)
    row = report['slots'][0]
    assert not row['valid_recorded_episode'] and not row['task_score_valid']
    assert 'terminal_scoring_contract' in row['pin_errors']
    assert report['cohorts'][0]['continuous_scores'] == []


def test_missing_continuous_slot_is_not_a_planned_binary_failure(environment):
    _, inventory, make = environment
    inventory['tasks'][0]['scoring_contract'] = continuous_contract()
    path, _ = make(modes=['joints_action'])
    report = suite.report_plan(path)
    row = report['slots'][0]
    assert row['status'] == 'missing' and row['task_scoring_mode'] == 'continuous'
    assert row['task_success'] is None and not row['task_score_valid']
    assert report['cohorts'][0]['planned_binary_slots'] == 0
    assert report['cohorts'][0]['success_fraction_of_planned_slots'] is None


def test_continuous_score_rejects_changed_archived_result(environment):
    _, inventory, make = environment
    inventory['tasks'][0]['scoring_contract'] = continuous_contract()
    path, _ = make(modes=['joints_action'])
    run, calls = fake_runner(environment, terminal_scoring=continuous_result(.125))
    assert suite.run_plan(path, run_one=run)['all_slots_valid']
    artifact = calls[0].output / 'run_report.json'
    value = suite.read(artifact)
    value['result']['scoring']['score'] = 123.
    suite.save(artifact, value)
    report = suite.report_plan(path)
    row = report['slots'][0]
    assert row['status'] == 'artifact_invalid' and not row['task_score_valid']
    assert row['task_success'] is None and row['artifact_errors']
    assert report['cohorts'][0]['continuous_scores'] == []


def test_mixed_task_suite_keeps_binary_efficiency_and_continuous_scores_separate(environment):
    project, inventory, make = environment
    release_file = project.parent / 'release/release_manifest.json'
    manifest = suite.read(release_file)
    manifest['tasks'].append({'id': 'height_task'})
    suite.save(release_file, manifest)
    inventory['tasks'].append({'task_id': 'height_task', 'scoring_contract': continuous_contract()})
    path, _ = make(modes=['joints_action'], tasks=['01_cube_push', 'height_task'])
    binary, _ = fake_runner(environment, outcome='success')
    continuous, _ = fake_runner(environment, terminal_scoring=continuous_result(.125))
    report = suite.run_plan(path, run_one=lambda args: (
        continuous(args) if args.task == 'height_task' else binary(args)))
    assert report['all_slots_valid'] and len(report['cohorts']) == 1
    cohort = report['cohorts'][0]
    assert cohort['valid_recorded_episodes'] == 2
    assert cohort['valid_binary_episodes'] == cohort['valid_continuous_episodes'] == 1
    assert cohort['planned_binary_slots'] == cohort['successes'] == 1
    assert cohort['success_rate_valid_episodes'] == cohort['success_fraction_of_planned_slots'] == 1.
    assert cohort['diagnostic_penalized_sim_cost_s'] == .4
    assert cohort['diagnostic_success_per_penalized_sim_second'] == 2.5
    assert cohort['diagnostic_penalized_wall_cost_s'] == 2.
    assert cohort['diagnostic_success_per_penalized_wall_second'] == .5
    assert cohort['continuous_scores'][0]['mean'] == .125


@pytest.mark.parametrize('manifest', [
    {'schema': 'robot_board_game_task_release.v1'},
    {'task_kind': 'board_game'},
    {'task_kind': 'turn_game'},
])
def test_suite_rejects_authenticated_games_before_ordinary_episode_setup(environment, monkeypatch, manifest):
    project, _, make = environment
    suite.save(project.parent / 'release/release_manifest.json', manifest)
    monkeypatch.setattr(suite, 'service_inventory', lambda _: pytest.fail('must reject before HTTP'))
    with pytest.raises(ValueError, match='function_runner'):
        make(modes=['joints_action'])


def test_default_four_mode_product_and_independent_budgets(environment):
    _,_,make=environment
    path,plan=make()
    assert [row['mode']for row in plan['slots']]==list(suite.PRESETS)
    assert len(plan['slots'])==4
    run,calls=fake_runner(environment)
    report=suite.run_plan(path,run_one=run)
    assert len(calls)==4
    assert [(a.control_mode,a.interaction_mode)for a in calls]==list(suite.PRESETS.values())
    assert all(a.max_decisions==2 and a.max_actions==4 and a.max_tool_calls_per_decision==8
               and a.max_program_actions==3 for a in calls)
    assert report['all_slots_valid'] and report['all_slots_attempted']
    assert len(report['cohorts'])==4 and report['leaderboard_eligible']is False
    assert all(row['primitive_execution_status_counts']=={'failed':1}for row in report['slots'])
    assert all(cohort['successes'] == 0 for cohort in report['cohorts'])


@pytest.mark.parametrize('layout,accepted', [
    ('historical', True), ('current', True), ('missing_archive_dependency', False),
    ('partial_plan_inventory', False)])
def test_scoring_dependencies_preserve_historical_archive_verification(environment, layout, accepted):
    _, _, make = environment
    path, plan = make(modes=['joints_action'])
    matrix = suite.Matrix.model_validate(plan['matrix'])
    slot = plan['slots'][0]
    output = path.parent / 'synthetic-archive'
    execute, _ = fake_runner(environment)
    args = suite.runner_args(matrix, slot, output)
    args.expected_spec_pins = plan['release_pins']
    execute(args)
    settings = suite.read(output / 'agent_config.json')
    additions = ('bench/protocol/allocations.py', 'bench/protocol/task_scoring.py')
    if layout == 'historical':
        for name in additions:
            plan['source_pins'].pop(name)
            settings['client_sources_sha256'].pop('dependencies/' + name)
            (output / 'client_source/dependencies' / name).unlink()
    elif layout == 'missing_archive_dependency':
        settings['client_sources_sha256'].pop('dependencies/' + additions[0])
    elif layout == 'partial_plan_inventory':
        plan['source_pins'].pop(additions[0])
    suite.save(output / 'agent_config.json', settings)
    before = suite.digest(output / 'agent_config.json')
    errors = suite._config_errors(plan, matrix, slot, output, suite.canonical_hash(settings))
    assert ('client_source_dependency_inventory' not in errors) is accepted
    if accepted:
        assert not errors
    assert suite.digest(output / 'agent_config.json') == before


@pytest.mark.parametrize('change',[
    {'max_decisions':True},{'max_actions':0},{'max_program_actions':-1},{'max_tool_calls_per_decision':33},
    {'max_program_actions':9},
    {'max_output_tokens':False},{'max_decisions':257},{'sim_budget':float('nan')},{'wall_budget':2401.},
    {'model_timeout':0.},{'http_timeout':float('inf')},{'modes':['unknown']},{'modes':['joints_action']*2},
    {'tasks':[]},{'seeds':[True]},{'seeds':[-1]},{'seeds':[0,0]},{'repeat_ids':['../escape']},
    {'repeat_ids':['r0','r0']},{'visual_inputs':['depth']},{'tactile_inputs':['yes']},
    {'task_release':'relative'},{'bench_url':'http://user:secret@localhost'},
    {'bench_url':'http://localhost/api'},{'reasoning_effort':'low'},{'extra':1},
    {'expected_runtime_pins':{'unexpected':'a'*64}},{'expected_runtime_pins':{'runtime_code_sha256':'bad'}},
    {'provider':'unadvertised'}, {'model_retries':True}, {'model_retries':-1}, {'model_retries':4},
    {'model_retry_policy':'unknown'},
])
def test_invalid_matrix_rejected(change):
    with pytest.raises((ValidationError,ValueError)):
        suite.Matrix.model_validate(change)


def test_matrix_size_and_duplicate_observation_validation():
    with pytest.raises(ValueError):
        suite.Matrix(repeat_ids=[f'r{i}'for i in range(65)])
    with pytest.raises(ValueError):
        suite.Matrix(visual_inputs=['rgb','rgb'])


def test_plan_complete_freeze_hash_and_output_not_reused(environment):
    _,_,make=environment
    path,plan=make(repeat_ids=['r0','r1'],visual_inputs=['rgb','rgbd'],tactile_inputs=['on','off'])
    assert len(plan['slots'])==32
    assert suite.load_plan(path)[0]==plan
    with pytest.raises(ValueError,match='already exist'):
        suite.create_plan(plan['matrix'],path.parent)
    edited=deepcopy(plan);edited['slots'].pop();suite.save(path,edited)
    with pytest.raises(ValueError,match='digest mismatch'):
        suite.load_plan(path)
    edited['plan_sha256']=suite.canonical_hash({k:v for k,v in edited.items()if k!='plan_sha256'})
    suite.save(path,edited)
    with pytest.raises(ValueError,match='entire frozen matrix'):
        suite.load_plan(path)


def test_missing_slots_are_reported_before_any_run(environment):
    _,_,make=environment
    path,_=make()
    report=suite.report_plan(path)
    assert report['planned_slots']==4 and report['attempted_slots']==0
    assert len(report['unattempted_slot_ids'])==4
    assert not report['all_slots_valid'] and not report['all_slots_attempted']
    assert all(c['success_rate_valid_episodes']is None for c in report['cohorts'])
    assert all(c['actual_elapsed_sim_s']is None for c in report['cohorts'])


def test_resume_does_not_retry_failed_or_successful_slots(environment):
    _,_,make=environment
    path,_=make()
    run,calls=fake_runner(environment,outcome='success')
    first=suite.run_plan(path,run_one=run)
    suite.run_plan(path,run_one=run)
    assert len(calls)==4 and sum(c['successes']for c in first['cohorts'])==4
    assert not first['model_call_budget_server_verified'] and not first['leaderboard_eligible']


def test_crashed_attempt_stays_visible_and_is_not_automatically_retried(environment):
    _,_,make=environment
    path,_=make(modes=['joints_action','fingertip_action'])
    def crash(_):raise KeyboardInterrupt()
    with pytest.raises(KeyboardInterrupt):suite.run_plan(path,run_one=crash)
    run,calls=fake_runner(environment)
    report=suite.run_plan(path,run_one=run)
    assert len(calls)==1 and report['incomplete_attempt_ids']==['slot_0000']
    assert not report['all_slots_attempted'] and not report['all_slots_valid']


@pytest.mark.parametrize('which',['source','release','policy'])
def test_changed_pins_prevent_model_runner_and_keep_every_failure(environment,which):
    project,inventory,make=environment
    path,plan=make()
    if which=='source':(project/'agent/model_client/runner.py').write_text('# changed\n')
    elif which=='release':
        manifest=Path(plan['matrix']['task_release'])/'release_manifest.json'
        value=suite.read(manifest);value['version']='changed';suite.save(manifest,value)
    else:inventory['policy']['version']='changed'
    run,calls=fake_runner(environment)
    report=suite.run_plan(path,run_one=run)
    assert not calls and report['attempted_slots']==4
    assert all(row['status']=='pin_mismatch'for row in report['slots'])
    assert not report['all_slots_valid']


def test_service_unavailable_preserves_attempts_without_secret_exception(environment,monkeypatch):
    _,_,make=environment
    path,_=make()
    def unavailable(_):raise suite.BenchHTTPError('secret-token-must-not-be-recorded')
    monkeypatch.setattr(suite,'service_inventory',unavailable)
    run,calls=fake_runner(environment)
    report=suite.run_plan(path,run_one=run)
    assert not calls and all(row['status']=='service_unavailable'for row in report['slots'])
    assert 'secret-token'not in(path.parent/'report.json').read_text()
    assert all('secret-token'not in p.read_text()for p in(path.parent/'attempts').glob('*.json'))


def test_unexpected_controller_cannot_be_counted_as_success(environment):
    _,_,make=environment
    path,_=make(modes=['fingertip_action'])
    run,_=fake_runner(environment,outcome='success',wrong_policy=True)
    report=suite.run_plan(path,run_one=run)
    assert report['slots'][0]['status']=='pin_mismatch'
    assert not report['slots'][0]['task_success']
    assert 'controller_descriptor'in report['slots'][0]['pin_errors']


def test_repeats_same_seed_are_preserved_and_runtime_changes_never_pool(environment):
    _,_,make=environment
    path,_=make(modes=['fingertip_action'],repeat_ids=['r0','r1'])
    first,_=fake_runner(environment,outcome='success',runtime_hash='b'*64)
    second,_=fake_runner(environment,outcome='cancelled',runtime_hash='d'*64)
    calls=[]
    def run(args):
        calls.append(args)
        return(first if len(calls)==1 else second)(args)
    report=suite.run_plan(path,run_one=run)
    assert len(report['cohorts'])==2
    assert [row['slot']['repeat_id']for row in report['slots']]==['r0','r1']
    assert all(row['slot']['seed'] == 0 for row in report['slots'])
    assert 'generalization'in report['seed_warning']


def test_expected_runtime_pin_mismatch_excludes_success(environment):
    _,_,make=environment
    path,_=make(modes=['joints_action'],expected_runtime_pins={'runtime_code_sha256':'e'*64})
    run,_=fake_runner(environment,outcome='success')
    report=suite.run_plan(path,run_one=run)
    assert 'spec_runtime_code_sha256'in report['slots'][0]['pin_errors']
    assert not report['slots'][0]['task_success']


def test_snapshot_tampering_is_rejected(environment):
    _,_,make=environment
    path,plan=make()
    (path.parent/'source_snapshot'/next(iter(plan['source_pins']))).write_text('changed')
    with pytest.raises(ValueError,match='snapshot digest'):
        suite.load_plan(path)


def test_source_change_during_runner_invalidates_result(environment):
    project,_,make=environment
    path,_=make(modes=['joints_action'])
    run,_=fake_runner(environment,outcome='success')
    def change(args):
        result=run(args)
        (project/'agent/model_client/runner.py').write_text('# changed mid-run')
        return result
    report=suite.run_plan(path,run_one=change)
    assert 'source_changed_or_unreadable_after_run'in report['slots'][0]['pin_errors']
    attempt=suite.read(path.parent/'attempts/slot_0000.json')
    assert attempt['source_pins_before_sha256'] != attempt['source_pins_after_sha256']
    assert not report['slots'][0]['task_success']


def test_missing_model_usage_is_null_not_zero(environment):
    _,_,make=environment
    path,_=make(modes=['fingertip_action'])
    run,_=fake_runner(environment,metadata=[{'duration_s':.5},{'duration_s':.7}])
    report=suite.run_plan(path,run_one=run)
    row=report['slots'][0];cohort=report['cohorts'][0]
    assert row['model_latency_client_s']==pytest.approx(1.2)
    assert all(row[key]is None and cohort[key]is None for key in
               ('gateway_prompt_tokens','gateway_completion_tokens','gateway_total_tokens','gateway_reasoning_tokens'))
    assert suite.model_statistics({})['model_latency_client_s']is None


def test_model_usage_fields_complete_and_partial():
    metadata={'duration_s':.5,'usage':{'prompt_tokens':10,'completion_tokens':20,'total_tokens':30,
                                     'completion_tokens_details':{'reasoning_tokens':15}}}
    value=suite.model_statistics({'model_call_attempts':2,'model_response_metadata':[metadata,metadata]})
    assert value['gateway_total_tokens']==60 and value['gateway_reasoning_tokens']==30
    assert value['model_latency_client_s']==1.
    partial=suite.model_statistics({'model_call_attempts':3,'model_response_metadata':[metadata,metadata]})
    assert partial['gateway_total_tokens']is None and partial['model_latency_client_s']is None
    bad=deepcopy(metadata);bad['usage']['total_tokens']=True
    assert suite.model_statistics({'model_call_attempts':1,'model_response_metadata':[bad]})['gateway_total_tokens']is None


def test_responses_usage_and_retries_do_not_invent_complete_costs():
    response = {'protocol':'responses', 'attempts':1, 'duration_s':2.,
                'usage':{'input_tokens':10, 'output_tokens':20, 'total_tokens':30,
                         'output_tokens_details':{'reasoning_tokens':15}}}
    report = {'model_call_attempts':1, 'model_response_metadata':[response]}
    stats = suite.model_statistics(report)
    assert stats['gateway_prompt_tokens'] == 10 and stats['gateway_completion_tokens'] == 20
    assert stats['gateway_reasoning_tokens'] == 15 and stats['gateway_total_tokens'] == 30
    response['attempts'] = 2
    stats = suite.model_statistics(report)
    assert not stats['token_usage_covers_all_http_attempts']
    assert stats['gateway_total_tokens'] is None and stats['gateway_prompt_tokens'] is None
    assert stats['model_latency_client_s'] == 2.


@pytest.mark.parametrize('retries, effective', [(None, None), (0, 0), (3, 3)])
def test_micu_matrix_pins_provider_protocol_and_passes_selection_to_runner(environment, retries, effective):
    _, _, make = environment
    path, _ = make(provider='micu', model_retries=retries, modes=['joints_action'])
    execute, calls = fake_runner(environment, outcome='success')
    report = suite.run_plan(path, run_one=execute)  # Synthetic archives only; no HTTP/model/physics.
    assert len(calls) == 1 and calls[0].provider == 'micu' and calls[0].model_retries == retries
    condition = report['cohorts'][0]['condition']
    assert condition['provider'] == 'micu' and condition['model_protocol'] == 'responses'
    assert condition['model_retries'] == effective and report['slots'][0]['task_success'] is True
    assert calls[0].model_retry_policy == condition['model_retry_policy'] == 'wait_10s_same_endpoint_v2'
    assert condition['model_retry_delay_s'] == 10 and condition['model_timeout_scope'] == 'per_attempt'
    assert suite.read(path)['matrix']['model_retry_policy'] == 'provider_default_v2'


def test_legacy_plan_without_provider_fields_keeps_gateway_and_zero_retries(environment):
    _, _, make = environment
    path, plan = make(modes=['joints_action'])
    plan['matrix'].pop('provider')
    plan['matrix'].pop('model_retries')
    plan['matrix'].pop('model_retry_policy')
    plan['plan_sha256'] = suite.canonical_hash({k:v for k,v in plan.items() if k!='plan_sha256'})
    suite.save(path, plan)
    execute, calls = fake_runner(environment, outcome='success', legacy_retry_metadata=True)
    report = suite.run_plan(path, run_one=execute)
    assert calls[0].provider == 'gateway' and calls[0].model_retries is None
    assert report['cohorts'][0]['condition']['model_retries'] == 0
    assert report['cohorts'][0]['condition']['model_retry_policy'] == 'legacy_gateway_v1'
    assert calls[0].model_retry_policy == 'legacy_gateway_v1'
    assert report['slots'][0]['task_success'] is True


@pytest.mark.parametrize('change, expected_error', [
    ({'provider':'gateway'}, 'agent_config_provider'),
    ({'model_retries':0}, 'agent_config_model_retries'),
    ({'gateway':{'protocol':'chat/completions'}}, 'agent_config_provider_protocol'),
    ({'gateway':{'protocol':'responses', 'endpoint':'https://other.invalid/v1/responses'}},
     'agent_config_micu_endpoint')])
def test_micu_attestation_rejects_protocol_provider_retry_or_endpoint_drift(environment, change, expected_error):
    _, _, make = environment
    path, plan = make(provider='micu', modes=['joints_action'])
    matrix = suite.Matrix.model_validate(plan['matrix'])
    slot = plan['slots'][0]
    output = path.parent/'synthetic-run'
    execute, _ = fake_runner(environment)
    args = suite.runner_args(matrix, slot, output)
    args.expected_spec_pins = plan['release_pins']
    execute(args)
    settings = suite.read(output/'agent_config.json')
    settings.update(change)
    suite.save(output/'agent_config.json', settings)
    errors = suite._config_errors(plan, matrix, slot, output, suite.canonical_hash(settings))
    assert expected_error in errors


@pytest.mark.parametrize('section, key, value, expected_error', [
    ('settings', 'model_retry_policy', None, 'agent_config_model_retry_policy'),
    ('settings', 'model_retry_policy', 'bounded_same_endpoint_model_request_v1', 'agent_config_model_retry_policy'),
    ('gateway', 'retry_policy', 'bounded_same_endpoint_model_request_v1', 'agent_config_gateway_retry_policy'),
    ('gateway', 'retry_delay_s', 1, 'agent_config_retry_delay'),
    ('gateway', 'timeout_scope', 'shared_call_budget', 'agent_config_retry_timeout_scope'),
    ('gateway', 'max_retries', 1, 'agent_config_gateway_model_retries'),
])
def test_new_micu_archive_must_attest_fixed_wait_and_per_attempt_timeout(
        environment, section, key, value, expected_error):
    _, _, make = environment
    path, plan = make(provider='micu', modes=['joints_action'])
    _, matrix = suite.load_plan(path)
    slot = plan['slots'][0]
    output = path.parent/'synthetic-run'
    execute, _ = fake_runner(environment)
    args = suite.runner_args(matrix, slot, output)
    args.expected_spec_pins = plan['release_pins']
    execute(args)
    settings = suite.read(output/'agent_config.json')
    target = settings if section=='settings' else settings['gateway']
    if value is None:
        target.pop(key)
    else:
        target[key] = value
    suite.save(output/'agent_config.json', settings)
    assert expected_error in suite._config_errors(plan, matrix, slot, output, suite.canonical_hash(settings))


@pytest.mark.parametrize('retries, expected', [(None, 1), (0, 0), (3, 3)])
def test_old_micu_plan_is_read_as_bounded_and_cannot_resume_under_new_runner(environment, retries, expected):
    _, _, make = environment
    path, plan = make(provider='micu', model_retries=retries, modes=['joints_action'])
    plan['matrix'].pop('model_retry_policy')
    plan['plan_sha256'] = suite.canonical_hash({k:v for k,v in plan.items() if k!='plan_sha256'})
    suite.save(path, plan)
    original = path.read_bytes()
    loaded, matrix = suite.load_plan(path)
    assert loaded == plan and path.read_bytes() == original
    assert matrix.model_retry_policy == 'legacy_v1'
    args = suite.runner_args(matrix, plan['slots'][0], path.parent/'not-started')
    assert args.model_retry_policy == 'bounded_same_endpoint_model_request_v1'
    report = suite.report_plan(path)
    condition = report['cohorts'][0]['condition']
    assert condition['model_retries'] == expected
    assert condition['model_retry_policy'] == 'bounded_same_endpoint_model_request_v1'
    assert condition['model_timeout_scope'] == 'shared_call_budget'
    assert 'model_retry_delay_s' not in condition
    execute, calls = fake_runner(environment)
    with pytest.raises(ValueError, match='create a new plan'):
        suite.run_plan(path, run_one=execute)
    assert not calls and not (path.parent/'attempts').exists() and path.read_bytes() == original


def test_old_micu_success_archive_keeps_original_retry_semantics_without_rewriting_evidence(environment):
    _, _, make = environment
    path, plan = make(provider='micu', modes=['joints_action'])
    plan['matrix'].pop('model_retry_policy')
    plan['plan_sha256'] = suite.canonical_hash({k:v for k,v in plan.items() if k!='plan_sha256'})
    suite.save(path, plan)
    plan, matrix = suite.load_plan(path)
    (path.parent/'attempts').mkdir()
    (path.parent/'runs').mkdir()
    historical_runner, calls = fake_runner(environment, outcome='success', legacy_retry_metadata=True)
    # Produce an old-format synthetic archive, bypassing the public run entry;
    # this fixture performs no model request, service call, or physics action.
    suite._run_worker(plan, matrix, plan['slots'], path.parent, None, historical_runner)
    archived = [path, *sorted((path.parent/'attempts').rglob('*.json')),
                *sorted((path.parent/'runs').rglob('*'))]
    before = {str(p):suite.digest(p) for p in archived if p.is_file()}
    report = suite.report_plan(path)
    assert report['slots'][0]['task_success'] is True
    assert report['cohorts'][0]['condition']['model_retries'] == 1
    assert report['cohorts'][0]['condition']['model_retry_policy'] == 'bounded_same_endpoint_model_request_v1'
    with pytest.raises(ValueError, match='read-only'):
        suite.run_plan(path, run_one=historical_runner)
    assert len(calls) == 1 and before == {str(p):suite.digest(p) for p in archived if p.is_file()}


@pytest.mark.parametrize('native, expected, valid', [
    ('cancelled', None, False), ('success', True, True), ('task_failure', False, True)])
def test_api_error_does_not_replace_terminal_evidence_with_failure(environment, native, expected, valid):
    _, _, make = environment
    path, _ = make(modes=['joints_action'])
    execute, _ = fake_runner(environment, outcome=native, error_type='ModelGatewayError')
    def run(args):
        report = execute(args)
        report.update(termination='client_error', model_error={'error_category':'transport_timeout'},
                      result_source='cleanup_finish' if native=='cancelled' else 'server_result')
        suite.save(args.output/'run_report.json', report)
        suite.save(args.output/'artifact_hashes.json', {str(p.relative_to(args.output)):suite.digest(p)
            for p in sorted(args.output.rglob('*')) if p.is_file() and p.name!='artifact_hashes.json'})
        return report
    report = suite.run_plan(path, run_one=run)
    row, = report['slots']
    cohort, = report['cohorts']
    assert row['task_success'] is expected and row['valid_recorded_episode'] is valid
    assert row['run_status'] == 'infra_error'
    assert cohort['unknown_task_outcomes'] == (0 if valid else 1)
    if not valid:
        assert row['diagnostic_penalized_sim_cost_s'] is None
        assert cohort['success_rate_valid_episodes'] is None
        assert cohort['diagnostic_success_per_penalized_sim_second'] is None
    else:
        assert cohort['success_rate_valid_episodes'] == int(expected)


def test_fast_failure_does_not_improve_diagnostic_efficiency(environment):
    _,_,make=environment
    path,_=make(modes=['fingertip_action'],repeat_ids=['success','fast_failure'])
    win,_=fake_runner(environment,outcome='success',elapsed_sim=1.,elapsed_wall=10.)
    lose,_=fake_runner(environment,outcome='task_failure',elapsed_sim=.001,elapsed_wall=.01)
    count=[]
    def run(args):
        count.append(args)
        return(win if len(count)==1 else lose)(args)
    report=suite.run_plan(path,run_one=run)
    cohort=report['cohorts'][0]
    assert len(report['cohorts'])==1 and cohort['diagnostic_penalized_sim_cost_s']==13.
    assert cohort['diagnostic_penalized_wall_cost_s']==610.
    assert cohort['diagnostic_success_per_penalized_sim_second']==pytest.approx(1/13)
    assert cohort['diagnostic_success_per_penalized_wall_second']==pytest.approx(1/610)
    assert cohort['diagnostic_task_macro_success_rate']==.5
    assert report['slots'][1]['completion_sim_s']is None and report['slots'][1]['completion_wall_s']is None


@pytest.mark.parametrize('changed',['run_report.json','agent_config.json','episode_ready.json','artifact_hashes.json'])
def test_changed_archives_are_invalid_on_report_reread(environment,changed):
    _,_,make=environment
    path,_=make(modes=['fingertip_action'])
    run,_=fake_runner(environment,outcome='success')
    first=suite.run_plan(path,run_one=run)
    assert first['slots'][0]['task_success']
    target=path.parent/'runs/slot_0000'/changed
    value=suite.read(target);value['injected']='changed';suite.save(target,value)
    second=suite.report_plan(path)
    assert second['slots'][0]['status']=='artifact_invalid'
    assert not second['slots'][0]['task_success'] and second['slots'][0]['artifact_errors']
    assert second['cohorts'][0]['diagnostic_penalized_sim_cost_s'] is None
    assert second['cohorts'][0]['diagnostic_task_macro_success_rate']is None


def test_changed_report_and_recomputed_manifest_cannot_replace_recorded_success(environment):
    _,_,make=environment
    path,_=make(modes=['joints_action'])
    run,_=fake_runner(environment,outcome='success')
    suite.run_plan(path,run_one=run)
    output=path.parent/'runs/slot_0000'
    report=suite.read(output/'run_report.json');report['result']['completion_sim_s']=.001
    suite.save(output/'run_report.json',report)
    manifest=suite.read(output/'artifact_hashes.json');manifest['run_report.json']=suite.digest(output/'run_report.json')
    suite.save(output/'artifact_hashes.json',manifest)
    result=suite.report_plan(path)
    assert result['slots'][0]['status']=='artifact_invalid'and not result['slots'][0]['task_success']
    assert 'archived_artifacts_changed'in result['slots'][0]['artifact_errors']


WORKER_URLS = [f'http://127.0.0.1:{8765+i}' for i in range(4)]


@pytest.mark.parametrize('urls', [
    ['http://127.0.0.1:8766', 'http://127.0.0.1:8765'],
    ['http://127.0.0.1:8765', 'http://localhost:8765/'],
    ['http://127.0.0.1:8765', 'http://127.0.0.1:8765/'],
    ['http://127.0.0.1:8765', 'http://user:secret@127.0.0.1:8766'],
    ['http://127.0.0.1:8765', 'http://127.0.0.1:8766/api'],
    ['http://127.0.0.1:8765', False],
])
def test_parallel_worker_origins_must_be_distinct_valid_and_primary_first(urls):
    with pytest.raises(ValueError):
        suite.Matrix(parallel_bench_urls=urls)


def test_parallel_plan_freezes_assignments_and_checks_every_inventory(environment, monkeypatch):
    _, inventory, make = environment
    queries = []
    def service(matrix):
        queries.append(matrix.bench_url)
        return deepcopy(inventory)
    monkeypatch.setattr(suite, 'service_inventory', service)
    path, plan = make(parallel_bench_urls=WORKER_URLS)
    assert queries == WORKER_URLS
    assert [row['bench_url'] for row in plan['slots']] == WORKER_URLS
    assert [row['worker_id'] for row in plan['slots']] == [f'worker_{i:02d}' for i in range(4)]
    changed = deepcopy(plan)
    changed['slots'][0]['bench_url'] = WORKER_URLS[1]
    changed['plan_sha256'] = suite.canonical_hash({k:v for k,v in changed.items() if k != 'plan_sha256'})
    suite.save(path, changed)
    with pytest.raises(ValueError, match='entire frozen matrix'):
        suite.load_plan(path)


def test_parallel_plan_rejects_different_worker_inventory(environment, monkeypatch):
    _, inventory, make = environment
    def service(matrix):
        value = deepcopy(inventory)
        if matrix.bench_url == WORKER_URLS[-1]:
            value['tasks'][0]['instruction'] = 'Changed humanlike grasp constraint'
        return value
    monkeypatch.setattr(suite, 'service_inventory', service)
    with pytest.raises(ValueError, match='identical selected tasks'):
        make(parallel_bench_urls=WORKER_URLS)


def test_four_workers_overlap_but_each_service_is_serial_and_resume_skips_all(environment):
    _, _, make = environment
    path, _ = make(parallel_bench_urls=WORKER_URLS, repeat_ids=['r0', 'r1'])
    ordinary_run, calls = fake_runner(environment)
    barrier = threading.Barrier(4, timeout=10)
    lock = threading.Lock()
    active = {url: 0 for url in WORKER_URLS}
    peaks = []
    def run(args):
        with lock:
            active[args.bench_url] += 1
            assert active[args.bench_url] == 1
            peaks.append(sum(active.values()))
        try:
            barrier.wait()  # All four workers must run; also checks the second queue round.
            return ordinary_run(args)
        finally:
            with lock:
                active[args.bench_url] -= 1
    first = suite.run_plan(path, run_one=run)
    assert first['all_slots_valid'] and first['all_slots_attempted']
    assert len(calls) == 8 and max(peaks) == 4 and not any(active.values())
    assert all(sum(args.bench_url == url for args in calls) == 2 for url in WORKER_URLS)
    second = suite.run_plan(path, run_one=run)
    assert len(calls) == 8 and second['all_slots_valid']
    for record in (path.parent/'attempts').glob('*.json'):
        attempt = suite.read(record)
        assert attempt['bench_url'] == attempt['slot']['bench_url']
        assert attempt['worker_id'] == attempt['slot']['worker_id']


def test_one_parallel_service_failure_does_not_drop_other_slots_or_retry(environment, monkeypatch):
    _, inventory, make = environment
    path, _ = make(parallel_bench_urls=WORKER_URLS)
    def service(matrix):
        if matrix.bench_url == WORKER_URLS[1]:
            raise suite.BenchHTTPError('credential-that-must-stay-private')
        return deepcopy(inventory)
    monkeypatch.setattr(suite, 'service_inventory', service)
    run, calls = fake_runner(environment)
    report = suite.run_plan(path, run_one=run)
    assert len(calls) == 3 and report['all_slots_attempted']
    assert [row['status'] for row in report['slots']] == ['finished', 'service_unavailable', 'finished', 'finished']
    monkeypatch.setattr(suite, 'service_inventory', lambda _: deepcopy(inventory))
    suite.run_plan(path, run_one=run)
    assert len(calls) == 3
    assert 'credential-that' not in (path.parent/'attempts/slot_0001.json').read_text()


def test_parallel_plan_lock_stays_held_inside_every_worker(environment):
    _, _, make = environment
    path, _ = make(parallel_bench_urls=WORKER_URLS)
    run, calls = fake_runner(environment)
    def check_lock(args):
        with pytest.raises(ValueError, match='already owns'):
            suite.run_plan(path, run_one=run)
        return run(args)
    report = suite.run_plan(path, run_one=check_lock)
    assert report['all_slots_valid'] and len(calls) == 4


def test_all_release_files_including_humanlike_constraints_checked_before_model(environment):
    _, _, make = environment
    path, plan = make()
    release = Path(plan['matrix']['task_release'])
    constraints = release/'humanlike_constraints.json'
    suite.save(constraints, {'thumb_opposition_required': True})
    manifest = suite.read(release/'release_manifest.json')
    manifest['files'][constraints.name] = {'sha256':suite.digest(constraints)}
    suite.save(release/'release_manifest.json', manifest)
    path, _ = make()
    suite.save(constraints, {'thumb_opposition_required': False})
    run, calls = fake_runner(environment)
    report = suite.run_plan(path, run_one=run)
    assert not calls and report['all_slots_attempted']
    assert all(row['status'] == 'preflight_failed' for row in report['slots'])
