"""Read-only attestation of both source layouts; no frozen Python is executed."""
import pytest

from agent.model_client import runner, suite


LEGACY_DEPENDENCIES = ('bench/agent/schemas.py', 'bench/agent/validation.py',
    'bench/geometry.py', 'scripts/maintenance/freeze_project_sources.py')
SDK_DEPENDENCIES = ('bench/sdk/__init__.py', 'bench/sdk/depth.py', 'bench/sdk/geometry.py')


def archive(tmp_path, layout, kind):
    root = 'interface/model_client' if layout == 'legacy' else 'agent/model_client'
    dependencies = LEGACY_DEPENDENCIES if layout == 'legacy' else runner.CLIENT_SOURCE_DEPENDENCIES
    matrix = suite.Matrix(modes=['joints_action'])
    slot = suite.expand_slots(matrix)[0]
    output = tmp_path/'run'
    output.mkdir()
    snapshot = tmp_path/'source_snapshot'
    source_pins, archived = {}, {}
    for name in (root+'/__init__.py', root+'/runner.py', root+'/nested/guard.py', *dependencies):
        source = snapshot/name
        source.parent.mkdir(parents=True, exist_ok=True)
        # Importing any historical source would fail, even with a matching hash.
        source.write_text(f'raise AssertionError("Do not execute archive: {name}")\n')
        source_pins[name] = suite.digest(source)
        if name.startswith(root+'/'):
            relative = name.removeprefix(root+'/')
            if kind == 'setup_v6' and '/' in relative:
                continue  # v6 runner copied only top-level client Python files.
        else:
            relative = 'dependencies/'+name
        copied = output/'client_source'/relative
        copied.parent.mkdir(parents=True, exist_ok=True)
        copied.write_bytes(source.read_bytes())
        archived[relative] = suite.digest(copied)
    plan = {'protocol': 'agent_development_suite_plan_v1', 'matrix': matrix.model_dump(mode='json'),
            'slots': [slot], 'source_pins': source_pins, 'release_pins': {},
            'service_inventory': {'tasks': [{'task_id': slot['task']}]}}
    plan['plan_sha256'] = suite.canonical_hash(plan)
    suite.save(tmp_path/'plan.json', plan)
    settings = {key: getattr(matrix, key) for key in ('model', 'reasoning_effort', 'max_decisions',
        'max_actions', 'max_tool_calls_per_decision', 'max_program_actions', 'max_output_tokens')}
    settings.update({key: slot[key] for key in ('control_mode', 'interaction_mode', 'visual_input', 'tactile_input')})
    settings.update(controller=None, provider='gateway', model_retries=0,
        model_retry_policy='legacy_gateway_v1', gateway={'protocol': 'chat/completions'},
        model_timeout_s=matrix.model_timeout, http_timeout_s=matrix.http_timeout,
        program_limits={'max_runtime_s': matrix.program_timeout}, client_sources_sha256=archived)
    budgets = {'max_actions': matrix.max_actions, 'max_model_calls': matrix.max_decisions,
               'sim_time_s': matrix.sim_budget, 'wall_time_s': matrix.wall_budget, 'max_action_sim_s': 5.}
    settings['budgets'] = budgets
    report = {'schema': 'online_visual_agent_run.v6' if kind == 'setup_v6' else 'online_visual_agent_run.v7'}
    if kind == 'run':
        spec = {'episode_id': 'archive_fixture', 'task_id': slot['task'], 'robot_id': matrix.robot,
            'seed': slot['seed'], 'scene_variant': 'original_v1', 'observation_profile': 'vision_proprio',
            'execution_mode': 'pause_physics_during_agent', 'track': 'direct_joints', 'system0': None,
            'scene_sha256': 'c'*64, 'budgets': budgets, **{name: 'b'*64 for name in suite.RUNTIME_PINS}}
        report.update(episode_id=spec['episode_id'], result={'spec': spec})
    else:
        report.update({key: settings[key] for key in ('controller', 'model', 'reasoning_effort',
            'control_mode', 'interaction_mode', 'visual_input', 'tactile_input')})
        report.update(episode_id=None, result=None, termination='client_error', error_type='BenchHTTPError',
            model_error=None, task_success=False if kind == 'setup_v6' else None, task_outcome='unknown',
            online_model_decision_loop_exercised=False, program_multiple_actions_exercised=False,
            model_input_observation_sequences=[], model_response_metadata=[], decision_status_counts={},
            interface_selection_counts={}, program_status_counts={}, primitive_termination_reason_counts={})
        report.update({key: 0 for key in ('model_call_attempts', 'model_responses', 'primitive_attempts',
            'submitted_actions', 'completed_actions', 'tool_calls', 'point_queries', 'read_calls')})
    value = {'plan': plan, 'matrix': matrix, 'slot': slot, 'report': report, 'output': output,
             'settings': settings, 'kind': kind, 'root': root, 'dependencies': dependencies}
    save_archive(value)
    return value


def save_archive(value):
    settings, output, report = value['settings'], value['output'], value['report']
    config_hash = suite.canonical_hash(settings)
    suite.save(output/'agent_config.json', settings)
    if value['kind'] == 'run':
        report['result']['spec']['agent_config_sha256'] = config_hash
        suite.save(output/'episode_ready.json', {'episode_id': report['episode_id'], 'spec': report['result']['spec']})
    else:
        report['config_sha256'] = config_hash
        matrix, slot = value['matrix'], value['slot']
        suite.save(output/'create_request.json', {'task_id': slot['task'], 'robot_id': matrix.robot,
            'seed': slot['seed'], 'scene_variant': 'original_v1', 'agent_id': 'online_visual_joints_action_agent',
            'agent_version': runner.AGENT_VERSION, 'agent_config_sha256': config_hash,
            'track': 'direct_joints', 'observation_profile': 'vision_proprio', 'budgets': settings['budgets']})
    suite.save(output/'run_report.json', report)
    suite.save(output/'artifact_hashes.json', {p.relative_to(output).as_posix(): suite.digest(p)
        for p in output.rglob('*') if p.is_file() and p.name != 'artifact_hashes.json'})


def verify(value):
    function = suite.verify_run if value['kind'] == 'run' else suite.verify_setup_failure
    return function(*(value[key] for key in ('plan', 'matrix', 'slot', 'report', 'output')))


@pytest.mark.parametrize('layout', ['legacy', 'current'])
@pytest.mark.parametrize('kind', ['run', 'setup_v6', 'setup_v7'])
def test_archives_verify_without_executing_or_rewriting_sources(tmp_path, monkeypatch, layout, kind):
    value = archive(tmp_path, layout, kind)
    before = {p.relative_to(tmp_path): p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}
    # Archive interpretation must not depend on the mutable working checkout.
    monkeypatch.setattr(suite, 'PROJECT', tmp_path/'nonexistent_live_project')
    loaded, matrix = suite.load_plan(tmp_path/'plan.json')
    assert loaded == value['plan'] and matrix == value['matrix']
    result = verify(value)
    assert result['passed'], result['errors']
    pins = (suite.artifact_pins if kind == 'run' else suite.setup_failure_pins)(value['output'])
    assert 'client_source/runner.py' in pins
    after = {p.relative_to(tmp_path): p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}
    assert after == before


@pytest.mark.parametrize('layout', ['legacy', 'current'])
@pytest.mark.parametrize('kind', ['run', 'setup_v7'])
@pytest.mark.parametrize('defect', ['wrong_root', 'mixed_roots', 'wrong_dependency'])
def test_wrong_or_mixed_layouts_are_rejected(tmp_path, layout, kind, defect):
    value = archive(tmp_path, layout, kind)
    pins = value['plan']['source_pins']
    if defect == 'wrong_root':
        for name in list(pins):
            if name.startswith(value['root']+'/'):
                pins['unexpected/model_client/'+name.removeprefix(value['root']+'/')] = pins.pop(name)
    elif defect == 'mixed_roots':
        other = 'agent/model_client' if layout == 'legacy' else 'interface/model_client'
        pins[other+'/runner.py'] = pins[value['root']+'/runner.py']
    else:
        old = value['dependencies'][0]
        other = 'bench/protocol/schemas.py' if layout == 'legacy' else 'bench/agent/schemas.py'
        digest = pins.pop(old)
        pins[other] = digest
        sources = value['settings']['client_sources_sha256']
        sources['dependencies/'+other] = sources.pop('dependencies/'+old)
        path = value['output']/'client_source/dependencies'
        (path/other).parent.mkdir(parents=True, exist_ok=True)
        (path/old).rename(path/other)
        save_archive(value)
    result = verify(value)
    assert not result['passed']
    assert ('client_source_root_inventory' if defect != 'wrong_dependency'
            else 'client_source_dependency_inventory') in result['errors']


@pytest.mark.parametrize('layout,kind', [('current', 'run'), ('current', 'setup_v7'), ('legacy', 'setup_v7')])
@pytest.mark.parametrize('where', ['plan', 'archive'])
def test_required_dependency_cannot_be_omitted(tmp_path, layout, kind, where):
    value = archive(tmp_path, layout, kind)
    name = 'bench/sdk/depth.py' if layout == 'current' else LEGACY_DEPENDENCIES[-1]
    if where == 'plan':
        value['plan']['source_pins'].pop(name)
    else:
        value['settings']['client_sources_sha256'].pop('dependencies/'+name)
        (value['output']/'client_source/dependencies'/name).unlink()
        save_archive(value)
    result = verify(value)
    assert not result['passed']
    assert ('client_source_dependency_inventory' if layout == 'current' else
            'setup_client_dependency_inventory' if where == 'plan' else
            'setup_client_source_inventory') in result['errors']


def test_legacy_run_keeps_historical_per_file_contract(tmp_path):
    value = archive(tmp_path, 'legacy', 'run')
    name = 'scripts/maintenance/freeze_project_sources.py'
    value['settings']['client_sources_sha256'].pop('dependencies/'+name)
    (value['output']/'client_source/dependencies'/name).unlink()
    save_archive(value)
    result = verify(value)
    assert result['passed'], result['errors']


def test_current_runner_archives_all_extracted_sdk_sources(tmp_path, monkeypatch):
    from test_online_model_runner import setup_loop
    args, _, _ = setup_loop(monkeypatch, tmp_path, [{'action': 'stop'}])
    report = runner.run(args)
    assert report['termination'] == 'model_requested_stop'
    settings = suite.read(args.output/'agent_config.json')
    for name in SDK_DEPENDENCIES:
        archived_name = 'dependencies/'+name
        assert settings['client_sources_sha256'][archived_name] == suite.digest(args.output/'client_source'/archived_name)
