"""Task-bound discovery and archive pinning, using only fake HTTP/runner data."""
from copy import deepcopy

import httpx
import pytest

from agent.model_client import runner, suite
from agent.model_client.bench_http import BenchHTTPError
from agent.model_client.free_interface import require_free_interface
from bench.protocol.schemas import DirectControlHandoff, PolicyDescriptor
from bench.runtime.catalog import interface_catalog
from bench.runtime.free_track import advertise_free_track
from test_bench_http_client import client_for
from test_model_attempt_accounting import setup_route
from test_model_suite import environment, fake_runner
from test_online_model_runner import setup_loop


TASKS = ['01_cube_push', '02_cube_pull']
PUBLIC_SERVICE_INVENTORY = suite.service_inventory


def catalog(policy):
    descriptor = PolicyDescriptor.model_validate(policy)
    return advertise_free_track(interface_catalog([descriptor]), [descriptor])


def task_inventory(base, *, different=True, free=False, tasks=TASKS):
    policies = {}
    for index, task in enumerate(tasks):
        descriptor = PolicyDescriptor.model_validate(base)
        descriptor.implementation_sha256 = ('ab'[index] if different else 'a') * 64
        if free:
            descriptor.direct_control_handoff = DirectControlHandoff()
        policies[task] = descriptor.model_dump(mode='json')
    inventory = {'tasks': [{'task_id': task, 'instruction': task} for task in tasks]}
    suite._store_task_inventory(inventory, 'policy', policies)
    if free:
        contracts = {task: require_free_interface(catalog(policy), policy) for task, policy in policies.items()}
        suite._store_task_inventory(inventory, 'free_interface', contracts)
    return inventory, policies


def configure_tasks(environment, *, free=False):
    project, inventory, _ = environment
    replacement, policies = task_inventory(inventory['policy'], free=free)
    inventory.clear()
    inventory.update(replacement)
    release = project.parent/'release/release_manifest.json'
    manifest = suite.read(release)
    manifest['tasks'] = [{'id': task} for task in TASKS]
    suite.save(release, manifest)
    return policies


@pytest.mark.parametrize('method', ['list_policies', 'list_interfaces'])
def test_discovery_query_preserves_optional_and_namespaced_task(method):
    seen = []
    def handler(request):
        seen.append(request.url)
        return httpx.Response(200, json=[])
    client = client_for(handler)
    getattr(client, method)()
    getattr(client, method)(task_id='cards/hex:7')
    assert seen[0].query == b''
    assert dict(seen[1].params) == {'task_id': 'cards/hex:7'}
    assert seen[0].path == seen[1].path == '/' + method.removeprefix('list_')
    assert b'%2F' in seen[1].query and b'%3A' in seen[1].query


@pytest.mark.parametrize('task', ['', False, 'task&private=true', 'task#other', 'x' * 161])
def test_invalid_discovery_task_never_reaches_transport(task):
    client = client_for(lambda _: pytest.fail('invalid selector reached transport'))
    for method in (client.list_policies, client.list_interfaces):
        with pytest.raises(BenchHTTPError, match='task identifier'):
            method(task_id=task)


@pytest.mark.parametrize('free', [False, True])
@pytest.mark.parametrize('different,tasks', [(False, TASKS[:1]), (False, TASKS), (True, TASKS)])
def test_inventory_discovers_each_task_and_only_collapses_identical_bindings(
        environment, monkeypatch, free, different, tasks):
    _, existing, _ = environment
    expected, policies = task_inventory(existing['policy'], different=different, free=free, tasks=tasks)
    queries = []
    class Client:
        def __init__(self, *args, **kwargs): pass
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def list_tasks(self): return deepcopy(expected['tasks'])
        def list_policies(self, task_id=None):
            queries.append(('policies', task_id))
            return [deepcopy(policies[task_id])]
        def list_interfaces(self, task_id=None):
            queries.append(('interfaces', task_id))
            return catalog(policies[task_id])
    monkeypatch.setattr(suite, 'BenchHTTPClient', Client)
    matrix = suite.Matrix(tasks=tasks, modes=['free_tool_agent' if free else 'fingertip_action'])
    actual = PUBLIC_SERVICE_INVENTORY(matrix)
    assert actual == expected
    assert queries == [('policies', task) for task in tasks] + (
        [('interfaces', task) for task in tasks] if free else [])
    if different:
        assert 'policy' not in actual and set(actual['policies_by_task']) == set(tasks)
    else:
        assert 'policy' in actual and 'policies_by_task' not in actual
    if free:
        assert ('free_interfaces_by_task' in actual) is different


def test_direct_only_inventory_does_not_require_geometry_discovery(environment, monkeypatch):
    _, inventory, _ = environment
    class Client:
        def __init__(self, *args, **kwargs): pass
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def list_tasks(self): return deepcopy(inventory['tasks'])
    monkeypatch.setattr(suite, 'BenchHTTPClient', Client)
    assert PUBLIC_SERVICE_INVENTORY(suite.Matrix(modes=['joints_action'])) == {
        'tasks': inventory['tasks'], 'policy': None}


@pytest.mark.parametrize('free', [False, True])
def test_online_runner_scopes_discovery_to_its_task(monkeypatch, tmp_path, free):
    if free:
        args, bench, _ = setup_route(monkeypatch, tmp_path, 'fingertip_action',
                                     [{'action': 'stop'}], free=True)
    else:
        args, bench, _ = setup_loop(monkeypatch, tmp_path, [{'action': 'stop'}], fingertip=True)
    report = runner.run(args)
    assert report['error_type'] is None
    assert bench.discovery_requests == [('policies', args.task)] + ([('interfaces', args.task)] if free else [])
    assert bench.create_requests[0]['task_id'] == args.task


@pytest.mark.parametrize('pin', ['controller', 'interface_inventory'])
def test_online_runner_rejects_other_tasks_frozen_binding_before_create(monkeypatch, tmp_path, pin):
    args, bench, model = setup_route(monkeypatch, tmp_path, 'fingertip_action',
                                    [{'action': 'stop'}], free=True)
    policy = PolicyDescriptor.model_validate(bench.policy_inventory[0]).model_dump(mode='json')
    if pin == 'controller':
        args.expected_controller = {**policy, 'implementation_sha256': 'e' * 64}
    else:
        args.expected_interface_inventory = require_free_interface(catalog(policy), policy)
        args.expected_interface_inventory['shared_contract']['completion'] = 'other task contract'
    report = runner.run(args)
    assert report['error_type'] == 'ValueError'
    assert not bench.create_requests and not model.requests


@pytest.mark.parametrize('mapped,shared', [('policies_by_task', 'policy'),
                                         ('free_interfaces_by_task', 'free_interface')])
@pytest.mark.parametrize('corruption', ['shared_and_mapped', 'missing_task', 'extra_task', 'invalid_descriptor'])
def test_plan_reload_rejects_ambiguous_or_incomplete_task_bindings(environment, mapped, shared, corruption):
    configure_tasks(environment, free=True)
    _, _, make = environment
    path, plan = make(tasks=TASKS, modes=['free_tool_agent'])
    inventory = plan['service_inventory']
    if corruption == 'shared_and_mapped':
        inventory[shared] = deepcopy(inventory[mapped][TASKS[0]])
    elif corruption == 'missing_task':
        inventory[mapped].pop(TASKS[1])
    elif corruption == 'extra_task':
        inventory[mapped]['not_selected'] = deepcopy(inventory[mapped][TASKS[0]])
    else:
        inventory[mapped][TASKS[1]] = None
    plan['plan_sha256'] = suite.canonical_hash({k: v for k, v in plan.items() if k != 'plan_sha256'})
    suite.save(path, plan)
    with pytest.raises(ValueError, match='exactly the selected tasks'):
        suite.load_plan(path)


def test_free_contract_cannot_pin_another_tasks_controller(environment):
    configure_tasks(environment, free=True)
    _, _, make = environment
    path, plan = make(tasks=TASKS, modes=['free_tool_agent'])
    contracts = plan['service_inventory']['free_interfaces_by_task']
    contracts[TASKS[1]] = deepcopy(contracts[TASKS[0]])
    plan['plan_sha256'] = suite.canonical_hash({k: v for k, v in plan.items() if k != 'plan_sha256'})
    suite.save(path, plan)
    with pytest.raises(ValueError, match='selected controller'):
        suite.load_plan(path)


@pytest.mark.parametrize('free', [False, True])
def test_each_suite_slot_executes_verifies_and_reports_its_task_binding(environment, free):
    policies = configure_tasks(environment, free=free)
    _, inventory, make = environment
    path, plan = make(tasks=TASKS, modes=['free_tool_agent' if free else 'fingertip_action'])
    execute, calls = fake_runner(environment, outcome='success')
    report = suite.run_plan(path, run_one=execute)
    assert report['all_slots_valid'] and all(row['task_success'] for row in report['slots'])
    assert [args.expected_controller for args in calls] == [policies[task] for task in TASKS]
    assert len(report['cohorts']) == 2
    for args in calls:
        if free:
            assert args.expected_interface_inventory == inventory['free_interfaces_by_task'][args.task]
        config = suite.read(args.output/'agent_config.json')
        assert config['controller'] == policies[args.task]
        cohort = next(row for row in report['cohorts'] if
                      row['condition']['actual_runtime_pins']['system0'] == policies[args.task])
        if free:
            assert cohort['condition']['public_interface_contract'] == inventory['free_interfaces_by_task'][args.task]
    # Reload and reverify the closed archive without consulting the live bench.
    assert suite.report_plan(path)['all_slots_valid']


@pytest.mark.parametrize('field', ['controller', 'interface_inventory'])
def test_agent_config_attestation_rejects_cross_task_swap(environment, field):
    configure_tasks(environment, free=True)
    _, inventory, make = environment
    path, plan = make(tasks=TASKS, modes=['free_tool_agent'])
    execute, calls = fake_runner(environment, outcome='success')
    suite.run_plan(path, run_one=execute)
    output = calls[1].output
    settings = suite.read(output/'agent_config.json')
    mapped = 'policies_by_task' if field == 'controller' else 'free_interfaces_by_task'
    settings[field] = deepcopy(inventory[mapped][TASKS[0]])
    suite.save(output/'agent_config.json', settings)
    errors = suite._config_errors(plan, suite.Matrix.model_validate(plan['matrix']),
                                  plan['slots'][1], output, suite.canonical_hash(settings))
    assert errors == ['agent_config_' + field]


def test_setup_failure_attests_selected_task_controller(environment):
    policies = configure_tasks(environment)
    project, _, make = environment
    path, plan = make(tasks=TASKS, modes=['fingertip_action'])
    matrix = suite.Matrix.model_validate(plan['matrix'])
    slot = plan['slots'][1]
    output = path.parent/'synthetic-setup-failure'
    args = suite.runner_args(matrix, slot, output)
    args.expected_spec_pins = plan['release_pins']
    execute, _ = fake_runner(environment)
    execute(args)
    settings = suite.read(output/'agent_config.json')
    budgets = {'max_actions': matrix.max_actions, 'max_model_calls': matrix.max_decisions,
               'sim_time_s': matrix.sim_budget, 'wall_time_s': matrix.wall_budget, 'max_action_sim_s': 5.}
    settings['budgets'] = budgets
    for name in runner.CLIENT_SOURCE_DEPENDENCIES:
        copied = output/'client_source/dependencies'/name
        copied.parent.mkdir(parents=True, exist_ok=True)
        copied.write_bytes((project/name).read_bytes())
        settings['client_sources_sha256']['dependencies/' + name] = suite.digest(copied)
    suite.save(output/'agent_config.json', settings)
    report = {key: settings[key] for key in ('controller', 'model', 'reasoning_effort',
              'control_mode', 'interaction_mode', 'visual_input', 'tactile_input')}
    report.update(schema='online_visual_agent_run.v7', episode_id=None, result=None,
        termination='client_error', error_type='BenchHTTPError', model_error=None,
        task_success=None, task_outcome='unknown', online_model_decision_loop_exercised=False,
        program_multiple_actions_exercised=False, config_sha256=suite.canonical_hash(settings),
        model_input_observation_sequences=[], model_response_metadata=[], decision_status_counts={},
        interface_selection_counts={}, program_status_counts={}, primitive_termination_reason_counts={})
    report.update({key: 0 for key in ('model_call_attempts', 'model_responses', 'primitive_attempts',
                   'submitted_actions', 'completed_actions', 'tool_calls', 'point_queries', 'read_calls')})
    suite.save(output/'create_request.json', {'task_id': slot['task'], 'robot_id': matrix.robot,
        'seed': slot['seed'], 'scene_variant': 'original_v1', 'agent_id': 'online_visual_fingertip_action_agent',
        'agent_version': runner.AGENT_VERSION, 'agent_config_sha256': report['config_sha256'],
        'track': 'system0', 'observation_profile': 'vision_proprio', 'budgets': budgets,
        'policy_id': policies[slot['task']]['policy_id'], 'policy_version': policies[slot['task']]['version']})
    verified = suite.verify_setup_failure(plan, matrix, slot, report, output)
    assert verified['passed'], verified['errors']
    report['controller'] = policies[TASKS[0]]
    assert 'setup_config_controller' in suite.verify_setup_failure(plan, matrix, slot, report, output)['errors']
