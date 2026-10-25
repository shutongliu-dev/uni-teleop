"""Rejected physical intent costs budget; synthetic transport is not physics."""
from copy import deepcopy

import pytest

from bench.paths import load_json
from bench.protocol.schemas import DirectControlHandoff, PolicyDescriptor
from bench.runtime.catalog import interface_catalog
from bench.runtime.free_track import advertise_free_track
from agent.model_client import runner
from test_multimode_runner import loop
from test_online_model_runner import setup_loop


INTERFACES = ('joints_action', 'fingertip_action', 'fingertip_tools', 'fingertip_code')


def hold_response(interface, duration):
    if interface.endswith('_code'):
        return f'hold(duration_s={duration!r})'
    if interface.endswith('_tools'):
        return {'tool': 'hold', 'arguments': {'duration_s': duration}}
    return {'action': 'hold', 'duration_s': duration}


def setup_route(monkeypatch, tmp_path, interface, responses, *, free=False):
    """Extend the existing fake server's public discovery for a free episode."""
    if free:
        responses = [{'interface': interface, 'payload': value} for value in responses]
        args, bench, model = loop(monkeypatch, tmp_path, responses, mode='free', rgbd=False)
        args.control_mode = 'hybrid'
        policy = bench.policy_inventory[0]
        policy['direct_control_handoff'] = DirectControlHandoff().model_dump(mode='json')
        descriptor = PolicyDescriptor.model_validate(policy)
        catalog = advertise_free_track(interface_catalog([descriptor]), [descriptor])
        def list_interfaces(task_id=None):
            bench.discovery_requests.append(('interfaces', task_id))
            return deepcopy(catalog)
        monkeypatch.setattr(bench, 'list_interfaces', list_interfaces, raising=False)
        create = bench.create

        def create_free(request):
            ready = create(request)
            ready['spec'].update(track='free_tool_agent', system0=deepcopy(policy))
            ready['capabilities']['system0'] = deepcopy(policy)
            ready['capabilities']['tracks'].append('free_tool_agent')
            return ready

        monkeypatch.setattr(bench, 'create', create_free)
    elif interface == 'joints_action':
        args, bench, model = setup_loop(monkeypatch, tmp_path, responses)
    else:
        args, bench, model = loop(monkeypatch, tmp_path, responses,
                                  mode=interface.rsplit('_', 1)[1], rgbd=False)
    return args, bench, model


def assert_receipts(output, attempts, submissions):
    paths = sorted((output / 'decisions').rglob('primitive_request.json'))
    rows = [load_json(path) for path in paths]
    assert len(rows) == attempts
    assert sorted(row['attempt'] for row in rows) == list(range(1, attempts + 1))
    assert sum(path.with_name('accepted.json').is_file() for path in paths) == submissions
    return rows


def assert_accounting_metadata(output, report):
    config = load_json(output / 'agent_config.json')
    ready = load_json(output / 'episode_ready.json')
    assert config['primitive_attempt_accounting'] == 'recognized_physical_request_v1'
    assert report['primitive_attempt_accounting'] == config['primitive_attempt_accounting']
    assert ready['spec']['agent_config_sha256'] == runner.canonical_hash(config)


@pytest.mark.parametrize('free', [False, True], ids=['fixed', 'free'])
@pytest.mark.parametrize('interface', INTERFACES)
def test_rejected_physical_parameters_exhaust_shared_budget(monkeypatch, tmp_path, interface, free):
    responses = [hold_response(interface, True), hold_response(interface, .02)]
    args, bench, model = setup_route(monkeypatch, tmp_path, interface, responses, free=free)
    args.max_actions = 1
    report = runner.run(args)
    assert report['error_type'] is None
    assert report['primitive_attempts'] == report['model_call_attempts'] == 1
    assert report['submitted_actions'] == report['completed_actions'] == 0
    assert report['termination'] == 'client_action_budget'
    assert len(model.requests) == 1 and not bench.commands
    assert_receipts(args.output, 1, 0)
    assert_accounting_metadata(args.output, report)


@pytest.mark.parametrize('free', [False, True], ids=['fixed', 'free'])
@pytest.mark.parametrize('interface', ['fingertip_action', 'fingertip_tools'])
def test_stale_physical_request_counts_without_post_or_double_charge(
        monkeypatch, tmp_path, interface, free):
    response = hold_response(interface, .02)
    args, bench, model = setup_route(monkeypatch, tmp_path, interface,
                                     [response, response], free=free)
    args.max_actions = 2
    generate = model.generate

    def stale_first_response(*a, **kw):
        result = generate(*a, **kw)
        if len(model.requests) == 1:
            bench.obs['sequence'] += 1
        return result

    monkeypatch.setattr(model, 'generate', stale_first_response)
    report = runner.run(args)
    assert report['error_type'] is None
    assert report['primitive_attempts'] == 2
    assert report['submitted_actions'] == report['completed_actions'] == len(bench.commands) == 1
    assert bench.commands[0]['observation_sequence'] == 1
    first = load_json(args.output / 'decisions/0000/execution.json')
    assert first['status'] == 'rejected_by_client'
    assert 'stale' in first['reason'].lower()
    assert_receipts(args.output, 2, 1)


def test_tool_cannot_rebase_to_a_sequence_changed_by_second_refresh(monkeypatch, tmp_path):
    args, bench, model = setup_route(monkeypatch, tmp_path, 'fingertip_tools',
                                     [hold_response('fingertip_tools', .02)])
    args.max_actions = 1
    get_observation = bench.get_observation
    reads_after_model = 0

    def change_on_move_refresh(episode):
        nonlocal reads_after_model
        if model.requests:
            reads_after_model += 1
            # runner latest, tool freshness check, then move's own refresh.
            if reads_after_model == 3:
                bench.obs['sequence'] += 1
        return get_observation(episode)

    monkeypatch.setattr(bench, 'get_observation', change_on_move_refresh)
    report = runner.run(args)
    assert report['error_type'] is None
    assert reads_after_model >= 3
    assert report['primitive_attempts'] == 1 and report['submitted_actions'] == 0
    assert not bench.commands
    first = load_json(args.output / 'decisions/0000/execution.json')
    assert first['status'] == 'rejected_by_client' and 'stale' in first['reason'].lower()
    assert assert_receipts(args.output, 1, 0)[0]['expected_sequence'] == 0


def test_program_keeps_completed_ack_and_rejected_attempt_in_receipts(monkeypatch, tmp_path):
    args, bench, _ = setup_route(monkeypatch, tmp_path, 'fingertip_code', [
        'hold(duration_s=0.02)\nhold(duration_s=True)\nhold(duration_s=0.02)',
        'hold(duration_s=0.02)'])
    args.max_actions = 2
    report = runner.run(args)
    assert report['error_type'] is None
    assert report['primitive_attempts'] == 2 and report['model_call_attempts'] == 1
    assert report['submitted_actions'] == report['completed_actions'] == len(bench.commands) == 1
    assert report['termination'] == 'client_action_budget'
    assert_receipts(args.output, 2, 1)
    program = load_json(args.output / 'decisions/0000/program_execution.json')
    assert program['status'] == 'tool_error' and program['move_calls'] == 2
    assert program['primitive_results'][0]['feedback']['status'] == 'completed'
    assert program['primitive_results'][1]['call_status'] == 'error'
    assert program['primitive_results'][1]['feedback'] is None


def test_readonly_unknown_and_stop_do_not_consume_physical_budget(monkeypatch, tmp_path):
    args, bench, _ = setup_route(monkeypatch, tmp_path, 'fingertip_tools', [
        {'tool': 'observe', 'arguments': {}},
        {'tool': 'point_at', 'arguments': {'camera_id': 'task', 'u': 0, 'v': 0}},
        {'tool': 'unknown', 'arguments': {}},
        {'tool': 'stop', 'arguments': {}}], free=True)
    args.max_actions = 1
    original = deepcopy(bench.obs)
    report = runner.run(args)
    assert report['error_type'] is None and report['termination'] == 'model_requested_stop'
    assert report['primitive_attempts'] == report['submitted_actions'] == 0
    assert report['model_call_attempts'] == 4 and not bench.commands
    assert bench.obs == original
    assert_receipts(args.output, 0, 0)


@pytest.mark.parametrize('source', [
    'hold(duration_s=0.02)\nimport os',
    'hold(duration_s=(1 / 0))',
])
def test_program_rejected_before_dispatch_does_not_consume_attempt(monkeypatch, tmp_path, source):
    args, bench, _ = setup_route(monkeypatch, tmp_path, 'fingertip_code', [source], free=True)
    report = runner.run(args)
    assert report['error_type'] is None
    assert report['primitive_attempts'] == report['submitted_actions'] == report['tool_calls'] == 0
    assert not bench.commands
    assert_receipts(args.output, 0, 0)
