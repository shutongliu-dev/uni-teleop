"""Explicit provider token-cap omission survives CLI, frozen plan and archives."""
from argparse import ArgumentTypeError
import json
import sys

import pytest

from agent.model_client import runner, suite
from bench.paths import load_json
from test_model_suite import environment, fake_runner
from test_online_model_runner import setup_loop


@pytest.mark.parametrize('value', ['none', 'null', 'disabled', 'NONE'])
def test_cli_null_output_tokens_reaches_runner(monkeypatch, tmp_path, value):
    seen = []
    monkeypatch.setattr(sys, 'argv', ['runner', '--max-output-tokens', value, '--output', str(tmp_path/'run')])
    monkeypatch.setattr(runner, 'run', lambda args: seen.append(args) or {'error_type': None})
    runner.main()
    assert seen[0].max_output_tokens is None


@pytest.mark.parametrize('value', ['0', '-1', '1.5', 'nan', 'inf', 'true', '32769'])
def test_cli_rejects_fake_unlimited_token_counts(value):
    with pytest.raises(ArgumentTypeError):
        runner.optional_output_tokens(value)


def test_cli_numeric_output_cap_preserves_value():
    assert runner.optional_output_tokens('16384') == 16384


def test_null_output_cap_survives_full_mock_online_loop(monkeypatch, tmp_path):
    args, bench, model = setup_loop(monkeypatch, tmp_path, [{'action': 'stop'}])
    args.max_output_tokens = None
    report = runner.run(args)
    assert report['error_type'] is None and len(model.requests) == 1
    assert model.requests[0][2]['max_output_tokens'] is None
    for relative in ['agent_config.json', 'decisions/0000/model_request.json']:
        value = load_json(args.output/relative)
        assert 'max_output_tokens' in value and value['max_output_tokens'] is None


def test_nullable_output_cap_plan_and_report_preserve_distinct_configuration(environment):
    _, _, make = environment
    path, plan = make(max_output_tokens=None)
    restored, matrix = suite.load_plan(path)
    assert restored['matrix']['max_output_tokens'] is matrix.max_output_tokens is None
    execute, calls = fake_runner(environment)
    report = suite.run_plan(path, run_one=execute)
    assert report['all_slots_valid'] and len(calls) == 4
    assert all(args.max_output_tokens is None for args in calls)
    assert all(group['condition']['budgets']['max_output_tokens'] is None for group in report['cohorts'])
    assert 'Infinity' not in json.dumps(report, allow_nan=False)


@pytest.mark.parametrize('value', ['none', 'null', 'disabled', True, 0])
def test_matrix_output_cap_requires_real_null_or_positive_integer(value):
    with pytest.raises(ValueError):
        suite.Matrix(max_output_tokens=value)


@pytest.mark.parametrize('missing', [False, True])
def test_null_cap_missing_or_replaced_in_archive_is_not_attested(environment, missing):
    _, _, make = environment
    path, plan = make(max_output_tokens=None, modes=['joints_action'])
    execute, _ = fake_runner(environment)
    matrix = suite.Matrix.model_validate(plan['matrix'])
    slot = plan['slots'][0]
    output = path.parent/'manual_run'
    args = suite.runner_args(matrix, slot, output)
    args.expected_spec_pins = plan['release_pins']
    report = execute(args)
    settings = suite.read(output/'agent_config.json')
    if missing:
        settings.pop('max_output_tokens')
    else:
        settings['max_output_tokens'] = 16384
    suite.save(output/'agent_config.json', settings)
    check = suite.verify_run(plan, matrix, slot, report, output)
    assert not check['passed'] and 'agent_config_max_output_tokens' in check['errors']
