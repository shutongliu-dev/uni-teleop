"""CPU recorded-input regressions must stop before counterfactual feedback."""
import json

import pytest

from bench.paths import load_json
from scripts.diagnostics import replay_privileged_decisions as replay_tool
from test_privileged_baseline import HoldStrategy, args, factory, harness


@pytest.fixture
def recorded(tmp_path, factory):
    make, _, _ = factory
    request = args(tmp_path, count=3)
    report = harness.run(request, service_factory=make, strategy_factory=HoldStrategy)
    assert report['trajectory_complete'] and report['error'] is None
    return request.output


def test_same_policy_matches_all_real_http_recorded_boundaries_without_physics(recorded):
    report = replay_tool.replay(recorded, strategy_factory=HoldStrategy)
    assert report['status'] == 'matched_recorded_decisions'
    assert report['matched_decisions'] == len(report['rows']) == 3
    assert report['actions_executed'] == report['native_invocations'] == report['model_calls'] == 0
    assert report['native_acceptance'] is False


def test_changed_action_stops_before_old_future_observation(recorded):
    calls = []
    class Changed(HoldStrategy):
        def next_decision(self, *arguments):
            calls.append(self.count)
            result = super().next_decision(*arguments)
            if self.count == 2:
                result['duration_s'] = .06
            return result
    report = replay_tool.replay(recorded, strategy_factory=Changed)
    assert report['status'] == 'first_decision_changed'
    assert report['matched_decisions'] == 1 and report['stop_index'] == 1
    assert report['changed_paths'] == ['decision.duration_s']
    assert report['candidate_command']['hold_duration_s'] == .06
    assert calls == [0, 1]


def test_invalid_candidate_stops_at_real_public_parser(recorded):
    class Invalid(HoldStrategy):
        def next_decision(self, *arguments):
            return {'action': 'hold', 'duration_s': 100.}
    report = replay_tool.replay(recorded, strategy_factory=Invalid)
    assert report['status'] == 'candidate_error'
    assert report['candidate_error']['type'] == 'ActionValidationError'
    assert report['matched_decisions'] == 0 and report['stop_index'] == 0


def rewrite_record(recorded, relative, change, *, update_hash):
    path = recorded / relative
    value = load_json(path)
    change(value)
    path.write_text(json.dumps(value))
    if update_hash:
        trajectory_path = recorded / 'trajectory.json'
        trajectory = load_json(trajectory_path)
        for row in trajectory['decisions']:
            for reference in row['artifacts'].values():
                if reference['path'] == str(path):
                    reference.update(replay_tool.fingerprint(path))
        trajectory_path.write_text(json.dumps(trajectory))
        report_path = recorded / 'report.json'
        report = load_json(report_path)
        report['trajectory'].update(replay_tool.fingerprint(trajectory_path))
        report_path.write_text(json.dumps(report))


def test_changed_recorded_oracle_hash_rejected_before_policy(recorded):
    rewrite_record(recorded, 'private/strategy_inputs/0000.json',
                   lambda value: value.update(observation_sequence=999), update_hash=False)
    with pytest.raises(ValueError, match='hash/path mismatch'):
        replay_tool.replay(recorded, strategy_factory=HoldStrategy)


@pytest.mark.parametrize('corruption', ['delete', 'change_bytes', 'change_and_rehash'])
def test_recorded_formal_command_must_exist_and_match_decision(recorded, corruption):
    relative = 'decisions/0000/action.json'
    if corruption == 'delete':
        (recorded / relative).unlink()
    else:
        rewrite_record(recorded, relative, lambda value: value.update(hold_duration_s=.08),
                       update_hash=corruption == 'change_and_rehash')
    with pytest.raises((ValueError, FileNotFoundError)):
        replay_tool.replay(recorded, strategy_factory=HoldStrategy)


@pytest.mark.parametrize('relative,change', [
    ('accepted.json', lambda value: value.update(action_id='other_action')),
    ('execution_source.json', lambda value: value.update(elapsed_sim_s=.08)),
    ('primitive_request.json', lambda value: value.update(expected_sequence=999)),
    ('after_observation.json', lambda value: value.update(episode_id='other_episode')),
])
def test_rehashed_but_inconsistent_execution_chain_cannot_supply_feedback(recorded, relative, change):
    rewrite_record(recorded, 'decisions/0000/' + relative, change, update_hash=True)
    with pytest.raises(ValueError):
        replay_tool.replay(recorded, strategy_factory=HoldStrategy)


@pytest.mark.parametrize('change', [
    lambda value: value.update(previous_feedback={'status': 'completed'}),
    lambda value: value.update(observation_sequence=999),
    lambda value: value['privileged_state'].update(sim_time_s=7.),
])
def test_self_consistent_hashes_do_not_hide_wrong_feedback_boundary(recorded, change):
    rewrite_record(recorded, 'private/strategy_inputs/0000.json', change, update_hash=True)
    with pytest.raises(ValueError, match='decision/feedback boundary'):
        replay_tool.replay(recorded, strategy_factory=HoldStrategy)


def test_success_claim_is_never_derived_from_old_report(recorded):
    path = recorded / 'report.json'
    old = load_json(path)
    old.update(accepted=True, task_success=True, physics_valid=True)
    path.write_text(json.dumps(old))
    report = replay_tool.replay(recorded, strategy_factory=HoldStrategy)
    assert report['status'] == 'matched_recorded_decisions'
    assert report['native_acceptance'] is False
    assert 'task_success' not in report


def test_recorded_policy_exception_can_generate_one_new_candidate(tmp_path, factory):
    make, _, _ = factory
    class Broken(HoldStrategy):
        def next_decision(self, *arguments):
            if self.count:
                raise ValueError('previous policy error')
            return super().next_decision(*arguments)
    request = args(tmp_path, count=3)
    old = harness.run(request, service_factory=make, strategy_factory=Broken)
    assert not old['trajectory_complete'] and old['completed_actions'] == 1
    report = replay_tool.replay(request.output, strategy_factory=HoldStrategy)
    assert report['status'] == 'candidate_at_missing_decision'
    assert report['matched_decisions'] == 1 and report['stop_index'] == 1
    assert report['candidate_decision'] == {'action': 'hold', 'duration_s': .04}


def test_cli_refuses_to_write_inside_original_run(recorded, monkeypatch):
    monkeypatch.setattr('sys.argv', ['replay', '--run-directory', str(recorded),
                                   '--output', str(recorded / 'new-report.json')])
    with pytest.raises(ValueError, match='sealed source episode'):
        replay_tool.main()
    assert not (recorded / 'new-report.json').exists()
