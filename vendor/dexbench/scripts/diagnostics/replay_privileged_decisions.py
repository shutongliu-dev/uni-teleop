"""Replay recorded decision boundaries on CPU without executing any actions.

Stops at the first changed decision: subsequent observations belong to the old
physical trajectory and cannot establish what the changed policy would do.
This is a policy regression diagnostic, never task or physics acceptance.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import inspect
import json
from pathlib import Path
import sys
import time

import bench

from bench.isaacsim.release import task_inputs
from agent.model_client.geometry_actions import GeometryActionLimits, build_geometry_action
from bench.paths import load_json


def read(path):
    return load_json(path, parse_constant=lambda value: (_ for _ in ()).throw(
        ValueError('Nonfinite JSON: ' + value)))


def fingerprint(path):
    return {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def checked(root, reference, expected):
    path = root / expected
    if (not isinstance(reference, dict) or Path(reference.get('path', '')).absolute() != path
            or not path.resolve().is_relative_to(root) or path.is_symlink()
            or fingerprint(path)['sha256'] != reference.get('sha256')):
        raise ValueError('Recording hash/path mismatch: ' + expected)
    return read(path)


def differences(a, b, prefix='decision'):
    if type(a) is not type(b):
        return [prefix]
    if isinstance(a, dict):
        return [path for key in sorted(a.keys() | b.keys()) for path in
                ([prefix + '.' + key] if key not in a or key not in b else
                 differences(a[key], b[key], prefix + '.' + key))]
    if isinstance(a, list):
        if len(a) != len(b):
            return [prefix]
        return [path for i, (x, y) in enumerate(zip(a, b))
                for path in differences(x, y, f'{prefix}[{i}]')]
    return [] if a == b else [prefix]


def replay(run_directory, *, strategy_factory=None):
    started = time.monotonic()
    root = Path(run_directory).resolve(strict=True)
    report = read(root / 'report.json')  # A final report is required, even for a failed episode.
    config = checked(root, report['config'], 'config.json')
    trajectory = checked(root, report['trajectory'], 'trajectory.json')
    ready = read(root / 'episode_ready.json')
    if (ready.get('spec') != trajectory.get('spec')
            or ready.get('episode_id') != ready['spec'].get('episode_id')):
        raise ValueError('Episode identity/spec does not match recorded trajectory')
    release = Path(config['release'])
    if fingerprint(release / 'release_manifest.json')['sha256'] != config['release_manifest_sha256']:
        raise ValueError('Frozen release manifest changed')
    _, scene, task, _ = task_inputs(release, config['task_id'])
    if strategy_factory is None:
        from scenes.cloth.privileged_policy import TASK_ID, PrivilegedClothDragPolicy
        from scenes.manipulation.privileged_policy import PrivilegedManipulationPolicy
        strategy_factory = PrivilegedClothDragPolicy if config['task_id'] == TASK_ID else PrivilegedManipulationPolicy
    policy = strategy_factory(deepcopy(scene), deepcopy(task))
    if hasattr(policy, 'configure_budget'):
        policy.configure_budget(config['budgets']['max_actions'])
    result = {'schema': 'privileged_decision_cpu_replay_v1', 'model_calls': 0,
        'native_invocations': 0, 'actions_executed': 0, 'native_acceptance': False,
        'scope': 'Policy regression on recorded inputs only; first divergence ends replay.',
        'recorded_report': fingerprint(root / 'report.json'),
        'recorded_trajectory': fingerprint(root / 'trajectory.json'),
        'recorded_config': fingerprint(root / 'config.json'),
        'recorded_ready': fingerprint(root / 'episode_ready.json'),
        'recorded_trajectory_complete': trajectory['integrity']['complete'],
        'candidate_policy': {'class': strategy_factory.__name__, 'version': getattr(policy, 'VERSION', None),
                             'source': fingerprint(Path(inspect.getfile(strategy_factory)).resolve())},
        'diagnostic_source': fingerprint(Path(__file__).resolve()),
        'status': 'matched_recorded_decisions', 'matched_decisions': 0, 'rows': []}
    previous_after = previous_feedback = None
    actions_used = 0
    rows = trajectory['decisions']
    if not rows:
        raise ValueError('No recorded decision boundaries')
    for i, row in enumerate(rows):
        if row['decision_index'] != i:
            raise ValueError('Decision indices must be continuous from zero')
        prefix = f'decisions/{i:04d}/'
        artifacts = row['artifacts']
        # Verify every available record, including commands that were actually
        # submitted. Matching decision JSON alone does not establish that the
        # recorded feedback resulted from that decision.
        values = {name: checked(root, reference,
            f'private/strategy_inputs/{i:04d}.json' if name == 'private_strategy_input' else prefix + name)
            for name, reference in artifacts.items()}
        before = values['before/observation.json']
        inputs = values['private_strategy_input']
        oracle = inputs['privileged_state']
        if (before.get('episode_id') != ready['episode_id']
                or before.get('sequence') != inputs.get('observation_sequence')
                or any(before.get(k) != oracle.get(k) for k in ('sequence', 'sim_time_s'))
                or inputs.get('previous_feedback') != previous_feedback
                or (previous_after is not None and before != previous_after)
                or (i == 0 and (before.get('sequence') != 0 or before.get('sim_time_s') != 0.))):
            raise ValueError(f'Broken recorded decision/feedback boundary: {i}')
        recorded = values.get('decision.json')
        after, feedback = values.get('after_observation.json'), values.get('execution.json')
        executed = values.get('action.json')
        complete = False
        if executed is not None:
            rebuilt = build_geometry_action(recorded, before, ready['capabilities'], ready['spec'],
                action_id=executed['action_id'], expected_sequence=before['sequence'],
                actions_used=actions_used, limits=GeometryActionLimits(**config.get('action_limits', {})))
            if rebuilt != executed:
                raise ValueError(f'Recorded formal command differs from decision: {i}')
            request = values.get('primitive_request.json') or {}
            if request.get('decision') != recorded or request.get('expected_sequence') != before['sequence']:
                raise ValueError(f'Recorded primitive request differs from decision: {i}')
            if feedback is not None:
                accepted, source = values.get('accepted.json') or {}, values.get('execution_source.json') or {}
                if (after is None or after.get('episode_id') != before['episode_id']
                        or after.get('sequence', -1) < before['sequence']
                        or any(value.get('action_id') != executed['action_id'] for value in (accepted, feedback))
                        or after.get('last_action_id') != executed['action_id']
                        or feedback.get('status') not in ('completed', 'failed', 'cancelled')
                        or after.get('last_action_status') != feedback.get('status')
                        or any(source.get(k) != feedback.get(k) for k in
                            ('action_id', 'status', 'sim_time_s', 'elapsed_sim_s', 'termination_reason', 'diagnostic'))
                        or feedback.get('sim_time_s') != after.get('sim_time_s')
                        or type(feedback.get('elapsed_sim_s')) not in (int, float)
                        or feedback['elapsed_sim_s'] < 0
                        or abs(after['sim_time_s'] - before['sim_time_s'] - feedback['elapsed_sim_s']) > 1e-6):
                    raise ValueError(f'Recorded command/ACK boundary is inconsistent: {i}')
                complete = True
        elif any(values.get(name) is not None for name in ('accepted.json', 'execution.json', 'execution_source.json')):
            raise ValueError(f'Recorded execution has no formal command: {i}')
        elif recorded is not None and recorded.get('action') == 'stop':
            if after != before or values.get('primitive_request.json') is not None:
                raise ValueError(f'Recorded stop changed the physical boundary: {i}')
            complete = True
        if recorded is None and any(values.get(name) is not None for name in
                                    ('action.json', 'primitive_request.json', 'rejection.json')):
            raise ValueError(f'Recorded action has no decision: {i}')
        entry = {'index': i, 'observation_sequence': before['sequence'], 'sim_time_s': before['sim_time_s']}
        result['rows'].append(entry)
        try:
            candidate = policy.next_decision(deepcopy(before), deepcopy(oracle), deepcopy(previous_feedback))
            # Validate exactly the public geometry ingress; no HTTP or native worker is constructed.
            command = build_geometry_action(candidate, before, ready['capabilities'], ready['spec'],
                action_id=f'cpu_replay_{i:04d}', expected_sequence=before['sequence'],
                actions_used=actions_used, limits=GeometryActionLimits(**config.get('action_limits', {})))
        except Exception as exc:
            result.update(status='candidate_error', stop_index=i,
                          candidate_error={'type': type(exc).__name__, 'detail': str(exc)})
            break
        entry['decision_sha256'] = hashlib.sha256(json.dumps(candidate, sort_keys=True, allow_nan=False).encode()).hexdigest()
        changed = differences(recorded, candidate)
        if changed:
            result.update(status='candidate_at_missing_decision' if recorded is None else 'first_decision_changed',
                stop_index=i, changed_paths=changed, recorded_decision=recorded,
                candidate_decision=candidate, candidate_command=command)
            break
        result['matched_decisions'] += 1
        # A later boundary is meaningful only after the exact recorded decision.
        if not complete:
            result.update(status='matched_partial_recording', stop_index=i)
            break
        previous_after, previous_feedback = after, feedback
        actions_used += executed is not None
        if i + 1 < len(rows) and (previous_after is None or previous_feedback is None or candidate['action'] == 'stop'):
            raise ValueError('Recording continues past an unexecuted or stop decision')
    result['candidate_diagnostic'] = policy.diagnostic()
    project = Path(bench.__file__).resolve().parent.parent
    result['loaded_project_sources'] = {str(path.relative_to(project)): fingerprint(path)['sha256']
        for module in tuple(sys.modules.values())
        if (source := getattr(module, '__file__', None))
        and (path := Path(source).resolve()).is_relative_to(project)
        and path.suffix == '.py' and path.is_file()}
    result['wall_s'] = time.monotonic() - started
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-directory', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True, help='New report outside the source episode.')
    args = parser.parse_args()
    output = args.output.resolve()
    if output.is_relative_to(args.run_directory.resolve()):
        raise ValueError('Do not add files to a sealed source episode')
    if output.exists():
        raise FileExistsError(output)
    result = replay(args.run_directory)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write('\n')
    print(json.dumps({key: result.get(key) for key in ('status', 'matched_decisions', 'stop_index', 'candidate_error')}))
    raise SystemExit(0 if result['status'] == 'matched_recorded_decisions' else 2)


if __name__ == '__main__':
    main()
