"""Run zero-model privileged manipulation through the public Agent action path.

Only the decision script reads a copied, synchronized private state. Its output
passes through InteractionSession, the HTTP routes and the pinned classical
executor. Native physics, task scoring, cameras and final audits remain active.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict
import inspect
import json
import math
import os
from pathlib import Path
import time

from fastapi.testclient import TestClient

from bench.isaacsim.release import digest, task_inputs
from bench.paths import load_json
from scenes.manipulation import DEFAULT_TASK_RELEASE
from bench.runtime.server import create_app
from bench.runtime.service import AgentService, canonical_hash
from agent.model_client.actions import ActionValidationError
from agent.model_client.bench_http import BenchHTTPClient
from agent.model_client.capture import save
from agent.model_client.geometry_actions import GeometryActionLimits, build_geometry_action
from agent.model_client.interaction import InteractionSession
from agent.model_client.runner import select_geometry_policy
from scripts.maintenance.freeze_project_sources import verify_if_snapshot


PROJECT = Path(__file__).absolute().parents[2]
SCHEMA = 'privileged_scripted_agent_baseline_v1'


def privileged_state(session, observation):
    """Read one acknowledged decision boundary; never hand out live state."""
    with session.lock:
        if observation.get('episode_id') != session.spec.episode_id:
            raise RuntimeError('Oracle and public observation belong to different episodes')
        if (type(observation.get('sequence')) is not int or observation['sequence'] < 0
                or type(observation.get('sim_time_s')) not in (int, float)
                or not math.isfinite(observation['sim_time_s']) or observation['sim_time_s'] < 0):
            raise RuntimeError('Oracle decision boundary has invalid time or sequence')
        if session.life.active_action_id is not None:
            raise RuntimeError('Cannot read oracle state while an action is active')
        if (session.sequence != observation['sequence']
                or session.raw['sim_time_s'] != observation['sim_time_s']
                or session.snapshot.sequence != observation['sequence']):
            raise RuntimeError('Oracle and public decision states are not synchronized')
        state = {'sequence': session.sequence, 'sim_time_s': session.raw['sim_time_s'],
                 'objects': session.raw.get('objects', {}), 'cloth': session.raw.get('cloth', {})}
        # Only scenes with deformable bodies add them; earlier oracles stay byte-identical.
        if session.raw.get('soft_bodies'):
            state['soft_bodies'] = session.raw['soft_bodies']
        return deepcopy(state)


def artifact(path):
    return {'path': str(path), 'sha256': digest(path)} if path.is_file() else None


def check_episode_artifacts(directory):
    directory = Path(directory)
    path = directory / 'artifact_hashes.json'
    if not path.is_file():
        return False
    entries = load_json(path)
    paths = list(directory.rglob('*'))
    if any(item.is_symlink() for item in paths):
        return False
    actual = {str(item.relative_to(directory)) for item in paths if item.is_file() and item != path}
    return (isinstance(entries, dict) and bool(entries) and set(entries) == actual
        and all(isinstance(expected, str) and len(expected) == 64
            and all(char in '0123456789abcdef' for char in expected) for expected in entries.values())
        and all(
        (directory / name).resolve().is_relative_to(directory.resolve())
        and (directory / name).is_file() and digest(directory / name) == expected
        for name, expected in entries.items()))


def package_trajectory(output, ready, config, *, destination=None, expected_counts=None):
    """Preserve partial decisions and explicitly verify the recorded action chain."""
    output = Path(output)
    rows, issues = [], []
    counts = {'primitive_attempts': 0, 'submitted_actions': 0, 'completed_actions': 0}
    if expected_counts is None and (output / 'report.json').is_file():
        report = load_json(output / 'report.json')
        expected_counts = {key: report.get(key) for key in counts}
    episode_id = ready.get('episode_id') if isinstance(ready, dict) else None
    previous_after = None
    previous_feedback = None

    def boundary(value, label):
        if (not isinstance(value, dict) or value.get('episode_id') != episode_id
                or type(value.get('sequence')) is not int or value['sequence'] < 0
                or type(value.get('sim_time_s')) not in (int, float)
                or not math.isfinite(value['sim_time_s']) or value['sim_time_s'] < 0):
            issues.append(label + ': invalid episode/sequence/time')
            return None
        return {key: value[key] for key in ('sequence', 'sim_time_s')}

    for index, folder in enumerate(sorted((output / 'decisions').glob('*'))):
        if not folder.is_dir():
            issues.append(folder.name + ': unexpected decision entry')
            continue
        if folder.name != f'{index:04d}':
            issues.append(folder.name + ': decision indices are not continuous from zero')
        paths = {name: folder / name for name in ('decision.json', 'before/observation.json',
            'after_observation.json', 'execution.json', 'execution_source.json', 'accepted.json',
            'action.json', 'rejection.json', 'primitive_request.json', 'strategy_diagnostic.json')}
        paths['private_strategy_input'] = output / 'private/strategy_inputs' / (folder.name + '.json')
        values = {}
        start_issues = len(issues)
        for name, path in paths.items():
            try:
                if not path.resolve().is_relative_to(output.resolve()):
                    raise ValueError('Artifact escapes episode output')
                value = load_json(path, parse_constant=lambda value: (_ for _ in ()).throw(
                    ValueError('Nonfinite JSON: ' + value))) if path.is_file() else None
                if value is not None and not isinstance(value, dict):
                    raise ValueError('Expected JSON object')
                values[name] = value
            except (ValueError, OSError) as exc:
                issues.append(folder.name + '/' + name + ': ' + str(exc))
                values[name] = None
        before, after, feedback = (values[name] for name in
            ('before/observation.json', 'after_observation.json', 'execution.json'))
        decision, command, accepted = (values[name] for name in ('decision.json', 'action.json', 'accepted.json'))
        before_boundary = boundary(before, folder.name + '/before')
        after_boundary = boundary(after, folder.name + '/after') if after is not None else None
        if previous_after is not None and before_boundary != previous_after:
            issues.append(folder.name + ': decision boundary does not match previous action end')
        request = values['primitive_request.json']
        counts['primitive_attempts'] += request is not None
        counts['submitted_actions'] += command is not None
        counts['completed_actions'] += feedback is not None and feedback.get('status') == 'completed'
        kind = 'incomplete'
        if decision is None:
            issues.append(folder.name + ': planner did not record a decision')
        elif decision.get('action') == 'stop':
            kind = 'stop'
            if any(value is not None for value in (command, accepted, feedback, request)) or before_boundary != after_boundary:
                issues.append(folder.name + ': stop has a physical action or changed observation')
        elif values['rejection.json'] is not None and accepted is None and feedback is None:
            kind = 'rejected'
        elif all(value is not None for value in (command, accepted, feedback, request, after_boundary)):
            kind = 'executed'
            action_id = command.get('action_id')
            if (not isinstance(action_id, str) or not action_id
                    or accepted.get('action_id') != action_id or feedback.get('action_id') != action_id
                    or after.get('last_action_id') != action_id):
                issues.append(folder.name + ': action identities disagree')
            if (before_boundary is None or command.get('observation_sequence') != before_boundary['sequence']
                    or request.get('expected_sequence') != before_boundary['sequence']
                    or request.get('decision') != decision):
                issues.append(folder.name + ': submitted action does not match the decision boundary')
            if feedback.get('status') not in ('completed', 'failed', 'cancelled') or after.get('last_action_status') != feedback.get('status'):
                issues.append(folder.name + ': terminal action status is inconsistent')
            elapsed, end_time = feedback.get('elapsed_sim_s'), feedback.get('sim_time_s')
            if (type(elapsed) not in (int, float) or not math.isfinite(elapsed) or elapsed < 0
                    or type(end_time) not in (int, float) or not math.isfinite(end_time)
                    or before_boundary is None or after_boundary['sequence'] < before_boundary['sequence']
                    or abs(end_time - after_boundary['sim_time_s']) > 1e-6
                    or abs(after_boundary['sim_time_s'] - before_boundary['sim_time_s'] - elapsed) > 1e-6):
                issues.append(folder.name + ': action timing is inconsistent')
            source_feedback = values['execution_source.json']
            if source_feedback is None or any(source_feedback.get(key) != feedback.get(key) for key in
                    ('action_id', 'status', 'sim_time_s', 'elapsed_sim_s', 'termination_reason', 'diagnostic')):
                issues.append(folder.name + ': source and public execution feedback disagree')
            try:
                rebuilt = build_geometry_action(decision, before, ready['capabilities'], ready['spec'],
                    action_id=command['action_id'], expected_sequence=before['sequence'],
                    actions_used=counts['submitted_actions']-1,
                    limits=GeometryActionLimits(**config.get('action_limits', {})))
                if rebuilt != command:
                    issues.append(folder.name + ': command differs from formal decision reconstruction')
            except Exception as exc:
                issues.append(folder.name + ': command reconstruction failed: ' + str(exc))
        else:
            issues.append(folder.name + ': physical decision is missing submission, execution, or observation files')
        strategy_input = values['private_strategy_input']
        if strategy_input is None:
            issues.append(folder.name + ': missing privileged strategy input')
        else:
            oracle = strategy_input.get('privileged_state', {})
            if (before_boundary is None or strategy_input.get('observation_sequence') != before_boundary['sequence']
                    or type(oracle.get('sequence')) is not int or oracle.get('sequence') != before_boundary['sequence']
                    or type(oracle.get('sim_time_s')) not in (int, float)
                    or oracle.get('sim_time_s') != before_boundary['sim_time_s']):
                issues.append(folder.name + ': privileged input is not synchronized with public observation')
            if strategy_input.get('previous_feedback') != previous_feedback:
                issues.append(folder.name + ': previous feedback differs from the recorded preceding execution')
        if kind == 'executed' and values['execution_source.json'] is None:
            issues.append(folder.name + ': missing source execution feedback')
        previous_after = after_boundary
        previous_feedback = feedback
        rows.append({'decision_index': int(folder.name) if folder.name.isdigit() else None,
            'record_kind': kind, 'record_complete': len(issues) == start_issues,
            'action': decision, 'before': before_boundary, 'after': after_boundary,
            'feedback': {k: feedback.get(k) for k in ('status', 'termination_reason', 'elapsed_sim_s', 'diagnostic')}
                if feedback else None,
            'artifacts': {name: artifact(path) for name, path in paths.items()
                if path.is_file() and path.resolve().is_relative_to(output.resolve())}})
    if not rows:
        issues.append('No recorded decisions')
    if expected_counts is not None:
        for key, value in counts.items():
            if type(expected_counts.get(key)) is not int or expected_counts[key] != value:
                issues.append(key + ': recorded count differs from runner count')
    integrity = {'complete': not issues, 'issues': issues, 'recorded_counts': counts,
                 'expected_counts': expected_counts, 'decision_count': len(rows)}
    value = {'schema': 'privileged_scripted_trajectory_v2', 'model_calls': 0,
        'privileged_planner': True, 'object_state_writeback': False,
        'scope': 'Recorded high-level actions and feedback; replay requires real physics and fresh validation.',
        'spec': ready['spec'] if ready else None, 'config': artifact(output / 'config.json'),
        'release_manifest_sha256': config['release_manifest_sha256'],
        'task_initial_state_sha256': config.get('task_initial_state_sha256'),
        'strategy_implementation': config['strategy_implementation'], 'integrity': integrity, 'decisions': rows}
    destination = output / 'trajectory.json' if destination is None else Path(destination)
    if destination.exists():
        raise FileExistsError('Choose a new trajectory sidecar: '+str(destination))
    save(destination, value)
    return {**artifact(destination), 'integrity': integrity}


def run(args, *, service_factory=AgentService, strategy_factory=None):
    from scenes.cloth.privileged_policy import TASK_ID, DEFAULT_CLOTH_TASK_RELEASE
    cloth_task = args.task == TASK_ID
    if args.release is None:
        args.release = DEFAULT_CLOTH_TASK_RELEASE if cloth_task else DEFAULT_TASK_RELEASE
    # The release scene pins the embodiment; the episode requests exactly that robot.
    _, release_scene, release_task, _ = task_inputs(args.release, args.task)
    robot_id = release_scene['robot'].get('hand_id', 'sharpa_w01_right')
    soft_mode = release_task.get('evaluation', {}).get('mode')
    if strategy_factory is None:
        if soft_mode == 'soft_press':
            from scenes.manipulation.soft_policy import PrivilegedSoftPressPolicy
            strategy_factory = PrivilegedSoftPressPolicy
        elif soft_mode == 'soft_push':
            from scenes.manipulation.soft_policy import PrivilegedSoftPushPolicy
            strategy_factory = PrivilegedSoftPushPolicy
        elif cloth_task:
            from scenes.cloth.privileged_policy import PrivilegedClothDragPolicy
            strategy_factory = PrivilegedClothDragPolicy
        elif robot_id == 'dual_fr3_sharpa_w01':
            from scenes.manipulation.bimanual_policy import PrivilegedBimanualPolicy
            strategy_factory = PrivilegedBimanualPolicy
        elif robot_id != 'sharpa_w01_right':
            from scenes.manipulation.embodiment_policy import PrivilegedEmbodimentPolicy
            strategy_factory = PrivilegedEmbodimentPolicy
        else:
            from scenes.manipulation.privileged_policy import PrivilegedManipulationPolicy
            strategy_factory = PrivilegedManipulationPolicy
    verification = verify_if_snapshot(PROJECT,
        required_modules=('scripts.validation.run_privileged_baseline',
            'bench.runtime.server', 'bench.runtime.service',
            'agent.model_client.interaction', strategy_factory.__module__),
        entrypoint_file=Path(__file__))
    if verification is None:
        raise ValueError('Native baseline requires a complete frozen source snapshot')
    if type(args.max_actions) is not int or not 1 <= args.max_actions <= 128:
        raise ValueError('This baseline permits between 1 and 128 primitive actions')
    if type(args.seed) is not int or not 0 <= args.seed < 2**32:
        raise ValueError('Seed must be an unsigned 32-bit integer')
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    save(output / 'runtime_imports.json', verification)
    limits = GeometryActionLimits()
    strategy_source = Path(inspect.getfile(strategy_factory)).resolve()
    config = {'schema': SCHEMA, 'task_id': args.task, 'release': str(args.release.resolve()),
        'release_manifest_sha256': digest(args.release / 'release_manifest.json'),
        'strategy_implementation': {'class': strategy_factory.__name__,
            'version': getattr(strategy_factory, 'VERSION', getattr(strategy_factory, 'version', None)),
            'source': artifact(strategy_source)},
        'source_snapshot_manifest': artifact(PROJECT / 'source_snapshot_manifest.json'),
        'execution': {'device': args.device,
            'isaac_allow_unvalidated_driver': os.environ.get('ISAAC_ALLOW_UNVALIDATED_DRIVER'),
            'resource_mode': os.environ.get('BENCH_RESOURCE_MODE', 'shared'),
            'resource_allocation_tag': os.environ.get('BENCH_RESOURCE_ALLOCATION_TAG', 'development_unreserved')},
        'seed': args.seed, 'robot_id': robot_id, 'model_calls': 0, 'privileged_state_used': True,
        'oracle_fields': ['objects', 'cloth'], 'object_state_writeback': False,
        'control_mode': 'fingertip', 'interaction_mode': 'action',
        'observation_profile': 'vision_proprio', 'action_limits': asdict(limits),
        'budgets': {'sim_time_s': None if cloth_task else 120., 'wall_time_s': None,
            'max_actions': args.max_actions, 'max_model_calls': 1, 'max_action_sim_s': 5.},
        'model_budget_note': 'Contract minimum is one; no model client is constructed or called.',
        'wall_budget_note': 'No wall/HTTP deadline; the finite primitive budget and 5-second primitive simulation timeout remain enforced.',
        'scope': 'Privileged scripted development baseline; not a visual Agent score or generalization evaluation.'}
    save(output / 'config.json', config)
    manager = None
    interaction = None
    native_session = None
    strategy = None
    ready = None
    error = None
    cleanup_errors = []
    termination = 'setup_failed'
    started = time.monotonic()
    try:
        entry, scene, task, initial = task_inputs(args.release, args.task)
        config['task_initial_state_sha256'] = canonical_hash(initial)
        config['task_file_sha256'] = digest(args.release / entry['task'])
        strategy = strategy_factory(deepcopy(scene), deepcopy(task))
        if hasattr(strategy, 'configure_budget'):
            strategy.configure_budget(args.max_actions)
        manager = service_factory(output / 'episodes', release=args.release, device=args.device)
        with TestClient(create_app(service=manager)) as client:
            with BenchHTTPClient('http://testserver', client=client,
                                 timeout_s=None, get_retries=0) as bench:
                controller = select_geometry_policy(bench.list_policies(), robot_id)
                config['controller'] = controller
                save(output / 'config.json', config)
                request = {'task_id': args.task, 'robot_id': robot_id,
                    'scene_variant': 'original_v1', 'seed': args.seed,
                    'agent_id': 'privileged_scripted_cloth' if cloth_task else 'privileged_scripted_manipulation', 'agent_version': 'v2',
                    'agent_config_sha256': canonical_hash(config),
                    'track': 'system0', 'observation_profile': 'vision_proprio',
                    'policy_id': controller['policy_id'], 'policy_version': controller['version'],
                    'budgets': config['budgets']}
                save(output / 'create_request.json', request)
                ready = bench.create(request)
                save(output / 'episode_ready.json', ready)
                if (ready['spec']['agent_config_sha256'] != canonical_hash(config)
                        or ready['spec']['system0'] != controller
                        or ready['spec']['budgets'] != config['budgets']):
                    raise RuntimeError('Episode did not pin the baseline configuration')
                episode_id = ready['episode_id']
                native_session = manager.get(episode_id)
                interaction = InteractionSession(bench, episode_id, ready['capabilities'],
                    ready['spec'], ready['observation'], output, control_mode='fingertip',
                    visual_input='rgbd', tactile_input='on', limits=limits,
                    deadline=None,
                    max_actions=args.max_actions)
                termination = 'primitive_budget_exhausted'
                for index in range(args.max_actions):
                    if interaction.http(bench.get_result, episode_id) is not None:
                        termination = 'server_finished'
                        break
                    folder = output / 'decisions' / f'{index:04d}'
                    captured = interaction.context(folder / 'before')
                    observation = captured['observation']
                    oracle = privileged_state(native_session, observation)
                    previous_feedback = deepcopy(interaction.feedback)
                    save(output / 'private' / 'strategy_inputs' / f'{index:04d}.json',
                         {'observation_sequence': observation['sequence'],
                          'privileged_state': oracle, 'previous_feedback': previous_feedback})
                    decision = strategy.next_decision(deepcopy(observation), deepcopy(oracle), previous_feedback)
                    save(folder / 'decision.json', decision)
                    diagnostic = strategy.diagnostic()
                    save(folder / 'strategy_diagnostic.json', diagnostic)
                    try:
                        feedback = interaction.move(decision, folder, f'scripted_{index:04d}',
                                                    observation['sequence'])
                    except ActionValidationError as exc:
                        save(folder / 'rejection.json', exc.public_metadata())
                        raise
                    save(folder / 'after_observation.json', interaction.observation)
                    after_oracle = privileged_state(native_session, interaction.observation)
                    object_id = task.get('public_goal', {}).get('object_id', task.get('object_id'))
                    progress = {'decision': index,
                        'sim_time_s': interaction.observation['sim_time_s'],
                        'action_status': feedback['status'], 'model_calls': 0,
                        'submitted': interaction.submitted, 'completed': interaction.completed,
                        'phase': diagnostic.get('phase', diagnostic.get('stage')),
                        'object_id': object_id,
                        'object_position_m': after_oracle['objects'].get(object_id, {}).get('position_m'),
                        'cloth_centroid_m': after_oracle.get('cloth', {}).get('centroid_m')}
                    save(output / 'progress.json', progress)
                    print(json.dumps({'event': 'primitive', **progress}), flush=True)
                    if interaction.result is not None:
                        termination = 'server_finished'
                        break
                    if interaction.stop_requested:
                        termination = 'strategy_requested_stop'
                        break
                # finish seals ordinary stop/budget exits; it never declares success.
                bench.finish(episode_id)
    except BaseException as exc:
        error = exc
        termination = 'baseline_error'
    finally:
        if manager is not None:
            try:
                manager.shutdown()
            except BaseException as exc:
                cleanup_errors.append({'type': type(exc).__name__, 'detail': str(exc)})
            for session in manager.sessions.values():
                if session.thread is not None:
                    session.thread.join(timeout=35.)
                    if session.thread.is_alive():
                        cleanup_errors.append({'type': 'WorkerThreadStillAlive'})
                if session.worker is not None and not session.worker.closed:
                    try:
                        session.cancelled.set()
                        session.worker.close(force=True)
                        cleanup_errors.append({'type': 'UnsealedWorkerRequiredForceClose'})
                    except BaseException as exc:
                        cleanup_errors.append({'stage': 'force_close', 'type': type(exc).__name__, 'detail': str(exc)})
        directory = native_session.directory if native_session is not None else None
        result = (native_session.result.model_dump(mode='json')
                  if native_session is not None and native_session.result is not None else None)
        audit_path = directory / 'private/physics_audit.json' if directory else None
        score_path = directory / 'private/evaluator_evidence.json' if directory else None
        def report_read(stage, function, default=None):
            try:
                return function()
            except Exception as exc:
                cleanup_errors.append({'stage': stage, 'type': type(exc).__name__, 'detail': str(exc)})
                return default
        audit = report_read('read_physics_audit', lambda: load_json(audit_path)) if audit_path and audit_path.is_file() else None
        score = report_read('read_task_evidence', lambda: load_json(score_path)) if score_path and score_path.is_file() else None
        native = bool(native_session is not None and native_session.metadata.get('native_isaacsim_app'))
        artifacts_verified = report_read('verify_episode_artifacts', lambda: check_episode_artifacts(directory), False) if directory else False
        diagnostic = report_read('strategy_diagnostic', strategy.diagnostic) if strategy else None
        counts = {'primitive_attempts': interaction.move_attempts if interaction else 0,
                  'submitted_actions': interaction.submitted if interaction else 0,
                  'completed_actions': interaction.completed if interaction else 0}
        trajectory = report_read('package_trajectory', lambda: package_trajectory(output, ready, config, expected_counts=counts))
        trajectory_complete = bool(trajectory and trajectory['integrity']['complete'])
        task_success = bool(score and score.get('success') is True)
        physics_valid = bool(native and result and result['physics_valid']
                             and audit and audit.get('physics_valid') is True)
        accepted = bool(task_success and physics_valid and result['outcome'] == 'success'
                        and artifacts_verified and trajectory_complete and error is None and not cleanup_errors)
        report = {'schema': SCHEMA, 'scope': config['scope'], 'model_calls': 0,
            'privileged_state_used': True, 'object_state_writeback': False,
            'native_isaac': native, 'task_success': task_success,
            'physics_valid': physics_valid, 'accepted': accepted,
            'termination': termination, 'wall_s': time.monotonic()-started,
            'error': {'type': type(error).__name__, 'detail': str(error)} if error else None,
            'cleanup_errors': cleanup_errors, 'result': result,
            'episode_directory': str(directory) if directory else None,
            'primitive_attempts': interaction.move_attempts if interaction else 0,
            'submitted_actions': interaction.submitted if interaction else 0,
            'completed_actions': interaction.completed if interaction else 0,
            'strategy_diagnostic': diagnostic,
            'physics_audit': artifact(audit_path) if audit_path else None,
            'task_evidence': artifact(score_path) if score_path else None,
            'episode_artifacts_verified': artifacts_verified,
            'source_snapshot': artifact(output / 'runtime_imports.json'),
            'config': artifact(output / 'config.json'),
            'strategy_implementation': config['strategy_implementation'],
            'source_snapshot_manifest': config['source_snapshot_manifest'],
            'execution': config['execution'],
            'trajectory': trajectory,
            'trajectory_complete': trajectory_complete,
            'video_scope': 'Original decision-boundary RGB-D frames; no continuous-video claim.'}
        save(output / 'report.json', report)
    if error is not None and not isinstance(error, Exception):
        raise error
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--release', type=Path, help='Default: current rigid manipulation release or preserved original cloth release, selected by --task.')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--task', default='01_cube_push')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--max-actions', type=int, default=128)
    parser.add_argument('--seed', type=int, default=0)
    report = run(parser.parse_args())
    print(json.dumps({key: report[key] for key in ('accepted', 'task_success', 'physics_valid',
        'model_calls', 'termination', 'primitive_attempts', 'episode_directory', 'error')},
        ensure_ascii=False), flush=True)
    if not report['accepted']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
