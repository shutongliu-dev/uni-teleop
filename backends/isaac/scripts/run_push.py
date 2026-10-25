#!/usr/bin/env python3
"""Run one RGB-D Astra/max fingertip push episode, or read an existing run.

Use an explicit source snapshot and a matching bench HTTP service. This command
does not start a server, queue, or background process.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import subprocess
import sys
from urllib.parse import urlsplit

from bench.paths import DATA_ROOT, load_json


PROJECT = Path(__file__).resolve().parents[1]
DATASETS = Path(f'{DATA_ROOT}/datasets')
CPU = Path(sys.executable)
PROFILE = PROJECT / 'configs/agent_bench/local_gateway_credentials.json'
RETAINED_RUN = PROJECT / 'configs/agent_bench/retained_run.json'
GATEWAY = 'http://127.0.0.1:8788/v1'


def read_json(path):
    return load_json(path)


def command(args):
    """Build the canonical runner CLI; no HTTP, model calls or output writes."""
    run = args.run.expanduser().absolute()
    root = args.project_root.expanduser().absolute()
    if (run.resolve() != run or not run.is_relative_to(DATASETS)
            or run == DATASETS or run.is_relative_to(root)):
        raise ValueError('Use a separate, non-aliased run directory under datasets')
    if run.exists():
        raise ValueError('Run directory already exists; use status or choose a new directory')
    if root.resolve() != root or not (root / 'source_snapshot_manifest.json').is_file():
        raise ValueError('project-root must be a complete frozen source snapshot')
    origin = urlsplit(args.bench_url)
    if (origin.scheme not in ('http', 'https') or not origin.hostname
            or origin.username or origin.password or origin.query or origin.fragment
            or origin.path not in ('', '/') or origin.port == 0):
        raise ValueError('bench-url must be an HTTP(S) origin without credentials or path')
    profile = read_json(args.credentials)
    if (profile.get('base_url', '').rstrip('/') != GATEWAY
            or not isinstance(profile.get('api_key_file'), str) or 'api_key' in profile):
        raise ValueError('Use the external-key codex2 credential profile for port 8788')
    argv = [str(CPU), '-B', '-m', 'agent.model_client.runner',
            '--bench-url', args.bench_url, '--credentials', str(args.credentials.absolute()),
            '--model', 'gpt-6-astra', '--reasoning-effort', 'max',
            '--task', '01_cube_push', '--robot', 'sharpa_w01_right',
            '--control-mode', 'fingertip', '--interaction-mode', 'action',
            '--visual-input', 'rgbd', '--tactile-input', 'on',
            '--max-decisions', '128', '--max-actions', '128',
            '--max-program-actions', '3', '--max-tool-calls-per-decision', '8',
            '--sim-budget', 'none', '--wall-budget', 'none',
            '--model-timeout', 'none', '--http-timeout', 'none',
            '--program-timeout', 'none', '--max-output-tokens', 'none',
            '--output', str(run)]
    return root, argv


def status(run):
    """Summarize public client artifacts, without claiming process liveness."""
    if run is None:
        try:
            current = read_json(RETAINED_RUN)
            directory = current.get('run_directory') if isinstance(current, dict) else None
            if not isinstance(directory, str) or not directory.strip():
                raise ValueError('Missing run_directory')
            run = Path(directory)
        except (OSError, ValueError):
            raise ValueError('No usable retained run in configs/agent_bench/retained_run.json; '
                             'set run_directory or pass status --run PATH') from None
    run = run.expanduser().absolute()
    if not run.is_dir():
        raise ValueError('Run directory does not exist')
    warnings = []

    def record(path):
        try:
            value = read_json(path)
            if not isinstance(value, dict):
                raise ValueError('Expected object')
            return value
        except (OSError, ValueError):
            warnings.append(str(path.relative_to(run)))
            return {}

    report_path = run / 'run_report.json'
    report = record(report_path) if report_path.exists() else {}
    ready_path = run / 'episode_ready.json'
    ready = record(ready_path) if ready_path.exists() else {}
    decisions = run / 'decisions'
    requests = list(decisions.glob('*/model_request.json'))
    responses = list(decisions.glob('*/model_response.json'))
    feedback = [record(path) for path in decisions.glob('**/execution.json')]
    # Code/free runs also have outer program receipts named execution.json.
    # Keep these separate from the public action lifecycle counts.
    action_states = {'accepted', 'running', 'completed', 'cancelled', 'failed'}
    counts = Counter(row['status'] for row in feedback if row.get('status') in action_states)
    result = report.get('result')
    result = result if isinstance(result, dict) else {}
    spec = result.get('spec') or ready.get('spec')
    spec = spec if isinstance(spec, dict) else {}
    return {
        'run': str(run),
        'state': 'report_available' if report else 'awaiting_terminal_report',
        'process_liveness': 'not_checked',
        'episode_id': report.get('episode_id') or ready.get('episode_id'),
        'task': spec.get('task_id'),
        'model_requests': report.get('model_call_attempts', len(requests)),
        'model_responses': report.get('model_responses', len(responses)),
        'primitive_attempts': report.get('primitive_attempts'),
        'submitted_actions': report.get('submitted_actions'),
        'completed_actions': report.get('completed_actions', counts.get('completed', 0)),
        'recorded_action_statuses': dict(counts),
        'termination': report.get('termination'),
        'error_type': report.get('error_type'),
        'reported_task_success': report.get('task_success'),
        'server_outcome': result.get('outcome'),
        'physics_valid': result.get('physics_valid'),
        'independent_acceptance': 'not_checked',
        'unreadable_or_incomplete_records': warnings,
    }


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    commands = p.add_subparsers(dest='command', required=True)
    start = commands.add_parser('start', help='Run one episode in the foreground')
    start.add_argument('--run', type=Path, required=True, help='New output directory under datasets')
    start.add_argument('--bench-url', required=True, help='Existing matching bench HTTP service origin')
    start.add_argument('--project-root', type=Path, required=True,
                       help='Complete frozen client source root; select explicitly for every new run')
    start.add_argument('--credentials', type=Path, default=PROFILE,
                       help='8788 profile referring to an external key file')
    start.add_argument('--dry-run', action='store_true', help='Verify source and print argv; do not execute')
    state = commands.add_parser('status', help='Read client artifacts; no polling or network')
    state.add_argument('--run', type=Path,
                       help='Client directory; default: retained_run.json run_directory')
    return p


def main(argv=None):
    p = parser()
    args = p.parse_args(argv)
    try:
        if args.command == 'status':
            print(json.dumps(status(args.run), ensure_ascii=False, indent=2))
            return
        root, command_argv = command(args)
        env = os.environ.copy()
        env.update(PYTHONPATH=str(root), PYTHONDONTWRITEBYTECODE='1')
        # Reuse the project's complete-source verifier; never copy a partial client.
        subprocess.run([str(CPU), '-B', str(PROJECT / 'scripts/maintenance/freeze_project_sources.py'),
                        'verify', str(root)], cwd=root, env=env, check=True, capture_output=True)
        if args.dry_run:
            print(json.dumps({'cwd': str(root), 'argv': command_argv,
                              'episodes': 1, 'executes': False}, ensure_ascii=False, indent=2))
            return
        os.chdir(root)
        os.execvpe(command_argv[0], command_argv, env)
    except (ValueError, OSError, subprocess.CalledProcessError) as exc:
        message = 'Frozen source verification failed' if isinstance(exc, subprocess.CalledProcessError) else str(exc)
        p.exit(2, message + '\n')


if __name__ == '__main__':
    main()
