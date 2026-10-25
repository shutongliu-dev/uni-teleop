"""Run an explicit bench command independently of an interactive tool session.

Only process completion is reported here. Physics/task acceptance stays with the
bench runner. Commands must reference credential files, never contain secrets.
There is no retry, resume, execution deadline, or process termination operation.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def now():
    return datetime.now(timezone.utc).isoformat()


def save(path, value):
    temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def identity(pid):
    try:
        fields = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
    except (FileNotFoundError, ProcessLookupError):
        return None
    return {'pid': pid, 'start_ticks': int(fields[19]), 'state': fields[0],
            'session_id': int(fields[3])}


def same_process(expected):
    if expected is None:
        return False
    current = identity(expected['pid'])
    return (current is not None and current['start_ticks'] == expected['start_ticks']
            and current['state'] != 'Z')


def session_members(session_id):
    result = []
    for path in Path('/proc').glob('[0-9]*/stat'):
        row = identity(int(path.parent.name))
        if row is not None and row['session_id'] == session_id and row['state'] != 'Z':
            result.append(row)
    return result


def status(directory):
    directory = Path(directory)
    finished = directory / 'completion.json'
    launched = directory / 'launched.json'
    if finished.exists():
        result = json.loads(finished.read_text())
        # Completion records the direct command exit, not native cleanup. A
        # surviving child retains its job session even after its parent exits.
        result['live_session_members'] = session_members(result['session_id'])
        return result
    if not launched.exists():
        return {'state': 'launch_incomplete', 'automatic_retry': False}
    info = json.loads(launched.read_text())
    supervisor_alive = same_process(info['supervisor'])
    worker = directory / 'command_started.json'
    worker_info = json.loads(worker.read_text()) if worker.exists() else None
    members = session_members(info['supervisor']['session_id'])
    return {'state': 'running' if supervisor_alive else 'interrupted_unsealed',
            'supervisor': info['supervisor'], 'supervisor_alive': supervisor_alive,
            'command': worker_info, 'live_session_members': members,
            'automatic_retry': False}


def supervise(directory):
    directory = Path(directory)
    plan = json.loads((directory / 'plan.json').read_text())
    own = identity(os.getpid())
    save(directory / 'launched.json', {'supervisor': own, 'started_at_utc': now()})
    started = time.monotonic()
    try:
        with (directory / 'command.log').open('xb') as stream:
            process = subprocess.Popen(plan['argv'], cwd=plan['cwd'],
                stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT)
            save(directory / 'command_started.json',
                 {'identity': identity(process.pid), 'started_at_utc': now()})
            code = process.wait()  # Intentionally no execution timeout.
        result = {'state': 'command_exited', 'exit_code': code}
    except Exception as error:
        result = {'state': 'launch_failed', 'error_type': type(error).__name__,
                  'message': str(error)}
    result.update(completed_at_utc=now(), elapsed_wall_s=time.monotonic() - started,
                  session_id=own['session_id'], task_acceptance_claimed=False,
                  native_cleanup_claimed=False, automatic_retry=False)
    save(directory / 'completion.json', result)


def launch(directory, argv, cwd):
    directory = Path(directory).resolve()
    cwd = Path(cwd).resolve(strict=True)
    if not cwd.is_dir() or not argv or not all(isinstance(v, str) and v for v in argv):
        raise ValueError('An existing working directory and nonempty command are required')
    directory.mkdir(parents=True, exist_ok=False)  # A job is never overwritten.
    save(directory / 'plan.json', {'argv': argv, 'cwd': str(cwd),
        'created_at_utc': now(), 'execution_timeout_s': None, 'automatic_retry': False})
    with (directory / 'supervisor.log').open('xb') as stream:
        supervisor = subprocess.Popen([sys.executable, '-B', str(Path(__file__).resolve()),
            '_supervise', str(directory)], cwd=cwd, stdin=subprocess.DEVNULL,
            stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
    return {'job_directory': str(directory), 'supervisor_pid': supervisor.pid,
            'automatic_retry': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='operation', required=True)
    start = sub.add_parser('launch')
    start.add_argument('--job-directory', type=Path, required=True)
    start.add_argument('--cwd', type=Path, required=True)
    start.add_argument('argv', nargs=argparse.REMAINDER)
    inspect = sub.add_parser('status')
    inspect.add_argument('job_directory', type=Path)
    internal = sub.add_parser('_supervise')
    internal.add_argument('job_directory', type=Path)
    args = parser.parse_args()
    if args.operation == '_supervise':
        supervise(args.job_directory)
    elif args.operation == 'status':
        print(json.dumps(status(args.job_directory)))
    else:
        argv = args.argv[1:] if args.argv[:1] == ['--'] else args.argv
        print(json.dumps(launch(args.job_directory, argv, args.cwd)))


if __name__ == '__main__':
    main()
