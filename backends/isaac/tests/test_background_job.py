import json
from pathlib import Path
import subprocess
import sys
import time

import pytest

from bench.paths import load_json
from scripts.maintenance.background_job import launch, status, identity, same_process, save


def wait_finished(job):
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        if (job / 'completion.json').exists():
            return status(job)
        time.sleep(.02)
    raise AssertionError('Test child did not finish')


def test_command_survives_launcher_exit_and_preserves_failure(tmp_path):
    job = tmp_path / 'job'
    script = Path(__file__).parents[1] / 'scripts/maintenance/background_job.py'
    subprocess.run([sys.executable, str(script), 'launch', '--job-directory', str(job),
        '--cwd', str(tmp_path), '--', sys.executable, '-c',
        'import time; time.sleep(.15); print("finished"); raise SystemExit(7)'], check=True)
    result = wait_finished(job)
    assert result['exit_code'] == 7
    assert result['task_acceptance_claimed'] is False
    assert result['native_cleanup_claimed'] is False
    assert (job / 'command.log').read_text() == 'finished\n'
    assert load_json(job / 'plan.json')['execution_timeout_s'] is None
    with pytest.raises(FileExistsError):
        launch(job, [sys.executable, '-c', 'pass'], tmp_path)


def test_launch_failure_is_sealed_and_not_success(tmp_path):
    job = tmp_path / 'bad'
    launch(job, ['/this/command/does/not/exist'], tmp_path)
    result = wait_finished(job)
    assert result['state'] == 'launch_failed'
    assert result['error_type'] == 'FileNotFoundError'


def test_pid_reuse_and_missing_launch_are_not_running(tmp_path):
    own = identity(__import__('os').getpid())
    assert same_process(own)
    different = {**own, 'start_ticks': own['start_ticks'] + 1}
    assert not same_process(different)
    assert status(tmp_path)['state'] == 'launch_incomplete'
    save(tmp_path / 'launched.json', {'supervisor': different})
    assert status(tmp_path)['state'] == 'interrupted_unsealed'


def test_invalid_command_does_not_create_job(tmp_path):
    job = tmp_path / 'empty'
    with pytest.raises(ValueError):
        launch(job, [], tmp_path)
    assert not job.exists()
