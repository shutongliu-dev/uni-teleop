"""Offline launcher checks; source verification uses a tiny synthetic snapshot."""
import json
import os
import socket
import subprocess
from types import SimpleNamespace

import pytest

from scripts import run_push
from scripts.maintenance import freeze_project_sources as freezer


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def forbidden(*args, **kwargs):
    raise AssertionError('This offline check must not execute a run or contact a service')


@pytest.fixture
def launch(tmp_path, monkeypatch):
    datasets = tmp_path/'datasets'
    datasets.mkdir()
    source = tmp_path/'editable'
    for relative in ('bench/adapters/__init__.py', 'bench/__init__.py',
                     'agent/__init__.py', 'scenes/__init__.py',
                     'bench/simulation/__init__.py', 'scripts/isaac_python.sh'):
        path = source/relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('# synthetic source\n')
    (source/'scripts/isaac_python.sh').chmod(0o755)
    snapshot = datasets/'snapshot'
    freezer.freeze(source, snapshot, ('scripts/isaac_python.sh',))
    credentials = tmp_path/'profile.json'
    # The external key deliberately does not exist: dry-run must not read it.
    write_json(credentials, {'base_url':run_push.GATEWAY,
                             'api_key_file':str(tmp_path/'missing_external_key.json')})
    monkeypatch.setattr(run_push, 'DATASETS', datasets)
    monkeypatch.setattr(run_push, 'os', SimpleNamespace(environ=os.environ, chdir=forbidden, execvpe=forbidden))
    monkeypatch.setattr(socket.socket, 'connect', forbidden)
    monkeypatch.setattr(socket, 'create_connection', forbidden)
    return SimpleNamespace(run=datasets/'new_run', snapshot=snapshot, credentials=credentials, datasets=datasets)


def start_args(launch):
    return ['start', '--run', str(launch.run), '--bench-url', 'http://127.0.0.1:8123',
            '--project-root', str(launch.snapshot), '--credentials', str(launch.credentials), '--dry-run']


def test_start_requires_explicit_source_snapshot(launch, capsys):
    argv = start_args(launch)
    del argv[argv.index('--project-root'):argv.index('--project-root')+2]
    with pytest.raises(SystemExit) as error:
        run_push.main(argv)
    assert error.value.code == 2
    assert '--project-root' in capsys.readouterr().err
    assert not launch.run.exists()


def test_dry_run_only_verifies_source_without_loading_keys_or_writing_run(launch, tmp_path, monkeypatch, capsys):
    before = {str(p.relative_to(tmp_path)):p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}
    calls = []

    def verify_only(argv, **kwargs):
        assert argv == [str(run_push.CPU), '-B',
                        str(run_push.PROJECT/'scripts/maintenance/freeze_project_sources.py'),
                        'verify', str(launch.snapshot)]
        assert kwargs['cwd'] == launch.snapshot
        assert kwargs['env']['PYTHONPATH'] == str(launch.snapshot)
        assert kwargs['env']['PYTHONDONTWRITEBYTECODE'] == '1'
        calls.append(argv)
        return subprocess.run(argv, **kwargs)

    monkeypatch.setattr(run_push, 'subprocess', SimpleNamespace(
        run=verify_only, CalledProcessError=subprocess.CalledProcessError))
    run_push.main(start_args(launch))
    planned = json.loads(capsys.readouterr().out)
    assert len(calls) == 1 and planned['executes'] is False and planned['episodes'] == 1
    assert planned['cwd'] == str(launch.snapshot)
    argv = planned['argv']
    assert argv[:4] == [str(run_push.CPU), '-B', '-m', 'agent.model_client.runner']
    for flag, value in {'--task':'01_cube_push', '--model':'gpt-6-astra', '--reasoning-effort':'max',
                        '--control-mode':'fingertip', '--interaction-mode':'action',
                        '--output':str(launch.run)}.items():
        assert argv[argv.index(flag)+1] == value
    assert not launch.run.exists()
    assert before == {str(p.relative_to(tmp_path)):p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}


@pytest.mark.parametrize('kind', ['outside', 'datasets_root', 'inside_snapshot', 'existing', 'symlink', 'dotdot'])
def test_start_rejects_unsafe_or_reused_output_before_verification(launch, tmp_path, monkeypatch, kind):
    if kind == 'outside':
        launch.run = tmp_path/'outside'
    elif kind == 'datasets_root':
        launch.run = launch.datasets
    elif kind == 'inside_snapshot':
        launch.run = launch.snapshot/'new_run'
    elif kind == 'existing':
        launch.run.mkdir()
        (launch.run/'keep').write_text('existing evidence')
    elif kind == 'symlink':
        alias = launch.datasets/'alias'
        alias.symlink_to(launch.snapshot, target_is_directory=True)
        launch.run = alias/'new_run'
    else:
        launch.run = launch.datasets/'unused'/'..'/'new_run'
    monkeypatch.setattr(run_push, 'subprocess', SimpleNamespace(
        run=forbidden, CalledProcessError=subprocess.CalledProcessError))
    before = {str(p):p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}
    with pytest.raises(SystemExit) as error:
        run_push.main(start_args(launch))
    assert error.value.code == 2
    assert before == {str(p):p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}


@pytest.mark.parametrize('change', ['missing_manifest', 'changed_source'])
def test_unverified_snapshot_cannot_start(launch, capsys, change):
    if change == 'missing_manifest':
        (launch.snapshot/'source_snapshot_manifest.json').unlink()
    else:
        (launch.snapshot/'bench/__init__.py').write_text('# changed source\n')
    with pytest.raises(SystemExit) as error:
        run_push.main(start_args(launch))
    assert error.value.code == 2
    expected = 'complete frozen source snapshot' if change == 'missing_manifest' else 'Frozen source verification failed'
    assert expected in capsys.readouterr().err
    assert not launch.run.exists()


@pytest.mark.parametrize('use_retained', [False, True])
def test_status_reads_historical_runs_without_snapshot_or_network(launch, tmp_path, monkeypatch, capsys, use_retained):
    history = tmp_path/'historical_run'
    write_json(history/'episode_ready.json', {'episode_id':'old_episode', 'spec':{'task_id':'01_cube_push'}})
    write_json(history/'run_report.json', {'schema':'online_visual_agent_run.v3',
        'model_call_attempts':9, 'model_responses':8, 'termination':'client_decision_budget',
        'task_success':False, 'result':{'outcome':'cancelled', 'physics_valid':False}})
    write_json(history/'decisions/0000/execution.json', {'status':'program_completed'})
    write_json(history/'decisions/0000/tools/0000/execution.json', {'status':'completed', 'action_id':'a0'})
    partial = history/'decisions/0001/execution.json'
    partial.parent.mkdir(parents=True)
    partial.write_text('{')
    retained = tmp_path/'retained_run.json'
    write_json(retained, {'run_directory':str(history)})
    monkeypatch.setattr(run_push, 'RETAINED_RUN', retained)
    monkeypatch.setattr(run_push, 'subprocess', SimpleNamespace(
        run=forbidden, CalledProcessError=subprocess.CalledProcessError))
    before = {str(p):p.read_bytes() for p in history.rglob('*') if p.is_file()}
    run_push.main(['status'] if use_retained else ['status', '--run', str(history)])
    result = json.loads(capsys.readouterr().out)
    assert result['state'] == 'report_available' and result['episode_id'] == 'old_episode'
    assert result['task'] == '01_cube_push' and result['model_requests'] == 9 and result['model_responses'] == 8
    assert result['completed_actions'] == 1 and result['recorded_action_statuses'] == {'completed':1}
    assert result['server_outcome'] == 'cancelled' and result['reported_task_success'] is False
    assert result['physics_valid'] is False
    assert result['process_liveness'] == result['independent_acceptance'] == 'not_checked'
    assert result['unreadable_or_incomplete_records'] == ['decisions/0001/execution.json']
    assert before == {str(p):p.read_bytes() for p in history.rglob('*') if p.is_file()}
