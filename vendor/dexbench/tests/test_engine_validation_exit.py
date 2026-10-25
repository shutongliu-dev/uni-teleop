"""A sealed native report cannot hide a crash or a rendering regression."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from bench.paths import load_json
from scripts.validation import validate_engine as runner


@pytest.mark.parametrize('exit_code,line,report,passed', [
    (0, '', {'passed': True}, True),
    (1, '', {'passed': True}, False),
    (0, '[Error] native solver failure', {'passed': True}, False),
    (0, 'Traceback (most recent call last):', {'passed': True}, False),
    (0, '[Warning] Fabric and geometry streaming are enabled together', {'passed': True}, False),
    (0, '', {'passed': False}, False),
    (0, '', {'passed': 'true'}, False),
    (0, '', None, False),
    (0, '', [], False),
])
def test_native_report_exit_and_log_must_all_pass(tmp_path, monkeypatch, exit_code, line, report, passed):
    source, output = tmp_path/'source', tmp_path/'result'
    source.mkdir()
    (output/'logs').mkdir(parents=True)
    monkeypatch.setattr(runner, 'verify', lambda source: None)
    monkeypatch.setattr(runner, 'gpu_inventory', lambda gpu: {'uuid': gpu})
    def native(command, *, cwd, env, stdout, stderr):
        target = Path(command[command.index('--output')+1])
        target.mkdir(parents=True)
        if report is not None:
            (target/'report.json').write_text(json.dumps(report))
        stdout.write(line+'\n')
        return SimpleNamespace(returncode=exit_code)
    monkeypatch.setattr(runner.subprocess, 'run', native)
    env = {key: 'probe' for key in (
        'CUDA_VISIBLE_DEVICES', 'ISAAC_BENCH_SINGLE_CUDA_DEVICE',
        'ISAAC_ALLOW_UNVALIDATED_DRIVER', 'ISAAC_BENCH_EXPERIENCE',
        'PYTHONPATH', 'ISAAC_BENCH_PYTHONPATH')}
    result = runner.run_case(source, output, env, 'probe', 'test.module', [])
    assert result['passed'] is passed
    assert result['exit_code'] == exit_code
    assert load_json(output/'logs/probe.exit.json') == result
