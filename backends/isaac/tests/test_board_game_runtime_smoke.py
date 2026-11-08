"""Checks of the operator smoke evidence guard; these are CPU-only fixtures."""
import json
import subprocess
from types import SimpleNamespace

import pytest

from scripts.validation.board_game_runtime_smoke import gpu_selection, native_coverage, run


def save_fixture(tmp_path, *, count=4, gap=False, unsupported=False, late=False):
    private = tmp_path/'private'
    board = private/'board_game'
    board.mkdir(parents=True)
    (private/'metadata.json').write_text(json.dumps({'physics_dt_s': .0005}))
    summary = {'samples': count, 'final_context_verified': True, 'failed': False,
               'unsupported_samples': 0, 'commits': 0, 'physical_steps_added': 0, 'batch_view_token_count': 2}
    (board/'evidence.json').write_text(json.dumps({'observer': summary, 'object_state_writes': False}))
    rows = []
    for index in range(count):
        rows.append({'frame': {'substep_index': 100+index+(1 if gap and index == 3 else 0),
            'simulation_time_s': (index+1)*.0005+(1. if late else 0.), 'dt_s': .0005,
            'contacts_complete': not unsupported, 'source': 'native_solver_substeps',
            'pieces': {'black_00': {}, 'white_00': {}}}, 'unsupported': []})
    (board/'native_frames.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in rows))
    return private


def test_saved_native_coverage_requires_complete_clock_and_inventory(tmp_path):
    result = native_coverage(save_fixture(tmp_path), .002)
    assert result['passed']
    assert result['recorded_samples'] == 4
    assert result['inventory_count'] == 2


@pytest.mark.parametrize('changes', [dict(count=3), dict(gap=True), dict(unsupported=True), dict(late=True)])
def test_missing_unsupported_or_mismatched_native_samples_fail(tmp_path, changes):
    assert not native_coverage(save_fixture(tmp_path, **changes), .002)['passed']


def test_smoke_refuses_editable_source_before_native_start(tmp_path):
    with pytest.raises(ValueError, match='complete snapshot root'):
        run(SimpleNamespace(output=tmp_path/'unused'))
    assert not (tmp_path/'unused').exists()


def test_gpu_identity_requires_uuid_and_matching_single_device_environment(monkeypatch):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', 'GPU-good')
    monkeypatch.setenv('ISAAC_BENCH_SINGLE_CUDA_DEVICE', '1')
    monkeypatch.setenv('ISAAC_ALLOW_UNVALIDATED_DRIVER', '1')
    with pytest.raises(ValueError, match='explicit --gpu-uuid'):
        gpu_selection('1')
    with pytest.raises(ValueError, match='explicit --gpu-uuid'):
        gpu_selection('GPU-other')


@pytest.mark.parametrize('pci,free', [('00000000:BB:00.0', '35000'), ('00000000:67:00.0', '35000'),
                                     ('00000000:AB:00.0', '15000')])
def test_gpu_health_exclusions_and_free_memory_guard(monkeypatch, pci, free):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', 'GPU-good')
    monkeypatch.setenv('ISAAC_BENCH_SINGLE_CUDA_DEVICE', '1')
    monkeypatch.setenv('ISAAC_ALLOW_UNVALIDATED_DRIVER', '1')
    monkeypatch.setattr('scripts.validation.board_game_runtime_smoke.subprocess.run',
        lambda *args, **kwargs: SimpleNamespace(stdout=f'GPU-good,{pci},RTX,48000,{free},580\n'))
    with pytest.raises(ValueError):
        gpu_selection('GPU-good')


def test_gpu_identity_query_retries_only_a_stalled_nvidia_smi(monkeypatch):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', 'GPU-good')
    monkeypatch.setenv('ISAAC_BENCH_SINGLE_CUDA_DEVICE', '1')
    monkeypatch.setenv('ISAAC_ALLOW_UNVALIDATED_DRIVER', '1')
    calls = []

    def stalled_once(argv, **kwargs):
        calls.append(kwargs['timeout'])
        if len(calls) == 1:
            raise subprocess.TimeoutExpired(argv, kwargs['timeout'])
        return SimpleNamespace(stdout='GPU-good,00000000:9B:00.0,RTX,48000,35000,580\n')
    monkeypatch.setattr('scripts.validation.board_game_runtime_smoke.subprocess.run', stalled_once)
    assert gpu_selection('GPU-good')['memory_free_mb'] == '35000' and len(calls) == 2

    def always_stalled(argv, **kwargs):
        raise subprocess.TimeoutExpired(argv, kwargs['timeout'])
    monkeypatch.setattr('scripts.validation.board_game_runtime_smoke.subprocess.run', always_stalled)
    with pytest.raises(subprocess.TimeoutExpired):
        gpu_selection('GPU-good')

    def failed(argv, **kwargs):
        raise subprocess.CalledProcessError(9, argv)
    monkeypatch.setattr('scripts.validation.board_game_runtime_smoke.subprocess.run', failed)
    with pytest.raises(subprocess.CalledProcessError):
        gpu_selection('GPU-good')
