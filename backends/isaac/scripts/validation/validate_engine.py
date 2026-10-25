"""Run native contact, material, rendering and room regressions from one snapshot.

No model calls or driver changes. Each check owns a separate Isaac process and
must pass its report, native log and observed process-exit checks.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import subprocess
import sys
import json

from bench.paths import load_json
from scripts.maintenance.freeze_project_sources import freeze, verify
from scripts.validation.validate_workroom_collection import (
    EXTRAS, digest, gpu_inventory, write_json,
)

PROJECT = Path(__file__).resolve().parents[2]
CHECKS = ('mesh_contacts', 'materials', 'render_sync', 'workrooms')


def cases(checks, families):
    result = []
    if 'mesh_contacts' in checks:
        result.append(('mesh_contacts', 'scripts.validation.validate_static_mesh_contacts', []))
    if 'materials' in checks:
        for mode in ('differentiated', 'default', 'bimanual_release'):
            result.append(('materials_' + mode, 'scripts.diagnostics.probe_robot_materials',
                           ['--mode', mode, '--family', families[0]]))
    if 'render_sync' in checks:
        result.append(('render_sync', 'scripts.validation.validate_render_sync',
                       ['--family', families[0]]))
    if 'workrooms' in checks:
        result.extend((family, 'scripts.validation.validate_workrooms', ['--family', family])
                      for family in families)
    return result


def run_case(source, output, env, name, module, arguments):
    verify(source)
    destination = output / 'runs' / name
    command = [str(source / 'scripts/isaac_python.sh'), '-m', module,
               '--output', str(destination), '--device', 'cuda:0', *arguments]
    log = output / 'logs' / (name + '.log')
    write_json(log.with_suffix('.launch.json'), {
        'command': command, 'cwd': str(source),
        'gpu': gpu_inventory(env['CUDA_VISIBLE_DEVICES']),
        'environment': {key: env[key] for key in (
            'CUDA_VISIBLE_DEVICES', 'ISAAC_BENCH_SINGLE_CUDA_DEVICE',
            'ISAAC_ALLOW_UNVALIDATED_DRIVER', 'ISAAC_BENCH_EXPERIENCE',
            'PYTHONPATH', 'ISAAC_BENCH_PYTHONPATH')}})
    print(json.dumps({'event': 'started', 'check': name, 'log': str(log)}), flush=True)
    with log.open('x') as handle:
        process = subprocess.run(command, cwd=source, env=env,
                                 stdout=handle, stderr=subprocess.STDOUT)
    lines = log.read_text(errors='replace').splitlines()
    errors = [line for line in lines if '[Error]' in line or 'Traceback (most recent call last)' in line]
    sync_warnings = [line for line in lines if 'geometry streaming are enabled together' in line]
    report_path = destination / 'report.json'
    report, error = {}, None
    try:
        report = load_json(report_path)
        if not isinstance(report, dict):
            raise ValueError('Native report must be a JSON object')
    except (OSError, ValueError) as exc:
        report, error = {}, f'{type(exc).__name__}: {exc}'
    result = {
        'check': name, 'module': module, 'exit_code': process.returncode,
        'passed': process.returncode == 0 and not errors and not sync_warnings
                  and error is None and report.get('passed') is True,
        'report': str(report_path), 'report_error': error,
        'report_sha256': digest(report_path) if report_path.is_file() else None,
        'log': str(log), 'log_sha256': digest(log),
        'native_error_count': len(errors), 'native_error_examples': errors[:10],
        'render_sync_warning_count': len(sync_warnings),
    }
    write_json(log.with_suffix('.exit.json'), result)
    verify(source)
    print(json.dumps({'event': 'finished', **{key: result[key] for key in (
        'check', 'passed', 'exit_code', 'native_error_count', 'render_sync_warning_count')}}), flush=True)
    return result


def main():
    from scenes.workrooms import FAMILIES
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--gpu', required=True, help='One full physical GPU UUID')
    parser.add_argument('--checks', nargs='+', choices=CHECKS, default=list(CHECKS))
    parser.add_argument('--families', nargs='+', choices=FAMILIES, default=list(FAMILIES))
    args = parser.parse_args()
    if not args.gpu.startswith('GPU-') or ',' in args.gpu:
        parser.error('One full physical GPU UUID is required')
    if len(set(args.checks)) != len(args.checks) or len(set(args.families)) != len(args.families):
        parser.error('Checks and families must be unique')
    source_status = subprocess.run(['git', '-C', str(PROJECT), 'status', '--porcelain'],
                                   text=True, capture_output=True, check=True).stdout
    if source_status.strip():
        raise RuntimeError('Commit the source changes before freezing native engine validation')
    gpu_before = gpu_inventory(args.gpu)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    source = output / 'source'
    links = ('bench/simulation/assets/shadow_hand',) if (PROJECT / 'bench/simulation/assets/shadow_hand').is_symlink() else ()
    manifest = freeze(PROJECT, source, EXTRAS, asset_links=links)
    for relative in manifest['files_sha256']:
        path = source / relative
        path.chmod(path.stat().st_mode & ~0o222)
    (source / 'source_snapshot_manifest.json').chmod(0o444)
    verify(source)
    (output / 'logs').mkdir()
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=args.gpu, ISAAC_BENCH_SINGLE_CUDA_DEVICE='1',
               ISAAC_ALLOW_UNVALIDATED_DRIVER='1', PYTHONDONTWRITEBYTECODE='1',
               ISAAC_BENCH_EXPERIENCE=str(source / 'bench/apps/workroom.headless.kit'),
               PYTHONPATH=str(source), ISAAC_BENCH_PYTHONPATH=str(source))
    selected = cases(args.checks, args.families)
    commit = subprocess.run(['git', '-C', str(PROJECT), 'rev-parse', 'HEAD'],
                            text=True, capture_output=True, check=True).stdout.strip()
    write_json(output / 'plan.json', {
        'schema': 'native_engine_validation_plan_v1',
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'checks': selected, 'source_project': str(PROJECT), 'source_commit': commit,
        'source_worktree_clean': True,
        'frozen_source': str(source),
        'source_manifest_sha256': digest(source / 'source_snapshot_manifest.json'),
        'gpu_before': gpu_before, 'driver_override': True,
        'driver_modified': False, 'model_calls': 0, 'task_qualification': False,
    })
    probe = subprocess.run([str(source / 'scripts/isaac_python.sh'), '-c',
        'import importlib,json,pathlib,sys; '
        'root=pathlib.Path.cwd().resolve(); '
        'paths=[str(pathlib.Path(importlib.import_module(m).__file__).resolve()) for m in sys.argv[1:]]; '
        'assert all(pathlib.Path(p).is_relative_to(root) for p in paths), paths; '
        'print(json.dumps({"root":str(root),"imports":paths}))',
        *sorted({module for _, module, _ in selected})],
        cwd=source, env=env, text=True, capture_output=True)
    write_json(output / 'sealed_entrypoint_probe.json', {
        'exit_code': probe.returncode, 'stdout': probe.stdout, 'stderr': probe.stderr,
        'launcher': str(source / 'scripts/isaac_python.sh')})
    if probe.returncode:
        raise RuntimeError('Sealed entrypoint/import probe failed')
    results = []
    for name, module, arguments in selected:
        results.append(run_case(source, output, env, name, module, arguments))
        write_json(output / 'results.json', {
            'schema': 'native_engine_validation_results_v1',
            'complete': len(results) == len(selected),
            'passed': len(results) == len(selected) and all(row['passed'] for row in results),
            'model_calls': 0, 'task_qualification': False, 'results': results})
    write_json(output / 'gpu_after.json', gpu_inventory(args.gpu))
    return 0 if all(row['passed'] for row in results) else 1


if __name__ == '__main__':
    sys.exit(main())
