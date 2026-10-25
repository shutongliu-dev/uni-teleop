"""Freeze the bench and validate Astra's six rooms sequentially on one GPU UUID.

Each native attempt gets its own log, source manifest and result. This is an
engineering dynamics check, with no model calls or manipulation task score.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

from bench.paths import load_json
from scripts.maintenance.freeze_project_sources import DEFAULT_EXTRAS, freeze, verify

PROJECT = Path(__file__).resolve().parents[2]
EXTRAS = (*DEFAULT_EXTRAS, 'bench/apps/workroom.headless.kit')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def gpu_inventory(uuid):
    query = ['nvidia-smi', '--id=' + uuid,
             '--query-gpu=uuid,pci.bus_id,memory.used,memory.total,utilization.gpu',
             '--format=csv,noheader,nounits']
    result = subprocess.run(query, capture_output=True, text=True, check=True)
    rows = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if len(rows) != 1 or rows[0].split(',')[0].strip() != uuid:
        raise ValueError('GPU UUID must resolve to exactly one physical device')
    fields = [part.strip() for part in rows[0].split(',')]
    if fields[1].upper().split(':')[-2] in ('67', 'BB', 'DB'):
        raise ValueError('This GPU is excluded by the recorded rendering failures')
    return dict(zip(('uuid', 'pci_bus_id', 'memory_used_mib', 'memory_total_mib',
                     'utilization_percent'), fields, strict=True))


def main():
    from scenes.workrooms import FAMILIES
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--gpu', required=True, help='Full physical GPU UUID')
    parser.add_argument('--families', nargs='+', choices=FAMILIES, default=list(FAMILIES))
    parser.add_argument('--check', choices=('dynamics', 'observations', 'base'), default='dynamics')
    parser.add_argument('--scene-json', type=Path, help='Single-family observation/base candidate override')
    args = parser.parse_args()
    if not args.gpu.startswith('GPU-') or ',' in args.gpu:
        parser.error('Supply one full physical GPU UUID')
    if len(args.families) != len(set(args.families)):
        parser.error('Each family may occur only once per collection')
    if args.scene_json and (args.check not in ('observations', 'base') or len(args.families) != 1):
        parser.error('--scene-json requires --check observations/base and exactly one family')
    if args.check == 'base' and args.families != ['mechanical_assembly']:
        parser.error('--check base requires --families mechanical_assembly')
    gpu_before = gpu_inventory(args.gpu)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    source = output / 'source'
    asset_links = ('bench/simulation/assets/shadow_hand',) if (PROJECT / 'bench/simulation/assets/shadow_hand').is_symlink() else ()
    manifest = freeze(PROJECT, source, EXTRAS, asset_links=asset_links)
    for relative in manifest['files_sha256']:
        path = source / relative
        path.chmod(path.stat().st_mode & ~0o222)
    (source / 'source_snapshot_manifest.json').chmod(0o444)
    verify(source)
    source_commit = subprocess.run(['git', '-C', str(PROJECT), 'rev-parse', 'HEAD'],
                                   text=True, capture_output=True, check=False)
    (output / 'logs').mkdir()
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=args.gpu, ISAAC_BENCH_SINGLE_CUDA_DEVICE='1',
               ISAAC_ALLOW_UNVALIDATED_DRIVER='1', PYTHONDONTWRITEBYTECODE='1',
               ISAAC_BENCH_EXPERIENCE=str(source / 'bench/apps/workroom.headless.kit'),
               PYTHONPATH=str(source), ISAAC_BENCH_PYTHONPATH=str(source))
    plan = {'schema': 'workroom_collection_plan.v1',
            'created_utc': datetime.now(timezone.utc).isoformat(),
            'families': args.families, 'gpu_before': gpu_before,
            'source_project': str(PROJECT), 'frozen_source': str(source),
            'source_commit': source_commit.stdout.strip() if source_commit.returncode == 0 else None,
            'source_manifest_sha256': digest(source / 'source_snapshot_manifest.json'),
            'driver_override': True, 'model_calls': 0,
            'check': args.check,
            'scope': 'Native '+args.check+' diagnostics; not manipulation task qualification'}
    write_json(output / 'plan.json', plan)
    probe = subprocess.run([str(source / 'scripts/isaac_python.sh'), '-c',
        'import json, pathlib, scenes.workrooms as rooms; '
        'import scripts.validation.validate_workrooms as validator; '
        'root = pathlib.Path.cwd().resolve(); '
        'paths = [str(pathlib.Path(m.__file__).resolve()) for m in (rooms, validator)]; '
        'assert all(pathlib.Path(p).is_relative_to(root) for p in paths), paths; '
        'print(json.dumps({"source_root": str(root), "imports": paths}))'],
        cwd=source, env=env, text=True, capture_output=True)
    write_json(output / 'sealed_entrypoint_probe.json', {
        'exit_code': probe.returncode, 'stdout': probe.stdout, 'stderr': probe.stderr,
        'launcher': str(source / 'scripts/isaac_python.sh')})
    if probe.returncode:
        raise RuntimeError('Sealed native entrypoint/import probe failed; see its record')
    module = {'observations': 'scripts.validation.validate_workroom_observations',
              'dynamics': 'scripts.validation.validate_workrooms',
              'base': 'scripts.diagnostics.probe_workroom_base'}[args.check]
    results = []
    for family in args.families:
        verify(source)
        command = [str(source / 'scripts/isaac_python.sh'), '-m',
                   module,
                   '--output', str(output / 'runs' / family), '--device', 'cuda:0']
        if args.check != 'base':
            command.extend(['--family', family])
        if args.scene_json:
            command.extend(['--scene-json', str(args.scene_json.resolve())])
        log = output / 'logs' / (family + '.log')
        launch = {'command': command, 'cwd': str(source), 'gpu': gpu_inventory(args.gpu),
                  'environment': {key: env[key] for key in ('CUDA_VISIBLE_DEVICES',
                      'ISAAC_BENCH_SINGLE_CUDA_DEVICE', 'ISAAC_ALLOW_UNVALIDATED_DRIVER',
                      'ISAAC_BENCH_EXPERIENCE',
                      'PYTHONPATH', 'ISAAC_BENCH_PYTHONPATH')}}
        write_json(output / 'logs' / (family + '.launch.json'), launch)
        print(json.dumps({'event': 'started', 'family': family, 'log': str(log)}), flush=True)
        with log.open('w') as handle:
            process = subprocess.run(command, cwd=source, env=env,
                                     stdout=handle, stderr=subprocess.STDOUT)
        native_errors = [line for line in log.read_text(errors='replace').splitlines()
                         if '[Error]' in line or 'Traceback (most recent call last)' in line]
        report_path = output / 'runs' / family / 'report.json'
        report, report_error = {}, None
        try:
            report = load_json(report_path)
            if not isinstance(report, dict):
                raise ValueError('Native report is not a JSON object')
        except (OSError, ValueError) as exc:
            report, report_error = {}, f'{type(exc).__name__}: {exc}'
        passed = (process.returncode == 0 and not native_errors and report_error is None
                  and (report.get('passed') is True or report.get('status') == 'passed'))
        result = {'family': family, 'exit_code': process.returncode, 'passed': passed,
                  'report': str(report_path), 'report_sha256': digest(report_path) if report_path.is_file() else None,
                  'report_error': report_error,
                  'log': str(log), 'log_sha256': digest(log),
                  'native_error_count': len(native_errors), 'native_error_examples': native_errors[:10]}
        write_json(output / 'logs' / (family + '.exit.json'), result)
        results.append(result)
        write_json(output / 'results.json', {'schema': 'workroom_collection_results.v1',
            'complete': len(results) == len(args.families),
            'passed': len(results) == len(args.families) and all(row['passed'] for row in results),
            'model_calls': 0, 'scope': plan['scope'], 'results': results})
        print(json.dumps({'event': 'finished', 'family': family, 'exit_code': process.returncode,
                          'passed': passed, 'native_errors': len(native_errors)}), flush=True)
    verify(source)
    write_json(output / 'gpu_after.json', gpu_inventory(args.gpu))
    return 0 if all(row['passed'] for row in results) else 1


if __name__ == '__main__':
    sys.exit(main())
