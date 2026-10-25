"""Run the CPU test suite with one pytest process per test file.

The shared environment has no pytest-xdist. Separate processes, each with its
own --basetemp, give the same isolation and finish the suite in minutes.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import re
import shutil
import stat
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[2]
COUNT = re.compile(r'(\d+) (passed|failed|errors?|skipped|xfailed|xpassed)\b')
NO_TESTS_COLLECTED = 5


def remove_tree(path):
    """Delete the temporary tree; some tests freeze read-only bundles inside it."""
    def writable(function, target, _):
        os.chmod(os.path.dirname(target), stat.S_IRWXU)
        if os.path.isdir(target) and not os.path.islink(target):
            os.chmod(target, stat.S_IRWXU)
        function(target)
    shutil.rmtree(path, onexc=writable)


def run_file(path, temp_root, log_dir):
    command = [sys.executable, '-B', '-m', 'pytest', '-W', 'ignore', '-p', 'no:cacheprovider',
               '-rfE', f'--basetemp={temp_root / path.stem}', str(path.relative_to(ROOT))]
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    output = result.stdout + result.stderr
    if log_dir is not None:
        (log_dir / f'{path.stem}.log').write_text(output)
    lines = output.strip().splitlines()
    counts = {}
    for number, kind in COUNT.findall(lines[-1] if lines else ''):
        kind = 'errors' if kind.startswith('error') else kind
        counts[kind] = counts.get(kind, 0) + int(number)
    return path, result.returncode, counts, output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('paths', nargs='*', type=Path, help='test files; default: tests/test_*.py')
    parser.add_argument('--jobs', type=int, default=min(32, os.cpu_count() or 1))
    parser.add_argument('--log-dir', type=Path, help='keep one log per test file here')
    args = parser.parse_args(argv)
    paths = [path.resolve() for path in args.paths] or sorted((ROOT / 'tests').glob('test_*.py'))
    # Larger files usually take longer; starting them first balances the pool.
    paths.sort(key=lambda path: path.stat().st_size, reverse=True)
    if args.log_dir is not None:
        args.log_dir.mkdir(parents=True, exist_ok=True)
    temp_root = Path(tempfile.mkdtemp(prefix='dexbench-pytest-'))
    started = time.monotonic()
    try:
        with ThreadPoolExecutor(max_workers=args.jobs) as pool:
            results = list(pool.map(lambda path: run_file(path, temp_root, args.log_dir), paths))
    finally:
        remove_tree(temp_root)
    totals, failed = {}, []
    for path, code, counts, output in sorted(results, key=lambda row: row[0].name):
        for kind, number in counts.items():
            totals[kind] = totals.get(kind, 0) + number
        if code not in (0, NO_TESTS_COLLECTED):
            failed.append(path)
            summary = [line for line in output.splitlines() if line.startswith(('FAILED', 'ERROR'))]
            print(f'--- {path.relative_to(ROOT)} (exit {code})', *(summary or output.splitlines()[-20:]), sep='\n')
    print(f"{len(paths)} files, {', '.join(f'{n} {k}' for k, n in sorted(totals.items()))} "
          f'in {time.monotonic() - started:.0f}s with {args.jobs} jobs')
    return 1 if failed else 0


if __name__ == '__main__':
    raise SystemExit(main())
