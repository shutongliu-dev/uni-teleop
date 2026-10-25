"""List, export or freeze explicit tabletop scene/task/initial-state candidates.

Examples (CPU only):
  python -m scripts.assets.prepare_bench_scenarios list
  python -m scripts.assets.prepare_bench_scenarios export --scene clear_table \
    --split development --seed 11 --output <data>/datasets/<new-dir>
  python -m scripts.assets.prepare_bench_scenarios prepare-release --scene clear_table \
    --task 01_cube_push --split evaluation --seed 101 --output <data>/datasets/<new-release>

For frozen-source use, include scenes/catalog.json in snapshot extras.
Selection is explicit; this command never switches the API's default release.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from bench.paths import DATA_ROOT, OUTPUT_ROOT
from scenes.registry import DEFAULT_REGISTRY, load_registry
from scenes.release import export_scene, prepare_release


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--registry', type=Path, default=DEFAULT_REGISTRY)
    commands = parser.add_subparsers(dest='command', required=True)
    commands.add_parser('list')
    for name in ('export', 'prepare-release'):
        sub = commands.add_parser(name)
        sub.add_argument('--scene', required=True)
        sub.add_argument('--split', choices=['development', 'evaluation', 'diagnostic'], required=True)
        sub.add_argument('--seed', type=int, required=True)
        sub.add_argument('--task', required=name == 'prepare-release')
        sub.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == 'list':
        registry = load_registry(args.registry)
        value = {k: registry[k] for k in ('schema', 'version', 'status', 'physics_accepted', 'split_seeds', 'catalogs', 'scenes')}
    else:
        if not args.output.resolve().is_relative_to(OUTPUT_ROOT):
            parser.error(f'Exported runtime artifacts must be under {DATA_ROOT}/datasets')
        function = export_scene if args.command == 'export' else prepare_release
        value = function(args.scene, args.split, args.seed, args.output,
                         task_id=args.task, registry_path=args.registry)
        value = {'output': str(args.output.resolve()), 'status': value['status'],
                 'physics_accepted': False, 'selection': {'scene': args.scene, 'task': args.task,
                 'split': args.split, 'seed': args.seed}}
    print(json.dumps(value, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
