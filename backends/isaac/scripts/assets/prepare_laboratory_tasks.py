"""Freeze a laboratory task release from a versioned task asset and frozen source."""
import argparse
import json
from pathlib import Path

from bench.isaacsim.release import digest
from scenes.workrooms.laboratory.composition import TASK_IDS
from scenes.workrooms.laboratory.release import prepare_task_release


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', choices=TASK_IDS, required=True)
    parser.add_argument('--asset-manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    manifest = prepare_task_release(args.task, args.output, asset_manifest=args.asset_manifest)
    print(json.dumps({'task_id': args.task, 'output': str(args.output), 'status': manifest['status'],
        'manifest_sha256': digest(args.output/'release_manifest.json')}))


if __name__ == '__main__':
    main()
