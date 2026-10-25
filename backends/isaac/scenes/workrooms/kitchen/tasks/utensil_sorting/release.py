"""Prepare a frozen kitchen task candidate; publication does not claim success."""
import argparse
import json
from pathlib import Path
from bench.isaacsim.release import digest
from scenes.workrooms.kitchen.release import prepare_task_release


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--asset-manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    manifest = prepare_task_release('utensil_sorting', args.output, asset_manifest=args.asset_manifest)
    print(json.dumps({'output': str(args.output), 'status': manifest['status'],
        'manifest_sha256': digest(args.output/'release_manifest.json')}))
