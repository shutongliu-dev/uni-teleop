"""Freeze revised task rules against an unchanged mechanical scene release."""
import argparse
import json
from pathlib import Path

from scenes.workrooms.mechanical_assembly.referee_release import revise_rules


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--source-manifest-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = revise_rules(args.source, args.output, source_manifest_sha256=args.source_manifest_sha256)
    print(json.dumps({'release_id': result['release_id'], 'status': result['status']}))


if __name__ == '__main__':
    main()
