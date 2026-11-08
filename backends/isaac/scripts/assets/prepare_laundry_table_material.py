"""Copy an immutable laundry release to a fresh, unqualified table-binding candidate.

Run from a complete frozen source snapshot. Works independently for each of the
five tasks and three installations; it does not update catalog selection.
"""
import argparse
import json
from pathlib import Path

from scenes.workrooms.laundry_room.table_material import PROFILE
from scenes.workrooms.laundry_room.table_material_release import prepare


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-release', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--profile', choices=(PROFILE,), default=PROFILE)
    args = parser.parse_args()
    manifest = prepare(args.source_release, args.output, profile=args.profile)
    print(json.dumps({'output': str(args.output.resolve()), 'release_id': manifest['release_id'],
                      'physics_accepted': manifest['physics_accepted'],
                      'table_material_derivation': manifest['table_material_derivation']}))


if __name__ == '__main__':
    main()
