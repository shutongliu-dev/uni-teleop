"""Copy an immutable laundry release to a fresh render-only fabric visual candidate.

Run from a complete frozen source snapshot. Each --profile pins one baked shared
fabric profile to a scene cloth id; physics, reset, rules and evaluator are
unchanged. It does not update catalog selection.
"""
import argparse
import json
from pathlib import Path

from scenes.workrooms.laundry_room.fabric_visual_release import prepare


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-release', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--profile', nargs=2, action='append', required=True, metavar=('CLOTH_ID', 'PROFILE_JSON'))
    args = parser.parse_args()
    profiles = dict(args.profile)
    if len(profiles) != len(args.profile):
        parser.error('Pin each cloth id once')
    manifest = prepare(args.source_release, args.output, profiles)
    print(json.dumps({'output': str(args.output.resolve()), 'release_id': manifest['release_id'],
                      'physics_accepted': manifest['physics_accepted'],
                      'fabric_visual_derivation': manifest['fabric_visual_derivation']['profiles']}))


if __name__ == '__main__':
    main()
