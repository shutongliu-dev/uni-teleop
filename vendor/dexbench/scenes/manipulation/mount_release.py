"""Re-pin a frozen release to the hands' flange_axis_centred_v2 assemblies.

The v1 identity mount left the natural top-down right-hand posture (palm toward
the robot, thumb to its left) beyond fr3_joint7's limit, so tasks were done
back-handed. v2 turns every hand 180 deg about the flange axis (robot_registry).
This copies a release and changes only each scene's robot assembly pin and the
mount fields derived from it. Tasks, evaluator files, physics profile, assets and
initial states stay byte-identical, so a v2 release differs from its source only
in where the hand sits on the arm. The public hand frame follows the hand, so
hand-relative observations keep their meaning; only joint-7 solutions move.
Separate strata from their v1 sources: never pool the two.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import tempfile

from bench.isaacsim.release import digest, dump, validate
from bench.paths import load_json

PROJECT = Path(__file__).resolve().parents[2]
CONVENTION = 'flange_axis_centred_v2'


def repinned_scene(scene):
    from bench.simulation.isaacsim_backend.robot_registry import DEFAULT_HAND, get_hand_spec
    result = deepcopy(scene)
    robot = result['robot']
    hand = robot.get('hand_id', DEFAULT_HAND)
    version = get_hand_spec(hand)['release_assembly_version']
    spec = get_hand_spec(hand, version)
    if spec['mount_convention'] != CONVENTION:
        raise ValueError(f'{hand} release assembly {version} is not {CONVENTION}')
    previous = robot.get('assembly_version') or get_hand_spec(hand)['assembly_version']
    if get_hand_spec(hand, previous)['mount_convention'] != 'flange_axis_centred_v1':
        raise ValueError('Source scene is not a flange_axis_centred_v1 scene')
    robot.update(hand_id=hand, assembly_version=version, mount_position_m=spec['mount_translation_m'],
                 mount_quaternion_wxyz=spec['mount_quaternion_wxyz'])
    return result, {'hand_id': hand, 'from_assembly': previous, 'to_assembly': version}


def prepare(source, output):
    source, output = Path(source).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError('Choose a new release path; destination exists: ' + str(output))
    old = validate(source)
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix='.' + output.name + '.preparing-', dir=output.parent))
    try:
        shutil.rmtree(staging)
        shutil.copytree(source, staging, symlinks=True, ignore=shutil.ignore_patterns('__pycache__'))
        manifest = load_json(staging / 'release_manifest.json')
        changes = {}
        for entry in manifest['tasks']:
            path = staging / entry['scene']
            scene, change = repinned_scene(load_json(path))
            dump(path, scene)
            entry['scene_sha256'] = manifest['files'][entry['scene']]['sha256'] = digest(path)
            changes[entry['id']] = change
        # The copied source_release_manifest.json is the source's own source; keep that
        # chain in the provenance and point this release at its immediate source.
        upstream = {'source_release': manifest.get('source_release'), 'source_release_sha256': manifest.get('source_release_sha256')}
        shutil.copy2(source / 'release_manifest.json', staging / 'source_release_manifest.json')
        provenance = {'protocol': 'mount_convention_release_v1', 'mount_convention': CONVENTION,
                      'source_release': str(source), 'source_release_sha256': digest(source / 'release_manifest.json'),
                      'source_release_upstream': upstream, 'scene_changes': changes,
                      'unchanged': ['Tasks, evaluator files, physics profile, asset hashes and initial states (byte-identical).'],
                      'generator_sources': {str(p): digest(p) for p in (Path(__file__),
                          PROJECT / 'bench/simulation/isaacsim_backend/robot_registry.py')},
                      'physics_accepted': False, 'task_policy_success_claimed': False}
        for name, data in (('mount_provenance.json', provenance),):
            dump(staging / name, data)
        manifest['files']['mount_provenance.json'] = {'sha256': digest(staging / 'mount_provenance.json')}
        manifest.update(release_id=f"{manifest['release_id']}__{CONVENTION}_{output.name}",
                        status='prepared_unvalidated', prepared_at_utc=datetime.now(timezone.utc).isoformat(),
                        source_release=str(source), source_release_sha256=digest(source / 'release_manifest.json'),
                        original_release_modified=False, mount_convention=CONVENTION,
                        physics_accepted=False, task_policy_success_claimed=False,
                        purpose=(manifest.get('purpose', '') + f' {CONVENTION}: hands turned 180 deg about the flange axis; '
                                 'separate stratum from the v1-mount source.').strip())
        dump(staging / 'release_manifest.json', manifest)
        validate(staging)
        staging.rename(output)
        return manifest
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    manifest = prepare(args.source, args.output)
    print(json.dumps({'release': str(args.output), 'release_id': manifest['release_id'],
                      'tasks': [t['id'] for t in manifest['tasks']], 'status': manifest['status']}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
