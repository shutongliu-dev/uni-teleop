"""Check the current living object contracts across optional installations.

This is a read-only identity check, not another native simulation. A shared
object asset does not establish hand-object interaction or robot clearance.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from bench.paths import OUTPUT_ROOT
from scenes.workrooms.catalog import load_task


TASKS = ('remote_operation', 'water_pouring', 'tower_building', 'tower_extraction')
CONFIGURATIONS = ('single_arm', 'dual_same_side', 'dual_opposed')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def object_contract(loaded):
    scene, task = loaded['scene'], loaded['task']
    bundle = deepcopy(scene.get('native_task_bundle'))
    if bundle:
        # The frozen configuration validator separately verifies approved hand
        # namespace changes and exact observer source derivation.
        bundle.pop('observer', None)
    evaluation = deepcopy(task['evaluation'])
    if task['id'] == 'remote_operation':
        for key in ('hand_link_prefixes', 'finger_link_prefixes', 'public_rules_sha256'):
            evaluation.pop(key)
    return {
        'environment': scene['workroom'],
        'physics': scene['physics'],
        'isaacsim': scene['isaacsim'],
        'table': scene['table'],
        'task_layers': [layer for layer in scene.get('task_layers', []) if layer['id'] == 'task'],
        'native_object_bundle': bundle,
        'asset_metadata': task.get('asset_metadata'),
        'evaluation_except_declared_hand_names': evaluation,
    }


def check():
    rows = []
    for task_id in TASKS:
        base = load_task('living_room', task_id)
        reference = object_contract(base)
        options = CONFIGURATIONS if task_id != 'tower_extraction' else ('two_players',)
        for option in options:
            loaded = base if option in ('single_arm', 'two_players') else load_task('living_room', task_id, option)
            root = Path(loaded['definition']['source_release']['path'])
            require(object_contract(loaded) == reference,
                    f'{task_id}/{option}: physical object contract changed')
            require(loaded['release_manifest']['physics_accepted'] is False
                    and loaded['release_manifest']['task_policy_success_claimed'] is False,
                    'Prepared release must not imply task or calibrated physics success')
            rows.append({
                'task_id': task_id, 'configuration_id': option,
                'release': str(root), 'release_manifest_sha256': digest(root / 'release_manifest.json'),
                'full_release_assets_and_frozen_evaluator_valid': True,
                'same_object_geometry_reset_physics_and_numeric_rules': True,
                'object_contract_sha256': hashlib.sha256(json.dumps(reference, sort_keys=True,
                    separators=(',', ':'), allow_nan=False).encode()).hexdigest(),
                'physics_dt_s': loaded['scene']['isaacsim']['physics_dt_s'],
            })
    require(len(rows) == 10, 'Expected three installations for Tasks1-3 and one two-player Task4')
    return {'schema': 'living_interaction_release_identity.v1', 'passed': True,
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'script_sha256': digest(__file__), 'configurations': rows,
        'native_runs_performed': 0,
        'scope': 'Full current release/asset validation and identical physical object contracts; '
                 'approved left/right operator adaptations checked by the frozen-release validator. '
                 'No transfer of robot interaction, trajectory, or whole-room dynamic qualification.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    require(args.output.resolve().is_relative_to(OUTPUT_ROOT),
            'Store generated audit evidence under shared datasets')
    require(not args.output.exists(), 'Refusing to overwrite historical evidence')
    result = check()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'passed': True, 'configurations': len(result['configurations']),
                      'output': str(args.output)}))


if __name__ == '__main__':
    main()
