"""Derive a release that removes measured embodiment and scoring distortions.

Scenes and initial states are copied byte-for-byte from the source release.
Changes are limited to: opt-in physics options for arm gravity compensation,
the manufacturer SHARPA actuator model and explicit robot contact materials;
a load-bearing pinch and longer hold for grasp_hold; and public robot/task
facts (table height, target reference, object mass). Thresholds of the other
tasks, assets and initial states are unchanged. Prior releases remain frozen.
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
from .contracts import make_success_contract
from bench.paths import DATA_ROOT, load_json

PROJECT = Path(__file__).resolve().parents[2]
SOURCE = Path(f'{DATA_ROOT}/datasets/fr3_sharpa_bench/isaacsim/releases/simple_manipulation/v7_contact_substeps')
GRAVITY = 9.81
# Physics options; each key is optional and absent keys keep legacy dynamics.
PHYSICS_OPTIONS = {
    'arm_gravity_compensation': 'physx_feedforward_v1',
    'hand_actuator_model': 'sharpa_official_position_mode_v1',
    # Current effective PhysX default for unbound colliders, made explicit.
    'robot_contact_materials': {'pad_friction': .5, 'shell_friction': .5},
}
GRASP_HOLD_S = 1.


def pinch_force(mass_kg, robot_friction, object_friction):
    """Pinch that carries the object by pad friction (two pads, average combine)."""
    mu = (robot_friction + object_friction) / 2.
    return mass_kg * GRAVITY / (2. * mu)


def upgrade_task(original, scene, asset, materials):
    task = deepcopy(original)
    cfg = task['evaluation']
    goal = task['public_goal']
    table = scene['table']
    table_top = float(table['position_m'][2] + table['half_extents_m'][2])
    target = next(obj for obj in scene['objects'] if obj['id'] == cfg['object'])
    mass = float(asset['mass_kg'])
    lo, hi = asset['bounds_min_m'], asset['bounds_max_m']
    above = (hi[2] - lo[2]) / 2.
    goal['workspace'] = {'table_top_z_m': table_top, 'world_up': '+z', 'gravity_m_s2': GRAVITY}
    goal['target_reference'] = ('object asset bounding-box centre (asset frame origin); upright at rest '
                                f'the centre is {above * 1000:g} mm above the table top')
    goal['object_mass_kg'] = round(mass, 4)
    facts = (f'桌面顶面世界高度为{table_top:g}米（世界坐标+z向上）；目标位置指物体包围盒中心，'
             f'直立静止时中心高出桌面{above * 1000:g}毫米；目标质量约{mass * 1000:.0f}克。')
    if cfg['mode'] == 'grasp_hold':
        force = pinch_force(mass, materials['pad_friction'], float(target.get('friction', .65)))
        cfg['minimum_pinch_normal_force_n'] = round(force, 4)
        cfg['hold_s'] = GRASP_HOLD_S
        facts += (f'抓稳指拇指指垫与食指、中指指垫之间的法向夹持力足以凭摩擦承托目标：'
                  f'保持期间平均夹持力不小于{force:.2f}牛，且每个20毫秒窗口不小于其一半；保持时间{GRASP_HOLD_S:g}秒。')
    for key in ('description',):
        goal[key] = goal[key].replace('挤压穿透', '超过2毫米的接触穿透') + facts
    task['steps'][-1]['instruction_zh'] = task['steps'][-1]['instruction_zh'].replace(
        '挤压穿透', '超过2毫米的接触穿透') + facts
    if cfg['mode'] != 'push':
        goal.get('grasp_constraints', {}).pop('push_grasp_scoring', None)
    from bench.evaluation.physics import physics_profile
    goal['success_contract'] = make_success_contract(task, physics_profile()['max_reported_rigid_penetration_m'])
    task['evaluation_scope'] += (' Embodiment-fidelity version: FR3 gravity compensation, manufacturer SHARPA '
        'position-mode actuators and explicit robot contact materials; grasp_hold additionally requires a '
        'load-bearing pad pinch over a 1 s hold. Other thresholds unchanged; prior releases remain frozen.')
    return task


def prepare(source=SOURCE, output=None, options=None):
    source, output = Path(source).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError('Choose a new release path; destination exists: ' + str(output))
    if output.is_relative_to(source) or source.is_relative_to(output):
        raise ValueError('Source and output trees must be separate')
    options = deepcopy(PHYSICS_OPTIONS if options is None else options)
    old = validate(source)
    catalog = load_json(Path(old['asset_catalog']['path']))
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix='.' + output.name + '.preparing-', dir=output.parent))
    try:
        files = {}
        for relative, item in old['files'].items():
            dest = staging / relative
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source / relative, dest)
            files[relative] = {'sha256': digest(dest), 'source': str(source / relative), 'source_sha256': item['sha256']}
        profile = load_json(source / old['physics_profile'])
        profile['isaacsim_options'].update(options)
        profile['embodiment_assumptions'] = {
            'arm_gravity_compensation': 'FR3 internal gravity and payload compensation; not a calibrated payload model',
            'hand_actuator_model': 'SHARPA vendor PhysX position-mode drive gains and armature from the pinned upstream USD',
            'robot_contact_materials': 'explicit binding at the previous effective PhysX default; not a measured friction',
        }
        dump(staging / old['physics_profile'], profile)
        files.pop(old['physics_profile'], None)
        entries = deepcopy(old['tasks'])
        materials = options.get('robot_contact_materials', {'pad_friction': .5})
        for entry in entries:
            task_path = staging / entry['task']
            scene = load_json(staging / entry['scene'])
            original = load_json(task_path)
            target = next(obj for obj in scene['objects'] if obj['id'] == original['evaluation']['object'])
            upgraded = upgrade_task(original, scene, catalog['assets'][target['asset_id']], materials)
            dump(task_path, upgraded)
            entry['task_sha256'] = files[entry['task']]['sha256'] = digest(task_path)
            if digest(staging / entry['scene']) != entry['scene_sha256']:
                raise ValueError('Scene changed during release derivation')
        for name in ('sequence_evaluator.py', 'humanlike.py'):
            scorer = PROJECT / 'scenes/manipulation' / ('evaluator.py' if name == 'sequence_evaluator.py' else name)
            shutil.copy2(scorer, staging / 'evaluator' / name)
            files['evaluator/' + name] = {'sha256': digest(scorer), 'source': str(scorer)}
        shutil.copy2(source / 'release_manifest.json', staging / 'source_release_manifest.json')
        provenance = {
            'protocol': 'embodiment_fidelity_release_v1',
            'source_release': str(source), 'source_release_sha256': digest(source / 'release_manifest.json'),
            'original_release_modified': False,
            'generator_sources': {str(p): digest(p) for p in (Path(__file__), PROJECT / 'scenes/manipulation/contracts.py',
                PROJECT / 'scenes/manipulation/evaluator.py', PROJECT / 'scenes/manipulation/humanlike.py',
                PROJECT / 'bench/simulation/isaacsim_backend/robot_registry.py', PROJECT / 'bench/simulation/isaacsim_backend/env.py')},
            'physics_options_added': options,
            'changes': ['FR3 gravity compensation feedforward (PhysX generalized gravity forces) on arm joints.',
                        'SHARPA manufacturer position-mode Kp/Kd/armature and URDF velocity caps on hand joints.',
                        'Explicit robot pad/shell physics materials at the previous effective values.',
                        'grasp_hold requires a load-bearing pad pinch and a 1 s terminal hold.',
                        'Public table height, target reference, object mass and neutral penetration wording.'],
            'unchanged': ['Scenes and initial states (byte-identical).', 'Assets.',
                          'Push, lift and pick-place thresholds.'],
            'physics_accepted': False, 'task_policy_success_claimed': False,
        }
        dump(staging / 'fidelity_provenance.json', provenance)
        files['fidelity_provenance.json'] = {'sha256': digest(staging / 'fidelity_provenance.json')}
        manifest = {
            'schema_version': old['schema_version'],
            'release_id': 'isaacsim_simple_manipulation_' + output.name,
            'status': 'prepared_unvalidated', 'prepared_at_utc': datetime.now(timezone.utc).isoformat(),
            'source_release': str(source), 'source_release_sha256': digest(source / 'release_manifest.json'),
            'original_release_modified': False, 'tasks': entries, 'files': files,
            'physics_profile': old['physics_profile'], 'physics_profile_sha256': digest(staging / old['physics_profile']),
            'asset_catalog': deepcopy(old['asset_catalog']), 'scene_stratum': old.get('scene_stratum'),
            'physics_accepted': False, 'task_policy_success_claimed': False,
            'purpose': 'Embodiment-fidelity release; do not pool with prior task, physics or evaluator versions.',
        }
        dump(staging / 'release_manifest.json', manifest)
        validate(staging)
        staging.rename(output)
        return manifest
    except BaseException:
        shutil.rmtree(staging)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=SOURCE)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--physics-options', type=Path,
                        help='JSON object replacing the default physics options (for A/B development releases)')
    args = parser.parse_args()
    options = load_json(args.physics_options) if args.physics_options else None
    manifest = prepare(args.source, args.output, options)
    print(json.dumps({'release': str(args.output), 'tasks': [t['id'] for t in manifest['tasks']],
                      'status': manifest['status']}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
