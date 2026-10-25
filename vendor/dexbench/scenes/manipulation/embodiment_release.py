"""Derive per-hand embodiment releases (FR3 + Allegro / LEAP / Shadow) from v8.

Each output release targets ONE registered hand: every scene goes through
scene_with_hand (versioned mount, self-collision, structural filters); initial
object states, assets and all task thresholds are unchanged. Hand-specific
scoring is replaced explicitly:

* the SHARPA-only humanlike grasp contract is removed (it is defined on SHARPA
  joints and links); lift/carry/grasp are scored by solver contacts, and
  grasp_hold by the same load-bearing pinch rule on the hand's own distal links;
* the SHARPA manufacturer actuator model is not applied (registry gains are
  used and recorded as uncalibrated); gravity compensation and explicit robot
  contact materials are kept.

These releases are separate embodiment strata: never pool them with SHARPA
scores. Prior releases remain frozen.
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
SOURCE = Path(f'{DATA_ROOT}/datasets/fr3_sharpa_bench/development/distortion_repair_20261014T043445Z/'
              'releases/v8_embodiment_fidelity_r2')
SHARPA_ONLY_PHYSICS = ('hand_actuator_model',)
HUMANLIKE_KEYS = ('humanlike_grasp', 'humanlike_grasp_role', 'humanlike_lift_start_m', 'push_grasp_violation_hold_s')
# Sentences of the SHARPA instructions that state the humanlike contract.
HUMANLIKE_SENTENCES = ('抓持姿态必须接近自然人手', '推移允许单指接触', '在抬高达到4厘米前先维持合规抓持')


def neutral_instruction(text, spec):
    sentences = [s for s in text.split('。') if s and not s.startswith(HUMANLIKE_SENTENCES)]
    sentences = [s.replace('以拇指对食指或中指自然抓稳方块', '以拇指对食指或中指抓稳方块')
                  .replace('拇指指垫与食指、中指指垫', '拇指末节与食指、中指末节') for s in sentences]
    fingers = '、'.join({'thumb': '拇指', 'index': '食指', 'middle': '中指', 'ring': '无名指', 'little': '小指'}[f]
                       for f in ('thumb', 'index', 'middle', 'ring', 'little') if f in spec['fingertip_links'])
    adapter = '经10毫米铝制转接盘' if spec.get('adapter') else '直接'
    embodiment = (f'机器人为FR3机械臂{adapter}安装{spec["label"]}（手指：{fingers}）；本任务不评估人手相似度，'
                  '只按手部连杆的求解器接触与目标状态判定，避免超过2毫米的接触穿透')
    return '。'.join(sentences[:1] + [embodiment] + sentences[1:]) + '。'


def convert_task(original, spec):
    task = deepcopy(original)
    cfg = task['evaluation']
    for key in HUMANLIKE_KEYS:
        cfg.pop(key, None)
    cfg['pinch_links'] = {finger: list(links) for finger, links in spec['pinch_links'].items()}
    goal = task['public_goal']
    goal.pop('grasp_constraints', None)
    text = neutral_instruction(task['steps'][-1]['instruction_zh'], spec)
    task['steps'][-1]['instruction_zh'] = goal['description'] = text
    goal['embodiment'] = {'hand_id': spec['hand_id'], 'label': spec['label'], 'fingers': list(spec['fingertip_links']),
                          'mount_convention': spec['mount_convention'], 'adapter': bool(spec.get('adapter'))}
    from bench.evaluation.physics import physics_profile
    goal['success_contract'] = make_success_contract(task, physics_profile()['max_reported_rigid_penetration_m'])
    task['evaluation_scope'] += (f' Embodiment stratum {spec["hand_id"]}: humanlike SHARPA contract removed; '
        'contacts and the pinch rule use this hand\'s distal links. Not comparable with SHARPA scores.')
    return task


def prepare(hand_id, source=SOURCE, output=None):
    from bench.simulation.isaacsim_backend.robot_registry import DEFAULT_HAND, get_hand_spec, scene_with_hand
    if hand_id == DEFAULT_HAND:
        raise ValueError('The default SHARPA hand uses the v8 release itself')
    spec = get_hand_spec(hand_id)
    source, output = Path(source).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError('Choose a new release path; destination exists: ' + str(output))
    old = validate(source)
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
        removed = {key: profile['isaacsim_options'].pop(key) for key in SHARPA_ONLY_PHYSICS if key in profile['isaacsim_options']}
        assumptions = profile.setdefault('embodiment_assumptions', {})
        for key in removed:
            assumptions.pop(key, None)
        assumptions['hand_drives'] = (f'{spec["label"]}: uniform registry drive gains kp={spec["hand_kp"]}, kd={spec["hand_kd"]} '
                                      '(uncalibrated simulation assumption); ' + spec['control_model'])
        assumptions['hand_mount'] = (f'{spec["mount_convention"]}; '
                                     + ('10 mm aluminium adapter disc with mass and collision' if spec.get('adapter') else 'direct flange mount'))
        # Audited structural exclusions are a release opt-in. task_inputs() replaces
        # scene['isaacsim'] with these profile options, so they belong here (v9 r3
        # wrote them into the scenes, where they had no effect).
        profile['isaacsim_options'].update(spec.get('release_collision_filters', {}))
        dump(staging / old['physics_profile'], profile)
        files.pop(old['physics_profile'], None)
        entries = deepcopy(old['tasks'])
        for entry in entries:
            scene_path, task_path = staging / entry['scene'], staging / entry['task']
            dump(scene_path, scene_with_hand(load_json(scene_path), hand_id))
            dump(task_path, convert_task(load_json(task_path), spec))
            entry['scene_sha256'] = files[entry['scene']]['sha256'] = digest(scene_path)
            entry['task_sha256'] = files[entry['task']]['sha256'] = digest(task_path)
        for name in ('sequence_evaluator.py', 'humanlike.py'):
            scorer = PROJECT / 'scenes/manipulation' / ('evaluator.py' if name == 'sequence_evaluator.py' else name)
            shutil.copy2(scorer, staging / 'evaluator' / name)
            files['evaluator/' + name] = {'sha256': digest(scorer), 'source': str(scorer)}
        shutil.copy2(source / 'release_manifest.json', staging / 'source_release_manifest.json')
        provenance = {
            'protocol': 'embodiment_stratum_release_v1', 'hand_id': hand_id,
            'source_release': str(source), 'source_release_sha256': digest(source / 'release_manifest.json'),
            'original_release_modified': False,
            'generator_sources': {str(p): digest(p) for p in (Path(__file__), PROJECT / 'scenes/manipulation/contracts.py',
                PROJECT / 'scenes/manipulation/evaluator.py', PROJECT / 'bench/simulation/isaacsim_backend/robot_registry.py',
                PROJECT / 'bench/simulation/isaacsim_backend/multihand_asset.py')},
            'physics_options_removed': removed,
            'changes': ['Scenes: scene_with_hand (versioned mount, robot self-collision, structural filters).',
                        f'Physics profile isaacsim_options += release collision filters {spec.get("release_collision_filters", {})}.',
                        'SHARPA humanlike grasp contract removed; contact and pinch channels use this hand\'s distal links.',
                        'SHARPA manufacturer actuator model not applied; registry gains used (uncalibrated).'],
            'unchanged': ['Initial object states and assets.', 'All task thresholds, including the pinch force.',
                          'Gravity compensation and explicit robot contact materials.'],
            'physics_accepted': False, 'task_policy_success_claimed': False,
        }
        dump(staging / 'embodiment_provenance.json', provenance)
        files['embodiment_provenance.json'] = {'sha256': digest(staging / 'embodiment_provenance.json')}
        manifest = {
            'schema_version': old['schema_version'],
            'release_id': 'isaacsim_simple_manipulation_' + output.name,
            'status': 'prepared_unvalidated', 'prepared_at_utc': datetime.now(timezone.utc).isoformat(),
            'source_release': str(source), 'source_release_sha256': digest(source / 'release_manifest.json'),
            'original_release_modified': False, 'tasks': entries, 'files': files,
            'physics_profile': old['physics_profile'], 'physics_profile_sha256': digest(staging / old['physics_profile']),
            'asset_catalog': deepcopy(old['asset_catalog']), 'scene_stratum': old.get('scene_stratum'),
            'embodiment': hand_id, 'physics_accepted': False, 'task_policy_success_claimed': False,
            'purpose': f'Embodiment stratum {hand_id}; do not pool with SHARPA or other embodiment scores.',
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
    parser.add_argument('--hand', required=True)
    parser.add_argument('--source', type=Path, default=SOURCE)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    manifest = prepare(args.hand, args.source, args.output)
    print(json.dumps({'release': str(args.output), 'embodiment': manifest['embodiment'],
                      'tasks': [t['id'] for t in manifest['tasks']], 'status': manifest['status']}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
