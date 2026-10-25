"""Freeze repaired scoring/public contracts on an unchanged curated scene."""
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
SOURCE = Path(f'{DATA_ROOT}/datasets/fr3_sharpa_bench/isaacsim/releases/simple_manipulation/v6_public_control')
# Historical v6 -> v7 builder; its output stays the frozen v7 path.
DEFAULT = Path(f'{DATA_ROOT}/datasets/fr3_sharpa_bench/isaacsim/releases/simple_manipulation/v7_contact_substeps')


def upgrade_task(original, penetration_limit, *, object_dimensions_m=None):
    task = deepcopy(original)
    cfg = task['evaluation']
    if task.get('schema_version') != 'simple-manipulation-task-v1' or cfg['mode'] not in (
            'push', 'lift_hold', 'pick_place', 'grasp_hold'):
        raise ValueError('Expected a simple manipulation task')
    humanlike = cfg['humanlike_grasp']
    cfg['physics_dt_s'] = humanlike['physics_dt_s']
    cfg['control_period_s'] = humanlike['control_period_s']
    # Tighten the public grasp proxy to the existing final physics bound. The
    # simulator, material values, timing and other task thresholds do not change.
    humanlike['maximum_robot_penetration_m'] = min(humanlike['maximum_robot_penetration_m'], penetration_limit)
    task['public_goal']['grasp_constraints']['criteria']['maximum_robot_penetration_m'] = humanlike['maximum_robot_penetration_m']
    had_public_contract = 'success_contract' in task['public_goal']
    task['public_goal']['success_contract'] = make_success_contract(task, penetration_limit)
    extra = ('操作过程中仅允许手部接触目标，机械臂本体接触目标会使本回合无效；'
             '需要释放时，整台机器人都必须与目标脱离。完成还需满足公开success_contract中的稳定与持续保持条件。')
    if not had_public_contract:
        task['steps'][-1]['instruction_zh'] += extra
        task['public_goal']['description'] += extra
    if cfg['mode'] == 'grasp_hold' and not had_public_contract:
        notice = (f"全回合方块相对初始位置位移不得超过{cfg['goal_xy_radius_m'] * 100:g}厘米、"
                  f"抬升不得超过{cfg['goal_z_tolerance_m'] * 1000:g}毫米；"
                  '移回原位不能撤销之前的超限。')
        task['steps'][-1]['instruction_zh'] += notice
        task['public_goal']['description'] += notice
    notice = ('倾角是物体本地正Z轴与世界正Z轴的夹角，绕世界竖直轴的转向本身不计入倾角。')
    if object_dimensions_m is not None:
        dimensions = [float(x) for x in object_dimensions_m]
        if len(dimensions) != 3 or any(not 0 < x < 1 for x in dimensions):
            raise ValueError('Expected finite, positive metre-scale object bounds')
        notice = ('目标外形为长方体，沿物体本地X、Y、Z轴尺寸约为'
                  + '×'.join(f'{x * 1000:g}' for x in dimensions) + '毫米；资产和任务ID保留cube命名。' + notice)
    task['steps'][-1]['instruction_zh'] += notice
    task['public_goal']['description'] += notice
    task['evaluation_scope'] += (' Contact-substep version: count each positive hand-contact physics '
        'substep once regardless of the number of contacting hand links. Require complete consecutive '
        'native contact sampling; a missing observation invalidates the episode. Prior releases remain frozen.')
    return task


def prepare(source=SOURCE, output=DEFAULT):
    source, output = Path(source).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError('Choose a new release path; destination exists: ' + str(output))
    if output.is_relative_to(source) or source.is_relative_to(output):
        raise ValueError('Source and output trees must be separate')
    old = validate(source)
    physics = load_json(source / old['physics_profile'])['isaacsim_options']
    catalog = load_json(Path(old['asset_catalog']['path']))
    from bench.evaluation.physics import physics_profile
    profile = physics_profile()
    limit = profile['max_reported_rigid_penetration_m']
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix='.' + output.name + '.preparing-', dir=output.parent))
    try:
        files = {}
        for relative, item in old['files'].items():
            dest = staging / relative
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source / relative, dest)
            files[relative] = {'sha256': digest(dest), 'source': str(source / relative),
                               'source_sha256': item['sha256']}
        entries = deepcopy(old['tasks'])
        for entry in entries:
            task_path = staging / entry['task']
            original = load_json(task_path)
            scene = load_json(staging / entry['scene'])
            target = next(obj for obj in scene['objects'] if obj['id'] == original['evaluation']['object'])
            if target['asset_id'] != 'collab_cube_small':
                raise ValueError('This simple-task release requires the reviewed rectangular target asset')
            asset = catalog['assets'][target['asset_id']]
            dimensions = [high - low for high, low in zip(asset['bounds_max_m'], asset['bounds_min_m'], strict=True)]
            upgraded = upgrade_task(original, limit, object_dimensions_m=dimensions)
            if (upgraded['evaluation']['physics_dt_s'] != physics['physics_dt_s']
                    or upgraded['evaluation']['control_period_s'] != .02):
                raise ValueError('Frozen task sampling clocks disagree with native physics profile')
            dump(task_path, upgraded)
            entry['task_sha256'] = files[entry['task']]['sha256'] = digest(task_path)
            if digest(staging / entry['scene']) != entry['scene_sha256']:
                raise ValueError('Scene changed during scoring repair')
        scorer = PROJECT / 'scenes/manipulation/evaluator.py'
        shutil.copy2(scorer, staging / 'evaluator/sequence_evaluator.py')
        files['evaluator/sequence_evaluator.py'] = {'sha256': digest(scorer), 'source': str(scorer)}
        shutil.copy2(source / old['physics_profile'], staging / old['physics_profile'])
        shutil.copy2(source / 'release_manifest.json', staging / 'source_release_manifest.json')
        provenance = {
            'protocol': 'contact_substep_contract_repair_v2',
            'source_release': str(source), 'source_release_sha256': digest(source / 'release_manifest.json'),
            'original_release_modified': False,
            'source_curation_provenance_describes_source_release_only': True,
            'generator_sources': {str(p): digest(p) for p in (Path(__file__),
                PROJECT / 'scenes/manipulation/contracts.py', PROJECT / 'bench/evaluation/physics.py')},
            'changes': ['Count unique hand-contact physics substeps instead of entire control windows.',
                        'Require complete consecutive native sampling evidence and reject duplicate pair rows.',
                        'Publish versioned static contact-duration and sampling semantics.',
                        'Describe the existing rectangular target dimensions and the unchanged local-Z tilt definition.'],
            'unchanged': ['Scenes and initial state bytes.', 'Solver/material profile and shared assets.',
                          'Other physical task thresholds; archived outcomes.'],
            'physics_acceptance_profile': profile,
            'physics_accepted': False, 'task_policy_success_claimed': False,
        }
        dump(staging / 'contract_repair_provenance.json', provenance)
        files['contract_repair_provenance.json'] = {'sha256': digest(staging / 'contract_repair_provenance.json')}
        manifest = {
            'schema_version': 'isaacsim-public-control-task-release-v1',
            'release_id': 'isaacsim_simple_manipulation_' + output.name,
            'status': 'prepared_unvalidated', 'prepared_at_utc': datetime.now(timezone.utc).isoformat(),
            'source_release': str(source), 'source_release_sha256': digest(source / 'release_manifest.json'),
            'original_release_modified': False, 'tasks': entries, 'files': files,
            'physics_profile': old['physics_profile'], 'physics_profile_sha256': old['physics_profile_sha256'],
            'asset_catalog': deepcopy(old['asset_catalog']), 'scene_stratum': old.get('scene_stratum'),
            'physics_accepted': False, 'task_policy_success_claimed': False,
            'purpose': 'Repaired contact-substep scoring; do not pool with prior task/evaluator versions.',
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
    parser.add_argument('--output', type=Path, default=DEFAULT)
    args = parser.parse_args()
    manifest = prepare(args.source, args.output)
    print(json.dumps({'release': str(args.output), 'tasks': [t['id'] for t in manifest['tasks']],
                      'status': manifest['status']}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
