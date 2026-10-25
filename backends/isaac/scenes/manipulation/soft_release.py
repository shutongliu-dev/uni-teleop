"""Derive the soft-body task release (FR3 + SHARPA, one foam sponge) from v8.

Tasks (v8 physics options reused; the sponge replaces the cube at the same
initial position and footprint 32 x 40 x 60 mm):
  07_sponge_press - press the sponge down to <= 75 % of its height for 1 s,
                    then release it (>= 90 %, at rest) without moving it.
  08_sponge_push  - push the sponge to the public goal (as in 01_cube_push).
The sponge is a PhysX volume deformable (FEM) with the mirror-symmetric
24-tet mesh: the shared-diagonal Kuhn split made a resting body creep about
0.9 mm/s (soft_bodies.md). PhysX reports no deformable contact impulses, so
these tasks are scored from the body's state only (SoftBodyTaskEvaluator).

r2 (2026-10-15): rest is judged from the body state over a 0.1 s window
(evaluator v7 ``windowed_body_state_v1``). r1 used PhysX's instantaneous
nodal speed, which never fell below the 0.02 m/s limit even for an untouched
sponge (0.04-0.09 m/s contact jitter), so neither r1 task could succeed.
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
from bench.paths import DATA_ROOT, load_json

PROJECT = Path(__file__).resolve().parents[2]
SOURCE = Path(f'{DATA_ROOT}/datasets/fr3_sharpa_bench/development/distortion_repair_20261014T043445Z/'
              'releases/v8_embodiment_fidelity_r2')
SPONGE = {'id': 'sponge', 'shape': 'box', 'size_m': [.032, .040, .060], 'cells_xyz': [4, 5, 8], 'tet_pattern': 'symmetric_24',
          'mass_kg': .008, 'youngs_modulus_pa': 5e4, 'poissons_ratio': .25, 'friction': .9, 'iterations': 32,
          'contact_offset_m': .002, 'rest_offset_m': 0., 'elasticity_damping': .005, 'visual_color': [.93, .78, .25]}
COMMON = {'initial_tolerance_m': .005, 'nominal_height_m': .060, 'hold_s': 1., 'maximum_rest_speed_m_s': .02,
          'max_observation_gap_s': .041, 'maximum_motion_speed_m_s': 1.5, 'maximum_lift_m': .005,
          'recovered_height_fraction': .9, 'rest_speed_source': 'windowed_body_state_v1', 'rest_window_s': .1}


def sponge_scene(source_scene, table_top, xy):
    scene = deepcopy(source_scene)
    spec = deepcopy(SPONGE)
    spec['position_m'] = [xy[0], xy[1], table_top + spec['size_m'][2] / 2 + .0015]
    scene['objects'] = []
    scene['goal_markers'] = []
    scene['soft_bodies'] = [spec]
    scene['description'] = '桌面上的软海绵块。'
    scene['scene_variant'] = {'id': 'soft_sponge_v1', 'layout_id': 'sponge_workbench', 'title_zh': '海绵工作台',
                              'description': 'One foam sponge (PhysX volume deformable); separate soft-body stratum',
                              'ranking_stratum': 'soft_v1', 'frozen_source_mutated': False}
    return scene, spec


def build_tasks(source_scene, source_push):
    table = source_scene['table']
    table_top = float(table['position_m'][2] + table['half_extents_m'][2])
    xy = source_push['evaluation']['initial_position_m'][:2]
    facts = (f'海绵为软体（有限元体积变形体），未压缩时沿世界X、Y、Z约32×40×60毫米、质量约8克，初始中心位于世界坐标'
             f'x={xy[0]:.2f}、y={xy[1]:.2f}米；桌面顶面世界高度为{table_top:g}米（世界坐标+z向上）。'
             '软体没有接触力读数，评分只看海绵自身的位置与形变。')
    tasks = []
    scene, spec = sponge_scene(source_scene, table_top, xy)
    centroid = list(spec['position_m'])
    press_text = ('把海绵向下压到原高度的75%以下并保持1秒，然后松开，让海绵恢复到原高度的90%以上并静止0.5秒；'
                  '全程海绵水平位移不得超过2厘米，不能把海绵拿离桌面。' + facts)
    tasks.append((scene, {'schema_version': 'simple-manipulation-task-v1', 'id': '07_sponge_press', 'title_zh': '按压海绵',
        'difficulty': 'elementary', 'cloth': False, 'target_position_m': centroid,
        'steps': [{'id': 'press_release', 'instruction_zh': press_text}],
        'evaluation': {'mode': 'soft_press', 'soft_body': 'sponge', 'initial_centroid_m': centroid, 'table_top_z_m': table_top,
                       'press_height_fraction': .75, 'release_hold_s': .5, 'maximum_displacement_m': .02, **COMMON},
        'public_goal': {'object_id': 'sponge', 'operation': 'soft_press', 'frame_id': 'world', 'description': press_text,
                        'workspace': {'table_top_z_m': table_top, 'world_up': '+z', 'gravity_m_s2': 9.81}, 'object_mass_kg': spec['mass_kg']},
        'recommended_budgets': {'sim_time_s': 120., 'max_actions': 128},
        'evaluation_scope': 'Soft press: deformation and pose of the simulated sponge only; no contact evidence for deformables.',
        'training_policy_included': False}))
    goal = source_push['evaluation']['goal_position_m']
    scene, spec = sponge_scene(source_scene, table_top, xy)
    push_text = (f'把海绵沿桌面推到世界坐标 x={goal[0]:.2f}、y={goal[1]:.2f} 米附近（中心偏差不超过2.5厘米），'
                 '移开手，让海绵不被压缩、贴着桌面并静止1秒。' + facts)
    tasks.append((scene, {'schema_version': 'simple-manipulation-task-v1', 'id': '08_sponge_push', 'title_zh': '推海绵',
        'difficulty': 'elementary', 'cloth': False, 'target_position_m': [goal[0], goal[1], centroid[2]],
        'steps': [{'id': 'push', 'instruction_zh': push_text}],
        'evaluation': {'mode': 'soft_push', 'soft_body': 'sponge', 'initial_centroid_m': centroid, 'table_top_z_m': table_top,
                       'goal_position_m': [goal[0], goal[1], centroid[2]], 'goal_xy_radius_m': .025, **COMMON},
        'public_goal': {'object_id': 'sponge', 'operation': 'soft_push', 'frame_id': 'world', 'description': push_text,
                        'target_position_m': [goal[0], goal[1], centroid[2]],
                        'workspace': {'table_top_z_m': table_top, 'world_up': '+z', 'gravity_m_s2': 9.81}, 'object_mass_kg': spec['mass_kg']},
        'recommended_budgets': {'sim_time_s': 120., 'max_actions': 128},
        'evaluation_scope': 'Soft push: centroid, height and rest of the simulated sponge only; no contact evidence for deformables.',
        'training_policy_included': False}))
    return tasks


def prepare(source=SOURCE, output=None):
    source, output = Path(source).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError('Choose a new release path; destination exists: ' + str(output))
    old = validate(source)
    lift = next(e for e in old['tasks'] if e['id'] == '02_cube_lift_hold')
    push = next(e for e in old['tasks'] if e['id'] == '01_cube_push')
    source_scene = load_json(source / lift['scene'])
    source_push = load_json(source / push['task'])
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix='.' + output.name + '.preparing-', dir=output.parent))
    try:
        files, entries = {}, []
        physics = load_json(source / old['physics_profile'])
        physics['embodiment_assumptions']['soft_bodies'] = ('PhysX GPU volume deformable FEM, uncalibrated foam (E 50 kPa, nu 0.25, '
            'dynamic friction 0.9; static friction unsupported); mirror-symmetric 24-tet box mesh.')
        dump(staging / old['physics_profile'], physics)
        for scene, task in build_tasks(source_scene, source_push):
            task['evaluation'].update(physics_dt_s=physics['isaacsim_options']['physics_dt_s'], control_period_s=.02)
            folder = Path('tasks') / task['id']
            dump(staging / folder / 'scene.json', scene)
            dump(staging / folder / 'task.json', task)
            for kind in ('scene', 'task'):
                files[str(folder / f'{kind}.json')] = {'sha256': digest(staging / folder / f'{kind}.json')}
            entries.append({'id': task['id'], 'title_zh': task['title_zh'], 'difficulty': task['difficulty'], 'cloth': False,
                            'scene': str(folder / 'scene.json'), 'scene_sha256': digest(staging / folder / 'scene.json'),
                            'task': str(folder / 'task.json'), 'task_sha256': digest(staging / folder / 'task.json')})
        (staging / 'evaluator').mkdir()
        for name in ('sequence_evaluator.py', 'humanlike.py'):
            scorer = PROJECT / 'scenes/manipulation' / ('evaluator.py' if name == 'sequence_evaluator.py' else name)
            shutil.copy2(scorer, staging / 'evaluator' / name)
            files['evaluator/' + name] = {'sha256': digest(scorer), 'source': str(scorer)}
        shutil.copy2(source / 'asset_file_hashes.json', staging / 'asset_file_hashes.json')
        files['asset_file_hashes.json'] = {'sha256': digest(staging / 'asset_file_hashes.json')}
        shutil.copy2(source / 'release_manifest.json', staging / 'source_release_manifest.json')
        provenance = {'protocol': 'soft_body_task_release_v1', 'source_release': str(source),
                      'source_release_sha256': digest(source / 'release_manifest.json'),
                      'generator_sources': {str(p): digest(p) for p in (Path(__file__), PROJECT / 'scenes/manipulation/evaluator.py',
                          PROJECT / 'bench/simulation/isaacsim_backend/scene.py', PROJECT / 'bench/simulation/isaacsim_backend/geometry.py')},
                      'physics_accepted': False, 'task_policy_success_claimed': False}
        dump(staging / 'soft_provenance.json', provenance)
        files['soft_provenance.json'] = {'sha256': digest(staging / 'soft_provenance.json')}
        manifest = {'schema_version': old['schema_version'], 'release_id': 'isaacsim_soft_' + output.name,
                    'status': 'prepared_unvalidated', 'prepared_at_utc': datetime.now(timezone.utc).isoformat(),
                    'source_release': str(source), 'source_release_sha256': digest(source / 'release_manifest.json'),
                    'original_release_modified': False, 'tasks': entries, 'files': files,
                    'physics_profile': old['physics_profile'], 'physics_profile_sha256': digest(staging / old['physics_profile']),
                    'asset_catalog': deepcopy(old['asset_catalog']), 'scene_stratum': 'soft_v1',
                    'physics_accepted': False, 'task_policy_success_claimed': False,
                    'purpose': 'Soft-body stratum; do not pool with rigid-object scores.'}
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
    args = parser.parse_args()
    manifest = prepare(args.source, args.output)
    print(json.dumps({'release': str(args.output), 'tasks': [t['id'] for t in manifest['tasks']],
                      'status': manifest['status']}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
