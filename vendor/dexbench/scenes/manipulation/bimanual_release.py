"""Derive the bimanual task release (dual FR3 + SHARPA W01) from the v8 release.

Tasks (new initial states; v8 physics options, contact substeps and robot
materials are reused unchanged):
  05_bimanual_box_lift   - lift a 24 cm carton with both hands (palms on its
                           +/-y faces) at least 8 cm and hold it 1 s.
  06_bimanual_dual_lift  - each hand lifts its own cube (same cube asset and
                           pinch geometry as the single-arm tasks, mirrored)
                           at least 8 cm and holds both for 1 s.
Scoring: scenes.manipulation.evaluator BimanualTaskEvaluator (solver contacts,
per-hand link prefixes, irrecoverable arm contact). Separate stratum; never
pool with single-arm scores.
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
from .contracts import make_bimanual_success_contract
from bench.paths import DATA_ROOT, load_json

PROJECT = Path(__file__).resolve().parents[2]
SOURCE = Path(f'{DATA_ROOT}/datasets/fr3_sharpa_bench/development/distortion_repair_20261014T043445Z/'
              'releases/v8_embodiment_fidelity_r2')
BIMANUAL_CATALOG = Path(f'{DATA_ROOT}/assets/task_parts/bimanual_collection/v1')
ROBOT = 'dual_fr3_sharpa_w01'
TABLE = {'position_m': [.35, 0., .725], 'half_extents_m': [.65, .65, .025], 'friction': .4}
TABLE_TOP = .75
COMMON = {'initial_tolerance_m': .005, 'minimum_lift_m': .08, 'hold_s': 1., 'maximum_speed_m_s': .04,
          'maximum_angular_speed_rad_s': .3, 'maximum_motion_speed_m_s': 1.5, 'minimum_contact_duration_s': .04,
          'max_observation_gap_s': .041, 'hand_sides': {'left': ['left_'], 'right': ['right_']}}
FACTS = '桌面顶面世界高度为0.75米（世界坐标+z向上）；机器人为两台FR3机械臂（基座分别位于机器人基座y=+0.30米左臂、y=-0.30米右臂），各自直接安装SHARPA左手/右手。'


def base_scene(source_scene, objects, description, layout):
    from bench.simulation.isaacsim_backend.robot_registry import scene_with_hand
    scene = scene_with_hand(deepcopy(source_scene), ROBOT)
    scene['robot'].pop('arm_home_rad', None)
    scene['table'] = deepcopy(TABLE)
    scene['objects'] = objects
    scene['goal_markers'] = []
    scene['fixtures'] = []
    scene['description'] = description
    scene['scene_variant'] = {'id': 'bimanual_v1', 'layout_id': layout, 'title_zh': '双臂工作台',
                              'description': 'Dual FR3 + SHARPA workbench; task objects only; separate bimanual stratum',
                              'ranking_stratum': 'bimanual_v1', 'frozen_source_mutated': False}
    scene['workcell_details'] = {'enabled': False}
    return scene


def box_lift(source_scene, catalog):
    box = catalog['assets']['carry_box']
    half_z = box['bounds_max_m'][2]
    position = [.55, 0., TABLE_TOP + half_z + .001]
    scene = base_scene(source_scene, [{'id': 'carry_box', 'asset_id': 'carry_box', 'asset_root': str(BIMANUAL_CATALOG),
                                       'position_m': position, 'quaternion_wxyz': [1, 0, 0, 0], 'friction': .65}],
                       '双手从两侧夹住纸箱并抬起。', 'carry_box_center')
    size = [round(2 * h * 1000) for h in box['bounds_max_m']]
    text = (f'用双手从纸箱左右两侧（世界y方向的两个侧面）同时夹住纸箱，把它从桌面抬高至少8厘米并稳定保持1秒；'
            f'两只手都必须一直接触纸箱，纸箱不能再接触桌面。纸箱沿世界X、Y、Z尺寸约为{size[0]}×{size[1]}×{size[2]}毫米，'
            f'质量约{box["mass_kg"] * 1000:.0f}克，初始中心位于世界坐标x={position[0]:.2f}、y={position[1]:.2f}米。'
            '只允许手部接触纸箱，机械臂本体接触会使本回合无效；避免超过2毫米的接触穿透。' + FACTS)
    task = {'schema_version': 'simple-manipulation-task-v1', 'id': '05_bimanual_box_lift', 'title_zh': '双手抬起纸箱',
            'difficulty': 'intermediate', 'cloth': False, 'target_position_m': position,
            'steps': [{'id': 'bimanual_lift', 'instruction_zh': text}],
            'evaluation': {'mode': 'bimanual_lift', 'object': 'carry_box', 'initial_position_m': position,
                           'maximum_tilt_rad': .35, **COMMON},
            'public_goal': {'object_id': 'carry_box', 'operation': 'bimanual_lift', 'frame_id': 'world', 'description': text,
                            'workspace': {'table_top_z_m': TABLE_TOP, 'world_up': '+z', 'gravity_m_s2': 9.81},
                            'object_mass_kg': box['mass_kg']},
            'recommended_budgets': {'sim_time_s': 120., 'max_actions': 128},
            'evaluation_scope': 'Bimanual lift: both hands in solver contact, object off the table, stable 1 s. '
                                'Dual FR3 + SHARPA stratum; not comparable with single-arm scores.',
            'training_policy_included': False}
    return scene, task


def dual_lift(source_scene, cube_object):
    # Mirror images of the single-arm cube pose relative to each arm base.
    cubes = {'cube_right': [.45, -.20], 'cube_left': [.45, .20]}
    objects = [{**deepcopy(cube_object), 'id': name, 'position_m': [x, y, cube_object['position_m'][2]]}
               for name, (x, y) in cubes.items()]
    scene = base_scene(source_scene, objects, '左右手各抓起一个方块并同时抬起。', 'dual_cubes')
    text = ('左手抓起左侧方块（世界坐标x=0.45、y=0.20米），右手抓起右侧方块（x=0.45、y=-0.20米），'
            '两块都抬高至少8厘米并同时稳定保持1秒；每个方块只能由指定的那只手抓持，不能接触桌面。'
            '方块沿物体本地X、Y、Z轴尺寸约为32×40×60毫米、质量约50克。只允许手部接触方块，机械臂本体接触会使本回合无效；'
            '避免超过2毫米的接触穿透。' + FACTS)
    initial = {name: [x, y, cube_object['position_m'][2]] for name, (x, y) in cubes.items()}
    task = {'schema_version': 'simple-manipulation-task-v1', 'id': '06_bimanual_dual_lift', 'title_zh': '双手各抬起一个方块',
            'difficulty': 'intermediate', 'cloth': False, 'target_position_m': initial['cube_left'],
            'steps': [{'id': 'dual_lift', 'instruction_zh': text}],
            'evaluation': {'mode': 'dual_lift', 'objects': {'cube_left': 'left', 'cube_right': 'right'},
                           'initial_positions_m': initial, 'maximum_tilt_rad': .7, **COMMON},
            'public_goal': {'object_id': 'cube_left', 'operation': 'dual_lift', 'frame_id': 'world', 'description': text,
                            'workspace': {'table_top_z_m': TABLE_TOP, 'world_up': '+z', 'gravity_m_s2': 9.81},
                            'object_mass_kg': .05},
            'recommended_budgets': {'sim_time_s': 120., 'max_actions': 128},
            'evaluation_scope': 'Dual lift: each cube off the table, stable and held only by its assigned hand for 1 s. '
                                'Dual FR3 + SHARPA stratum; not comparable with single-arm scores.',
            'training_policy_included': False}
    return scene, task


def prepare(source=SOURCE, output=None):
    from bench.evaluation.physics import physics_profile
    source, output = Path(source).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError('Choose a new release path; destination exists: ' + str(output))
    old = validate(source)
    lift_entry = next(e for e in old['tasks'] if e['id'] == '02_cube_lift_hold')
    source_scene = load_json(source / lift_entry['scene'])
    cube_object = next(o for o in source_scene['objects'] if o['id'] == 'cube')
    catalog = load_json(BIMANUAL_CATALOG / 'manifest.json')
    penetration = physics_profile()['max_reported_rigid_penetration_m']
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix='.' + output.name + '.preparing-', dir=output.parent))
    try:
        files, entries = {}, []
        physics = load_json(source / old['physics_profile'])
        physics['embodiment_assumptions']['bimanual'] = ('Two independent FR3 articulations (PhysX 64-link limit), each with the '
            'official SHARPA with_flange hand at the identity mount; base offsets +/-0.30 m in y.')
        dump(staging / old['physics_profile'], physics)
        for builder in (lambda: box_lift(source_scene, catalog), lambda: dual_lift(source_scene, cube_object)):
            scene, task = builder()
            physics_options = physics['isaacsim_options']
            task['evaluation'].update(physics_dt_s=physics_options['physics_dt_s'], control_period_s=.02)
            task['public_goal']['bimanual_success_contract'] = make_bimanual_success_contract(task, penetration)
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
        hashes = load_json(source / 'asset_file_hashes.json')
        for path in [BIMANUAL_CATALOG / 'manifest.json', *sorted(BIMANUAL_CATALOG.glob('*.obj'))]:
            hashes[str(path)] = {'sha256': digest(path)}
        dump(staging / 'asset_file_hashes.json', hashes)
        files['asset_file_hashes.json'] = {'sha256': digest(staging / 'asset_file_hashes.json')}
        shutil.copy2(source / 'release_manifest.json', staging / 'source_release_manifest.json')
        provenance = {'protocol': 'bimanual_task_release_v1', 'robot_id': ROBOT, 'source_release': str(source),
                      'source_release_sha256': digest(source / 'release_manifest.json'),
                      'generator_sources': {str(p): digest(p) for p in (Path(__file__), PROJECT / 'scenes/manipulation/evaluator.py',
                          PROJECT / 'scenes/manipulation/contracts.py', PROJECT / 'bench/simulation/isaacsim_backend/robot_registry.py',
                          PROJECT / 'bench/simulation/isaacsim_backend/bimanual_asset.py')},
                      'bimanual_catalog': {'path': str(BIMANUAL_CATALOG / 'manifest.json'),
                                           'sha256': digest(BIMANUAL_CATALOG / 'manifest.json')},
                      'physics_accepted': False, 'task_policy_success_claimed': False}
        dump(staging / 'bimanual_provenance.json', provenance)
        files['bimanual_provenance.json'] = {'sha256': digest(staging / 'bimanual_provenance.json')}
        manifest = {'schema_version': old['schema_version'], 'release_id': 'isaacsim_bimanual_' + output.name,
                    'status': 'prepared_unvalidated', 'prepared_at_utc': datetime.now(timezone.utc).isoformat(),
                    'source_release': str(source), 'source_release_sha256': digest(source / 'release_manifest.json'),
                    'original_release_modified': False, 'tasks': entries, 'files': files,
                    'physics_profile': old['physics_profile'], 'physics_profile_sha256': digest(staging / old['physics_profile']),
                    'asset_catalog': deepcopy(old['asset_catalog']), 'scene_stratum': 'bimanual_v1', 'embodiment': ROBOT,
                    'physics_accepted': False, 'task_policy_success_claimed': False,
                    'purpose': 'Bimanual stratum (dual FR3 + SHARPA); do not pool with single-arm scores.'}
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
