"""Prepare immutable same-side/opposed derivatives of the eligible tasks.

Run from a complete frozen project. The emitted index is a publication candidate;
it does not change the catalog or promote IK/native/manipulation qualification.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from bench.isaacsim.release import checked, digest, dump, task_inputs
from bench.paths import DATA_ROOT, load_json, OUTPUT_ROOT
from scenes.workrooms.bimanual_layout import (
    layout_id, prepare_stands, robot_inputs, workspace_contract,
)
from scenes.workrooms.catalog import task_definition
from scenes.workrooms.robot_configuration_release import prepare_release
from scenes.workrooms.robot_configurations import ELIGIBLE_TASKS, SCHEMA, VERSION
from scripts.maintenance.freeze_project_sources import verify_if_snapshot


def prepare(output: Path, assets: Path, report: Path, families=None, tasks=None, base_index=None):
    project = Path(__file__).resolve().parents[2]
    if verify_if_snapshot(project, required_modules=('scenes.workrooms.robot_configuration_release',),
                          entrypoint_file=Path(__file__)) is None:
        raise ValueError('Preparation requires complete frozen project source')
    output, assets, report = (p.resolve() for p in (output, assets, report))
    if (not output.is_relative_to(OUTPUT_ROOT)
            or not report.is_relative_to(OUTPUT_ROOT)
            or not assets.is_relative_to(f'{DATA_ROOT}/assets')):
        raise ValueError('Releases/evidence belong in shared datasets; stands in shared assets')
    if output.exists() or assets.exists() or report.exists():
        raise FileExistsError('Use fresh versioned paths; never overwrite a preparation')
    families = tuple(families) if families is not None else tuple(ELIGIBLE_TASKS)
    if (not families or len(set(families)) != len(families)
            or any(f not in ELIGIBLE_TASKS for f in families)):
        raise ValueError('Select distinct eligible workrooms')
    selected = {family: tuple(ELIGIBLE_TASKS[family]) for family in families}
    if tasks is not None:
        tasks = tuple(tasks)
        allowed = {f'{family}/{task_id}' for family in families
                   for task_id in ELIGIBLE_TASKS[family]}
        if not tasks or len(set(tasks)) != len(tasks) or not set(tasks) <= allowed:
            raise ValueError('Select distinct eligible family/task IDs from the selected workrooms')
        selected = {family: tuple(task_id for task_id in task_ids
                                 if f'{family}/{task_id}' in tasks)
                    for family, task_ids in selected.items()}
        selected = {family: task_ids for family, task_ids in selected.items() if task_ids}
    overrides = {}
    if base_index is not None:
        candidates = load_json(Path(base_index))
        if candidates.get('complete') is not True:
            raise ValueError('Base release candidates must be complete')
        for row in candidates['tasks']:
            pair = (row['family'], row['task_id'])
            if pair in overrides:
                raise ValueError('Duplicate base release candidate')
            overrides[pair] = row['source_release']
        if set(overrides) != {(family, task_id) for family, task_ids in selected.items()
                              for task_id in task_ids}:
            raise ValueError('Base release candidates must exactly match the selected task set')
    report.mkdir(parents=True)
    index = {'schema': 'workroom_robot_configuration_candidates.v1',
             'source_snapshot': {'path': str(project/'source_snapshot_manifest.json'),
                                 'sha256': digest(project/'source_snapshot_manifest.json')},
             'families': list(families), 'complete': False, 'tasks': [], 'variants': 0,
             'native_qualified': False, 'manipulation_qualified': False}
    try:
        for family, task_ids in selected.items():
            stands = {}
            for cid in ('dual_same_side', 'dual_opposed'):
                root = assets/family/cid
                prepare_stands(family, task_ids[0], cid, root)
                stands[cid] = root
            for task_id in task_ids:
                definition = task_definition(family, task_id)
                reference = overrides.get((family, task_id), definition['source_release'])
                source = Path(reference['path'])
                checked(source/'release_manifest.json', reference['manifest_sha256'])
                if overrides:
                    _, candidate_scene, _, _ = task_inputs(source, task_id, seed=17)
                    original_scene = load_json(Path(definition['source_release']['path'])/'scene.json')
                    if candidate_scene['workroom'] != original_scene['workroom']:
                        raise ValueError('A base release candidate cannot change the fixed room')
                source_scene = load_json(source/'scene.json')
                original_robot = source_scene['robot']
                expected_cameras = (set(source_scene['cameras']) - {'wrist'}) | {'left_wrist', 'right_wrist'}
                w, x, y, z = original_robot['base_quaternion_wxyz']
                yaw = math.atan2(2*(w*z+x*y), 1-2*(y*y+z*z))
                single_layout = {'robot_layout_id': definition['layout'].get(
                        'robot_layout_id', f'{family}_{task_id}_original_single_arm'),
                    'placements': {'right': {
                        'base_position_m': original_robot['base_position_m'],
                        'base_quaternion_wxyz': original_robot['base_quaternion_wxyz'],
                        'base_yaw_rad': yaw, 'hand_chirality': 'right'}},
                    'reference_surface': 'original fixed task installation',
                    'frame': 'world', 'units': 'metres', 'control_scope': 'one_agent_29_dof'}
                row = {'family': family, 'task_id': task_id,
                       'base_release': reference,
                       'single_arm_robot_requirements': {
                           'hand_id': original_robot['hand_id'],
                           'assembly_version': original_robot['assembly_version'],
                           'articulations': 1},
                       'single_arm_layout': single_layout, 'configurations': {}}
                index['tasks'].append(row)
                for cid in ('dual_same_side', 'dual_opposed'):
                    robot, initial, ready = robot_inputs(family, task_id, cid)
                    workspace = workspace_contract(family, task_id, cid)
                    configuration = {
                        'family': family, 'task_id': task_id, 'configuration_id': cid,
                        'layout_id': layout_id(family, task_id, cid), 'robot': robot,
                        'initial_state': initial, 'ready_design': ready, 'workspace': workspace,
                        'stand_fixture': {'id': 'bimanual_robot_stand',
                            'asset_id': 'bimanual_option_stand', 'asset_root': str(stands[cid]),
                            'position_m': [0., 0., 0.]},
                    }
                    root = output/family/task_id/cid/'v1'
                    prepare_release(source, root, configuration)
                    _, scene, task, reset = task_inputs(root, task_id, seed=17)
                    if (len(reset['joint_positions_rad']) != 58
                            or task['active_sides'] != ['left', 'right']
                            or set(scene['cameras']) != expected_cameras):
                        raise ValueError('Prepared installation lost its dual-hand contract')
                    row['configurations'][cid] = {
                        'source_release': {'path': str(root),
                            'manifest_sha256': digest(root/'release_manifest.json')},
                        'release_schema': SCHEMA, 'release_schema_version': VERSION,
                        'robot_requirements': {'hand_id': robot['hand_id'],
                            'assembly_version': robot['assembly_version'],
                            'articulations': 2, 'active_sides': ['left', 'right']},
                        'layout': {'robot_layout_id': configuration['layout_id'],
                            **{k: workspace[k] for k in ('placements', 'reference_surface',
                                                       'frame', 'units', 'control_scope')}},
                        'qualification': {'native_preflight': False,
                            'agent_bench_interface': False, 'robot_manipulation': False,
                            'physics_accepted': False},
                    }
                    index['variants'] += 1
                    dump(report/'candidate_index.json', index)
                    print(json.dumps({'prepared': index['variants'], 'family': family,
                                      'task_id': task_id, 'configuration_id': cid}), flush=True)
        index['complete'] = True
        dump(report/'candidate_index.json', index)
    except BaseException as exc:
        index['failure'] = {'type': type(exc).__name__, 'message': str(exc)}
        dump(report/'candidate_index.json', index)
        raise
    return index


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--assets', required=True, type=Path)
    parser.add_argument('--report', required=True, type=Path)
    parser.add_argument('--families', nargs='+', choices=tuple(ELIGIBLE_TASKS))
    parser.add_argument('--tasks', nargs='+', help='Optional exact family/task IDs; publish only those derivatives')
    parser.add_argument('--base-index', type=Path, help='Complete exact-task candidate index; prepares derivatives before catalog publication')
    args = parser.parse_args()
    prepare(args.output, args.assets, args.report, args.families, args.tasks, args.base_index)


if __name__ == '__main__':
    main()
