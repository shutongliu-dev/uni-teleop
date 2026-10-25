"""Reproducible CPU-only placement, FK and discrete wrist IK diagnostics.

Save raw successes and failures in a new datasets directory. Local IK failures
are not unreachability proofs; successful wrist endpoints are neither finger
grasp targets nor collision-free simultaneous motions. No native engine runs.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import itertools
import json
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np

from bench.geometry import rpy_matrix, transform
from bench.paths import DATA_ROOT
from scenes.workrooms.living_room import dual_robot_layout as layout
from scripts.maintenance.freeze_project_sources import verify_if_snapshot

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASETS_ROOT = Path(f'{DATA_ROOT}/datasets')
DEFAULT_EXPERIMENT = DATASETS_ROOT/'fr3_sharpa_bench/development/living_dual_layout_20261018/cpu01'
WRIST_GRID_M = {'x': [.35, .50, .65, .80, .95],
                'y': [-.20, -.10, 0., .10, .20],
                'z': [.98, 1.08, 1.18, 1.28]}
SOLVER_SETTINGS = {'iterations': 180, 'multi_starts': 0,
                   'position_tolerance': .0001, 'rotation_tolerance': .002,
                   'rotation_weight': .2, 'damping': .025, 'return_trace': True}
NEAR_LIMIT_RAD = .15
PROPOSED_ROOM_BOUNDS_XY_M = ((-2., -1.8), (2.8, 2.7))
REQUIRED_MODULES = (
    'scripts.diagnostics.check_living_dual_layout',
    'scenes.workrooms.living_room.dual_robot_layout',
    'scenes.workrooms.mechanical_assembly.robot_layout',
    'bench.control.factory', 'bench.control.kinematics', 'bench.control.fingertip',
)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def joint_margins(chain, joints):
    return {joint.name: float(min(joints[joint.name]-joint.lower, joint.upper-joints[joint.name]))
            for joint in chain.active}


def wrist_probe(model, point, rpy, initial):
    offset = np.eye(4) if model.wrist_offset is None else model.wrist_offset
    target = np.linalg.inv(model.arm_base) @ transform(point, rpy_matrix(rpy)) @ np.linalg.inv(offset)
    solved = model.arm.inverse(target, model.arm_tip,
        dict(zip(model.arm.names, layout.source_layout.ARM_SEED_RAD)), **SOLVER_SETTINGS)
    joints = {**initial['joint_positions_rad'], **solved['joint_positions_rad']}
    model.checked_joints(joints, enforce_limits=True)
    wrist = model.arm.forward(joints, model.arm_base)[model.arm_tip] @ offset
    margins = joint_margins(model.arm, joints)
    return {'wrist_position_m': point, 'wrist_rpy_rad': rpy,
            'wrist_fk_matrix': wrist.tolist(), 'arm_joint_margins_rad': margins,
            'minimum_arm_joint_margin_rad': min(margins.values()),
            'near_joint_limit': min(margins.values()) < NEAR_LIMIT_RAD,
            'joints_within_limits': True, 'solver': solved}


def check_placement(half_spacing, source_robot, source_initial):
    robots = layout.robot_configurations(half_spacing, source_robot)
    stands = layout.stand_geometry(half_spacing)
    ready, initials, frames, models = {}, {}, {}, {}
    checks = {}
    for side, robot in robots.items():
        changed = [key for key in source_robot if robot[key] != source_robot[key]]
        if set(changed) != {'base_position_m', 'base_quaternion_wxyz'}:
            raise ValueError('Robot fields other than base placement changed: '+repr(changed))
        initials[side], ready[side] = layout.ready_configuration(robot, source_initial)
        if initials[side] != source_initial:
            raise ValueError('Ready joints changed from the mechanical source')
        models[side] = layout.geometry_model(robot)
        frames[side] = layout.link_world_transforms(robot, initials[side])
        model, joints = models[side], initials[side]['joint_positions_rad']
        if any(model.hand_posture_prior.violation(finger, joints) is not None
               for finger in model.finger_joints):
            raise ValueError('Ready hand violates its original posture prior')
        for matrix in frames[side].values():
            np.testing.assert_allclose(matrix[:3, :3].T @ matrix[:3, :3], np.eye(3), atol=1e-10)
            np.testing.assert_allclose(np.linalg.det(matrix[:3, :3]), 1., atol=1e-10)
        boxes = stands[side]['boxes']
        ordered = sorted(boxes, key=lambda box: box['position_m'][2])
        bottoms = [box['position_m'][2]-box['size_m'][2]/2 for box in ordered]
        tops = [box['position_m'][2]+box['size_m'][2]/2 for box in ordered]
        np.testing.assert_allclose(bottoms[0], stands[side]['ground_datum_m'], atol=1e-12)
        np.testing.assert_allclose(bottoms[1:], tops[:-1], atol=1e-12)
        np.testing.assert_allclose(tops[-1], layout.TABLE_TOP_Z_M, atol=1e-12)
        low = np.min([np.array(b['position_m'])-np.array(b['size_m'])/2 for b in boxes], axis=0)
        high = np.max([np.array(b['position_m'])+np.array(b['size_m'])/2 for b in boxes], axis=0)
        room_low, room_high = np.array(PROPOSED_ROOM_BOUNDS_XY_M)
        wall_margin = min(np.min(low[:2]-room_low), np.min(room_high-high[:2]))
        if wall_margin <= 0 or stands[side]['foot_to_table_edge_m'] <= 0:
            raise ValueError('Stand overlaps the tabletop XY footprint or proposed room wall')
        checks[side] = {'only_base_fields_changed': changed, 'ready_joints_preserved': True,
            'ready_joints_within_limits': True, 'ready_hand_posture_prior_valid': True,
            'ready_arm_joint_margins_rad': joint_margins(model.arm, joints),
            'ready_hand_joint_margins_rad': joint_margins(model.hand, joints),
            'fk_frame_count': len(frames[side]), 'fk_rotations_proper': True,
            'stand_bounds_m': [low.tolist(), high.tolist()],
            'stand_ground_datum_m': stands[side]['ground_datum_m'],
            'stand_vertical_support_continuous': True,
            'stand_to_table_xy_gap_m': stands[side]['foot_to_table_edge_m'],
            'stand_to_proposed_room_wall_xy_gap_m': float(wall_margin)}
    symmetry = np.diag([-1., -1., 1., 1.])
    symmetry[0, 3] = 2*layout.TABLE_CENTER_XY_M[0]
    if frames['front'].keys() != frames['rear'].keys():
        raise ValueError('Robot FK link inventories differ')
    error = max(float(np.max(np.abs(symmetry @ first-frames['rear'][name])))
                for name, first in frames['front'].items())
    if error > 1e-10:
        raise ValueError('Ready link transforms do not have central rotational symmetry')
    rows = []
    placements = layout.placements(half_spacing)
    for index, point in enumerate(itertools.product(*(WRIST_GRID_M[key] for key in ('x', 'y', 'z')))):
        for side in layout.SIDES:
            row = wrist_probe(models[side], list(point), placements[side]['wrist_probe_rpy_rad'], initials[side])
            rows.append({'id': f'{side}_{index:03d}', 'side': side, **row})
    index = {(row['side'], tuple(row['wrist_position_m'])): row for row in rows}
    common = []
    opposed = []
    for point in itertools.product(*(WRIST_GRID_M[key] for key in ('x', 'y', 'z'))):
        front, rear = index['front', point], index['rear', point]
        if front['solver']['converged'] and rear['solver']['converged']:
            common.append({'wrist_position_m': list(point), 'probe_ids': [front['id'], rear['id']],
                           'minimum_arm_joint_margin_rad': min(front['minimum_arm_joint_margin_rad'], rear['minimum_arm_joint_margin_rad'])})
        if point[1] < 0:
            opposite = (round(1.3-point[0], 2), -point[1], point[2])
            rear = index['rear', opposite]
            opposed.append({'front_wrist_position_m': list(point), 'rear_wrist_position_m': list(opposite),
                'probe_ids': [front['id'], rear['id']],
                'both_converged': bool(front['solver']['converged'] and rear['solver']['converged']),
                'minimum_arm_joint_margin_rad': min(front['minimum_arm_joint_margin_rad'], rear['minimum_arm_joint_margin_rad'])})
    successful = [row for row in rows if row['solver']['converged']]
    failures = [row for row in rows if not row['solver']['converged']]
    centerline = [row for row in rows if row['wrist_position_m'][0] == .65 and row['wrist_position_m'][1] == 0]
    limits = {side: {joint.name: {'lower_rad': joint.lower, 'upper_rad': joint.upper}
                    for joint in [*model.arm.active, *model.hand.active]}
              for side, model in models.items()}
    return {'base_half_spacing_m': half_spacing, 'placements': placements,
        'robots': robots, 'ready': ready, 'initials': initials, 'stands': stands,
        'joint_limits': limits, 'checks': checks,
        'ready_symmetry': {'type': '180_degree_rotation_about_table_center',
            'maximum_matrix_element_error': error, 'world_transform': symmetry.tolist(),
            'same_right_hand_both_sides': True, 'reflection_or_geometry_scaling_used': False},
        'summary': {'wrist_targets_per_robot': len(rows)//2,
            'converged_per_robot': {side: sum(row['solver']['converged'] for row in rows if row['side'] == side) for side in layout.SIDES},
            'same_point_independently_reachable_count': len(common),
            'distinct_opposed_pairs_both_reachable_count': sum(row['both_converged'] for row in opposed),
            'distinct_opposed_pair_count': len(opposed),
            'centerline_converged_probe_ids': [row['id'] for row in centerline if row['solver']['converged']],
            'failed_probe_ids': [row['id'] for row in failures],
            'near_limit_probe_ids': [row['id'] for row in rows if row['near_joint_limit']],
            'lowest_success_margins': [{'id': row['id'], 'margin_rad': row['minimum_arm_joint_margin_rad'], 'wrist_position_m': row['wrist_position_m']}
                for row in sorted(successful, key=lambda row: row['minimum_arm_joint_margin_rad'])[:10]]},
        'same_point_independent_solutions': common, 'distinct_opposed_pairs': opposed,
        'raw_probes': rows}


def run(half_spacings):
    source = layout.source_layout.robot_configuration()
    initial, source_ready = layout.source_layout.ready_configuration(source)
    model = layout.source_layout.geometry_model(source)
    snapshot_before = verify_if_snapshot(PROJECT_ROOT, required_modules=REQUIRED_MODULES,
                                         entrypoint_file=Path(__file__))
    manifest_path = PROJECT_ROOT/'source_snapshot_manifest.json'
    manifest_sha256 = digest(manifest_path) if snapshot_before is not None else None
    # A relocated snapshot has no .git and the freezer does not record a Git
    # commit. Its manifest hash is the identity; do not query the editable source.
    git_head = None
    if snapshot_before is None and (PROJECT_ROOT/'.git').exists():
        git_head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=PROJECT_ROOT, text=True).strip()
    candidates = []
    for spacing in half_spacings:
        candidate = check_placement(spacing, source, initial)
        candidates.append(candidate)
        print(json.dumps({'base_half_spacing_m': spacing, **candidate['summary']}, allow_nan=False), flush=True)
    imported_sources = {}
    for module in list(sys.modules.values()):
        path = getattr(module, '__file__', None)
        if path and Path(path).suffix == '.py' and Path(path).resolve().is_relative_to(PROJECT_ROOT):
            imported_sources[str(Path(path).resolve().relative_to(PROJECT_ROOT))] = digest(path)
    asset_reference = layout.source_layout.robot_asset_reference(source)
    snapshot_after = verify_if_snapshot(PROJECT_ROOT, required_modules=REQUIRED_MODULES,
                                        entrypoint_file=Path(__file__))
    if snapshot_before is not None and digest(manifest_path) != manifest_sha256:
        raise RuntimeError('Complete source snapshot manifest changed during diagnostic')
    return {'schema': 'living_dual_layout_cpu_v2', 'created_utc': datetime.now(timezone.utc).isoformat(),
        'layout_id': layout.LAYOUT_ID, 'default_half_spacing_m': layout.BASE_HALF_SPACING_M,
        'assumptions': {'table_center_xy_m': list(layout.TABLE_CENTER_XY_M),
            'table_bounds_xy_m': layout.TABLE_BOUNDS_XY_M, 'table_top_z_m': layout.TABLE_TOP_Z_M,
            'proposed_room_bounds_xy_m': PROPOSED_ROOM_BOUNDS_XY_M,
            'room_bounds_provenance': 'Current room dimensions supplied by scene owner; room geometry is not loaded by this diagnostic',
            'room_geometry_loaded': False,
            'wooden_object_dimensions': None, 'hand_posture': 'Unchanged source waiting soft curl',
            'probe_frame': 'public wrist; not fingertips or object grasp center',
            'probe_orientation': 'source down-facing wrist rotated with each base',
            'no_mirrored_or_scaled_robot_geometry': True},
        'sampling': {'grid_m': WRIST_GRID_M, 'solver_settings': SOLVER_SETTINGS,
            'warm_start_arm_seed_rad': list(layout.source_layout.ARM_SEED_RAD),
            'restarts': 'none; exactly the original mechanical diagnostic warm seed and solver tolerances',
            'near_joint_limit_threshold_rad': NEAR_LIMIT_RAD},
        'source': {'robot': source, 'ready': source_ready, 'initial': initial,
            'asset_reference': asset_reference,
            'urdf_sha256': {str(path): digest(path) for path in (model.arm_path, model.hand_path)},
            'project_root': str(PROJECT_ROOT),
            'git_head': git_head,
            'git_head_scope': 'Editable checkout only; complete snapshot manifests do not record Git commits',
            'complete_snapshot_manifest': str(manifest_path) if snapshot_before is not None else None,
            'complete_snapshot_manifest_sha256': manifest_sha256,
            'snapshot_verification_before': snapshot_before,
            'snapshot_verification_after': snapshot_after,
            'evidence_scope': 'complete_frozen_source_cpu_diagnostic' if snapshot_after is not None else 'exploratory_editable_source_cpu_diagnostic',
            'imported_project_sources_sha256': dict(sorted(imported_sources.items()))},
        'candidates': candidates, 'geometry_checks_passed': True,
        'native_runtime_started': False, 'model_calls': 0, 'native_qualified': False,
        'manipulation_qualified': False,
        'limitations': [
            'Endpoint IK only; finite-seed failures do not prove global unreachability.',
            'Independent same-point solutions are an overlap diagnostic, not a command to occupy one point simultaneously.',
            'Distinct paired wrist solutions do not establish collision-free robot, hand or object configurations.',
            'No finger grasp transform, wooden object dimensions, approach path or coordinated motion is specified.',
            'Stand footprint gaps exclude tabletop legs, sofa, robot meshes, contact offsets and native cooked colliders; the USD preview must check actual room shapes.',
            'All existing living-room tasks remain planned; this is no task implementation or qualification.',
        ]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_EXPERIMENT)
    parser.add_argument('--half-spacings', type=float, nargs='+', default=[.88, .80, .76, .74])
    args = parser.parse_args()
    output = args.output_dir.expanduser().resolve()
    if not output.is_relative_to(DATASETS_ROOT.resolve()):
        parser.error('--output-dir must be under shared datasets, outside project source')
    if len(set(args.half_spacings)) != len(args.half_spacings):
        parser.error('--half-spacings must be unique')
    for spacing in args.half_spacings:
        layout.placements(spacing)
    output.mkdir(parents=True, exist_ok=False)
    report = run(args.half_spacings)
    if report['source']['snapshot_verification_after'] is not None:
        report['source']['snapshot_directory'] = str(PROJECT_ROOT)
    else:
        # Retain an explicitly partial import copy for exploratory editable runs;
        # it must never be presented as a complete frozen benchmark source tree.
        for relative, expected in report['source']['imported_project_sources_sha256'].items():
            source = PROJECT_ROOT/relative
            destination = output/'imported_source_copy'/relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            if digest(destination) != expected:
                raise RuntimeError('Source changed during CPU diagnostic: '+relative)
        report['source']['snapshot_directory'] = None
        report['source']['imported_source_copy_directory'] = str(output/'imported_source_copy')
    with (output/'report.json').open('x') as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
        handle.write('\n')
    with (output/'raw_probes.jsonl').open('x') as handle:
        for candidate in report['candidates']:
            for row in candidate['raw_probes']:
                handle.write(json.dumps({'base_half_spacing_m': candidate['base_half_spacing_m'], **row}, allow_nan=False)+'\n')
    print(json.dumps({'report': str(output/'report.json'), 'report_sha256': digest(output/'report.json'),
                      'raw_probes_sha256': digest(output/'raw_probes.jsonl'), 'geometry_checks_passed': True}), flush=True)


if __name__ == '__main__':
    main()
