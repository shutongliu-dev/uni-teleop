"""Offline actual-USD bounds, mass inputs and discrete original-robot IK audit.

Private development diagnostics only. Enclosing shape boxes are conservative;
neither their overlap nor an IK endpoint establishes a physical grasp or path.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from bench.isaacsim.release import digest, dump, task_inputs
from bench.paths import load_json, OUTPUT_ROOT
from scripts.diagnostics.check_interaction_workspace import (
    ArmSelfCollision, RecordedCollisionBounds, content_digest, free_surface_approach as box_surface_approach,
    installation, separating_axis_gap, solve_target,
)

TASKS = ('remote_operation', 'water_pouring', 'storage_box')
CONFIGURATIONS = ('single_arm', 'dual_same_side', 'dual_opposed')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def free_surface_approach(checker, reference, point, *, owner_paths=()):
    """Legacy enclosing-box approach, unless the case bound a declared approach rule.

    Raw samples and every supplement query the same checker, so they all use
    the rule an adapter bound to it; legacy validators bind none.
    """
    rule = getattr(checker, 'approach_context', None)
    if rule is None:
        return box_surface_approach(checker, reference, point, owner_paths=owner_paths)
    return box_surface_approach(checker, reference, point, owner_paths=owner_paths,
        static_owner_paths=rule['static_owner_paths'], exact_clearance=rule['exact_clearance'])


def refine_goal_approach(checker, target, reference, surface):
    """A bound rule may only remove blockers of a goal; the approach point is never changed."""
    rule = getattr(checker, 'approach_context', None)
    refine = None if rule is None else rule.get('goal_refinement')
    return surface if refine is None else refine(target, reference, surface)


def read_request(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, 'Duplicate request key')
            result[key] = value
        return result
    value = load_json(Path(path), object_pairs_hook=unique, parse_constant=lambda _: (_ for _ in ()).throw(ValueError('Nonfinite request')))
    require(set(value) <= {'schema', 'cases', 'duration_s'} and
            value.get('schema') == 'living_randomized_validation_request.v1', 'Unexpected request schema')
    cases = value.get('cases')
    require(isinstance(cases, list) and 1 <= len(cases) <= 90, 'Require 1–90 explicit cases')
    seen = set()
    for case in cases:
        require(set(case) == {'task_id', 'configuration_id', 'seed', 'release'}, 'Unexpected case fields')
        require(case['task_id'] in TASKS and case['configuration_id'] in CONFIGURATIONS,
                'Unsupported ordinary task/configuration')
        require(type(case['seed']) is int and 0 <= case['seed'] < 10, 'Expected one of ten declared seeds')
        require(Path(case['release']).is_absolute(), 'Use explicit absolute release path')
        key = (case['task_id'], case['configuration_id'], case['seed'])
        require(key not in seen, 'Duplicate case')
        seen.add(key)
    duration = value.get('duration_s', .8)
    require(type(duration) in (int, float) and math.isfinite(duration) and .6 <= duration <= 3.
            and math.isclose(round(duration / .02) * .02, duration, abs_tol=1e-9),
            'Passive duration must be 0.6–3 s in 20 ms multiples')
    return {**value, 'duration_s': float(duration)}


def case_name(case):
    return f"{case['task_id']}/{case['configuration_id']}/seed_{case['seed']:02d}"


def body_paths(scene, task):
    paths = (scene.get('native_task_bundle', {}).get('rigid_bodies')
             or task.get('evaluation', {}).get('body_paths'))
    require(isinstance(paths, dict) and paths and len(set(paths.values())) == len(paths),
            'Require explicit unique task rigid-body paths')
    return paths


def case_targets(task):
    # Seed-zero legacy metadata may retain a different historical target list.
    # Only the explicitly frozen reset-case feature list belongs to this audit.
    targets = task['reset_case']['interaction_targets']
    require(isinstance(targets, list) and targets and all(isinstance(row, dict) for row in targets),
            'Require explicit frozen case interaction features')
    require(len({row['label'] for row in targets}) == len(targets), 'Duplicate case feature label')
    for row in targets:
        require(row.get('type') in ('actual_surface', 'goal', 'pouring_edge')
                and row.get('reference_state') in ('initial', 'opened'), 'Feature must declare surface/goal and mechanism state')
        require(isinstance(row.get('operation_group'), str) and row['operation_group']
                and type(row.get('required')) is bool, 'Feature must declare operation group and requirement')
    return {row['label']: row for row in targets}


def operation_coverage(samples):
    groups = {}
    for sample in samples:
        target = sample['feature']
        name = target['operation_group']
        identity = (target['required'], sample['requires_prior_mechanism_change'], target['reference_state'])
        group = groups.setdefault(name, {'identity': identity, 'features': [], 'covered_features': []})
        require(group['identity'] == identity, 'Operation group mixes requirements or mechanism stages')
        group['features'].append(sample['label'])
        covered = (any(hand['converged'] and hand['clearance_passed'] for hand in sample['geometric_endpoint_ik'].values())
                   if sample['requires_prior_mechanism_change'] else sample['initial_endpoint_covered'])
        if covered:
            group['covered_features'].append(sample['label'])
    return {name: {'required': row['identity'][0], 'conditional': row['identity'][1],
                  'reference_state': row['identity'][2], 'feature_labels': row['features'],
                  'covered_feature_labels': row['covered_features'], 'endpoint_covered': bool(row['covered_features']),
                  'scope': 'At least one authored alternative; conditional groups cover robot-only IK, not opened-scene clearance.'}
            for name, row in groups.items()}


def verify_authored_poses(stage, paths, recorded):
    from pxr import Gf, UsdGeom
    require(isinstance(recorded, dict) and set(recorded) == set(paths), 'Case pose inventory differs from actual body inventory')
    cache = UsdGeom.XformCache()
    result = {}
    for name, path in paths.items():
        row = recorded[name]
        require(row['prim_path'] == path, 'Case body path differs')
        transform = Gf.Transform(cache.GetLocalToWorldTransform(stage.GetPrimAtPath(path)))
        q = transform.GetRotation().GetQuat()
        position = np.asarray(transform.GetTranslation())
        quaternion = np.asarray([q.GetReal(), *q.GetImaginary()])
        expected = np.asarray(row['quaternion_wxyz'])
        position_error = float(np.linalg.norm(position - row['position_m']))
        orientation_error = float(min(np.linalg.norm(quaternion - expected), np.linalg.norm(quaternion + expected)))
        require(position_error < 2e-7 and orientation_error < 2e-6, 'Case metadata differs from final USD body pose')
        result[name] = {'position_error_m': position_error, 'quaternion_component_error_norm': orientation_error}
    return result


def transformed_feature(target, stage):
    """Use the actual composed body frame, including authored variant scale."""
    from pxr import UsdGeom
    path = target['prim_path']
    prim = stage.GetPrimAtPath(path)
    require(bool(prim), 'Authored feature prim is absent: ' + path)
    matrix = np.asarray(UsdGeom.XformCache().GetLocalToWorldTransform(prim)).T
    local = np.asarray(target['local_position_m'], dtype=float)
    direction = np.asarray(target['approach_direction_local'], dtype=float)
    distance = target['approach_distance_m']
    require(local.shape == direction.shape == (3,) and np.isfinite([local, direction]).all()
            and abs(np.linalg.norm(direction) - 1) < 1e-6
            and type(distance) in (int, float) and math.isfinite(distance) and 0 < distance <= .15,
            'Invalid explicit surface/approach feature')
    world = (matrix @ [*local, 1])[:3]
    outward = matrix[:3, :3] @ direction
    outward /= np.linalg.norm(outward)
    return world, world + outward * distance


def authored_mass_properties(stage, paths):
    from pxr import UsdPhysics
    result = {}
    for name, path in paths.items():
        prim = stage.GetPrimAtPath(path)
        require(bool(prim) and prim.HasAPI(UsdPhysics.RigidBodyAPI), 'Missing native rigid body: ' + path)
        api = UsdPhysics.MassAPI(prim)
        mass = api.GetMassAttr().Get()
        inertia = api.GetDiagonalInertiaAttr().Get()
        com = api.GetCenterOfMassAttr().Get()
        axes = api.GetPrincipalAxesAttr().Get()
        result[name] = {'prim_path': path, 'mass_kg': None if mass is None else float(mass),
            'diagonal_inertia_kg_m2': None if inertia is None else list(inertia),
            'center_of_mass_m': None if com is None or not np.isfinite(list(com)).all() else list(com),
            'principal_axes_wxyz': None if axes is None else [float(axes.GetReal()), *axes.GetImaginary()],
            'mass_authored': bool(api.GetMassAttr().HasAuthoredValueOpinion()),
            'inertia_authored': bool(api.GetDiagonalInertiaAttr().HasAuthoredValueOpinion())}
        require(mass is not None and math.isfinite(mass) and mass > 0., 'Missing positive authored mass')
    return result


def shape_inventory(checker, paths):
    """Record every final USD collider, including joint support and signed gaps.

    AABB broad phase followed by full oriented-box SAT. Negative bounds gaps
    remain unresolved geometry, not fabricated native mesh penetration.
    """
    from pxr import UsdPhysics
    owners = sorted(paths.items(), key=lambda item: len(item[1]), reverse=True)
    shapes = []
    for path, corners in zip(checker.obstacle_names, checker.obstacle_corners):
        owner = next((name for name, root in owners if path == root or path.startswith(root + '/')), None)
        shapes.append({'prim_path': path, 'body_id': owner, 'corners_m': corners.tolist(),
                       'bounds_m': [corners.min(0).tolist(), corners.max(0).tolist()]})
    joints = []
    for prim in checker.stage.Traverse():
        if prim.IsA(UsdPhysics.Joint):
            joint = UsdPhysics.Joint(prim)
            joints.append({'prim_path': str(prim.GetPath()), 'type': prim.GetTypeName(),
                'body0': [str(x) for x in joint.GetBody0Rel().GetTargets()],
                'body1': [str(x) for x in joint.GetBody1Rel().GetTargets()]})
    candidates, overlaps = [], []
    for i, first in enumerate(shapes):
        if first['body_id'] is None:
            continue
        a = np.asarray(first['corners_m']); alo, ahi = a.min(0), a.max(0)
        for j, second in enumerate(shapes):
            if i == j or first['body_id'] == second['body_id']:
                continue
            b = np.asarray(second['corners_m']); blo, bhi = b.min(0), b.max(0)
            if np.any(np.minimum(ahi, bhi) < np.maximum(alo, blo) - .003):
                continue
            gap = separating_axis_gap(a, b)
            row = {'body_id': first['body_id'], 'shape': first['prim_path'],
                   'other_body_id': second['body_id'], 'other_shape': second['prim_path'],
                   'signed_enclosing_obb_gap_m': gap,
                   'vertical_bottom_to_top_gap_m': float(alo[2] - bhi[2])}
            if gap < -.0001 and (second['body_id'] is None or i < j):
                overlaps.append(row)
            if abs(alo[2] - bhi[2]) <= .003 and np.all(np.minimum(ahi[:2], bhi[:2]) > np.maximum(alo[:2], blo[:2])):
                candidates.append(row)
    supported = {row['body_id'] for row in candidates}
    constrained = {name for name, path in paths.items() for joint in joints
                   if path in joint['body0'] + joint['body1']}
    return {'shapes': shapes, 'joints': joints, 'support_candidates': candidates,
            'unresolved_enclosing_shape_overlaps': overlaps,
            'bodies_without_near_support_or_joint': sorted(set(paths) - supported - constrained),
            'scope': 'Final composed CollisionAPI shape enclosures. SAT is exact for these OBBs, not for non-box meshes/SDFs. Near support is geometric only; native contacts required.'}


def run_case(case, *, adapter=None):
    release = Path(case['release'])
    _, scene, task, initial = task_inputs(release, case['task_id'], seed=case['seed'])
    return run_inputs(case, scene, task, initial, adapter=adapter)


def run_inputs(case, scene, task, initial, *, adapter=None):
    """Audit explicit resolved inputs; adapters must validate their own identities."""
    release = Path(case['release'])
    context = getattr(adapter, 'audit_context', None)
    reset = task.get('reset_case', {}) if context is None else context(case, scene, task)
    if context is None:
        require(reset.get('seed') == case['seed'] and reset.get('configuration_id') == case['configuration_id'],
                'Selected seed/configuration differs from frozen case')
    if adapter is not None:
        adapter.validate_case(case, scene, task)
    family = 'living_room' if adapter is None else reset['family']
    models, stands, yaws = (getattr(adapter, 'installation', installation) if adapter else installation)(scene, initial, family, case['task_id'], case['configuration_id'])
    joints = initial['joint_positions_rad']
    checker_type = RecordedCollisionBounds if adapter is None else adapter.CollisionBounds
    checker = checker_type(scene, scene['robot'], models, joints, stands)
    paths = (body_paths if adapter is None else adapter.body_paths)(scene, task)
    approach_rule = getattr(adapter, 'approach_context', None)
    if approach_rule is not None:
        checker.approach_context = approach_rule(checker, scene, task, paths)
    pose_binding = verify_authored_poses(checker.stage, paths, reset['actual_body_poses_m'])
    mass = authored_mass_properties(checker.stage, paths)
    geometry_binding = ({} if adapter is None else {'actual_geometry_binding':
        adapter.verify_case_geometry(checker.stage, scene, task, mass),
        'validator_source': adapter.source_identity()})
    inventory = shape_inventory(checker, paths)
    arm = getattr(adapter, 'SelfCollision', ArmSelfCollision)(checker)
    target_solver = getattr(checker, 'solve_target', solve_target)
    ready = {side: checker.boxes(side, joints) for side in models}
    samples = []
    for label, target in getattr(adapter, 'case_targets', case_targets)(task).items():
        selected_models = (adapter.target_models(target, models) if adapter is not None
                           and hasattr(adapter, 'target_models') else models)
        reference, point = transformed_feature(target, checker.stage)
        point, surface = free_surface_approach(checker, reference, point, owner_paths=tuple(paths.values()))
        surface = refine_goal_approach(checker, target, reference, surface)
        hands = {side: target_solver(model, joints, point, side, yaws[side], target.get('approach', 'above'),
                                   checker, ready, joint_check=arm.check) for side, model in selected_models.items()}
        accessible = not surface['blocked']
        conditional = target['reference_state'] != 'initial' or bool(target.get('requires_open'))
        geometric_ik = (hands if not conditional else {
            side: target_solver(model, joints, point, side, yaws[side], target.get('approach', 'above'),
                               joint_check=arm.check) for side, model in selected_models.items()})
        usable = [side for side, row in hands.items() if accessible and row['converged'] and row['clearance_passed']]
        samples.append({'label': label, 'feature': target, 'reference_position_m': reference.tolist(),
            'free_approach_position_m': point.tolist(), 'surface': surface, 'hands': hands,
            'geometric_endpoint_ik': geometric_ik,
            'requires_prior_mechanism_change': conditional,
            'required_mechanism_state': target['reference_state'],
            'conditional_clearance_scope': ('Robot-only endpoint IK; actual initial closed-state obstacles retained separately. Opened-scene clearance and opening path not verified.' if conditional else None),
            'initially_accessible_geometry': accessible, 'usable_sides': usable,
            'initial_endpoint_covered': bool(usable)})
    ready_clearance, ready_arm, stand = checker.check(ready), arm.check(joints), checker.stand_check()
    initial_samples = [row for row in samples if not row['requires_prior_mechanism_change']]
    conditional_samples = [row for row in samples if row['requires_prior_mechanism_change']]
    groups = operation_coverage(samples)
    initial_groups = [row for row in groups.values() if row['required'] and not row['conditional']]
    conditional_groups = [row for row in groups.values() if row['required'] and row['conditional']]
    report = {'schema': ('living_randomized' if adapter is None else adapter.schema_prefix)+'_workspace.v1', 'case': case,
        'case_fingerprint': reset.get('case_fingerprint'), 'release_manifest_sha256': digest(release / 'release_manifest.json'),
        'resolved_inputs_sha256': content_digest({'scene': scene, 'task': task, 'initial': initial}),
        'robot_configuration_sha256': content_digest(scene['robot']), 'initial_sha256': content_digest(initial),
        'asset_dependencies': checker.dependencies, 'authored_mass_properties': mass,
        'authored_pose_binding': pose_binding, **geometry_binding,
        **({} if approach_rule is None else {'approach_rule': adapter.describe_approach(checker.approach_context)}),
        'actual_shape_inventory': inventory, 'samples': samples, 'operation_groups': groups,
        'ready_clearance': ready_clearance, 'ready_arm_self_collision': ready_arm,
        'stand_clearance': stand,
        'checks': {'ready_geometry_clear': ready_clearance['clear'] and ready_arm['clear'] and stand['clear'],
            'initial_features_covered': bool(initial_samples) and all(row['initial_endpoint_covered'] for row in initial_samples),
            'initial_required_operation_groups_covered': bool(initial_groups) and all(row['endpoint_covered'] for row in initial_groups),
            'conditional_required_endpoint_groups_covered': all(row['endpoint_covered'] for row in conditional_groups),
            'conditional_endpoint_ik_covered': all(any(hand['converged'] and hand['clearance_passed']
                for hand in row['geometric_endpoint_ik'].values()) for row in conditional_samples),
            'native_support_verified': False, 'conditional_scene_clearance_verified': not conditional_samples},
        'model_calls': 0, 'native_steps': 0,
        'scope': 'Original robot discrete endpoint IK and final-USD enclosing-shape clearance. No path, grasp, operation or native qualification.',
        'robot_manipulation_qualified': False, 'continuous_path_qualified': False}
    if adapter is not None and hasattr(adapter, 'augment_interaction_checks'):
        adapter.augment_interaction_checks(report, task=task, checker=checker, paths=paths,
            models=models, joints=joints, yaws=yaws, ready=ready, arm=arm)
    return report


def main(*, adapter=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    request = (read_request if adapter is None else adapter.read_request)(args.request)
    name_case = case_name if adapter is None else adapter.case_name
    require(args.output.resolve().is_relative_to(OUTPUT_ROOT), 'Evidence belongs under shared datasets')
    args.output.mkdir(parents=True, exist_ok=False)
    summary = {'schema': ('living_randomized' if adapter is None else adapter.schema_prefix)+'_workspace_batch.v1', 'request_sha256': digest(args.request), 'cases': []}
    for case in request['cases']:
        path = args.output / name_case(case) / 'workspace.json'
        path.parent.mkdir(parents=True, exist_ok=False)
        try:
            report = run_case(case, adapter=adapter)
            row = {'case': case, 'completed': True, 'initial_endpoint_covered': sum(s['initial_endpoint_covered'] for s in report['samples']),
                   'features': len(report['samples']), 'checks': report['checks']}
        except Exception as exc:
            report = row = {'case': case, 'completed': False, 'error': {'type': type(exc).__name__, 'message': str(exc)}}
            if adapter is not None:
                report['validator_source'] = adapter.source_identity()
                manifest = Path(case['release'])/'release_manifest.json'
                if manifest.is_file():
                    report['release_manifest_sha256'] = digest(manifest)
        dump(path, report)
        summary['cases'].append({**row, 'report': str(path), 'sha256': digest(path)})
        summary['all_reports_completed'] = all(r['completed'] for r in summary['cases'])
        summary['all_initial_features_covered'] = all(r.get('checks', {}).get('initial_features_covered', False) for r in summary['cases'])
        summary['all_initial_required_operation_groups_covered'] = all(r.get('checks', {}).get('initial_required_operation_groups_covered', False) for r in summary['cases'])
        summary['all_conditional_required_endpoint_groups_covered'] = all(r.get('checks', {}).get('conditional_required_endpoint_groups_covered', False) for r in summary['cases'])
        summary['all_ready_geometry_clear'] = all(r.get('checks', {}).get('ready_geometry_clear', False) for r in summary['cases'])
        summary['native_or_manipulation_qualification'] = False
        dump(args.output / 'summary.json', summary)
        print(json.dumps(row), flush=True)
    return 0 if all(r['completed'] for r in summary['cases']) else 1


if __name__ == '__main__':
    raise SystemExit(main())
