"""Static geometric-support and robot-free dynamic settle admission of parameterized candidates.

G_canonical_support_regression
   The static checker's ``no_missing_geometric_support`` inventory on the
   composed, robot-free task stage: the same collider inventory as the workroom
   CollisionBounds (robot and stand colliders excluded) and the same
   parameterized source-geometry review. It is a regression check against the
   canonical world: a candidate is rejected only for bodies that have support in
   the same installation's canonical seed 0 but lack it in the candidate. When
   seed 0 has full support (e.g. NIST connector) this is exactly the original
   rule; bodies the checker cannot support even at seed 0 (surface_wiping sponge
   and cells) are left to the dynamic settle admission. Both inventories are
   recorded. A fixed checker changes both automatically.

D  Robot-free passive settle on GPU (Isaac Sim 6.1) for the native protocol hold
   (0.8 s; laundry 1.6 s). Every object must meet the unchanged native passive
   thresholds over the terminal 0.5 s window: displacement <= 5 mm, linear speed
   <= 0.02 m/s, angular speed <= 0.2 rad/s (the frozen NIST twist-free centreline
   exemption applies only where the native referee declares it); cloth P95 node
   speed <= 0.08 m/s with drift <= 15 mm. Native support evidence must be complete
   and the frozen referee must not report a physical failure during the hold.
   Thresholds are never relaxed here.

The GPU runner is scripts/validation/parameterized_settle.py; this module holds the
CPU-side rule and the verdict so both are testable without a simulator.
"""
from __future__ import annotations

from itertools import product
import math

import numpy as np

from bench.paths import load_json

SCHEMA = 'parameterized_settle_admission.v1'
SUPPORT_SCHEMA = 'parameterized_geometric_support_admission.v1'
RIGID_THRESHOLDS = {'displacement_m': .005, 'linear_speed_m_s': .02, 'angular_speed_rad_s': .2}
CLOTH_THRESHOLDS = {'p95_node_speed_m_s': .08, 'drift_m': .015}
TERMINAL_WINDOW_S = .5
HOLD_S = {'laundry_room': 1.6}
DEFAULT_HOLD_S = .8
SETTLED_STATE_TASKS = ('nist_cable_routing', 'solid_weighing')


def hold_seconds(family):
    """Native protocol hold of the family (laundry 1.6 s, otherwise 0.8 s)."""
    return HOLD_S.get(family, DEFAULT_HOLD_S)


def support_checker(scene):
    """Robot-free stand-in with exactly the workroom CollisionBounds obstacle inventory."""
    from types import SimpleNamespace
    from pxr import Usd, UsdGeom, UsdPhysics
    from scripts.validation.workroom_randomized_workspace import composed_support_stage
    stage, sources = composed_support_stage(scene)
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ['default', 'render', 'proxy', 'guide'], False, True)
    transforms = UsdGeom.XformCache()
    corners, names = [], []
    for prim in Usd.PrimRange(stage.GetPseudoRoot(), Usd.TraverseInstanceProxies()):
        path = str(prim.GetPath())
        if path.startswith('/Robots/') or 'robot_stand' in path.lower():
            continue
        if not prim.HasAPI(UsdPhysics.CollisionAPI) or not prim.IsA(UsdGeom.Boundable):
            continue
        if UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get() is False:
            continue
        bound = cache.ComputeUntransformedBound(prim).ComputeAlignedRange()
        points = np.array([(*p, 1.) for p in product(*zip(bound.GetMin(), bound.GetMax()))])
        world = np.asarray(transforms.GetLocalToWorldTransform(prim)).T
        points = (world@points.T).T[:, :3]
        if not np.isfinite(points).all():
            raise ValueError('Nonfinite actual collision shape')
        corners.append(points); names.append(path)
    return SimpleNamespace(stage=stage, obstacle_names=names, obstacle_corners=np.asarray(corners),
                           dependencies=sources)


GEOMETRIC_SUPPORT_RULE = 'G_canonical_support_regression'


def support_review(scene, task, *, checker=None):
    """The static checker's missing-support inventory on the composed robot-free stage."""
    from scripts.validation.parameterized_scene_checks import source_geometry
    from scripts.validation.workroom_randomized_workspace import body_paths
    checker = support_checker(scene) if checker is None else checker
    paths = body_paths(scene, task)
    review = source_geometry(checker, paths, task)
    chain = 'bodies_without_geometric_support_chain' in review
    missing = sorted(review['bodies_without_geometric_support_chain'] if chain
                     else review['bodies_without_near_support_or_joint'])
    return {'criterion': 'bodies_without_geometric_support_chain' if chain else 'bodies_without_near_support_or_joint',
            'bodies': sorted(paths), 'missing': missing, 'collider_count': len(checker.obstacle_names)}


def support_admission(candidate, canonical):
    """Rule G from two support reviews: reject only bodies supported at canonical seed 0."""
    if candidate['criterion'] != canonical['criterion'] or candidate['bodies'] != canonical['bodies']:
        raise ValueError('Candidate and canonical seed-0 support inventories are not comparable')
    regression = sorted(set(candidate['missing']) - set(canonical['missing']))
    return {'schema': SUPPORT_SCHEMA, 'rule': GEOMETRIC_SUPPORT_RULE, 'passed': not regression,
            'regression': regression, 'candidate_missing': list(candidate['missing']),
            'canonical_seed0_missing': list(canonical['missing']),
            'canonical_seed0_supported': sorted(set(canonical['bodies'])-set(canonical['missing'])),
            'criterion': candidate['criterion'], 'collider_count': candidate['collider_count'],
            'canonical_collider_count': canonical['collider_count'],
            'scope': 'Static checker no_missing_geometric_support inventory on the composed robot-free task stage, '
                     'as a regression check against the same installation\'s canonical seed 0; identical to the '
                     'original rule whenever seed 0 has full support. Geometry only, not native support or settling.'}


def geometric_support(scene, task, canonical_scene, canonical_task):
    """Rule G for a composed candidate against its installation's canonical seed-0 world."""
    return support_admission(support_review(scene, task), support_review(canonical_scene, canonical_task))


def _finite(value, name):
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise ValueError('Invalid settle statistic: '+name)
    return float(value)


def rigid_verdict(trace, placement=None):
    """Per-object terminal-window stability with the unchanged native thresholds."""
    if trace.get('diagnostic_thresholds') != RIGID_THRESHOLDS or trace.get('terminal_window_s') != TERMINAL_WINDOW_S:
        raise ValueError('Native passive thresholds/window differ')
    exempt = set(placement.get('twist_free_ids', [])) if placement is not None else set()
    objects, failures = {}, []
    for name, row in sorted(trace['objects'].items()):
        drift = _finite(row['window_displacement_from_first_m'], name)
        linear = _finite(row['max_linear_speed_m_s'], name)
        angular = _finite(row['max_angular_speed_rad_s'], name)
        stable = (drift <= RIGID_THRESHOLDS['displacement_m'] and linear <= RIGID_THRESHOLDS['linear_speed_m_s']
                  and (name in exempt or angular <= RIGID_THRESHOLDS['angular_speed_rad_s']))
        supported = row['native_supported_or_joint_rooted'] is True
        objects[name] = {'window_displacement_m': drift, 'max_linear_speed_m_s': linear,
                         'max_angular_speed_rad_s': angular, 'angular_exempt_by_frozen_referee': name in exempt,
                         'stable': stable, 'native_supported_or_joint_rooted': supported}
        if not stable or not supported:
            failures.append(name)
    return objects, failures


def cloth_verdict(cloths):
    """Cloth P95 node speed and drift with the unchanged native thresholds."""
    rows, failures = {}, []
    for name, row in sorted((cloths or {}).items()):
        speed = _finite(row['p95_node_speed_m_s'], name)
        drift = _finite(row['drift_m'], name)
        stable = speed <= CLOTH_THRESHOLDS['p95_node_speed_m_s'] and drift <= CLOTH_THRESHOLDS['drift_m']
        rows[name] = {'p95_node_speed_m_s': speed, 'drift_m': drift, 'stable': stable}
        if not stable:
            failures.append(name)
    return rows, failures


# The frozen native passive validation (workroom_randomized_native.validate_chain_integrity) requires
# every flexible joint chain to stay intact and of nominal length; it raises this exact message.
CHAIN_INTEGRITY_FAILURE = 'Original native joint-chain integrity or length condition failed'


def passive_observation_failure(validate, observation, score, task):
    """The native passive validation of one control, with a broken/overstretched joint chain as a physical failure.

    The native protocol aborts such a hold, so the world fails it; an admission must record that as the
    candidate's passive physical failure instead of aborting the whole settle batch. Every other
    validation error (observer/contact/evidence invariants) is raised unchanged.
    """
    try:
        return validate(observation, score, task)
    except ValueError as error:
        if str(error) != CHAIN_INTEGRITY_FAILURE:
            raise
        return {'status': 'task_failure', 'reason': 'joint_chain_integrity_or_length',
                'irrecoverable_reason': 'joint_chain_integrity_or_length', 'native_validation_message': str(error)}


def placement_review_applicable(row):
    """Whether the frozen task placement review can replay this hold.

    That review re-validates every recorded control with the same native passive validation, so after a
    broken/overstretched joint chain at any control (``joint_chain_failure``, also when another passive
    failure such as a physics safety stop was recorded first) it would raise that error again; the
    candidate is already rejected by its passive physical failure and the review is skipped (and the
    skip recorded).
    """
    failure = row.get('passive_physical_failure') or {}
    return 'joint_chain_failure' not in row and failure.get('native_validation_message') != CHAIN_INTEGRITY_FAILURE


def settle_verdict(row):
    """Admission of one settled candidate row produced by the GPU runner."""
    placement = row.get('task_specific_placement_stability')
    objects, failed = rigid_verdict(row['trace'], placement)
    cloths, cloth_failed = cloth_verdict(row.get('cloth_settle'))
    physical = row.get('passive_physical_failure')
    admitted = not failed and not cloth_failed and physical is None
    return {'schema': SCHEMA, 'admitted': admitted, 'objects': objects, 'unstable_objects': failed,
            'cloths': cloths, 'unstable_cloths': cloth_failed, 'passive_physical_failure': physical,
            'hold_s': row['hold_s'], 'thresholds': {**RIGID_THRESHOLDS, **CLOTH_THRESHOLDS,
                                                    'terminal_window_s': TERMINAL_WINDOW_S},
            'robot_free': row.get('robot_free')}


# ------------------------------------------------- settled-state publication
SETTLED_SCHEMA = 'parameterized_settled_initial_state.v1'
MAXIMUM_SETTLE_HOLD_S = 3.


def settled_state_record(case, report_ref, object_names, object_q, paths, verdict):
    """Settled body poses after an extended robot-free hold (cable/granular tasks only)."""
    if case['task_id'] not in SETTLED_STATE_TASKS:
        raise ValueError('Settled-state publication is declared only for cable and granular tasks')
    q = np.asarray(object_q, dtype=float)
    if q.shape != (len(object_names), 7) or not np.isfinite(q).all() or set(object_names) != set(paths):
        raise ValueError('Invalid settled native state')
    poses = {}
    for index, name in enumerate(object_names):
        quaternion = np.asarray([q[index, 6], q[index, 3], q[index, 4], q[index, 5]])
        quaternion /= np.linalg.norm(quaternion)
        poses[name] = {'prim_path': paths[name], 'position_m': q[index, :3].tolist(),
                       'quaternion_wxyz': quaternion.tolist()}
    return {'schema': SETTLED_SCHEMA, 'task_id': case['task_id'], 'family': case['family'],
            'configuration_id': case['configuration_id'], 'seed': case['seed'], 'attempt': case['attempt'],
            'layout_amplitude_scale': case.get('layout_amplitude_scale', 1.), 'hold_s': case['hold_s'],
            'settled': verdict['admitted'] is True,
            'source': {'settle_inputs': case['inputs'], 'source_asset': case.get('asset'), 'settle_report': report_ref},
            'body_poses': poses,
            'scope': 'Final state of a robot-free passive hold whose terminal window met the unchanged native '
                     'thresholds; used as the published initial state only for cable/granular tasks.'}


def checked_settled_state(reference, *, task_id, seed, attempt, layout_amplitude_scale):
    """A pinned settled state that may replace the dynamic reset poses of exactly this candidate."""
    import hashlib
    import json
    from pathlib import Path
    path = Path(reference['path'])
    if not path.is_absolute() or hashlib.sha256(path.read_bytes()).hexdigest() != reference['sha256']:
        raise ValueError('Settled initial state changed')
    record = load_json(path)
    if (record.get('schema') != SETTLED_SCHEMA or task_id not in SETTLED_STATE_TASKS or record['settled'] is not True
            or (record['task_id'], record['seed'], record['attempt'], record['layout_amplitude_scale'])
            != (task_id, seed, attempt, layout_amplitude_scale)
            or not hold_seconds(record['family']) <= record['hold_s'] <= MAXIMUM_SETTLE_HOLD_S):
        raise ValueError('Settled initial state does not belong to this cable/granular candidate')
    for name, row in record['body_poses'].items():
        q, p = np.asarray(row['quaternion_wxyz'], dtype=float), np.asarray(row['position_m'], dtype=float)
        if q.shape != (4,) or p.shape != (3,) or not np.isfinite([*q, *p]).all() or abs(np.linalg.norm(q)-1) > 1e-6:
            raise ValueError('Invalid settled pose: '+name)
    return record


# ------------------------------------------------------------- laundry/cloth
def cloth_rows(trace):
    """Per-cloth P95 node speed (the laundry native speed policy) and terminal all-node drift."""
    limits = trace['original_thresholds']
    if (limits.get('max_node_speed_m_s') != CLOTH_THRESHOLDS['p95_node_speed_m_s']
            or limits.get('stable_node_drift_m') != CLOTH_THRESHOLDS['drift_m']
            or limits.get('speed_node_percentile') != 95.):
        raise ValueError('Laundry task cloth thresholds differ from the declared admission thresholds')
    interval = trace['speed_policy'] == 'position_interval'
    return {name: {'p95_node_speed_m_s': row['max_interval_speed_percentile_m_s'] if interval else
                   max(row['max_raw_speed_percentile_m_s'], row['max_pose_speed_percentile_m_s']),
                   'drift_m': row['max_node_drift_m']} for name, row in trace['cloths'].items()}


def laundry_settle_verdict(row):
    """Cloth admission: unchanged P95 speed/drift plus the native terminal contact geometry and table support."""
    trace = row.get('laundry_trace')
    if trace is None:
        return {'schema': SCHEMA, 'admitted': False, 'cloths': {}, 'unstable_cloths': [],
                'laundry_failure': row.get('laundry_failure'), 'hold_s': row['hold_s'],
                'thresholds': {**CLOTH_THRESHOLDS, 'terminal_window_s': TERMINAL_WINDOW_S}, 'robot_free': row.get('robot_free')}
    cloths, failed = cloth_verdict(cloth_rows(trace))
    supported = [name for name, value in trace['cloths'].items() if value['geometric_support_present'] is not True]
    geometry = trace['terminal_contact_geometry_acceptable'] is True
    admitted = not failed and not supported and geometry and trace['passed'] is True
    return {'schema': SCHEMA, 'admitted': admitted, 'cloths': cloths, 'unstable_cloths': failed,
            'unsupported_cloths': supported, 'terminal_contact_geometry_acceptable': geometry,
            'hold_s': row['hold_s'], 'thresholds': {**CLOTH_THRESHOLDS, 'terminal_window_s': trace['terminal_window_s']},
            'robot_free': row.get('robot_free')}
