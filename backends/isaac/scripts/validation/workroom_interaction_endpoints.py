"""Private endpoint diagnostics for seven placement-only kitchen/lab tasks.

These use the original ready-hand geometry, collision checker and IK. A
precontact offset is not a contact measurement, and independently reachable
endpoints are not a trajectory or a grasp. No controller state is changed.
"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import json

import numpy as np

from bench.paths import load_json
from scripts.validation import living_randomized_workspace as living

TASKS = frozenset(('breakfast_serving', 'pot_lid_placement', 'surface_wiping',
    'cap_placement', 'solid_weighing', 'pipette_transfer', 'tube_clamp_transfer'))
PRECONTACT_M = .010
LIFT_M = .030
SURFACE_TOLERANCE_M = .000020

# Explicitly reviewed original mechanics: the referee consumes measured axial
# piston displacement; it does not require a ThumbPad-specific contact. A future
# rule change must be reviewed instead of inheriting this geometric supplement.
PIPETTE_SEMANTIC_SOURCES = {
    'evaluator/sequence_evaluator.py': ('scenes/workrooms/laboratory/evaluation.py',
        '1b317f66ba0ce365c77ca9b7526dc3c2e247e2c529e07a2b2e92401686ea2dac'),
    'evaluator/liquid_dynamics.py': ('scenes/workrooms/laboratory/tasks/pipette_transfer/liquid.py',
        'b9591de5bc732c6f36824bdb8e15d12f3a6555ab3eb96b7d9eb3222688c243e6'),
    'evaluator/rules.py': ('scenes/workrooms/laboratory/rules.py',
        '8c29a1f381d4a1929eb3af7e39bf2bc86e2e94bccbfc7b8824d96c622d75fefc'),
}
# Reviewed as one complete source tuple, never independent interchangeable pins.
# This successor changes public dry masses to actual asset masses and passes
# metadata to public_rules. Its piston dynamics and all scoring code/thresholds
# retain the original semantics; the solid-weighing text branch is inapplicable.
PIPETTE_SEMANTIC_PROFILES = {
    'original_piston_v1': {name: row[1] for name, row in PIPETTE_SEMANTIC_SOURCES.items()},
    'parameterized_public_dry_mass_v1': {
        'evaluator/sequence_evaluator.py': '4f5f66fa849541373a9da644234db2a5a1c48ed54698253119e295ba5e31ac1c',
        'evaluator/liquid_dynamics.py': 'b9591de5bc732c6f36824bdb8e15d12f3a6555ab3eb96b7d9eb3222688c243e6',
        'evaluator/rules.py': '6765a59c73af4362df934f43b6f6c8121727010fdba70250f8c86749ee568414',
    },
}


def pipette_semantic_sources(release, task):
    """Admit only fully reviewed original or explicit parameterized mechanics."""
    source = Path(__file__).resolve().parents[2]
    manifest = load_json(release/'release_manifest.json')
    actual = {name: living.digest(release/name) for name in PIPETTE_SEMANTIC_SOURCES}
    profiles = [name for name, pins in PIPETTE_SEMANTIC_PROFILES.items() if actual == pins]
    living.require(len(profiles) == 1,
                   'Original piston displacement/contact semantics require renewed review')
    profile = profiles[0]
    if profile == 'parameterized_public_dry_mass_v1':
        parameters = task.get('reset_case', {}).get('parameterized_physics', {}).get('parameters', {})
        living.require(parameters.get('schema') == 'parameterized_physics_v1'
            and parameters.get('family') == 'laboratory'
            and parameters.get('task_id') == task.get('id') == 'pipette_transfer',
            'Reviewed parameterized piston semantics require explicit matching task parameters')
    records = {}
    for filename, (relative, expected_original) in PIPETTE_SEMANTIC_SOURCES.items():
        living.require(manifest['files_sha256'].get(filename) == actual[filename]
            and living.digest(source/relative) == expected_original,
            'Original piston displacement/contact semantics require renewed review')
        records[filename] = {'path': str(release/filename), 'sha256': actual[filename],
            'source': relative, 'original_source_sha256': expected_original}
    return profile, records


def _mesh_triangles(prim):
    """Authored collision surface, including the source convex-hull operation.

    This deliberately does not claim to reproduce PhysX cooking or an SDF.
    """
    from pxr import UsdGeom, UsdPhysics
    from scipy.spatial import ConvexHull
    mesh = UsdGeom.Mesh(prim)
    points = np.asarray(mesh.GetPointsAttr().Get(), dtype=float)
    living.require(points.ndim == 2 and points.shape[1] == 3 and np.isfinite(points).all(),
                   'Invalid source mesh points')
    if UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr().Get() == 'convexHull':
        return points[ConvexHull(points).simplices]
    counts = list(mesh.GetFaceVertexCountsAttr().Get())
    indices = np.asarray(mesh.GetFaceVertexIndicesAttr().Get(), dtype=int)
    living.require(sum(counts) == len(indices) and all(n >= 3 for n in counts)
                   and np.all((indices >= 0) & (indices < len(points))), 'Invalid source mesh topology')
    result, start = [], 0
    for count in counts:
        face = indices[start:start + count]; start += count
        # Existing task source polygons are convex; non-triangles are checked
        # before fan triangulation, so a concave polygon cannot invent surface.
        vertices = points[face]
        if count > 3:
            normal = np.cross(vertices[1]-vertices[0], vertices[2]-vertices[0])
            length = np.linalg.norm(normal)
            living.require(length > 1e-14, 'Degenerate source polygon')
            normal /= length
            living.require(np.max(np.abs((vertices-vertices[0]) @ normal)) <= 1e-7,
                           'Nonplanar source polygon needs explicit triangulation')
            turns = [np.dot(np.cross(vertices[(i+1) % count]-vertices[i],
                                    vertices[(i+2) % count]-vertices[(i+1) % count]), normal)
                     for i in range(count)]
            living.require(min(turns) >= -1e-12 or max(turns) <= 1e-12,
                           'Concave source polygon needs explicit triangulation')
        result.extend(points[[face[0], face[i], face[i+1]]] for i in range(1, count-1))
    return np.asarray(result)


def surface_witness(stage, target, world_point):
    """Reject a purported grip reference in a hollow interior or visual mesh."""
    from pxr import Usd, UsdGeom, UsdPhysics
    import trimesh
    root = stage.GetPrimAtPath(target['prim_path'])
    living.require(bool(root), 'Missing interaction reference prim')
    declared = target.get('source_collider_path', target.get('surface_witness', {}).get('collider_path'))
    if declared is not None:
        living.require(isinstance(declared, str) and (declared == target['prim_path']
            or declared.startswith(target['prim_path']+'/')), 'Surface witness is outside its declared object')
        root = stage.GetPrimAtPath(declared)
        living.require(bool(root) and root.HasAPI(UsdPhysics.CollisionAPI), 'Declared source witness is not a collider')
    cache = UsdGeom.XformCache()
    witnesses = []
    for prim in ([root] if declared is not None else Usd.PrimRange(root)):
        if not prim.HasAPI(UsdPhysics.CollisionAPI) or UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get() is False:
            continue
        matrix = np.asarray(cache.GetLocalToWorldTransform(prim)).T
        linear = matrix[:3, :3]
        scales = np.linalg.norm(linear, axis=0)
        living.require(np.allclose(scales, scales[0], atol=1e-8, rtol=0) and scales[0] > 0
                       and np.allclose(linear.T @ linear, np.eye(3)*scales[0]**2, atol=1e-8, rtol=0),
                       'Surface diagnostic requires a rigid or uniform-scale collider')
        point = (np.linalg.inv(matrix) @ [*world_point, 1.])[:3]
        kind = prim.GetTypeName()
        if kind == 'Mesh':
            triangles = _mesh_triangles(prim)
            nearest = trimesh.triangles.closest_point(triangles, np.repeat(point[None, :], len(triangles), axis=0))
            distance = float(np.linalg.norm(nearest-point, axis=1).min())
        elif kind == 'Cube':
            half = float(UsdGeom.Cube(prim).GetSizeAttr().Get())/2
            q = np.abs(point)-half
            distance = abs(float(np.linalg.norm(np.maximum(q, 0)) + min(float(q.max()), 0.)))
        elif kind == 'Sphere':
            distance = abs(float(np.linalg.norm(point))-float(UsdGeom.Sphere(prim).GetRadiusAttr().Get()))
        elif kind in ('Cylinder', 'Capsule'):
            shape = UsdGeom.Cylinder(prim) if kind == 'Cylinder' else UsdGeom.Capsule(prim)
            axis = 'XYZ'.index(str(shape.GetAxisAttr().Get()))
            half = float(shape.GetHeightAttr().Get())/2
            radius = float(shape.GetRadiusAttr().Get())
            radial = np.linalg.norm(np.delete(point, axis))
            if kind == 'Capsule':
                distance = abs(float(np.hypot(radial, max(abs(point[axis])-half, 0)))-radius)
            else:
                q = np.array([radial-radius, abs(point[axis])-half])
                distance = abs(float(np.linalg.norm(np.maximum(q, 0)) + min(float(q.max()), 0.)))
        else:
            raise ValueError('Unsupported source surface primitive: '+kind)
        witnesses.append({'prim_path': str(prim.GetPath()), 'distance_m': distance*float(scales[0])})
    living.require(bool(witnesses), 'Actual surface has no enabled source collider')
    closest = min(witnesses, key=lambda row: row['distance_m'])
    living.require(closest['distance_m'] <= SURFACE_TOLERANCE_M,
                   'Actual surface reference is not on enabled collision geometry: '+target['label'])
    return {**closest, 'tolerance_m': SURFACE_TOLERANCE_M,
        'scope': 'Source CollisionAPI surface; not native cooked geometry or a measured contact.'}


def same_hand_coverage(endpoints, *, conditional):
    """Three different hands, or three different feature alternatives, do not add up."""
    common = None
    for row in endpoints.values():
        usable = {side for side, result in row['hands'].items()
                  if result['converged'] and result['clearance_passed']
                  and (conditional or not row['surface']['blocked'])}
        common = usable if common is None else common & usable
    return sorted(common or ())


def pipette_press_supplement(report, task, checker, sequences, groups):
    """Same rigid plunger's pull-ring top is also a real push surface.

    This is a separate private diagnostic candidate. Neither the frozen feature
    list nor the original unsuccessful ThumbPad result is overwritten.
    """
    group = groups.get('plunger_press')
    if task.get('id') != 'pipette_transfer' or group is None or group['covered']:
        return
    from pxr import UsdPhysics
    metadata = task['asset_metadata']; hydraulics = metadata['task_parameters']['hydraulics']
    actor = metadata['actors']['plunger']; tool = metadata['actors']['pipette']
    living.require(hydraulics['plunger_id'] == 'plunger' and hydraulics['tool_id'] == 'pipette'
        and hydraulics['joint_path'] == metadata['joints']['plunger'], 'Unknown piston body/axis binding')
    joint_prim = checker.stage.GetPrimAtPath(hydraulics['joint_path'])
    living.require(bool(joint_prim) and joint_prim.IsA(UsdPhysics.PrismaticJoint), 'Missing original piston joint')
    joint = UsdPhysics.PrismaticJoint(joint_prim)
    living.require([str(x) for x in joint.GetBody0Rel().GetTargets()] == [tool]
        and [str(x) for x in joint.GetBody1Rel().GetTargets()] == [actor]
        and joint.GetJointEnabledAttr().Get() is not False and joint.GetAxisAttr().Get() == 'Z'
        and joint.GetLowerLimitAttr().Get() == 0.
        and np.isclose(joint.GetUpperLimitAttr().Get(), hydraulics['stroke_m'], atol=1e-8, rtol=0),
        'Supplement requires the original positive-Z piston travel')
    for rotation in (joint.GetLocalRot0Attr().Get(), joint.GetLocalRot1Attr().Get()):
        living.require(abs(abs(float(rotation.GetReal()))-1.) < 1e-7
            and np.linalg.norm(rotation.GetImaginary()) < 1e-7, 'Piston joint axis frame changed')
    candidates = [row for row in report['samples'] if row['feature']['label'] == 'plunger_pull_ring']
    living.require(len(candidates) == 1, 'Missing unique original pull-ring feature')
    sample = candidates[0]; feature = sample['feature']
    living.require(feature['type'] == 'actual_surface' and feature['reference_state'] == 'initial'
        and feature['operation_group'] == 'plunger_pull' and feature['prim_path'] == actor
        and feature.get('source_collider_path') == actor+'/PullRing'
        and feature['local_position_m'] == [0., 0., .1125]
        and feature['approach_direction_local'] == [0., 0., 1.], 'Unknown actual ring push surface')
    for path in (actor+'/PullRing', actor+'/ThumbPad'):
        prim = checker.stage.GetPrimAtPath(path)
        living.require(bool(prim) and prim.HasAPI(UsdPhysics.CollisionAPI)
            and UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get() is not False,
            'Piston push surface is not an enabled collider')
        owner = prim
        while owner and not owner.HasAPI(UsdPhysics.RigidBodyAPI):
            owner = owner.GetParent()
        living.require(bool(owner) and str(owner.GetPath()) == actor, 'Ring and pad must share the actual rigid piston')
    witness = surface_witness(checker.stage, feature, sample['reference_position_m'])
    semantic_profile, semantic_sources = pipette_semantic_sources(Path(report['case']['release']), task)
    original = sequences[feature['label']]
    living.require(not original['conditional'] and original['source_surface'] == witness,
                   'Pull-ring sequence geometry differs from its same-input witness')
    label = 'plunger_press_ring_top_supplement'
    living.require(label not in sequences, 'Duplicate piston push supplement')
    supplemental = deepcopy(original)
    supplemental.update(feature_label=label, operation_group='plunger_press', required=group['required'],
        supplemental=True, semantic_basis={
            'source_feature_label': feature['label'], 'source_feature_sha256': living.content_digest(feature),
            'resolved_inputs_sha256': report['resolved_inputs_sha256'],
            'source_sequence_sha256': living.content_digest(original),
            'shared_rigid_actor': actor, 'joint_path': str(joint_prim.GetPath()),
            'semantic_source_profile': semantic_profile,
            'frozen_semantic_sources': semantic_sources,
            'reason': 'The original ring and pad are colliders of the same prismatic piston. A downward push on its real ring top can drive the same measured axial displacement; original rules do not prescribe a pad-only contact.',
            'reuse_scope': 'Exact existing ring surface, approach/precontact/lift coordinates, hand geometry and collision checks in this same resolved input; no new IK success is inferred for the original pad.',
            'original_press_feature_labels': list(group['feature_labels'])})
    sequences[label] = supplemental
    group['original_endpoint_covered_without_supplement'] = group['covered']
    group['feature_labels'].append(label)
    if supplemental['covered']:
        group['covered_feature_labels'].append(label)
    group['covered'] = bool(group['covered_feature_labels'])


def augment(report, *, task, checker, paths, models, joints, yaws, ready, arm):
    sequences, groups = {}, {}
    for sample in report['samples']:
        target = sample['feature']
        if target['type'] != 'actual_surface':
            continue
        reference = np.asarray(sample['reference_position_m'])
        witness = surface_witness(checker.stage, target, reference)
        living.require(target['approach_distance_m'] >= PRECONTACT_M,
                       'Approach must not be shorter than the explicit precontact offset')
        near_target = {**target, 'approach_distance_m': PRECONTACT_M}
        _, near = living.transformed_feature(near_target, checker.stage)
        near, near_surface = living.free_surface_approach(checker, reference, near, owner_paths=tuple(paths.values()))
        # An independently checked endpoint, not a claim that the connecting
        # vertical segment is collision-free or that an object follows the hand.
        lift = near + [0., 0., LIFT_M]
        conditional = sample['requires_prior_mechanism_change']
        endpoints = {'approach': {'position_m': sample['free_approach_position_m'],
            'surface': deepcopy(sample['surface']),
            'hands': deepcopy(sample['geometric_endpoint_ik'] if conditional else sample['hands'])}}
        for name, point in (('precontact', near), ('lift', lift)):
            hands = {side: getattr(checker, 'solve_target', living.solve_target)(model, joints, point, side, yaws[side],
                target.get('approach', 'above'), None if conditional else checker,
                None if conditional else ready, joint_check=arm.check) for side, model in models.items()}
            endpoints[name] = {'position_m': point.tolist(), 'surface': deepcopy(near_surface), 'hands': hands}
        usable = same_hand_coverage(endpoints, conditional=conditional)
        row = {'feature_label': target['label'], 'operation_group': target['operation_group'],
            'required': target['required'], 'conditional': conditional, 'source_surface': witness,
            'precontact_requested_offset_m': PRECONTACT_M, 'lift_world_z_offset_m': LIFT_M,
            'precontact_actual_reference_distance_m': float(np.linalg.norm(near-reference)),
            'endpoints': endpoints, 'usable_sides': usable, 'covered': bool(usable),
            'scene_clearance_checked': not conditional,
            'scope': ('Robot-only conditional endpoints; no opened-state geometry or mechanism trajectory.' if conditional else
                      'Original ready-hand geometry at three discrete endpoints; no contact, grasp, held-object or continuous-path claim.')}
        retry = getattr(checker, 'retry_interaction_sequence', None)
        if retry is not None and not row['covered'] and not conditional:
            fallback = retry(row, checker=checker, models=models, joints=joints,
                             ready=ready, joint_check=arm.check)
            if fallback is not None:
                row.update(original_covered=False, original_usable_sides=list(usable),
                           orientation_fallback=fallback)
                usable = fallback['usable_sides']
                row.update(usable_sides=usable, covered=bool(usable),
                           coverage_basis='same_orientation_fallback' if usable else 'original_and_fallback_failed')
        sequences[target['label']] = row
        group = groups.setdefault(target['operation_group'], {'required': target['required'],
            'conditional': conditional, 'feature_labels': [], 'covered_feature_labels': []})
        living.require((group['required'], group['conditional']) == (target['required'], conditional),
                       'Interaction group mixes requirements or mechanism stages')
        group['feature_labels'].append(target['label'])
        if usable:
            group['covered_feature_labels'].append(target['label'])
    for group in groups.values():
        group['covered'] = bool(group['covered_feature_labels'])
    pipette_press_supplement(report, task, checker, sequences, groups)
    initial = [g for g in groups.values() if g['required'] and not g['conditional']]
    conditional = [g for g in groups.values() if g['required'] and g['conditional']]
    living.require(bool(initial), 'New kitchen/lab task needs explicit required initial grip references')
    report['interaction_endpoint_sequences'] = sequences
    report['interaction_endpoint_groups'] = groups
    report['checks']['initial_required_interaction_sequences_covered'] = all(g['covered'] for g in initial)
    report['checks']['conditional_required_interaction_sequences_covered'] = all(g['covered'] for g in conditional)
    from scripts.validation.workroom_source_interference import review_source_geometry
    report['source_geometry_review'] = review_source_geometry(checker, paths, task)
