"""Read native mixed spoon geometry without advancing physics or changing state.

The original mixed collector export is retained as unchecked for its unsupported
SDF rows. Separate complete native inventory, convex and actual SDF field checks
are explicit here. This is geometry instrumentation, not dynamics acceptance.
"""
from pathlib import Path
import argparse
import json
import traceback
import numpy as np
import trimesh
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from bench.paths import OUTPUT_ROOT

PROJECT = Path(__file__).resolve().parents[2]
CONFIG = {'convex_corner_error_m': 1e-7, 'convex_relative_volume_error': 1e-6,
          'sdf_error_in_requested_pitches': 2., 'finite_difference_step_in_pitches': .5,
          'smooth_normal_difference': 1e-4, 'gradient_fd_error': .15,
          'documented_distance_channel': 0, 'documented_gradient_channels': [1, 2, 3],
          'layout_calibration': 'unique_original_mesh_sign_distance_and_finite_difference_v1'}


def exact_mesh_distances(mesh, points):
    """Negative-inside distances from original triangles and solid-angle winding."""
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError('Finite Nx3 points required')
    if not mesh.is_watertight or not mesh.is_winding_consistent or mesh.volume <= 0:
        raise ValueError('Closed positive original collision surface required')
    closest, distances, triangles = trimesh.proximity.closest_point_naive(mesh, points)
    winding = []
    for start in range(0, len(points), 32):
        a, b, c = np.moveaxis(mesh.triangles[None]-points[start:start+32, None, None], 2, 0)
        la, lb, lc = (np.linalg.norm(v, axis=-1) for v in (a, b, c))
        numerator = np.einsum('...i,...i->...', a, np.cross(b, c))
        denominator = la*lb*lc + np.einsum('...i,...i->...', a, b)*lc + np.einsum('...i,...i->...', b, c)*la + np.einsum('...i,...i->...', c, a)*lb
        winding.extend(np.sum(2*np.arctan2(numerator, denominator), axis=1)/(4*np.pi))
    winding = np.asarray(winding)
    away = distances > 1e-10
    if np.any(away & (np.minimum(abs(winding), abs(abs(winding)-1)) > 1e-5)):
        raise ValueError('Ambiguous closed-mesh winding away from boundary')
    signed = distances*np.where(abs(winding) > .5, -1., 1.)
    signed[~away] = 0.
    return signed, closest, triangles, winding


def make_samples(row, bowl):
    """Interior/cavity witnesses, local finite differences and separate boundary points."""
    points = [[0., -.053, z] for z in (.001, .005, .010, -.001, .013)]
    labels = ['floor_material', 'center_cavity_low', 'center_cavity_high', 'below_bottom', 'above_rim']
    signs = [-1, 1, 1, 1, 1]
    for i in range(16):
        angle = 2*np.pi*(i+.37)/16
        for radius, label, sign in ((.017, 'wall_material', -1), (.010, 'side_cavity', 1), (.021, 'outside_wall', 1)):
            points.append([radius*np.cos(angle), 1.45*radius*np.sin(angle)-.053, .008])
            labels.append(label+'_'+str(i)); signs.append(sign)
    transform = np.asarray(row.get('natural_rest', {}).get('source_to_rest_transform', np.eye(4)), float)
    scale = np.asarray(row['dimension_profile']['linear_scale_xyz'], float)
    centers = trimesh.transform_points(np.asarray(points), transform)*scale
    pitch = float(np.ptp(bowl.vertices, axis=0).max())/int(row['spoon_split_bowl_collision_partition']['bowl_sdf_resolution'])
    h = CONFIG['finite_difference_step_in_pitches']*pitch
    all_points = list(centers); differences = []
    for center in centers:
        pairs = []
        for axis in range(3):
            delta = np.eye(3)[axis]*h
            pairs.append([len(all_points), len(all_points)+1]); all_points.extend((center+delta, center-delta))
        differences.append(pairs)
    boundary_first = len(all_points)
    for face in np.linspace(0, len(bowl.faces)-1, 24, dtype=int):
        center = bowl.triangles_center[face]; normal = bowl.face_normals[face]
        all_points.extend(center+k*h*normal for k in (-1, 0, 1))
    actual_points = np.asarray(all_points, np.float32)
    return actual_points, {'labels': labels, 'expected_signs': signs, 'witness_count': len(centers),
        'finite_difference_pairs': differences, 'boundary_first': boundary_first,
        'requested_pitch_m': pitch, 'requested_pitch_is_not_native_spacing_measurement': True}


def evaluate_layout(raw4, points, exact, normals, samples, *, distance_channel, gradient_channels):
    raw4, points, exact = np.asarray(raw4, float), np.asarray(points, float), np.asarray(exact, float)
    if raw4.shape != (len(points), 4) or not np.isfinite(raw4).all():
        raise ValueError('Complete finite native SDF raw4 required')
    count = samples['witness_count']; boundary = samples['boundary_first']; pitch = samples['requested_pitch_m']
    distances = raw4[:, distance_channel]; gradient = raw4[:, gradient_channels]
    limit = CONFIG['sdf_error_in_requested_pitches']*pitch
    fd = np.zeros((count, 3)); smooth = np.ones(count, dtype=bool)
    for i, pairs in enumerate(samples['finite_difference_pairs']):
        local_normals = [normals[i]]
        for axis, (plus, minus) in enumerate(pairs):
            fd[i, axis] = (distances[plus]-distances[minus])/(points[plus, axis]-points[minus, axis])
            local_normals.extend((normals[plus], normals[minus]))
        smooth[i] = np.max(np.linalg.norm(np.asarray(local_normals)-local_normals[0], axis=1)) <= CONFIG['smooth_normal_difference']
    signs = np.asarray(samples['expected_signs'])
    valid_witnesses = np.all(np.sign(exact[:count]) == signs) and np.all(abs(exact[:count]) > limit)
    sign_passed = valid_witnesses and np.all(signs*distances[:count] > 0)
    errors = abs(distances-exact)
    gradient_errors = np.linalg.norm(gradient[:count]-fd, axis=1)
    gradient_passed = bool(smooth.any() and np.max(gradient_errors[smooth]) <= CONFIG['gradient_fd_error'])
    return {'distance_channel': distance_channel, 'gradient_channels': list(gradient_channels),
        'passed': bool(sign_passed and errors.max() <= limit and gradient_passed),
        'original_mesh_witnesses_valid': bool(valid_witnesses), 'material_cavity_signs_passed': bool(sign_passed),
        'maximum_distance_error_m': float(errors.max()), 'distance_error_limit_m': limit,
        'maximum_witness_distance_error_m': float(errors[:count].max()),
        'maximum_near_boundary_distance_error_m': float(errors[boundary:].max()),
        'smooth_witnesses': int(smooth.sum()), 'finite_difference_gradient_passed': gradient_passed,
        'maximum_smooth_gradient_error': float(gradient_errors[smooth].max()) if smooth.any() else None,
        'per_point_distance_error_m': errors.tolist(), 'witness_signs_native': np.sign(distances[:count]).astype(int).tolist(),
        'finite_difference_gradients': fd.tolist(), 'smooth_witness_mask': smooth.tolist()}


def calibrate_layout(raw4, points, exact, normals, samples):
    """Explicitly identify one layout by independent metric/sign/gradient evidence.

    Preserve every candidate and the documented layout. No smallest-error
    selection, rescaling, sign flipping or fallback after an ambiguous result.
    """
    candidates = [evaluate_layout(raw4, points, exact, normals, samples,
        distance_channel=i, gradient_channels=[j for j in range(4) if j != i]) for i in range(4)]
    passing = [x for x in candidates if x['passed']]
    selected = passing[0] if len(passing) == 1 else None
    return {'passed': selected is not None, 'method': CONFIG['layout_calibration'],
        'candidate_layouts': candidates, 'passing_layout_count': len(passing),
        'selected_distance_channel': None if selected is None else selected['distance_channel'],
        'selected_gradient_channels': None if selected is None else selected['gradient_channels'],
        'matches_documentation': None if selected is None else selected['distance_channel'] == 0,
        'scale_fitted': False, 'sign_flipped': False, 'unqualified_layout_fallback': False}


def _mesh(stage, path):
    from pxr import UsdGeom
    mesh = UsdGeom.Mesh(stage.GetPrimAtPath(path))
    return trimesh.Trimesh(np.asarray(mesh.GetPointsAttr().Get()), np.asarray(mesh.GetFaceVertexIndicesAttr().Get()).reshape(-1, 3), process=False)


def read_native_sdf_fields(sim_view, paths, points, tensor, copy_array):
    """Use exact string views: installed native binding does not accept lists."""
    points = np.asarray(points, np.float32)
    if not paths or len(set(paths)) != len(paths) or points.ndim != 3 or points.shape[0] != len(paths) or points.shape[2] != 3 or not np.isfinite(points).all():
        raise ValueError('Unique shape paths and complete finite sample points required')
    fields = []
    for path, samples in zip(paths, points):
        view = sim_view.create_sdf_shape_view(path, len(samples))
        if view.count != 1 or list(view.object_paths) != [path] or view.max_num_points != len(samples) or not view.check():
            raise ValueError('Native SDF shape inventory differs: '+path)
        raw = copy_array(view.get_sdf_and_gradients(tensor(samples[None])))
        if raw.shape != (1, len(samples), 4) or not np.isfinite(raw).all():
            raise ValueError('Complete finite native SDF raw4 required: '+path)
        fields.append(np.array(raw[0], copy=True))
    return np.stack(fields)


def inspect(env, task, output):
    from pxr import UsdGeom
    from bench.evaluation.cloth.cloth_robot_collision import collect_colliders, state_digest, apply_transform
    from bench.simulation.isaacsim_backend.contact_sample import copy_array
    from bench.isaacsim.release import dump, digest
    before = state_digest(env.capture_state()); clock = (env.world.current_time_step_index, env.world.current_time)
    layers = {l.identifier: __import__('hashlib').sha256(l.ExportToString().encode()).hexdigest() for l in env.stage.GetUsedLayers()}
    rows = {r['id']: r for r in task['asset_metadata']['objects'] if r.get('spoon_split_bowl_collision_partition')}
    if not rows or any(r['collision_count'] != 5 or len(set(r['collider_paths'])) != 5 for r in rows.values()):
        raise ValueError('Explicit five-shape spoon inventory required')
    export = collect_colliders(env, body_paths={n: r['body_path'] for n, r in rows.items()})
    dump(output/'mixed_cooked_export.json', export)
    expected = {p for r in rows.values() for p in r['collider_paths']}
    bowls = {r['body_path']+'/Collision/spoon_bowl_sdf' for r in rows.values()}
    issues = {(r.get('path'), r['reason']) for r in export['issues']}
    expected_issues = {(p, 'unsupported_native_mesh_approximation') for p in bowls}
    checks = {'mixed_inventory': export['status'] == 'unchecked' and export['requested_body_paths'] == {n:r['body_path'] for n,r in rows.items()} and issues == expected_issues and len(export['issues']) == len(bowls)
        and set(x['path'] for x in export['colliders']) == expected
        and len(export['colliders']) == len(expected)
        and all(q['finished'] and {x['path'] for x in q['colliders']} == set(rows[q['body']]['collider_paths']) and len(q['colliders']) == 5 and all(x['valid'] for x in q['colliders']) for q in export['queries']),
        'original_colliders_inactive': all(not env.stage.GetPrimAtPath(r['body_path']+'/Collision/utensil').IsActive() for r in rows.values())}
    convex = []; actual_mass = {}
    for name, row in rows.items():
        view = env._object_views[name]
        mass = copy_array(view.get_masses()).reshape(-1); com = copy_array(view.get_coms())[0, :3]; inertia = copy_array(view.get_inertias()).reshape(3, 3)
        rotation = Rotation.from_quat(row['principal_axes_xyzw']).as_matrix()
        wanted = rotation@np.diag(row['diagonal_inertia_kg_m2'])@rotation.T
        ok = mass.shape == (1,) and np.allclose(mass, row['mass_kg'], rtol=2e-6, atol=1e-10) and np.allclose(com, row['center_of_mass_m'], rtol=0, atol=2e-8) and np.allclose(inertia, wanted, rtol=2e-4, atol=1e-10)
        actual_mass[name] = {'passed': bool(ok), 'mass_kg': mass.tolist(), 'COM_m': com.tolist(), 'inertia_matrix_kg_m2': inertia.tolist()}
    checks['native_explicit_mass_COM_inertia'] = all(x['passed'] for x in actual_mass.values())
    for record in export['colliders']:
        if record['path'] in bowls: continue
        ok = record['status'] == 'verified_cooking_representation' and len(record['pieces']) == 1 and record['approximation'] == 'convexHull'
        if not ok: raise ValueError('Unverified native handle convex: '+record['path'])
        matrix = np.asarray(record['mesh_to_body_column_matrix'])
        authored = apply_transform(_mesh(env.stage, record['path']).vertices, matrix)
        cooked = apply_transform(record['pieces'][0]['vertices'], matrix)
        error = max(float(cKDTree(authored).query(cooked)[0].max()), float(cKDTree(cooked).query(authored)[0].max()))
        a, c = trimesh.convex.convex_hull(authored), trimesh.convex.convex_hull(cooked)
        relative = abs(c.volume/a.volume-1.)
        convex.append({'path': record['path'], 'maximum_corner_set_error_body_m': error,
            'relative_volume_error': float(relative), 'passed': bool(error <= CONFIG['convex_corner_error_m'] and relative <= CONFIG['convex_relative_volume_error'])})
    checks['native_convex_geometry'] = len(convex) == 4*len(rows) and all(x['passed'] for x in convex)
    points, descriptions, meshes = [], [], []
    cache = UsdGeom.XformCache()
    for row in rows.values():
        relative = cache.GetLocalToWorldTransform(env.stage.GetPrimAtPath(row['body_path']+'/Collision/spoon_bowl_sdf')) * cache.GetLocalToWorldTransform(env.stage.GetPrimAtPath(row['body_path'])).GetInverse()
        if not np.allclose(np.asarray(relative), np.eye(4), rtol=0, atol=1e-12):
            raise ValueError('Bowl witnesses require verified identity mesh-to-body transform')
        mesh = _mesh(env.stage, row['body_path']+'/Collision/spoon_bowl_sdf')
        sample, desc = make_samples(row, mesh); points.append(sample); descriptions.append(desc); meshes.append(mesh)
    tensor_points = np.stack(points)
    paths = [r['body_path']+'/Collision/spoon_bowl_sdf' for r in rows.values()]
    raw = read_native_sdf_fields(env._sim_view, paths, tensor_points, env._tensor, copy_array)
    np.savez_compressed(output/'native_sdf_fields.npz', native_shape_local_points_m=tensor_points, native_sdf_raw4=raw)
    analyses = []; arrays = {}
    for i, path in enumerate(paths):
        mesh, desc = meshes[i], descriptions[i]
        exact, closest, triangles, winding = exact_mesh_distances(mesh, tensor_points[i])
        normals = mesh.face_normals[triangles]
        calibration = calibrate_layout(raw[i], tensor_points[i], exact, normals, desc)
        analyses.append({'path': paths[i], 'samples': desc,
            'documented_layout': calibration['candidate_layouts'][0], 'explicit_independent_layout_calibration': calibration})
        arrays['exact_distance_'+str(i)] = exact; arrays['exact_closest_point_'+str(i)] = closest
        arrays['exact_triangle_index_'+str(i)] = triangles; arrays['exact_winding_'+str(i)] = winding
    np.savez_compressed(output/'exact_sdf_reference.npz', **arrays)
    dump(output/'native_sdf_samples.json', analyses)
    checks['actual_SDF_unique_calibration_and_geometry'] = all(x['explicit_independent_layout_calibration']['passed'] for x in analyses)
    checks['SDF_layout_consistent_across_shapes'] = len({x['explicit_independent_layout_calibration']['selected_distance_channel'] for x in analyses}) == 1
    after_layers = {l.identifier: __import__('hashlib').sha256(l.ExportToString().encode()).hexdigest() for l in env.stage.GetUsedLayers()}
    checks['read_only_state_time_layers'] = before == state_digest(env.capture_state()) and clock == (env.world.current_time_step_index, env.world.current_time) and layers == after_layers
    return {'passed': all(checks.values()), 'checks': checks, 'configuration': CONFIG,
        'mass_properties': actual_mass, 'convex_geometry': convex, 'sdf_shape_paths': paths,
        'state_hash_before': before, 'state_hash_after': state_digest(env.capture_state()),
        'layers_before': layers, 'layers_after': after_layers,
        'native_sdf_fields_sha256': digest(output/'native_sdf_fields.npz'),
        'original_mixed_export_status_preserved': export['status'],
        'scope': 'Geometry/reset readback only; no force, stability, material or robot-manipulation qualification. SDF layout is explicitly calibrated by unique independent original-mesh signs/distances and finite-difference gradients. Every candidate including the documented layout is retained; ambiguous layouts fail, with no fitted scale or sign correction.'}


def run(args):
    from bench.isaacsim.release import task_inputs, dump, digest
    from scripts.maintenance.freeze_project_sources import verify_if_snapshot
    source = verify_if_snapshot(PROJECT, required_modules=('scripts.validation.kitchen_hybrid_geometry_probe',), entrypoint_file=Path(__file__))
    if source is None or Path.cwd().resolve() != PROJECT: raise ValueError('Use complete frozen source')
    output = args.output.resolve()
    if not output.is_relative_to(OUTPUT_ROOT): raise ValueError('Use shared evidence storage')
    output.mkdir(parents=True, exist_ok=False); env = None
    report = {'schema': 'kitchen_hybrid_native_geometry.v1', 'passed': False, 'physics_accepted': False,
        'source': source, 'source_manifest_sha256': digest(PROJECT/'source_snapshot_manifest.json'),
        'release_manifest_sha256': digest(args.release/'release_manifest.json'), 'force_corrections': False}
    try:
        _, scene, task, initial = task_inputs(args.release, args.task_id)
        dump(output/'task.json', task); dump(output/'scene.json', scene)
        from bench.simulation.isaacsim_backend.env import FR3DexterousIsaacEnv
        env = FR3DexterousIsaacEnv(scene, device='cuda:0', headless=True)
        env.reset(seed=0, initial_state=initial)
        dump(output/'runtime_metadata.json', env.metadata())
        if int(env._sim_view.device_ordinal) < 0: raise ValueError('Actual GPU native geometry required')
        report.update(inspect(env, task, output))
        report['source_final'] = verify_if_snapshot(PROJECT, required_modules=('scripts.validation.kitchen_hybrid_geometry_probe',), entrypoint_file=Path(__file__))
    except BaseException:
        report['exception'] = traceback.format_exc(); traceback.print_exc()
    finally:
        report['evidence_sha256'] = {p.name: digest(p) for p in output.iterdir() if p.is_file()}
        dump(output/'qualification.json', report)
        if env is not None:
            if not report['passed']:
                try: raise RuntimeError('Native hybrid geometry diagnostic did not pass; evidence preserved')
                finally: env.close()
            env.close()
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--release', type=Path, required=True); parser.add_argument('--task-id', required=True)
    parser.add_argument('--output', type=Path, required=True)
    raise SystemExit(run(parser.parse_args()))
