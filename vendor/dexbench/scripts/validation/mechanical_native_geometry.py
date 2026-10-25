"""Read live task collider geometry and mass properties without advancing physics.

Source meshes, native SDF samples and convex cooking are distinct evidence. No
visual mesh fallback or object setter is used. This module imports no simulator
until collect_task_geometry is called with an existing environment.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
import numpy as np


def inertia_checks(mass, inertia, com):
    tensor, pose = np.asarray(inertia, float), np.asarray(com, float)
    valid = (np.isfinite(mass) and mass > 0 and tensor.shape == (3, 3)
             and pose.shape == (7,) and np.isfinite(tensor).all() and np.isfinite(pose).all())
    if not valid:
        return {'passed': False, 'reason': 'missing_or_nonfinite_mass_properties'}
    eigen = np.linalg.eigvalsh((tensor+tensor.T)/2)
    checks = {'symmetric': bool(np.max(abs(tensor-tensor.T)) <= 1e-8),
              'positive_definite': bool(eigen.min() > 0),
              'principal_moment_triangle_inequality': bool(eigen.max() <= eigen.sum()-eigen.max()+1e-9),
              'normalized_principal_axes': bool(abs(np.linalg.norm(pose[3:])-1) <= 1e-5)}
    return {'passed': all(checks.values()), 'checks': checks,
            'principal_moments_kg_m2': eigen.tolist()}


def sdf_distance_column(values, expected_signs):
    """Resolve SDK layout from independent inside/outside witness points.

Installed SDK prose and historical interfaces disagree on component order.
Fail closed unless one endpoint column has the expected signs and unit gradients.
"""
    values = np.asarray(values, float)
    candidates = []
    for column in (0, 3):
        gradients = np.delete(values, column, axis=1)
        if (np.all(values[:, column]*np.asarray(expected_signs) > 1e-5)
                and np.all(abs(np.linalg.norm(gradients, axis=1)-1) < .15)):
            candidates.append(column)
    if len(candidates) != 1:
        raise ValueError('SDF component convention not independently resolved')
    return candidates[0]


def pitch_from_profiles(z, distances, scale):
    """Measure repeated SDF thread geometry; not a dynamic lead qualification."""
    from scipy.signal import find_peaks
    z, values = np.asarray(z, float), np.asarray(distances, float)
    periods = []
    for row in values:
        peaks, _ = find_peaks(row, prominence=.00005, distance=round(.0015/(z[1]-z[0])))
        periods.extend(np.diff(z[peaks]).tolist())
    periods = np.asarray(periods)*scale
    passed = len(periods) >= 12 and np.max(abs(periods-.003)) <= .00012
    return {'passed': bool(passed), 'measured_period_count': len(periods),
            'median_pitch_m': float(np.median(periods)) if len(periods) else None,
            'maximum_pitch_error_m': float(np.max(abs(periods-.003))) if len(periods) else None,
            'nominal_pitch_m': .003, 'tolerance_m': .00012,
            'scope': 'Axial spacing of live SDF extrema at 24 azimuths; not contact transmission accuracy.'}


def source_surface_errors(mesh, points):
    """Nearest triangles with unit-normalized triangle kernels and AABB pruning.

    trimesh's closest-point region tests use an absolute tolerance on quantities
    with units up to length**4. Millimetre triangles can therefore be wrongly
    classified as edges. Normalize EACH triangle before that kernel, then undo
    the scale on the returned coordinates. The original mesh is never changed.
    """
    import trimesh
    from scipy.spatial import cKDTree
    vertices, triangles = np.asarray(mesh.vertices), np.asarray(mesh.triangles)
    tree = cKDTree(vertices); low, high = triangles.min(1), triangles.max(1)
    distances = []
    for point in points:
        # The nearest vertex is an upper bound on nearest-surface distance.
        upper = tree.query(point)[0]
        bound = np.maximum(np.maximum(low-point, point-high), 0.)
        chosen = np.flatnonzero(np.linalg.norm(bound, axis=1) <= upper+1e-10)
        chosen_triangles = triangles[chosen]
        origins = chosen_triangles[:, 0]
        lengths = np.max(np.linalg.norm(chosen_triangles-np.roll(chosen_triangles, 1, axis=1), axis=2), axis=1)
        lengths = np.where(lengths > 0, lengths, 1.)
        normalized = (chosen_triangles-origins[:, None])/lengths[:, None, None]
        query = (point-origins)/lengths[:, None]
        nearest = trimesh.triangles.closest_point(normalized, query)*lengths[:, None]+origins
        distances.append(float(np.min(np.linalg.norm(nearest-point, axis=1))))
    return np.asarray(distances)


def zero_surface_check(z, radius, scale):
    """Compare entire sampled zero surfaces one nominal pitch apart."""
    shift = round(.002/(z[1]-z[0]))
    difference = np.abs(radius[:, shift:]-radius[:, :-shift])*scale
    return {'passed': bool(np.max(difference) <= 50e-6),
            'maximum_one_pitch_surface_difference_m': float(np.max(difference)),
            'rms_one_pitch_surface_difference_m': float(np.sqrt(np.mean(difference**2))),
            'all_sampled_azimuths_and_axial_pairs_used': True,
            'point_pairs': int(difference.size), 'nominal_pitch_m': .003,
            'surface_tolerance_m': 50e-6,
            'scope': 'Finite native SDF zero-surface witnesses, including every sampled pair; no extrema pruning.'}


def collect_task_geometry(env, output):
    from pxr import Usd, UsdGeom, UsdPhysics, UsdUtils, PhysicsSchemaTools
    from omni.physx import get_physx_cooking_interface, get_physx_property_query_interface
    from omni.physx.bindings._physx import (PhysxCollisionRepresentationResult,
        PhysxPropertyQueryResult, PhysxPropertyQueryMode)
    from bench.simulation.isaacsim_backend.env import _np
    from bench.evaluation.cloth.cloth_robot_collision import (state_digest,
        verify_native_pose, cooked_piece_geometry, piece_in_body)
    import torch

    output = Path(output)
    stage, cache = env.stage, UsdGeom.XformCache()
    before = state_digest(env.capture_state())
    layers_before = {l.identifier: hashlib.sha256(l.ExportToString().encode()).hexdigest()
                     for l in stage.GetUsedLayers()}
    stage_id = UsdUtils.StageCache.Get().GetId(stage).ToLongInt()
    query, cooking = get_physx_property_query_interface(), get_physx_cooking_interface()
    bodies, colliders, issues = {}, [], []
    selected = {name: env.scene_info[group][name] for name, group in
                [('screwdriver', 'objects'), ('bolt', 'objects'), ('threaded_fixture', 'fixtures')]}
    for name, info in selected.items():
        path = info['path']; found, rb, finished = [], [], []
        def collider_callback(r):
            found.append({'valid': r.result == PhysxPropertyQueryResult.VALID,
                'path': str(PhysicsSchemaTools.intToSdfPath(r.path_id)), 'stage_id': int(r.stage_id),
                'local_pos': list(r.local_pos), 'local_rot_xyzw': list(r.local_rot),
                'aabb_min': list(r.aabb_local_min), 'aabb_max': list(r.aabb_local_max), 'volume': float(r.volume)})
        def body_callback(r):
            rb.append({'valid': r.result == PhysxPropertyQueryResult.VALID, 'mass_kg': float(r.mass),
                'com_m': list(r.center_of_mass), 'principal_inertia_kg_m2': list(r.inertia),
                'principal_axes_xyzw': list(r.principal_axes)})
        query.query_prim(stage_id=stage_id, prim_id=PhysicsSchemaTools.sdfPathToInt(path),
            query_mode=PhysxPropertyQueryMode.QUERY_RIGID_BODY_WITH_COLLIDERS, timeout_ms=-1,
            finished_fn=lambda: finished.append(True), rigid_body_fn=body_callback, collider_fn=collider_callback)
        view = env._sim_view.create_rigid_body_view(path)
        if view.count != 1:
            raise ValueError('Expected one body: '+path)
        mass = float(_np(view.get_masses()).reshape(-1)[0])
        inertia = _np(view.get_inertias()).reshape(3, 3).T
        com = _np(view.get_coms()).reshape(7)
        prim = stage.GetPrimAtPath(path); api = UsdPhysics.MassAPI(prim)
        authored = {'mass_kg': api.GetMassAttr().Get(), 'center_of_mass': str(api.GetCenterOfMassAttr().Get()),
                    'diagonal_inertia': str(api.GetDiagonalInertiaAttr().Get()),
                    'principal_axes': str(api.GetPrincipalAxesAttr().Get())}
        bodies[name] = {'path': path, 'body_mode': info['body_mode'], 'mass_kg': mass,
            'inertia_about_com_in_body_frame_kg_m2': inertia.tolist(), 'com_principal_frame_xyzw': com.tolist(),
            'checks': inertia_checks(mass, inertia, com), 'usd_mass_attributes': authored,
            'property_query': rb, 'property_query_finished': bool(finished),
            'kinematic_dynamic_inertia_qualified': info['body_mode'] != 'kinematic_fixture'}
        if not bodies[name]['checks']['passed'] or len(rb) != 1 or not rb[0]['valid'] or not finished:
            issues.append({'body': name, 'reason': 'mass_or_property_query_invalid'})
        if len(rb) == 1:
            agreement = {'mass': bool(np.isclose(mass, rb[0]['mass_kg'], atol=1e-7, rtol=0)),
                'com': bool(np.allclose(com[:3], rb[0]['com_m'], atol=1e-7, rtol=0)),
                'principal_moments': bool(np.allclose(np.linalg.eigvalsh((inertia+inertia.T)/2),
                    sorted(rb[0]['principal_inertia_kg_m2']), atol=1e-9, rtol=1e-5))}
            bodies[name]['tensor_property_query_agreement'] = agreement
            if not all(agreement.values()):
                issues.append({'body': name, 'reason': 'independent_native_mass_queries_disagree'})
        native = {r['path']: r for r in found}
        active = []; disabled = []
        for p in Usd.PrimRange(prim):
            if not p.HasAPI(UsdPhysics.CollisionAPI):
                continue
            ppath = str(p.GetPath())
            if UsdPhysics.CollisionAPI(p).GetCollisionEnabledAttr().Get() is False:
                disabled.append(ppath); continue
            active.append(ppath)
            row = {'body': name, 'path': ppath, 'status': 'unchecked'}; colliders.append(row)
            if ppath not in native or not native[ppath]['valid']:
                issues.append({'path': ppath, 'reason': 'native_collider_not_found'}); continue
            points = np.asarray(UsdGeom.Mesh(p).GetPointsAttr().Get(), float)
            # Native rigid poses have no scale. Preserve an inherited 1.5 USD
            # scale when expressing mesh vertices in the physical body frame.
            body_matrix = np.asarray(cache.GetLocalToWorldTransform(prim)).T.copy()
            body_matrix[:3, :3] /= np.linalg.norm(body_matrix[:3, :3], axis=0)
            matrix = np.linalg.inv(body_matrix) @ np.asarray(cache.GetLocalToWorldTransform(p)).T
            approx = str(UsdPhysics.MeshCollisionAPI(p).GetApproximationAttr().Get())
            row.update(approximation=approx, native=native[ppath], mesh_to_body_column_matrix=matrix.tolist(),
                source_points_sha256=hashlib.sha256(points.tobytes()).hexdigest(), source_vertex_count=len(points),
                source_bounds=[points.min(0).tolist(), points.max(0).tolist()],
                native_pose=verify_native_pose(matrix, native[ppath]),
                effective_collision_attributes={a.GetName(): str(a.Get()) for a in p.GetAttributes()
                                                if a.GetName().startswith(('physics:', 'physx'))})
            if approx in ('convexHull', 'convexDecomposition'):
                pieces, received = [], []
                def cooked_callback(result, convexes):
                    received.append(result)
                    if result == PhysxCollisionRepresentationResult.RESULT_VALID:
                        for c in convexes:
                            pieces.append({'vertices': [list(v) for v in c.vertices],
                                'indices': [int(i) for i in c.indices],
                                'polygons': [{'index_base': int(v.index_base), 'num_vertices': int(v.num_vertices)} for v in c.polygons],
                                'planes': [list(v.plane) for v in c.polygons]})
                cooking.request_convex_collision_representation(stage_id=stage_id,
                    collision_prim_id=PhysicsSchemaTools.sdfPathToInt(p.GetPath()),
                    run_asynchronously=False, on_result=cooked_callback)
                if received != [PhysxCollisionRepresentationResult.RESULT_VALID] or not pieces:
                    issues.append({'path': ppath, 'reason': 'convex_cooking_invalid'}); continue
                vertices = np.concatenate([np.asarray(v['vertices']) for v in pieces])
                error = float(np.max(abs(np.stack([vertices.min(0), vertices.max(0)])-
                                         np.array([native[ppath]['aabb_min'], native[ppath]['aabb_max']]))))
                row.update(pieces=[cooked_piece_geometry(v) for v in pieces], native_aabb_error_m=error)
                for piece in row['pieces']:
                    piece['native_topology_counts'] = {'vertices': len(piece['vertices']),
                        'polygons': len(piece['polygons']),
                        'maximum_vertices_per_polygon': max(p['num_vertices'] for p in piece['polygons'])}
                    piece_in_body(piece, matrix, planes_key='support_hull_planes')
                if error > 1e-5:
                    issues.append({'path': ppath, 'reason': 'cooked_native_bounds_disagree'}); continue
                if approx == 'convexHull' and len(pieces) == 1:
                    from scripts.validation.mechanical_convex_geometry import convex_surface_error
                    # Include inherited body scale, unlike a mesh-parent-only
                    # comparison: these tolerances are physical metres.
                    boundary_error = convex_surface_error(points @ matrix[:3, :3].T,
                        vertices @ matrix[:3, :3].T)
                    topology = row['pieces'][0]['native_topology_counts']
                    envelope = (topology['vertices'] <= 64 and topology['polygons'] <= 64
                                and topology['maximum_vertices_per_polygon'] <= 32)
                    row['source_cooked_boundary'] = {
                        'passed': bool(boundary_error <= 20e-6 and envelope),
                        'maximum_surface_halfspace_error_m': boundary_error,
                        'tolerance_m': 20e-6, 'gpu_convex_topology_envelope': envelope,
                        'coordinate_frame': 'physical body frame; inherited scales applied once',
                        'scope': 'Necessary surface/topology checks; GPU fallback warnings must also be absent.'}
                    if not row['source_cooked_boundary']['passed']:
                        issues.append({'path': ppath, 'reason': 'convex_boundary_or_topology_not_verified'})
                else:
                    issues.append({'path': ppath, 'reason': 'unsupported_compound_source_boundary_comparison'})
                row['status'] = 'verified_convex_cooking'
            elif approx == 'sdf':
                is_bolt = name == 'bolt'; is_thread = p.GetName() in ('collisions_thread', 'thread_insert')
                # Analytical witnesses lie well inside/outside the known central shaft/ring.
                z0 = .012 if is_bolt else .032
                witnesses = np.array([[0., 0., z0], [.011, 0., z0], [0., .011, z0]])
                signs = [-1, 1, 1] if is_bolt else [1, -1, -1]
                theta = np.linspace(0, 2*np.pi, 24, endpoint=False)
                z = np.linspace(.005, .023, 721) if is_bolt else np.linspace(.027, .036, 361)
                radius = .0075 if is_thread else .0098
                grid = np.stack(np.broadcast_arrays(radius*np.cos(theta)[:, None],
                    radius*np.sin(theta)[:, None], z[None, :]), axis=-1).reshape(-1, 3)
                sample = np.concatenate([witnesses, grid])
                sdf = env._sim_view.create_sdf_shape_view(ppath, len(sample))
                if sdf.count != 1 or list(sdf.object_paths) != [ppath]:
                    raise ValueError('SDF view identity mismatch')
                raw = _np(sdf.get_sdf_and_gradients(torch.as_tensor(sample[None], dtype=torch.float32,
                                                                          device=env.device)))[0]
                column = sdf_distance_column(raw[:3], signs)
                filename = name+'_'+p.GetName()+'_sdf.npz'
                np.savez_compressed(output/filename, points_prim_local=sample, native_values=raw,
                                    z_prim_local=z, theta_rad=theta, distance_column=column)
                row.update(status='verified_native_sdf_samples', sdf_sample_count=len(sample),
                    sdf_distance_column=column, sdf_witness_points=witnesses.tolist(),
                    sdf_witness_results=raw[:3].tolist(), samples_file=filename,
                    samples_sha256=hashlib.sha256((output/filename).read_bytes()).hexdigest(),
                    sdf_shape_paths=list(sdf.object_paths))
                if is_thread:
                    import trimesh
                    scale = row['native_pose']['mesh_scale'][2]
                    # Independent planar witnesses determine whether this API
                    # expects unscaled mesh coordinates, physical shape-local
                    # coordinates or a second inverse scale. Native/USD pose
                    # agreement above excludes an unrecorded collider offset.
                    plane_z = .030 if is_bolt else .032
                    top_z = float(points[:, 2].max())
                    plane_points = np.array([[.011, 0, plane_z], [.012, 0, plane_z],
                        [.013, 0, plane_z], [.011, 0, top_z-.0005],
                        [.011, 0, top_z], [.011, 0, top_z+.0005]])
                    expected_distances = np.array([-.001, 0, .001, -.0005, 0, .0005])
                    calibration = env._sim_view.create_sdf_shape_view(ppath, len(plane_points))
                    coordinate_trials = {}
                    for label, factor in [('mesh_local', 1.), ('scaled_shape_local', scale),
                                          ('inverse_scaled_local', 1./scale)]:
                        trial = _np(calibration.get_sdf_and_gradients(torch.as_tensor(
                            (plane_points*factor)[None], dtype=torch.float32, device=env.device)))[0]
                        coordinate_trials[label] = {'query_points': (plane_points*factor).tolist(),
                            'raw_sdf_results': trial.tolist(), 'input_scale': factor,
                            'maximum_expected_plane_distance_error_m': float(np.max(abs(trial[:, column]-expected_distances))*scale)}
                    row['sdf_coordinate_calibration'] = {'passed': coordinate_trials['mesh_local']['maximum_expected_plane_distance_error_m'] <= 50e-6,
                        'expected_mesh_local_signed_distances_m': expected_distances.tolist(),
                        'trials': coordinate_trials, 'physical_scale_applied_to_error_once': scale,
                        'scope': 'Original flat outer hex side and flat top witnesses; no coordinate fitting or source modification.'}
                    if not row['sdf_coordinate_calibration']['passed']:
                        issues.append({'path': ppath, 'reason': 'sdf_query_coordinate_calibration_failed'})
                    row['fixed_radius_extrema_diagnostic'] = pitch_from_profiles(z,
                        raw[3:, column].reshape(len(theta), len(z)), scale)
                    # Sparse-SDF far-field values can be coarser than the actual
                    # collision boundary. Independently probe the zero surface;
                    # retain the full fixed-radius diagnostic, including failures.
                    direction = np.c_[np.cos(np.repeat(theta, len(z))), np.sin(np.repeat(theta, len(z)))]
                    lower, upper = np.full(len(grid), .006), np.full(len(grid), .0086)
                    def query_radii(rad):
                        values = np.c_[direction*rad[:, None], np.tile(z, len(theta))]
                        batch = np.concatenate([witnesses, values])
                        return _np(sdf.get_sdf_and_gradients(torch.as_tensor(batch[None],
                            dtype=torch.float32, device=env.device)))[0, 3:, column]
                    low_values, high_values = query_radii(lower), query_radii(upper)
                    outward = 1 if is_bolt else -1
                    if not (np.all(low_values*outward < 0) and np.all(high_values*outward > 0)):
                        raise ValueError('Native SDF radial zero surface is not bracketed')
                    for _ in range(12):
                        middle = (lower+upper)/2; values = query_radii(middle)
                        inside = values*outward < 0
                        lower = np.where(inside, middle, lower); upper = np.where(inside, upper, middle)
                    radii = (lower+upper)/2
                    surface = np.c_[direction*radii[:, None], np.tile(z, len(theta))]
                    zero_file = filename.replace('_sdf.npz', '_zero_surface.npz')
                    np.savez_compressed(output/zero_file, points_prim_local=surface,
                        radii_prim_local=radii.reshape(len(theta), len(z)), z_prim_local=z, theta_rad=theta,
                        bracketing_width_m=(upper-lower)*scale)
                    row['zero_surface_file'] = zero_file
                    row['zero_surface_sha256'] = hashlib.sha256((output/zero_file).read_bytes()).hexdigest()
                    row['zero_surface_periodicity'] = zero_surface_check(z, radii.reshape(len(theta), len(z)), scale)
                    counts = np.asarray(UsdGeom.Mesh(p).GetFaceVertexCountsAttr().Get())
                    faces = np.asarray(UsdGeom.Mesh(p).GetFaceVertexIndicesAttr().Get())
                    if not np.all(counts == 3):
                        raise ValueError('Independent nearest-source check requires triangle mesh')
                    mesh = trimesh.Trimesh(vertices=points, faces=faces.reshape(-1, 3), process=False)
                    witness_ids = np.unique(np.linspace(0, len(surface)-1, 192, dtype=int))
                    source_errors = source_surface_errors(mesh, surface[witness_ids])*scale
                    row['zero_surface_source_comparison'] = {'passed': bool(source_errors.max() <= 50e-6),
                        'distance_algorithm': 'per_triangle_unit_normalized_closest_point_with_conservative_aabb_pruning_v2',
                        'sample_indices': witness_ids.tolist(), 'absolute_errors_m': source_errors.tolist(),
                        'maximum_error_m': float(source_errors.max()), 'surface_tolerance_m': 50e-6,
                        'source_watertight': bool(mesh.is_watertight), 'source_winding_consistent': bool(mesh.is_winding_consistent),
                        'scope': '192 deterministic live zero-surface witnesses versus exact nearest source triangles; not all surface points.'}
                    if not row['zero_surface_periodicity']['passed'] or not row['zero_surface_source_comparison']['passed']:
                        issues.append({'path': ppath, 'reason': 'native_sdf_zero_surface_not_verified'})
                    # The pre-engaged reset contacts the bolt's lower end and
                    # fixture's upper end. Mid-thread periodicity alone does
                    # not cover these run-in/run-out regions. Compare their
                    # complete fixed grid against source distance; do not
                    # require a deliberately terminated thread to be periodic.
                    engaged_z = (np.linspace(.0005, .0065, 241) if is_bolt
                                 else np.linspace(.032, .03775, 231))
                    engaged_direction = np.c_[np.cos(np.repeat(theta, len(engaged_z))),
                                               np.sin(np.repeat(theta, len(engaged_z)))]
                    engaged_count = len(engaged_direction)
                    engaged_view = env._sim_view.create_sdf_shape_view(ppath, engaged_count)
                    def engaged_query(rad):
                        positions = np.c_[engaged_direction*rad[:, None], np.tile(engaged_z, len(theta))]
                        return _np(engaged_view.get_sdf_and_gradients(torch.as_tensor(
                            positions[None], dtype=torch.float32, device=env.device)))[0, :, column]
                    lo, hi = np.full(engaged_count, .004), np.full(engaged_count, .009)
                    if not (np.all(engaged_query(lo)*outward < 0)
                            and np.all(engaged_query(hi)*outward > 0)):
                        raise ValueError('Engaged end-region native SDF radial zero is not bracketed')
                    for _ in range(14):
                        mid = (lo+hi)/2; inside = engaged_query(mid)*outward < 0
                        lo = np.where(inside, mid, lo); hi = np.where(inside, hi, mid)
                    engaged_radius = (lo+hi)/2
                    engaged_points = np.c_[engaged_direction*engaged_radius[:, None],
                                           np.tile(engaged_z, len(theta))]
                    engaged_file = filename.replace('_sdf.npz', '_engaged_zero_surface.npz')
                    np.savez_compressed(output/engaged_file, points_prim_local=engaged_points,
                        radii_prim_local=engaged_radius.reshape(len(theta), len(engaged_z)),
                        z_prim_local=engaged_z, theta_rad=theta, bracketing_width_m=(hi-lo)*scale)
                    engaged_ids = np.unique(np.linspace(0, engaged_count-1, 384, dtype=int))
                    engaged_errors = source_surface_errors(mesh, engaged_points[engaged_ids])*scale
                    row['engaged_end_source_comparison'] = {
                        'passed': bool(engaged_errors.max() <= 50e-6),
                        'source_axial_range_prim_m': [float(engaged_z[0]), float(engaged_z[-1])],
                        'sample_indices': engaged_ids.tolist(), 'absolute_errors_m': engaged_errors.tolist(),
                        'maximum_error_m': float(engaged_errors.max()), 'surface_tolerance_m': 50e-6,
                        'zero_surface_file': engaged_file,
                        'zero_surface_sha256': hashlib.sha256((output/engaged_file).read_bytes()).hexdigest(),
                        'source_mesh_unchanged': True,
                        'scope': '384 deterministic native zero-surface witnesses across the actual pre-engaged end region; no periodicity assumed at thread termination.'}
                    if not row['engaged_end_source_comparison']['passed']:
                        issues.append({'path': ppath, 'reason': 'native_engaged_sdf_surface_not_verified'})
            else:
                issues.append({'path': ppath, 'reason': 'unsupported_approximation'})
        bodies[name].update(active_colliders=active, disabled_colliders=disabled,
                            native_property_colliders=sorted(native))
        # Property queries retain authored disabled shapes (as material tensors
        # do). Their presence is not proof of simulation/query participation.
        # native_task_parameters separately performs the enclosing scene query.
        bodies[name]['disabled_shapes_retained_by_property_query'] = sorted(set(native)&set(disabled))
        bodies[name]['property_query_is_simulation_shape_flag_readback'] = False
        if set(active)-set(native) or set(native)-set(active)-set(disabled):
            issues.append({'body': name, 'reason': 'native_active_collider_inventory_mismatch',
                           'native_only': sorted(set(native)-set(active)), 'source_only': sorted(set(active)-set(native))})
    after = state_digest(env.capture_state())
    layers_after = {l.identifier: hashlib.sha256(l.ExportToString().encode()).hexdigest()
                    for l in stage.GetUsedLayers()}
    if before != after or layers_before != layers_after:
        issues.append({'reason': 'read_only_export_changed_state_or_stage'})
    return {'schema': 'mechanical_native_geometry.v1', 'passed': not issues,
            'bodies': bodies, 'colliders': colliders, 'issues': issues,
            'physical_state_unchanged': before == after, 'stage_layers_unchanged': layers_before == layers_after,
            'scope': 'Live shape queries, convex representations, SDF samples and physical inertia sanity. '
                     'Kinematic fixture inertia is recorded only; SDF pitch does not prove dynamic thread lead.',
            'object_state_writes': 0, 'simulation_steps': 0}
