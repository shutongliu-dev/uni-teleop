"""Finite native SDF witnesses for the complete faceted L5 gear union."""
from itertools import permutations
import math
from pathlib import Path
import numpy as np

from scripts.assets.prepare_gear_sdf import SDF_PARAMETERS
from scripts.validation.gear_union_geometry import GEOMETRY, _polar, gear_asset

TOLERANCE_M = 20.e-6
ZERO_RESIDUAL_M = 1.e-6
PROJECTION_ITERATIONS = 8
LAYOUT_IDENTITY_INDICES = (0, 1, 2, 3, 5, 6, 7, 8)


def calibration_points(*, input_hub):
    """Known flat source faces, safely away from all edges and micro-ledges."""
    g = GEOMETRY
    centers = [[.030, 0., 0.], [.020, 0., .040]]
    normals = [[0., 0., -1.], [0., 0., 1.]]
    for angle in (math.pi/6, math.pi/2):
        if input_hub:
            centers.append([*_polar(.027, angle), .029])
            normals.append([math.cos(angle), math.sin(angle), 0.])
        else:
            index = round(angle/(2*math.pi/48))
            # Actual 48-gon edges lie between integer angle vertices.
            first, second = np.array([_polar(.025, a) for a in (index*2*math.pi/48, (index+1)*2*math.pi/48)])
            centers.append([*((first+second)/2), .029])
            normal = (first+second)/np.linalg.norm(first+second)
            normals.append([*normal, 0.])
    # Hub bore inner faces are outward toward the empty hole.
    a, b = [_polar(g['bore_radius_m']/math.cos(math.pi/48), v) for v in (0., 2*math.pi/48)]
    center = (a+b)/2
    centers.append([*center, .029])
    normals.append([*(-center/np.linalg.norm(center)), 0.])
    centers, normals = np.asarray(centers), np.asarray(normals)
    points = np.concatenate([centers-.0005*normals, centers+.0005*normals])
    expected = np.r_[np.full(len(centers), -.0005), np.full(len(centers), .0005)]
    return points, expected, np.vstack([normals, normals])


def calibration_source_audit(mesh, points, expected, *, input_hub):
    """Check the declared metric/sign convention against the actual source."""
    import trimesh
    triangles = mesh.triangles
    origins = triangles[:, 0]
    scales = np.max(np.linalg.norm(triangles-np.roll(triangles, 1, axis=1), axis=2), axis=1)
    normalized = (triangles-origins[:, None])/scales[:, None, None]
    distances = []
    for point in points:
        nearest = trimesh.triangles.closest_point(normalized, (point-origins)/scales[:, None])
        nearest = nearest*scales[:, None]+origins
        distances.append(np.linalg.norm(nearest-point, axis=1).min())
    inside = np.zeros(len(points), dtype=bool)
    for piece, _, _ in gear_asset('audit', input_hub=input_hub)._physical_pieces:
        normals = piece.face_normals
        offsets = np.einsum('ij,ij->i', normals, piece.triangles[:, 0])
        inside |= np.max(points @ normals.T-offsets, axis=1) <= 1.e-12
    actual = np.asarray(distances)*np.where(inside, -1., 1.)
    error = float(np.max(abs(actual-expected)))
    return {'passed': error <= 1.e-10, 'maximum_signed_distance_error_m': error,
        'source_signed_distances_m': actual.tolist(), 'source_inside': inside.tolist()}


def calibrate_layout(query, points, expected, normals, *, identity_indices=None):
    """Separate column identity from unchanged all-point metric thresholds.

    The declared eight top/bottom/external-hub plane witnesses may identify
    columns even if a bore witness fails precision. Every calibration witness
    still participates in the original 20 um and .05 accuracy gates.
    """
    native_query = query
    def query(p):
        raw = native_query(p)
        if raw.shape != (len(p), 4) or not np.isfinite(raw).all():
            raise ValueError('Every calibration query requires finite Nx4 raw data')
        return raw
    raw = query(points)
    if raw.shape != (len(points), 4) or not np.isfinite(raw).all():
        raise ValueError('Invalid native SDF calibration tensor')
    identity = np.arange(len(points)) if identity_indices is None else np.asarray(identity_indices)
    candidates = [i for i in range(4) if np.max(abs(raw[identity, i]-expected[identity])) <= TOLERANCE_M]
    if len(candidates) != 1:
        raise ValueError('Native signed-distance column is not uniquely calibrated')
    column = candidates[0]
    h = 1.e-5
    finite = np.column_stack([(query(points+np.eye(3)[i]*h)[:, column]
        -query(points-np.eye(3)[i]*h)[:, column])/(2*h) for i in range(3)])
    choices = []
    for order in permutations(i for i in range(4) if i != column):
        if np.max(abs(raw[:, order][identity]-normals[identity])) <= .05:
            choices.append(order)
    if len(choices) != 1:
        raise ValueError('Native XYZ gradient columns are not uniquely calibrated')
    order = choices[0]
    checks = {'all_distance_calibration_within_20um': bool(np.max(abs(raw[:, column]-expected)) <= TOLERANCE_M),
        'all_gradient_finite_difference_errors_within_005': bool(np.max(abs(raw[:, order]-finite)) <= .05),
        'all_gradient_source_normal_errors_within_005': bool(np.max(abs(raw[:, order]-normals)) <= .05)}
    return {'passed': all(checks.values()), 'checks': checks,
        'layout_identity_point_indices': identity.tolist(), 'metric_point_indices': list(range(len(points))),
        'distance_column': column, 'gradient_columns_xyz': list(order),
        'maximum_distance_calibration_error_m': float(np.max(abs(raw[:, column]-expected))),
        'maximum_gradient_finite_difference_error': float(np.max(abs(raw[:, order]-finite))),
        'maximum_gradient_source_normal_error': float(np.max(abs(raw[:, order]-normals))),
        'raw': raw, 'finite_difference_gradient': finite}


def project_all(query, points, column, gradient_columns):
    """Bounded read-only Newton queries; retain every point and every failure."""
    projected = np.array(points, dtype=float, copy=True)
    original = query(projected)
    failed = ~np.isfinite(original).all(axis=1)
    histories = [original.copy()]
    for _ in range(PROJECTION_ITERATIONS):
        raw = histories[-1]
        gradients = raw[:, gradient_columns]
        squared = np.einsum('ij,ij->i', gradients, gradients)
        active = abs(raw[:, column]) > ZERO_RESIDUAL_M
        failed |= active & ((squared < 1.e-12) | ~np.isfinite(squared))
        delta = raw[:, column, None]*gradients/np.maximum(squared[:, None], 1.e-30)
        failed |= active & (np.linalg.norm(delta, axis=1) > .0001)
        selected = active & ~failed
        projected[selected] -= delta[selected]
        current = query(projected)
        failed |= ~np.isfinite(current).all(axis=1)
        histories.append(current.copy())
    residual = np.abs(histories[-1][:, column])
    displacement = np.linalg.norm(projected-points, axis=1)
    qualified = (~failed & (abs(original[:, column]) <= TOLERANCE_M)
        & (residual <= ZERO_RESIDUAL_M) & (displacement <= TOLERANCE_M))
    return {'passed': bool(qualified.all()), 'projected': projected, 'original': original,
        'iterations_raw': np.stack(histories), 'failed': failed, 'qualified': qualified,
        'residual': residual, 'displacement': displacement}


def surface_points(mesh):
    weights = np.array([[1/3]*3, [2/3,1/6,1/6], [1/6,2/3,1/6], [1/6,1/6,2/3],
        [.9,.05,.05], [.05,.9,.05], [.05,.05,.9]])
    return np.vstack([np.einsum('wa,tac->twc', weights, mesh.triangles).reshape(-1, 3), mesh.vertices])


def bore_points(mesh):
    points = [[r*math.cos(a), r*math.sin(a), z]
        for r in (0., .006, .0105) for a in np.arange(16)*math.pi/8
        for z in np.linspace(-.0005, .0405, 25)]
    centers, normals = mesh.triangles_center, mesh.face_normals
    inward_radial = np.einsum('ij,ij->i', centers[:, :2], normals[:, :2]) < -.001
    points.extend(centers[inward_radial]+normals[inward_radial]*.00004)
    return np.asarray(points)


def check_gear_sdf(env, name, source_mesh, output, *, sdf_parameters=None):
    """All queries occur after reset, without stepping or authoring live state."""
    import torch
    import trimesh
    from pxr import UsdGeom, UsdPhysics
    from bench.simulation.isaacsim_backend.env import _np
    from scripts.validation.gear_sector_sdf_checks import QueryJournal
    from scripts.assets.prepare_gear_dense_sdf import DENSE_PARAMETERS
    expected_parameters = SDF_PARAMETERS if sdf_parameters is None else sdf_parameters
    if expected_parameters not in (SDF_PARAMETERS, DENSE_PARAMETERS):
        raise ValueError('Only the original sparse or single-variable dense SDF1024 contract is permitted')
    mesh = trimesh.load(source_mesh, force='mesh', process=False)
    info = env.scene_info['objects'][name]
    if len(info['collision_paths']) != 1:
        raise ValueError('Exactly one SDF collider per candidate gear is required')
    path = info['collision_paths'][0]
    prim = env.stage.GetPrimAtPath(path)
    authored = UsdGeom.Mesh(prim)
    if (UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr().Get() != 'sdf'
            or not np.array_equal(authored.GetPointsAttr().Get(), np.asarray(mesh.vertices, np.float32))
            or not np.array_equal(authored.GetFaceVertexIndicesAttr().Get(), mesh.faces.reshape(-1))
            or not np.array_equal(UsdGeom.Xformable(prim).GetLocalTransformation(), np.eye(4))):
        raise ValueError('Single SDF must retain the complete identity-scaled union surface')
    parameters = {k: prim.GetAttribute('physxSDFMeshCollision:'+k).Get() for k in SDF_PARAMETERS}
    points = surface_points(mesh)
    bore = bore_points(mesh)
    convention, expected, normals = calibration_points(input_hub=name == 'input_gear')
    source_audit = calibration_source_audit(mesh, convention, expected, input_hub=name == 'input_gear')
    if not source_audit['passed']:
        raise ValueError('Calibration points are not the declared source signed distances')
    capacity = max(len(points), len(bore))
    view = env._sim_view.create_sdf_shape_view(path, capacity)
    if view.count != 1 or list(view.object_paths) != [path]:
        raise ValueError('Native SDF shape inventory mismatch')
    def query(p):
        padded = np.repeat(np.asarray(p)[-1:], capacity, axis=0)
        padded[:len(p)] = p
        return _np(view.get_sdf_and_gradients(torch.as_tensor(padded[None], dtype=torch.float32,
            device=env.device)))[0][:len(p)].copy()
    # Save calibration even if convention/gradient decoding fails.
    output = Path(output)
    output.mkdir()
    query = QueryJournal(query, output/'query_journal')
    np.savez_compressed(output/'calibration_raw.npz', points=convention, expected=expected,
        expected_gradients=normals, raw=query(convention))
    # Raw surface and bore data survive even an undecodable layout, and are
    # distinct from successful projection/acceptance evidence.
    raw_bore = query(bore)
    np.savez_compressed(output/'uninterpreted_surface_and_bore.npz', source_points=points,
        source_raw=query(points), bore_points=bore, bore_raw=raw_bore)
    layout = calibrate_layout(query, convention, expected, normals,
        identity_indices=LAYOUT_IDENTITY_INDICES)
    projection = project_all(query, points, layout['distance_column'], layout['gradient_columns_xyz'])
    np.savez_compressed(output/'witnesses.npz', source_points=points, bore_points=bore,
        bore_raw=raw_bore, calibration_points=convention, calibration_expected=expected,
        calibration_raw=layout.pop('raw'), finite_difference_gradient=layout.pop('finite_difference_gradient'),
        **{k: v for k, v in projection.items() if k != 'passed'})
    source_distance = projection['original'][:, layout['distance_column']]
    checks = {'sdf_parameters_preserved': parameters == expected_parameters,
        'every_query_receipt_finite_Nx4': query.all_receipts_finite,
        'all_calibration_metric_thresholds': layout['passed'],
        'all_source_samples_and_zero_projections': projection['passed'],
        'bore_empty': bool(np.isfinite(raw_bore).all() and np.all(raw_bore[:, layout['distance_column']] > 0))}
    return {'passed': all(checks.values()), 'checks': checks, 'native_sdf_path': path,
        'source_mesh': str(source_mesh), 'source_triangles': len(mesh.faces), 'source_vertices': len(mesh.vertices),
        'surface_witness_count': len(points), 'bore_witness_count': len(bore),
        'query_receipt_count': query.index, 'invalid_query_receipt_indices': query.invalid_receipt_indices,
        'source_calibration_audit': source_audit, 'layout_calibration': layout, 'sdf_parameters_usd_readback': parameters,
        'maximum_source_sample_distance_m': float(np.max(abs(source_distance))) if np.isfinite(source_distance).all() else None,
        'maximum_zero_residual_m': float(np.max(projection['residual'])) if np.isfinite(projection['residual']).all() else None,
        'maximum_zero_displacement_from_source_m': float(np.max(projection['displacement'])) if np.isfinite(projection['displacement']).all() else None,
        'failed_projection_count': int(projection['failed'].sum()),
        'unqualified_witness_count': int((~projection['qualified']).sum()),
        'tolerance_m': TOLERANCE_M, 'zero_residual_tolerance_m': ZERO_RESIDUAL_M,
        'scope': 'Finite union-boundary witnesses only. Gradient projection need not identify a unique face at micro-ledges; displacement to the known source point bounds nearest-source error. No continuous Hausdorff, contact or force qualification.'}
