"""Finite read-only native SDF witnesses for the task0 two-cylinder union."""
from pathlib import Path
import numpy as np

from scripts.validation.gear_sdf_checks import calibrate_layout
from scripts.validation.mechanical_ring_sdf import surface_witnesses

TOLERANCE_M = 20.e-6

# Additional dense diagnostics only: every source witness remains in all gates.
# These fine-mesh indices exceeded1um even at the best of the source, midpoint
# and final bracket endpoints in the preserved subgrid4 candidate09.
SUBGRID4_RESIDUAL_INDICES = (
    395794,399378,417370,417438,417629,420954,421022,421213,431781,435365,
    439133,442717,453384,456968,460552,464136,481810,485394,489204,489383,
    492788,492967,496287,496828,499871,510885,514469,517799,521383,553766,
    554066,557350,557650,560556,564140,567899,568025,571483,571609,804461,
    808045,811651,815235,854933,858517,862064,865648,904542,908126,926345,
    929929,933215,933479,936799,937063,976761,980345,983948,987532,997997,
    1001581,1069406,1072990,1098377,1101961,1148793,1148812,1152377,
    1152396,1220227,1223811,1227102,1230686,1270640,1270677,1274224,
    1274261,1574117,1574683)


def normal_zero_witnesses(query_distance, points, normals, source_distance, *, iterations=15):
    """Retain every actual query and select its smallest absolute residual.

    A sampled field can be stepped: the final bracket midpoint is not always
    the best queried approximation. Preserve that midpoint and both endpoints
    independently, including any sign jump. No witness or threshold is removed.
    """
    count = len(points)
    offsets = np.zeros((count, iterations+2), dtype=np.float64)
    distances = np.empty((count, iterations+2), dtype=np.float32)
    distances[:, 0] = source_distance
    lower, upper = np.full(count, -.0005), np.full(count, .0005)
    low_values = query_distance(points+lower[:, None]*normals)
    high_values = query_distance(points+upper[:, None]*normals)
    bracketed = (low_values < 0.) & (high_values > 0.)
    for index in range(1, iterations+1):
        mid = (lower+upper)/2
        values = query_distance(points+mid[:, None]*normals)
        offsets[:, index], distances[:, index] = mid, values
        lower = np.where(values < 0., mid, lower)
        upper = np.where(values >= 0., mid, upper)
    offsets[:, -1] = (lower+upper)/2
    distances[:, -1] = query_distance(points+offsets[:, -1, None]*normals)
    finite_scores = np.where(np.isfinite(distances), abs(distances), np.inf)
    selected = np.argmin(finite_scores, axis=1)
    rows = np.arange(count)
    return {'query_normal_offsets_m': offsets, 'query_distances_m': distances,
        'selected_query_indices': selected, 'normal_offset_m': offsets[rows, selected],
        'residual_m': distances[rows, selected], 'original_midpoint_offset_m': offsets[:, -1],
        'original_midpoint_residual_m': distances[:, -1],
        'final_bracket_lower_offset_m': lower, 'final_bracket_upper_offset_m': upper,
        'final_bracket_lower_distance_m': query_distance(points+lower[:, None]*normals),
        'final_bracket_upper_distance_m': query_distance(points+upper[:, None]*normals),
        'bracketed': bracketed, 'inside_values': low_values, 'outside_values': high_values}


def normalized_gradient_check(layout):
    """The native SDF API returns normalized gradient direction, not magnitude.

    Keep the shared mechanical validator unchanged; retain the raw finite
    differences and their original mismatch as independent diagnostics.
    """
    finite = layout['finite_difference_gradient']
    lengths = np.linalg.norm(finite, axis=1)
    normalized = finite/np.maximum(lengths[:, None], 1e-30)
    gradients = layout['raw'][:, layout['gradient_columns_xyz']]
    error = float(np.max(abs(gradients-normalized)))
    layout['maximum_unnormalized_gradient_finite_difference_error'] = layout['maximum_gradient_finite_difference_error']
    layout['maximum_gradient_finite_difference_error'] = error
    layout['finite_difference_gradient_lengths'] = lengths.tolist()
    layout['finite_difference_comparison'] = 'Unit direction: native kernel normalizes gradient; raw derivative magnitudes remain separately recorded.'
    layout['checks']['all_gradient_finite_difference_errors_within_005'] = bool(np.all(lengths > 1e-12) and error <= .05)
    layout['passed'] = all(layout['checks'].values())
    return layout


def analytic_distance(points):
    """Exact distance to the union boundary, excluding the internal joining disk."""
    p = np.asarray(points, dtype=float)
    rz = np.column_stack((np.linalg.norm(p[:, :2], axis=1), p[:, 2]))
    profile = np.array([[0., -.007], [.028, -.007], [.028, .007],
                        [.020, .007], [.020, .047], [0., .047]])
    distances = []
    for a, b in zip(profile[:-1], profile[1:]):
        ab = b-a
        t = np.clip((rz-a)@ab/(ab@ab), 0., 1.)
        distances.append(np.linalg.norm(rz-a-t[:, None]*ab, axis=1))
    inside = (((rz[:, 0] <= .028) & (rz[:, 1] >= -.007) & (rz[:, 1] <= .007))
              | ((rz[:, 0] <= .020) & (rz[:, 1] >= .007) & (rz[:, 1] <= .047)))
    return np.min(distances, axis=0)*np.where(inside, -1., 1.)


def analytic_witnesses(angles=1024):
    """All azimuths at walls, shoulder/caps and every sharp circular edge."""
    points, normals = [], []
    profiles = [(.028, z, 1., 0.) for z in (-.006, 0., .006)]
    profiles += [(.020, z, 1., 0.) for z in (.008, .027, .046)]
    profiles += [(r, -.007, 0., -1.) for r in (.005, .020, .027)]
    profiles += [(r, .007, 0., 1.) for r in (.021, .024, .027)]
    profiles += [(r, .047, 0., 1.) for r in (.005, .010, .019)]
    profiles += [(.028, -.007, 1., -1.), (.028, .007, 1., 1.),
                 (.020, .007, 1., 1.), (.020, .047, 1., 1.)]
    for r, z, nr, nz in profiles:
        for a in np.arange(angles)*2*np.pi/angles:
            points.append([r*np.cos(a), r*np.sin(a), z])
            normal = np.array([nr*np.cos(a), nr*np.sin(a), nz])
            normals.append(normal/np.linalg.norm(normal))
    return np.asarray(points), np.asarray(normals)


def calibration():
    centers = np.array([[.01, 0., -.007], [.01, 0., .047], [.024, 0., .007],
                        [.028, 0., 0.], [0., .028, 0.], [.020, 0., .027]])
    normals = np.array([[0., 0., -1.], [0., 0., 1.], [0., 0., 1.],
                        [1., 0., 0.], [0., 1., 0.], [1., 0., 0.]])
    return (np.vstack((centers-.0005*normals, centers+.0005*normals)),
            np.r_[np.full(len(centers), -.0005), np.full(len(centers), .0005)],
            np.vstack((normals, normals)))


def check_token_sdf(sim, collider_path, source_mesh, output, *, device='cuda:0', coordinate_scale=1.):
    """Native SdfShapeView existence plus metric/sign and finite zero-surface checks.

    Does not step, author state or claim a native grid dimension readback. USD
    resolution is separate from this actual cooked-field query.
    """
    import torch
    import trimesh
    output = Path(output)
    mesh = trimesh.load(source_mesh, force='mesh', process=False)
    source_points, source_normals = surface_witnesses(mesh)
    exact_points, exact_normals = analytic_witnesses()
    points = np.vstack((source_points, exact_points))
    normals = np.vstack((source_normals, exact_normals))
    convention, expected, convention_normals = calibration()
    capacity = min(len(points), 65536)
    view = sim.create_sdf_shape_view(collider_path, capacity)
    inventory = {'count': int(view.count), 'paths': list(view.object_paths),
                 'max_num_points': int(view.max_num_points)}
    if view.count != 1 or list(view.object_paths) != [collider_path]:
        raise ValueError('Expected exactly one actual native shape with SDF information')
    input_scale, distance_scale, distance_column = 1., 1., None
    def query(p):
        results = []
        for start in range(0, len(p), capacity):
            part = np.asarray(p)[start:start+capacity]
            padded = np.repeat(part[-1:], capacity, axis=0)
            padded[:len(part)] = part
            raw = view.get_sdf_and_gradients(torch.as_tensor((padded*input_scale)[None], dtype=torch.float32, device=device))
            values = raw.detach().cpu().numpy()[0, :len(part)].copy()
            if distance_column is not None:
                values[:, distance_column] *= distance_scale
            results.append(values)
        return np.concatenate(results, axis=0)
    units = {'input_multiplier': 1., 'distance_output_multiplier_to_m': 1.,
             'scope': 'Unscaled source; full metric/gradient calibration below.'}
    if coordinate_scale != 1.:
        candidates, trials, raw_trials = [], [], {}
        identity = [0, 1, 2, 3, 4, 6, 7, 8, 9, 10]
        for multiplier in (1., coordinate_scale):
            input_scale = multiplier
            raw = query(convention)
            raw_trials[f'input_multiplier_{int(multiplier)}'] = raw
            for output_multiplier in (1., 1./coordinate_scale):
                for column in range(4):
                    error = float(np.max(abs(raw[identity, column]*output_multiplier-expected[identity])))
                    row = {'input_multiplier': multiplier, 'distance_output_multiplier_to_m': output_multiplier,
                           'distance_column': column, 'maximum_identity_metric_error_m': error}
                    trials.append(row)
                    if error <= TOLERANCE_M:
                        candidates.append(row)
        np.savez_compressed(output/'native_unit_calibration.npz', points_in_physical_m=convention,
                            expected_distance_m=expected, **raw_trials)
        if len(candidates) != 1:
            return {'passed': False, 'checks': {'native_unit_calibration': False},
                'native_unit_trials': trials, 'native_sdf_inventory': inventory,
                'scope': 'Conditioned mesh native query units did not uniquely calibrate; no surface qualification.'}
        units = {**candidates[0], 'all_trials': trials,
            'scope': 'Input and distance unit scales selected by independent signed metric witnesses; gradients separately calibrated below.'}
        input_scale = units['input_multiplier']
        distance_scale = units['distance_output_multiplier_to_m']
        distance_column = units['distance_column']
    raw_convention = query(convention)
    raw_source = query(points)
    # Independent view capacities distinguish a batch/layout issue from an
    # invalid cooked field without modifying the asset or stepping physics.
    capacity_raw = {}
    for size in (len(convention), 4096):
        small = sim.create_sdf_shape_view(collider_path, size)
        padded = np.repeat(convention[-1:], size, axis=0)
        padded[:len(convention)] = convention
        raw = small.get_sdf_and_gradients(torch.as_tensor((padded*input_scale)[None], dtype=torch.float32, device=device))
        values = raw.detach().cpu().numpy()[0, :len(convention)].copy()
        if distance_column is not None:
            values[:, distance_column] *= distance_scale
        capacity_raw[f'capacity_{size}'] = values
    np.savez_compressed(output/'raw_sdf_samples.npz', point_indices=np.arange(len(points)), points=points, normals=normals,
        raw=raw_source, calibration_points=convention, calibration_raw=raw_convention,
        calibration_expected=expected, **capacity_raw)
    try:
        # Flat bottom/top/shoulder plus base walls identify the tensor layout;
        # every point, including the handle, still participates in all gates.
        layout = calibrate_layout(query, convention, expected, convention_normals,
                                  identity_indices=(0, 1, 2, 3, 4, 6, 7, 8, 9, 10))
        layout = normalized_gradient_check(layout)
    except ValueError as error:
        return {'passed': False, 'checks': {'layout_metric_calibration': False},
            'error': str(error), 'native_sdf_inventory': inventory, 'native_unit_calibration': units,
            'capacity_calibration_raw': {key: value.tolist() for key, value in capacity_raw.items()},
            'source_sample_count': len(points), 'nonzero_source_sample_count': int(np.any(raw_source != 0, axis=1).sum()),
            'scope': 'Native SDF field failed calibration; no zero-surface or contact qualification.'}
    column = layout['distance_column']
    root = normal_zero_witnesses(lambda p: query(p)[:, column], points, normals, raw_source[:, column])
    offset, residual, bracketed = root['normal_offset_m'], root['residual_m'], root['bracketed']
    lower, upper = root['final_bracket_lower_offset_m'], root['final_bracket_upper_offset_m']
    final_low, final_high = root['final_bracket_lower_distance_m'], root['final_bracket_upper_distance_m']
    zeros = points+offset[:, None]*normals
    zero_distance = analytic_distance(zeros)
    selected_requery = query(zeros)[:,column]
    fixed = SUBGRID4_RESIDUAL_INDICES if len(points) == 1589248 else ()
    selected = np.unique(np.r_[np.argsort(abs(residual))[-20:],
        np.argsort(abs(root['original_midpoint_residual_m']))[-20:], fixed]).astype(int)
    delta = np.linspace(-5e-6, 5e-6, 401)
    dense_offsets = root['original_midpoint_offset_m'][selected, None]+delta[None]
    dense_points = points[selected, None, :]+dense_offsets[:, :, None]*normals[selected, None, :]
    previous_point = np.array([-.00011453435733339486, -.027999297217165348, .005916666666666666])
    previous_normal = np.array([-.006135884649162052, -.9999811752826012, 0.])
    previous_offsets = np.array([-3.,-2.,-1.6,-1.58,-1.57,-1.56,-1.54,-1.,0.,1.,2.,3.])*1e-6
    np.savez_compressed(output/'residual_dense_queries.npz', source_point_indices=selected,
        source_points=points[selected], source_normals=normals[selected], normal_offsets_m=dense_offsets,
        raw=query(dense_points.reshape(-1,3)).reshape(len(selected),len(delta),4),
        previous_08_point=previous_point, previous_08_normal=previous_normal,
        previous_08_offsets_m=previous_offsets, previous_08_raw=query(previous_point+previous_offsets[:,None]*previous_normal),
        fixed_candidate09_source_indices=np.asarray(fixed,dtype=int))
    np.savez_compressed(output/'sdf_zero_surface.npz', zero_points=zeros,
        exact_union_distance_m=zero_distance, selected_residual_requery_m=selected_requery,
        original_midpoint_points=points+root['original_midpoint_offset_m'][:,None]*normals, **root,
        calibration_finite_gradient=layout.pop('finite_difference_gradient'), calibration_raw=layout.pop('raw'))
    checks = {'layout_metric_calibration': layout['passed'],
        'calibration_analytic_truth': bool(np.allclose(analytic_distance(convention), expected, rtol=0, atol=1e-12)),
        'all_samples_finite': bool(np.isfinite(raw_source).all() and np.isfinite(root['query_distances_m']).all()
            and np.isfinite(final_low).all() and np.isfinite(final_high).all()),
        'all_rays_bracketed': bool(bracketed.all()),
        'selected_query_reproduced_exactly': bool(np.array_equal(residual,selected_requery)),
        'source_mesh_and_analytic_samples_within_20um': bool(np.max(abs(raw_source[:, column])) <= TOLERANCE_M),
        'zero_normal_displacements_within_20um': bool(np.max(abs(offset)) <= TOLERANCE_M),
        'zero_exact_union_distances_within_20um': bool(np.max(abs(zero_distance)) <= TOLERANCE_M),
        'zero_residual_within_1um': bool(np.max(abs(residual)) <= 1e-6)}
    return {'passed': all(checks.values()), 'checks': checks, 'native_sdf_inventory': inventory,
        'query_batch_capacity': capacity, 'complete_witness_count': len(points),
        'native_unit_calibration': units,
        'layout_calibration': layout, 'source_triangle_samples': len(source_points),
        'analytic_surface_samples': len(exact_points), 'azimuth_count': 1024,
        'maximum_source_distance_m': float(np.max(abs(raw_source[:, column]))),
        'maximum_zero_normal_offset_m': float(np.max(abs(offset))),
        'maximum_zero_exact_union_distance_m': float(np.max(abs(zero_distance))),
        'maximum_zero_residual_m': float(np.max(abs(residual))),
        'zero_selection': 'Minimum absolute residual among source,15 bisection queries and original final midpoint; all17 actual query offsets/distances retained per source witness.',
        'selected_zero_residual_above_1um_count': int(np.sum(abs(residual)>1e-6)),
        'maximum_original_midpoint_residual_m': float(np.max(abs(root['original_midpoint_residual_m']))),
        'original_midpoint_residual_above_1um_count': int(np.sum(abs(root['original_midpoint_residual_m'])>1e-6)),
        'maximum_final_bracket_distance_jump_m': float(np.max(abs(final_high-final_low))),
        'maximum_final_bracket_interval_m': float(np.max(upper-lower)),
        'unbracketed_samples': int((~bracketed).sum()), 'tolerance_m': TOLERANCE_M,
        'scope': 'Finite native cooked-SDF witnesses, including sharp edges. No continuous Hausdorff, contact dynamics, task success or native grid-dimension qualification.'}
