"""Append-only, finite scalar-root diagnostic for the frozen L5 witnesses.

This implements bounded_zero_search_proposal03's query schedule. It neither
changes the old Newton projection nor grants original-body exterior credit.
Callbacks receive exactly the local float32 coordinates recorded here; they
must submit those coordinates unchanged and return the complete native Nx4.
"""
from pathlib import Path
import json
import math

import numpy as np

from scripts.validation.gear_sdf_checks import TOLERANCE_M, ZERO_RESIDUAL_M

SEARCH_RADIUS_M = TOLERANCE_M - 20.e-9
OFFSETS_M = np.linspace(-SEARCH_RADIUS_M, SEARCH_RADIUS_M, 9)
BISECTION_ITERATIONS = 16


def frozen_directions(mesh, vertex_indices):
    """Reproduce the fixed 64 hemisphere lines plus incident normal lines."""
    indices = np.arange(64)
    z = (indices + .5) / 64
    phi = indices * math.pi * (3 - math.sqrt(5))
    sphere = np.c_[np.sqrt(1-z*z)*np.cos(phi), np.sqrt(1-z*z)*np.sin(phi), z]
    result = []
    for vertex in vertex_indices:
        faces = np.flatnonzero(np.any(mesh.faces == vertex, axis=1))
        unique = []
        for normal in mesh.face_normals[faces]:
            normal = normal.copy()
            nonzero = np.flatnonzero(abs(normal) > 1.e-12)
            if not len(nonzero):
                raise ValueError('Incident face has no finite nonzero normal')
            if normal[nonzero[-1]] < 0:
                normal = -normal
            if not any(np.linalg.norm(normal-other) < 1.e-9 for other in unique):
                unique.append(normal)
        result.append(np.concatenate([sphere, np.asarray(unique).reshape(-1, 3)]))
    return tuple(result)


def _initial_samples(origins, directions):
    points, labels = [], []
    for origin_index, (origin, lines) in enumerate(zip(origins, directions, strict=True)):
        for direction_index, direction in enumerate(lines):
            points.append(origin + OFFSETS_M[:, None]*direction)
            labels.extend((origin_index, direction_index, offset) for offset in range(9))
    return (np.concatenate(points) if points else np.empty((0, 3)),
            np.asarray(labels, dtype=np.int64).reshape(-1, 3))


def load_frozen_field(witnesses, field):
    """Load one existing NPZ population without generating replacement seeds.

    The caller verifies the frozen file/source hashes. Checking exact initial
    arrays here catches frame, ordering and offset mismatches before querying.
    """
    origins = np.asarray(witnesses[field+'_origins'], dtype=float)
    directions = tuple(np.asarray(witnesses[f'{field}_directions_{i}'], dtype=float)
                       for i in range(len(origins)))
    source_indices = np.asarray(witnesses[field+'_origin_indices'], dtype=np.int64)
    points, labels = _initial_samples(origins, directions)
    if (not np.array_equal(points, witnesses[field+'_points'])
            or not np.array_equal(labels, witnesses[field+'_labels'])):
        raise ValueError('Frozen initial witness coordinates or labels changed')
    return {'origins': origins, 'directions': directions, 'source_indices': source_indices}


def bounded_zero_search(queries, origins, directions, *, output, rotations=None,
                        distance_columns=None, gradient_columns=None, source_indices=None):
    """Query all fixed brackets and retain success, failure and actual submits.

    ``origins`` and ``directions`` use one field frame (piece-local or body).
    ``rotations[k]`` maps field to collider k via ``local = points @ R``.
    Every local float32 point is inverse-transformed and checked against its
    original witness before any callback runs. Compound candidate ordering uses
    the maximum displacement over these actual coordinates, conservatively.

    Each nonempty round calls every collider once: initial, 16 synchronous
    bisections, final requery. Invalid brackets cannot regain validity. Exact
    zeros collapse their bracket and are still queried for all 16 rounds.
    All four columns must be finite even though no API gradient is a derivative.

    The JSON-friendly result and result.npz are numeric diagnostics only.
    Nearby internal caps and multiple zeros require independent classification;
    this helper always leaves exterior classification ambiguous.
    """
    queries = tuple(queries)
    count = len(queries)
    origins = np.asarray(origins, dtype=float)
    directions = tuple(np.asarray(d, dtype=float) for d in directions)
    if not count or origins.ndim != 2 or origins.shape[1] != 3 or not np.isfinite(origins).all():
        raise ValueError('Require native callbacks and finite Nx3 original witnesses')
    if len(directions) != len(origins):
        raise ValueError('Each witness needs its own direction population')
    for lines in directions:
        if (lines.ndim != 2 or lines.shape[1] != 3 or not len(lines)
                or not np.isfinite(lines).all()
                or not np.allclose(np.linalg.norm(lines, axis=1), 1., rtol=0., atol=1.e-12)):
            raise ValueError('Every search direction must be a finite unit vector')
    rotations = np.asarray([np.eye(3)]*count if rotations is None else rotations, dtype=float)
    if (rotations.shape != (count, 3, 3) or not np.isfinite(rotations).all()
            or not np.allclose(rotations @ rotations.transpose(0, 2, 1), np.eye(3), rtol=0., atol=1.e-12)
            or not np.allclose(np.linalg.det(rotations), 1., rtol=0., atol=1.e-12)):
        raise ValueError('Collider transforms must be proper orthonormal rotations')
    columns = np.asarray([3]*count if distance_columns is None else distance_columns)
    if columns.shape != (count,) or not np.isin(columns, np.arange(4)).all():
        raise ValueError('One decoded distance column is required per collider')
    orders = ([tuple(i for i in range(4) if i != column) for column in columns]
              if gradient_columns is None else gradient_columns)
    if len(orders) != count or any(sorted([int(column), *order]) != list(range(4))
                                    for column, order in zip(columns, orders, strict=True)):
        raise ValueError('Each native layout must cover four distinct columns')
    indices = np.arange(len(origins)) if source_indices is None else np.asarray(source_indices)
    if (indices.shape != (len(origins),) or not np.issubdtype(indices.dtype, np.integer)
            or np.any(indices < 0) or len(np.unique(indices)) != len(indices)):
        raise ValueError('Original source indices must be unique nonnegative integers')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    calls = 0
    all_finite = True
    all_bounded = True
    records = []

    def sample(points, labels, iteration, *, eligible=None, **metadata):
        nonlocal calls, all_finite, all_bounded
        points = np.asarray(points, dtype=float).reshape(-1, 3)
        labels = np.asarray(labels, dtype=np.int64).reshape(-1, 3)
        origin_rows = origins[labels[:, 0]]
        submitted = np.asarray([points @ rotation for rotation in rotations], dtype=np.float32)
        actual = np.asarray([local.astype(float) @ rotation.T
                             for local, rotation in zip(submitted, rotations, strict=True)])
        displacement = np.linalg.norm(actual-origin_rows[None], axis=2)
        bounded = np.isfinite(actual).all(axis=(0, 2)) & np.all(displacement <= TOLERANCE_M, axis=0)
        all_bounded &= bool(np.all(bounded))
        queried = bounded.copy() if eligible is None else bounded & eligible
        number = len(records)
        prefix = output / f'{number:02d}'
        # Requests survive callback errors, including rejected out-of-ball rows.
        np.savez_compressed(str(prefix)+'_request.npz', requested_points=points,
            labels=labels, source_indices=indices[labels[:, 0]], iteration=iteration,
            submitted_local_float32=submitted, actual_field_points=actual,
            displacement_by_shape_m=displacement, bounded=bounded, queried=queried, **metadata)
        raw = np.full((count, len(points), 4), np.nan)
        raw_responses = {}
        schema_valid = np.ones(count, dtype=bool)
        try:
            if np.any(queried):
                for shape, query in enumerate(queries):
                    calls += 1
                    response = np.asarray(query(submitted[shape, queried]))
                    raw_responses[f'native_raw_{shape}'] = response
                    schema_valid[shape] = response.shape == (int(queried.sum()), 4)
                    if schema_valid[shape]:
                        raw[shape, queried] = response
                    all_finite &= bool(schema_valid[shape] and np.isfinite(response).all())
        except Exception as exc:
            np.savez_compressed(str(prefix)+'_response.npz', all_native_raw=raw, **raw_responses)
            (Path(str(prefix)+'_error.json')).write_text(json.dumps(
                {'type': type(exc).__name__, 'message': str(exc), 'native_calls_so_far': calls}, indent=2)+'\n')
            raise
        distances = np.stack([r[:, int(column)] for r, column in zip(raw, columns, strict=True)])
        finite = queried & np.isfinite(raw).all(axis=(0, 2))
        winner = np.argmin(np.where(np.isfinite(distances), distances, np.inf), axis=0)
        winner[~finite] = -1
        selected = np.full((len(points), 4), np.nan)
        for shape, (rotation, column, order) in enumerate(zip(rotations, columns, orders, strict=True)):
            chosen = finite & (winner == shape)
            selected[chosen, :3] = raw[shape, chosen][:, order] @ rotation.T
            selected[chosen, 3] = raw[shape, chosen, int(column)]
        record = dict(points=points, labels=labels, iteration=iteration, submitted=submitted,
            actual=actual, displacement=displacement.max(axis=0), raw=raw, selected=selected,
            finite=finite, bounded=bounded, winner=winner, queried=queried)
        records.append(record)
        np.savez_compressed(str(prefix)+'_response.npz', all_native_raw=raw,
            selected_raw=selected, winner=winner, finite=finite, schema_valid=schema_valid,
            **raw_responses)
        return record

    points, initial_labels = _initial_samples(origins, directions)
    # Initial labels identify offsets; later labels identify subintervals.
    initial = sample(points, initial_labels, 0)
    line_rows = np.arange(len(points)).reshape(-1, 9)
    left = line_rows[:, :-1].reshape(-1)
    right = line_rows[:, 1:].reshape(-1)
    left_value, right_value = initial['selected'][left, 3], initial['selected'][right, 3]
    brackets = (initial['finite'][left] & initial['finite'][right]
                & (((left_value < 0.) & (right_value > 0.))
                   | ((left_value > 0.) & (right_value < 0.))))
    left, right = left[brackets], right[brackets]
    labels = initial_labels[left].copy()
    lo, hi = OFFSETS_M[labels[:, 2]].copy(), OFFSETS_M[labels[:, 2]+1].copy()
    lo_raw, hi_raw = initial['selected'][left].copy(), initial['selected'][right].copy()
    lo_record, hi_record = np.zeros(len(left), dtype=int), np.zeros(len(right), dtype=int)
    lo_row, hi_row = left.copy(), right.copy()
    valid = np.ones(len(left), dtype=bool)
    np.savez_compressed(output/'initial_brackets.npz', left_row=left, right_row=right,
        labels=labels, all_subintervals_sign_changing=brackets)
    line_directions = np.asarray([directions[origin][direction]
        for origin, direction, _ in labels]).reshape(-1, 3)
    for iteration in range(1, BISECTION_ITERATIONS+1):
        mid = (lo+hi)/2
        current = sample(origins[labels[:, 0]]+mid[:, None]*line_directions, labels, iteration,
            eligible=valid, left_offset_m=lo, right_offset_m=hi, left_raw=lo_raw, right_raw=hi_raw,
            left_record=lo_record, right_record=hi_record, left_row=lo_row, right_row=hi_row,
            bracket_valid_before=valid)
        valid &= current['finite']
        value = current['selected'][:, 3]
        zero = valid & (value == 0.)
        move_left = valid & ((np.signbit(value) == np.signbit(lo_raw[:, 3])) | zero)
        move_right = valid & (~move_left | zero)
        lo[move_left], hi[move_right] = mid[move_left], mid[move_right]
        lo_raw[move_left], hi_raw[move_right] = current['selected'][move_left], current['selected'][move_right]
        lo_record[move_left], hi_record[move_right] = iteration, iteration
        lo_row[move_left] = np.flatnonzero(move_left)
        hi_row[move_right] = np.flatnonzero(move_right)

    # Keep every candidate in its immutable round file; this only chooses an
    # index for each origin, with no replacement search after the final query.
    best = [None]*len(origins)
    for record_index, record in enumerate(records):
        candidate = record['finite'] & (abs(record['selected'][:, 3]) <= ZERO_RESIDUAL_M)
        for row in np.flatnonzero(candidate):
            origin, direction, interval = record['labels'][row]
            if record_index == 0:
                interval = min(interval, 7)
            key = (record['displacement'][row], abs(record['selected'][row, 3]),
                   int(direction), int(interval), record['iteration'])
            if best[origin] is None or key < best[origin][0]:
                best[origin] = (key, record_index, row)
    chosen_origins = np.asarray([i for i, choice in enumerate(best) if choice is not None], dtype=int)
    chosen_records = np.asarray([best[i][1] for i in chosen_origins], dtype=int)
    chosen_rows = np.asarray([best[i][2] for i in chosen_origins], dtype=int)
    final_points = np.asarray([records[r]['points'][i] for r, i in zip(chosen_records, chosen_rows, strict=True)]).reshape(-1, 3)
    final_labels = np.asarray([records[r]['labels'][i] for r, i in zip(chosen_records, chosen_rows, strict=True)], dtype=int).reshape(-1, 3)
    final = sample(final_points, final_labels, BISECTION_ITERATIONS+1,
        selected_record=chosen_records, selected_row=chosen_rows)
    same_submit = np.ones(len(chosen_origins), dtype=bool)
    for row, (record, candidate_row) in enumerate(zip(chosen_records, chosen_rows, strict=True)):
        same_submit[row] = np.array_equal(final['submitted'][:, row], records[record]['submitted'][:, candidate_row])
    qualified = np.zeros(len(origins), dtype=bool)
    qualified[chosen_origins] = (final['finite'] & same_submit
        & (abs(final['selected'][:, 3]) <= ZERO_RESIDUAL_M))
    np.savez_compressed(output/'result.npz', origins=origins, source_indices=indices,
        qualified=qualified, chosen_origin_indices=chosen_origins,
        chosen_record=chosen_records, chosen_row=chosen_rows,
        selected_points=final_points, submitted_local_float32=final['submitted'],
        actual_field_points=final['actual'], all_native_raw=final['raw'],
        selected_raw=final['selected'], winner=final['winner'],
        displacement_m=final['displacement'], same_submitted_coordinate=same_submit,
        final_bracket_valid=valid, final_left_offset_m=lo, final_right_offset_m=hi,
        final_left_raw=lo_raw, final_right_raw=hi_raw,
        final_left_record=lo_record, final_right_record=hi_record,
        final_left_row=lo_row, final_right_row=hi_row)
    summary = {'schema': 'gear_bounded_zero_diagnostic.v1',
        'numeric_passed': bool(len(origins) and qualified.all() and all_finite and all_bounded),
        'origin_count': len(origins), 'source_indices': indices.tolist(),
        'numeric_qualified_count': int(qualified.sum()),
        'unqualified_source_indices': indices[~qualified].tolist(),
        'all_query_receipts_finite': all_finite,
        'all_attempted_coordinates_within_bound': all_bounded,
        'native_query_call_count': calls, 'native_query_call_upper': (BISECTION_ITERATIONS+2)*count,
        'query_round_count': len(records), 'bisection_iterations': BISECTION_ITERATIONS,
        'initial_sign_changing_bracket_count': int(brackets.sum()),
        'selected_candidate_count': len(chosen_origins),
        'maximum_selected_displacement_m': (float(final['displacement'].max()) if len(final_points) else None),
        'maximum_selected_residual_m': (float(abs(final['selected'][:, 3]).max())
            if len(final_points) and np.isfinite(final['selected']).all() else None),
        'geometry_qualified': False, 'exterior_classification': 'ambiguous',
        'used_for_legacy_acceptance': False, 'result_npz': str(output/'result.npz'),
        'candidate_order': 'maximum actual shape displacement, absolute residual, direction, subinterval, iteration',
        'initial_sample_subinterval_rule': 'offset index maps to min(offset index, 7); iteration is 0',
        'scope': 'Finite numeric feasibility only. No bracket does not prove absence of a zero. '
                 'Internal-cap, original-body exterior, original projection, metric and dynamics gates remain independent.'}
    (output/'report.json').write_text(json.dumps(summary, indent=2)+'\n')
    return summary
