"""Read-only shape and release diagnostics for raw native garment streams.

This writes a separate analysis artifact. It neither changes a native report
nor grants task/robot success. Sampled geometry is not continuous collision
evidence; projected union area is explicitly rasterized at the declared pitch.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import inspect
import json
from pathlib import Path
import sys

import numpy as np

from bench.paths import load_json


def pin(path):
    path = Path(path).resolve()
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4*1024*1024), b''):
            h.update(block)
    return {'path': str(path), 'sha256': h.hexdigest()}


def _quantiles(values):
    values = np.asarray(values, dtype=float)
    return dict(zip(('min', 'p01', 'p50', 'p95', 'p99', 'max'),
                    map(float, np.quantile(values, (0., .01, .5, .95, .99, 1.)))))


def projected_union_area(q, triangles, pitch=.002):
    from PIL import Image, ImageDraw
    lo, hi = q[:, :2].min(0)-pitch, q[:, :2].max(0)+pitch
    size = np.ceil((hi-lo)/pitch).astype(int)+1
    if size.max() > 4096:
        raise ValueError('Observed footprint exceeds the bounded diagnostic raster')
    image = Image.new('1', tuple(map(int, size)), 0)
    draw = ImageDraw.Draw(image)
    pixels = (q[:, :2]-lo)/pitch
    for tri in triangles:
        draw.polygon([tuple(p) for p in pixels[tri]], fill=1)
    return {'area_m2': float(np.count_nonzero(np.asarray(image)))*pitch*pitch,
        'raster_pitch_m': pitch, 'raster_size_xy': size.tolist(),
        'method': 'union of projected surface triangles on a binary pixel raster; includes a boundary-pixel discretization bias'}


def analyze(run_dir, *, task_path=None, source_root=None, motion_path=None,
            sample_every_s=1., raster_pitch_m=.002, self_crossing=True):
    run_dir = Path(run_dir).resolve()
    report_path = run_dir/'native_report.json'
    report = load_json(report_path)
    stream = report['states']
    valid = int(stream['valid_samples'])
    if valid < 1:
        raise ValueError('Native report has no observed samples')
    source_root = Path(source_root or report['source']['snapshot_root']).resolve()
    sys.path.insert(0, str(source_root))
    evaluation = importlib.import_module('scenes.workrooms.laundry_room.evaluation')
    intersections = importlib.import_module('bench.evaluation.cloth.audit_cloth_self_intersection')
    if not Path(evaluation.__file__).resolve().is_relative_to(source_root):
        raise ValueError('Scorer imports did not resolve to the explicitly chosen source tree')
    default_task = report.get('inputs', {}).get('task.json', {}).get('path')
    if default_task is None and task_path is None:
        default_task = str(Path(report['scene_input']['path']).parent/'task.json')
    task_path = Path(task_path or default_task).resolve()
    task = load_json(task_path)
    evaluator = evaluation.SequenceEvaluator(task)
    name = 'towel'
    spec = evaluator.specs[name]
    mesh = evaluator.meshes[name]
    rest, triangles = mesh['rest'], mesh['triangles']
    arrays, checked_files = {}, {}
    for key in ('cloth_q', 'cloth_qd', 'control_mask', 'time_s'):
        item = stream['files'][key]
        file_pin = pin(item['path'])
        if file_pin['sha256'] != item['sha256']:
            raise ValueError('Raw native stream changed: '+key)
        a = np.load(item['path'], mmap_mode='r', allow_pickle=False)
        if list(a.shape) != item['shape'] or len(a) < valid:
            raise ValueError('Native stream shape or valid sample count is inconsistent')
        arrays[key] = a[:valid]
        checked_files[key] = file_pin
    q, qd, mask = (arrays[k] for k in ('cloth_q', 'cloth_qd', 'control_mask'))
    times = np.asarray(arrays['time_s'], dtype=float)
    if q.shape[1:] != rest.shape or qd.shape != q.shape or mask.shape != q.shape[:2]:
        raise ValueError('Raw position, velocity and control topology disagree')
    if not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError('Native time must be strictly increasing and finite')
    if sample_every_s <= 0 or raster_pitch_m <= 0:
        raise ValueError('Positive sampling interval and raster pitch required')
    motion = report.get('motion', report.get('motion_metadata', {}))
    if motion_path:
        motion = load_json(Path(motion_path))
    phases = motion.get('phases', []) if isinstance(motion, dict) else []
    indices = {0, valid-1}
    indices.update(np.searchsorted(times, np.arange(times[0], times[-1], sample_every_s)).tolist())
    events = {}
    attachment_events = report.get('attachment_events', [])
    phase_starts = {}
    for event in attachment_events:
        if event.get('event') == 'enable':
            phase_starts.setdefault(int(event['phase_index']), float(event['time_s']))
    for phase in phases:
        phase_index = int(phase['phase_index'])
        if attachment_events and phase_index not in phase_starts:
            continue
        offset = phase_starts.get(phase_index, phase['grasp_time_s'])-phase['grasp_time_s']
        for key in ('grasp_time_s', 'move_begin_s', 'move_end_s', 'hold_release_time_s', 'fold_release_time_s'):
            if key == 'hold_release_time_s' and 'hold' not in phase['anchor_roles']:
                continue
            event_time = phase.get(key, float('inf'))+offset
            if key.endswith('release_time_s') and attachment_events:
                role = 'hold' if key.startswith('hold') else 'fold'
                release_events = [e for e in attachment_events if e.get('event') == 'disable'
                    and e['phase_index'] == phase_index and phase['anchor_roles'][e['anchor_index']] == role]
                if not release_events:
                    continue
                event_time = max(e['time_s'] for e in release_events)
            if times[0] <= event_time <= times[-1]:
                # Event records are written after the sample at that instant;
                # choose the next actual sample to show its resulting mask.
                side = 'right' if key.endswith('release_time_s') else 'left'
                index = min(int(np.searchsorted(times, event_time, side=side)), valid-1)
                indices.add(index)
                events.setdefault(index, []).append(f"{phase['name']}:{key}")
    target = np.asarray(evaluator.meta['target_center_xy_m'])
    references = getattr(evaluator, 'garment_reference_variants', {}).get(name, [('primary', mesh)])
    stage_references = []
    assets = importlib.import_module('scenes.workrooms.laundry_room.garment_folding_assets')
    if phases and hasattr(assets, '_fold_reference_positions') and hasattr(assets, '_goal_row'):
        flat = np.asarray(spec['unfolded_reference_vertices_m'], dtype=float)
        sequence = task['asset_metadata']['design']['fold_sequence_axis_hinge_sign_radius_m']
        fold_frame = task['asset_metadata']['design'].get('fold_coordinate_frame')
        for phase_index, phase in enumerate(phases):
            folds = sequence[:phase_index+1]
            args = [flat, folds, float(spec['physical_thickness_m'])]
            if 'triangles' in inspect.signature(assets._fold_reference_positions).parameters:
                args.append(triangles)
            frame_options = {'fold_frame': fold_frame} if fold_frame is not None else {}
            reference_q = assets._fold_reference_positions(*args, **frame_options)
            goal_args = [task['id'], rest, triangles, flat, reference_q, float(spec['physical_thickness_m'])]
            if 'folds' in inspect.signature(assets._goal_row).parameters:
                goal_args.append(folds)
            goal = assets._goal_row(*goal_args, **frame_options)
            reference_mesh = evaluator._garment_mesh({**spec, **goal}, rest, triangles,
                mesh['weights'], mesh['areas'], mesh['edges'], mesh['lengths'])
            reference_target = (reference_q[:, :2].min(0)+reference_q[:, :2].max(0))/2
            reference_area = projected_union_area(reference_q, triangles, raster_pitch_m)['area_m2']
            stage_references.append((phase['name'], reference_mesh, reference_target, reference_area))
    table_z = float(evaluator.meta['table_top_z_m'])
    thickness = float(spec['physical_thickness_m'])
    rect = evaluator.meta['clear_rectangle_xy_m']
    rows = []
    for i in sorted(indices):
        points = np.asarray(q[i], dtype=float)
        velocities = np.asarray(qd[i], dtype=float)
        if not np.isfinite(points).all() or not np.isfinite(velocities).all():
            raise ValueError('Nonfinite observed cloth state at sample '+str(i))
        cross = np.cross(points[triangles[:, 1]]-points[triangles[:, 0]],
                         points[triangles[:, 2]]-points[triangles[:, 0]])
        areas = np.linalg.norm(cross, axis=1)/2
        edge_ratio = np.linalg.norm(points[mesh['edges'][:, 1]]-points[mesh['edges'][:, 0]], axis=1)/mesh['lengths']
        shape = {}
        for reference_id, reference_mesh in references:
            kwargs = {'table_supported': True}
            if len(references) > 1 or hasattr(evaluator, 'garment_reference_variants'):
                kwargs['reference_mesh'] = reference_mesh
            good, detail = evaluator._garment_shape(name, points, target, **kwargs)
            shape[reference_id] = {'geometric_goal_good': bool(good), **detail}
        footprint = projected_union_area(points, triangles, raster_pitch_m)
        stage_shape = {}
        for phase_name, reference_mesh, reference_target, reference_area in stage_references:
            good, detail = evaluator._garment_shape(name, points, reference_target,
                table_supported=True, reference_mesh=reference_mesh)
            stage_shape[phase_name] = {'partial_fold_geometry_good': bool(good), **detail,
                'projected_footprint_union_ratio': footprint['area_m2']/reference_area}
        window = max(0, int(np.searchsorted(times, times[i]-.5)))
        drift = np.linalg.norm(points-np.asarray(q[window]), axis=1)
        inside = ((points[:, 0] >= rect[0]) & (points[:, 0] <= rect[1])
                  & (points[:, 1] >= rect[2]) & (points[:, 1] <= rect[3]))
        penetration = np.maximum(table_z+thickness/2-points[inside, 2], 0.)
        row = {'sample_index': i, 'time_s': float(times[i]), 'events': events.get(i, []),
            'extent_xyz_m': np.ptp(points, axis=0).tolist(),
            'bounds_min_m': points.min(0).tolist(), 'bounds_max_m': points.max(0).tolist(),
            'summed_projected_surface_area_m2': float(abs(cross[:, 2]).sum()/2),
            'projected_footprint_union': footprint,
            'edge_length_ratio': _quantiles(edge_ratio), 'triangle_area_ratio': _quantiles(areas/mesh['areas']),
            'raw_node_speed_m_s': _quantiles(np.linalg.norm(velocities, axis=1)),
            'drift_from_half_second_anchor_m': _quantiles(drift),
            'drift_window_actual_s': float(times[i]-times[window]),
            'controlled_vertices': int(np.count_nonzero(mask[i])),
            'reference_geometry': shape,
            'partial_phase_reference_geometry': stage_shape,
            'table_height_envelope': {'max_penetration_m': float(penetration.max()) if len(penetration) else None,
                'vertices_beyond_half_mm_penetration': int(np.count_nonzero(penetration > .0005)),
                'vertices_inside_declared_work_rectangle': int(np.count_nonzero(inside)),
                'vertices_outside_declared_work_rectangle': int(np.count_nonzero(~inside)),
                'scope': 'Material midplane minus half nominal thickness against the table plane, restricted to declared clear rectangle; not a full room collider audit'}}
        rows.append(row)
    counts = np.count_nonzero(mask, axis=1)
    controlled = np.flatnonzero(counts)
    release_index = int(controlled[-1]+1) if len(controlled) and controlled[-1]+1 < valid else (0 if not len(controlled) else None)
    final_window = max(0, int(np.searchsorted(times, times[-1]-.5)))
    final_window_p95 = [float(np.quantile(np.linalg.norm(qd[i], axis=1), .95)) for i in range(final_window, valid)]
    window_anchor = np.asarray(q[final_window], dtype=float)
    final_window_drift_p95 = [float(np.quantile(np.linalg.norm(np.asarray(q[i])-window_anchor, axis=1), .95))
                              for i in range(final_window, valid)]
    crossing = {'checked': False}
    if self_crossing:
        hits, candidates = intersections.self_intersections(np.asarray(q[-1], dtype=float), triangles)
        crossing = {'checked': True, 'sample_index': valid-1, 'time_s': float(times[-1]),
            'triangle_pair_count': len(hits), 'candidate_pairs': candidates,
            'first_128_triangle_pairs': hits[:128],
            'scope': 'Exact sampled triangle crossings and coplanar overlaps with structural-boundary exclusion; final state only, no between-step or finite-thickness certification'}
    return {'schema': 'laundry_garment_native_shape_analysis.v1',
        'inputs': {'native_report': pin(report_path), 'task': pin(task_path), 'streams': checked_files,
            'analysis_source': pin(__file__), 'evaluator_source': pin(evaluation.__file__),
            'intersection_source': pin(intersections.__file__),
            'motion_metadata': pin(motion_path) if motion_path else None},
        'task_id': task['id'], 'recording_complete': bool(report.get('complete')),
        'valid_samples': valid, 'duration_observed_s': float(times[-1]-times[0]),
        'task_success_claimed': False, 'physics_accepted': False,
        'phase_metadata_available': bool(phases), 'sampled_frames': rows,
        'observed_phase_start_times_s': phase_starts,
        'phase_event_timing_basis': 'Actual attachment enable/disable timestamps; movement times offset from actual enable. Robot approach/park time is retained in the raw native clock.',
        'release': {'max_controlled_vertices': int(counts.max()),
            'ever_controlled': bool(len(controlled)), 'final_controlled_vertices': int(counts[-1]),
            'final_all_released_since_s': float(times[release_index]) if release_index is not None else None,
            'final_all_released_duration_s': float(times[-1]-times[release_index]) if release_index is not None else 0.,
            'final_half_second_all_released': bool(np.all(counts[final_window:] == 0)),
            'peak_raw_speed_p95_over_final_half_second_m_s': max(final_window_p95),
            'peak_drift_p95_from_final_half_second_anchor_m': max(final_window_drift_p95)},
        'final_self_crossing': crossing,
        'scope': 'Observed native geometry and masks only. Geometric goal diagnostics do not grant manipulation credit, robot grasp qualification or full task success.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--task', type=Path)
    parser.add_argument('--source-root', type=Path)
    parser.add_argument('--motion-metadata', type=Path)
    parser.add_argument('--sample-every-s', type=float, default=1.)
    parser.add_argument('--raster-pitch-m', type=float, default=.002)
    parser.add_argument('--skip-self-crossing', action='store_true')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.resolve() == args.run_dir.resolve()/'native_report.json' or args.output.exists():
        raise ValueError('Choose a fresh separate analysis artifact; native reports are read-only')
    result = analyze(args.run_dir, task_path=args.task, source_root=args.source_root,
        motion_path=args.motion_metadata, sample_every_s=args.sample_every_s,
        raster_pitch_m=args.raster_pitch_m, self_crossing=not args.skip_self_crossing)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({'output': str(args.output), 'task_id': result['task_id'],
        'valid_samples': result['valid_samples'], 'release': result['release'],
        'final_self_crossing': result['final_self_crossing'], 'final': result['sampled_frames'][-1]}, indent=2))


if __name__ == '__main__':
    main()
