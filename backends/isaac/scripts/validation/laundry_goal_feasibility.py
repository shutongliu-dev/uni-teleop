"""Prepare passive garment goal resets and audit them with the original referee.

No simulation is launched here. Run a fresh frozen laundry_material_settle
process on each prepared input. Reset geometry earns no operation/task credit.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import importlib
import json
import math
from pathlib import Path
import sys

import numpy as np

from bench.paths import DATA_ROOT, load_json

DATASETS = Path(f'{DATA_ROOT}/datasets')
PROFILE = 'laundry_single_garment_folding_task.v4'


def pin(path):
    path = Path(path).resolve()
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4*1024*1024), b''):
            digest.update(block)
    return {'path': str(path), 'sha256': digest.hexdigest()}


def checked(record):
    path = Path(record['path']).resolve()
    if pin(path)['sha256'] != record['sha256']:
        raise ValueError('Changed pinned input: '+str(path))
    return path


def write_json(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def derive_inputs(task, scene, initial, reset, surface_materials=None, candidate_vertices=None):
    """Change only authored reset positions and an explicitly supplied surface prior."""
    if task['id'] not in ('shirt_folding', 'trousers_folding') or task['evaluation']['rule_profile'] != PROFILE:
        raise ValueError('An original single-garment v4 task is required')
    if reset not in ('unfolded', 'analytic-goal', 'geometry-candidate'):
        raise ValueError('Unknown reset')
    if (reset == 'geometry-candidate') != (candidate_vertices is not None):
        raise ValueError('Explicit candidate vertices are required only for a geometry-candidate reset')
    if (scene.get('named_cloth_initialization') != 'author-only'
            or len(scene.get('cloths', [])) != 1 or scene.get('cloth', {}).get('enabled', False)
            or not scene.get('robot', {}).get('hand_id') or not scene['robot'].get('enabled', True)):
        raise ValueError('One authored named cloth and the unchanged installed robot required')
    spec, meta = scene['cloths'][0], task['asset_metadata']['cloths'][0]
    name = spec['id']
    if name != meta['id'] or set(initial['cloths_vertices_m']) != {name}:
        raise ValueError('Reset material identity differs')
    for key in ('rest_vertices_m', 'triangles'):
        if spec[key] != meta[key]:
            raise ValueError('Task and simulation rest topology differ: '+key)
    if spec['origin_m'] != [0., 0., 0.] or spec['quaternion_wxyz'] != [1., 0., 0., 0.]:
        raise ValueError('Expected original world-coordinate authored garment')
    if not np.array_equal(spec['initial_vertices_m'], initial['cloths_vertices_m'][name]):
        raise ValueError('Scene and initial reset positions differ')
    if not math.isclose(scene['physics']['dt_s'], task['evaluation']['physics_dt_s'], abs_tol=1.e-12):
        raise ValueError('Native and referee dt differ')
    candidate_scene, candidate_initial = deepcopy(scene), deepcopy(initial)
    changes = []
    if reset in ('analytic-goal', 'geometry-candidate'):
        points = np.asarray(meta['folded_reference_vertices_m'] if reset == 'analytic-goal' else candidate_vertices, dtype=float)
        if points.shape != np.asarray(spec['rest_vertices_m']).shape or not np.isfinite(points).all():
            raise ValueError('Finite original target with identical topology required')
        candidate_scene['cloths'][0]['initial_vertices_m'] = points.tolist()
        candidate_initial['cloths_vertices_m'][name] = points.tolist()
        changes.extend(['scene.cloths[0].initial_vertices_m', 'initial.cloths_vertices_m.'+name])
    if surface_materials is not None:
        # Validation belongs to the existing generic material contract.
        from bench.simulation.isaacsim_backend.workroom_materials import material_options
        candidate_scene['isaacsim']['workroom_surface_materials'] = material_options(surface_materials)
        changes.append('scene.isaacsim.workroom_surface_materials')
    return candidate_scene, candidate_initial, changes


def load_candidate(vertices_path, evidence_path, task_path):
    """Require the candidate's saved research provenance and original task binding."""
    evidence = load_json(Path(evidence_path))
    if (evidence.get('schema') != 'laundry_legal_goal_reset_geometry_research.v1'
            or evidence.get('native_simulation_executed') is not False or evidence.get('task_success') is not False):
        raise ValueError('Explicit static geometry-candidate evidence required')
    checked(evidence['original_task'])
    if evidence['original_task']['sha256'] != pin(task_path)['sha256']:
        raise ValueError('Candidate belongs to a different original task')
    generator = checked(evidence['script'])
    helper = checked(evidence['tool'])
    vertices_pin = pin(vertices_path)
    rows = [row for row in evidence['candidates'] if row['reset_coordinates'] == vertices_pin]
    if len(rows) != 1:
        raise ValueError('Candidate vertices do not match exactly one pinned research row')
    checked(rows[0]['reset_coordinates'])
    points = np.load(vertices_path, allow_pickle=False)
    if points.dtype.kind not in 'fi' or not np.isfinite(points).all():
        raise ValueError('Finite numeric candidate vertices required')
    return points, {'vertices': vertices_pin, 'research': pin(evidence_path),
        'generation_script': pin(generator), 'generation_helper': pin(helper), 'candidate_name': rows[0]['name'],
        'scope': 'Authored reset candidate only; original task/scoring reference is unchanged.'}


def frozen_modules(record):
    manifest_path = checked(record)
    source = manifest_path.parent
    manifest = load_json(manifest_path)
    sys.path.insert(0, str(source))
    names = ('scenes.workrooms.laundry_room.evaluation',
             'scenes.workrooms.laundry_room.terminal_window',
             'bench.evaluation.cloth.audit_cloth_self_intersection',
             'bench.evaluation.cloth.cloth_robot_collision',
             'scenes.workrooms.laundry_room.native_observer')
    modules = [importlib.import_module(name) for name in names]
    pins = {}
    # Check all imported referee/geometry dependencies, not only its entrypoint.
    for name, module in tuple(sys.modules.items()):
        path = getattr(module, '__file__', None)
        if path and (name == 'scenes' or name.startswith('scenes.') or name == 'bench' or name.startswith('bench.')):
            path = Path(path).resolve()
            if not path.is_relative_to(source):
                raise ValueError('Run in a fresh process; frozen imports escaped: '+name)
            item = pin(path)
            if manifest['files_sha256'].get(str(path.relative_to(source))) != item['sha256']:
                raise ValueError('Frozen source hash mismatch: '+name)
            pins[name] = item
    return modules, pins


def strain(rest, points, triangles):
    """Independent 3D edge-Gram stretch diagnostics, with rest-area weighting."""
    def gram(q):
        a = q[triangles[:, 1]]-q[triangles[:, 0]]
        b = q[triangles[:, 2]]-q[triangles[:, 0]]
        return (np.sum(a*a, axis=1), np.sum(a*b, axis=1), np.sum(b*b, axis=1),
                np.linalg.norm(np.cross(a, b), axis=1)/2)
    r00, r01, r11, area = gram(rest)
    q00, q01, q11, new_area = gram(points)
    det = r00*r11-r01*r01
    if np.any(det <= 0):
        raise ValueError('Degenerate rest triangles')
    other = np.maximum(q00*q11-q01*q01, 0.)
    trace = q00*r11+q11*r00-2*q01*r01
    hi = (trace+np.sqrt(np.maximum(trace*trace-4*det*other, 0.)))/(2*det)
    lo = other/(det*np.maximum(hi, 1.e-300))
    stretches = np.sqrt(np.maximum(np.column_stack([lo, hi]), 0.))
    error = np.max(abs(stretches-1.), axis=1)
    order = np.argsort(error)
    i = min(np.searchsorted(np.cumsum(area[order]), .95*area.sum()), len(order)-1)
    return {'principal_deviation_rest_area_p95': float(error[order[i]]),
        'rest_area_fraction_above_5pct_strain': float(area[error > .05].sum()/area.sum()),
        'principal_stretch_min': float(stretches.min()), 'principal_stretch_max': float(stretches.max()),
        'triangle_area_ratio_min': float((new_area/area).min())}


def static_diagnostics(points, evaluator, intersection_module):
    name = next(iter(evaluator.specs))
    mesh = evaluator.meshes[name]
    good, geometry = evaluator._garment_fold({name: points})
    hits, candidates = intersection_module.self_intersections(points, mesh['triangles'])
    return {'original_goal_geometry_passed': bool(good), 'original_geometry': geometry,
        'original_mesh_metrics': evaluator._mesh_metrics(name, points),
        'strain': strain(mesh['rest'], points, mesh['triangles']),
        'self_crossings': {'pair_count': len(hits), 'candidate_pairs': candidates,
            'pairs': hits, 'scope': 'Exact sampled center-surface intersections; no finite-thickness or continuous-time certificate.'}}


def robot_checker(observer_module, export, body_names, evaluator, task):
    """Reuse the original observer's 5 mm + half-thickness release envelope."""
    return observer_module.CookedRobotGeometry(export, body_names,
        {name: mesh['triangles'] for name, mesh in evaluator.meshes.items()},
        {name: spec['physical_thickness_m'] for name, spec in evaluator.specs.items()},
        penetration_tolerances_m=(evaluator.limits['max_penetration_m'], evaluator.limits['hard_penetration_m']),
        active_sides=task.get('active_sides', ['left', 'right']))


def prepare(args):
    native = load_json(args.original_report)
    paths = {key: checked(native['inputs'][key]) for key in ('task.json', 'scene.json', 'initial_state.json')}
    task, scene, initial = [load_json(paths[key]) for key in paths]
    # Import frozen code before any project-dependent validation.
    modules, source_pins = frozen_modules(native['source_manifest'])
    evaluator = modules[0].SequenceEvaluator(task)
    surface = load_json(args.surface_materials_json) if args.surface_materials_json else None
    candidate, candidate_provenance = None, None
    if args.reset == 'geometry-candidate':
        if args.candidate_vertices is None or args.candidate_evidence is None:
            raise ValueError('Geometry-candidate requires --candidate-vertices and --candidate-evidence')
        candidate, candidate_provenance = load_candidate(args.candidate_vertices, args.candidate_evidence, paths['task.json'])
    elif args.candidate_vertices is not None or args.candidate_evidence is not None:
        raise ValueError('Candidate paths apply only to a geometry-candidate reset')
    new_scene, new_initial, changes = derive_inputs(task, scene, initial, args.reset, surface, candidate)
    name = scene['cloths'][0]['id']
    points = np.asarray(new_initial['cloths_vertices_m'][name], float)
    preflight = static_diagnostics(points, evaluator, modules[2])
    delta = np.linalg.norm(points-np.asarray(task['asset_metadata']['cloths'][0]['folded_reference_vertices_m'], float), axis=1)
    preflight['difference_from_unchanged_scoring_reference'] = {
        'vertex_rms_m': float(np.sqrt(np.mean(delta*delta))), 'vertex_p95_m': float(np.quantile(delta, .95)),
        'vertex_max_m': float(delta.max()), 'alignment': 'Raw world-coordinate material correspondence; original scorer performs its own unchanged alignment.'}
    # This only predicts clearance using the old observed resting robot. The new
    # native process must export its own cooked geometry/poses for actual release.
    if native.get('robot_colliders'):
        export_path = checked(native['robot_colliders'])
        body_path = checked(native['states']['files']['robot_body_q'])
        body = np.load(body_path, mmap_mode='r', allow_pickle=False)[0:1]
        preflight['old_robot_pose_clearance'] = modules[3].audit_cloth_robot(
            load_json(export_path), {'particle_q': points[None], 'body_q': body},
            evaluator.meshes[name]['triangles'], native['body_names'])
        preflight['old_robot_pose_clearance']['scope'] += ' Uses original native sample 0 robot poses, not a new-run separation certificate.'
        checker = robot_checker(modules[4], load_json(export_path), native['body_names'], evaluator, task)
        preflight['old_robot_pose_release'] = checker.check({name: points}, body[0])
    output = args.output.resolve()
    if not output.is_relative_to(DATASETS) or output.exists():
        raise ValueError('Choose a fresh shared dataset input directory')
    output.mkdir(parents=True)
    (output/'task.json').write_bytes(paths['task.json'].read_bytes())
    write_json(output/'scene.json', new_scene)
    write_json(output/'initial_state.json', new_initial)
    result = {'schema': 'laundry_goal_feasibility_inputs.v1', 'script': pin(__file__),
        'original_report': pin(args.original_report), 'original_inputs': {k: pin(v) for k, v in paths.items()},
        'source_manifest': native['source_manifest'], 'frozen_source_pins': source_pins,
        'reset': args.reset, 'changes': changes, 'cloth_id': name,
        'geometry_candidate': candidate_provenance,
        'surface_materials_input': pin(args.surface_materials_json) if surface is not None else None,
        'inputs': {k: pin(output/k) for k in paths}, 'initial_diagnostics': preflight,
        'task_success': False, 'physics_accepted': False, 'native_simulation_executed': False,
        'scope': 'Object-only passive reset diagnostic. Task bytes/rest/material/robot/control/thresholds preserved; only listed authored reset/surface-prior fields differ. No operation credit or cloth state setter.'}
    write_json(output/'diagnostic_inputs.json', result)
    return {'output': str(output), 'reset': args.reset, 'initial_geometry_passed': preflight['original_goal_geometry_passed'],
            'initial_self_crossing_pairs': preflight['self_crossings']['pair_count']}


def load_arrays(native, keys):
    valid = int(native['states']['valid_samples'])
    if valid < 1:
        raise ValueError('No observed native samples')
    arrays, pins = {}, {}
    for key in keys:
        item = native['states']['files'][key]
        path = checked(item)
        value = np.load(path, mmap_mode='r', allow_pickle=False)
        if (list(value.shape) != item['shape'] or len(value) < valid
                or ('dtype' in item and str(value.dtype) != item['dtype'])):
            raise ValueError('Truncated or inconsistent raw stream: '+key)
        if not np.isfinite(value[:valid]).all():
            raise ValueError('Nonfinite raw stream: '+key)
        arrays[key], pins[key] = value[:valid], pin(path)
    if (arrays['native_step'].dtype.kind not in 'iu'
            or not np.array_equal(arrays['native_step'], np.arange(valid))
            or not np.isfinite(arrays['time_s']).all() or np.any(np.diff(arrays['time_s']) <= 0)):
        raise ValueError('Missing or inconsistent native samples')
    return arrays, pins


def passive_run_checks(native, arrays):
    """Evidence completeness/absence of assistance; not a geometry acceptance gate."""
    return {
        'native_report_complete': native.get('complete') is True,
        'native_stream_complete': native['states'].get('complete') is True,
        'all_allocated_samples_observed': native['states'].get('allocated_samples') == len(arrays['time_s']),
        'native_step_count_matches': native.get('native_steps') == len(arrays['time_s'])-1,
        'passive_settle_mode': native.get('mode') == 'settle',
        'no_attachment_prims': native.get('actual_attachment_prims') == [],
        'no_attachments_created': native.get('attachments_created') == 0,
        'no_cloth_position_velocity_writes': native.get('cloth_position_velocity_writes') == 0,
        'no_controlled_vertices': not bool(np.any(arrays['control_mask'])),
        'no_task_success_claim': native.get('task_success_claimed') is False,
    }


def analyze(args):
    prepared = load_json(args.prepared)
    paths = {k: checked(v) for k, v in prepared['inputs'].items()}
    native = load_json(args.report)
    native_source = checked(native['source_manifest'])
    for key, report_key in [('task.json', 'task_input'), ('scene.json', 'scene_input'), ('initial_state.json', 'initial_input')]:
        if native[report_key]['sha256'] != prepared['inputs'][key]['sha256']:
            raise ValueError('Native run did not use the prepared input: '+key)
        checked(native[report_key])
    task = load_json(paths['task.json'])
    modules, source_pins = frozen_modules(prepared['source_manifest'])
    evaluator = modules[0].SequenceEvaluator(task)
    name = prepared['cloth_id']; mesh = evaluator.meshes[name]
    arrays, stream_pins = load_arrays(native, ('native_step', 'time_s', 'cloth_q', 'cloth_qd', 'control_mask', 'robot_body_q'))
    dt = evaluator.c['physics_dt_s']
    if not np.allclose(arrays['time_s'], arrays['native_step']*dt, atol=1.e-8, rtol=0):
        raise ValueError('Native and original rule clocks differ')
    if not np.isin(arrays['control_mask'], [0, 1]).all():
        raise ValueError('Invalid recorded control mask')
    terminal_arrays = {'time_s': arrays['time_s'], 'native_step': arrays['native_step'],
        name+'_q': arrays['cloth_q'], name+'_qd': arrays['cloth_qd'],
        name+'_control_mask': np.asarray(arrays['control_mask'], bool), name+'_triangles': mesh['triangles']}
    window = modules[1].terminal_window(terminal_arrays, evaluator)
    passive_checks = passive_run_checks(native, arrays)
    count = math.ceil(evaluator.limits['stable_s']/dt)+1
    first = max(0, len(arrays['time_s'])-count)
    indices = sorted({0, *range(first, len(arrays['time_s']))})
    rows = []
    for index in indices:
        q = np.asarray(arrays['cloth_q'][index], float)
        rows.append({'sample_index': index, 'time_s': float(arrays['time_s'][index]),
            'original_mesh_metrics': evaluator._mesh_metrics(name, q), 'strain': strain(mesh['rest'], q, mesh['triangles'])})
    endpoints = {label: static_diagnostics(np.asarray(arrays['cloth_q'][index], float), evaluator, modules[2])
        for label, index in [('initial', 0), ('final', len(arrays['time_s'])-1)]}
    collision = {'status': 'unchecked', 'reason': 'New-run native cooked collider export missing', 'frames_checked': 0}
    robot_release = {'checked': False, 'terminal_fully_separated': False}
    if native.get('robot_colliders'):
        export_path = checked(native['robot_colliders'])
        collision = modules[3].audit_cloth_robot(load_json(export_path),
            {'particle_q': np.asarray(arrays['cloth_q'][indices]), 'body_q': np.asarray(arrays['robot_body_q'][indices])},
            mesh['triangles'], native['body_names'])
        collision.update(export=pin(export_path), sample_indices=indices)
        checker = robot_checker(modules[4], load_json(export_path), native['body_names'], evaluator, task)
        observed = [{'sample_index': i, 'time_s': float(arrays['time_s'][i]),
            **checker.check({name: np.asarray(arrays['cloth_q'][i], float)}, arrays['robot_body_q'][i])} for i in indices]
        robot_release = {'checked': True, 'samples': observed,
            'terminal_fully_separated': all(row['fully_separated'] for row in observed if row['sample_index'] >= first),
            'scope': 'Original frozen native observer release envelope at initial and every terminal sample; no force measurement.'}
    release = {'attachment_prims': native.get('actual_attachment_prims'),
        'cloth_position_velocity_writes': native.get('cloth_position_velocity_writes'),
        'controlled_nodes_anywhere': int(np.count_nonzero(arrays['control_mask'])),
        'new_run_cooked_robot_clearance': collision, 'original_robot_release': robot_release}
    result = {'schema': 'laundry_goal_feasibility_analysis.v1', 'script': pin(__file__),
        'prepared': pin(args.prepared), 'native_report': pin(args.report), 'native_complete': bool(native.get('complete')),
        'native_source_manifest': pin(native_source), 'original_referee_source_manifest': prepared['source_manifest'],
        'frozen_source_pins': source_pins, 'streams': stream_pins, 'unchanged_limits': evaluator.limits,
        'valid_samples': len(arrays['time_s']), 'duration_s': float(arrays['time_s'][-1]),
        'reset': prepared['reset'], 'geometry_candidate': prepared.get('geometry_candidate'),
        'initial_authoring_diagnostics': prepared['initial_diagnostics'],
        'native_endpoints': endpoints, 'original_terminal_window': window,
        'passive_run_checks': passive_checks, 'passive_run_valid': all(passive_checks.values()),
        'initial_and_every_terminal_frame_strain': rows, 'release_evidence': release,
        'task_success': False, 'physics_accepted': False, 'robot_grasp_qualified': False,
        'limits': ['Object-only passive reset; no operation credit, task score or manipulation acceptance. A terminal-window pass in an incomplete/nonpassive run is unqualified.',
            'Original terminal geometry/stability/control-mask rule runs on every final-window frame; no new scorer or thresholds.',
            'Cooked robot audit checks initial and every terminal frame; center surfaces only, not measured contact forces or finite-thickness clearance.',
            'Self-intersections checked at native initial/final only; no whole-trajectory or continuous-time collision certificate.',
            'Failure of exact-target relaxation does not prove the original tolerance region is unreachable.']}
    if args.output.resolve() in {args.report.resolve(), args.prepared.resolve()} or args.output.exists():
        raise ValueError('Choose a fresh separate analysis path')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, result)
    return {'output': str(args.output), 'native_complete': result['native_complete'],
        'passive_run_valid': result['passive_run_valid'],
        'terminal': {k: window[k] for k in ('passed', 'stable', 'geometry_passed', 'complete_release')},
        'robot_clearance': collision['status'], 'robot_terminal_fully_separated': robot_release['terminal_fully_separated'],
        'task_success': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    p = commands.add_parser('prepare')
    p.add_argument('--original-report', type=Path, required=True)
    p.add_argument('--reset', choices=('unfolded', 'analytic-goal', 'geometry-candidate'), required=True)
    p.add_argument('--candidate-vertices', type=Path)
    p.add_argument('--candidate-evidence', type=Path)
    p.add_argument('--surface-materials-json', type=Path)
    p.add_argument('--output', type=Path, required=True)
    p.set_defaults(func=prepare)
    p = commands.add_parser('analyze')
    p.add_argument('--prepared', type=Path, required=True)
    p.add_argument('--report', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.set_defaults(func=analyze)
    args = parser.parse_args()
    print(json.dumps(args.func(args), indent=2))


if __name__ == '__main__':
    main()
