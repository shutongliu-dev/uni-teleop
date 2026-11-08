"""Audit an immutable, complete native laundry stream with CUDA and its frozen scorer.

No native simulation or policy runs here. Compressed NPZ members are expanded to
temporary memory-mapped arrays for sequential I/O. Missing provenance or native
readback yields unknown, never an invented gravity/constraint/collision pass.
An earlier latched task success must also survive an independent check of the
actual terminal stability interval with the same frozen goal and thresholds.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import re
import shutil
import tempfile
import time
import zipfile

import numpy as np

from bench.isaacsim.release import checked, digest, dump
from bench.paths import DATA_ROOT, load_json
from scripts.maintenance.freeze_project_sources import verify, verify_if_snapshot
from scenes.workrooms.laundry_room.terminal_window import terminal_window
from scenes.workrooms.laundry_room.rules import (thresholds, LEGACY_PROFILE, PRACTICAL_PROFILE,
    GARMENT_PROFILES, GARMENT_TASK_IDS, GARMENT_FOLDING_TASK_IDS, PRACTICAL_PROFILES, rule_profile)
from scripts.validation.laundry_initialization import initialization_mode, validate_author_only, authored_arrays

PROJECT = Path(__file__).resolve().parents[2]
DATASETS = Path(f'{DATA_ROOT}/datasets')
SCHEMA = 'laundry_complete_stream_audit.v1'
NATIVE_SCHEMA = 'laundry_native_stream.v1'
ATTACHMENT_MODES = {'native_plain_xform_attachment', 'native_omniphysics_vertex_xform_attachment'}
NODAL_MODE = 'engineering_local_nodal_boundary'


class EvidenceUnknown(ValueError):
    """A required observation or its provenance cannot be established."""


def require(condition, message):
    if not condition:
        raise EvidenceUnknown(message)


def artifact(path, expected=None):
    path = Path(path).resolve(strict=True)
    require(path.is_file(), 'Evidence must be a regular file: '+str(path))
    sha = digest(path)
    if expected is not None:
        require(sha == expected, 'Recorded evidence hash changed: '+str(path))
    return {'path': str(path), 'sha256': sha, 'bytes': path.stat().st_size}


def bound_artifact(row, inputs):
    require(isinstance(row, dict) and set(row) >= {'path', 'sha256'}, 'Missing path/hash evidence binding')
    value = artifact(row['path'], row['sha256'])
    inputs[value['path']] = value['sha256']
    return Path(value['path'])


class NativeArchive:
    """Read NPZ safely without retaining every full cloth array in RAM."""

    def __init__(self, path, directory):
        self.path, self.directory = Path(path), Path(directory)

    def __enter__(self):
        self.temp = tempfile.TemporaryDirectory(prefix='.native-array-cache-', dir=self.directory)
        self.arrays = {}
        try:
            with zipfile.ZipFile(self.path) as archive:
                names = archive.namelist()
                require(len(set(names)) == len(names), 'Duplicate NPZ members')
                for name in names:
                    require(name == Path(name).name and name.endswith('.npy'), 'Unsafe or unsupported NPZ member')
                    destination = Path(self.temp.name)/name
                    with archive.open(name) as source, destination.open('xb') as target:
                        shutil.copyfileobj(source, target, 8*1024*1024)
                    array = np.load(destination, mmap_mode='r', allow_pickle=False)
                    require(array.dtype.kind in 'biuf', 'Only numeric native arrays are supported')
                    self.arrays[name[:-4]] = array
            return self.arrays
        except BaseException:
            self.__exit__(None, None, None)
            raise

    def __exit__(self, *_):
        for array in self.arrays.values():
            if getattr(array, '_mmap', None) is not None:
                array._mmap.close()
        self.arrays.clear()
        self.temp.cleanup()


def verify_native_bindings(run, states, release_root, scene, task, inputs):
    """Bind actual producer inputs/source; a summary 'passed' is never read."""
    require(run.get('audit_contract') == NATIVE_SCHEMA, 'Native run lacks the complete-stream evidence contract')
    require(run.get('task_id') == task['id'], 'Native run and frozen task differ')
    archive = bound_artifact(run['state_archive'], inputs)
    require(archive == Path(states).resolve(), 'Requested states differ from the native report')
    require(run.get('complete') is True and 'error' not in run,
            'Native producer did not complete its full recording and final checks')
    reference = run['release']
    reference_path = Path(reference['path']).resolve()
    if reference_path.name == 'release_manifest.json':
        reference_path = reference_path.parent
    require(reference_path == release_root, 'Native run used a different release directory')
    sha = reference.get('manifest_sha256', reference.get('sha256'))
    require(isinstance(sha, str), 'Native report lacks the release manifest hash')
    bound_artifact({'path': str(release_root/'release_manifest.json'), 'sha256': sha}, inputs)
    for key, filename in (('scene_input', 'scene.json'), ('task_input', 'task.json')):
        path = bound_artifact(run[key], inputs)
        require(path == release_root/filename, 'Native input differs from its frozen release: '+key)
    source_manifest = bound_artifact(run['source_snapshot_manifest'], inputs)
    require(source_manifest.name == 'source_snapshot_manifest.json', 'Native source manifest identity differs')
    original_source = verify(source_manifest.parent)
    source = load_json(source_manifest)
    for key in ('source_snapshot', 'source_snapshot_after_run'):
        receipt = run.get(key)
        require(isinstance(receipt, dict) and receipt.get('verified') is True
                and receipt.get('runtime_imports_verified') is True
                and receipt.get('snapshot_root') == str(source_manifest.parent),
                'Native source/runtime import verification receipt is missing: '+key)
    entry = 'scripts/validation/laundry_cloth_interactions.py'
    require(entry in source['files_sha256'], 'Native source snapshot omits the engineering entrypoint')
    bound_artifact({'path': str(source_manifest.parent/entry), 'sha256': source['files_sha256'][entry]}, inputs)
    if initialization_mode(run) == 'author-only':
        helper = 'scripts/validation/laundry_initialization.py'
        require(helper in source['files_sha256'], 'Native source omits the author-only initialization contract')
        bound_artifact({'path': str(source_manifest.parent/helper), 'sha256': source['files_sha256'][helper]}, inputs)
    require(run.get('physics_dt_s') == task['evaluation']['physics_dt_s'], 'Native and frozen physics periods differ')
    native = run['native_evidence']
    require(isinstance(native.get('tensor_device'), str) and re.fullmatch(r'cuda(?::[0-9]+)?', native['tensor_device']),
            'Actual native tensor device does not establish CUDA execution')
    require(native.get('gpu_dynamics_enabled') is True and native.get('broadphase_type') == 'GPU',
            'Actual native PhysicsScene does not establish GPU dynamics/broadphase')
    gravity = np.asarray(native.get('gravity_m_s2'), dtype=float)
    require(gravity.shape == (3,) and np.isfinite(gravity).all()
            and np.allclose(gravity, scene['physics']['gravity_m_s2'], atol=1.e-6, rtol=0)
            and gravity[2] < 0, 'Native gravity readback is missing or differs')
    cloths = native['cloths']
    specs = {row['id']: row for row in task['asset_metadata']['cloths']}
    require(set(cloths) == set(specs), 'Native cloth inventory differs')
    for name, spec in specs.items():
        row = cloths[name]
        require(row.get('enabled') is True and row.get('schema') == 'native_surface_deformable',
                'Native live surface-deformable identity is missing: '+name)
        require(row.get('disable_gravity') is False and row.get('self_collision') is True,
                'Native cloth gravity/self-collision readback is missing: '+name)
        require(row.get('kinematic_enabled') is False,
                'Native cloth is kinematic or its dynamic flag is unknown: '+name)
        require(isinstance(row.get('applied_schemas'), list)
                and {'OmniPhysicsDeformableBodyAPI', 'OmniPhysicsSurfaceDeformableSimAPI',
                     'PhysxSurfaceDeformableBodyAPI'} <= set(row['applied_schemas']),
                'Actual native surface-deformable schemas are missing: '+name)
        require(type(row.get('vertex_count')) is int
                and row['vertex_count'] == len(spec['rest_vertices_m']),
                'Actual native tensor vertex count differs: '+name)
        require(type(row.get('surface_thickness_m')) in (float, int)
                and abs(row['surface_thickness_m']-spec['physical_thickness_m']) <= 1.e-8,
                'Actual native material thickness differs: '+name)
        verify_solver_options(scene, name, row)
    boundary = run['engineering_boundary']
    mode = boundary.get('mode')
    require(mode in ATTACHMENT_MODES | {NODAL_MODE}, 'Unsupported or undeclared local engineering control mode')
    if mode == NODAL_MODE:
        require(boundary.get('uncontrolled_input_arrays_bitwise_unchanged') is True,
                'Uncontrolled native input positions/velocities were not verified unchanged')
    else:
        require(boundary.get('cloth_position_writes_after_reset') is False
                and boundary.get('cloth_velocity_writes_after_reset') is False,
                'Native attachment control must not write cloth state after reset')
        attachments = attachment_registry(boundary, specs)
        require(native.get('attachment_inventory') == attachments,
                'Actual stage attachment inventory differs from the declared cloth/vertex/anchor registry')
    require(type(boundary.get('post_release_nodal_write_steps')) is int
            and boundary['post_release_nodal_write_steps'] == 0,
            'No-write release has not been established')
    require(type(boundary.get('nodal_write_steps')) is int and boundary['nodal_write_steps'] >= 0,
            'Actual nodal write count is missing')
    require(run.get('native_evidence_final') == native,
            'Final native readback is missing or physics/cloth/constraint configuration changed')
    return {'original_source_verified': original_source, 'native_readback': deepcopy(native),
        'native_readback_final': deepcopy(run['native_evidence_final']),
        'native_source_import_receipts': {key: deepcopy(run[key])
            for key in ('source_snapshot', 'source_snapshot_after_run')},
        'engineering_boundary': deepcopy(boundary),
        'trust_scope': 'Recorded native readback and actual write/constraint receipts bound to the complete original source snapshot; not inferred from the reported task outcome.'}


def verify_solver_options(scene, name, native_cloth):
    """Validate explicit solver priors without inventing legacy native receipts."""
    spec = next((row for row in scene.get('cloths', ()) if row['id'] == name), None)
    require(spec is not None, 'Frozen scene omits the native cloth: '+name)
    options = {**scene.get('isaacsim', {}), **spec.get('isaacsim', {})}
    verify_material_damping(options, name, native_cloth)
    verify_surface_sleep_settings(options, name, native_cloth)
    filter_key = 'cloth_self_collision_filter_distance_m'
    settling_key = 'cloth_settling_threshold_m_s'
    if filter_key not in options and settling_key not in options and 'solver_attributes' not in native_cloth:
        return  # Historical v1 streams did not emit this optional readback.
    attributes = native_cloth.get('solver_attributes')
    require(isinstance(attributes, dict), 'Actual native cloth solver attributes are missing: '+name)
    fields = (
        ('cloth_iterations', 'solverPositionIterationCount', int, 32),
        ('cloth_linear_damping', 'linearDamping', float, .10),
        ('cloth_collision_pair_update_frequency', 'collisionPairUpdateFrequency', int, None),
        ('cloth_collision_iteration_multiplier', 'collisionIterationMultiplier', int, None),
        ('cloth_speculative_ccd', 'enableSpeculativeCCD', bool, None),
        (filter_key, 'selfCollisionFilterDistance', float, None),
        (settling_key, 'settlingThreshold', float, None),
    )
    for key, suffix, kind, default in fields:
        expected = options.get(key, default)
        if expected is None:
            continue  # An unspecified native default is not a frozen numeric prior.
        attr = 'physxDeformableBody:'+suffix
        actual = attributes.get(attr)
        if kind is float:
            matches = (type(actual) in (int, float) and np.isfinite(actual)
                       and np.isclose(actual, expected, rtol=1.e-6, atol=1.e-8))
        else:
            matches = type(actual) is kind and actual == expected
        require(matches, 'Actual native solver option differs or is missing: '+name+'/'+attr)


def verify_surface_sleep_settings(options, name, native_cloth):
    """Require the explicit threshold and unchanged default sleep policy receipts."""
    from bench.simulation.isaacsim_backend.cloth_sleep import (
        SETTLING_THRESHOLD_OPTION, SLEEP_DEFAULTS, surface_settling_properties,
    )
    if SETTLING_THRESHOLD_OPTION not in options and 'usd_sleep_settings' not in native_cloth:
        return  # Historical releases did not record schema fallback values.
    try:
        properties = surface_settling_properties(options)
    except ValueError as error:
        raise EvidenceUnknown(str(error)) from error
    settings = native_cloth.get('usd_sleep_settings')
    require(isinstance(settings, dict), 'Actual USD sleep settings are missing: '+name)
    for suffix, fallback in SLEEP_DEFAULTS.items():
        attr = 'physxDeformableBody:'+suffix
        row = settings.get(suffix)
        require(isinstance(row, dict), 'Actual USD sleep setting is missing: '+name+'/'+attr)
        actual, expected = row.get('value'), properties.get(attr, fallback)
        require(type(actual) in (int, float) and np.isfinite(actual)
                and np.isclose(actual, expected, rtol=1.e-6, atol=1.e-8),
                'Actual USD sleep setting differs or is missing: '+name+'/'+attr)
        require(row.get('authored') is (attr in properties),
                'Actual USD sleep authored opinion differs or is missing: '+name+'/'+attr)


def verify_material_damping(options, name, native_cloth):
    """Explicit material priors require actual initial/final native readback."""
    from bench.simulation.isaacsim_backend.cloth_materials import (
        DAMPING_OPTIONS, surface_damping_properties,
    )
    if not any(key in options for key in DAMPING_OPTIONS) and 'material_attributes' not in native_cloth:
        return  # Preserve old streams without newly introduced priors/receipts.
    actual = native_cloth.get('material_attributes')
    require(isinstance(actual, dict), 'Actual native material attributes are missing: '+name)
    try:
        expected = surface_damping_properties(options)
    except ValueError as error:
        raise EvidenceUnknown(str(error)) from error
    for attribute, value in expected.items():
        observed = actual.get(attribute)
        require(type(observed) in (int, float) and np.isfinite(observed) and observed >= 0
                and np.isclose(observed, value, rtol=1.e-6, atol=1.e-8),
                'Actual native material option differs or is missing: '+name+'/'+attribute)


def attachment_registry(boundary, specs):
    rows = boundary.get('attachments')
    require(isinstance(rows, list) and len(rows) > 0, 'Native attachment registry is missing')
    paths = []
    for row in rows:
        require(isinstance(row, dict) and set(row) == {'path', 'cloth_id', 'vertex_ids', 'anchor_path'},
                'Every attachment requires exact path, cloth, node and anchor identities')
        require(row['cloth_id'] in specs and all(isinstance(row[k], str) and row[k].startswith('/World/') for k in ('path', 'anchor_path')),
                'Attachment references an unknown cloth or non-world anchor/path')
        ids = row['vertex_ids']
        n = len(specs[row['cloth_id']]['rest_vertices_m'])
        require(isinstance(ids, list) and len(ids) > 0 and len(set(ids)) == len(ids)
                and all(type(i) is int and 0 <= i < n for i in ids), 'Attachment node IDs are invalid')
        paths.append(row['path'])
    require(len(set(paths)) == len(paths) and boundary.get('native_attachment_paths') == paths,
            'Attachment paths are duplicated or disagree with their enabled-column order')
    return rows


def validate_attachment_masks(arrays, boundary, specs, count):
    rows = attachment_registry(boundary, specs)
    enabled = arrays.get('attachment_enabled')
    require(enabled is not None and enabled.shape == (count, len(rows)) and enabled.dtype.kind == 'b',
            'Actual per-step attachment enabled readback is required')
    require(not enabled[-1].any(), 'Final native attachment state is not fully released')
    for name, spec in specs.items():
        owners = {}
        for index, row in enumerate(rows):
            if row['cloth_id'] == name:
                for vertex in row['vertex_ids']:
                    owners.setdefault(vertex, []).append(index)
        allowed = np.zeros(len(spec['rest_vertices_m']), dtype=bool)
        allowed[list(owners)] = True
        mask = arrays[name+'_control_mask']
        require(not mask[:, ~allowed].any(), 'Control mask contains vertices absent from actual attachment inventory')
        for vertex, columns in owners.items():
            active = enabled[:, columns].sum(axis=1)
            require(np.all(active <= 1), 'A cloth node has multiple simultaneously enabled attachments')
            require(np.array_equal(mask[:, vertex], active.astype(bool)),
                    'Control mask differs from actual enabled native attachment nodes')


def validate_stream(arrays, run, task, *, scene=None, inputs=None):
    initialization = initialization_mode(run)
    specs = {row['id']: row for row in task['asset_metadata']['cloths']}
    required = {'time_s', 'native_step', 'position_write_calls', 'velocity_write_calls', 'hidden_constraints'}
    required |= {name+suffix for name in specs for suffix in ('_q', '_qd', '_control_mask', '_triangles')}
    require(required <= arrays.keys(), 'Missing complete-stream arrays: '+str(sorted(required-arrays.keys())))
    times, steps = arrays['time_s'], arrays['native_step']
    require(times.ndim == steps.ndim == 1 and len(times) >= 2 and steps.dtype.kind in 'iu',
            'Time/native-step axes must contain reset and at least one real step')
    count = len(times)
    require(len(steps) == count and np.array_equal(steps, np.arange(count)),
            'Native stream must include reset zero and every consecutive step')
    dt = task['evaluation']['physics_dt_s']
    require(np.isfinite(times).all() and np.allclose(times, steps*dt, rtol=0, atol=1.e-8),
            'Native timestamps disagree with the frozen period')
    require(run.get('native_steps') == count-1, 'Native report and archive step counts differ')
    active_cloths = np.zeros(count, dtype=np.int64)
    mode = run['engineering_boundary'].get('mode')
    require(mode in ATTACHMENT_MODES | {NODAL_MODE}, 'Missing or unsupported native control mode')
    for name, spec in specs.items():
        n = len(spec['rest_vertices_m'])
        q, qd, mask, triangles = (arrays[name+suffix] for suffix in ('_q', '_qd', '_control_mask', '_triangles'))
        require(q.shape == qd.shape == (count, n, 3) and q.dtype.kind == qd.dtype.kind == 'f',
                'Native cloth position/velocity shape or dtype differs: '+name)
        require(mask.shape == (count, n) and mask.dtype.kind == 'b', 'Explicit per-vertex boolean control mask required: '+name)
        require(np.all(mask.sum(axis=1) <= .10*n), 'Engineering control exceeded ten percent of a cloth: '+name)
        require(triangles.dtype.kind in 'iu' and np.array_equal(triangles, np.asarray(spec['triangles'])),
                'Native cloth topology differs from frozen semantic vertex IDs: '+name)
        if initialization == 'tensor-reset':
            require(np.isfinite(q[0]).all() and np.max(np.abs(q[0]-np.asarray(spec['initial_vertices_m']))) <= 5.e-5,
                    'Reset-zero native cloth differs from the frozen canonical initialization: '+name)
        if mode == NODAL_MODE:
            require(not mask[0].any(), 'Reset baseline cannot include a nodal engineering write')
        active_cloths += mask.any(axis=1)
    for key in ('position_write_calls', 'velocity_write_calls', 'hidden_constraints'):
        value = arrays[key]
        require(value.shape == (count,) and value.dtype.kind in 'iu' and np.all(value >= 0),
                'Actual per-step integer receipt required: '+key)
    if mode in ATTACHMENT_MODES:
        validate_attachment_masks(arrays, run['engineering_boundary'], specs, count)
        for key in ('position_write_calls', 'velocity_write_calls'):
            require(not arrays[key].any(), 'Native attachment run contains nodal state writes: '+key)
    else:
        for key in ('position_write_calls', 'velocity_write_calls'):
            require(np.array_equal(arrays[key], active_cloths),
                    'Actual nodal setter count disagrees with active cloth control masks: '+key)
    require(np.all(arrays['hidden_constraints'] == 0), 'Native constraint inventory contains undeclared constraints')
    require(run['engineering_boundary']['nodal_write_steps'] == int(arrays['position_write_calls'].sum()),
            'Native accumulated write count differs from per-step receipts')
    require(not active_cloths[-1], 'The native stream ends before all engineering control is released')
    initialization_receipt = None
    if initialization == 'author-only':
        require(scene is not None and inputs is not None, 'Author-only audit requires frozen scene and bound evidence inputs')
        initialization_receipt = validate_author_only(arrays, run, scene, task, lambda row: bound_artifact(row, inputs))
    result = {'frames_including_reset': count, 'native_steps': count-1, 'physics_dt_s': dt,
        'duration_s': float(times[-1]), 'control_mode': mode,
        'controlled_native_steps': int(np.count_nonzero(active_cloths[1:])),
        'reset_attachments_enabled': bool(active_cloths[0]) if mode in ATTACHMENT_MODES else False,
        'position_write_calls': int(arrays['position_write_calls'].sum()),
        'velocity_write_calls': int(arrays['velocity_write_calls'].sum()),
        'hidden_constraints_max': int(arrays['hidden_constraints'].max())}
    if initialization_receipt is not None:
        result['initialization'] = initialization_receipt
    return result


def table_penetration(vertices, scene, thickness_by_id):
    center = np.asarray(scene['table']['position_m'])
    half = np.asarray(scene['table']['half_extents_m'])
    top = float(center[2]+half[2])
    maximum = 0.
    for name, points in vertices.items():
        inside = np.all(np.abs(points[:, :2]-center[:2]) <= half[:2], axis=1)
        if inside.any():
            maximum = max(maximum, float(np.max(top+thickness_by_id[name]/2-points[inside, 2])))
    return max(0., maximum)


def audit_authored_geometry(vertices, scene, task, evaluator, oracle, controlled_vertices):
    """Same physical limits before warmup; setup cannot erase an invalid asset."""
    specs = {row['id']: row for row in task['asset_metadata']['cloths']}
    require(set(vertices) == set(specs) and all(np.isfinite(v).all() for v in vertices.values()),
            'Invalid authored geometry inventory or coordinates')
    try:
        for name in specs:
            evaluator._check_local_mask(name, controlled_vertices[name])
        evaluator._check_world_grasp_regions(vertices, controlled_vertices)
    except ValueError as error:
        raise EvidenceUnknown('Authored warmup control violates frozen grasp locality: '+str(error)) from error
    geometry = oracle.check(vertices)
    require(geometry.get('finite_thickness_compression_checked') is True,
            'GPU oracle did not check authored finite thickness')
    keys = ('self_intersections', 'intercloth_intersections', 'degenerate_triangles')
    require(all(type(geometry.get(key)) is int and geometry[key] >= 0 for key in keys),
            'Invalid authored CUDA collision evidence')
    cloth_depth = float(geometry['max_cloth_envelope_penetration_m'])
    require(np.isfinite(cloth_depth) and cloth_depth >= 0, 'Invalid authored thickness evidence')
    depth = table_penetration(vertices, scene, {name: row['physical_thickness_m'] for name, row in specs.items()})
    mesh = {name: evaluator._mesh_metrics(name, v) for name, v in vertices.items()}
    limits = thresholds('towel_stacking' if task['id'] in (*GARMENT_TASK_IDS, *GARMENT_FOLDING_TASK_IDS) else task['id'], LEGACY_PROFILE)
    valid = (not any(geometry[key] for key in keys) and max(depth, cloth_depth) <= limits['max_penetration_m']
        and all(row['max_edge_stretch_ratio'] <= limits['max_edge_stretch_ratio']
                and row['min_triangle_area_ratio'] >= limits['min_triangle_area_ratio'] for row in mesh.values()))
    return {'passed': bool(valid), 'geometry': geometry, 'mesh': mesh,
        'table_envelope_penetration_m': depth, 'operation_credit': False,
        'scope': 'Actual authored USD geometry before the two internal warmup steps; warmup intermediate states remain unobserved.'}


def audit_frames(arrays, scene, task, evaluator, oracle, trace, *, authored_vertices=None):
    """Sequential complete-stream replay; a terminal score never stops audit."""
    specs = {row['id']: row for row in task['asset_metadata']['cloths']}
    thickness = {name: row['physical_thickness_m'] for name, row in specs.items()}
    counts = {'self_intersections': 0, 'intercloth_intersections': 0, 'degenerate_triangles': 0}
    maxima = {'table_envelope_penetration_m': 0., 'cloth_envelope_penetration_m': 0.,
              'max_penetration_m': 0., 'max_edge_stretch_ratio': 0.}
    first_bad = first_success = None
    authored_geometry = None
    if authored_vertices is not None:
        authored_geometry = audit_authored_geometry(authored_vertices, scene, task, evaluator, oracle,
            {name: np.flatnonzero(arrays[name+'_control_mask'][0]).tolist() for name in specs})
        if not authored_geometry['passed']:
            first_bad = {'phase': 'authored_initial', 'native_step': None, 'time_s': None, **authored_geometry}
    total = len(arrays['time_s'])
    for index in range(total):
        vertices = {name: np.asarray(arrays[name+'_q'][index]) for name in specs}
        velocities = {name: np.asarray(arrays[name+'_qd'][index]) for name in specs}
        require(all(np.isfinite(v).all() for v in (*vertices.values(), *velocities.values())),
                'Nonfinite native cloth state at step '+str(index))
        geometry = oracle.check(vertices)
        if 'crossing_rest_area_fraction_max' in task['evaluation']['thresholds']:
            from scenes.workrooms.laundry_room.garment_crossings import crossing_measure
            geometry['self_crossing_measure'] = crossing_measure(vertices['towel'], specs['towel']['triangles'],
                specs['towel']['rest_vertices_m'], geometry.get('self_intersection_pairs', []),
                pair_count=geometry['self_intersections'],
                complete=geometry.get('intersection_pair_inventory_complete', False))
        require(geometry.get('finite_thickness_compression_checked') is True,
                'GPU oracle did not provide finite-thickness evidence')
        for key in counts:
            require(type(geometry.get(key)) is int and geometry[key] >= 0, 'Invalid CUDA collision result: '+key)
            counts[key] += geometry[key]
        table_depth = table_penetration(vertices, scene, thickness)
        cloth_depth = float(geometry['max_cloth_envelope_penetration_m'])
        require(np.isfinite(cloth_depth) and cloth_depth >= 0., 'Invalid CUDA thickness result')
        depth = max(table_depth, cloth_depth)
        mesh = {name: evaluator._mesh_metrics(name, v) for name, v in vertices.items()}
        for name in specs:
            evaluator._check_local_mask(name, np.flatnonzero(arrays[name+'_control_mask'][index]).tolist())
        evaluator._check_world_grasp_regions(vertices, {name: np.flatnonzero(arrays[name+'_control_mask'][index]).tolist() for name in specs})
        maxima['table_envelope_penetration_m'] = max(maxima['table_envelope_penetration_m'], table_depth)
        maxima['cloth_envelope_penetration_m'] = max(maxima['cloth_envelope_penetration_m'], cloth_depth)
        maxima['max_penetration_m'] = max(maxima['max_penetration_m'], depth)
        maxima['max_edge_stretch_ratio'] = max(maxima['max_edge_stretch_ratio'], *(row['max_edge_stretch_ratio'] for row in mesh.values()))
        limits = thresholds('towel_stacking' if task['id'] in (*GARMENT_TASK_IDS, *GARMENT_FOLDING_TASK_IDS) else task['id'], LEGACY_PROFILE)
        valid = (not any(geometry[key] for key in counts) and depth <= limits['max_penetration_m']
            and all(row['max_edge_stretch_ratio'] <= limits['max_edge_stretch_ratio']
                    and row['min_triangle_area_ratio'] >= limits['min_triangle_area_ratio'] for row in mesh.values()))
        if not valid and first_bad is None:
            first_bad = {'native_step': index, 'time_s': float(arrays['time_s'][index]),
                         'geometry': geometry, 'mesh': mesh, 'table_envelope_penetration_m': table_depth}
        frame = {'schema': 'laundry_native_frame.v1', 'complete': True,
            'native_step': index, 'time_s': float(arrays['time_s'][index]),
            'control_mode': 'engineering_local_grasp',
            'controlled_vertices': {name: np.flatnonzero(arrays[name+'_control_mask'][index]).tolist() for name in specs},
            'cloths': {name: {'enabled': True, 'vertices_m': vertices[name],
                'velocities_m_s': velocities[name], 'triangles': arrays[name+'_triangles']} for name in specs},
            # These booleans are conclusions of verify_native_bindings and
            # validate_stream; missing producer evidence never reaches replay.
            'physics': {'complete': True, 'valid': bool(valid), 'validity_scope': 'strict_geometry', 'gravity_enabled': True,
                'hidden_constraints': int(arrays['hidden_constraints'][index]),
                'self_intersections': geometry['self_intersections'],
                'intercloth_intersections': geometry['intercloth_intersections'], 'max_penetration_m': depth,
                **({'self_crossing_measure': geometry['self_crossing_measure']} if 'self_crossing_measure' in geometry else {})}}
        score = evaluator.observe(frame)
        if score['success'] and first_success is None:
            first_success = {'native_step': index, 'time_s': frame['time_s']}
        row = {'native_step': index, 'time_s': frame['time_s'], 'geometry': geometry, 'mesh': mesh,
            'table_envelope_penetration_m': table_depth, 'max_penetration_m': depth,
            'physical_sample_valid': bool(valid), 'frozen_score_status': score['status'],
            'frozen_score_reason': score['reason'], 'released_stable_s': score['released_stable_s']}
        trace.write(json.dumps(row, allow_nan=False, separators=(',', ':'))+'\n')
        if index and index % 2000 == 0:
            print(json.dumps({'event': 'audit_progress', 'native_step': index,
                'frames': total, 'physical_failure_seen': first_bad is not None}), flush=True)
    result = evaluator.report()
    terminal = terminal_window(arrays, evaluator)
    result = {'samples_checked': total, 'all_native_samples_checked': True,
        'task_rule_profile': rule_profile(task), 'physical_diagnostic_profile': LEGACY_PROFILE,
        'physical_geometry_stream_passed': first_bad is None, 'intersection_pair_observations': counts,
        'maxima': maxima, 'first_physical_failure': first_bad, 'first_frozen_task_success': first_success,
        'frozen_evaluator_result': result,
        'terminal_window_required': True, 'terminal_window': terminal,
        'accepted_engineering_task_success': bool((first_bad is None or rule_profile(task) in PRACTICAL_PROFILES)
            and result['success'] and terminal['passed']),
        'scope': 'Sampled triangle intersection and finite-thickness envelopes plus frozen task scoring and the actual terminal goal/release/stability window; no continuous CCD, force calibration, full-room collision audit or robot grasp proof.',
        'crease_thickness_certified': False}
    if authored_geometry is not None:
        result['authored_initial_geometry'] = authored_geometry
        result['intermediate_warmup_states_certified'] = False
    return result


def run(args):
    required_modules = ('scripts.validation.audit_laundry_cloth', 'bench.isaacsim.release',
                        'scenes.workrooms.laundry_room.terminal_window', 'scripts.validation.laundry_initialization')
    source = verify_if_snapshot(PROJECT, required_modules=required_modules, entrypoint_file=Path(__file__))
    if source is None:
        raise ValueError('Complete-stream audit must run from a complete frozen project source snapshot')
    output = args.output.resolve()
    require(output.is_relative_to(DATASETS) and not output.exists(), 'Choose a fresh output under shared datasets')
    output.mkdir(parents=True)
    started = time.monotonic()
    inputs = {}
    report = {'schema': SCHEMA, 'status': 'unknown', 'passed': False,
        'accepted_engineering_task_success': False, 'robot_manipulation_qualified': False,
        'terminal_window_required': True,
        'physics_accepted': False, 'new_native_simulation_performed': False,
        'audit_source': source, 'audit_source_manifest': artifact(PROJECT/'source_snapshot_manifest.json')}
    try:
        states, metadata, root = args.states.resolve(), args.run_metadata.resolve(), args.release.resolve()
        for path in (states, metadata, root):
            require(path.is_relative_to(DATASETS), 'Native evidence and release must reside under shared datasets')
        run_identity = artifact(metadata)
        inputs[run_identity['path']] = run_identity['sha256']
        state_identity = artifact(states)
        inputs[state_identity['path']] = state_identity['sha256']
        run_metadata = load_json(metadata)
        from bench.isaacsim.release import task_inputs, load_evaluator, validate
        manifest = validate(root)
        require(manifest.get('schema') == 'laundry_cloth_task_release.v1', 'Expected a frozen laundry release')
        _, scene, task, _ = task_inputs(root, manifest['task_id'])
        report['task_id'] = task['id']
        report['native_provenance'] = verify_native_bindings(run_metadata, states, root, scene, task, inputs)
        for relative, sha in manifest['files_sha256'].items():
            inputs[str(root/relative)] = checked(root/relative, sha)
        for path, sha in manifest['asset_files_sha256'].items():
            inputs[path] = checked(path, sha)
        evaluator = load_evaluator(root)(task)
        with NativeArchive(states, output) as arrays:
            report['stream_contract'] = validate_stream(arrays, run_metadata, task, scene=scene, inputs=inputs)
            authored_vertices = None
            if initialization_mode(run_metadata) == 'author-only':
                from pxr import Usd
                authored_stage = Usd.Stage.Open(str(bound_artifact(run_metadata['initialization']['authored_stage'], inputs)))
                authored = authored_arrays(authored_stage, {row['id']: row for row in task['asset_metadata']['cloths']})
                authored_vertices = {row['id']: authored[row['id']+'_authored_q'] for row in task['asset_metadata']['cloths']}
            from bench.evaluation.cloth.gpu_intersections import ClothIntersectionAudit
            required_modules += ('bench.evaluation.cloth.gpu_intersections',)
            specs = {row['id']: row for row in task['asset_metadata']['cloths']}
            oracle = ClothIntersectionAudit({name: row['triangles'] for name, row in specs.items()},
                {name: len(row['rest_vertices_m']) for name, row in specs.items()},
                thickness_by_id={name: row['physical_thickness_m'] for name, row in specs.items()}, device=args.device,
                **({'intersection_pair_capacity': 1024} if 'crossing_rest_area_fraction_max' in task['evaluation']['thresholds'] else {}))
            require(oracle.device.is_cuda, 'Full-stream geometry audit requires an approved CUDA device')
            report['audit_device'] = str(oracle.device)
            with (output/'frame_audit.jsonl').open('x') as trace:
                report.update(audit_frames(arrays, scene, task, evaluator, oracle, trace, authored_vertices=authored_vertices))
        for path, sha in inputs.items():
            checked(path, sha)
        report['original_source_final'] = verify(Path(report['native_provenance']['original_source_verified']['snapshot_root']))
        report['audit_source_final'] = verify_if_snapshot(PROJECT, required_modules=required_modules, entrypoint_file=Path(__file__))
        report['frame_audit'] = artifact(output/'frame_audit.jsonl')
        report['passed'] = report['accepted_engineering_task_success']
        report['status'] = 'passed' if report['passed'] else 'unknown' if report['frozen_evaluator_result']['status'] == 'unknown' else 'failed'
    except (EvidenceUnknown, KeyError, TypeError, ValueError, OSError, RuntimeError, zipfile.BadZipFile) as error:
        report.update(status='unknown', passed=False, accepted_engineering_task_success=False,
                      error={'type': type(error).__name__, 'message': str(error)})
        if (output/'frame_audit.jsonl').is_file():
            report['partial_frame_audit'] = artifact(output/'frame_audit.jsonl')
            report['all_native_samples_checked'] = False
    report['inputs_sha256'] = inputs
    report['elapsed_wall_s'] = time.monotonic()-started
    dump(output/'report.json', report)
    print(json.dumps({'status': report['status'], 'passed': report['passed'], 'output': str(output)}), flush=True)
    return 0 if report['passed'] else 2 if report['status'] == 'unknown' else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--states', type=Path, required=True)
    parser.add_argument('--run-metadata', type=Path, required=True)
    parser.add_argument('--release', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    raise SystemExit(run(parser.parse_args()))


if __name__ == '__main__':
    main()
