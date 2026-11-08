"""Derive a stack reset from strict or explicit practical native-fold evidence.

Only terminal geometry is copied. Original evidence remains external, and the
new two-cloth reset requires its own complete native qualification.
"""
from copy import deepcopy
from collections import OrderedDict
import json
from pathlib import Path
import threading
import zipfile

import numpy as np

from bench.isaacsim.release import checked, digest
from bench.paths import load_json

SCHEMA = 'laundry_towel_asset.v2'

# Keep only small terminal slices. Every caller still verifies the complete
# archive and all evidence on each validation; mtime/size never authorize reuse.
_TAIL_CACHE_BYTES = 64 * 1024 * 1024
_TAIL_CACHE_ENTRIES = 32
_TAIL_CACHE = OrderedDict()
_TAIL_CACHE_SIZE = 0
_TAIL_CACHE_LOCK = threading.Lock()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def tail_array(archive, name, count, *, total_rows=None, verified_archive_sha256=None):
    """Read a bounded terminal slice without loading/copying the large NPZ.

    An internal caller may reuse a slice only after validating the current full
    archive SHA in this validation pass. Standalone calls remain uncached. The
    return value always belongs to the caller, never to the immutable cache.
    """
    global _TAIL_CACHE_SIZE
    key = None
    if verified_archive_sha256 is not None:
        require(isinstance(verified_archive_sha256, str) and len(verified_archive_sha256) == 64
                and all(c in '0123456789abcdef' for c in verified_archive_sha256),
                'A verified archive SHA256 is required for terminal slice reuse')
        key = (str(Path(archive).resolve()), verified_archive_sha256, name, count, total_rows)
        with _TAIL_CACHE_LOCK:
            cached = _TAIL_CACHE.get(key)
            if cached is not None:
                _TAIL_CACHE.move_to_end(key)
        if cached is not None:
            return cached.copy()
    with zipfile.ZipFile(archive) as files, files.open(name+'.npy') as stream:
        version = np.lib.format.read_magic(stream)
        require(version in ((1, 0), (2, 0)), 'Unsupported native array encoding')
        reader = np.lib.format.read_array_header_1_0 if version == (1, 0) else np.lib.format.read_array_header_2_0
        shape, fortran, dtype = reader(stream)
        require(not fortran and not dtype.hasobject and dtype.kind in 'biuf'
                and shape and shape[0] >= count, 'Incomplete numeric terminal native array')
        require(total_rows is None or shape[0] == total_rows, 'Native array frame count differs from its complete audit')
        width = int(np.prod(shape[1:]))*dtype.itemsize
        stream.seek(stream.tell()+(shape[0]-count)*width)
        data = stream.read(count*width)
        require(len(data) == count*width, 'Truncated native terminal array')
        result = np.frombuffer(data, dtype=dtype).reshape((count, *shape[1:])).copy()
    if key is not None and 0 < result.nbytes <= _TAIL_CACHE_BYTES:
        cached = result.copy()
        cached.flags.writeable = False
        with _TAIL_CACHE_LOCK:
            previous = _TAIL_CACHE.pop(key, None)
            if previous is not None:
                _TAIL_CACHE_SIZE -= previous.nbytes
            while _TAIL_CACHE and (_TAIL_CACHE_SIZE + cached.nbytes > _TAIL_CACHE_BYTES
                                   or len(_TAIL_CACHE) >= _TAIL_CACHE_ENTRIES):
                _, evicted = _TAIL_CACHE.popitem(last=False)
                _TAIL_CACHE_SIZE -= evicted.nbytes
            _TAIL_CACHE[key] = cached
            _TAIL_CACHE_SIZE += cached.nbytes
    return result


def _practical_evidence(review_path, native_path, audit_path, native, audit, archive,
                        release, scene, task, evidence, bind):
    """Rebind an accepted review without relabeling its failed strict audit."""
    from scripts.validation import review_laundry_practical as practical
    from scripts.maintenance.freeze_project_sources import verify

    review_path = bind(review_path)
    review = load_json(review_path)
    require(review.get('schema') == practical.SCHEMA
            and review.get('practical_acceptance_profile') == practical.PROFILE
            and review.get('task_id') == 'towel_folding' and review.get('status') == 'accepted'
            and review.get('practical_operation_accepted') is True
            and review.get('reason_codes') == [] and 'error' not in review,
            'An accepted practical folding review with the supported profile is required')
    for key, expected in (('native_report', native_path), ('full_stream_audit', audit_path)):
        row = review.get(key, {})
        require(Path(row.get('path', '')).resolve() == expected
                and row.get('sha256') == evidence[str(expected)],
                'Practical review binds a different '+key)
    practical.audit.verify_native_bindings(native, archive, release, scene, task, evidence)
    trace = practical.verify_prior_audit(audit, native, native_path, archive, release, task, evidence)
    manifest = load_json(release/'release_manifest.json')
    for relative, sha in manifest['files_sha256'].items():
        bind(release/relative, sha)
    for path, sha in manifest['asset_files_sha256'].items():
        bind(path, sha)
    required_inputs = {path: sha for path, sha in evidence.items() if path != str(review_path)}
    inputs = review.get('inputs_sha256')
    require(isinstance(inputs, dict) and inputs, 'Practical review input bindings are missing')
    require(all(inputs.get(path) == sha for path, sha in required_inputs.items()),
            'Practical review does not bind the same complete source/native/release/frame evidence')
    for path, sha in inputs.items():
        require(str(Path(path).resolve()) == path, 'Practical review input path is not canonical')
        bind(path, sha)
    row = review['review_source_manifest']
    source_manifest = bind(row['path'], row['sha256'])
    require(source_manifest.name == 'source_snapshot_manifest.json', 'Missing practical review source identity')
    verify(source_manifest.parent)
    frozen = load_json(source_manifest)
    entry = 'scripts/validation/review_laundry_practical.py'
    entry_path = bind(source_manifest.parent/entry, frozen['files_sha256'][entry])
    for key in ('review_source', 'review_source_final'):
        receipt = review.get(key, {})
        require(receipt.get('verified') is True and receipt.get('runtime_imports_verified') is True
                and receipt.get('snapshot_root') == str(source_manifest.parent)
                and receipt.get('entrypoint', {}).get('path') == str(entry_path)
                and receipt.get('entrypoint', {}).get('sha256') == evidence[str(entry_path)],
                'Practical review source/import verification is missing or differs: '+key)
    total = native['native_steps']+1
    clock = {key: tail_array(archive, key, total, total_rows=total,
        verified_archive_sha256=evidence[str(archive)]) for key in ('native_step', 'time_s')}
    require(clock['native_step'].dtype.kind in 'iu'
            and np.array_equal(clock['native_step'], np.arange(total))
            and clock['time_s'].shape == (total,) and np.isfinite(clock['time_s']).all()
            and np.allclose(clock['time_s'], np.arange(total)*task['evaluation']['physics_dt_s'], atol=1.e-8, rtol=0),
            'Practical source native clock is not complete and consecutive')
    coverage = practical.verify_frame_trace(trace, clock, audit)
    require(review.get('prior_frame_coverage') == coverage, 'Practical review frame coverage differs')
    for key in ('frames_including_reset', 'native_steps', 'physics_dt_s'):
        require(review.get('stream_contract', {}).get(key) == audit['stream_contract'][key],
                'Practical review native stream contract differs: '+key)
    strict = {key: deepcopy(audit.get(key)) for key in ('status', 'passed',
        'accepted_engineering_task_success', 'frozen_evaluator_result', 'terminal_window')}
    require(review.get('strict_diagnostics') == strict, 'Practical review changed the original strict diagnostics')
    history = review.get('operation_history', {})
    require(history.get('operation_seen') is True and history.get('operation_displacement_seen') is True
            and history.get('reset_credit') is False,
            'Practical review does not establish an operation without reset credit')
    return review_path, review, coverage


def metadata_from_native_fold(native_report, fullstream_audit, *, cells_xy=(30, 20), physics_options=None,
                              source_fold_practical_review=None):
    from bench.isaacsim.release import validate, load_evaluator
    from scripts.maintenance.freeze_project_sources import verify
    from .assets import (candidate_metadata, DERIVED_TEXTURES, DERIVED_TEXTURE_HASHES,
                         DERIVED_TEXTURE_TILE_M)

    evidence = {}
    def bind(path, sha=None):
        path = Path(path).resolve(strict=True)
        actual = checked(path, sha) if sha is not None else digest(path)
        evidence[str(path)] = actual
        return path
    native_path, audit_path = bind(native_report), bind(fullstream_audit)
    native, audit = (load_json(path) for path in (native_path, audit_path))
    require(native.get('audit_contract') == 'laundry_native_stream.v1'
            and native.get('task_id') == 'towel_folding' and native.get('complete') is True
            and 'error' not in native, 'Source must be a completed native towel_folding run')
    practical = source_fold_practical_review is not None
    require(audit.get('schema') == 'laundry_complete_stream_audit.v1'
            and audit.get('task_id') == 'towel_folding'
            and (audit.get('status') in ('passed', 'failed') if practical else audit.get('status') == 'passed')
            and all(audit.get(key) is True for key in (('all_native_samples_checked', 'physical_geometry_stream_passed')
                if practical else ('passed', 'accepted_engineering_task_success',
                    'all_native_samples_checked', 'physical_geometry_stream_passed')))
            and audit.get('first_physical_failure') is None
            and str(audit.get('audit_device', '')).startswith('cuda:')
            and audit.get('samples_checked') == native['native_steps']+1,
            ('A complete physical geometry folding audit is required for practical derivation' if practical else
             'A passed complete-stream folding audit is required; partial or failed evidence is ineligible'))
    terminal = audit.get('terminal_window', {})
    require(audit.get('terminal_window_required') is True
            and all(terminal.get(key) is True for key in (('complete', 'complete_release', 'geometry_passed')
                if practical else ('passed', 'complete', 'complete_release', 'stable', 'geometry_passed'))),
            'Source fold audit must independently pass the actual terminal window')
    archive = bind(native['state_archive']['path'], native['state_archive']['sha256'])
    release = Path(native['release']['path']).resolve()
    if release.name == 'release_manifest.json':
        release = release.parent
    release_sha = native['release'].get('manifest_sha256', native['release'].get('sha256'))
    require(isinstance(release_sha, str) and len(release_sha) == 64, 'Missing native release identity')
    bind(release/'release_manifest.json', release_sha)
    validate(release)
    scene_path = bind(release/'scene.json', native['scene_input']['sha256'])
    task_path = bind(release/'task.json', native['task_input']['sha256'])
    require(Path(native['scene_input']['path']).resolve() == scene_path
            and Path(native['task_input']['path']).resolve() == task_path, 'Native release/input paths disagree')
    scene, task = (load_json(path) for path in (scene_path, task_path))
    require(task['id'] == 'towel_folding', 'Source release is not a folding task')
    source_profile = deepcopy(scene['isaacsim'])
    if physics_options is not None:
        require(physics_options == source_profile, 'Requested stack physics profile differs from successful native fold')
    source_manifest = load_json(release/'release_manifest.json')
    bind(release/'physics_profile.json', source_manifest['files_sha256']['physics_profile.json'])
    require(load_json(release/'physics_profile.json')['isaacsim_options'] == source_profile,
            'Source physics profile does not match its scene')
    for row in (native['source_snapshot_manifest'], audit['audit_source_manifest']):
        path = bind(row['path'], row['sha256'])
        require(path.name == 'source_snapshot_manifest.json', 'Missing complete source identity')
        verify(path.parent)
    bind(audit['frame_audit']['path'], audit['frame_audit']['sha256'])
    audited = audit.get('inputs_sha256', {})
    for path in (native_path, archive, release/'release_manifest.json', scene_path, task_path,
                 release/'physics_profile.json'):
        require(audited.get(str(path)) == evidence[str(path)], 'Audit does not bind the same native input: '+str(path))
    expected = candidate_metadata('towel_stacking', cells_xy=cells_xy)
    source_spec = task['asset_metadata']['cloths']
    require(len(source_spec) == 1 and source_spec[0]['id'] == 'towel', 'Source fold requires one live towel')
    source_spec = source_spec[0]
    for field in ('cells_xy', 'size_m', 'rest_area_m2', 'rest_vertices_m', 'triangles',
                  'mass_kg', 'areal_density_kg_m2', 'physical_thickness_m', 'dynamic_model', 'fixed_vertices'):
        require(source_spec[field] == expected['cloths'][0][field], 'Source towel physical/rest property differs: '+field)
    source_cloth = next(row for row in scene['cloths'] if row['id'] == 'towel')
    merged = {**source_profile, **source_cloth.get('isaacsim', {})}
    merged = {key: value for key, value in merged.items() if not key.startswith('cloth_visual_')}
    require(merged == {**source_profile, 'cloth_thickness_m': .003}
            and source_cloth['mass_kg'] == .108 and source_cloth['friction'] == .45
            and source_cloth['self_contact'] is True,
            'Source cloth has incompatible mass, contact or per-cloth physical overrides')
    scorer = load_evaluator(release)(task)
    if practical:
        require(scorer.limits['stable_s'] == .5 and scorer.limits['max_node_speed_m_s'] == .035
                and scorer.limits['stable_node_drift_m'] == .006, 'Practical reset requires unchanged v1 terminal limits')
        review_path, review, coverage = _practical_evidence(source_fold_practical_review,
            native_path, audit_path, native, audit, archive, release, scene, task, evidence, bind)
    dt = task['evaluation']['physics_dt_s']
    count = int(np.ceil(scorer.limits['stable_s']/dt))+1
    total_rows = native['native_steps']+1 if practical else None
    def terminal(name, rows, *, total_rows=None):
        return tail_array(archive, name, rows, total_rows=total_rows,
                          verified_archive_sha256=evidence[str(archive)])
    q = terminal('towel_q', count, total_rows=total_rows)
    qd = terminal('towel_qd', count, total_rows=total_rows)
    controls = terminal('towel_control_mask', count, total_rows=total_rows)
    steps = terminal('native_step', count)
    times = terminal('time_s', count)
    triangles = terminal('towel_triangles', len(source_spec['triangles']),
        total_rows=len(source_spec['triangles']) if practical else None)
    require(triangles.dtype.kind in 'iu' and np.array_equal(triangles, source_spec['triangles']),
            'Native topology differs from the successful source release')
    require(q.shape == qd.shape == (count, len(source_spec['rest_vertices_m']), 3)
            and q.dtype.kind == qd.dtype.kind == 'f'
            and np.isfinite(q).all() and np.isfinite(qd).all(), 'Invalid terminal native cloth arrays')
    require(controls.shape == q.shape[:2] and controls.dtype.kind == 'b' and not controls.any()
            and np.array_equal(steps, np.arange(native['native_steps']-count+1, native['native_steps']+1))
            and np.allclose(times, steps*dt, atol=1.e-8, rtol=0), 'Terminal native window is not complete and released')
    speed = float(np.linalg.norm(qd, axis=-1).max())
    pose_speed = float(np.linalg.norm(np.diff(q, axis=0), axis=-1).max()/dt)
    drift = float(np.linalg.norm(q-q[0], axis=-1).max())
    if practical:
        from scripts.validation.review_laundry_practical import practical_terminal_window
        practical_window = practical_terminal_window({'towel_q': q, 'towel_qd': qd,
            'towel_control_mask': controls, 'towel_triangles': triangles, 'native_step': steps, 'time_s': times}, scorer)
        require(practical_window['passed'], 'Final native fold fails practical terminal checks: '+','.join(practical_window['reason_codes']))
        require(practical_window == review.get('terminal_window'), 'Recomputed practical terminal window differs from its review')
        strict_window = practical_window['strict_terminal_window']
        speed, pose_speed, drift = (strict_window[key] for key in
            ('max_node_speed_m_s', 'pose_speed_m_s', 'max_node_drift_m'))
    else:
        require(speed <= scorer.limits['max_node_speed_m_s'] and pose_speed <= scorer.limits['max_node_speed_m_s']
                and drift <= scorer.limits['stable_node_drift_m'], 'Final native fold is not released and stable')
    good, geometry = scorer._fold('towel', q[-1], np.asarray(task['asset_metadata']['target_center_xy_m']), table_supported=True)
    require(good, 'Last native fold shape no longer meets its frozen geometry rules')
    center = (q[-1].astype(float).min(axis=0)[:2]+q[-1].astype(float).max(axis=0)[:2])/2
    translations = {}
    for row, target in zip(expected['cloths'], ([.30, -.55], [.95, -.55])):
        shift = np.r_[np.asarray(target)-center, 0.]
        row['initial_vertices_m'] = (q[-1].astype(float)+shift).tolist()
        row['visual_texture_paths'] = deepcopy(DERIVED_TEXTURES)
        row['visual_texture_tile_size_m'] = deepcopy(DERIVED_TEXTURE_TILE_M)
        translations[row['id']] = shift.tolist()
    expected['schema'] = SCHEMA
    expected['visual_texture_provenance'] = {
        'dependencies_sha256': deepcopy(DERIVED_TEXTURE_HASHES),
        'tile_size_basis': 'Poly Haven API_INFO dimensions in millimetres, divided by 1000; rounded to 1 nm.',
        'tile_unit_source': 'https://github.com/Poly-Haven/Public-API/blob/master/swagger.yml',
        'albedo_color_space': 'sRGB',
        'albedo_normalization': 'Per-channel linear RGB mean normalized to each cloth display_color.',
        'scope': 'Explicit new derived-stack appearance; no geometry, thickness or physical material change.'}
    expected['fold_reset_resolution'] = 'Terminal geometry copied from a successful released native fold with unchanged material topology; this new two-cloth reset still needs its own complete native validation.'
    expected['settled_fold_reset'] = {'schema': 'laundry_settled_fold_reset.v1',
        'native_report_path': str(native_path), 'fullstream_audit_path': str(audit_path),
        'evidence_sha256': dict(sorted(evidence.items())), 'physics_options': source_profile,
        'source_native_step': int(steps[-1]), 'source_time_s': float(times[-1]),
        'terminal_window_samples': count, 'terminal_max_speed_m_s': speed,
        'terminal_pose_speed_m_s': pose_speed, 'terminal_max_drift_m': drift,
        'terminal_geometry': geometry, 'translations_m': translations,
        'rotation_or_scaling_applied': False, 'reset_task_credit': False,
        'new_canonical_physics_accepted': False, 'large_archive_copied': False}
    if practical:
        expected['fold_reset_resolution'] = ('Terminal geometry translated from an explicitly accepted practical native-fold review '
            'with unchanged material topology. Original strict diagnostics are retained; this new two-cloth reset remains unqualified.')
        expected['settled_fold_reset'].update(schema='laundry_settled_fold_reset.v2',
            acceptance_basis='explicit_practical_operation_review', source_fold_practical_review=str(review_path),
            practical_acceptance_profile=review['practical_acceptance_profile'],
            practical_terminal_window=practical_window, prior_frame_coverage=coverage,
            source_strict_diagnostics=deepcopy(review['strict_diagnostics']))
    for path, sha in evidence.items():
        checked(path, sha)
    return expected


def validate_settled_metadata(metadata):
    require(metadata.get('schema') == SCHEMA and metadata.get('task_id') == 'towel_stacking',
            'Native-fold derivation is only supported for stack assets')
    provenance = metadata['settled_fold_reset']
    practical = provenance.get('source_fold_practical_review')
    require(provenance.get('schema') == ('laundry_settled_fold_reset.v2' if practical is not None else 'laundry_settled_fold_reset.v1'),
            'Derived reset schema and explicit practical review disagree')
    expected = metadata_from_native_fold(provenance['native_report_path'], provenance['fullstream_audit_path'],
        cells_xy=metadata['cloths'][0]['cells_xy'], physics_options=provenance['physics_options'],
        source_fold_practical_review=practical)
    require(metadata == expected, 'Derived stack reset differs from successful native terminal geometry or provenance')
    return metadata
