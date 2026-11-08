"""Synthetic CPU evidence only; practical derivation never qualifies a new reset."""
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest

from test_laundry_settled_reset import fixture
from bench.paths import load_json
from scenes.workrooms.laundry_room import assets, evaluation, settled_reset
from scripts.validation import review_laundry_practical as practical
from scripts.validation.laundry_initialization import SCHEMA as INITIALIZATION_SCHEMA


def write(path, data):
    Path(path).write_text(json.dumps(data))


def practical_fixture(tmp_path, monkeypatch):
    native_path, audit_path, profile, _ = fixture(tmp_path, monkeypatch)
    native, audit = (load_json(p) for p in (native_path, audit_path))
    root, source = tmp_path/'source_release', tmp_path/'source'
    task = load_json(root/'task.json')
    source_files = {}
    for name in ('scripts/validation/laundry_cloth_interactions.py', 'scripts/validation/laundry_initialization.py',
                 'scripts/validation/audit_laundry_cloth.py', 'bench/evaluation/cloth/gpu_intersections.py'):
        path = source/name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('# Synthetic CPU source receipt only.\n')
        source_files[name] = assets.digest(path)
    write(source/'source_snapshot_manifest.json', {'files_sha256': source_files})
    write(root/'release_manifest.json', {'files_sha256': {name: assets.digest(root/name)
        for name in ('scene.json', 'task.json', 'physics_profile.json')}, 'asset_files_sha256': {}})
    archive = Path(native['state_archive']['path'])
    with np.load(archive, allow_pickle=False) as data:
        arrays = {name: data[name] for name in data.files}
    # Real operation history outside the tail; only one node jitters in the tail.
    arrays['towel_q'][0, :, 0] += .03
    arrays['towel_control_mask'][1, :2] = True
    arrays['towel_q'][-200::2, 0, 0] += .0001
    arrays['towel_qd'][-200:, 0, 0] = .08
    np.savez_compressed(archive, **arrays)
    artifact = practical.audit.artifact
    receipt = {'verified': True, 'runtime_imports_verified': True, 'snapshot_root': str(source)}
    native.update(state_archive=artifact(archive), release=artifact(root/'release_manifest.json'),
        source_snapshot_manifest=artifact(source/'source_snapshot_manifest.json'), physics_dt_s=.002,
        source_snapshot=receipt, source_snapshot_after_run=deepcopy(receipt),
        initialization={'schema': INITIALIZATION_SCHEMA, 'mode': 'author-only'},
        native_evidence={'tensor_device': 'cuda:0', 'gpu_dynamics_enabled': True, 'broadphase_type': 'GPU',
            'gravity_m_s2': [0., 0., -9.81], 'cloths': {'towel': {'enabled': True,
                'schema': 'native_surface_deformable', 'disable_gravity': False,
                'self_collision': True, 'surface_thickness_m': .003, 'kinematic_enabled': False,
                'applied_schemas': ['OmniPhysicsDeformableBodyAPI', 'OmniPhysicsSurfaceDeformableSimAPI',
                    'PhysxSurfaceDeformableBodyAPI'], 'vertex_count': len(arrays['towel_q'][0])}}},
        engineering_boundary={'mode': practical.audit.NODAL_MODE,
            'uncontrolled_input_arrays_bitwise_unchanged': True, 'post_release_nodal_write_steps': 0,
            'nodal_write_steps': 1})
    native['native_evidence_final'] = deepcopy(native['native_evidence'])
    write(native_path, native)
    trace = Path(audit['frame_audit']['path'])
    trace.write_text(''.join(json.dumps({'native_step': int(i), 'time_s': float(t),
        'physical_sample_valid': True})+'\n' for i, t in zip(arrays['native_step'], arrays['time_s'])))
    terminal = practical.practical_terminal_window(arrays, evaluation.SequenceEvaluator(task))
    audit.update(status='failed', passed=False, accepted_engineering_task_success=False,
        terminal_window=terminal['strict_terminal_window'], stream_contract={
            'frames_including_reset': len(arrays['time_s']), 'native_steps': native['native_steps'], 'physics_dt_s': .002},
        audit_source_manifest=artifact(source/'source_snapshot_manifest.json'), audit_source=receipt,
        audit_source_final=deepcopy(receipt), frozen_evaluator_result={'evidence_status': 'complete'},
        authored_initial_geometry={'passed': True}, frame_audit=artifact(trace))
    audit['inputs_sha256'] = {str(p): assets.digest(p) for p in (
        native_path, archive, root/'release_manifest.json', root/'scene.json', root/'task.json', root/'physics_profile.json')}
    write(audit_path, audit)
    review_source = tmp_path/'review_source'
    entry = review_source/'scripts/validation/review_laundry_practical.py'
    entry.parent.mkdir(parents=True)
    entry.write_text('# Synthetic practical review source receipt only.\n')
    write(review_source/'source_snapshot_manifest.json', {'files_sha256': {
        'scripts/validation/review_laundry_practical.py': assets.digest(entry)}})
    review_receipt = {'verified': True, 'runtime_imports_verified': True, 'snapshot_root': str(review_source),
                      'entrypoint': artifact(entry)}
    bound = {**audit['inputs_sha256'], str(audit_path): assets.digest(audit_path), str(trace): assets.digest(trace),
             str(source/'source_snapshot_manifest.json'): assets.digest(source/'source_snapshot_manifest.json')}
    bound.update({str(source/p): sha for p, sha in source_files.items()})
    review = {'schema': practical.SCHEMA, 'practical_acceptance_profile': practical.PROFILE,
        'task_id': 'towel_folding', 'status': 'accepted', 'practical_operation_accepted': True, 'reason_codes': [],
        'native_report': artifact(native_path), 'full_stream_audit': artifact(audit_path), 'inputs_sha256': bound,
        'review_source_manifest': artifact(review_source/'source_snapshot_manifest.json'),
        'review_source': review_receipt, 'review_source_final': deepcopy(review_receipt),
        'stream_contract': deepcopy(audit['stream_contract']),
        'prior_frame_coverage': practical.verify_frame_trace(trace, arrays, audit),
        'strict_diagnostics': {key: deepcopy(audit.get(key)) for key in ('status', 'passed',
            'accepted_engineering_task_success', 'frozen_evaluator_result', 'terminal_window')},
        'operation_history': practical.operation_history(arrays, evaluation.SequenceEvaluator(task)),
        'terminal_window': terminal}
    review_path = tmp_path/'practical_review.json'
    write(review_path, review)
    # Source manifest semantics are unit-tested by the frozen-source verifier;
    # all evidence path/hash binding and native readback checks stay real here.
    for module in (practical, practical.audit):
        monkeypatch.setattr(module, 'verify', lambda p: {'verified': True, 'fixture_only': True})
    return native_path, audit_path, review_path, profile, arrays


def derive(native, audit, review, profile):
    return settled_reset.metadata_from_native_fold(native, audit, physics_options=profile,
                                                   source_fold_practical_review=review)


def rebind(native_path, audit_path, review_path):
    """Allow adversarial but internally hash-consistent fixture changes."""
    native, audit, review = (load_json(p) for p in (native_path, audit_path, review_path))
    native['state_archive'] = practical.audit.artifact(native['state_archive']['path'])
    write(native_path, native)
    audit['frame_audit'] = practical.audit.artifact(audit['frame_audit']['path'])
    audit['inputs_sha256'] = {p: assets.digest(p) for p in audit['inputs_sha256']}
    write(audit_path, audit)
    review['native_report'] = practical.audit.artifact(native_path)
    review['full_stream_audit'] = practical.audit.artifact(audit_path)
    review['inputs_sha256'] = {p: assets.digest(p) for p in review['inputs_sha256']}
    review['strict_diagnostics'] = {key: deepcopy(audit.get(key)) for key in review['strict_diagnostics']}
    write(review_path, review)


def test_explicit_practical_reset_keeps_strict_failure_and_only_translates(tmp_path, monkeypatch):
    native, audit, review, profile, arrays = practical_fixture(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match='passed complete-stream'):
        settled_reset.metadata_from_native_fold(native, audit)
    output = tmp_path/'derived_practical'
    assets.prepare_assets('towel_stacking', output, source_fold_native_report=native, source_fold_audit=audit,
        source_fold_practical_review=review, physics_options=profile)
    manifest, metadata = assets.load_asset_manifest('towel_stacking', output/'manifest.json')
    provenance = metadata['settled_fold_reset']
    assert provenance['schema'] == 'laundry_settled_fold_reset.v2'
    assert provenance['acceptance_basis'] == 'explicit_practical_operation_review'
    assert provenance['practical_terminal_window']['passed']
    assert not provenance['source_strict_diagnostics']['passed']
    assert provenance['terminal_max_speed_m_s'] == .08
    assert provenance['terminal_pose_speed_m_s'] == pytest.approx(.05)
    assert not provenance['reset_task_credit'] and not provenance['new_canonical_physics_accepted']
    assert str(review) in manifest['files_sha256']
    for cloth in metadata['cloths']:
        points = np.asarray(cloth['initial_vertices_m'])
        assert np.array_equal(points[:, 2], arrays['towel_q'][-1, :, 2])
        shift = provenance['translations_m'][cloth['id']]
        assert shift[2] == 0 and np.array_equal(points, arrays['towel_q'][-1]+shift)
    assert assets.validate_metadata(metadata) == metadata
    changed = deepcopy(metadata)
    changed['settled_fold_reset'].pop('source_fold_practical_review')
    with pytest.raises(ValueError, match='schema and explicit practical review'):
        assets.validate_metadata(changed)
    for mutation in ('node', 'z', 'rotate', 'scale'):
        changed = deepcopy(metadata)
        q = np.asarray(changed['cloths'][0]['initial_vertices_m'])
        if mutation == 'node':
            q[0, 0] += .001
        elif mutation == 'z':
            q[:, 2] += .001
        elif mutation == 'rotate':
            q[:, :2] = q[:, :2] @ np.array([[0., -1.], [1., 0.]])
        else:
            q *= 1.01
        changed['cloths'][0]['initial_vertices_m'] = q.tolist()
        with pytest.raises(ValueError, match='differs from successful native'):
            assets.validate_metadata(changed)


@pytest.mark.parametrize('field,value', [('status', 'rejected'), ('practical_operation_accepted', False),
    ('practical_acceptance_profile', 'unsupported'), ('reason_codes', ['failure']), ('schema', 'unknown')])
def test_ineligible_review_rejected(tmp_path, monkeypatch, field, value):
    native, audit, review, profile, _ = practical_fixture(tmp_path, monkeypatch)
    data = load_json(review); data[field] = value; write(review, data)
    with pytest.raises(ValueError, match='accepted practical folding review'):
        derive(native, audit, review, profile)


@pytest.mark.parametrize('key', ['native_report', 'full_stream_audit'])
def test_review_of_other_same_task_record_rejected(tmp_path, monkeypatch, key):
    native, audit, review, profile, _ = practical_fixture(tmp_path, monkeypatch)
    data = load_json(review)
    other = tmp_path/'other_same_task.json'; other.write_text(Path(data[key]['path']).read_text())
    data[key] = practical.audit.artifact(other); write(review, data)
    with pytest.raises(ValueError, match='binds a different'):
        derive(native, audit, review, profile)


@pytest.mark.parametrize('mutation,reason', [('geometry', 'terminal_goal_geometry_failed'),
    ('drift', 'terminal_drift_exceeded'), ('raw', 'terminal_raw_speed_p95_exceeded'),
    ('pose', 'terminal_pose_speed_p95_exceeded'), ('control', 'not complete and released')])
def test_tail_recomputed_even_when_accepted_report_and_hashes_match(tmp_path, monkeypatch, mutation, reason):
    native, audit, review, profile, arrays = practical_fixture(tmp_path, monkeypatch)
    if mutation == 'geometry':
        arrays['towel_q'][-100] += [.1, 0, 0]
    elif mutation == 'drift':
        arrays['towel_q'][-100, 0, 0] += .007
    elif mutation == 'raw':
        arrays['towel_qd'][-100, :, 0] = .04
    elif mutation == 'pose':
        arrays['towel_q'][-100] += [.00008, 0, 0]
    else:
        arrays['towel_control_mask'][-100, 0] = True
    np.savez_compressed(load_json(native)['state_archive']['path'], **arrays)
    rebind(native, audit, review)
    with pytest.raises(ValueError, match=reason):
        derive(native, audit, review, profile)


@pytest.mark.parametrize('mutation', ['missing', 'step', 'physical'])
def test_rebound_frame_trace_must_cover_every_valid_frame(tmp_path, monkeypatch, mutation):
    native, audit, review, profile, _ = practical_fixture(tmp_path, monkeypatch)
    trace = Path(load_json(audit)['frame_audit']['path'])
    rows = [json.loads(line) for line in trace.read_text().splitlines()]
    if mutation == 'missing':
        rows.pop()
    elif mutation == 'step':
        rows[100]['native_step'] += 1
    else:
        rows[100]['physical_sample_valid'] = False
    trace.write_text(''.join(json.dumps(row)+'\n' for row in rows))
    rebind(native, audit, review)
    with pytest.raises(ValueError, match='frame audit|physical pass contradicts'):
        derive(native, audit, review, profile)


@pytest.mark.parametrize('mutation', ['source', 'source_receipt', 'review_binding', 'physical_audit', 'authored_geometry'])
def test_practical_source_and_physics_binding_required(tmp_path, monkeypatch, mutation):
    native, audit, review, profile, _ = practical_fixture(tmp_path, monkeypatch)
    data = load_json(review)
    if mutation == 'source':
        Path(data['review_source']['entrypoint']['path']).write_text('# changed source\n')
    elif mutation == 'source_receipt':
        data['review_source_final']['runtime_imports_verified'] = False; write(review, data)
    elif mutation == 'review_binding':
        data['inputs_sha256'].pop(str(native)); write(review, data)
    else:
        data = load_json(audit)
        if mutation == 'physical_audit':
            data['physical_geometry_stream_passed'] = False
        else:
            data['authored_initial_geometry']['passed'] = False
        write(audit, data); rebind(native, audit, review)
    with pytest.raises(ValueError):
        derive(native, audit, review, profile)


def test_practical_cannot_change_topology_profile_or_omit_evidence(tmp_path, monkeypatch):
    native, audit, review, profile, _ = practical_fixture(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match='physical/rest property differs'):
        settled_reset.metadata_from_native_fold(native, audit, cells_xy=(60, 40), source_fold_practical_review=review)
    with pytest.raises(ValueError, match='physics profile differs'):
        derive(native, audit, review, {**profile, 'cloth_thickness_m': .001})
    with pytest.raises(FileNotFoundError):
        derive(native, audit, tmp_path/'missing.json', profile)
    with pytest.raises(ValueError, match='Only stacking'):
        assets.prepare_assets('towel_stacking', tmp_path/'unused', source_fold_practical_review=review)
    assert not (tmp_path/'unused').exists()


@pytest.mark.parametrize('name', ['towel_q', 'towel_triangles'])
def test_rebound_archive_cannot_hide_extra_prefix_rows(tmp_path, monkeypatch, name):
    native, audit, review, profile, arrays = practical_fixture(tmp_path, monkeypatch)
    arrays[name] = np.concatenate((arrays[name][:1], arrays[name]))
    np.savez_compressed(load_json(native)['state_archive']['path'], **arrays)
    rebind(native, audit, review)
    with pytest.raises(ValueError, match='Native array frame count differs'):
        derive(native, audit, review, profile)


@pytest.mark.parametrize('section', ['strict_diagnostics', 'terminal_window'])
def test_reported_terminal_results_cannot_replace_recomputation(tmp_path, monkeypatch, section):
    native, audit, review, profile, _ = practical_fixture(tmp_path, monkeypatch)
    data = load_json(review)
    if section == 'strict_diagnostics':
        data[section]['passed'] = True
    else:
        data[section]['max_frame_raw_speed_p95_m_s'] = .001
    write(review, data)
    with pytest.raises(ValueError, match='strict diagnostics|terminal window differs'):
        derive(native, audit, review, profile)
