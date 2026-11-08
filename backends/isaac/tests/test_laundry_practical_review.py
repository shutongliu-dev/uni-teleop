"""CPU counterexamples for the explicit practical profile; no native physics."""
from copy import deepcopy
import json
from types import SimpleNamespace

import numpy as np
import pytest

from scenes.workrooms.laundry_room.composition import build_task_inputs
from scenes.workrooms.laundry_room.evaluation import SequenceEvaluator
from scripts.validation import review_laundry_practical as practical


def fixture(count=260):
    value = build_task_inputs('towel_flattening', physics_options={'physics_dt_s': .002}, rule_profile='laundry_strict_task.v1')
    cloth = value['task']['asset_metadata']['cloths'][0]
    n = len(cloth['rest_vertices_m'])
    q = np.tile(np.asarray(cloth['initial_vertices_m']), (count, 1, 1))
    q[1:] = np.asarray(cloth['rest_vertices_m'])+[.65, -.55, .9015]
    mask = np.zeros((count, n), dtype=bool)
    mask[1, :2] = True
    arrays = {'time_s': np.arange(count)*.002, 'native_step': np.arange(count),
        'towel_q': q, 'towel_qd': np.zeros_like(q), 'towel_control_mask': mask,
        'towel_triangles': np.asarray(cloth['triangles'])}
    prior = {'physical_geometry_stream_passed': True,
             'all_native_samples_checked': True, 'samples_checked': count}
    return value, arrays, prior


def review(arrays, prior, value):
    return practical.review_arrays(arrays, SequenceEvaluator(value['task']), prior)


def test_single_node_micro_jitter_passes_only_explicit_practical_profile():
    value, arrays, prior = fixture()
    arrays['towel_q'][-200::2, 0, 0] += .0001
    arrays['towel_qd'][-200:, 0, 0] = .08
    result = review(arrays, prior, value)
    assert result['practical_operation_accepted']
    terminal = result['terminal_window']
    assert terminal['practical_acceptance_profile'] == practical.PROFILE
    assert terminal['max_frame_raw_speed_p95_m_s'] == 0
    assert terminal['max_frame_pose_speed_p95_m_s'] == 0
    strict = terminal['strict_terminal_window']
    assert not strict['passed']
    assert strict['geometry_passed'] and strict['complete_release']
    assert strict['max_node_speed_m_s'] == pytest.approx(.08)
    assert strict['pose_speed_m_s'] == pytest.approx(.05)
    assert strict['max_node_drift_m'] == pytest.approx(.0001)
    assert set(strict['reason_codes']) == {'terminal_raw_speed_exceeded', 'terminal_pose_speed_exceeded'}
    assert 'passed' not in result and 'physics_accepted' not in result


@pytest.mark.parametrize('change, reason', [
    # One brief frame must fail: no percentile pooling across time.
    (lambda a: a['towel_qd'].__setitem__((-100, slice(None), 0), .04), 'terminal_raw_speed_p95_exceeded'),
    (lambda a: a['towel_q'].__setitem__(-100, a['towel_q'][-100]+[.00008, 0, 0]), 'terminal_pose_speed_p95_exceeded'),
    # Use speed norm, not the largest component.
    (lambda a: a['towel_qd'].__setitem__(-100, .025), 'terminal_raw_speed_p95_exceeded'),
    # A single node cannot evade the unchanged all-node drift rule.
    (lambda a: a['towel_q'].__setitem__((-100, 0, 0), a['towel_q'][-100, 0, 0]+.007), 'terminal_drift_exceeded'),
    # Geometry/release are checked throughout the window, even if the last frame recovers.
    (lambda a: a['towel_q'].__setitem__(-100, a['towel_q'][-100]+[.1, 0, 0]), 'terminal_goal_geometry_failed'),
    (lambda a: a['towel_control_mask'].__setitem__((-100, 0), True), 'terminal_control_retained'),
])
def test_terminal_counterexamples_reject(change, reason):
    value, arrays, prior = fixture()
    change(arrays)
    result = review(arrays, prior, value)
    assert not result['practical_operation_accepted']
    assert reason in result['reason_codes']


def test_more_than_five_percent_jittering_nodes_reject():
    value, arrays, prior = fixture()
    arrays['towel_qd'][-10, :40, 0] = .04
    result = review(arrays, prior, value)
    assert not result['practical_operation_accepted']
    assert 'terminal_raw_speed_p95_exceeded' in result['reason_codes']


@pytest.mark.parametrize('task_id', ('towel_folding', 'towel_stacking'))
def test_fold_and_stack_keep_frozen_geometry_and_do_not_dilute_one_cloth_speed(task_id):
    from scenes.workrooms.laundry_room.assets import folded_vertices
    value = build_task_inputs(task_id, physics_options={'physics_dt_s': .002}, rule_profile='laundry_strict_task.v1')
    task, count = value['task'], 260
    arrays = {'time_s': np.arange(count)*.002, 'native_step': np.arange(count)}
    for spec in task['asset_metadata']['cloths']:
        name, rest = spec['id'], np.asarray(spec['rest_vertices_m'])
        lower_z = .9105 if task_id == 'towel_stacking' and name == 'towel' else .9015
        goal = folded_vertices(rest, task['asset_metadata']['target_center_xy_m'], lower_z=lower_z)
        arrays[name+'_q'] = np.tile(goal, (count, 1, 1))
        arrays[name+'_q'][0] = spec['initial_vertices_m']
        arrays[name+'_qd'] = np.zeros((count, len(rest), 3))
        arrays[name+'_control_mask'] = np.zeros((count, len(rest)), dtype=bool)
        arrays[name+'_triangles'] = np.asarray(spec['triangles'])
    arrays['towel_control_mask'][1, :2] = True
    prior = {'physical_geometry_stream_passed': True,
             'all_native_samples_checked': True, 'samples_checked': count}
    assert review(arrays, prior, value)['practical_operation_accepted']
    arrays['towel_qd'][-10, :40, 0] = .04
    result = review(arrays, prior, value)
    assert not result['practical_operation_accepted']
    assert 'terminal_raw_speed_p95_exceeded' in result['reason_codes']
    if task_id == 'towel_stacking':
        stationary = [row for name, row in result['terminal_window']['per_cloth'].items() if name != 'towel']
        assert len(stationary) == 1 and stationary[0]['max_frame_raw_speed_p95_m_s'] == 0


@pytest.mark.parametrize('change, reason', [
    (lambda p: p.update(physical_geometry_stream_passed=False), 'full_stream_physical_geometry_failed'),
    (lambda p: p.pop('physical_geometry_stream_passed'), 'full_stream_physical_geometry_failed'),
    (lambda p: p.update(all_native_samples_checked=False), 'full_stream_frame_coverage_incomplete'),
    (lambda p: p.update(samples_checked=259), 'full_stream_frame_coverage_incomplete'),
])
def test_prior_physics_and_complete_frame_coverage_remain_mandatory(change, reason):
    value, arrays, prior = fixture()
    change(prior)
    result = review(arrays, prior, value)
    assert not result['practical_operation_accepted']
    assert reason in result['reason_codes']


def test_short_window_and_reset_only_cannot_accept():
    value, arrays, prior = fixture(count=250)
    result = review(arrays, prior, value)
    assert not result['practical_operation_accepted']
    assert 'terminal_window_incomplete' in result['reason_codes']
    value, arrays, prior = fixture()
    arrays['towel_q'][:] = arrays['towel_q'][-1]
    arrays['towel_control_mask'][:] = False
    result = review(arrays, prior, value)
    assert not result['practical_operation_accepted']
    assert 'no_engineering_operation' in result['reason_codes']
    assert 'operation_displacement_not_established' in result['reason_codes']


def binding_fixture(tmp_path, monkeypatch):
    value, arrays, report = fixture(count=4)
    release, source = tmp_path/'release', tmp_path/'source'
    release.mkdir()
    source.mkdir()
    for name in ('release_manifest.json', 'scene.json', 'task.json'):
        (release/name).write_text('{}')
    source_manifest = source/'source_snapshot_manifest.json'
    source_manifest.write_text(json.dumps({'files_sha256': {
        'scripts/validation/audit_laundry_cloth.py': 'fixture',
        'bench/evaluation/cloth/gpu_intersections.py': 'fixture'}}))
    monkeypatch.setattr(practical, 'verify', lambda p: {'verified': True, 'fixture_only': True})
    native, states = tmp_path/'native_report.json', tmp_path/'states.npz'
    native.write_text('{}')
    np.savez(states, **arrays)
    trace = tmp_path/'frames.jsonl'
    rows = [{'native_step': i, 'time_s': float(t), 'physical_sample_valid': True}
            for i, t in enumerate(arrays['time_s'])]
    trace.write_text(''.join(json.dumps(row)+'\n' for row in rows))
    receipt = {'verified': True, 'runtime_imports_verified': True, 'snapshot_root': str(source)}
    report.update(schema=practical.audit.SCHEMA, status='failed', task_id=value['task']['id'],
        audit_device='cuda:0', stream_contract={'frames_including_reset': 4, 'native_steps': 3, 'physics_dt_s': .002},
        inputs_sha256={str(path): practical.audit.digest(path) for path in
            (native, states, release/'release_manifest.json', release/'scene.json', release/'task.json')},
        audit_source_manifest=practical.audit.artifact(source_manifest),
        audit_source=receipt, audit_source_final=deepcopy(receipt),
        frozen_evaluator_result={'evidence_status': 'complete'}, frame_audit=practical.audit.artifact(trace),
        first_physical_failure=None, authored_initial_geometry={'passed': True})
    run = {'task_id': value['task']['id'], 'native_steps': 3}
    return report, run, native, states, release, value['task'], arrays, trace


def test_complete_failed_strict_audit_is_valid_evidence_but_changed_inputs_are_not(tmp_path, monkeypatch):
    report, run, native, states, release, task, arrays, trace = binding_fixture(tmp_path, monkeypatch)
    inventory = {}
    assert practical.verify_prior_audit(report, run, native, states, release, task, inventory) == trace
    assert practical.verify_frame_trace(trace, arrays, report)['all_native_frames_covered']
    assert str(trace) in inventory
    native.write_text('{"changed": true}')
    with pytest.raises(practical.audit.EvidenceUnknown, match='hash changed'):
        practical.verify_prior_audit(report, run, native, states, release, task, {})


def test_author_only_cannot_omit_pre_warmup_physics(tmp_path, monkeypatch):
    from scripts.validation.laundry_initialization import SCHEMA
    report, run, native, states, release, task, _, _ = binding_fixture(tmp_path, monkeypatch)
    run['initialization'] = {'schema': SCHEMA, 'mode': 'author-only'}
    report.pop('authored_initial_geometry')
    with pytest.raises(practical.audit.EvidenceUnknown, match='pre-warmup geometry'):
        practical.verify_prior_audit(report, run, native, states, release, task, {})


@pytest.mark.parametrize('change', [
    lambda r: r.update(status='unknown'),
    lambda r: r.update(all_native_samples_checked=False),
    lambda r: r.update(samples_checked=3),
    lambda r: r.update(task_id='wrong_task'),
    lambda r: r.update(audit_device='cpu'),
    lambda r: r['audit_source_final'].update(runtime_imports_verified=False),
    lambda r: r['frozen_evaluator_result'].update(evidence_status='incomplete'),
    lambda r: r.update(inputs_sha256={}),
    lambda r: r['frame_audit'].update(sha256='0'*64),
])
def test_incomplete_or_unbound_prior_audit_cannot_be_reused(tmp_path, monkeypatch, change):
    report, run, native, states, release, task, _, _ = binding_fixture(tmp_path, monkeypatch)
    change(report)
    with pytest.raises(practical.audit.EvidenceUnknown):
        practical.verify_prior_audit(report, run, native, states, release, task, {})


@pytest.mark.parametrize('change', [
    lambda rows: rows.pop(),
    lambda rows: rows[1].update(native_step=2),
    lambda rows: rows[1].update(time_s=.003),
    lambda rows: rows[1].update(physical_sample_valid=False),
])
def test_trace_checks_every_native_frame_and_rejects_physics_contradictions(tmp_path, monkeypatch, change):
    report, _, _, _, _, _, arrays, trace = binding_fixture(tmp_path, monkeypatch)
    rows = [json.loads(line) for line in trace.read_text().splitlines()]
    change(rows)
    trace.write_text(''.join(json.dumps(row)+'\n' for row in rows))
    with pytest.raises(practical.audit.EvidenceUnknown):
        practical.verify_frame_trace(trace, arrays, report)


def test_cli_requires_frozen_review_source_before_creating_output(tmp_path, monkeypatch):
    output = tmp_path/'output'
    monkeypatch.setattr(practical, 'verify_if_snapshot', lambda *a, **kw: None)
    with pytest.raises(ValueError, match='complete frozen project source'):
        practical.run(SimpleNamespace(output=output))
    assert not output.exists()
