"""Short synthetic audit-contract tests; no GPU or native physics is launched."""
from copy import deepcopy
import io
import json
from pathlib import Path
from types import SimpleNamespace
import zipfile

import numpy as np
import pytest

from bench.paths import load_json
from scripts.validation import audit_laundry_cloth as audit
from scenes.workrooms.laundry_room.composition import build_task_inputs
from scenes.workrooms.laundry_room.evaluation import SequenceEvaluator


def stream_fixture(count=4):
    inputs = build_task_inputs('towel_flattening', physics_options={'physics_dt_s': .002}, rule_profile='laundry_strict_task.v1')
    task = inputs['task']
    cloth = task['asset_metadata']['cloths'][0]
    n = len(cloth['rest_vertices_m'])
    q = np.tile(np.asarray(cloth['initial_vertices_m']), (count, 1, 1))
    goal = np.asarray(cloth['rest_vertices_m'])+[.65, -.55, .9015]
    q[1:] = goal
    mask = np.zeros((count, n), dtype=bool)
    mask[1, :2] = True
    calls = mask.any(axis=1).astype(np.int64)
    arrays = {'time_s': np.arange(count)*.002, 'native_step': np.arange(count),
        'towel_q': q, 'towel_qd': np.zeros_like(q), 'towel_control_mask': mask,
        'towel_triangles': np.asarray(cloth['triangles']), 'hidden_constraints': np.zeros(count, dtype=int),
        'position_write_calls': calls.copy(), 'velocity_write_calls': calls.copy()}
    run = {'native_steps': count-1, 'engineering_boundary': {'nodal_write_steps': 1, 'mode': audit.NODAL_MODE}}
    return inputs, arrays, run


def native_binding_fixture(tmp_path, monkeypatch):
    value, arrays, run = stream_fixture()
    root, source = tmp_path/'release', tmp_path/'source'
    root.mkdir(); (source/'scripts/validation').mkdir(parents=True)
    for name, data in [('scene.json', value['scene']), ('task.json', value['task']), ('release_manifest.json', {'fixture': True})]:
        (root/name).write_text(json.dumps(data))
    script = source/'scripts/validation/laundry_cloth_interactions.py'
    script.write_text('# CPU provenance fixture; no native claims\n')
    source_manifest = source/'source_snapshot_manifest.json'
    source_manifest.write_text(json.dumps({'files_sha256': {'scripts/validation/laundry_cloth_interactions.py': audit.digest(script)}}))
    monkeypatch.setattr(audit, 'verify', lambda p: {'verified': True, 'snapshot_root': str(p), 'fixture_only': True})
    states = tmp_path/'states.npz'
    np.savez_compressed(states, **arrays)
    run.update(audit_contract=audit.NATIVE_SCHEMA, task_id=value['task']['id'], physics_dt_s=.002,
        complete=True,
        release={'path': str(root), 'manifest_sha256': audit.digest(root/'release_manifest.json')},
        scene_input=audit.artifact(root/'scene.json'), task_input=audit.artifact(root/'task.json'),
        source_snapshot_manifest=audit.artifact(source_manifest), state_archive=audit.artifact(states),
        native_evidence={'tensor_device': 'cuda:0', 'gpu_dynamics_enabled': True, 'broadphase_type': 'GPU',
            'gravity_m_s2': [0., 0., -9.81], 'cloths': {'towel': {'enabled': True,
                'schema': 'native_surface_deformable', 'disable_gravity': False,
                'self_collision': True, 'surface_thickness_m': .003, 'kinematic_enabled': False,
                'applied_schemas': ['OmniPhysicsDeformableBodyAPI', 'OmniPhysicsSurfaceDeformableSimAPI',
                    'PhysxSurfaceDeformableBodyAPI'], 'vertex_count': len(arrays['towel_q'][0])}}},
        engineering_boundary={'mode': 'engineering_local_nodal_boundary',
            'uncontrolled_input_arrays_bitwise_unchanged': True, 'post_release_nodal_write_steps': 0,
            'nodal_write_steps': 1})
    run['native_evidence_final'] = deepcopy(run['native_evidence'])
    for key in ('source_snapshot', 'source_snapshot_after_run'):
        run[key] = {'verified': True, 'runtime_imports_verified': True,
                    'snapshot_root': str(source), 'fixture_only': True}
    return value, run, states, root


def test_binds_native_source_inputs_and_release_without_trusting_passed(tmp_path, monkeypatch):
    value, run, states, root = native_binding_fixture(tmp_path, monkeypatch)
    run['passed'] = False
    inventory = {}
    result = audit.verify_native_bindings(run, states, root, value['scene'], value['task'], inventory)
    assert result['original_source_verified']['verified']
    assert str(states) in inventory and str(root/'release_manifest.json') in inventory
    (root/'scene.json').write_text('{}')
    with pytest.raises(audit.EvidenceUnknown, match='hash changed'):
        audit.verify_native_bindings(run, states, root, value['scene'], value['task'], {})


@pytest.mark.parametrize('change', [
    lambda r: r.update(audit_contract='old_prototype'),
    lambda r: r['native_evidence'].pop('tensor_device'),
    lambda r: r['native_evidence'].update(tensor_device='cpu'),
    lambda r: r['native_evidence'].update(gpu_dynamics_enabled=False),
    lambda r: r['native_evidence'].update(gravity_m_s2=[0,0,0]),
    lambda r: r['native_evidence']['cloths']['towel'].update(disable_gravity=True),
    lambda r: r['native_evidence']['cloths']['towel'].update(surface_thickness_m=.001),
    lambda r: r['engineering_boundary'].pop('uncontrolled_input_arrays_bitwise_unchanged'),
    lambda r: r['engineering_boundary'].update(post_release_nodal_write_steps=1),
    lambda r: r.update(complete=False),
    lambda r: r.update(error={'message': 'final export failed'}),
    lambda r: r.pop('source_snapshot_after_run'),
    lambda r: r['source_snapshot_after_run'].update(runtime_imports_verified=False),
    lambda r: r['source_snapshot_after_run'].update(snapshot_root='/wrong/source'),
    lambda r: r['native_evidence']['cloths']['towel'].update(kinematic_enabled=None),
    lambda r: r['native_evidence']['cloths']['towel'].update(applied_schemas=[]),
    lambda r: r['native_evidence']['cloths']['towel'].update(vertex_count=1),
    lambda r: r.pop('native_evidence_final'),
    lambda r: r['native_evidence_final'].update(gpu_dynamics_enabled=False),
])
def test_missing_or_wrong_native_attestation_is_unknown(tmp_path, monkeypatch, change):
    value, run, states, root = native_binding_fixture(tmp_path, monkeypatch)
    change(run)
    with pytest.raises((audit.EvidenceUnknown, KeyError)):
        audit.verify_native_bindings(run, states, root, value['scene'], value['task'], {})


def test_complete_axis_masks_and_actual_nodal_call_receipts():
    value, arrays, run = stream_fixture()
    result = audit.validate_stream(arrays, run, value['task'])
    assert result['frames_including_reset'] == 4
    assert result['controlled_native_steps'] == result['position_write_calls'] == 1


def solver_fixture():
    value, _, _ = stream_fixture()
    scene = value['scene']
    scene['isaacsim'].update(cloth_iterations=64, cloth_linear_damping=2.,
        cloth_collision_pair_update_frequency=8, cloth_collision_iteration_multiplier=8,
        cloth_speculative_ccd=True, cloth_self_collision_filter_distance_m=.003)
    attrs = {'physxDeformableBody:'+key: val for key, val in {
        'solverPositionIterationCount': 64, 'linearDamping': 2., 'collisionPairUpdateFrequency': 8,
        'collisionIterationMultiplier': 8, 'enableSpeculativeCCD': True,
        'selfCollisionFilterDistance': float(np.float32(.003))}.items()}
    return scene, {'solver_attributes': attrs}


def test_explicit_solver_options_and_per_cloth_override_require_actual_readback():
    scene, native = solver_fixture()
    audit.verify_solver_options(scene, 'towel', native)
    scene['cloths'][0].setdefault('isaacsim', {})['cloth_linear_damping'] = 3.
    with pytest.raises(audit.EvidenceUnknown, match='linearDamping'):
        audit.verify_solver_options(scene, 'towel', native)
    native['solver_attributes']['physxDeformableBody:linearDamping'] = 3.
    audit.verify_solver_options(scene, 'towel', native)


@pytest.mark.parametrize('key', ['solverPositionIterationCount', 'linearDamping',
    'collisionPairUpdateFrequency', 'collisionIterationMultiplier', 'enableSpeculativeCCD',
    'selfCollisionFilterDistance'])
def test_missing_or_wrong_explicit_solver_field_is_unknown(key):
    scene, native = solver_fixture()
    attr = 'physxDeformableBody:'+key
    native['solver_attributes'].pop(attr)
    with pytest.raises(audit.EvidenceUnknown, match=key):
        audit.verify_solver_options(scene, 'towel', native)
    native['solver_attributes'][attr] = False if key == 'enableSpeculativeCCD' else -1
    with pytest.raises(audit.EvidenceUnknown, match=key):
        audit.verify_solver_options(scene, 'towel', native)


def test_new_filter_requires_solver_receipt_but_legacy_v1_stays_compatible():
    scene, _ = solver_fixture()
    with pytest.raises(audit.EvidenceUnknown, match='solver attributes'):
        audit.verify_solver_options(scene, 'towel', {})
    scene['isaacsim'].pop('cloth_self_collision_filter_distance_m')
    audit.verify_solver_options(scene, 'towel', {})


def attachment_stream_fixture():
    value, arrays, run = stream_fixture()
    boundary = run['engineering_boundary']
    boundary.update(mode='native_plain_xform_attachment', nodal_write_steps=0,
        native_attachment_paths=['/World/Attachment0'],
        attachments=[{'path': '/World/Attachment0', 'cloth_id': 'towel',
            'vertex_ids': [0, 1], 'anchor_path': '/World/Anchor0'}])
    # The initial constraint is explicitly reported as active; reset receives
    # no task credit. It is released before the last two physics steps.
    arrays['towel_control_mask'][0, :2] = True
    arrays['attachment_enabled'] = np.array([[True], [True], [False], [False]])
    arrays['position_write_calls'][:] = 0
    arrays['velocity_write_calls'][:] = 0
    return value, arrays, run


def test_native_attachments_bind_actual_enabled_nodes_without_nodal_writes():
    value, arrays, run = attachment_stream_fixture()
    result = audit.validate_stream(arrays, run, value['task'])
    assert result['control_mode'] == 'native_plain_xform_attachment'
    assert result['reset_attachments_enabled'] is True
    assert result['position_write_calls'] == result['velocity_write_calls'] == 0
    assert result['controlled_native_steps'] == 1


@pytest.mark.parametrize('change', [
    lambda a, r: a['attachment_enabled'].__setitem__((0, 0), False),
    lambda a, r: a['attachment_enabled'].__setitem__((-1, 0), True),
    lambda a, r: a['towel_control_mask'].__setitem__((1, 50), True),
    lambda a, r: a['position_write_calls'].__setitem__(1, 1),
    lambda a, r: r['engineering_boundary']['attachments'][0].update(vertex_ids=[0, 3]),
])
def test_native_attachment_mismatch_or_hidden_nodal_write_is_unknown(change):
    value, arrays, run = attachment_stream_fixture()
    change(arrays, run)
    with pytest.raises(audit.EvidenceUnknown):
        audit.validate_stream(arrays, run, value['task'])


def test_actual_stage_attachment_inventory_must_match_declaration(tmp_path, monkeypatch):
    value, run, states, root = native_binding_fixture(tmp_path, monkeypatch)
    _, _, attachment_run = attachment_stream_fixture()
    run['engineering_boundary'].update(attachment_run['engineering_boundary'],
        cloth_position_writes_after_reset=False, cloth_velocity_writes_after_reset=False)
    run['native_evidence']['attachment_inventory'] = deepcopy(run['engineering_boundary']['attachments'])
    run['native_evidence_final'] = deepcopy(run['native_evidence'])
    assert audit.verify_native_bindings(run, states, root, value['scene'], value['task'], {})
    run['native_evidence']['attachment_inventory'][0]['vertex_ids'] = [30, 31]
    with pytest.raises(audit.EvidenceUnknown, match='Actual stage attachment inventory'):
        audit.verify_native_bindings(run, states, root, value['scene'], value['task'], {})


@pytest.mark.parametrize('change', [
    lambda a: a.pop('hidden_constraints'),
    lambda a: a['hidden_constraints'].__setitem__(2, 1),
    lambda a: a['native_step'].__setitem__(2, 5),
    lambda a: a['time_s'].__setitem__(0, .002),
    lambda a: a['position_write_calls'].__setitem__(2, 1),
    lambda a: a['velocity_write_calls'].__setitem__(1, 0),
    lambda a: a['towel_control_mask'].__setitem__((0, 0), True),
    lambda a: a['towel_control_mask'].__setitem__((-1, 0), True),
    lambda a: a['towel_triangles'].__setitem__((0, 0), 10),
    lambda a: a['towel_q'].__setitem__((0, 0, 2), 5.),
])
def test_incomplete_or_inconsistent_stream_cannot_reach_oracle(change):
    value, arrays, run = stream_fixture()
    change(arrays)
    with pytest.raises(audit.EvidenceUnknown):
        audit.validate_stream(arrays, run, value['task'])


class SyntheticOracle:
    """Controlled test outputs only; deliberately no CUDA qualification."""
    def __init__(self, bad_at=None, penetration=0.):
        self.calls = 0
        self.bad_at, self.penetration = bad_at, penetration

    def check(self, vertices):
        index = self.calls
        self.calls += 1
        return {'self_intersections': int(index == self.bad_at), 'intercloth_intersections': 0,
            'degenerate_triangles': 0, 'finite_thickness_compression_checked': True,
            'max_cloth_envelope_penetration_m': self.penetration,
            'fixture_only': True}


def test_complete_audit_continues_after_success_and_rejects_later_physics():
    value, arrays, run = stream_fixture(count=260)
    audit.validate_stream(arrays, run, value['task'])
    scorer = SequenceEvaluator(value['task'])
    oracle = SyntheticOracle(bad_at=259)
    output = io.StringIO()
    result = audit.audit_frames(arrays, value['scene'], value['task'], scorer, oracle, output)
    assert oracle.calls == result['samples_checked'] == 260
    assert len(output.getvalue().splitlines()) == 260
    assert result['first_frozen_task_success'] is not None
    assert result['frozen_evaluator_result']['success'] is True
    assert result['accepted_engineering_task_success'] is False
    assert result['first_physical_failure']['native_step'] == 259


def test_complete_audit_requires_actual_terminal_window_not_only_latched_success():
    value, arrays, _ = stream_fixture(count=260)
    result = audit.audit_frames(arrays, value['scene'], value['task'], SequenceEvaluator(value['task']),
        SyntheticOracle(), io.StringIO())
    assert result['accepted_engineering_task_success']
    assert result['terminal_window_required']
    assert result['terminal_window']['passed']
    assert result['terminal_window']['samples_checked'] == 251
    assert result['terminal_window']['duration_s'] == pytest.approx(.5)


@pytest.mark.parametrize('change, reason', [
    (lambda a: a['towel_qd'].__setitem__((-1, 0, 0), .10), 'terminal_raw_speed_exceeded'),
    (lambda a: a['towel_q'].__setitem__(-1, a['towel_q'][-1]+[.0001, 0, 0]), 'terminal_pose_speed_exceeded'),
    (lambda a: a['towel_q'].__setitem__(-1, a['towel_q'][-1]+[.1, 0, 0]), 'terminal_goal_geometry_failed'),
    (lambda a: a['towel_control_mask'].__setitem__((-1, 0), True), 'terminal_control_retained'),
])
def test_early_success_cannot_accept_late_motion_shape_loss_or_regrasp(change, reason):
    value, arrays, _ = stream_fixture(count=260)
    change(arrays)
    result = audit.audit_frames(arrays, value['scene'], value['task'], SequenceEvaluator(value['task']),
        SyntheticOracle(), io.StringIO())
    assert result['frozen_evaluator_result']['success']
    assert result['first_frozen_task_success'] is not None
    assert result['physical_geometry_stream_passed']
    assert not result['accepted_engineering_task_success']
    assert reason in result['terminal_window']['reason_codes']


def test_terminal_gate_checks_entire_last_window_even_if_last_shape_recovers():
    value, arrays, _ = stream_fixture(count=600)
    arrays['towel_q'][400] += [.1, 0, 0]
    result = audit.audit_frames(arrays, value['scene'], value['task'], SequenceEvaluator(value['task']),
        SyntheticOracle(), io.StringIO())
    assert result['frozen_evaluator_result']['success']
    terminal = result['terminal_window']
    assert not terminal['passed']
    assert terminal['first_geometry_failure']['native_step'] == 400
    assert terminal['terminal_geometry']['target_rms_m'] < 1.e-10


def test_terminal_window_rejects_slow_accumulated_drift():
    value, arrays, _ = stream_fixture(count=600)
    arrays['towel_q'][349:, :, 0] += np.linspace(0., .007, 251)[:, None]
    result = audit.audit_frames(arrays, value['scene'], value['task'], SequenceEvaluator(value['task']),
        SyntheticOracle(), io.StringIO())
    terminal = result['terminal_window']
    assert result['frozen_evaluator_result']['success']
    assert terminal['geometry_passed'] and terminal['pose_speed_m_s'] < .035
    assert terminal['reason_codes'] == ['terminal_drift_exceeded']
    assert not result['accepted_engineering_task_success']


@pytest.mark.parametrize('task_id', ('towel_flattening', 'towel_folding', 'towel_stacking'))
def test_terminal_geometry_uses_each_frozen_task_shape(task_id):
    from scenes.workrooms.laundry_room.assets import folded_vertices
    value = build_task_inputs(task_id, physics_options={'physics_dt_s': .002}, rule_profile='laundry_strict_task.v1')
    task = value['task']; count = 251
    arrays = {'time_s': np.arange(count)*.002, 'native_step': np.arange(count)}
    for spec in task['asset_metadata']['cloths']:
        name = spec['id']; rest = np.asarray(spec['rest_vertices_m'])
        if task_id == 'towel_flattening':
            points = rest+[.65, -.55, .9015]
        else:
            lower_z = .9105 if task_id == 'towel_stacking' and name == 'towel' else .9015
            points = folded_vertices(rest, task['asset_metadata']['target_center_xy_m'], lower_z=lower_z)
        arrays[name+'_q'] = np.tile(points, (count, 1, 1))
        arrays[name+'_qd'] = np.zeros((count, len(rest), 3))
        arrays[name+'_control_mask'] = np.zeros((count, len(rest)), dtype=bool)
        arrays[name+'_triangles'] = np.asarray(spec['triangles'])
    scorer = SequenceEvaluator(task)
    assert audit.terminal_window(arrays, scorer)['passed']
    # A terminal interval beyond the original frozen budget is not rescued
    # by a prior latched success either.
    arrays['native_step'] += 60000
    arrays['time_s'] = arrays['native_step']*.002
    terminal = audit.terminal_window(arrays, scorer)
    assert terminal['reason_codes'] == ['terminal_budget_exhausted']


def test_terminal_window_cannot_create_reset_or_short_record_credit():
    value, arrays, _ = stream_fixture(count=260)
    arrays['towel_q'][:] = arrays['towel_q'][1]
    arrays['towel_control_mask'][:] = False
    result = audit.audit_frames(arrays, value['scene'], value['task'], SequenceEvaluator(value['task']),
        SyntheticOracle(), io.StringIO())
    assert result['terminal_window']['passed']
    assert not result['accepted_engineering_task_success']
    value, arrays, _ = stream_fixture(count=4)
    terminal = audit.terminal_window(arrays, SequenceEvaluator(value['task']))
    assert not terminal['complete'] and not terminal['passed']
    assert terminal['reason_codes'] == ['terminal_window_incomplete']


def test_thickness_envelope_includes_half_thickness_above_table():
    value, arrays, _ = stream_fixture()
    points = arrays['towel_q'][1].copy()
    points[:, 2] = .9008
    depth = audit.table_penetration({'towel': points}, value['scene'], {'towel': .003})
    assert depth == pytest.approx(.0007)
    oracle = SyntheticOracle(penetration=.0012)
    result = audit.audit_frames(arrays, value['scene'], value['task'], SequenceEvaluator(value['task']), oracle, io.StringIO())
    assert result['maxima']['max_penetration_m'] == pytest.approx(.0012)
    assert not result['physical_geometry_stream_passed']


def test_full_audit_rechecks_late_stretch_after_scorer_latches():
    value, arrays, _ = stream_fixture(count=260)
    arrays['towel_q'][-1, 0, 0] -= .1
    oracle = SyntheticOracle()
    result = audit.audit_frames(arrays, value['scene'], value['task'], SequenceEvaluator(value['task']), oracle, io.StringIO())
    assert result['first_frozen_task_success'] is not None
    assert result['first_physical_failure']['native_step'] == 259
    assert result['maxima']['max_edge_stretch_ratio'] > 1.25
    assert not result['accepted_engineering_task_success']


def test_safe_npz_extraction_and_cleanup(tmp_path):
    _, arrays, _ = stream_fixture()
    path = tmp_path/'states.npz'
    np.savez_compressed(path, **arrays)
    with audit.NativeArchive(path, tmp_path) as loaded:
        assert isinstance(loaded['towel_q'], np.memmap)
        assert np.array_equal(loaded['towel_q'], arrays['towel_q'])
        assert len(list(tmp_path.glob('.native-array-cache-*'))) == 1
    assert not list(tmp_path.glob('.native-array-cache-*'))
    bad = tmp_path/'unsafe.npz'
    with zipfile.ZipFile(bad, 'w') as archive:
        archive.writestr('../escape.npy', b'not an array')
    with pytest.raises(audit.EvidenceUnknown, match='Unsafe'):
        with audit.NativeArchive(bad, tmp_path):
            pass
    assert not list(tmp_path.glob('.native-array-cache-*'))


def test_cli_rejects_editable_source_before_gpu_or_output(tmp_path, monkeypatch):
    monkeypatch.setattr(audit, 'verify_if_snapshot', lambda *a, **k: None)
    args = SimpleNamespace(output=tmp_path/'output')
    with pytest.raises(ValueError, match='complete frozen'):
        audit.run(args)
    assert not args.output.exists()


def test_cli_old_prototype_returns_unknown_before_constructing_cuda_oracle(tmp_path, monkeypatch):
    from bench.isaacsim import release as canonical_release
    monkeypatch.setattr(audit, 'verify_if_snapshot', lambda *a, **k: {'verified': True, 'fixture_only': True})
    monkeypatch.setattr(audit, 'DATASETS', tmp_path)
    source = tmp_path/'source'; source.mkdir()
    (source/'source_snapshot_manifest.json').write_text('{"fixture_only":true}')
    monkeypatch.setattr(audit, 'PROJECT', source)
    value, _, _ = stream_fixture()
    root = tmp_path/'release'; root.mkdir()
    metadata = tmp_path/'native_report.json'
    metadata.write_text('{"audit_contract":"old_prototype"}')
    states = tmp_path/'native_states.npz'; states.write_bytes(b'not read before provenance gate')
    monkeypatch.setattr(canonical_release, 'validate', lambda *a, **k: {
        'schema': 'laundry_cloth_task_release.v1', 'task_id': value['task']['id']})
    monkeypatch.setattr(canonical_release, 'task_inputs', lambda *a, **k: (
        {}, value['scene'], value['task'], value['initial_state']))
    args = SimpleNamespace(states=states, run_metadata=metadata, release=root,
        output=tmp_path/'audit', device='cuda:0')
    assert audit.run(args) == 2
    report = load_json(args.output/'report.json')
    assert report['status'] == 'unknown' and report['passed'] is False
    assert 'complete-stream evidence contract' in report['error']['message']
    assert report['inputs_sha256'][str(states)] == audit.digest(states)
    assert not (args.output/'frame_audit.jsonl').exists()


def test_bad_authored_geometry_cannot_be_washed_away_by_warmup_or_latched_success():
    value, arrays, _ = stream_fixture(count=260)
    vertices = {'towel': arrays['towel_q'][0].copy()}
    oracle = SyntheticOracle(bad_at=0)  # Only the authored pre-warmup state is bad.
    result = audit.audit_frames(arrays, value['scene'], value['task'], SequenceEvaluator(value['task']),
        oracle, io.StringIO(), authored_vertices=vertices)
    assert oracle.calls == 261 and result['samples_checked'] == 260
    assert result['frozen_evaluator_result']['success'] and result['terminal_window']['passed']
    assert result['first_physical_failure']['phase'] == 'authored_initial'
    assert result['authored_initial_geometry']['operation_credit'] is False
    assert result['intermediate_warmup_states_certified'] is False
    assert not result['accepted_engineering_task_success']


def test_author_only_requires_its_helper_inside_bound_native_source(tmp_path, monkeypatch):
    value, run, states, root = native_binding_fixture(tmp_path, monkeypatch)
    from scripts.validation.laundry_initialization import SCHEMA
    run['initialization'] = {'schema': SCHEMA, 'mode': 'author-only'}
    with pytest.raises(audit.EvidenceUnknown, match='source omits the author-only'):
        audit.verify_native_bindings(run, states, root, value['scene'], value['task'], {})


def test_authored_warmup_control_must_fit_the_same_frozen_world_grasp_regions():
    value, arrays, _ = stream_fixture()
    vertices = {'towel': arrays['towel_q'][0].copy()}
    # Three isolated corner fragments meet the <=10% count and material
    # locality rules, but cannot form <=2 world-space clamps before warmup.
    controlled = {'towel': [0, 30, 650]}
    oracle = SyntheticOracle()
    with pytest.raises(audit.EvidenceUnknown, match='Authored warmup control.*world-space grasp regions'):
        audit.audit_authored_geometry(vertices, value['scene'], value['task'], SequenceEvaluator(value['task']),
            oracle, controlled)
    assert oracle.calls == 0
