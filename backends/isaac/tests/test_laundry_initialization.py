"""CPU USD/provenance fixtures only; no native simulation or GPU qualification."""
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
from pxr import Gf, Sdf, Usd, UsdGeom, Vt

from scenes.workrooms.laundry_room.composition import build_task_inputs
from scripts.validation import audit_laundry_cloth as audit
from scripts.validation import laundry_initialization as init


def fixture(tmp_path):
    value = build_task_inputs('towel_folding', physics_options={'physics_dt_s': .002})
    task, scene = value['task'], value['scene']
    spec = task['asset_metadata']['cloths'][0]
    q = np.asarray(spec['initial_vertices_m'], np.float32)
    n, count, dt = len(q), 4, .002
    tick = init.world_clock_tick(dt)
    stage = Usd.Stage.CreateNew(str(tmp_path/'initial.usda'))
    UsdGeom.Xform.Define(stage, '/World')
    mesh = UsdGeom.Mesh.Define(stage, '/World/Cloths/towel')
    mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(q))
    tri = np.asarray(spec['triangles'], np.int32)
    mesh.CreateFaceVertexCountsAttr([3]*len(tri))
    mesh.CreateFaceVertexIndicesAttr(tri.reshape(-1).tolist())
    anchor = UsdGeom.Xform.Define(stage, '/World/Anchor0')
    anchor.AddTranslateOp().Set(Gf.Vec3d(*map(float, q[0])))
    prim = stage.DefinePrim('/World/Attachment0', 'OmniPhysicsVtxXformAttachment')
    prim.CreateRelationship('omniphysics:src0').SetTargets(['/World/Cloths/towel'])
    prim.CreateRelationship('omniphysics:src1').SetTargets(['/World/Anchor0'])
    prim.CreateAttribute('omniphysics:attachmentEnabled', Sdf.ValueTypeNames.Bool).Set(True)
    prim.CreateAttribute('omniphysics:vtxIndicesSrc0', Sdf.ValueTypeNames.IntArray).Set([0, 1])
    stage.GetRootLayer().Save()
    declared = [{'path': '/World/Attachment0', 'cloth_id': 'towel', 'vertex_ids': [0, 1], 'anchor_path': '/World/Anchor0'}]
    targets = init.anchor_targets(stage, declared)
    # A real warmup can differ from canonical; use an unmistakable CPU witness.
    warmup = q+np.float32(.0002)
    qd = np.full_like(q, .003)
    mask = np.zeros((count, n), bool); mask[:2, :2] = True
    arrays = {'time_s': np.arange(count)*dt, 'native_step': np.arange(count),
        'absolute_world_time_s': (np.arange(count)+2)*tick, 'absolute_world_step': np.arange(count)+2,
        'towel_q': np.tile(warmup, (count, 1, 1)), 'towel_qd': np.tile(qd, (count, 1, 1)),
        'towel_triangles': tri, 'towel_control_mask': mask,
        'anchor_targets': np.tile(targets, (count, 1, 1)),
        'attachment_enabled': np.array([[True], [True], [False], [False]]),
        'position_write_calls': np.zeros(count, int), 'velocity_write_calls': np.zeros(count, int),
        'hidden_constraints': np.zeros(count, int)}
    trace = {'world_clock_tick_s': np.asarray(tick, np.float64), 'world_step_before_reset': np.asarray(0), 'world_time_before_reset': np.asarray(0.),
        'world_step_after_warmup': np.asarray(2), 'world_time_after_warmup': np.asarray(2*tick),
        'world_step_at_baseline': np.asarray(2), 'world_time_at_baseline': np.asarray(2*tick),
        'warmup_anchor_targets': targets.copy(), 'baseline_anchor_targets': targets.copy(),
        'towel_authored_q': q, 'towel_triangles': tri,
        'towel_warmup_q': warmup.copy(), 'towel_baseline_q': warmup.copy(),
        'towel_warmup_qd': qd.copy(), 'towel_baseline_qd': qd.copy(),
        'towel_warmup_control_mask': mask[0].copy(), 'towel_setter_calls': np.array([0, 0])}
    np.savez_compressed(tmp_path/'native_reset_trace.npz', **trace)
    np.savez_compressed(tmp_path/'native_states.npz', **arrays)
    trace_binding = audit.artifact(tmp_path/'native_reset_trace.npz')
    run = {'native_steps': count-1, 'state_archive': audit.artifact(tmp_path/'native_states.npz'),
        'initialization': {'schema': init.SCHEMA, 'mode': 'author-only',
            'authored_stage': audit.artifact(tmp_path/'initial.usda'), 'trace': trace_binding,
            'warmup_policy': 'isaacsim_6_1_world_reset_two_steps', 'world_clock_tick_s': tick,
            'intermediate_warmup_states_recorded': False, 'warmup_operation_credit': False},
        'initialization_trace': trace_binding,
        'reset_position_velocity_calls_by_cloth': {'towel': [0, 0]},
        'final_position_velocity_calls_by_cloth': {'towel': [0, 0]},
        'native_world_reset_offset': {'absolute_step': 2, 'absolute_time_s': 2*tick},
        'engineering_boundary': {'mode': 'native_plain_xform_attachment', 'nodal_write_steps': 0,
            'attachments': declared, 'native_attachment_paths': ['/World/Attachment0']}}
    return value, arrays, run, trace


def check(value, arrays, run):
    return audit.validate_stream(arrays, run, value['task'], scene=value['scene'], inputs={})


def write_trace(run, trace):
    path = run['initialization']['trace']['path']
    np.savez_compressed(path, **trace)
    run['initialization']['trace'] = run['initialization_trace'] = audit.artifact(path)


def test_exact_bound_warmup_can_be_new_baseline_without_relaxing_canonical_gate(tmp_path):
    value, arrays, run, _ = fixture(tmp_path)
    result = check(value, arrays, run)
    assert result['initialization']['verified']
    assert result['initialization']['warmup_operation_credit'] is False
    assert result['initialization']['intermediate_warmup_states_recorded'] is False
    run.pop('initialization')
    with pytest.raises(ValueError, match='canonical initialization'):
        check(value, arrays, run)
    arrays['towel_q'][0] = np.asarray(value['task']['asset_metadata']['cloths'][0]['initial_vertices_m'], np.float32)
    legacy = check(value, arrays, run)
    assert 'initialization' not in legacy


@pytest.mark.parametrize('mutate', [
    lambda a, r: r.update(initialization=None),
    lambda a, r: r['initialization'].update(mode='auto'),
    lambda a, r: r['initialization'].update(warmup_operation_credit=True),
    lambda a, r: r['initialization'].update(intermediate_warmup_states_recorded=True),
    lambda a, r: r['initialization'].update(warmup_policy='settle-until-valid'),
    lambda a, r: r['initialization'].update(world_clock_tick_s=.002),
    lambda a, r: r['reset_position_velocity_calls_by_cloth']['towel'].__setitem__(0, 1),
    lambda a, r: r['final_position_velocity_calls_by_cloth']['towel'].__setitem__(1, 1),
    lambda a, r: r['final_position_velocity_calls_by_cloth']['towel'].__setitem__(1, False),
    lambda a, r: r.pop('final_position_velocity_calls_by_cloth'),
    lambda a, r: r['native_world_reset_offset'].update(absolute_step=3),
    lambda a, r: a['absolute_world_step'].__setitem__(2, 5),
    lambda a, r: a['absolute_world_time_s'].__setitem__(2, .02),
    lambda a, r: a.pop('absolute_world_step'),
    lambda a, r: a['towel_qd'].__setitem__(0, 0),
    lambda a, r: a['towel_q'].__setitem__((0, 3, 0), .1),
    lambda a, r: a['anchor_targets'].__setitem__((0, 0, 2), 1.2),
    lambda a, r: r['initialization']['authored_stage'].update(sha256='0'*64),
])
def test_warmup_contract_rejects_missing_or_falsified_receipts(tmp_path, mutate):
    value, arrays, run, _ = fixture(tmp_path)
    mutate(arrays, run)
    with pytest.raises((ValueError, KeyError)):
        check(value, arrays, run)


@pytest.mark.parametrize('mutate', [
    lambda t: t.__setitem__('world_clock_tick_s', np.asarray(.002)),
    lambda t: t.__setitem__('world_step_before_reset', np.asarray(1)),
    lambda t: t.__setitem__('world_time_before_reset', np.asarray(.002)),
    lambda t: t.__setitem__('world_step_after_warmup', np.asarray(3)),
    lambda t: t.__setitem__('world_time_after_warmup', np.asarray(.006)),
    lambda t: t.__setitem__('world_step_at_baseline', np.asarray(3)),
    lambda t: t['towel_baseline_qd'].__setitem__((0, 0), 0),
    lambda t: t['towel_warmup_q'].__setitem__((0, 0), .2),
    lambda t: t['towel_authored_q'].__setitem__((0, 0), .2),
    lambda t: t['towel_triangles'].__setitem__((0, 0), 10),
    lambda t: t['towel_warmup_control_mask'].__setitem__(0, False),
    lambda t: t['towel_setter_calls'].__setitem__(1, 1),
    lambda t: t['warmup_anchor_targets'].__setitem__((0, 0), .2),
])
def test_rehashing_trace_cannot_hide_warmup_or_reset_changes(tmp_path, mutate):
    value, arrays, run, trace = fixture(tmp_path)
    mutate(trace); write_trace(run, trace)
    with pytest.raises((ValueError, KeyError)):
        check(value, arrays, run)


def test_missing_trace_or_scene_is_unknown_not_canonical_fallback(tmp_path):
    value, arrays, run, _ = fixture(tmp_path)
    with pytest.raises(ValueError, match='frozen scene'):
        audit.validate_stream(arrays, run, value['task'])
    run['initialization'].pop('trace')
    with pytest.raises(KeyError):
        check(value, arrays, run)


def test_usd_points_are_read_even_when_stage_and_trace_are_rehashed(tmp_path):
    value, arrays, run, trace = fixture(tmp_path)
    stage = Usd.Stage.Open(str(tmp_path/'initial.usda'))
    mesh = UsdGeom.Mesh.Get(stage, '/World/Cloths/towel')
    q = np.asarray(mesh.GetPointsAttr().Get()).copy(); q[0, 0] += .01
    mesh.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(q)); stage.GetRootLayer().Save()
    run['initialization']['authored_stage'] = audit.artifact(tmp_path/'initial.usda')
    trace['towel_authored_q'] = q; write_trace(run, trace)
    with pytest.raises(ValueError, match='frozen initial vertices'):
        check(value, arrays, run)


def test_usd_disabled_warmup_attachment_cannot_match_an_enabled_mask(tmp_path):
    value, arrays, run, _ = fixture(tmp_path)
    stage = Usd.Stage.Open(str(tmp_path/'initial.usda'))
    stage.GetPrimAtPath('/World/Attachment0').GetAttribute('omniphysics:attachmentEnabled').Set(False)
    stage.GetRootLayer().Save()
    run['initialization']['authored_stage'] = audit.artifact(tmp_path/'initial.usda')
    with pytest.raises(ValueError, match='warmup attachment differs'):
        check(value, arrays, run)


def test_same_bits_detects_zero_sign_dtype_and_shape():
    q = np.array([0.], dtype=np.float32)
    assert not init.same_bits(q, -q)
    assert not init.same_bits(q, q.astype(np.float64))
    assert not init.same_bits(q, q[None])
    assert init.same_bits(q, q.copy())


def test_counted_handle_records_real_setter_dispatches_and_delegates_reads():
    events = []
    native = SimpleNamespace(count=1, get_simulation_nodal_positions=lambda: 'q',
        set_simulation_nodal_positions=lambda *a: events.append(('q', a)),
        set_simulation_nodal_velocities=lambda *a: events.append(('qd', a)))
    view = init.CountedSurfaceView(native)
    assert view.count == 1 and view.get_simulation_nodal_positions() == 'q'
    assert view.setter_calls == [0, 0]
    view.set_simulation_nodal_positions('new-q', 0)
    view.set_simulation_nodal_velocities('new-qd', 0)
    assert view.setter_calls == [1, 1] and [row[0] for row in events] == ['q', 'qd']


def test_formal_cli_keeps_old_default_and_exposes_only_explicit_new_options(tmp_path, monkeypatch):
    import sys
    from scripts.validation import laundry_cloth_interactions as harness
    observed = []
    monkeypatch.setattr(harness, 'run_task', lambda args: observed.append(args) or 0)
    argv = ['probe', '--case', 'towel_folding', '--output', str(tmp_path)]
    monkeypatch.setattr(sys, 'argv', argv)
    with pytest.raises(SystemExit) as first:
        harness.main()
    assert first.value.code == 0
    assert observed[-1].initialization == 'tensor-reset'
    assert observed[-1].fold_edge_alignment is False and observed[-1].fold_grip_span_m is None
    assert observed[-1].fold_trajectory == 'arc' and observed[-1].fold_peak_height_m is None
    assert observed[-1].passive_after_release_s is None
    monkeypatch.setattr(sys, 'argv', argv+['--initialization', 'author-only', '--fold-edge-alignment', '--fold-grip-span-m', '.04', '.04',
        '--fold-trajectory', 'bezier', '--fold-peak-height-m', '.16', '--passive-after-release-s', '12.'])
    with pytest.raises(SystemExit) as second:
        harness.main()
    assert second.value.code == 0
    assert observed[-1].initialization == 'author-only'
    assert observed[-1].fold_edge_alignment is True and observed[-1].fold_grip_span_m == [.04, .04]
    assert observed[-1].fold_trajectory == 'bezier' and observed[-1].fold_peak_height_m == .16
    assert observed[-1].passive_after_release_s == 12.


def long_clock_fixture(tmp_path):
    value, arrays, run, trace = fixture(tmp_path)
    count, dt = 31001, .0005
    tick = init.world_clock_tick(dt)
    value['task']['evaluation']['physics_dt_s'] = dt
    value['scene']['physics']['dt_s'] = dt
    for key, array in list(arrays.items()):
        if not key.endswith('_triangles'):
            arrays[key] = np.broadcast_to(array[0], (count, *array.shape[1:]))
    arrays['native_step'] = np.arange(count)
    arrays['time_s'] = arrays['native_step']*dt
    arrays['absolute_world_step'] = arrays['native_step']+2
    arrays['absolute_world_time_s'] = arrays['absolute_world_step']*tick
    arrays['attachment_enabled'] = np.zeros((count, 1), bool)
    arrays['attachment_enabled'][:2] = True
    arrays['towel_control_mask'] = np.zeros((count, len(arrays['towel_q'][0])), bool)
    arrays['towel_control_mask'][:2, :2] = True
    run['native_steps'] = count-1
    run['initialization']['world_clock_tick_s'] = tick
    run['native_world_reset_offset']['absolute_time_s'] = 2*tick
    trace['world_clock_tick_s'] = np.asarray(tick, np.float64)
    for phase in ('after_warmup', 'at_baseline'):
        trace['world_time_'+phase] = np.asarray(2*tick)
    write_trace(run, trace)
    return value, arrays, run


def test_31000_step_native_clock_accepts_exact_float32_callback_tick(tmp_path):
    value, arrays, run = long_clock_fixture(tmp_path)
    assert arrays['absolute_world_time_s'][30998] == 15.500000736210495
    assert run['native_world_reset_offset']['absolute_time_s'] == .0010000000474974513
    result = check(value, arrays, run)
    assert result['initialization']['world_clock_tick_s'] == float(np.float32(.0005))
    assert result['initialization']['relative_task_period_s'] == .0005
    assert arrays['time_s'][-1] == 15.5  # Frozen task clock remains float64 dt.


@pytest.mark.parametrize('change', [
    lambda a: a['absolute_world_step'].__setitem__(25000, 25003),
    lambda a: a['absolute_world_time_s'].__setitem__(25000, a['absolute_world_time_s'][25000]+1.e-5),
    lambda a: a.__setitem__('absolute_world_time_s', a['absolute_world_step']*.0005),
    lambda a: a.__setitem__('time_s', a['native_step']*float(np.float32(.0005))),
])
def test_long_clock_does_not_hide_missing_steps_or_arbitrary_time_drift(tmp_path, change):
    value, arrays, run = long_clock_fixture(tmp_path)
    change(arrays)
    with pytest.raises(ValueError):
        check(value, arrays, run)


@pytest.mark.parametrize('case, options', [
    ('towel_folding', ['--fold-peak-height-m', '.16']),
    ('towel_folding', ['--fold-trajectory', 'bezier', '--fold-peak-height-m', '-.1']),
    ('towel_folding', ['--fold-trajectory', 'bezier', '--fold-peak-height-m', 'nan']),
    ('towel_stacking', ['--fold-trajectory', 'bezier']),
    ('lift_release', ['--fold-trajectory', 'bezier']),
    ('lift_release', ['--passive-after-release-s', '12.']),
    ('towel_folding', ['--passive-after-release-s', 'nan']),
    ('towel_folding', ['--passive-after-release-s', '-1.']),
])
def test_invalid_bezier_cli_never_starts_the_native_harness(tmp_path, monkeypatch, case, options):
    import sys
    from scripts.validation import laundry_cloth_interactions as harness
    observed = []
    monkeypatch.setattr(harness, 'run_task', lambda args: observed.append(args))
    monkeypatch.setattr(harness, 'run', lambda args: observed.append(args))
    monkeypatch.setattr(sys, 'argv', ['probe', '--case', case, '--output', str(tmp_path), *options])
    with pytest.raises(SystemExit) as failure:
        harness.main()
    assert failure.value.code == 2 and not observed
