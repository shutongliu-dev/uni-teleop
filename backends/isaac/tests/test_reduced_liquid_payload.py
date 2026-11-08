from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from bench.simulation.isaacsim_backend.reduced_liquid_payload import (
    SCHEMA, NativeLiquidPayload, composite_properties)


class NativeView:
    """CPU transport double, never a claim of native physical validation."""
    count = 1

    def __init__(self, mass=.12):
        self.mass = np.array([[mass]], dtype=np.float32)
        self.com = np.array([[0., 0., .04, 0., 0., 0., 1.]], dtype=np.float32)
        self.inertia = np.diag([8.e-5, 9.e-5, 6.e-5]).reshape(1, 9).astype(np.float32)
        self.writes = []

    def get_masses(self): return self.mass.copy()
    def get_coms(self): return self.com.copy()
    def get_inertias(self): return self.inertia.copy()
    def set_masses(self, data, indices): self.writes.append('mass'); self.mass = data.copy()
    def set_coms(self, data, indices): self.writes.append('com'); self.com = data.copy()
    def set_inertias(self, data, indices): self.writes.append('inertia'); self.inertia = data.copy()


def setup():
    spec = {'schema': SCHEMA, 'state_schema': 'example-conserved-volume-v1',
        'density_kg_m3': 1000., 'initial_total_ml': 80.,
        'unbound_volume_paths': [['flight'], ['spill']],
        'bindings': [{'body_id': 'vessel', 'volume_path': ['bodies', 'vessel'],
            'dry_mass_kg': .12, 'radius_m': .032, 'floor_center_m': [.008, 0., .004],
            'axis_body': [0., 0., 1.], 'maximum_volume_ml': 200.}]}
    view = NativeView()
    payload = NativeLiquidPayload(spec, {'vessel': view}, lambda x, device=None: np.asarray(x))
    return payload, view, spec


def state(volume=80., step=1):
    return {'schema': 'example-conserved-volume-v1', 'bodies': {'vessel': volume},
            'flight': 0., 'spill': 80.-volume, 'native_step': step, 'native_time_s': step*.0005}


def sync(payload, value, baseline=False):
    payload.sync(value, native_step=value['native_step'], native_time_s=value['native_time_s'], baseline=baseline)


def test_mass_com_inertia_follow_volume_and_empty_returns_exact_dry_baseline():
    p, view, _ = setup()
    dry = deepcopy(p.dry['vessel'])
    sync(p, state(), baseline=True)
    assert view.mass[0, 0] == pytest.approx(.20)
    assert view.com[0, 0] > 0.  # Real offset liquid shifts COM; no weight-only surrogate.
    assert view.inertia[0, 2] != 0.  # Full body-frame parallel-axis product of inertia.
    assert view.writes == ['mass', 'com', 'inertia']
    first = p.capture()
    assert int(first['liquid_payload_effective_from_native_step']) == 2
    sync(p, state(0., 2))
    np.testing.assert_array_equal(view.mass, np.array([[dry['mass_kg']]], np.float32))
    np.testing.assert_allclose(view.com[0, :3], dry['com_m'], atol=0., rtol=0.)
    np.testing.assert_allclose(view.inertia.reshape(3, 3), dry['inertia_kg_m2'], atol=0., rtol=0.)
    p.begin_episode()
    sync(p, state(), baseline=True)
    assert view.mass[0, 0] == pytest.approx(.20)  # Repeated reset never counts the water twice.


def test_composition_is_covariant_under_a_changed_body_coordinate_frame():
    p, _, spec = setup()
    dry = p.dry['vessel']; row = spec['bindings'][0]
    expected = composite_properties(dry, 50., 1000., row)
    angle = .7
    rotation = np.array([[np.cos(angle), 0., np.sin(angle)], [0., 1., 0.], [-np.sin(angle), 0., np.cos(angle)]])
    moved_dry = {**dry, 'com_m': rotation@dry['com_m'], 'inertia_kg_m2': rotation@dry['inertia_kg_m2']@rotation.T}
    moved_row = {**row, 'axis_body': rotation@row['axis_body'], 'floor_center_m': rotation@row['floor_center_m']}
    actual = composite_properties(moved_dry, 50., 1000., moved_row)
    np.testing.assert_allclose(actual['com_m'], rotation@expected['com_m'], atol=1.e-15)
    np.testing.assert_allclose(actual['inertia_kg_m2'], rotation@expected['inertia_kg_m2']@rotation.T, atol=1.e-15)
    assert np.linalg.eigvalsh(actual['inertia_kg_m2']).min() > 0.


@pytest.mark.parametrize('bad', [float('nan'), -1., 201., True])
def test_invalid_volume_rejected_before_any_native_write(bad):
    p, view, _ = setup(); value = state(); value['bodies']['vessel'] = bad
    with pytest.raises(ValueError): sync(p, value, baseline=True)
    assert view.writes == []


def test_unclosed_duplicate_inventory_and_particle_double_count_rejected():
    p, view, spec = setup(); value = state(30.); value['spill'] = 40.
    with pytest.raises(ValueError, match='conserved'): sync(p, value, baseline=True)
    assert view.writes == []
    with pytest.raises(ValueError, match='particle'):
        NativeLiquidPayload(spec, {'vessel': view}, np.asarray, physical_particle_sets=True)
    spec['unbound_volume_paths'] = [['bodies', 'vessel']]
    with pytest.raises(ValueError, match='unique'):
        NativeLiquidPayload(spec, {'vessel': view}, np.asarray)


def test_native_tick_is_exact_and_provider_cannot_replay_a_frame():
    p, view, _ = setup(); sync(p, state(), baseline=True)
    old_count = len(view.writes)
    with pytest.raises(ValueError, match='consecutive'): sync(p, state())
    value = state(70., 2)
    with pytest.raises(ValueError, match='stale'):
        p.sync(value, native_step=2, native_time_s=.010)
    assert len(view.writes) == old_count


def test_explicit_provider_density_must_agree_with_payload_but_is_optional():
    p, view, _ = setup(); value = {**state(), 'density_kg_m3': 900.}
    with pytest.raises(ValueError, match='density'): sync(p, value, baseline=True)
    assert not view.writes
    value['density_kg_m3'] = 1000.
    sync(p, value, baseline=True)
    assert view.mass[0, 0] == pytest.approx(.20)


def test_dropped_native_setter_fails_instead_of_reporting_requested_values():
    p, view, _ = setup()
    view.set_masses = lambda data, indices: None
    with pytest.raises(RuntimeError, match='readback mismatch'):
        sync(p, state(), baseline=True)
    assert p.last is None


def test_initial_mass_must_be_dry_and_body_must_be_declared():
    _, _, spec = setup()
    with pytest.raises(ValueError, match='dry mass'):
        NativeLiquidPayload(spec, {'vessel': NativeView(.20)}, np.asarray)
    with pytest.raises(ValueError, match='distinct task rigid objects'):
        NativeLiquidPayload(spec, {}, np.asarray)


def test_environment_samples_provider_then_updates_next_step_payload_only_when_enabled():
    from bench.simulation.isaacsim_backend.env import FR3DexterousIsaacEnv
    calls = []
    reader = SimpleNamespace(sample=lambda **kw: calls.append(('sample', kw['native_step'])),
        reduced_dynamics=SimpleNamespace(on_observe=lambda: state()))
    env = SimpleNamespace(_object_views={}, _task_reader=reader, time_s=0., dt=.0005,
        world=SimpleNamespace(current_time=.0005, current_time_step_index=1))
    FR3DexterousIsaacEnv._sample_task_state(env, [], baseline=True)
    assert calls == [('sample', 1)]
    env._liquid_payload = SimpleNamespace(sync=lambda value, **kw: calls.append(('payload', value['native_step'], kw)))
    FR3DexterousIsaacEnv._sample_task_state(env, [], baseline=True)
    assert [x[0] for x in calls] == ['sample', 'sample', 'payload']
    assert calls[-1][2] == {'native_time_s': .0005, 'native_step': 1, 'baseline': True}


def audit_context():
    p, view, spec = setup(); sync(p, state(), baseline=True)
    initial = p.capture(); properties = p.metadata()
    metadata = {'scene_spec': {'reduced_liquid_payload': spec}, 'reduced_liquid_payload': properties,
        'scene_info': {'objects': {'vessel': {'mass_kg': .12}, 'dry_other': {'mass_kg': .05}}},
        'object_names': ['vessel', 'dry_other'], 'native_robot_link_masses_kg': {'link': 1.},
        'native_object_masses_kg': {'vessel': float(view.mass[0, 0]), 'dry_other': .05},
        'native_object_com_pose_xyzw': {'vessel': view.com[0].tolist()},
        'native_object_inertias_kg_m2': {'vessel': view.inertia.reshape(3, 3).tolist()}}
    sync(p, state(0., 2)); final = p.capture()
    states = {key: np.stack([initial[key], final[key]]) for key in initial}
    states.update(native_physics_step_index=np.array([1, 2]), native_world_time_s=np.array([.0005, .001]))
    return {'metadata': metadata, 'states': states}


def test_archive_audit_accepts_actual_wet_metadata_and_all_frame_empty_transition():
    from bench.isaacsim.audit import check_native_masses
    result = check_native_masses(audit_context())
    assert result['native_object_masses_kg']['vessel'] == pytest.approx(.20)
    assert result['reduced_liquid_payload']['captured_frames'] == 2
    assert result['object_mass_errors_kg']['vessel'] == pytest.approx(0., abs=1.e-8)


@pytest.mark.parametrize('field', ['mass_kg', 'com_pose_xyzw', 'inertia_kg_m2', 'applied_after_native_step', 'unbound_volume_ml'])
def test_archive_audit_rejects_missing_weight_inertia_clock_or_inventory(field):
    from bench.isaacsim.audit import check_native_masses
    context = audit_context()
    context['states']['liquid_payload_'+field][1] += 1 if field == 'applied_after_native_step' else .001
    with pytest.raises((ValueError, AssertionError)):
        check_native_masses(context)


def test_old_release_mass_audit_still_uses_original_dry_readback_without_payload_fields():
    from bench.isaacsim.audit import check_native_masses
    context = audit_context(); metadata = context['metadata']
    metadata['scene_spec'] = {}; del metadata['reduced_liquid_payload']
    metadata['native_object_masses_kg']['vessel'] = .12
    result = check_native_masses(context)
    assert 'reduced_liquid_payload' not in result
    metadata['native_object_masses_kg']['vessel'] = .20
    with pytest.raises(ValueError): check_native_masses(context)


@pytest.mark.parametrize('mutation', [None, 'missing', 'mass', 'inertia'])
def test_ordinary_http_archive_requires_opted_in_payload_readback(tmp_path, mutation):
    from test_agent_physics_audit import Recording
    from bench.evaluation.physics import audit_native_episode
    recording = Recording(tmp_path, cloth=False)
    context = audit_context()
    recording.scene.update(context['metadata']['scene_spec'])
    recording.meta.update({k: v for k, v in context['metadata'].items() if k != 'scene_spec'})
    recording.meta['reduced_liquid_payload']['applied_after_native_step'] = 24000
    recording.meta['reduced_liquid_payload']['native_time_s'] = 12.
    for key, value in context['states'].items():
        if key.startswith('liquid_payload_'):
            recording.states[key] = value
    recording.states['liquid_payload_applied_after_native_step'] = np.array([24000, 24040])
    recording.states['liquid_payload_effective_from_native_step'] = np.array([24001, 24041])
    recording.states['liquid_payload_native_time_s'] = np.array([12., 12.02])
    recording.states['object_q'] = np.tile([[[0, 0, 1., 0, 0, 0, 1.], [1., 0, 1., 0, 0, 0, 1.]]], (2, 1, 1))
    recording.states['object_qd'] = np.zeros((2, 2, 6))
    if mutation == 'missing':
        del recording.states['liquid_payload_inertia_kg_m2']
    elif mutation in ('mass', 'inertia'):
        key = 'liquid_payload_'+('mass_kg' if mutation == 'mass' else 'inertia_kg_m2')
        recording.states[key][1] += .001
    result = audit_native_episode(recording.write())
    check = result['checks']['native_liquid_payload']
    assert check['required'] is True
    assert check['status'] == ('passed' if mutation is None else 'unchecked' if mutation == 'missing' else 'failed')
    assert result['physics_valid'] is (mutation is None), result


def test_old_http_archive_check_inventory_has_no_optional_payload_gate(tmp_path):
    from test_agent_physics_audit import Recording
    from bench.evaluation.physics import audit_native_episode, audit_source_paths
    result = audit_native_episode(Recording(tmp_path, cloth=False).write())
    assert result['physics_valid'] is True
    assert 'native_liquid_payload' not in result['checks']
    assert 'reduced_liquid_payload.py' in {p.name for p in audit_source_paths()}


@pytest.mark.parametrize('mutation', [None, 'usd_hash', 'body_identity', 'missing_mass', 'wet_mass', 'unbound_mass', 'unpinned_layer'])
def test_referenced_native_bundle_audits_pinned_dry_masses_including_unbound_body(tmp_path, mutation):
    import hashlib
    from pxr import Usd, UsdGeom, UsdPhysics
    from bench.isaacsim.audit import check_native_masses

    context = audit_context()
    meta = context['metadata']
    meta['scene_info']['objects']['plunger'] = meta['scene_info']['objects'].pop('dry_other')
    meta['native_object_masses_kg']['plunger'] = meta['native_object_masses_kg'].pop('dry_other')
    for name, mass in [('source', .10), ('receiver_a', .06), ('receiver_b', .06)]:
        meta['scene_info']['objects'][name] = {'mass_kg': mass}
        meta['native_object_masses_kg'][name] = mass
    meta['object_names'] = list(meta['scene_info']['objects'])
    path = tmp_path/'task.usda'
    stage = Usd.Stage.CreateNew(str(path))
    for name, row in meta['scene_info']['objects'].items():
        prim = UsdGeom.Xform.Define(stage, '/Task/'+name).GetPrim()
        UsdPhysics.RigidBodyAPI.Apply(prim)
        if mutation != 'missing_mass' or name != 'plunger':
            UsdPhysics.MassAPI.Apply(prim).CreateMassAttr(row['mass_kg'])
    if mutation == 'unpinned_layer':
        dependency = Usd.Stage.CreateNew(str(tmp_path/'dependency.usda'))
        UsdGeom.Xform.Define(dependency, '/MutableDependency')
        dependency.GetRootLayer().Save()
        stage.GetRootLayer().subLayerPaths = ['dependency.usda']
    stage.GetRootLayer().Save()
    source_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    bundle = {'usd_path': str(path), 'usd_sha256': source_hash,
              'rigid_bodies': {name: '/Task/'+name for name in meta['object_names']}}
    meta['scene_spec']['native_task_bundle'] = bundle
    for name in meta['object_names']:
        meta['scene_info']['objects'][name] = {'kind': 'referenced_native_task_body',
            'path': '/Task/'+name, 'source_usd_sha256': source_hash}
    if mutation == 'usd_hash':
        bundle['usd_sha256'] = '0'*64
    elif mutation == 'body_identity':
        bundle['rigid_bodies']['plunger'] = '/Task/vessel'
    elif mutation == 'wet_mass':
        meta['native_object_masses_kg']['vessel'] = .12
    elif mutation == 'unbound_mass':
        meta['native_object_masses_kg']['plunger'] = .08
    if mutation is None:
        result = check_native_masses(context)
        assert set(result['object_mass_errors_kg']) == {'vessel', 'plunger', 'source', 'receiver_a', 'receiver_b'}
        assert result['native_object_masses_kg']['vessel'] == pytest.approx(.20)
        assert result['reduced_liquid_payload']['captured_frames'] == 2
    else:
        with pytest.raises(ValueError):
            check_native_masses(context)
    assert hashlib.sha256(path.read_bytes()).hexdigest() == source_hash


def test_parameterized_release_rebinds_only_exact_canonical_dry_masses():
    from bench.simulation.isaacsim_backend.reduced_liquid_payload import parameterized_dry_masses
    spec = {'schema': 'reduced-liquid-payload-v1', 'bindings': [
        {'body_id': 'source', 'dry_mass_kg': .12}, {'body_id': 'receiver_a', 'dry_mass_kg': .06},
        {'body_id': 'receiver_b', 'dry_mass_kg': .07}, {'body_id': 'unrecorded', 'dry_mass_kg': .05}]}
    record = {'source': {'before': {'mass_kg': .11999999731779099}, 'mass_kg': .1381487101316452},
              'receiver_a': {'before': {'mass_kg': .05999999865889549}, 'mass_kg': .0690743550658226},
              # Frozen payload mass is not the recorded canonical value: never silently replaced.
              'receiver_b': {'before': {'mass_kg': .06}, 'mass_kg': .069}}
    reduced = {'configuration': {'reset_case': {'parameterized_physics': {'applied': {'body_physics': record}}}}}
    bound, rebound = parameterized_dry_masses(spec, reduced)
    assert [row['dry_mass_kg'] for row in bound['bindings']] == [.1381487101316452, .0690743550658226, .07, .05]
    assert [row['body_id'] for row in rebound] == ['source', 'receiver_a']
    assert spec['bindings'][0]['dry_mass_kg'] == .12  # the frozen input itself is untouched
    assert parameterized_dry_masses(spec, {'configuration': {'reset_case': {}}}) == (spec, [])
    assert parameterized_dry_masses(spec, None) == (spec, [])
    # Seed 0 records unscaled masses (float32 readback); the frozen value is kept.
    canonical = {'source': {'before': {'mass_kg': .11999999731779099}, 'mass_kg': .11999999731779099}}
    reduced0 = {'configuration': {'reset_case': {'parameterized_physics': {'applied': {'body_physics': canonical}}}}}
    assert parameterized_dry_masses(spec, reduced0) == (spec, [])
