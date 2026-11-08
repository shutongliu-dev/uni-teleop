import numpy as np
import pytest

from scripts.validation.sponge_freeflight import TABLE, author
from scripts.validation.sponge_freeflight_metrics import evaluate, force_for_step, lumped_masses


def ideal_flight():
    foam = {'rest_points_m': [[0., 0., 0.], [.01, 0., 0.], [0., .01, 0.], [0., 0., .01]],
            'tetrahedra': [[0, 1, 2, 3]], 'total_mass_kg': .07128,
            'damping_receipt': {'settling_threshold_m_s': .1}}
    dt = float(np.float32(.0005)); time = np.arange(401)*dt
    force = np.array([force_for_step(i) for i in range(401)], dtype=np.float32)
    velocity = np.zeros((401, 3))
    velocity[1:] = np.cumsum((force[1:]/.0792+[0., 0., -9.81])*dt, axis=0)
    position = np.zeros((401, 3)); position[1:] = np.cumsum((velocity[1:]+velocity[:-1])*.5*dt, axis=0)
    pad = np.tile([0., 0., 1.3, 0., 0., 0., 1.], (401, 1)); pad[:, :3] += position
    nodes = np.asarray(foam['rest_points_m'])[None]+[0., 0., 1.3]+position[:, None]
    raw = {'time_s': time, 'native_step': np.arange(2, 403), 'pad_q': pad,
           'pad_qd': np.column_stack((velocity, np.zeros_like(velocity))),
           'foam_q': nodes, 'foam_qd': np.broadcast_to(velocity[:, None], nodes.shape).copy(),
           'force_world_n': force}
    return raw, foam


def test_exact_pulse_and_float64_declared_mass_momentum():
    raw, foam = ideal_flight()
    checks, measured, derived = evaluate(raw, foam, .00792, [0., 0., .002], [0., 0., -9.81])
    assert all(checks.values()), checks
    assert np.isclose(measured['actual_force_pulse_impulse_ns'][0], .02, rtol=1e-6)
    assert np.isclose(measured['total_accounted_mass_kg'], .0792)
    assert measured['damping_correction_applied'] is False
    assert derived['momentum_ns'].dtype == np.float64
    assert np.count_nonzero(raw['force_world_n'][:, 0]) == 100


@pytest.mark.parametrize('failure', ['heavier_inertia', 'bad_gravity', 'release_loss', 'fixed_pose', 'fixed_height', 'quiet_node'])
def test_motion_counterexamples_do_not_pass(failure):
    raw, foam = ideal_flight()
    if failure == 'heavier_inertia':
        raw['foam_qd'][:, :, 0] /= 1.62; raw['pad_qd'][:, 0] /= 1.62
        expected = 'horizontal_pulse_momentum'
    elif failure == 'bad_gravity':
        raw['foam_qd'][:, :, 2] *= 1.62; raw['pad_qd'][:, 2] *= 1.62
        expected = 'gravity_momentum_all_windows'
    elif failure == 'release_loss':
        raw['foam_qd'][201:, :, 0] *= .8; raw['pad_qd'][201:, 0] *= .8
        expected = 'release_horizontal_momentum'
    elif failure == 'fixed_pose':
        raw['foam_q'][:] = raw['foam_q'][0]; raw['pad_q'][:] = raw['pad_q'][0]
        expected = 'actual_horizontal_com_motion'
    elif failure == 'fixed_height':
        raw['foam_q'][:, :, 2] = raw['foam_q'][0, :, 2]; raw['pad_q'][:, 2] = raw['pad_q'][0, 2]
        expected = 'actual_gravity_com_motion'
    else:
        raw['foam_qd'][150, 0] = 0.
        expected = 'settling_not_indicated_during_pulse_release'
    checks, _, _ = evaluate(raw, foam, .00792, [0., 0., .002], [0., 0., -9.81])
    assert not checks[expected]


@pytest.mark.parametrize('failure', ['missing_step', 'force_outside_pulse', 'nonfinite'])
def test_invalid_native_archive_is_rejected(failure):
    raw, foam = ideal_flight()
    if failure == 'missing_step': raw['native_step'][200] += 1
    elif failure == 'force_outside_pulse': raw['force_world_n'][100, 0] = .4
    else: raw['foam_qd'][201, 0, 0] = np.nan
    with pytest.raises(ValueError): evaluate(raw, foam, .00792, [0., 0., .002], [0., 0., -9.81])


def test_rest_volume_weights_include_all_nodes_and_reject_inversion():
    _, foam = ideal_flight()
    assert np.allclose(lumped_masses(foam), .07128/4)
    foam['tetrahedra'] = [[0, 2, 1, 3]]
    with pytest.raises(ValueError, match='Positive rest'): lumped_masses(foam)
    foam['tetrahedra'] = [[0, 1, 2, 3]]; foam['rest_points_m'].append([1., 1., 1.])
    with pytest.raises(ValueError, match='Every archived'): lumped_masses(foam)


def test_cpu_author_lifts_both_layers_without_spring_or_mass_override(tmp_path):
    Usd = pytest.importorskip('pxr.Usd'); UsdGeom = pytest.importorskip('pxr.UsdGeom')
    UsdPhysics = pytest.importorskip('pxr.UsdPhysics'); Gf = pytest.importorskip('pxr.Gf')
    file = tmp_path/'source.usda'; source = Usd.Stage.CreateNew(str(file))
    root = '/World/KitchenTask'; pad = root+'/Actors/sponge'; foam = root+'/VolumeBodies/foam'
    for path in (pad, foam):
        prim = UsdGeom.Xform.Define(source, path)
        prim.AddTranslateOp().Set(Gf.Vec3d(.25, -.67, .9045))
        UsdPhysics.MassAPI.Apply(prim.GetPrim()).CreateMassAttr(.00792 if path == pad else .07128)
    source.GetRootLayer().Save(); before = file.read_bytes()
    meta = {'actors': {'sponge': pad}, 'volume_bodies': {'foam': {'prim_path': foam}}}
    stage = Usd.Stage.CreateInMemory(); assert author(stage, {'root_prim_path': root}, file, meta) == pad
    for path in (pad, foam):
        assert np.allclose(stage.GetPrimAtPath(path).GetAttribute('xformOp:translate').Get(), [.25, -.67, 1.2045])
    assert not any(p.IsA(UsdPhysics.Joint) for p in stage.Traverse())
    assert not UsdPhysics.RigidBodyAPI(stage.GetPrimAtPath(TABLE))
    assert file.read_bytes() == before
    assert 'physics:mass' not in stage.GetRootLayer().ExportToString()
    assert np.isclose(stage.GetPrimAtPath(TABLE).GetAttribute('physxCollision:contactOffset').Get(), .002)


def test_origin_translation_is_opt_in_and_rejects_existing_root_transform(tmp_path):
    from scripts.validation.sponge_freeflight import translate_instrument_frame
    Usd = pytest.importorskip('pxr.Usd'); UsdGeom = pytest.importorskip('pxr.UsdGeom'); Gf = pytest.importorskip('pxr.Gf')
    stage = Usd.Stage.CreateInMemory(); root = UsdGeom.Xform.Define(stage, '/World/KitchenTask')
    root.AddTranslateOp().Set(Gf.Vec3d(1., 0., 0.)); before = stage.GetRootLayer().ExportToString()
    m = {'root_prim_path': '/World/KitchenTask'}
    receipt = translate_instrument_frame(stage, m, {}, center_origin=False)
    assert receipt['applied_translation_m'] == [0., 0., 0.]
    assert stage.GetRootLayer().ExportToString() == before
    with pytest.raises(ValueError, match='untransformed task root'):
        translate_instrument_frame(stage, m, {}, center_origin=True)
    assert stage.GetRootLayer().ExportToString() == before
