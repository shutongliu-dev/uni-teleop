from pathlib import Path
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from bench.paths import DATA_ROOT
from scripts.validation.spoon_freeflight import evaluate, load_asset, author, collision_bounds, physical_inventory, SHIFT
from scripts.validation.sponge_freeflight_metrics import force_for_step

MASS = .048668
COM = np.array([0., .004697, .004694])


def ideal(response=1.):
    dt = float(np.float32(.0005)); time = np.arange(401)*dt
    force = np.array([force_for_step(i) for i in range(401)], np.float32)
    v = np.zeros((401, 6)); v[1:, :3] = np.cumsum((response*force[1:]/MASS+[0., 0., -9.81])*dt, axis=0)
    q = np.tile([.2, -.7, 3., 0., 0., 0., 1.], (401, 1))
    q[1:, :3] += np.cumsum(.5*(v[:-1, :3]+v[1:, :3])*dt, axis=0)
    positions = np.zeros((401, 3), np.float32); positions[1:] = q[:-1, :3]+COM
    return dict(time_s=time, native_step=np.arange(401)+2, body_q=q, body_qd=v,
                force_world_n=force, force_position_world_m=positions, torque_world_nm=np.zeros((401, 3)))


def test_actual_mass_and_nonzero_com():
    checks, measured, derived = evaluate(ideal(), MASS, COM, -9.81, MASS)
    assert all(checks.values()), checks
    assert measured['measured_inertial_mass_kg'] == pytest.approx(MASS)
    assert measured['actual_force_pulse_impulse_ns'][0] == pytest.approx(.02)
    assert derived['momentum_ns'].dtype == np.float64


@pytest.mark.parametrize('response', [1/7, 7])
def test_sevenfold_solver_response_cannot_pass(response):
    checks, measured, _ = evaluate(ideal(response), MASS, COM, -9.81, MASS)
    assert not checks['horizontal_pulse_momentum']
    assert measured['measured_inertial_mass_kg'] == pytest.approx(MASS/response)


def test_wrong_force_point_rejected():
    raw = ideal(); raw['force_position_world_m'][101] -= COM
    with pytest.raises(ValueError, match='COM'): evaluate(raw, MASS, COM, -9.81, MASS)


def test_native_report_schema_empty_relation_is_instrumentation_only():
    Usd = pytest.importorskip('pxr.Usd')
    stage = Usd.Stage.CreateInMemory(); p = stage.DefinePrim('/Body', 'Xform')
    before = physical_inventory(stage, '/Body')
    rel = p.CreateRelationship('physxContactReport:reportPairs')
    assert physical_inventory(stage, '/Body') == before
    rel.SetTargets(['/Other'])
    assert physical_inventory(stage, '/Body') != before


def test_original_asset_only_target_translation_changes():
    Usd = pytest.importorskip('pxr.Usd')
    path = Path(f'{DATA_ROOT}/assets/task_parts/kitchen/utensil_sorting/household_contactconvex_v5/manifest.json')
    if not path.exists(): pytest.skip('Original immutable v5 fixture not available')
    manifest, usd, meta, spec = load_asset(path, 'spoon_0')
    stage = Usd.Stage.CreateInMemory(); receipt = author(stage, manifest, usd, meta, 'spoon_0')
    assert receipt['original_composed_property_graph_preserved']
    assert np.array_equal(receipt['applied_translation_m'], SHIFT)
    local, statics, colliders = collision_bounds(stage, list(meta['actors'].values()))
    assert len(colliders[meta['actors']['spoon_0']]) == 220
    assert len(colliders[meta['actors']['spoon_1']]) == 220
    assert statics and all(np.isfinite(x).all() for x in local.values())
    body = stage.GetPrimAtPath(meta['actors']['spoon_0'])
    assert body.GetAttribute('physics:mass').Get() == pytest.approx(spec['mass_kg'])
    assert np.allclose(body.GetAttribute('physics:centerOfMass').Get(), COM, atol=1e-7)
