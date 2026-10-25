"""Failure cases for the diagnostic ledger, not native fluid qualification."""
from copy import deepcopy
import numpy as np
import pytest
from scenes.workrooms.living_room.tasks.water_pouring.liquid_ledger import LiquidLedger


def make():
    profile = [(0, .1, 0), (.1, .1, 0)]
    metadata = {'water': {'particle_count': 2, 'particle_mass_kg': .001, 'reference_density_kg_m3': 1000},
        'containers': {k: {'inner_profile_z_radius_spout_m': profile} for k in ('kettle', 'cup')},
        'tray': {'bounds_xy_m': [[.385, -.175], [.915, .175]], 'interior_floor_z_m': .786}, 'table_top_z_m': .78}
    inverse = {'kettle': np.eye(4), 'cup': np.eye(4)}
    inverse['cup'][3, 0] = -1.
    return LiquidLedger(metadata, dt=.01, initial_time=0), inverse


def sample(ledger, inverse, points, time):
    return ledger.update(time_s=time, ids=[0, 1], points=points, velocities=np.zeros((2, 3)), world_to_local=inverse)


def test_spill_remains_after_recapture_and_is_not_double_counted():
    ledger, inverse = make()
    first = sample(ledger, inverse, [[0, 0, .05], [.65, 0, .789]], .01)
    assert first['cumulative_spill_ml'] == 1
    second = sample(ledger, inverse, [[0, 0, .05], [1, 0, .05]], .02)
    assert second['counts']['cup'] == 1
    assert second['cumulative_spill_ml'] == 1
    assert second['recaptured_spilled_particles'] == 1
    assert sum(second['current_mass_kg'].values()) == .002


def test_above_rim_and_outside_wall_never_credit_cup():
    ledger, inverse = make()
    row = sample(ledger, inverse, [[1, 0, .101], [1.11, 0, .05]], .01)
    assert row['counts']['cup'] == 0
    assert not row['task_success']


@pytest.mark.parametrize('bad', ['missing', 'repeat', 'id', 'nan', 'overlap'])
def test_bad_observations_fail_closed_without_advancing_ledger(bad):
    ledger, inverse = make()
    kwargs = dict(time_s=.01, ids=[0, 1], points=[[0, 0, .05], [1, 0, .05]], velocities=np.zeros((2, 3)), world_to_local=deepcopy(inverse))
    if bad == 'missing': kwargs['time_s'] = .02
    if bad == 'repeat': kwargs['time_s'] = 0
    if bad == 'id': kwargs['ids'] = [1, 0]
    if bad == 'nan': kwargs['points'][0][2] = float('nan')
    if bad == 'overlap': kwargs['world_to_local']['cup'] = np.eye(4)
    with pytest.raises(ValueError): ledger.update(**kwargs)
    assert ledger.tick == 0
    assert ledger.previous_time == 0


def test_water_resting_on_tray_rim_is_spill_not_perpetual_flight():
    ledger, inverse = make()
    row = sample(ledger, inverse, [[.8, .17028, .8055], [.8, .17, .9]], .01)
    assert row['surface_interception_counts']['tray_rim'] == 1
    assert row['counts']['spilled_outside'] == 1
    assert row['counts']['in_flight_or_unresolved'] == 1
    assert row['cumulative_spill_ml'] == 1
