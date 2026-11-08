"""Particle-center surface geometry regressions; these are CPU fixtures only."""
import math

import numpy as np
import pytest
from pxr import Usd, UsdGeom

from scenes.workrooms.living_room.tasks.water_pouring import assets, rules
from scenes.workrooms.living_room.tasks.water_pouring.liquid_ledger import LiquidLedger
from scenes.workrooms.living_room.tasks.water_pouring.native_observer import NativeTaskObserver
from scripts.validation.living_pour_probe import native_inverse


def test_original_three_mm_geometry_and_public_thresholds_are_retained():
    config = rules.evaluation_configuration()
    assert config['geometry_spill_envelope_m'] == .006
    assert 'geometry_spill_envelope_basis' not in config
    assert rules.public_rules()['goal']['max_spill_ml'] == 5.
    assert rules.thresholds()['dwell_s'] == .3


@pytest.mark.parametrize('mutate', [
    lambda w: w.pop('collider_rest_offset_m'),
    lambda w: w.pop('particle_spacing_m'),
    lambda w: w.update(collider_rest_offset_m=float('nan')),
    lambda w: w.update(collider_rest_offset_m=-.1),
    lambda w: w.update(collider_rest_offset_m=.05),
    lambda w: w.update(particle_spacing_m=.02),
    lambda w: w.update(particle_spacing_m=True),
])
def test_unknown_or_invalid_offset_is_not_guessed(mutate):
    water = {'particle_spacing_m': .008, 'collider_rest_offset_m': .008 * 5 / 6}
    mutate(water)
    with pytest.raises(ValueError):
        rules.spill_geometry_parameters(water)


def test_native_and_diagnostic_classifiers_resolve_real_rest_height_and_recapture():
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageMetersPerUnit(stage, 1.)
    meta = assets.author_task(stage, particle_spacing_m=.008)
    configuration = {'evaluation': rules.evaluation_configuration(asset_metadata=meta)}
    assert configuration['evaluation']['geometry_spill_envelope_m'] == pytest.approx(.007666666666666667)
    original = np.array(UsdGeom.Points(stage.GetPrimAtPath(meta['water']['particles_path'])).GetPointsAttr().Get())
    objects, inverse = {}, {}
    for name, body in meta['containers'].items():
        yaw = body['initial_yaw_rad']
        pose = [*body['initial_position_m'], 0, 0, math.sin(yaw / 2), math.cos(yaw / 2)]
        inverse[name] = native_inverse(pose)
        objects[name] = {'position_m': pose[:3], 'quaternion_wxyz': [pose[6], *pose[3:6]],
                         'linear_velocity_m_s': [0, 0, 0], 'angular_velocity_rad_s': [0, 0, 0]}
    frame = {'baseline': True, 'time_s': 0., 'native_time_s': 0., 'native_step': 0,
             'physics_dt_s': .0005, 'objects': objects, 'contacts': [],
             'particle_sets': {'water': {'positions_m': original.copy(),
                 'velocities_m_s': np.zeros_like(original), 'ids': np.arange(488), 'total_mass_kg': .25}}}
    observer = NativeTaskObserver(configuration)
    observer.on_frame(frame)
    assert observer.on_observe()['water']['counts']['kettle'] == 488
    ledger = LiquidLedger(meta, dt=.0005, initial_time=0.)
    points = original.copy()
    rest = meta['water']['collider_rest_offset_m']
    points[0] = [.65, 0, .786 + rest]          # Actual coarse-particle tray rest.
    points[1] = [.65, 0, .836]                # Free flight, five centimetres up.
    points[2] = [.918, .17, .803 + rest]      # Resting over the outside rim edge.
    points[3] = [.95, .3, .78 + rest]         # Table outside the tray.
    points[4] = [1.5, .3, rest]              # Floor outside the table.
    points[5] = [.65, 0, .786 + rest + .0015]  # Even nearby flight above the envelope stays pending.
    for tick, current in enumerate((points, original), start=1):
        frame.update(baseline=False, time_s=tick * .0005, native_time_s=tick * .0005, native_step=tick)
        frame['particle_sets']['water']['positions_m'] = current
        observer.on_frame(frame)
        observed = observer.on_observe()
        assert not observed['errors']
        measured = ledger.update(time_s=tick * .0005, ids=np.arange(488), points=current,
                                 velocities=np.zeros_like(current), world_to_local=inverse)
        assert observed['water']['counts'] == measured['counts']
        assert observed['water']['cumulative_unique_spilled_particles'] == 4
        if tick == 1:
            assert measured['counts']['spilled_outside'] == 4
            assert measured['counts']['in_flight_or_unresolved'] == 2
        else:
            assert measured['counts']['kettle'] == 488
            assert measured['recaptured_spilled_particles'] == 4
            assert measured['cumulative_spill_ml'] == pytest.approx(4 * 250 / 488)
