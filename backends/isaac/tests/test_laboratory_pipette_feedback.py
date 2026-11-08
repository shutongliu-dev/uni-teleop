import math

import numpy as np
import pytest
from pxr import Usd, UsdGeom

from scenes.workrooms.laboratory.tasks.pipette_transfer.assets import author_task, PISTON_AREA_M2
from scenes.workrooms.laboratory.tasks.pipette_transfer.liquid import (
    FEEDBACK_SCHEMA, PistonVolumeModel, ReducedLiquidDynamics, SCHEMA,
)


def setup():
    metadata = author_task(Usd.Stage.CreateInMemory())
    p = metadata['task_parameters']['hydraulics']
    poses = {o['id']: o['initial_pose_xyzw'] for o in metadata['objects'] if o['id'] in p['containers']}
    return metadata, p, poses


def stroke(model, poses, tip, start, end, steps=100):
    for q in np.linspace(start, end, steps+1)[1:]:
        model.step(.005, piston_q_m=q, tool_pose_xyzw=[*tip, 0., 0., 0., 1.], container_poses=poses)


def test_v2_retains_identical_v1_water_equations_and_limits():
    _, parameters, poses = setup(); newer = dict(parameters, schema=FEEDBACK_SCHEMA)
    legacy, feedback = PistonVolumeModel(dict(parameters, schema=SCHEMA)), PistonVolumeModel(newer)
    q20 = 20e-6/PISTON_AREA_M2
    phases = [(np.array(poses['source'][:3])+[0,0,.01],0,q20),
              (np.array(poses['receiver_a'][:3])+[0,0,.085],q20,q20/2),
              (np.array(poses['receiver_a'][:3])+[0,0,.085],q20/2,q20/2),
              ([.7,1.3,1.1],q20/2,0),([.7,1.3,1.1],0,0)]
    for tip, start, end in phases:
        for model in (legacy, feedback): stroke(model, poses, tip, start, end)
        a, b = legacy.state(), feedback.state()
        for key in a.keys()-{'schema', 'limitations', 'native_water_weight_reaction_applied'}:
            assert a[key] == b[key], key
    assert legacy.state()['schema'] == SCHEMA
    assert 'water_difference_ml' not in legacy.state()
    assert 'body_liquid_volumes_ml' not in legacy.state()
    assert 'density_kg_m3' not in legacy.state()
    assert legacy.state()['native_water_weight_reaction_applied'] is False
    assert feedback.state()['receivers_ml']['receiver_a'] == pytest.approx(10.)
    assert feedback.state()['cumulative_spill_ml'] == pytest.approx(10.)


def test_v2_density_uses_explicit_key_and_old_fixture_default():
    _, p, _ = setup(); p=dict(p,schema=FEEDBACK_SCHEMA); p.pop('density_kg_m3',None)
    assert PistonVolumeModel(p).state()['density_kg_m3'] == 1000.
    assert PistonVolumeModel(dict(p,density_kg_m3=997.)).state()['density_kg_m3'] == 997.
    for invalid in (0., -1., float('nan'), float('inf'), True):
        with pytest.raises(ValueError): PistonVolumeModel(dict(p,density_kg_m3=invalid))


def test_authored_wet_picture_supports_match_pinned_room_collision_tops():
    from scenes.workrooms.laboratory.scene import environment_reference
    metadata, _, _ = setup(); room=Usd.Stage.Open(environment_reference()['usd_path'])
    cache=UsdGeom.BBoxCache(Usd.TimeCode.Default(),['default'])
    paths=('/World/Geometry/worktop/Collider','/World/Geometry/rear_lab_counter/Collider')
    for bounds,path in zip(metadata['packet_visual_bindings']['spill_surface_bounds_m'],paths,strict=True):
        actual=cache.ComputeWorldBound(room.GetPrimAtPath(path)).ComputeAlignedRange()
        low,high=np.asarray(actual.GetMin()),np.asarray(actual.GetMax())
        assert bounds[0] == pytest.approx([*low[:2],high[2]],abs=1.e-6)
        assert bounds[1] == pytest.approx(high,abs=1.e-6)


def test_water_difference_separates_flight_and_does_not_reduce_spill():
    _, p, poses = setup(); model = PistonVolumeModel(dict(p, schema=FEEDBACK_SCHEMA))
    q10 = 10e-6/PISTON_AREA_M2
    stroke(model, poses, np.array(poses['source'][:3])+[0,0,.01], 0, q10)
    model.step(.005, piston_q_m=q10-.0001, tool_pose_xyzw=[.7,1.3,1.1,0,0,0,1], container_poses=poses)
    state = model.state()
    assert state['water_difference_ml'] == pytest.approx(state['in_flight_ml'])
    assert state['current_spill_ml'] == pytest.approx(0.)
    assert sum(state['body_liquid_volumes_ml'].values())+state['in_flight_ml'] == pytest.approx(80.)
    stroke(model, poses, [.7,1.3,1.1], q10-.0001, 0)
    stroke(model, poses, [.7,1.3,1.1], 0, 0)
    state = model.state()
    assert state['in_flight_ml'] == 0.
    assert state['water_difference_ml'] == pytest.approx(10.)
    assert state['current_spill_ml'] == pytest.approx(state['cumulative_spill_ml'])
    saved = model.visual_state(); previous = state['cumulative_spill_ml']
    assert saved['spill_marks'] and all(x['position_m'][2] == .9 for x in saved['spill_marks'])
    assert len(saved['spill_marks']) <= 64
    saved['spill_marks'][0]['position_m'][0] = 123.
    assert model.visual_state()['spill_marks'][0]['position_m'][0] != 123.
    stroke(model, poses, [.7,1.3,1.1], 0, 0)
    assert model.state()['cumulative_spill_ml'] == previous
    assert model.visual_state()['packets'] == []


def test_empty_air_and_stopped_piston_never_create_visible_flow():
    _, p, poses = setup(); model = PistonVolumeModel(dict(p, schema=FEEDBACK_SCHEMA))
    stroke(model, poses, [.7,1.3,1.1], 0, .03)
    stroke(model, poses, [.7,1.3,1.1], .03, 0)
    assert model.visual_state()['packets'] == []
    assert model.visual_state()['spill_marks'] == []
    assert model.state()['tool_ml'] == 0.
    q10 = 10e-6/PISTON_AREA_M2
    stroke(model, poses, np.array(poses['source'][:3])+[0,0,.01], 0, q10)
    stroke(model, poses, [.7,1.3,1.1], q10, q10)
    assert model.state()['tool_ml'] == pytest.approx(10.)
    assert model.visual_state()['packets'] == []


def test_tilted_overflow_without_ballistic_impact_gets_no_invented_wet_mark():
    _, p, poses = setup(); model = PistonVolumeModel(dict(p, schema=FEEDBACK_SCHEMA))
    poses['source'] = [*poses['source'][:3], math.sin(math.pi/4),0,0,math.cos(math.pi/4)]
    stroke(model, poses, [.7,1.3,1.1], 0, 0, steps=1)
    assert model.state()['cumulative_spill_ml'] == 80.
    assert model.visual_state()['spill_marks'] == []
    assert model.visual_state()['unmarked_spill_ml'] == 80.


def test_private_capture_coordinates_do_not_leak_through_provider_observe():
    metadata, _, _ = setup(); metadata['task_parameters']['hydraulics']['schema'] = FEEDBACK_SCHEMA
    provider = ReducedLiquidDynamics({'asset_metadata': metadata})
    objects = {o['id']: {'position_m': o['initial_position_m'], 'quaternion_wxyz': [1,0,0,0],
        'linear_velocity_m_s': [0,0,0], 'angular_velocity_rad_s': [0,0,0]} for o in metadata['objects']}
    provider.on_frame({'objects': objects, 'native_step': 17, 'native_time_s': .0085,
                       'physics_dt_s': .0005, 'baseline': True})
    publicish = provider.on_observe(); private = provider.capture_visual_state()
    assert publicish['schema'] == FEEDBACK_SCHEMA
    assert set(publicish['body_liquid_volumes_ml']) == {'source','receiver_a','receiver_b','pipette'}
    assert 'packets' not in publicish and 'spill_marks' not in publicish
    assert private['native_step'] == publicish['native_step'] == 17
    assert private['native_time_s'] == publicish['native_time_s'] == .0085
    assert private['fluid_model'] == 'finite_volume_not_pbd'
    assert 'native_water_weight_reaction_applied' not in publicish
