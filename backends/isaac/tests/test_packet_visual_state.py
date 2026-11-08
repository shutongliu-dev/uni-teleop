from copy import deepcopy
import math

import pytest
from pxr import Usd, UsdGeom, UsdPhysics, UsdShade

from bench.simulation.isaacsim_backend.packet_visual_state import PacketVisualBindings, SCHEMA, STATE_SCHEMA


def scene():
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.Xform.Define(stage, '/World/LiquidPictures')
    UsdShade.Material.Define(stage, '/World/Water')
    body = UsdGeom.Xform.Define(stage, '/World/Cup')
    UsdPhysics.RigidBodyAPI.Apply(body.GetPrim())
    UsdPhysics.MassAPI.Apply(body.GetPrim()).CreateMassAttr(.1)
    collider = UsdGeom.Cube.Define(stage, '/World/Cup/Collider')
    UsdPhysics.CollisionAPI.Apply(collider.GetPrim())
    config = {'schema': SCHEMA, 'root_path': '/World/LiquidPictures', 'material_path': '/World/Water',
              'max_packets': 32, 'max_spill_marks': 8, 'spill_film_thickness_m': .00015,
              'spill_surface_bounds_m': [[[0.,0.,.9],[1.,.5,.9]], [[0.,1.,.9],[1.,1.5,.9]]],
              'maximum_spill_radius_m': .04}
    return stage, config


def state(step=1):
    return {'schema': STATE_SCHEMA, 'fluid_model': 'finite_volume_not_pbd',
            'native_step': step, 'native_time_s': step*.0005,
            'packets': [{'id': 4, 'volume_ml': .005, 'position_m': [.7, .2, 1.1],
                         'velocity_m_s': [0., 0., -.3]}],
            'spill_marks': [{'id': 0, 'volume_ml': .01, 'position_m': [.8, .2, .9]}],
            'in_flight_ml': .005, 'marked_spill_ml': .01,
            'unmarked_spill_ml': .02, 'cumulative_spill_ml': .03}


def test_actual_packet_radius_and_impact_location_are_session_only():
    stage, config = scene(); before = stage.GetRootLayer().ExportToString()
    bindings = PacketVisualBindings(stage, config)
    report = bindings.sync(state(), native_step=1)
    packet = UsdGeom.PointInstancer(stage.GetPrimAtPath('/World/LiquidPictures/Packets'))
    wet = UsdGeom.PointInstancer(stage.GetPrimAtPath('/World/LiquidPictures/WetMarks'))
    assert list(packet.GetIdsAttr().Get()) == [4]
    assert list(packet.GetPositionsAttr().Get()[0]) == pytest.approx([.7, .2, 1.1])
    radius = (3*.005e-6/(4*math.pi))**(1/3)
    assert list(packet.GetScalesAttr().Get()[0]) == pytest.approx([radius]*3)
    assert list(wet.GetPositionsAttr().Get()[0]) == pytest.approx([.8, .2, .900125])
    assert stage.GetRootLayer().ExportToString() == before
    assert report['physics_state_writeback'] is False
    for prim in Usd.PrimRange(stage.GetPrimAtPath('/World/LiquidPictures')):
        assert not prim.HasAPI(UsdPhysics.CollisionAPI)
        assert not prim.HasAPI(UsdPhysics.RigidBodyAPI)
    session = stage.GetSessionLayer().ExportToString()
    assert bindings.sync(state(), native_step=1)['changed'] is False
    assert stage.GetSessionLayer().ExportToString() == session


def test_fresh_reset_removes_prior_packets_and_marks_without_inventing_flow():
    stage, config = scene(); PacketVisualBindings(stage, config).sync(state(), native_step=1)
    fresh = PacketVisualBindings(stage, config)
    empty = state(0)
    empty.update(packets=[], spill_marks=[], in_flight_ml=0., marked_spill_ml=0.,
                 unmarked_spill_ml=0., cumulative_spill_ml=0.)
    report = fresh.sync(empty, native_step=0)
    assert report['visible_packet_count'] == report['visible_spill_mark_count'] == 0
    for name in ('Packets', 'WetMarks'):
        prim = stage.GetPrimAtPath('/World/LiquidPictures/'+name)
        assert UsdGeom.Imageable(prim).ComputeVisibility() == 'invisible'
        assert len(UsdGeom.PointInstancer(prim).GetPositionsAttr().Get()) == 0


@pytest.mark.parametrize('fault', ['nan', 'duplicate', 'clock', 'inventory', 'negative'])
def test_bad_or_stale_model_state_cannot_partially_write_pictures(fault):
    stage, config = scene(); bindings = PacketVisualBindings(stage, config)
    bad = state()
    if fault == 'nan': bad['packets'][0]['position_m'][0] = float('nan')
    elif fault == 'duplicate': bad['packets'].append(deepcopy(bad['packets'][0]))
    elif fault == 'clock': bad['native_step'] = 2
    elif fault == 'inventory': bad['in_flight_ml'] = 1.
    else: bad['spill_marks'][0]['volume_ml'] = -1.
    before = stage.GetSessionLayer().ExportToString()
    with pytest.raises(ValueError): bindings.sync(bad, native_step=1)
    assert stage.GetSessionLayer().ExportToString() == before


def test_destination_rejects_physics_and_unowned_children():
    stage, config = scene()
    UsdPhysics.RigidBodyAPI.Apply(stage.GetPrimAtPath(config['root_path']))
    with pytest.raises(ValueError, match='physical ancestor'): PacketVisualBindings(stage, config)
    stage, config = scene(); UsdGeom.Sphere.Define(stage, config['root_path']+'/Foreign')
    with pytest.raises(ValueError, match='empty or owned'): PacketVisualBindings(stage, config)
    stage, config = scene(); bindings = PacketVisualBindings(stage, config)
    bindings.sync(state(), native_step=1)
    with Usd.EditContext(stage, stage.GetSessionLayer()):
        UsdPhysics.CollisionAPI.Apply(stage.GetPrimAtPath(config['root_path']+'/Packets/Prototype'))
    with pytest.raises(ValueError, match='physical geometry'): bindings.sync(state(), native_step=1)


def test_capacity_bounds_only_picture_and_does_not_inflate_packets():
    stage, config = scene(); config['max_packets'] = 2
    sample = state(); sample['packets'] = [dict(sample['packets'][0], id=i) for i in range(5)]
    sample['in_flight_ml'] = .025; original = deepcopy(sample)
    result = PacketVisualBindings(stage, config).sync(sample, native_step=1)
    assert result['packet_count'] == 5 and result['visible_packet_count'] == 2
    assert result['omitted_packet_volume_ml'] == pytest.approx(.015)
    assert sample == original


def test_wet_marks_do_not_float_over_aisle_or_outside_table_and_stop_at_edge():
    stage, config = scene(); sample = state()
    sample['spill_marks'] = [dict(sample['spill_marks'][0], id=i, position_m=point)
        for i, point in enumerate(([.8,.2,.9],[.8,.75,.9],[1.1,.2,.9],[.999,.2,.9],[.8,.2,.8]))]
    sample.update(marked_spill_ml=.05, unmarked_spill_ml=.02, cumulative_spill_ml=.07)
    original = deepcopy(sample)
    report = PacketVisualBindings(stage, config).sync(sample, native_step=1)
    wet = UsdGeom.PointInstancer(stage.GetPrimAtPath(config['root_path']+'/WetMarks'))
    assert list(wet.GetIdsAttr().Get()) == [0,3]
    assert wet.GetScalesAttr().Get()[1][0] == pytest.approx(.001)
    assert report['undrawn_spill_mark_volume_ml'] == pytest.approx(.03)
    assert sample == original  # Cropping changes no flight/spill accounting.


def test_private_model_coordinates_never_enter_observer_or_public_ledger(monkeypatch):
    from bench.simulation.isaacsim_backend import native_task_bundle as bundle
    stage, config = scene()
    class Observer:
        def on_frame(self, frame): self.frame = frame
        def on_observe(self): return {'volume_ml': self.frame['reduced_dynamics']['in_flight_ml']}
    class Model:
        def on_frame(self, frame): self.frame = frame
        def on_observe(self): return {'in_flight_ml': .005}
        def capture_visual_state(self): return state(self.frame['native_step'])
    observer, model = Observer(), Model()
    monkeypatch.setattr(bundle, 'load_observer', lambda spec: observer if spec == 'observer' else model)
    reader = bundle.NativeTaskReader(stage, {'particle_sets': {}, 'joints': {}},
        {'observer': 'observer', 'reduced_dynamics': 'model', 'packet_visual_bindings': config})
    reader.sample(objects={}, contacts=[], time_s=.0005, native_time_s=.0005,
                  native_step=1, physics_dt_s=.0005)
    assert reader.observe() == {'volume_ml': .005}
    assert set(observer.frame['reduced_dynamics']) == {'in_flight_ml'}
    assert 'packets' not in observer.frame
    assert reader._reduced_visual_state['packets'][0]['position_m'] == [.7, .2, 1.1]
    receipt = reader.sync_visual_state()
    assert receipt['packet_visuals']['visible_packet_count'] == 1
    assert 'position_m' not in repr(receipt)


def test_refraction_budget_is_opt_in_for_real_non_pbd_bindings():
    from bench.simulation.isaacsim_backend.particle_surfaces import ParticleSurfaceRenderer
    class Settings(dict):
        def set(self, key, value): self[key] = value
    stage, config = scene(); renderer = ParticleSurfaceRenderer(stage); settings = Settings()
    renderer.configure_refraction(settings)
    assert settings == {} and renderer.refraction_settings == {}
    settings['/rtx/pathtracing/maxSpecularAndTransmissionBounces'] = 20
    settings['/rtx/pathtracing/maxBounces'] = 16
    settings['/rtx/rtpt/maxBounces'] = 3
    renderer.configure_refraction(settings, packet_visual_bindings=config)
    assert settings['/rtx/translucency/maxRefractionBounces'] == 12
    assert settings['/rtx/rtpt/maxSpecularAndTransmissionBounces'] == 12
    assert settings['/rtx/pathtracing/maxSpecularAndTransmissionBounces'] == 20
    assert settings['/rtx/pathtracing/maxBounces'] == 16
    assert settings['/rtx/rtpt/maxBounces'] == 12
    assert len(renderer.groups) == 0  # Declared finite volumes never become PBD.


def test_volume_only_scenes_keep_their_original_total_bounce_budget():
    from bench.simulation.isaacsim_backend.particle_surfaces import ParticleSurfaceRenderer
    from scenes.workrooms.laboratory.tasks.pipette_transfer.assets import author_task
    class Settings(dict):
        def set(self,key,value): self[key]=value
    stage=Usd.Stage.CreateInMemory();metadata=author_task(stage)
    settings=Settings({'/rtx/rtpt/maxBounces':3,'/rtx/pathtracing/maxBounces':4})
    ParticleSurfaceRenderer(stage).configure_refraction(settings,
        volume_visual_bindings=metadata['volume_visual_bindings'])
    assert settings['/rtx/rtpt/maxBounces'] == 3
    assert settings['/rtx/pathtracing/maxBounces'] == 4
    assert settings['/rtx/pathtracing/maxSpecularAndTransmissionBounces'] == 12
