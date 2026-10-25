import hashlib
from pathlib import Path

import numpy as np
import pytest

from bench.paths import DATA_ROOT, load_json
from bench.simulation.isaacsim_backend.native_task_bundle import (
    SCHEMA, NativeTaskReader, add_bundle, joint_positions, load_observer, readonly,
)


ASSETS = Path(f'{DATA_ROOT}/assets/task_parts/living_room')


def spec(task, version):
    from pxr import Usd, UsdPhysics
    root = ASSETS / task / version
    path = root / 'initial.usda'
    if not path.exists():
        pytest.skip('Shared candidate assets are unavailable')
    stage = Usd.Stage.Open(str(path))
    row = {'schema': SCHEMA, 'usd_path': str(path),
        'usd_sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'root_prim_path': '/World/LivingTask',
        'rigid_bodies': {p.GetName().lower(): str(p.GetPath()) for p in stage.Traverse()
                         if p.HasAPI(UsdPhysics.RigidBodyAPI)},
        'joints': {p.GetName().lower(): str(p.GetPath()) for p in stage.Traverse() if p.IsA(UsdPhysics.Joint)},
        'particle_sets': {}}
    if task == 'water_pouring':
        water = load_json(root / 'metadata.json')['water']
        row['particle_sets']['water'] = {'prim_path': water['particles_path'], 'system_path': water['system_path'],
            'particle_count': water['particle_count'], 'total_mass_kg': water['total_mass_kg'], 'position_attribute': 'points'}
    return row


@pytest.mark.parametrize('task,version,count', [('remote_operation', 'v5', 4), ('water_pouring', 'v1', 2)])
def test_real_frozen_assemblies_keep_inventory_geometry_and_source(task, version, count):
    from pxr import Usd, UsdGeom, UsdPhysics
    row = spec(task, version)
    source = Path(row['usd_path']).read_bytes()
    stage = Usd.Stage.CreateInMemory()
    UsdPhysics.Scene.Define(stage, '/physicsScene')
    result = add_bundle(stage, row, physics_scene_path='/physicsScene')
    assert len(result['objects']) == count
    assert set(result['joints']) == set(row['joints'])
    assert Path(row['usd_path']).read_bytes() == source
    original = Usd.Stage.Open(row['usd_path'])
    for p in original.Traverse():
        if p.IsA(UsdGeom.Mesh):
            assert p.GetAttribute('points').Get() == stage.GetPrimAtPath(p.GetPath()).GetAttribute('points').Get()
    if task == 'water_pouring':
        assert [str(p) for p in stage.GetPrimAtPath(row['particle_sets']['water']['system_path']).GetRelationship('simulationOwner').GetTargets()] == ['/physicsScene']


def test_missing_inventory_and_changed_source_fail():
    from pxr import Usd
    row = spec('remote_operation', 'v5')
    row['rigid_bodies'].pop('powerbutton')
    with pytest.raises(ValueError, match='inventory'):
        add_bundle(Usd.Stage.CreateInMemory(), row, physics_scene_path='/physicsScene')
    row['usd_sha256'] = '0' * 64
    with pytest.raises(ValueError, match='hash'):
        add_bundle(Usd.Stage.CreateInMemory(), row, physics_scene_path='/physicsScene')


def test_absent_bundle_is_an_exact_noop():
    from pxr import Usd
    stage = Usd.Stage.CreateInMemory()
    before = stage.GetRootLayer().ExportToString()
    assert add_bundle(stage, None, physics_scene_path='/physicsScene') is None
    assert stage.GetRootLayer().ExportToString() == before


def test_joint_readback_uses_body_orientations_and_anchors():
    # A 180-degree Y rotation reverses world Z while local press stays negative.
    objects = {'remote': {'position_m': [3, 2, 1], 'quaternion_wxyz': [0, 0, 1, 0]},
        'button': {'position_m': [3, 2, .9915], 'quaternion_wxyz': [0, 0, 1, 0]}}
    rows = {'power': {'body0': 'remote', 'body1': 'button', 'anchor0_m': [0, 0, .01],
                      'anchor1_m': [0, 0, 0], 'axis0': [0, 0, 1]}}
    assert joint_positions(objects, rows)['power']['position_m'] == pytest.approx(-.0015)


def test_observer_imports_frozen_helpers_and_rejects_drift(tmp_path):
    main, helper = tmp_path / 'observer.py', tmp_path / 'rules.py'
    helper.write_text('VALUE = 3\n')
    main.write_text('from .rules import VALUE\nclass NativeTaskObserver:\n'
        ' def __init__(self, configuration): self.configuration = configuration\n'
        ' def on_frame(self, frame): self.frame = frame\n'
        ' def on_observe(self): return {"value": VALUE}\n')
    files = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (main, helper)}
    row = {'path': str(main), 'sha256': files[main.name], 'files': files, 'configuration': {}}
    assert load_observer(row).on_observe() == {'value': 3}
    helper.write_text('VALUE = 4\n')
    with pytest.raises(ValueError, match='hash'):
        load_observer(row)


def test_readonly_frame_and_repeated_observation_cache():
    frame = readonly({'points': np.ones((2, 3)), 'contacts': [{'value': 1}]})
    with pytest.raises(ValueError):
        frame['points'][0, 0] = 0
    with pytest.raises(TypeError):
        frame['contacts'][0]['value'] = 2
    class Observer:
        calls = 0
        def on_observe(self):
            self.calls += 1
            return {'count': self.calls}
    reader = NativeTaskReader.__new__(NativeTaskReader)
    reader.observer = Observer()
    reader._last_observed_step = None
    reader._step = 3
    assert reader.observe() == reader.observe() == {'count': 1}
    reader._step = 4
    assert reader.observe() == {'count': 2}


def test_bundle_reset_restores_pinned_particles_and_rigid_transforms():
    from pxr import Usd, UsdGeom, UsdPhysics, Vt
    from bench.simulation.isaacsim_backend.native_task_bundle import restore_bundle_before_world_reset
    row = spec('water_pouring', 'v1')
    stage = Usd.Stage.CreateInMemory()
    UsdPhysics.Scene.Define(stage, '/physicsScene')
    info = add_bundle(stage, row, physics_scene_path='/physicsScene')
    source = Usd.Stage.Open(row['usd_path'])
    body = stage.GetPrimAtPath(next(iter(info['objects'].values()))['path'])
    xform = UsdGeom.Xformable(body)
    expected = xform.GetLocalTransformation()
    xform.ClearXformOpOrder()
    xform.AddTranslateOp(opSuffix='reset_test').Set((4., 5., 6.))
    particle = stage.GetPrimAtPath(row['particle_sets']['water']['prim_path'])
    original = source.GetPrimAtPath(particle.GetPath())
    particle.GetAttribute('points').Set(Vt.Vec3fArray([(3., 2., 1.)]))
    particle.GetAttribute('ids').Set(Vt.Int64Array([999]))
    restore_bundle_before_world_reset(stage, info)
    assert UsdGeom.Xformable(body).GetLocalTransformation() == expected
    for name in ('points', 'velocities', 'ids'):
        assert particle.GetAttribute(name).Get() == original.GetAttribute(name).Get()


def test_bundle_particle_capture_preserves_original_ids():
    reader = NativeTaskReader.__new__(NativeTaskReader)
    reader._particles = {'water': {'positions_m': np.zeros((2, 3)), 'velocities_m_s': np.ones((2, 3)),
                                  'ids': np.array([8, 92], dtype=np.int64)}}
    state = reader.capture_particles()
    assert np.array_equal(state['task_particle_water_ids'], [8, 92])
    state['task_particle_water_ids'][0] = 0
    assert reader._particles['water']['ids'][0] == 8


def reduced_reader_spec(tmp_path, *, invalid_output=False):
    observer_dir=tmp_path/'observer';observer_dir.mkdir()
    model_dir=tmp_path/'dynamics';model_dir.mkdir()
    observer=observer_dir/'observer.py'
    observer.write_text('class NativeTaskObserver:\n'
        ' def __init__(self, configuration): self.frames = 0\n'
        ' def on_frame(self, frame):\n'
        '  self.frames += 1\n'
        '  self.state = {"volume": frame["reduced_dynamics"]["volume"], "step": frame["native_step"]}\n'
        '  try: frame["reduced_dynamics"]["volume"] = 0\n'
        '  except TypeError: self.state["readonly"] = True\n'
        ' def on_observe(self): return self.state\n')
    helper=model_dir/'dimensions.py';helper.write_text('AREA = 2.0\n')
    model=model_dir/'dynamics.py'
    model.write_text('from .dimensions import AREA\nclass ReducedModel:\n'
        ' def __init__(self, configuration): self.volume = configuration["initial"]; self.steps = []\n'
        ' def on_frame(self, frame):\n'
        '  assert "reduced_dynamics" not in frame\n'
        '  self.steps.append(frame["native_step"])\n'
        '  if not frame["baseline"]: self.volume += AREA * frame["objects"]["piston"]["position_m"][2]\n'
        '  try: frame["objects"]["piston"]["position_m"][2] = 99\n'
        '  except TypeError: pass\n'
        '  else: raise RuntimeError("Native frame was writable")\n'
        ' def on_observe(self): return {"volume": '+('float("nan")' if invalid_output else 'self.volume')+'}\n')
    def package(entrypoint):
        files={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in entrypoint.parent.glob('*.py')}
        return {'path':str(entrypoint),'sha256':files[entrypoint.name],'files':files,'configuration':{}}
    result={'observer':package(observer),'reduced_dynamics':package(model)}
    result['reduced_dynamics'].update(class_name='ReducedModel',configuration={'initial':5.})
    return result


def sample_reduced_reader(reader, step, *, z=.25):
    obj={'piston':{'position_m':[0.,0.,z],'quaternion_wxyz':[1.,0.,0.,0.]}}
    reader.sample(objects=obj,contacts=[],time_s=step*.001,native_time_s=step*.001,
        native_step=step,physics_dt_s=.001,baseline=step==0)
    assert obj['piston']['position_m']==[0.,0.,z]


def test_reduced_model_precedes_readonly_referee_and_reset_has_fresh_state(tmp_path):
    from pxr import Usd
    spec=reduced_reader_spec(tmp_path)
    info={'particle_sets':{},'joints':{}}
    reader=NativeTaskReader(Usd.Stage.CreateInMemory(),info,spec)
    sample_reduced_reader(reader,0);assert reader.observe()['volume']==5.
    sample_reduced_reader(reader,1)
    assert reader.observe()=={'volume':5.5,'step':1,'readonly':True}
    assert reader.reduced_dynamics.steps==[0,1]
    assert reader.observe()==reader.observe()  # observing does not advance physics
    assert reader.reduced_dynamics.steps==[0,1]
    reset=NativeTaskReader(Usd.Stage.CreateInMemory(),info,spec)
    sample_reduced_reader(reset,0)
    assert reset.observe()['volume']==5.
    assert reset.reduced_dynamics is not reader.reduced_dynamics


def test_reduced_model_rejects_nonfinite_output_before_referee(tmp_path):
    from pxr import Usd
    reader=NativeTaskReader(Usd.Stage.CreateInMemory(),{'particle_sets':{},'joints':{}},
        reduced_reader_spec(tmp_path,invalid_output=True))
    with pytest.raises(ValueError,match='JSON compliant'):
        sample_reduced_reader(reader,0)
    assert reader.observer.frames==0


def test_reduced_model_pins_every_python_dependency(tmp_path):
    from pxr import Usd
    spec=reduced_reader_spec(tmp_path)
    (tmp_path/'dynamics'/'dimensions.py').write_text('AREA = 10.0\n')
    with pytest.raises(ValueError,match='hash'):
        NativeTaskReader(Usd.Stage.CreateInMemory(),{'particle_sets':{},'joints':{}},spec)
