"""CPU USD checks for observer-driven visuals, separate from RTX qualification."""
from copy import deepcopy
import hashlib
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
from pxr import Usd, UsdGeom, UsdPhysics, UsdShade

from bench.simulation.isaacsim_backend.native_task_bundle import NativeTaskReader
from bench.simulation.isaacsim_backend.task_visual_state import (
    BooleanMaterialBindings, ROOT_PATH, SCHEMA,
)


def configuration(target='/World/Screen'):
    return {'schema': SCHEMA, 'bindings': [{
        'id': 'display_power', 'target_prim_path': target,
        'state_path': ['display', 'power_on'],
        'false_material': {'diffuse_color_rgb': [.008, .012, .015],
            'emissive_color_rgb': [0, 0, 0], 'roughness': .11, 'metallic': .12},
        'true_material': {'diffuse_color_rgb': [.12, .35, .65],
            'emissive_color_rgb': [.15, .65, 1.5], 'roughness': .3, 'metallic': 0.}}]}


def screen_stage():
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.Cube.Define(stage, '/World/Screen')
    return stage


def material_path(stage, target='/World/Screen'):
    return str(UsdShade.MaterialBindingAPI(stage.GetPrimAtPath(target)).ComputeBoundMaterial()[0].GetPath())


def test_unconfigured_visuals_preserve_stage_exactly():
    stage = screen_stage()
    before = stage.GetRootLayer().ExportToString()
    bindings = BooleanMaterialBindings(stage, None)
    assert bindings.sync({}, native_step=0) is None
    assert stage.GetRootLayer().ExportToString() == before


def test_boolean_state_changes_material_and_repeated_cameras_do_not_write():
    stage = screen_stage()
    bindings = BooleanMaterialBindings(stage, configuration())
    off = bindings.sync({'display': {'power_on': False}}, native_step=3)
    assert off['source_native_step'] == 3
    assert off['bindings'][0]['value'] is False
    assert material_path(stage) == ROOT_PATH + '/display_power/false'
    on = bindings.sync({'display': {'power_on': True}}, native_step=4)
    assert on['bindings'][0]['binding_changed'] is True
    assert material_path(stage) == ROOT_PATH + '/display_power/true'
    shader = UsdShade.Shader.Get(stage, material_path(stage) + '/Surface')
    assert list(shader.GetInput('emissiveColor').Get()) == pytest.approx([.15, .65, 1.5])
    before = stage.GetRootLayer().ExportToString()
    repeated = bindings.sync({'display': {'power_on': True}}, native_step=4)
    assert repeated['bindings'][0]['binding_changed'] is False
    assert stage.GetRootLayer().ExportToString() == before
    bindings.sync({'display': {'power_on': False}}, native_step=5)
    assert material_path(stage).endswith('/false')


@pytest.mark.parametrize('state', [{}, {'display': {}}, {'display': {'power_on': 1}},
                                  {'display': {'power_on': 'true'}}, {'display': {'power_on': None}}])
def test_missing_or_nonboolean_observer_state_fails_before_binding_changes(state):
    stage = screen_stage()
    bindings = BooleanMaterialBindings(stage, configuration())
    bindings.sync({'display': {'power_on': False}}, native_step=0)
    before = stage.GetRootLayer().ExportToString()
    with pytest.raises(ValueError, match='Visual'):
        bindings.sync(state, native_step=1)
    assert stage.GetRootLayer().ExportToString() == before


@pytest.mark.parametrize('change', ['missing', 'collider', 'transform', 'bad_rgb', 'nonfinite',
                                    'arbitrary_field', 'duplicate', 'bad_state_path'])
def test_invalid_configuration_never_authors_materials(change):
    stage = screen_stage()
    config = configuration()
    row = config['bindings'][0]
    if change == 'missing':
        row['target_prim_path'] = '/World/Missing'
    elif change == 'collider':
        UsdPhysics.CollisionAPI.Apply(stage.GetPrimAtPath('/World/Screen'))
    elif change == 'transform':
        UsdGeom.Xform.Define(stage, '/World/Transform')
        row['target_prim_path'] = '/World/Transform'
    elif change == 'bad_rgb':
        row['true_material']['diffuse_color_rgb'] = [2, 0, 0]
    elif change == 'nonfinite':
        row['true_material']['roughness'] = float('nan')
    elif change == 'arbitrary_field':
        row['true_material']['physics:mass'] = 2
    elif change == 'duplicate':
        config['bindings'].append(deepcopy(row))
    elif change == 'bad_state_path':
        row['state_path'] = 'display.power_on'
    before = stage.GetRootLayer().ExportToString()
    with pytest.raises(ValueError):
        BooleanMaterialBindings(stage, config)
    assert stage.GetRootLayer().ExportToString() == before


def test_multiple_bindings_resolve_all_states_before_any_switch():
    stage = screen_stage()
    UsdGeom.Cube.Define(stage, '/World/SecondScreen')
    config = configuration()
    second = deepcopy(config['bindings'][0])
    second.update(id='second_display', target_prim_path='/World/SecondScreen', state_path=['second'])
    config['bindings'].append(second)
    bindings = BooleanMaterialBindings(stage, config)
    bindings.sync({'display': {'power_on': False}, 'second': False}, native_step=0)
    before = stage.GetRootLayer().ExportToString()
    with pytest.raises(ValueError):
        bindings.sync({'display': {'power_on': True}}, native_step=1)
    assert stage.GetRootLayer().ExportToString() == before


def test_pinned_room_screen_switch_preserves_all_original_attributes_and_source():
    from scenes.workrooms.living_room.scene import ENVIRONMENT_ROOT, ENVIRONMENT_USD_SHA256
    source = Path(ENVIRONMENT_ROOT) / 'scene.usda'
    if not source.is_file():
        pytest.skip('Pinned living room is unavailable')
    original_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    assert original_hash == ENVIRONMENT_USD_SHA256
    stage = Usd.Stage.CreateInMemory()
    stage.DefinePrim('/World/Room', 'Xform').GetReferences().AddReference(str(source), '/World')
    target = '/World/Room/Geometry/television_screen/Visual'
    original = {str(prim.GetPath()): {attr.GetName(): attr.Get() for attr in prim.GetAttributes()}
                for prim in stage.Traverse()}
    bindings = BooleanMaterialBindings(stage, configuration(target))
    for step, power in enumerate((False, True, False)):
        bindings.sync({'display': {'power_on': power}}, native_step=step)
        assert material_path(stage, target).endswith('/true' if power else '/false')
    for path, attributes in original.items():
        prim = stage.GetPrimAtPath(path)
        assert {attr.GetName(): attr.Get() for attr in prim.GetAttributes()} == attributes
    assert hashlib.sha256(source.read_bytes()).hexdigest() == original_hash


def test_reader_uses_cached_native_tick_and_reset_restores_off(tmp_path):
    entrypoint = tmp_path / 'observer.py'
    entrypoint.write_text('class NativeTaskObserver:\n'
        ' def __init__(self, configuration): self.calls = 0\n'
        ' def on_frame(self, frame): self.step = frame["native_step"]\n'
        ' def on_observe(self):\n'
        '  self.calls += 1\n'
        '  return {"display": {"power_on": self.step == 1}}\n')
    sha = hashlib.sha256(entrypoint.read_bytes()).hexdigest()
    spec = {'observer': {'path': str(entrypoint), 'sha256': sha,
        'files': {'observer.py': sha}, 'configuration': {}}, 'visual_state_bindings': configuration()}
    stage = screen_stage()
    reader = NativeTaskReader(stage, {'particle_sets': {}, 'joints': {}}, spec)
    def sample(target, step):
        target.sample(objects={}, contacts=[], time_s=step * .001, native_time_s=step * .001,
                      native_step=step, physics_dt_s=.001, baseline=step == 0)
    sample(reader, 0)
    reader.observe()
    reader.sync_visual_state()
    assert reader.observer.calls == 1
    assert material_path(stage).endswith('/false')
    sample(reader, 1)
    assert reader.sync_visual_state()['source_native_step'] == 1
    reader.sync_visual_state()
    assert reader.observer.calls == 2
    assert material_path(stage).endswith('/true')
    reset_reader = NativeTaskReader(stage, {'particle_sets': {}, 'joints': {}}, spec)
    sample(reset_reader, 0)
    assert reset_reader.sync_visual_state()['bindings'][0]['value'] is False
    assert material_path(stage).endswith('/false')


def test_native_render_entrypoint_syncs_material_before_every_render(monkeypatch):
    from bench.simulation.isaacsim_backend.env import FR3DexterousIsaacEnv
    from bench.simulation.isaacsim_backend import render_sync
    # Keep the real rendering entrypoint and USD material mapping, replacing
    # only the unavailable CPU-side Kit/RTX dependencies and image annotators.
    for name in ('carb', 'omni', 'omni.replicator', 'omni.replicator.core'):
        monkeypatch.setitem(sys.modules, name, ModuleType(name))
    sys.modules['carb'].settings = SimpleNamespace(get_settings=lambda: {})
    monkeypatch.setattr(render_sync, 'verify_settings', lambda settings: {})
    monkeypatch.setattr(render_sync, 'camera_calibration', lambda *args, **kwargs: {})
    stage = screen_stage()
    bindings = BooleanMaterialBindings(stage, configuration())
    power = False
    rendered = []
    env = FR3DexterousIsaacEnv.__new__(FR3DexterousIsaacEnv)
    env.physics_device = 'cuda:0'
    env.cameras = {'display': {'resolution': (2, 2)}}
    env.render_width = env.render_height = 2
    env.time_s = 0.
    env._render_products = {('display', 2, 2): (object(), {
        'rgb': SimpleNamespace(get_data=lambda: np.zeros((2, 2, 4), dtype=np.uint8)),
        'camera_params': SimpleNamespace(get_data=lambda: {}),
    })}
    env._task_reader = SimpleNamespace(sync_visual_state=lambda:
        bindings.sync({'display': {'power_on': power}}, native_step=0))
    env.world = SimpleNamespace(current_time_step_index=0, current_time=0.,
        render=lambda: rendered.append(material_path(stage)))
    for power in (False, True, False):
        frame = env._render_frames(('display',), 2, 2, include_depth=False)['display']
        expected = ROOT_PATH + '/display_power/' + ('true' if power else 'false')
        assert rendered[-2:] == [expected, expected]
        assert frame['task_visual_state']['bindings'][0]['value'] is power
        assert frame['rgb'].shape == (2, 2, 3)
    assert env.world.current_time_step_index == 0
    assert env.world.current_time == 0.
