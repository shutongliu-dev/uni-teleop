"""CPU rejection checks for the display-only native RTX acceptance harness."""
from copy import deepcopy
from itertools import product
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.validation.living_television_visual import (
    actual_cuda_identity, assess_display, depth_receipt, display_fixture, same_physics, screen_roi,
    validate_fixture_receipt,
)


def calibration():
    # Independent pinhole camera: fx=fy=320, optical axis -Z, 320x240 image.
    return {'resolution': [320, 240], 'meters_per_scene_unit': 1.,
        'matrix_convention': 'row_vector_world_to_view_then_view_to_clip',
        'world_to_view': np.eye(4).tolist(),
        'projection': [[2., 0, 0, 0], [0, 8/3, 0, 0],
                       [0, 0, 0, -1], [0, 0, .01, 0]]}


def screen():
    return np.array(list(product((-.4, .4), (-.225, .225), (-2.01, -2.))))


def test_fixture_only_replaces_render_hook_and_restores_it_on_failure():
    canonical = {'television': {'power_on': False}}
    reader = SimpleNamespace(sync_visual_state=lambda: canonical, observe=lambda: canonical)
    original = reader.sync_visual_state
    bindings = SimpleNamespace(sync=lambda value, native_step: {'state': value, 'step': native_step})
    fixture = {'television': {'power_on': True}}
    with pytest.raises(RuntimeError, match='capture failed'):
        with display_fixture(reader, bindings, fixture, 17) as receipt:
            assert reader.sync_visual_state() == receipt == {'state': fixture, 'step': 17}
            assert reader.observe() is canonical
            raise RuntimeError('capture failed')
    assert reader.sync_visual_state is original
    assert reader.sync_visual_state() is canonical


def test_real_material_receipts_preserve_fixture_across_repeated_render_syncs():
    from pxr import Usd, UsdGeom
    from bench.simulation.isaacsim_backend.task_visual_state import BooleanMaterialBindings
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.Cube.Define(stage, '/World/Screen')
    material = {'diffuse_color_rgb': [.1, .2, .3], 'emissive_color_rgb': [0., 0., 0.],
                'roughness': .3, 'metallic': 0.}
    bindings = BooleanMaterialBindings(stage, {'schema': 'boolean-material-bindings-v1',
        'bindings': [{'id': 'power', 'target_prim_path': '/World/Screen',
            'state_path': ['television', 'power_on'], 'false_material': material,
            'true_material': {**material, 'emissive_color_rgb': [1., 1., 1.]}}]})
    canonical = {'television': {'power_on': False}}
    reader = SimpleNamespace(sync_visual_state=lambda: None, observe=lambda: canonical)
    for power_on in (False, True, False):
        with display_fixture(reader, bindings, {'television': {'power_on': power_on}}, 17) as first:
            assert first['bindings'][0]['binding_changed'] is True
            for _ in range(2):
                actual = reader.sync_visual_state()
                assert actual['bindings'][0]['binding_changed'] is False
                validate_fixture_receipt(actual, first)
                assert reader.observe() is canonical
            for field, value in [('source_native_step', 18), ('bindings', [])]:
                with pytest.raises(ValueError, match='differs'):
                    validate_fixture_receipt({**actual, field: value}, first)
            wrong = deepcopy(actual)
            wrong['bindings'][0]['value'] = not power_on
            with pytest.raises(ValueError, match='differs'):
                validate_fixture_receipt(wrong, first)


def test_roi_tracks_measured_camera_translation_and_rejects_wrong_view():
    # Face edges project to x=96/224, y=84/156; 15% inset avoids the bezel.
    assert screen_roi(screen(), calibration(), 320, 240) == [116, 95, 204, 145]
    moved = screen()+[2., 0., 0.]
    with pytest.raises(ValueError, match='fully inside'):
        screen_roi(moved, calibration(), 320, 240)
    measured = deepcopy(calibration())
    measured['world_to_view'][3][0] = -2.
    assert screen_roi(moved, measured, 320, 240) == [116, 95, 204, 145]


def test_roi_rejects_hidden_small_and_mismatched_captures():
    with pytest.raises(ValueError, match='behind'):
        screen_roi(screen()+[0, 0, 4.], calibration(), 320, 240)
    with pytest.raises(ValueError, match='too few pixels'):
        screen_roi(screen()+[0, 0, -8.], calibration(), 320, 240)
    with pytest.raises(ValueError, match='resolution'):
        screen_roi(screen(), calibration(), 640, 480)


def test_wrong_surface_depth_and_occlusion_are_rejected():
    roi = [116, 95, 204, 145]
    depth = np.full((240, 320), 2., dtype=np.float32)
    assert depth_receipt(depth, screen(), calibration(), roi)['passed']
    assert not depth_receipt(depth-.1, screen(), calibration(), roi)['passed']
    assert not depth_receipt(depth+.1, screen(), calibration(), roi)['passed']
    # A partially occluding foreground object must fail even with valid depth
    # everywhere and the correct median depth over the entire image.
    depth[95:145, 116:150] = .7
    assert not depth_receipt(depth, screen(), calibration(), roi)['passed']
    depth[95:145, 116:204] = 0
    assert not depth_receipt(depth, screen(), calibration(), roi)['passed']


def test_display_requires_screen_change_and_return_to_off():
    off = np.full((240, 320, 3), 12, dtype=np.uint8)
    on = off.copy()
    on[95:145, 116:204] = [70, 110, 160]
    returned = off+1
    roi = [116, 95, 204, 145]
    assert assess_display(off, on, returned, roi)['passed']
    assert not assess_display(off, off, off, roi)['passed']  # stale image
    assert not assess_display(off, on, on, roi)['passed']  # failed off transition
    wrong_surface = np.full_like(off, 180)
    wrong_surface[95:145, 116:204] = 12
    assert not assess_display(off, wrong_surface, returned, roi)['passed']


@pytest.mark.parametrize('field', ['object_q', 'object_qd', 'joint_target_q',
                                 'native_physics_step_index', 'native_world_time_s'])
def test_unchanged_clock_does_not_hide_pose_velocity_or_target_writes(field):
    state = {'object_q': np.zeros((1, 7)), 'object_qd': np.zeros((1, 6)),
        'joint_target_q': np.zeros(29), 'native_physics_step_index': np.asarray(0),
        'native_world_time_s': np.asarray(0.)}
    after = deepcopy(state)
    assert same_physics(state, after)
    after[field] = after[field]+1
    with pytest.raises(ValueError, match='changed physical state'):
        same_physics(state, after)


def test_gpu_uuid_must_come_from_actual_cuda_properties(monkeypatch):
    cuda = SimpleNamespace(device_count=lambda: 1, current_device=lambda: 0,
        get_device_properties=lambda _: SimpleNamespace(uuid='abcd-1234', name='test', total_memory=1024))
    monkeypatch.setitem(sys.modules, 'torch', SimpleNamespace(cuda=cuda, __version__='cpu-test-double'))
    assert actual_cuda_identity('GPU-abcd-1234')['uuid'] == 'abcd-1234'
    with pytest.raises(ValueError, match='Actual CUDA device UUID'):
        actual_cuda_identity('GPU-other-device')
    cuda.device_count = lambda: 2
    with pytest.raises(ValueError, match='exactly one'):
        actual_cuda_identity('GPU-abcd-1234')
