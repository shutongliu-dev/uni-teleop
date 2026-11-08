from types import SimpleNamespace

import numpy as np
import pytest

from scripts.media.native_press_media import _RenderOnlyAdapter, _state, press_cameras


def fixtures():
    data = {'q': np.arange(12, dtype=float).reshape(1, 4, 3), 'qd': np.zeros((1, 4, 3)),
            'pose': np.array([[.6, -.8, .9, 0., 0., 0., 1.]]), 'velocity': np.zeros((1, 6))}
    volume = SimpleNamespace(get_simulation_nodal_positions=lambda: data['q'],
                             get_simulation_nodal_velocities=lambda: data['qd'])
    pad = SimpleNamespace(get_transforms=lambda: data['pose'], get_velocities=lambda: data['velocity'])
    world = SimpleNamespace(current_time_step_index=1600, current_time=.8, render=lambda: None)
    return data, world, volume, pad


@pytest.mark.parametrize('mutation', ['step', 'time', 'q', 'qd', 'pose', 'velocity'])
def test_render_adapter_rejects_any_physical_change(mutation):
    data, world, volume, pad = fixtures()
    def render():
        if mutation == 'step':
            world.current_time_step_index += 1
        elif mutation == 'time':
            world.current_time += .0005
        else:
            data[mutation].flat[0] += 1e-10
    world.render = render
    adapter = _RenderOnlyAdapter(world, volume, pad, _state(world, volume, pad))
    with pytest.raises(RuntimeError, match='Presentation'):
        adapter.update()


def test_render_adapter_accepts_read_only_world_render():
    _, world, volume, pad = fixtures()
    adapter = _RenderOnlyAdapter(world, volume, pad, _state(world, volume, pad))
    adapter.update()
    adapter.update()
    assert adapter.render_calls == 2
    assert world.current_time_step_index == 1600


@pytest.mark.parametrize('field', ['pose', 'velocity'])
def test_render_adapter_also_checks_the_independent_dynamic_platen(field):
    _, world, volume, pad = fixtures()
    data = {'pose': np.array([[0., 0., 1., 0., 0., 0., 1.]]), 'velocity': np.zeros((1, 6))}
    platen = SimpleNamespace(get_transforms=lambda: data['pose'], get_velocities=lambda: data['velocity'])
    extra = {'platen': platen}
    adapter = _RenderOnlyAdapter(world, volume, pad, _state(world, volume, pad, extra), extra)
    adapter.update()
    world.render = lambda: data[field].__setitem__((0, 0), 1e-10)
    with pytest.raises(RuntimeError, match='extra_platen'):
        adapter.update()


def test_fixed_camera_views_frame_the_original_tool_and_test_platen():
    pose = np.array([.6, -.8, .9, 0., 0., 0., 1.])
    cameras = press_cameras(pose)
    # Both views contain the uncompressed tool and the withdrawn test platen.
    points = np.array([[x, y, z] for x in (-.06, .06) for y in (-.039, .039)
                       for z in (0., .059)])+pose[:3]
    for camera in cameras.values():
        forward = np.array(camera['target'])-camera['eye']
        forward /= np.linalg.norm(forward)
        right = np.cross(forward, [0., 0., 1.]); right /= np.linalg.norm(right)
        up = np.cross(right, forward)
        relative = points-camera['eye']
        depth = relative@forward
        assert depth.min() > camera['clipping_m'][0]
        assert np.max(abs(relative@right)/depth) < camera['horizontal_aperture_mm']/(2*camera['focal_length_mm'])
        assert np.max(abs(relative@up)/depth) < camera['vertical_aperture_mm']/(2*camera['focal_length_mm'])
