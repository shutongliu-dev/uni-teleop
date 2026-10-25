"""CPU rejection checks for the native rendering acceptance harness."""
from copy import deepcopy

import numpy as np
import pytest

from scripts.validation.validate_render_sync import assess_frame, box_depth
from bench.simulation.isaacsim_backend.cameras import LEGACY_CAMERAS, project_point, resolve_cameras
from bench.simulation.isaacsim_backend.env import FR3DexterousIsaacEnv
from bench.simulation.isaacsim_backend.render_sync import (
    RENDER_SETTINGS, STREAMING_ACTIVE, camera_calibration, startup_arguments, verify_settings,
)


def calibration():
    return camera_calibration({
        'cameraModel': 'pinhole', 'renderProductResolution': [320, 240],
        'cameraViewTransform': np.eye(4).reshape(-1),
        'cameraProjection': np.array([[2., 0, 0, 0], [0, 8/3, 0, 0],
                                      [0, 0, 0, -1], [0, 0, .01, 0]]).reshape(-1),
        'cameraAperture': [20.955, 15.2908], 'cameraApertureOffset': [0., 0.],
        'cameraFocalLength': 20.955, 'metersPerSceneUnit': 1.,
    }, width=320, height=240)


def front_face_capture():
    # An independently specified flat visible face: z=.965 and half-size .035.
    rows, columns = np.mgrid[:240, :320]
    mask = (np.abs(columns+.5-160) < 320*.035/.965) & (np.abs(rows+.5-120) < 320*.035/.965)
    rgb = np.zeros((240, 320, 3), dtype=np.uint8)
    rgb[mask] = [255, 0, 255]
    depth = np.zeros((240, 320), dtype=np.float32)
    depth[mask] = .965
    return rgb, depth


def test_measured_projection_overrides_nominal_vertical_aperture():
    result = calibration()
    assert result['intrinsics_px'] == {'fx': 320., 'fy': 320., 'cx': 160., 'cy': 120.}
    assert result['intrinsics_px']['fy'] != pytest.approx(240*20.955/15.2908)
    pose = {'position_m': [0., 0., -1.], 'quaternion_wxyz': [1., 0., 0., 0.]}
    rgb, depth = front_face_capture()
    assert assess_frame(rgb, depth, result, pose)['passed']
    predicted = box_depth(result, pose)
    np.testing.assert_allclose(predicted[depth > 0], .965)


def test_stale_image_and_wrong_depth_are_rejected_even_when_finite():
    pose = {'position_m': [0., 0., -1.], 'quaternion_wxyz': [1., 0., 0., 0.]}
    rgb, depth = front_face_capture()
    stale = {**pose, 'position_m': [0., .10, -1.]}
    assert not assess_frame(rgb, depth, calibration(), stale)['passed']
    assert not assess_frame(rgb, depth+.02, calibration(), pose)['passed']


def test_world_to_view_matrix_is_used_in_ray_intersection():
    original = calibration()
    translated = deepcopy(original)
    translated['world_to_view'][3][0] = -2.
    pose = {'position_m': [2., 0., -1.], 'quaternion_wxyz': [1., 0., 0., 0.]}
    rgb, depth = front_face_capture()
    assert assess_frame(rgb, depth, translated, pose)['passed']


def test_pixel_projection_uses_actual_conformed_matrix_and_world_view():
    measured = calibration()
    measured['world_to_view'][3][0] = -2.
    # No USD stage is accessed: this is exactly the captured camera, including
    # fy=320 despite a nominal vertical aperture that would imply another fy.
    assert project_point(None, None, [2., .1, -1.], 320, 240,
                         calibration=measured) == pytest.approx([160., 88.])
    assert project_point(None, None, [2., 0., 1.], 320, 240, calibration=measured) is None
    assert project_point(None, None, [2., 0., 0.], 320, 240, calibration=measured) is None
    measured['projection'][2][0] = .1
    assert project_point(None, None, [2., 0., -1.], 320, 240,
                         calibration=measured) == pytest.approx([144., 120.])


def test_pixel_projection_rejects_mismatched_or_nonfinite_calibration():
    measured = calibration()
    with pytest.raises(ValueError, match='resolution'):
        project_point(None, None, [0., 0., -1.], 640, 480, calibration=measured)
    measured['projection'][0][0] = float('nan')
    with pytest.raises(ValueError, match='Invalid captured'):
        project_point(None, None, [0., 0., -1.], 320, 240, calibration=measured)


@pytest.mark.parametrize('field', ['horizontal_aperture_mm', 'vertical_aperture_mm'])
@pytest.mark.parametrize('value', [0., -1., float('nan'), float('inf')])
def test_camera_apertures_reject_invalid_values(field, value):
    with pytest.raises(ValueError, match='positive and finite'):
        resolve_cameras({'cameras': {'camera': {**LEGACY_CAMERAS['task'], field: value}}})


def test_optional_apertures_preserve_legacy_defaults():
    assert all('horizontal_aperture_mm' not in c and 'vertical_aperture_mm' not in c
               for c in resolve_cameras({}).values())
    result = resolve_cameras({'cameras': {'room': {**LEGACY_CAMERAS['task'],
        'horizontal_aperture_mm': 36., 'vertical_aperture_mm': 24.}}})
    assert result['room']['horizontal_aperture_mm'] == 36.
    assert result['room']['vertical_aperture_mm'] == 24.


@pytest.mark.parametrize('kind', ['world', 'link'])
@pytest.mark.parametrize('clipping', [[0., 1.], [-.1, 1.], [1., 1.], [2., 1.],
                                     [.01, float('inf')], [float('nan'), 1.], [.01]])
def test_camera_clipping_rejects_invalid_frusta(kind, clipping):
    entry = {'kind': kind, 'mount_link': 'wrist', 'clipping_m': clipping}
    with pytest.raises(ValueError, match='clipping_m'):
        resolve_cameras({'cameras': {'camera': entry}})


def test_explicit_world_clipping_reaches_usd_without_changing_legacy_default():
    pytest.importorskip('pxr')
    from pxr import Usd, UsdGeom
    env = FR3DexterousIsaacEnv.__new__(FR3DexterousIsaacEnv)
    env.stage = Usd.Stage.CreateInMemory()
    env.cameras = resolve_cameras({'cameras': {
        'legacy': dict(LEGACY_CAMERAS['task']),
        'bounded': {**LEGACY_CAMERAS['task'], 'clipping_m': [.025, 8.]},
    }})
    env._author_cameras()
    legacy = UsdGeom.Camera(env.stage.GetPrimAtPath('/World/Cameras/legacy'))
    bounded = UsdGeom.Camera(env.stage.GetPrimAtPath('/World/Cameras/bounded'))
    assert list(legacy.GetClippingRangeAttr().Get()) == pytest.approx([.01, 100.])
    assert list(bounded.GetClippingRangeAttr().Get()) == pytest.approx([.025, 8.])
    assert 'clipping_m' not in resolve_cameras({})['task']


def test_usd_authors_only_explicit_apertures():
    pytest.importorskip('pxr')
    from pxr import Usd, UsdGeom
    env = FR3DexterousIsaacEnv.__new__(FR3DexterousIsaacEnv)
    env.stage = Usd.Stage.CreateInMemory()
    env.cameras = resolve_cameras({'cameras': {
        'legacy': dict(LEGACY_CAMERAS['task']),
        'wide': {**LEGACY_CAMERAS['task'], 'horizontal_aperture_mm': 36.},
    }})
    env._author_cameras()
    legacy = UsdGeom.Camera(env.stage.GetPrimAtPath('/World/Cameras/legacy'))
    wide = UsdGeom.Camera(env.stage.GetPrimAtPath('/World/Cameras/wide'))
    assert not legacy.GetHorizontalApertureAttr().HasAuthoredValueOpinion()
    assert not legacy.GetVerticalApertureAttr().HasAuthoredValueOpinion()
    assert wide.GetHorizontalApertureAttr().Get() == pytest.approx(36.)
    assert not wide.GetVerticalApertureAttr().HasAuthoredValueOpinion()


def test_direct_fabric_shortcut_disabled_before_app_start_and_conflict_rejected():
    args = startup_arguments()
    assert not any(arg.startswith('--/UJITSO/') for arg in args)
    assert '--/app/useFabricSceneDelegate=true' in args
    assert '--/rtx/hydra/readTransformsFromFabricInRenderDelegate=false' in args
    # pilot01 proves the streaming manager stays active when caching is off.
    # The safe path keeps Fabric, but does not directly read it in RTX.
    safe = {**RENDER_SETTINGS, STREAMING_ACTIVE: True}
    assert verify_settings(safe)['effective_settings'][STREAMING_ACTIVE] is True
    with pytest.raises(RuntimeError, match='Unsafe native'):
        verify_settings({**safe, '/rtx/hydra/readTransformsFromFabricInRenderDelegate': True})
    with pytest.raises(RuntimeError, match='Unsafe native'):
        verify_settings({**safe, '/app/useFabricSceneDelegate': False})


def test_legacy_rgb_also_rejects_untimed_physical_pose_changes():
    env = FR3DexterousIsaacEnv.__new__(FR3DexterousIsaacEnv)
    states = iter([{'body_q': np.zeros((1, 7))}, {'body_q': np.ones((1, 7))}])
    env.capture_state = lambda: next(states)
    env._render_frames = lambda *args, **kwargs: {'rgb': np.zeros((1, 1, 3))}
    with pytest.raises(RuntimeError, match='changed physical state'):
        env.render('overview', 1, 1)


@pytest.mark.parametrize('physics_device, repeated_updates', [('cuda:0', 2), ('cpu', 1)])
def test_camera_batch_shares_updates_and_reads_only_after_all_products_are_ready(monkeypatch, physics_device, repeated_updates):
    import sys
    from types import ModuleType, SimpleNamespace
    from bench.simulation.isaacsim_backend import render_sync
    for name in ('carb', 'omni', 'omni.replicator', 'omni.replicator.core'):
        monkeypatch.setitem(sys.modules, name, ModuleType(name))
    sys.modules['carb'].settings = SimpleNamespace(get_settings=lambda: {})
    monkeypatch.setattr(render_sync, 'verify_settings', lambda settings: {})
    monkeypatch.setattr(render_sync, 'camera_calibration', lambda *args, **kwargs: {})
    events, products = [], []
    updates = 0

    def create(path, resolution):
        events.append(('create', path))
        products.append(path)
        return path

    class Annotator:
        def __init__(self, name):
            self.name = name

        def attach(self, product):
            self.product = product[0]

        def get_data(self):
            events.append(('read', self.product))
            return np.full((2, 2, 4), updates, dtype=np.uint8) if self.name == 'rgb' else {}

    rep = sys.modules['omni.replicator.core']
    rep.create = SimpleNamespace(render_product=create)
    rep.AnnotatorRegistry = SimpleNamespace(get_annotator=Annotator)
    env = FR3DexterousIsaacEnv.__new__(FR3DexterousIsaacEnv)
    env.physics_device = physics_device
    env.cameras = {name: {'kind': 'world', 'resolution': (2, 2)} for name in ('left', 'right')}
    env.render_width = env.render_height = 2
    env._render_products, env._cameras_authored = {}, True
    env.stage = SimpleNamespace(GetPrimAtPath=lambda path: True)
    env.time_s = .4
    env.capture_state = lambda: {'time_s': np.array(.4), 'joint_q': np.zeros(29)}

    def render():
        nonlocal updates
        assert len(products) == 2  # Both eyes exist before any rendering.
        events.append(('render', updates))
        updates += 1

    env.world = SimpleNamespace(current_time_step_index=800, current_time=.4, render=render)
    for expected in (8, 8 + repeated_updates):
        frames = env.render_many(('left', 'right'))
        assert updates == expected  # One warm-up sequence for the entire batch.
        assert list(frames) == ['left', 'right']
        for rgb in frames.values():
            np.testing.assert_array_equal(rgb, np.full((2, 2, 3), expected, dtype=np.uint8))
        assert all(event[0] == 'read' for event in events[-4:])
    env.world.render = lambda: setattr(env.world, 'current_time_step_index', 801)
    with pytest.raises(RuntimeError, match='advanced native PhysX'):
        env.render_many(('left', 'right'))


@pytest.mark.parametrize('changed', ['joint_q', 'object_q', 'native_physics_step_index'])
def test_camera_batch_rejects_physical_changes(changed):
    env = FR3DexterousIsaacEnv.__new__(FR3DexterousIsaacEnv)
    states = iter([{changed: np.zeros(1)}, {changed: np.ones(1)}])
    env.capture_state = lambda: next(states)
    env._render_frames = lambda *args, **kwargs: {}
    with pytest.raises(RuntimeError, match='changed physical state'):
        env.render_many(('left', 'right'))
