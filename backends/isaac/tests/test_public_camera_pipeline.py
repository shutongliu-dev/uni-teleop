"""CPU evidence for scene camera -> native capture -> HTTP RGB-D publication.

The renderer is a double. These checks do not qualify native images or physics.
"""
from copy import deepcopy
from io import BytesIO

import numpy as np
import pytest
from PIL import Image

from bench.geometry import quaternion_matrix
from bench.paths import load_json
from bench.protocol.schemas import Pose
from bench.runtime import service
from bench.runtime.worker import camera_sensor_configuration, capture_camera_frames
from bench.sdk.geometry import camera_z_to_world
from bench.simulation.isaacsim_backend.camera_calibration import public_camera_calibration
from bench.simulation.isaacsim_backend.cameras import resolve_cameras
from bench.simulation.isaacsim_backend.render_sync import camera_calibration
import test_agent_service as service_tests
from test_agent_service import frozen_inputs, runner  # reuse actual frozen task/lifecycle fixtures


def measured_calibration(width, height, *, eye=(0., 0., 0.), rotation=None, units=1.):
    """Synthetic RTX data with deliberately noncentral principal point."""
    world_from_view = np.eye(4)
    world_from_view[:3, :3] = np.eye(3) if rotation is None else rotation
    world_from_view[:3, 3] = np.asarray(eye) / units
    return camera_calibration({
        'cameraModel': 'pinhole', 'renderProductResolution': [width, height],
        'cameraViewTransform': np.linalg.inv(world_from_view).T.reshape(-1),
        'cameraProjection': np.array([[2., 0., 0., 0.], [0., 3., 0., 0.],
                                      [.12, -.2, 0., -1.], [0., 0., .01, 0.]]).reshape(-1),
        'cameraAperture': [20.955, 15.2908], 'cameraApertureOffset': [0., 0.],
        'cameraFocalLength': 20.955, 'metersPerSceneUnit': units,
    }, width=width, height=height)


class CameraEnv:
    """World and mounted camera views; no USD stage or physics execution API."""
    def __init__(self, *, legacy=False):
        self.scene_spec = {} if legacy else {'cameras': {
            'overview': {'eye': [1., 2., 3.], 'target': [0., 0., 0.], 'resolution': [12, 8]},
            'task': {'eye': [0., 0., 2.], 'target': [0., 0., 0.], 'resolution': [10, 6]},
            'room': {'eye': [3., 3., 3.], 'target': [0., 0., 0.], 'resolution': [16, 9]},
            'wrist_left': {'mount_link': 'fr3_left_link8', 'resolution': [8, 6]},
            'wrist_right': {'mount_link': 'fr3_right_link8', 'resolution': [6, 4]},
        }}
        self.cameras = resolve_cameras(self.scene_spec)
        self.camera_ids = list(self.cameras)
        self.render_width, self.render_height = 640, 480
        self.sim_time_s = 0.
        self.calls = []
        self.mutate_physics = False
        self.capture_updates = {}

    def capture_state(self):
        return {'time': np.array([self.sim_time_s]), 'body_q': np.zeros((2, 7))}

    def render_rgbd(self, *, camera, width, height):
        self.calls.append((camera, width, height))
        if self.mutate_physics:
            self.sim_time_s += .02
        eye = (self.sim_time_s, 0., 1.) if camera.startswith('wrist') else (2., 0., 3.)
        rgb = np.zeros((height, width, 3), dtype=np.uint8)
        rgb[:, :, 0] = 255
        return {'rgb': rgb, 'depth_m': np.full((height, width), .75, dtype=np.float32),
                'sim_time_s': 0., 'depth_sim_time_s': 0.,
                'camera_calibration': measured_calibration(width, height, eye=eye),
                'depth_encoding': 'float32_npy', 'depth_unit': 'm', 'depth_semantics': 'camera_z',
                'depth_invalid_value': 0., 'depth_rgb_aligned': True,
                'depth_statistics': {'private': True}, 'depth_source': 'test',
                'task_visual_state': {'private': True},
                'render_synchronization': {'private': True}, **self.capture_updates}


@pytest.mark.parametrize('angle', [0., np.pi, .71])
@pytest.mark.parametrize('units', [1., .01])
def test_rtx_optical_calibration_backprojects_same_pixel_and_metric_world_pose(angle, units):
    c, s = np.cos(angle), np.sin(angle)
    rotation = np.array([[c, 0., s], [0., 1., 0.], [-s, 0., c]])
    eye = np.array([2., -.4, 1.2])
    measured = measured_calibration(320, 240, eye=eye, rotation=rotation, units=units)
    public = public_camera_calibration(measured, width=320, height=240)
    pose = Pose(**public['world_from_camera'])
    np.testing.assert_allclose(pose.position_m, eye)
    np.testing.assert_allclose(quaternion_matrix(pose.quaternion_wxyz),
                               rotation @ np.diag([1., -1., -1.]), atol=1e-12)
    point = camera_z_to_world(u=131, v=109, depth_m=.75,
        intrinsics_3x3=public['intrinsics_3x3'], world_from_camera=pose)
    # Compare against native row-vector projection, independently of the public K.
    view_point = np.r_[point / units, 1.] @ np.asarray(measured['world_to_view'])
    clip = view_point @ np.asarray(measured['projection'])
    uv = [(clip[0] / clip[3] + 1) * 160, (1 - clip[1] / clip[3]) * 120]
    np.testing.assert_allclose(uv, [131.5, 109.5], atol=1e-10)
    assert -view_point[2] * units == pytest.approx(.75)


@pytest.mark.parametrize('field,value', [
    ('resolution', [640, 480]), ('matrix_convention', 'column_vector'),
    ('camera_model', 'fisheye'), ('meters_per_scene_unit', 0.),
    ('meters_per_scene_unit', float('nan')), ('world_to_view', np.zeros((4, 4)).tolist()),
    ('world_to_view', np.diag([2., 1., 1., 1.]).tolist()),
    ('projection', np.eye(4).tolist()), ('projection', np.full((4, 4), float('nan')).tolist()),
])
def test_public_calibration_rejects_unusable_or_mismatched_native_matrices(field, value):
    measured = measured_calibration(320, 240)
    measured[field] = value
    with pytest.raises(RuntimeError):
        public_camera_calibration(measured, width=320, height=240)


def test_worker_captures_all_scene_views_at_declared_sizes_and_current_wrist_pose(tmp_path):
    env = CameraEnv()
    config = camera_sensor_configuration(env)
    assert service.sensor_configuration({'agent_sensor_config': config}) == config
    before = env.capture_state()
    frames = capture_camera_frames(env, tmp_path, 0, {'sim_time_s': 0.}, config)
    assert [frame['camera_id'] for frame in frames] == env.camera_ids
    assert env.calls == [(name, *env.cameras[name]['resolution']) for name in env.camera_ids]
    assert config['cameras']['wrist_left']['mount_link'] == 'fr3_left_link8'
    for frame in frames:
        assert Image.open(frame['path']).size == (frame['width'], frame['height'])
        depth = np.load(frame['depth_path'], allow_pickle=False)
        assert depth.shape == (frame['height'], frame['width']) and depth.dtype == np.float32
    assert all(np.array_equal(before[key], env.capture_state()[key]) for key in before)
    # Next decision: wrist transform comes from the captured frame, with no USD
    # stage on this double. Static room pose stays fixed while wrist pose changes.
    env.sim_time_s = .02
    env.capture_updates.update(sim_time_s=.02, depth_sim_time_s=.02)
    second = capture_camera_frames(env, tmp_path, 1, {'sim_time_s': .02}, config)
    assert second[2]['world_from_camera'] == frames[2]['world_from_camera']
    assert second[3]['world_from_camera']['position_m'] == pytest.approx([.02, 0., 1.])


def test_backend_resolves_world_and_link_camera_paths_without_a_fixed_world_path_guess():
    from bench.simulation.isaacsim_backend.env import FR3DexterousIsaacEnv
    env = FR3DexterousIsaacEnv.__new__(FR3DexterousIsaacEnv)
    env.cameras = CameraEnv().cameras
    env._link_camera_paths = {}
    env.scene_info = {'robot_link_paths': {'fr3_left_link8': '/World/Left/hand',
                                          'fr3_right_link8': '/World/Right/hand'}}
    assert env._camera_prim_path('room') == '/World/Cameras/room'
    assert env._camera_prim_path('wrist_left') == '/World/Left/hand/bench_camera_wrist_left'
    assert env._camera_prim_path('wrist_right') == '/World/Right/hand/bench_camera_wrist_right'


def test_legacy_scene_preserves_v1_inventory_dimensions_and_hash_fields(tmp_path):
    env = CameraEnv(legacy=True)
    config = camera_sensor_configuration(env)
    assert config == {'protocol': 'agent_rgbd_v1', 'rgb': ['overview', 'task'],
        'width': 640, 'height': 480, 'depth': True, 'depth_encoding': 'float32_npy',
        'depth_unit': 'm', 'depth_semantics': 'camera_z', 'depth_invalid_value': 0.,
        'depth_rgb_aligned': True, 'camera_frame': 'optical_x_right_y_down_z_forward',
        'capture': 'decision_and_policy_steps'}
    assert service.sensor_configuration({'agent_sensor_config': config}) == config
    frames = capture_camera_frames(env, tmp_path, 0, {'sim_time_s': 0.}, config)
    assert [(frame['camera_id'], frame['width'], frame['height']) for frame in frames] == [
        ('overview', 640, 480), ('task', 640, 480)]


@pytest.mark.parametrize('updates', [{'sim_time_s': .02}, {'depth_sim_time_s': .02},
    {'rgb': np.zeros((2, 2, 3), np.uint8)}, {'depth_m': np.zeros((2, 2), np.float32)}])
def test_worker_rejects_clock_or_dimensions_not_matching_observation(tmp_path, updates):
    env = CameraEnv()
    env.capture_updates = updates
    with pytest.raises(RuntimeError):
        capture_camera_frames(env, tmp_path, 0, {'sim_time_s': 0.}, camera_sensor_configuration(env))


def test_worker_rejects_rendering_that_changes_physical_state(tmp_path):
    env = CameraEnv()
    env.mutate_physics = True
    with pytest.raises(RuntimeError, match='changed physical state'):
        capture_camera_frames(env, tmp_path, 0, {'sim_time_s': 0.}, camera_sensor_configuration(env))


@pytest.mark.parametrize('change', ['missing', 'extra', 'duplicate', 'wrong_size', 'bool_size',
                                  'no_mount', 'world_mount', 'no_role', 'extra_field'])
def test_v2_sensor_inventory_rejects_ambiguous_or_incomplete_channel_metadata(change):
    config = camera_sensor_configuration(CameraEnv())
    if change == 'missing':
        del config['cameras']['room']
    elif change == 'extra':
        config['cameras']['hidden'] = deepcopy(config['cameras']['room'])
    elif change == 'duplicate':
        config['rgb'].append('room')
    elif change == 'wrong_size':
        config['cameras']['room']['width'] = 0
    elif change == 'bool_size':
        config['cameras']['room']['width'] = True
    elif change == 'no_mount':
        config['cameras']['wrist_left']['mount_link'] = None
    elif change == 'world_mount':
        config['cameras']['room']['mount_link'] = 'unexpected_link'
    elif change == 'no_role':
        config['cameras']['room']['role'] = ''
    else:
        config['cameras']['room']['private_canary'] = 'not_public'
    with pytest.raises(RuntimeError):
        service.sensor_configuration({'agent_sensor_config': config})


class SceneCameraWorker(service_tests.FakeWorker):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.camera_env = CameraEnv()
        self.initial['metadata']['agent_sensor_config'] = camera_sensor_configuration(self.camera_env)

    def request(self, operation, **payload):
        if operation == 'cameras':
            self.camera_env.sim_time_s = self.raw['sim_time_s']
            self.camera_env.capture_updates = {'sim_time_s': self.raw['sim_time_s'],
                                               'depth_sim_time_s': self.raw['sim_time_s']}
            return {'ok': True, 'frames': capture_camera_frames(self.camera_env, self.output,
                payload['sequence'], self.raw, self.initial['metadata']['agent_sensor_config'])}
        return super().request(operation, **payload)


def test_http_serves_all_five_declared_views_and_keeps_renderer_metadata_private(runner, monkeypatch):
    from fastapi.testclient import TestClient
    from bench.runtime.server import create_app
    monkeypatch.setattr(service_tests, 'FakeWorker', SceneCameraWorker)
    make, _, _ = runner
    api = make()
    with TestClient(create_app(service=api), raise_server_exceptions=False) as client:
        response = client.post('/episodes', json=service_tests.request().model_dump(mode='json'))
        assert response.status_code == 201, response.text
        reply = response.json()
        session = api.get(reply['episode_id'])
        config = load_json(session.directory/'sensor_config.json')
        assert service.canonical_hash(config) == reply['spec']['sensor_config_sha256']
        assert reply['capabilities']['camera_ids'] == config['rgb']
        assert len(reply['observation']['cameras']) == 5
        for frame in reply['observation']['cameras']:
            expected = config['cameras'][frame['camera_id']]
            assert (frame['width'], frame['height']) == (expected['width'], expected['height'])
            assert frame['sim_time_s'] == frame['depth_sim_time_s'] == reply['observation']['sim_time_s']
            assert not {'camera_calibration', 'render_synchronization', 'depth_statistics',
                        'task_visual_state',
                        'path', 'depth_path'} & set(frame)
            rgb = client.get(frame['rgb_uri'])
            depth = client.get(frame['depth_uri'])
            assert rgb.status_code == depth.status_code == 200
            assert Image.open(BytesIO(rgb.content)).size == (expected['width'], expected['height'])
            assert np.load(BytesIO(depth.content), allow_pickle=False).shape == (expected['height'], expected['width'])
        assert str(session.directory) not in response.text


@pytest.mark.parametrize('change', ['missing', 'duplicate', 'wrong_size', 'stale'])
def test_http_rejects_v2_frames_that_disagree_with_pinned_channels(runner, monkeypatch, change):
    class IncorrectCameraWorker(SceneCameraWorker):
        def request(self, operation, **payload):
            result = super().request(operation, **payload)
            if operation == 'cameras':
                if change == 'missing':
                    result['frames'].pop()
                elif change == 'duplicate':
                    result['frames'][-1] = result['frames'][0]
                elif change == 'wrong_size':
                    result['frames'][-1]['width'] += 1
                else:
                    result['frames'][-1]['sim_time_s'] += .02
                    result['frames'][-1]['depth_sim_time_s'] += .02
            return result
    monkeypatch.setattr(service_tests, 'FakeWorker', IncorrectCameraWorker)
    make, _, workers = runner
    with pytest.raises(RuntimeError):
        service_tests.session_for(make())
    assert workers[-1].closed_force
