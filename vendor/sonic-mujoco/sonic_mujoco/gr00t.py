from pathlib import Path

import mujoco
import numpy as np

from .controllers.sonic.parameters import DEFAULT_ANGLES
from .envs.mujoco.g1 import RobotState
from .tactile_skin import TACTILE_DEVICE_NAMES, JuQiaoTactileFrame

PROMPT = (
    "Use your forearm to sweep all objects across the blue divider from one side "
    "of the table to the other"
)
INITIAL_MOTION_TOKEN = np.array(
    [
        -0.0625,
        0.0,
        -0.0625,
        -0.125,
        -0.1875,
        -0.0625,
        0.1875,
        0.25,
        0.1875,
        -0.125,
        0.0625,
        -0.0625,
        -0.25,
        -0.25,
        -0.3125,
        -0.0625,
        0.0,
        -0.0625,
        -0.125,
        -0.1875,
        0.0,
        -0.25,
        0.0,
        -0.25,
        -0.0625,
        0.0625,
        0.125,
        -0.125,
        0.25,
        0.1875,
        0.25,
        -0.125,
        0.125,
        0.1875,
        -0.0625,
        0.0,
        -0.1875,
        -0.1875,
        0.25,
        0.0,
        0.0,
        -0.125,
        0.0625,
        0.0,
        -0.0625,
        -0.0625,
        0.1875,
        -0.0625,
        0.0,
        0.0625,
        0.125,
        0.0625,
        0.125,
        0.0625,
        0.125,
        0.0,
        0.125,
        0.1875,
        0.0,
        0.0,
        0.0625,
        0.0625,
        0.1875,
        0.0625,
    ],
    dtype=np.float32,
)


class Gr00tClient:
    def __init__(self, host: str, port: int, timeout_ms: int = 60_000) -> None:
        import msgpack
        import msgpack_numpy
        import zmq

        self._msgpack = msgpack
        self._msgpack_numpy = msgpack_numpy
        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.REQ)
        self._socket.setsockopt(zmq.RCVTIMEO, timeout_ms)
        self._socket.setsockopt(zmq.SNDTIMEO, timeout_ms)
        self._socket.connect(f"tcp://{host}:{port}")

    def ping(self) -> bool:
        return self._call("ping")["status"] == "ok"

    def reset(self) -> None:
        self._call("reset", {"options": None})

    def get_action(self, observation: dict) -> dict[str, np.ndarray]:
        action, _ = self._call(
            "get_action", {"observation": observation, "options": None}
        )
        return {
            key.removeprefix("action."): np.asarray(value)
            for key, value in action.items()
        }

    def close(self) -> None:
        self._socket.close(linger=0)
        self._context.term()

    def _call(self, endpoint: str, data: dict | None = None):
        request = {"endpoint": endpoint}
        if data is not None:
            request["data"] = data
        payload = self._msgpack.packb(request, default=self._msgpack_numpy.encode)
        self._socket.send(payload)
        response = self._msgpack.unpackb(
            self._socket.recv(), object_hook=self._msgpack_numpy.decode, raw=False
        )
        if isinstance(response, dict) and "error" in response:
            raise RuntimeError(response["error"])
        return response


class StereoCamera:
    def __init__(self, model, data, width: int = 640, height: int = 480) -> None:
        self._data = data
        self._renderer = mujoco.Renderer(model, width=width, height=height)
        self._observer = mujoco.MjvCamera()
        self._observer.type = mujoco.mjtCamera.mjCAMERA_FREE
        self._observer.lookat[:] = (0.55, 0.0, 0.75)
        self._observer.distance = 2.5
        self._observer.azimuth = -45.0
        self._observer.elevation = -22.0

    def render(self) -> dict[str, np.ndarray]:
        images = {}
        for key, camera in (
            ("ego_view_left", "head_camera_left"),
            ("ego_view_right", "head_camera_right"),
        ):
            self._renderer.update_scene(self._data, camera=camera)
            images[key] = self._renderer.render().copy()
        return images

    def render_observer(self) -> np.ndarray:
        self._renderer.update_scene(self._data, camera=self._observer)
        return self._renderer.render().copy()

    def close(self) -> None:
        self._renderer.close()


class VideoWriter:
    def __init__(self, path: str | Path, width: int, height: int, fps: int) -> None:
        import av

        self._av = av
        self._container = av.open(str(path), mode="w")
        self._stream = self._container.add_stream("libx264", rate=fps)
        self._stream.width = width
        self._stream.height = height
        self._stream.pix_fmt = "yuv420p"
        self._stream.options = {"preset": "fast", "crf": "20"}

    def write(self, image: np.ndarray) -> None:
        frame = self._av.VideoFrame.from_ndarray(image, format="rgb24")
        for packet in self._stream.encode(frame):
            self._container.mux(packet)

    def close(self) -> None:
        for packet in self._stream.encode():
            self._container.mux(packet)
        self._container.close()


def build_observation(
    state: RobotState,
    images: dict[str, np.ndarray],
    prompt: str = PROMPT,
    tactile: JuQiaoTactileFrame | None = None,
) -> dict:
    joints = (state.joint_position - DEFAULT_ANGLES).astype(np.float32)
    tactile_values = (
        np.zeros((len(TACTILE_DEVICE_NAMES), 256), dtype=np.uint8)
        if tactile is None
        else tactile.values
    )
    return {
        "video": {key: value[None, None] for key, value in images.items()},
        "tactile": {
            key: tactile_values[index][None, None].copy()
            for index, key in enumerate(TACTILE_DEVICE_NAMES)
        },
        "state": {
            "left_leg": _batch(joints[0:6]),
            "right_leg": _batch(joints[6:12]),
            "waist": _batch(joints[12:15]),
            "left_arm": _batch(joints[15:22]),
            "right_arm": _batch(joints[22:29]),
            "left_hand": np.zeros((1, 1, 7), dtype=np.float32),
            "right_hand": np.zeros((1, 1, 7), dtype=np.float32),
            "projected_gravity": _batch(projected_gravity(state.base_quaternion)),
        },
        "language": {"annotation.human.task_description": [[prompt]]},
    }


def save_image(path: str | Path, image: np.ndarray) -> None:
    from PIL import Image

    Image.fromarray(image).save(path)


def _batch(value: np.ndarray) -> np.ndarray:
    return np.asarray(value, dtype=np.float32)[None, None]


def projected_gravity(quaternion: np.ndarray) -> np.ndarray:
    w = quaternion[0]
    vector = -quaternion[1:]
    gravity = np.array([0.0, 0.0, -1.0])
    return (
        gravity * (2.0 * w * w - 1.0)
        + 2.0 * w * np.cross(vector, gravity)
        + 2.0 * vector * np.dot(vector, gravity)
    )
