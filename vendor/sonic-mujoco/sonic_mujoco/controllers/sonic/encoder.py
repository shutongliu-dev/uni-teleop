from collections.abc import Callable
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from ...envs.mujoco.g1 import RobotState
from ...teleop import TeleopCommand

Array = NDArray[np.float64]
EncoderModel = Callable[[Array], Array]
ENCODER_OBSERVATION_DIM = 1762
TOKEN_DIM = 64
WRIST_JOINTS = (23, 24, 25, 26, 27, 28)

# Offsets in the deployed SONIC encoder observation_config.yaml.
SMPL_JOINTS_OFFSET = 922
ANCHOR_ORIENTATION_OFFSET = 1642
WRIST_JOINTS_OFFSET = 1702


def _normalize(quaternion: Array) -> Array:
    value = np.asarray(quaternion, dtype=np.float64)
    norm = np.linalg.norm(value)
    if value.shape != (4,) or not np.isfinite(norm) or norm < 1e-8:
        raise ValueError("quaternion must be a finite non-zero vector of shape (4,)")
    return value / norm


def _multiply(left: Array, right: Array) -> Array:
    w1, x1, y1, z1 = left
    w2, x2, y2, z2 = right
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ]
    )


def _conjugate(quaternion: Array) -> Array:
    return quaternion * np.array([1.0, -1.0, -1.0, -1.0])


def _heading(quaternion: Array) -> Array:
    w, x, y, z = _normalize(quaternion)
    yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return np.array([np.cos(yaw / 2.0), 0.0, 0.0, np.sin(yaw / 2.0)])


def _orientation_6d(quaternion: Array) -> Array:
    w, x, y, z = _normalize(quaternion)
    return np.array(
        [
            1.0 - 2.0 * (y * y + z * z),
            2.0 * (x * y - w * z),
            2.0 * (x * y + w * z),
            1.0 - 2.0 * (x * x + z * z),
            2.0 * (x * z - w * y),
            2.0 * (y * z + w * x),
        ]
    )


class _MotionBuffer:
    def __init__(self) -> None:
        self.frames: dict[int, tuple[Array, Array, Array]] = {}
        self.current: int | None = None

    def reset(self) -> None:
        self.frames.clear()
        self.current = None

    def update(self, command: TeleopCommand) -> bool | None:
        first = int(command.frame_index[0])
        last = int(command.frame_index[-1])
        reset = not self.frames or first > max(self.frames) + 1
        if reset:
            self.reset()
            self.current = first
        elif last <= max(self.frames):
            return None

        for index, smpl, root, joints in zip(
            command.frame_index,
            command.smpl_joints,
            command.root_quaternion,
            command.joint_position,
        ):
            self.frames[int(index)] = (smpl, root, joints)
        self._trim()
        return reset

    def sample(self, count: int = 10) -> tuple[Array, Array, Array]:
        if self.current is None:
            raise RuntimeError("waiting for the first teleop command")
        indices = np.asarray(sorted(self.frames))
        selected = []
        for target in range(self.current, self.current + count):
            position = min(np.searchsorted(indices, target), len(indices) - 1)
            selected.append(self.frames[int(indices[position])])
        smpl, root, joints = zip(*selected)
        return np.stack(smpl), np.stack(root), np.stack(joints)

    def advance(self) -> None:
        if self.current is not None and self.current + 10 in self.frames:
            self.current += 1
            self._trim()

    def _trim(self) -> None:
        if self.current is None:
            return
        for index in tuple(self.frames):
            if index < self.current - 5:
                del self.frames[index]


class SonicEncoderObservationBuilder:
    """Build only the protocol-v3/SMPL branch of the deployed encoder input."""

    def __init__(self) -> None:
        self._motion = _MotionBuffer()
        self._initial_base: Array | None = None
        self._initial_reference: Array | None = None
        self._delta_heading = 0.0

    def reset(self) -> None:
        self._motion.reset()
        self._initial_base = None
        self._initial_reference = None
        self._delta_heading = 0.0

    def update(self, command: TeleopCommand) -> None:
        reset = self._motion.update(command)
        if reset is None:
            return
        if reset:
            self._initial_base = None
            self._initial_reference = None
            self._delta_heading = 0.0
        else:
            self._delta_heading += command.heading_increment

    def build(self, state: RobotState) -> Array:
        smpl_joints, root_quaternion, joint_position = self._motion.sample()
        base = _normalize(state.base_quaternion)
        if self._initial_base is None:
            self._initial_base = base
            self._initial_reference = _normalize(root_quaternion[0])

        assert self._initial_reference is not None
        apply_heading = _multiply(
            _heading(self._initial_base),
            _conjugate(_heading(self._initial_reference)),
        )
        if self._delta_heading:
            delta = np.array(
                [
                    np.cos(self._delta_heading / 2.0),
                    0.0,
                    0.0,
                    np.sin(self._delta_heading / 2.0),
                ]
            )
            apply_heading = _multiply(delta, apply_heading)
        base_inverse = _conjugate(base)
        orientations = [
            _orientation_6d(
                _multiply(base_inverse, _multiply(apply_heading, reference))
            )
            for reference in root_quaternion
        ]

        observation = np.zeros(ENCODER_OBSERVATION_DIM, dtype=np.float32)
        observation[:4] = (2.0, 0.0, 0.0, 0.0)
        observation[SMPL_JOINTS_OFFSET:ANCHOR_ORIENTATION_OFFSET] = smpl_joints.reshape(
            -1
        )
        observation[ANCHOR_ORIENTATION_OFFSET:WRIST_JOINTS_OFFSET] = np.asarray(
            orientations
        ).reshape(-1)
        observation[WRIST_JOINTS_OFFSET:] = joint_position[:, WRIST_JOINTS].reshape(-1)
        return observation.astype(np.float64)

    def advance(self) -> None:
        self._motion.advance()


class OnnxEncoder:
    def __init__(self, model_path: str | Path) -> None:
        try:
            import onnxruntime as ort
        except ImportError as error:
            raise RuntimeError(
                "install sonic-mujoco[sonic] to run ONNX inference"
            ) from error

        self._session = ort.InferenceSession(
            str(model_path), providers=["CPUExecutionProvider"]
        )
        self._input = self._session.get_inputs()[0].name
        self._output = self._session.get_outputs()[0].name
        if self._session.get_inputs()[0].shape != [1, ENCODER_OBSERVATION_DIM]:
            raise ValueError("SONIC encoder input must have shape [1, 1762]")
        if self._session.get_outputs()[0].shape != [1, TOKEN_DIM]:
            raise ValueError("SONIC encoder output must have shape [1, 64]")

    def __call__(self, observation: Array) -> Array:
        value = np.asarray(observation, dtype=np.float32)
        if value.shape != (ENCODER_OBSERVATION_DIM,):
            raise ValueError(
                f"observation must have shape ({ENCODER_OBSERVATION_DIM},)"
            )
        output = self._session.run([self._output], {self._input: value[None]})[0]
        return np.asarray(output[0], dtype=np.float64)


class SonicEncoder:
    def __init__(self, model: EncoderModel) -> None:
        self._model = model
        self.observations = SonicEncoderObservationBuilder()

    @classmethod
    def from_onnx(cls, model_path: str | Path) -> "SonicEncoder":
        return cls(OnnxEncoder(model_path))

    def reset(self) -> None:
        self.observations.reset()

    def encode(self, state: RobotState, command: TeleopCommand | None = None) -> Array:
        if command is not None:
            self.observations.update(command)
        observation = self.observations.build(state)
        token = np.asarray(self._model(observation), dtype=np.float64)
        if token.shape != (TOKEN_DIM,) or not np.isfinite(token).all():
            raise ValueError("encoder output must be a finite vector of shape (64,)")
        self.observations.advance()
        return token
