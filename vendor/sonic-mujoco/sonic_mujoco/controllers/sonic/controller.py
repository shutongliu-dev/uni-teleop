from collections.abc import Callable
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from ...envs.mujoco.g1 import RobotCommand, RobotState
from .observation import SonicObservationBuilder
from .parameters import (
    ACTION_SCALE,
    DEFAULT_ANGLES,
    HARDWARE_FROM_ISAACLAB,
    KD,
    KP,
    OBSERVATION_DIM,
    PHYSICS_STEPS_PER_CONTROL,
)

Array = NDArray[np.float64]
Policy = Callable[[Array], Array]


def action_to_command(action: Array) -> RobotCommand:
    raw = np.asarray(action, dtype=np.float32)
    if raw.shape != (29,) or not np.isfinite(raw).all():
        raise ValueError("action must be a finite vector of shape (29,)")
    raw = raw.astype(np.float64)
    return RobotCommand(
        joint_position=(
            DEFAULT_ANGLES + raw[HARDWARE_FROM_ISAACLAB] * ACTION_SCALE
        ),
        joint_velocity=np.zeros(29),
        feedforward_torque=np.zeros(29),
        kp=KP,
        kd=KD,
    )


class OnnxPolicy:
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
        if self._session.get_inputs()[0].shape != [1, OBSERVATION_DIM]:
            raise ValueError("SONIC decoder input must have shape [1, 994]")
        if self._session.get_outputs()[0].shape != [1, 29]:
            raise ValueError("SONIC decoder output must have shape [1, 29]")

    def __call__(self, observation: Array) -> Array:
        value = np.asarray(observation, dtype=np.float32)
        if value.shape != (OBSERVATION_DIM,):
            raise ValueError(f"observation must have shape ({OBSERVATION_DIM},)")
        output = self._session.run([self._output], {self._input: value[None]})[0]
        return np.asarray(output[0], dtype=np.float64)


class SonicController:
    steps_per_action = PHYSICS_STEPS_PER_CONTROL

    def __init__(self, policy: Policy) -> None:
        self._policy = policy
        self._observations = SonicObservationBuilder()
        self.last_action = np.zeros(29)
        self.last_observation = np.zeros(OBSERVATION_DIM)

    @classmethod
    def from_onnx(cls, model_path: str | Path) -> "SonicController":
        return cls(OnnxPolicy(model_path))

    def reset(self) -> None:
        self._observations.reset()
        self.last_action.fill(0.0)
        self.last_observation.fill(0.0)

    def act(self, state: RobotState, token_state: Array) -> RobotCommand:
        self.last_observation = self._observations.build(
            state, token_state, self.last_action
        )
        action = np.asarray(self._policy(self.last_observation), dtype=np.float32)
        command = action_to_command(action)
        self.last_action = action.astype(np.float64)
        return command
