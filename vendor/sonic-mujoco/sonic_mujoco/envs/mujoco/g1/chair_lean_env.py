from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np
from numpy.typing import NDArray

from .g1_env import MujocoG1Env


@dataclass(frozen=True, slots=True)
class ChairLeanState:
    chair_position: NDArray[np.float64]
    chair_quaternion: NDArray[np.float64]
    backrest_width: float
    backrest_height: float
    backrest_tilt: float
    backrest_softness: float


class MujocoG1ChairLeanEnv(MujocoG1Env):
    """G1 approaches and leans against a compliant, movable armchair."""

    def __init__(self, timestep: float = 0.005) -> None:
        package_root = Path(__file__).resolve().parents[3]
        scene = package_root / "assets" / "mujoco" / "scenes" / "g1" / "chair_lean.xml"
        super().__init__(scene, timestep)

        self._robot_qpos = self._freejoint_qpos_address("floating_base_joint")
        self._chair_qpos = self._freejoint_qpos_address("chair_joint")
        self._chair_body_id = self._body_id("chair")
        self._back_id = self._geom_id("chair_back")
        self._back_shell_id = self._geom_id("chair_back_shell")
        self._chair_geom_ids = np.flatnonzero(
            (self.model.geom_bodyid == self._chair_body_id)
            & (self.model.geom_contype != 0)
        )
        self._base_mass = float(self.model.body_mass[self._chair_body_id])
        self._base_inertia = self.model.body_inertia[self._chair_body_id].copy()
        self._base_friction = self.model.geom_friction[self._chair_geom_ids].copy()
        self._backrest_width = 0.51
        self._backrest_height = 0.31
        self._backrest_tilt = np.deg2rad(10.0)
        self._backrest_softness = 0.035

    def reset(self, seed: int | None = None) -> None:
        super().reset(seed=seed)
        rng = np.random.default_rng(seed)
        self._randomize_chair(rng)
        self._place_chair_and_robot(rng)
        mujoco.mj_forward(self.model, self.data)

    def get_scene_state(self) -> ChairLeanState:
        return ChairLeanState(
            chair_position=self.data.qpos[
                self._chair_qpos : self._chair_qpos + 3
            ].copy(),
            chair_quaternion=self.data.qpos[
                self._chair_qpos + 3 : self._chair_qpos + 7
            ].copy(),
            backrest_width=self._backrest_width,
            backrest_height=self._backrest_height,
            backrest_tilt=self._backrest_tilt,
            backrest_softness=self._backrest_softness,
        )

    def _randomize_chair(self, rng: np.random.Generator) -> None:
        self._backrest_width = rng.uniform(0.48, 0.54)
        self._backrest_height = rng.uniform(0.29, 0.34)
        self._backrest_tilt = rng.uniform(np.deg2rad(7), np.deg2rad(14))
        self._backrest_softness = rng.uniform(0.025, 0.050)

        half_height = self._backrest_height / 2
        rotation = self._y_quaternion(self._backrest_tilt)
        center_x = half_height * np.sin(self._backrest_tilt)
        center_z = 0.35 + half_height * np.cos(self._backrest_tilt)

        self.model.geom_size[self._back_id] = (
            rng.uniform(0.035, 0.050),
            self._backrest_width / 2,
            half_height,
        )
        self.model.geom_pos[self._back_id] = (center_x - 0.040, 0.0, center_z)
        self.model.geom_quat[self._back_id] = rotation
        self.model.geom_solref[self._back_id, 0] = self._backrest_softness

        self.model.geom_size[self._back_shell_id] = (
            0.058,
            self._backrest_width / 2 + 0.025,
            half_height + 0.015,
        )
        self.model.geom_pos[self._back_shell_id] = (
            center_x + 0.005,
            0.0,
            center_z,
        )
        self.model.geom_quat[self._back_shell_id] = rotation

        mass_scale = rng.uniform(0.9, 1.1)
        self.model.body_mass[self._chair_body_id] = self._base_mass * mass_scale
        self.model.body_inertia[self._chair_body_id] = self._base_inertia * mass_scale
        friction_scale = rng.uniform(0.9, 1.1)
        self.model.geom_friction[self._chair_geom_ids] = (
            self._base_friction * friction_scale
        )
        mujoco.mj_setConst(self.model, self.data)

    def _place_chair_and_robot(self, rng: np.random.Generator) -> None:
        chair_yaw = rng.uniform(np.deg2rad(-5), np.deg2rad(5))
        self.data.qpos[self._chair_qpos : self._chair_qpos + 3] = (
            rng.uniform(1.38, 1.52),
            rng.uniform(-0.06, 0.06),
            0.0,
        )
        self.data.qpos[self._chair_qpos + 3 : self._chair_qpos + 7] = (
            self._z_quaternion(chair_yaw)
        )

        robot_yaw = rng.uniform(np.deg2rad(-8), np.deg2rad(8))
        self.data.qpos[self._robot_qpos : self._robot_qpos + 3] = (
            rng.uniform(-0.25, 0.05),
            rng.uniform(-0.16, 0.16),
            0.793,
        )
        self.data.qpos[self._robot_qpos + 3 : self._robot_qpos + 7] = (
            self._z_quaternion(robot_yaw)
        )

    @staticmethod
    def _y_quaternion(angle: float) -> NDArray[np.float64]:
        return np.array([np.cos(angle / 2), 0.0, np.sin(angle / 2), 0.0])

    @staticmethod
    def _z_quaternion(angle: float) -> NDArray[np.float64]:
        return np.array([np.cos(angle / 2), 0.0, 0.0, np.sin(angle / 2)])

    def _body_id(self, name: str) -> int:
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
        if body_id < 0:
            raise ValueError(f"missing chair body: {name}")
        return body_id

    def _geom_id(self, name: str) -> int:
        geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name)
        if geom_id < 0:
            raise ValueError(f"missing chair geom: {name}")
        return geom_id

    def _freejoint_qpos_address(self, name: str) -> int:
        joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint_id < 0 or self.model.jnt_type[joint_id] != mujoco.mjtJoint.mjJNT_FREE:
            raise ValueError(f"missing chair freejoint: {name}")
        return int(self.model.jnt_qposadr[joint_id])
