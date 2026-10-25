from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np
from numpy.typing import NDArray

from .g1_env import MujocoG1Env


@dataclass(frozen=True, slots=True)
class BasketLoadingState:
    basket_position: NDArray[np.float64]
    basket_quaternion: NDArray[np.float64]
    basket_mass: float
    object_masses: NDArray[np.float64]
    loaded_count: int


class MujocoG1BasketLoadingEnv(MujocoG1Env):
    """G1 steadies a basket while varied objects are loaded into it."""

    LOAD_TIMES = np.arange(5.0, 35.0, 5.0)
    _OBJECTS = ("can", "bottle", "box", "ball", "weight", "soft")
    _MASS_RANGES = np.array(
        [
            (0.35, 0.75),
            (0.55, 1.15),
            (0.75, 1.50),
            (0.25, 0.60),
            (1.60, 2.80),
            (0.50, 1.20),
        ]
    )
    _DROP_OFFSETS = np.array(
        [
            (-0.06, -0.12, 0.38),
            (-0.05, 0.12, 0.40),
            (0.05, -0.08, 0.39),
            (0.03, 0.10, 0.40),
            (-0.02, -0.03, 0.42),
            (0.06, 0.02, 0.41),
        ]
    )

    def __init__(self, timestep: float = 0.005) -> None:
        package_root = Path(__file__).resolve().parents[3]
        scene = (
            package_root / "assets" / "mujoco" / "scenes" / "g1" / "basket_loading.xml"
        )
        super().__init__(scene, timestep)

        self._robot_qpos = self._freejoint_addresses("floating_base_joint")[0]
        self._basket_qpos, _ = self._freejoint_addresses("loading_basket_joint")
        self._basket_body = self._body_id("loading_basket")
        joints = [
            self._freejoint_addresses(f"load_{name}_joint") for name in self._OBJECTS
        ]
        self._object_qpos = np.array([address[0] for address in joints])
        self._object_qvel = np.array([address[1] for address in joints])
        self._object_bodies = np.array(
            [self._body_id(f"load_{name}") for name in self._OBJECTS]
        )
        self._base_object_inertia = self.model.body_inertia[self._object_bodies].copy()
        self._object_geoms = np.array(
            [self._geom_id(f"load_{name}_geom") for name in self._OBJECTS]
        )
        self._source_qpos = self._freejoint_addresses("source_table_joint")[0]
        self._target_qpos = self._freejoint_addresses("target_table_joint")[0]
        self._loading_order = np.arange(len(self._OBJECTS))
        self._object_masses = np.zeros(len(self._OBJECTS))
        self._basket_mass = 1.35
        self._loaded_count = 0
        self._loading_started = False

    def reset(self, seed: int | None = None) -> None:
        super().reset(seed=seed)
        rng = np.random.default_rng(seed)
        self._randomize_physics(rng)
        self._place_scene(rng)
        self._loading_order = rng.permutation(len(self._OBJECTS))
        self._loaded_count = 0
        self._loading_started = False
        mujoco.mj_forward(self.model, self.data)

    def start_loading(self) -> None:
        self._loading_started = True

    def advance_loading(self, elapsed_seconds: float) -> None:
        if not self._loading_started:
            return
        previous_count = self._loaded_count
        while (
            self._loaded_count < len(self.LOAD_TIMES)
            and elapsed_seconds >= self.LOAD_TIMES[self._loaded_count]
        ):
            object_index = self._loading_order[self._loaded_count]
            self._drop_object(object_index, self._loaded_count)
            self._loaded_count += 1
        if self._loaded_count != previous_count:
            mujoco.mj_forward(self.model, self.data)

    def get_scene_state(self) -> BasketLoadingState:
        return BasketLoadingState(
            basket_position=self.data.qpos[
                self._basket_qpos : self._basket_qpos + 3
            ].copy(),
            basket_quaternion=self.data.qpos[
                self._basket_qpos + 3 : self._basket_qpos + 7
            ].copy(),
            basket_mass=self._basket_mass,
            object_masses=self._object_masses.copy(),
            loaded_count=self._loaded_count,
        )

    def _randomize_physics(self, rng: np.random.Generator) -> None:
        self._basket_mass = rng.uniform(1.0, 1.8)
        basket_scale = self._basket_mass / 1.35
        self.model.body_mass[self._basket_body] = self._basket_mass
        self.model.body_inertia[self._basket_body] = (
            np.array([0.045, 0.065, 0.085]) * basket_scale
        )

        self._object_masses = rng.uniform(
            self._MASS_RANGES[:, 0], self._MASS_RANGES[:, 1]
        )
        base_masses = np.array([0.55, 0.85, 1.10, 0.40, 2.20, 0.75])
        for index, body_id in enumerate(self._object_bodies):
            scale = self._object_masses[index] / base_masses[index]
            self.model.body_mass[body_id] = self._object_masses[index]
            self.model.body_inertia[body_id] = self._base_object_inertia[index] * scale
        self.model.geom_friction[self._object_geoms, 0] = rng.uniform(
            0.45, 0.90, len(self._OBJECTS)
        )
        mujoco.mj_setConst(self.model, self.data)

    def _place_scene(self, rng: np.random.Generator) -> None:
        source_position = np.array([0.94, rng.uniform(-0.04, 0.04), 0.0])
        target_position = np.array([0.96, rng.uniform(1.10, 1.22), 0.0])
        self._set_freejoint(self._source_qpos, source_position, 0.0)
        self._set_freejoint(self._target_qpos, target_position, 0.0)

        basket_position = source_position + np.array(
            [rng.uniform(-0.03, 0.03), rng.uniform(-0.04, 0.04), 0.762]
        )
        self._set_freejoint(
            self._basket_qpos,
            basket_position,
            rng.uniform(np.deg2rad(-3), np.deg2rad(3)),
        )

        staging = np.array(
            [
                (-0.17, -0.10, 0.827),
                (0.00, -0.10, 0.899),
                (0.18, -0.10, 0.817),
                (-0.17, 0.10, 0.822),
                (0.01, 0.10, 0.807),
                (0.19, 0.10, 0.827),
            ]
        )
        for index, address in enumerate(self._object_qpos):
            position = target_position + staging[index]
            self._set_freejoint(address, position, rng.uniform(-np.pi, np.pi))

        robot_position = np.array(
            [rng.uniform(-0.16, -0.06), source_position[1], 0.793]
        )
        self._set_freejoint(
            self._robot_qpos,
            robot_position,
            rng.uniform(np.deg2rad(-4), np.deg2rad(4)),
        )

    def _drop_object(self, object_index: int, drop_index: int) -> None:
        basket_position = self.data.qpos[self._basket_qpos : self._basket_qpos + 3]
        basket_quaternion = self.data.qpos[
            self._basket_qpos + 3 : self._basket_qpos + 7
        ]
        offset = np.empty(3)
        mujoco.mju_rotVecQuat(offset, self._DROP_OFFSETS[drop_index], basket_quaternion)
        qpos = self._object_qpos[object_index]
        qvel = self._object_qvel[object_index]
        self.data.qpos[qpos : qpos + 3] = basket_position + offset
        self.data.qpos[qpos + 3 : qpos + 7] = self._z_quaternion(
            0.7 * object_index + 0.4 * drop_index
        )
        self.data.qvel[qvel : qvel + 6] = 0.0
        self.data.qvel[qvel + 2] = -0.10

    def _set_freejoint(
        self, address: int, position: NDArray[np.float64], yaw: float
    ) -> None:
        self.data.qpos[address : address + 3] = position
        self.data.qpos[address + 3 : address + 7] = self._z_quaternion(yaw)

    @staticmethod
    def _z_quaternion(angle: float) -> NDArray[np.float64]:
        return np.array([np.cos(angle / 2), 0.0, 0.0, np.sin(angle / 2)])

    def _body_id(self, name: str) -> int:
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
        if body_id < 0:
            raise ValueError(f"missing basket-loading body: {name}")
        return body_id

    def _geom_id(self, name: str) -> int:
        geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name)
        if geom_id < 0:
            raise ValueError(f"missing basket-loading geom: {name}")
        return geom_id

    def _freejoint_addresses(self, name: str) -> tuple[int, int]:
        joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint_id < 0 or self.model.jnt_type[joint_id] != mujoco.mjtJoint.mjJNT_FREE:
            raise ValueError(f"missing basket-loading freejoint: {name}")
        return int(self.model.jnt_qposadr[joint_id]), int(
            self.model.jnt_dofadr[joint_id]
        )
