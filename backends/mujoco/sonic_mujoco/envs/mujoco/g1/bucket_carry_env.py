from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np
from numpy.typing import NDArray

from .g1_env import MujocoG1Env


@dataclass(frozen=True, slots=True)
class BucketCarryState:
    source_table_position: NDArray[np.float64]
    target_table_position: NDArray[np.float64]
    container_position: NDArray[np.float64]
    container_quaternion: NDArray[np.float64]
    container_size: NDArray[np.float64]
    container_mass: float
    container_center_of_mass: NDArray[np.float64]
    container_friction: float


class MujocoG1BucketCarryEnv(MujocoG1Env):
    """G1 carries a large container between two identical folding tables."""

    TABLE_TOP_HEIGHT = 0.762

    def __init__(self, timestep: float = 0.005) -> None:
        package_root = Path(__file__).resolve().parents[3]
        scene = (
            package_root / "assets" / "mujoco" / "scenes" / "g1" / "bucket_carry.xml"
        )
        super().__init__(scene, timestep)

        self._robot_qpos = self._freejoint_qpos_address("floating_base_joint")
        self._source_qpos = self._freejoint_qpos_address("source_table_joint")
        self._target_qpos = self._freejoint_qpos_address("target_table_joint")
        self._container_qpos = self._freejoint_qpos_address("carry_jug_joint")
        self._source_body_id = self._body_id("source_table")
        self._target_body_id = self._body_id("target_table")
        self._container_body_id = self._body_id("carry_jug")

        self._table_body_ids = np.array(
            [self._source_body_id, self._target_body_id], dtype=np.int32
        )
        self._base_table_mass = self.model.body_mass[self._table_body_ids].copy()
        self._base_table_inertia = self.model.body_inertia[self._table_body_ids].copy()
        self._table_geom_ids = np.flatnonzero(
            np.isin(self.model.geom_bodyid, self._table_body_ids)
            & (self.model.geom_contype != 0)
        )
        self._base_table_friction = self.model.geom_friction[
            self._table_geom_ids
        ].copy()

        self._container_geom_ids = np.array(
            [
                self._geom_id("jug_body"),
                self._geom_id("jug_shoulder"),
                self._geom_id("jug_neck"),
                self._geom_id("jug_cap"),
            ]
        )
        self._visual_mesh_id = self._mesh_id("water_gallon_mesh")
        self._container_radius = 0.14
        self._container_height = 0.51
        self._container_mass = 5.5
        self._container_center_of_mass = np.array([0.0, 0.0, 0.20])
        self._container_friction = 0.68

    def reset(self, seed: int | None = None) -> None:
        super().reset(seed=seed)
        rng = np.random.default_rng(seed)
        self._randomize_physics(rng)
        self._place_scene(rng)
        mujoco.mj_forward(self.model, self.data)

    def get_scene_state(self) -> BucketCarryState:
        return BucketCarryState(
            source_table_position=self.data.qpos[
                self._source_qpos : self._source_qpos + 3
            ].copy(),
            target_table_position=self.data.qpos[
                self._target_qpos : self._target_qpos + 3
            ].copy(),
            container_position=self.data.qpos[
                self._container_qpos : self._container_qpos + 3
            ].copy(),
            container_quaternion=self.data.qpos[
                self._container_qpos + 3 : self._container_qpos + 7
            ].copy(),
            container_size=np.array(
                [2 * self._container_radius, self._container_height]
            ),
            container_mass=self._container_mass,
            container_center_of_mass=self._container_center_of_mass.copy(),
            container_friction=self._container_friction,
        )

    def _randomize_physics(self, rng: np.random.Generator) -> None:
        self._container_radius = rng.uniform(0.125, 0.150)
        self._container_height = rng.uniform(0.42, 0.50)
        self._container_mass = rng.uniform(5.0, 10.0)
        self._container_friction = rng.uniform(0.48, 0.82)
        self._container_center_of_mass = np.array(
            [
                rng.uniform(-0.025, 0.025),
                rng.uniform(-0.025, 0.025),
                rng.uniform(0.40, 0.58) * self._container_height,
            ]
        )
        self._resize_container()

        radius = self._container_radius
        height = self._container_height
        transverse_inertia = self._container_mass * (3 * radius**2 + height**2) / 12
        axial_inertia = 0.5 * self._container_mass * radius**2
        self.model.body_mass[self._container_body_id] = self._container_mass
        self.model.body_ipos[self._container_body_id] = self._container_center_of_mass
        self.model.body_inertia[self._container_body_id] = (
            transverse_inertia,
            transverse_inertia,
            axial_inertia,
        )
        self.model.geom_friction[self._container_geom_ids, 0] = self._container_friction

        table_scale = rng.uniform(0.95, 1.05)
        self.model.body_mass[self._table_body_ids] = self._base_table_mass * table_scale
        self.model.body_inertia[self._table_body_ids] = (
            self._base_table_inertia * table_scale
        )
        friction_scale = rng.uniform(0.95, 1.05)
        self.model.geom_friction[self._table_geom_ids] = (
            self._base_table_friction * friction_scale
        )
        mujoco.mj_setConst(self.model, self.data)

    def _resize_container(self) -> None:
        radius = self._container_radius
        height = self._container_height
        body_id, shoulder_id, neck_id, cap_id = self._container_geom_ids

        self.model.geom_pos[body_id] = (0.0, 0.0, 0.33 * height)
        self.model.geom_size[body_id] = (radius, 0.31 * height, 0.0)
        self.model.geom_pos[shoulder_id] = (0.0, 0.0, 0.69 * height)
        self.model.geom_size[shoulder_id] = (radius, radius, 0.10 * height)
        self.model.geom_pos[neck_id] = (0.0, 0.0, 0.86 * height)
        self.model.geom_size[neck_id] = (0.36 * radius, 0.075 * height, 0.0)
        self.model.geom_pos[cap_id] = (0.0, 0.0, 0.97 * height)
        self.model.geom_size[cap_id] = (0.40 * radius, 0.03 * height, 0.0)

        self.model.mesh_scale[self._visual_mesh_id] = (
            2 * radius / 0.29,
            2 * radius / 0.29,
            height / 0.50,
        )

    def _place_scene(self, rng: np.random.Generator) -> None:
        source_position = np.array(
            [rng.uniform(0.90, 1.00), rng.uniform(-0.90, -0.76), 0.0]
        )
        target_position = np.array(
            [rng.uniform(0.90, 1.05), rng.uniform(0.76, 0.94), 0.0]
        )
        source_yaw = rng.uniform(np.deg2rad(-3), np.deg2rad(3))
        target_yaw = rng.uniform(np.deg2rad(-3), np.deg2rad(3))
        self._set_freejoint(self._source_qpos, source_position, source_yaw)
        self._set_freejoint(self._target_qpos, target_position, target_yaw)

        container_position = source_position + (
            rng.uniform(-0.06, 0.06),
            rng.uniform(-0.07, 0.07),
            self.TABLE_TOP_HEIGHT,
        )
        container_yaw = rng.uniform(-np.pi, np.pi)
        self._set_freejoint(self._container_qpos, container_position, container_yaw)

        robot_position = np.array(
            [
                rng.uniform(-0.14, -0.02),
                source_position[1] + rng.uniform(-0.08, 0.08),
                0.793,
            ]
        )
        robot_yaw = rng.uniform(np.deg2rad(-6), np.deg2rad(6))
        self._set_freejoint(self._robot_qpos, robot_position, robot_yaw)

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
            raise ValueError(f"missing bucket-carry body: {name}")
        return body_id

    def _geom_id(self, name: str) -> int:
        geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name)
        if geom_id < 0:
            raise ValueError(f"missing bucket-carry geom: {name}")
        return geom_id

    def _mesh_id(self, name: str) -> int:
        mesh_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_MESH, name)
        if mesh_id < 0:
            raise ValueError(f"missing bucket-carry mesh: {name}")
        return mesh_id

    def _freejoint_qpos_address(self, name: str) -> int:
        joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint_id < 0 or self.model.jnt_type[joint_id] != mujoco.mjtJoint.mjJNT_FREE:
            raise ValueError(f"missing bucket-carry freejoint: {name}")
        return int(self.model.jnt_qposadr[joint_id])
