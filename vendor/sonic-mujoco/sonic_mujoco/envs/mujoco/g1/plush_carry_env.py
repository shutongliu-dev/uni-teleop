from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np
from numpy.typing import NDArray

from .g1_env import MujocoG1Env


@dataclass(frozen=True, slots=True)
class PlushCarryState:
    source_table_position: NDArray[np.float64]
    target_table_position: NDArray[np.float64]
    plush_position: NDArray[np.float64]
    plush_quaternion: NDArray[np.float64]
    plush_extent: NDArray[np.float64]
    plush_mass: float
    plush_stiffness: float
    plush_friction: float


class MujocoG1PlushCarryEnv(MujocoG1Env):
    """G1 carries a deformable plush toy between two folding tables."""

    TABLE_TOP_HEIGHT = 0.762
    PLUSH_HALF_HEIGHT = 0.370

    def __init__(self, timestep: float = 0.005) -> None:
        package_root = Path(__file__).resolve().parents[3]
        scene = package_root / "assets" / "mujoco" / "scenes" / "g1" / "plush_carry.xml"
        super().__init__(scene, timestep)

        self._robot_qpos = self._freejoint_qpos_address("floating_base_joint")
        self._source_qpos = self._freejoint_qpos_address("source_table_joint")
        self._target_qpos = self._freejoint_qpos_address("target_table_joint")
        self._plush_qpos = self._freejoint_qpos_address("carry_plush_joint")

        self._table_body_ids = np.array(
            [self._body_id("source_table"), self._body_id("target_table")]
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

        self._plush_root_id = self._body_id("carry_plush")
        self._flex_id = self._flex_index("plush_flex")
        node_address = int(self.model.flex_nodeadr[self._flex_id])
        node_count = int(self.model.flex_nodenum[self._flex_id])
        if node_count:
            self._plush_body_ids = self.model.flex_nodebodyid[
                node_address : node_address + node_count
            ].copy()
        else:
            vertex_address = int(self.model.flex_vertadr[self._flex_id])
            vertex_count = int(self.model.flex_vertnum[self._flex_id])
            self._plush_body_ids = np.unique(
                self.model.flex_vertbodyid[
                    vertex_address : vertex_address + vertex_count
                ]
            )
            self._plush_body_ids = self._plush_body_ids[self._plush_body_ids > 0]
        self._plush_body_ids = np.concatenate(
            ([self._plush_root_id], self._plush_body_ids)
        )
        self._base_plush_mass = self.model.body_mass[self._plush_body_ids].copy()
        self._base_plush_inertia = self.model.body_inertia[self._plush_body_ids].copy()
        self._flex_equality_id = int(
            np.flatnonzero(
                (self.model.eq_type == mujoco.mjtEq.mjEQ_FLEX)
                & (self.model.eq_obj1id == self._flex_id)
            )[0]
        )
        self._base_flex_solref = self.model.eq_solref[self._flex_equality_id].copy()
        self._vertex_slice = slice(
            int(self.model.flex_vertadr[self._flex_id]),
            int(self.model.flex_vertadr[self._flex_id])
            + int(self.model.flex_vertnum[self._flex_id]),
        )

        self._plush_mass = float(self._base_plush_mass.sum())
        self._plush_stiffness = 1.0
        self._plush_friction = float(self.model.flex_friction[self._flex_id, 0])

    def reset(self, seed: int | None = None) -> None:
        super().reset(seed=seed)
        rng = np.random.default_rng(seed)
        self._randomize_physics(rng)
        self._place_scene(rng)
        mujoco.mj_forward(self.model, self.data)

    def get_scene_state(self) -> PlushCarryState:
        vertices = self.data.flexvert_xpos[self._vertex_slice]
        return PlushCarryState(
            source_table_position=self.data.qpos[
                self._source_qpos : self._source_qpos + 3
            ].copy(),
            target_table_position=self.data.qpos[
                self._target_qpos : self._target_qpos + 3
            ].copy(),
            plush_position=self.data.qpos[
                self._plush_qpos : self._plush_qpos + 3
            ].copy(),
            plush_quaternion=self.data.qpos[
                self._plush_qpos + 3 : self._plush_qpos + 7
            ].copy(),
            plush_extent=np.ptp(vertices, axis=0),
            plush_mass=self._plush_mass,
            plush_stiffness=self._plush_stiffness,
            plush_friction=self._plush_friction,
        )

    def _randomize_physics(self, rng: np.random.Generator) -> None:
        self._plush_mass = rng.uniform(0.9, 2.0)
        mass_scale = self._plush_mass / self._base_plush_mass.sum()
        self.model.body_mass[self._plush_body_ids] = self._base_plush_mass * mass_scale
        self.model.body_inertia[self._plush_body_ids] = (
            self._base_plush_inertia * mass_scale
        )

        self._plush_stiffness = rng.uniform(0.70, 1.30)
        self._plush_friction = rng.uniform(0.60, 0.95)

        table_scale = rng.uniform(0.95, 1.05)
        self.model.body_mass[self._table_body_ids] = self._base_table_mass * table_scale
        self.model.body_inertia[self._table_body_ids] = (
            self._base_table_inertia * table_scale
        )
        self.model.geom_friction[self._table_geom_ids] = (
            self._base_table_friction * rng.uniform(0.95, 1.05)
        )
        mujoco.mj_setConst(self.model, self.data)
        self.model.eq_solref[self._flex_equality_id] = self._base_flex_solref
        self.model.eq_solref[self._flex_equality_id, 0] /= self._plush_stiffness
        self.model.flex_friction[self._flex_id, 0] = self._plush_friction

    def _place_scene(self, rng: np.random.Generator) -> None:
        source = np.array([rng.uniform(0.90, 1.00), rng.uniform(-2.54, -2.38), 0.0])
        target = np.array([rng.uniform(0.90, 1.05), rng.uniform(2.38, 2.56), 0.0])
        self._set_freejoint(self._source_qpos, source, rng.uniform(-0.05, 0.05))
        self._set_freejoint(self._target_qpos, target, rng.uniform(-0.05, 0.05))
        plush = source + (
            rng.uniform(-0.04, 0.04),
            rng.uniform(-0.04, 0.04),
            self.TABLE_TOP_HEIGHT + self.PLUSH_HALF_HEIGHT,
        )
        self._set_freejoint(self._plush_qpos, plush, rng.uniform(-0.16, 0.16))
        robot = np.array(
            [
                source[0] + rng.uniform(-0.08, 0.08),
                source[1] + rng.uniform(-0.92, -0.82),
                0.793,
            ]
        )
        self._set_freejoint(
            self._robot_qpos,
            robot,
            np.pi / 2 + rng.uniform(-0.08, 0.08),
        )

    def _set_freejoint(
        self, address: int, position: NDArray[np.float64], yaw: float
    ) -> None:
        self.data.qpos[address : address + 3] = position
        self.data.qpos[address + 3 : address + 7] = (
            np.cos(yaw / 2),
            0.0,
            0.0,
            np.sin(yaw / 2),
        )

    def _body_id(self, name: str) -> int:
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
        if body_id < 0:
            raise ValueError(f"missing plush-carry body: {name}")
        return body_id

    def _flex_index(self, name: str) -> int:
        flex_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_FLEX, name)
        if flex_id < 0:
            raise ValueError(f"missing plush-carry flex: {name}")
        return flex_id

    def _freejoint_qpos_address(self, name: str) -> int:
        joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint_id < 0 or self.model.jnt_type[joint_id] != mujoco.mjtJoint.mjJNT_FREE:
            raise ValueError(f"missing plush-carry freejoint: {name}")
        return int(self.model.jnt_qposadr[joint_id])
