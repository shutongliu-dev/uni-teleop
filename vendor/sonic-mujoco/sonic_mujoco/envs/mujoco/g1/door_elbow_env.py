from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

from .g1_env import MujocoG1Env


@dataclass(frozen=True, slots=True)
class DoorElbowState:
    door_angle: float
    door_angular_velocity: float
    door_mass: float
    hinge_damping: float
    hinge_friction: float
    surface_friction: float
    execution_side: int


class MujocoG1DoorElbowEnv(MujocoG1Env):
    """G1 opens a resisted residential door with an arm or shoulder."""

    def __init__(self, timestep: float = 0.005) -> None:
        package_root = Path(__file__).resolve().parents[3]
        scene = package_root / "assets" / "mujoco" / "scenes" / "g1" / "door_elbow.xml"
        super().__init__(scene, timestep)

        self._robot_qpos = self._freejoint_qpos_address("floating_base_joint")
        self._door_joint_id = self._joint_id("door_hinge")
        self._door_qpos = int(self.model.jnt_qposadr[self._door_joint_id])
        self._door_dof = int(self.model.jnt_dofadr[self._door_joint_id])
        self._door_body_id = self._body_id("door")
        self._door_geom_id = self._geom_id("door_leaf")
        self._base_mass = float(self.model.body_mass[self._door_body_id])
        self._base_inertia = self.model.body_inertia[self._door_body_id].copy()
        self._door_mass = self._base_mass
        self._hinge_damping = 1.6
        self._hinge_friction = 2.8
        self._surface_friction = 0.62
        self._execution_side = 1

    def reset(self, seed: int | None = None) -> None:
        super().reset(seed=seed)
        rng = np.random.default_rng(seed)
        self._randomize_door(rng)
        self._place_robot(rng)
        mujoco.mj_forward(self.model, self.data)

    def get_scene_state(self) -> DoorElbowState:
        return DoorElbowState(
            door_angle=-float(self.data.qpos[self._door_qpos]),
            door_angular_velocity=-float(self.data.qvel[self._door_dof]),
            door_mass=self._door_mass,
            hinge_damping=self._hinge_damping,
            hinge_friction=self._hinge_friction,
            surface_friction=self._surface_friction,
            execution_side=self._execution_side,
        )

    def _randomize_door(self, rng: np.random.Generator) -> None:
        self._door_mass = rng.uniform(20.0, 30.0)
        self._hinge_damping = rng.uniform(0.7, 2.4)
        self._hinge_friction = rng.uniform(1.0, 4.0)
        self._surface_friction = rng.uniform(0.45, 0.75)
        scale = self._door_mass / self._base_mass
        self.model.body_mass[self._door_body_id] = self._door_mass
        self.model.body_inertia[self._door_body_id] = self._base_inertia * scale
        self.model.dof_damping[self._door_dof] = self._hinge_damping
        self.model.dof_frictionloss[self._door_dof] = self._hinge_friction
        self.model.geom_friction[self._door_geom_id, 0] = self._surface_friction
        mujoco.mj_setConst(self.model, self.data)
        self.data.qpos[self._door_qpos] = -rng.uniform(
            np.deg2rad(1.0), np.deg2rad(10.0)
        )
        self.data.qvel[self._door_dof] = 0.0

    def _place_robot(self, rng: np.random.Generator) -> None:
        self._execution_side = int(rng.choice((-1, 1)))
        yaw = self._execution_side * rng.uniform(np.deg2rad(8.0), np.deg2rad(20.0))
        self.data.qpos[self._robot_qpos : self._robot_qpos + 3] = (
            rng.uniform(0.48, 0.70),
            self._execution_side * rng.uniform(0.12, 0.24),
            0.793,
        )
        self.data.qpos[self._robot_qpos + 3 : self._robot_qpos + 7] = (
            np.cos(yaw / 2),
            0.0,
            0.0,
            np.sin(yaw / 2),
        )

    def _body_id(self, name: str) -> int:
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
        if body_id < 0:
            raise ValueError(f"missing door-elbow body: {name}")
        return body_id

    def _joint_id(self, name: str) -> int:
        joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint_id < 0:
            raise ValueError(f"missing door-elbow joint: {name}")
        return joint_id

    def _geom_id(self, name: str) -> int:
        geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name)
        if geom_id < 0:
            raise ValueError(f"missing door-elbow geom: {name}")
        return geom_id

    def _freejoint_qpos_address(self, name: str) -> int:
        joint_id = self._joint_id(name)
        if self.model.jnt_type[joint_id] != mujoco.mjtJoint.mjJNT_FREE:
            raise ValueError(f"expected free joint: {name}")
        return int(self.model.jnt_qposadr[joint_id])
