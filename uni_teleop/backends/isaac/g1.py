"""G1 SONIC state/effort adapter over Isaac PhysX; floating-base model."""
from importlib.resources import files
import numpy as np
from scipy.spatial.transform import Rotation
from uni_teleop.backends.isaac.mjcf import IsaacMjcfRobot
from uni_teleop.core.g1 import SONIC_JOINT_NAMES
from uni_teleop.core.robot import RobotState
from uni_teleop.mapping.dexhand import DEXHAND_JOINT_NAMES


class IsaacG1Env(IsaacMjcfRobot):
    def __init__(self, *, output, headless=True):
        xml = files('sonic_mujoco') / 'assets/mujoco/scenes/g1/empty.xml'
        super().__init__(xml, output=output, headless=headless)
        from sonic_mujoco.neck import NECK_JOINT_NAMES
        self.body_indices = self.indices(SONIC_JOINT_NAMES)
        self.hand_indices = self.indices(DEXHAND_JOINT_NAMES)
        self.neck_indices = self.indices(NECK_JOINT_NAMES)
        # Source actuator limits, resolved by joint names rather than backend order.
        import mujoco
        source = mujoco.MjModel.from_xml_path(str(xml))
        self.effort_limits = np.zeros(len(self.names))
        for aid in range(source.nu):
            name = source.joint(int(source.actuator_trnid[aid,0])).name
            if name in self.names:
                self.effort_limits[self.names.index(name)] = max(abs(source.actuator_ctrlrange[aid]))
        if np.any(self.effort_limits <= 0): raise ValueError('Missing source effort limits')
        self._previous_velocity = np.zeros(3)
        self._previous_time = self.time

    def get_robot_state(self):
        root = self.view.get_root_transforms()[0]
        velocity = self.view.get_root_velocities()[0]
        rotation = Rotation.from_quat(root[3:7])
        quat = root[[6,3,4,5]].copy()
        dt = self.time-self._previous_time
        acceleration = (velocity[:3]-self._previous_velocity)/dt if dt > 0 else np.zeros(3)
        self._previous_velocity, self._previous_time = velocity[:3].copy(), self.time
        # Both backends expose local IMU gyro and specific force; root velocity is world-frame.
        return RobotState(self.time, root[:3].copy(), quat, velocity[:3].copy(), velocity[3:].copy(),
            self.view.get_dof_positions()[0,self.body_indices].copy(),
            self.view.get_dof_velocities()[0,self.body_indices].copy(),
            self._efforts[0,self.body_indices].copy(), quat,
            rotation.inv().apply(velocity[3:]), rotation.inv().apply(acceleration-np.array([0,0,-9.81])))

    def step(self, command, steps=1):
        q = self.view.get_dof_positions()[0].copy()
        vel, kp, kd, ff = (np.zeros(len(self.names)) for _ in range(4))
        i = self.body_indices
        q[i], vel[i], kp[i], kd[i], ff[i] = (command.joint_position, command.joint_velocity,
                                          command.kp, command.kd, command.feedforward_torque)
        for indices, target, p, d in ((self.hand_indices, command.hand_joint_position, 2.5, .15),
                                       (self.neck_indices, command.neck_joint_position, 8., .4)):
            if target is not None: q[indices] = target
            kp[indices], kd[indices] = p, d
        self.step_targets(q, vel, kp, kd, self.effort_limits, ff, steps)
