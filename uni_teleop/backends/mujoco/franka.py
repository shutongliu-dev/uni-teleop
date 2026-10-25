"""Dynamic Franka/SHARPA workbench built from the same pinned URDF as Isaac.

MuJoCo renders the collision meshes; this is a portable practice scene, not a
conversion or qualification of the DexBench benchmark tasks.
"""
from pathlib import Path
import tempfile
import xml.etree.ElementTree as ET

import mujoco
import numpy as np


def build_model(assets, *, base=(-.32, 0., .77), quaternion=(1., 0., 0., 0.)):
    path = Path(assets) / 'robots/assemblies/fr3_sharpa/isaacsim_6_1_v2/fr3_sharpa.urdf'
    urdf = ET.parse(path).getroot()
    for mesh in urdf.findall('.//mesh'):
        mesh.set('filename', str((path.parent / mesh.get('filename')).resolve()))
    compiler = ET.SubElement(urdf, 'mujoco')
    ET.SubElement(compiler, 'compiler', discardvisual='true', fusestatic='false', balanceinertia='true')
    imported = mujoco.MjModel.from_xml_string(ET.tostring(urdf, encoding='unicode'))
    with tempfile.TemporaryDirectory(prefix='uni-franka-') as temporary:
        mjcf = Path(temporary) / 'robot.xml'
        mujoco.mj_saveLastXML(str(mjcf), imported)
        root = ET.parse(mjcf).getroot()
    ET.SubElement(root, 'option', timestep='0.001', integrator='implicitfast', iterations='50')
    world = root.find('worldbody')
    robot = world.find('body')
    robot.set('pos', ' '.join(map(str, base)))
    robot.set('quat', ' '.join(map(str, quaternion)))
    ET.SubElement(world, 'light', pos='0 -1 3', dir='0 0 -1')
    ET.SubElement(world, 'geom', name='floor', type='plane', size='3 3 .1', rgba='.15 .19 .23 1')
    ET.SubElement(world, 'geom', name='table', type='box', pos='.35 0 .72', size='.8 .6 .03', rgba='.65 .56 .41 1')
    for name, y, kind, size, z in [('block', -.14, 'box', '.035 .035 .06', .812),
                                  ('can', .14, 'cylinder', '.035 .06', .812)]:
        body = ET.SubElement(world, 'body', name=name, pos=f'.22 {y} {z}')
        ET.SubElement(body, 'freejoint', name=name+'_free')
        ET.SubElement(body, 'geom', type=kind, size=size, mass='.16', friction='.7 .005 .0001', rgba='.15 .7 .6 1')
    actuators = ET.SubElement(root, 'actuator')
    for joint in root.findall('.//joint'):
        name = joint.get('name')
        if name is None:
            continue
        arm = name.startswith('fr3_')
        ET.SubElement(actuators, 'position', name=name+'_drive', joint=name,
                      kp='2000' if arm else '3', kv='140' if arm else '.1',
                      forcelimited='true', forcerange=joint.get('actuatorfrcrange', '-3 3'))
    return mujoco.MjModel.from_xml_string(ET.tostring(root, encoding='unicode'))


class FrankaWorkbench:
    control_dt = .02

    def __init__(self, assets, geometry, initial, *, unrestricted_arm=False):
        self.geometry = geometry
        from scipy.spatial.transform import Rotation
        q = Rotation.from_matrix(geometry.base[:3, :3]).as_quat()
        self.model = build_model(assets, base=geometry.base[:3, 3], quaternion=q[[3, 0, 1, 2]])
        self.data = mujoco.MjData(self.model)
        self.names = tuple(geometry.names)
        ids = [self.model.joint(n).id for n in self.names]
        if unrestricted_arm:
            self.model.jnt_limited[ids[:7]] = 0
        self.q_indices = self.model.jnt_qposadr[ids]
        self.v_indices = self.model.jnt_dofadr[ids]
        self.actuators = np.array([self.model.actuator(n+'_drive').id for n in self.names])
        self.initial = dict(initial)
        self.renderer = None
        self.reset()

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        q = np.array([self.initial[n] for n in self.names])
        self.data.qpos[self.q_indices] = q
        self.data.ctrl[self.actuators] = q
        mujoco.mj_forward(self.model, self.data)
        return self.positions()

    def positions(self):
        return dict(zip(self.names, map(float, self.data.qpos[self.q_indices])))

    def step(self, targets):
        if set(targets) != set(self.names):
            raise ValueError('Target joint names must match the complete robot')
        q = np.array([targets[n] for n in self.names])
        if not np.isfinite(q).all():
            raise ValueError('Non-finite joint target')
        self.data.ctrl[self.actuators] = q
        for _ in range(round(self.control_dt / self.model.opt.timestep)):
            # Robot weight support in simulation; no physical leader motor write.
            self.data.qfrc_applied[self.v_indices] = self.data.qfrc_bias[self.v_indices]
            mujoco.mj_step(self.model, self.data)
        if not np.isfinite(self.data.qpos).all():
            raise RuntimeError('MuJoCo produced non-finite state')
        return self.positions()

    def render(self, width=640, height=480):
        if self.renderer is None:
            self.renderer = mujoco.Renderer(self.model, height=height, width=width)
        camera = mujoco.MjvCamera()
        camera.lookat[:] = [.1, 0., 1.05]
        camera.distance, camera.azimuth, camera.elevation = 2.1, 135., -25.
        self.renderer.update_scene(self.data, camera=camera)
        return self.renderer.render().copy()

    def close(self):
        if self.renderer:
            self.renderer.close()
