"""A free SHARPA articulation and a table, with contact-respecting wrist PD."""
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from uni_teleop.backends.sharpa import PALM, INITIAL_POSITION, INITIAL_QUATERNION
from uni_teleop.control.floating_hand import wrist_wrench


def build_xml(hand, output):
    urdf = ET.parse(hand.urdf).getroot()
    compiler = ET.SubElement(urdf, 'mujoco')
    ET.SubElement(compiler, 'compiler', discardvisual='false', fusestatic='false', balanceinertia='true')
    imported = mujoco.MjModel.from_xml_string(ET.tostring(urdf, encoding='unicode'))
    path = Path(output) / 'sharpa_table.xml'
    mujoco.mj_saveLastXML(str(path), imported)
    root = ET.parse(path).getroot()
    ET.SubElement(root, 'option', timestep='.002', integrator='implicitfast', iterations='50')
    visual = ET.SubElement(root, 'visual')
    ET.SubElement(visual, 'global', offwidth='1280', offheight='960')
    world = root.find('worldbody')
    palm = world.find('body')
    if palm.get('name') != PALM:
        raise ValueError('Standalone model root must be the SHARPA palm')
    palm.set('pos', ' '.join(map(str, INITIAL_POSITION)))
    palm.set('quat', ' '.join(map(str, INITIAL_QUATERNION[[3, 0, 1, 2]])))
    ET.SubElement(palm, 'freejoint', name='palm_free')
    # Gravity support for the virtual hand only. Props still fall normally.
    for body in palm.iter('body'):
        body.set('gravcomp', '1')
    for geom in palm.iter('geom'):
        if geom.get('contype') == '0' and geom.get('conaffinity') == '0':
            continue  # Original CAD visuals, not extra collision geometry.
        geom.set('contype', '2')
        geom.set('conaffinity', '1')
        geom.set('friction', '1 .01 .001')
        geom.set('rgba', '0 0 0 0')
        geom.set('group', '3')
    ET.SubElement(world, 'light', pos='0 -1 2.5', dir='0 .3 -1')
    ET.SubElement(world, 'geom', name='floor', type='plane', size='2 2 .05', rgba='.18 .2 .24 1')
    ET.SubElement(world, 'geom', name='table', type='box', pos='.15 0 .72', size='.55 .45 .03', rgba='.6 .48 .32 1')
    body = ET.SubElement(world, 'body', name='block', pos='.16 0 .785')
    ET.SubElement(body, 'freejoint', name='block_free')
    ET.SubElement(body, 'geom', name='block_geom', type='box', size='.03 .03 .035', mass='.08',
                  friction='1 .01 .001', rgba='.12 .65 .48 1')
    actuators = ET.SubElement(root, 'actuator')
    for name, effort in zip(hand.names, hand.effort):
        joint = palm.find(f'.//joint[@name="{name}"]')
        joint.set('armature', '.0002')
        ET.SubElement(actuators, 'position', name=name+'_drive', joint=name, kp='3', kv='.08',
                      forcelimited='true', forcerange=f'{-effort} {effort}')
    ET.ElementTree(root).write(path, encoding='unicode')
    return path


class FloatingHand:
    control_dt = .02

    def __init__(self, hand, output, *, headless=True):
        self.hand = hand
        self.xml = build_xml(hand, output)
        self.model = mujoco.MjModel.from_xml_path(str(self.xml))
        self.data = mujoco.MjData(self.model)
        self.root_id = self.model.body(PALM).id
        self.joint_ids = np.array([self.model.joint(n).id for n in hand.names])
        self.q_indices = self.model.jnt_qposadr[self.joint_ids]
        self.actuators = np.array([self.model.actuator(n+'_drive').id for n in hand.names])
        self.data.qpos[self.q_indices] = hand.initial
        self.data.ctrl[self.actuators] = hand.initial
        mujoco.mj_forward(self.model, self.data)
        self.renderer = self.viewer = None
        self.events = []
        if not headless:
            from mujoco import viewer
            self.viewer = viewer.launch_passive(self.model, self.data, key_callback=self._key)
            self._camera(self.viewer.cam)

    def _key(self, code):
        if code == 32: self.events.append('toggle')
        if code in (82, 114): self.events.append('recenter')

    @staticmethod
    def _camera(camera):
        camera.lookat[:] = [.08, 0., .95]
        camera.distance, camera.azimuth, camera.elevation = .95, 135., -25.

    def pose(self):
        return self.data.xpos[self.root_id].copy(), self.data.xquat[self.root_id][[1, 2, 3, 0]].copy()

    def view_forward(self):
        scene = mujoco.MjvScene(self.model, maxgeom=0)
        camera = mujoco.MjvCamera()
        self._camera(camera)
        if self.viewer:
            with self.viewer.lock():
                mujoco.mjv_updateCamera(self.model, self.data, self.viewer.cam, scene)
        else:
            mujoco.mjv_updateCamera(self.model, self.data, camera, scene)
        return np.array(scene.camera[0].forward)

    def positions(self):
        return self.data.qpos[self.q_indices].copy()

    def step(self, position, quaternion, fingers):
        self.data.ctrl[self.actuators] = fingers
        jacp, jacr = np.zeros((3, self.model.nv)), np.zeros((3, self.model.nv))
        for _ in range(round(self.control_dt/self.model.opt.timestep)):
            p, q = self.pose()
            mujoco.mj_jacBody(self.model, self.data, jacp, jacr, self.root_id)
            velocity = np.r_[jacp @ self.data.qvel, jacr @ self.data.qvel]
            force, torque = wrist_wrench(p, q, velocity, position, quaternion)
            self.data.qfrc_applied[:] = 0
            mujoco.mj_applyFT(self.model, self.data, force, torque, p, self.root_id, self.data.qfrc_applied)
            mujoco.mj_step(self.model, self.data)
        if not np.isfinite(self.data.qpos).all():
            raise RuntimeError('Non-finite MuJoCo hand state')

    def render(self, status=''):
        if self.viewer: self.viewer.sync()

    def running(self):
        return self.viewer is None or self.viewer.is_running()

    def screenshot(self, path):
        from PIL import Image
        if self.renderer is None:
            self.renderer = mujoco.Renderer(self.model, height=960, width=1280)
        camera = mujoco.MjvCamera()
        self._camera(camera)
        self.renderer.update_scene(self.data, camera=camera)
        Image.fromarray(self.renderer.render()).save(path)

    def close(self):
        if self.viewer: self.viewer.close()
        if self.renderer: self.renderer.close()
