"""Robot-mounted or optional head-tracked stereo cameras and XRoboToolkit frame transport."""
from __future__ import annotations

import mmap
import os
from pathlib import Path
import struct
import subprocess
import tempfile

import numpy as np

from .retarget import pose_matrix, wrist_matrix

HEADER = struct.Struct('<4sIIII')


class FrameWriter:
    def __init__(self, width, height, *, listen='0.0.0.0:13579', python='/usr/bin/python3'):
        self.width, self.height = width*2, height
        self.file = tempfile.NamedTemporaryFile(prefix='dexbench-pico-', dir='/dev/shm')
        self.file.truncate(HEADER.size + self.width*height*3)
        self.memory = mmap.mmap(self.file.fileno(), 0)
        self.sequence = 0
        blank = np.zeros((height, width, 3), dtype=np.uint8)
        self.publish(blank, blank)
        bridge = Path(__file__).resolve().parents[3] / 'scripts/teleop/pico_video.py'
        environment = os.environ.copy()
        for name in ('LD_LIBRARY_PATH', 'LD_PRELOAD', 'PYTHONHOME', 'PYTHONPATH'):
            environment.pop(name, None)
        self.process = subprocess.Popen([python, '-B', str(bridge), '--frames', self.file.name, '--listen', listen], env=environment)

    def publish(self, left, right):
        frame = np.concatenate((left, right), axis=1)
        if frame.dtype != np.uint8 or frame.shape != (self.height, self.width, 3):
            raise ValueError('Stereo frame dimensions or dtype changed')
        self.sequence += 1
        self.memory[:HEADER.size] = HEADER.pack(b'SMVF', self.sequence, self.width, self.height, frame.nbytes)
        self.memory[HEADER.size:] = frame.tobytes()
        self.sequence += 1
        self.memory[:HEADER.size] = HEADER.pack(b'SMVF', self.sequence, self.width, self.height, frame.nbytes)

    def check(self):
        if self.process.poll() is not None:
            raise RuntimeError('PICO video bridge exited; inspect its terminal output')

    def close(self):
        self.process.terminate()
        try:
            self.process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait()
        self.memory.close()
        self.file.close()


class _CameraMount:
    def __init__(self, env, camera, robot_poses=None):
        from pxr import Gf, UsdGeom
        spec = env.cameras[camera]
        self.source = env.stage.GetPrimAtPath(env.prepare_cameras()[camera])
        self.XformCache = UsdGeom.XformCache
        self.mount_link = spec.get('mount_link')
        self.local = np.array(UsdGeom.Xformable(self.source).GetLocalTransformation()).T
        if self.mount_link and spec.get('aim_target_m') is not None:
            from bench.simulation.isaacsim_backend.cameras import aimed_link_matrix
            poses = robot_poses if robot_poses is not None else env.observation()['robot']
            mount = wrist_matrix({'wrist': poses[self.mount_link]})
            self.local = np.array(aimed_link_matrix(Gf.Matrix4d(mount.T.tolist()),
                spec['offset_xyz_m'], spec['aim_target_m'], spec['standoff_m'])).T
        self.env, self.Gf = env, Gf

    def camera(self, name, width, height):
        from pxr import UsdGeom
        source = UsdGeom.Camera(self.source)
        path = '/World/Cameras/'+name
        self.env.cameras[name] = {'kind': 'world', 'id': name, 'resolution': [width, height]}
        self.env.camera_paths[name] = path
        camera = UsdGeom.Camera.Define(self.env.stage, path)
        aperture = source.GetHorizontalApertureAttr().Get()
        self.focal_ratio = source.GetFocalLengthAttr().Get() / aperture
        camera.CreateFocalLengthAttr(source.GetFocalLengthAttr().Get())
        camera.CreateHorizontalApertureAttr(aperture)
        camera.CreateVerticalApertureAttr(aperture*height/width)
        camera.CreateClippingRangeAttr(source.GetClippingRangeAttr().Get())
        return UsdGeom.Xformable(camera).AddTransformOp()

    def matrix(self, robot_poses):
        # Fabric renders live robot poses without writing them back into USD.
        # Use the same measured physics state as control, never stale USD xforms.
        if self.mount_link:
            poses = robot_poses if robot_poses is not None else self.env.observation()['robot']
            return wrist_matrix({'wrist': poses[self.mount_link]}) @ self.local
        return np.array(self.XformCache().GetLocalToWorldTransform(self.source)).T


class StereoView:
    def __init__(self, env, camera='wrist', *, width=640, height=480, ipd=.064,
                 head_tracked=False, robot_poses=None, inset_camera=None, rear_camera=False):
        from pxr import Gf
        spec = env.cameras[camera]
        if head_tracked and spec['kind'] != 'world':
            raise ValueError('Head tracking requires a world camera; a wrist camera follows its mount')
        self.mount = _CameraMount(env, camera, robot_poses)
        self.head_tracked = head_tracked
        self.env, self.Gf = env, Gf
        self.width, self.height, self.ipd = width, height, ipd
        self.base = self.mount.matrix(robot_poses)
        self.reference = None
        self.relative = np.eye(4)
        # Operator-only cameras preserve the source optics and leave scene cameras intact.
        self.ops = {side: self.mount.camera('pico_'+side, width, height) for side in ('left', 'right')}
        self.inset = None
        self.rear_op = None
        if inset_camera and inset_camera != camera:
            self.inset = _CameraMount(env, inset_camera, robot_poses)
            inset_width, inset_height = width*2//5, height*2//5
            self.inset_op = self.inset.camera('pico_inset', inset_width, inset_height)
            self.inset_label = inset_camera.upper()
        if rear_camera:
            from pxr import UsdGeom
            from .camera import rear_camera_pose
            self.rear_op = self.mount.camera('pico_rear', width*3//4, height*3//4)
            rear = UsdGeom.Camera(env.stage.GetPrimAtPath(env.camera_paths['pico_rear']))
            rear.GetHorizontalApertureAttr().Set(36.)
            rear.GetVerticalApertureAttr().Set(36.*height/width)
            rear.GetFocalLengthAttr().Set(22.)
            rear.GetClippingRangeAttr().Set(Gf.Vec2f(.02, 20.))
            eye, target = rear_camera_pose(env.scene_spec)
            world = Gf.Matrix4d().SetLookAt(Gf.Vec3d(*eye), Gf.Vec3d(*target), Gf.Vec3d(0, 0, 1)).GetInverse()
            self.rear_op.Set(world)
        self.update(None, robot_poses)

    def recenter(self):
        self.reference = None
        self.relative = np.eye(4)

    def update(self, head, robot_poses=None):
        self.base = self.mount.matrix(robot_poses)
        if head is not None:
            current = pose_matrix(head)
            if self.reference is None:
                self.reference = current
            if self.head_tracked:
                self.relative = np.linalg.inv(self.reference) @ current
        for side, sign in (('left', -1), ('right', 1)):
            eye = np.eye(4)
            eye[0, 3] = sign*self.ipd/2
            world = self.base @ self.relative @ eye
            self.ops[side].Set(self.Gf.Matrix4d(world.T.tolist()))

        if self.inset:
            world = self.inset.matrix(robot_poses)
            self.inset_op.Set(self.Gf.Matrix4d(world.T.tolist()))

    def render(self, markers=()):
        cameras = ('pico_left', 'pico_right') + (('pico_inset',) if self.inset else ())
        if self.rear_op is not None:
            cameras += ('pico_rear',)
        # Render all views at their own resolution in one non-physics update.
        frames = self.env.render_many(cameras)
        if markers:
            for side, op in self.ops.items():
                name = 'pico_'+side
                frames[name] = control_point_overlay(frames[name], markers, np.array(op.Get()).T, self.mount.focal_ratio)
            if self.inset:
                frames['pico_inset'] = control_point_overlay(frames['pico_inset'], markers,
                    np.array(self.inset_op.Get()).T, self.inset.focal_ratio)
        self.frames = frames
        eyes = frames['pico_left'], frames['pico_right']
        if self.inset:
            return tuple(picture_in_picture(eye, frames['pico_inset'], self.inset_label) for eye in eyes)
        return eyes


def picture_in_picture(main, inset, label):
    """A screen-fixed mono inset shared by both eyes; preserve the main stereo pair."""
    from PIL import Image, ImageDraw
    image = Image.fromarray(main)
    width, height = inset.shape[1], inset.shape[0]
    x, y = main.shape[1]-width-8, 24
    draw = ImageDraw.Draw(image)
    draw.rectangle((x-2, y-18, x+width+1, y+height+1), fill=(25, 30, 35))
    draw.text((x+4, y-15), label, fill=(255, 255, 255))
    image.paste(Image.fromarray(inset), (x, y))
    return np.asarray(image).copy()


def control_point_overlay(frame, points, camera_world, focal_ratio):
    """Project measured control references; crosshairs are not contact indicators."""
    from PIL import Image, ImageDraw
    image = Image.fromarray(frame)
    draw = ImageDraw.Draw(image)
    height, width = frame.shape[:2]
    focal = width*focal_ratio
    for point in points:
        local = (np.asarray(point)-camera_world[:3, 3]) @ camera_world[:3, :3]
        if local[2] >= -.001:
            continue
        x, y = width/2 + focal*local[0]/-local[2], height/2 - focal*local[1]/-local[2]
        if not (0 <= x < width and 0 <= y < height):
            continue
        for color, stroke in (((15, 25, 30), 4), ((255, 220, 60), 2)):
            draw.line((x-7, y, x+7, y), fill=color, width=stroke)
            draw.line((x, y-7, x, y+7), fill=color, width=stroke)
    return np.asarray(image).copy()
