"""Floating-hand stereo overview plus palm inset, using the existing PICO bridge."""
import json
import time

import numpy as np
from scipy.spatial.transform import Rotation

from .isaac import FrameWriter, picture_in_picture
from .isaac_camera import IN_HAND_WRIST_CAMERA_V1


class HandPicoVideo:
    def __init__(self, env, output, *, listen='0.0.0.0:13579'):
        from isaacsim.sensors.camera import Camera
        from pxr import UsdGeom

        self.env, self.output = env, output
        self.cameras = {}
        for name, resolution in (('left', (640, 480)), ('right', (640, 480)),
                                 ('wrist', (256, 192))):
            camera = Camera('/World/PicoVideo/'+name, resolution=resolution)
            camera.initialize()
            camera.set_clipping_range(.015, 20.)
            optics = UsdGeom.Camera(env.stage.GetPrimAtPath('/World/PicoVideo/'+name))
            optics.CreateHorizontalApertureAttr(36.)
            optics.CreateVerticalApertureAttr(27.)
            optics.CreateFocalLengthAttr(18. if name == 'wrist' else 26.)
            self.cameras[name] = camera
        mount = IN_HAND_WRIST_CAMERA_V1
        self.local = np.eye(4)
        self.local[:3, :3] = Rotation.from_euler('xyz', mount['offset_rpy_rad']).as_matrix()
        self.local[:3, 3] = mount['offset_xyz_m']
        self.writer = FrameWriter(640, 480, listen=listen)
        self.published = 0
        self.started = time.monotonic()
        self.metadata = dict(layout='overview_stereo_with_wrist_inset',
            eye_resolution=[640, 480], inset_resolution=[256, 192], ipd_m=.064,
            head_tracked=False, listen=listen, published_frames=0)
        self._record()

    def update(self):
        from pxr import Usd, UsdGeom

        stage = self.env.stage
        base = np.array(UsdGeom.Xformable(stage.GetPrimAtPath(self.env.camera_path))
                        .ComputeLocalToWorldTransform(Usd.TimeCode.Default())).T
        p, q = self.env.pose()
        palm = np.eye(4)
        palm[:3, :3] = Rotation.from_quat(q).as_matrix()
        palm[:3, 3] = p
        for name, sign in (('left', -.5), ('right', .5), ('wrist', 0.)):
            if name == 'wrist':
                transform = palm @ self.local
            else:
                offset = np.eye(4)
                offset[0, 3] = .064 * sign
                transform = base @ offset
            xyzw = Rotation.from_matrix(transform[:3, :3]).as_quat()
            self.cameras[name].set_world_pose(transform[:3, 3], xyzw[[3, 0, 1, 2]], camera_axes='usd')

    def publish(self):
        self.writer.check()
        frames = {name: camera.get_rgba() for name, camera in self.cameras.items()}
        if any(frame is None or frame.size == 0 for frame in frames.values()):
            return  # Render products warm up during the first application frames.
        rgb = {name: frame[:, :, :3].astype(np.uint8) for name, frame in frames.items()}
        eyes = [picture_in_picture(rgb[name], rgb['wrist'], 'WRIST') for name in ('left', 'right')]
        self.writer.publish(*eyes)
        self.published += 1
        if self.published == 30:
            from PIL import Image
            Image.fromarray(np.concatenate(eyes, axis=1)).save(self.output/'pico_preview.png')
            self._record()

    def _record(self):
        self.metadata.update(published_frames=self.published,
                             elapsed_s=time.monotonic()-self.started)
        (self.output/'pico_video.json').write_text(json.dumps(self.metadata, indent=2)+'\n')

    def close(self):
        self._record()
        self.writer.close()
