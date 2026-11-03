"""Desktop auxiliary views for the floating hand; no PICO video changes."""
import numpy as np
from scipy.spatial.transform import Rotation

from .isaac_camera import IN_HAND_WRIST_CAMERA_V1, rear_camera_pose


class HandDesktopViews:
    def __init__(self, stage, scene, pose):
        from pxr import Gf, UsdGeom
        from omni.kit.viewport.utility import create_viewport_window

        self.Gf = Gf
        self.windows = []
        wrist_spec = scene['cameras']['wrist']
        mount = IN_HAND_WRIST_CAMERA_V1
        self.local = np.eye(4)
        self.local[:3, :3] = Rotation.from_euler('xyz', mount['offset_rpy_rad']).as_matrix()
        self.local[:3, 3] = mount['offset_xyz_m']
        for name, title, x, y, width, height, spec in (
            ('wrist', 'Wrist camera', 980, 220, 360, 300, wrist_spec),
            ('rear', 'Rear camera - workbench', 30, 480, 460, 370,
             dict(focal_length_mm=22., horizontal_aperture_mm=36.,
                  vertical_aperture_mm=27., clipping_m=[.02, 20.])),
        ):
            path = '/World/OperatorViews/'+name
            camera = UsdGeom.Camera.Define(stage, path)
            camera.CreateFocalLengthAttr(spec['focal_length_mm'])
            camera.CreateHorizontalApertureAttr(spec['horizontal_aperture_mm'])
            camera.CreateVerticalApertureAttr(spec['vertical_aperture_mm'])
            camera.CreateClippingRangeAttr(Gf.Vec2f(*spec['clipping_m']))
            op = camera.AddTransformOp()
            if name == 'wrist':
                self.wrist_op = op
            else:
                eye, target = rear_camera_pose(scene)
                op.Set(Gf.Matrix4d().SetLookAt(Gf.Vec3d(*eye), Gf.Vec3d(*target),
                                              Gf.Vec3d(0, 0, 1)).GetInverse())
            window = create_viewport_window(title, width=width, height=height,
                position_x=x, position_y=y, camera_path=path)
            if window is None:
                raise RuntimeError('Could not create desktop camera viewport')
            window.viewport_api.set_texture_resolution((640, 480))
            self.windows.append(window)
        self.update(pose)

    def update(self, pose):
        # Fabric does not update USD link transforms: use measured palm state.
        position, quaternion = pose
        world = np.eye(4)
        world[:3, :3] = Rotation.from_quat(quaternion).as_matrix()
        world[:3, 3] = position
        self.wrist_op.Set(self.Gf.Matrix4d((world @ self.local).T.tolist()))

    def close(self):
        for window in self.windows:
            window.destroy()
        self.windows.clear()
