"""Camera calibration in optical coordinates: x right, y down, z forward."""
import math

import numpy as np


def rotation_quaternion(matrix):
    # Numerically stable branch form; avoids an optional scipy dependency in Kit.
    m = matrix
    if m[0][0]+m[1][1]+m[2][2] > 0:
        s = math.sqrt(1+m[0][0]+m[1][1]+m[2][2])*2
        q = [s/4,(m[2][1]-m[1][2])/s,(m[0][2]-m[2][0])/s,(m[1][0]-m[0][1])/s]
    else:
        i = max(range(3),key=lambda j:m[j][j]);j=(i+1)%3;k=(i+2)%3
        s = math.sqrt(1+m[i][i]-m[j][j]-m[k][k])*2
        xyz=[0.,0.,0.];xyz[i]=s/4;xyz[j]=(m[j][i]+m[i][j])/s;xyz[k]=(m[k][i]+m[i][k])/s
        q=[(m[k][j]-m[j][k])/s,*xyz]
    norm=math.sqrt(sum(x*x for x in q))
    return [x/norm for x in q]


def public_camera_calibration(captured, *, width, height):
    """Convert the *same RGB-D frame's* RTX matrices into the public optical frame.

    RTX uses row-vector world-to-view coordinates (+y up, -z forward). The
    public camera frame is column-vector +x right, +y down, +z forward in metres.
    A moving link's USD transform can lag Fabric; never read USD for this pose.
    """
    if (not isinstance(captured, dict)
            or captured.get('camera_model') != 'pinhole'
            or captured.get('matrix_convention') != 'row_vector_world_to_view_then_view_to_clip'
            or captured.get('resolution') != [width, height]):
        raise RuntimeError('Captured calibration does not match the RGB-D camera contract')
    view = np.asarray(captured.get('world_to_view'), dtype=float)
    projection = np.asarray(captured.get('projection'), dtype=float)
    units = float(captured.get('meters_per_scene_unit', float('nan')))
    if (view.shape != (4, 4) or projection.shape != (4, 4)
            or not np.isfinite(view).all() or not np.isfinite(projection).all()
            or not np.isfinite(units) or units <= 0):
        raise RuntimeError('Invalid captured camera matrices or units')
    rotation = view[:3, :3]
    if (not np.allclose(view[:, 3], [0., 0., 0., 1.], atol=1e-6, rtol=0.)
            or not np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-5, rtol=0.)
            or not math.isclose(float(np.linalg.det(rotation)), 1., abs_tol=1e-5)):
        raise RuntimeError('Captured camera view must be a rigid world transform')
    # These are the perspective terms supported by the public pinhole contract.
    # Principal-point offsets remain allowed and are taken from the render product.
    if (not np.allclose(projection[[0, 0, 0, 1, 1, 1, 3, 3, 3],
                                   [1, 2, 3, 0, 2, 3, 0, 1, 3]], 0., atol=1e-8, rtol=0.)
            or not math.isclose(float(projection[2, 3]), -1., abs_tol=1e-8)):
        raise RuntimeError('Captured projection is not a supported pinhole')
    fx, fy = width * projection[0, 0] / 2, height * projection[1, 1] / 2
    cx, cy = width * (1 - projection[2, 0]) / 2, height * (1 + projection[2, 1]) / 2
    if min(fx, fy) <= 0:
        raise RuntimeError('Captured camera focal scales must be positive')
    world_from_optical = np.linalg.inv(view).T @ np.diag([1., -1., -1., 1.])
    return {'intrinsics_3x3': [float(fx), 0., float(cx), 0., float(fy), float(cy), 0., 0., 1.],
            'world_from_camera': {'frame_id': 'world',
                'position_m': (world_from_optical[:3, 3] * units).tolist(),
                'quaternion_wxyz': rotation_quaternion(world_from_optical[:3, :3])}}


def calibrate_camera(stage, camera_id, width, height):
    """Legacy nominal world-camera calibration; never use for runtime RGB-D poses."""
    from pxr import UsdGeom, Usd, Gf
    camera=UsdGeom.Camera(stage.GetPrimAtPath('/World/Cameras/'+camera_id))
    horizontal=float(camera.GetHorizontalApertureAttr().Get())
    vertical=horizontal*height/width
    # Match sensor aspect explicitly instead of relying on an implicit renderer
    # conform-window setting. Caller renders again after this change.
    camera.GetVerticalApertureAttr().Set(vertical)
    focal=float(camera.GetFocalLengthAttr().Get())
    world=UsdGeom.Xformable(camera).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    origin=world.Transform(Gf.Vec3d(0,0,0))
    axes=[world.Transform(Gf.Vec3d(*v))-origin for v in ((1,0,0),(0,-1,0),(0,0,-1))]
    rotation=[[float(axes[column][row]) for column in range(3)] for row in range(3)]
    return {'intrinsics_3x3':[width*focal/horizontal,0.,width/2,0.,height*focal/vertical,height/2,0.,0.,1.],
            'world_from_camera':{'frame_id':'world','position_m':[float(v) for v in origin],
                                 'quaternion_wxyz':rotation_quaternion(rotation)}}
