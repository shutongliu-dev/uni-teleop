"""Synchronous RTX presentation settings and measured render-product calibration.

No physics state setters belong here. Isaac 6.1 World.render() performs native
articulation kinematics and PhysX Fabric force_update before its non-physics
Kit update. The Fabric scene delegate presents these transforms to RTX; the
RTX render delegate's separate direct-Fabric shortcut conflicts with geometry
streaming in the bundled Hydra 1.0.4, so that shortcut is explicitly disabled.
"""
from __future__ import annotations

import numpy as np


RENDER_SYNC_VERSION = 'fabric_scene_delegate_sync_v2'
# Supply these before SimulationApp constructs its first render delegate.
# Native pilot01 established that UJITSO/geometry=false disables caching, not
# the streaming manager: its /active flag remains true. Keep that readback as
# evidence and remove the incompatible direct-transform shortcut instead.
RENDER_SETTINGS = {
    '/app/useFabricSceneDelegate': True,
    '/rtx/hydra/readTransformsFromFabricInRenderDelegate': False,
    '/rtx/hydra/readGPUTransformsFromFabricInRenderDelegate': False,
    '/rtx/hydra/geometrySyncLoads': True,
    '/rtx-transient/hydra/geometrystreaming/syncLoad': True,
    '/app/asyncRendering': False,
    '/app/asyncRenderingLowLatency': False,
    '/omni/replicator/asyncRendering': False,
}
STREAMING_ACTIVE = '/rtx-transient/hydra/geometrystreaming/active'


def startup_arguments():
    return [f'--{key}={str(value).lower()}' for key, value in RENDER_SETTINGS.items()]


def verify_settings(settings):
    effective = {key: settings.get(key) for key in (*RENDER_SETTINGS, STREAMING_ACTIVE,
        '/UJITSO/geometry', '/persistent/UJITSO/geometry', '/UJITSO/geometryTypes')}
    incorrect = {key: value for key, value in RENDER_SETTINGS.items() if effective[key] != value}
    if incorrect:
        raise RuntimeError(f'Unsafe native render synchronization settings: {effective}')
    return {'version': RENDER_SYNC_VERSION, 'effective_settings': effective,
            'sync_operation': 'Isaac 6.1 World.render: kinematics, PhysX Fabric force_update, non-physics Kit update',
            'transform_transport': 'Fabric scene delegate; direct RTX Fabric transform reads disabled',
            'geometry_streaming_disabled_claimed': False,
            'physics_state_writeback': False}


def camera_calibration(data, *, width, height):
    """Record actual RTX projection, including render-product aspect conformance.

camera_params supplies row-vector world-to-view and view-to-clip matrices.
Do not reconstruct projection from USD's default vertical aperture: the
render product conforms that aperture to its own pixel aspect ratio.
"""
    if not isinstance(data, dict) or data.get('cameraModel') != 'pinhole':
        raise RuntimeError('Native camera_params must describe a pinhole camera')
    resolution = np.asarray(data.get('renderProductResolution'))
    if resolution.shape != (2,) or resolution.tolist() != [width, height]:
        raise RuntimeError('Native camera calibration resolution differs from RGB-D')
    result = {'source': 'RTX camera_params annotator on the RGB-D render product',
              'matrix_convention': 'row_vector_world_to_view_then_view_to_clip',
              'resolution': [width, height], 'camera_model': 'pinhole'}
    for key, destination in [('cameraViewTransform', 'world_to_view'),
                             ('cameraProjection', 'projection')]:
        value = np.asarray(data.get(key), dtype=float)
        if value.size != 16 or not np.isfinite(value).all():
            raise RuntimeError(f'Invalid native camera parameter {key}')
        result[destination] = value.reshape(4, 4).tolist()
    projection = np.asarray(result['projection'])
    # Supported cameras are centered pinholes. These terms come from the actual
    # projection, not the nominal film aperture stored on the USD camera.
    result['intrinsics_px'] = {'fx': float(width * projection[0, 0] / 2),
        'fy': float(height * projection[1, 1] / 2),
        'cx': float(width * (1 - projection[2, 0]) / 2),
        'cy': float(height * (1 + projection[2, 1]) / 2)}
    if min(result['intrinsics_px']['fx'], result['intrinsics_px']['fy']) <= 0:
        raise RuntimeError('Invalid native camera focal scale')
    for key, destination, count in [('cameraAperture', 'authored_aperture_mm', 2),
                                    ('cameraApertureOffset', 'authored_aperture_offset_mm', 2)]:
        value = np.asarray(data.get(key), dtype=float)
        if value.shape != (count,) or not np.isfinite(value).all():
            raise RuntimeError(f'Invalid native camera parameter {key}')
        result[destination] = value.tolist()
    result['focal_length_mm'] = float(data['cameraFocalLength'])
    result['meters_per_scene_unit'] = float(data['metersPerSceneUnit'])
    if not np.isfinite([result['focal_length_mm'], result['meters_per_scene_unit']]).all():
        raise RuntimeError('Nonfinite native camera units')
    return result
