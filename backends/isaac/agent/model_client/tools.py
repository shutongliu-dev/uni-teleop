"""Strict public tools: observation, metric pixel projection, moves and timed hold.

This module never imports a simulator, reads object truth, or executes model
code. Tool calls are ordinary JSON objects; the runner owns their execution
and all action/model/time budgets.
"""
from __future__ import annotations

import hashlib
import re

import numpy as np

from bench.protocol.schemas import CameraObservation
from bench.sdk.depth import decode_metric_depth_npy
from bench.sdk.geometry import camera_z_to_world
from .actions import ActionValidationError, parse_json_object, public_observation
from .geometry_actions import parse_geometry_action


class ToolError(ActionValidationError):
    """Invalid tool request or unavailable public sensor evidence."""


def _pixel_arguments(camera_id, u, v):
    if (not isinstance(camera_id, str)
            or re.fullmatch(r'[A-Za-z0-9_-][A-Za-z0-9_.-]{0,159}', camera_id) is None
            or camera_id in ('.', '..')):
        raise ToolError('camera_id must name an advertised public camera')
    if type(u) is not int or type(v) is not int or u < 0 or v < 0:
        raise ToolError('u and v must be nonnegative zero-based integer pixel indices')


def parse_tool_call(value):
    """Parse exactly one of observe, point_at, move_geometry, hold, or stop.

    ``observe`` accepts no arguments. ``point_at`` takes camera_id/u/v.
    ``move_geometry`` uses the same arguments as the sparse-geometry action,
    without its action discriminator, including optional positive wrist
    position_tolerance_m and orientation_tolerance_rad. ``hold`` requires duration_s with an
    optional reason. ``stop`` accepts an optional reason.
    No semantic locate/grasp tool or arbitrary function name is available.
    """
    try:
        call = parse_json_object(value)
        if set(call) != {'tool', 'arguments'}:
            raise ToolError('Tool call must contain only tool and arguments')
        name, arguments = call['tool'], call['arguments']
        if not isinstance(name, str) or name not in ('observe', 'point_at', 'move_geometry', 'hold', 'stop'):
            raise ToolError('Unknown public tool')
        if not isinstance(arguments, dict):
            raise ToolError('Tool arguments must be an object')
        if name == 'observe':
            if arguments:
                raise ToolError('observe accepts no arguments')
            return {'tool': name, 'arguments': {}}
        if name == 'point_at':
            if set(arguments) != {'camera_id', 'u', 'v'}:
                raise ToolError('point_at requires only camera_id, u and v')
            _pixel_arguments(arguments['camera_id'], arguments['u'], arguments['v'])
            return {'tool': name, 'arguments': dict(arguments)}
        if 'action' in arguments:
            raise ToolError('Tool arguments cannot override the action type')
        parsed = parse_geometry_action({'action': name, **arguments})
        parsed.pop('action')
        return {'tool': name, 'arguments': parsed}
    except ActionValidationError as error:
        if isinstance(error, ToolError):
            raise
        raise ToolError(str(error)) from None


def project_depth_point(camera_rows, observation, camera_id, u, v):
    """Back-project one current public RGB-D pixel into world coordinates.

    Array indices are zero-based integers and the calibrated image location is
    (u + 0.5, v + 0.5). Depth is optical camera z in meters, not ray range.
    The result identifies a visible sensor pixel, never an object or grasp.
    """
    _pixel_arguments(camera_id, u, v)
    try:
        obs = public_observation(observation)
    except ActionValidationError:
        raise ToolError('point_at requires a valid public observation') from None
    if obs.last_action_status in ('accepted', 'running'):
        raise ToolError('point_at requires a current decision-boundary observation')
    cameras = [camera for camera in obs.cameras if camera.camera_id == camera_id]
    if len(cameras) != 1:
        raise ToolError('Camera is absent or duplicated in this public observation')
    camera = cameras[0]
    if (camera.depth_uri is None or camera.depth_sha256 is None or camera.depth_encoding != 'float32_npy'
            or camera.depth_semantics != 'camera_z' or camera.depth_unit != 'm'
            or camera.depth_invalid_value != 0. or camera.depth_rgb_aligned is not True
            or camera.intrinsics_3x3 is None or camera.world_from_camera is None):
        raise ToolError('Camera has no aligned calibrated metric depth')
    if abs(camera.sim_time_s - obs.sim_time_s) > 1e-9 or abs(camera.depth_sim_time_s - obs.sim_time_s) > 1e-9:
        raise ToolError('Depth does not belong to the current observation time')
    pattern = rf'/episodes/{re.escape(obs.episode_id)}/images/(0|[1-9][0-9]*)/{re.escape(camera_id)}'
    match = re.fullmatch(pattern, camera.rgb_uri)
    if match is None or camera.depth_uri != f'/episodes/{obs.episode_id}/depth/{match.group(1)}/{camera_id}':
        raise ToolError('Camera RGB and depth must belong to this episode and revision')
    if not isinstance(camera_rows, list):
        raise ToolError('Downloaded camera rows are unavailable')
    matches = [row for row in camera_rows if isinstance(row, dict)
               and isinstance(row.get('metadata'), dict) and row['metadata'].get('camera_id') == camera_id]
    if len(matches) != 1:
        raise ToolError('Downloaded camera is absent or duplicated')
    row = matches[0]
    try:
        downloaded = CameraObservation.model_validate(row['metadata'])
    except (ValueError, TypeError):
        raise ToolError('Downloaded camera metadata is invalid') from None
    if downloaded != camera:
        raise ToolError('Downloaded camera metadata does not match this observation')
    if u >= camera.width or v >= camera.height:
        raise ToolError('Pixel is outside the current depth image')
    raw = row.get('depth_npy_bytes')
    if not isinstance(raw, bytes) or hashlib.sha256(raw).hexdigest() != camera.depth_sha256:
        raise ToolError('Depth bytes do not match the current public depth digest')
    try:
        depth = decode_metric_depth_npy(raw, width=camera.width, height=camera.height)
    except Exception:
        raise ToolError('Current depth payload is invalid') from None
    provided = row.get('depth_m')
    if (not isinstance(provided, np.ndarray) or provided.dtype != np.dtype('float32')
            or provided.shape != depth.shape or not np.array_equal(provided, depth)):
        raise ToolError('Decoded depth values do not match the verified depth bytes')
    if 'png_bytes' in row and (not isinstance(row['png_bytes'], bytes)
            or hashlib.sha256(row['png_bytes']).hexdigest() != camera.rgb_sha256):
        raise ToolError('RGB bytes do not match the selected public camera')
    z = float(depth[v, u])
    try:
        point = camera_z_to_world(u=u, v=v, depth_m=z,
            intrinsics_3x3=camera.intrinsics_3x3, world_from_camera=camera.world_from_camera)
    except ValueError as error:
        raise ToolError(str(error)) from None
    return {'tool': 'point_at', 'episode_id': obs.episode_id, 'sequence': obs.sequence,
            'sim_time_s': obs.sim_time_s, 'camera_id': camera_id, 'camera_revision': int(match.group(1)),
            'u': u, 'v': v, 'pixel_center_uv': [u + .5, v + .5],
            'depth_m': z, 'depth_semantics': 'camera_z', 'depth_unit': 'm',
            'depth_sha256': camera.depth_sha256, 'world_position_m': point.tolist(), 'frame_id': 'world',
            'scope': 'visible RGB-D pixel; no object identity or grasp inference'}
