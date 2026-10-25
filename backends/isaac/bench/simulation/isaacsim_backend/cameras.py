"""Scene-supplied fixed world views and rigidly mounted wrist cameras.

A camera is either world-fixed (an eye/target pair, as every earlier release
used) or mounted on a robot link, in which case its prim lives under that link
and rides the hand automatically. Nothing here advances physics: a camera is
posed once at build time (world cameras) or by its parent link (wrist cameras).

Backwards compatibility is explicit: a scene with no ``cameras`` block resolves
to the legacy pair, so frozen releases and their recorded calibrations are
unchanged.
"""
from __future__ import annotations

import numpy as np


LEGACY_CAMERAS = {
    'overview': {'eye': [1.90, -2.20, 2.0], 'target': [.25, .25, 1.1], 'focal_length_mm': 22.},
    'task': {'eye': [.74, -.79, 1.59], 'target': [.52, .17, .77], 'focal_length_mm': 27.},
}
KINDS = ('world', 'link')
# Three world views plus one wrist on each independent robot in a dual task.
# Existing releases still resolve their declared camera set unchanged.
MAX_CAMERAS = 5
# A hand-mounted camera placed at its link origin sees only the hand (measured
# depth 2-6 cm in the room contract probe). Pushed this far along the line
# to its target it clears the fingers and frames the work surface.
DEFAULT_LINK_STANDOFF_M = 0.12


def _work_table_top(scene):
    """The centre of the scene's work-table top, used as a default camera aim."""
    table = scene.get('table')
    if not table:
        return None
    position = [float(v) for v in table['position_m']]
    half = [float(v) for v in table['half_extents_m']]
    return [position[0], position[1], position[2] + half[2]]


def camera_ids(specs):
    return list(specs)


def resolve_cameras(scene, *, default=None):
    """Normalise a scene's ``cameras`` block.

    ``None`` (no block) returns the legacy overview/task pair unchanged (no
    ``resolution``: they render at the env's size). A block replaces the pair; each
    of its cameras records its own ``resolution`` (default 640 x 480).
    """
    raw = scene.get('cameras')
    if raw is None:
        return {name: {'id': name, 'kind': 'world', **dict(value)} for name, value in (default or LEGACY_CAMERAS).items()}
    entries = raw.get('channels', raw) if isinstance(raw, dict) else raw
    if isinstance(entries, dict):
        entries = [dict(value, id=key) for key, value in entries.items()]
    if not isinstance(entries, list) or not entries:
        raise ValueError('cameras must be a non-empty list of camera specifications')
    if len(entries) > MAX_CAMERAS:
        raise ValueError(f'At most {MAX_CAMERAS} cameras are supported, got {len(entries)}')
    resolved = {}
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError('Each camera must be an object')
        name = entry.get('id')
        if not isinstance(name, str) or not name.isidentifier():
            raise ValueError(f'Camera id must be an identifier, got {name!r}')
        if name in resolved:
            raise ValueError(f'Duplicate camera id {name!r}')
        kind = entry.get('kind', 'link' if entry.get('mount_link') else 'world')
        if kind not in KINDS:
            raise ValueError(f'Camera {name!r} kind must be one of {KINDS}')
        focal = float(entry.get('focal_length_mm', 24.))
        if not np.isfinite(focal) or focal <= 0:
            raise ValueError(f'Camera {name!r} focal_length_mm must be positive')
        resolution = entry.get('resolution', [640, 480])
        if len(resolution) != 2 or any(int(v) <= 0 for v in resolution):
            raise ValueError(f'Camera {name!r} resolution must be two positive integers')
        record = {'id': name, 'kind': kind, 'focal_length_mm': focal,
                  'resolution': [int(resolution[0]), int(resolution[1])],
                  'role': entry.get('role', 'third_person' if kind == 'world' else 'wrist'),
                  'title_zh': entry.get('title_zh')}
        if 'clipping_m' in entry or kind == 'link':
            clipping = np.asarray(entry.get('clipping_m', [.02, 20.]), dtype=float)
            if (clipping.shape != (2,) or not np.isfinite(clipping).all()
                    or not 0 < clipping[0] < clipping[1]):
                raise ValueError(f'Camera {name!r} clipping_m must satisfy 0 < near < far')
            record['clipping_m'] = clipping.tolist()
        for field in ('horizontal_aperture_mm', 'vertical_aperture_mm'):
            if field in entry:
                value = float(entry[field])
                if not np.isfinite(value) or value <= 0:
                    raise ValueError(f'Camera {name!r} {field} must be positive and finite')
                record[field] = value
        if kind == 'world':
            eye = [float(v) for v in entry.get('eye', [0., 0., 1.])]
            target = [float(v) for v in entry.get('target', [0., 0., 0.])]
            if len(eye) != 3 or len(target) != 3 or not np.isfinite(eye + target).all():
                raise ValueError(f'Camera {name!r} eye/target must be finite triples')
            if float(np.linalg.norm(np.asarray(eye) - np.asarray(target))) < 1e-6:
                raise ValueError(f'Camera {name!r} eye and target coincide')
            record.update(eye=eye, target=target)
        else:
            link = entry.get('mount_link')
            if not isinstance(link, str) or not link:
                raise ValueError(f'Camera {name!r} needs mount_link')
            offset = [float(v) for v in entry.get('offset_xyz_m', [0., 0., 0.])]
            rpy = [float(v) for v in entry.get('offset_rpy_rad', [0., 0., 0.])]
            if len(offset) != 3 or len(rpy) != 3 or not np.isfinite(offset + rpy).all():
                raise ValueError(f'Camera {name!r} offset must be finite triples')
            aim = entry.get('aim_target_m')
            aim_source = 'declared' if aim is not None else None
            if aim is None and entry.get('aim', True) is not False:
                # An unaimed hand camera at a link origin films the inside of the
                # hand; aim it at the work surface unless the scene opts out.
                aim = _work_table_top(scene)
                aim_source = 'default_work_table_top' if aim is not None else None
            if aim is not None:
                aim = [float(v) for v in aim]
                if len(aim) != 3 or not np.isfinite(aim).all():
                    raise ValueError(f'Camera {name!r} aim_target_m must be a finite triple')
            standoff = float(entry.get('standoff_m', DEFAULT_LINK_STANDOFF_M if aim is not None else 0.))
            if not np.isfinite(standoff) or standoff < 0:
                raise ValueError(f'Camera {name!r} standoff_m must be non-negative')
            record.update(mount_link=link, offset_xyz_m=offset, offset_rpy_rad=rpy,
                          aim_target_m=aim, aim_source=aim_source, standoff_m=standoff,
                          look_axis=entry.get('look_axis', '+z'))
        resolved[name] = record
    return resolved


def aimed_link_matrix(link_world, offset, target, standoff_m=0.0):
    """A link-mounted camera's local transform, aimed at a world point.

    ``link_world`` is the mount link's world transform (a 4x4 as a nested list or
    a ``Gf.Matrix4d``). The camera sits at ``link_world @ offset``, pushed
    ``standoff_m`` further along the direction to ``target``, and is rotated so
    its view axis points at ``target``. The standoff is what keeps a wrist camera
    outside the hand it is mounted on: the hand's own links would otherwise fill
    the frame. The returned transform is expressed in the link frame, so the
    camera rides the hand afterwards.
    """
    from pxr import Gf
    link = Gf.Matrix4d(*link_world) if not isinstance(link_world, Gf.Matrix4d) else link_world
    world_position = link.Transform(Gf.Vec3d(*offset))
    if standoff_m:
        direction = Gf.Vec3d(*target) - world_position
        length = direction.GetLength()
        if length > 1e-6:
            world_position = world_position + direction / length * float(standoff_m)
    view = Gf.Matrix4d().SetLookAt(world_position, Gf.Vec3d(*target), Gf.Vec3d(0, 0, 1)).GetInverse()
    # Gf multiplies row-vectors (v * M), so composing local * link puts the camera
    # in the world; solving for local gives view * link^-1. The other order places
    # the camera at the wrong point and facing away from the target.
    return view * link.GetInverse()


def project_point(stage, prim_path, point_m, width, height, *, calibration=None):
    """Project a world point; ``None`` if behind the camera.

    With ``calibration`` from a captured frame, use that render product's actual
    RTX world-to-view and projection matrices, including aspect conformance.
    Without it the legacy USD-filmback calculation below is only a *nominal*
    framing estimate; it does not guarantee correspondence to rendered pixels.
    Pixel (0, 0) is the top-left corner.
    """
    if calibration is not None:
        if calibration.get('resolution') != [width, height]:
            raise ValueError('Projection resolution differs from captured calibration')
        if calibration.get('matrix_convention') != 'row_vector_world_to_view_then_view_to_clip':
            raise ValueError('Unsupported captured camera matrix convention')
        point = np.asarray(point_m, dtype=float)
        view = np.asarray(calibration.get('world_to_view'), dtype=float)
        projection = np.asarray(calibration.get('projection'), dtype=float)
        units = float(calibration.get('meters_per_scene_unit', float('nan')))
        if (point.shape != (3,) or view.shape != (4, 4) or projection.shape != (4, 4)
                or not np.isfinite(point).all() or not np.isfinite(view).all()
                or not np.isfinite(projection).all() or not np.isfinite(units) or units <= 0):
            raise ValueError('Invalid captured camera projection or world point')
        camera = np.r_[point/units, 1.] @ view
        if camera[2] >= -1e-6:
            return None
        clip = camera @ projection
        if clip[3] <= 1e-12:
            return None
        ndc = clip[:2]/clip[3]
        return [float((ndc[0]+1)*width/2), float((1-ndc[1])*height/2)]
    from pxr import Gf, Usd, UsdGeom
    prim = stage.GetPrimAtPath(prim_path)
    if not prim:
        return None
    world = UsdGeom.XformCache(Usd.TimeCode.Default()).GetLocalToWorldTransform(prim)
    camera = UsdGeom.Camera(prim)
    focal = float(camera.GetFocalLengthAttr().Get())
    horizontal = float(camera.GetHorizontalApertureAttr().Get() or 20.955)
    vertical = float(camera.GetVerticalApertureAttr().Get() or 15.2908)
    eye = world.Transform(Gf.Vec3d(0, 0, 0))
    right = world.TransformDir(Gf.Vec3d(1, 0, 0)).GetNormalized()
    up = world.TransformDir(Gf.Vec3d(0, 1, 0)).GetNormalized()
    forward = -(world.TransformDir(Gf.Vec3d(0, 0, 1)).GetNormalized())
    delta = Gf.Vec3d(*[float(v) for v in point_m]) - eye
    depth = delta * forward
    if depth <= 1e-6:
        return None
    x = (delta * right) / depth * focal
    y = (delta * up) / depth * focal
    return [float(width / 2. + x / horizontal * width), float(height / 2. - y / vertical * height)]


def look_at_matrix(eye, target, up=(0., 0., 1.)):
    from pxr import Gf
    matrix = Gf.Matrix4d().SetLookAt(Gf.Vec3d(*eye), Gf.Vec3d(*target), Gf.Vec3d(*up))
    return matrix.GetInverse()


def link_camera_matrix(window):
    """Camera-to-link transform from an xyz offset and an rpy rotation."""
    from pxr import Gf
    matrix = Gf.Matrix4d(1.)
    matrix.SetRotateOnly(Gf.Rotation(Gf.Vec3d(1, 0, 0), np.degrees(window[0]))
                         * Gf.Rotation(Gf.Vec3d(0, 1, 0), np.degrees(window[1]))
                         * Gf.Rotation(Gf.Vec3d(0, 0, 1), np.degrees(window[2])))
    matrix.SetTranslateOnly(Gf.Vec3d(*window[3]))
    return matrix
