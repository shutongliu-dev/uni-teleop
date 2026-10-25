"""Static wooden reserve trays, separate from the fixed room and game pieces.

All mesh and collider coordinates use the room world origin. Install each
returned Asset with an identity fixture transform. This is an asset recipe,
not a claim that a robot can grasp every supplied reserve position.
"""
from __future__ import annotations

import math

import numpy as np

from bench.simulation.assets.inlay_geometry import rounded_box
from bench.simulation.assets.scene_parts import Asset

TRAY_MARGIN_M = .040  # 28 mm token radius plus 12 mm margin.
TRAY_THICKNESS_M = .020
LEG_WIDTH_M = .028
TABLE_LOW_M = np.array([.15, -.50, .74])
TABLE_HIGH_M = np.array([1.15, .50, .78])
TABLE_LOW_M.setflags(write=False)
TABLE_HIGH_M.setflags(write=False)
WOOD = [.235, .137, .067]
TIMBER = [.118, .067, .034]
GRAIN = [.207, .117, .055]
MARK = [.155, .086, .039]


def _overlap(a, b):
    return bool((np.minimum(a[1], b[1]) - np.maximum(a[0], b[0]) > 1.e-9).all())


def _add_box(asset, label, center, size, color, *, corner=.003, bevel=.0007):
    mesh = rounded_box(size, min(corner, min(size) / 4), bevel)
    mesh.apply_translation(center)
    asset.add_display(mesh, color, label, roughness=.66, metallic=0.)
    asset.add_box_collision(np.asarray(size) / 2, label, center)


def _flat_details(low, high, positions, z):
    """Fine irregular grain and flat position rings, with no grasp obstruction."""
    import trimesh

    vertices, faces = [], []
    rng = np.random.default_rng(6127)
    x = low[0] + .009
    while x < high[0] - .009:
        y = np.linspace(low[1] + .006, high[1] - .006, 19)
        t = np.linspace(0., 1., len(y))
        center = x + .0011 * np.sin(t * 5.1 + x * 9) + .0004 * np.sin(t * 13. + x)
        width = rng.uniform(.00025, .00065) * np.minimum(1., np.minimum(t, 1 - t) * 8)
        start = len(vertices)
        for xx, yy, ww in zip(center, y, width):
            vertices.extend(((xx - ww / 2, yy, z + .00003),
                             (xx + ww / 2, yy, z + .00003)))
        for i in range(len(y) - 1):
            a = start + 2 * i
            faces.extend(((a, a + 3, a + 2), (a, a + 1, a + 3)))
        x += rng.uniform(.010, .022)
    grain = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    grain.update_faces(grain.nondegenerate_faces())
    grain.remove_unreferenced_vertices()
    vertices, faces = [], []
    for point in positions:
        start = len(vertices)
        for angle in np.arange(64) * math.tau / 64:
            for radius in (.030, .03065):
                vertices.append((point[0] + radius * math.cos(angle),
                                 point[1] + radius * math.sin(angle), z + .00006))
        for i in range(64):
            a, b = start + 2 * i, start + 2 * ((i + 1) % 64)
            faces.extend(((a, a + 1, b + 1), (a, b + 1, b)))
    marks = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    return grain, marks


def _outboard_legs(low, high, bottom):
    """Place a trestle wholly outside one original table edge, preferring X."""
    half = LEG_WIDTH_M / 2
    inset, gap = half + .006, .008
    for axis in (0, 1):
        other = 1 - axis
        candidates = ((low[axis] + inset, TABLE_LOW_M[axis] - gap, -1),
                      (high[axis] - inset, TABLE_HIGH_M[axis] + gap, 1))
        for coordinate, edge, sign in candidates:
            if (coordinate + half <= edge if sign < 0 else coordinate - half >= edge):
                points = []
                for along in (low[other] + inset, high[other] - inset):
                    center = np.zeros(3)
                    center[axis], center[other], center[2] = coordinate, along, bottom / 2
                    points.append(center)
                return points, axis, sign
    return [], None, None


def reserve_tray(asset_id, positions, surface_z=.80):
    """Build a tray around one reserve group; inputs are world XY or base XYZ.

    The positions' XY bounds gain 40 mm on each side. Their Z coordinates do
    not set the support height: the caller must initialize each 14 mm token
    base centre at ``surface_z + .007``. A 20 mm plate rests on the table at
    the default height. Its outboard trestle stands on the original floor.
    Small groups wholly borne by the tabletop need no floor legs.
    """
    points = np.asarray(list(positions), dtype=float)
    if (points.ndim != 2 or points.shape[1] not in (2, 3)
            or not len(points) or not np.isfinite(points).all()):
        raise ValueError('Reserve positions must be a nonempty finite list of XY or XYZ points')
    if not isinstance(asset_id, str) or not asset_id:
        raise ValueError('A reserve tray requires a nonempty asset id')
    surface_z = float(surface_z)
    bottom = surface_z - TRAY_THICKNESS_M
    if not np.isfinite(surface_z) or bottom < TABLE_HIGH_M[2] - 1.e-9:
        raise ValueError('Tray underside must remain at or above the unchanged table top')
    xy = points[:, :2]
    low, high = xy.min(0) - TRAY_MARGIN_M, xy.max(0) + TRAY_MARGIN_M
    legs, axis, sign = _outboard_legs(low, high, bottom)
    if not legs:
        # A small margin can overhang an edge, but the plate centre must have
        # actual tabletop bearing. Never create a floating high shelf.
        center = (low + high) / 2
        if (not np.all(center >= TABLE_LOW_M[:2]) or not np.all(center <= TABLE_HIGH_M[:2])
                or abs(bottom - TABLE_HIGH_M[2]) > 1.e-9):
            raise ValueError('This reserve group has no tabletop bearing or room for an outboard trestle')
    asset = Asset(asset_id, dynamic=False, mass_kg=0., friction=.4)
    center = [*(low + high) / 2, surface_z - TRAY_THICKNESS_M / 2]
    _add_box(asset, 'reserve_platform', center, [*(high - low), TRAY_THICKNESS_M], WOOD,
             corner=.007, bevel=.001)
    for i, point in enumerate(legs):
        _add_box(asset, 'outboard_leg_' + str(i), point, [LEG_WIDTH_M, LEG_WIDTH_M, bottom], TIMBER)
    if legs:
        # Low stretcher reinforces the two-leg trestle without adding any
        # structure near the handles or inside the original table footprint.
        brace_center = (legs[0] + legs[1]) / 2
        brace_center[2] = .18
        brace_size = np.array([.022, .022, .024])
        brace_size[1 - axis] = abs(legs[1][1 - axis] - legs[0][1 - axis])
        _add_box(asset, 'outboard_stretcher', brace_center, brace_size, TIMBER)
    for label, mesh, color in zip(('walnut_grain', 'reserve_position_rings'),
                                  _flat_details(low, high, xy, surface_z), (GRAIN, MARK)):
        asset.add_display(mesh, color, label, roughness=.72, metallic=0.)
    # Assert against authored colliders, not rounded display bounds.
    for label, _, part in asset.collision:
        p, half = np.asarray(part['position_m']), np.asarray(part['half_extents_m'])
        if _overlap((p - half, p + half), (TABLE_LOW_M, TABLE_HIGH_M)):
            raise ValueError('Reserve fixture intersects the original worktop: ' + label)
    asset.data.update(
        origin='room_world_origin; fixture transform must be identity',
        surface_z_m=surface_z, platform_thickness_m=TRAY_THICKNESS_M,
        platform_bounds_xy_m=[low.tolist(), high.tolist()],
        reserve_positions_xy_m=xy.tolist(), required_token_base_center_z_m=surface_z + .007,
        reserve_margin_m=TRAY_MARGIN_M, outboard_leg_count=len(legs),
        support_edge=None if axis is None else {'axis': 'xy'[axis], 'side': sign},
        installation='Task-owned rigid fixture: tabletop bearing where overlapping and floor-supported outboard trestle; fixed room unchanged.',
        appearance={'version': 'reserve_walnut_v1', 'finish': 'matte warm walnut; flat grain and subtle position rings',
                    'maximum_visual_surface_offset_m': .00006, 'raised_rims_or_slots': False},
        material_prior='Static support friction .4, restitution 0; preserved simulation prior, not measured wood friction.',
        scope='CPU fixture geometry only; no native grasp, reach or neighboring-piece clearance qualification')
    return asset
