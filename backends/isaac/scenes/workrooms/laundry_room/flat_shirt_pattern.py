"""Project-authored flat sewn T-shirt pattern, CC0-1.0.

This is a genuine connected front/back surface with four open boundaries.
It is an alternative rest mesh, not a projection of Newton's worn shirt.
"""
from functools import lru_cache

import numpy as np
from scipy.spatial import Delaunay

from .garments import _contains

PROFILE = 'procedural_flat_sewn_shirt_v1'


def _segment_distance(points, a, b):
    delta = b-a
    fraction = np.clip((points-a)@delta/(delta@delta), 0., 1.)
    return np.linalg.norm(points-a-fraction[:, None]*delta, axis=1)


@lru_cache(maxsize=1)
def build():
    # Ordered CCW. Open segments are the hem, both cuffs and the round neck.
    points = [(-.21, -.30), (.21, -.30), (.21, .14), (.30, .075),
              (.35, .19), (.21, .29), (.075, .31)]
    open_edges = {0: 'lower_hem', 3: 'right_cuff'}
    for angle in np.linspace(0., np.pi, 13)[1:]:
        open_edges[len(points)-1] = 'neckline'
        points.append((.075*np.cos(angle), .31-.065*np.sin(angle)))
    points.extend([(-.21, .29), (-.35, .19), (-.30, .075), (-.21, .14)])
    open_edges[len(points)-3] = 'left_cuff'
    polygon = np.asarray(points, dtype=float)
    pitch, half_gap = .012, .0015
    boundary = []
    for a, b in zip(polygon, np.roll(polygon, -1, axis=0)):
        count = int(np.ceil(np.linalg.norm(b-a)/pitch))
        boundary.extend(a+np.arange(count)[:, None]/count*(b-a))
    xs = np.arange(-.35, .35001, pitch)
    ys = np.arange(-.30, .31001, pitch*np.sqrt(3)/2)
    grid = np.array([[x+(i % 2)*pitch/2, y] for i, y in enumerate(ys) for x in xs])
    def boundary_distance(q):
        return np.min([_segment_distance(q, a, b)
                       for a, b in zip(polygon, np.roll(polygon, -1, axis=0))], axis=0)
    grid = grid[_contains(grid, polygon) & (boundary_distance(grid) >= .38*pitch)]
    planar = np.unique(np.round(np.r_[grid, boundary], 10), axis=0)
    tris = Delaunay(planar).simplices
    mids = ((planar[tris]+np.roll(planar[tris], -1, axis=1))/2).reshape(-1, 2)
    inside = _contains(planar[tris].mean(1), polygon)
    inside &= (_contains(mids, polygon) | (boundary_distance(mids) < 1.e-8)).reshape(-1, 3).all(1)
    tris = tris[inside]
    cross = np.cross(np.c_[planar[tris[:, 1]]-planar[tris[:, 0]], np.zeros(len(tris))],
                     np.c_[planar[tris[:, 2]]-planar[tris[:, 0]], np.zeros(len(tris))])[:, 2]
    # Rounded collinear samples can make Qhull emit zero-area boundary slivers.
    # Remove only those numerical triangles before sewing, keeping real cells.
    keep = abs(cross) > 1.e-10
    tris, cross = tris[keep], cross[keep]
    tris[cross < 0] = tris[cross < 0, ::-1]
    used = np.unique(tris)
    remap = np.zeros(len(planar), dtype=int)
    remap[used] = np.arange(len(used))
    planar, tris = planar[used], remap[tris]
    count = len(planar)
    rest = np.r_[np.c_[planar, np.full(count, half_gap)],
                 np.c_[planar, np.full(count, -half_gap)]]
    faces = [*tris.tolist(), *(tris[:, ::-1]+count).tolist()]
    edges = np.sort(np.concatenate([tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]]]), axis=1)
    edges, incidence = np.unique(edges, axis=0, return_counts=True)
    features = {name: set() for name in ('lower_hem', 'neckline', 'left_cuff', 'right_cuff')}
    directed = set(map(tuple, np.concatenate([tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]]]).tolist()))
    for a, b in edges[incidence == 1]:
        mid = (planar[a]+planar[b])/2
        opening = next((name for edge, name in open_edges.items()
                        if _segment_distance(mid[None], polygon[edge], polygon[(edge+1) % len(polygon)])[0] < 1.e-8), None)
        if opening:
            features[opening].update((int(a), int(b), int(a+count), int(b+count)))
            continue
        if (int(a), int(b)) not in directed:
            a, b = b, a
        faces.extend([[int(b), int(a), int(a+count)], [int(b), int(a+count), int(b+count)]])
    provenance = {'kind': 'project_authored_flat_sewn_two_panel_surface', 'profile': PROFILE,
        'license': 'CC0-1.0', 'source_outline_xy_m': polygon.tolist(),
        'nominal_mesh_pitch_m': pitch, 'initial_panel_midplane_gap_m': 2*half_gap,
        'sewn_edges': 'shoulders, sleeve upper/lower sides and torso side seams',
        'openings': ['neckline', 'left_cuff', 'right_cuff', 'lower_hem'],
        'vertices_projected': False, 'derived_from_official_worn_mesh': False,
        'scope': 'Simplified project-authored flat T-shirt tailoring pattern, not a calibrated garment or inherited solver success.'}
    return rest, np.asarray(faces, dtype=np.int32), {k: sorted(v) for k, v in features.items()}, provenance
