"""Single unfolded clothing shells, with separate fold-goal references.

The official Newton shirt retains its complete sewn surface and material
metric.  Reading that USD does not import or run Newton.  The trousers are an
explicit project-authored two-panel shell with open waist and cuffs.  Goal
meshes are evaluator references only; no runtime writes them into cloth state.
"""
from __future__ import annotations

from copy import deepcopy
from functools import lru_cache
import json
import math
from pathlib import Path

import numpy as np
from scipy.spatial import Delaunay, cKDTree

from bench.isaacsim.release import checked, digest, dump
from .garments import _area, _contains, _face_groups, _write_weave
from .flat_shirt_pattern import PROFILE as FLAT_SHIRT_PROFILE
from bench.paths import DATA_ROOT, load_json, undistributed

ASSETS = Path(f'{DATA_ROOT}/assets')
SCHEMA = 'laundry_single_garment_folding_asset.v1'
MANIFEST_SCHEMA = 'laundry_single_garment_folding_asset_manifest.v1'
INPUT_MANIFEST_SCHEMA = 'laundry_single_garment_folding_asset_manifest.v2'
TASK_KINDS = {'shirt_folding': 'shirt', 'trousers_folding': 'trousers'}
SHIRT_PROFILE = 'official_newton_1_6_1_sewn_shirt_v1'
TROUSERS_PROFILE = 'procedural_sewn_trousers_v1'
NEWTON_PACKAGE = ASSETS/'simulators/newton/1.6.1/venv/lib/python3.12/site-packages'
SHIRT_USD = NEWTON_PACKAGE/'newton/examples/assets/unisex_shirt.usd'
SHIRT_SHA256 = '9eb7f161154ce8d3f00253650e78a97d022ebe99b5017656ee965a536d5e64c5'
NEWTON_LICENSE = NEWTON_PACKAGE/'newton-1.6.1.dist-info/licenses/LICENSE.md'
NEWTON_LICENSE_SHA256 = 'a6cba85bc92e0cff7a450b1d873c0eaa2e9fc96bf472df0247a26bec77bf3ff9'
THICKNESS = {'shirt': .0012, 'trousers': .0012}
TABLE_Z = .90


def _edges(triangles):
    t = np.asarray(triangles, dtype=np.int64)
    return np.unique(np.sort(np.concatenate((t[:, [0, 1]], t[:, [1, 2]], t[:, [2, 0]])), axis=1),
                     axis=0, return_counts=True)


def _boundary_loops(triangles):
    edges, counts = _edges(triangles)
    neighbors = {}
    for a, b in edges[counts == 1]:
        neighbors.setdefault(int(a), []).append(int(b))
        neighbors.setdefault(int(b), []).append(int(a))
    if any(len(row) != 2 for row in neighbors.values()) or np.any(counts > 2):
        raise ValueError('Expected manifold cloth with closed boundary loops')
    pending, loops = set(neighbors), []
    while pending:
        first = min(pending)
        ordered, previous, current = [first], None, first
        while True:
            nxt = next(value for value in sorted(neighbors[current]) if value != previous)
            if nxt == first:
                break
            ordered.append(nxt)
            previous, current = current, nxt
        pending.difference_update(ordered)
        loops.append(ordered)
    return loops


@lru_cache(maxsize=1)
def _shirt():
    checked(SHIRT_USD, SHIRT_SHA256)
    checked(NEWTON_LICENSE, NEWTON_LICENSE_SHA256)
    from pxr import Usd, UsdGeom
    stage = Usd.Stage.Open(str(SHIRT_USD))
    prim = stage.GetPrimAtPath('/root/shirt')
    mesh = UsdGeom.Mesh(prim)
    counts = np.asarray(mesh.GetFaceVertexCountsAttr().Get(), dtype=int)
    if not np.all(counts == 3):
        raise ValueError('Pinned Newton shirt unexpectedly stopped being triangular')
    points = np.asarray(mesh.GetPointsAttr().Get(), dtype=float)
    transform = np.asarray(UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(0), dtype=float)
    if not np.array_equal(transform, np.eye(4)):
        raise ValueError('Pinned shirt local transform changed')
    points *= .01  # The pinned source and official example use centimetres.
    points -= (points.min(axis=0)+points.max(axis=0))/2
    triangles = np.asarray(mesh.GetFaceVertexIndicesAttr().Get(), dtype=np.int32).reshape(-1, 3)
    loops = _boundary_loops(triangles)
    if len(loops) != 4:
        raise ValueError('Official shirt must retain hem, neckline and both cuff openings')
    by_y = sorted(loops, key=lambda ids: points[ids, 1].mean())
    cuffs = sorted(by_y[1:3], key=lambda ids: points[ids, 0].mean())
    features = dict(lower_hem=by_y[0], neckline=by_y[-1], left_cuff=cuffs[0], right_cuff=cuffs[1])
    provenance = {'kind': 'official_sewn_garment_surface', 'profile': SHIRT_PROFILE,
        'source_usd': {'path': str(SHIRT_USD), 'sha256': SHIRT_SHA256},
        'source_prim': '/root/shirt', 'source_stage_up_axis': str(UsdGeom.GetStageUpAxis(stage)),
        'source_metres_per_unit': float(UsdGeom.GetStageMetersPerUnit(stage)),
        'applied_uniform_scale': .01, 'applied_rotation': 'identity: source XY body footprint lies on the table, as in the official cloth example',
        'applied_translation': 'subtract source scaled axis-aligned bounding-box centre, then translate into the laundry room',
        'topology_changed': False, 'vertices_projected': False,
        'distribution_license': {'path': str(NEWTON_LICENSE), 'sha256': NEWTON_LICENSE_SHA256,
                                 'spdx': 'Apache-2.0'},
        'source_example': 'newton/examples/cloth/example_cloth_franka.py',
        'scope': 'Retains the official connected front/back shirt shell; no Newton solver or example success is transferred to PhysX.'}
    return points, triangles, features, provenance


def _distance_to_polygon(points, polygon):
    edge = np.roll(polygon, -1, axis=0)-polygon
    rel = points[:, None]-polygon
    along = np.clip(np.einsum('nki,ki->nk', rel, edge)/np.sum(edge*edge, axis=1), 0., 1.)
    return np.linalg.norm(points[:, None]-(polygon[None]+along[:, :, None]*edge), axis=-1).min(axis=1)


@lru_cache(maxsize=1)
def _trousers():
    # 74 cm waist-to-cuff length leaves table margin. The outline is a project
    # pattern, not a scaled claim about a commercial garment size.
    polygon = np.array([[-.19, .37], [-.18, .13], [-.165, -.37], [-.055, -.37],
        [-.035, .015], [0., .08], [.035, .015], [.055, -.37], [.165, -.37],
        [.18, .13], [.19, .37]])[::-1]
    pitch = .012
    boundary = []
    for a, b in zip(polygon, np.roll(polygon, -1, axis=0)):
        boundary.extend(a+np.arange(int(np.ceil(np.linalg.norm(b-a)/pitch)))[:, None]
                        /int(np.ceil(np.linalg.norm(b-a)/pitch))*(b-a))
    xs = np.arange(-.19, .19001, pitch)
    ys = np.arange(-.37, .37001, pitch*np.sqrt(3)/2)
    grid = np.array([[x+(i % 2)*pitch/2, y] for i, y in enumerate(ys) for x in xs])
    grid = grid[_contains(grid, polygon) & (_distance_to_polygon(grid, polygon) >= pitch*.38)]
    planar = np.unique(np.round(np.r_[grid, boundary], 10), axis=0)
    tris = Delaunay(planar).simplices
    centres = planar[tris].mean(axis=1)
    mids = (planar[tris]+np.roll(planar[tris], -1, axis=1))/2
    inside = _contains(centres, polygon)
    inside &= (_contains(mids.reshape(-1, 2), polygon)
               | (_distance_to_polygon(mids.reshape(-1, 2), polygon) < 1.e-8)).reshape(-1, 3).all(axis=1)
    tris = tris[inside]
    normal_z = np.cross(np.c_[planar[tris[:, 1]]-planar[tris[:, 0]], np.zeros(len(tris))],
                        np.c_[planar[tris[:, 2]]-planar[tris[:, 0]], np.zeros(len(tris))])[:, 2]
    tris[normal_z < 0] = tris[normal_z < 0, ::-1]
    used = np.unique(tris)
    remap = np.zeros(len(planar), dtype=int)
    remap[used] = np.arange(len(used))
    planar, tris = planar[used], remap[tris]
    count, half_gap = len(planar), .003
    points = np.r_[np.c_[planar, np.full(count, half_gap)],
                   np.c_[planar, np.full(count, -half_gap)]]
    faces = [*tris.tolist(), *(tris[:, ::-1]+count).tolist()]
    edges, incidence = _edges(tris)
    for a, b in edges[incidence == 1]:
        # Leave the real waist and two cuff openings; sew all side/crotch edges.
        ys = planar[[a, b], 1]
        if np.allclose(ys, .37, atol=1.e-8) or np.allclose(ys, -.37, atol=1.e-8):
            continue
        # Match the panel boundary direction to preserve manifold orientation.
        directed = any((int(a), int(b)) in ((int(t[0]), int(t[1])), (int(t[1]), int(t[2])),
                       (int(t[2]), int(t[0]))) for t in tris if a in t and b in t)
        if not directed:
            a, b = b, a
        faces += [[int(b), int(a), int(a+count)], [int(b), int(a+count), int(b+count)]]
    triangles = np.asarray(faces, dtype=np.int32)
    loops = _boundary_loops(triangles)
    if len(loops) != 3:
        raise ValueError('Sewn trousers must have one waist and two cuff openings')
    waist = max(loops, key=lambda ids: points[ids, 1].mean())
    cuffs = sorted((ids for ids in loops if ids != waist), key=lambda ids: points[ids, 0].mean())
    features = dict(waistband=waist, left_cuff=cuffs[0], right_cuff=cuffs[1],
                    crotch=np.flatnonzero((abs(points[:, 0]) < .045) & (abs(points[:, 1]-.05) < .05)).tolist())
    provenance = {'kind': 'project_authored_sewn_two_panel_surface', 'profile': TROUSERS_PROFILE,
        'license': 'CC0-1.0', 'source_outline_xy_m': polygon.tolist(), 'nominal_mesh_pitch_m': pitch,
        'sewn_edges': 'all side and crotch panel boundaries; waist and cuffs remain open',
        'initial_panel_midplane_gap_m': 2*half_gap, 'vertices_projected': False,
        'scope': 'Simplified trouser shell with connected front/back panels and three real openings; no measured tailoring or material calibration.'}
    return points, triangles, features, provenance


def topology(task_id, mesh_profile=None):
    if task_id not in TASK_KINDS:
        raise ValueError('Unknown single garment folding task')
    allowed = (SHIRT_PROFILE, FLAT_SHIRT_PROFILE) if task_id == 'shirt_folding' else (TROUSERS_PROFILE,)
    if mesh_profile is not None and mesh_profile not in allowed:
        raise ValueError('Mesh profile does not belong to this single garment task')
    if task_id == 'shirt_folding' and mesh_profile == FLAT_SHIRT_PROFILE:
        from .flat_shirt_pattern import build
        q, t, features, provenance = build()
    else:
        q, t, features, provenance = _shirt() if task_id == 'shirt_folding' else _trousers()
    return q.copy(), t.copy(), deepcopy(features), deepcopy(provenance)


def _fold_up(points, axis, hinge, sign, radius, plane_z):
    """Evaluator reference only: finite-radius fold above the support plane."""
    result = points.copy()
    s, z = sign*(points[:, axis]-hinge), points[:, 2]-plane_z
    a = np.pi*radius/2
    theta = np.clip((s+a)/radius, 0., np.pi)
    along = np.where(s <= -a, s, np.where(s >= a, -s, -a+radius*np.sin(theta)))
    result[:, axis] = hinge+sign*(along-z*np.sin(theta))
    result[:, 2] = plane_z+radius*(1-np.cos(theta))+z*np.cos(theta)
    return result


def _rotate_xy(points, quarter_turns, pivot):
    matrices = (np.array([[1., 0.], [0., 1.]]), np.array([[0., -1.], [1., 0.]]),
                np.array([[-1., 0.], [0., -1.]]), np.array([[0., 1.], [-1., 0.]]))
    result = points.copy()
    result[:, :2] = (result[:, :2]-pivot)@matrices[quarter_turns % 4].T+pivot
    return result


def _rotate_degrees(points, degrees, pivot):
    """Rigid Z rotation; exact quarter turns keep the existing arithmetic."""
    quarter = round(degrees/90.)
    if abs(degrees-90.*quarter) < 1.e-12:
        if quarter % 4 == 0:
            return np.asarray(points, dtype=float).copy()
        return _rotate_xy(points, quarter, pivot)
    angle = math.radians(degrees)
    c, s = math.cos(angle), math.sin(angle)
    result = np.asarray(points, dtype=float).copy()
    # Separate scalar ufunc operations avoid a BLAS/FMA-dependent last bit
    # when the frozen native reference is reloaded by Isaac's NumPy build.
    x, y = result[:, 0]-pivot[0], result[:, 1]-pivot[1]
    result[:, 0] = x*c-y*s+pivot[0]
    result[:, 1] = x*s+y*c+pivot[1]
    return result


def transform_fold_frame(points, frame=None, *, inverse=False):
    """Coordinate conversion only, never a physical cloth-state assignment."""
    if frame is None:
        return np.asarray(points, dtype=float).copy()
    if (set(frame) != {'schema', 'yaw_degrees', 'pivot_xy_m'}
            or frame['schema'] != 'laundry_fold_coordinate_frame.v1'):
        raise ValueError('Explicit garment fold coordinate frame required')
    return _rotate_degrees(np.asarray(points, dtype=float),
        (-1. if inverse else 1.)*frame['yaw_degrees'], np.asarray(frame['pivot_xy_m']))


def _rotate_fold(fold, quarter_turns, pivot):
    axis, hinge, sign, radius = fold
    vector = np.zeros((1, 3))
    vector[0, axis] = 1.
    normal = _rotate_xy(vector, quarter_turns, np.zeros(2))[0, :2]
    new_axis = int(np.argmax(abs(normal)))
    factor = float(normal[new_axis])
    return new_axis, float(pivot[new_axis]+factor*(hinge-pivot[axis])), int(sign*factor), radius


def _reference(task_id, initial, thickness, features, unfolded_reference_path=None,
               yaw_quarter_turns=0, orientation_pivot_xy=None, triangles=None, fold_frame=None):
    source = None
    if unfolded_reference_path is None:
        flat = initial.copy()
        # This deliberately unqualified reference is only sufficient to create
        # a settling experiment. The native shell itself is never projected.
        low, high = initial[:, 2].min(), initial[:, 2].max()
        flat[:, 2] = TABLE_Z+thickness/2+thickness*(initial[:, 2]-low)/(high-low)
        status = 'provisional_projection_not_qualified'
    else:
        path = Path(unfolded_reference_path).resolve()
        record = load_json(path)
        from .initial_state_assets import REFERENCE_SCHEMA
        if record.get('schema') not in ('laundry_unfolded_native_reference.v1', REFERENCE_SCHEMA) or record.get('task_id') != task_id:
            raise ValueError('Wrong unfolded native reference')
        flat = np.asarray(record['vertices_m'], dtype=float)
        if flat.shape != initial.shape or not np.isfinite(flat).all():
            raise ValueError('Incomplete finite native unfolded surface required')
        source_key = 'input_source' if record['schema'] == REFERENCE_SCHEMA else 'native_source'
        if source_key == 'input_source' and set(record) != {'schema', 'task_id', 'vertices_m', 'initial_state_input'}:
            raise ValueError('Published input must not contain experiment reports or outcome fields')
        source = {'path': str(path), 'sha256': digest(path), source_key:
                  deepcopy(record['initial_state_input'] if source_key == 'input_source' else record['native_source'])}
        if not isinstance(source[source_key], dict) or not source[source_key]:
            raise ValueError('Native unfolded reference requires source provenance')
        status = 'native_unfolded_reference_fold_target'
    pivot = np.asarray(orientation_pivot_xy, dtype=float)
    canonical = (transform_fold_frame(flat, fold_frame, inverse=True) if fold_frame is not None
                 else _rotate_xy(flat, -yaw_quarter_turns, pivot))
    centre = (canonical[:, :2].min(axis=0)+canonical[:, :2].max(axis=0))/2
    if task_id == 'shirt_folding':
        hem = canonical[features['lower_hem']]
        neck = canonical[features['neckline']]
        # Fold just the sleeves, not the entire long torso side strips. Their
        # true flat width comes from the sewn hem, never a worn-shell shadow.
        right_hinge = float(np.quantile(hem[:, 0], .98)+.005)
        left_hinge = float(np.quantile(hem[:, 0], .02)-.005)
        transverse_hinge = float((np.median(hem[:, 1])+np.median(neck[:, 1]))/2)
        folds = [(0, right_hinge, 1, .007),
                 (0, left_hinge, -1, .007), (1, transverse_hinge, -1, .010)]
    else:
        folds = [(0, float(centre[0]), -1, .007),
                 (1, float(centre[1]), -1, .012)]
    if fold_frame is None:
        folds = [_rotate_fold(f, yaw_quarter_turns, pivot) for f in folds]
    folded = _fold_reference_positions(flat, folds, thickness,
        topology(task_id)[1] if triangles is None else triangles, fold_frame=fold_frame)
    target = (folded[:, :2].min(axis=0)+folded[:, :2].max(axis=0))/2
    return flat, folded, folds, target, status, source


def _vertex_normals(points, triangles):
    normals = np.zeros_like(points)
    face_normals = np.cross(points[triangles[:, 1]]-points[triangles[:, 0]],
                            points[triangles[:, 2]]-points[triangles[:, 0]])
    for k in range(3):
        np.add.at(normals, triangles[:, k], face_normals)
    return normals/np.maximum(np.linalg.norm(normals, axis=1)[:, None], 1.e-12)


def _fold_reference_positions(flat, folds, thickness, triangles, *, return_normals=False, fold_frame=None):
    if fold_frame is not None:
        canonical = transform_fold_frame(flat, fold_frame, inverse=True)
        folded, normals = _fold_reference_positions(canonical, folds, thickness, triangles, return_normals=True)
        folded = transform_fold_frame(folded, fold_frame)
        normals = _rotate_degrees(normals, fold_frame['yaw_degrees'], np.zeros(2))
        return (folded, normals) if return_normals else folded
    folded = flat.copy()
    # Contact envelopes in a reference engine can leave millimetre-to-centimetre
    # air gaps. Compact only the *evaluator target* vertically before folding;
    # the physical reset/reference vertices above remain the exact native q.
    low, high = float(flat[:, 2].min()), float(flat[:, 2].max())
    plane = TABLE_Z+thickness/2
    # Compress the mid-surface wrinkle envelope while separately retaining
    # nominal front/back wall clearance. Uniform Z scaling would turn a real
    # 1.2 mm shirt wall gap into 0.1 mm when a 7 cm wrinkle is flattened.
    envelope = min(fold[3] for fold in folds)
    scale = min(1., (envelope-thickness)/max(high-low, 1.e-12))
    upper_wall = np.clip(.5+2*_vertex_normals(flat, triangles)[:, 2], 0., 1.)
    folded[:, 2] = plane+scale*(flat[:, 2]-low)+(1-scale)*thickness*upper_wall
    normals = _vertex_normals(folded, triangles)
    for axis, hinge, sign, radius in folds:
        s = sign*(folded[:, axis]-hinge)
        theta = np.clip((s+np.pi*radius/2)/radius, 0., np.pi)
        old_axis, old_z = normals[:, axis].copy(), normals[:, 2].copy()
        normals[:, axis] = old_axis*np.cos(theta)-sign*old_z*np.sin(theta)
        normals[:, 2] = sign*old_axis*np.sin(theta)+old_z*np.cos(theta)
        folded = _fold_up(folded, axis, hinge, sign, radius, plane)
    return (folded, normals) if return_normals else folded


def _fold_layer_pairs(unfolded, folded, thickness):
    """Pairs created by a fold, excluding a garment's natural front/back wall."""
    tree = cKDTree(folded[:, :2])
    result = []
    for upper, neighbors in enumerate(tree.query_ball_point(folded[:, :2], .008)):
        lower = [i for i in neighbors if folded[upper, 2]-folded[i, 2] >= thickness*.8
                 and np.linalg.norm(unfolded[upper, :2]-unfolded[i, :2]) >= .060]
        if lower:
            i = min(lower, key=lambda i: (np.linalg.norm(folded[upper, :2]-folded[i, :2]),
                                          folded[upper, 2]-folded[i, 2], i))
            result.append([int(i), upper])
    return result


def _shell_layer_pairs(task_id, rest, triangles, unfolded, folded, thickness, table_facing=None):
    """Natural garment wall pairs, independently of fold-created overlap.

    The scalar clearance is directed Z, never 3-D distance: a mismatched XY
    sample must not disguise two collapsed material surfaces as thick cloth.
    """
    half = len(rest)//2
    authored_panels = len(rest) % 2 == 0 and np.allclose(rest[:half, :2], rest[half:, :2], atol=1.e-10, rtol=0.)
    if authored_panels:
        candidates = [(i, i+half) for i in range(half)]
    else:
        normals = _vertex_normals(unfolded, triangles)
        candidates = []
        for i, neighbors in enumerate(cKDTree(unfolded[:, :2]).query_ball_point(unfolded[:, :2], .008)):
            opposite = [j for j in neighbors if j > i
                and np.linalg.norm(rest[i]-rest[j]) > .025
                and normals[i, 2]*normals[j, 2] < -.3]
            if opposite:
                j = min(opposite, key=lambda j: (np.linalg.norm(unfolded[i, :2]-unfolded[j, :2]), j))
                candidates.append((i, j))
    result = []
    if table_facing is None:
        table_facing = abs(_vertex_normals(folded, triangles)[:, 2]) >= .5
    for i, j in candidates:
        gap = folded[j, 2]-folded[i, 2]
        # Sample table-facing surface patches, excluding vertical seam/fold
        # walls by normal direction, never by whether their gap would pass.
        if (table_facing[i] and table_facing[j]
                and np.linalg.norm(folded[i, :2]-folded[j, :2]) <= .010):
            result.append([i, j] if gap > 0 else [j, i])
    return result


def _goal_row(task_id, rest, triangles, flat, folded, thickness, folds, fold_frame=None):
    _, transported_normals = _fold_reference_positions(flat, folds, thickness, triangles,
                                                       return_normals=True, fold_frame=fold_frame)
    return {'folded_reference_vertices_m': folded.tolist(),
        'folded_layer_pairs': _fold_layer_pairs(flat, folded, thickness),
        'shell_layer_pairs': _shell_layer_pairs(task_id, rest, triangles, flat, folded, thickness,
                                               abs(transported_normals[:, 2]) >= .5),
        'support_bottom_vertex_indices': np.flatnonzero(folded[:, 2] <= folded[:, 2].min()+.0004).tolist()}


def _equivalent_goal_variants(task_id, rest, triangles, flat, folded, folds, thickness, target, fold_frame=None):
    alternative = list(folds)
    if task_id == 'trousers_folding':
        axis, hinge, sign, radius = alternative[0]
        alternative[0] = (axis, hinge, -sign, radius)
        variant_id = 'right_leg_over_left_then_cuffs_to_waist'
    else:
        alternative[:2] = alternative[1::-1]
        variant_id = 'left_sleeve_then_right_sleeve_then_hem'
    points = _fold_reference_positions(flat, alternative, thickness, triangles, fold_frame=fold_frame)
    shift = target-(points[:, :2].min(axis=0)+points[:, :2].max(axis=0))/2
    points[:, :2] += shift
    if np.allclose(points, folded, atol=1.e-8, rtol=0.):
        return []
    row = _goal_row(task_id, rest, triangles, flat, points, thickness, alternative, fold_frame)
    row.update(id=variant_id)
    return [row]


def _recording_shipped(report_path):
    """Whether a native settle recording (report and state arrays) is part of the data package."""
    if undistributed(report_path):
        return False
    files = load_json(Path(report_path))['states']['files']
    return not any(undistributed(item['path']) for item in files.values())


def _native_settle_state(task_id, report_path, translation_m, yaw_quarter_turns=0,
                         rotation_pivot_xy_m=None, mesh_profile=None, yaw_degrees=None):
    """Read and verify a completed passive PhysX stream, without state setters."""
    report_path = Path(report_path).resolve()
    report = load_json(report_path)
    if (report.get('schema') != 'laundry_single_garment_native.v1'
            or not report.get('complete') or report.get('mode') != 'settle'
            or report.get('cloth_position_velocity_writes') != 0
            or report.get('cloth_id') != 'towel'):
        raise ValueError('A complete passive native settle with zero cloth state writes is required')
    stream = report['states']
    if not stream.get('complete') or stream['valid_samples'] < 2:
        raise ValueError('Incomplete native settle state stream')
    arrays = {}
    for name in ('cloth_q', 'cloth_qd', 'control_mask', 'time_s'):
        item = stream['files'][name]
        path = Path(item['path'])
        checked(path, item['sha256'])
        arrays[name] = np.load(path, mmap_mode='r', allow_pickle=False)
        if list(arrays[name].shape) != item['shape']:
            raise ValueError('Native state stream shape differs from report')
    end = int(stream['valid_samples'])-1
    q = np.asarray(arrays['cloth_q'][end], dtype=float)
    rest, triangles, _, _ = topology(task_id, mesh_profile)
    if q.shape != rest.shape or not np.isfinite(q).all():
        raise ValueError('Native settled state does not match the garment topology')
    if np.any(arrays['control_mask'][:end+1]):
        raise ValueError('Passive reset source cannot contain controlled cloth points')
    shift = np.asarray(translation_m, dtype=float)
    if shift.shape != (3,) or not np.isfinite(shift).all():
        raise ValueError('Native reset needs a finite explicit rigid translation')
    times = np.asarray(arrays['time_s'][:end+1], dtype=float)
    if not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError('Native settle timestamps must be finite and strictly increasing')
    first = int(np.searchsorted(times, times[-1]-.5))
    velocity = np.linalg.norm(np.asarray(arrays['cloth_qd'][end], dtype=float), axis=1)
    if not np.isfinite(velocity).all():
        raise ValueError('Native settled velocity must be finite')
    drift = np.linalg.norm(q-np.asarray(arrays['cloth_q'][first]), axis=1)
    edges, _ = _edges(triangles)
    ratios = (np.linalg.norm(q[edges[:, 0]]-q[edges[:, 1]], axis=1)
              /np.linalg.norm(rest[edges[:, 0]]-rest[edges[:, 1]], axis=1))
    metrics = {'vertical_extent_m': float(np.ptp(q[:, 2])),
        'speed_p95_m_s': float(np.quantile(velocity, .95)),
        'final_half_second_drift_p95_m': float(np.quantile(drift, .95)),
        'final_window_s': float(times[-1]-times[first]),
        'edge_ratio_min': float(ratios.min()), 'edge_ratio_max': float(ratios.max())}
    if (metrics['vertical_extent_m'] > .08 or metrics['speed_p95_m_s'] > .08
            or metrics['final_half_second_drift_p95_m'] > .015
            or metrics['final_window_s'] < .49
            or metrics['edge_ratio_min'] < .8 or metrics['edge_ratio_max'] > 1.2):
        raise ValueError('Native source has not reached a stable unfolded state: '+str(metrics))
    source = {'engine': 'physx', 'mode': 'passive_settle',
        'native_report': {'path': str(report_path), 'sha256': digest(report_path)},
        'cloth_state': deepcopy(stream['files']['cloth_q']), 'sample_index': end,
        'sample_time_s': float(times[-1]), 'translation_m': shift.tolist(),
        'rotation_wxyz': [1., 0., 0., 0.], 'settle_metrics': metrics,
        'cloth_position_velocity_writes': 0, 'controlled_cloth_vertices': 0,
        'operation_credit': False, 'rest_geometry_or_topology_modified': False}
    if mesh_profile is not None:
        source['mesh_profile'] = mesh_profile
    # A native run may already have been authored in a rotated layout. Read
    # that orientation through the report -> scene -> asset-metadata hash
    # chain, separately from any *additional* export transform below.
    input_orientation = None
    base_degrees = 0.
    scene_pin = report.get('scene_input') or report.get('inputs', {}).get('scene.json')
    if scene_pin:
        scene_path = Path(scene_pin['path'])
        checked(scene_path, scene_pin['sha256'])
        scene = load_json(scene_path)
        candidate = scene.get('laundry_task_provenance', {}).get('asset_candidate', {})
        if candidate:
            metadata_path = Path(candidate['manifest_path']).parent/'metadata.json'
            expected = candidate['files_sha256'][str(metadata_path)]
            checked(metadata_path, expected)
            input_meta = load_json(metadata_path)
            base_yaw = input_meta.get('source_yaw_quarter_turns', 0)
            if type(base_yaw) is not int or base_yaw not in (0, 1, 2, 3):
                raise ValueError('Native source asset has an invalid authored orientation')
            authored_degrees = input_meta.get('source_yaw_degrees')
            if authored_degrees is not None and (base_yaw != 0
                    or type(authored_degrees) not in (int, float) or not math.isfinite(authored_degrees)
                    or not 0. <= authored_degrees < 360.):
                raise ValueError('Native source asset has an invalid general orientation')
            base_degrees = float(base_yaw*90 if authored_degrees is None else authored_degrees)
            if base_yaw or authored_degrees is not None:
                input_orientation = {'scene': deepcopy(scene_pin),
                    'asset_metadata': {'path': str(metadata_path), 'sha256': expected},
                    'source_yaw_quarter_turns': base_yaw,
                    'source_center_xy_m': deepcopy(input_meta['source_center_xy_m'])}
                if authored_degrees is not None:
                    input_orientation['source_yaw_degrees'] = float(authored_degrees)
    if type(yaw_quarter_turns) is not int or yaw_quarter_turns not in (0, 1, 2, 3):
        raise ValueError('Native reset rotation must be an explicit 0/1/2/3 quarter turn')
    if yaw_degrees is not None and (yaw_quarter_turns != 0
            or type(yaw_degrees) not in (int, float) or not math.isfinite(yaw_degrees)
            or not -360. <= yaw_degrees <= 360.):
        raise ValueError('General native rotation must be finite degrees, without a second quarter rotation')
    if yaw_quarter_turns or rotation_pivot_xy_m is not None or yaw_degrees is not None:
        pivot = np.asarray(rotation_pivot_xy_m if rotation_pivot_xy_m is not None
                           else (q[:, :2].min(0)+q[:, :2].max(0))/2, dtype=float)
        if pivot.shape != (2,) or not np.isfinite(pivot).all():
            raise ValueError('A finite XY native reset rotation pivot is required')
        if yaw_degrees is None:
            q = _rotate_xy(q, yaw_quarter_turns, pivot)
            angle = yaw_quarter_turns*np.pi/2
            source.update(yaw_quarter_turns=yaw_quarter_turns, rotation_pivot_xy_m=pivot.tolist(),
                          rotation_wxyz=[float(np.cos(angle/2)), 0., 0., float(np.sin(angle/2))])
        else:
            q = _rotate_degrees(q, yaw_degrees, pivot)
            angle = math.radians(yaw_degrees)
            source.update(yaw_degrees=float(yaw_degrees), rotation_pivot_xy_m=pivot.tolist(),
                          rotation_wxyz=[math.cos(angle/2), 0., 0., math.sin(angle/2)])
    if input_orientation:
        source['input_orientation'] = input_orientation
        if yaw_degrees is None and 'source_yaw_degrees' not in input_orientation:
            source['result_yaw_quarter_turns'] = (input_orientation['source_yaw_quarter_turns']+yaw_quarter_turns) % 4
    if yaw_degrees is not None or (input_orientation and 'source_yaw_degrees' in input_orientation):
        source['result_yaw_degrees'] = (base_degrees+(yaw_quarter_turns*90. if yaw_degrees is None else yaw_degrees)) % 360.
    return q+shift, source


def _same_native_source(observed, pinned):
    """Strict provenance with only numerical noise in recomputed diagnostics.

    NumPy's float32 percentile interpolation changed between supported host
    and Isaac environments. This tolerance cannot alter any source hash, pose,
    flag or physics acceptance gate; raw q is still checked bit-for-bit.
    """
    observed = deepcopy(observed)
    pinned = deepcopy(pinned)
    a, b = observed.pop('settle_metrics', None), pinned.pop('settle_metrics', None)
    if observed != pinned or not isinstance(a, dict) or not isinstance(b, dict) or a.keys() != b.keys():
        return False
    return all(type(a[key]) in (int, float) and type(b[key]) in (int, float)
               and np.isfinite(a[key]) and np.isfinite(b[key])
               and abs(a[key]-b[key]) <= 1.e-8 for key in a)


def _bind_native_material_profile(source, profile_id, task_id):
    """Reject a thicker reset inherited from physics run with thinner cloth."""
    from .material_profiles import material_profile as get_material
    profile = get_material(profile_id, TASK_KINDS[task_id])
    report = load_json(_checked_path(source['native_report']))
    scene_pin = report.get('scene_input') or report.get('inputs', {}).get('scene.json')
    if not scene_pin:
        raise ValueError('New material reset requires its native source scene identity')
    scene = load_json(_checked_path(scene_pin))
    candidate = scene.get('laundry_task_provenance', {}).get('asset_candidate') or {}
    if not candidate:
        raise ValueError('New material reset requires a frozen source asset')
    metadata_path = Path(candidate['manifest_path']).parent/'metadata.json'
    metadata_pin = {'path': str(metadata_path),
                    'sha256': candidate['files_sha256'][str(metadata_path)]}
    metadata = load_json(_checked_path(metadata_pin))
    rows = metadata.get('cloths', [])
    authored = scene.get('cloths', [])
    if (metadata.get('task_id') != task_id or metadata.get('material_profile') != profile_id
            or len(rows) != 1 or len(authored) != 1
            or rows[0]['physical_thickness_m'] != profile['physical_thickness_m']
            or rows[0]['mass_kg'] != profile['mass_kg']
            or authored[0]['mass_kg'] != profile['mass_kg']
            or authored[0]['isaacsim']['cloth_thickness_m'] != profile['physical_thickness_m']
            or any({**scene['isaacsim'], **authored[0]['isaacsim']}.get(key) != value
                   for key, value in metadata['physics_options'].items())):
        raise ValueError('Native settled reset was not simulated with the declared thicker material')
    return {**deepcopy(source), 'material_profile': profile_id,
            'material_source': {'scene': deepcopy(scene_pin), 'asset_metadata': metadata_pin,
                'physical_thickness_m': profile['physical_thickness_m'], 'mass_kg': profile['mass_kg'],
                'physics_options': deepcopy(metadata['physics_options'])}}


def _checked_path(pin):
    path = Path(pin['path'])
    checked(path, pin['sha256'])
    return path


def export_native_settled_reference(task_id, native_report_path, output, *, translation_m=(0., 0., 0.),
                                    yaw_quarter_turns=0, rotation_pivot_xy_m=None, mesh_profile=None,
                                    yaw_degrees=None, material_profile=None):
    """Freeze an actual passive settled frame as an explicitly named reset prior."""
    q, source = _native_settle_state(task_id, native_report_path, translation_m,
                                    yaw_quarter_turns, rotation_pivot_xy_m, mesh_profile, yaw_degrees)
    if material_profile is not None:
        source = _bind_native_material_profile(source, material_profile, task_id)
    output = Path(output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    record = {'schema': 'laundry_unfolded_native_reference.v1', 'task_id': task_id,
        'vertices_m': q.tolist(), 'native_source': source}
    output.write_text(json.dumps(record, indent=2)+'\n')
    return record


def candidate_metadata(task_id, *, texture_root=None, source_center_xy_m=None,
                       mesh_profile=None, unfolded_reference_path=None, physics_dt_s=.001,
                       physical_thickness_m=None, reset_profile='source_shell_gravity_settle_v1',
                       cloth_surface_bend_stiffness=100., solver_iterations=64,
                       source_yaw_quarter_turns=0, cloth_dynamic_friction=None,
                       source_yaw_degrees=None, material_profile=None):
    if task_id not in TASK_KINDS:
        raise ValueError('Unknown single garment folding task')
    kind = TASK_KINDS[task_id]
    profile = None
    if material_profile is not None:
        from .material_profiles import material_profile as get_material
        profile = get_material(material_profile, kind)
        if physical_thickness_m is not None and physical_thickness_m != profile['physical_thickness_m']:
            raise ValueError('Explicit thickness differs from the versioned material profile')
        physical_thickness_m = profile['physical_thickness_m']
    default_profile = SHIRT_PROFILE if kind == 'shirt' else TROUSERS_PROFILE
    allowed_profiles = (SHIRT_PROFILE, FLAT_SHIRT_PROFILE) if kind == 'shirt' else (TROUSERS_PROFILE,)
    if mesh_profile is not None and mesh_profile not in allowed_profiles:
        raise ValueError('Wrong explicitly versioned folding mesh profile')
    expected_profile = mesh_profile or default_profile
    rest, triangles, features, provenance = topology(task_id, expected_profile)
    centre = np.asarray((.65, -.55) if source_center_xy_m is None else source_center_xy_m, dtype=float)
    if centre.shape != (2,) or not np.isfinite(centre).all() or physics_dt_s not in (.002, .001, .0005):
        raise ValueError('Finite source XY and supported explicit physics step required')
    if (type(solver_iterations) is not int or solver_iterations not in (32, 64)
            or not np.isfinite(cloth_surface_bend_stiffness)
            or not 0. <= cloth_surface_bend_stiffness <= 1000.):
        raise ValueError('Explicit 32/64 solver iterations and finite 0..1000 bend stiffness required')
    if type(source_yaw_quarter_turns) is not int or source_yaw_quarter_turns not in (0, 1, 2, 3):
        raise ValueError('Source orientation must explicitly use 0/1/2/3 quarter turns')
    if source_yaw_degrees is not None and (source_yaw_quarter_turns != 0
            or type(source_yaw_degrees) not in (int, float) or not math.isfinite(source_yaw_degrees)
            or not 0. <= source_yaw_degrees < 360.):
        raise ValueError('General source orientation requires 0..360 degrees without a second quarter rotation')
    fold_frame = (None if source_yaw_degrees is None else
        {'schema': 'laundry_fold_coordinate_frame.v1', 'yaw_degrees': float(source_yaw_degrees),
         'pivot_xy_m': centre.tolist()})
    if cloth_dynamic_friction is not None and (not np.isfinite(cloth_dynamic_friction)
            or not 0. <= cloth_dynamic_friction <= 1.):
        raise ValueError('Explicit cloth dynamic friction must be finite and between zero and one')
    thickness = THICKNESS[kind] if physical_thickness_m is None else float(physical_thickness_m)
    if not np.isfinite(thickness) or not .0005 <= thickness <= .002:
        raise ValueError('Declare a finite clothing material thickness between 0.5 and 2 mm')
    initial = rest+np.r_[centre, TABLE_Z+.03-rest[:, 2].min()]
    initial = (transform_fold_frame(initial, fold_frame) if fold_frame is not None
               else _rotate_xy(initial, source_yaw_quarter_turns, centre))
    flat, folded, folds, target, status, reference_source = _reference(
        task_id, initial, thickness, features, unfolded_reference_path,
        source_yaw_quarter_turns, centre, triangles, fold_frame)
    native_reset_source = None
    from .initial_state_assets import RESET_PROFILE
    published_reset = reset_profile == RESET_PROFILE
    if reset_profile == 'physx_native_settled_unfolded_v1':
        native = (reference_source or {}).get('native_source', {})
        if native.get('engine') != 'physx' or native.get('mode') != 'passive_settle':
            raise ValueError('Native settled reset requires an exported passive PhysX reference')
        expected_yaw = source_yaw_quarter_turns*90. if source_yaw_degrees is None else source_yaw_degrees
        observed_yaw = native.get('result_yaw_degrees',
            90.*native.get('result_yaw_quarter_turns', native.get('yaw_quarter_turns', 0)))
        if abs(observed_yaw-expected_yaw) > 1.e-10:
            raise ValueError('Explicit reset source orientation differs from native reference transform')
        if native.get('mesh_profile', default_profile) != expected_profile:
            raise ValueError('Native settled reset mesh profile differs from the declared rest shell')
        if not _recording_shipped(native['native_report']['path']):
            # The native recording is provenance outside the data package; the
            # pinned reference already holds the observed settled state.
            observed = flat
        else:
            checked(Path(native['native_report']['path']), native['native_report']['sha256'])
            observed, native_reset_source = _native_settle_state(
                task_id, native['native_report']['path'], native['translation_m'],
                native.get('yaw_quarter_turns', 0), native.get('rotation_pivot_xy_m'), native.get('mesh_profile'),
                native.get('yaw_degrees'))
            if profile:
                native_reset_source = _bind_native_material_profile(native_reset_source, material_profile, task_id)
            if not _same_native_source(native_reset_source, native) or not np.array_equal(observed, flat):
                raise ValueError('Native reset/reference differs from its pinned observed state')
        # Preserve the exact pinned diagnostics after independently checking
        # the physical limits above, so release metadata stays reproducible.
        native_reset_source = deepcopy(native)
        initial = observed
    elif published_reset:
        if not reference_source or 'input_source' not in reference_source:
            raise ValueError('Published settled reset requires its independent physical input')
        initial = flat.copy()
    elif reset_profile != 'source_shell_gravity_settle_v1':
        raise ValueError('Unknown explicitly versioned single garment reset profile')
    color = [.19, .39, .40] if kind == 'shirt' else [.12, .18, .29]
    texture_paths = ({role: str(Path(texture_root)/name) for role, name in
        [('normal', 'woven_normal.png'), ('roughness', 'woven_roughness.png')]} if texture_root else {})
    area = float(_area(rest, triangles).sum())
    mass = profile['mass_kg'] if profile else (.18 if kind == 'shirt' else .26)
    physics = {'physics_dt_s': float(physics_dt_s), 'cloth_iterations': solver_iterations,
        'cloth_linear_damping': 2., 'cloth_thickness_m': thickness,
        'cloth_youngs_modulus_pa': 667000., 'cloth_poissons_ratio': .3,
        'cloth_surface_bend_stiffness': float(cloth_surface_bend_stiffness), 'cloth_collision_pair_update_frequency': 8,
        'cloth_collision_iteration_multiplier': 8, 'cloth_speculative_ccd': True,
        'cloth_self_collision_filter_distance_m': thickness,
        'cloth_contact_offset_m': thickness+.0002, 'cloth_rest_offset_m': thickness/2}
    if published_reset:
        from .initial_state_assets import validate_source
        validate_source(reference_source['input_source'], task_id=task_id, mesh_profile=expected_profile,
            rest=rest, triangles=triangles, material_prior=profile, physics_options=physics,
            centre=centre, quarter_turns=source_yaw_quarter_turns, yaw_degrees=source_yaw_degrees,
            vertices=flat)
    goal = _goal_row(task_id, rest, triangles, flat, folded, thickness, folds, fold_frame)
    row = {'id': 'towel', 'mesh_type': 'custom_surface', 'garment_kind': kind,
        'public_label_zh': '展开T恤' if kind == 'shirt' else '展开长裤',
        'size_m': np.ptp(rest[:, :2], axis=0).tolist(), 'rest_area_m2': area,
        'mass_kg': mass, 'areal_density_kg_m2': mass/area,
        'physical_thickness_m': thickness, 'rest_vertices_m': rest.tolist(),
        'triangles': triangles.tolist(), 'initial_vertices_m': initial.tolist(),
        'unfolded_reference_vertices_m': flat.tolist(), **goal,
        'folded_reference_variants': _equivalent_goal_variants(
            task_id, rest, triangles, flat, folded, folds, thickness, target, fold_frame),
        'folded_layer_pair_basis': 'Directed folded-target neighbors, at least 60 mm apart in unfolded native XY; natural front/back garment wall pairs are excluded.',
        'shell_layer_pair_basis': 'Natural opposing garment walls: exact authored panel correspondence for trousers; unfolded XY-near, normal-opposed, rest-space-distant samples for the shirt. Final table-facing normals select clearance samples; no minimum-gap filter selects only passing samples. Directed target Z clearance is scored independently of folding layers.',
        'boundary_features': features, 'visual_face_groups': _face_groups(kind, rest, triangles, color, features),
        'visual_texture_paths': texture_paths, 'visual_texture_tile_size_m': [.032, .032],
        'display_color': color, 'dynamic_model': 'native_surface_deformable', 'fixed_vertices': [],
        'material_basis': 'Explicit uncalibrated clothing simulation prior; original connected shell is free after reset.'}
    if cloth_dynamic_friction is not None:
        # The native material builder reads the per-cloth friction field,
        # not a solver option. Omission preserves historical assets exactly.
        # PhysX ignores the static-friction property of surface deformables.
        row['friction'] = float(cloth_dynamic_friction)
    metadata = {'schema': SCHEMA, 'task_id': task_id, 'garment_kind': kind, 'cloths': [row],
        'mesh_profile': expected_profile,
        'initial_state_kind': ('published_settled_unfolded_shell' if published_reset else
                               'native_settled_unfolded_shell' if native_reset_source else 'unfolded_shell'),
        'reset_profile': reset_profile, 'native_settled_source': native_reset_source,
        'source_yaw_quarter_turns': source_yaw_quarter_turns,
        'source_center_xy_m': centre.tolist(), 'target_center_xy_m': target.tolist(),
        'table_top_z_m': TABLE_Z, 'clear_rectangle_xy_m': [-.10, 1.40, -.94, -.16],
        'physics_options': physics, 'physical_thickness_m': thickness, 'goal_reference_status': status,
        'unfolded_reference_path': str(Path(unfolded_reference_path).resolve()) if unfolded_reference_path else None,
        'unfolded_reference_source': reference_source,
        'procedural_texture_root': str(Path(texture_root)) if texture_root else None,
        'design': {'representation': provenance, 'fold_sequence_axis_hinge_sign_radius_m': [list(f) for f in folds],
            'equivalent_reference_policy': 'Either trouser leg may fold over the other; either sleeve order is allowed. Additional targets are rigidly translated in XY to the primary target bounding-box centre, each with independent fold/shell/support samples. No task physics state is changed.',
            'initial_height_basis': ('Pinned native passive settled q plus declared rigid translation; original rest geometry retained; reset earns no fold credit.'
                if native_reset_source or published_reset else 'Minimum original shell point 30 mm above the actual table; gravity settling earns no fold credit.'),
            'goal_vertical_compaction': 'Evaluator target only: retain actual native XY/material indices; compress the wrinkle mid-surface envelope while independently retaining nominal front/back wall clearance using the sewn surface normal. No uniform Z squeeze of material thickness and no physical reset write.',
            'goal_only': 'Analytic fold reference is never written into the simulator; provisional projection is blocked from task success.'},
        'thickness_semantics': 'Native and read-only visual material thickness agree; initial shell depth and later layer gaps are separate.',
        'material_calibrated': False, 'native_physics_accepted': False,
        'ownership': 'Task-owned single garment; original fixed room and historical stacking assets remain unchanged.'}
    if fold_frame is not None:
        metadata['source_yaw_degrees'] = float(source_yaw_degrees)
        metadata['design']['fold_coordinate_frame'] = fold_frame
    if profile:
        metadata['material_profile'] = material_profile
        metadata['material_prior'] = profile
    if published_reset:
        metadata['initial_state_input'] = deepcopy(reference_source['input_source'])
    return metadata


def validate_metadata(metadata):
    expected = candidate_metadata(metadata.get('task_id'), texture_root=metadata.get('procedural_texture_root'),
        source_center_xy_m=metadata.get('source_center_xy_m'), mesh_profile=metadata.get('mesh_profile'),
        unfolded_reference_path=metadata.get('unfolded_reference_path'),
        physics_dt_s=metadata.get('physics_options', {}).get('physics_dt_s'),
        physical_thickness_m=metadata.get('physical_thickness_m'),
        reset_profile=metadata.get('reset_profile', 'source_shell_gravity_settle_v1'),
        cloth_surface_bend_stiffness=metadata.get('physics_options', {}).get('cloth_surface_bend_stiffness'),
        solver_iterations=metadata.get('physics_options', {}).get('cloth_iterations'),
        source_yaw_quarter_turns=metadata.get('source_yaw_quarter_turns', 0),
        cloth_dynamic_friction=metadata.get('cloths', [{}])[0].get('friction'),
        source_yaw_degrees=metadata.get('source_yaw_degrees'),
        material_profile=metadata.get('material_profile'))
    if metadata.get('schema') != SCHEMA or metadata != expected:
        raise ValueError('Single garment asset differs from its pinned shell and reference design')
    return metadata


def geometry_report(metadata):
    row = metadata['cloths'][0]
    q, t = np.asarray(row['rest_vertices_m']), np.asarray(row['triangles'])
    edges, counts = _edges(t)
    lengths, areas = np.linalg.norm(q[edges[:, 0]]-q[edges[:, 1]], axis=1), _area(q, t)
    return {'schema': 'laundry_single_garment_geometry.v1', 'task_id': metadata['task_id'],
        'vertices': len(q), 'triangles': len(t), 'boundary_loops': len(_boundary_loops(t)),
        'nonmanifold_edges': int(np.count_nonzero(counts > 2)),
        'min_edge_m': float(lengths.min()), 'median_edge_m': float(np.median(lengths)),
        'max_edge_m': float(lengths.max()), 'min_triangle_area_m2': float(areas.min()),
        'finite': bool(np.isfinite(q).all()), 'goal_reference_status': metadata['goal_reference_status'],
        'self_intersections_checked': False, 'native_validation_performed': False,
        'scope': 'Topology and finite rest geometry only; native contact, settling and folding remain unqualified.'}


def prepare_assets(task_id, output, **options):
    output = Path(output).resolve()
    if not output.is_relative_to(ASSETS) or output.exists():
        raise ValueError('Choose a fresh asset version under shared assets')
    metadata = candidate_metadata(task_id, texture_root=output, **options)
    output.mkdir(parents=True)
    _write_weave(output)
    dump(output/'metadata.json', metadata)
    from .initial_state_assets import RESET_PROFILE
    published_reset = metadata['reset_profile'] == RESET_PROFILE
    if not published_reset:
        dump(output/'cpu_geometry.json', geometry_report(metadata))
    attribution = ('Single unfolded garment folding task. Woven maps are project-authored.\n'
        'Geometry and source transformations are explicitly recorded in metadata.design.representation.\n'
        'No native solver success or calibrated material properties are inherited from another engine.\n')
    dependencies = {str(output/name): digest(output/name) for name in ['woven_normal.png', 'woven_roughness.png']}
    if metadata.get('material_profile'):
        from . import material_profiles
        material_source = Path(material_profiles.__file__).resolve()
        dependencies[str(material_source)] = digest(material_source)
    if metadata['mesh_profile'] == SHIRT_PROFILE:
        (output/'LICENSE-Newton-Apache-2.0.txt').write_bytes(NEWTON_LICENSE.read_bytes())
        attribution += 'Shirt surface: Newton 1.6.1 examples/assets/unisex_shirt.usd, Apache-2.0 distribution; original topology retained.\n'
        dependencies.update({str(SHIRT_USD): SHIRT_SHA256, str(NEWTON_LICENSE): NEWTON_LICENSE_SHA256,
            str(output/'LICENSE-Newton-Apache-2.0.txt'): digest(output/'LICENSE-Newton-Apache-2.0.txt')})
    elif task_id == 'trousers_folding':
        attribution += 'Trouser surface: project-authored sewn two-panel approximation, CC0-1.0.\n'
    else:
        attribution += 'T-shirt surface: project-authored flat sewn two-panel pattern, CC0-1.0; distinct from the retained official Newton profile.\n'
        from . import flat_shirt_pattern
        pattern_source = Path(flat_shirt_pattern.__file__).resolve()
        dependencies[str(pattern_source)] = digest(pattern_source)
    if metadata['unfolded_reference_source']:
        source = metadata['unfolded_reference_source']
        dependencies[source['path']] = source['sha256']
    if metadata['native_settled_source']:
        source = metadata['native_settled_source']['native_report']
        dependencies[source['path']] = source['sha256']
        report = load_json(Path(source['path']))
        for name in ('cloth_q', 'cloth_qd', 'control_mask', 'time_s'):
            item = report['states']['files'][name]
            dependencies[item['path']] = item['sha256']
        orientation = metadata['native_settled_source'].get('input_orientation')
        if orientation:
            for item in (orientation['scene'], orientation['asset_metadata']):
                dependencies[item['path']] = item['sha256']
        material = metadata['native_settled_source'].get('material_source')
        if material:
            for item in (material['scene'], material['asset_metadata']):
                dependencies[item['path']] = item['sha256']
    (output/'ATTRIBUTION.txt').write_text(attribution)
    manifest = {'schema': INPUT_MANIFEST_SCHEMA if published_reset else MANIFEST_SCHEMA, 'task_id': task_id,
        'metadata_sha256': digest(output/'metadata.json'), 'attribution_sha256': digest(output/'ATTRIBUTION.txt'),
        'dependencies_sha256': dependencies,
        'authoring_source': {'path': str(Path(__file__).resolve()), 'sha256': digest(__file__)},
        'native_physics_accepted': False, 'material_calibrated': False}
    if not published_reset:
        manifest['cpu_geometry_sha256'] = digest(output/'cpu_geometry.json')
    dump(output/'manifest.json', manifest)
    return manifest


def load_assets(task_id, path):
    path = Path(path).resolve()
    manifest = load_json(path)
    if manifest.get('schema') not in (MANIFEST_SCHEMA, INPUT_MANIFEST_SCHEMA) or manifest.get('task_id') != task_id:
        raise ValueError('Wrong single garment folding manifest')
    inventory = {str(path): digest(path)}
    files = [('metadata.json', 'metadata_sha256'), ('ATTRIBUTION.txt', 'attribution_sha256')]
    if manifest['schema'] == MANIFEST_SCHEMA:
        files.append(('cpu_geometry.json', 'cpu_geometry_sha256'))
    for name, field in files:
        inventory[str(path.parent/name)] = checked(path.parent/name, manifest[field])
    for dependency, sha in manifest['dependencies_sha256'].items():
        inventory[dependency] = checked(dependency, sha)
    metadata = validate_metadata(load_json(path.parent/'metadata.json'))
    from .initial_state_assets import RESET_PROFILE
    if (manifest['schema'] == INPUT_MANIFEST_SCHEMA) != (metadata['reset_profile'] == RESET_PROFILE):
        raise ValueError('Garment input manifest and reset generation disagree')
    source = manifest['authoring_source']
    inventory[source['path']] = checked(source['path'], source['sha256'])
    return {'manifest_path': str(path), 'manifest_sha256': digest(path), 'files_sha256': inventory}, metadata


load_asset_manifest = load_assets
