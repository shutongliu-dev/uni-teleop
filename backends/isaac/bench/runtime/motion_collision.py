"""Static robot-only convex advice; no task stage, simulator or object state.

The nonadjacent convex intersection LP is the same construction used by
scripts.diagnostics.check_bimanual_arm_self_collision. This loader deliberately
does not use CollisionBounds, whose broader offline scope reads task geometry.
"""
from __future__ import annotations

import hashlib
from itertools import combinations
import json
from pathlib import Path
import re

import numpy as np

from bench.paths import load_json


class CollisionSourceUnavailable(ValueError):
    def __init__(self, reason):
        self.reason = reason
        super().__init__(reason)


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class Fr3SelfCollision:
    """Original imported link0..7 convexes matched to the pinned arm URDF.

    Unsupported imports fail closed. Each check is static and intentionally
    ignores contact offsets, adjacent links, hands, other arms and environment.
    """
    def __init__(self, model):
        try:
            from pxr import Usd, UsdGeom, UsdPhysics
            from scipy.spatial import ConvexHull
        except ImportError as error:
            raise CollisionSourceUnavailable('dependency_unavailable') from error
        self.model = model
        source = Path(model.arm_path)
        side = model.embodiment.side
        candidates = [source.parent / f'USD_IMPORT_{side}.json', source.parent / 'USD_IMPORT.json']
        manifest_path = next((p for p in candidates if p.is_file()), None)
        if manifest_path is None:
            raise CollisionSourceUnavailable('source_import_unavailable')
        try:
            manifest = load_json(manifest_path)
            expected = model.asset_sha256[str(source)]
            if _sha(source) != expected or manifest['urdf_sha256'] != expected:
                raise ValueError('Import does not match the pinned URDF')
            inputs = {str(source): expected, str(manifest_path): _sha(manifest_path),
                      manifest['usd_path']: manifest['usd_sha256'], **manifest['derived_files']}
            if any(_sha(path) != sha for path, sha in inputs.items()):
                raise ValueError('Changed imported robot geometry')
            stage = Usd.Stage.Open(manifest['usd_path'])
            if stage is None:
                raise ValueError('Unavailable imported robot USD')
            # Reject imports pulling unverified external USD layers. Texture
            # files are visual only and never enter this geometry calculation.
            if any(layer.realPath and str(Path(layer.realPath)) not in inputs
                   for layer in stage.GetUsedLayers()):
                raise ValueError('Unverified imported robot layer')
            self.shapes = []
            xforms = UsdGeom.XformCache()
            names = set(model.arm.forward(dict.fromkeys(model.arm.names, 0.)))
            for prim in Usd.PrimRange(stage.GetPseudoRoot(), Usd.TraverseInstanceProxies()):
                if not prim.HasAPI(UsdPhysics.CollisionAPI):
                    continue
                body = prim
                while body and not body.HasAPI(UsdPhysics.RigidBodyAPI):
                    body = body.GetParent()
                if not body or body.GetName() not in names:
                    continue
                match = re.fullmatch(r'fr3_(?:[a-z][a-z0-9_]*_)?link([0-7])', body.GetName())
                if match is None:
                    continue
                if (not prim.IsA(UsdGeom.Mesh) or
                    UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr().Get() != 'convexHull'):
                    raise ValueError('Expected original convexHull arm meshes')
                relative = (np.linalg.inv(np.asarray(xforms.GetLocalToWorldTransform(body)).T)
                            @ np.asarray(xforms.GetLocalToWorldTransform(prim)).T)
                points = np.asarray(UsdGeom.Mesh(prim).GetPointsAttr().Get(), dtype=float)
                local = (relative @ np.c_[points, np.ones(len(points))].T).T[:, :3]
                hull = ConvexHull(local)
                self.shapes.append({'link': body.GetName(), 'index': int(match[1]),
                                    'vertices': local[hull.vertices], 'planes': hull.equations})
            if sorted(s['index'] for s in self.shapes) != list(range(8)):
                raise ValueError('Incomplete original arm convex coverage')
            self.shapes.sort(key=lambda s: s['index'])
            self.source_sha256 = hashlib.sha256(json.dumps(inputs, sort_keys=True,
                separators=(',', ':')).encode()).hexdigest()
        except (KeyError, OSError, TypeError, ValueError, RuntimeError) as error:
            raise CollisionSourceUnavailable('source_import_unverified') from error

    def check(self, joint_positions):
        """Return only public robot link names and geometric intersection radii."""
        from scipy.optimize import linprog
        q = {name: float(joint_positions[name]) for name in self.model.arm.names}
        if not all(np.isfinite(list(q.values()))):
            raise ValueError('Finite arm coordinates required')
        poses = self.model.arm.forward(q, self.model.arm_base)
        world = []
        for shape in self.shapes:
            pose = poses[shape['link']]
            points = (pose[:3, :3] @ shape['vertices'].T).T + pose[:3, 3]
            a = shape['planes'][:, :3] @ pose[:3, :3].T
            planes = np.c_[a, shape['planes'][:, 3] - a @ pose[:3, 3]]
            world.append((shape, np.array([points.min(0), points.max(0)]), planes))
        intersections = []
        for (first, a, ap), (second, b, bp) in combinations(world, 2):
            if abs(first['index'] - second['index']) <= 1:
                continue
            if not np.all(np.minimum(a[1], b[1]) - np.maximum(a[0], b[0]) > 1e-8):
                continue
            planes = np.vstack([ap, bp])
            solved = linprog([0., 0., 0., -1.],
                A_ub=np.c_[planes[:, :3], np.ones(len(planes))], b_ub=-planes[:, 3],
                bounds=[(None, None)] * 3 + [(0., None)], method='highs')
            if solved.status == 2:
                continue
            if not solved.success:
                raise RuntimeError('Robot convex intersection solve failed')
            if solved.x[3] > .001:
                intersections.append({'links': [first['link'], second['link']],
                                      'intersection_radius_m': float(solved.x[3])})
        return intersections

    def check_path(self, initial, goal):
        """65 fixed nominal samples; neither a continuous nor native sweep."""
        rows, count, previous, largest = {}, 0, None, 0.
        for fraction in np.linspace(0., 1., 65):
            alpha = 10 * fraction**3 - 15 * fraction**4 + 6 * fraction**5
            q = {n: float(initial[n] + alpha * (goal[n] - initial[n])) for n in self.model.arm.names}
            values = np.array(list(q.values()))
            if previous is not None:
                largest = max(largest, float(np.max(np.abs(values - previous))))
            previous = values
            pairs = self.check(q)
            count += bool(pairs)
            for pair in pairs:
                key = tuple(pair['links'])
                row = rows.setdefault(key, {'links': pair['links'], 'maximum_intersection_radius_m': 0.,
                    'first_sample_fraction': float(fraction), 'at_goal': False})
                row['maximum_intersection_radius_m'] = max(row['maximum_intersection_radius_m'], pair['intersection_radius_m'])
                row['at_goal'] |= bool(fraction == 1.)
        return {'status': 'sampled', 'detected': bool(rows), 'goal_detected': any(r['at_goal'] for r in rows.values()),
                'sample_count': 65, 'colliding_sample_count': count,
                'maximum_sample_joint_step_rad': largest, 'pairs': list(rows.values()),
                'robot_source_sha256': self.source_sha256}


def check_nominal_arm_path(model, initial, goal):
    try:
        return Fr3SelfCollision(model).check_path(initial, goal)
    except CollisionSourceUnavailable as error:
        return {'status': 'unsupported', 'reason': error.reason}
