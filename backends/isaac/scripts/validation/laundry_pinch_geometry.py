"""Read-only thumb/index cloth pinch geometry from native cooked pad surfaces.

This diagnostic neither commands fingers nor creates attachments. An eligible
result is only a local geometric prerequisite: the caller must additionally
observe closing, convergence and material retention. It is not a contact-force,
friction, or grasp-success measurement. Body poses must be fresh native xyz/xyzw.
"""
from __future__ import annotations

import hashlib
import json

import numpy as np
from scipy.spatial import ConvexHull, HalfspaceIntersection

from bench.convex_geometry import clip_polygon, convex_triangle_audit
from bench.evaluation.cloth.cloth_robot_collision import (
    SCHEMA, SUPPORT_SCHEMA, apply_transform, planes_in_body, transform_xyzw,
)


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def _surface(point, vertices, faces, planes, *, contact_normal_profile='facet_v1'):
    """Exact point-to-triangulated-convex-boundary distance and closest face."""
    from scripts.validation.laundry_contact_observation import validate_normal_profile
    validate_normal_profile(contact_normal_profile)
    mesh = vertices[faces]
    a, b, c = mesh[:, 0], mesh[:, 1], mesh[:, 2]
    n = planes[:, :3]
    projection = point - (point @ n.T + planes[:, 3])[:, None]*n
    ab, ac, ap = b-a, c-a, projection-a
    d00, d01, d11 = (ab*ab).sum(1), (ab*ac).sum(1), (ac*ac).sum(1)
    denominator = d00*d11-d01*d01
    valid = denominator > 1.e-30
    # The inner halfspace intersection may contain almost coincident coplanar
    # vertices. Its QHull facets can then have zero area at floating precision.
    # Such facets still contribute their edges, never an invalid projection.
    v = np.divide(d11*(ap*ab).sum(1)-d01*(ap*ac).sum(1), denominator,
                  out=np.zeros(len(mesh)), where=valid)
    w = np.divide(d00*(ap*ac).sum(1)-d01*(ap*ab).sum(1), denominator,
                  out=np.zeros(len(mesh)), where=valid)
    inside = valid & (v >= -1.e-10) & (w >= -1.e-10) & (v+w <= 1+1.e-10)
    choices, face_ids = [projection[inside]], [np.flatnonzero(inside)]
    for x, y in ((a, b), (b, c), (c, a)):
        delta = y-x
        length2 = (delta*delta).sum(1)
        t = np.clip(np.divide(((point-x)*delta).sum(1), length2,
                    out=np.zeros(len(mesh)), where=length2 > 1.e-30), 0., 1.)
        choices.append(x+t[:, None]*delta)
        face_ids.append(np.arange(len(faces)))
    choices, face_ids = np.concatenate(choices), np.concatenate(face_ids)
    distances = np.linalg.norm(choices-point, axis=1)
    i = int(distances.argmin())
    signed = float(distances[i])
    if np.max(point @ planes[:, :3].T+planes[:, 3]) < 0:
        signed = -signed
    if contact_normal_profile=='distance_gradient_v1':
        # Outside a convex edge/vertex, use the unique distance gradient in
        # its normal cone. On-surface and interior medial-axis ties have no
        # unique gradient and cannot supply a witness in this explicit mode.
        ambiguous = abs(signed)<=1.e-12
        if signed<0:
            nearest=choices[abs(distances-abs(signed))<=1.e-12]
            ambiguous |= bool(np.any(np.linalg.norm(nearest-choices[i],axis=1)>1.e-9))
        return signed,choices[i],(None if ambiguous else (point-choices[i])/signed)
    return signed, choices[i], planes[face_ids[i], :3]


def _area(polygon):
    if len(polygon) < 3:
        return 0.
    return float(np.linalg.norm(np.cross(polygon[1:-1]-polygon[0],
                                        polygon[2:]-polygon[0]), axis=1).sum()/2)


def _diameter(points):
    return float(np.linalg.norm(points[:, None]-points[None, :], axis=2).max())


class PinchGeometry:
    """Cache the official left/right thumb and index elastomer convex surfaces.

    ``collider_export`` is the unmodified result of ``collect_colliders(env)``.
    A single original side is supported. All selected pads must have exactly one
    verified convex; unsupported representations fail closed. Source identity
    is retained in the receipt; callers remain responsible for binding an export
    to the same frozen robot/native run as the supplied body poses.
    """
    def __init__(self, collider_export, body_names, *, _surface_pair=('elastomer', 'elastomer'),
                 _include_distal=True):
        export = collider_export
        if (export.get('schema') != SCHEMA or export.get('status') != 'complete'
                or export.get('issues') or export.get('visual_mesh_fallback') is not False
                or not export.get('state_hash_before')
                or export['state_hash_before'] != export.get('state_hash_after')
                or not export.get('layer_hashes_before')
                or export['layer_hashes_before'] != export.get('layer_hashes_after')):
            raise ValueError('Complete unchanged native cooked collision export required')
        self.body_names = list(body_names)
        if len(set(self.body_names)) != len(self.body_names):
            raise ValueError('Unique native body names required')
        self.export_sha256 = _digest(export)
        if len(_surface_pair) != 2 or any(x not in ('elastomer', 'DP') for x in _surface_pair):
            raise ValueError('Only original thumb/index distal bodies are supported')
        self.surface_pair = tuple(_surface_pair)
        self.pads = {}
        for side in ('left', 'right'):
            sides = {}
            for finger, suffix in zip(('thumb', 'index'), self.surface_pair):
                body = side+'_'+finger+'_'+suffix
                rows = [r for r in export['colliders'] if r.get('body_name') == body]
                if not rows:
                    continue
                if (len(rows) != 1 or body not in self.body_names
                        or rows[0].get('status') != 'verified_cooking_representation'
                        or '/'+body+'/' not in rows[0]['path']
                        or rows[0].get('approximation') != 'convexHull'
                        or len(rows[0].get('pieces', [])) != 1):
                    raise ValueError('Original side-specific elastomer convex required')
                row, piece = rows[0], rows[0]['pieces'][0]
                if piece.get('geometry_representation') != SUPPORT_SCHEMA:
                    raise ValueError('Cooked inner/outer uncertainty bounds required')
                matrix = np.asarray(row['mesh_to_body_column_matrix'], float)
                outer = apply_transform(piece['conservative_outer_vertices'], matrix)
                outer_planes = planes_in_body(piece['conservative_outer_planes'], matrix)
                inner_planes = planes_in_body(piece['conservative_inner_planes'], matrix)
                center = apply_transform(np.asarray(piece['vertices'], float), matrix).mean(0)
                if np.max(center @ inner_planes[:, :3].T+inner_planes[:, 3]) >= -1.e-10:
                    raise ValueError('No verified interior point for cooked pad')
                inner = HalfspaceIntersection(inner_planes, center).intersections
                hull = ConvexHull(inner)
                if not np.isfinite(inner).all():
                    raise ValueError('Invalid cooked pad bound')
                sides[finger] = {'body': self.body_names.index(body), 'path': row['path'],
                    'body_name': body,
                    'outer': outer, 'outer_planes': outer_planes, 'inner': inner,
                    'inner_planes': hull.equations.copy(), 'faces': hull.simplices.copy()}
            if sides and set(sides) != {'thumb', 'index'}:
                raise ValueError('Both original thumb and index pads are required per side')
            if sides:
                self.pads[side] = sides
        if not self.pads:
            raise ValueError('No verified original SHARPA thumb/index pad pair')
        self._variants = []
        if _include_distal:
            # Each *actual* pair stays separate; never convex-hull a whole finger
            # or bridge a non-contact gap between its shell and elastomer.
            inventory = {r.get('body_name') for r in export['colliders']}
            for pair in (('DP', 'elastomer'), ('elastomer', 'DP'), ('DP', 'DP')):
                if any(all(side+'_'+finger+'_'+suffix in inventory for finger, suffix in
                           zip(('thumb', 'index'), pair)) for side in self.pads):
                    self._variants.append(PinchGeometry(export, self.body_names,
                        _surface_pair=pair, _include_distal=False))
        self._topology_key, self._material_components = None, None

    def contact_motion_reference(self, side):
        """Original cooked outer vertices, in each actual rigid-body frame."""
        from scripts.validation.laundry_contact_stability import REFERENCE_SCHEMA
        surfaces = {}
        for geometry in [self, *self._variants]:
            for pad in geometry.pads.get(side, {}).values():
                points = pad['outer'].tolist()
                if pad['body_name'] in surfaces and surfaces[pad['body_name']] != points:
                    raise ValueError('Conflicting original cooked contact surfaces')
                surfaces[pad['body_name']] = points
        return {'schema': REFERENCE_SCHEMA, 'side': side,
                'cooked_export_sha256': self.export_sha256,
                'body_names_sha256': _digest(self.body_names), 'surfaces': surfaces}

    def _multilayer(self, pads, q, tri, rest, axis, alo, bhi, thickness, band,
                    witness_penetration, severe, diameter, max_vertices, min_area, normal_dot_max=-.5,
                    contact_normal_profile='facet_v1', contact_witness_profile='centroid_v1'):
        """Separate two-face diagnostic, never automatic attachment eligibility.

        Two distinct rest-local surfaces must belong to the same mesh component.
        This permits sewn front/back panels but rejects coincident disconnected
        sheets and distant garment material. No vertices or topology are changed.
        """
        topology_key = (len(q), hashlib.sha256(tri.tobytes()).hexdigest())
        if topology_key != self._topology_key:
            parent = np.arange(len(q))
            def find(i):
                while parent[i] != i:
                    parent[i] = parent[parent[i]]; i = parent[i]
                return int(i)
            for ids in tri:
                root = find(int(ids[0]))
                for i in ids[1:]: parent[find(int(i))] = root
            self._material_components = np.array([find(i) for i in range(len(q))])
            self._topology_key = topology_key
        each = {};ambiguous=[]
        xyz = q[tri]
        for finger, orientation in (('thumb', 1), ('index', -1)):
            pad, rows = pads[finger], []
            lo, hi = pad['bounds']; candidates = np.flatnonzero(
                np.all(xyz.max(1) >= lo-band, axis=1) & np.all(xyz.min(1) <= hi+band, axis=1))
            if contact_witness_profile in ('facet_regions_v1', 'surface_regions_v2'):
                from scripts.validation.laundry_facet_regions import material_facet_witnesses
                each[finger] = material_facet_witnesses(pad, xyz, tri, rest, candidates, axis, orientation,
                    alo, bhi, thickness, band, witness_penetration, min_area,
                    surface_query=_surface, polygon_area=_area)
                if contact_witness_profile == 'surface_regions_v2':
                    from scripts.validation.laundry_edge_regions import material_edge_witnesses
                    each[finger].extend(material_edge_witnesses(pad, xyz, tri, rest, candidates, axis, orientation,
                        alo, bhi, thickness, band, witness_penetration, min_area,
                        surface_query=_surface, polygon_area=_area))
                continue
            for tid in candidates:
                polygon = xyz[tid].copy()
                for plane in pad['inner_planes']:
                    polygon = clip_polygon(polygon, plane[:3], plane[3]-band)
                    if len(polygon) < 3: break
                if len(polygon) < 3 or _area(polygon) < min_area: continue
                point = polygon.mean(0)
                distance, surface, normal = _surface(point, pad['inner'], pad['faces'], pad['inner_planes'],
                    **({'contact_normal_profile':contact_normal_profile} if contact_normal_profile!='facet_v1' else {}))
                if normal is None:
                    ambiguous.append({'finger':finger,'triangle_id':int(tid),'point_m':point.tolist(),
                        'signed_inner_distance_m':distance,'reason':'undefined_signed_distance_gradient'})
                    continue
                along = float(point@axis)
                if (distance > band+1.e-9 or distance < -witness_penetration
                        or float(normal@axis)*orientation < .25
                        or along < alo-thickness/2-witness_penetration
                        or along > bhi+thickness/2+witness_penetration): continue
                material = rest[tri[tid]]
                weights = np.linalg.lstsq((xyz[tid, 1:]-xyz[tid, 0]).T, point-xyz[tid, 0], rcond=None)[0]
                material_point = material[0]+weights@(material[1:]-material[0])
                rows.append({'triangle_id': int(tid), 'point_m': point.tolist(),
                    'rest_point_m': material_point.tolist(), 'signed_inner_distance_m': distance,
                    'surface_point_m': surface.tolist(), 'normal_world': normal.tolist(),
                    'clipped_area_m2': _area(polygon)})
            each[finger] = rows
        pairs = []
        for a in each['thumb']:
            for b in each['index']:
                ia, ib = a['triangle_id'], b['triangle_id']
                if ia == ib: continue
                ids = np.unique(tri[[ia, ib]])
                if (len(ids) > max_vertices or len(set(self._material_components[ids])) != 1
                        or _diameter(q[ids]) > diameter or _diameter(rest[ids]) > diameter): continue
                delta = np.array(b['point_m'])-a['point_m']; axial = float(delta@axis)
                lateral = float(np.linalg.norm(delta-axial*axis))
                material_distance = float(np.linalg.norm(np.array(b['rest_point_m'])-a['rest_point_m']))
                normal_dot = float(np.array(a['normal_world'])@b['normal_world'])
                if (axial <= thickness*.25 or axial > min(.015,4*thickness)
                        or lateral > .010 or material_distance > .020 or normal_dot > normal_dot_max): continue
                pairs.append({'thumb': a, 'index': b, 'local_vertex_ids': ids.tolist(),
                    'same_connected_garment': True, 'local_topology_connected': len(set(tri[ia]) & set(tri[ib])) >= 2,
                    'world_diameter_m': _diameter(q[ids]), 'material_diameter_m': _diameter(rest[ids]),
                    'material_witness_distance_m': material_distance, 'axial_material_span_m': axial,
                    'lateral_witness_separation_m': lateral, 'opposing_normal_dot': normal_dot})
        result = {'plausible': bool(pairs) and not severe and bhi >= alo-witness_penetration,
            'automatic_attachment_eligible': False,
            'scope': 'Distinct local material faces between opposed surfaces; not a force, retention, local weld or grasp-success claim.',
            'per_finger_witnesses': each, 'paired_witnesses': pairs,
            'maximum_layer_span_m': min(.015,4*thickness)}
        if contact_normal_profile!='facet_v1':result['undefined_normal_witnesses']=ambiguous
        return result

    def diagnose(self, side, body_q, cloth_q, triangles, rest_vertices, thickness_m,
                 *, contact_band_m=.0015, max_patch_diameter_m=.035,
                 max_patch_vertices=32, severe_penetration_m=.002,
                 witness_penetration_m=.0005, minimum_witness_area_m2=5.e-7,
                 engineering_profile=None, contact_normal_profile='facet_v1',
                 contact_witness_profile='centroid_v1'):
        """Return conservative geometry eligibility, local IDs and raw evidence.

        Contact-band distance is an upper bound measured to each *inner* cooked
        convex; proximity to an AABB or expanded halfspace alone never qualifies.
        Cloth triangle interiors are clipped/tested, including a triangle with
        vertices outside the pad. A selected connected material patch must fit
        the diameter bound in both rest and observed world coordinates.
        """
        if engineering_profile not in (None, 'angled_adjacent_patch_v4'):
            raise ValueError('Unknown explicit engineering geometry profile')
        from scripts.validation.laundry_contact_observation import validate_normal_profile
        validate_normal_profile(contact_normal_profile)
        from scripts.validation.laundry_contact_observation import validate_witness_profile
        validate_witness_profile(contact_witness_profile, contact_normal_profile)
        normal_dot_max = -.25 if engineering_profile else -.5
        scalars = (thickness_m, contact_band_m, max_patch_diameter_m,
                   severe_penetration_m, witness_penetration_m, minimum_witness_area_m2)
        if any(not np.isfinite(x) or x <= 0 for x in scalars):
            raise ValueError('Positive finite pinch thresholds required')
        if (side not in self.pads or not isinstance(max_patch_vertices, int)
                or not 3 <= max_patch_vertices <= 32 or max_patch_diameter_m > .075
                or contact_band_m > .003 or witness_penetration_m > severe_penetration_m):
            raise ValueError('Unsupported side or excessive pinch thresholds')
        poses, q, rest = np.asarray(body_q, float), np.asarray(cloth_q, float), np.asarray(rest_vertices, float)
        raw_tri = np.asarray(triangles)
        if (poses.shape != (len(self.body_names), 7) or q.ndim != 2 or q.shape[1] != 3
                or len(q) < 3 or rest.shape != q.shape or raw_tri.ndim != 2 or raw_tri.shape[1] != 3
                or not np.issubdtype(raw_tri.dtype, np.integer) or len(raw_tri) == 0
                or raw_tri.min() < 0 or raw_tri.max() >= len(q)
                or not all(np.isfinite(x).all() for x in (poses, q, rest))):
            raise ValueError('Complete finite native poses and indexed cloth surface required')
        tri = raw_tri.astype(np.int64, copy=False)
        pads = {}
        severe = []
        for finger, pad in self.pads[side].items():
            matrix = transform_xyzw(poses[pad['body']])
            row = {key: apply_transform(pad[key], matrix) for key in ('inner', 'outer')}
            row.update({key: planes_in_body(pad[key], matrix) for key in ('inner_planes', 'outer_planes')})
            row['faces'], row['path'], row['body_name'] = pad['faces'], pad['path'], pad['body_name']
            if contact_witness_profile in ('facet_regions_v1', 'surface_regions_v2'):
                from scripts.validation.laundry_facet_regions import coplanar_facets, world_facets
                if 'witness_facets_local' not in pad:
                    pad['witness_facets_local'] = coplanar_facets(pad['inner'], pad['faces'], pad['inner_planes'])
                row['witness_facets'] = world_facets(pad['witness_facets_local'], matrix)
                if contact_witness_profile == 'surface_regions_v2':
                    from scripts.validation.laundry_edge_regions import convex_edges, world_edges
                    if 'witness_edges_local' not in pad:
                        pad['witness_edges_local'] = convex_edges(pad['inner'], pad['faces'], pad['inner_planes'])
                    row['witness_edges'] = world_edges(pad['witness_edges_local'], matrix)
            row['bounds'] = np.stack([row['outer'].min(0), row['outer'].max(0)])
            penetration = convex_triangle_audit(q, tri, row['outer_planes'], row['bounds'], severe_penetration_m)
            # Either outer-bound deep intersection is disqualifying; uncertain
            # raw-plane/support-hull disagreement cannot enable an attachment.
            row['severe_penetration'] = penetration
            if not penetration['all_triangles_certified_clear_within_tolerance']:
                severe.append(finger)
            pads[finger] = row
        a, b = pads['thumb'], pads['index']
        ca, cb = a['inner'].mean(0), b['inner'].mean(0)
        axis_length = float(np.linalg.norm(cb-ca))
        if axis_length < 1.e-8:
            raise ValueError('Coincident pad centers do not define a pinch axis')
        axis = (cb-ca)/axis_length
        alo, bhi = float((a['outer']@axis).max()), float((b['outer']@axis).min())
        inner_a, inner_b = float((a['inner']@axis).max()), float((b['inner']@axis).min())
        band = thickness_m/2+contact_band_m
        witnesses = [];ambiguous=[]
        lower = np.maximum(a['bounds'][0]-band, b['bounds'][0]-band)
        upper = np.minimum(a['bounds'][1]+band, b['bounds'][1]+band)
        xyz = q[tri]
        candidate = np.flatnonzero(np.all(xyz.max(1) >= lower, axis=1) & np.all(xyz.min(1) <= upper, axis=1))
        if bhi >= alo-witness_penetration_m:
            for tid in candidate:
                polygon = xyz[tid].copy()
                for pad in (a, b):
                    for plane in pad['inner_planes']:
                        polygon = clip_polygon(polygon, plane[:3], plane[3]-band)
                        if len(polygon) < 3:
                            break
                    if len(polygon) < 3:
                        break
                if len(polygon) < 3 or _area(polygon) < minimum_witness_area_m2:
                    continue
                point = polygon.mean(0)
                normal_options=({'contact_normal_profile':contact_normal_profile} if contact_normal_profile!='facet_v1' else {})
                da, pa, na = _surface(point, a['inner'], a['faces'], a['inner_planes'],**normal_options)
                db, pb, nb = _surface(point, b['inner'], b['faces'], b['inner_planes'],**normal_options)
                if na is None or nb is None:
                    ambiguous.append({'triangle_id':int(tid),'point_m':point.tolist(),
                        'undefined_fingers':[f for f,n in [('thumb',na),('index',nb)] if n is None],
                        'reason':'undefined_signed_distance_gradient'})
                    continue
                along = float(point@axis)
                if (max(da, db) > band+1.e-9 or min(da, db) < -witness_penetration_m
                        or along < alo-thickness_m/2-witness_penetration_m
                        or along > bhi+thickness_m/2+witness_penetration_m
                        or float(na@nb) > -.5 or float(na@axis) < .25 or float(nb@axis) > -.25):
                    continue
                witnesses.append({'triangle_id': int(tid), 'point_m': point.tolist(),
                    'clipped_area_m2': _area(polygon), 'thumb_signed_inner_distance_m': da,
                    'index_signed_inner_distance_m': db, 'opposing_normal_dot': float(na@nb),
                    'thumb_surface_point_m': pa.tolist(), 'index_surface_point_m': pb.tolist()})
        # Material connectivity uses shared edges, never only world-space nearness.
        by_id = {w['triangle_id']: w for w in witnesses}
        remaining, components = set(by_id), []
        while remaining:
            group, pending = set(), [min(remaining)]
            while pending:
                tid = pending.pop()
                if tid not in remaining:
                    continue
                remaining.remove(tid); group.add(tid)
                pending.extend(other for other in remaining if len(set(tri[tid]) & set(tri[other])) >= 2)
            components.append(sorted(group))
        components.sort(key=lambda ids: -sum(by_id[i]['clipped_area_m2'] for i in ids))
        selected, local = [], None
        for ids in components:
            vertices = np.unique(tri[ids])
            area = float(np.linalg.norm(np.cross(rest[tri[ids, 1]]-rest[tri[ids, 0]],
                                                 rest[tri[ids, 2]]-rest[tri[ids, 0]]), axis=1).sum()/2)
            world_diameter, material_diameter = _diameter(q[vertices]), _diameter(rest[vertices])
            if (len(vertices) <= max_patch_vertices and world_diameter <= max_patch_diameter_m
                    and material_diameter <= max_patch_diameter_m):
                selected = vertices.tolist()
                local = {'triangle_ids': ids, 'vertex_count': len(selected), 'material_connected': True,
                    'rest_area_m2': area, 'world_diameter_m': world_diameter,
                    'material_diameter_m': material_diameter,
                    'witness_area_m2': sum(by_id[i]['clipped_area_m2'] for i in ids)}
                break
        reasons = []
        if severe: reasons.append('cloth_deeply_intersects_pad_or_uncertain_outer_bound')
        if bhi < alo-witness_penetration_m: reasons.append('pad_support_projections_overlap')
        if not witnesses: reasons.append('no_same_material_surface_between_opposing_near_pads')
        elif not selected: reasons.append('no_connected_material_patch_within_locality_limits')
        eligible = not reasons
        result = {'schema': 'laundry_pinch_geometry.v1', 'eligible': eligible, 'reasons': reasons,
            'state_identity': {
                'body_q_float64_sha256': hashlib.sha256(poses.astype('<f8').tobytes()).hexdigest(),
                'cloth_q_float64_sha256': hashlib.sha256(q.astype('<f8').tobytes()).hexdigest()},
            'contact_force_measured': False, 'friction_grasp_qualified': False,
            'scope': 'Fresh native cooked-pad/cloth geometry only; caller must verify closing, dwell and retention.',
            'side': side, 'local_vertex_ids': selected if eligible else [],
            'candidate_vertex_ids': selected, 'locality': local,
            'pad_paths': {f: p['path'] for f, p in pads.items()},
            'contact_body_names': {f: p['body_name'] for f, p in pads.items()},
            'surface_pair': dict(zip(('thumb', 'index'), self.surface_pair)),
            'cooked_export_sha256': self.export_sha256,
            'body_names_sha256': _digest(self.body_names),
            'pinch_axis_world': axis.tolist(), 'pad_center_distance_m': axis_length,
            'pad_support_gap_along_axis_lower_m': bhi-alo,
            'pad_support_gap_along_axis_upper_m': inner_b-inner_a,
            'gap_scope': 'Conservative support-plane separation along pad-center axis, not minimum Euclidean distance.',
            'near_surface_distance_upper_bound_m': band,
            'severe_penetration_pads': severe,
            'severe_penetration_diagnostics': {f: p['severe_penetration'] for f, p in pads.items()},
            'material_component_count': len(components), 'witnesses': witnesses,
            'thresholds': {'cloth_thickness_m': thickness_m, 'contact_band_m': contact_band_m,
                'max_patch_diameter_m': max_patch_diameter_m, 'max_patch_vertices': max_patch_vertices,
                'severe_penetration_m': severe_penetration_m, 'witness_penetration_m': witness_penetration_m,
                'minimum_witness_area_m2': minimum_witness_area_m2}}
        result['multilayer_diagnostic'] = self._multilayer(pads, q, tri, rest, axis, alo, bhi,
            thickness_m, band, witness_penetration_m, severe, max_patch_diameter_m,
            max_patch_vertices, minimum_witness_area_m2, normal_dot_max,contact_normal_profile,contact_witness_profile)
        if contact_normal_profile!='facet_v1':
            result.update(contact_normal_profile=contact_normal_profile,
                undefined_normal_witnesses=ambiguous,
                normal_scope='Actual signed-distance gradient; on-surface or interior medial-axis ambiguity rejects the witness.')
        if contact_witness_profile != 'centroid_v1':
            result.update(contact_witness_profile=contact_witness_profile,
                witness_scope='Finite original cloth subregions projected inside verified cooked facets; exact distance-gradient witness normals.')
            if contact_witness_profile == 'surface_regions_v2':
                result['witness_scope'] = ('Finite original cloth subregions inside verified cooked facet prisms '
                    'or conservative exterior convex-edge Voronoi regions; exact distance-gradient normals.')
        if engineering_profile:
            result['engineering_geometry_profile'] = engineering_profile
            result['thresholds']['maximum_opposing_normal_dot'] = normal_dot_max
        if self._variants:
            options = []
            for geometry in self._variants:
                if side not in geometry.pads: continue
                options.append(geometry.diagnose(side, poses, q, tri, rest, thickness_m,
                    contact_band_m=contact_band_m, max_patch_diameter_m=max_patch_diameter_m,
                    max_patch_vertices=max_patch_vertices, severe_penetration_m=severe_penetration_m,
                    witness_penetration_m=witness_penetration_m, minimum_witness_area_m2=minimum_witness_area_m2,
                    engineering_profile=engineering_profile,
                    **({'contact_normal_profile':contact_normal_profile} if contact_normal_profile!='facet_v1' else {}),
                    **({'contact_witness_profile':contact_witness_profile} if contact_witness_profile!='centroid_v1' else {})))
            result['distal_contact_diagnostics'] = options
            deep_group = bool(severe) or any(row['severe_penetration_pads'] for row in options)
            result['distal_group_severe_penetration'] = deep_group
            result['distal_contact_eligible'] = not deep_group and (eligible or any(row['eligible'] for row in options))
            result['distal_multilayer_plausible'] = not deep_group and (
                result['multilayer_diagnostic']['plausible'] or any(
                    row['multilayer_diagnostic']['plausible'] for row in options))
            if deep_group and eligible:
                result['eligible'] = False
                result['local_vertex_ids'] = []
                result['reasons'].append('severe_penetration_elsewhere_in_thumb_index_distal_group')
        else:
            result['distal_contact_diagnostics'] = []
            result['distal_contact_eligible'] = eligible
            result['distal_multilayer_plausible'] = result['multilayer_diagnostic']['plausible']
            result['distal_group_severe_penetration'] = bool(severe)
        return result
