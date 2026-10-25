"""Exact exterior of the existing faceted L5 physical solids, for diagnostics.

This constructs a surface union, not an approximation of the nominal gear.
It preserves the differing root and hub bore polygons and their small ledges.
The visual-only paint marker is deliberately absent. No task asset is replaced.
"""
from __future__ import annotations

import math
import argparse
import hashlib
import json
import shutil
from pathlib import Path
import numpy as np

from bench.paths import load_json
from scenes.workrooms.mechanical_assembly.tasks.gear_train_assembly.assets import (
    LEGACY_ASSET_ROOT as ASSET_ROOT, GEOMETRY, STEEL_DENSITY, _polar, _root_boundary, gear_asset, tooth_half_angle)

SOURCE_MANIFEST_SHA256 = '32079d5e2db8a788ccc16ce1a52cfad3769fe90dd1b97dedcfda72da692f5f33'


def _outline():
    g = GEOMETRY
    radii = [g['root_radius_m'], *np.linspace(g['base_radius_m'],
        g['outside_radius_m'], g['involute_flank_intervals']+1)]
    result = []
    pitch = 2*math.pi/g['teeth']
    for index in range(g['teeth']):
        center = index*pitch
        result.extend(_polar(r, center-tooth_half_angle(r)) for r in radii)
        tip = tooth_half_angle(radii[-1])
        result.extend(_polar(radii[-1], a) for a in
            np.linspace(center-tip, center+tip, g['tooth_tip_arc_intervals']+1)[1:])
        result.extend(_polar(r, center+tooth_half_angle(r)) for r in radii[-2::-1])
        half = tooth_half_angle(radii[0])
        result.extend(_polar(radii[0], a) for a in
            np.linspace(center+half, center+pitch-half, g['root_gap_intervals']+1)[1:-1])
    return np.asarray(result)


def _angular_vertices(polygon):
    angles = np.mod(np.arctan2(polygon[:, 1], polygon[:, 0]), 2*math.pi)
    # Equal-angle radial root relief edges have two distinct vertices. Keep
    # their original counterclockwise traversal order, including the jump.
    order = np.argsort(np.round(angles, 12), kind='stable')
    angles = angles[order]
    for i in range(1, len(angles)):
        if abs(angles[i]-angles[i-1]) <= 1.e-12:
            angles[i] = angles[i-1]
    return angles, polygon[order]


def _at(polygon, angles, sides=None):
    """Intersect rays with the original straight edges; never round a profile."""
    knots, points = _angular_vertices(polygon)
    indices = (np.searchsorted(knots, np.mod(angles, 2*math.pi), side='right')-1) % len(knots)
    first, second = points[indices], points[(indices+1) % len(points)]
    rays = np.c_[np.cos(angles), np.sin(angles)]
    edge = second-first
    cross = lambda a, b: a[..., 0]*b[..., 1]-a[..., 1]*b[..., 0]
    with np.errstate(divide='ignore', invalid='ignore'):
        result = rays*(cross(first, second)/cross(rays, edge))[:, None]
    # Return the source vertex exactly at a knot. At a radial edge select the
    # incoming or outgoing endpoint rather than collapsing the real step.
    sides = ['right']*len(angles) if sides is None else sides
    for i, angle in enumerate(np.mod(angles, 2*math.pi)):
        at_knot = np.flatnonzero(np.abs((knots-angle+math.pi) % (2*math.pi)-math.pi) < 1.e-12)
        if len(at_knot):
            result[i] = points[at_knot[0] if sides[i] == 'left' else at_knot[-1]]
    return result


def _unique_angles(values):
    values = sorted(float(v % (2*math.pi)) for v in values)
    result = []
    for value in values:
        if not result or value-result[-1] > 1.e-12:
            result.append(value)
    if len(result) > 1 and result[0]+2*math.pi-result[-1] <= 1.e-12:
        result.pop()
    return np.asarray(result)


def _conform_edges(mesh):
    """Split cap edges at existing relief vertices, without moving a vertex.

    A radial root relief makes two consecutive outer vertices share a ray.
    Its zero-area cap wedge leaves a T junction in adjacent cap triangles;
    subdivision of those straight edges restores a conforming triangulation.
    """
    import trimesh
    for _ in range(4):
        counts = np.bincount(mesh.edges_unique_inverse)
        boundary = mesh.edges_unique[counts == 1]
        if not len(boundary):
            return mesh
        candidates = np.unique(boundary)
        boundary_keys = {tuple(edge) for edge in boundary}
        faces, changed = [], False
        for triangle in mesh.faces:
            split = False
            for k in range(3):
                i, j, opposite = triangle[k], triangle[(k+1) % 3], triangle[(k+2) % 3]
                if tuple(sorted((i, j))) not in boundary_keys:
                    continue
                delta = mesh.vertices[j]-mesh.vertices[i]
                points = mesh.vertices[candidates]-mesh.vertices[i]
                fractions = points @ delta/np.dot(delta, delta)
                distance = np.linalg.norm(points-fractions[:, None]*delta, axis=1)
                mask = (fractions > 1.e-10) & (fractions < 1.-1.e-10) & (distance < 1.e-12)
                if not np.any(mask):
                    continue
                middle = candidates[mask][np.argsort(fractions[mask])]
                sequence = [i, *middle, j]
                faces.extend([a, b, opposite] for a, b in zip(sequence, sequence[1:]))
                split = changed = True
                break
            if not split:
                faces.append(triangle)
        if not changed:
            return mesh
        mesh = trimesh.Trimesh(vertices=mesh.vertices, faces=faces, process=False)
    return mesh


def gear_exterior(*, input_hub=False):
    """Return one watertight genus-one surface with the exact source boundary."""
    import trimesh
    g = GEOMETRY
    root_angles, _ = _root_boundary()
    factor = 1/math.cos(np.diff(np.r_[root_angles, root_angles[0]+2*math.pi]).max()/2)
    root_bore = np.array([_polar(g['bore_radius_m']*factor, a) for a in root_angles])
    hub_angles = np.arange(48)*2*math.pi/48
    hub_bore = np.array([_polar(g['bore_radius_m']/math.cos(math.pi/48), a) for a in hub_angles])
    if input_hub:
        normals = np.arange(6)*math.pi/3+math.pi/6
        hub_radii = [g['input_hub_across_flats_m']/2/max(math.cos(a-n) for n in normals) for a in hub_angles]
    else:
        hub_radii = np.full(48, g['hub_radius_m'])
    hub_outer = np.array([_polar(r, a) for r, a in zip(hub_radii, hub_angles)])
    outer = _outline()
    angles = _unique_angles(np.concatenate([_angular_vertices(p)[0] for p in (root_bore, hub_bore, hub_outer, outer)]))
    # At the face/hub interface the original bore polygons cross. Split at
    # their actual edge intersections so both upward and downward ledges stay.
    crossings = []
    closed = np.r_[angles, angles[0]+2*math.pi]
    a, b = _at(root_bore, closed), _at(hub_bore, closed)
    difference = np.linalg.norm(a, axis=1)-np.linalg.norm(b, axis=1)
    for i in range(len(angles)):
        if difference[i]*difference[i+1] < 0:
            da, db = a[i+1]-a[i], b[i+1]-b[i]
            determinant = da[0]*db[1]-da[1]*db[0]
            if abs(determinant) > 1.e-20:
                delta = b[i]-a[i]
                point = a[i]+da*((delta[0]*db[1]-delta[1]*db[0])/determinant)
                crossings.append(math.atan2(point[1], point[0]))
    angles = _unique_angles([*angles, *crossings])
    radial_knots, _ = _angular_vertices(outer)
    radial_knots = radial_knots[:-1][np.diff(radial_knots) < 1.e-12]
    expanded, sides = [], []
    for angle in angles:
        if np.any(np.abs((radial_knots-angle+math.pi) % (2*math.pi)-math.pi) < 1.e-12):
            expanded.extend([angle, angle])
            sides.extend(['left', 'right'])
        else:
            expanded.append(angle)
            sides.append('right')
    angles = np.asarray(expanded)
    root, hub, outside, shoulder = (_at(p, angles, sides) for p in (root_bore, hub_bore, outer, hub_outer))
    vertices, faces = [], []

    def ring(points, height):
        start = len(vertices)
        vertices.extend(np.c_[points, np.full(len(points), height)])
        return np.arange(start, start+len(points))

    def strip(first, second, reverse=False, selected=None):
        for i in range(len(angles)):
            if selected is not None and not selected[i]:
                continue
            j = (i+1) % len(angles)
            triangles = [[first[i], first[j], second[j]], [first[i], second[j], second[i]]]
            faces.extend([t[::-1] for t in triangles] if reverse else triangles)

    zface, ztop = g['face_width_m'], g['face_width_m']+g['hub_height_m']
    outer0, outer1 = ring(outside, 0.), ring(outside, zface)
    inner0 = ring(root*g['entry_radius_m']/g['bore_radius_m'], 0.)
    inner1, inner2 = ring(root, g['entry_depth_m']), ring(root, zface)
    hub0, hub1 = ring(hub, zface), ring(hub, ztop)
    shoulder0, shoulder1 = ring(shoulder, zface), ring(shoulder, ztop)
    strip(outer0, outer1)
    strip(inner0, inner1, reverse=True)
    strip(inner1, inner2, reverse=True)
    strip(hub0, hub1, reverse=True)
    strip(shoulder0, shoulder1)
    strip(inner0, outer0)  # bottom, -Z
    strip(shoulder0, outer1, reverse=True)  # exposed gear face, +Z
    strip(hub1, shoulder1, reverse=True)  # hub top, +Z
    # strip(inner2, hub0) is -Z when hub bore is larger, so reverse it.
    mid = (angles+np.r_[angles[1:], angles[0]+2*math.pi])/2
    root_radius = np.linalg.norm(_at(root_bore, mid), axis=1)
    hub_radius = np.linalg.norm(_at(hub_bore, mid), axis=1)
    strip(inner2, hub0, reverse=True, selected=hub_radius >= root_radius)
    strip(hub0, inner2, selected=hub_radius < root_radius)
    mesh = trimesh.Trimesh(vertices=np.asarray(vertices), faces=np.asarray(faces), process=False)
    mesh.merge_vertices(digits_vertex=12)
    mesh.update_faces(mesh.nondegenerate_faces(height=1.e-13))
    mesh.remove_unreferenced_vertices()
    mesh = _conform_edges(mesh)
    if not mesh.is_watertight or not mesh.is_winding_consistent or mesh.euler_number != 0 or mesh.body_count != 1 or mesh.volume <= 0:
        raise ValueError('Exact union surface is not a single closed consistently oriented through-bore solid')
    return mesh


def geometry_report(*, input_hub=False):
    """Compare the union integrals with the unchanged 448 physical pieces."""
    mesh = gear_exterior(input_hub=input_hub)
    source = gear_asset('input_gear' if input_hub else 'idler_gear', input_hub=input_hub)
    reference = source.data
    mass = mesh.volume*STEEL_DENSITY
    inertia = mesh.moment_inertia*STEEL_DENSITY
    com_error = float(np.linalg.norm(mesh.center_mass-source.center_of_mass_m))
    mass_error = abs(mass-source.mass_kg)
    inertia_error = float(np.max(np.abs(inertia-np.asarray(reference['inertia_body_frame_kg_m2']))))
    # Every output vertex, triangle center and unique edge midpoint must lie
    # on a source solid boundary. At face centers, opposite normal offsets
    # also reject a retained internal seam. This supplements the exact
    # construction and mass integrals; it is not an SDF sampling certificate.
    centers = mesh.triangles_center
    witnesses = np.vstack([mesh.vertices, centers, mesh.vertices[mesh.edges_unique].mean(axis=1)])
    probes = np.vstack([witnesses, centers+mesh.face_normals*1.e-10,
                        centers-mesh.face_normals*1.e-10])
    signed = np.full(len(probes), np.inf)
    for piece, _, _ in source._physical_pieces:
        low, high = piece.bounds
        mask = np.all(probes >= low-2.e-10, axis=1) & np.all(probes <= high+2.e-10, axis=1)
        indices = np.flatnonzero(mask)
        normals = piece.face_normals
        offsets = np.einsum('ij,ij->i', normals, piece.triangles[:, 0])
        distance = np.max(probes[indices] @ normals.T-offsets, axis=1)
        signed[indices] = np.minimum(signed[indices], distance)
    boundary_error = float(np.max(np.abs(signed[:len(witnesses)])))
    outward = signed[len(witnesses):len(witnesses)+len(centers)]
    inward = signed[len(witnesses)+len(centers):]
    outward_failures = int(np.sum(outward < -1.e-12))
    inward_failures = int(np.sum(inward > 1.e-12))
    return mesh, {'schema': 'gear_exact_union_cpu.v1', 'input_hub': input_hub,
        'passed': bool(mass_error < 1.e-10 and com_error < 1.e-12 and inertia_error < 1.e-12
            and boundary_error < 1.e-12 and outward_failures == 0 and inward_failures == 0),
        'original_physical_piece_count': len(source._physical_pieces),
        'vertices': len(mesh.vertices), 'triangles': len(mesh.faces),
        'watertight': bool(mesh.is_watertight), 'winding_consistent': bool(mesh.is_winding_consistent),
        'body_count': int(mesh.body_count), 'euler_number': int(mesh.euler_number),
        'mass_kg': float(mass), 'mass_absolute_error_kg': float(mass_error),
        'center_of_mass_m': mesh.center_mass.tolist(), 'com_error_m': com_error,
        'inertia_body_frame_kg_m2': inertia.tolist(), 'inertia_max_absolute_error_kg_m2': inertia_error,
        'source_boundary_witnesses': len(witnesses), 'source_boundary_halfspace_error_m': boundary_error,
        'outward_face_centers_inside_source_union': outward_failures,
        'inward_face_centers_outside_source_union': inward_failures,
        'normal_side_probe_distance_m': 1.e-10,
        'visual_only_marker_excluded': True,
        'scope': 'CPU exact faceted physical union only. No SDF cooking, dynamics or task qualification.'}


def source_asset_audit(name, *, source_root=ASSET_ROOT):
    """Bind the mathematical source to every immutable ten-decimal OBJ cell."""
    import trimesh
    from scipy.spatial import cKDTree
    source_root = Path(source_root).resolve()
    manifest_path = source_root/'manifest.json'
    digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    if digest(manifest_path) != SOURCE_MANIFEST_SHA256:
        raise ValueError('Original v2 source manifest changed')
    entry = load_json(manifest_path)['assets'][name]
    generated = gear_asset(name, input_hub=name == 'input_gear')
    errors, files = [], {str(manifest_path): SOURCE_MANIFEST_SHA256}
    for (mesh, _, label), part in zip(generated._physical_pieces, entry['collision_parts'], strict=True):
        path = source_root/part['mesh_path']
        if label != part['label'] or digest(path) != part['sha256']:
            raise ValueError('Source physical piece changed: '+str(path))
        if part['position_m'] != [0., 0., 0.] or part['quaternion_xyzw'] != [0., 0., 0., 1.] or part['scale'] != [1, 1, 1]:
            raise ValueError('Unexpected source piece transform')
        actual = trimesh.load(path, force='mesh', process=False)
        if len(actual.vertices) != len(mesh.vertices) or len(actual.faces) != len(mesh.faces):
            raise ValueError('Original source topology differs')
        error = max(cKDTree(actual.vertices).query(mesh.vertices)[0].max(),
                    cKDTree(mesh.vertices).query(actual.vertices)[0].max())
        errors.append(float(error))
        files[str(path)] = part['sha256']
    for key, value in [('mass_kg', generated.mass_kg), ('center_of_mass_m', generated.center_of_mass_m),
            ('inertia_body_frame_kg_m2', generated.data['inertia_body_frame_kg_m2'])]:
        if not np.array_equal(np.asarray(entry[key]), np.asarray(value)):
            raise ValueError('Source physical parameters changed: '+key)
    # export_obj(digits=10) rounds each coordinate by at most 0.5e-10 m.
    bound = math.sqrt(3)*.5e-10+1.e-15
    return {'passed': max(errors) <= bound, 'asset_id': name,
        'physical_piece_count': len(errors), 'maximum_source_vertex_rounding_error_m': max(errors),
        'declared_ten_decimal_rounding_bound_m': bound, 'source_files_sha256': files,
        'scope': 'Exact generated source union; original OBJ coordinates differ only by their recorded ten-decimal serialization precision.'}


def write_candidate(output):
    """Write a separate surface candidate; leave the original library untouched."""
    import trimesh
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    provenance = {}
    for source_name, target_name in [('LICENSE.txt', 'LICENSE.txt'), ('SOURCE.json', 'ORIGINAL_SOURCE.json')]:
        source_path, target_path = ASSET_ROOT/source_name, output/target_name
        shutil.copyfile(source_path, target_path)
        provenance[str(source_path)] = hashlib.sha256(source_path.read_bytes()).hexdigest()
        provenance[str(target_path)] = hashlib.sha256(target_path.read_bytes()).hexdigest()
    reports = {}
    for name in ('input_gear', 'idler_gear', 'output_gear'):
        mesh, report = geometry_report(input_hub=name == 'input_gear')
        source = source_asset_audit(name)
        if not report['passed'] or not source['passed']:
            raise ValueError('CPU union admission failed for '+name)
        path = output/(name+'.obj')
        path.write_text(trimesh.exchange.obj.export_obj(mesh, digits=17, include_normals=False))
        reports[name] = {'geometry': report, 'source': source, 'mesh_path': str(path),
            'mesh_sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    value = {'schema': 'gear_surface_union_candidate.v1', 'passed': True,
        'source_manifest_sha256': SOURCE_MANIFEST_SHA256, 'assets': reports,
        'source_license_provenance_sha256': provenance,
        'license': 'CC0-1.0; original procedural source by this project',
        'native_sdf_validated': False, 'physics_accepted': False, 'task_success_claimed': False}
    (output/'cpu_geometry.json').write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    value = write_candidate(args.output)
    print(json.dumps({'passed': value['passed'], 'output': str(args.output.resolve()),
        'native_sdf_validated': False, 'physics_accepted': False}))


if __name__ == '__main__':
    main()
