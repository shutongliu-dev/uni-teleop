"""Procedural bored solids with explicit convex pieces and mass integrals.

The polygonal collision solid, rather than its bounding box, determines mass,
COM and inertia. Every bore is retained in the union of convex wedges.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
from numpy.polynomial import Polynomial

from bench.simulation.assets.scene_parts import Asset, lathe_mesh, write_library

SEGMENTS = 64
CONTACT_OFFSET_M = .0001
# Conservative PhysX 5.4/5.6.1 documented GPU convex envelope. These are source
# topology checks; cooked runtime data and GPU eligibility require native audit.
GPU_CONVEX_LIMITS = {'vertices': 64, 'polygons': 64, 'vertices_per_polygon': 32}
MATERIAL_PRIOR = {
    'static_friction': .4, 'dynamic_friction': .4, 'restitution': 0.,
    'friction_combine': 'average',
    'source': 'Explicit synthetic rigid-material task prior; same coefficient as workroom supports.',
    'measured_or_hardware_calibrated': False,
}
DENSITY_PRIOR_KG_M3 = 1200.


def _profile(profile):
    rows = np.asarray(profile, dtype=float)
    if (rows.ndim != 2 or rows.shape[1] != 3 or len(rows) < 2
            or not np.isfinite(rows).all() or np.any(np.diff(rows[:, 0]) <= 0)
            or np.any(rows[:, 1] < 0) or np.any(rows[:, 2] <= rows[:, 1])):
        raise ValueError('Profile requires increasing z and 0 <= inner radius < outer radius')
    # A whole wedge is convex only for convex inner and concave outer profiles.
    slopes = np.diff(rows[:, 1:], axis=0) / np.diff(rows[:, 0])[:, None]
    if len(slopes) > 1 and (np.any(np.diff(slopes[:, 0]) < -1e-10)
                            or np.any(np.diff(slopes[:, 1]) > 1e-10)):
        raise ValueError('Split nonconvex profiles into disjoint axial pieces before making hulls')
    return rows


def polygonal_mass_properties(profile, *, density_kg_m3=DENSITY_PRIOR_KG_M3, segments=SEGMENTS):
    """Exact integrals for coaxial regular-polygon lofts with linear radii.

    Outer radii are circumradii; positive inner radii are apothems, so the
    guaranteed hole diameter is exactly the requested diameter at every z.
    """
    rows = _profile(profile)
    if isinstance(segments, bool) or int(segments) != segments or segments < 8:
        raise ValueError('At least eight integral angular segments are required')
    if not math.isfinite(density_kg_m3) or density_kg_m3 <= 0:
        raise ValueError('Density must be finite and positive')
    count = int(segments)
    angle = 2 * math.pi / count
    inner_factor = 1 / math.cos(math.pi / count)
    area_factor = count * math.sin(angle) / 2
    polar_factor = count * math.sin(angle) * (2 + math.cos(angle)) / 12
    volume = first_z = second_z = polar = 0.
    for lower, upper in zip(rows[:-1], rows[1:]):
        height = upper[0] - lower[0]
        inner = Polynomial([lower[1], (upper[1] - lower[1]) / height]) * inner_factor
        outer = Polynomial([lower[2], (upper[2] - lower[2]) / height])
        z = Polynomial([lower[0], 1.])
        area = area_factor * (outer ** 2 - inner ** 2)
        def integral(poly): return float(poly.integ()(height) - poly.integ()(0.))
        volume += integral(area)
        first_z += integral(z * area)
        second_z += integral(z ** 2 * area)
        polar += integral(polar_factor * (outer ** 4 - inner ** 4))
    mass = density_kg_m3 * volume
    center_z = first_z / volume
    axial = density_kg_m3 * polar
    transverse = axial / 2 + density_kg_m3 * second_z - mass * center_z ** 2
    return {'volume_m3': volume, 'mass_kg': mass, 'center_of_mass_m': [0., 0., center_z],
            'inertia_diag_kg_m2': [transverse, transverse, axial]}


def revolved_meshes(profile, *, segments=SEGMENTS, split_axial=False):
    """Return a display surface and independently convex collision pieces."""
    import trimesh
    rows = _profile(profile)
    polygonal_mass_properties(rows, segments=segments)
    inner_factor = 1 / math.cos(math.pi / segments)
    boundary = [(r, z) for z, _, r in rows]
    boundary += [(r * inner_factor, z) for z, r, _ in rows[::-1]]
    display = lathe_mesh(boundary, segments=segments)
    pieces = []
    if split_axial:
        # Preserve the original display surface and mass profile. Only the
        # collision union is partitioned at existing profile breakpoints.
        for lower,upper in zip(rows[:-1],rows[1:]):
            _,cells=revolved_meshes([lower,upper],segments=segments)
            pieces.extend(cells)
    elif np.all(rows[:, 1] == 0):
        # Keep every original boundary vertex. A single 64-gon cylinder has
        # 128 vertices and 64 vertices on each cap; quarter sectors preserve
        # its exact solid while keeping individual hulls inside GPU limits.
        for indices in np.array_split(np.arange(segments), 4):
            angles = np.r_[indices, indices[-1] + 1] * 2 * math.pi / segments
            points = []
            for z, _, outer in rows:
                points.append([0., 0., z])
                points.extend(np.c_[outer * np.cos(angles), outer * np.sin(angles),
                                    np.full(len(angles), z)])
            pieces.append(trimesh.convex.convex_hull(np.asarray(points)))
    else:
        if np.any(rows[:, 1] == 0):
            raise ValueError('A changing zero-radius bore must be split into separate pieces')
        for index in range(segments):
            angles = np.array([index, index + 1]) * 2 * math.pi / segments
            points = []
            for z, inner, outer in rows:
                for radius in (inner * inner_factor, outer):
                    points.extend(np.c_[radius * np.cos(angles), radius * np.sin(angles),
                                        np.full(2, z)])
            pieces.append(trimesh.convex.convex_hull(np.asarray(points)))
    if not display.is_watertight or any(not p.is_watertight or not p.is_convex for p in pieces):
        raise ValueError('Procedural solid is not watertight or has nonconvex collision pieces')
    return display, pieces


def convex_topology(mesh):
    """Count coplanar polygon faces, not triangulated OBJ face records."""
    from scipy.spatial import ConvexHull
    hull = ConvexHull(mesh.vertices)
    polygons = {}
    for equation, triangle in zip(hull.equations, hull.simplices):
        # Hull equations are normalized. Mesh coordinates are metres; this
        # groups numerical triangulation of one plane without merging facets.
        key = tuple(np.round(equation, decimals=10))
        polygons.setdefault(key, set()).update(map(int, triangle))
    return {'vertices': len(hull.vertices), 'polygons': len(polygons),
            'vertices_per_polygon': max(map(len, polygons.values()))}


def make_asset(asset_id, *, dynamic, density_kg_m3=DENSITY_PRIOR_KG_M3):
    asset = Asset(asset_id, dynamic=dynamic, mass_kg=1., friction=.4)
    asset.data.update(density_kg_m3=float(density_kg_m3),
        density_provenance='Synthetic uniform-density engineering prior; not a measured material.',
        material_prior=dict(MATERIAL_PRIOR),
        contact_offset_m=CONTACT_OFFSET_M, rest_offset_m=0.,
        geometry_features=[], mass_property_components=[])
    return asset


def add_revolved(asset, label, profile, *, offset=(0., 0., 0.), color='plastic_blue', segments=SEGMENTS,
                 split_axial=False):
    rows = _profile(profile)
    display, pieces = revolved_meshes(rows, segments=segments, split_axial=split_axial)
    display.apply_translation(offset)
    asset.add_display(display, color, label)
    for index, piece in enumerate(pieces):
        piece.apply_translation(offset)
        data = {'geometry': 'convex_mesh', 'position_m': [0., 0., 0.]}
        if np.all(rows[:, 1] > 0):
            data.update(bore=True, bore_radius_m=float(rows[:, 1].min()),
                        clearance_radius_m=float(rows[:, 1].min()),
                        bore_axis_xy_m=list(offset[:2]))
        asset.add_collision(piece, f'{label}_{index:03d}', **data)
    mass = polygonal_mass_properties(rows, density_kg_m3=asset.data['density_kg_m3'], segments=segments)
    mass['center_of_mass_m'] = (np.asarray(mass['center_of_mass_m']) + offset).tolist()
    asset.data['mass_property_components'].append(mass)
    asset.data['geometry_features'].append({'label': label, 'kind': 'regular_polygon_revolved_profile',
        'profile_z_inner_apothem_outer_circumradius_m': rows.tolist(),
        'offset_m': list(offset), 'segments': segments, 'collision_parts': len(pieces),
        'outer_radial_faceting_error_bound_m': float(rows[:, 2].max() * (1-math.cos(math.pi/segments))),
        'bore_radius_is_guaranteed_minimum': bool(np.all(rows[:, 1] > 0))})
    if split_axial:
        asset.data['geometry_features'][-1]['collision_split_at_profile_z_breakpoints']=True
    return asset


def add_box(asset, label, size, *, center, color='steel'):
    size = np.asarray(size, dtype=float)
    if size.shape != (3,) or not np.isfinite(size).all() or np.any(size <= 0):
        raise ValueError('Box dimensions must be positive')
    asset.add_part({'geometry': 'box', 'size_m': size.tolist(), 'color': color}, label=label, offset=center)
    volume = float(np.prod(size)); mass = volume * asset.data['density_kg_m3']
    asset.data['mass_property_components'].append({'volume_m3': volume, 'mass_kg': mass,
        'center_of_mass_m': list(center),
        'inertia_diag_kg_m2': (mass/12*(np.sum(size**2)-size**2)).tolist()})
    asset.data['geometry_features'].append({'label':label,'kind':'box','size_m':size.tolist(),'center_m':list(center)})
    return asset


def finalize_mass(asset):
    """Combine nonoverlapping piece integrals with the parallel-axis theorem."""
    components = asset.data['mass_property_components']
    mass = sum(x['mass_kg'] for x in components)
    center = sum(x['mass_kg'] * np.asarray(x['center_of_mass_m']) for x in components) / mass
    tensor = np.zeros((3,3))
    for item in components:
        delta = np.asarray(item['center_of_mass_m']) - center
        tensor += np.diag(item['inertia_diag_kg_m2']) + item['mass_kg'] * (np.dot(delta,delta)*np.eye(3)-np.outer(delta,delta))
    if np.max(np.abs(tensor-np.diag(np.diag(tensor)))) > 1e-10:
        raise ValueError('Loader supports diagonal principal inertia; arrange the solid on its symmetry axes')
    asset.mass_kg = float(mass);asset.center_of_mass_m = center.tolist()
    asset.data.update(inertia_diag_kg_m2=np.diag(tensor).tolist(),
        inertia_method='Exact regular-polygon cross-section polynomial integrals plus parallel-axis theorem; uniform design density.',
        volume_m3=sum(x['volume_m3'] for x in components),
        origin='Declared geometric datum; COM explicitly authored independently.',
        mass_properties_are_measured=False,
        kinematic_fixture=not asset.dynamic)
    asset.data['cpu_geometry_audit'] = audit_geometry(asset)
    return asset


def audit_geometry(asset):
    """Independently integrate actual meshes and reject a filled/nonconvex bore."""
    import trimesh
    from scipy.spatial import ConvexHull
    collision = [mesh for _, mesh, _ in asset.collision]
    if any(mesh is None or not mesh.is_watertight or not mesh.is_convex for mesh in collision):
        raise ValueError('Every collision cell must be an explicit watertight convex mesh')
    display = trimesh.util.concatenate([mesh for _, mesh, _ in asset.visual])
    volume = sum(mesh.volume for mesh in collision)
    if not np.isclose(volume, asset.data['volume_m3'], rtol=1e-9, atol=1e-15):
        raise ValueError('Collision volume differs from exact profile integral; bore may be filled')
    topology = [convex_topology(mesh) for mesh in collision]
    maxima = {key: max(row[key] for row in topology) for key in GPU_CONVEX_LIMITS}
    if any(maxima[key] > limit for key, limit in GPU_CONVEX_LIMITS.items()):
        raise ValueError('Source convex topology exceeds conservative GPU convex envelope')
    density = asset.data['density_kg_m3']
    if not np.isclose(display.volume*density, asset.mass_kg, rtol=1e-9, atol=1e-12):
        raise ValueError('Visual-solid mass and collision integral disagree')
    com_error = float(np.max(np.abs(display.center_mass-asset.center_of_mass_m)))
    inertia_error = float(np.max(np.abs(display.moment_inertia*density-np.diag(asset.data['inertia_diag_kg_m2']))))
    if com_error>1e-10 or inertia_error>1e-10:
        raise ValueError('Independent mesh COM/inertia integral differs from declared mass properties')
    halfspaces=[ConvexHull(mesh.vertices).equations for mesh in collision]
    checked=0
    for feature in asset.data['geometry_features']:
        if not feature.get('bore_radius_is_guaranteed_minimum'): continue
        rows=np.asarray(feature['profile_z_inner_apothem_outer_circumradius_m'])
        # Exclude physical end faces. For blind sockets the floor is intentional.
        for z in np.linspace(rows[0,0]+1e-6, rows[-1,0]-1e-6, 9):
            radius=float(np.interp(z,rows[:,0],rows[:,1]))-1e-6
            for angle in np.arange(2*feature['segments'])*math.pi/feature['segments']:
                point=np.array([radius*math.cos(angle),radius*math.sin(angle),z])+feature['offset_m']
                if any(np.all(eq[:,:3]@point+eq[:,3]<=1e-10) for eq in halfspaces):
                    raise ValueError('A collision hull intrudes into the required open channel')
                checked+=1
    return {'passed':True,'scope':'Source mesh geometry/integrals only; not native cooking or contact qualification.',
        'collision_parts':len(collision),'collision_volume_m3':float(volume),
        'source_convex_topology_max':maxima,
        'conservative_gpu_convex_limits':dict(GPU_CONVEX_LIMITS),
        'cooked_gpu_eligibility_verified':False,
        'independent_mesh_com_max_error_m':com_error,
        'independent_mesh_inertia_max_error_kg_m2':inertia_error,
        'open_channel_points_checked':checked}


def write_assets(output, assets, *, generator, parameters, source_paths):
    """Create a fresh immutable candidate library and its provenance."""
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError('Use a fresh asset version: ' + str(output))
    for _, asset in assets: finalize_mass(asset)
    source = {'schema':'mechanical_procedural_asset_source.v1','generator':generator,
        'sources_sha256':{str(Path(p).resolve()):hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in source_paths},
        'parameters':parameters,'units':'metres','source_kind':'original_procedural',
        'license':'CC0-1.0','material_prior':MATERIAL_PRIOR,
        'contact_offsets':{'contact_m':CONTACT_OFFSET_M,'rest_m':0.,
            'scope':'Must also be applied through explicit scene object/fixture parameters and checked natively.'},
        'mass_scope':'Synthetic density, exact mass/inertia for the polygonal source collision solids; not hardware calibration.',
        'collision_scope':'Union of individually convex cells, never a global hull across a bore; native cooking unvalidated.',
        'native_validated':False}
    result = write_library(output, assets, generator=generator,
        extra={'qualification':'prepared_unvalidated','physics_accepted':False,
               'source_provenance':'SOURCE.json','source_metadata':source})
    (output/'SOURCE.json').write_text(json.dumps(source,ensure_ascii=False,indent=2)+'\n')
    (output/'LICENSE.txt').write_text('Original procedural geometry: CC0 1.0 Universal.\nhttps://creativecommons.org/publicdomain/zero/1.0/\n')
    return result


def physical_spec(asset_id, asset_root, object_id, position):
    return {'id':object_id,'asset_id':asset_id,'asset_root':str(Path(asset_root).resolve()),
            'position_m':list(position),'quaternion_wxyz':[1.,0.,0.,0.],
            'friction':.4,'contact_offset_m':CONTACT_OFFSET_M,'rest_offset_m':0.}
