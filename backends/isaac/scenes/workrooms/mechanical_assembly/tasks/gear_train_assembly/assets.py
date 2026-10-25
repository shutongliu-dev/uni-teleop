"""Original, bore-preserving involute gears and free hand tools for L5.

All dimensions are simulation design values. Convex pieces tile the actual
faceted tooth profiles; there is no convex hull around a complete gear, no
filled bore, and no gear/hinge constraint. Native contact remains unqualified.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from bench.geometry import matrix_quaternion
from bench.paths import DATA_ROOT
from bench.simulation.assets.scene_parts import Asset, write_library


ASSET_ROOT = Path(f'{DATA_ROOT}/assets/task_parts/mechanical_assembly/gear_train_assembly/v3_4')
LEGACY_ASSET_ROOT = ASSET_ROOT.with_name('v2')
STEEL_DENSITY = 7850.
HANDLE_DENSITY = 1150.
GEOMETRY = {
    'module_m': .005, 'teeth': 20, 'pressure_angle_degrees': 20.,
    'pitch_radius_m': .05, 'base_radius_m': .05*math.cos(math.radians(20.)),
    'outside_radius_m': .055, 'root_radius_m': .04375,
    'face_width_m': .018, 'bore_radius_m': .012,
    'entry_radius_m': .019, 'entry_depth_m': .010,
    'hub_height_m': .022, 'hub_radius_m': .025, 'input_hub_across_flats_m': .054,
    'pair_tangential_backlash_m': .0008, 'center_distance_m': .100,
    'shaft_radius_m': .0117, 'shaft_tip_radius_m': .005,
    'shaft_cylinder_top_m': .062, 'shaft_tip_top_m': .078,
    'workplate_thickness_m': .018, 'thrust_top_m': .022,
    'involute_flank_intervals': 12, 'root_tooth_sector_intervals': 4,
    'root_gap_intervals': 3, 'tooth_tip_arc_intervals': 4,
}
GEAR_TEETH = {'input_gear': 20, 'idler_gear': 32, 'output_gear': 20}
TRAIN_GEOMETRY = {
    'version': 'v3_large_idler_20_32_20', 'module_m': .005,
    'gear_teeth': deepcopy(GEAR_TEETH), 'adjacent_center_distance_m': .130,
    'shaft_centers_local_xy_m': [[0., -.130], [0., 0.], [0., .130]],
    'workplate_size_m': [.200, .410, .018],
    'gear_supply_positions_xy_m': [[.21, -.20], [.25, .32], [.21, .10]],
    'supply_support': 'Directly on the original table; no display pads or supply rests',
    'signed_angle_ratios': {'idler_per_input': -.625, 'output_per_input': 1.},
}


def gear_geometry(teeth=20):
    """Use the same module/pressure angle; a larger wheel is not a scaled mesh."""
    if type(teeth) is not int or teeth not in (20, 32):
        raise ValueError('Only the frozen 20/32-tooth profiles are supported')
    g = deepcopy(GEOMETRY)
    # A centre distance belongs to a gear pair, not to either wheel profile.
    # GEOMETRY retains the old 20/20 diagnostic fixture's value separately.
    g.pop('center_distance_m')
    if teeth == 20:
        # Preserve the original literal float values and mass integration
        # bit-for-bit; recomputing .05 - 1.25*.005 differs by one float ULP.
        return g
    radius = g['module_m']*teeth/2
    g.update(teeth=teeth, pitch_radius_m=radius,
             base_radius_m=radius*math.cos(math.radians(g['pressure_angle_degrees'])),
             outside_radius_m=radius+g['module_m'], root_radius_m=radius-1.25*g['module_m'])
    return g
SOURCE_REFERENCES = [
    {'url': 'https://khkgears.net/gear-knowledge/gear-technical-reference/calculation-gear-dimensions/',
     'scope': 'Standard unshifted spur gear size relations: d=m*z, addendum=m, whole depth=2.25*m. No source mesh copied.'},
    {'url': 'https://www.khkgears.us/media/1224/a-primer-on-backlash-and-its-purpose-in-gear-designs.pdf',
     'scope': 'Backlash comes from the sum of pitch-circle tooth thickness reductions; each gear here loses 0.4 mm.'},
]


def _hull(points):
    import trimesh
    mesh = trimesh.convex.convex_hull(np.asarray(points, dtype=float))
    if not mesh.is_watertight or not mesh.is_convex or not math.isfinite(mesh.volume) or mesh.volume <= 1.e-15:
        raise ValueError('Every collision piece must be a closed, positive-volume convex solid')
    return mesh


def _prism(polygon, bottom, top):
    """Extrude an already convex 2-D polygon; reject any lost concavity."""
    polygon = np.asarray(polygon, dtype=float)
    first = np.roll(polygon, -1, axis=0)-polygon
    second = np.roll(polygon, -2, axis=0)-np.roll(polygon, -1, axis=0)
    cross = first[:, 0]*second[:, 1]-first[:, 1]*second[:, 0]
    if not (np.all(cross >= -1.e-13) or np.all(cross <= 1.e-13)):
        raise ValueError('Nonconvex tooth band cannot be replaced by a convex hull')
    mesh = _hull(np.vstack([np.c_[polygon, np.full(len(polygon), z)] for z in (bottom, top)]))
    area = abs(np.dot(polygon[:, 0], np.roll(polygon[:, 1], -1))
               -np.dot(polygon[:, 1], np.roll(polygon[:, 0], -1)))/2
    if abs(mesh.volume-area*(top-bottom)) > max(1.e-14, mesh.volume*1.e-8):
        raise ValueError('Convex piece changed the declared tooth profile')
    return mesh


def _polar(radius, angle):
    return np.array([radius*math.cos(angle), radius*math.sin(angle)])


def _involute(angle):
    return math.tan(angle)-angle


def tooth_half_angle(radius, geometry=None):
    """Angular half-thickness, including half of the pair's backlash."""
    g = GEOMETRY if geometry is None else geometry
    at_pitch = math.pi/(2*g['teeth'])-g['pair_tangential_backlash_m']/(4*g['pitch_radius_m'])
    at_base = at_pitch+_involute(math.radians(g['pressure_angle_degrees']))
    angle = math.acos(min(1., g['base_radius_m']/radius))
    return at_base-_involute(angle)


def tooth_polygons(*, split_tip=False, geometry=None):
    """Exact 2-D polygons used by the native convex tooth pieces."""
    g = GEOMETRY if geometry is None else geometry
    radii = [g['root_radius_m'], *np.linspace(g['base_radius_m'], g['outside_radius_m'],
                                           g['involute_flank_intervals']+1)]
    polygons = []
    for tooth in range(g['teeth']):
        center = tooth*2*math.pi/g['teeth']
        for low, high in zip(radii, radii[1:]):
            a, b = tooth_half_angle(low, g), tooth_half_angle(high, g)
            polygons.append(np.array([_polar(low, center-a), _polar(high, center-b),
                                      _polar(high, center+b), _polar(low, center+a)]))
        tip = tooth_half_angle(g['outside_radius_m'], g)
        angles = np.linspace(center-tip, center+tip, g['tooth_tip_arc_intervals']+1)
        tip_polygon = np.array([_polar(g['outside_radius_m'], a) for a in angles])
        if split_tip:
            polygons.append(tip_polygon)
        else:
            # The last band and tip share their entire chord. Their exact
            # union is convex: preserve every outer-arc vertex and both lower
            # band vertices, eliminating the 21-micrometre-thin isolated cap.
            band = polygons.pop()
            merged = np.vstack([band[:1], tip_polygon, band[-1:]])
            _prism(merged, 0., g['face_width_m'])  # Reject any lost concavity.
            def area(p):
                return abs(np.dot(p[:, 0], np.roll(p[:, 1], -1))
                    -np.dot(p[:, 1], np.roll(p[:, 0], -1)))/2
            if abs(area(merged)-area(band)-area(tip_polygon)) > 1.e-14:
                raise ValueError('Tip union changed the original tooth boundary')
            polygons.append(merged)
    return polygons


def pair_overlap_count(angle_a, angle_b, *, distance=.100, geometry_a=None, geometry_b=None):
    """CPU convex-polygon SAT on actual faceted teeth; no physics inference."""
    return _pair_polygon_overlap_count(tooth_polygons(geometry=geometry_a),
        tooth_polygons(geometry=geometry_b), angle_a, angle_b, distance)


def _pair_polygon_overlap_count(polygons_a, polygons_b, angle_a, angle_b, distance):
    def transformed(polygons, angle, offset):
        c, s = math.cos(angle), math.sin(angle)
        rotation = np.array([[c, -s], [s, c]])
        return [p @ rotation.T+offset for p in polygons]
    first = transformed(polygons_a, angle_a, np.zeros(2))
    second = transformed(polygons_b, angle_b, np.array([0., distance]))
    other_low = np.array([p.min(axis=0) for p in second])
    other_high = np.array([p.max(axis=0) for p in second])
    overlaps = 0
    for a in first:
        low, high = a.min(axis=0), a.max(axis=0)
        candidates = np.flatnonzero(np.all(high > other_low+1.e-10, axis=1)
                                    & np.all(other_high > low+1.e-10, axis=1))
        for index in candidates:
            b = second[index]
            edges = np.vstack([np.roll(a, -1, axis=0)-a, np.roll(b, -1, axis=0)-b])
            axes = np.c_[-edges[:, 1], edges[:, 0]]
            axes /= np.linalg.norm(axes, axis=1)[:, None]
            aa, bb = a @ axes.T, b @ axes.T
            depth = np.minimum(aa.max(axis=0), bb.max(axis=0))-np.maximum(aa.min(axis=0), bb.min(axis=0))
            overlaps += int(np.all(depth > 1.e-10))
    return overlaps


def geometry_audit():
    """Standard proportions and exact discretized mesh compatibility at reset."""
    g = GEOMETRY
    samples = np.linspace(0., 2*math.pi/g['teeth'], 73)
    polygons = tooth_polygons()
    overlaps = [_pair_polygon_overlap_count(polygons, polygons, a, math.pi/g['teeth']-a, .1) for a in samples]
    flank_errors = []
    radii = np.linspace(g['base_radius_m'], g['outside_radius_m'], g['involute_flank_intervals']+1)
    for low, high in zip(radii, radii[1:]):
        a, b = _polar(low, tooth_half_angle(low)), _polar(high, tooth_half_angle(high))
        direction = b-a
        for radius in np.linspace(low, high, 33):
            point = _polar(radius, tooth_half_angle(radius))
            fraction = np.clip(np.dot(point-a, direction)/np.dot(direction, direction), 0., 1.)
            flank_errors.append(float(np.linalg.norm(point-(a+fraction*direction))))
    report = {'schema': 'mechanical_gear_cpu_geometry_audit.v1',
        'passed': not any(overlaps) and max(flank_errors) <= 2.e-5,
        'profile': deepcopy(g), 'full_tooth_cycle_samples': len(samples),
        'maximum_intersecting_tooth_piece_pairs': max(overlaps),
        'tip_to_tip_negative_control_overlaps': pair_overlap_count(0., 0.),
        'maximum_sampled_involute_chord_error_m': max(flank_errors),
        'pitch_tooth_thickness_m': 2*g['pitch_radius_m']*tooth_half_angle(g['pitch_radius_m']),
        'nominal_pair_backlash_m': math.pi*g['module_m']-4*g['pitch_radius_m']*tooth_half_angle(g['pitch_radius_m']),
        'nominal_radial_root_clearance_m': g['center_distance_m']-g['outside_radius_m']-g['root_radius_m'],
        'minimum_radial_shaft_clearance_m': g['bore_radius_m']-g['shaft_radius_m'],
        'scope': 'CPU facet geometry at nominal centers, with a collision negative control. Does not establish native cooked contacts, torque transfer, numerical convergence or robot manipulation.'}
    report['passed'] &= report['tip_to_tip_negative_control_overlaps'] > 0
    return report


def train_geometry_audit():
    """Check both unequal-tooth meshes through an input tooth period."""
    small, large = gear_geometry(20), gear_geometry(32)
    distance = TRAIN_GEOMETRY['adjacent_center_distance_m']
    samples = np.linspace(0., 2*math.pi/20, 73)
    phase = math.pi/32
    records = {}
    polygons = {g['teeth']: tooth_polygons(geometry=g) for g in (small, large)}
    for name, first, second, angles in (
            ('input_idler', small, large, [(a, phase-a*20/32) for a in samples]),
            ('idler_output', large, small, [(phase-a*20/32, a) for a in samples])):
        overlaps = [_pair_polygon_overlap_count(polygons[first['teeth']], polygons[second['teeth']],
                                                a, b, distance) for a, b in angles]
        records[name] = {
            'teeth': [first['teeth'], second['teeth']], 'center_distance_m': distance,
            'sample_count': len(samples), 'maximum_intersecting_tooth_piece_pairs': max(overlaps),
            'tip_to_tip_negative_control_overlaps': pair_overlap_count(0., 0., distance=distance,
                geometry_a=first, geometry_b=second),
            'radial_tip_root_clearance_m': min(distance-first['outside_radius_m']-second['root_radius_m'],
                                               distance-second['outside_radius_m']-first['root_radius_m']),
            'nominal_pair_tangential_backlash_m': math.pi*first['module_m']
                -2*first['pitch_radius_m']*tooth_half_angle(first['pitch_radius_m'], first)
                -2*second['pitch_radius_m']*tooth_half_angle(second['pitch_radius_m'], second)}
    errors = {}
    for g in (small, large):
        values = []
        radii = np.linspace(g['base_radius_m'], g['outside_radius_m'], g['involute_flank_intervals']+1)
        for low, high in zip(radii, radii[1:]):
            a, b = _polar(low, tooth_half_angle(low, g)), _polar(high, tooth_half_angle(high, g))
            direction = b-a
            for radius in np.linspace(low, high, 33):
                point = _polar(radius, tooth_half_angle(radius, g))
                fraction = np.clip(np.dot(point-a, direction)/np.dot(direction, direction), 0., 1.)
                values.append(float(np.linalg.norm(point-(a+fraction*direction))))
        errors[str(g['teeth'])] = max(values)
    return {'schema': 'mechanical_unequal_gear_cpu_audit.v1',
        'passed': all(r['maximum_intersecting_tooth_piece_pairs'] == 0
                      and r['tip_to_tip_negative_control_overlaps'] > 0 for r in records.values())
                  and max(errors.values()) <= 2.e-5,
        'train': deepcopy(TRAIN_GEOMETRY), 'pairs': records,
        'maximum_sampled_involute_chord_error_m_by_teeth': errors,
        'scope': 'Actual faceted profiles sampled through one input tooth period, with wrong-phase overlap controls; no native contact or transmission qualification.'}


def _root_boundary(geometry=None):
    g = GEOMETRY if geometry is None else geometry
    half = tooth_half_angle(g['root_radius_m'], g)
    angles, radii = [], []
    pitch = 2*math.pi/g['teeth']
    for tooth in range(g['teeth']):
        center = tooth*pitch
        for relative in np.linspace(-half, half, g['root_tooth_sector_intervals']+1)[:-1]:
            angles.append(center+relative)
            radii.append(g['root_radius_m']*math.cos(half)/math.cos(relative))
        for angle in np.linspace(center+half, center+pitch-half, g['root_gap_intervals']+1)[:-1]:
            angles.append(angle)
            radii.append(g['root_radius_m'])
    return np.array(angles), np.array(radii)


def _ring_pieces(angles, outer_radii, inner_profile):
    """Convex ring sectors whose inner chords never enter the declared bore."""
    closed_angles = np.r_[angles, angles[0]+2*math.pi]
    factor = 1/math.cos(float(np.diff(closed_angles).max())/2)
    result = []
    for i, (a, b) in enumerate(zip(closed_angles, closed_angles[1:])):
        j = (i+1) % len(angles)
        points = []
        for z, inner_radius in inner_profile:
            for angle, outer in ((a, outer_radii[i]), (b, outer_radii[j])):
                points.extend([[*_polar(inner_radius*factor, angle), z], [*_polar(outer, angle), z]])
        result.append(_hull(points))
    return result, factor


def _hex_ring(outer_af, inner_af, bottom, top, *, sections=6):
    angles = np.arange(sections)*2*math.pi/sections
    # Both hexes use vertices along +X; across-flats dimension is exact.
    outer = outer_af/(2*math.cos(math.pi/sections))
    inner = inner_af/2
    return _ring_pieces(angles, np.full(sections, outer), [(bottom, inner), (top, inner)])[0]


def _add(asset, mesh, label, density=STEEL_DENSITY, color='steel'):
    asset.add_display(mesh, color, label).add_collision(mesh, label, geometry='convex_mesh', position_m=[0., 0., 0.])
    asset._physical_pieces.append((mesh, float(density), label))


def _asset(name, dynamic=True):
    asset = Asset(name, dynamic=dynamic, mass_kg=1. if dynamic else 0., friction=.3)
    asset._physical_pieces = []
    return asset


def _finish(asset):
    """Integrate nonoverlapping solid pieces; preserve non-diagonal inertia."""
    if not asset.dynamic:
        asset.data.update(inertia_method='Fixed fixture; no dynamic inertia qualification',
                          material_prior={'friction': .3, 'source': 'Explicit uncalibrated scene simulation prior'})
        return asset
    pieces = [(mesh.mass_properties, density, label) for mesh, density, label in asset._physical_pieces]
    masses = np.array([p.volume*d for p, d, _ in pieces])
    mass = float(masses.sum())
    center = sum(m*p.center_mass for m, (p, _, _) in zip(masses, pieces))/mass
    inertia = np.zeros((3, 3))
    for m, (p, density, _) in zip(masses, pieces):
        delta = p.center_mass-center
        inertia += density*p.inertia+m*(np.dot(delta, delta)*np.eye(3)-np.outer(delta, delta))
    values, axes = np.linalg.eigh(inertia)
    if np.linalg.det(axes) < 0:
        axes[:, 0] *= -1
    if not np.all(values > 0) or values[0]+values[1] < values[2]-1.e-12:
        raise ValueError('Invalid integrated mass inertia')
    quaternion = matrix_quaternion(axes)[[1, 2, 3, 0]].tolist()
    asset.mass_kg = mass
    asset.center_of_mass_m = center.tolist()
    asset.data.update(
        inertia_diag_kg_m2=values.tolist(), principal_axes_xyzw=quaternion,
        inertia_body_frame_kg_m2=inertia.tolist(), principal_axes_columns=axes.tolist(),
        inertia_method='Volume integrals of nonoverlapping closed convex solids, with parallel-axis theorem; simulation densities, not measured',
        density_priors_kg_m3={'steel': STEEL_DENSITY, 'polymer_handle': HANDLE_DENSITY},
        material_prior={'friction': .3, 'source': 'Explicit uncalibrated scene simulation prior'},
        integrated_piece_count=len(pieces), solid_volume_m3=float(sum(p.volume for p, _, _ in pieces)),
        mass_components=[{'label': label, 'volume_m3': float(p.volume), 'density_kg_m3': density,
                          'mass_kg': float(p.volume*density)} for p, density, label in pieces])
    return asset


def gear_asset(name, *, input_hub=False, teeth=20):
    g = gear_geometry(teeth)
    asset = _asset(name)
    angles, radii = _root_boundary(g)
    root, factor = _ring_pieces(angles, radii,
        [(0., g['entry_radius_m']), (g['entry_depth_m'], g['bore_radius_m']),
         (g['face_width_m'], g['bore_radius_m'])])
    for i, mesh in enumerate(root):
        _add(asset, mesh, f'root_web_{i:03d}')
    for i, polygon in enumerate(tooth_polygons(geometry=g)):
        _add(asset, _prism(polygon, 0., g['face_width_m']), f'tooth_band_{i:03d}')
    sections = 48
    hub_angles = np.arange(sections)*2*math.pi/sections
    if input_hub:
        # Intersections of 48 radial rays with the regular hex boundary.
        apothem = g['input_hub_across_flats_m']/2
        normals = np.arange(6)*math.pi/3+math.pi/6
        outer = np.array([apothem/max(math.cos(a-n) for n in normals) for a in hub_angles])
    else:
        outer = np.full(sections, g['hub_radius_m'])
    hubs, _ = _ring_pieces(hub_angles, outer,
        [(g['face_width_m'], g['bore_radius_m']),
         (g['face_width_m']+g['hub_height_m'], g['bore_radius_m'])])
    for i, mesh in enumerate(hubs):
        _add(asset, mesh, f'grip_hub_{i:03d}', color='brass' if input_hub else 'steel')
    # Paint only; no extra mass or collision and no change to the bore.
    import trimesh
    marker = trimesh.creation.box(extents=[.012, .003, .0001])
    marker.apply_translation([.036, 0., g['face_width_m']+.00005])
    asset.add_display(marker, 'plastic_red', 'output_reference_mark')
    asset.data.update(gear_geometry=deepcopy(g), body_origin='Bottom face, shaft axis at local XY=0',
        bore_minimum_radius_m=g['bore_radius_m'], bore_faceting_maximum_radius_error_m=g['bore_radius_m']*(1/math.cos(math.pi/48)-1),
        root_web_bore_factor=factor,
        root_profile='Radial relief below the base circle, faceted root arc; loaded nominal involute starts above base radius. No rack-generated fillet claim.',
        tooth_collision='Each radial band is independently checked convex and volume-preserving; final band includes its exact convex tip-arc union. No complete-tooth or complete-gear hull.',
        decomposition_revision='v3: same-module 20/32-tooth profiles, retaining the v2 exact convex tip union; each body mass and inertia reintegrated from its actual solids')
    return _finish(asset)


def _box(size, position):
    import trimesh
    mesh = trimesh.creation.box(extents=size)
    mesh.apply_translation(position)
    return mesh


def _round_sections(radius, bottom, top, xy=(0., 0.), *, top_radius=None, sections=96):
    """Tile the same faceted cylinder/cone in four GPU-sized convex sectors.

    Each cap has at most 26 vertices, each hull at most 52. This preserves the
    circular boundary without the >64-vertex CPU contact fallback in PhysX.
    """
    if sections % 4 or sections > 96:
        raise ValueError('Round profiles require a multiple of four, at most 96 facets')
    top_radius = radius if top_radius is None else top_radius
    result = []
    for quarter in range(4):
        angles = np.linspace(quarter*math.pi/2, (quarter+1)*math.pi/2, sections//4+1)
        points = []
        for z, r in ((bottom, radius), (top, top_radius)):
            points.append([*xy, z])
            points.extend([[r*math.cos(a)+xy[0], r*math.sin(a)+xy[1], z] for a in angles])
        result.append(_hull(points))
    return result


def workplate_asset():
    g = GEOMETRY
    asset = _asset('gear_workplate', False)
    _add(asset, _box(TRAIN_GEOMETRY['workplate_size_m'], [0., 0., .009]), 'workplate')
    for i, (_, y) in enumerate(TRAIN_GEOMETRY['shaft_centers_local_xy_m']):
        for label, radius, bottom, top, top_radius in (
                ('thrust', .039, .018, .022, .039),
                ('guide_shaft', g['shaft_radius_m'], .022, .062, g['shaft_radius_m']),
                ('tapered_tip', .0117, .062, .078, .005)):
            for part, mesh in enumerate(_round_sections(radius, bottom, top, (0., y), top_radius=top_radius)):
                _add(asset, mesh, f'{label}_{i}_{part}')
    asset.data.update(shaft_centers_local_xy_m=deepcopy(TRAIN_GEOMETRY['shaft_centers_local_xy_m']),
                      shaft_geometry=deepcopy(g), constraints='Fixed fixture only; gears have no joints or angular constraints')
    return _finish(asset)


def socket_crank_asset():
    asset = _asset('socket_crank')
    # A 3-mm entry chamfer opens from 62 to 58 mm AF; remaining cavity is 58 mm AF.
    angles = np.arange(6)*math.pi/3
    walls, _ = _ring_pieces(angles, np.full(6, .072/(2*math.cos(math.pi/6))),
                            [(0., .031), (.003, .029), (.025, .029)])
    for i, mesh in enumerate(walls):
        _add(asset, mesh, f'hex_socket_wall_{i}')
    # Circular 26 mm through bore in the roof, not a closed cap over the shaft.
    angles = np.arange(48)*2*math.pi/48
    normals = np.arange(6)*math.pi/3+math.pi/6
    outer = np.array([.036/max(math.cos(a-n) for n in normals) for a in angles])
    roof, _ = _ring_pieces(angles, outer, [(.025, .013), (.032, .013)])
    for i, mesh in enumerate(roof):
        _add(asset, mesh, f'through_bore_roof_{i:02d}')
    _add(asset, _box([.057, .022, .012], [-.0585, 0., .038]), 'crank_lever')
    for i, mesh in enumerate(_round_sections(.017, .044, .154, (-.07, 0.), sections=64)):
        _add(asset, mesh, f'polymer_handle_{i}', density=HANDLE_DENSITY, color='plastic_blue')
    asset.data.update(body_origin='Bottom socket opening center, +Z along socket/shaft; lever points -X',
        socket_geometry={'inner_across_flats_m': .058, 'entry_across_flats_m': .062,
            'entry_depth_m': .003, 'cavity_depth_m': .025, 'roof_top_m': .032,
            'roof_through_bore_diameter_m': .026, 'outer_across_flats_m': .072,
            'handle_diameter_m': .034, 'handle_length_m': .110, 'lever_radius_m': .070})
    return _finish(asset)


def wrench_asset():
    asset = _asset('open_end_wrench')
    # Open +X mouth, 58-mm jaw separation. All regions meet without overlap.
    for label, size, center in [
        ('upper_jaw', [.060, .012, .012], [.005, .035, .006]),
        ('lower_jaw', [.060, .012, .012], [.005, -.035, .006]),
        ('jaw_back', [.015, .082, .012], [-.0325, 0., .006]),
        ('handle', [.150, .032, .012], [-.115, 0., .006])]:
        _add(asset, _box(size, center), label)
    asset.data.update(body_origin='Jaw axis at XY=0, lower face Z=0; mouth opens +X',
                      jaw_across_flats_m=.058, handle_width_m=.032, thickness_m=.012)
    return _finish(asset)


def rest_asset(name, dimensions):
    asset = _asset(name, False)
    _add(asset, _box(dimensions, [0., 0., dimensions[2]/2]), 'open_rest', color='dark_steel')
    return _finish(asset)


def make_assets():
    assets = [gear_asset('input_gear', input_hub=True), gear_asset('idler_gear', teeth=32),
              gear_asset('output_gear'), workplate_asset(), socket_crank_asset(), wrench_asset()]
    return [(asset.asset_id, asset) for asset in assets]


def prepare_assets(output=ASSET_ROOT):
    """Author a fresh shared version; never overwrite an asset or old evidence."""
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError('Choose a new gear asset version: '+str(output))
    audit = train_geometry_audit()
    if not audit['passed']:
        raise ValueError('The faceted tooth cycle or negative control failed its CPU geometry audit')
    assets = make_assets()
    from .clearance import tool_sweep_audit
    sweep = tool_sweep_audit(dict(assets))
    if not sweep['passed']:
        raise ValueError('The upright tool full-yaw enclosure does not clear the new geometry')
    source = {'schema': 'mechanical_gear_geometry_source.v1', 'source_kind': 'original_procedural',
              'license': 'CC0-1.0', 'source_references': deepcopy(SOURCE_REFERENCES),
              'generator_path': str(Path(__file__).resolve()),
              'generator_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'geometry': deepcopy(GEOMETRY), 'train_geometry': deepcopy(TRAIN_GEOMETRY),
              'gear_geometries': {name: gear_geometry(teeth) for name, teeth in GEAR_TEETH.items()},
              'cpu_geometry_audit': audit,
              'cpu_continuous_tool_sweep_audit': sweep,
              'clearance_helper_sha256': hashlib.sha256(Path(__file__).with_name('clearance.py').read_bytes()).hexdigest(),
              'physical_densities_are_simulation_priors': True,
              'collision_geometry': 'Faceted involute tooth bands plus bore-preserving convex ring sectors; no hidden joints.',
              'native_physics_validated': False, 'robot_manipulation_validated': False,
              'existing_screwdriver': 'Referenced separately from pinned threaded_sharpa/v3; not copied or relicensed.'}
    result = write_library(output, assets, generator=__name__+'.prepare_assets', extra={'source': source})
    (output/'SOURCE.json').write_text(json.dumps(source, indent=2, allow_nan=False)+'\n')
    (output/'LICENSE.txt').write_text('Original procedural geometry dedicated to the public domain under CC0-1.0.\nhttps://creativecommons.org/publicdomain/zero/1.0/\nReferenced third-party screwdriver keeps its original source license.\n')
    return result
