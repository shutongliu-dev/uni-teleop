"""Small original-procedural steel hardware for workshop table clutter.

These are passive rigid bodies with real bores and chamfers, not grasp targets
or manufacturer CAD. Density is an explicit engineering prior. Mass and inertia
are exact for the authored polygonal solids; native contact is unqualified.
"""
from pathlib import Path
import math

import numpy as np
from numpy.polynomial import Polynomial

from bench.paths import DATA_ROOT
from bench.simulation.assets import scene_parts
from . import assembly_geometry as geometry


ASSET_ROOT = Path(f'{DATA_ROOT}/assets/task_parts/mechanical_assembly/table_hardware/v1')
STEEL_DENSITY_KG_M3 = 7850.
NUT_ID = 'mechanical_table_hex_nut'
WASHER_ID = 'mechanical_table_washer'
BOLT_ID = 'mechanical_table_short_bolt'


def _regular_loft_integrals(z, circumradii, sides):
    """Volume, first/second Z moment and polar area moment of a polygon loft."""
    angle = 2 * math.pi / sides
    area_factor = sides * math.sin(angle) / 2
    polar_factor = sides * math.sin(angle) * (2 + math.cos(angle)) / 12
    totals = np.zeros(4)
    for lower, upper, r0, r1 in zip(z[:-1], z[1:], circumradii[:-1], circumradii[1:]):
        height = upper - lower
        radius = Polynomial([r0, (r1-r0)/height])
        position = Polynomial([lower, 1.])
        area = area_factor * radius**2
        terms = (area, position*area, position**2*area, polar_factor*radius**4)
        totals += [float(p.integ()(height)-p.integ()(0.)) for p in terms]
    return totals


def _mass_from_integrals(integrals):
    volume, first_z, second_z, polar = integrals
    mass = STEEL_DENSITY_KG_M3 * volume
    center_z = first_z / volume
    axial = STEEL_DENSITY_KG_M3 * polar
    transverse = axial/2 + STEEL_DENSITY_KG_M3*second_z - mass*center_z**2
    return {'volume_m3': float(volume), 'mass_kg': float(mass),
            'center_of_mass_m': [0., 0., float(center_z)],
            'inertia_diag_kg_m2': [float(transverse), float(transverse), float(axial)]}


def _nut():
    import trimesh
    asset = geometry.make_asset(NUT_ID, dynamic=True, density_kg_m3=STEEL_DENSITY_KG_M3)
    # The outer six flats are exact. Four angular cells per flat preserve a
    # 24-sided open bore; inward radial faces are never bridged by a global hull.
    count = 24
    z = np.array([0., .0008, .0092, .010])
    outer_af = np.array([.0182, .019, .019, .0182])
    inner_apothem = np.array([.00685, .0065, .0065, .00685])
    angles = np.arange(count) * 2*math.pi/count
    normals = math.pi/6 + np.arange(6)*math.pi/3
    outer_radii = outer_af[:, None]/2 / np.cos(angles[:, None]-normals).max(axis=1)
    inner_radii = np.broadcast_to(inner_apothem[:, None]/math.cos(math.pi/count), (len(z), count))
    rings = []
    for radii in (outer_radii, inner_radii):
        rings.append(np.stack([radii*np.cos(angles), radii*np.sin(angles),
                               np.broadcast_to(z[:, None], radii.shape)], axis=-1))
    vertices = np.array(rings).reshape(-1, 3)
    def index(side, level, angle): return side*len(z)*count + level*count + angle % count
    faces = []
    def quad(a,b,c,d,reverse=False):
        faces.extend(((a,c,b),(a,d,c)) if reverse else ((a,b,c),(a,c,d)))
    for i in range(count):
        for j in range(len(z)-1):
            for side in (0,1):
                quad(index(side,j,i),index(side,j,i+1),index(side,j+1,i+1),index(side,j+1,i),reverse=side==1)
        for j in (0,len(z)-1):
            quad(index(0,j,i),index(0,j,i+1),index(1,j,i+1),index(1,j,i),reverse=j==0)
    display = trimesh.Trimesh(vertices=vertices, faces=faces, process=True)
    if not display.is_watertight or not display.is_winding_consistent or display.volume <= 0:
        raise ValueError('Nut must be a closed positive-volume bored surface')
    asset.add_display(display, 'steel', 'hex_nut_with_open_bore')
    for i in range(count):
        points = np.concatenate([rings[s][:, [i,(i+1)%count]].reshape(-1,3) for s in (0,1)])
        cell = trimesh.convex.convex_hull(points)
        asset.add_collision(cell, f'hex_nut_sector_{i:02d}', geometry='convex_mesh',
                            position_m=[0.,0.,0.], bore=True, bore_radius_m=.0065,
                            clearance_radius_m=.0065, bore_axis_xy_m=[0.,0.])
    integrals = (_regular_loft_integrals(z, outer_af/(2*math.cos(math.pi/6)), 6)
                 - _regular_loft_integrals(z, inner_apothem/math.cos(math.pi/count), count))
    asset.data['mass_property_components'].append(_mass_from_integrals(integrals))
    asset.data['geometry_features'].append({'kind':'hexagonal_outer_polygon_with_open_polygonal_bore',
        'label':'hex_nut', 'outer_across_flats_m':.019, 'bore_minimum_diameter_m':.013,
        'height_m':.010, 'segments':count, 'bore_radius_is_guaranteed_minimum':True,
        # Shared audit uses this only to test the open channel at every height.
        'profile_z_inner_apothem_outer_circumradius_m':np.c_[z,inner_apothem,outer_af/(2*math.cos(math.pi/6))].tolist(),
        'offset_m':[0.,0.,0.], 'thread_geometry':'Unthreaded clearance bore; not a mating thread model.'})
    return asset


def _washer():
    asset = geometry.make_asset(WASHER_ID, dynamic=True, density_kg_m3=STEEL_DENSITY_KG_M3)
    return geometry.add_revolved(asset, 'chamfered_open_washer',
        [[0.,.0068,.0137],[.0003,.0065,.014],[.0022,.0065,.014],[.0025,.0068,.0137]],
        segments=32, color='dark_steel')


def _bolt():
    import trimesh
    asset = geometry.make_asset(BOLT_ID, dynamic=True, density_kg_m3=STEEL_DENSITY_KG_M3)
    geometry.add_revolved(asset, 'short_shank', [[0.,0.,.0048],[.0012,0.,.006],[.024,0.,.006]],
                          segments=32, color='steel')
    z = np.array([.024,.0247,.0313,.032])
    radii = np.array([.018,.019,.019,.018])/(2*math.cos(math.pi/6))
    angles = np.arange(6)*math.pi/3
    points = np.concatenate([np.c_[radius*np.cos(angles),radius*np.sin(angles),np.full(6,height)]
                             for height,radius in zip(z,radii)])
    head = trimesh.convex.convex_hull(points)
    asset.add_display(head, 'steel', 'chamfered_hex_head')
    asset.add_collision(head, 'hex_head', geometry='convex_mesh', position_m=[0.,0.,0.])
    asset.data['mass_property_components'].append(_mass_from_integrals(_regular_loft_integrals(z,radii,6)))
    asset.data['geometry_features'].append({'kind':'chamfered_hex_head', 'across_flats_m':.019,
        'head_height_m':.008, 'shank_diameter_m':.012, 'shank_length_m':.024,
        'thread_geometry':'Plain short shank; no helical mating/contact qualification.'})
    return asset


def prepare_assets(output=ASSET_ROOT):
    """Write a fresh versioned library; refuse any existing destination."""
    assets = [(asset.asset_id, asset) for asset in (_nut(), _washer(), _bolt())]
    return geometry.write_assets(output, assets,
        generator='scenes.workrooms.mechanical_assembly.distractor_assets',
        parameters={'density_kg_m3':STEEL_DENSITY_KG_M3,
            'density_scope':'Explicit uniform steel-like design prior; not a hardware measurement.',
            'count_per_task':5, 'roles':['hex_nut','washer','short_hex_head_bolt'],
            'dimensions_scope':'Original small hardware proportions, not a specified commercial fastener standard.',
            'native_contact_qualified':False},
        source_paths=[__file__, geometry.__file__, scene_parts.__file__])
