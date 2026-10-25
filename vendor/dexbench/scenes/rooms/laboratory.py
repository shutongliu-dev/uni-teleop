"""The ``laboratory`` scene family: a chemistry / biology bench with real glassware.

What is shipped (asset library ``assets/laboratory/v2``; ``v1`` is kept untouched
as the evidence of the version it replaces):

* Dynamic objects (authored OBJ assets, all with a positive design mass, a
  uniform-density bounding-box inertia and only convex collision pieces):
  twelve 12 x 100 mm round-bottom test tubes standing in a two-tier rack, eight
  72 mm 1000 ul pipette tips standing point-down in a tip rack, a 50 / 150 /
  250 ml beaker trio, a 125 ml Erlenmeyer flask, a 50 ml graduated cylinder on a
  hexagonal foot, a hexagonal glass stirring rod lying flat, and a cork flask
  ring (the one real through bore in this family).
* Static props (immovable primitive boxes): the bench carcass with cabinet doors
  under a black epoxy top, end lips, a reagent riser, the tube rack and the tip
  rack (both drilled through, see below), a hot plate / stirrer, a fume hood, a
  sink unit with an eyewash, a framed and glazed window, a door casing, a safety
  shower, a wall shelf, a yellow flammables cabinet, a first-aid box, wall
  notices and signs, a waste bin and a lab stool.
* Fixtures (static authored assets): reagent bottles on the wall shelf and the
  riser, a micropipette on its stand, and a fire extinguisher.

Collision geometry, as built by ``scene.add_task_asset`` (OBJ vertices are used
as written and then moved by the part's ``position_m``; every convex piece is
therefore written centred on its own bounding box and placed by ``position_m``,
so no offset is ever applied twice):

* Open vessels (beakers, flask, graduated cylinder) are blind cavities, not
  through holes, so they declare no bore. Each has a solid floor piece (a convex
  solid of revolution over the heel, or the hexagonal foot) and a wall of convex
  sectors whose outer vertices lie on the visible glass and whose inner faces
  are tangent to a declared ``cavity_radius_m``. The collision wall is grown
  INWARD to >= 3 mm (the visible glass is 1.2 - 1.5 mm), so the collision mouth
  is a few millimetres narrower than the visible mouth; the floor top is the
  visible inner floor. The flask is a hollow cone: four conical wall stages and a
  straight neck stage.
* The cork ring is the one through hole: sixteen convex sectors declared
  ``bore: true`` around a 25 mm radius bore, each sector's inner face tangent to
  the bore, so no piece seals it.
* Test tube: an analytic cylinder plus an analytic sphere (the round bottom).
  Pipette tip: a convex frustum (body) plus an analytic cylinder (collar); the
  tip's internal channel is not modelled (its narrowest passage is the 0.6 mm
  orifice and no pipette nozzle is simulated). Stirring rod: the convex hull of
  its own hexagonal prism, flat face down.
* Every convex piece is at least 3 mm across in every direction.

Racks: the tube rack is a base plate (the declared support the tubes stand on),
two end walls and an upper plate of rails and ribs leaving a 14 x 14 mm hole
around each tube 62 - 66 mm above the base, so a tube can lean about 1 degree
before the plate holds it. The tip rack is a floor plate (the declared support
the tip points stand on), four walls and a deck of rails and ribs leaving a
closed 7.4 x 7.4 mm hole per tip; each tip's 9 mm collar rests on the deck
around its hole at exactly the same time as its point touches the floor, so the
tip is held like a real rack holds it and its lowest point is on a declared
support.

Supports declared to ``family.support_surfaces``: the ``lab_bench`` static prop
declares the work-table top (the table box is the only collider there - no
duplicate slab is built), the tube rack declares its base plate and the tip rack
its floor plate (both real parts of the same prop).

Masses are design values for a simulation task, not manufacturer measurements;
each asset records its source. Visual roughness and metallic are written into
each manifest visual part (``write_library`` has no field for them and
``scene.add_task_asset`` would otherwise bind 0.35 / 0.65, i.e. metallic glass).

Shared-code note: ``room.add_room`` binds room surfaces by the surface key
(``materials['floor']``) and ignores ``room.surfaces``; this family therefore
names its room materials ``floor``/``wall``/``ceiling``/``baseboard`` (and the
pedestal's ``top``/``side``) so they are actually bound.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np

from bench.paths import DATA_ROOT, load_json
from scenes.rooms import DEFAULT_LIBRARY_ROOT
from bench.simulation.assets.scene_parts import Asset, annulus_mesh, box_mesh, cylinder_mesh, lathe_mesh, write_library

FAMILY = 'laboratory'
ASSET_VERSION = 'v2'
ASSET_BASE = DEFAULT_LIBRARY_ROOT / FAMILY
ASSET_ROOT = str(ASSET_BASE / ASSET_VERSION)
# v1 is superseded, not deleted: the tests use it as a known-bad control
# (version history: ASSET_BASE/../VERSIONS.md).
SUPERSEDED_ASSET_ROOTS = {'v1': str(ASSET_BASE / 'v1')}
V1_MANIFEST_SHA256 = '695bfef9c4dfa3ea91d306cc544c3e186e314b3446b00965f6f04725a41d1327'
GENERATOR = 'scenes/rooms/laboratory.py'
LICENSE_ID = 'CC0-1.0'

GLASS_DENSITY_KG_M3 = 2230.0          # borosilicate 3.3
VISUAL_SEGMENTS = 96                  # every sector count below divides it
MIN_COLLISION_WIDTH_M = 0.003         # rule 4: nothing a contact can approach is thinner
COLLISION_WALL_M = 0.0033             # radial wall of every vessel collider (>= 3 mm)

# --- room ----------------------------------------------------------------------
ROOM_SIZE_M = (5.0, 4.0, 2.7)
ROOM_HALF = (ROOM_SIZE_M[0] / 2., ROOM_SIZE_M[1] / 2.)
WALL_THICKNESS_M = 0.12

# --- the work bench: the table box is the only collider of the bench top ------
TABLE_X = (0.30, 1.10)
TABLE_Y = (-0.30, 0.70)
TABLE_TOP_Z = 0.750
TABLE_THICKNESS_M = 0.030
TABLE_CENTER = ((TABLE_X[0] + TABLE_X[1]) / 2., (TABLE_Y[0] + TABLE_Y[1]) / 2.,
                TABLE_TOP_Z - TABLE_THICKNESS_M / 2.)
TABLE_HALF = ((TABLE_X[1] - TABLE_X[0]) / 2., (TABLE_Y[1] - TABLE_Y[0]) / 2., TABLE_THICKNESS_M / 2.)
TABLE_BOTTOM_Z = TABLE_TOP_Z - TABLE_THICKNESS_M

# --- racks (static props) -------------------------------------------------------
TUBE_RACK_CENTER = (0.620, 0.380)
TUBE_PITCH_M = 0.020
TUBE_COLUMNS, TUBE_ROWS = 6, 2
TUBE_RACK_HALF = (0.063, 0.023)
TUBE_RACK_BASE_M = 0.004
TUBE_RACK_TOP_Z = TABLE_TOP_Z + TUBE_RACK_BASE_M            # 0.754: tubes stand here
TUBE_HOLE_HALF_M = 0.007                                    # 14 x 14 mm openings
TUBE_PLATE_BOTTOM_Z = TUBE_RACK_TOP_Z + 0.062
TUBE_PLATE_TOP_Z = TUBE_PLATE_BOTTOM_Z + 0.004

TIP_RACK_CENTER = (0.440, 0.380)
TIP_PITCH_M = 0.013
TIP_COLUMNS, TIP_ROWS = 4, 2
TIP_RACK_HALF = (0.030, 0.017)
TIP_RACK_FLOOR_M = 0.004
TIP_RACK_FLOOR_TOP_Z = TABLE_TOP_Z + TIP_RACK_FLOOR_M       # 0.754: tip points stand here
TIP_HOLE_HALF_M = 0.0037                                    # 7.4 x 7.4 mm closed holes
TIP_RACK_WALL_M = 0.003

# --- authored dimensions (metres) ----------------------------------------------
TUBE = {'radius': 0.006, 'height': 0.100, 'wall': 0.0010, 'mass': 0.012}
TIP = {'length': 0.072, 'collar_radius': 0.0045, 'collar_height': 0.010,
       'body_top_radius': 0.0033, 'point_radius': 0.0005, 'mass': 0.0008}
TIP_DECK_TOP_Z = TIP_RACK_FLOOR_TOP_Z + TIP['length'] - TIP['collar_height']   # 0.816
TIP_DECK_BOTTOM_Z = TIP_DECK_TOP_Z - 0.004

BEAKERS = {   # ISO 3819 low form: 50 ml 42 x 62, 150 ml 60 x 82, 250 ml 70 x 97 (max height)
    'beaker_50ml': {'outer': 0.0210, 'height': 0.060, 'wall': 0.0012, 'mass': 0.035},
    'beaker_150ml': {'outer': 0.0300, 'height': 0.080, 'wall': 0.0013, 'mass': 0.070},
    'beaker_250ml': {'outer': 0.0350, 'height': 0.095, 'wall': 0.0015, 'mass': 0.100},
}
VESSEL_FLOOR_M = 0.0032       # visible inner floor height == collision floor top
VESSEL_HEEL_M = 0.0030        # heel radius at the base
BEAKER_SECTORS = 24
FLASK = {'base_outer': 0.0335, 'height': 0.115, 'neck_outer': 0.0135, 'neck_height': 0.025,
         'wall': 0.0015, 'mass': 0.085, 'cone_stages': 4, 'cone_sectors': 24, 'neck_sectors': 16,
         'cone_wall_horizontal': 0.0034}   # 3.31 mm normal to the 13 deg cone wall
GRAD_CYLINDER = {'outer': 0.0120, 'height': 0.200, 'wall': 0.0014, 'foot_circumradius': 0.033,
                 'foot_height': 0.010, 'mass': 0.100, 'sectors': 16}
STIR_ROD = {'across_flats': 0.007, 'length': 0.200}
STIR_ROD['circumradius'] = STIR_ROD['across_flats'] / 2. / math.cos(math.pi / 6.)
STIR_ROD['mass'] = round(1.5 * math.sqrt(3.) * STIR_ROD['circumradius'] ** 2 * STIR_ROD['length']
                         * GLASS_DENSITY_KG_M3, 4)          # 0.0189 kg
CORK_RING = {'outer': 0.040, 'inner': 0.025, 'height': 0.025, 'bevel': 0.001, 'sectors': 16,
             'mass': 0.015}
BOTTLES = {
    'reagent_bottle_amber_500ml': {'radius': 0.043, 'body_top': 0.115, 'neck_radius': 0.0225,
                                   'shoulder_top': 0.150, 'cap_radius': 0.025, 'cap_top': 0.178,
                                   'glass': 'amber_glass', 'cap': 'cap_blue'},
    'reagent_bottle_clear_250ml': {'radius': 0.035, 'body_top': 0.088, 'neck_radius': 0.016,
                                   'shoulder_top': 0.115, 'cap_radius': 0.0175, 'cap_top': 0.138,
                                   'glass': 'bottle_glass', 'cap': 'cap_red'},
}
MICROPIPETTE_SIZE = (0.100, 0.090, 0.300)
EXTINGUISHER = {'radius': 0.075, 'body_top': 0.400, 'dome_top': 0.450, 'valve_radius': 0.018,
                'valve_top': 0.500, 'lever_top': 0.515}

# Visual appearance of authored parts. ``write_library`` records only a colour,
# so build_library writes these roughness/metallic values into each visual part.
APPEARANCE = {
    'frosted_glass': {'color': [0.84, 0.89, 0.91], 'roughness': 0.45, 'metallic': 0.0},
    'graduation_white': {'color': [0.95, 0.95, 0.95], 'roughness': 0.60, 'metallic': 0.0},
    'tip_polypropylene': {'color': [0.28, 0.50, 0.78], 'roughness': 0.45, 'metallic': 0.0},
    'cork': {'color': [0.66, 0.50, 0.33], 'roughness': 0.92, 'metallic': 0.0},
    'amber_glass': {'color': [0.42, 0.22, 0.07], 'roughness': 0.25, 'metallic': 0.0},
    'bottle_glass': {'color': [0.80, 0.86, 0.86], 'roughness': 0.30, 'metallic': 0.0},
    'cap_blue': {'color': [0.15, 0.32, 0.62], 'roughness': 0.50, 'metallic': 0.0},
    'cap_red': {'color': [0.68, 0.16, 0.14], 'roughness': 0.50, 'metallic': 0.0},
    'pipette_body': {'color': [0.88, 0.89, 0.90], 'roughness': 0.45, 'metallic': 0.0},
    'pipette_accent': {'color': [0.20, 0.40, 0.75], 'roughness': 0.40, 'metallic': 0.0},
    'stand_grey': {'color': [0.35, 0.37, 0.40], 'roughness': 0.50, 'metallic': 0.20},
    'extinguisher_red': {'color': [0.72, 0.08, 0.06], 'roughness': 0.35, 'metallic': 0.10},
    'extinguisher_black': {'color': [0.06, 0.06, 0.06], 'roughness': 0.50, 'metallic': 0.0},
    'chrome': {'color': [0.75, 0.76, 0.78], 'roughness': 0.20, 'metallic': 0.90},
}
GLASSWARE = ('test_tube', 'beaker_50ml', 'beaker_150ml', 'beaker_250ml', 'erlenmeyer_flask',
             'graduated_cylinder', 'stirring_rod')


# --- small geometry helpers ------------------------------------------------------

def _r(value):
    return round(float(value), 6)


def _hull(points):
    import trimesh
    return trimesh.convex.convex_hull(np.asarray(points, dtype=float))


def _ring_points(radius, z, count, phase=0.0):
    angles = phase + np.arange(count) * 2. * math.pi / count
    return np.c_[radius * np.cos(angles), radius * np.sin(angles), np.full(count, float(z))]


def _revolved_hull(levels, segments=VISUAL_SEGMENTS):
    """Convex hull of rings ``(radius, z)``: exact when radius is concave in z."""
    return _hull(np.vstack([_ring_points(r, z, segments) for r, z in levels]))


def _wall_sector(levels, index, count):
    """One convex sector of a revolved wall, between two adjacent sector angles.

    ``levels`` is a list of ``(z, r_outer, r_inner)``. Outer vertices lie on the
    outer (visible) surface; inner vertices are pushed out to
    ``r_inner / cos(pi / count)`` so the sector's inner face is tangent to the
    ``r_inner`` circle at the sector's mid angle: its exact distance from the
    axis is ``r_inner`` and no smaller. A sector whose inner vertex would reach
    its outer one is refused (it would turn inside out and cross the axis side).
    """
    half = math.pi / count
    angles = (2. * half * index, 2. * half * (index + 1))
    points = []
    for z, r_outer, r_inner in levels:
        r_vertex = r_inner / math.cos(half)
        if not r_vertex < r_outer - 1e-4:
            raise ValueError(f'sector inner vertex {r_vertex:.5f} m is not inside its outer '
                             f'radius {r_outer:.5f} m')
        for angle in angles:
            points.append((r_outer * math.cos(angle), r_outer * math.sin(angle), z))
            points.append((r_vertex * math.cos(angle), r_vertex * math.sin(angle), z))
    return _hull(points)


def _profile_moments(profile):
    """Volume and centroid z of the solid of revolution of a closed (r, z) polygon."""
    p = np.asarray(profile, dtype=float)
    r, z = p[:, 0], p[:, 1]
    r1, z1 = np.roll(r, -1), np.roll(z, -1)
    cross = r * z1 - r1 * z
    a_r = float(np.sum((r + r1) * cross) / 6.)
    a_rz = float(np.sum(cross * (r * (2. * z + z1) + r1 * (z + 2. * z1))) / 24.)
    return 2. * math.pi * abs(a_r), a_rz / a_r


def _combined_com(pieces):
    """``pieces`` = [(volume, centroid_z)]; the volume-weighted centroid z."""
    volume = sum(v for v, _ in pieces)
    return sum(v * z for v, z in pieces) / volume


def _heel(radius, z_bottom, heel, steps=6):
    """Points of a quarter-round heel from (radius - heel, z_bottom) to (radius, z_bottom + heel)."""
    centre_r, centre_z = radius - heel, z_bottom + heel
    return [(centre_r + heel * math.cos(theta), centre_z + heel * math.sin(theta))
            for theta in np.linspace(-math.pi / 2., 0., steps + 1)]


def _display(asset, mesh, appearance, label):
    asset.add_display(mesh, APPEARANCE[appearance]['color'], label)
    asset.data.setdefault('visual_appearance', {})[label] = appearance


def _convex(asset, label, mesh, **component):
    """Add a convex collision piece written centred on its own box, placed by position_m.

    ``scene.add_task_asset`` loads the OBJ vertices as written and then moves the
    piece by ``position_m``; writing the mesh already shifted *and* recording
    the shift in ``position_m`` would apply it twice (the v1 flask defect).
    """
    lo, hi = np.asarray(mesh.bounds, dtype=float)
    centre = (lo + hi) / 2.
    piece = mesh.copy()
    piece.apply_translation(-centre)
    asset.add_collision(piece, label, geometry='convex_mesh',
                        position_m=[float(v) for v in centre], **component)


def _cylinder(asset, label, radius, z0, z1, **component):
    asset.add_collision(None, label, geometry='cylinder', radius_m=float(radius),
                        half_height_m=float((z1 - z0) / 2.), position_m=[0., 0., float((z0 + z1) / 2.)],
                        **component)


def _box_collision(asset, label, lo, hi):
    lo, hi = np.asarray(lo, dtype=float), np.asarray(hi, dtype=float)
    asset.add_collision(None, label, geometry='box', half_extents_m=[float(v) for v in (hi - lo) / 2.],
                        position_m=[float(v) for v in (hi + lo) / 2.])


def _box_display(asset, label, lo, hi, appearance):
    lo, hi = np.asarray(lo, dtype=float), np.asarray(hi, dtype=float)
    mesh = box_mesh(hi - lo)
    mesh.apply_translation((hi + lo) / 2.)
    _display(asset, mesh, appearance, label)


def _vessel_wall(asset, label, levels, count, *, role='wall'):
    """A revolved wall as ``count`` convex sectors; returns their exact clearance."""
    clearance = min(r_inner for _, _, r_inner in levels)
    for index in range(count):
        _convex(asset, f'{label}_{index:02d}', _wall_sector(levels, index, count),
                sector=f'{index:02d}/{count}', role=role, clearance_radius_m=float(clearance))
    return clearance


# --- dynamic assets -----------------------------------------------------------

def _test_tube():
    """12 x 100 mm round-bottom tube; collider = analytic cylinder + sphere (the bottom)."""
    radius, height, wall = TUBE['radius'], TUBE['height'], TUBE['wall']
    half = height / 2.
    centre_z = -half + radius
    inner = radius - wall
    outer_arc = [(radius * math.sin(a), centre_z - radius * math.cos(a))
                 for a in np.linspace(0., math.pi / 2., 13)]
    inner_arc = [(inner * math.sin(a), centre_z - inner * math.cos(a))
                 for a in np.linspace(math.pi / 2., 0., 13)]
    profile = outer_arc + [(radius, half), (inner, half)] + inner_arc
    asset = Asset('test_tube', dynamic=True, mass_kg=TUBE['mass'], friction=0.45,
                  center_of_mass_m=(0., 0., _profile_moments(profile)[1]))
    _display(asset, lathe_mesh(profile, segments=48), 'frosted_glass', 'tube')
    _cylinder(asset, 'body', radius, centre_z, half, role='body')
    asset.add_collision(None, 'round_bottom', geometry='sphere', radius_m=float(radius),
                        position_m=[0., 0., float(centre_z)], role='round_bottom')
    asset.data.update(
        nominal_size_m=DESIGN_SIZES['test_tube'],
        size_note='12 mm outside diameter, 100 mm long, 1 mm wall, hemispherical bottom',
        mass_source='Design value from the brief (a test tube 0.012 kg)',
        mass_design_value_kg=TUBE['mass'], closed_bottom=True, open_top=True,
        outer_profile_m=[[0., -half]] + [[float(r), float(z)] for r, z in outer_arc] + [[radius, half]],
        collision_note=('Analytic cylinder over the straight body plus an analytic sphere for the '
                        'round bottom: the collider is the visible tube, closed (no handled bore).'),
        rest_note='Stands on its round bottom on the rack base plate; the upper plate keeps it upright')
    return asset


def _pipette_tip():
    """A 1000 ul tip, 72 mm long with a 9 mm collar; stands point down in its rack."""
    length = TIP['length']
    z0, z1 = -length / 2., length / 2.
    z_collar = z1 - TIP['collar_height']
    outer_r, top_r, point_r = TIP['collar_radius'], TIP['body_top_radius'], TIP['point_radius']
    profile = [(0.0003, z0), (point_r, z0), (top_r, z_collar), (outer_r, z_collar), (outer_r, z1),
               (0.0038, z1), (0.0029, z_collar - 0.001), (0.0003, z0 + 0.001)]
    asset = Asset('pipette_tip', dynamic=True, mass_kg=TIP['mass'], friction=0.45,
                  center_of_mass_m=(0., 0., _profile_moments(profile)[1]))
    _display(asset, lathe_mesh(profile, segments=48), 'tip_polypropylene', 'tip')
    _convex(asset, 'body', _revolved_hull([(point_r, z0), (top_r, z_collar)], segments=48), role='body')
    _cylinder(asset, 'collar', outer_r, z_collar, z1, role='collar')
    asset.data.update(
        nominal_size_m=DESIGN_SIZES['pipette_tip'],
        size_note='1000 ul tip: 72 mm long, 9 mm collar 10 mm tall, 6.6 mm body under the collar',
        mass_source=('Design value 0.8 g, as quoted for a 72 x 9 mm 1000 ul tip (Sarstedt 70.3050); '
                     'not weighed here'),
        mass_design_value_kg=TIP['mass'], collar_underside_z_m=float(z_collar),
        outer_profile_m=[[point_r, z0], [top_r, z_collar], [outer_r, z_collar], [outer_r, z1]],
        collision_note=('Convex frustum for the body plus an analytic cylinder for the collar. The '
                        'tip is open at both ends, but its narrowest passage is the 0.6 mm orifice '
                        'and no pipette nozzle is simulated, so the channel is not modelled.'),
        rest_note=('Point down: the point stands on the tip-rack floor while the collar rests on '
                   'the deck around its hole'))
    return asset


def _beaker(asset_id):
    """Griffin low-form beaker: floor piece over the heel + 24 wall sectors grown inward."""
    spec = BEAKERS[asset_id]
    outer, height, wall = spec['outer'], spec['height'], spec['wall']
    z_bottom, z_top = -height / 2., height / 2.
    z_floor = z_bottom + VESSEL_FLOOR_M
    heel = _heel(outer, z_bottom, VESSEL_HEEL_M)
    profile = [(0., z_bottom)] + heel + [(outer, z_top), (outer - wall, z_top), (outer - wall, z_floor),
                                         (0., z_floor)]
    volume, com_z = _profile_moments(profile)
    asset = Asset(asset_id, dynamic=True, mass_kg=spec['mass'], friction=0.42,
                  center_of_mass_m=(0., 0., com_z))
    _display(asset, lathe_mesh(profile, segments=VISUAL_SEGMENTS), 'frosted_glass', 'body')
    _convex(asset, 'floor', _revolved_hull(heel + [(outer, z_floor)]), role='floor')
    cavity = outer * math.cos(math.pi / BEAKER_SECTORS) - COLLISION_WALL_M
    _vessel_wall(asset, 'wall', [(z_floor, outer, cavity), (z_top, outer, cavity)], BEAKER_SECTORS)
    asset.data.update(
        nominal_size_m=DESIGN_SIZES[asset_id],
        size_note=(f'{asset_id.split("_")[1]} low-form beaker: {2 * outer * 1000:.0f} mm outside '
                   f'diameter, {height * 1000:.0f} mm tall, {wall * 1000:.1f} mm visible wall, '
                   f'{VESSEL_FLOOR_M * 1000:.1f} mm floor'),
        mass_source=(f'Design value (borosilicate shell of this geometry is about '
                     f'{volume * GLASS_DENSITY_KG_M3 * 1000:.0f} g; the brief quotes a beaker at 0.10 kg)'),
        mass_design_value_kg=spec['mass'], open_top=True, closed_bottom=True,
        cavity_radius_m=float(cavity), cavity_floor_z_m=float(z_floor),
        visual_inner_radius_m=float(outer - wall),
        outer_profile_m=[[0., z_bottom]] + [[float(r), float(z)] for r, z in heel] + [[outer, z_top]],
        collision_note=('Blind cavity, not a through hole: a solid floor piece plus 24 convex wall '
                        'sectors (no bore declared). Outer vertices lie on the visible glass.'),
        collision_visual_difference=(f'The collision wall is {COLLISION_WALL_M * 1000:.1f} mm thick, '
                                     f'grown inward from the {wall * 1000:.1f} mm glass, so the collision '
                                     f'mouth radius is {cavity * 1000:.2f} mm against a visible '
                                     f'{(outer - wall) * 1000:.2f} mm; the floor top is the visible floor.'))
    return asset


def _erlenmeyer_flask():
    """125 ml narrow-mouth Erlenmeyer: hollow cone (4 conical stages) + neck + floor."""
    spec = FLASK
    base, height, neck = spec['base_outer'], spec['height'], spec['neck_outer']
    wall = spec['wall']
    z_bottom, z_top = -height / 2., height / 2.
    z_floor = z_bottom + VESSEL_FLOOR_M
    z_cone0 = z_bottom + VESSEL_HEEL_M
    z_shoulder = z_top - spec['neck_height']

    def cone(z):
        return base + (neck - base) * (z - z_cone0) / (z_shoulder - z_cone0)

    heel = _heel(base, z_bottom, VESSEL_HEEL_M)
    profile = ([(0., z_bottom)] + heel +
               [(neck, z_shoulder), (neck, z_top), (neck - wall, z_top), (neck - wall, z_shoulder),
                (cone(z_floor) - wall, z_floor), (0., z_floor)])
    volume, com_z = _profile_moments(profile)
    asset = Asset('erlenmeyer_flask', dynamic=True, mass_kg=spec['mass'], friction=0.42,
                  center_of_mass_m=(0., 0., com_z))
    _display(asset, lathe_mesh(profile, segments=VISUAL_SEGMENTS), 'frosted_glass', 'body')
    _convex(asset, 'floor', _revolved_hull(heel + [(cone(z_floor), z_floor)]), role='floor')
    clearances = []
    count = spec['cone_sectors']
    for stage in range(spec['cone_stages']):
        za = z_floor + (z_shoulder - z_floor) * stage / spec['cone_stages']
        zb = z_floor + (z_shoulder - z_floor) * (stage + 1) / spec['cone_stages']
        levels = [(z, cone(z), cone(z) * math.cos(math.pi / count) - spec['cone_wall_horizontal'])
                  for z in (za, zb)]
        clearances.append(_vessel_wall(asset, f'cone_{stage:02d}', levels, count))
    neck_count = spec['neck_sectors']
    neck_inner = neck * math.cos(math.pi / neck_count) - COLLISION_WALL_M
    clearances.append(_vessel_wall(asset, 'neck', [(z_shoulder, neck, neck_inner),
                                                   (z_top, neck, neck_inner)], neck_count))
    asset.data.update(
        nominal_size_m=DESIGN_SIZES['erlenmeyer_flask'],
        size_note=(f'125 ml narrow mouth: {2 * base * 1000:.0f} mm base, {height * 1000:.0f} mm tall, '
                   f'{2 * neck * 1000:.0f} mm neck ({2 * (neck - wall) * 1000:.0f} mm bore), '
                   f'{spec["neck_height"] * 1000:.0f} mm neck height'),
        mass_source=(f'Design value (borosilicate shell of this geometry is about '
                     f'{volume * GLASS_DENSITY_KG_M3 * 1000:.0f} g)'),
        mass_design_value_kg=spec['mass'], open_top=True, closed_bottom=True,
        cavity_radius_m=float(min(clearances)), cavity_floor_z_m=float(z_floor),
        visual_inner_radius_m=float(neck - wall),
        outer_profile_m=([[0., z_bottom]] + [[float(r), float(z)] for r, z in heel] +
                         [[neck, z_shoulder], [neck, z_top]]),
        collision_note=('Blind hollow cone: a solid floor piece, four conical wall stages of 24 '
                        'sectors whose inner faces follow the cone, and a 16-sector neck. No bore '
                        'is declared (the flask is closed at the bottom).'),
        collision_visual_difference=('Collision walls are grown inward to >= 3 mm, so the collision '
                                     f'mouth radius is {min(clearances) * 1000:.2f} mm against a visible '
                                     f'{(neck - wall) * 1000:.2f} mm.'))
    return asset


def _graduated_cylinder():
    """50 ml graduated cylinder on a hexagonal foot; the foot is the cavity floor."""
    spec = GRAD_CYLINDER
    outer, height, wall = spec['outer'], spec['height'], spec['wall']
    foot_r, foot_h = spec['foot_circumradius'], spec['foot_height']
    z_bottom, z_top = -height / 2., height / 2.
    z_floor = z_bottom + foot_h
    hexagon = [(foot_r * math.cos(k * math.pi / 3.), foot_r * math.sin(k * math.pi / 3.))
               for k in range(6)]
    foot = _hull([(x, y, z) for x, y in hexagon for z in (z_bottom, z_floor)])
    tube_profile = [(0., z_floor), (outer, z_floor), (outer, z_top), (outer - wall, z_top),
                    (outer - wall, z_floor + 0.0005), (0., z_floor + 0.0005)]
    foot_volume = 1.5 * math.sqrt(3.) * foot_r ** 2 * foot_h
    tube_volume, tube_z = _profile_moments(tube_profile)
    com_z = _combined_com([(foot_volume, z_bottom + foot_h / 2.), (tube_volume, tube_z)])
    asset = Asset('graduated_cylinder', dynamic=True, mass_kg=spec['mass'], friction=0.42,
                  center_of_mass_m=(0., 0., com_z))
    _display(asset, foot, 'frosted_glass', 'foot')
    _display(asset, lathe_mesh(tube_profile, segments=VISUAL_SEGMENTS), 'frosted_glass', 'tube')
    graduation = []
    for index in range(1, 9):
        z = z_floor + (height - foot_h) * index / 9.
        ring = annulus_mesh(outer + 0.0002, outer - 0.0004, 0.0008, segments=48)
        ring.apply_translation((0., 0., z))
        _display(asset, ring, 'graduation_white', f'grad_{index:02d}')
        graduation.append(_r(z))
    _convex(asset, 'foot', foot, role='floor')
    cavity = outer * math.cos(math.pi / spec['sectors']) - COLLISION_WALL_M
    _vessel_wall(asset, 'wall', [(z_floor, outer, cavity), (z_top, outer, cavity)], spec['sectors'])
    asset.data.update(
        nominal_size_m=DESIGN_SIZES['graduated_cylinder'],
        size_note=(f'50 ml cylinder: {2 * outer * 1000:.0f} mm tube, {height * 1000:.0f} mm tall, on a '
                   f'hexagonal foot {2 * foot_r * 1000:.0f} mm across corners'),
        mass_source='Design value (a 50 ml borosilicate graduated cylinder is about 0.1 kg)',
        mass_design_value_kg=spec['mass'], open_top=True, closed_bottom=True,
        cavity_radius_m=float(cavity), cavity_floor_z_m=float(z_floor),
        visual_inner_radius_m=float(outer - wall), graduation_z_m=graduation,
        graduation_note='Display-only printed rings; no liquid is modelled',
        outer_profile_m=[[foot_r, z_bottom], [foot_r, z_floor], [outer + 0.0002, z_floor],
                         [outer + 0.0002, z_top]],
        collision_note=('The hexagonal foot (convex hull of the visible foot) is the cavity floor; the '
                        'tube is 16 convex wall sectors grown inward to >= 3 mm. No bore is declared.'))
    return asset


def _stirring_rod():
    """A 7 mm across-flats hexagonal glass rod, flat face down, along +x."""
    radius, length = STIR_ROD['circumradius'], STIR_ROD['length']
    points = [(x, radius * math.cos(k * math.pi / 3.), radius * math.sin(k * math.pi / 3.))
              for x in (-length / 2., length / 2.) for k in range(6)]
    rod = _hull(points)
    asset = Asset('stirring_rod', dynamic=True, mass_kg=STIR_ROD['mass'], friction=0.40)
    _display(asset, rod, 'frosted_glass', 'rod')
    _convex(asset, 'rod', rod.copy(), role='body')
    asset.data.update(
        nominal_size_m=DESIGN_SIZES['stirring_rod'],
        size_note=('7 mm across-flats hexagonal glass rod, 200 mm long, lying along +x with a flat '
                   'face down (it cannot roll)'),
        mass_source=(f'Design value: borosilicate volume of this prism x 2230 kg/m3 = '
                     f'{STIR_ROD["mass"] * 1000:.1f} g'),
        mass_design_value_kg=STIR_ROD['mass'],
        collision_note='The convex hull of the very prism that is displayed: collider == mesh')
    return asset


def _cork_ring():
    """An 80 mm cork flask ring: the family's real through bore (16 bore-tangent sectors)."""
    spec = CORK_RING
    outer, inner, height, bevel = spec['outer'], spec['inner'], spec['height'], spec['bevel']
    asset = Asset('cork_ring', dynamic=True, mass_kg=spec['mass'], friction=0.70)
    _display(asset, annulus_mesh(outer, inner, height, segments=VISUAL_SEGMENTS, bevel=bevel), 'cork', 'ring')
    levels = [(-height / 2., outer - bevel, inner + bevel), (-height / 2. + bevel, outer, inner),
              (height / 2. - bevel, outer, inner), (height / 2., outer - bevel, inner + bevel)]
    count = spec['sectors']
    for index in range(count):
        _convex(asset, f'ring_{index:02d}', _wall_sector(levels, index, count),
                sector=f'{index:02d}/{count}', role='ring', bore=True, bore_radius_m=float(inner),
                clearance_radius_m=float(inner))
    asset.data.update(
        nominal_size_m=DESIGN_SIZES['cork_ring'],
        size_note='Cork flask ring: 80 mm outside, 50 mm bore, 25 mm tall, 1 mm edge bevel',
        mass_source='Design value (cork at about 200 kg/m3 over this ring is about 0.015 kg)',
        mass_design_value_kg=spec['mass'], bore_radius_m=float(inner),
        outer_profile_m=[[outer - bevel, -height / 2.], [outer, -height / 2. + bevel],
                         [outer, height / 2. - bevel], [outer - bevel, height / 2.]],
        collision_note=('A real through bore: 16 convex sectors declared bore: true, each inner face '
                        'tangent to the 25 mm bore, so no piece seals it.'))
    return asset


# --- static fixture assets (origin at the base bottom centre) --------------------

def _reagent_bottle(asset_id):
    spec = BOTTLES[asset_id]
    radius, body_top, neck = spec['radius'], spec['body_top'], spec['neck_radius']
    shoulder_top, cap_r, cap_top = spec['shoulder_top'], spec['cap_radius'], spec['cap_top']
    heel = _heel(radius, 0., 0.004)
    shoulder = [(neck + (radius - neck) * math.cos(t), body_top + (shoulder_top - body_top) * math.sin(t))
                for t in np.linspace(0., math.pi / 2., 7)]
    profile = [(0., 0.)] + heel + [(radius, body_top)] + shoulder[1:] + [(0., shoulder_top)]
    asset = Asset(asset_id, dynamic=False, mass_kg=0.0, friction=0.5)
    _display(asset, lathe_mesh(profile, segments=VISUAL_SEGMENTS), spec['glass'], 'bottle')
    cap = cylinder_mesh(cap_r, cap_top - shoulder_top, sections=VISUAL_SEGMENTS)
    cap.apply_translation((0., 0., (shoulder_top + cap_top) / 2.))
    _display(asset, cap, spec['cap'], 'cap')
    _convex(asset, 'bottle', _revolved_hull(heel + [(radius, body_top)] + shoulder[1:]), role='body')
    _cylinder(asset, 'cap', cap_r, shoulder_top, cap_top, role='cap')
    asset.data.update(nominal_size_m=DESIGN_SIZES[asset_id], origin='base_bottom_centre',
                      role='Static reagent bottle on a shelf; never a manipulation target',
                      collision_note='Convex hull of the bottle body (heel, body, shoulder) plus the cap')
    return asset


def _micropipette():
    """A single-channel micropipette hanging in its stand by the handle (static)."""
    asset = Asset('micropipette_stand', dynamic=False, mass_kg=0.0, friction=0.5)
    boxes = (('stand_base', (-0.050, -0.045, 0.000), (0.050, 0.045, 0.012)),
             ('stand_post', (0.027, -0.008, 0.012), (0.043, 0.008, 0.200)),
             ('prong_a', (-0.014, 0.008, 0.151), (0.027, 0.014, 0.165)),
             ('prong_b', (-0.014, -0.014, 0.151), (0.027, -0.008, 0.165)))
    for label, lo, hi in boxes:
        _box_display(asset, label, lo, hi, 'stand_grey')
        _box_collision(asset, label, lo, hi)
    shaft = _revolved_hull([(0.0025, 0.040), (0.0060, 0.100)], segments=48)
    _display(asset, shaft, 'pipette_body', 'shaft')
    _convex(asset, 'shaft', shaft.copy(), role='pipette')
    for label, radius, z0, z1, appearance in (('ejector', 0.009, 0.100, 0.150, 'pipette_body'),
                                              ('neck', 0.007, 0.150, 0.165, 'pipette_body'),
                                              ('handle', 0.015, 0.165, 0.265, 'pipette_body'),
                                              ('plunger', 0.006, 0.265, 0.290, 'pipette_body'),
                                              ('button', 0.008, 0.290, 0.300, 'pipette_accent')):
        mesh = cylinder_mesh(radius, z1 - z0, sections=48)
        mesh.apply_translation((0., 0., (z0 + z1) / 2.))
        _display(asset, mesh, appearance, label)
        _cylinder(asset, label, radius, z0, z1, role='pipette')
    asset.data.update(nominal_size_m=DESIGN_SIZES['micropipette_stand'], origin='base_bottom_centre',
                      role=('Static micropipette for the tip rack, hanging by its handle on the stand '
                            'prongs; tips cannot be mounted on it (no nozzle socket is simulated)'),
                      collision_note='Boxes for the stand, convex/cylinder pieces for every pipette section')
    return asset


def _fire_extinguisher():
    spec = EXTINGUISHER
    radius, body_top, dome_top = spec['radius'], spec['body_top'], spec['dome_top']
    heel = _heel(radius, 0., 0.006)
    dome = [(spec['valve_radius'] + (radius - spec['valve_radius']) * math.cos(t),
             body_top + (dome_top - body_top) * math.sin(t)) for t in np.linspace(0., math.pi / 2., 7)]
    profile = [(0., 0.)] + heel + [(radius, body_top)] + dome[1:] + [(0., dome_top)]
    asset = Asset('fire_extinguisher', dynamic=False, mass_kg=0.0, friction=0.6)
    _display(asset, lathe_mesh(profile, segments=VISUAL_SEGMENTS), 'extinguisher_red', 'body')
    valve = cylinder_mesh(spec['valve_radius'], spec['valve_top'] - dome_top, sections=48)
    valve.apply_translation((0., 0., (dome_top + spec['valve_top']) / 2.))
    _display(asset, valve, 'chrome', 'valve')
    lever_lo, lever_hi = (-0.060, -0.010, spec['valve_top']), (0.030, 0.010, spec['lever_top'])
    _box_display(asset, 'lever', lever_lo, lever_hi, 'extinguisher_black')
    _convex(asset, 'body', _revolved_hull(heel + [(radius, body_top)] + dome[1:]), role='body')
    _cylinder(asset, 'valve', spec['valve_radius'], dome_top, spec['valve_top'], role='valve')
    _box_collision(asset, 'lever', lever_lo, lever_hi)
    asset.data.update(nominal_size_m=DESIGN_SIZES['fire_extinguisher'], origin='base_bottom_centre',
                      role='Static 6 kg-class extinguisher standing by the door')
    return asset


def _bottle_size(spec):
    return [2 * spec['radius'], 2 * spec['radius'], spec['cap_top']]


# Bounding boxes the builders must reproduce exactly (the CPU test compares them
# with the manifest); rest poses use the manifest once it exists.
DESIGN_SIZES = {
    'test_tube': [2 * TUBE['radius'], 2 * TUBE['radius'], TUBE['height']],
    'pipette_tip': [2 * TIP['collar_radius'], 2 * TIP['collar_radius'], TIP['length']],
    **{key: [2 * spec['outer'], 2 * spec['outer'], spec['height']] for key, spec in BEAKERS.items()},
    'erlenmeyer_flask': [2 * FLASK['base_outer'], 2 * FLASK['base_outer'], FLASK['height']],
    'graduated_cylinder': [2 * GRAD_CYLINDER['foot_circumradius'],
                           2 * GRAD_CYLINDER['foot_circumradius'] * math.sin(math.pi / 3.),
                           GRAD_CYLINDER['height']],
    'stirring_rod': [STIR_ROD['length'], 2 * STIR_ROD['circumradius'], STIR_ROD['across_flats']],
    'cork_ring': [2 * CORK_RING['outer'], 2 * CORK_RING['outer'], CORK_RING['height']],
    **{key: _bottle_size(spec) for key, spec in BOTTLES.items()},
    'micropipette_stand': list(MICROPIPETTE_SIZE),
    'fire_extinguisher': [2 * EXTINGUISHER['radius'], 2 * EXTINGUISHER['radius'], EXTINGUISHER['lever_top']],
}

ASSET_BUILDERS = (
    ('test_tube', _test_tube),
    ('pipette_tip', _pipette_tip),
    ('beaker_50ml', lambda: _beaker('beaker_50ml')),
    ('beaker_150ml', lambda: _beaker('beaker_150ml')),
    ('beaker_250ml', lambda: _beaker('beaker_250ml')),
    ('erlenmeyer_flask', _erlenmeyer_flask),
    ('graduated_cylinder', _graduated_cylinder),
    ('stirring_rod', _stirring_rod),
    ('cork_ring', _cork_ring),
    ('reagent_bottle_amber_500ml', lambda: _reagent_bottle('reagent_bottle_amber_500ml')),
    ('reagent_bottle_clear_250ml', lambda: _reagent_bottle('reagent_bottle_clear_250ml')),
    ('micropipette_stand', _micropipette),
    ('fire_extinguisher', _fire_extinguisher),
)


def _apply_visual_appearance(root):
    """Write roughness/metallic into every visual part of the manifest just written.

    ``scene.add_task_asset`` reads ``part.get('roughness', .35)`` and
    ``part.get('metallic', .65)``; without these keys every glass vessel would be
    bound 65 % metallic. The mesh hashes are untouched (only manifest fields are
    added), so ``load_task_asset(verify_hashes=True)`` still verifies every OBJ.
    """
    path = Path(root) / 'manifest.json'
    catalogue = load_json(path)
    for asset_id, record in catalogue['assets'].items():
        mapping = record.get('visual_appearance', {})
        for part in record['visual_parts']:
            name = mapping[part['label']]
            part.update(appearance=name, roughness=APPEARANCE[name]['roughness'],
                        metallic=APPEARANCE[name]['metallic'])
    catalogue['visual_appearance_note'] = (
        'roughness/metallic were added to every visual part by laboratory.build_library after '
        'write_library (which records colour only), so scene.add_task_asset binds the authored '
        'appearance instead of its 0.35/0.65 defaults')
    path.write_text(json.dumps(catalogue, indent=1, ensure_ascii=False, allow_nan=False) + '\n')
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_library(asset_root=ASSET_ROOT):
    """Write the authored OBJ library into a NEW version directory; returns the record.

    ``write_library`` refuses a non-empty target, so this never overwrites a
    version: rebuild into the next ``vN`` and repoint ``ASSET_ROOT``.
    """
    assets = [(asset_id, builder()) for asset_id, builder in ASSET_BUILDERS]
    record = write_library(Path(asset_root), assets, generator=GENERATOR, license_id=LICENSE_ID,
                           extra={'family': FAMILY, 'units': 'metres',
                                  'source': 'Procedural geometry authored for the laboratory scene family',
                                  'dimension_status': 'Simulation task design values; not manufacturer CAD',
                                  'mass_status': 'Design values recorded per asset with their source',
                                  'supersedes': {'version': 'v1', 'root': SUPERSEDED_ASSET_ROOTS['v1'],
                                                 'manifest_sha256': V1_MANIFEST_SHA256,
                                                 'kept_as_evidence': True},
                                  'collision_note': ('Convex collision pieces are the physical shape; '
                                                     'each OBJ is centred on its own box and placed by '
                                                     'position_m, so no offset is applied twice')})
    record['manifest_sha256'] = _apply_visual_appearance(asset_root)
    record['visual_appearance_patched'] = True
    return record


def load_manifest(asset_root=ASSET_ROOT):
    """Read the authored manifest, or ``None`` if the library is not built yet."""
    path = Path(asset_root) / 'manifest.json'
    if not path.is_file():
        return None
    return load_json(path)


def asset_sizes(asset_root=ASSET_ROOT):
    """AABB size of every asset: from the manifest when it exists, else the design sizes."""
    manifest = load_manifest(asset_root)
    if not manifest:
        return {key: list(value) for key, value in DESIGN_SIZES.items()}
    return {key: list(record['size_m']) for key, record in manifest['assets'].items()}


# --- scene helpers ------------------------------------------------------------------

def _box(part_id, lo, hi, material, **extra):
    lo, hi = np.asarray(lo, dtype=float), np.asarray(hi, dtype=float)
    if np.any(hi - lo <= 0):
        raise ValueError(f'{part_id}: empty box {lo} {hi}')
    return {'id': part_id, 'geometry': 'box', 'position_m': [_r(v) for v in (lo + hi) / 2.],
            'half_extents_m': [_r(v) for v in (hi - lo) / 2.], 'material': material, **extra}


def _prop(prop_id, title_zh, material, parts, friction=0.6, support=None, **extra):
    """A static prop; ``support`` (lo, hi) declares a support box to family.support_surfaces."""
    spec = {'id': prop_id, 'title_zh': title_zh, 'material': material, 'friction': friction,
            'parts': parts, **extra}
    if support is not None:
        lo, hi = (np.asarray(v, dtype=float) for v in support)
        spec['position_m'] = [_r(v) for v in (lo + hi) / 2.]
        spec['half_extents_m'] = [_r(v) for v in (hi - lo) / 2.]
    return spec


def _opening(opening_id, wall, along, z_range, kind):
    """A wall opening from world coordinates: ``along`` is x on front/back walls, y on left/right."""
    lo, hi = along
    centre = (lo + hi) / 2.
    u = {'front': centre, 'back': -centre, 'right': -centre, 'left': centre}[wall]
    v = (z_range[0] + z_range[1]) / 2. - ROOM_SIZE_M[2] / 2.
    return {'id': opening_id, 'wall': wall, 'center_m': [_r(u), _r(v)],
            'size_m': [_r(hi - lo), _r(z_range[1] - z_range[0])], 'kind': kind,
            'world_along_m': [lo, hi], 'world_z_m': list(z_range)}


DOOR = _opening('door', 'back', (1.30, 2.20), (0.0, 2.10), 'door')
WINDOW = _opening('window', 'front', (-0.20, 0.90), (1.15, 2.25), 'window')


def _rack_grid(centre, pitch, columns, rows):
    return [(centre[0] + (c - (columns - 1) / 2.) * pitch, centre[1] + (r - (rows - 1) / 2.) * pitch)
            for r in range(rows) for c in range(columns)]


TUBE_SLOTS = _rack_grid(TUBE_RACK_CENTER, TUBE_PITCH_M, TUBE_COLUMNS, TUBE_ROWS)
TIP_SLOTS = _rack_grid(TIP_RACK_CENTER, TIP_PITCH_M, TIP_COLUMNS, TIP_ROWS)


def _perforated_plate(prefix, centre, half, pitch, columns, rows, hole_half, z0, z1, material):
    """A plate of rails (along x) and ribs (along y) that leaves a CLOSED square hole per slot."""
    cx, cy = centre
    xs = [cx + (c - (columns - 1) / 2.) * pitch for c in range(columns)]
    ys = [cy + (r - (rows - 1) / 2.) * pitch for r in range(rows)]
    parts = []
    y_edges = [cy - half[1]]
    for y in ys:
        y_edges += [y - hole_half, y + hole_half]
    y_edges.append(cy + half[1])
    for index in range(0, len(y_edges), 2):          # rails between and outside the rows
        parts.append(_box(f'{prefix}_rail_{index // 2:02d}', (cx - half[0], y_edges[index], z0),
                          (cx + half[0], y_edges[index + 1], z1), material))
    x_edges = [cx - half[0]]
    for x in xs:
        x_edges += [x - hole_half, x + hole_half]
    x_edges.append(cx + half[0])
    for row, y in enumerate(ys):                     # ribs between the holes of each row
        for index in range(0, len(x_edges), 2):
            parts.append(_box(f'{prefix}_rib_{row}_{index // 2:02d}',
                              (x_edges[index], y - hole_half, z0), (x_edges[index + 1], y + hole_half, z1),
                              material))
    return parts


# --- static props -------------------------------------------------------------------

def _lab_bench():
    """Cabinets under the epoxy top. Declares the table top as a support; builds no slab."""
    doors, handles = [], []
    edges = [(-0.258, -0.032), (-0.028, 0.198), (0.202, 0.428), (0.432, 0.658)]
    for index, (y0, y1) in enumerate(edges):
        doors.append(_box(f'door_{index:02d}', (0.34, y0, 0.12), (0.36, y1, 0.70), 'cabinet_door'))
        hy = y1 - 0.030 if index % 2 == 0 else y0 + 0.018
        handles.append(_box(f'handle_{index:02d}', (0.325, hy, 0.52), (0.34, hy + 0.012, 0.64),
                            'handle_steel'))
    parts = ([_box('carcass', (0.36, -0.26, 0.10), (1.06, 0.66, TABLE_BOTTOM_Z), 'cabinet_grey'),
              _box('plinth', (0.42, -0.22, 0.0), (1.02, 0.62, 0.10), 'rubber_black')] + doors + handles)
    lo = (TABLE_X[0], TABLE_Y[0], TABLE_BOTTOM_Z)
    hi = (TABLE_X[1], TABLE_Y[1], TABLE_TOP_Z)
    return _prop('lab_bench', '实验台柜体', 'cabinet_grey', parts, friction=0.55, support=(lo, hi),
                 support_of='table',
                 support_note=('Declares the work-table top (SCENE["table"], the only collider there) '
                               'to family.support_surfaces, which does not read the table; no second '
                               'slab is built.'))


def _bench_lip():
    return _prop('bench_lip', '实验台挡边', 'stainless', [
        _box('lip_left_end', (0.30, -0.30, TABLE_TOP_Z), (0.96, -0.28, TABLE_TOP_Z + 0.02), 'stainless'),
        _box('lip_right_end', (0.30, 0.68, TABLE_TOP_Z), (0.96, 0.70, TABLE_TOP_Z + 0.02), 'stainless')],
        friction=0.45)


RISER_BOARD_TOP_Z = 1.10


def _reagent_riser():
    parts = [_box(f'upright_{index:02d}', (1.00, y, TABLE_TOP_Z), (1.08, y + 0.02, RISER_BOARD_TOP_Z - 0.02),
                  'stainless') for index, y in enumerate((0.10, 0.38, 0.66))]
    parts.append(_box('board', (0.98, 0.10, RISER_BOARD_TOP_Z - 0.02), (1.10, 0.68, RISER_BOARD_TOP_Z),
                      'stainless'))
    parts.append(_box('back_rail', (1.08, 0.10, RISER_BOARD_TOP_Z), (1.10, 0.68, RISER_BOARD_TOP_Z + 0.04),
                      'stainless'))
    return _prop('reagent_riser', '台面试剂架', 'stainless', parts, friction=0.5)


def _tube_rack():
    cx, cy = TUBE_RACK_CENTER
    hx, hy = TUBE_RACK_HALF
    base = _box('base_plate', (cx - hx, cy - hy, TABLE_TOP_Z), (cx + hx, cy + hy, TUBE_RACK_TOP_Z), 'rack_white')
    walls = [_box(f'end_wall_{name}', (x0, cy - hy, TUBE_RACK_TOP_Z), (x1, cy + hy, TUBE_PLATE_BOTTOM_Z),
                  'rack_white')
             for name, x0, x1 in (('a', cx - hx, cx - hx + 0.006), ('b', cx + hx - 0.006, cx + hx))]
    plate = _perforated_plate('upper', TUBE_RACK_CENTER, TUBE_RACK_HALF, TUBE_PITCH_M, TUBE_COLUMNS,
                              TUBE_ROWS, TUBE_HOLE_HALF_M, TUBE_PLATE_BOTTOM_Z, TUBE_PLATE_TOP_Z, 'rack_white')
    return _prop('tube_rack', '试管架', 'rack_white', [base] + walls + plate, friction=0.6,
                 support=((cx - hx, cy - hy, TABLE_TOP_Z), (cx + hx, cy + hy, TUBE_RACK_TOP_Z)),
                 support_part='base_plate',
                 holds={'asset_id': 'test_tube', 'slots_xy_m': [[_r(x), _r(y)] for x, y in TUBE_SLOTS],
                        'hole_half_m': TUBE_HOLE_HALF_M, 'plate_z_m': [TUBE_PLATE_BOTTOM_Z, TUBE_PLATE_TOP_Z]},
                 note=('12-place two-tier rack: tubes stand on the base plate and pass through closed '
                       '14 mm holes in the upper plate 62-66 mm above it'))


def _tip_rack():
    cx, cy = TIP_RACK_CENTER
    hx, hy = TIP_RACK_HALF
    w = TIP_RACK_WALL_M
    floor = _box('floor_plate', (cx - hx, cy - hy, TABLE_TOP_Z), (cx + hx, cy + hy, TIP_RACK_FLOOR_TOP_Z),
                 'rack_blue')
    walls = [_box('wall_front', (cx - hx, cy - hy, TIP_RACK_FLOOR_TOP_Z), (cx + hx, cy - hy + w, TIP_DECK_BOTTOM_Z),
                  'rack_blue'),
             _box('wall_back', (cx - hx, cy + hy - w, TIP_RACK_FLOOR_TOP_Z), (cx + hx, cy + hy, TIP_DECK_BOTTOM_Z),
                  'rack_blue'),
             _box('wall_left', (cx - hx, cy - hy + w, TIP_RACK_FLOOR_TOP_Z), (cx - hx + w, cy + hy - w,
                                                                               TIP_DECK_BOTTOM_Z), 'rack_blue'),
             _box('wall_right', (cx + hx - w, cy - hy + w, TIP_RACK_FLOOR_TOP_Z), (cx + hx, cy + hy - w,
                                                                                   TIP_DECK_BOTTOM_Z), 'rack_blue')]
    deck = _perforated_plate('deck', TIP_RACK_CENTER, TIP_RACK_HALF, TIP_PITCH_M, TIP_COLUMNS, TIP_ROWS,
                             TIP_HOLE_HALF_M, TIP_DECK_BOTTOM_Z, TIP_DECK_TOP_Z, 'rack_blue')
    return _prop('tip_rack', '移液枪头盒', 'rack_blue', [floor] + walls + deck, friction=0.6,
                 support=((cx - hx, cy - hy, TABLE_TOP_Z), (cx + hx, cy + hy, TIP_RACK_FLOOR_TOP_Z)),
                 support_part='floor_plate',
                 holds={'asset_id': 'pipette_tip', 'slots_xy_m': [[_r(x), _r(y)] for x, y in TIP_SLOTS],
                        'hole_half_m': TIP_HOLE_HALF_M, 'deck_z_m': [TIP_DECK_BOTTOM_Z, TIP_DECK_TOP_Z]},
                 note=('8-place tip box: each tip point stands on the floor plate while its collar '
                       'rests on the deck around a closed 7.4 mm hole'))


def _hot_plate():
    top = TABLE_TOP_Z
    return _prop('hot_plate_stirrer', '加热磁力搅拌器', 'hotplate_body', [
        _box('body', (0.82, -0.22, top), (0.98, 0.04, top + 0.09), 'hotplate_body'),
        _box('top_plate', (0.835, -0.155, top + 0.09), (0.965, -0.025, top + 0.098), 'ceramic_white'),
        _box('knob_heat', (0.805, -0.19, top + 0.03), (0.82, -0.16, top + 0.06), 'knob_black'),
        _box('knob_stir', (0.805, -0.10, top + 0.03), (0.82, -0.07, top + 0.06), 'knob_black'),
        _box('display', (0.815, -0.05, top + 0.035), (0.82, 0.01, top + 0.065), 'display_dark')],
        friction=0.6)


def _fume_hood():
    x0, x1, y0, y1 = 1.25, 2.45, 1.20, 2.00
    return _prop('fume_hood', '通风橱', 'hood_white', [
        _box('plinth', (1.30, 1.32, 0.0), (2.40, y1, 0.10), 'rubber_black'),
        _box('base_cabinet', (1.27, 1.26, 0.10), (2.43, y1, 0.86), 'cabinet_grey'),
        _box('cabinet_door_a', (1.28, 1.24, 0.12), (1.84, 1.26, 0.84), 'cabinet_door'),
        _box('cabinet_door_b', (1.86, 1.24, 0.12), (2.42, 1.26, 0.84), 'cabinet_door'),
        _box('work_surface', (x0, y0, 0.86), (x1, y1, 0.90), 'black_bench_top'),
        _box('side_left', (x0, y0, 0.90), (1.33, y1, 2.20), 'hood_white'),
        _box('side_right', (2.37, y0, 0.90), (x1, y1, 2.20), 'hood_white'),
        _box('back_baffle', (1.33, 1.94, 0.90), (2.37, y1, 2.20), 'hood_white'),
        _box('canopy', (x0, y0, 2.20), (x1, y1, 2.40), 'hood_white'),
        _box('fascia', (1.33, y0, 1.95), (2.37, 1.24, 2.20), 'hood_white'),
        _box('airfoil_sill', (1.33, y0, 0.90), (2.37, 1.24, 0.92), 'stainless'),
        _box('sash_rail', (1.33, 1.245, 1.33), (2.37, 1.275, 1.37), 'stainless'),
        _box('sash_glass', (1.35, 1.25, 1.37), (2.35, 1.27, 1.95), 'sash_glass'),
        _box('control_panel', (2.37, 1.18, 1.10), (2.45, y0, 1.30), 'display_dark'),
        _box('duct', (1.70, 1.55, 2.40), (2.00, 1.85, ROOM_SIZE_M[2]), 'hood_white')], friction=0.6)


def _sink_unit():
    counter_z0, counter_z1 = 0.87, 0.90
    basin = (0.10, 0.50, 1.52, 1.84)
    bx0, bx1, by0, by1 = basin
    return _prop('sink_unit', '水槽与洗眼器', 'cabinet_grey', [
        _box('plinth', (-0.36, 1.50, 0.0), (0.96, 2.00, 0.10), 'rubber_black'),
        _box('cabinet_left', (-0.38, 1.44, 0.10), (0.08, 2.00, counter_z0), 'cabinet_grey'),
        _box('cabinet_right', (0.52, 1.44, 0.10), (0.98, 2.00, counter_z0), 'cabinet_grey'),
        _box('door_left', (-0.38, 1.42, 0.12), (0.08, 1.44, 0.85), 'cabinet_door'),
        _box('door_right', (0.52, 1.42, 0.12), (0.98, 1.44, 0.85), 'cabinet_door'),
        _box('door_under_sink', (0.08, 1.42, 0.10), (0.52, 1.44, 0.66), 'cabinet_door'),
        _box('counter_left', (-0.40, 1.40, counter_z0), (bx0, 2.00, counter_z1), 'black_bench_top'),
        _box('counter_right', (bx1, 1.40, counter_z0), (1.00, 2.00, counter_z1), 'black_bench_top'),
        _box('counter_front', (bx0, 1.40, counter_z0), (bx1, by0, counter_z1), 'black_bench_top'),
        _box('counter_back', (bx0, by1, counter_z0), (bx1, 2.00, counter_z1), 'black_bench_top'),
        _box('basin_bottom', (bx0, by0, 0.68), (bx1, by1, 0.70), 'stainless'),
        _box('basin_front', (bx0, by0, 0.70), (bx1, by0 + 0.02, counter_z0), 'stainless'),
        _box('basin_back', (bx0, by1 - 0.02, 0.70), (bx1, by1, counter_z0), 'stainless'),
        _box('basin_left', (bx0, by0 + 0.02, 0.70), (bx0 + 0.02, by1 - 0.02, counter_z0), 'stainless'),
        _box('basin_right', (bx1 - 0.02, by0 + 0.02, 0.70), (bx1, by1 - 0.02, counter_z0), 'stainless'),
        _box('faucet_post', (0.285, 1.85, counter_z1), (0.315, 1.88, 1.10), 'stainless'),
        _box('faucet_spout', (0.285, 1.65, 1.07), (0.315, 1.85, 1.10), 'stainless'),
        _box('faucet_outlet', (0.285, 1.65, 1.03), (0.315, 1.68, 1.07), 'stainless'),
        _box('eyewash_body', (0.70, 1.60, counter_z1), (0.88, 1.84, 0.96), 'safety_green'),
        _box('eyewash_bowls', (0.715, 1.62, 0.96), (0.865, 1.76, 0.98), 'safety_yellow'),
        _box('eyewash_sign', (0.70, 1.99, 0.97), (0.88, 2.00, 1.08), 'safety_green')], friction=0.6)


def _window_frame():
    (x0, x1), (z0, z1) = WINDOW['world_along_m'], WINDOW['world_z_m']
    y_face = ROOM_HALF[1]
    mid = (x0 + x1) / 2.
    glazing = {'glazing_of': 'window'}
    return _prop('window_frame', '窗框与窗台', 'window_frame', [
        _box('stile_left', (x0 - 0.06, y_face - 0.04, z0), (x0, y_face, z1), 'window_frame'),
        _box('stile_right', (x1, y_face - 0.04, z0), (x1 + 0.06, y_face, z1), 'window_frame'),
        _box('head', (x0 - 0.06, y_face - 0.04, z1), (x1 + 0.06, y_face, z1 + 0.06), 'window_frame'),
        _box('sill', (x0 - 0.10, y_face - 0.12, z0 - 0.04), (x1 + 0.10, y_face, z0), 'window_frame'),
        _box('mullion', (mid - 0.02, y_face + 0.04, z0), (mid + 0.02, y_face + 0.08, z1), 'window_frame',
             **glazing),
        _box('pane_left', (x0, y_face + 0.055, z0), (mid - 0.02, y_face + 0.065, z1), 'window_glass', **glazing),
        _box('pane_right', (mid + 0.02, y_face + 0.055, z0), (x1, y_face + 0.065, z1), 'window_glass',
             **glazing)], friction=0.6,
        note='Casing, sill and head sit on the wall face around the opening; only the glazing is in it')


def _door_casing():
    (x0, x1), (_, z1) = DOOR['world_along_m'], DOOR['world_z_m']
    y_face = -ROOM_HALF[1]
    return _prop('door_casing', '门框', 'window_frame', [
        _box('stile_a', (x0 - 0.06, y_face, 0.0), (x0, y_face + 0.04, z1 + 0.06), 'window_frame'),
        _box('stile_b', (x1, y_face, 0.0), (x1 + 0.06, y_face + 0.04, z1 + 0.06), 'window_frame'),
        _box('head', (x0, y_face, z1), (x1, y_face + 0.04, z1 + 0.06), 'window_frame')], friction=0.6,
        note='The head sits directly on the opening (no gap); the stiles flank it, outside the opening')


def _safety_shower():
    y_face = -ROOM_HALF[1]
    return _prop('safety_shower', '紧急喷淋', 'safety_green', [
        _box('pipe', (0.88, y_face + 0.02, 1.20), (0.92, y_face + 0.06, 2.35), 'safety_green'),
        _box('arm', (0.88, y_face + 0.06, 2.31), (0.92, y_face + 0.40, 2.35), 'safety_green'),
        _box('head', (0.775, y_face + 0.275, 2.25), (1.025, y_face + 0.525, 2.31), 'safety_yellow'),
        _box('pull_rod', (0.89, y_face + 0.39, 1.70), (0.91, y_face + 0.41, 2.25), 'stainless'),
        _box('pull_handle', (0.86, y_face + 0.38, 1.66), (0.94, y_face + 0.42, 1.70), 'safety_yellow'),
        _box('sign', (0.80, y_face, 2.40), (1.00, y_face + 0.01, 2.55), 'safety_green')], friction=0.6)


WALL_SHELF_BOARDS_Z = (1.40, 1.75)


def _wall_shelf():
    y_face = ROOM_HALF[1]
    parts = []
    for index, top in enumerate(WALL_SHELF_BOARDS_Z):
        parts.append(_box(f'board_{index}', (-2.05, 1.78, top - 0.02), (-0.75, y_face - 0.01, top),
                          'stainless'))
        for side, x in (('a', -1.95), ('b', -0.87)):
            parts.append(_box(f'bracket_{index}{side}', (x, 1.80, top - 0.06), (x + 0.02, y_face - 0.01, top - 0.02),
                              'dark_steel'))
    for side, x in (('a', -1.95), ('b', -0.87)):
        parts.append(_box(f'standard_{side}', (x, y_face - 0.01, 1.20), (x + 0.02, y_face, 1.95), 'dark_steel'))
    return _prop('wall_shelf', '墙面试剂架', 'stainless', parts, friction=0.5)


def _flammables_cabinet():
    x_face = ROOM_HALF[0]
    return _prop('flammables_cabinet', '易燃品安全柜', 'safety_yellow', [
        _box('plinth', (2.10, -0.55, 0.0), (x_face - 0.01, 0.25, 0.05), 'rubber_black'),
        _box('carcass', (2.06, -0.58, 0.05), (x_face - 0.01, 0.28, 1.65), 'safety_yellow'),
        _box('door_a', (2.04, -0.58, 0.08), (2.06, -0.152, 1.62), 'safety_yellow'),
        _box('door_b', (2.04, -0.148, 0.08), (2.06, 0.28, 1.62), 'safety_yellow'),
        _box('handle_a', (2.025, -0.172, 0.80), (2.04, -0.160, 0.98), 'handle_steel'),
        _box('handle_b', (2.025, -0.140, 0.80), (2.04, -0.128, 0.98), 'handle_steel'),
        _box('placard', (2.035, -0.45, 1.30), (2.04, -0.25, 1.45), 'fire_red')], friction=0.6)


def _first_aid_box():
    x_face = ROOM_HALF[0]
    return _prop('first_aid_box', '急救箱', 'safety_green', [
        _box('box', (x_face - 0.10, 0.60, 1.40), (x_face, 0.90, 1.70), 'safety_green'),
        _box('cross_bar_h', (x_face - 0.105, 0.70, 1.535), (x_face - 0.10, 0.80, 1.565), 'paper'),
        _box('cross_bar_v_low', (x_face - 0.105, 0.735, 1.50), (x_face - 0.10, 0.765, 1.535), 'paper'),
        _box('cross_bar_v_high', (x_face - 0.105, 0.735, 1.565), (x_face - 0.10, 0.765, 1.60), 'paper')],
        friction=0.6)


def _wall_notices():
    x_face = -ROOM_HALF[0]
    return _prop('wall_notices', '墙面资料', 'paper', [
        _box('sheet_a', (x_face, 0.24, 1.40), (x_face + 0.016, 0.46, 1.70), 'paper'),
        _box('sheet_b', (x_face, 0.53, 1.38), (x_face + 0.016, 0.71, 1.62), 'paper'),
        _box('sheet_c', (x_face, 0.00, 1.35), (x_face + 0.016, 0.20, 1.61), 'cardboard'),
        _box('periodic_table', (x_face, -0.80, 1.30), (x_face + 0.012, -0.20, 1.75), 'poster_blue')],
        friction=0.7)


def _safety_signs():
    x_face = -ROOM_HALF[0]
    parts = [_box('panel', (x_face, -1.60, 1.40), (x_face + 0.02, -1.08, 1.80), 'paper')]
    for name, material, y in (('sign_fire', 'fire_red', -1.51), ('sign_caution', 'safety_yellow', -1.34),
                              ('sign_ppe', 'poster_blue', -1.17)):
        parts.append(_box(name, (x_face + 0.02, y - 0.06, 1.54), (x_face + 0.026, y + 0.06, 1.66), material))
    return _prop('safety_signs', '安全标识', 'paper', parts, friction=0.6)


def _waste_bin():
    x0, x1, y0, y1, h, w = -1.70, -1.40, 1.36, 1.64, 0.45, 0.012
    return _prop('waste_bin', '废弃物桶', 'plastic_grey', [
        _box('floor', (x0, y0, 0.0), (x1, y1, w), 'plastic_grey'),
        _box('wall_front', (x0, y0, w), (x1, y0 + w, h), 'plastic_grey'),
        _box('wall_back', (x0, y1 - w, w), (x1, y1, h), 'plastic_grey'),
        _box('wall_left', (x0, y0 + w, w), (x0 + w, y1 - w, h), 'plastic_grey'),
        _box('wall_right', (x1 - w, y0 + w, w), (x1, y1 - w, h), 'plastic_grey'),
        _box('rim_band', (x0 + w, y0 + w, h - 0.03), (x1 - w, y0 + 2 * w, h), 'plastic_red')], friction=0.7)


def _lab_stool():
    cx, cy, seat_top = -0.95, -0.95, 0.62
    parts = [_box('seat', (cx - 0.17, cy - 0.17, seat_top - 0.04), (cx + 0.17, cy + 0.17, seat_top), 'stool_seat')]
    for index, (sx, sy) in enumerate(((-1, -1), (-1, 1), (1, -1), (1, 1))):
        x, y = cx + sx * 0.13, cy + sy * 0.13
        parts.append(_box(f'leg_{index:02d}', (x - 0.016, y - 0.016, 0.0), (x + 0.016, y + 0.016, seat_top - 0.04),
                          'dark_steel'))
    parts.append(_box('ring_front', (cx - 0.114, cy - 0.146, 0.22), (cx + 0.114, cy - 0.114, 0.24), 'dark_steel'))
    parts.append(_box('ring_back', (cx - 0.114, cy + 0.114, 0.22), (cx + 0.114, cy + 0.146, 0.24), 'dark_steel'))
    return _prop('lab_stool', '实验凳', 'dark_steel', parts, friction=0.7)


def _static_props():
    return [_lab_bench(), _bench_lip(), _reagent_riser(), _tube_rack(), _tip_rack(), _hot_plate(),
            _fume_hood(), _sink_unit(), _window_frame(), _door_casing(), _safety_shower(), _wall_shelf(),
            _flammables_cabinet(), _first_aid_box(), _wall_notices(), _safety_signs(), _waste_bin(),
            _lab_stool()]


# --- fixtures ---------------------------------------------------------------------

def _fixture(fixture_id, asset_id, xyz, yaw=0.0):
    return {'id': fixture_id, 'asset_id': asset_id, 'asset_root': ASSET_ROOT,
            'position_m': [_r(v) for v in xyz],
            'quaternion_wxyz': [round(math.cos(yaw / 2.), 9), 0.0, 0.0, round(math.sin(yaw / 2.), 9)],
            'friction': 0.6}


def _fixtures():
    shelf_y = 1.885
    fixtures = [
        _fixture('bottle_shelf_00', 'reagent_bottle_amber_500ml', (-1.85, shelf_y, WALL_SHELF_BOARDS_Z[0])),
        _fixture('bottle_shelf_01', 'reagent_bottle_amber_500ml', (-1.62, shelf_y, WALL_SHELF_BOARDS_Z[0])),
        _fixture('bottle_shelf_02', 'reagent_bottle_clear_250ml', (-1.40, shelf_y, WALL_SHELF_BOARDS_Z[0])),
        _fixture('bottle_shelf_03', 'reagent_bottle_clear_250ml', (-1.75, shelf_y, WALL_SHELF_BOARDS_Z[1])),
        _fixture('bottle_shelf_04', 'reagent_bottle_amber_500ml', (-1.30, shelf_y, WALL_SHELF_BOARDS_Z[1])),
        _fixture('bottle_shelf_05', 'reagent_bottle_clear_250ml', (-1.05, shelf_y, WALL_SHELF_BOARDS_Z[1])),
        _fixture('bottle_riser_00', 'reagent_bottle_clear_250ml', (1.035, 0.18, RISER_BOARD_TOP_Z)),
        _fixture('bottle_riser_01', 'reagent_bottle_amber_500ml', (1.035, 0.30, RISER_BOARD_TOP_Z)),
        _fixture('bottle_riser_02', 'reagent_bottle_clear_250ml', (1.035, 0.55, RISER_BOARD_TOP_Z)),
        _fixture('micropipette', 'micropipette_stand', (0.90, 0.50, TABLE_TOP_Z)),
        _fixture('fire_extinguisher', 'fire_extinguisher', (1.12, -1.88, 0.0)),
    ]
    return fixtures


# --- objects ----------------------------------------------------------------------

def _object(object_id, asset_id, xy, support_top, sizes, friction):
    """A rest pose whose lowest collision point (its box bottom) is exactly ``support_top``."""
    size = sizes[asset_id]
    return {'id': object_id, 'asset_id': asset_id, 'asset_root': ASSET_ROOT,
            'position_m': [_r(xy[0]), _r(xy[1]), _r(support_top + size[2] / 2.)],
            'quaternion_wxyz': [1.0, 0.0, 0.0, 0.0], 'friction': friction,
            'rest_geometry': {'size_m': [_r(v) for v in size], 'origin': 'centre'}}


LOOSE_GLASSWARE = (
    # id, asset, xy; >= 50 mm between neighbouring footprints, all inside the reach envelope.
    ('beaker_50ml_00', 'beaker_50ml', (0.420, -0.020)),
    ('cork_ring_00', 'cork_ring', (0.540, 0.000)),
    ('beaker_250ml_00', 'beaker_250ml', (0.670, 0.000)),
    ('beaker_150ml_00', 'beaker_150ml', (0.450, 0.130)),
    ('erlenmeyer_flask_00', 'erlenmeyer_flask', (0.580, 0.140)),
    ('graduated_cylinder_00', 'graduated_cylinder', (0.720, 0.150)),
    ('stirring_rod_00', 'stirring_rod', (0.550, 0.260)),
)


def _build_objects(sizes):
    objects = []
    for index, (x, y) in enumerate(TUBE_SLOTS):
        objects.append(_object(f'test_tube_{index:02d}', 'test_tube', (x, y), TUBE_RACK_TOP_Z, sizes, 0.45))
    for index, (x, y) in enumerate(TIP_SLOTS):
        objects.append(_object(f'pipette_tip_{index:02d}', 'pipette_tip', (x, y), TIP_RACK_FLOOR_TOP_Z, sizes,
                               0.45))
    for object_id, asset_id, xy in LOOSE_GLASSWARE:
        friction = 0.70 if asset_id == 'cork_ring' else 0.42
        objects.append(_object(object_id, asset_id, xy, TABLE_TOP_Z, sizes, friction))
    return objects


# --- materials and cameras -------------------------------------------------------

def _materials():
    """Flat colours only (no texture dependency). Room/pedestal keys match add_room/add_pedestal."""
    def color(rgb, roughness, metallic=0.0):
        return {'kind': 'color', 'color': list(rgb), 'roughness': roughness, 'metallic': metallic}
    return {
        # Bound by surface key in room.add_room / room.add_pedestal.
        'floor': color([0.40, 0.44, 0.42], 0.45),
        'wall': color([0.86, 0.87, 0.84], 0.88),
        'ceiling': color([0.92, 0.92, 0.91], 0.94),
        'baseboard': color([0.22, 0.23, 0.24], 0.60),
        'top': color([0.20, 0.21, 0.23], 0.45, 0.50),
        'side': color([0.30, 0.33, 0.37], 0.55, 0.20),
        # Furniture and props.
        'black_bench_top': color([0.070, 0.075, 0.080], 0.30),
        'cabinet_grey': color([0.78, 0.79, 0.78], 0.50),
        'cabinet_door': color([0.70, 0.73, 0.76], 0.45),
        'handle_steel': color([0.62, 0.63, 0.65], 0.30, 0.80),
        'stainless': color([0.58, 0.60, 0.62], 0.32, 0.75),
        'dark_steel': color([0.22, 0.24, 0.26], 0.45, 0.60),
        'rubber_black': color([0.06, 0.06, 0.07], 0.80),
        'rack_white': color([0.90, 0.90, 0.88], 0.55),
        'rack_blue': color([0.22, 0.45, 0.72], 0.50),
        'hood_white': color([0.88, 0.89, 0.88], 0.50),
        'sash_glass': color([0.74, 0.82, 0.86], 0.15),
        'window_glass': color([0.80, 0.86, 0.90], 0.12),
        'window_frame': color([0.93, 0.93, 0.92], 0.50),
        'ceramic_white': color([0.92, 0.92, 0.90], 0.25),
        'hotplate_body': color([0.85, 0.86, 0.86], 0.45),
        'knob_black': color([0.08, 0.08, 0.09], 0.50),
        'display_dark': color([0.05, 0.10, 0.12], 0.20),
        'safety_green': color([0.10, 0.55, 0.30], 0.50),
        'safety_yellow': color([0.92, 0.72, 0.10], 0.45),
        'fire_red': color([0.75, 0.10, 0.08], 0.40),
        'paper': color([0.93, 0.92, 0.88], 0.85),
        'cardboard': color([0.62, 0.48, 0.32], 0.90),
        'poster_blue': color([0.25, 0.42, 0.68], 0.70),
        'plastic_grey': color([0.45, 0.47, 0.48], 0.60),
        'plastic_red': color([0.68, 0.22, 0.20], 0.55),
        'stool_seat': color([0.12, 0.13, 0.15], 0.60),
    }


THIRD_PERSON_EYE = (0.25, -1.45, 1.62)      # 1.92 m from the bench centre, 0.87 m above it
BENCH_VIEW_EYE = (1.75, -1.05, 1.60)        # 1.84 m; frames all four bench corners
WRIST_AIM = (0.58, 0.20, TABLE_TOP_Z)


def _cameras():
    table_top_centre = [_r(TABLE_CENTER[0]), _r(TABLE_CENTER[1]), TABLE_TOP_Z]
    return [
        {'id': 'third_person', 'kind': 'world', 'role': 'third_person',
         'eye': list(THIRD_PERSON_EYE), 'target': table_top_centre, 'focal_length_mm': 24.0,
         'resolution': [960, 720], 'title_zh': '第三人称视角'},
        {'id': 'bench_view', 'kind': 'world', 'role': 'table',
         'eye': list(BENCH_VIEW_EYE), 'target': [0.66, 0.20, 0.76], 'focal_length_mm': 28.0,
         'resolution': [960, 720], 'title_zh': '台面视角'},
        # Aimed at the bench (env solves the camera transform from the mount link's
        # world pose) and stood off along the view axis. offset_rpy_rad = [pi, 0, 0]
        # is the un-aimed fallback: a USD camera looks along its local -Z, and the
        # flipped camera then looks along C_MC +Z, i.e. along the fingers.
        {'id': 'wrist', 'kind': 'link', 'role': 'wrist', 'mount_link': 'right_hand_C_MC',
         'offset_xyz_m': [0.0, 0.0, 0.06], 'offset_rpy_rad': [math.pi, 0.0, 0.0],
         'aim_target_m': list(WRIST_AIM), 'standoff_m': 0.10,
         'clipping_m': [0.02, 20.0], 'resolution': [640, 480], 'title_zh': '手腕相机'},
    ]


SCENE = {
    'schema_version': 'isaacsim-scene-family-v1',
    'family': FAMILY,
    'title_zh': '化学实验室台',
    'description_zh': ('化学/生物实验室：黑色环氧树脂台面下为钢制柜体，台面放有双层试管架（12 支圆底试管穿过上层孔板）、'
                       '8 孔移液枪头盒（枪头尖朝下、卡环搁在孔板上）、50/150/250 ml 烧杯、125 ml 锥形瓶、'
                       '50 ml 量筒、六棱玻璃搅拌棒与软木烧瓶垫圈；台面远端有加热磁力搅拌器、移液器支架与试剂架。'
                       '房间有通风橱、带洗眼器的水槽、带窗框窗台的窗户、门框、紧急喷淋、墙面试剂架、'
                       '易燃品安全柜、急救箱、灭火器、废弃物桶与实验凳。玻璃为磨砂不透明材质。'),
    'room': {
        'enabled': True,
        'size_m': list(ROOM_SIZE_M),
        'origin_m': [0.0, 0.0, 0.0],
        'wall_thickness_m': WALL_THICKNESS_M,
        'baseboard_m': 0.09,
        'uv_scale': {'floor': [1.0, 1.0], 'wall': [1.0, 1.0]},
        # add_room binds by these keys, so each surface uses the material of the same name.
        'surfaces': {'floor': 'floor', 'wall': 'wall', 'ceiling': 'ceiling', 'baseboard': 'baseboard'},
        'openings': [DOOR, WINDOW],
        'ceiling_lights': [
            {'id': 'lamp_bench', 'position_m': [0.70, 0.20, 2.62], 'extent_m': [1.20, 0.60],
             'intensity': 9500.0, 'color': [1.0, 0.98, 0.95]},
            {'id': 'lamp_room', 'position_m': [-1.30, -0.80, 2.62], 'extent_m': [0.60, 0.60],
             'intensity': 5200.0, 'color': [0.98, 0.99, 1.0]},
            {'id': 'lamp_wall', 'position_m': [1.20, 1.40, 2.62], 'extent_m': [0.60, 0.60],
             'intensity': 5200.0, 'color': [0.98, 0.99, 1.0]},
        ],
        'dome': {'enabled': True, 'intensity': 350.0, 'color': [0.85, 0.90, 1.0], 'hdri_visible': False},
        'collide': {'floor': True, 'walls': True, 'ceiling': False},
    },
    'materials': _materials(),
    'cameras': _cameras(),
    'robot': {'base_position_m': [0.0, 0.0, 0.75], 'hand_id': 'sharpa_w01_right',
              'assembly_version': 'isaacsim_6_1_v2', 'assets_root': f'{DATA_ROOT}/assets',
              'arm_home_rad': [0.4, -0.5, 0.0, -2.3, 0.0, 1.8, 0.8],
              'arm_kp': 2000.0, 'arm_kd': 140.0, 'hand_kp': 3.0, 'hand_kd': 0.1},
    # The skirt fills the 0.65 m under the stand so it does not float above the floor.
    'pedestal': {'position_m': [0.0, 0.0, 0.70], 'half_extents_m': [0.22, 0.22, 0.05],
                 'skirt_height_m': 0.65, 'friction': 0.8},
    # The epoxy bench top: the only collider of the work surface (see lab_bench).
    'table': {'position_m': [_r(v) for v in TABLE_CENTER], 'half_extents_m': [_r(v) for v in TABLE_HALF],
              'friction': 0.55, 'material': 'black_bench_top'},
    'surfaces': [],
    'static_props': _static_props(),
    'fixtures': _fixtures(),
    'objects': _build_objects(asset_sizes()),
    'cloth': {'enabled': False},
    'physics': {'dt_s': 0.0005, 'control_dt_s': 0.02, 'gravity_m_s2': [0.0, 0.0, -9.81]},
    'isaacsim': {'physics_dt_s': 0.0005, 'robot_self_collisions': True,
                 'filter_thumb_cmc_structure': True},
}
