"""Heritage Hex furniture: ivory inlays, turned lacquer and dark timber.

The token is the union of two nonoverlapping analytic cylinders. Its mass is
a simulation design prior; COM and inertia use their actual volumes, not a
bounding box. The small visual roundovers, inlays and finishes are appearance
only: the v1 collision dimensions and material priors remain unchanged.
"""
from __future__ import annotations

import argparse
import hashlib
import math
from pathlib import Path

import numpy as np

from bench.simulation.assets.scene_parts import Asset, box_mesh, cylinder_mesh, write_library
from bench.simulation.assets import inlay_geometry
from bench.simulation.assets.inlay_geometry import (
    convex_outline, convex_prism, offset_outline, rounded_box,
    rounded_cylinder, rounded_rectangle, rounded_strip,
)

COLORS = {'red': [.42, .055, .037], 'blue': [.040, .145, .175]}
TIMBER = [.105, .061, .036]
IVORY = [.72, .68, .58]
INLAY = [.33, .27, .17]
GRID = [.055, .048, .035]
APPEARANCE = 'hex_heritage_v3'
GRID_WIDTH_M = .0015

# Original single-stroke lettering, normalized to a one-unit writing box.
# Geometry avoids a font or texture dependency in the native asset loader.
_LETTERING = {
    'A': [[(0, 0), (.5, 1), (1, 0)], [(.2, .38), (.8, .38)]],
    'B': [[(0, 0), (0, 1), (.65, 1), (1, .82), (1, .65), (.65, .5), (0, .5)],
          [(.65, .5), (1, .35), (1, .18), (.65, 0), (0, 0)]],
    'C': [[(1, .85), (.75, 1), (.25, 1), (0, .75), (0, .25), (.25, 0), (.75, 0), (1, .15)]],
    'D': [[(0, 0), (0, 1), (.6, 1), (1, .75), (1, .25), (.6, 0), (0, 0)]],
    'E': [[(1, 1), (0, 1), (0, 0), (1, 0)], [(0, .5), (.8, .5)]],
    'F': [[(1, 1), (0, 1), (0, 0)], [(0, .5), (.8, .5)]],
    'G': [[(1, .85), (.75, 1), (.25, 1), (0, .75), (0, .25), (.25, 0), (.75, 0),
           (1, .25), (1, .48), (.55, .48)]],
    '1': [[(.15, .72), (.55, 1), (.55, 0)], [(.15, 0), (.95, 0)]],
    '2': [[(0, .8), (.2, 1), (.75, 1), (1, .8), (1, .6), (0, 0), (1, 0)]],
    '3': [[(0, 1), (1, 1), (.5, .55), (.8, .55), (1, .35), (1, .2), (.75, 0), (.15, 0), (0, .15)]],
    '4': [[(.78, 0), (.78, 1), (0, .35), (1, .35)]],
    '5': [[(1, 1), (0, 1), (0, .55), (.75, .55), (1, .35), (1, .2), (.75, 0), (.15, 0), (0, .15)]],
    '6': [[(.9, .9), (.7, 1), (.25, 1), (0, .7), (0, .2), (.2, 0), (.75, 0), (1, .2),
           (1, .45), (.75, .6), (0, .6)]],
    '7': [[(0, 1), (1, 1), (.3, 0)]],
}


def _translated(mesh, position):
    result = mesh.copy()
    result.apply_translation(position)
    return result


def token_mass_properties(geometry):
    g = geometry
    cylinders = [(g.piece_radius_m, g.piece_base_height_m, 0.),
                 (g.handle_radius_m, g.handle_height_m,
                  (g.piece_base_height_m + g.handle_height_m)/2)]
    volume = sum(math.pi*r*r*h for r, h, _ in cylinders)
    parts = [(r, h, z, g.piece_mass_kg*math.pi*r*r*h/volume) for r, h, z in cylinders]
    center = sum(z*m for _, _, z, m in parts)/g.piece_mass_kg
    ix = sum(m*(3*r*r+h*h)/12 + m*(z-center)**2 for r, h, z, m in parts)
    iz = sum(m*r*r/2 for r, _, _, m in parts)
    return {'center_of_mass_m': [0., 0., center], 'inertia_diag_kg_m2': [ix, ix, iz],
            'inertia_method': 'Analytic uniform-density two-cylinder union about COM; parallel-axis theorem.',
            'design_density_kg_m3': g.piece_mass_kg/volume,
            'mass_status': 'Simulation design prior, not a measured manufactured piece',
            'origin': 'Centre of circular base; COM is explicitly offset along +Z'}


def token(color, geometry):
    g = geometry
    props = token_mass_properties(g)
    asset = Asset('hex_token_'+color, dynamic=True, mass_kg=g.piece_mass_kg,
                  friction=.65, center_of_mass_m=props['center_of_mass_m'])
    for label, radius, height, z in (
            ('base', g.piece_radius_m, g.piece_base_height_m, 0.),
            ('handle', g.handle_radius_m, g.handle_height_m,
             (g.piece_base_height_m+g.handle_height_m)/2)):
        # The rounded display stays inside the unchanged analytic collider.
        # Most of the handle remains a full-radius straight pinch surface.
        rounding = min(.0015, height*.12, radius*.10)
        shade = (np.asarray(COLORS[color]) * (.86 if label == 'base' else 1.)).tolist()
        asset.add_display(_translated(rounded_cylinder(radius, height, rounding), [0, 0, z]),
                          shade, label, roughness=.25 if label == 'base' else .22, metallic=0.)
        asset.add_collision(None, label, geometry='cylinder', radius_m=radius,
                            half_height_m=height/2, position_m=[0., 0., z])
    asset.data.update(props)
    asset.data.update(design_dimensions={'base_diameter_m': 2*g.piece_radius_m,
                                        'base_height_m': g.piece_base_height_m,
                                        'handle_diameter_m': 2*g.handle_radius_m,
                                        'handle_height_m': g.handle_height_m},
                      material_prior='Existing task rigid material friction0.65, restitution0; no grasp tuning.',
                      appearance={'version': APPEARANCE, 'finish': 'Glazed cinnabar / indigo lacquer ceramic',
                                  'visual_roundover_max_m': .0015,
                                  'collision_shape': 'Unchanged two analytic cylinders',
                                  'visual_mass_is_not_used': True})
    return asset


def _strip(a, b, width, z):
    import trimesh
    a, b = np.array(a), np.array(b)
    direction = b-a
    length = np.linalg.norm(direction)
    mesh = trimesh.creation.box([length, width, .0001])
    angle = math.atan2(direction[1], direction[0])
    matrix = trimesh.transformations.rotation_matrix(angle, [0, 0, 1])
    matrix[:3, 3] = [*(a+b)/2, z]
    mesh.apply_transform(matrix)
    return mesh


def _coordinate_labels(asset, geometry):
    import trimesh
    g = geometry
    height, width, stroke_width = .0105, .0066, .00125
    positions = []
    for index in range(7):
        x, _ = g.cell_local_xy(index, 0)
        positions.append((chr(ord('A')+index), 'q', index, [x, -g.board_size_xy_m[1]/2+.0072]))
        corner = np.asarray(g.cell_local_xy(0, index))
        normal = np.array([-math.sqrt(3)/2, .5])
        point = corner + normal*(g.pitch_m/math.sqrt(3)+.012)
        positions.append((str(index+1), 'r', index, point.tolist()))
    metadata = []
    for text, axis, index, center in positions:
        lines = []
        for stroke in _LETTERING[text]:
            path = (np.array(stroke)-.5)*[width, height] + center
            lines.extend(_strip(a, b, stroke_width, g.board_thickness_m+.00007)
                         for a, b in zip(path[:-1], path[1:]))
        asset.add_display(trimesh.util.concatenate(lines), [.83, .79, .67],
                          'coordinate_'+text, roughness=.85, metallic=0.)
        metadata.append({'text': text, 'axis': axis, 'index': index,
                         'anchor_local_xy_m': center, 'height_m': height,
                         'stroke_width_m': stroke_width,
                         'classification': 'visual_only_coordinate_inlay'})
    return metadata


def _frame_pinstripe(geometry):
    """Fine mitred frame with intentional bottom-edge gaps for legible letters."""
    import trimesh
    g = geometry
    sx, sy = g.board_size_xy_m
    outline = offset_outline(rounded_rectangle([sx-.012, sy-.012], .005), -.000325)
    letters = [g.cell_local_xy(q, 0)[0] for q in range(7)]
    strips = []
    for a, b in zip(outline, np.roll(outline, -1, axis=0)):
        if abs(a[1]-b[1]) < 1.e-12 and a[1] < -sy/2+.007:
            left, right = sorted([a[0], b[0]])
            for x in letters:
                if left < x-.0065 < right:
                    strips.append(_strip([left, a[1]], [x-.0065, a[1]], .00065,
                                         g.board_thickness_m+.000055))
                    left = x+.0065
            if right > left:
                strips.append(_strip([left, a[1]], [right, a[1]], .00065,
                                     g.board_thickness_m+.000055))
        else:
            strips.append(_strip(a, b, .00065, g.board_thickness_m+.000055))
    return trimesh.util.concatenate(strips)


def _timber_grain(field, size, z):
    """Sparse irregular veneer grain, well clear of the playing field."""
    import trimesh
    sx, sy = size
    edges = np.roll(field, -1, axis=0)-field
    outward = np.c_[edges[:, 1], -edges[:, 0]]
    outward /= np.linalg.norm(outward, axis=1)[:, None]

    def wood_only(point):
        return np.sum((point-field)*outward, axis=1).max() > .020

    lines = []
    for index, y in enumerate((.016, .041, .074, .096, .121, .149, .170, .198, .223)):
        x = np.linspace(-sx/2+.020, -.06, 65)
        curve = np.c_[x, y+.0018*np.sin(27*x+index*.79)+.0008*np.sin(75*x+index*1.3)]
        for sign in (-1, 1):
            for a, b in zip(sign*curve[:-1], sign*curve[1:]):
                if wood_only(a) and wood_only(b) and max(abs(a[1]), abs(b[1])) < sy/2-.020:
                    lines.append(_strip(a, b, .00028, z+.000005))
    return trimesh.util.concatenate(lines)


def board(geometry):
    import trimesh
    g = geometry
    sx, sy = g.board_size_xy_m
    asset = Asset('hex_board', dynamic=False, mass_kg=0, friction=.4)
    top = g.board_thickness_m
    asset.add_display(_translated(rounded_box([sx, sy, top], .009, min(.0011, top*.1)),
                                 [0, 0, top/2]),
                      TIMBER, 'wood_board', roughness=.46, metallic=0.)
    asset.add_box_collision([sx/2, sy/2, g.board_thickness_m/2], 'solid_board',
                            [0, 0, g.board_thickness_m/2])
    edges, vertices = set(), []
    for q in range(7):
        for r in range(7):
            x, y = g.cell_local_xy(q, r)
            radius = g.pitch_m/math.sqrt(3)
            corners = [(x+radius*math.cos(math.pi/6+i*math.pi/3),
                        y+radius*math.sin(math.pi/6+i*math.pi/3)) for i in range(6)]
            vertices.extend(corners)
            for a, b in zip(corners, corners[1:]+corners[:1]):
                edges.add(tuple(sorted(tuple(round(v, 9) for v in p) for p in (a, b))))
    # A single neutral playing field follows the cells' convex outline. The
    # unused triangular corners remain timber instead of a large yellow mat.
    field = convex_outline(vertices)
    asset.add_display(_timber_grain(field, [sx, sy], top), [.128, .076, .044],
                      'walnut_grain', roughness=.54, metallic=0.)
    asset.add_display(convex_prism(offset_outline(field, .0080), top-.0002, top+.00002),
                      INLAY, 'field_border', roughness=.60, metallic=0.)
    asset.add_display(convex_prism(offset_outline(field, .0066), top-.0001, top+.00004),
                      IVORY, 'ivory_playing_field', roughness=.68, metallic=0.)
    lines = [_strip(a, b, GRID_WIDTH_M, top+.00008) for a, b in sorted(edges)]
    asset.add_display(trimesh.util.concatenate(lines), GRID, 'hex_grid', roughness=.88, metallic=0.)
    asset.add_display(_frame_pinstripe(g),
                      INLAY, 'frame_pinstripe', roughness=.60, metallic=0.)
    # Two restrained key-fret inlays occupy the otherwise empty wood corners.
    fret = np.array([[-.018, -.018], [-.018, .018], [.018, .018], [.018, -.018],
                     [-.008, -.018], [-.008, .008], [.008, .008], [.008, -.007]])
    corner_lines = []
    for sign in (-1, 1):
        center = sign*np.array([sx/2-.067, -(sy/2-.067)])
        path = center + sign*fret
        corner_lines.extend(_strip(a, b, .0007, top+.00007) for a, b in zip(path[:-1], path[1:]))
    asset.add_display(trimesh.util.concatenate(corner_lines), INLAY, 'corner_frets', roughness=.58, metallic=0.)
    pins = [_translated(cylinder_mesh(.0014, .00008, 32), [x*(sx/2-.012), y*(sy/2-.012), top+.00006])
            for x in (-1, 1) for y in (-1, 1)]
    asset.add_display(trimesh.util.concatenate(pins), [.39, .31, .18], 'corner_pins', roughness=.50, metallic=.42)
    # Parallel goal inlays remain outside every complete hexagon. Colour and
    # opposing-side semantics are identical to the rules; only finish changes.
    for color, axis in (('red', 1), ('blue', 0)):
        bands = []
        for side in (0, 6):
            q0, r0 = (side, 0) if axis == 0 else (0, side)
            q1, r1 = (side, 6) if axis == 0 else (6, side)
            a, b = np.array(g.cell_local_xy(q0, r0)), np.array(g.cell_local_xy(q1, r1))
            direction = (b-a)/np.linalg.norm(b-a)
            normal = np.array([-direction[1], direction[0]])
            mid = (a+b)/2
            if np.dot(normal, mid) < 0:
                normal = -normal
            shift = normal*(g.pitch_m/math.sqrt(3)+.0023)
            bands.append(rounded_strip(a+shift-direction*g.pitch_m*.26,
                                       b+shift+direction*g.pitch_m*.26, .0028,
                                       top+.000045, top+.00012))
        asset.add_display(trimesh.util.concatenate(bands), COLORS[color], color+'_goal_edges',
                          roughness=.32, metallic=0.)
    coordinates = _coordinate_labels(asset, g)
    asset.data.update(geometry=g.specification(),
                      coordinate_labels=coordinates,
                      surface_note='Unchanged flat solid support. Inlays rise at most 0.13mm and do not collide.',
                      material_prior='Workroom support simulation prior friction0.4, restitution0.',
                      appearance={'version': APPEARANCE, 'grid_width_m': GRID_WIDTH_M,
                                  'field_finish': 'Warm ivory with fine smoke-brown hexagonal inlays',
                                  'frame_finish': 'Dark walnut with fine key-fret corner inlays',
                                  'maximum_visual_top_offset_m': .00013,
                                  'collision_shape': 'Unchanged single solid rectangular box'})
    return asset


def stand(sign, base_height=.95):
    """Cantilever column stays outside the table; cap supports the real base."""
    color = 'red' if sign < 0 else 'blue'
    asset = Asset('hex_stand_'+color, dynamic=False, mass_kg=0, friction=.4)
    # Origin is room world XY origin. The same four exact solid box colliders
    # remain; modest edge roundovers and wood panels are appearance only.
    for name, xyz, size, shade in (
        ('foot', [.65, sign*.74, .025], [.40, .38, .05], [.065, .053, .039]),
        ('column', [.65, sign*.74, (base_height-.04)/2], [.19, .19, base_height-.04], [.145, .085, .046]),
        ('cantilever', [.65, sign*.625, base_height-.035], [.22, .41, .03], [.075, .060, .042]),
        ('mount_plate', [.65, sign*.52, base_height-.0125], [.22, .20, .025], [.17, .18, .16]),
    ):
        asset.add_display(_translated(rounded_box(size, .003, .0008), xyz), shade, name,
                          roughness=.45 if name == 'column' else .39,
                          metallic=0. if name == 'column' else .35)
        asset.add_box_collision(np.asarray(size)/2, name, xyz)
    # Flush panel veneers break up the engineered column without changing its
    # silhouette or making a slender visual hide a large invisible collider.
    panels, pinstripes = [], []
    panel_height = base_height-.23
    for axis in (0, 1):
        for side in (-1, 1):
            center = np.array([.65, sign*.74, (base_height-.04)/2])
            center[axis] += side*.095045
            size = np.array([.142, .142, panel_height])
            size[axis] = .00007
            panels.append(_translated(box_mesh(size), center))
            for offset in (-.073, .073):
                line = center.copy()
                line[1-axis] += offset
                size = np.array([.0011, .0011, panel_height+.012])
                size[axis] = .00007
                pinstripes.append(_translated(box_mesh(size), line))
    import trimesh
    asset.add_display(trimesh.util.concatenate(panels), TIMBER, 'recessed_timber_panels', roughness=.49, metallic=0.)
    asset.add_display(trimesh.util.concatenate(pinstripes), [.28, .22, .13], 'panel_pinstripes', roughness=.55, metallic=.2)
    asset.data['installation'] = 'Ideal rigid anchored pedestal; column clears table edge, original robot geometry unchanged.'
    asset.data['appearance'] = {'version': APPEARANCE, 'finish': 'Walnut panels over dark bronze support',
                               'maximum_veneer_offset_m': .00008,
                               'collision_shape': 'Unchanged four solid boxes'}
    return asset


def build_assets(output, geometry):
    values = [board(geometry), token('red', geometry), token('blue', geometry), stand(-1), stand(1)]
    generator = 'scenes/workrooms/card_room/tasks/hex7x7/assets.py'
    result = write_library(output, [(a.asset_id, a) for a in values], generator=generator,
                           extra={'task': 'hex7x7_dual_robot_v1', 'geometry': geometry.specification(),
                                  'appearance_version': APPEARANCE,
                                  'generator_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                                  'geometry_helper_sha256': hashlib.sha256(Path(inlay_geometry.__file__).read_bytes()).hexdigest()})
    Path(output, 'LICENSE.txt').write_text('Original procedural task geometry and inlay designs. CC0-1.0.\n'
                                         'No external textures or third-party visual assets.\n')
    return result


def main():
    from scenes.board_games.hex import HexGeometry
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    import json
    print(json.dumps(build_assets(args.output, HexGeometry()), indent=2))


if __name__ == '__main__':
    main()
