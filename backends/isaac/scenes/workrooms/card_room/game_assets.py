"""Black and white wooden game sets for the fixed heritage card room.

The piece mass, analytic two-cylinder collisions, COM, inertia and material prior
remain those of the original Hex task. The three boards have their declared
single solid box support. New pedestal positions and heights match the task's
robot bases; their feet remain on the floor. No external textures, simulator imports or mutable
module configuration are needed by this deterministic CPU recipe.
"""
from __future__ import annotations

import argparse
import hashlib
import math
from pathlib import Path

import numpy as np

from bench.simulation.assets import inlay_geometry
from bench.simulation.assets.inlay_geometry import (
    outline_ring, rounded_box,
    rounded_cylinder, rounded_rectangle, rounded_strip,
)
from bench.simulation.assets.scene_parts import Asset, write_library
from .tasks.hex7x7 import assets as heritage

APPEARANCE = 'tabletop_black_white_wood_v3'
COLORS = {'black': [.023, .026, .023], 'white': [.80, .82, .78]}
BOARD_WOOD = [.245, .135, .063]
BOARD_EDGE = [.125, .064, .029]
GRID = [.025, .018, .011]
GRID_WIDTH_M = .0022
MAXIMUM_BOARD_TOP_OFFSET_M = .00013
HEX_COORDINATE_LAYOUT = 'clear_goal_frame_v2'
# Growth rings have irregular widths. All material meshes share one surface;
# there are no floating lines or added collision shapes on a wooden token.
_GROWTH_RADII = np.cumsum(np.random.default_rng(1946).uniform(.0016, .0037, 100))
_GROWTH_RADII.setflags(write=False)


def _translated(mesh, position):
    result = mesh.copy()
    result.apply_translation(position)
    return result


def _flat_mesh(vertices, faces):
    import trimesh
    mesh = trimesh.Trimesh(vertices=np.asarray(vertices, dtype=float),
                           faces=np.asarray(faces, dtype=np.int64), process=False)
    mesh.update_faces(mesh.nondegenerate_faces())
    mesh.remove_unreferenced_vertices()
    return mesh


def _ribbon(path, widths, z):
    """A thin display surface, not a row of little three-dimensional boxes."""
    path = np.asarray(path, dtype=float)
    normal = np.gradient(path, axis=0)
    normal = np.c_[-normal[:, 1], normal[:, 0]]
    normal /= np.linalg.norm(normal, axis=1)[:, None]
    offset = normal * np.asarray(widths)[:, None] / 2
    vertices = np.empty((len(path)*2, 3))
    vertices[::2, :2], vertices[1::2, :2] = path-offset, path+offset
    vertices[:, 2] = z
    faces = []
    for i in range(len(path)-1):
        a = 2*i
        faces.extend([(a, a+2, a+3), (a, a+3, a+1)])
    return _flat_mesh(vertices, faces)


def _veneer_grain(size, z):
    """Restrained, nonuniform straight grain, in two compact tonal layers.

    A shared local warp bends adjacent fibres together. Random growth spacing,
    varied lengths and tapered ends avoid a uniform comb or repeated sine-wave
    motif. These veneers are open visual meshes only, below the game markings.
    """
    import trimesh
    sx, sy = size
    rng = np.random.default_rng(741)
    groups = [[], []]
    y = -sy/2+.014
    while y < sy/2-.014:
        start = -sx/2+rng.uniform(.010, .055)
        end = sx/2-rng.uniform(.010, .055)
        if rng.random() < .28:
            # Discontinuous pores let the medium-tone wood remain dominant.
            if rng.random() < .5:
                start += sx*rng.uniform(.10, .25)
            else:
                end -= sx*rng.uniform(.10, .25)
        x = np.linspace(start, end, 42)
        t = (x-start)/(end-start)
        warp = (.0065*np.tanh((y-.04*sy)/(.10*sy))
                * np.exp(-((x-.13*sx)/(.27*sx))**2)
                + .0013*np.sin(11*x/sx + 4*y/sy)
                + .0004*np.sin(29*x/sx + 17*y/sy))
        path = np.c_[x, y+warp]
        width = rng.uniform(.00030, .00075)
        taper = np.minimum(1., np.minimum(t, 1-t)*9.)
        varying_width = width*taper*(1.+.28*np.sin(7*t+rng.uniform(0, 6)))
        # A softer wide edge and a narrow centre read as veneer, without
        # lighting-dependent shading from raised or bevelled stripe geometry.
        groups[0].append(_ribbon(path, varying_width*3.2, z))
        groups[1].append(_ribbon(path, varying_width, z+.000006))
        y += rng.uniform(.007, .015)
    return [trimesh.util.concatenate(parts) for parts in groups]


def _cap(radius, z, segments=160, rings=10):
    """Top disc with enough radial samples for subtle end grain on a token."""
    angles = np.arange(segments)*2*math.pi/segments
    vertices = [[0., 0., z]]
    for r in np.linspace(radius/rings, radius, rings):
        vertices.extend(np.c_[r*np.cos(angles), r*np.sin(angles), np.full(segments, z)])
    faces = [(0, 1+j, 1+(j+1)%segments) for j in range(segments)]
    for i in range(rings-1):
        a, b = 1+i*segments, 1+(i+1)*segments
        for j in range(segments):
            k = (j+1)%segments
            faces.extend([(a+j, b+j, b+k), (a+j, b+k, a+k)])
    return _flat_mesh(vertices, faces)


def _wooden_cylinder(radius, height, rounding, z):
    """Partition an unchanged rounded envelope into three matte wood tones."""
    import trimesh
    surface = rounded_cylinder(radius, height, rounding)
    # The original top is a triangle fan; replace it with a radial surface so
    # colour boundaries never become long artificial spokes through the cap.
    top_faces = np.all(np.isclose(surface.triangles[:, :, 2], height/2,
                                 rtol=0., atol=1.e-12), axis=1)
    surface.update_faces(~top_faces)
    surface.remove_unreferenced_vertices()
    surface = trimesh.util.concatenate([surface, _cap(radius-rounding, height/2)])
    surface.merge_vertices()
    surface.fix_normals()
    center = surface.triangles_center
    # An off-centre log section produces gently curved top grain and largely
    # longitudinal fibres on the turned handle. Growth radii are not periodic.
    x = center[:, 0]+.033+.0010*np.sin(65*(center[:, 2]+z))
    y = (center[:, 1]+.109)*.79
    distance = np.sqrt(x*x+y*y)
    ring = np.searchsorted(_GROWTH_RADII, distance)
    lower = _GROWTH_RADII[ring-1]
    phase = (distance-lower)/(_GROWTH_RADII[ring]-lower)
    tone = np.where(phase < .11, 0, np.where(phase < .27, 1, 2))
    # The underside is not decorated. Its mesh is still present and unchanged.
    tone[center[:, 2] < -height/2+1.e-12] = 2
    result = []
    for index in range(3):
        part = surface.submesh([np.flatnonzero(tone == index)], append=True, repair=False)
        part.apply_translation([0., 0., z])
        result.append(part)
    return result


def token_mass_properties(geometry):
    """Exactly the established analytic uniform-density two-cylinder prior."""
    return heritage.token_mass_properties(geometry)


def token(color, geometry):
    if color not in COLORS:
        raise ValueError('Wooden game pieces must be black or white')
    g = geometry
    props = token_mass_properties(g)
    asset = Asset('token_'+color, dynamic=True, mass_kg=g.piece_mass_kg,
                  friction=.65, center_of_mass_m=props['center_of_mass_m'])
    for label, radius, height, z in (
            ('base', g.piece_radius_m, g.piece_base_height_m, 0.),
            ('handle', g.handle_radius_m, g.handle_height_m,
             (g.piece_base_height_m+g.handle_height_m)/2)):
        rounding = min(.0015, height*.12, radius*.10)
        factors = (.84, .94, 1.) if color == 'white' else (.67, .84, 1.)
        for index, (mesh, factor) in enumerate(zip(_wooden_cylinder(radius, height, rounding, z), factors)):
            shade = np.asarray(COLORS[color])*factor*(.96 if label == 'base' else 1.)
            asset.add_display(mesh, shade.tolist(), f'{label}_wood_{index}',
                              roughness=.68 if index == 0 else .61, metallic=0.)
        asset.add_collision(None, label, geometry='cylinder', radius_m=radius,
                            half_height_m=height/2, position_m=[0., 0., z])
    asset.data.update(props)
    asset.data.update(
        design_dimensions={'base_diameter_m': 2*g.piece_radius_m,
                           'base_height_m': g.piece_base_height_m,
                           'handle_diameter_m': 2*g.handle_radius_m,
                           'handle_height_m': g.handle_height_m},
        material_prior='Preserved simulation prior friction0.65, restitution0; not measured wood friction.',
        appearance={'version': APPEARANCE,
                    'finish': ('Matte ebony-stained wood' if color == 'black' else 'Matte white-stained wood'),
                    'woodgrain': 'Irregular growth-ring tones partition the original turned display envelope',
                    'visual_roundover_max_m': .0015,
                    'collision_shape': 'Unchanged two analytic cylinders',
                    'visual_mass_is_not_used': True})
    return asset


def _strip(a, b, width, z):
    return _ribbon([a, b], [width, width], z)


def _grid_edges(g):
    """Local cell boundaries, deduplicated before mesh authoring."""
    if g.shape == 'square':
        half = g.size*g.pitch_m/2
        edges = []
        for i in range(g.size+1):
            p = -half+i*g.pitch_m
            edges.extend([((-half, p), (half, p)), ((p, -half), (p, half))])
        return edges
    edges = set()
    for q in range(g.size):
        for r in range(g.size):
            x, y = g.cell_local_xy(q, r)
            radius = g.pitch_m/math.sqrt(3)
            vertices = [(x+radius*math.cos(math.pi/6+i*math.pi/3),
                         y+radius*math.sin(math.pi/6+i*math.pi/3)) for i in range(6)]
            for a, b in zip(vertices, vertices[1:]+vertices[:1]):
                edges.add(tuple(sorted(tuple(round(v, 9) for v in p) for p in (a, b))))
    return sorted(edges)


def _coordinate_labels(asset, g):
    """Small edge coordinates reuse the original authored single-stroke glyphs."""
    import trimesh
    height, width, stroke = .0105, .0066, .00125
    top = g.board_thickness_m
    positions = []
    for index in range(g.size):
        x, _ = g.cell_local_xy(index, 0)
        # Hex has a goal band below its cells: leave a real gap above the
        # shorter letter, while keeping its lower strokes on the flat wood rim.
        q_height = .009 if g.shape == 'hex' else height
        q_margin = .0065 if g.shape == 'hex' else .008
        positions.append((chr(ord('A')+index), 'q', index,
                          [x, -g.board_size_xy_m[1]/2+q_margin], q_height))
        if g.shape == 'hex':
            corner = np.asarray(g.cell_local_xy(0, index))
            normal = np.array([-math.sqrt(3)/2, .5])
            tangent = np.array([.5, math.sqrt(3)/2])
            # Move outward from the white goal, then along its edge so the
            # lowest numeral still clears the left frame and rounded board.
            point = corner+normal*(g.pitch_m/math.sqrt(3)+.016)+tangent*.009
        else:
            _, y = g.cell_local_xy(0, index)
            point = np.array([-g.board_size_xy_m[0]/2+.008, y])
        positions.append((str(index+1), 'r', index, point.tolist(), height))
    metadata = []
    for text, axis, index, center, glyph_height in positions:
        lines = []
        for line in heritage._LETTERING[text]:
            path = (np.asarray(line)-.5)*[width, glyph_height]+center
            lines.extend(_strip(a, b, stroke, top+.00009) for a, b in zip(path[:-1], path[1:]))
        asset.add_display(trimesh.util.concatenate(lines), [.69, .61, .45],
                          'coordinate_'+text, roughness=.76, metallic=0.)
        metadata.append({'text': text, 'axis': axis, 'index': index,
                         'anchor_local_xy_m': center, 'height_m': glyph_height,
                         'stroke_width_m': stroke,
                         'classification': 'visual_only_coordinate_inlay'})
    return metadata


def _goal_edges(asset, g):
    import trimesh
    if g.shape != 'hex':
        return []
    top = g.board_thickness_m
    goals = []
    for color, axis in (('black', 1), ('white', 0)):
        bands = []
        for side in (0, g.size-1):
            q0, r0 = (side, 0) if axis == 0 else (0, side)
            q1, r1 = (side, g.size-1) if axis == 0 else (g.size-1, side)
            a, b = np.array(g.cell_local_xy(q0, r0)), np.array(g.cell_local_xy(q1, r1))
            direction = (b-a)/np.linalg.norm(b-a)
            normal = np.array([-direction[1], direction[0]])
            if np.dot(normal, (a+b)/2) < 0:
                normal = -normal
            # At least 1.6 mm of actual walnut separates goals from full cell
            # boundaries, so the black edge cannot merge with a black grid line.
            shift = normal*(g.pitch_m/math.sqrt(3)+.0046)
            bands.append(rounded_strip(a+shift-direction*g.pitch_m*.22,
                                       b+shift+direction*g.pitch_m*.22, .0040,
                                       top+.000065, top+.00012, segments=6))
            goals.append({'color': color, 'axis': 'q' if axis == 0 else 'r',
                          'side': side, 'width_m': .0040,
                          'classification': 'visual_only_goal_inlay'})
        asset.add_display(trimesh.util.concatenate(bands), COLORS[color],
                          color+'_goal_edges', roughness=.65, metallic=0.)
    return goals


def board(geometry):
    import trimesh
    g = geometry
    sx, sy = g.board_size_xy_m
    top = g.board_thickness_m
    asset = Asset('board', dynamic=False, mass_kg=0, friction=.4)
    asset.add_display(_translated(rounded_box([sx, sy, top], .007, .0010), [0, 0, top/2]),
                      BOARD_EDGE, 'walnut_board_edge', roughness=.62, metallic=0.)
    # Medium walnut covers the entire field, including the unused Hex corners.
    # Nothing light enough to merge with a white piece is used as the field.
    surface = inlay_geometry.convex_prism(rounded_rectangle([sx-.005, sy-.005], .005),
                                         top-.00004, top+.000012)
    asset.add_display(surface, BOARD_WOOD, 'walnut_playing_surface', roughness=.68, metallic=0.)
    grain = _veneer_grain([sx-.005, sy-.005], top+.000025)
    for index, (mesh, factor) in enumerate(zip(grain, (.94, .86))):
        asset.add_display(mesh, (np.asarray(BOARD_WOOD)*factor).tolist(),
                          f'walnut_veneer_grain_{index}', roughness=.71, metallic=0.)
    asset.add_box_collision([sx/2, sy/2, top/2], 'solid_board', [0, 0, top/2])
    edges = _grid_edges(g)
    lines = [_strip(a, b, GRID_WIDTH_M, top+.000075) for a, b in edges]
    asset.add_display(trimesh.util.concatenate(lines), GRID,
                      'hex_grid' if g.shape == 'hex' else 'square_grid', roughness=.80, metallic=0.)
    # A fine dark perimeter frames the board without competing with cell lines.
    outline = rounded_rectangle([sx-.010, sy-.010], .004)
    # The established Hex frame helper leaves deliberate gaps around A-G.
    # A continuous bottom pinstripe would cut through their lower strokes.
    frame = (heritage._frame_pinstripe(g) if g.shape == 'hex' else
             outline_ring(outline, .0006, top+.00004, top+.00006))
    asset.add_display(frame, BOARD_EDGE, 'frame_inlay', roughness=.72, metallic=0.)
    goals = _goal_edges(asset, g)
    coordinates = _coordinate_labels(asset, g)
    asset.data.update(
        geometry=g.specification(), coordinate_labels=coordinates, goal_edges=goals,
        surface_note='Flat solid box support; display veneers and inlays do not collide.',
        material_prior='Preserved workroom support simulation prior friction0.4, restitution0.',
        appearance={'version': APPEARANCE, 'grid_width_m': GRID_WIDTH_M,
                    'grid_edge_count': len(edges),
                    'coordinate_layout': (HEX_COORDINATE_LAYOUT if g.shape == 'hex'
                                          else 'square_edge_coordinates_v1'),
                    'field_finish': 'Medium warm walnut with restrained irregular veneer grain',
                    'piece_contrast': 'Pale white-stained wood is lighter than every field tone',
                    'maximum_visual_top_offset_m': MAXIMUM_BOARD_TOP_OFFSET_M,
                    'collision_shape': 'One solid rectangular box at the declared board dimensions'})
    return asset


def stand(sign, base_height=None, base_distance=None):
    """Rigid pedestal with a floor-supported column and task-matched mount."""
    from .robot_layout import ROBOT_BASE_DISTANCE_M, ROBOT_BASE_HEIGHT_M
    base_height = ROBOT_BASE_HEIGHT_M if base_height is None else base_height
    base_distance = ROBOT_BASE_DISTANCE_M if base_distance is None else base_distance
    if sign not in (-1, 1):
        raise ValueError('Pedestal side must be -1 or +1')
    asset = heritage.stand(sign, base_height)
    offset = np.array([0., sign*(base_distance-.52), 0.])
    for _, mesh, _ in asset.visual:
        mesh.apply_translation(offset)
    for _, mesh, component in asset.collision:
        if mesh is not None:
            mesh.apply_translation(offset)
        else:
            component['position_m'] = (np.asarray(component['position_m'])+offset).tolist()
    asset.asset_id = 'stand_left' if sign < 0 else 'stand_right'
    asset.data['mount_position_m'] = [.65, sign*base_distance, base_height]
    asset.data['installation'] = 'Ideal rigid anchored pedestal; lowered column keeps foot on floor; mount matches task robot base.'
    asset.data['appearance'] = dict(asset.data['appearance'],
                                    original_template_version=heritage.APPEARANCE,
                                    version=APPEARANCE,
                                    collision_shape='Four solid boxes with versioned mount height and position')
    return asset


def build_assets(output, geometry):
    """Write one immutable task-parts library; never overwrite existing data."""
    values = [board(geometry), token('black', geometry), token('white', geometry), stand(-1), stand(1)]
    from . import reserve_assets
    values.extend(reserve_assets.reserve_tray(name, positions, surface_z=geometry.reserve_surface_z_m)
                  for name, positions in geometry.reserve_stations().items())
    generator = 'scenes/workrooms/card_room/game_assets.py'
    from scenes.board_games.tabletop import geometry as geometry_module
    from . import robot_layout
    dependencies = [Path(__file__), Path(inlay_geometry.__file__), Path(heritage.__file__),
                    Path(geometry_module.__file__), Path(robot_layout.__file__), Path(reserve_assets.__file__)]
    result = write_library(
        output, [(a.asset_id, a) for a in values], generator=generator,
        extra={'task': geometry.game_id, 'geometry': geometry.specification(),
               'appearance_version': APPEARANCE,
               'generator_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               'geometry_helper_sha256': hashlib.sha256(Path(inlay_geometry.__file__).read_bytes()).hexdigest(),
               'recipe_sources_sha256': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in dependencies},
               'native_visual_acceptance': 'Not established by CPU asset generation or unit tests'})
    Path(output, 'LICENSE.txt').write_text(
        'Original procedural task geometry, grain and inlay designs. CC0-1.0.\n'
        'No external textures or third-party visual assets.\n'
        'Original Hex analytic piece parameters preserved; pedestal mount height and position are versioned.\n')
    return result


def main():
    import json
    from scenes.board_games.tabletop.geometry import geometry_for
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--game', choices=['tic_tac_toe_3x3', 'connect_four_5x5', 'hex7x7'], required=True)
    args = parser.parse_args()
    print(json.dumps(build_assets(args.output, geometry_for(args.game)), indent=2))


if __name__ == '__main__':
    main()
