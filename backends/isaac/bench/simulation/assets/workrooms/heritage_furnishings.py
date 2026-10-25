"""Grouped, real-geometry furnishings for the fixed scholar's game room.

These are static perimeter decorations, never task objects. Books, rolled
scrolls, a framed game study and foliage are modeled; no new image is needed.
"""
from __future__ import annotations

import math

import numpy as np
from pxr import Usd, UsdGeom

from .common import light


def _book(b, name, center, size, cover):
    x, y, z = center
    dx, dy, dz = size
    b.box(name + '_pages', center, (dx - .008, dy - .010, dz - .006),
          'heritage_paper', .001, collision=False)
    for i, zz in enumerate((z - dz / 2 + .0015, z + dz / 2 - .0015)):
        b.box(name + '_cover_' + str(i), (x, y, zz), (dx, dy, .003),
              cover, .001, collision=False)
    b.box(name + '_spine', (x, y + dy / 2 - .004, z), (dx, .008, dz),
          cover, .001, collision=False)
    # Four visible thread stations bind a substantial paper block.
    for i in range(4):
        xx = x - dx * .34 + i * dx * .225
        b.tube(name + '_stitch_' + str(i), [(xx, y + dy / 2 - .025, z + dz / 2 + .001),
               (xx, y + dy / 2 + .002, z + dz / 2 + .001),
               (xx, y + dy / 2 + .002, z - dz / 2)], .0011, 'heritage_linen')
    b.box(name + '_title_slip', (x + dx * .23, y, z + dz / 2 + .0004),
          (dx * .18, dy * .62, .0007), 'heritage_paper', .0001, collision=False)


def _go_bowl(b, name, center, material):
    x, y, z = center
    b.lathe(name + '_body', center, [(0, 0), (.052, 0), (.077, .014),
            (.094, .048), (.090, .087), (.073, .103), (0, .103)], material)
    b.lathe(name + '_lid', (x, y, z + .103), [(0, 0), (.078, 0),
            (.080, .009), (.066, .023), (.035, .030), (0, .031)], material)
    b.torus(name + '_lid_rim', (x, y, z + .109), .078, .0025, 'heritage_table')


def _game_library(b):
    """A low asymmetric display cabinet faces into the visible left wall."""
    x, y = -1.49, -1.17
    for i, xx in enumerate((x - .15, x + .15)):
        for j, yy in enumerate((y - .50, y + .50)):
            b.box(f'game_library_foot_{i}_{j}', (xx, yy, .075),
                  (.045, .045, .15), 'heritage_walnut', .003)
    for i, yy in enumerate((y - .535, y + .535)):
        b.box('game_library_side_' + str(i), (x, yy, .90),
              (.36, .035, 1.50), 'heritage_walnut', .003)
    for i, zz in enumerate((.165, .65, 1.13, 1.65)):
        b.box('game_library_shelf_' + str(i), (x, y, zz),
              (.39, 1.12, .030), 'heritage_walnut', .004)
    # The short staggered partitions make several generous display bays.
    for i, (yy, zz, height) in enumerate(((y + .04, .89, .45), (y - .16, 1.39, .49))):
        b.box('game_library_partition_' + str(i), (x, yy, zz),
              (.34, .026, height), 'heritage_walnut', .002)
    b.box('game_library_lower_back', (x - .174, y, .405),
          (.018, 1.04, .45), 'heritage_panel', .001, collision=False)
    for i, yy in enumerate((y - .27, y + .27)):
        b.box('game_library_lower_door_' + str(i), (x + .174, yy, .405),
              (.022, .51, .43), 'heritage_walnut', .003)
        b.box('game_library_door_inset_' + str(i), (x + .187, yy, .405),
              (.006, .43, .34), 'heritage_table', .002, collision=False)
        b.torus('game_library_door_pull_' + str(i), (x + .205, yy + (.17 if i == 0 else -.17), .43),
                .016, .0028, 'heritage_brass', rotation=(0, 90, 0))
    for i in range(3):
        _book(b, 'game_library_manual_' + str(i), (x + .012, y - .245, .665 + .018 + .037 * i),
              (.275, .30, .036), ('heritage_ink', 'heritage_silk', 'heritage_cinnabar')[i])
    for i, yy in enumerate((y + .19, y + .40)):
        _go_bowl(b, 'game_library_go_bowl_' + str(i), (x + .015, yy, .665),
                 'heritage_walnut' if i == 0 else 'heritage_paper')
    # A fitted cradle supports three scrolls; their roller ends are visible.
    for i, xx in enumerate((x - .10, x + .10)):
        b.box('game_library_scroll_cradle_' + str(i), (xx, y + .22, 1.162),
              (.025, .36, .034), 'heritage_table', .003, collision=False)
    for i, yy in enumerate((y + .10, y + .22, y + .34)):
        zz = 1.207
        b.cylinder('game_library_scroll_paper_' + str(i), (x, yy, zz), .028, .275,
                   'heritage_paper', rotation=(0, 90, 0), collision=False)
        for j, xx in enumerate((x - .151, x + .151)):
            b.cylinder(f'game_library_scroll_roller_{i}_{j}', (xx, yy, zz), .013, .028,
                       'heritage_walnut', rotation=(0, 90, 0), collision=False)
        b.cylinder('game_library_scroll_band_' + str(i), (x, yy, zz), .0288, .018,
                   'heritage_silk', rotation=(0, 90, 0), collision=False)
    b.lathe('game_library_vase', (x, y - .34, 1.145), [(0, 0), (.042, 0),
            (.076, .07), (.062, .15), (.027, .24), (.023, .24),
            (.024, .19), (.050, .13), (.055, .065), (.032, .008), (0, .008)], 'heritage_celadon')


def _floor_lantern(b):
    x, y = -1.38, 2.08
    b.lathe('floor_lantern_base', (x, y, 0), [(0, 0), (.18, 0),
            (.18, .035), (.15, .065), (.09, .09), (.065, .11), (0, .11)],
            'heritage_walnut', collision=True)
    b.cylinder('floor_lantern_stem', (x, y, .62), .025, 1.02, 'heritage_walnut')
    for i, zz in enumerate((1.12, 1.68)):
        b.box('floor_lantern_cap_' + str(i), (x, y, zz), (.34, .34, .040), 'heritage_walnut', .006)
    for i, xx in enumerate((x - .145, x + .145)):
        for j, yy in enumerate((y - .145, y + .145)):
            b.box(f'floor_lantern_rib_{i}_{j}', (xx, yy, 1.40),
                  (.016, .016, .52), 'heritage_walnut', .002, collision=False)
    for axis in (0, 1):
        for sign in (-1, 1):
            center = [x, y, 1.40]
            center[axis] += sign * .144
            size = [.280, .280, .49]
            size[axis] = .002
            b.box(f'floor_lantern_silk_{axis}_{sign}', center, size,
                  'heritage_lantern_silk', .0004, collision=False)
            center[axis] += sign * .004
            size = [.010, .010, .49]
            b.box(f'floor_lantern_mullion_{axis}_{sign}', center, size,
                  'heritage_walnut', .001, collision=False)
    light(b, 'floor_lantern_glow', (x, y, 1.094), (x, y, .15),
          (.24, .24), 2200., (1., 1., 1.), shape='disk', role='accent', temperature=3000.)


def _game_record(b):
    """A framed nine-line game study fills the blank wall left of the window."""
    x, y, z = -.77, 2.452, 1.92
    b.box('game_record_back', (x, y, z), (.65, .025, .75), 'heritage_walnut', .004, collision=False)
    b.box('game_record_paper', (x, y - .014, z), (.58, .002, .68), 'heritage_paper', .0004, collision=False)
    for i, xx in enumerate((x - .312, x + .312)):
        b.box('game_record_frame_side_' + str(i), (xx, y - .024, z),
              (.026, .032, .75), 'heritage_walnut', .003, collision=False)
    for i, zz in enumerate((z - .362, z + .362)):
        b.box('game_record_frame_rail_' + str(i), (x, y - .024, zz),
              (.60, .032, .026), 'heritage_walnut', .003, collision=False)
    for i in range(9):
        offset = (i - 4) * .053
        b.box('game_record_grid_u_' + str(i), (x + offset, y - .016, z),
              (.0018, .0007, .424), 'heritage_ink', .0001, collision=False)
        b.box('game_record_grid_v_' + str(i), (x, y - .016, z + offset),
              (.424, .0007, .0018), 'heritage_ink', .0001, collision=False)
    for i, (column, row) in enumerate(((-2, -1), (-1, -1), (-2, 0), (-1, 1), (0, 0), (1, 0), (0, 2), (2, 1), (1, 2))):
        material = 'heritage_ink' if i % 2 == 0 else 'heritage_linen'
        # Shallow discs read as an inlaid study, not playable task pieces.
        b.cylinder('game_record_mark_' + str(i), (x + column * .053, y - .019, z + row * .053),
                   .018, .002, material, rotation=(90, 0, 0), collision=False)
    b.box('game_record_seal', (x + .245, y - .017, z - .285),
          (.034, .001, .034), 'heritage_cinnabar', .0003, collision=False)
    # Dark upper cartouche is part of the frame, with no tiny unreadable text.
    b.box('game_record_heading', (x, y - .017, z + .285),
          (.17, .001, .018), 'heritage_silk', .0003, collision=False)


def _garden_screen(b):
    # Back corner only: the folding screen is behind the chair and plant stand.
    for panel, (cx, cy, angle) in enumerate(((2.055, 2.30, -10.), (2.605, 2.30, 10.))):
        radians = math.radians(angle)

        def part(suffix, u, v, z, size, material='heritage_walnut', collision=False):
            point = (cx + u * math.cos(radians) - v * math.sin(radians),
                     cy + u * math.sin(radians) + v * math.cos(radians), z)
            return b.box(f'garden_screen_{panel}_{suffix}', point, size, material, .002,
                         rotation=(0, 0, angle), collision=collision)

        for i, u in enumerate((-.26, .26)):
            part('post_' + str(i), u, 0, .97, (.034, .045, 1.94), collision=True)
            part('foot_' + str(i), u, 0, .028, (.080, .22, .056), collision=True)
        for i, zz in enumerate((.17, .68, 1.86)):
            part('rail_' + str(i), 0, 0, zz, (.52, .040, .034), collision=True)
        part('lower_panel', 0, .006, .425, (.48, .020, .47), 'heritage_silk', collision=True)
        # Open stepped lattice, fewer than ten members per panel.
        for i, u in enumerate((-.17, 0., .17)):
            part('lattice_' + str(i), u, 0, 1.27, (.012, .022, 1.14))
        for i, zz in enumerate((.93, 1.61)):
            part('cross_' + str(i), 0, 0, zz, (.48, .022, .012))
        part('inner_top', 0, -.002, 1.75, (.24, .025, .012))
        for i, u in enumerate((-.12, .12)):
            part('step_' + str(i), u, -.002, 1.68, (.012, .025, .14))


def _bonsai(b):
    x, y = 2.39, 1.99
    b.box('bonsai_stand_top', (x, y, .73), (.46, .41, .035), 'heritage_walnut', .005)
    for i, xx in enumerate((x - .17, x + .17)):
        for j, yy in enumerate((y - .15, y + .15)):
            b.box(f'bonsai_stand_leg_{i}_{j}', (xx, yy, .35625),
                  (.035, .035, .7125), 'heritage_walnut', .003)
    for i, yy in enumerate((y - .15, y + .15)):
        b.box('bonsai_stand_apron_' + str(i), (x, yy, .68), (.36, .025, .075), 'heritage_walnut', .002)
    b.lathe('bonsai_planter', (x, y, .7475), [(0, 0), (.14, 0), (.175, .09),
            (.18, .11), (.164, .11), (.137, .025), (0, .025)], 'heritage_celadon')
    b.cylinder('bonsai_soil', (x, y, .835), .159, .018, 'heritage_moss', collision=False)
    trunk = [(x, y, .84), (x - .055, y + .01, .96), (x - .035, y, 1.11),
             (x + .055, y, 1.25), (x + .07, y + .015, 1.42)]
    for i in range(len(trunk) - 1):
        b.tube('bonsai_trunk_' + str(i), trunk[i:i + 2], .022 - i * .004, 'heritage_walnut')
    clusters = [(-.24, -.045, 1.15, .18), (.20, .005, 1.30, .19),
                (-.12, .055, 1.43, .18), (.07, .018, 1.55, .16)]
    rng = np.random.default_rng(1729)
    for i, (dx, dy, zz, radius) in enumerate(clusters):
        start = trunk[min(i + 1, 4)]
        tip = (x + dx, y + dy, zz - .026)
        b.tube('bonsai_branch_' + str(i), [start, (x + dx * .65, y + dy * .6, zz - .075), tip],
               .009 - i * .0013, 'heritage_walnut')
        # Faceted leaf sprays form broad, irregular pine pads, not green balls.
        vertices, faces = [], []
        for leaf in range(62):
            theta = rng.uniform(0., math.tau)
            reach = radius * math.sqrt(rng.uniform(0., 1.))
            center = np.array((tip[0] + reach * math.cos(theta),
                               tip[1] + reach * .64 * math.sin(theta), zz + rng.uniform(-.022, .032)))
            direction = rng.uniform(0., math.tau)
            half = np.array((math.cos(direction), math.sin(direction), rng.uniform(-.25, .4))) * rng.uniform(.018, .033)
            width = np.array((-math.sin(direction), math.cos(direction), .12)) * .006
            first = len(vertices)
            vertices.extend((center - half, center + width, center + half, center - width))
            faces.extend(((first, first + 1, first + 2), (first, first + 2, first + 3)))
        b.mesh(b.path('bonsai_foliage_' + str(i)), vertices, faces, 'heritage_leaf')


def build_furnishings(b):
    """Build five coherent perimeter groups and describe their actual bounds."""
    groups = []
    for name, recipe, purpose in (
            ('game_library', _game_library, 'display cabinet with bound game manuals, closed Go bowls and rolled scrolls'),
            ('floor_lantern', _floor_lantern, 'wood and silk reading lamp in the rear left corner'),
            ('game_record', _game_record, 'framed game study on the back wall'),
            ('garden_screen', _garden_screen, 'two open lattice panels behind the tea chair'),
            ('bonsai', _bonsai, 'pine bonsai on a tall display stand')):
        before = set(b._names)
        recipe(b)
        paths = ['/World/Geometry/' + name for name in sorted(set(b._names) - before)]
        cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ['default', 'render'])
        ranges = [cache.ComputeWorldBound(b.stage.GetPrimAtPath(path)).ComputeAlignedRange() for path in paths]
        groups.append({'name': name, 'purpose': purpose, 'geometry_paths': paths,
                       'bounds_m': [np.min([list(box.GetMin()) for box in ranges], axis=0).tolist(),
                                    np.max([list(box.GetMax()) for box in ranges], axis=0).tolist()]})
    return {'groups': groups, 'static_perimeter_only': True,
            'scope': 'Authored geometry and CPU bounds; visibility requires the native room and overview images'}
