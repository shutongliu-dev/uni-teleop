"""Additional kitchen inventories and opt-in contextual dressing, CC0-1.0.

Sealed pantry contents are rigid packaged goods, with no fluid simulation. All
open dishes retain their physical cavities. Label bands, grip ribs, seams and
fine decorative rings are visual detail, with masses included in the declared
body mass. Fixed context is deliberately outside the manipulation workspace.
"""
from __future__ import annotations

import math
import numpy as np
from . import geometry as g


def _ring(radius, z, thickness=.0006):
    return g.lathe([(radius+thickness*math.cos(t), z+thickness*math.sin(t))
                    for t in np.linspace(0, 2*math.pi, 9)], 64)


def sealed_container(kind, *, radius=.027, height=.11, color='cream'):
    """Distinct closed package silhouettes; contents/caps are one rigid object."""
    if kind == 'bottle':
        profile = [(0, 0), (radius*.78, 0), (radius, .006),
                   (radius, height*.66), (radius*.85, height*.73),
                   (radius*.39, height*.82), (radius*.36, height*.94),
                   (radius*.42, height*.94), (radius*.42, height), (0, height)]
        cap_r, cap_z = radius*.43, height*.975
    elif kind == 'can':
        profile = [(0, 0), (radius*.93, 0), (radius, .002),
                   (radius, .005), (radius*.96, .006),
                   (radius*.96, height-.005), (radius, height-.004),
                   (radius, height-.001), (radius*.93, height), (0, height)]
        cap_r, cap_z = radius*.90, height-.001
    elif kind == 'grinder':
        profile = [(0, 0), (radius*.90, 0), (radius, .004),
                   (radius*.87, height*.22), (radius*.67, height*.48),
                   (radius*.80, height*.68), (radius, height*.79),
                   (radius*.94, height*.94), (radius*.65, height), (0, height)]
        cap_r, cap_z = radius*.95, height*.85
    else:
        profile = [(0, 0), (radius*.84, 0), (radius, .005),
                   (radius, height-.020), (radius*.86, height-.014),
                   (radius*.86, height-.010), (radius*.95, height-.010),
                   (radius*.95, height-.002), (radius*.87, height), (0, height)]
        cap_r, cap_z = radius*.96, height-.006
    mesh = g.lathe(profile, 64)
    details = [('base_seam', _ring(radius*.985, .005), 'steel' if kind == 'can' else color),
               ('cap_edge', _ring(cap_r, cap_z), 'steel' if kind == 'can' else 'rubber')]
    if kind == 'can':
        for i, z in enumerate((.016, height-.016)):
            details.append(('rolled_bead_'+str(i), _ring(radius*.97, z), 'steel'))
        # Pull tab and recessed top score are appearance, not a separate lid.
        tab = _ring(.007, height+.0004, .001)
        tab.apply_scale((1., 1.6, 1.))
        details += [('pull_tab', tab, 'steel'), ('top_score', _ring(radius*.78, height+.0002, .00035), 'steel')]
    elif kind != 'grinder':
        for i in range(16):
            t = i*math.tau/16
            details.append((f'cap_grip_{i}', g.tube([
                (cap_r*math.cos(t), cap_r*math.sin(t), cap_z-.003),
                (cap_r*math.cos(t), cap_r*math.sin(t), cap_z+.003)], .00055, 6), 'rubber'))
    # An inset-looking paper front label and a contrasting product mark.
    # Front is -Y, visible from the room and task cameras.
    label_z = height*.43
    details += [('paper_label', g.box((radius*1.35, .0007, height*.28),
                                     (0, -radius-.0003, label_z)), 'cream'),
                ('product_mark', g.box((radius*.65, .0008, height*.08),
                                      (0, -radius-.0008, label_z+.005)), color)]
    return [('shell', mesh, color, 'convexHull')], details


def _carton(width, depth, height, color):
    # Folded top shoulders and seal seam distinguish cartons from simple cubes.
    profile = [(0, 0, 0, 0), (width/2-.001, depth/2-.001, .003, 0),
               (width/2, depth/2, .004, .001),
               (width/2, depth/2, .004, height-.013),
               (width/2-.002, depth/2-.006, .003, height), (0, 0, 0, height)]
    mesh = g.rounded_loft(profile)
    details = [('folded_seal', g.box((width-.008, .003, .0014), (0, 0, height+.0003)), 'cream'),
               ('front_label', g.box((width*.76, .0006, height*.57), (0, -depth/2-.0002, height*.48)), 'cream'),
               ('product_band', g.box((width*.66, .0007, height*.11), (0, -depth/2-.0006, height*.58)), color)]
    for i in range(5):
        details.append((f'printed_grain_{i}', g.ellipsoid((.0023, .0005, .0047),
            ((i-2)*.006, -depth/2-.001, height*.35), subdivisions=1), color))
    return [('package', mesh, color, 'convexHull')], details


def _bin(a, name, description, xy, size, color, category):
    from .assets import _pose
    sx, sy = size
    a.body(name, 'group_destination', _pose(*xy, .9001),
           [('shell', g.tray((sx, sy, .028), wall=.005, floor=.005), color, 'none')], 0,
           category=category, label=description, public_description=description,
           floor_top_local_z_m=.005,
           # One millimetre inside the actual floor/wall transition.
           interior_bounds_xy_m=[[-sx/2+.006, -sy/2+.006], [sx/2-.006, sy/2-.006]])
    # Three short rim stripes are real markings, never invisible slot fixtures.
    for i in range(3):
        a.decoration(name+f'_rim_mark_{i}', g.box((.025, .002, .0002),
            (xy[0]+(i-1)*.035, xy[1]-sy/2+.0025, .9283)), 'brass')


def _public_goal(a, descriptions):
    a.meta['public_goal'] = {
        'objects': [{'object_id': item['id'], 'description': item['public_description'],
                     'category': item['category'], 'destination': item['assigned_fixture']}
                    for item in a.meta['objects']],
        'destinations': [{'destination_id': name, 'description': description, 'category': category}
                         for name, description, category in descriptions],
    }
    a.meta['required_object_names'] = list(a.meta['actors'])
    a.meta['placements'] = {item['id']: item['assigned_fixture'] for item in a.meta['objects']}
    a.meta['source'].update(contents_approximation='Sealed packages and empty tableware are rigid; no contents or liquid dynamics.',
                            source_revision='kitchen_six_inventory_v1')


def pantry_stocking(a, presentation):
    from .assets import _pose
    a.body('source_mat', 'pantry_delivery_mat', _pose(.25, -.86, .9001),
           [('mat', g.board((.42, .44, .004), bevel=.0008), 'rubber', 'none')], 0,
           floor_top_local_z_m=.004, interior_bounds_xy_m=[[-.204, -.214], [.204, .214]])
    destinations = [('bottles_bin', 'Deep blue open bin for sealed oil and vinegar bottles', 'bottles'),
                    ('seasonings_bin', 'Cream open bin for salt, pepper and paprika jars', 'seasonings'),
                    ('dry_goods_bin', 'Wooden open bin for rice, pasta and the tomato can', 'dry_goods')]
    _bin(a, *destinations[0][:2], (.62, -.88), (.23, .40), 'blue', 'bottles')
    _bin(a, *destinations[1][:2], (.87, -.75), (.23, .18), 'cream', 'seasonings')
    _bin(a, *destinations[2][:2], (.87, -.99), (.23, .26), 'wood', 'dry_goods')
    specs = [
        ('olive_oil', 'bottles', 'enamel', .255, 'upright', 'Green sealed olive-oil bottle with narrow neck and ribbed cap',
         sealed_container('bottle', radius=.028, height=.178, color='enamel'), (.12, -.98), (.62, -.98)),
        ('vinegar', 'bottles', 'toast', .230, 'upright', 'Amber sealed vinegar bottle with tall shoulders and ribbed cap',
         sealed_container('bottle', radius=.026, height=.161, color='toast'), (.25, -.98), (.62, -.78)),
        ('salt', 'seasonings', 'cream', .105, 'upright', 'Short cream salt jar with a sealed screw cap and paper band',
         sealed_container('jar', radius=.023, height=.076, color='cream'), (.38, -.98), (.795, -.75)),
        ('pepper', 'seasonings', 'rubber', .085, 'upright', 'Dark waisted pepper grinder, closed and rigid',
         sealed_container('grinder', radius=.022, height=.104, color='rubber'), (.12, -.84), (.870, -.75)),
        ('paprika', 'seasonings', 'tomato', .075, 'upright', 'Red paprika spice jar with shoulder, cap ribs and label',
         sealed_container('jar', radius=.022, height=.085, color='tomato'), (.25, -.84), (.945, -.75)),
        ('rice_carton', 'dry_goods', 'cream', .180, 'any', 'Cream rice carton with folded top seam and grain emblem',
         _carton(.058, .046, .112, 'cream'), (.38, -.84), (.810, -.935)),
        ('pasta_carton', 'dry_goods', 'blue', .140, 'any', 'Blue pasta carton with folded top seam and printed grain band',
         _carton(.056, .043, .128, 'blue'), (.12, -.70), (.930, -.935)),
        ('tomato_can', 'dry_goods', 'tomato', .220, 'any', 'Red tomato can with rolled steel rims, ring pull and paper label',
         sealed_container('can', radius=.029, height=.078, color='tomato'), (.25, -.70), (.870, -1.055)),
    ]
    goals, supports = {}, {}
    for name, category, color, mass, orientation, description, geometry, source_xy, target_xy in specs:
        parts, details = geometry
        target = _pose(*target_xy, .9054)
        fixture = {'bottles': 'bottles_bin', 'seasonings': 'seasonings_bin', 'dry_goods': 'dry_goods_bin'}[category]
        a.body(name, name, target if presentation else _pose(*source_xy, .9045), parts, mass,
               details=details, category=category, assigned_fixture=fixture,
               placement_orientation=orientation, label=name.replace('_', ' ').title(),
               public_description=description, package_state='sealed', material_description=color)
        goals[name], supports[name] = target, fixture
    a.meta['goals'] = {'body_poses_xyzw': goals, 'support_by_actor': supports,
                       'class_to_fixture': {category: name for name, _, category in destinations}}
    a.meta['source_support_by_actor'] = {name: 'source_mat' for name in a.meta['actors']}
    _public_goal(a, destinations)


def _bowl(radius, height):
    return g.lathe([(0, 0), (radius*.48, 0), (radius*.55, .003),
                    (radius*.78, height*.42), (radius, height-.004),
                    (radius, height-.001), (radius*.97, height),
                    (radius*.91, height-.001), (radius*.90, height-.006),
                    (radius*.70, height*.43), (radius*.46, .005), (0, .005)], 96)


def table_clearing(a, presentation):
    from .assets import _pose, _plate, _cup, _utensil
    a.body('source_mat', 'used_tableware_mat', _pose(.245, -.875, .9001),
           [('mat', g.board((.48, .49, .004), bevel=.0008), 'blue', 'none')], 0,
           floor_top_local_z_m=.004, interior_bounds_xy_m=[[-.234, -.239], [.234, .239]])
    destinations = [('dishes_tray', 'Large cream open tray for plates and bowls; safe stacking is allowed', 'dishware'),
                    ('drinkware_tray', 'Blue open tray for the two empty handled cups', 'drinkware'),
                    ('cutlery_tray', 'Wooden open tray for the spoon and fork, with free yaw', 'cutlery')]
    _bin(a, *destinations[0][:2], (.65, -.895), (.30, .43), 'cream', 'dishware')
    _bin(a, *destinations[1][:2], (.925, -.75), (.22, .20), 'blue', 'drinkware')
    _bin(a, *destinations[2][:2], (.925, -.985), (.22, .25), 'wood', 'cutlery')
    dinner = _plate(); dinner.apply_scale((1.075, 1.075, 1.))
    side = _plate(); side.apply_scale((.85, .85, 1.))
    tea_parts = []
    # A low flared teacup has a genuine open handle and cavity, unlike the mug.
    tea_shell = g.lathe([(0, 0), (.022, 0), (.025, .003), (.031, .038),
                        (.035, .055), (.034, .058), (.031, .058),
                        (.028, .040), (.022, .005), (0, .005)], 96)
    tea_handle = g.tube([(0., .032+.024*math.cos(t), .032+.018*math.sin(t))
                         for t in np.linspace(-math.pi/2, math.pi/2, 33)], .0035, 12)
    tea_parts = [('shell', tea_shell, 'blue', 'sdf'), ('handle', tea_handle, 'blue', 'sdf')]
    mug_parts = _cup()
    specs = [
        ('dinner_plate', 'dishware', .145, 'upright', 'Wide cream dinner plate with raised rim and an open eating surface',
         [('shell', dinner, 'cream', 'sdf')], [], (.095, -1.000), (.650, -1.000), .9054, 'dishes_tray'),
        ('side_plate', 'dishware', .100, 'upright', 'Smaller blue side plate; may sit in the dinner plate or directly in the dish tray',
         [('shell', side, 'blue', 'sdf')], [], (.265, -1.000), (.650, -1.000), .9107, 'dinner_plate'),
        ('cereal_bowl', 'dishware', .125, 'upright', 'Deep green cereal bowl with an open cavity and flat supporting foot',
         [('shell', _bowl(.055, .050), 'enamel', 'sdf')], [], (.095, -.780), (.570, -.780), .9054, 'dishes_tray'),
        ('small_bowl', 'dishware', .085, 'upright', 'Small cream condiment bowl with a shallower open cavity',
         [('shell', _bowl(.046, .034), 'cream', 'sdf')], [], (.240, -.800), (.720, -.780), .9054, 'dishes_tray'),
        ('mug', 'drinkware', .095, 'upright', 'Tall ceramic mug with a true open handle; empty and upright',
         mug_parts, [], (.380, -.720), (.870, -.770), .9054, 'drinkware_tray'),
        ('tea_cup', 'drinkware', .080, 'upright', 'Low blue flared teacup with a separate curved open handle; empty',
         tea_parts, [], (.400, -.855), (.980, -.770), .9054, 'drinkware_tray'),
    ]
    for name, x in (('spoon', .370), ('fork', .435)):
        mesh, details = _utensil(name)
        specs.append((name, 'cutlery', .032 if name == 'spoon' else .030, 'any',
                      'Steel spoon with a concave bowl and engraved handle' if name == 'spoon' else
                      'Steel four-tine fork with open tine gaps and engraved handle',
                      [('utensil', mesh, 'steel', 'sdf')], details,
                      (x, -1.000), (.885 if name == 'spoon' else .965, -.985), .9054, 'cutlery_tray'))
    goals, supports = {}, {}
    for name, category, mass, orientation, description, parts, details, source_xy, target_xy, target_z, support in specs:
        fixture = {'dishware': 'dishes_tray', 'drinkware': 'drinkware_tray', 'cutlery': 'cutlery_tray'}[category]
        target = _pose(*target_xy, target_z)
        a.body(name, name, target if presentation else _pose(*source_xy, .9045), parts, mass,
               details=details, category=category, assigned_fixture=fixture,
               placement_orientation=orientation, label=name.replace('_', ' ').title(),
               public_description=description, floor_top_local_z_m=.005 if category != 'cutlery' else 0.,
               interior_radius_m={'dinner_plate': .059*1.075, 'side_plate': .059*.85,
                                  'cereal_bowl': .055*.90, 'small_bowl': .046*.90,
                                  'mug': .0275, 'tea_cup': .031}.get(name, 0.))
        goals[name], supports[name] = target, support
    a.meta['goals'] = {'body_poses_xyzw': goals, 'support_by_actor': supports,
                       'class_to_fixture': {category: name for name, _, category in destinations},
                       'witness_note': 'The side plate rests inside the dinner plate; this is an optional real stack, not a required slot layout.'}
    a.meta['source_support_by_actor'] = {name: 'source_mat' for name in a.meta['actors']}
    _public_goal(a, destinations)


def enrich_context(a):
    """Twelve fixed everyday supplies on the free outer edge of the bar.

    They are physical fixed scenery, never reset/scoring actors or destination
    fixtures. Every contact mesh remains outside y >= -1.20 m by > 20 mm.
    The unchanged room's existing scanned/detailed kitchen inventory remains.
    """
    from .assets import _pose
    specs = [
        ('jam_jar', 'preserves', 'Blue preserve jar with a ribbed lid and paper label', 'jar', .028, .074, 'blue'),
        ('honey_jar', 'preserves', 'Amber honey jar with rolled base, shoulder and sealed lid', 'jar', .027, .090, 'bread'),
        ('small_oil_cruet', 'condiment', 'Green oil cruet with narrow neck and screw cap', 'bottle', .025, .145, 'enamel'),
        ('salt_mill', 'seasoning', 'Light waisted salt mill', 'grinder', .023, .110, 'cream'),
        ('pepper_mill', 'seasoning', 'Dark waisted pepper mill', 'grinder', .023, .122, 'rubber'),
        ('sugar_jar', 'dry_storage', 'Ceramic sugar storage jar with broad screw lid', 'jar', .031, .102, 'ceramic'),
        ('tea_tin', 'dry_storage', 'Blue tea tin with rolled seams and metal lid', 'can', .030, .090, 'blue'),
        ('coffee_tin', 'dry_storage', 'Brown coffee tin with steel rim and paper label', 'can', .031, .112, 'toast'),
    ]
    for i, (name, category, description, kind, radius, height, color) in enumerate(specs):
        parts, details = sealed_container(kind, radius=radius, height=height, color=color)
        a.body('context_'+name, name, _pose(.025+i*.090, -1.275, .9001), parts, 0,
               details=details, static_context=True, category=category, label=name.replace('_', ' ').title(),
               public_description=description, declared_support='bar outer strip', support_path='/World/Geometry/worktop/Collider')
    parts, details = _carton(.060, .055, .092, 'cream')
    a.body('context_flour_packet', 'flour_packet', _pose(.755, -1.275, .9001), parts, 0,
           details=details, static_context=True, category='dry_storage', label='Flour packet',
           public_description='Small closed flour carton with folded top and printed grain label',
           declared_support='bar outer strip', support_path='/World/Geometry/worktop/Collider')
    parts, details = _carton(.058, .052, .070, 'blue')
    a.body('context_tea_box', 'tea_box', _pose(.845, -1.275, .9001), parts, 0,
           details=details, static_context=True, category='dry_storage', label='Tea box',
           public_description='Blue tea box with folded seal and paper label',
           declared_support='bar outer strip', support_path='/World/Geometry/worktop/Collider')
    # Folded linen is a fixed rigid approximation, with geometric hems and seams.
    linen = g.board((.070, .060, .012), bevel=.001)
    details = [('hem_'+str(i), g.box((.061, .001, .0005), (0, side*.025, .0122)), 'cream') for i, side in enumerate((-1, 1))]
    for i in range(4):
        details.append(('fold_'+str(i), g.box((.069, .0004, .0007), (0, -.0299, .002+i*.0025)), 'ceramic'))
    a.body('context_folded_napkin', 'folded_napkin', _pose(.940, -1.275, .9001),
           [('linen', linen, 'ceramic', 'none')], 0, details=details, static_context=True,
           category='table_linen', label='Folded napkin',
           public_description='Folded sage cloth napkin with layered edges and stitched hems; fixed rigid scenery',
           declared_support='bar outer strip', support_path='/World/Geometry/worktop/Collider')
    eggcup = g.lathe([(0, 0), (.021, 0), (.022, .003), (.014, .007), (.011, .021),
                     (.024, .044), (.025, .049), (.023, .051), (.020, .048), (.011, .029), (0, .026)], 64)
    egg = g.ellipsoid((.018, .018, .026), (0, 0, .058))
    a.body('context_egg_cup', 'egg_cup', _pose(1.035, -1.275, .9001),
           [('cup', eggcup, 'cream', 'none'), ('egg', egg, 'cream', 'none')], 0,
           static_context=True, category='breakfast_accessory', label='Egg cup',
           public_description='Footed ceramic egg cup with an egg seated in its open bowl; fixed scenery',
           declared_support='bar outer strip', support_path='/World/Geometry/worktop/Collider')
    a.meta['dressing_profile'] = 'enriched_v1'
    a.meta['context_policy'] = {'scoring': False, 'required': False, 'dynamic': False,
        'region_xy_m': [[-.01, -1.32], [1.07, -1.23]],
        'protected_bar_workspace_xy_m': [[-.10, -1.20], [1.10, -.60]],
        'scope': 'Fixed everyday supplies supplement the pinned furnished room; task payloads and destinations remain separately identified.'}
    if 'public_goal' in a.meta:
        a.meta['public_goal']['context_objects'] = [
            {'object_id': name, 'description': item['public_description']}
            for name, item in a.meta['contextual_prop_metadata'].items()]


def enrich_public_goal(a):
    """Add inspectable descriptions without altering legacy default metadata."""
    if 'public_goal' in a.meta:
        return
    descriptions = {
        'breakfast_serving': {
            'tray': ('serving_equipment', 'destination_zone', 'Wooden serving tray with open cavity and brass carry handle; load at source before carrying'),
            'plate': ('tableware', 'tray', 'Cream plate with shallow open eating surface and raised rim'),
            'cup': ('drinkware', 'tray', 'Empty ceramic cup with open cavity and curved finger handle'),
            'bread_0': ('bread', 'plate', 'First small bread roll with sesame seeds, scored crust and flat supporting underside'),
            'bread_1': ('bread', 'plate', 'Second small bread roll with a different seeded crust pattern'),
        },
        'produce_transfer': {name: (category, category+'_tray',
            ('Red ribbed tomato with green calyx and small stem' if category == 'tomato' else
             'Short tapered carrot with root rings and stem, naturally resting on a real hull facet'))
            for category in ('tomato', 'carrot') for name in (f'{category}_{i}' for i in range(3))},
        'utensil_sorting': {f'{kind}_{i}': (kind, 'cutlery_tray/'+kind,
            {'spoon': 'Steel spoon with a concave bowl and engraved handle',
             'fork': 'Steel four-tine fork with real open tine gaps and engraved handle',
             'knife': 'Blunt table knife with rounded blade and engraved handle'}[kind])
            for kind in ('spoon', 'fork', 'knife') for i in range(2)},
        'pot_lid_placement': {
            'pot': ('cookware', 'cold_burner', 'Green enamel pot with a physically open cavity, rolled rim and two open metal ear handles'),
            'lid': ('cookware', 'pot', 'Separate domed steel lid with locating lip and graspable black knob'),
        },
    }[a.task_id]
    targets = {
        'breakfast_serving': [('destination_zone', 'Blue marked serving zone; move the actually loaded tray here', 'serving'),
                              ('tray', 'Interior of the movable serving tray', 'tableware'),
                              ('plate', 'Eating surface of the movable plate', 'bread')],
        'produce_transfer': [('tomato_tray', 'Cream open tray marked with a tomato pictogram', 'tomato'),
                             ('carrot_tray', 'Blue open tray marked with a carrot pictogram', 'carrot')],
        'utensil_sorting': [('cutlery_tray/'+kind, 'Wooden tray channel marked with a '+kind+' silhouette', kind)
                            for kind in ('spoon', 'fork', 'knife')],
        'pot_lid_placement': [('cold_burner', 'Front-left cold cooktop burner; seat and release the pot first', 'cookware'),
                              ('pot', 'Actual rolled rim of the separately supported pot', 'cookware')],
    }[a.task_id]
    objects = []
    for item in a.meta['objects']:
        category, destination, description = descriptions[item['id']]
        item.update(category=category, public_description=description, label=item['id'].replace('_', ' ').title())
        objects.append({'object_id': item['id'], 'description': description,
                        'category': category, 'destination': destination})
    a.meta['public_goal'] = {'objects': objects,
        'destinations': [{'destination_id': name, 'description': description, 'category': category}
                         for name, description, category in targets],
        'context_objects': [{'object_id': name, 'description': item['public_description']}
                            for name, item in a.meta['contextual_prop_metadata'].items()]}
