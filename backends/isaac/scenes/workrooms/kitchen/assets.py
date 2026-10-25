"""Detailed original kitchen task assets; task-owned passive rigid bodies only.

All dimensions are metres and all local origins are at the nominal support
plane. These recipes author candidates; only a frozen native report can qualify
them. The presentation option changes reset poses, never physical behavior.
"""
from __future__ import annotations

import math
import numpy as np
import trimesh
from pxr import Gf, Sdf, UsdGeom, UsdPhysics, UsdShade

from . import geometry as g
from bench.paths import DATA_ROOT

SCHEMA = 'kitchen_task_assets.v1'
ASSET_ROOT = f'{DATA_ROOT}/assets/task_parts/kitchen'
TASK_IDS = ('breakfast_serving', 'produce_transfer', 'utensil_sorting', 'pot_lid_placement',
            'pantry_stocking', 'table_clearing')
SURFACE_Z = .90
BAR_SUPPORT = '/World/Geometry/worktop/Collider'
PREP_SUPPORT = '/World/Geometry/kitchen_counter_middle/Collider'
COOKTOP_SUPPORT = '/World/Geometry/kitchen_cooktop/Collider'
PHYSICS_PRIOR = {'static_friction': .4, 'dynamic_friction': .35, 'restitution': .02,
    'friction_combine': 'average', 'restitution_combine': 'average',
    'contact_offset_m': .0002, 'rest_offset_m': 0.,
    'source': 'Declared synthetic dry kitchen contact prior; not measured and not increased for stability.',
    'mass_source': 'Declared plausible lightweight household mass priors, not density measurements.',
    'inertia': 'Computed from the closed physical geometry using declared total mass.',
    'collision': 'One SDF shell for each open vessel; handles preserve openings. No convex hull seals a cavity.'}

PALETTE = {
    'wood': ((.46, .26, .12), .46, 0.), 'ceramic': ((.80, .88, .83), .24, 0.),
    'cream': ((.91, .87, .73), .36, 0.), 'steel': ((.48, .52, .55), .24, .85),
    'brass': ((.57, .37, .12), .27, .75), 'enamel': ((.12, .32, .29), .30, .1),
    'rubber': ((.10, .12, .13), .65, 0.), 'tomato': ((.65, .045, .022), .30, 0.),
    'carrot': ((.88, .23, .025), .47, 0.), 'leaf': ((.11, .25, .038), .58, 0.),
    'bread': ((.66, .31, .085), .80, 0.), 'crumb': ((.91, .68, .33), .86, 0.),
    'toast': ((.38, .14, .025), .75, 0.), 'blue': ((.15, .34, .48), .48, 0.),
}


def _pose(x, y, z, yaw=0.):
    return [float(x), float(y), float(z), 0., 0., math.sin(math.radians(yaw)/2), math.cos(math.radians(yaw)/2)]


def _world_child(parent, xyz):
    angle = 2*math.atan2(parent[5], parent[6])
    x, y, z = xyz
    return _pose(parent[0]+math.cos(angle)*x-math.sin(angle)*y,
                 parent[1]+math.sin(angle)*x+math.cos(angle)*y, parent[2]+z,
                 math.degrees(angle))


class _Author:
    def __init__(self, stage, task_id, root, presentation, texture_assets=None):
        self.stage, self.root, self.task_id = stage, root, task_id
        self.texture_assets = dict(texture_assets or {})
        UsdGeom.Xform.Define(stage, root)
        self.materials = {name: g.material(stage, root+'/Materials/'+name, *values,
            vertex_color=name in ('bread', 'tomato', 'carrot'),
            albedo_texture=self.texture_assets.get('wood_albedo') if name == 'wood' else None)
            for name, values in PALETTE.items()}
        self.physical_material = UsdShade.Material.Define(stage, root+'/Materials/Contact')
        physical = UsdPhysics.MaterialAPI.Apply(self.physical_material.GetPrim())
        physical.CreateStaticFrictionAttr(PHYSICS_PRIOR['static_friction'])
        physical.CreateDynamicFrictionAttr(PHYSICS_PRIOR['dynamic_friction'])
        physical.CreateRestitutionAttr(PHYSICS_PRIOR['restitution'])
        g.schema(self.physical_material.GetPrim(), 'PhysxMaterialAPI')
        g.attribute(self.physical_material.GetPrim(), 'physxMaterial:frictionCombineMode', Sdf.ValueTypeNames.Token, 'average')
        g.attribute(self.physical_material.GetPrim(), 'physxMaterial:restitutionCombineMode', Sdf.ValueTypeNames.Token, 'average')
        self.meta = {'schema': SCHEMA, 'task_id': task_id, 'root_path': root,
            'source': {'kind': 'original_procedural', 'license': 'CC0-1.0',
                'authoring_module': __name__, 'external_assets': [],
                'food_approximation': 'Food is rigid and non-deforming; no heat, fluid, cutting or fracture.'},
            'actors': {}, 'fixtures': {}, 'objects': [], 'fixture_metadata': {}, 'goals': {},
            'surface_z_m': SURFACE_Z, 'workspace_bounds_m': [[-.10, -1.20, .90], [1.10, -.60, 1.40]],
            'recovery_workspace_bounds_m': [[-.95, -1.35, .60], [1.85, -.55, 1.80]],
            'work_surface_paths': [BAR_SUPPORT], 'physics_prior': dict(PHYSICS_PRIOR),
            'presentation': bool(presentation), 'physics_accepted': False,
            'geometry_units': 'metres', 'pose_convention': 'position xyz then quaternion xyzw',
            'qualification': 'Authored physical candidate; no native acceptance or manipulation success implied.'}

    def body(self, name, role, pose, parts, mass, *, details=(), static_context=False, **extra):
        if static_context and mass:
            raise ValueError('Context objects must be fixed, non-scoring scenery')
        path = self.root+('/Context/' if static_context else '/Actors/' if mass else '/Fixtures/')+name
        body = UsdGeom.Xform.Define(self.stage, path)
        body.AddTranslateOp().Set(Gf.Vec3d(*pose[:3]))
        body.AddOrientOp().Set(Gf.Quatf(pose[6], Gf.Vec3f(*pose[3:6])))
        collision_geometry, colliders, all_geometry = [], [], []
        for label, mesh, material, approximation in parts:
            colors = None
            if material in ('bread', 'tomato', 'carrot'):
                xyz = np.asarray(mesh.vertices)
                variation = .035*np.sin(xyz[:, 0]*117+xyz[:, 1]*205)*np.sin(xyz[:, 2]*240)
                colors = np.clip(np.asarray(PALETTE[material][0])+variation[:, None], .005, .95)
            texcoords = g.wood_texcoords(mesh) if material == 'wood' else None
            g.write_mesh(self.stage, path+'/Visual/'+label, mesh, self.materials[material],
                         colors=colors, texcoords=texcoords)
            collider_path = path+'/Collision/'+label
            collider = g.write_mesh(self.stage, collider_path, mesh, self.materials[material],
                                    collision=approximation if mass else 'none')
            UsdGeom.Imageable(collider).CreateVisibilityAttr('invisible')
            UsdShade.MaterialBindingAPI(collider).Bind(self.physical_material, materialPurpose='physics')
            collision_geometry.append(mesh)
            colliders.append(collider_path)
            all_geometry.append(mesh)
        for label, mesh, material in details:
            g.write_mesh(self.stage, path+'/Visual/'+label, mesh, self.materials[material])
            all_geometry.append(mesh)
        bounds = trimesh.util.concatenate(all_geometry).bounds.tolist()
        physical_solid = trimesh.util.concatenate(collision_geometry)
        contact_bounds = physical_solid.bounds.tolist()
        entry = {'id': name, 'role': role, 'body_path': path, 'path': path,
            'mass_kg': float(mass), 'mass': float(mass), 'local_bounds_m': bounds,
            'collision_bounds_m': contact_bounds, 'initial_pose_xyzw': pose,
            # Exact extremal vertices avoid the nonexistent corners of a
            # rotated AABB when judging support height and containment.
            'collision_support_points_m': trimesh.convex.convex_hull(physical_solid.vertices).vertices.tolist(),
            'collider_paths': colliders, 'collision_count': len(colliders),
            'support_shape': {'type': 'closed_contact_mesh', 'origin_at_bottom': True,
                              'bottom_local_z_m': contact_bounds[0][2]}, **extra}
        if mass:
            UsdPhysics.RigidBodyAPI.Apply(body.GetPrim()).CreateRigidBodyEnabledAttr(True)
            UsdPhysics.RigidBodyAPI(body.GetPrim()).CreateKinematicEnabledAttr(False)
            props = g.mass_properties(collision_geometry, mass)
            entry.update(props)
            mass_api = UsdPhysics.MassAPI.Apply(body.GetPrim())
            mass_api.CreateMassAttr(mass)
            mass_api.CreateCenterOfMassAttr(Gf.Vec3f(*props['center_of_mass_m']))
            mass_api.CreateDiagonalInertiaAttr(Gf.Vec3f(*props['diagonal_inertia_kg_m2']))
            q = props['principal_axes_xyzw']
            mass_api.CreatePrincipalAxesAttr(Gf.Quatf(q[3], Gf.Vec3f(*q[:3])))
            g.schema(body.GetPrim(), 'PhysxRigidBodyAPI')
            g.attribute(body.GetPrim(), 'physxRigidBody:enableSpeculativeCCD', Sdf.ValueTypeNames.Bool, True)
            self.meta['actors'][name] = path
            self.meta['objects'].append(entry)
        elif static_context:
            entry.update(static_dressing=True, scoring=False, required=False,
                         physical_model='Fixed task-owned contextual prop; no dynamic mass.')
            self.meta.setdefault('contextual_props', {})[name] = path
            self.meta.setdefault('contextual_prop_metadata', {})[name] = entry
        else:
            self.meta['fixtures'][name] = path
            self.meta['fixture_metadata'][name] = entry
        return entry

    def decoration(self, name, mesh, material):
        g.write_mesh(self.stage, self.root+'/Markings/'+name, mesh, self.materials[material])


def _cup():
    body = g.lathe([(0, 0), (.025, 0), (.029, .002), (.031, .008),
        (.032, .073), (.0315, .079), (.0305, .080), (.0285, .080),
        (.0275, .078), (.028, .009), (.024, .005), (0, .005)])
    # Open handle in YZ, attached to the rear (+Y) wall at both ends.
    angles = np.linspace(-math.pi/2, math.pi/2, 40)
    handle = g.tube([(0., .029+.026*math.cos(t), .043+.024*math.sin(t)) for t in angles], .0045)
    return [('shell', body, 'ceramic', 'sdf'), ('handle', handle, 'ceramic', 'sdf')]


def _plate():
    return g.lathe([(0, 0), (.049, 0), (.052, .001), (.061, .005),
        (.069, .011), (.070, .013), (.069, .015), (.065, .015),
        (.059, .009), (.051, .005), (0, .005)])


def _bread(index):
    mesh = g.ellipsoid((.025, .015, .015), (0, 0, .0135))
    xyz = np.asarray(mesh.vertices)
    # Flatten the underside at its real local support plane.
    xyz[:, 2] = np.maximum(0., xyz[:, 2])
    mesh.vertices = xyz
    mesh.merge_vertices()
    mesh.remove_unreferenced_vertices()
    mesh.fix_normals()
    details = []
    rng = np.random.default_rng(901+index)
    for i in range(34):
        x, y = rng.uniform(-.018, .018), rng.uniform(-.009, .009)
        z = .0135+.015*math.sqrt(max(0, 1-(x/.025)**2-(y/.015)**2))
        details.append((f'sesame_{i:02}', g.ellipsoid((.0014, .00065, .00035), (x, y, z), subdivisions=1), 'crumb'))
    for i, x in enumerate((-.010, 0, .010)):
        yy = np.linspace(-.007, .007, 12)
        curve = [(x+.001*math.cos(y*200), y, .0135+.015*math.sqrt(max(0, 1-(x/.025)**2-(y/.015)**2))+.0001) for y in yy]
        details.append((f'score_{i}', g.tube(curve, .00055, 8), 'crumb'))
    return mesh, details


def _breakfast(a, presentation):
    source = _pose(.16, -.84, .901)
    target = _pose(.88, -.84, .9003)
    tray_pose = target if presentation else source
    tray_mesh = g.tray((.34, .24, .020), wall=.005, floor=.005)
    # D handle has a 98 x 42 mm clear XZ opening and true finger access.
    handle = g.tube([(-.057, .117, .016), (-.057, .149, .022),
        (-.057, .166, .032), (-.057, .166, .073), (-.050, .166, .080),
        (.050, .166, .080), (.057, .166, .073), (.057, .166, .032),
        (.057, .149, .022), (.057, .117, .016)], .008, 16)
    rivets = [(f'rivet_{i}', g.ellipsoid((.003, .0012, .003), (x, .126, .015), subdivisions=2), 'brass')
              for i, x in enumerate((-.057, .057))]
    tray = a.body('tray', 'serving_tray', tray_pose,
        [('shell', tray_mesh, 'wood', 'sdf'), ('handle', handle, 'brass', 'sdf')], .185,
        details=rivets, interior_bounds_xy_m=[[-.164, -.114], [.164, .114]],
        floor_top_local_z_m=.005, handle_bounds_m=[[-.065, .117, .008], [.065, .174, .088]],
        handle_opening_xz_m=[.098, .042])
    plate_goal = _world_child(target, (-.070, -.005, .0053))
    cup_goal = _world_child(target, (.090, -.005, .0053))
    bread_goals = [_world_child(plate_goal, (x, 0., .0053)) for x in (-.026, .026)]
    a.body('plate', 'breakfast_plate', plate_goal if presentation else _pose(.51, -.88, .901),
        [('shell', _plate(), 'cream', 'sdf')], .115,
        interior_radius_m=.059, interior_bounds_xy_m=[[-.059, -.059], [.059, .059]],
        floor_top_local_z_m=.005)
    a.body('cup', 'empty_cup', cup_goal if presentation else _pose(.49, -.68, .901),
        _cup(), .095, interior_radius_m=.0275, floor_top_local_z_m=.005,
        interior_bounds_xy_m=[[-.0275, -.0275], [.0275, .0275]], handle_axis_local=[0., 1., 0.])
    for i in range(2):
        mesh, details = _bread(i)
        a.body(f'bread_{i}', 'bread', bread_goals[i] if presentation else _pose(.64, -.72-i*.14, .901, i*12-6),
            [('crumb_body', mesh, 'bread', 'convexHull')], .028+i*.002, details=details)
    for name, pose, color in (('source', source, 'cream'), ('destination', target, 'blue')):
        for side in (-1, 1):
            a.decoration(name+f'_mark_{side+1}', g.box((.345, .002, .0001),
                (pose[0], pose[1]+side*.125, .90008)), color)
    a.meta['goals'] = {'source_center_xy_m': source[:2], 'destination_center_xy_m': target[:2],
        'source_region_xy_m': [[-.06, -1.025], [.38, -.61]],
        'destination_region_xy_m': [[.66, -1.025], [1.10, -.61]],
        'body_poses_xyzw': {'tray': target, 'plate': plate_goal, 'cup': cup_goal,
                           **{f'bread_{i}': pose for i, pose in enumerate(bread_goals)}},
        'support_by_actor': {'tray': BAR_SUPPORT, 'plate': 'tray', 'cup': 'tray',
                             'bread_0': 'plate', 'bread_1': 'plate'},
        'loaded_tray_local_centers_xy_m': {'plate': [-.070, -.005], 'cup': [.090, -.005]},
        'carry_axis_world': [1., 0., 0.], 'nominal_transport_distance_m': .72,
        'source_tray_pose_xyzw': source, 'destination_tray_pose_xyzw': target}
    a.meta['source_support_by_actor'] = {name: BAR_SUPPORT for name in a.meta['actors']}


def _tomato(index):
    mesh = g.lathe([(0, 0), (.011, 0), (.020, .003), (.0265, .010),
        (.028, .021), (.0265, .034), (.020, .043), (.010, .047), (0, .046)], 96)
    vertices = np.asarray(mesh.vertices)
    theta = np.arctan2(vertices[:, 1], vertices[:, 0])
    radial_scale = 1.+.018*np.cos(6*theta+index*.8)*np.sin(np.clip(vertices[:, 2]/.047, 0, 1)*math.pi)
    vertices[:, :2] *= radial_scale[:, None]
    mesh.vertices = vertices
    details = [('stem', g.tube([(0, 0, .045), (.001, .001, .053)], .0018), 'leaf')]
    for i in range(5):
        angle = i*2*math.pi/5+index*.2
        leaf = g.ellipsoid((.008, .002, .00065), (.006, 0, .047), subdivisions=2)
        leaf.apply_transform(trimesh.transformations.rotation_matrix(angle, [0, 0, 1]))
        details.append((f'sepal_{i}', leaf, 'leaf'))
    return mesh, details


def _carrot(index):
    # Preserve the original tapered root, then give it a natural resting frame.
    mesh = g.lathe([(0, 0), (.002, 0), (.006, .012), (.011, .036),
        (.0148, .065), (.0155, .086), (.013, .098), (0, .101)], 64)
    mesh.apply_transform(trimesh.transformations.rotation_matrix(math.pi/2, [1, 0, 0]))
    mesh.apply_translation((0, .0505, .015))
    vertices = np.asarray(mesh.vertices)
    vertices[:, 2] = np.maximum(vertices[:, 2], 0.)
    mesh.vertices = vertices
    mesh.merge_vertices()
    details = []
    for i, y in enumerate((-.039, -.024, -.005, .018, .035)):
        radius = float(np.interp(y, [-.0505, -.0355, -.0145, .0145, .0385, .0505], [.013, .0155, .0148, .011, .006, .002]))
        curve = [(radius*.98*math.cos(t), y+.0006*math.sin(t*3), .015+radius*.98*math.sin(t))
                 for t in np.linspace(.20, math.pi-.20, 24)]
        details.append((f'root_ring_{i}', g.tube(curve, .00038, 8), 'toast'))
    for i in range(3):
        details.append((f'stem_{i}', g.tube([(i*.002-.002, -.047, .019), (i*.003-.003, -.054, .026)], .0011), 'leaf'))
    transform, receipt = g.natural_rest_transform(mesh)
    mesh.apply_transform(transform)
    for _, detail, _ in details:
        detail.apply_transform(transform)
    mesh.metadata['natural_rest'] = receipt
    return mesh, details


def _produce(a, presentation):
    a.body('source_board', 'source_board', _pose(.255, -.815, .9001),
        [('board', g.board((.36, .34, .012)), 'wood', 'none')], 0,
        floor_top_local_z_m=.012, interior_bounds_xy_m=[[-.174, -.164], [.174, .164]])
    tray_centers = {'tomato': (.60, -.90), 'carrot': (.94, -.90)}
    for category, (x, y) in tray_centers.items():
        name = category+'_tray'
        a.body(name, category+'_target', _pose(x, y, .9001),
            [('shell', g.tray((.26, .34, .035), wall=.005, floor=.005), 'cream' if category == 'tomato' else 'blue', 'none')], 0,
            floor_top_local_z_m=.005, interior_bounds_xy_m=[[-.124, -.164], [.124, .164]], category=category)
        # On the outer rim, visible food pictograms without fake fixtures.
        if category == 'tomato':
            icon = g.ellipsoid((.012, .007, .0003), (x, y-.162, .9353), subdivisions=2)
        else:
            icon = g.ellipsoid((.004, .014, .0003), (x, y-.158, .9353), subdivisions=2)
        a.decoration(name+'_icon', icon, category)
    source_slots = [(x, y) for y in (-.925, -.815, -.705) for x in (.20, .34)]
    goal_poses, supports = {}, {}
    for i in range(3):
        for category, slot in (('tomato', i*2), ('carrot', i*2+1)):
            name = f'{category}_{i}'
            x, y = tray_centers[category]
            target_y = y+(i-1)*.090 if category == 'tomato' else -.880+(i-1)*.085
            target = _pose(x, target_y, .9054,
                           90. if category == 'carrot' else i*33.)
            source = _pose(*source_slots[slot], .9131, 90. if category == 'carrot' else 0.)
            mesh, details = _tomato(i) if category == 'tomato' else _carrot(i)
            a.body(name, category, target if presentation else source,
                [('body', mesh, category, 'convexHull')], .085+i*.005 if category == 'tomato' else .050+i*.003,
                details=details, category=category, assigned_fixture=category+'_tray',
                **({'natural_rest': mesh.metadata['natural_rest']} if category == 'carrot' else {}))
            goal_poses[name], supports[name] = target, category+'_tray'
    a.meta['goals'] = {'body_poses_xyzw': goal_poses, 'support_by_actor': supports,
        'class_to_fixture': {'tomato': 'tomato_tray', 'carrot': 'carrot_tray'}}
    a.meta['source_support_by_actor'] = {name: 'source_board' for name in a.meta['actors']}
    a.meta['layout_revision'] = 'Source board compacted and moved toward aisle; tomato/carrot columns X=0.20/0.34, Y=-0.925/-0.815/-0.705, horizontal carrots. Candidate points passed read-only original-robot IK screening; no grasp or swept-path qualification.'


def _utensil(kind):
    # Handle points +Y, food-contact end points -Y. Edges are rounded in silhouette.
    handle = [(-.007, -.015), (-.008, .045), (-.010, .068), (-.008, .078),
              (-.004, .081), (.004, .081), (.008, .078), (.010, .068), (.008, .045), (.007, -.015)]
    if kind == 'fork':
        outline = [(-.007, -.015), (-.017, -.040), (-.018, -.074), (-.016, -.081),
            (-.013, -.081), (-.012, -.054), (-.007, -.054), (-.007, -.080),
            (-.003, -.081), (-.002, -.054), (.003, -.054), (.004, -.081),
            (.008, -.080), (.008, -.054), (.013, -.054), (.014, -.081),
            (.017, -.080), (.018, -.074), (.017, -.040), (.007, -.015)]
        outline += list(reversed(handle[1:-1]))
        mesh = g.extruded_outline(outline, .006)
    elif kind == 'knife':
        outline = [(-.007, -.013), (-.010, -.072), (-.008, -.082),
            (-.003, -.085), (.006, -.083), (.015, -.069), (.016, -.027), (.007, -.013)]
        outline += list(reversed(handle[1:-1]))
        mesh = g.extruded_outline(outline, .006)
    else:
        bowl = g.lathe([(0, 0), (.006, 0), (.012, .002), (.0175, .007),
            (.018, .009), (.0165, .010), (.014, .006), (.008, .0035), (0, .003)],
            72, scale_xy=(1., 1.45))
        bowl.apply_translation((0, -.053, 0))
        stem = g.extruded_outline(handle, .006, .002)
        # A single SDF mesh containing two intersecting closed solids; the
        # 3 mm overlap at the neck is disclosed in the effective-density model.
        bridge = g.box((.014, .020, .006), (0, -.020, .005))
        mesh = trimesh.util.concatenate((bowl, bridge, stem))
    details = []
    for i, y in enumerate((.025, .031, .037)):
        details.append((f'handle_engraving_{i}', g.box((.009, .0005, .0001), (0, y, .0061 if kind != 'spoon' else .0081)), 'brass'))
    return mesh, details


def _utensils(a, presentation):
    a.body('source_pad', 'source_pad', _pose(.33, -.805, .9001),
        [('pad', g.board((.42, .40, .004), bevel=.0008), 'rubber', 'none')], 0,
        floor_top_local_z_m=.004, interior_bounds_xy_m=[[-.206, -.196], [.206, .196]])
    parts = [('shell', g.tray((.34, .32, .030), wall=.005, floor=.005), 'wood', 'none')]
    for i, x in enumerate((-.055, .055)):
        parts.append((f'divider_{i}', g.box((.005, .300, .022), (x, 0, .016)), 'wood', 'none'))
    channels = {kind: {'interior_bounds_xy_m': [[x-.047, -.148], [x+.047, .148]],
                       'floor_top_local_z_m': .005, 'center_local_xy_m': [x, 0.]}
                for kind, x in (('spoon', -.110), ('fork', 0.), ('knife', .110))}
    a.body('cutlery_tray', 'cutlery_tray', _pose(.80, -.90, .9001), parts, 0,
        floor_top_local_z_m=.005, interior_bounds_xy_m=[[-.164, -.154], [.164, .154]], channels=channels)
    goal_poses, supports = {}, {}
    for k, kind in enumerate(('spoon', 'fork', 'knife')):
        for i in range(2):
            name = f'{kind}_{i}'
            target = _pose(.80+(k-1)*.110+(i-.5)*.044, -.90, .9054)
            source = _pose(.20+k*.13, -.70-i*.21, .9051, 180. if i == 0 else 0.)
            mesh, details = _utensil(kind)
            a.body(name, kind, target if presentation else source,
                [('utensil', mesh, 'steel', 'sdf')], {'spoon': .032, 'fork': .030, 'knife': .038}[kind],
                details=details, category=kind, assigned_fixture='cutlery_tray', assigned_channel=kind,
                handle_axis_local=[0., 1., 0.], food_contact_axis_local=[0., -1., 0.])
            goal_poses[name], supports[name] = target, 'cutlery_tray'
    # Matching silhouettes at the far end are visible visual-only engravings.
    for k, kind in enumerate(('spoon', 'fork', 'knife')):
        mesh, _ = _utensil(kind)
        mesh.apply_scale((.25, .25, .01))
        mesh.apply_translation((.80+(k-1)*.110, -1.028, .9052))
        a.decoration(kind+'_icon', mesh, 'brass')
    a.meta['goals'] = {'body_poses_xyzw': goal_poses, 'support_by_actor': supports,
        'class_to_channel': {'spoon': 'spoon', 'fork': 'fork', 'knife': 'knife'},
        'handle_axis_world': [0., 1., 0.]}
    a.meta['source_support_by_actor'] = {name: 'source_pad' for name in a.meta['actors']}
    a.meta['layout_revision'] = 'Source pad compacted and shifted toward aisle, X=0.20/0.33/0.46, Y=-0.70/-0.91; handles face into the source gap. Candidate points passed read-only original-robot IK screening; no grasp or swept-path qualification.'


def _pot(a, presentation):
    a.meta.update(work_surface_paths=[PREP_SUPPORT, COOKTOP_SUPPORT],
        workspace_bounds_m=[[-.5025, .57, .90], [.98, 1.31, 1.50]],
        recovery_workspace_bounds_m=[[-.5025, .57, .60], [.98, 1.31, 1.80]])
    a.body('source_mat', 'cookware_source_mat', _pose(.07, .735, .9001),
        [('mat', g.board((.49, .32, .003), bevel=.0006), 'rubber', 'none')], 0,
        floor_top_local_z_m=.003, interior_bounds_xy_m=[[-.240, -.155], [.240, .155]])
    shell = g.lathe([(0, 0), (.075, 0), (.083, .002), (.089, .010),
        (.090, .094), (.090, .098), (.0885, .100), (.0855, .100),
        (.084, .098), (.084, .010), (.079, .006), (0, .006)], 128)
    pot_parts = [('shell', shell, 'enamel', 'sdf')]
    for side in (-1, 1):
        points = [(side*.087, -.029, .070), (side*.109, -.029, .070),
            (side*.120, -.019, .070), (side*.120, .019, .070),
            (side*.109, .029, .070), (side*.087, .029, .070)]
        pot_parts.append((f'handle_{"left" if side<0 else "right"}', g.tube(points, .0045, 16), 'steel', 'sdf'))
    # Thin rolled steel rim is decorative; the enamel shell includes its contact surface.
    # Rolled toroidal trim does not add a solid disk over the cavity.
    rim = g.tube([(.0872*math.cos(t), .0872*math.sin(t), .0993)
                  for t in np.linspace(0, 2*math.pi, 129)], .00065, 10)
    pot_target = _pose(.485, .815, .9010)
    lid_target = _world_child(pot_target, (0, 0, .1003))
    pot = a.body('pot', 'cooking_pot', pot_target if presentation else _pose(-.03, .705, .9041, 90),
        pot_parts, .48, details=[('rolled_rim', rim, 'steel')],
        interior_radius_m=.084, interior_bounds_xy_m=[[-.084, -.084], [.084, .084]],
        floor_top_local_z_m=.006, rim_local_z_m=.100,
        rim_radius_range_m=[.0855, .0885], handle_axis_local=[1., 0., 0.])
    # The lid underside is flat at z=0 and rests on the rim. The shallow
    # locating ring extends 2 mm below the seat; actual reset origin is lowered
    # by that amount so every object's local minimum remains zero.
    lid_shell = g.lathe([(0, .002), (.080, .002), (.080, 0), (.082, 0),
        (.082, .002), (.0915, .002), (.094, .004), (.094, .006),
        (.090, .010), (.070, .017), (.040, .021), (0, .022)], 128)
    # Need ordered profile that does not fold: bottom boundary from pole to
    # locating lip, then outer edge and domed top back to the pole.
    knob = g.lathe([(0, .021), (.009, .021), (.010, .025), (.016, .032),
        (.018, .041), (.017, .047), (.014, .050), (0, .050)], 64)
    lid_target[2] -= .002
    a.body('lid', 'pot_lid', lid_target if presentation else _pose(.21, .705, .9041),
        [('shell', lid_shell, 'steel', 'sdf'), ('knob', knob, 'rubber', 'convexHull')], .145,
        seat_local_z_m=.002, locating_outer_radius_m=.082,
        seat_radius_range_m=[.082, .0915], knob_bounds_m=[[-.018, -.018, .025], [.018, .018, .050]])
    a.meta['goals'] = {'burner_center_xy_m': [.485, .815],
        'burner_support_paths': [COOKTOP_SUPPORT],
        'pot_rim_local_z_m': .100, 'lid_seat_local_z_m': .002,
        'lid_base_relative_pot_m': .098,
        'body_poses_xyzw': {'pot': pot_target, 'lid': lid_target},
        'support_by_actor': {'pot': COOKTOP_SUPPORT, 'lid': 'pot'}}
    a.meta['source_support_by_actor'] = {'pot': 'source_mat', 'lid': 'source_mat'}
    a.meta['layout_revision'] = 'Pot and lid moved to preparation-counter front at (-0.03,0.705), (0.21,0.705); pot ear axis Y gives 56 mm source-footprint separation. Reach remains unqualified.'


def author_task(stage, task_id, root_path='/World/KitchenTask', *, presentation=False,
                texture_assets=None, dressing_profile='legacy'):
    """Author one candidate task into an existing stage and return JSON metadata.

    No asset is written to disk here. A release builder must freeze source,
    export this stage and metadata, and attach independent native evidence.
    """
    if task_id not in TASK_IDS:
        raise ValueError('Unknown kitchen task: '+str(task_id))
    if stage.GetPrimAtPath(root_path):
        raise ValueError('Task root already exists: '+root_path)
    if dressing_profile not in ('legacy', 'enriched_v1'):
        raise ValueError('Unknown kitchen dressing profile: '+str(dressing_profile))
    from .expanded_assets import pantry_stocking, table_clearing, enrich_context, enrich_public_goal
    author = _Author(stage, task_id, root_path, presentation, texture_assets)
    {'breakfast_serving': _breakfast, 'produce_transfer': _produce,
     'utensil_sorting': _utensils, 'pot_lid_placement': _pot,
     'pantry_stocking': pantry_stocking, 'table_clearing': table_clearing}[task_id](author, presentation)
    if dressing_profile == 'enriched_v1':
        enrich_context(author)
        enrich_public_goal(author)
    return author.meta
