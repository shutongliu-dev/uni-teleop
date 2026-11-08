"""Opt-in native rigid-cell elastic sponge candidate; no runtime state writes.

The 27 convex cells and green pad are dynamic rigid bodies. Their 63 D6 links
are passive, finite force springs. This is an explicitly discrete constitutive
prior, not calibrated continuum FEM or a controller. The profile is explicit.
The detailed skin is a hidden immutable rest template; the native Bench renderer
uses actual sampled cell transforms to display its deformation.
"""
from __future__ import annotations

import itertools
import numpy as np
import trimesh
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation
from pxr import Gf, Sdf, UsdGeom, UsdPhysics, UsdShade

from . import geometry as g
from .sponge_deformable import TOTAL_MASS_KG, FOAM_MASS_KG, SCOURING_MASS_KG

PROFILE = 'elastic_cell_lattice_v1'
Z_LAYERS = (.004, .022, .038, .045)
YOUNG = 10000.
POISSON_PRIOR = .25
SHEAR = YOUNG/(2*(1+POISSON_PRIOR))
DAMPING_RATIO = .15
LIMITATIONS = [
    'Uncalibrated 27-cell discrete beam/spring approximation, not isotropic continuum FEM.',
    'Poisson ratio only defines the shear-modulus prior; lateral Poisson response is not guaranteed.',
    'Rigid cells do not change volume; filtered internal overlap approximates porous compression.',
    'Internal collision is disabled for this entire composite; severe collapse must be rejected by geometry checks.',
    'Finite face springs omit intrarigid-cell strain and require separate compression/shear calibration.',
    'The green 4 mm scouring layer is rigid. No absorption, fracture, plasticity or fluid model.',
    'Detailed skin follows sampled native cells; its fine visual pores are not collision geometry.',
    'Joint reaction reporting excludes drive forces; use external contact impulses and independent momentum checks.',
]


def _half_widths(z):
    return np.array([np.interp(z, Z_LAYERS, [.060, .060, .0396, .0372]),
                     np.interp(z, Z_LAYERS, [.039, .039, .024, .0228])])


def _outline(z):
    hx, hy = _half_widths(z)
    radius = .0036
    points = []
    for sx, sy in itertools.product((-1, 1), repeat=2):
        for angle in (0., np.pi/4, np.pi/2):
            points.append([sx*(hx-radius+radius*np.cos(angle)),
                           sy*(hy-radius+radius*np.sin(angle))])
    points = np.asarray(points)
    return points[ConvexHull(points).vertices]


def _clip(poly, axis, bound, sign):
    """Clip a convex polygon to sign*(coordinate-bound) <= 0."""
    result = []
    for start, end in zip(poly, np.roll(poly, -1, axis=0)):
        a, b = sign*(start[axis]-bound), sign*(end[axis]-bound)
        if a <= 1e-14:
            result.append(start)
        if (a < -1e-14 and b > 1e-14) or (a > 1e-14 and b < -1e-14):
            result.append(start+(end-start)*a/(a-b))
    return np.asarray(result)


def _section(i, j, z):
    poly = _outline(z)
    half = _half_widths(z)
    for axis, index in enumerate((i, j)):
        low, high = (-1+2*index/3)*half[axis], (-1+2*(index+1)/3)*half[axis]
        poly = _clip(_clip(poly, axis, low, -1), axis, high, 1)
    return np.column_stack((poly, np.full(len(poly), z)))


def lattice_geometry():
    """Exact convex partition of the baseline two-chord/taper foam boundary."""
    cells = {}
    for i, j, k in itertools.product(range(3), repeat=3):
        vertices = np.vstack((_section(i, j, Z_LAYERS[k]), _section(i, j, Z_LAYERS[k+1])))
        cells[i, j, k] = trimesh.convex.convex_hull(vertices)
    return cells


def _face_geometry(key, axis):
    i, j, k = key
    if axis == 2:
        return _section(i, j, Z_LAYERS[k+1])
    vertices = []
    for z in (Z_LAYERS[k], Z_LAYERS[k+1]):
        hx, hy = _half_widths(z)
        if axis == 0:
            x = (-1+2*(i+1)/3)*hx
            vertices.extend([(x, (-1+2*j/3)*hy, z), (x, (-1+2*(j+1)/3)*hy, z)])
        else:
            y = (-1+2*(j+1)/3)*hy
            vertices.extend([((-1+2*i/3)*hx, y, z), ((-1+2*(i+1)/3)*hx, y, z)])
    return np.asarray(vertices)


def _face_properties(vertices, direction):
    """Area, centroid and second moments of one planar material face."""
    origin = vertices.mean(0)
    _, _, basis = np.linalg.svd(vertices-origin)
    normal = basis[-1]
    if normal@direction < 0:
        normal *= -1
    tangent = np.cross([0., 0., 1.], normal)
    if np.linalg.norm(tangent) < 1e-10:
        tangent = np.array([1., 0., 0.])
    tangent /= np.linalg.norm(tangent)
    frame = np.column_stack((normal, tangent, np.cross(normal, tangent)))
    uv = (vertices-origin)@frame[:, 1:]
    uv = uv[ConvexHull(uv).vertices]
    u, v = uv.T
    un, vn = np.roll(uv, -1, axis=0).T
    cross = u*vn-un*v
    area = cross.sum()/2
    center = np.array([np.sum((u+un)*cross), np.sum((v+vn)*cross)])/(6*area)
    iu = np.sum((u*u+u*un+un*un)*cross)/12-area*center[0]**2
    iv = np.sum((v*v+v*vn+vn*vn)*cross)/12-area*center[1]**2
    return area, origin+frame[:, 1:]@center, frame, np.array([iu, iv])


def _body(stage, path, mesh, mass, position, orientation, visual_material, physics_material):
    center = np.asarray(mesh.center_mass)
    local = mesh.copy()
    local.vertices -= center
    props = g.mass_properties([local], mass)
    xform = UsdGeom.Xform.Define(stage, path)
    xform.AddTranslateOp().Set(Gf.Vec3d(*(position+orientation.apply(center))))
    q = orientation.as_quat()
    xform.AddOrientOp().Set(Gf.Quatf(q[3], Gf.Vec3f(*q[:3])))
    prim = xform.GetPrim()
    UsdPhysics.RigidBodyAPI.Apply(prim).CreateKinematicEnabledAttr(False)
    mp = UsdPhysics.MassAPI.Apply(prim)
    mp.CreateMassAttr(float(mass))
    mp.CreateCenterOfMassAttr(Gf.Vec3f(*props['center_of_mass_m']))
    mp.CreateDiagonalInertiaAttr(Gf.Vec3f(*props['diagonal_inertia_kg_m2']))
    q = props['principal_axes_xyzw']
    mp.CreatePrincipalAxesAttr(Gf.Quatf(q[3], Gf.Vec3f(*q[:3])))
    g.schema(prim, 'PhysxRigidBodyAPI')
    for name, kind, value in (
        ('linearDamping', Sdf.ValueTypeNames.Float, 0.),
        ('angularDamping', Sdf.ValueTypeNames.Float, 0.),
        ('solverPositionIterationCount', Sdf.ValueTypeNames.Int, 32),
        ('solverVelocityIterationCount', Sdf.ValueTypeNames.Int, 1),
    ):
        g.attribute(prim, 'physxRigidBody:'+name, kind, value)
    g.write_mesh(stage, path+'/Visual/cell', local, visual_material)
    collider_path = path+'/Collision/cell'
    collider = g.write_mesh(stage, collider_path, local, visual_material, collision='convexHull')
    UsdGeom.Imageable(collider).CreateVisibilityAttr('invisible')
    UsdShade.MaterialBindingAPI(collider).Bind(physics_material, materialPurpose='physics')
    return {'id': path.rsplit('/', 1)[-1], 'body_path': path, 'path': path,
        'role': 'passive_elastic_sponge_cell', 'mass_kg': float(mass), 'mass': float(mass), **props,
        'initial_pose_xyzw': [*(position+orientation.apply(center)).tolist(), *orientation.as_quat().tolist()],
        'composite_rest_center_m': center.tolist(), 'volume_m3': float(mesh.volume),
        'local_bounds_m': local.bounds.tolist(), 'collision_bounds_m': local.bounds.tolist(),
        'collision_support_points_m': local.vertices.tolist(), 'collider_paths': [collider_path],
        'collision_count': 1, 'component_of': 'sponge', 'physics_accepted': False}


def _joint(stage, path, first, second, vertices, inertias):
    c0, c1 = (np.asarray(row['composite_rest_center_m']) for row in (first, second))
    area, anchor, frame, moments = _face_properties(vertices, c1-c0)
    length = float((c1-c0)@frame[:, 0])
    if length <= 0:
        raise ValueError('Elastic material link has nonpositive rest length')
    stiffness = np.array([YOUNG*area/length, SHEAR*area/length, SHEAR*area/length,
                          SHEAR*moments.sum()/length, YOUNG*moments[1]/length,
                          YOUNG*moments[0]/length])
    # Omni PhysX swing Y/Z drive coefficients must match. Preserve their mean
    # and record the discarded anisotropic face-moment distinction explicitly.
    rotational_unaveraged = stiffness[3:].copy()
    stiffness[4:] = stiffness[4:].mean()
    mass0, mass1 = first['mass_kg'], second['mass_kg']
    reduced_mass = mass0*mass1/(mass0+mass1)
    inverse_inertia = np.linalg.inv(inertias[first['id']])+np.linalg.inv(inertias[second['id']])
    reduced_inertias = 1/np.einsum('ij,ij->j', frame, inverse_inertia@frame)
    damping = 2*DAMPING_RATIO*np.sqrt(stiffness*np.r_[np.full(3, reduced_mass), reduced_inertias])
    damping[4:] = damping[4:].mean()
    joint = UsdPhysics.Joint.Define(stage, path)
    joint.CreateBody0Rel().SetTargets([first['body_path']])
    joint.CreateBody1Rel().SetTargets([second['body_path']])
    # Both links share the sponge's initial orientation. The pad origin remains
    # its original bottom plane; each foam actor origin is its own COM.
    for index, row in enumerate((first, second)):
        body_origin = np.asarray(row.get('composite_rest_origin_m', row['composite_rest_center_m']))
        getattr(joint, 'CreateLocalPos'+str(index)+'Attr')().Set(Gf.Vec3f(*(anchor-body_origin)))
        q = Rotation.from_matrix(frame).as_quat()
        getattr(joint, 'CreateLocalRot'+str(index)+'Attr')().Set(Gf.Quatf(q[3], Gf.Vec3f(*q[:3])))
    joint.CreateExcludeFromArticulationAttr(True)
    joint.CreateCollisionEnabledAttr(False)
    drives = {}
    for index, axis in enumerate(('transX', 'transY', 'transZ', 'rotX', 'rotY', 'rotZ')):
        scale = np.pi/180 if index >= 3 else 1.
        drive = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), axis)
        drive.CreateTypeAttr('force')
        drive.CreateStiffnessAttr(float(stiffness[index]*scale))
        drive.CreateDampingAttr(float(damping[index]*scale))
        drive.CreateTargetPositionAttr(0.)
        drive.CreateTargetVelocityAttr(0.)
        drive.CreateMaxForceAttr(float('inf'))
        drives[axis] = {'stiffness_si': float(stiffness[index]), 'damping_si': float(damping[index]),
                        'usd_stiffness': float(stiffness[index]*scale), 'usd_damping': float(damping[index]*scale)}
    return {'prim_path': path, 'body0': first['id'], 'body1': second['id'],
        'face_area_m2': float(area), 'rest_normal_distance_m': length,
        'face_second_moments_m4': moments.tolist(), 'anchor_rest_m': anchor.tolist(),
        'joint_frame_columns': frame.tolist(), 'reduced_mass_kg': reduced_mass,
        'reduced_angular_inertia_kg_m2': reduced_inertias.tolist(),
        'unaveraged_angular_stiffness_nm_rad': rotational_unaveraged.tolist(),
        'swing_y_z_coefficients_averaged': True, 'drives': drives}


def author_lattice_sponge(stage, metadata):
    """Explicitly replace a fresh household rigid sponge; defaults stay intact.

    Call after ``author_task(..., prop_profile='household_v1')``. Return the
    candidate record and update its actor inventory for native diagnostics.
    This does not select or publish a task release or modify public contracts.
    """
    if metadata['task_id'] != 'surface_wiping' or metadata.get('prop_profile') != 'household_v1':
        raise ValueError('Elastic lattice requires household surface_wiping')
    if metadata.get('sponge_physics_profile', 'rigid') != 'rigid' or metadata.get('volume_bodies'):
        raise ValueError('Elastic lattice requires a fresh rigid sponge')
    stage_units = UsdGeom.GetStageMetersPerUnit(stage)
    if not np.isclose(stage_units, 1.):
        raise ValueError('Elastic lattice requires metre stage units')
    root = metadata['root_path']
    row = next(x for x in metadata['objects'] if x['id'] == 'sponge')
    body_path = row['body_path']
    skin_path = body_path+'/Visual/finish_porous_foam'
    green_path = body_path+'/Visual/finish_green_scouring_layer'
    if not stage.GetPrimAtPath(skin_path) or not stage.GetPrimAtPath(green_path):
        raise ValueError('Elastic lattice requires detailed household skin')
    pose = np.asarray(row['initial_pose_xyzw'], float)
    rotation = Rotation.from_quat(pose[3:])
    contact = UsdShade.Material(stage.GetPrimAtPath(root+'/Materials/Contact'))
    yellow = UsdShade.Material(stage.GetPrimAtPath(root+'/Materials/crumb'))
    green = UsdShade.Material(stage.GetPrimAtPath(root+'/Materials/leaf'))
    # Preserve all original porous skin attributes without pretending the
    # static parent will deform it; native cell visuals remain visible.
    UsdGeom.Imageable(stage.GetPrimAtPath(skin_path)).CreateVisibilityAttr('invisible')
    stage.RemovePrim(body_path+'/Collision')
    pad = g.rounded_loft([(0, 0, 0, 0), (.060, .039, .0036, 0),
                          (.060, .039, .0036, .004), (0, 0, 0, .004)])
    collider_path = body_path+'/Collision/scouring'
    collider = g.write_mesh(stage, collider_path, pad, green, collision='convexHull')
    UsdGeom.Imageable(collider).CreateVisibilityAttr('invisible')
    UsdShade.MaterialBindingAPI(collider).Bind(contact, materialPurpose='physics')
    props = g.mass_properties([pad], SCOURING_MASS_KG)
    mp = UsdPhysics.MassAPI(stage.GetPrimAtPath(body_path))
    mp.GetMassAttr().Set(SCOURING_MASS_KG)
    mp.GetCenterOfMassAttr().Set(Gf.Vec3f(*props['center_of_mass_m']))
    mp.GetDiagonalInertiaAttr().Set(Gf.Vec3f(*props['diagonal_inertia_kg_m2']))
    q = props['principal_axes_xyzw']
    mp.GetPrincipalAxesAttr().Set(Gf.Quatf(q[3], Gf.Vec3f(*q[:3])))
    row.update(props, mass_kg=SCOURING_MASS_KG, mass=SCOURING_MASS_KG,
        role='rigid_scouring_layer_of_elastic_sponge', collider_paths=[collider_path], collision_count=1,
        collision_bounds_m=pad.bounds.tolist(), local_bounds_m=pad.bounds.tolist(),
        collision_support_points_m=pad.vertices.tolist(), composite_mass_kg=TOTAL_MASS_KG,
        composite_rest_center_m=np.asarray(pad.center_mass).tolist(), composite_rest_origin_m=[0., 0., 0.],
        physical_approximation='Passive 27-cell elastic lattice and rigid 4 mm green layer; uncalibrated discrete constitutive prior.',
        public_description='Yellow elastic sponge candidate with green scouring underside')
    row['dimension_profile'].update(component_mass_kg=SCOURING_MASS_KG,
        composite_mass_kg=TOTAL_MASS_KG,
        mass_rule='Existing 79.2 g composite partitioned 90% volume-weighted rigid cells and 10% green layer.')
    prim = stage.GetPrimAtPath(body_path)
    for name, kind, value in (('linearDamping', Sdf.ValueTypeNames.Float, 0.),
                             ('angularDamping', Sdf.ValueTypeNames.Float, 0.),
                             ('solverPositionIterationCount', Sdf.ValueTypeNames.Int, 32),
                             ('solverVelocityIterationCount', Sdf.ValueTypeNames.Int, 1)):
        g.attribute(prim, 'physxRigidBody:'+name, kind, value)
    geometry = lattice_geometry()
    volume = sum(mesh.volume for mesh in geometry.values())
    cells, inertias = {}, {'sponge': np.asarray(pad.moment_inertia)*SCOURING_MASS_KG/pad.mass}
    for key, mesh in geometry.items():
        mass = FOAM_MASS_KG*mesh.volume/volume
        name = 'sponge_cell_'+'_'.join(map(str, key))
        cell = _body(stage, root+'/Actors/'+name, mesh, mass, pose[:3], rotation, yellow, contact)
        cell['grid_index'] = list(key)
        cells[key] = cell
        inertias[name] = np.asarray(mesh.moment_inertia)*mass/mesh.mass
        metadata['objects'].append(cell)
        metadata['actors'][name] = cell['body_path']
    all_rows = [row, *cells.values()]
    for first in all_rows:
        # Whole-composite pair filtering allows compressing porous cells without
        # redundant hard contacts. External table/hand collisions remain on.
        UsdPhysics.FilteredPairsAPI.Apply(stage.GetPrimAtPath(first['body_path'])).CreateFilteredPairsRel().SetTargets(
            [Sdf.Path(other['body_path']) for other in all_rows if other is not first])
    links = []
    for key, cell in cells.items():
        for axis in range(3):
            next_key = list(key)
            next_key[axis] += 1
            other = cells.get(tuple(next_key))
            if other is not None:
                links.append(_joint(stage, root+'/ElasticLinks/link_'+str(len(links)), cell, other,
                                    _face_geometry(key, axis), inertias))
        if key[2] == 0:
            links.append(_joint(stage, root+'/ElasticLinks/link_'+str(len(links)), row, cell,
                                _section(*key[:2], Z_LAYERS[0]), inertias))
    metadata.setdefault('joints', {}).update({link['prim_path'].rsplit('/', 1)[-1]: link['prim_path']
                                            for link in links})
    goal = metadata['goals']['body_poses_xyzw']['sponge']
    goal_rotation = Rotation.from_quat(goal[3:])
    for cell in cells.values():
        center = np.asarray(cell['composite_rest_center_m'])
        metadata['goals']['body_poses_xyzw'][cell['id']] = [
            *(np.asarray(goal[:3])+goal_rotation.apply(center)).tolist(), *goal[3:]]
    # Compact read-only rest binding for the render-only skin consumer. At
    # rest, each influence transforms the same original vertex exactly. These
    # weights never enter collision, mass, spring or force computations.
    skin_points = np.asarray(UsdGeom.Mesh(stage.GetPrimAtPath(skin_path)).GetPointsAttr().Get(), float)
    import hashlib
    skin_binding = {'prim_path': skin_path,
        'binding_schema': 'eight_nearest_inverse_squared_v1',
        'neighbor_count': 8, 'minimum_squared_distance_m2': 1e-12,
        'point_count': len(skin_points),
        'rest_points_sha256': hashlib.sha256(skin_points.tobytes()).hexdigest(),
        'member_ids': [cell['id'] for cell in cells.values()],
        'mechanism': 'Eight-nearest-cell inverse-square rigid-transform blend from the immutable USD rest template; rendering only.',
        'runtime_deformation_verified': False, 'physical_surface_equivalence_verified': False}
    record = {'profile': PROFILE, 'candidate_only': True, 'physics_accepted': False,
        'total_mass_kg': TOTAL_MASS_KG, 'foam_mass_kg': FOAM_MASS_KG, 'pad_mass_kg': SCOURING_MASS_KG,
        'foam_volume_m3': float(volume), 'foam_density_kg_m3': float(FOAM_MASS_KG/volume),
        'grid': [3, 3, 3], 'z_layers_m': list(Z_LAYERS), 'cell_count': len(cells),
        'joint_count': len(links), 'cells': list(cells.values()), 'links': links,
        'parent_rigid_body': 'sponge', 'member_ids': [x['id'] for x in all_rows],
        'logical_object_count': 1, 'native_body_count': len(all_rows),
        'rest_local_bounds_m': [[-.06, -.039, 0.], [.06, .039, .045]],
        'visual_skin_bindings': [skin_binding],
        'material_prior': {'youngs_modulus_pa': YOUNG, 'poissons_ratio_for_shear_prior': POISSON_PRIOR,
            'shear_modulus_pa': SHEAR, 'modal_damping_ratio': DAMPING_RATIO,
            'linear_stiffness': 'normal E*A/L; tangent G*A/L',
            'angular_stiffness': 'twist G*(Iu+Iv)/L; bend E*Iv/L and E*Iu/L, then average swing Y/Z',
            'damping': '2*zeta*sqrt(k*reduced_mass_or_axis_inertia), then average swing Y/Z',
            'force_drive': True, 'angular_usd_conversion': 'SI per radian times pi/180 to per degree',
            'global_continuum_modulus_calibrated': False, 'mass_scaling': False,
            'rigid_drag': 0., 'hard_joint_constraints': 0},
        'internal_collision_filter': {'all_component_pairs': True, 'body_count': len(all_rows),
            'unique_pair_count': len(all_rows)*(len(all_rows)-1)//2},
        'visual': {'rest_skin_path': skin_path, 'rest_skin_preserved': True, 'rest_skin_hidden': True,
            'green_skin_path': green_path, 'visible_geometry': 'Native rigid cell surfaces and original green skin',
            'runtime_skinning_implemented': True,
            'runtime_visual_consumer': 'bench.simulation.isaacsim_backend.elastic_visual_skin.ElasticVisualSkin',
            'state_writes': 0},
        'limitations': list(LIMITATIONS), 'native_validation_complete': False}
    metadata['sponge_physics_profile'] = PROFILE
    metadata['elastic_composites'] = {'sponge': record}
    metadata['composite_objects'] = {'sponge': {'rigid_body': 'sponge',
        'rigid_components': [x['id'] for x in all_rows], 'total_mass_kg': TOTAL_MASS_KG,
        'rest_local_bounds_m': [[-.06, -.039, 0.], [.06, .039, .045]],
        'release_requires_all_components': True}}
    metadata['physics_prior']['sponge_elastic_lattice'] = record['material_prior']
    metadata['source']['cleaning_approximation'] = 'Passive elastic cell sponge candidate; only actual loaded green-layer contact and travel may clear soil cells.'
    metadata['visual_finish']['sponge_approximation'] = 'Native cells supply physics; the ordinary Bench renderer blends their sampled transforms into the preserved porous skin without writing physical state.'
    return record
