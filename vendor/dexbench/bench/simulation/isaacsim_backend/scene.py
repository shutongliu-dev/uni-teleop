"""USD scene authoring for native Isaac Sim; import only after SimulationApp.

Surface deformable authoring follows NVIDIA Omni Physics 110.1 single mesh
surface example. Material parameters are explicit PhysX assumptions, not a
numerical equivalence to the old Newton VBD material.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from pxr import Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdLux, UsdPhysics, UsdShade, Vt

from .asset_catalog import (load_task_asset, load_obj_mesh, collision_offsets, inertia_principal_axes,
                            collision_mesh_coordinate_scale, condition_collision_mesh,
                            validate_triangle_mesh_role, prepare_triangle_mesh)
from .geometry import cloth_grid
from .robot_asset import collision_configuration
from .robot_materials import (apply_pad_compliance, bind_collider_instances, material_kind, material_options,
                              resolved_collider_materials)
from .visuals import uv_box
from bench.paths import DATA_ROOT, load_json

DEFORMABLE_DOC = 'https://docs.omniverse.nvidia.com/kit/docs/omni_physics/110.1/dev_guide/deformables/deformable_bodies.html'


def transform(prim, position=(0, 0, 0), quaternion=(1, 0, 0, 0), scale=None):
    x = UsdGeom.Xformable(prim)
    x.ClearXformOpOrder()
    x.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in position]))
    q = np.asarray(quaternion, dtype=float)
    q /= np.linalg.norm(q)
    x.AddOrientOp().Set(Gf.Quatf(float(q[0]), Gf.Vec3f(*q[1:].tolist())))
    if scale is not None:
        x.AddScaleOp().Set(Gf.Vec3f(*[float(v) for v in scale]))


def visual_material(stage, name, color, roughness=.55, metallic=0.):
    path = '/World/Looks/' + name
    mat = UsdShade.Material.Define(stage, path)
    shader = UsdShade.Shader.Define(stage, path + '/Surface')
    shader.CreateIdAttr('UsdPreviewSurface')
    shader.CreateInput('diffuseColor', Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*map(float, color)))
    shader.CreateInput('roughness', Sdf.ValueTypeNames.Float).Set(float(roughness))
    shader.CreateInput('metallic', Sdf.ValueTypeNames.Float).Set(float(metallic))
    mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), 'surface')
    return mat


def bind_visual(prim, material):
    UsdShade.MaterialBindingAPI.Apply(prim).Bind(material)


def rigid_material(stage, name, friction):
    mat = UsdShade.Material.Define(stage, '/World/PhysicsMaterials/' + name)
    api = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
    api.CreateStaticFrictionAttr(float(friction))
    api.CreateDynamicFrictionAttr(float(friction))
    api.CreateRestitutionAttr(0.)
    PhysxSchema.PhysxMaterialAPI.Apply(mat.GetPrim()).CreateFrictionCombineModeAttr('average')
    return mat


def collision(prim, material, *, convex=False, contact_offset=.001, rest_offset=0.):
    UsdPhysics.CollisionAPI.Apply(prim)
    if convex:
        UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr('convexHull')
    physx = PhysxSchema.PhysxCollisionAPI.Apply(prim)
    physx.CreateContactOffsetAttr(float(contact_offset))
    physx.CreateRestOffsetAttr(float(rest_offset))
    UsdShade.MaterialBindingAPI.Apply(prim).Bind(material, UsdShade.Tokens.weakerThanDescendants, 'physics')


def mesh(stage, path, vertices, triangles, normals=None):
    m = UsdGeom.Mesh.Define(stage, path)
    vertices = np.asarray(vertices, dtype=np.float32)
    triangles = np.asarray(triangles, dtype=np.int32).reshape(-1, 3)
    m.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(vertices))
    m.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(triangles.reshape(-1)))
    m.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(np.full(len(triangles), 3, dtype=np.int32)))
    m.CreateSubdivisionSchemeAttr('none')
    m.CreateDoubleSidedAttr(True)
    if normals is not None:
        m.CreateNormalsAttr(Vt.Vec3fArray.FromNumpy(np.asarray(normals, dtype=np.float32)))
        m.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
    return m


def box(stage, path, position, half, material, physics_material=None):
    g = UsdGeom.Cube.Define(stage, path)
    g.CreateSizeAttr(2.)
    transform(g.GetPrim(), position, scale=half)
    bind_visual(g.GetPrim(), material)
    if physics_material:
        collision(g.GetPrim(), physics_material)
    return g.GetPrim()


def add_task_asset(stage, spec, *, dynamic):
    info = _add_task_asset(stage, spec, dynamic=dynamic)
    if spec.get('presentation') is not None:
        from .presentation import apply_presentation
        root = '/World/' + ('Objects/' if dynamic else 'Fixtures/') + spec['id']
        info['presentation'] = apply_presentation(stage, root, spec)
    return info


def _add_task_asset(stage, spec, *, dynamic):
    asset = load_task_asset(spec['asset_id'], asset_root=spec.get('asset_root'), verify_hashes=True)
    coordinate_scale = collision_mesh_coordinate_scale(spec)
    if asset.get('backend') == 'usd_reference':
        if coordinate_scale != 1.:
            raise ValueError('Collision mesh conditioning supports plain mesh assets only')
        from .usd_asset import add_usd_asset
        return add_usd_asset(stage, spec, asset, dynamic=dynamic)
    validate_triangle_mesh_role(asset, dynamic=dynamic)
    if coordinate_scale != 1. and any(p.get('geometry','convex_mesh') not in ('mesh','convex_mesh')
                                      for p in asset['collision_parts']):
        raise ValueError('Collision mesh conditioning requires explicit mesh collision parts')
    offsets = collision_offsets(spec)
    # Preflight every explicit raw triangle part before creating any prim or
    # material. Retain the checked arrays; never reload different bytes later.
    triangle_meshes = {}
    for kind, parts in [('visual', asset['visual_parts']), ('collision', asset['collision_parts'])]:
        for index, part in enumerate(parts):
            if part.get('geometry') == 'triangle_mesh':
                vertices, faces = load_obj_mesh(part['mesh_path'])
                triangle_meshes[kind, index] = prepare_triangle_mesh(vertices, faces, part.get('scale', [1, 1, 1]))
    path = '/World/' + ('Objects/' if dynamic else 'Fixtures/') + spec['id']
    root = UsdGeom.Xform.Define(stage, path).GetPrim()
    transform(root, spec['position_m'], spec.get('quaternion_wxyz', [1, 0, 0, 0]))
    physmat = rigid_material(stage, spec['id'], spec.get('friction', .65))
    if dynamic:
        UsdPhysics.RigidBodyAPI.Apply(root)
        mass = UsdPhysics.MassAPI.Apply(root)
        mass.CreateMassAttr(float(asset['mass_kg']))
        mass.CreateCenterOfMassAttr(Gf.Vec3f(*asset['center_of_mass_m']))
        mass.CreateDiagonalInertiaAttr(Gf.Vec3f(*asset['inertia_diag_kg_m2']))
        principal = inertia_principal_axes(asset)
        mass.CreatePrincipalAxesAttr(Gf.Quatf(principal[0], Gf.Vec3f(*principal[1:])))
        body = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
        body.CreateSolverPositionIterationCountAttr(16)
        body.CreateSolverVelocityIterationCountAttr(4)
        body.CreateEnableSpeculativeCCDAttr(True)
        PhysxSchema.PhysxContactReportAPI.Apply(root).CreateThresholdAttr(0.)
    for kind, parts in [('visual', asset['visual_parts']), ('collision', asset['collision_parts'])]:
        for i, part in enumerate(parts):
            child = f'{path}/{kind}_{i:03d}'
            geometry = part.get('geometry', 'convex_mesh')
            if geometry == 'triangle_mesh':
                v, f = triangle_meshes[kind, i]
                g, scale = mesh(stage, child, v, f), None
            elif geometry in ('mesh', 'convex_mesh'):
                v, f = load_obj_mesh(part['mesh_path'])
                v *= np.asarray(part.get('scale', [1, 1, 1]), dtype=np.float32)
                v, scale = condition_collision_mesh(v, coordinate_scale) if kind == 'collision' else (v, None)
                g = mesh(stage, child, v, f)
            elif geometry == 'box':
                g = UsdGeom.Cube.Define(stage, child)
                g.CreateSizeAttr(2.)
                scale = part['half_extents_m']
            elif geometry == 'cylinder':
                g = UsdGeom.Cylinder.Define(stage, child)
                g.CreateRadiusAttr(float(part['radius_m']))
                g.CreateHeightAttr(2 * float(part['half_height_m']))
                g.CreateAxisAttr('Z')
                scale = None
            elif geometry == 'sphere':
                g = UsdGeom.Sphere.Define(stage, child)
                g.CreateRadiusAttr(float(part['radius_m']))
                scale = None
            else:
                raise ValueError(f'Unknown task collision geometry: {geometry}')
            q = part.get('quaternion_xyzw', [0, 0, 0, 1])
            transform(g.GetPrim(), part.get('position_m', [0, 0, 0]), [q[3], *q[:3]], scale)
            if kind == 'visual':
                color = part.get('color', [.55, .60, .65])
                mat = visual_material(stage, f'{spec["id"]}_{i}', color, part.get('roughness', .35), part.get('metallic', .65))
                bind_visual(g.GetPrim(), mat)
            else:
                collision(g.GetPrim(), physmat, convex=geometry in ('mesh', 'convex_mesh'), **offsets)
                if geometry == 'triangle_mesh':
                    UsdPhysics.MeshCollisionAPI.Apply(g.GetPrim()).CreateApproximationAttr('none')
                # Invisible display does not disable physical collision.
                UsdGeom.Imageable(g).CreateVisibilityAttr(UsdGeom.Tokens.invisible)
    return {'path': path, 'asset_id': spec['asset_id'], 'manifest_sha256': asset['manifest_sha256'], 'asset_root': asset['asset_root'],
            'collision_paths': [f'{path}/collision_{i:03d}' for i in range(len(asset['collision_parts']))],
            'collision_parts': len(asset['collision_parts']), 'visual_parts': len(asset['visual_parts']),
            'collision_mesh_coordinate_scale': coordinate_scale,
            'mass_kg': float(asset['mass_kg']) if dynamic else None,
            'center_of_mass_m': asset.get('center_of_mass_m') if dynamic else None,
            'inertia_diag_kg_m2': asset.get('inertia_diag_kg_m2') if dynamic else None,
            **({'principal_axes_xyzw': list(asset['principal_axes_xyzw'])}
               if dynamic and 'principal_axes_xyzw' in asset else {}),
            'rigid_friction': float(spec.get('friction', .65)), 'physics_material_purpose': 'physics'}


def required_api(prim, name):
    if not prim.ApplyAPI(name) or not prim.HasAPI(name):
        raise RuntimeError(f'Required native Isaac Sim schema unavailable: {name}')


def set_attr(prim, name, value):
    attr = prim.GetAttribute(name)
    if not attr:
        raise RuntimeError(f'Required native schema attribute unavailable: {prim.GetPath()}.{name}')
    if not attr.Set(value):
        raise RuntimeError(f'Cannot author {name}')


def add_cloth(stage, spec, options):
    v, triangles = cloth_grid(spec)
    path = '/World/Cloth'
    m = mesh(stage, path, v, triangles)
    p = m.GetPrim()
    transform(p, spec['origin_m'], spec.get('quaternion_wxyz', [1, 0, 0, 0]))
    for schema in ('OmniPhysicsDeformableBodyAPI', 'OmniPhysicsSurfaceDeformableSimAPI', 'PhysxSurfaceDeformableBodyAPI'):
        required_api(p, schema)
    set_attr(p, 'omniphysics:mass', float(spec['mass_kg']))
    set_attr(p, 'omniphysics:restShapePoints', Vt.Vec3fArray.FromNumpy(v))
    set_attr(p, 'omniphysics:restTriVtxIndices', Vt.Vec3iArray.FromNumpy(triangles))
    set_attr(p, 'physxDeformableBody:selfCollision', bool(spec.get('self_contact', True)))
    set_attr(p, 'physxDeformableBody:disableGravity', False)
    set_attr(p, 'physxDeformableBody:solverPositionIterationCount', int(options.get('cloth_iterations', 32)))
    set_attr(p, 'physxDeformableBody:linearDamping', float(options.get('cloth_linear_damping', .10)))
    set_attr(p, 'physxDeformableBody:maxLinearVelocity', -1.)
    mat = UsdShade.Material.Define(stage, '/World/PhysicsMaterials/cloth')
    mp = mat.GetPrim()
    for schema in ('OmniPhysicsBaseMaterialAPI', 'OmniPhysicsDeformableMaterialAPI',
                   'OmniPhysicsSurfaceDeformableMaterialAPI', 'PhysxSurfaceDeformableMaterialAPI'):
        required_api(mp, schema)
    thickness = float(options.get('cloth_thickness_m', .001))
    young = float(options.get('cloth_youngs_modulus_pa', 2.e6))
    poisson = float(options.get('cloth_poissons_ratio', .3))
    bend = float(options.get('cloth_surface_bend_stiffness', 2.e4))
    properties = {'omniphysics:dynamicFriction': float(spec.get('friction', .45)),
                  'omniphysics:staticFriction': float(spec.get('friction', .45)),
                  'omniphysics:density': float(spec['mass_kg'] / (np.prod(spec['size_m']) * thickness)),
                  'omniphysics:youngsModulus': young, 'omniphysics:poissonsRatio': poisson,
                  'omniphysics:surfaceThickness': thickness, 'omniphysics:surfaceBendStiffness': bend,
                  'physxDeformableMaterial:elasticityDamping': .005,
                  'physxDeformableMaterial:bendDamping': .005}
    for key, value in properties.items():
        set_attr(mp, key, value)
    collision(p, mat, contact_offset=float(options.get('cloth_contact_offset_m', .003)),
              rest_offset=float(options.get('cloth_rest_offset_m', .0015)))
    # Optional contact-quality settings (NVIDIA surface demo uses 4/4 + CCD);
    # absent keys keep the PhysX defaults used by earlier releases.
    contact_quality = {}
    for key, attr, kind in (('cloth_collision_pair_update_frequency', 'physxDeformableBody:collisionPairUpdateFrequency', int),
                            ('cloth_collision_iteration_multiplier', 'physxDeformableBody:collisionIterationMultiplier', int),
                            ('cloth_speculative_ccd', 'physxDeformableBody:enableSpeculativeCCD', bool)):
        if key in options:
            set_attr(p, attr, kind(options[key]))
            contact_quality[attr] = kind(options[key])
    color = options.get('cloth_visual_color', [.82, .76, .62])
    bind_visual(p, visual_material(stage, 'cloth', color, .88))
    return {'path': path, 'triangles': triangles, 'rest_vertices': v,
            'material': properties, 'material_calibrated': False, 'api_reference': DEFORMABLE_DOC,
            'material_units': {'youngs_modulus': 'Pa', 'surface_thickness': 'm',
                               'density': 'kg/m^3', 'surface_bend_stiffness': 'Pa; PhysX multiplies by thickness^3',
                               'elasticity_damping': '1/s', 'bend_damping': '1/s', 'linear_damping': '1/s'},
            'total_mass_kg': float(spec['mass_kg']), 'visual_color': color,
            'material_limitations': ['Static friction is unsupported by PhysX deformables; dynamic friction is used.',
                                     'Old Newton VBD stretch/area/bend coefficients are not copied into PhysX material fields.'],
            'solver': 'PhysX surface deformable XPBD FEM', 'fixed_vertices': 0,
            'contact_quality': contact_quality,
            'vertex_order': 'x-fast; identical named vertex indices to source task'}


def add_soft_body(stage, spec, index):
    """Volume (tetrahedral FEM) soft body, e.g. a sponge. No attachments, no pinned nodes."""
    from .geometry import TET_PATTERNS, tet_surface_triangles
    if spec.get('shape', 'box') != 'box':
        raise ValueError('Only box soft bodies are supported')
    name = spec['id']
    if not name.replace('_', '').isalnum():
        raise ValueError(f'Invalid soft body id: {name!r}')
    pattern = spec.get('tet_pattern', 'kuhn_6')
    if pattern not in TET_PATTERNS:
        raise ValueError(f'Unknown soft body tet pattern: {pattern!r}')
    v, tets = TET_PATTERNS[pattern](spec['size_m'], spec['cells_xyz'])
    path = f'/World/SoftBodies/{name}'
    UsdGeom.Xform.Define(stage, '/World/SoftBodies')
    tet = UsdGeom.TetMesh.Define(stage, path)
    tet.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(v))
    tet.GetTetVertexIndicesAttr().Set(Vt.Vec4iArray.FromNumpy(tets))
    surface = tet_surface_triangles(tets)
    tet.GetSurfaceFaceVertexIndicesAttr().Set(Vt.Vec3iArray.FromNumpy(surface))
    p = tet.GetPrim()
    transform(p, spec['position_m'], spec.get('quaternion_wxyz', [1, 0, 0, 0]))
    for schema in ('OmniPhysicsDeformableBodyAPI', 'OmniPhysicsVolumeDeformableSimAPI', 'PhysxBaseDeformableBodyAPI'):
        required_api(p, schema)
    set_attr(p, 'omniphysics:restShapePoints', Vt.Vec3fArray.FromNumpy(v))
    set_attr(p, 'omniphysics:restTetVtxIndices', Vt.Vec4iArray.FromNumpy(tets))
    set_attr(p, 'omniphysics:mass', float(spec['mass_kg']))
    set_attr(p, 'physxDeformableBody:selfCollision', bool(spec.get('self_collision', False)))
    set_attr(p, 'physxDeformableBody:solverPositionIterationCount', int(spec.get('iterations', 32)))
    set_attr(p, 'physxDeformableBody:linearDamping', float(spec.get('linear_damping', .05)))
    # Optional settling of near-static vertices (PhysX default: 0.1 m/s, 10/s).
    for key, attr in (('settling_threshold_m_s', 'settlingThreshold'), ('settling_damping_per_s', 'settlingDamping')):
        if key in spec:
            set_attr(p, 'physxDeformableBody:' + attr, float(spec[key]))
    mat = UsdShade.Material.Define(stage, f'/World/PhysicsMaterials/soft_{name}')
    mp = mat.GetPrim()
    for schema in ('OmniPhysicsBaseMaterialAPI', 'OmniPhysicsDeformableMaterialAPI', 'PhysxDeformableMaterialAPI'):
        required_api(mp, schema)
    volume = float(np.prod(spec['size_m']))
    properties = {'omniphysics:dynamicFriction': float(spec.get('friction', .9)),
                  'omniphysics:staticFriction': float(spec.get('friction', .9)),
                  'omniphysics:density': float(spec['mass_kg']) / volume,
                  'omniphysics:youngsModulus': float(spec['youngs_modulus_pa']),
                  'omniphysics:poissonsRatio': float(spec.get('poissons_ratio', .25)),
                  'physxDeformableMaterial:elasticityDamping': float(spec.get('elasticity_damping', .005))}
    for key, value in properties.items():
        set_attr(mp, key, value)
    collision(p, mat, contact_offset=float(spec.get('contact_offset_m', .002)),
              rest_offset=float(spec.get('rest_offset_m', 0.)))
    color = spec.get('visual_color', [.93, .78, .25])
    bind_visual(p, visual_material(stage, f'soft_{name}', color, .9))
    return {'path': path, 'id': name, 'tets': tets, 'surface_triangles': surface, 'rest_vertices': v,
            'size_m': [float(x) for x in spec['size_m']], 'cells_xyz': [int(x) for x in spec['cells_xyz']],
            'tet_pattern': pattern,
            'material': properties, 'material_calibrated': False, 'total_mass_kg': float(spec['mass_kg']),
            'solver': 'PhysX volume deformable FEM (GPU)', 'fixed_nodes': 0, 'attachments': 0,
            'api_reference': DEFORMABLE_DOC,
            'material_limitations': ['Static friction is not modelled by PhysX deformables; dynamic friction is used.',
                                     'Uncalibrated foam parameters; no plasticity, absorption or pressure.']}


def add_workbench(stage, scene):
    mat = visual_material(stage, 'table_gray', [.34, .36, .38], .6, .12)
    dark = visual_material(stage, 'frame', [.09, .11, .13], .35, .7)
    physmat = rigid_material(stage, 'table', scene['table'].get('friction', .4))
    table = scene['table']
    box(stage, '/World/Table', table['position_m'], table['half_extents_m'], mat, physmat)
    center = np.asarray(table['position_m'])
    half = np.asarray(table['half_extents_m'])
    for ix in (-1, 1):
        for iy in (-1, 1):
            x, y = center[:2] + np.array([ix, iy]) * (half[:2] - .07)
            box(stage, f'/World/Workbench/leg_{ix+1}_{iy+1}', [x, y, .35], [.03, .03, .35], dark)
    # Flat base plate places FR3 base .77 m directly on the .75 m worktop.
    base = scene['robot'].get('base_position_m', [0, 0, .77])
    from .robot_registry import get_robot_spec
    arms = get_robot_spec(scene['robot']).get('arms')
    if arms:
        # Visual plates (no collider) under each arm of a multi-arm robot.
        from .robot_layout import world_base_poses
        for side, pose in world_base_poses(scene['robot']).items():
            x, y, z = pose['position_m']
            plate = box(stage, f'/World/Workbench/robot_mount_plate_{side}', [x, y, z - .01], [.11, .10, .01], dark)
            transform(plate, [x, y, z - .01], pose['quaternion_wxyz'], [.11, .10, .01])
    else:
        box(stage, '/World/Workbench/robot_mount_plate', [base[0], base[1], base[2] - .01], [.11, .10, .01], dark)
    ground = visual_material(stage, 'floor', [.14, .16, .19], .8)
    box(stage, '/World/Ground', [0, 0, -.03], [5, 5, .03], ground, physmat)
    # Offline exported original geometry preserves the richer visual workcell.
    assets = Path(scene['robot'].get('assets_root', f'{DATA_ROOT}/assets'))
    root = assets / 'scenes/workcell_details/isaacsim_6_1_v1'
    metadata_path = root / 'mesh_manifest.json'
    groups = []
    if scene.get('workcell_details', {}).get('enabled') and metadata_path.is_file():
        meta = load_json(metadata_path)
        arrays = np.load(root / 'meshes.npz')
        for i, item in enumerate(meta['meshes']):
            g = mesh(stage, '/World/Details/' + item['name'], arrays[f'v_{i}'], arrays[f'f_{i}'], arrays[f'n_{i}'])
            bind_visual(g.GetPrim(), visual_material(stage, 'detail_' + str(i), item['color'], item['roughness'], item['metallic']))
            groups.append(item['name'])
    dome = UsdLux.DomeLight.Define(stage, '/World/Lights/Ambient')
    dome.CreateIntensityAttr(700.)
    dome.CreateColorAttr(Gf.Vec3f(.82, .89, 1.))
    key = UsdLux.DistantLight.Define(stage, '/World/Lights/Key')
    key.CreateIntensityAttr(2200.)
    key.CreateAngleAttr(12.)
    UsdGeom.Xformable(key).AddRotateXYZOp().Set(Gf.Vec3f(-30, -35, 15))
    return {'gray_table': True, 'detail_groups': groups, 'decorative_collision': False}


def add_static_prop(stage, spec, materials, *, prefix='/World/Props'):
    """A solid, immovable prop: a primitive collider with no rigid body.

    Used for room furniture that must be touched (bench tops, vices, pegboard
    panel, cabinets). It has no rigid body, so it never moves, but it does
    produce solver contact. The visual surface is textured and may be cut; the
    collider is a primitive box, because PhysX's GPU contact filter supports
    primitives only and a mesh collider would drop out of the contact view.
    """
    from .room import _box_collider
    path = f'{prefix}/{spec["id"]}'
    root = UsdGeom.Xform.Define(stage, path).GetPrim()
    physmat = rigid_material(stage, 'prop_' + spec['id'], spec.get('friction', .7))
    body = []
    for index, part in enumerate(spec['parts']):
        child = f'{path}/{part["id"]}'
        if part.get('geometry', 'box') != 'box':
            raise ValueError(f'Static prop {spec["id"]} part {part["id"]} must be a box for now')
        half = [float(v) for v in part['half_extents_m']]
        center = [float(v) for v in part.get('position_m', [0., 0., 0.])]
        prim = uv_box(stage, child, center, half, scale=part.get('uv_scale', [1., 1.]))
        material = materials.get(part.get('material', spec.get('material', 'prop')))
        if material is not None:
            from .visuals import bind as bind_material
            bind_material(stage, prim, material)
        _box_collider(stage, child, center, half, physics_material=physmat)
        body.append(child)
    return {'id': spec['id'], 'path': path, 'parts': body, 'body_mode': 'immovable_static',
            'friction': float(spec.get('friction', .7))}


def add_structured_scene(stage, scene, room_spec):
    """Build a furnished, textured room around the robot: shell, pedestal, props.

    Kept separate from ``add_workbench`` so a scene without a ``room`` block
    keeps producing exactly the earlier stage.
    """
    from .room import add_pedestal, add_room, room_materials

    definitions = dict(scene.get('materials', {}))
    for name in list(room_spec['surfaces'].values()):
        definitions.setdefault(name, {'kind': 'color', 'color': [.55, .56, .58]})
    definitions.setdefault('prop', {'kind': 'color', 'color': [.42, .45, .48], 'roughness': .6})

    # Surfaces referenced by fixtures/static props may already exist as task
    # asset physics materials; the visual ones are built here.
    materials, material_specs = room_materials(stage, definitions, prefix='mat')
    floor_physmat = rigid_material(stage, 'room_floor', room_spec.get('floor_friction', .8))
    room_record = add_room(stage, room_spec, materials, physics_material=floor_physmat,
                           collide=room_spec.get('collide'))

    pedestal = None
    if scene.get('pedestal'):
        pedestal = add_pedestal(stage, scene['pedestal'], materials)
        top = pedestal['top_z_m']
        declared = float(scene.get('pedestal', {}).get('robot_base_z_m', top))
        if abs(declared - top) > 1e-6:
            raise ValueError(f'pedestal robot_base_z_m {declared} does not match its top {top}')
        base = scene['robot'].get('base_position_m', [0, 0, .77])
        if abs(float(base[2]) - top) > 1e-6:
            raise ValueError(f'robot base_position_m[2] {base[2]} must equal the pedestal top {top}')

    props = [add_static_prop(stage, spec, materials) for spec in scene.get('static_props', [])]
    solid = list(room_record['solid_paths'])
    solid += [p for spec in props for p in spec['parts']]

    # The work table: a separate solid bench in front of the robot. The robot
    # stands on its own pedestal, never on this table.
    table_record = None
    if scene.get('table'):
        table = scene['table']
        table_record = add_static_prop(stage, {'id': 'work_table', 'friction': table.get('friction', .5),
                                              'material': table.get('material', 'table'),
                                              'parts': [{'id': 'top', 'geometry': 'box',
                                                         'position_m': table['position_m'],
                                                         'half_extents_m': table['half_extents_m'],
                                                         'uv_scale': table.get('uv_scale', [1., 1.]),
                                                         'material': table.get('material', 'table')}]},
                                       materials)
        solid += table_record['parts']
    # Extra support plates (a low shelf, a machine bed, a tray floor) are solid.
    surfaces = []
    for spec in scene.get('surfaces', []) or []:
        record = add_static_prop(stage, {'id': spec['id'], 'friction': spec.get('friction', .7),
                                         'material': spec.get('material', 'prop'),
                                         'parts': [{'id': 'plate', 'geometry': 'box',
                                                    'position_m': spec['position_m'],
                                                    'half_extents_m': spec['half_extents_m'],
                                                    'material': spec.get('material', 'prop')}]},
                                 materials)
        surfaces.append(record)
        solid += record['parts']
    if pedestal:
        solid += [p['path'] for p in pedestal['parts']]

    return {'kind': 'structured_room', 'room': room_record, 'pedestal': pedestal,
            'table': table_record, 'surfaces': surfaces, 'static_props': props,
            'solid_paths': solid, 'materials': material_specs, 'decorative_collision': False}


def build_scene(stage, scene, robot_usd, *, physics_scene_path='/World/PhysicsScene'):
    if ('arm_base_poses' in scene['robot']
            and scene.get('isaacsim', {}).get('robot_base_constraint') != 'native_fixed_base_v1'):
        raise ValueError('An explicit dual arm layout requires native_fixed_base_v1')
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.)
    world = UsdGeom.Xform.Define(stage, '/World')
    stage.SetDefaultPrim(world.GetPrim())
    robot = UsdGeom.Xform.Define(stage, '/World/Robot').GetPrim()
    from .robot_layout import reference_robot
    reference_robot(stage, robot, robot_usd, scene['robot'])
    from .robot_base import apply_base_constraint
    base_constraint = apply_base_constraint(stage, robot, scene)
    arm_root = None
    arm_roots = []
    joint_paths = {}
    collision_config = collision_configuration(scene)
    link_paths = {}
    for p in Usd.PrimRange(robot):
        if p.HasAPI(UsdPhysics.ArticulationRootAPI):
            arm_root = str(p.GetPath())
            arm_roots.append(arm_root)
            api = PhysxSchema.PhysxArticulationAPI.Apply(p)
            api.CreateEnabledSelfCollisionsAttr(collision_config['enabled'])
            api.CreateSolverPositionIterationCountAttr(16)
            api.CreateSolverVelocityIterationCountAttr(4)
        if p.IsA(UsdPhysics.RevoluteJoint):
            joint_paths[p.GetName()] = str(p.GetPath())
            drive = UsdPhysics.DriveAPI.Apply(p, 'angular')
            drive.CreateTypeAttr('force')
        if p.HasAPI(UsdPhysics.RigidBodyAPI):
            link_paths[p.GetName()] = str(p.GetPath())
            PhysxSchema.PhysxContactReportAPI.Apply(p).CreateThresholdAttr(0.)
    from .robot_registry import get_robot_spec
    expected = get_robot_spec(scene['robot'])['joint_count']
    if not arm_root or len(joint_paths) != expected or len(arm_roots) != (len(robot_usd) if isinstance(robot_usd, dict) else 1):
        raise RuntimeError(f'Expected FR3+hand articulation(s) with {expected} physical joints: {arm_roots}, {len(joint_paths)}')
    hand_spec = get_robot_spec(scene['robot'])
    if hand_spec.get('hand_armature_kg_m2', 0.):
        for name, path in joint_paths.items():
            if not name.startswith('fr3_'):
                joint_prim = stage.GetPrimAtPath(path)
                axis = 'rot' + str(UsdPhysics.RevoluteJoint(joint_prim).GetAxisAttr().Get())
                if not joint_prim.ApplyAPI('PhysxJointAxisAPI', axis):
                    raise RuntimeError('Missing native per-axis armature schema')
                joint_prim.CreateAttribute(f'physxJointAxis:{axis}:armature', Sdf.ValueTypeNames.Float).Set(hand_spec['hand_armature_kg_m2'])
    mimic_records = []
    for follower, config in hand_spec['mimic_joints'].items():
        prim = stage.GetPrimAtPath(joint_paths[follower])
        prim.RemoveAPI('NewtonMimicAPI')
        axis = 'rot' + str(UsdPhysics.RevoluteJoint(prim).GetAxisAttr().Get())
        native = PhysxSchema.PhysxMimicJointAPI.Apply(prim, axis)
        native.CreateReferenceJointRel().SetTargets([Sdf.Path(joint_paths[config['joint']])])
        reference_prim = stage.GetPrimAtPath(joint_paths[config['joint']])
        reference_axis = 'rot' + str(UsdPhysics.RevoluteJoint(reference_prim).GetAxisAttr().Get())
        native.CreateReferenceJointAxisAttr(reference_axis)
        native.CreateGearingAttr(-float(config['multiplier']))
        native.CreateOffsetAttr(-float(config['offset']))
        mimic_records.append({'follower': follower, **config, 'api': 'PhysxMimicJointAPI', 'axis': axis, 'reference_axis': reference_axis, 'simulation_approximation': '1:1 distal coupling; no tendon differential'})
    robot_materials = None
    materials = scene.get('isaacsim', {}).get('robot_contact_materials')
    material_choice = material_options(scene.get('isaacsim', {}))
    if material_choice['binding'] is not None and materials is None:
        raise ValueError('robot_material_binding needs robot_contact_materials')
    if materials is not None:
        # Explicit, recorded robot friction: elastomer pads vs every other link.
        # Without this option robot colliders keep the unrecorded PhysX default.
        if set(materials) != {'pad_friction', 'shell_friction'}:
            raise ValueError('robot_contact_materials needs pad_friction and shell_friction')
        pad = rigid_material(stage, 'robot_pad', materials['pad_friction'])
        shell = rigid_material(stage, 'robot_shell', materials['shell_friction'])
        bound = {'robot_pad': [], 'robot_shell': []}
        if material_choice['binding'] == 'collider_instance_v1':
            holders = bind_collider_instances(stage, '/World/Robot', set(link_paths), pad, shell)
            for link in link_paths:
                bound[material_kind(link)].append(link)
            binding = {'mode': 'collider_instance_v1', 'bound_instance_prims': len(holders),
                       'pad_instance_prims': sum(kind == 'robot_pad' for kind in holders.values())}
        else:
            # Legacy (no opt-in): the links are nested, so the root-most link's binding
            # wins and every collider resolves to robot_shell (resolved_collider_materials).
            for link, path in link_paths.items():
                name = 'robot_pad' if link.endswith('_elastomer') else 'robot_shell'
                UsdShade.MaterialBindingAPI.Apply(stage.GetPrimAtPath(path)).Bind(
                    pad if name == 'robot_pad' else shell, UsdShade.Tokens.strongerThanDescendants, 'physics')
                bound[name].append(link)
            binding = {'mode': 'link_stronger_than_descendants_legacy'}
        if material_choice['pad_compliance'] is not None:
            apply_pad_compliance(pad, material_choice['pad_compliance'])
        robot_materials = {'values': dict(materials), 'combine': 'average', 'restitution': 0.,
                           'bound_links': bound, 'binding': binding,
                           'resolved_collider_materials': resolved_collider_materials(stage, '/World/Robot'),
                           'pad_compliance': material_choice['pad_compliance']}
    authored_pairs = []
    for first, second in collision_config['excluded_link_pairs']:
        if first not in link_paths or second not in link_paths:
            raise RuntimeError('Imported robot lacks the rigid links of a validated structural exclusion')
        first_path, second_path = link_paths[first], link_paths[second]
        UsdPhysics.FilteredPairsAPI.Apply(stage.GetPrimAtPath(first_path)).CreateFilteredPairsRel().AddTarget(Sdf.Path(second_path))
        authored_pairs.append([first_path, second_path])
    collision_config['authored_filtered_body_paths'] = authored_pairs
    # A scene with a room block builds a furnished, textured room; without one
    # the legacy grey workbench is produced and the stage is unchanged.
    from .room import validate_room
    room_spec = validate_room(scene.get('room'))
    workroom_materials = scene.get('isaacsim', {}).get('workroom_surface_materials')
    workroom_lighting = scene.get('isaacsim', {}).get('workroom_lighting')
    lighting_record = None
    if workroom_materials is not None and 'workroom' not in scene:
        raise ValueError('workroom_surface_materials requires a workroom reference')
    if workroom_lighting is not None and 'workroom' not in scene:
        raise ValueError('workroom_lighting requires a workroom reference')
    if 'workroom' in scene:
        if room_spec['enabled']:
            raise ValueError('workroom reference and procedural room cannot be enabled together')
        from .workroom import add_workroom, apply_runtime_lighting
        workbench = add_workroom(stage, scene['workroom'], surface_materials=workroom_materials)
        if scene.get('workroom_presentation') is not None:
            from .presentation import apply_presentation
            workbench['presentation'] = apply_presentation(stage, workbench['path'],
                {'presentation': scene['workroom_presentation']})
        lighting_record = apply_runtime_lighting(stage, scene['workroom'], workroom_lighting)
        table_paths = list(workbench['contact_filter_paths'])
    elif room_spec['enabled']:
        workbench = add_structured_scene(stage, scene, room_spec)
        table_paths = list(workbench['solid_paths'])
    else:
        workbench = add_workbench(stage, scene)
        table_paths = ['/World/Table', '/World/Ground']
    goal_markers = []
    for marker in scene.get('goal_markers', []):
        # Public task cues are visual geometry only, never collision supports.
        marker_path = '/World/GoalMarkers/' + marker['id']
        marker_material = visual_material(stage, 'goal_' + marker['id'], marker.get('color', [.12, .65, .28]), .8)
        box(stage, marker_path, marker['position_m'], marker['half_extents_m'], marker_material)
        goal_markers.append({'path': marker_path, 'collision_enabled': False})
    objects = {x['id']: add_task_asset(stage, x, dynamic=True) for x in scene['objects']}
    fixtures = {x['id']: add_task_asset(stage, x, dynamic=False) for x in scene.get('fixtures', [])}
    from .task_layers import mount_task_layers
    task_layers = mount_task_layers(stage, scene.get('task_layers', []),
        physics_scene_path=physics_scene_path, occupied_objects=objects)
    for item in task_layers['objects'].values():
        PhysxSchema.PhysxContactReportAPI.Apply(stage.GetPrimAtPath(item['path'])).CreateThresholdAttr(0.)
    objects.update(task_layers['objects'])
    table_paths.extend(task_layers['static_collision_paths'])
    from .native_task_bundle import add_bundle
    physics_scenes = [str(p.GetPath()) for p in stage.Traverse() if p.IsA(UsdPhysics.Scene)]
    bundle = add_bundle(stage, scene.get('native_task_bundle'),
                        physics_scene_path=physics_scenes[0] if len(physics_scenes) == 1 else '')
    if bundle is not None:
        if len(physics_scenes) != 1:
            raise ValueError('A native task bundle requires exactly one bench PhysicsScene')
        if set(objects) & set(bundle['objects']) or set(fixtures) & set(bundle['fixtures']):
            raise ValueError('Native task bundle identifiers conflict with scene objects')
        objects.update(bundle['objects'])
        fixtures.update(bundle['fixtures'])
    cloth = add_cloth(stage, scene['cloth'], scene.get('isaacsim', {})) if scene.get('cloth', {}).get('enabled') else None
    soft_bodies = {}
    for index, spec in enumerate(scene.get('soft_bodies', [])):
        if spec['id'] in soft_bodies:
            raise ValueError(f'Duplicate soft body id: {spec["id"]}')
        soft_bodies[spec['id']] = add_soft_body(stage, spec, index)
    from .static_mesh_contacts import adapt_static_meshes
    fixed_contacts = adapt_static_meshes(stage, scene.get('isaacsim', {}).get('static_mesh_contact_mode'),
        exclude_paths=['/World/Robot', '/World/Cloth'] + [info['path'] for info in soft_bodies.values()])
    extra = {}
    if fixed_contacts is not None:
        mapping = fixed_contacts['collider_actor_paths']
        table_paths = sorted({mapping.get(path, path) for path in table_paths}
                             | set(fixed_contacts['kinematic_body_paths']))
        for info in fixtures.values():
            info['contact_filter_paths'] = sorted({mapping.get(path, path) for path in
                info.get('contact_filter_paths', info['collision_paths'])})
        extra = {key: fixed_contacts[key] for key in ('kinematic_body_paths', 'collider_actor_paths')}
        extra['static_mesh_contacts'] = fixed_contacts
        workbench.update(extra)
        if workbench.get('kind') == 'referenced_workroom':
            workbench['contact_filter_paths'] = table_paths
            workbench['contact_unobserved_paths'] = sorted(set(workbench['contact_unobserved_paths']) - set(mapping))
            workbench['contact_scope'] = 'Primitive static filters and opt-in fixed kinematic actors for original collision meshes.'
    return {**extra, **({'workroom_lighting': lighting_record} if lighting_record is not None else {}),
            **({'native_task_bundle': bundle} if bundle is not None else {}),
            **({'robot_base_constraint': base_constraint} if base_constraint is not None else {}),
            'robot_path': '/World/Robot', 'articulation_path': arm_roots if isinstance(robot_usd, dict) else arm_root,
            'joint_paths': joint_paths,
            'objects': objects, 'fixtures': fixtures, 'cloth': cloth, 'workbench': workbench, 'goal_markers': goal_markers,
            **({'task_layers': task_layers['layers']} if task_layers['layers'] else {}),
            'robot_self_collision': collision_config, 'mimic_constraints': mimic_records, 'robot_link_paths': link_paths,
            'robot_contact_materials': robot_materials, 'soft_bodies': soft_bodies,
            'support_paths': table_paths}
