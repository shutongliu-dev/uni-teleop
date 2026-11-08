"""Original bonded volume-FEM sponge candidate, authored without Isaac/GPU.

The yellow material is a native tetrahedral deformable, permanently bonded to
the green rigid scouring layer. The bond is a material interface, never a hand
grasp. Parameters are uncalibrated priors; the scouring face cannot bend and
neither layer models water, plasticity or pore-scale air flow.
"""
from __future__ import annotations

import numpy as np
from pxr import Gf, Sdf, UsdGeom, UsdPhysics, UsdShade, Vt

from . import geometry as g

PROFILE = 'bonded_fem_v1'
COARSE_PROFILE = 'bonded_fem_coarse_v1'
PROFILES = (PROFILE, COARSE_PROFILE)
TOTAL_MASS_KG = .0792
SCOURING_MASS_KG = .00792
FOAM_MASS_KG = TOTAL_MASS_KG - SCOURING_MASS_KG
MATERIAL = {'youngs_modulus_pa': 10000., 'poissons_ratio': .25,
            'dynamic_friction': .35, 'elasticity_damping': .005,
            'linear_damping': .05, 'contact_offset_m': .0002,
            'rest_offset_m': 0., 'solver_position_iterations': 32,
            'calibrated': False}
LIMITATIONS = [
    'Uniform dry FEM foam with uncalibrated elastic parameters; not measured sponge material.',
    'The 4 mm green scouring layer is rigid and cannot bend.',
    'Permanent two-way material bond; no robot attachment or runtime node targets.',
    'No absorption, fluid, air pressure, fracture or plasticity.',
    'PhysX deformable contact impulses are unavailable; proximity is not contact force.',
    'Only deformable dynamic friction is effective; no static-friction claim.',
]


def _signed_tet_volumes(points, tets):
    p = points[tets]
    return np.einsum('ij,ij->i', np.cross(p[:, 1]-p[:, 0], p[:, 2]-p[:, 0]),
                     p[:, 3]-p[:, 0]) / 6.


def foam_topology(profile=PROFILE):
    """Freeze a regular conforming Kuhn grid without thin radial spokes.

    The original detailed visual skin remains unchanged. Collision corners
    approximate each 3.6 mm radius quarter-circle by two chords, inset by at
    most r*(1-cos(pi/8)) = 0.2741 mm. This small explicit physical-boundary
    approximation avoids carrying submillimetre visual curvature into badly
    conditioned tetrahedra. All long faces, support planes and taper heights
    retain their metric coordinates.
    """
    from bench.simulation.isaacsim_backend.geometry import box_tet_grid, tet_surface_triangles
    if profile not in PROFILES:
        raise ValueError('Unknown bonded sponge discretization: '+str(profile))
    # The opt-in comparison halves the simultaneous rigid/vertex constraints.
    # Keep every vertical layer and all physical boundary coordinates. This
    # does not improve the worst corner cell or establish native convergence.
    nx, ny = (14, 9) if profile == COARSE_PROFILE else (20, 13)
    z_layers = np.array([.004, .010, .016, .022, .027, .032, .038, .0415, .045])
    _, tets = box_tet_grid([.120, .078, .041], [nx, ny, len(z_layers)-1])
    points = []
    for z in z_layers:
        hx = np.interp(z, [.004, .022, .038, .045], [.060, .060, .0396, .0372])
        hy = np.interp(z, [.004, .022, .038, .045], [.039, .039, .024, .0228])
        radius = .0036
        xs = np.r_[-hx, np.linspace(-hx+radius, hx-radius, nx-1), hx]
        ys = np.r_[-hy, np.linspace(-hy+radius, hy-radius, ny-1), hy]
        for j, yy in enumerate(ys):
            for i, xx in enumerate(xs):
                x, y = xx, yy
                if i in (0, nx) and j in (0, ny):
                    x = np.sign(x)*(hx-radius+radius/np.sqrt(2))
                    y = np.sign(y)*(hy-radius+radius/np.sqrt(2))
                points.append((x, y, z))
    points = np.asarray(points, dtype=np.float32)
    volume = _signed_tet_volumes(points.astype(float), tets)
    # PhysX updateNodeInverseVolumes rejects determinant <= 1e-9 (metre units),
    # even if it is positive. Keep >4x margin after float32 serialization.
    if np.any(volume*6 <= 4e-9):
        raise ValueError('Sponge tetrahedron below the native PhysX determinant margin')
    surface = tet_surface_triangles(tets)
    return points, tets, surface, np.arange((nx+1)*(ny+1), dtype=np.int32)


def topology_quality(points, tets):
    """Geometry-only conditioning diagnostics; not a native stability result."""
    p = np.asarray(points, float)[np.asarray(tets)]
    matrix = (p[:, 1:]-p[:, :1]).transpose(0, 2, 1)
    singular = np.linalg.svd(matrix, compute_uv=False)
    volume = _signed_tet_volumes(np.asarray(points, float), np.asarray(tets))
    squared_edges = sum(np.sum((p[:, a]-p[:, b])**2, axis=1) for a in range(4) for b in range(a))
    mean_ratio = 12*(3*volume)**(2/3)/squared_edges
    lumped = np.zeros(len(points))
    for column in range(4):
        np.add.at(lumped, np.asarray(tets)[:, column], volume/4)
    return {'minimum_determinant_m3': float((volume*6).min()),
        'maximum_rest_matrix_condition_number': float((singular[:, 0]/singular[:, -1]).max()),
        'minimum_tetrahedron_mean_ratio': float(mean_ratio.min()),
        'geometric_lumped_node_volume_max_min_ratio': float(lumped.max()/lumped.min()),
        'hypothetical_mass_ratio_if_inverse_mass_clamped_at_50': float(np.maximum(lumped, lumped.max()/50).sum()/lumped.sum()),
        'mass_clamp_scope': 'CPU sensitivity calculation only; does not assert the native solver applies this clamp.'}


def _bind_pose(prim, points):
    g.schema(prim, 'OmniPhysicsDeformablePoseAPI:default')
    g.attribute(prim, 'deformablePose:default:omniphysics:purposes', Sdf.ValueTypeNames.TokenArray, ['bindPose'])
    g.attribute(prim, 'deformablePose:default:omniphysics:points', Sdf.ValueTypeNames.Point3fArray,
                Vt.Vec3fArray.FromNumpy(np.asarray(points, dtype=np.float32)))


def apply_bonded_fem_profile(author, profile=PROFILE):
    """Replace an already household-sized sponge and return its volume record."""
    stage, meta = author.stage, author.meta
    if meta['task_id'] != 'surface_wiping' or meta.get('prop_profile') != 'household_v1':
        raise ValueError('Bonded FEM sponge requires household_v1 surface_wiping')
    if profile not in PROFILES:
        raise ValueError('Unknown bonded sponge discretization: '+str(profile))
    row = next(r for r in meta['objects'] if r['id'] == 'sponge')
    body_path = row['body_path']
    # Preserve all prior pore geometry and pigmentation, but move the foam skin
    # under its deformable owner so the solver, not the rigid base, drives it.
    skin_old = body_path+'/Visual/finish_porous_foam'
    green = body_path+'/Visual/finish_green_scouring_layer'
    if not stage.GetPrimAtPath(skin_old) or not stage.GetPrimAtPath(green):
        raise ValueError('FEM profile requires the detailed household sponge skin')
    root_path = author.root+'/VolumeBodies/foam'
    root = UsdGeom.Xform.Define(stage, root_path)
    pose = row['initial_pose_xyzw']
    root.AddTranslateOp().Set(Gf.Vec3d(*pose[:3]))
    root.AddOrientOp().Set(Gf.Quatf(pose[6], Gf.Vec3f(*pose[3:6])))
    for api in ('OmniPhysicsDeformableBodyAPI', 'PhysxBaseDeformableBodyAPI'):
        g.schema(root.GetPrim(), api)
    for name, kind, value in (
        ('omniphysics:deformableBodyEnabled', Sdf.ValueTypeNames.Bool, True),
        ('omniphysics:kinematicEnabled', Sdf.ValueTypeNames.Bool, False),
        ('omniphysics:startsAsleep', Sdf.ValueTypeNames.Bool, False),
        ('omniphysics:mass', Sdf.ValueTypeNames.Float, FOAM_MASS_KG),
        ('physxDeformableBody:selfCollision', Sdf.ValueTypeNames.Bool, False),
        ('physxDeformableBody:solverPositionIterationCount', Sdf.ValueTypeNames.Int, MATERIAL['solver_position_iterations']),
        ('physxDeformableBody:linearDamping', Sdf.ValueTypeNames.Float, MATERIAL['linear_damping']),
    ):
        g.attribute(root.GetPrim(), name, kind, value)
    points, tets, surface, bonded = foam_topology(profile)
    sim_path = root_path+'/SimulationMesh'
    sim = UsdGeom.TetMesh.Define(stage, sim_path)
    sim.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(points))
    sim.CreateTetVertexIndicesAttr(Vt.Vec4iArray.FromNumpy(tets))
    sim.CreateSurfaceFaceVertexIndicesAttr(Vt.Vec3iArray.FromNumpy(surface))
    sim.CreateExtentAttr([Gf.Vec3f(*points.min(0).tolist()), Gf.Vec3f(*points.max(0).tolist())])
    sim.CreatePurposeAttr('guide')
    g.schema(sim.GetPrim(), 'OmniPhysicsVolumeDeformableSimAPI')
    g.attribute(sim.GetPrim(), 'omniphysics:restShapePoints', Sdf.ValueTypeNames.Point3fArray,
                Vt.Vec3fArray.FromNumpy(points))
    g.attribute(sim.GetPrim(), 'omniphysics:restTetVtxIndices', Sdf.ValueTypeNames.Int4Array,
                Vt.Vec4iArray.FromNumpy(tets))
    _bind_pose(sim.GetPrim(), points)
    UsdPhysics.CollisionAPI.Apply(sim.GetPrim()).CreateCollisionEnabledAttr(True)
    g.schema(sim.GetPrim(), 'PhysxCollisionAPI')
    for name in ('contactOffset', 'restOffset'):
        key = 'contact_offset_m' if name == 'contactOffset' else 'rest_offset_m'
        g.attribute(sim.GetPrim(), 'physxCollision:'+name, Sdf.ValueTypeNames.Float, MATERIAL[key])
    physics_mat = UsdShade.Material.Define(stage, author.root+'/Materials/FoamFEM')
    for api in ('OmniPhysicsBaseMaterialAPI', 'OmniPhysicsDeformableMaterialAPI', 'PhysxDeformableMaterialAPI'):
        g.schema(physics_mat.GetPrim(), api)
    volume = float(_signed_tet_volumes(points.astype(float), tets).sum())
    for name, value in {
        'omniphysics:density': FOAM_MASS_KG/volume,
        'omniphysics:youngsModulus': MATERIAL['youngs_modulus_pa'],
        'omniphysics:poissonsRatio': MATERIAL['poissons_ratio'],
        'omniphysics:dynamicFriction': MATERIAL['dynamic_friction'],
        'physxDeformableMaterial:elasticityDamping': MATERIAL['elasticity_damping'],
    }.items():
        g.attribute(physics_mat.GetPrim(), name, Sdf.ValueTypeNames.Float, value)
    for prim in (root.GetPrim(), sim.GetPrim()):
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(physics_mat, materialPurpose='physics')
    UsdGeom.Xform.Define(stage, root_path+'/Visual')
    skin_path = root_path+'/Visual/porous_foam'
    if not Sdf.CopySpec(stage.GetRootLayer(), skin_old, stage.GetRootLayer(), skin_path):
        raise ValueError('Could not transfer original foam skin')
    stage.RemovePrim(skin_old)
    skin = UsdGeom.Mesh(stage.GetPrimAtPath(skin_path))
    _bind_pose(skin.GetPrim(), np.asarray(skin.GetPointsAttr().Get()))

    # The original sponge actor now represents only the green contact layer.
    stage.RemovePrim(body_path+'/Collision')
    base = g.rounded_loft([(0, 0, 0, 0), (.060, .039, .0036, 0),
                          (.060, .039, .0036, .004), (0, 0, 0, .004)])
    collider_path = body_path+'/Collision/scouring'
    collider = g.write_mesh(stage, collider_path, base, author.materials['leaf'], collision='convexHull')
    UsdGeom.Imageable(collider).CreateVisibilityAttr('invisible')
    UsdShade.MaterialBindingAPI(collider).Bind(author.physical_material, materialPurpose='physics')
    props = g.mass_properties([base], SCOURING_MASS_KG)
    mass_api = UsdPhysics.MassAPI(stage.GetPrimAtPath(body_path))
    mass_api.GetMassAttr().Set(SCOURING_MASS_KG)
    mass_api.GetCenterOfMassAttr().Set(Gf.Vec3f(*props['center_of_mass_m']))
    mass_api.GetDiagonalInertiaAttr().Set(Gf.Vec3f(*props['diagonal_inertia_kg_m2']))
    q = props['principal_axes_xyzw']
    mass_api.GetPrincipalAxesAttr().Set(Gf.Quatf(q[3], Gf.Vec3f(*q[:3])))
    row.update(props, mass_kg=SCOURING_MASS_KG, mass=SCOURING_MASS_KG,
        role='rigid_scouring_layer_of_bonded_sponge', collider_paths=[collider_path], collision_count=1,
        collision_bounds_m=base.bounds.tolist(), collision_support_points_m=base.convex_hull.vertices.tolist(),
        local_bounds_m=base.bounds.tolist(), composite_mass_kg=TOTAL_MASS_KG,
        physical_approximation='Native FEM yellow foam bonded to a rigid 4 mm green scouring layer; no fluid model.',
        public_description='Yellow deformable foam sponge with a bonded green flat scouring underside')
    row['dimension_profile'] = {
        'name': 'household_v1', 'linear_scale_xyz': [1.2, 1.2, 1.],
        'composite_source_mass_kg': .055, 'composite_mass_kg': TOTAL_MASS_KG,
        'component_mass_kg': SCOURING_MASS_KG,
        'mass_rule': 'Household composite mass partitioned 10% rigid scouring / 90% FEM foam; neither fraction is a measured density.'}
    meta['dimension_changes']['sponge']['scope'] = 'Composite envelope before partition into separately simulated rigid and FEM components'

    attachment_path = author.root+'/Attachments/foam_to_sponge'
    bond = stage.DefinePrim(attachment_path, 'OmniPhysicsVtxXformAttachment')
    bond.CreateRelationship('omniphysics:src0').SetTargets([sim_path])
    bond.CreateRelationship('omniphysics:src1').SetTargets([body_path])
    g.attribute(bond, 'omniphysics:attachmentEnabled', Sdf.ValueTypeNames.Bool, True)
    g.attribute(bond, 'omniphysics:vtxIndicesSrc0', Sdf.ValueTypeNames.IntArray, bonded.tolist())
    g.attribute(bond, 'omniphysics:localPositionsSrc1', Sdf.ValueTypeNames.Point3fArray,
                Vt.Vec3fArray.FromNumpy(points[bonded]))
    # Remove only bonded interface triangles and their shared-node neighbours
    # from collisions with the attached base. Other foam/table/hand pairs remain.
    filtered_faces = np.flatnonzero(np.isin(surface, bonded).any(axis=1))
    filter_path = author.root+'/Attachments/foam_scouring_interface_filter'
    filt = stage.DefinePrim(filter_path, 'OmniPhysicsElementCollisionFilter')
    filt.CreateRelationship('omniphysics:src0').SetTargets([sim_path])
    filt.CreateRelationship('omniphysics:src1').SetTargets([collider_path])
    g.attribute(filt, 'omniphysics:filterEnabled', Sdf.ValueTypeNames.Bool, True)
    for name, value in {'groupElemCounts0': [len(filtered_faces)], 'groupElemIndices0': filtered_faces.tolist(),
                        'groupElemCounts1': [], 'groupElemIndices1': []}.items():
        g.attribute(filt, 'omniphysics:'+name, Sdf.ValueTypeNames.UIntArray, value)
    record = {'prim_path': root_path, 'simulation_mesh_path': sim_path, 'collision_mesh_path': sim_path,
        'parent_rigid_body': 'sponge', 'rest_points_m': points.tolist(), 'tetrahedra': tets.tolist(),
        'surface_triangles': surface.tolist(), 'total_mass_kg': FOAM_MASS_KG,
        'initial_pose_xyzw': list(pose), 'rest_volume_m3': volume,
        'topology_profile': ('regular_kuhn_chord_corners_14x9_v1' if profile == COARSE_PROFILE
                             else 'regular_kuhn_chord_corners_v3'),
        'minimum_rest_tetrahedron_determinant_m3': float((_signed_tet_volumes(points.astype(float), tets)*6).min()),
        'topology_quality': topology_quality(points, tets),
        'physical_boundary_change': {
            'corner_approximation': 'Two chords per 3.6 mm radius quarter-circle; original detailed skin remains unchanged.',
            'maximum_analytic_cross_section_inset_m': float(.0036*(1-np.cos(np.pi/8))),
            'physical_outer_dimensions_m': [.120, .078, .041],
            'support_planes_and_taper_heights_preserved': True},
        'attachment': {'prim_path': attachment_path, 'node_indices': bonded.tolist(),
            'rigid_local_positions_m': points[bonded].tolist(), 'type': 'OmniPhysicsVtxXformAttachment',
            'two_way': True, 'meaning': 'Permanent material-layer bond; never a robot grasp'},
        'interface_collision_filter': {'prim_path': filter_path, 'surface_triangle_indices': filtered_faces.tolist(),
            'rigid_collider_path': collider_path},
        'visual_bindings': [{'prim_path': skin_path, 'simulation_mesh_path': sim_path,
            'mechanism': 'Native OmniPhysicsDeformableBodyAPI hierarchy plus explicit default bindPose',
            'runtime_deformation_verified': False}],
        'material': {**MATERIAL, 'material_path': str(physics_mat.GetPath()), 'density_kg_m3': FOAM_MASS_KG/volume},
        'provenance': {'kind': 'original_procedural', 'license': 'CC0-1.0', 'authoring_module': __name__,
            'runtime': 'Isaac Sim 6.1.0 / omni.physx 110.3.2+110.3.0',
            'reference': 'https://docs.omniverse.nvidia.com/kit/docs/omni_physics/110.1/dev_guide/deformables/deformable_authoring.html'},
        'limitations': list(LIMITATIONS), 'physics_accepted': False}
    if profile == COARSE_PROFILE:
        record['discretization_comparison'] = {
            'candidate_only': True, 'baseline_profile': PROFILE,
            'grid_cells_xy': [14, 9], 'baseline_grid_cells_xy': [20, 13],
            'z_layers_m': np.unique(points[:, 2]).astype(float).tolist(),
            'node_count': len(points), 'tet_count': len(tets),
            'attachment_node_count': len(bonded), 'baseline_attachment_node_count': 294,
            'hypothesis': 'Reduce simultaneous rigid-vertex constraints at unchanged physical parameters; no convergence result is assumed.',
            'physical_boundary': 'Original nine vertical planes, piecewise taper, two-chord corners and support planes retained.',
            'conditioning_limit': 'Worst corner-cell quality is unchanged; unclamped node mass ratio increases. Read topology_quality and its explicit clamp50 sensitivity, not a native mass measurement.',
            'physical_parameters_unchanged': True, 'native_validation_complete': False}
    meta['volume_bodies'] = {'foam': record}
    meta['sponge_physics_profile'] = profile
    meta['composite_objects'] = {'sponge': {'rigid_body': 'sponge', 'volume_bodies': ['foam'],
        'total_mass_kg': TOTAL_MASS_KG, 'rest_local_bounds_m': [[-.06, -.039, 0.], [.06, .039, .045]],
        'release_requires_all_components': True}}
    meta['source']['cleaning_approximation'] = 'Bonded native FEM foam and rigid scouring layer; actual loaded rigid scouring contact and travel clear finite soil cells.'
    meta['visual_finish']['sponge_approximation'] = 'Native deformable porous foam skin; rigid green scouring layer. No absorption.'
    meta['physics_prior']['sponge_volume_material'] = dict(record['material'])
    return record
