"""Read-only finite native SDF witnesses for unchanged closed L3 annular parts."""
from pathlib import Path
import hashlib
import numpy as np

from scripts.validation.mechanical_native_geometry import sdf_distance_column, source_surface_errors


def surface_witnesses(mesh):
    """Cover every original triangle at interior and near-corner locations."""
    weights = np.array([[1/3]*3, [2/3,1/6,1/6], [1/6,2/3,1/6], [1/6,1/6,2/3],
                        [.9,.05,.05], [.05,.9,.05], [.05,.05,.9]])
    points = np.einsum('wa,tac->twc', weights, mesh.triangles).reshape(-1,3)
    normals = np.repeat(mesh.face_normals, len(weights), axis=0)
    if not mesh.is_watertight or not mesh.is_winding_consistent or mesh.volume <= 0:
        raise ValueError('Closed outward-wound source mesh required')
    return points, normals


def convention_and_bore_witnesses(mesh):
    """Derive independent sign probes from each part's actual annular bounds."""
    zlo, zhi = mesh.bounds[:,2]
    radii = np.linalg.norm(mesh.vertices[:,:2],axis=1)
    inner, outer = radii.min(), radii.max()
    bottom_inner = radii[np.isclose(mesh.vertices[:,2],zlo,rtol=0,atol=1e-10)].min()
    middle, zmid, eps = (bottom_inner+outer)/2, (zlo+zhi)/2, .0005
    witnesses = np.array([[middle,0,zhi+eps], [middle,0,zhi-eps],
        [middle,0,zlo-eps], [middle,0,zlo+eps], [outer+eps,0,zmid],
        [outer-eps,0,zmid], [inner-eps,0,zmid], [inner+eps,0,zmid]])
    signs = np.array([1,-1,1,-1,1,-1,1,-1])
    bore = np.array([[r*np.cos(a),r*np.sin(a),z] for r in (0., inner*.5, inner*.875)
        for a in np.arange(8)*np.pi/4 for z in (zlo-eps, zlo+(zhi-zlo)/6,
            zmid, zhi-(zhi-zlo)/6, zhi+eps)])
    return witnesses, signs, bore


def check_ring_sdf(env, name, source_mesh, output, *, subgrid_resolution=6):
    if subgrid_resolution not in (0,6):
        raise ValueError('Only original sparse6 or explicit dense0 SDF1024 witnesses are supported')
    from pxr import UsdGeom, UsdPhysics, UsdUtils, PhysicsSchemaTools
    from omni.physx import get_physx_property_query_interface
    from omni.physx.bindings._physx import PhysxPropertyQueryResult, PhysxPropertyQueryMode
    from bench.simulation.isaacsim_backend.env import _np
    from bench.evaluation.cloth.cloth_robot_collision import state_digest
    import torch
    import trimesh
    output = Path(output)
    mesh = trimesh.load(source_mesh, force='mesh', process=False)
    info = env.scene_info['objects'][name]
    if len(info['collision_paths']) != 1:
        raise ValueError('Exactly one SDF shape must replace the original compound')
    path = info['collision_paths'][0]
    prim = env.stage.GetPrimAtPath(path)
    actual_mesh = UsdGeom.Mesh(prim)
    if (UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr().Get() != 'sdf'
            or not np.array_equal(actual_mesh.GetPointsAttr().Get(), np.asarray(mesh.vertices,np.float32))
            or not np.array_equal(actual_mesh.GetFaceVertexIndicesAttr().Get(), mesh.faces.reshape(-1))
            or not np.allclose(UsdGeom.Xformable(prim).GetLocalTransformation(),np.eye(4),rtol=0,atol=0)):
        raise ValueError('Candidate must retain the complete unscaled source surface')
    before = state_digest(env.capture_state())
    before_step = int(env.world.current_time_step_index)
    def layers():
        return {l.identifier: hashlib.sha256(l.ExportToString().encode()).hexdigest()
                for l in env.stage.GetUsedLayers()}
    before_layers = layers()
    points, normals = surface_witnesses(mesh)
    witnesses, signs, bore = convention_and_bore_witnesses(mesh)
    capacity = max(len(points), len(witnesses), len(bore))
    view = env._sim_view.create_sdf_shape_view(path, capacity)
    if view.count != 1 or list(view.object_paths) != [path]:
        raise ValueError('Native SDF shape inventory mismatch')
    def query(p):
        p = np.asarray(p)
        padded = np.repeat(p[-1:], capacity, axis=0)
        padded[:len(p)] = p
        return _np(view.get_sdf_and_gradients(torch.as_tensor(padded[None],
            dtype=torch.float32, device=env.device)))[0][:len(p)].copy()
    raw_witnesses = query(witnesses)
    column = sdf_distance_column(raw_witnesses, signs)
    expected = source_surface_errors(mesh,witnesses)*signs
    calibration_error = float(np.max(abs(raw_witnesses[:,column]-expected)))
    # The whole bore must remain empty; these include both openings and the axis.
    raw_bore = query(bore)
    lower, upper = np.full(len(points),-.0005), np.full(len(points),.0005)
    low_values = query(points+lower[:,None]*normals)[:,column]
    high_values = query(points+upper[:,None]*normals)[:,column]
    bracketed = (low_values < 0)&(high_values > 0)
    source_values = query(points)[:,column]
    for _ in range(14):
        mid = (lower+upper)/2
        values = query(points+mid[:,None]*normals)[:,column]
        lower = np.where(values < 0, mid, lower)
        upper = np.where(values >= 0, mid, upper)
    offset = (lower+upper)/2
    zero = points+offset[:,None]*normals
    errors = source_surface_errors(mesh,zero)
    np.savez_compressed(output/'sdf_witnesses.npz', source_points=points, normals=normals,
        source_distances=source_values, bracket_inside_distances=low_values,
        bracket_outside_distances=high_values, bracketed=bracketed, zero_points=zero,
        zero_normal_offset_m=offset, zero_nearest_source_distance_m=errors,
        convention_points=witnesses, convention_raw=raw_witnesses, convention_expected=expected,
        bore_points=bore,bore_raw=raw_bore)
    body_records, collider_records, finished = [], [], []
    def rigid(r):
        body_records.append({'valid': r.result == PhysxPropertyQueryResult.VALID,
            'mass_kg': float(r.mass), 'com_m': list(r.center_of_mass),
            'principal_inertia_kg_m2': list(r.inertia), 'principal_axes_xyzw': list(r.principal_axes)})
    def collider(r):
        collider_records.append({'valid': r.result == PhysxPropertyQueryResult.VALID,
            'path': str(PhysicsSchemaTools.intToSdfPath(r.path_id)), 'volume_m3': float(r.volume),
            'aabb_min': list(r.aabb_local_min), 'aabb_max': list(r.aabb_local_max)})
    get_physx_property_query_interface().query_prim(
        stage_id=UsdUtils.StageCache.Get().GetId(env.stage).ToLongInt(),
        prim_id=PhysicsSchemaTools.sdfPathToInt(info['path']),
        query_mode=PhysxPropertyQueryMode.QUERY_RIGID_BODY_WITH_COLLIDERS, timeout_ms=-1,
        finished_fn=lambda: finished.append(True), rigid_body_fn=rigid, collider_fn=collider)
    body = env._object_views[name]
    mass = float(_np(body.get_masses()).reshape(-1)[0])
    com = _np(body.get_coms()).reshape(7)
    inertia = _np(body.get_inertias()).reshape(3,3).T
    independent = (finished == [True] and len(body_records)==1 and body_records[0]['valid']
        and len(collider_records)==1 and collider_records[0]['valid'] and collider_records[0]['path']==path
        and np.isclose(body_records[0]['mass_kg'],mass,rtol=1e-5,atol=1e-7)
        and np.allclose(body_records[0]['com_m'],com[:3],rtol=0,atol=1e-7)
        and np.allclose(sorted(body_records[0]['principal_inertia_kg_m2']), np.linalg.eigvalsh(inertia),rtol=2e-5,atol=1e-9))
    unchanged = (state_digest(env.capture_state()) == before and layers()==before_layers
                 and int(env.world.current_time_step_index)==before_step)
    sdf_parameters = {key:prim.GetAttribute('physxSDFMeshCollision:'+key).Get() for key in (
        'sdfResolution','sdfSubgridResolution','sdfBitsPerSubgridPixel','sdfEnableRemeshing',
        'sdfTriangleCountReductionFactor')}
    expected_parameters = {'sdfResolution':1024,'sdfSubgridResolution':subgrid_resolution,
        'sdfBitsPerSubgridPixel':'BitsPerPixel16','sdfEnableRemeshing':False,
        'sdfTriangleCountReductionFactor':1.}
    checks = {'all_rays_bracketed': bool(bracketed.all()),
        'sdf_authored_parameters_retained': sdf_parameters == expected_parameters,
        'coordinate_and_distance_convention': calibration_error <= 20e-6,
        'bore_remains_empty': bool(np.all(raw_bore[:,column] > 0)),
        'all_source_samples_within_tolerance': bool(np.max(abs(source_values)) <= 20e-6),
        'all_zero_normal_offsets_within_tolerance': bool(np.max(abs(offset)) <= 20e-6),
        'all_zero_nearest_surface_errors_within_tolerance': bool(errors.max() <= 20e-6),
        'independent_native_mass_query': bool(independent), 'query_read_only': unchanged}
    return {'passed': all(checks.values()), 'checks': checks, 'native_sdf_path': path,
        'source_mesh': str(source_mesh), 'sample_count': len(points), 'source_triangle_count':len(mesh.faces),
        'distance_column':column, 'calibration_maximum_error_m':calibration_error,
        'maximum_source_sample_distance_m':float(np.max(abs(source_values))),
        'maximum_zero_normal_offset_m':float(np.max(abs(offset))),
        'maximum_zero_nearest_surface_error_m':float(errors.max()),
        'unbracketed_count':int((~bracketed).sum()), 'tolerance_m':20e-6,
        'sdf_parameters_usd_readback':sdf_parameters, 'bore_sample_count':len(bore),
        'storage_scope':('Dense float32; sdfBitsPerSubgridPixel retained but inapplicable.' if subgrid_resolution==0 else 'Sparse subgrid storage with BitsPerPixel16.'),
        'native_property_query':{'bodies':body_records,'colliders':collider_records},
        'before_state_digest':before,'before_step':before_step,'before_layer_hashes':before_layers,
        'scope':'Finite native SDF surface/bore witnesses; not exhaustive continuous geometry or contact acceptance.'}
