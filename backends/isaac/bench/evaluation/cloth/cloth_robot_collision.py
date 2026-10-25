"""Independent cooked-collider export and cloth/robot geometry diagnostics.

Importing this module is CPU-only. Native queries occur only in collect_colliders,
when the caller supplies an already running environment. No fallback visual mesh,
collision authoring, attachment, simulation update or state setter is used here.
"""
from __future__ import annotations

import hashlib
import json
from copy import deepcopy
import numpy as np

from bench.convex_geometry import convex_triangle_audit

SCHEMA = 'cloth_robot_cooked_colliders_v1'
SUPPORT_SCHEMA = 'physx_cooked_vertex_support_with_raw_plane_bounds_v1'


def digest_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def state_digest(state):
    h = hashlib.sha256()
    for key in sorted(state):
        value = np.ascontiguousarray(state[key])
        h.update(key.encode()); h.update(str(value.dtype).encode())
        h.update(str(value.shape).encode()); h.update(value.tobytes())
    return h.hexdigest()


def transform_xyzw(pose):
    value = np.asarray(pose, dtype=float)
    if value.shape != (7,) or not np.isfinite(value).all():
        raise ValueError('Expected finite xyz/xyzw pose')
    x, y, z, w = value[3:]
    if abs(np.linalg.norm(value[3:]) - 1.) > 1e-5:
        raise ValueError('Recorded body quaternion is not normalized')
    r = np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                  [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                  [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])
    result = np.eye(4); result[:3, :3] = r; result[:3, 3] = value[:3]
    return result


def apply_transform(points, matrix):
    p, m = np.asarray(points, float), np.asarray(matrix, float)
    if p.ndim != 2 or p.shape[1] != 3 or m.shape != (4, 4) or not np.isfinite(p).all() or not np.isfinite(m).all():
        raise ValueError('Malformed finite transform or points')
    if not np.allclose(m[3], [0, 0, 0, 1], atol=1e-12):
        raise ValueError('Affine transform required')
    return p @ m[:3, :3].T + m[:3, 3]


def planes_in_body(planes,matrix):
    result=np.asarray(planes,float)@np.linalg.inv(np.asarray(matrix,float))
    lengths=np.linalg.norm(result[:,:3],axis=1)
    if not np.isfinite(result).all() or (lengths<1e-12).any():raise ValueError('Invalid transformed halfspaces')
    return result/lengths[:,None]


def piece_in_body(piece, mesh_to_body, *, planes_key='planes'):
    """Transform native halfspaces, preserving non-uniform scale in meters."""
    vertices = np.asarray(piece['vertices'], float)
    planes = np.asarray(piece[planes_key], float)
    matrix = np.asarray(mesh_to_body, float)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) < 4:
        raise ValueError('A native convex needs at least four vertices')
    if planes.ndim != 2 or planes.shape[1] != 4 or len(planes) < 4:
        raise ValueError('Native polygon halfspaces are missing')
    if not np.isfinite(planes).all() or not np.isfinite(vertices).all():
        raise ValueError('Nonfinite native convex')
    transformed = apply_transform(vertices, matrix)
    transformed_planes = planes_in_body(planes,matrix)
    if (transformed @ transformed_planes[:, :3].T + transformed_planes[:, 3]).max() > 1e-5:
        raise ValueError('Cooked polygon planes do not contain their vertices')
    return transformed, transformed_planes


def cooked_piece_geometry(piece):
    """Preserve SDK planes; derive the support hull from SDK cooked vertices.

    PhysX supportLocal uses cooked vertices, but cooked polygons can be slightly
    non-planar. Never widen their containment tolerance. Build inner/outer bounds
    covering both representations, so a boundary disagreement stays unchecked.
    """
    from scipy.spatial import ConvexHull, HalfspaceIntersection
    result=deepcopy(piece)
    vertices=np.asarray(piece['vertices'],float);raw=np.asarray(piece['planes'],float)
    if vertices.ndim!=2 or vertices.shape[1]!=3 or len(vertices)<4 or not np.isfinite(vertices).all():
        raise ValueError('Malformed cooked vertices')
    if raw.ndim!=2 or raw.shape[1]!=4 or len(raw)<4 or not np.isfinite(raw).all():
        raise ValueError('Malformed cooked planes')
    norms=np.linalg.norm(raw[:,:3],axis=1)
    if (norms<1e-12).any():raise ValueError('Degenerate cooked plane')
    raw=raw/norms[:,None]
    center=vertices.mean(0)
    if (center@raw[:,:3].T+raw[:,3]>=-1e-12).any():
        raise ValueError('Cooked centroid is not strictly inside all raw planes')
    hull=ConvexHull(vertices)
    support=hull.equations
    if (vertices@support[:,:3].T+support[:,3]).max()>1e-10:
        raise ValueError('Derived cooked-vertex support hull containment failed')
    plane_vertices=HalfspaceIntersection(raw,center).intersections
    if not np.isfinite(plane_vertices).all():raise ValueError('Raw plane intersection is unbounded or invalid')
    outer_vertices=np.concatenate([vertices,plane_vertices])
    outer=ConvexHull(outer_vertices)
    polygon_deviation=[]
    if 'polygons' in piece or 'indices' in piece:
        if len(piece['polygons'])!=len(raw):raise ValueError('Cooked plane/polygon count differs')
        indices=piece['indices']
        for i,p in enumerate(piece['polygons']):
            first,count=p['index_base'],p['num_vertices']
            if type(first) is not int or type(count) is not int or first<0 or count<3 or first+count>len(indices):
                raise ValueError('Invalid cooked polygon index range')
            ids=indices[first:first+count]
            if any(type(k) is not int or k<0 or k>=len(vertices) for k in ids):raise ValueError('Invalid cooked vertex index')
            polygon_deviation.append(float(abs(vertices[ids]@raw[i,:3]+raw[i,3]).max()))
    result.update(geometry_representation=SUPPORT_SCHEMA,support_hull_planes=support.tolist(),
        raw_plane_intersection_vertices=plane_vertices.tolist(),
        conservative_outer_vertices=outer_vertices.tolist(),conservative_outer_planes=outer.equations.tolist(),
        conservative_inner_planes=np.concatenate([support,raw]).tolist(),
        representation_difference={'raw_planes_max_vertex_support_violation_m':float((vertices@raw[:,:3].T+raw[:,3]).max()),
            'maximum_polygon_nonplanarity_m':max(polygon_deviation,default=None),
            'support_hull_volume_m3':float(hull.volume),'raw_plane_volume_m3':float(ConvexHull(plane_vertices).volume),
            'decision_rule':'Clear only outside conservative outer hull; penetration only inside both representations; intervening band unchecked'})
    return result


def verify_native_pose(mesh_to_body, native_record, *, tolerance=1e-5):
    """Check USD relative pose against PhysX, leaving scale in cooked geometry.

    Shear/reflection cannot be reconciled by this diagnostic and fail closed.
    The native query's unscaled prim-local AABB is checked separately.
    """
    matrix = np.asarray(mesh_to_body, float)
    apply_transform(np.zeros((1, 3)), matrix)
    scale = np.linalg.norm(matrix[:3, :3], axis=0)
    if (scale < 1e-12).any():
        raise ValueError('Degenerate collider scale')
    rotation = matrix[:3, :3] / scale
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=tolerance, rtol=0) or np.linalg.det(rotation) <= 0:
        raise ValueError('Collider shear or reflected transform is not verified')
    native = transform_xyzw([*native_record['local_pos'], *native_record['local_rot_xyzw']])
    position_error = float(np.max(abs(matrix[:3, 3] - native[:3, 3])))
    rotation_error = float(np.max(abs(rotation - native[:3, :3])))
    if max(position_error, rotation_error) > tolerance:
        raise ValueError('Native collider pose disagrees with USD collider-to-body transform')
    return {'position_max_abs_error_m': position_error, 'rotation_matrix_max_abs_error': rotation_error,
            'verification_tolerance': tolerance, 'mesh_scale': scale.tolist()}


def collect_colliders(env):
    """Read official PhysX cooking and property-query APIs; never mutate stage.

    Unavailable callbacks, unsupported shapes and pose/bounds mismatch explicitly
    make the export unchecked. Cooking representation is not a force sensor or
    direct PxShape memory dump; matching native inventory/bounds is recorded.
    """
    from pxr import Usd, UsdGeom, UsdPhysics, UsdUtils, PhysicsSchemaTools
    from omni.physx import get_physx_cooking_interface, get_physx_property_query_interface
    from omni.physx.bindings._physx import (PhysxCollisionRepresentationResult,
        PhysxPropertyQueryResult, PhysxPropertyQueryMode)

    stage = env.stage
    before = state_digest(env.capture_state())
    layers_before = {l.identifier: hashlib.sha256(l.ExportToString().encode()).hexdigest() for l in stage.GetUsedLayers()}
    stage_id = UsdUtils.StageCache.Get().GetId(stage).ToLongInt()
    native = {}; queries = []; issues = []; output = []
    if UsdGeom.GetStageMetersPerUnit(stage)!=1.:
        issues.append({'reason':'stage_units_not_meters'})
    paths = dict(env.scene_info['robot_link_paths'])
    cooking = get_physx_cooking_interface()
    query = get_physx_property_query_interface()
    cache = UsdGeom.XformCache()
    for body_name, body_path in paths.items():
        found = []; finished = []; body_results = []
        def collider_callback(response, sink=found):
            sink.append({'result':str(response.result), 'valid':response.result == PhysxPropertyQueryResult.VALID,
                'path':str(PhysicsSchemaTools.intToSdfPath(response.path_id)),
                'stage_id':int(response.stage_id), 'local_pos':list(response.local_pos),
                'local_rot_xyzw':list(response.local_rot), 'aabb_min':list(response.aabb_local_min),
                'aabb_max':list(response.aabb_local_max), 'volume':float(response.volume)})
        try:
            query.query_prim(stage_id=stage_id, prim_id=PhysicsSchemaTools.sdfPathToInt(body_path),
                query_mode=PhysxPropertyQueryMode.QUERY_RIGID_BODY_WITH_COLLIDERS, timeout_ms=-1,
                finished_fn=lambda sink=finished:sink.append(True), rigid_body_fn=lambda r,sink=body_results:sink.append(str(r.result)),
                collider_fn=collider_callback)
        except Exception as exc:
            issues.append({'body':body_name,'reason':'native_property_query_error','error_type':type(exc).__name__})
        queries.append({'body':body_name,'path':body_path,'finished':bool(finished),'body_results':body_results,'colliders':found})
        if not finished or not body_results or any(r!=str(PhysxPropertyQueryResult.VALID) for r in body_results):
            issues.append({'body':body_name,'reason':'native_property_query_not_finished_without_advancing_simulation'})
        for row in found:
            native.setdefault(row['path'],[]).append((body_name,row))
    inventory = []
    robot = stage.GetPrimAtPath('/World/Robot')
    for prim in Usd.PrimRange(robot, Usd.TraverseInstanceProxies()):
        if not prim.HasAPI(UsdPhysics.CollisionAPI):
            continue
        if UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get() is False:
            continue
        path = str(prim.GetPath()); inventory.append(path)
        record = {'path':path,'type':prim.GetTypeName(),'status':'unchecked','pieces':[]}
        output.append(record)
        candidates = [(name, p) for name,p in paths.items() if path==p or path.startswith(p+'/')]
        if not candidates:
            issues.append({'path':path,'reason':'missing_robot_body_mapping'});continue
        body_name, body_path = max(candidates,key=lambda x:len(x[1]))
        record.update(body_name=body_name,body_path=body_path)
        if not prim.IsA(UsdGeom.Mesh):
            issues.append({'path':path,'reason':'unsupported_non_mesh_collider'});continue
        approximation = UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr().Get()
        record['approximation'] = str(approximation)
        if approximation not in ('convexHull','convexDecomposition'):
            issues.append({'path':path,'reason':'unsupported_native_mesh_approximation'});continue
        try:
            relative = cache.GetLocalToWorldTransform(prim) * cache.GetLocalToWorldTransform(stage.GetPrimAtPath(body_path)).GetInverse()
            matrix = np.array(relative,dtype=float).T
            record['mesh_to_body_column_matrix'] = matrix.tolist()
            record['authored_collision_attributes'] = {a.GetName():str(a.Get()) for a in prim.GetAttributes()
                if a.GetName().startswith(('physics:','physx'))}
            record['filtered_pairs'] = [str(x) for x in UsdPhysics.FilteredPairsAPI(prim).GetFilteredPairsRel().GetTargets()]
            received=[]
            def cooked_callback(result,convexes):
                received.append(result)
                if result != PhysxCollisionRepresentationResult.RESULT_VALID:return
                for convex in convexes:
                    record['pieces'].append({'vertices':[list(v) for v in convex.vertices],
                        'indices':[int(i) for i in convex.indices],
                        'polygons':[{'index_base':int(p.index_base),'num_vertices':int(p.num_vertices)} for p in convex.polygons],
                        'planes':[list(p.plane) for p in convex.polygons]})
            cooking.request_convex_collision_representation(stage_id=stage_id,
                collision_prim_id=PhysicsSchemaTools.sdfPathToInt(prim.GetPath()),
                run_asynchronously=False,on_result=cooked_callback)
            record['cooking_results']=[str(r) for r in received]
            if received != [PhysxCollisionRepresentationResult.RESULT_VALID] or not record['pieces']:
                issues.append({'path':path,'reason':'cooking_unavailable_or_invalid'});continue
            record['pieces'].sort(key=digest_json)
            # Official property-query tests specify AABB in unscaled prim-local
            # coordinates; apply no native local_pose or scale a second time.
            local_vertices=np.concatenate([np.asarray(p['vertices'],float) for p in record['pieces']])
            record['pieces']=[cooked_piece_geometry(piece) for piece in record['pieces']]
            for piece in record['pieces']:piece_in_body(piece,matrix,planes_key='support_hull_planes')
            cooked_bounds=np.stack([local_vertices.min(0),local_vertices.max(0)])
            queried=native.get(path,[])
            record['native_query_records']=[r for _,r in queried]
            if not queried or any(name!=body_name or not r['valid'] or r['stage_id']!=stage_id for name,r in queried):
                issues.append({'path':path,'reason':'native_collider_identity_not_verified'});continue
            if len(queried)!=1:
                issues.append({'path':path,'reason':'expected_single_native_report_per_collision_prim'});continue
            record['native_pose_verification']=verify_native_pose(matrix,queried[0][1])
            query_bounds=np.array([queried[0][1]['aabb_min'],queried[0][1]['aabb_max']],float)
            if query_bounds.shape!=(2,3) or not np.isfinite(query_bounds).all() or np.any(query_bounds[0]>query_bounds[1]):
                issues.append({'path':path,'reason':'invalid_native_query_bounds'});continue
            error=float(np.max(abs(cooked_bounds-query_bounds)))
            record.update(cooked_prim_local_aabb=cooked_bounds.tolist(),native_query_prim_local_aabb=query_bounds.tolist(),
                bounds_max_abs_error_m=error,bounds_verification_tolerance_m=1e-5)
            if error>1e-5:
                issues.append({'path':path,'reason':'native_bounds_or_coordinate_convention_mismatch'});continue
            record['status']='verified_cooking_representation'
        except Exception as exc:
            issues.append({'path':path,'reason':'collider_export_error','error_type':type(exc).__name__,'message':str(exc)})
    if set(native)!=set(inventory):
        issues.append({'reason':'native_authored_inventory_mismatch','native_only':sorted(set(native)-set(inventory)),
                       'authored_only':sorted(set(inventory)-set(native))})
    after=state_digest(env.capture_state())
    layers_after={l.identifier:hashlib.sha256(l.ExportToString().encode()).hexdigest() for l in stage.GetUsedLayers()}
    if before!=after:issues.append({'reason':'physical_state_changed_during_export'})
    if layers_before!=layers_after:issues.append({'reason':'usd_layers_changed_during_export'})
    result = {'schema':SCHEMA,'status':'complete' if output and not issues else 'unchecked','stage_id':stage_id,
        'method':'Official PhysX cooking representation for the live USD CollisionAPI prim, cross-checked against native property-query inventory, prim-local bounds and collider-to-body pose',
        'scope':'Cooked-vertex support hulls with raw-plane uncertainty bounds at recorded poses; no contact force or continuous-time collision certificate',
        'visual_mesh_fallback':False,'state_hash_before':before,'state_hash_after':after,
        'layer_hashes_before':layers_before,'layer_hashes_after':layers_after,'queries':queries,
        'colliders':output,'issues':issues}
    return json.loads(json.dumps(result,allow_nan=False))


def reanalyse_export(export):
    """Recheck saved native queries, retaining the original failed assessment.

    This never edits an original export or issues a native query. The only old
    error superseded here is the disproven exact raw-plane containment premise.
    """
    result=deepcopy(export)
    result['original_assessment']={'status':export.get('status'),'issues':deepcopy(export.get('issues')),
                                  'canonical_export_sha256':digest_json(export)}
    issues=[deepcopy(x) for x in export.get('issues',[]) if not (
        x.get('reason')=='collider_export_error' and x.get('message')=='Cooked polygon planes do not contain their vertices')]
    native={}
    for query in export['queries']:
        if not query['finished'] or query['body_results']!=['PhysxPropertyQueryResult.VALID']:
            issues.append({'body':query['body'],'reason':'saved_native_query_incomplete'})
        for record in query['colliders']:native.setdefault(record['path'],[]).append((query['body'],record))
    for row in result['colliders']:
        row['status']='unchecked'
        try:
            if row['type']!='Mesh' or row['approximation'] not in ('convexHull','convexDecomposition'):
                raise ValueError('Unsupported saved native collider')
            queries=native.get(row['path'],[])
            if len(queries)!=1 or queries[0][0]!=row['body_name'] or not queries[0][1]['valid'] or queries[0][1]['stage_id']!=export['stage_id']:
                raise ValueError('Saved native collider identity mismatch')
            row['native_query_records']=[queries[0][1]]
            row['native_pose_verification']=verify_native_pose(row['mesh_to_body_column_matrix'],queries[0][1])
            if row['cooking_results']!=['PhysxCollisionRepresentationResult.RESULT_VALID'] or not row['pieces']:
                raise ValueError('Saved native cooking result invalid')
            row['pieces']=[cooked_piece_geometry(p) for p in row['pieces']]
            vertices=np.concatenate([np.asarray(p['vertices'],float) for p in row['pieces']])
            bounds=np.stack([vertices.min(0),vertices.max(0)])
            expected=np.asarray([queries[0][1]['aabb_min'],queries[0][1]['aabb_max']],float)
            error=float(np.max(abs(bounds-expected)))
            if expected.shape!=(2,3) or not np.isfinite(expected).all() or error>1e-5:
                raise ValueError('Saved native AABB mismatch')
            row.update(cooked_prim_local_aabb=bounds.tolist(),native_query_prim_local_aabb=expected.tolist(),
                bounds_max_abs_error_m=error,bounds_verification_tolerance_m=1e-5,status='verified_cooking_representation')
        except Exception as exc:
            issues.append({'path':row['path'],'reason':'saved_export_reanalysis_error','message':str(exc),'error_type':type(exc).__name__})
    if set(native)!={x['path'] for x in result['colliders']}:
        issues.append({'reason':'saved_native_inventory_mismatch'})
    if export.get('state_hash_before')!=export.get('state_hash_after') or export.get('layer_hashes_before')!=export.get('layer_hashes_after'):
        issues.append({'reason':'saved_export_mutated_state_or_layer'})
    result.update(issues=issues,status='complete' if result['colliders'] and not issues else 'unchecked',
        geometry_representation=SUPPORT_SCHEMA,
        reanalysis_scope='Read-only saved native cooking and property-query records; no new native execution; original assessment retained')
    return result


def audit_cloth_robot(export, states, triangles, body_names, *, tolerance_m=1e-4):
    """Full triangles against every exported convex; unresolved stays unchecked."""
    if type(tolerance_m) not in (int,float) or not np.isfinite(tolerance_m) or tolerance_m<=0:
        raise ValueError('Positive finite tolerance required')
    cloth=np.asarray(states['particle_q'],float);poses=np.asarray(states['body_q'],float)
    triangles=np.asarray(triangles)
    if cloth.ndim!=3 or cloth.shape[-1]!=3 or not cloth.shape[1] or not np.isfinite(cloth).all():
        raise ValueError('Finite nonempty recorded cloth vertices required')
    if poses.shape!=(len(cloth),len(body_names),7) or not np.isfinite(poses).all():
        raise ValueError('Body pose inventory or frame count mismatch')
    if triangles.ndim!=2 or triangles.shape[1]!=3 or not len(triangles) or not np.issubdtype(triangles.dtype,np.integer) or triangles.min()<0 or triangles.max()>=cloth.shape[1]:
        raise ValueError('Valid integer cloth topology required')
    if len(set(body_names))!=len(body_names):raise ValueError('Duplicate body names')
    unresolved=[];pieces=[];violations=[];uncertainties=[]
    for row in export.get('colliders',[]):
        if row.get('status')!='verified_cooking_representation' or row.get('body_name') not in body_names:
            unresolved.append(row.get('path'));continue
        try:
            for i,piece in enumerate(row['pieces']):
                matrix=row['mesh_to_body_column_matrix']
                if piece.get('geometry_representation')==SUPPORT_SCHEMA:
                    geometry=cooked_piece_geometry(piece)
                    vertices=apply_transform(geometry['conservative_outer_vertices'],matrix)
                    planes=planes_in_body(geometry['conservative_outer_planes'],matrix)
                    inner=planes_in_body(geometry['conservative_inner_planes'],matrix)
                else:
                    vertices,planes=piece_in_body(piece,matrix);inner=planes
                pieces.append((row['path'],i,body_names.index(row['body_name']),planes,inner,np.stack([vertices.min(0),vertices.max(0)])))
        except (ValueError,KeyError,np.linalg.LinAlgError):unresolved.append(row.get('path'))
    for frame,vertices in enumerate(cloth):
        local={}
        for path,piece,body,planes,inner,bounds in pieces:
            if body not in local:
                local[body]=apply_transform(vertices,np.linalg.inv(transform_xyzw(poses[frame,body])))
            result=convex_triangle_audit(local[body],triangles,planes,bounds,tolerance_m)
            if not result['all_triangles_certified_clear_within_tolerance']:
                confirmed=convex_triangle_audit(local[body],triangles,inner,bounds,tolerance_m)
                if not confirmed['all_triangles_certified_clear_within_tolerance']:
                    violations.append({'frame':frame,'collider':path,'piece':piece,**confirmed})
                uncertain=sorted(set(result['triangles_penetrating_tolerance'])-set(confirmed['triangles_penetrating_tolerance']))
                if uncertain:uncertainties.append({'frame':frame,'collider':path,'piece':piece,'triangle_ids':uncertain})
    verified=(export.get('schema')==SCHEMA and export.get('status')=='complete' and not export.get('issues')
        and export.get('visual_mesh_fallback') is False and bool(pieces) and not unresolved
        and isinstance(export.get('state_hash_before'),str) and len(export['state_hash_before'])==64
        and export.get('state_hash_before')==export.get('state_hash_after')
        and bool(export.get('layer_hashes_before')) and export.get('layer_hashes_before')==export.get('layer_hashes_after'))
    return {'status':'failed' if violations else 'passed' if verified and not uncertainties else 'unchecked',
        'frames_checked':len(cloth),'triangles_per_frame':len(triangles),'convex_pieces_checked':len(pieces),
        'tolerance_m':tolerance_m,'unresolved_colliders':unresolved,'export_verified':verified,
        'violations':violations,'representation_boundary_uncertainties':uncertainties,
        'scope':'Recorded cloth center-surface triangles vs robot cooked convexes; conservative raw-plane/support-hull boundary uncertainty; not between-frame CCD, textile calibration or cloth force'}
