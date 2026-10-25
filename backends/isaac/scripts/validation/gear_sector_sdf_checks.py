"""Finite external, cap, seam, occupancy and bore gates for gear sector SDFs."""
from pathlib import Path
import json
import numpy as np
import trimesh

from bench.isaacsim.release import digest, dump
from bench.paths import load_json
from scripts.assets.prepare_gear_sector_sdf import PARAMETERS, SECTOR_COUNT, ASSET_SUFFIX
from scripts.assets.prepare_gear_sector_subgrid import validated_parameters
from scripts.validation.gear_sector_geometry import nearest_surface, signed_surface_distance
from scripts.validation.gear_sdf_checks import (TOLERANCE_M, ZERO_RESIDUAL_M,
    surface_points, bore_points, calibration_points, calibrate_layout, project_all)

BRACKET_M = .0001
CAP_EXTERNAL_CLEARANCE_M = .0002


def validate_cooking_profile(contract, asset_root):
    """Admit the declared weld-only contrast without changing legacy contracts."""
    profile=contract.get('cooking_profile')
    source_path=Path(asset_root)/'SOURCE.json'
    source=load_json(source_path) if source_path.is_file() else {}
    if profile is None:
        if 'cooking_profile' in source:
            raise ValueError('Candidate cooking profile is missing from the frozen contract')
        return None
    from scripts.assets.prepare_gear_sector_weld import WELD0_COOKING_PROFILE
    same=lambda value: json.dumps(value,sort_keys=True)==json.dumps(WELD0_COOKING_PROFILE,sort_keys=True)
    if not same(profile) or contract['parameters']!=profile['sdf_parameters']:
        raise ValueError('Only the frozen subgrid2 weld0 cooking profile is permitted')
    if not same(source.get('cooking_profile')):
        raise ValueError('Candidate provenance does not match the frozen cooking profile')
    return profile


def cooking_schema_readback(prim, profile):
    """Read composed USD opinions, without claiming access to cooked topology."""
    attribute='physxTriangleMeshCollision:weldTolerance'
    api='PhysxTriangleMeshCollisionAPI'
    attr=prim.GetAttribute(attribute)
    authored=bool(attr and attr.HasAuthoredValueOpinion())
    value=attr.Get() if attr else None
    finite=value is not None and np.isfinite(value)
    observed={'api_applied':api in prim.GetAppliedSchemas(), 'attribute':attribute,
        'value_authored':authored,'value':float(value) if finite else None,
        'nonfinite_or_missing_fallback':None if finite else str(value),
        'scope':'Composed USD readback before queries; not a direct cooked-mesh or welding-execution readback.'}
    observed['declared_profile_matches']=bool(profile is None or
        (observed['api_applied'] and authored and finite and value==profile['weld_tolerance']))
    return observed


def finite_max(values):
    values = np.asarray(values)
    return float(np.max(values)) if values.size and np.isfinite(values).all() else None


class QueryJournal:
    """Persist each request before invoking the native API, then its raw reply."""
    def __init__(self, query, root):
        self.query, self.root, self.index = query, Path(root), 0
        self.all_receipts_finite = True
        self.invalid_receipt_indices = []
        self.root.mkdir()

    def __call__(self, points):
        number = self.index
        self.index += 1
        prefix = self.root / f'{number:04d}'
        np.savez_compressed(str(prefix)+'_request.npz', points=np.asarray(points))
        try:
            result = self.query(points)
            np.savez_compressed(str(prefix)+'_response.npz', raw=result)
            valid=bool(result.shape==(len(points),4) and np.isfinite(result).all())
            self.all_receipts_finite &= valid
            if not valid:self.invalid_receipt_indices.append(number)
            dump(str(prefix)+'_receipt.json', {'completed': True, 'index': number,'finite_Nx4':valid})
            return result
        except Exception as exc:
            self.all_receipts_finite = False
            if number not in self.invalid_receipt_indices:self.invalid_receipt_indices.append(number)
            dump(str(prefix)+'_receipt.json', {'completed': False, 'index': number,
                'error_type': type(exc).__name__, 'error': str(exc)})
            raise


SEAM_OFFSETS_M = np.array([-40., -20., 0., 20., 40.])*1.e-6


def finite_layout(query, points, expected, normals):
    """Reject any nonfinite column in all seven calibration API receipts.

    The supplied journal stores each raw reply before this gate rejects it.
    Derivative queries must not ignore invalid gradient columns just because
    the finite-difference calculation only consumes the distance column.
    """
    def checked(p):
        raw=query(p)
        if raw.shape != (len(p),4) or not np.isfinite(raw).all():
            raise ValueError('Every calibration query requires finite Nx4 raw data')
        return raw
    return calibrate_layout(checked,points,expected,normals)


def root_bracket(query, centers, normals, *, increasing):
    """Signed distance zero on a predeclared internal-cap normal segment."""
    centers, normals = np.asarray(centers), np.asarray(normals)
    lo, hi = np.full(len(centers), -BRACKET_M), np.full(len(centers), BRACKET_M)
    first_raw = query(centers+lo[:,None]*normals)
    last_raw = query(centers+hi[:,None]*normals)
    first,last=first_raw[:,3],last_raw[:,3]
    bracket = ((first < 0.) & (last > 0.)) if increasing else ((first > 0.) & (last < 0.))
    bracket &= np.isfinite(first_raw).all(axis=1) & np.isfinite(last_raw).all(axis=1)
    histories = [np.c_[lo, first], np.c_[hi, last]]
    for _ in range(16):
        mid = (lo+hi)/2
        raw = query(centers+mid[:,None]*normals)
        value = raw[:,3]
        bracket &= np.isfinite(raw).all(axis=1)
        move_lo = value < 0. if increasing else value > 0.
        lo = np.where(move_lo, mid, lo)
        hi = np.where(move_lo, hi, mid)
        histories.append(np.c_[mid, value])
    root = (lo+hi)/2
    final_raw=query(centers+root[:,None]*normals)
    residual = np.abs(final_raw[:,3])
    qualified = bracket & np.isfinite(final_raw).all(axis=1) & (residual <= ZERO_RESIDUAL_M)
    return {'root_offset_m': root, 'residual_m': residual, 'qualified': qualified,
            'history': np.stack(histories), 'bracket_valid': bracket}


def gap_metrics(first, second):
    gap = second['root_offset_m']-first['root_offset_m']
    valid = first['qualified'] & second['qualified'] & np.isfinite(gap)
    return {'passed': bool(len(gap) and np.all(valid & (abs(gap) <= TOLERANCE_M))),
        'signed_gap_m': gap, 'qualified': valid & (abs(gap) <= TOLERANCE_M),
        'maximum_gap_m': finite_max(np.maximum(gap,0.)),
        'maximum_overlap_m': finite_max(np.maximum(-gap,0.))}


def occupancy_metrics(native, cpu_piece_min, cpu_body):
    """Separate metric tolerance, unambiguous membership and seam ambiguity."""
    native, cpu_piece_min, cpu_body = map(np.asarray, (native,cpu_piece_min,cpu_body))
    if native.shape != cpu_piece_min.shape or native.shape != cpu_body.shape:
        raise ValueError('Occupancy populations must align')
    finite = np.isfinite(native) & np.isfinite(cpu_piece_min) & np.isfinite(cpu_body)
    error = abs(native-cpu_piece_min)
    inside, outside = cpu_body < -1.e-10, cpu_body > 1.e-10
    inside_clear = (cpu_body < -TOLERANCE_M) & (cpu_piece_min < -TOLERANCE_M)
    outside_clear = cpu_body > TOLERANCE_M
    membership = (~inside_clear | (native < 0.)) & (~outside_clear | (native > 0.))
    qualified = finite & (error <= TOLERANCE_M) & membership
    return {'passed': bool(len(native) and np.all(qualified)),
        'piece_metric_passed': bool(len(native) and np.all(finite & (error <= TOLERANCE_M))),
        'unambiguous_membership_passed': bool(len(native) and np.all(finite & membership)),
        'unambiguous_inside_count': int(inside_clear.sum()),
        'unambiguous_outside_count': int(outside_clear.sum()),
        'membership_ambiguous_within_20um_of_body_or_cut_count': int((~inside_clear & ~outside_clear).sum()),
        'maximum_piece_union_metric_error_m': finite_max(error),
        'strict_native_wrong_sign_inside_count': int(np.sum(inside & (native > 0.))),
        'strict_native_wrong_sign_outside_count': int(np.sum(outside & (native < 0.))),
        'cpu_internal_interface_zero_count': int(np.sum(inside & (abs(cpu_piece_min) < 1.e-10))),
        'maximum_positive_distance_inside_m': finite_max(np.maximum(native[inside],0.)),
        'maximum_negative_distance_outside_m': finite_max(np.maximum(-native[outside],0.)),
        'qualified': qualified,
        'scope': 'Metric compares the closed-piece representation. Membership requires correct sign outside the fixed 20um ambiguity band of body/cut surfaces; exact-cut zeros are separately bounded by seam-gap gates, not claimed as original-body exterior.'}


def legacy_union_metrics(raw, expected, normals):
    """Retain the former body-field gate, including expected representation failures."""
    raw, expected, normals = map(np.asarray, (raw, expected, normals))
    distance = abs(raw[:,3]-expected)
    gradient = np.max(abs(raw[:,:3]-normals),axis=1)
    finite = np.isfinite(raw).all(axis=1)
    return {'original_union_metric_passed': bool(np.all(finite & (distance <= TOLERANCE_M) & (gradient <= .05))),
        'original_union_distance_failed_indices': np.flatnonzero(~finite | (distance > TOLERANCE_M)).tolist(),
        'original_union_gradient_failed_indices': np.flatnonzero(~finite | (gradient > .05)).tolist(),
        'maximum_original_union_distance_error_m': finite_max(distance),
        'maximum_original_union_gradient_error': finite_max(gradient),
        'used_for_sector_acceptance': False,
        'reason': 'Several old inside-body anchors lie on new internal cuts. Even exact CPU sectors have min(piece distance)=0 there, unlike the original negative body distance. This original metric is retained as a representation diagnostic, not called unchanged calibration. Each sector has its own independently verified plane calibration.'}


def build_contract(asset_root, output):
    """CPU witnesses are fixed and hashed before any native cooking/query."""
    root, output = Path(asset_root).resolve(), Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    name = 'input_gear'
    manifest = load_json(root/'manifest.json')
    asset = manifest['assets'][name+ASSET_SUFFIX]
    authored = load_json(root/'cpu_sector_geometry.json')['assets'][name]
    source = trimesh.load(authored['source_mesh'], force='mesh', process=False)
    pieces, evidence = [], []
    for spec in asset['sector_pieces']:
        e = np.load(spec['evidence_path'])
        pieces.append(trimesh.Trimesh(vertices=e['vertices_body_m'], faces=e['faces'],process=False))
        evidence.append(e)
    cloud, cloud_labels, line_centers, line_normals, line_first, line_second = [],[],[],[],[],[]
    all_caps, cap_owners = [],[]
    for i,(mesh,e) in enumerate(zip(pieces,evidence,strict=True)):
        mask = e['cap_mask']
        # Every cap is retained; only the known-clearance subset is suitable
        # for an unambiguous internal normal-line bracket.
        centers = mesh.triangles_center[mask]
        normals = mesh.face_normals[mask]
        distances = signed_surface_distance(source,centers)
        for p,n,d in zip(centers,normals,distances,strict=True):
            all_caps.append(p); cap_owners.append(i)
            if d < -CAP_EXTERNAL_CLEARANCE_M:
                end = 2*np.pi*(i+1)/SECTOR_COUNT
                end_out = np.array([-np.sin(end),np.cos(end),0.])
                other = (i+1)%SECTOR_COUNT if n@end_out > .99 else (i-1)%SECTOR_COUNT
                line_centers.append(p);line_normals.append(n)
                line_first.append(i);line_second.append(other)
        for edge_index,(pair,edge) in enumerate(zip(mesh.face_adjacency,mesh.face_adjacency_edges,strict=True)):
            if bool(mask[pair[0]]) == bool(mask[pair[1]]): continue
            exterior = pair[0] if not mask[pair[0]] else pair[1]
            cap = pair[1] if mask[pair[1]] else pair[0]
            n, across = mesh.face_normals[exterior], mesh.face_normals[cap]
            a,b = mesh.vertices[edge]
            for t in (0.,.125,.25,.5,.75,.875,1.):
                center=a+(b-a)*t
                for u in SEAM_OFFSETS_M:
                    for v in SEAM_OFFSETS_M:
                        cloud.append(center+u*across+v*n)
                        cloud_labels.append((i,edge_index,t,u,v))
    cloud = np.asarray(cloud)
    caps = np.asarray(all_caps)
    legacy, legacy_body, legacy_normals = calibration_points(input_hub=True)
    groups = {'seam_cloud':cloud,'internal_caps':caps,'legacy':legacy}
    audit = {}
    for key,points in groups.items():
        body_signed = signed_surface_distance(source,points)
        minimum = np.full(len(points),np.inf)
        # Outside the exact union, min piece distance equals its unsigned
        # original-source distance. Inside, only angular owners can be negative.
        outside = body_signed >= -1.e-12
        minimum[outside] = abs(body_signed[outside])
        for i,mesh in enumerate(pieces):
            a,b=2*np.pi*np.array([i,i+1])/SECTOR_COUNT
            keep=np.array([[-np.sin(a),np.cos(a),0.],[np.sin(b),-np.cos(b),0.]])
            selected=(~outside)&np.all(points@keep.T>=-1.e-12,axis=1)
            ids=np.flatnonzero(selected)
            if len(ids):
                distances,_=nearest_surface(mesh,points[ids])
                minimum[ids]=np.minimum(minimum[ids],-distances)
        if not np.isfinite(minimum).all():raise ValueError('CPU union occupancy incomplete')
        audit[key+'_body_signed_m']=body_signed
        audit[key+'_piece_min_signed_m']=minimum
    centers,normals=np.asarray(line_centers),np.asarray(line_normals)
    if len(centers)==0:raise ValueError('No interior seam brackets')
    cpu_brackets=[]
    for j,(i,k) in enumerate(zip(line_first,line_second,strict=True)):
        ends=centers[j]+np.array([-BRACKET_M,BRACKET_M])[:,None]*normals[j]
        left=signed_surface_distance(pieces[i],ends)
        right=signed_surface_distance(pieces[k],ends)
        cpu_brackets.append([left,right])
        if not(left[0]<0<left[1] and right[0]>0>right[1]):
            raise ValueError('CPU internal cap bracket has incorrect side membership')
    np.savez_compressed(output/'witnesses.npz',source_points=surface_points(source),
        bore_points=bore_points(source),seam_cloud=cloud,seam_labels=np.asarray(cloud_labels),internal_caps=caps,
        cap_owners=np.asarray(cap_owners),line_centers=centers,line_normals=normals,
        line_first=np.asarray(line_first),line_second=np.asarray(line_second),cpu_bracket_endpoint_distances_m=np.asarray(cpu_brackets),
        legacy=legacy,legacy_original_expected_m=legacy_body,legacy_original_normals_body=legacy_normals,**audit)
    contract={'schema':'gear_sector_geometry_contract.v1','asset_root':str(root),
        'manifest_sha256':digest(root/'manifest.json'),'source_mesh':authored['source_mesh'],
        'source_mesh_sha256':digest(authored['source_mesh']),
        'witnesses_sha256':digest(output/'witnesses.npz'),'generator_sha256':digest(__file__),
        'sector_count':SECTOR_COUNT,'parameters':PARAMETERS,
        'source_witness_count':len(surface_points(source)),'bore_witness_count':len(bore_points(source)),
        'seam_cloud_count':len(cloud),'all_cap_centers_count':len(caps),
        'internal_line_count':len(centers),'internal_line_external_clearance_m':CAP_EXTERNAL_CLEARANCE_M,
        'seam_label_columns':['piece_index','edge_index','fraction_along_edge','across_cut_offset_m','exterior_normal_offset_m'],
        'membership_ambiguity_band_m':TOLERANCE_M,
        'legacy_scope':'Original union-distance/gradient checks retained as diagnostics; min-piece field is not the body signed distance at internal cuts.',
        'line_half_bracket_m':BRACKET_M,'seam_offsets_m':SEAM_OFFSETS_M.tolist(),
        'geometry_tolerance_m':TOLERANCE_M,'zero_residual_tolerance_m':ZERO_RESIDUAL_M,
        'scope':'Input gear only; no observation step/load/full-three-gear qualification.'}
    dump(output/'contract.json',contract)
    return contract


def check_native(env, asset, contract_root, output, *, bounded_search=None):
    """Read each local SDF and transform its queried gradient back to the body."""
    import torch
    from pxr import UsdGeom, UsdPhysics
    from bench.simulation.isaacsim_backend.env import _np
    output=Path(output);output.mkdir()
    contract_root=Path(contract_root)
    contract=load_json(contract_root/'contract.json')
    expected_parameters=validated_parameters(contract['parameters'])
    cooking_profile=validate_cooking_profile(contract,contract['asset_root'])
    witness=np.load(contract_root/'witnesses.npz')
    if digest(contract_root/'witnesses.npz')!=contract['witnesses_sha256']:
        raise ValueError('Frozen CPU witnesses changed')
    info=env.scene_info['objects']['input_gear']
    if len(info['collision_paths'])!=SECTOR_COUNT:raise ValueError('Eight SDF shapes required')
    queries={};journals={};piece_reports={};raw_queries={};layouts={};projections={}
    capacity=max(len(witness['source_points']),len(witness['seam_cloud']))
    for spec in asset['sector_pieces']:
        i=spec['index'];row={'passed':False};folder=output/('piece_'+str(i).zfill(2));folder.mkdir()
        try:
            e=np.load(spec['evidence_path']);rotation=np.asarray(spec['mesh_to_body_rotation'])
            path=next(p for p in info['collision_paths'] if p.endswith('/'+spec['usd_prim_path'].split('/')[-1]))
            prim=env.stage.GetPrimAtPath(path);geometry=UsdGeom.Mesh(prim)
            expected_matrix=np.eye(4);expected_matrix[:3,:3]=rotation.T
            matrix=np.asarray(UsdGeom.Xformable(prim).GetLocalTransformation())
            transform_ok=np.allclose(matrix,expected_matrix,rtol=0.,atol=1.e-12)
            mesh_ok=(np.array_equal(geometry.GetPointsAttr().Get(),np.asarray(e['vertices_local_m'],np.float32))
                and np.array_equal(geometry.GetFaceVertexIndicesAttr().Get(),e['faces'].reshape(-1))
                and UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr().Get()=='sdf')
            parameters={k:prim.GetAttribute('physxSDFMeshCollision:'+k).Get() for k in PARAMETERS}
            cooking=cooking_schema_readback(prim,cooking_profile)
            row.update(native_path=path,mesh_and_topology_preserved=bool(mesh_ok),
                mesh_to_body_transform_verified=bool(transform_ok),native_usd_local_matrix=matrix.tolist(),
                schema_parameters=parameters,cooking_schema=cooking)
            dump(folder/'schema_before_queries.json',row)
            if not cooking['declared_profile_matches']:
                raise ValueError('Actual scene collider does not retain the declared cooking profile')
            view=env._sim_view.create_sdf_shape_view(path,capacity)
            if view.count!=1 or list(view.object_paths)!=[path]:raise ValueError('SDF singleton identity mismatch')
            def raw_query(points,view=view):
                p=np.asarray(points)
                padded=np.repeat(p[-1:],capacity,axis=0);padded[:len(p)]=p
                return _np(view.get_sdf_and_gradients(torch.as_tensor(padded[None],dtype=torch.float32,
                    device=env.device)))[0,:len(p)].copy()
            raw_query=QueryJournal(raw_query,folder/'query_journal')
            journals[i]=raw_query
            local=trimesh.Trimesh(vertices=e['vertices_local_m'],faces=e['faces'],process=False)
            own=surface_points(local)
            cp,cg,ce=(e[k] for k in ('calibration_points_local_m','calibration_gradients_local','calibration_expected_m'))
            # Every raw population survives even an undecodable/failed calibration.
            populations={'piece_surface':own,'calibration':cp,
                **{key:witness[key]@rotation for key in ('source_points','bore_points','seam_cloud','internal_caps','legacy')}}
            np.savez_compressed(folder/'all_population_inputs.npz',**populations,mesh_to_body_rotation=rotation)
            for key,points in populations.items():
                raw=raw_query(points)
                np.savez_compressed(folder/(key+'_raw.npz'),local_points=points,raw=raw)
            layout=finite_layout(raw_query,cp,ce,cg)
            column,order=layout['distance_column'],layout['gradient_columns_xyz']
            raw_queries[i]=raw_query
            layouts[i]={'distance_column':column,'gradient_columns':order}
            def body_query(points,raw_query=raw_query,rotation=rotation,column=column,order=order):
                raw=raw_query(np.asarray(points)@rotation)
                return np.c_[raw[:,order]@rotation.T,raw[:,column]]
            queries[i]=body_query
            projection=project_all(raw_query,own,column,order)
            projections['piece_'+str(i).zfill(2)]=(own,projection)
            np.savez_compressed(folder/'piece_projection.npz',source_local_points=own,
                **{k:v for k,v in projection.items() if k!='passed'},
                calibration_raw=layout.pop('raw'),calibration_finite_difference=layout.pop('finite_difference_gradient'))
            face_samples=len(local.faces)*7
            cap=np.repeat(e['cap_mask'],7)
            source_distance=abs(projection['original'][:,column])
            row={**row,'passed':bool(transform_ok and mesh_ok and parameters==expected_parameters and layout['passed'] and projection['passed'] and raw_query.all_receipts_finite),
                'native_path':path,'mesh_and_topology_preserved':bool(mesh_ok),
                'mesh_to_body_transform_verified':bool(transform_ok),'native_usd_local_matrix':matrix.tolist(),
                'schema_parameters':parameters,'layout':layout,
                'whole_piece_boundary_including_caps_passed':projection['passed'],
                'all_piece_samples':len(own),'cap_triangle_samples':int(cap.sum()),
                'maximum_source_boundary_distance_m':finite_max(source_distance),
                'maximum_cap_sample_distance_m':finite_max(source_distance[:face_samples][cap]),
                'maximum_exterior_face_sample_distance_m':finite_max(source_distance[:face_samples][~cap]),
                'maximum_zero_residual_m':finite_max(projection['residual']),
                'maximum_zero_displacement_m':finite_max(projection['displacement']),
                'unqualified_count':int((~projection['qualified']).sum())}
        except Exception as exc:
            import traceback
            row['error']={'type':type(exc).__name__,'detail':str(exc),'traceback':traceback.format_exc()}
        piece_reports[str(i)]=row;dump(folder/'report.json',row)
    report={'passed':False,'pieces':piece_reports,'all_eight_layouts_decoded':len(queries)==SECTOR_COUNT}
    if len(queries)!=SECTOR_COUNT:
        report['incomplete_reason']='Raw populations retained; missing local column/frame identity prevents compound interpretations.'
        return report
    compound_histories=[]
    def compound(points,*,retain=False):
        raw=np.stack([queries[i](points) for i in range(SECTOR_COUNT)])
        if retain:compound_histories.append(raw)
        index=np.argmin(raw[:,:,3],axis=0)
        selected=raw[index,np.arange(len(points))]
        # Nonfinite values are never silently replaced by a different shape.
        selected[~np.isfinite(raw).all(axis=(0,2))]=np.nan
        return selected
    source_points=witness['source_points']
    projection=project_all(lambda p:compound(p,retain=True),source_points,3,[0,1,2])
    projections['original_union']=(source_points,projection)
    source=trimesh.load(contract['source_mesh'],force='mesh',process=False)
    np.savez_compressed(output/'original_exterior_projection_before_cpu_distance.npz',
        source_points=source_points,all_piece_queries=np.stack(compound_histories),
        **{k:v for k,v in projection.items() if k!='passed'})
    nearest=np.full(len(source_points),np.nan)
    finite_points=np.isfinite(projection['projected']).all(axis=1)
    if np.any(finite_points):
        nearest[finite_points],_=nearest_surface(source,projection['projected'][finite_points])
    exterior_passed=bool(projection['passed'] and np.all(nearest<=TOLERANCE_M))
    np.savez_compressed(output/'original_exterior_projection.npz',source_points=source_points,
        all_piece_queries=np.stack(compound_histories),nearest_original_surface_m=nearest,
        **{k:v for k,v in projection.items() if k!='passed'})
    report['original_exterior']={'passed':exterior_passed,'witness_count':len(source_points),
        'maximum_native_source_distance_m':finite_max(abs(projection['original'][:,3])),
        'maximum_projection_displacement_m':finite_max(projection['displacement']),
        'maximum_nearest_original_surface_m':finite_max(nearest),
        'maximum_zero_residual_m':finite_max(projection['residual']),
        'unqualified_count':int((~projection['qualified']).sum())}
    for key in ('seam_cloud','internal_caps','legacy'):
        raw=np.stack([queries[i](witness[key]) for i in range(SECTOR_COUNT)])
        minimum=np.min(raw[:,:,3],axis=0)
        minimum[~np.isfinite(raw).all(axis=(0,2))]=np.nan
        metrics=occupancy_metrics(minimum,witness[key+'_piece_min_signed_m'],witness[key+'_body_signed_m'])
        np.savez_compressed(output/(key+'.npz'),points_body_m=witness[key],all_piece_queries=raw,
            native_piece_min_signed_m=minimum,cpu_body_signed_m=witness[key+'_body_signed_m'],
            cpu_piece_min_signed_m=witness[key+'_piece_min_signed_m'],qualified=metrics.pop('qualified'))
        report[key]=metrics
        if key=='legacy':
            index=np.argmin(raw[:,:,3],axis=0)
            report['legacy_original_union_diagnostic']=legacy_union_metrics(raw[index,np.arange(len(index))],
                witness['legacy_original_expected_m'],witness['legacy_original_normals_body'])
    bore_raw=np.stack([queries[i](witness['bore_points']) for i in range(SECTOR_COUNT)])
    bore_min=np.min(bore_raw[:,:,3],axis=0)
    np.savez_compressed(output/'bore.npz',points_body_m=witness['bore_points'],all_piece_queries=bore_raw)
    report['bore']={'passed':bool(np.isfinite(bore_raw).all() and np.all(bore_min>0.)),
        'witness_count':len(bore_min),'minimum_signed_clearance_m':(-finite_max(-bore_min) if np.isfinite(bore_min).all() else None),
        'nonpositive_count':int(np.sum(bore_min<=0.))}
    gap_reports=[]
    for first in range(SECTOR_COUNT):
        for second in ((first-1)%SECTOR_COUNT,(first+1)%SECTOR_COUNT):
            indices=np.flatnonzero((witness['line_first']==first)&(witness['line_second']==second))
            if len(indices)==0:raise ValueError('Each side requires known-clearance cap brackets')
            c,n=witness['line_centers'][indices],witness['line_normals'][indices]
            a=root_bracket(queries[first],c,n,increasing=True)
            b=root_bracket(queries[second],c,n,increasing=False)
            gap=gap_metrics(a,b)
            np.savez_compressed(output/('gap_'+str(first)+'_'+str(second)+'.npz'),
                centers_body_m=c,normals_body=n,indices=indices,
                **{'first_'+k:v for k,v in a.items()},**{'second_'+k:v for k,v in b.items()},
                signed_gap_m=gap.pop('signed_gap_m'),qualified=gap.pop('qualified'))
            gap_reports.append({'first':first,'second':second,'count':len(indices),**gap})
    report['internal_seam_gap']={'passed':all(r['passed'] for r in gap_reports),'pairs':gap_reports,
        'scope':'Both positive void gaps and negative overlaps must be <=20um in magnitude; caps are farther than200um from the original exterior and roots must stay in the fixed+-100um bracket.'}
    for index,journal in journals.items():
        row=piece_reports[str(index)]
        row['query_receipts']={'count':journal.index,'all_finite_Nx4':journal.all_receipts_finite,
            'invalid_receipt_indices':journal.invalid_receipt_indices}
        row['passed']=bool(row['passed'] and journal.all_receipts_finite)
        dump(output/('piece_'+str(index).zfill(2))/'report.json',row)
    report['all_query_receipts_finite']=bool(len(journals)==SECTOR_COUNT and all(j.all_receipts_finite for j in journals.values()))
    report['passed']=bool(report['all_query_receipts_finite'] and all(r['passed'] for r in piece_reports.values()) and exterior_passed
        and all(report[k]['passed'] for k in ('seam_cloud','internal_caps','legacy','bore','internal_seam_gap')))
    if bounded_search is not None:
        # Append a separately declared numerical query experiment. In particular,
        # neither a different root nor a closer endpoint changes the old gates.
        from scripts.validation.gear_bounded_probe import run_bounded_probe
        dump(output/'original_geometry_report.json',report)
        original_count=sum(j.index for j in journals.values())
        try:
            report['bounded_search']=run_bounded_probe(
                bounded_search,asset,raw_queries,layouts,projections,output/'bounded_search')
        except Exception as exc:
            import traceback
            report['bounded_search']={'geometry_qualified':False,'numeric_feasibility_passed':False,
                'error':{'type':type(exc).__name__,'detail':str(exc),'traceback':traceback.format_exc()}}
            report['passed']=False
        report['bounded_search']['original_native_query_count']=original_count
        report['bounded_search']['additional_native_query_count']=sum(j.index for j in journals.values())-original_count
        for index,journal in journals.items():
            piece_reports[str(index)]['query_receipts_after_bounded_search']={
                'count':journal.index,'all_finite_Nx4':journal.all_receipts_finite,
                'invalid_receipt_indices':journal.invalid_receipt_indices}
            dump(output/('piece_'+str(index).zfill(2))/'report.json',piece_reports[str(index)])
        report['all_query_receipts_finite']=all(j.all_receipts_finite for j in journals.values())
        report['passed']=bool(report['passed'] and report['all_query_receipts_finite'])
    report['scope']='Finite input-gear geometry only. Internal piece surfaces do not certify original-body exterior, transmission or quantitative contact forces.'
    return report
