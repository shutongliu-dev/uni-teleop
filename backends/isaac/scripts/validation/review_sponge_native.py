"""Independently recompute native sponge evidence from immutable raw arrays.

CPU, read only with respect to the simulator/assets/trajectory. This reviewer
never imports the online sponge evaluator or advances a native simulator.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from bench.paths import load_json


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def determinant(points, tets):
    p = np.asarray(points, dtype=np.float64)[..., tets, :]
    return np.einsum('...i,...i->...', np.cross(p[..., 1, :]-p[..., 0, :],
        p[..., 2, :]-p[..., 0, :]), p[..., 3, :]-p[..., 0, :])


def raw_metrics(q, qd, pad, platen, foam, support, block_size=32):
    rest, tets = np.asarray(foam['rest_points_m']), np.asarray(foam['tetrahedra'])
    det0 = determinant(rest, tets)
    if np.any(det0 <= 0):
        raise ValueError('Nonpositive authored rest volume')
    edges = np.unique(np.sort(tets[:, [[0,1],[0,2],[0,3],[1,2],[1,3],[2,3]]].reshape(-1, 2), 1), axis=0)
    lengths = np.linalg.norm(rest[edges[:, 1]]-rest[edges[:, 0]], axis=1)
    top = np.flatnonzero((rest[:, 2] >= rest[:, 2].max()-1e-6)
        & (abs(rest[:, 0]) <= .033) & (abs(rest[:, 1]) <= .020))
    bonds = np.asarray(foam['attachment']['node_indices'])
    anchors = np.asarray(foam['attachment']['rigid_local_positions_m'])
    metrics = {name: [] for name in ('min_j', 'max_j', 'volume_ratio', 'inverted_count',
        'min_edge_ratio', 'max_edge_ratio', 'bond_error_m', 'central_height_m',
        'pad_lowest_z_m', 'max_raw_node_speed_m_s', 'platen_bbox_gap_m',
        'platen_top_node_penetration_m', 'platen_surface_node_penetration_m',
        'central_top_mean_local_x_m', 'central_top_mean_local_y_m')}
    all_local = np.empty(q.shape, dtype=np.float32)
    first_inverted = None
    half = np.array([.035, .022, .004])
    surface_ids = np.unique(np.asarray(foam['surface_triangles']).reshape(-1))
    for start in range(0, len(q), block_size):
        end = min(start+block_size, len(q))
        x, v = np.asarray(q[start:end], float), np.asarray(qd[start:end], float)
        j = determinant(x, tets)/det0
        bad = j <= 0
        if first_inverted is None and bad.any():
            frame = int(np.flatnonzero(bad.any(axis=1))[0])
            ids = np.flatnonzero(bad[frame])
            first_inverted = {'frame': start+frame, 'tetrahedron_indices': ids.tolist(),
                'ratios': j[frame, ids].tolist(), 'rest_centroids_m': rest[tets[ids]].mean(1).tolist(),
                'world_centroids_m': x[frame, tets[ids]].mean(1).tolist()}
        stretch = np.linalg.norm(x[:, edges[:, 1]]-x[:, edges[:, 0]], axis=2)/lengths
        r = Rotation.from_quat(pad[start:end, 3:7]).as_matrix()
        local = np.einsum('bni,bij->bnj', x-pad[start:end, None, :3], r)
        all_local[start:end] = local
        actual_anchors = np.einsum('ni,bji->bnj', anchors, r)+pad[start:end, None, :3]
        pad_points = np.einsum('ni,bji->bnj', support, r)+pad[start:end, None, :3]
        values = {'min_j': j.min(1), 'max_j': j.max(1),
            'volume_ratio': (j*det0).sum(1)/det0.sum(), 'inverted_count': bad.sum(1),
            'min_edge_ratio': stretch.min(1), 'max_edge_ratio': stretch.max(1),
            'bond_error_m': np.linalg.norm(x[:, bonds]-actual_anchors, axis=2).max(1),
            'central_height_m': np.median(local[:, top, 2], axis=1),
            'pad_lowest_z_m': pad_points[:, :, 2].min(1),
            'max_raw_node_speed_m_s': np.linalg.norm(v, axis=2).max(1),
            'central_top_mean_local_x_m': local[:, top, 0].mean(1),
            'central_top_mean_local_y_m': local[:, top, 1].mean(1)}
        if platen is not None and len(platen):
            pr = Rotation.from_quat(platen[start:end, 3:7]).as_matrix()
            plocal = np.einsum('bni,bij->bnj', x-platen[start:end, None, :3], pr)
            combined = np.concatenate((x, pad_points), axis=1)
            phalf = np.abs(pr) @ half
            delta = np.maximum(np.maximum(platen[start:end, :3]-phalf-combined.max(1),
                combined.min(1)-(platen[start:end, :3]+phalf)), 0)
            # Sampled nodes inside the true rotated platen box. This is a
            # positive penetration witness, not full triangle intersection proof.
            depth = np.maximum(np.min(half-np.abs(plocal), axis=2), 0)
            values.update(platen_bbox_gap_m=np.linalg.norm(delta, axis=1),
                platen_top_node_penetration_m=depth[:, top].max(1),
                platen_surface_node_penetration_m=depth[:, surface_ids].max(1))
        else:
            values.update({name: np.full(end-start, np.nan) for name in (
                'platen_bbox_gap_m', 'platen_top_node_penetration_m', 'platen_surface_node_penetration_m')})
        for name, value in values.items():
            metrics[name].extend(value.tolist())
    return {k: np.asarray(v) for k,v in metrics.items()}, all_local, first_inverted


def contact_metrics(rows, dt):
    forces, max_error, min_sep, normal_error, points = [], 0., np.inf, 0., 0
    actors = set()
    for row in rows:
        total = np.zeros(3)
        for contact in row['contacts']:
            f = np.asarray(contact['raw_scalar_normal_impulses_ns'], float)
            n = np.asarray(contact['raw_normals'], float)
            sep = np.asarray(contact['raw_separations_m'], float)
            if n.shape != (len(f),3) or len(f) != len(sep) or len(f) != contact['point_count']:
                raise ValueError('Native point contact length mismatch')
            if not (np.isfinite(f).all() and np.isfinite(n).all() and np.isfinite(sep).all()):
                raise ValueError('Nonfinite native contact')
            vec = (f[:,None]*n).sum(0)
            max_error = max(max_error, float(abs(vec-np.asarray(contact['impulse_vector_ns'])).max()))
            max_error = max(max_error, abs(float(np.linalg.norm(f[:,None]*n, axis=1).sum())-
                contact['impulse_magnitude_sum_ns']))
            min_sep = min(min_sep, float(sep.min()))
            if (f != 0).any():
                normal_error = max(normal_error, float(abs(np.linalg.norm(n[f != 0],axis=1)-1).max()))
            actors.add((contact['actor0'], contact['actor1']))
            points += len(f)
            total += vec
        forces.append(total/dt)
    return np.asarray(forces), {'point_samples': points, 'actor_pairs': sorted(actors),
        'max_online_impulse_aggregate_error_ns': max_error,
        'minimum_native_separation_m': min_sep if np.isfinite(min_sep) else None,
        'maximum_active_normal_length_error': normal_error}


def review(run, output):
    run, output = Path(run).resolve(), Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    online = load_json(run/'qualification.json')
    manifest_path = Path(online['asset_manifest'])
    manifest = load_json(manifest_path)
    meta_path = manifest_path.parent/'metadata.json'
    meta = load_json(meta_path)
    foam = meta['volume_bodies']['foam']
    sponge = next(x for x in meta['objects'] if x['id'] == 'sponge')
    with (run/'native_frames.jsonl').open() as stream:
        rows = []
        for line in stream:
            row = json.loads(line)
            row.pop('coverage', None)
            rows.append(row)
    with np.load(run/'native_states.npz') as archive:
        raw = {key: archive[key] for key in archive.files}
    q, qd, pad, pd = (raw[k] for k in ('foam_q','foam_qd','scouring_q','scouring_qd'))
    platen = raw['platen_state'] if len(raw['platen_state']) else None
    time, step = raw['time_s'], raw['native_step']
    dt = float(online['physics_dt_s'])
    expected_steps = sum(round(float(length)/dt) for _,length in online.get('plan',[]))
    checks = {
        'asset_manifest_hash': digest(manifest_path) == online['asset_manifest_sha256'],
        'asset_metadata_hash': digest(meta_path) == manifest['metadata_sha256'],
        'asset_usd_hash': digest(manifest_path.parent/'task.usda') == manifest['usd_sha256'],
        'raw_file_hashes': all(digest(run/name)==sha for name,sha in online.get('evidence_sha256',{}).items()),
        'all_saved_arrays_finite': all(np.isfinite(v).all() for v in raw.values()),
        'jsonl_raw_frame_count': len(rows)==len(time)==len(q)==len(qd)==len(pad)==len(pd),
        'native_steps_consecutive': bool(np.all(np.diff(step)==1)),
        'raw_actual_step_matches_jsonl': np.array_equal(step,[r['native_step'] for r in rows]),
        'raw_actual_time_matches_jsonl': np.array_equal(time,[r['time_s'] for r in rows]),
        'native_dt_matches': bool(np.allclose(np.diff(time),dt,rtol=2e-6,atol=1e-10)),
        'planned_trajectory_complete': len(time)==expected_steps+1 and expected_steps>0,
        'rest_points_match_asset': np.array_equal(raw['rest_points_m'],foam['rest_points_m']),
        'tetrahedra_match_asset': np.array_equal(raw['tetrahedra'],foam['tetrahedra'])}
    if platen is not None:
        checks['platen_raw_matches_jsonl'] = np.array_equal(platen,
            [r['platen']['pose_xyzw']+r['platen']['velocity'] for r in rows])
    chunks_report = []
    cm = load_json(run/'native_chunks_manifest.json')
    for chunk in cm['chunks']:
        a,b = chunk['first_frame'],chunk['last_frame']+1
        with np.load(chunk['path']) as data:
            same = all(np.array_equal(data[k],raw[k][a:b]) for k in data.files)
        chunks_report.append({'path':chunk['path'],'first_frame':a,'last_frame':b-1,
            'sha256_matches':digest(chunk['path'])==chunk['sha256'],'matches_final_raw':same})
    checks['checkpoint_chunks_match_full_archive'] = all(x['sha256_matches'] and x['matches_final_raw'] for x in chunks_report)
    checks['checkpoint_chunks_cover_full_archive'] = cm['frames']==len(time) and \
        [x['first_frame'] for x in chunks_report]==[0]+[x['last_frame']+1 for x in chunks_report[:-1]]
    metrics, local, first_inverted = raw_metrics(q,qd,pad,platen,foam,np.asarray(sponge['collision_support_points_m']))
    forces, contact = contact_metrics(rows,dt)
    scalar_names = {'min_j':'minimum_tetrahedron_volume_ratio','max_j':'maximum_tetrahedron_volume_ratio',
        'volume_ratio':'total_volume_ratio','min_edge_ratio':'minimum_edge_length_ratio',
        'max_edge_ratio':'maximum_edge_length_ratio','bond_error_m':'maximum_material_bond_error_m',
        'central_height_m':'central_top_height_in_pad_frame_m','pad_lowest_z_m':'scouring_lowest_z_m',
        'max_raw_node_speed_m_s':'maximum_native_node_speed_m_s'}
    if platen is not None:
        scalar_names['platen_bbox_gap_m']='platen_to_composite_bbox_gap_m'
    scalar_errors = {key:float(abs(metrics[key]-[r['geometry'][name] for r in rows]).max())
        for key,name in scalar_names.items()}
    scalar_errors['normal_table_force_n'] = float(abs(forces[:,2]-[r['normal_table_force_n'] for r in rows]).max())
    checks['independent_scalar_recomputation_matches'] = max(scalar_errors.values())<2e-6
    tail = time>=time[-1]-.4-1e-10
    settle_end = float(online['plan'][0][1])
    reference = np.array([r['phase']=='settle' for r in rows]) & (time>=settle_end-.2)
    loaded = np.array([r['phase']=='loaded_hold' for r in rows])
    pose_speed = np.concatenate([np.linalg.norm((np.asarray(q[i+1:min(i+257,len(q))],float)-
        np.asarray(q[i:min(i+256,len(q)-1)],float))/np.diff(time)[i:min(i+256,len(q)-1),None,None],
        axis=2).max(1) for i in range(0,len(q)-1,256)])
    checks['complete_settled_reference_window'] = bool(reference.any() and time[-1] >= settle_end-1e-7)
    if not reference.any():
        reference = np.zeros(len(time),bool)
        reference[0] = True
    ref_local = np.mean(local[reference],axis=0,dtype=np.float64)
    recovered = np.linalg.norm(local[tail]-ref_local,axis=2)
    height = metrics['central_height_m']
    mass = foam['total_mass_kg']+sponge['mass_kg']
    restdet = determinant(raw['rest_points_m'],raw['tetrahedra'])
    lumped = np.zeros(q.shape[1])
    np.add.at(lumped,raw['tetrahedra'].reshape(-1),np.repeat(restdet/restdet.sum()*foam['total_mass_kg']/4,4))
    momentum = np.einsum('n,tnc->tc',lumped,qd)+sponge['mass_kg']*pd[:,:3]
    windows = {}
    for label, mask in [('settle_last_200ms',reference),('loaded_hold',loaded),('recover_last_400ms',tail)]:
        ids=np.flatnonzero(mask)
        if not len(ids):
            continue
        rate=((momentum[ids[-1]]-momentum[ids[0]])/(time[ids[-1]]-time[ids[0]])
              if len(ids)>1 else np.zeros(3))
        windows[label]={'frame_count':len(ids),'time_s':[float(time[ids[0]]),float(time[ids[-1]])],
            'mean_table_force_n':forces[mask].mean(0).tolist(),
            'median_central_height_m':float(np.median(height[mask])),
            'max_raw_node_speed_m_s':float(metrics['max_raw_node_speed_m_s'][mask].max()),
            'max_adjacent_step_node_speed_m_s':float(pose_speed[mask[1:]].max()) if mask[1:].any() else 0.,
            'nominal_lumped_momentum_rate_n':rate.tolist(),
            'table_z_minus_nominal_weight_minus_momentum_rate_n':float(forces[mask,2].mean()-mass*9.81-rate[2]),
            'positive_table_contact_fraction':float((forces[mask,2]>0).mean())}
    checks.update(no_inverted_tetrahedra=not np.any(metrics['inverted_count']),
        no_gross_foam_distortion=bool(metrics['min_j'].min()>=.1 and metrics['min_edge_ratio'].min()>=.5 and metrics['max_edge_ratio'].max()<=1.5),
        material_bond_error_at_most_1mm=bool(metrics['bond_error_m'].max()<=.001),
        recovered_raw_all_node_speed_at_most_25mm_s=bool(metrics['max_raw_node_speed_m_s'][tail].max()<=.025),
        recovered_adjacent_step_all_node_speed_at_most_25mm_s=bool(pose_speed[tail[1:]].max()<=.025),
        nominal_gravity_transmission_within_15pct=bool(abs(forces[tail,2].mean()-mass*9.81)<=.15*mass*9.81),
        recovered_central_height_within_1mm=bool(abs(np.median(height[tail])-np.median(height[reference]))<=.001))
    measurements = {'nominal_mass_kg':mass,'nominal_weight_n':mass*9.81,
        'minimum_tet_volume_ratio':float(metrics['min_j'].min()),'maximum_tet_volume_ratio':float(metrics['max_j'].max()),
        'minimum_edge_ratio':float(metrics['min_edge_ratio'].min()),'maximum_edge_ratio':float(metrics['max_edge_ratio'].max()),
        'maximum_bond_error_m':float(metrics['bond_error_m'].max()),
        'recovered_all_node_shape_max_error_m':float(recovered.max()),
        'recovered_all_node_shape_rms_error_m':float(np.sqrt(np.mean(recovered**2))),
        'windows':windows}
    if loaded.any():
        compression=float(np.median(height[reference])-np.median(height[loaded]))
        measurements['central_compression_m']=compression
        checks['actual_central_compression_1_to_12mm']=.001<=compression<=.012
        checks['loaded_table_force_increase_above_150mN']=forces[loaded,2].mean()>forces[reference,2].mean()+.15
    if platen is not None:
        checks['recovered_platen_composite_gap_at_least_5mm']=bool(metrics['platen_bbox_gap_m'][tail].min()>=.005)
        measurements.update(maximum_sampled_surface_node_platen_penetration_m=float(metrics['platen_surface_node_penetration_m'].max()),
            maximum_sampled_central_top_node_platen_penetration_m=float(metrics['platen_top_node_penetration_m'].max()),
            recovered_minimum_platen_composite_bbox_gap_m=float(metrics['platen_bbox_gap_m'][tail].min()),
            actual_platen_maximum_downward_travel_m=float(platen[0,2]-platen[:,2].min()),
            platen_maximum_xy_drift_m=float(np.linalg.norm(platen[:,:2]-platen[0,:2],axis=1).max()),
            platen_maximum_rotation_from_identity_rad=float(np.linalg.norm(Rotation.from_quat(platen[:,3:7]).as_rotvec(),axis=1).max()))
    np.savez_compressed(output/'recomputed_metrics.npz',time_s=time,native_step=step,
        table_force_n=forces,nominal_lumped_momentum_kg_m_s=momentum,adjacent_step_max_node_speed_m_s=pose_speed,**metrics)
    result={'schema':'independent_sponge_native_raw_review.v1','run':str(run),
        'reviewer_source':str(Path(__file__).resolve()),'reviewer_sha256':digest(__file__),
        'run_qualification_sha256':digest(run/'qualification.json'),'online_passed':online['passed'],
        'online_process_exit_code':online['process_exit_code'],'online_visual_readback':online.get('visual_readback'),
        'native_frame_count':len(time),'native_steps':len(time)-1,'dt_s':dt,
        'checks':{k:bool(v) for k,v in checks.items()},'passed':all(checks.values()),
        'scalar_recomputation_max_abs_errors':scalar_errors,'measurements':measurements,
        'first_inverted_tetrahedra':first_inverted,'contacts':contact,'checkpoints':chunks_report,
        'scope':['Raw every-native-step node q/qd, rigid states and per-point table normal impulses are independently recomputed.',
            'Checkpoint trajectory_complete=false is a crash-safety manifest marker; completeness is verified from final files/steps/plan.',
            'Nominal lumped mass momentum uses declared rest-volume mass; actual native nodal masses are not available in this archive.',
            'Node-in-platen tests are penetration witnesses, not a proof that no surface triangle intersects the platen.',
            'All-node recovery is compared in the moving pad frame; no additional best-fit alignment removes residual deformation.',
            'No robot task completion, grasp success, material calibration or exact prescribed-load claim.'],
        'files_sha256':{str(p):digest(p) for p in (run/'native_states.npz',run/'native_frames.jsonl',output/'recomputed_metrics.npz')}}
    (output/'report.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    for path in output.iterdir():
        path.chmod(0o444)
    print(json.dumps({'output':str(output),'passed':result['passed'],'checks':result['checks']}))
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    review(a.run,a.output)


if __name__=='__main__':
    main()
