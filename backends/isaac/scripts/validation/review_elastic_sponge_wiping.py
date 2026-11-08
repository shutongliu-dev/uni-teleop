"""Read-only full native wiping replay, preserving the original process outcome."""
import argparse,json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from pxr import Usd
from bench.isaacsim.release import digest,checked,dump
from bench.paths import load_json, OUTPUT_ROOT
from scripts.maintenance.freeze_project_sources import verify_if_snapshot
from scripts.validation.elastic_sponge_clearance import certify_stage
from scripts.validation import elastic_sponge_wiping as motion
from scripts.validation.elastic_sponge_compression import quiet
from scenes.workrooms.kitchen.wiping import ContactWiping
from scenes.workrooms.kitchen.rules import thresholds
PROJECT=Path(__file__).resolve().parents[2]
TABLE='/World/FreeflightTableBoundary'


def run(args):
    source=verify_if_snapshot(PROJECT,required_modules=('scripts.validation.review_elastic_sponge_wiping','scripts.validation.elastic_sponge_clearance','scripts.validation.elastic_sponge_wiping','scenes.workrooms.kitchen.wiping'),entrypoint_file=Path(__file__))
    if source is None or Path.cwd().resolve()!=PROJECT:raise ValueError('Use complete sealed review source')
    directory=args.run.resolve();old=load_json(directory/'qualification.json');init=load_json(directory/'native_initialization.json')
    if old['case']!='wiping':raise ValueError('Exact native wiping case required')
    for name,sha in old['evidence_sha256'].items():checked(directory/name,sha)
    oldroot=Path(old['source']['snapshot_root']);oldsource=load_json(oldroot/'source_snapshot_manifest.json')
    checked(oldroot/'source_snapshot_manifest.json',old['identity']['source_manifest_sha256'])
    for name,sha in oldsource['files_sha256'].items():checked(oldroot/name,sha)
    # The only review behavior changed is conservative geometric refinement.
    unchanged=('scripts/validation/elastic_sponge_wiping.py','scripts/validation/elastic_sponge_compression.py','scenes/workrooms/kitchen/wiping.py','scenes/workrooms/kitchen/rules.py')
    for name in unchanged:assert digest(PROJECT/name)==digest(oldroot/name),name
    asset=Path(old['asset_manifest']);checked(asset,old['identity']['asset_manifest_sha256']);manifest=load_json(asset)
    checked(asset.parent/'metadata.json',manifest['metadata_sha256']);checked(asset.parent/'task.usda',manifest['usd_sha256']);meta=load_json(asset.parent/'metadata.json')
    rows=[json.loads(x) for x in (directory/'native_frames.jsonl').read_text().splitlines()];raw=dict(np.load(directory/'native_states.npz'))
    count=motion.STEPS+1
    assert len(rows)==len(raw['time_s'])==count==13201 and np.all(np.diff(raw['native_step'])==1)
    assert all(np.isfinite(x).all() for x in raw.values())
    assert np.allclose(np.diff(raw['time_s']),float(np.float32(.0005)),atol=1e-10,rtol=0)
    chunk_manifest=load_json(directory/'native_chunks_manifest.json');parts=[];promotions=[]
    for chunk in chunk_manifest['chunks']:
        path=Path(chunk['path']);checked(path,chunk['sha256']);part=dict(np.load(path));parts.append(part)
        for key in raw:
            if part[key].dtype!=raw[key].dtype:promotions.append({'chunk':str(path),'key':key,'chunk_dtype':str(part[key].dtype),'archive_dtype':str(raw[key].dtype)})
    for key in raw:assert np.array_equal(raw[key],np.concatenate([x[key] for x in parts])),key
    receipt=load_json(directory.parent/(directory.name+'_launch')/'receipts.json');assert receipt[-1]['exit_code']==old['process_exit_code']
    paths=init['native_members']['paths'];bypath={x['body_path']:x for x in meta['objects']};specs=[bypath[x] for x in paths]
    assert len(paths)==28 and set(paths)==set(meta['actors'].values())
    mass=np.asarray(init['native_members']['masses_kg'],float);com=np.asarray(init['native_members']['com_poses_local'],float)[:,:3];inertia=np.asarray(init['native_members']['inertias_kg_m2'],float)
    for i,spec in enumerate(specs):
        assert np.isclose(mass[i],spec['mass_kg'],rtol=2e-6,atol=1e-10)
        assert np.allclose(com[i],spec['center_of_mass_m'],atol=2e-8,rtol=1e-5)
        rot=Rotation.from_quat(spec['principal_axes_xyzw']).as_matrix();expect=rot@np.diag(spec['diagonal_inertia_kg_m2'])@rot.T
        assert np.allclose(inertia[i],expect,rtol=2e-4,atol=1e-10)
    assert np.isclose(mass.sum(),.0792,atol=1e-9,rtol=0)
    pad=paths.index(meta['actors']['sponge']);q=raw['body_q'];v=raw['body_qd'];centers=Rotation.from_quat(q[:,pad,3:].astype(float)).apply(np.tile(com[pad],(count,1)))+q[:,pad,:3]
    point_error=np.linalg.norm(raw['force_position_world_m'][1:].astype(float)-centers[:-1].astype(np.float32),axis=1)
    assert point_error.max()<2e-7
    expected_force=np.array([np.zeros(3,np.float32)]+[motion.force(i,meta,q[i-1,pad],v[i-1,pad])[0] for i in range(1,count)])
    assert np.array_equal(raw['force_world_n'],expected_force)
    wiping=ContactWiping(meta,thresholds('surface_wiping'),.0005);coverage_identical=True;maximum_contact_residual=0.
    for i,row in enumerate(rows):
        contact=row['pad_contact'];loaded=TABLE in contact['other_actor_paths'] and contact['net_contact_force_n'][2]>0
        scalar=np.array(contact['scalar_impulses_ns']);normals=np.asarray(contact['normals']).reshape(-1,3)
        point_force=(scalar[:,None]*normals).sum(0)/.0005
        maximum_contact_residual=max(maximum_contact_residual,float(np.linalg.norm(point_force-contact['net_contact_force_n'])))
        result=wiping.update({'position_m':q[i,pad,:3].tolist(),'quaternion_wxyz':[float(q[i,pad,6]),*q[i,pad,3:6].tolist()]},loaded_contact=loaded,baseline=i==0)
        coverage_identical&=result==row['coverage']
    assert coverage_identical
    checks,measures,derived=motion.evaluate(raw,rows,mass,pad,TABLE)
    stage=Usd.Stage.Open(str(directory/'initial.usda'));proof,certs=certify_stage(stage,specs,q,init['static_collision_bounds'],excluded_paths=(TABLE,),threshold=.005)
    checks.update(actual_convex_clearance_5mm=proof['passed'],all_native_inputs_and_complete_raw_verified=True,actual_pre_step_pad_com_force_points=bool(point_error.max()<2e-7),
        force_law_replayed_exactly=True,soil_all_frames_replayed_exactly=coverage_identical,
        native_contact_point_buffer_consistent=maximum_contact_residual<=1e-4,
        pad_contacts_only_actual_table=all(set(r['pad_contact']['other_actor_paths'])<={TABLE} and r['pad_contact']['finite'] for r in rows),
        no_gross_lattice_distortion=all(x['minimum_neighbor_center_distance_ratio']>=.5 and x['maximum_neighbor_center_distance_ratio']<=1.5 and x['maximum_relative_neighbor_rotation_rad']<=.5 for x in rows),
        original_solver_tgs=init['scene_properties']['physxScene:solverType']=='TGS')
    out=args.output.resolve()
    if not out.is_relative_to(OUTPUT_ROOT):raise ValueError('Use shared evidence storage')
    out.mkdir(parents=True,exist_ok=False);np.savez_compressed(out/'clearance_certificates.npz',**certs)
    logs=[]
    for p in sorted((directory.parent/(directory.name+'_launch')).glob('*.log')):
        text=p.read_text(errors='replace');logs.append({'path':str(p),'sha256':digest(p),'error_lines':[line for line in text.splitlines() if '[Error]' in line or 'Traceback' in line]})
    refs=[directory/'qualification.json',directory/'native_initialization.json',directory/'native_states.npz',directory/'native_chunks_manifest.json',directory/'native_frames.jsonl',directory/'initial.usda',directory.parent/(directory.name+'_launch')/'receipts.json',asset]
    report={'schema':'elastic_wiping_full_raw_refinement.v1','independent_physical_review_passed':bool(all(checks.values())),
        'original_qualification_passed':old['passed'],'original_native_exit_code':old['process_exit_code'],'original_failed_checks':[k for k,x in old['checks'].items() if not x],
        'review_source':source,'review_source_manifest_sha256':digest(PROJECT/'source_snapshot_manifest.json'),'native_source_identity':old['identity'],
        'unchanged_motion_scoring_modules':{name:digest(PROJECT/name) for name in unchanged},'native_frames':count,'native_steps':count-1,'native_member_count':28,
        'all_chunk_values_identical_to_archive':True,'dtype_promotions_preserving_exact_values':promotions,'checks':{k:bool(x) for k,x in checks.items()},'measurements':measures,
        'native_mass_kg':float(mass.sum()),'maximum_force_point_vs_pre_step_com_m':float(point_error.max()),'maximum_raw_contact_point_sum_vs_net_force_n':maximum_contact_residual,
        'actual_convex_clearance':proof,'certificate_sha256':digest(out/'clearance_certificates.npz'),'logs':logs,'references':[{'path':str(p),'sha256':digest(p)} for p in refs],
        'model_calls':0,'new_gpu_steps':0,'controller_changes':False,'threshold_changes':False,
        'scope':'Original native process exit1 and qualificationfalse remain permanently unchanged. Its sole failed AABB screen was conservative; complete actual-convex separation now verifies the same5mm gate with1micrometre guard over every native frame. Force, physical motion, all28member quietness and original soil scoring are replayed without modification. This is object engineering evidence, not robot task success.'}
    dump(out/'report.json',report);dump(out/'seal.json',{'report_sha256':digest(out/'report.json'),'certificate_sha256':digest(out/'clearance_certificates.npz')})
    for p in out.iterdir():p.chmod(0o444)
    out.chmod(0o555);print(json.dumps({'passed':report['independent_physical_review_passed'],'original_native_exit_code':old['process_exit_code'],'output':str(out),'report_sha256':digest(out/'report.json')}))
    return 0 if report['independent_physical_review_passed'] else 1


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    raise SystemExit(run(parser.parse_args()))
