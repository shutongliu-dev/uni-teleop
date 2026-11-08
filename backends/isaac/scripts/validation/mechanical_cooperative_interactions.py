"""Passive insertion and native support of the exact cooperative task geometry.

Drop-insertion starts already inserted by12mm. Outside-entry instead starts
with the aligned peg tip10mm above the real socket mouth. Both change only the
initial reset; subsequent motion is entirely gravity/contact driven, without
pose writes, external tool loads, robot strategy or cooperation scoring credit.
"""
from collections import deque
from copy import deepcopy
import argparse,json,math,os,traceback
from pathlib import Path
import numpy as np
from bench.isaacsim.release import digest,dump,task_inputs
from bench.paths import load_json, OUTPUT_ROOT
from bench.simulation.isaacsim_backend.native_task_bundle import add_bundle,joint_positions
from scenes.workrooms.mechanical_assembly.cooperative_composition import TASK_IDS,PHYSICS_DT
from scenes.workrooms.mechanical_assembly.nist_native_evidence import rotation,local,points
from scripts.maintenance.freeze_project_sources import verify_if_snapshot
from scripts.validation.board_game_runtime_smoke import gpu_selection
from scripts.validation.mechanical_nist_interactions import (array,components,signed_external_load,
    contact_records,friction_records,native_clock_check,passive_anchor_checks,terminal_checks)
PROJECT=Path(__file__).resolve().parents[2]
TABLE_PATH='/World/ProbeTable'
CAPACITY=65536
DWELL_S=.3


def cooperative_checks(rows,metadata,joints,masses,*,goal,mating_gap_m):
    result=terminal_checks(rows,metadata,joints,masses,goal=False,mating_gap_m=mating_gap_m)
    if not goal:return result
    p=metadata['task_parameters'];frames=list(rows)
    depths=[];geometry=[];support_pairs=set()
    for row in frames:
        sleeve,peg=row['objects']['sleeve'],row['objects']['peg']
        rel=local(peg['position_m'],sleeve);depth=.060-rel[2];depths.append(float(depth))
        axis=float(np.dot(rotation(sleeve['quaternion_wxyz'])[:,2],rotation(peg['quaternion_wxyz'])[:,2]))
        inside=all(np.all(np.abs(points(obj,next(x for x in metadata['objects'] if x['id']==name)['local_bounds_m'])-p['collection_center_m'])<=np.asarray(p['collection_half_extents_m'])+.006) for name,obj in row['objects'].items())
        geometry.append(bool(.030<=depth<=.058 and np.linalg.norm(rel[:2])<=.006 and axis>=math.cos(.30) and inside))
        for contact in row['raw_contacts']:
            vector=contact['impulse_vector_ns']
            if vector[2]>1e-10:
                support_pairs.add((contact['actor0'],contact['actor1']))
            elif vector[2]<-1e-10:support_pairs.add((contact['actor1'],contact['actor0']))
    paths=metadata['actors'];tray=metadata['fixtures']['collection_tray']
    sleeve_supported=any(a==paths['sleeve'] and (b==tray or b.startswith(tray+'/')) for a,b in support_pairs)
    peg_supported=(paths['peg'],paths['sleeve']) in support_pairs
    result['cooperative_geometry']={'passed':all(geometry) and sleeve_supported and peg_supported,
        'insertion_depth_range_m':[min(depths),max(depths)],'sleeve_tray_support':sleeve_supported,
        'peg_sleeve_support':peg_supported,'complete_terminal_geometry':all(geometry),
        'task_cooperation_credit':False}
    result['passed'] &= result['cooperative_geometry']['passed']
    return result


def review_passive_insertion(probe_output):
    """Reproduce the measured insertion distance from every retained native row."""
    root=Path(probe_output).resolve();source=root/'native_frames.jsonl'
    frames=[json.loads(line) for line in source.read_text().splitlines()]
    qualification=load_json(root/'qualification.json')
    if qualification.get('case')!='drop_insert':raise ValueError('Expected the passive insertion probe')
    depths=[.060-float(local(row['objects']['peg']['position_m'],row['objects']['sleeve'])[2]) for row in frames]
    complete=len(frames)==round(qualification['duration_s']/PHYSICS_DT)
    result={'schema':'mechanical_cooperative_passive_insertion_review.v1',
        'source_frames':str(source),'source_frames_sha256':digest(source),
        'source_probe_qualification_sha256':digest(root/'qualification.json'),
        'review_source_sha256':digest(__file__),'samples':len(frames),
        'initial_native_depth_m':depths[0],'final_native_depth_m':depths[-1],
        'actual_passive_insertion_progress_m':depths[-1]-depths[0],
        'maximum_reported_penetration_m':qualification['maximum_reported_penetration_m'],
        'native_support_stability_passed':qualification['passed'],
        'post_reset_pose_writes':qualification['post_reset_pose_writes'],
        'external_force_calls':qualification['external_force_calls'],
        'passed':qualification['passed'] and complete and depths[-1]-depths[0]>.035,
        'scope':'Actual gravity-driven insertion after a shallow reset, then two-body tray support; no robot cooperation or force calibration claim.'}
    return result


def outside_entry_checks(frames, metadata, *, duration_s, initial_objects):
    """Check the complete outside-to-inside path, independently of final fit."""
    parameters=metadata['task_parameters']
    mouth=np.asarray(parameters['socket_mouth_local_m'],float)
    tip=np.asarray(parameters['peg_tip_local_m'],float)
    axis=np.asarray(parameters['socket_axis_local'],float)
    def depth(objects):
        peg,sleeve=objects['peg'],objects['sleeve']
        tip_world=np.asarray(peg['position_m'])+rotation(peg['quaternion_wxyz'])@tip
        return float(np.dot(mouth-local(tip_world,sleeve),axis))
    depths=[depth(row['objects']) for row in frames]
    initial_depth=depth(initial_objects)
    indices={str(threshold):next((i for i,value in enumerate(depths) if value>=threshold),None)
             for threshold in (0.,.030,.0515)}
    continuous=(len(frames)==round(duration_s/PHYSICS_DT)
        and all(b['native_step']==a['native_step']+1 and b['native_time_s']>a['native_time_s']
                for a,b in zip(frames,frames[1:])))
    ordered=(all(i is not None for i in indices.values())
        and 0<indices['0.0']<indices['0.03']<=indices['0.0515'])
    tail=frames[-round(DWELL_S/PHYSICS_DT)-1:]
    paths=metadata['actors'];tray=metadata['fixtures']['collection_tray']
    supports={'peg_sleeve':0,'sleeve_tray':0}
    for row in tail:
        pairs=set()
        for contact in row['raw_contacts']:
            vertical=contact['impulse_vector_ns'][2]
            if vertical>1e-10:pairs.add((contact['actor0'],contact['actor1']))
            elif vertical<-1e-10:pairs.add((contact['actor1'],contact['actor0']))
        supports['peg_sleeve']+=(paths['peg'],paths['sleeve']) in pairs
        supports['sleeve_tray']+=any(a==paths['sleeve'] and (b==tray or b.startswith(tray+'/')) for a,b in pairs)
    dwell=tail[-1]['native_time_s']-tail[0]['native_time_s']
    complete_tail_support=all(count==len(tail) for count in supports.values()) and dwell>=DWELL_S-1e-9
    passed=(continuous and -.0101<=initial_depth<=-.0099 and depths[0]<-.009
        and ordered and .0515<=depths[-1]<=.0525 and complete_tail_support)
    return {'passed':bool(passed),'complete_contiguous_native_trace':bool(continuous),
        'initial_native_depth_m':initial_depth,'first_sample_depth_m':depths[0],
        'final_native_depth_m':depths[-1],'depth_range_m':[min(depths),max(depths)],
        'actual_progress_m':depths[-1]-initial_depth,'ordered_entry_milestones':bool(ordered),
        'first_crossings':{key:({'native_step':frames[i]['native_step'],'time_s':frames[i]['time_s'],
                               'depth_m':depths[i]} if i is not None else None) for key,i in indices.items()},
        'terminal_support_samples':len(tail),'terminal_support_duration_s':dwell,
        'terminal_loaded_support_sample_counts':supports,'complete_terminal_loaded_support':bool(complete_tail_support),
        'scope':'Axis-aligned, centered entry from outside the mouth through actual independent body motion; not arbitrary misalignment or robot insertion.'}


def review_outside_entry(probe_output):
    """Recheck every saved native frame and the actual post-reset baseline."""
    root=Path(probe_output).resolve();qualification=load_json(root/'qualification.json')
    if qualification.get('case')!='outside_entry':raise ValueError('Expected the outside-entry probe')
    frames_path=root/'native_frames.jsonl'
    frames=[json.loads(line) for line in frames_path.read_text().splitlines()]
    metadata=load_json(root/'metadata.json')
    checks=outside_entry_checks(frames,metadata,duration_s=qualification['duration_s'],
        initial_objects=qualification['native_initial_objects'])
    unforced=(qualification['post_reset_pose_writes']==0 and qualification['external_force_calls']==0
        and qualification['robot_present'] is False)
    penetration=qualification['maximum_reported_penetration_m']
    return {'schema':'mechanical_cooperative_outside_entry_review.v1',
        'passed':bool(qualification['passed'] and checks['passed'] and unforced and penetration<=.002),
        'checks':checks,'source_frames':str(frames_path),'source_frames_sha256':digest(frames_path),
        'source_probe_qualification_sha256':digest(root/'qualification.json'),
        'source_metadata_sha256':digest(root/'metadata.json'),'review_source_sha256':digest(__file__),
        'native_support_stability_passed':qualification['checks']['passed'],
        'maximum_reported_penetration_m':penetration,'post_reset_pose_writes':qualification['post_reset_pose_writes'],
        'external_force_calls':qualification['external_force_calls'],'robot_present':qualification['robot_present'],
        'scope':'Complete aligned outside-to-inside gravity motion and final real support; no robot strategy, task success or quantitative force calibration.'}


def run(args):
    source = verify_if_snapshot(PROJECT, required_modules=(
        'scripts.validation.mechanical_cooperative_interactions',
        'bench.simulation.isaacsim_backend.native_task_bundle',
        'scenes.workrooms.mechanical_assembly.nist_native_evidence'), entrypoint_file=Path(__file__))
    if source is None:
        raise ValueError('Run from a complete frozen project snapshot')
    output = args.output.resolve()
    if not output.is_relative_to(OUTPUT_ROOT):
        raise ValueError('Native evidence belongs under shared datasets')
    if not .5 <= args.duration <= 5. or abs(args.duration/PHYSICS_DT-round(args.duration/PHYSICS_DT)) > 1.e-8:
        raise ValueError('Bounded duration must be 0.5–5 seconds on the native grid')
    output.mkdir(parents=True, exist_ok=False)
    report = {'schema': 'mechanical_cooperative_object_interactions.v1', 'passed': False,
        'task_id': args.task_id, 'case': args.case, 'source_snapshot': str(PROJECT),
        'source': source, 'physics_dt_s': PHYSICS_DT, 'duration_s': args.duration,
        'robot_present': False, 'model_calls': 0, 'post_reset_pose_writes': 0,
        'external_force_calls': 0, 'task_success_claimed': False,
        'physics_accepted': False, 'full_functional_operation_qualified': False,
        'scope': 'Exact independent parts under gravity: passive insertion and released support. No hand manipulation or benchmark success.'}
    app = None
    try:
        report['gpu'] = gpu_selection(args.gpu_uuid)
        _, scene, task, _ = task_inputs(args.release, args.task_id)
        if scene['isaacsim'].get('solver_type', 'TGS') != 'TGS':
            raise ValueError('Probe requires the selected TGS task configuration')
        metadata = task['asset_metadata']
        bundle = deepcopy(scene['native_task_bundle'])
        report['release'] = {'path': str(args.release.resolve()),
                            'manifest_sha256': digest(args.release/'release_manifest.json')}
        report['asset'] = {'path': bundle['usd_path'], 'sha256': bundle['usd_sha256']}
        source_hash = digest(bundle['usd_path'])
        from isaacsim import SimulationApp
        app = SimulationApp({'headless': True, 'width': 64, 'height': 64, 'multi_gpu': False,
            'active_cuda_gpus': [0], 'max_gpu_count': 1, 'physics_gpu': 0,
            'fast_shutdown': True, 'extra_args': ['--/rtx/verifyDriverVersion/enabled=false']},
            experience=str(PROJECT/'bench/apps/workroom.headless.kit'))
        import omni.usd
        from isaacsim.core.api import World
        from pxr import Gf, UsdGeom, UsdPhysics, PhysxSchema
        from bench.simulation.isaacsim_backend.scene import rigid_material
        from pxr import UsdShade
        omni.usd.get_context().new_stage()
        stage = omni.usd.get_context().get_stage()
        world = World(stage_units_in_meters=1., physics_dt=PHYSICS_DT, rendering_dt=.02,
            backend='torch', device='cuda:0', physics_prim_path='/World/PhysicsScene')
        context = world.get_physics_context()
        context.enable_gpu_dynamics(True)
        context.set_broadphase_type('GPU')
        context.set_solver_type('TGS')
        physics = PhysxSchema.PhysxSceneAPI.Apply(stage.GetPrimAtPath(context.prim_path))
        physics.CreateEnableCCDAttr(False)
        physics.CreateGpuMaxRigidContactCountAttr(2**20)
        physics.CreateGpuMaxRigidPatchCountAttr(2**18)
        physics.CreateGpuFoundLostAggregatePairsCapacityAttr(2**14)
        physics.CreateGpuTotalAggregatePairsCapacityAttr(2**18)
        cube = UsdGeom.Cube.Define(stage, TABLE_PATH)
        cube.CreateSizeAttr(1.)
        cube.AddTranslateOp().Set(Gf.Vec3d(.65, 0., .76))
        cube.AddScaleOp().Set(Gf.Vec3f(1., 1., .04))
        UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
        material = rigid_material(stage, 'probe_table', .4)
        UsdShade.MaterialBindingAPI.Apply(cube.GetPrim()).Bind(
            material, UsdShade.Tokens.weakerThanDescendants, 'physics')
        info = add_bundle(stage, bundle, physics_scene_path=context.prim_path)
        if args.case in ('goal','drop_insert','outside_entry'):
            poses = deepcopy(metadata['diagnostic_goal_poses'])
            if args.case == 'drop_insert':poses['peg'][2]+=.040
            if args.case == 'outside_entry':
                # This case intentionally tests centered/aligned entry only.
                # The source peg origin is its tip; the source mouth is60mm up
                # from the sleeve datum. Do not reuse the preinserted start.
                parameters=metadata['task_parameters']
                if (parameters['peg_tip_local_m'] != [0,0,0]
                    or parameters['socket_axis_local'] != [0,0,1]
                    or poses['sleeve'][3:] != [0,0,0,1] or poses['peg'][3:] != [0,0,0,1]):
                    raise ValueError('Outside-entry requires the declared aligned source frames')
                poses['peg'][:3]=(np.asarray(poses['sleeve'][:3])
                    +np.asarray(parameters['socket_mouth_local_m'])+[0,0,.010]).tolist()
            if set(poses) != set(metadata['actors']):
                raise ValueError('Goal probe requires an explicit reset pose for every task body')
            for name, pose in poses.items():
                values = np.asarray(pose, float)
                if values.shape != (7,) or not np.isfinite(values).all() or abs(np.linalg.norm(values[3:])-1.) > 1.e-4:
                    raise ValueError('Goal reset requires finite world xyzw poses')
                xform = UsdGeom.Xformable(stage.GetPrimAtPath(metadata['actors'][name]))
                xform.ClearXformOpOrder()
                xform.AddTranslateOp().Set(Gf.Vec3d(*values[:3]))
                xform.AddOrientOp().Set(Gf.Quatf(float(values[6]), Gf.Vec3f(*values[3:6])))
            report['reset_only_goal_poses'] = poses
        dump(output/'metadata.json', metadata)
        stage.Export(str(output/'probe_initial.usda'))
        world.reset()
        sim = world.physics_sim_view
        body_view = sim.create_rigid_body_view(list(metadata['actors'].values()))
        body_paths = list(body_view.prim_paths)
        reverse = {path: name for name, path in metadata['actors'].items()}
        if body_view.count != len(reverse) or set(body_paths) != set(reverse):
            raise ValueError('Native body inventory is incomplete')
        names = [reverse[path] for path in body_paths]
        initial_transforms=array(body_view.get_transforms()).reshape(len(names),7)
        report['native_initial_objects']={name:{'position_m':pose[:3].tolist(),
            'quaternion_wxyz':[float(pose[6]),*pose[3:6].tolist()]}
            for name,pose in zip(names,initial_transforms)}
        masses = dict(zip(names, array(body_view.get_masses()).reshape(-1).tolist()))
        if any(not math.isfinite(value) or value <= 0 for value in masses.values()):
            raise ValueError('Native mass inventory is invalid')
        fixed = info['fixtures'].get('native_bundle_fixed', {}).get('collision_paths', [])
        filters = [*metadata['actors'].values(), *fixed, TABLE_PATH]
        contact = sim.create_rigid_contact_view(list(metadata['actors'].values()),
            filter_patterns=[list(filters) for _ in metadata['actors']], max_contact_data_count=CAPACITY)
        sensors = list(contact.sensor_paths)
        mapped = [list(row) for row in contact.filter_paths]
        if set(sensors) != set(metadata['actors'].values()) or any(row != mapped[0] for row in mapped):
            raise ValueError('Native contact sensor/filter mapping differs')
        filters = mapped[0]
        if set(filters) != set([*metadata['actors'].values(), *fixed, TABLE_PATH]):
            raise ValueError('Native contact filters are incomplete')
        sensor_names = [reverse[path] for path in sensors]
        groups = components(metadata['actors'], info['joints'])
        report['contact_mapping'] = {'sensor_paths': sensors, 'filter_paths': filters,
            'sparse_matrix_semantics': 'Unlisted sensor/filter entries are exact zero vectors. All signs retained.',
            'contact_force_matrix_semantics': 'Native normal-contact matrix; independently retained as originally reported.',
            'legacy_component_external_force_n_semantics': 'Normal-only compatibility field, identical to component_normal_force_n.',
            'friction_semantics': 'Native get_friction_data(dt=1) signed world-space impulses, divided by the same declared dt as the normal matrix.',
            'support_force_semantics': 'Complete signed external normal plus friction vector sum; internal joint-connected component pairs excluded.'}
        report['native_masses_kg'] = masses
        report['joint_connected_components'] = [list(group) for group in groups]
        baseline_step, baseline_time = int(world.current_time_step_index), float(world.current_time)
        configured_dt = float(world.get_physics_dt())
        report['native_clock'] = {'baseline_step': baseline_step, 'baseline_time_s': baseline_time,
            'configured_dt_s': configured_dt, 'declared_dt_s': PHYSICS_DT,
            'callback_dt_s': float(np.float32(PHYSICS_DT)),
            'clock_source': 'World physics callback accumulates its float32 step_size in binary64; actual times are retained.'}
        previous_time = baseline_time
        tail = deque(maxlen=round(DWELL_S/PHYSICS_DT)+1)
        max_penetration = 0.
        with (output/'native_frames.jsonl').open('x') as stream:
            for tick in range(1, round(args.duration/PHYSICS_DT)+1):
                world.step(render=False)
                native_step, native_time = int(world.current_time_step_index), float(world.current_time)
                clock_check = native_clock_check(tick=tick, step=native_step, time_s=native_time,
                    baseline_step=baseline_step, baseline_time=baseline_time,
                    previous_time=previous_time, configured_dt=float(world.get_physics_dt()))
                if not clock_check['passed']:
                    report['native_clock_failure'] = clock_check
                    dump(output/'native_clock_failure.json', clock_check)
                    raise ValueError('Missing or inconsistent native sampling step: '+json.dumps(clock_check))
                previous_time = native_time
                objects = {}
                poses = array(body_view.get_transforms()).reshape(len(names), 7)
                velocities = array(body_view.get_velocities()).reshape(len(names), 6)
                if not np.isfinite(poses).all() or not np.isfinite(velocities).all():
                    raise ValueError('Nonfinite native body state')
                for name, pose, velocity in zip(names, poses, velocities):
                    objects[name] = {'position_m': pose[:3].tolist(),
                        'quaternion_wxyz': [float(pose[6]), *pose[3:6].tolist()],
                        'linear_velocity_m_s': velocity[:3].tolist(),
                        'angular_velocity_rad_s': velocity[3:].tolist()}
                matrix = array(contact.get_contact_force_matrix(dt=PHYSICS_DT))
                if matrix.shape != (len(sensors), len(filters), 3) or not np.isfinite(matrix).all():
                    raise ValueError('Native contact matrix shape/values changed')
                raw = contact_records(contact, sensors, filters)
                raw_friction, friction_matrix = friction_records(contact, sensors, filters)
                total_matrix = matrix.astype(np.float64)+friction_matrix
                max_penetration = max(max_penetration, max(
                    [-min(row['separations_m']) for row in raw], default=0.))
                joint_state = joint_positions(objects, info['joints'])
                nonzero = np.argwhere(np.any(matrix != 0., axis=2))
                friction_nonzero = np.argwhere(np.any(friction_matrix != 0., axis=2))
                total_nonzero = np.argwhere(np.any(total_matrix != 0., axis=2))
                row = {'native_step': native_step, 'native_time_s': native_time,
                    'time_s': native_time-baseline_time, 'nominal_time_s': tick*PHYSICS_DT,
                    'native_increment_s': clock_check['actual_increment_s'],
                    'native_clock_cumulative_error_s': clock_check['cumulative_error_s'],
                    'objects': objects, 'joints': joint_state,
                    'passive_links': passive_anchor_checks(joint_state),
                    'raw_contacts': raw,
                    'raw_friction': raw_friction,
                    'contact_matrix_nonzero': [[int(i), int(j), *matrix[i, j].tolist()] for i, j in nonzero],
                    'friction_matrix_nonzero': [[int(i), int(j), *friction_matrix[i, j].tolist()]
                                               for i, j in friction_nonzero],
                    'total_contact_matrix_nonzero': [[int(i), int(j), *total_matrix[i, j].tolist()]
                                                     for i, j in total_nonzero],
                    'component_normal_force_n': {str(i): signed_external_load(
                        matrix, sensor_names, filters, group, metadata['actors']).tolist()
                        for i, group in enumerate(groups)},
                    'component_friction_force_n': {str(i): signed_external_load(
                        friction_matrix, sensor_names, filters, group, metadata['actors']).tolist()
                        for i, group in enumerate(groups)},
                    'component_external_force_n': {str(i): signed_external_load(
                        matrix, sensor_names, filters, group, metadata['actors']).tolist()
                        for i, group in enumerate(groups)},
                    'component_total_external_force_n': {str(i): signed_external_load(
                        total_matrix, sensor_names, filters, group, metadata['actors']).tolist()
                        for i, group in enumerate(groups)}}
                if (int(world.current_time_step_index), float(world.current_time)) != (native_step, native_time):
                    raise ValueError('Observation unexpectedly advanced native time')
                stream.write(json.dumps(row, separators=(',', ':'), allow_nan=False)+'\n')
                tail.append(row)
                if max_penetration > .006:
                    raise ValueError('Native penetration exceeds the declared practical 6 mm bound')
        report['checks'] = cooperative_checks(tail, metadata, info['joints'], masses, goal=args.case in ('goal','drop_insert','outside_entry'),
            mating_gap_m=task['public_rules']['referee']['thresholds'].get('maximum_mating_gap_m', .0005))
        report['maximum_reported_penetration_m'] = max_penetration
        report['native_substeps'] = round(args.duration/PHYSICS_DT)
        report['source_asset_unchanged'] = digest(bundle['usd_path']) == source_hash
        report['passed'] = bool(report['checks']['passed'] and report['source_asset_unchanged'])
        if args.case == 'outside_entry':
            frames=[json.loads(line) for line in (output/'native_frames.jsonl').read_text().splitlines()]
            report['outside_entry_checks']=outside_entry_checks(frames,metadata,duration_s=args.duration,
                initial_objects=report['native_initial_objects'])
            report['passed'] &= report['outside_entry_checks']['passed']
    except BaseException as exc:
        traceback.print_exc()
        report['error'] = {'type': type(exc).__name__, 'message': str(exc)}
    finally:
        report['files_sha256'] = {p.name: digest(p) for p in output.iterdir() if p.is_file()}
        dump(output/'qualification.json', report)
        print(json.dumps({'passed': report['passed'], 'output': str(output)}), flush=True)
        if app is not None:
            app.close()
    return 0 if report['passed'] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--release', required=True, type=Path)
    parser.add_argument('--task-id', choices=TASK_IDS, required=True)
    parser.add_argument('--case', choices=('canonical', 'goal', 'drop_insert', 'outside_entry'), required=True)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--gpu-uuid', required=True)
    parser.add_argument('--duration', type=float, default=1.)
    raise SystemExit(run(parser.parse_args()))


if __name__ == '__main__':
    main()
