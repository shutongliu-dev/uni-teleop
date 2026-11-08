"""Audit finite pinch poses of all explicitly pinned laundry installations."""
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import argparse
import json
import math
import numpy as np
from bench.control.factory import public_robot_geometry, robot_configuration
from bench.geometry import rpy_matrix
from bench.isaacsim.release import checked, digest, dump, validate
from bench.paths import load_json
from scenes.workrooms.laundry_room.robot_configurations import stand_geometry
from scripts.diagnostics.check_bimanual_options import CollisionBounds, gaps
from scripts.diagnostics.check_bimanual_arm_self_collision import ArmSelfCollision
from scripts.validation.laundry_contact_pinch import pinch_postures, hand_collision_vertices, local_hand_points


def targets(task):
    rows = []
    meta = task['asset_metadata']
    for cloth in meta['cloths']:
        points = np.asarray(cloth['initial_vertices_m'])
        groups = cloth.get('boundary_features', {'outline': list(range(len(points)))})
        for name, ids in groups.items():
            p = points[ids]
            chosen = set()
            for axis in (0, 1):
                chosen.update((ids[int(p[:,axis].argmin())], ids[int(p[:,axis].argmax())]))
            if name == 'outline' and 'cells_xy' in cloth:
                nx, ny = cloth['cells_xy']
                chosen.update((0, nx, ny*(nx+1), (ny+1)*(nx+1)-1))
            _, _, basis = np.linalg.svd(p[:,:2]-p[:,:2].mean(0), full_matrices=False)
            yaw = math.atan2(basis[0,1], basis[0,0])
            angles = ([0, math.pi/2, math.pi, 3*math.pi/2] if name in ('outline', 'crotch') else
                      [yaw, yaw+math.pi, yaw+math.pi/6, yaw-math.pi/6, yaw+math.pi+math.pi/6, yaw+math.pi-math.pi/6])
            for index in sorted(chosen):
                rows.append({'label': cloth['id']+'/'+name+'/'+str(index), 'point_m': points[index].tolist(),
                             'yaw_candidates_rad': angles, 'physical_thickness_m': cloth['physical_thickness_m']})
    return rows


def clearance(checker, ready, model, side, joints, meshes):
    boxes = {**ready, side: checker.boxes(side, joints)}
    raw = checker.check(boxes)
    fk = model.arm.forward(joints, model.arm_base)[model.arm_tip] @ model.wrist_offset
    actual_minimum_z = min(float((q @ fk[:3,:3].T+fk[:3,3])[:,2].min())
                          for q in local_hand_points(model, meshes, joints).values())
    kept = gaps(boxes[side], checker.obstacles)
    replaced = []
    for i, (body, _) in enumerate(checker.robot_shapes[side]):
        if not body.startswith(side+'_'):
            continue
        for j, name in enumerate(checker.obstacle_names):
            if name.endswith('/Geometry/worktop/Collider'):
                kept[i,j] = np.inf
                replaced.append([i,j])
    other = {k:v for k,v in raw['checks'].items() if k != side+'_environment'}
    good = bool(kept.min()>.001 and actual_minimum_z>=.90 and all(v['clear'] for v in other.values()))
    return {'passed': good, 'raw_usd_bounds': raw, 'other_environment_gap_m': float(kept.min()),
            'actual_hand_mesh_minimum_z_m': actual_minimum_z, 'hand_table_replaced_pairs': replaced}


def audit(entry):
    reference = entry['source_release']; root = Path(reference['path'])
    checked(root/'release_manifest.json', reference['manifest_sha256'])
    manifest = validate(root, assets=False)
    scene, task, initial = [load_json(root/name) for name in ('scene.json','task.json','initial_state.json')]
    config = task['robot_configuration_id']
    if manifest['configuration_id'] != config:
        raise ValueError('Configuration identity differs')
    return audit_inputs(scene, task, initial, reference)


def _arm_chains(own, initial, wrist, center, tcp, restart_seeds=None):
    """Original three endpoint solves, with an optional bounded audit restart."""
    def chain(seed):
        seed=dict(seed); solutions=[]; calls=0
        for rise in (.10,0.,.05):
            pose=wrist.copy();pose[2,3]+=rise
            local=np.linalg.inv(own.arm_base) @ pose @ np.linalg.inv(own.wrist_offset)
            solved=own.arm.inverse(local,own.arm_tip,seed,iterations=180,multi_starts=0,
                position_tolerance=.0005,rotation_tolerance=.005)
            calls+=1
            if not solved['converged']:break
            seed.update(solved['joint_positions_rad'])
            fk=own.arm.forward(seed,own.arm_base)[own.arm_tip] @ own.wrist_offset
            error=float(np.linalg.norm(fk[:3,3]+fk[:3,:3]@center-(tcp+[0,0,rise])))
            solutions.append({'rise_m':rise,'actual_pinch_error_m':error,**solved})
        return solutions,calls
    first,calls=chain({name:initial[name] for name in own.arm.names})
    yield first,calls,None
    if restart_seeds is not None:
        origin={**initial,**(first[0]['joint_positions_rad'] if first else {})}
        for seed in restart_seeds(own,origin):
            solutions,calls=chain(seed['arm_joint_positions_rad'])
            yield solutions,calls,seed


def audit_inputs(scene, task, initial, reference, *, checker_type=None,
                 self_checker_type=None, arm_restart_seeds=None):
    """Check resolved authored inputs without assuming a canonical release file."""
    config = task['robot_configuration_id']
    model = public_robot_geometry(robot_configuration(scene))
    models = {'right':model} if config=='single_arm' else model.sides
    initial = initial['joint_positions_rad']
    version = scene['robot_configuration'].get('layout_version', 1)
    checker = (CollisionBounds if checker_type is None else checker_type)(
        scene, scene['robot'], models, initial, stand_geometry(config, version=version))
    self_checker = (ArmSelfCollision if self_checker_type is None else self_checker_type)(checker)
    ready = {side:checker.boxes(side, initial) for side in models}
    prepared = {}
    for side, own in models.items():
        center, postures = pinch_postures(own, side)
        meshes = hand_collision_vertices(own)
        clouds = [np.concatenate(list(local_hand_points(own, meshes, p['hand_posture_rad']).values()))-center for p in postures]
        heights = {pitch: .9003-min(float((p @ rpy_matrix([math.pi,math.radians(pitch),0]).T)[:,2].min())
                                  for p in clouds) for pitch in (60,45,75)}
        prepared[side] = (center, postures, meshes, heights)
    result = {'task_id':task['id'], 'configuration_id':config, 'source_release':reference,
              'layout_version':version, 'stand_clearance':checker.stand_check(),
              'ready_clearance':checker.check(ready), 'ready_arm_self_clearance':self_checker.check(initial), 'targets':[]}
    for target in targets(task):
        sides = {}
        p = np.asarray(target['point_m'])
        # Keep the calibrated relative pad height for raised folds as well as flat cloth.
        cloth_rise = max(0., float(p[2]-(task['asset_metadata']['table_top_z_m']+target['physical_thickness_m']/2)))
        for side, own in models.items():
            center, postures, meshes, heights = prepared[side]
            best = None; failures = []; calls = 0
            for pitch in (60,45,75):
                for yaw in target['yaw_candidates_rad']:
                    rotation = rpy_matrix([math.pi,math.radians(pitch),yaw])
                    tcp = np.asarray([*p[:2], heights[pitch]+cloth_rise])
                    wrist = np.eye(4); wrist[:3,:3]=rotation; wrist[:3,3]=tcp-rotation@center
                    for solutions,count,restart in _arm_chains(own,initial,wrist,center,tcp,arm_restart_seeds):
                        calls+=count
                        if len(solutions)!=3 or max(s['actual_pinch_error_m'] for s in solutions)>.001:
                            continue
                        audits=[]
                        for arm_index, hand_index in ((0,0),(1,0),(1,30),(1,60),(2,60)):
                            joints={**initial,**solutions[arm_index]['joint_positions_rad'],**postures[hand_index]['hand_posture_rad']}
                            audits.append({'arm_index':arm_index,'hand_index':hand_index,
                                           **clearance(checker,ready,own,side,joints,meshes)})
                            audits[-1]['nonadjacent_arm_convexes'] = self_checker.side(side, joints)
                            audits[-1]['passed'] &= audits[-1]['nonadjacent_arm_convexes']['clear']
                        record={'yaw_deg':math.degrees(yaw),'pitch_deg':pitch,'tcp_m':tcp.tolist(),
                                'cloth_height_compensation_m':cloth_rise,'arm_ik':solutions,'clearance':audits}
                        if restart is not None:record['restart_selection']=restart
                        if all(a['passed'] for a in audits):
                            best=record; break
                        failures.append(record)
                    if best is not None:break
                if best is not None: break
            sides[side]={'passed':best is not None,'solution':best,'ik_solver_calls':calls,
                         'first_rejected_clearance_candidate':failures[0] if failures else None}
        result['targets'].append({**target,'sides':sides,'passed':any(s['passed'] for s in sides.values())})
    result['dependencies']=checker.dependencies
    result['hand_collision_meshes']={s:[r['source'] for r in prepared[s][2]] for s in models}
    result['target_count']=len(result['targets'])
    result['targets_passed']=sum(t['passed'] for t in result['targets'])
    result['all_discrete_checks_passed']=(result['stand_clearance']['clear'] and result['ready_clearance']['clear']
                                         and result['ready_arm_self_clearance']['clear']
                                         and result['targets_passed']==result['target_count'])
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--index',type=Path,required=True); parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--workers',type=int,default=5)
    args=parser.parse_args()
    if args.output.exists(): raise ValueError('Choose a new report')
    from scripts.maintenance.freeze_project_sources import verify_if_snapshot
    snapshot=verify_if_snapshot(Path(__file__).resolve().parents[2],
        required_modules=('scripts.validation.laundry_configuration_workspace',
                          'scripts.validation.laundry_contact_pinch'), entrypoint_file=Path(__file__))
    source=load_json(args.index)
    if not source['complete']: raise ValueError('Incomplete configuration matrix')
    options=[option for row in source['tasks'] for option in row['configurations'].values()]
    report={'schema':'laundry_configured_pinch_workspace.v1','complete':False,
            'index':{'path':str(args.index.resolve()),'sha256':digest(args.index)},
            'source_script':{'path':str(Path(__file__).resolve()),'sha256':digest(__file__)},
            'source_snapshot':snapshot,
            'scope':'Actual original thumb/index FK and original collision meshes at finite approach, contact, closure and lift poses. Raised folds retain pad-height compensation. Each target requires one usable hand while the other stays ready. Table contact uses original hand mesh world bounds; other body pairs use conservative USD AABB clearance >1mm. Nonadjacent FR3 arm convexes reject common interior sphere radius >1mm. This does not qualify continuous paths, hand internal or hand-vs-arm self collision, native cooked contacts, cooperative simultaneous motion, successful material capture or task manipulation.',
            'native_qualified':False,'manipulation_qualified':False,'results':[]}
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        pending=[pool.submit(audit, option) for option in options]
        for future in as_completed(pending):
            row=future.result();report['results'].append(row);dump(args.output,report)
            print(json.dumps({k:row[k] for k in ('task_id','configuration_id','targets_passed','target_count','all_discrete_checks_passed')}),flush=True)
    report['complete']=True;dump(args.output,report)

if __name__=='__main__':main()
