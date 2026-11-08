"""Native engineering folds with local attachments and measured dual hands.

The optional parked-robot mode isolates cloth feasibility. Only tracking mode
links attachment motion to measured native fingertips; neither is friction
grasping or an autonomous Bench policy. No cloth state setters are used.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import itertools
import json
from pathlib import Path
import time
import traceback

import numpy as np

from bench.paths import load_json
from scripts.validation.laundry_single_garment_native import (
    DATASETS, PROJECT, NativeArrays, advance, dump, pin, robot_readback)
from scripts.validation.laundry_garment_folding_motion import build_folding_motion
from scripts.validation.laundry_engineering_motion import minimum_jerk


class RobotDriver:
    def __init__(self, env, scene, clearance, wrist_yaw_offset=0., wrist_yaw_by_side=None,
                 approach_height=.12):
        from bench.control.factory import public_robot_geometry, robot_configuration
        self.env, self.clearance = env, float(clearance)
        self.wrist_yaw_offset = float(wrist_yaw_offset)
        self.wrist_yaw_by_side = dict(wrist_yaw_by_side or {})
        self.approach_height = float(approach_height)
        self.models = public_robot_geometry(robot_configuration({'robot': scene['robot']})).sides
        self.tip_indices = {side: env.body_names.index(side+'_middle_fingertip') for side in self.models}
        measured = robot_readback(env)['joint_q']
        self.seed = {name: float(np.clip(value, *env.joint_limits[name]))
                     for name,value in zip(env.joint_names, measured)}
        self.park = dict(self.seed)
        self.receipts = []
        self.assignment_failures = []

    def tips(self):
        body = robot_readback(self.env)['robot_body_q']
        return {side: body[index, :3].astype(float) for side, index in self.tip_indices.items()}

    def solve(self, points, *, seed=None):
        from scenes.workrooms.laundry_room.robot_layout import hand_probe_transform
        result = dict(self.seed if seed is None else seed)
        errors = {}
        for side, point in points.items():
            model = self.models[side]
            wrist = hand_probe_transform(side, point, fingertip_clearance_m=self.clearance)
            yaw = self.wrist_yaw_by_side.get(side, self.wrist_yaw_offset)
            c, s = np.cos(yaw), np.sin(yaw)
            wrist[:3,:3] = np.array([[c,-s,0],[s,c,0],[0,0,1]]) @ wrist[:3,:3]
            tip = model.public_geometry({name: result[name] for name in model.names})['fingertip_positions_m']['middle']
            wrist[:3,3] = np.asarray(point)+[0,0,self.clearance]-wrist[:3,:3] @ np.asarray(tip)
            offset = model.wrist_offset if model.wrist_offset is not None else np.eye(4)
            target = np.linalg.inv(model.arm_base) @ wrist @ np.linalg.inv(offset)
            solved = model.arm.inverse(target, model.arm_tip, result,
                iterations=180, multi_starts=0, position_tolerance=.001, rotation_tolerance=.008)
            if not solved['converged']:
                raise ValueError('Unreachable '+side+' hand target: '+json.dumps(solved))
            result.update(solved['joint_positions_rad'])
            errors[side] = solved
        return result, errors

    def assign(self, origins, candidate=None):
        candidates = []
        current = self.tips()
        assignments = [tuple(candidate['anchor_assignment'])] if candidate else itertools.permutations(('left','right'))
        for sides in assignments:
            if len(sides) != 2 or set(sides) != {'left','right'}:
                raise ValueError('A warm-start candidate must assign one distinct hand to each patch')
            points = {side: np.asarray(point)+[0, 0, self.approach_height] for side, point in zip(sides, origins)}
            try:
                seed = dict(self.seed)
                if candidate:
                    for side, arm_seed in candidate['arm_joint_positions_rad'].items():
                        if set(arm_seed) != set(self.models[side].arm.names):
                            raise ValueError('Warm-start candidates may only override original arm joints')
                        seed.update(arm_seed)
                joints, errors = self.solve(points, seed=seed)
                self.solve(dict(zip(sides, origins)), seed=joints)
            except ValueError as error:
                self.assignment_failures.append({'sides':list(sides),
                    'observed_origins_m':np.asarray(origins).tolist(),
                    'hover_points_m':{key:value.tolist() for key,value in points.items()},
                    'message':str(error)})
                continue
            cost = sum(np.linalg.norm(current[side]-point) for side, point in points.items())
            candidates.append((cost, sides, joints, errors))
        if not candidates:
            raise ValueError('Neither dual-hand assignment reaches the observed grasp patches')
        cost, sides, joints, errors = min(candidates, key=lambda row: row[0])
        self.receipts.append({'sides': sides, 'approach_distance_sum_m': cost, 'approach_ik': errors,
            'wrist_yaw_by_side_rad':dict(self.wrist_yaw_by_side)})
        return sides, joints

    def command(self, joints):
        env = self.env
        active = {name: float(joints[name]) for name in env.actuated_joint_names}
        for name, value in active.items():
            if not env.joint_limits[name][0] <= value <= env.joint_limits[name][1]:
                raise ValueError('Joint target outside original limit: '+name)
        expanded = env._expand_mimic_targets(active)
        env._targets = np.asarray([[expanded[name] for name in env.joint_names]], np.float32)
        env._robot_view.set_dof_position_targets(env._tensor(env._targets), env._one)
        self.seed = {**joints, **expanded}


def run(args):
    from scripts.maintenance.freeze_project_sources import verify_if_snapshot
    source = verify_if_snapshot(PROJECT, required_modules=(
        'scripts.validation.laundry_single_garment_fold',), entrypoint_file=Path(__file__))
    if source is None:
        raise ValueError('Use a complete frozen source snapshot')
    release = args.release.resolve()
    task, scene, initial = [load_json(release/name)
                           for name in ('task.json', 'scene.json', 'initial_state.json')]
    if scene['robot'].get('hand_id') != 'dual_fr3_sharpa_w01' or scene.get('named_cloth_initialization') != 'author-only':
        raise ValueError('Actual dual robot and native authored initialization required')
    motion = build_folding_motion(task, scene, settle_s=args.settle,
        fold_duration_s=args.fold_duration, final_passive_s=args.final_passive,
        grip_rotation=args.grip_rotation, peak_height_m=args.fold_peak_height,
        body_fold_peak_height_m=args.body_fold_peak_height,
        first_leg_grasp_material_x_m=args.first_leg_grasp_material_x,
        second_hold_material_region_xy_m=args.second_hold_material_region)
    phases = motion.phases[:args.max_phases] if args.max_phases else motion.phases
    dt = float(scene['physics']['dt_s'])
    phase_duration = args.fold_duration+.75+1.
    approach_duration = 3. if args.robot_control == 'track' else 0.
    park_duration = 2.*max(0,len(phases)-1) if args.robot_control == 'track' else 0.
    duration = args.settle+len(phases)*(approach_duration+phase_duration)+park_duration+args.final_passive
    steps = round(duration/dt)
    if abs(steps*dt-duration) > 1e-8:
        raise ValueError('All events must land on the native time grid')
    output = args.output.resolve()
    if not output.is_relative_to(DATASETS) or output.exists():
        raise ValueError('Use a fresh shared dataset directory')
    output.mkdir(parents=True)
    wrist_yaws = {side: value for side,value in
                  (('left',args.left_wrist_yaw_offset),('right',args.right_wrist_yaw_offset)) if value is not None}
    ik_candidates = load_json(args.ik_seeds) if args.ik_seeds else {}
    report = {'schema': 'laundry_single_garment_fold.v1', 'complete': False,
        'source': source, 'source_manifest': pin(PROJECT/'source_snapshot_manifest.json'),
        'inputs': {name: pin(release/name) for name in ('task.json','scene.json','initial_state.json','release_manifest.json')},
        'task_id': task['id'], 'robot_present': True, 'robot_control': args.robot_control,
        'attachment_control': 'native measured middle fingertip minus declared clearance' if args.robot_control == 'track'
            else 'independent engineering anchors; parked robots do not operate cloth',
        'fingertip_clearance_m': args.clearance, 'cloth_position_velocity_writes': 0,
        'wrist_yaw_offset_rad': args.wrist_yaw_offset,
        'wrist_yaw_by_side_rad': wrist_yaws,
        'approach_height_m': args.approach_height,
        'ik_seed_input': pin(args.ik_seeds) if args.ik_seeds else None,
        'phase_transition': 'All constraints disabled; move through original parked joints over 2 seconds before each subsequent approach.',
        'hold_withdraw_toward_base_y_m': args.hold_withdraw_toward_base,
        'grip_rotation': args.grip_rotation,
        'grip_orientation_semantics': 'Declared local engineering clamp rolls about the fold hinge; this is not measured wrist rotation or friction grasping.',
        'local_hand_collision_filtering': args.local_hand_collision_filter,
        'collision_filter_scope': 'Only fixed local material triangles against their assigned hand colliders while the engineering clamp is present or withdrawing; table, other material, other hand and self contact remain enabled.',
        'collision_filter_events': [],
        'collision_filter_physical_toggle_independently_qualified': False,
        'model_calls': 0, 'task_success_claimed': False, 'robot_friction_grasp_qualified': False,
        'ordinary_bench_policy_qualified': False, 'physics_accepted': False,
        'motion': motion.metadata, 'attachment_events': [], 'duration_s': duration,
        'completed_all_phases_requested': len(phases) == len(motion.phases)}
    env = capture = writer = driver = contact_log = filters = None
    filter_items = []
    code, step, active, mask = 1, 0, [], None
    desired = np.zeros((2, 3), dtype=np.float64)
    anchor_actual = np.zeros((2, 3), dtype=np.float64)
    anchor_rotation = np.asarray([[1.,0.,0.,0.],[1.,0.,0.,0.]])
    applied_rotation = anchor_rotation.copy()
    phase_index = -1
    camera_ids = ('overview', 'task') if args.camera == 'both' else (args.camera,)
    try:
        from bench.simulation.isaacsim_backend.env import FR3DexterousIsaacEnv, _np
        from scripts.validation.laundry_cloth_interactions import attachment
        env = FR3DexterousIsaacEnv(scene)
        env.reset(initial_state=initial)
        # Kit registers USD/PhysX modules during environment construction.
        from pxr import Gf, UsdGeom
        import omni.physx
        report['native_initialization'] = env.named_cloth_initialization_receipt
        report['joint_names'], report['body_names'] = env.joint_names, env.body_names
        driver = RobotDriver(env, scene, args.clearance, args.wrist_yaw_offset, wrist_yaws,
                             args.approach_height)
        if args.local_hand_collision_filter:
            from scripts.validation.laundry_local_grip_filters import LocalGripFilters
            filters = LocalGripFilters(env, motion.spec['rest_vertices_m'], motion.spec['triangles'])
            report['collision_filter_configuration'] = filters.receipt
        view = env._cloth_views['towel']
        q = lambda: _np(view.get_simulation_nodal_positions())[0]
        mask = np.zeros(len(q()), dtype=np.uint8)
        writer = NativeArrays(output/'states', steps+1)
        native_start = int(env.world.current_time_step_index)
        world_start = float(env.world.current_time)
        env._begin_contact_sampling()
        contact_log = (output/'native_rigid_contacts.jsonl').open('x')
        contact_count, contact_impulse = 0, 0.
        env.stage.Export(str(output/'initial.usda'))
        if args.media_output:
            from scripts.media.laundry_cloth_frames import LaundryClothFrameCapture
            capture = LaundryClothFrameCapture(stage=env.stage, world=env.world, app=env.app,
                cloth_views=env._cloth_views, cloth_specs=scene['cloths'], triangles=env.cloth_triangles,
                cloth_paths={k:v['path'] for k,v in env.scene_info['cloths'].items()},
                output=args.media_output, cameras={key: scene['cameras'][key] for key in camera_ids})
        last_print = time.monotonic()
        def photograph(tag=None):
            before = robot_readback(env)
            capture.capture(tag or f'step_{step:06d}', camera_ids=camera_ids)
            if any(not np.array_equal(value, robot_readback(env)[key]) for key,value in before.items()):
                raise RuntimeError('Rendering changed native robot state')
        def record():
            native = robot_readback(env)
            writer.append({**native, 'cloth_q': q(), 'cloth_qd': _np(view.get_simulation_nodal_velocities())[0],
                'control_mask': mask.copy(), 'time_s': np.asarray(step*dt),
                'native_step': np.asarray(step, np.int64),
                'world_step': np.asarray(env.world.current_time_step_index, np.int64),
                'world_time_s': np.asarray(env.world.current_time),
                'phase_index': np.asarray(phase_index, np.int64), 'desired_anchor_m': desired.copy(),
                'actual_anchor_m': anchor_actual.copy(), 'joint_target': env._targets[0].copy(),
                'commanded_anchor_quaternion_wxyz': anchor_rotation.copy(),
                'applied_anchor_quaternion_wxyz': applied_rotation.copy(),
                'local_collision_filter_count': np.asarray(sum(item.active for item in filter_items),np.int64),
                'rigid_contact_pair_count': np.asarray(contact_count,np.int64),
                'rigid_contact_impulse_ns': np.asarray(contact_impulse,dtype=np.float64)})
        def tick():
            nonlocal step, last_print, contact_count, contact_impulse
            mask[:] = 0
            tips = driver.tips() if args.robot_control == 'track' and active else None
            for index, item in enumerate(active):
                if not item['enabled']:
                    continue
                position = tips[item['side']]-[0, 0, args.clearance] if tips is not None else desired[index]
                item['translate'].Set(Gf.Vec3d(*map(float, position)))
                quat = anchor_rotation[index]
                item['orient'].Set(Gf.Quatd(float(quat[0]),Gf.Vec3d(*map(float,quat[1:]))))
                applied_rotation[index] = quat
                anchor_actual[index] = position
                mask[item['ids']] = 1
            env._current_step_time = env.time_s+env.dt
            advance(env)
            env._contact_reports.clear()
            env._read_rigid_contacts()
            rows = env._contact_reports
            contact_count = len(rows)
            contact_impulse = sum(row['impulse_magnitude_sum_ns'] for row in rows)
            if rows:
                contact_log.write(json.dumps({'native_step':step+1,'time_s':(step+1)*dt,'contacts':rows})+'\n')
            step += 1
            if filters is not None and step % round(.02/dt) == 0:
                native_bodies = robot_readback(env)['robot_body_q']
                for item in filter_items:
                    if item.active and item.released and item.try_restore(q(), native_bodies):
                        report['collision_filter_events'].append({'event':'restore',
                            'time_s':step*dt,'native_step':step,**item.receipt})
            record()
            # 8 Hz is not an integer multiple of a 2 ms native step. Alternate
            # adjacent native steps instead of accumulating a video-clock drift.
            if capture and int(step*dt*args.media_fps+1.e-9) > int((step-1)*dt*args.media_fps+1.e-9):
                photograph()
            if time.monotonic()-last_print > 20:
                print(json.dumps({'event':'fold_progress', 'step':step, 'time_s':step*dt,
                    'phase_index':phase_index, 'controlled_nodes':int(mask.sum())}), flush=True)
                last_print = time.monotonic()
        def wait(seconds):
            for _ in range(round(seconds/dt)):
                tick()
        record()
        if capture:
            photograph()
        wait(args.settle)
        for phase_index, phase in enumerate(phases):
            if active:
                raise RuntimeError('Old attachments must be disabled before regrasp')
            if args.robot_control == 'track' and phase_index:
                begin = dict(driver.seed)
                for local in range(round(2./dt)):
                    if local % round(.02/dt) == 0:
                        alpha = minimum_jerk((local+1)*dt,0.,2.)
                        driver.command({name:begin[name]+alpha*(driver.park[name]-begin[name]) for name in begin})
                    tick()
            if any(item.active for item in filter_items):
                raise RuntimeError('Local hand collision filters must be restored before the next grasp')
            preview = deepcopy(motion).bind_phase(phase_index, q())
            origins = [row['origin_m'] for row in preview['anchors']]
            sides = ('left','right')
            if args.robot_control == 'track':
                candidate = ik_candidates.get('phase_candidates',{}).get(str(phase_index))
                if candidate and 'wrist_yaw_offsets_rad' in candidate:
                    driver.wrist_yaw_by_side = dict(candidate['wrist_yaw_offsets_rad'])
                sides, target = driver.assign(origins,candidate)
                begin = dict(driver.seed)
                for local in range(round(2./dt)):
                    if local % round(.02/dt) == 0:
                        alpha = minimum_jerk((local+1)*dt, 0., 2.)
                        driver.command({name: begin[name]+alpha*(target[name]-begin[name]) for name in begin})
                    tick()
                for local in range(round(1./dt)):
                    if local % round(.02/dt) == 0:
                        alpha = minimum_jerk((local+1)*dt, 0., 1.)
                        targets = {side: np.asarray(origin)+[0,0,args.approach_height*(1-alpha)] for side,origin in zip(sides,origins)}
                        driver.command(driver.solve(targets)[0])
                    tick()
            bound = motion.bind_phase(phase_index, q())
            tips = driver.tips()
            for index, (side, row) in enumerate(zip(sides, bound['anchors'])):
                origin, ids = np.asarray(row['origin_m']), row['vertex_ids']
                position = tips[side]-[0,0,args.clearance] if args.robot_control == 'track' else origin
                if args.robot_control == 'track' and np.linalg.norm(position-origin) > .025:
                    raise ValueError('Measured hand is too far from the selected local patch')
                path = f'/World/EngineeringFold/p{phase_index}_a{index}'
                transform = UsdGeom.Xform.Define(env.stage, path)
                translate = transform.AddTranslateOp()
                translate.Set(Gf.Vec3d(*map(float, position)))
                orient = transform.AddOrientOp(UsdGeom.XformOp.PrecisionDouble)
                orient.Set(Gf.Quatd(1.,Gf.Vec3d(0.,0.,0.)))
                anchor_rotation[index] = [1.,0.,0.,0.]
                applied_rotation[index] = [1.,0.,0.,0.]
                prim = attachment(env.stage, path+'_attachment', env.scene_info['cloths']['towel']['path'],
                    path, ids, q()[ids]-position)
                collision_filter = None
                if filters is not None:
                    collision_filter = filters.bind(side, ids, q(), path+'_local_filter')
                    filter_items.append(collision_filter)
                    report['collision_filter_events'].append({'event':'enable',
                        'time_s':step*dt,'native_step':step,**collision_filter.receipt})
                active.append({'prim':prim,'translate':translate,'orient':orient,'ids':ids,'side':side,
                    'enabled':True,'collision_filter':collision_filter})
                desired[index], anchor_actual[index] = origin, position
                report['attachment_events'].append({'event':'enable','time_s':step*dt,
                    'phase_index':phase_index,'anchor_index':index,'side':side,'vertex_ids':ids,
                    'measured_tip_m':tips[side].tolist(), 'anchor_m':position.tolist(),
                    'maximum_node_tip_distance_m':float(np.linalg.norm(q()[ids]-tips[side],axis=1).max())})
            omni.physx.get_physx_simulation_interface().flush_changes()
            for local in range(round(phase_duration/dt)):
                t = phase['grasp_time_s']+local*dt
                for index, item in enumerate(active):
                    desired[index], anchor_rotation[index] = motion.pose(phase_index,index,t)
                    enabled = motion.active(phase_index,index,t)
                    if item['enabled'] and not enabled:
                        item['prim'].GetAttribute('omniphysics:attachmentEnabled').Set(False)
                        item['enabled'] = False
                        if item['collision_filter'] is not None:
                            item['collision_filter'].released = True
                        report['attachment_events'].append({'event':'disable','time_s':step*dt,
                            'phase_index':phase_index,'anchor_index':index})
                        omni.physx.get_physx_simulation_interface().flush_changes()
                if args.robot_control == 'track' and local % round(.02/dt) == 0:
                    hand_targets = {}
                    for index,side in enumerate(sides):
                        point = desired[index].copy()
                        anchor = bound['anchors'][index]
                        if anchor['role'] == 'hold' and not active[index]['enabled']:
                            fraction = minimum_jerk(t,anchor['release_time_s'],anchor['release_time_s']+.6)
                            point[1] += (1 if side=='left' else -1)*args.hold_withdraw_toward_base*fraction
                        hand_targets[side] = point
                    driver.command(driver.solve(hand_targets)[0])
                tick()
            if any(item['enabled'] for item in active):
                raise RuntimeError('Unreleased attachment at phase boundary')
            active.clear()
            dump(output/'progress.json', {'time_s':step*dt,'completed_phase':phase_index,
                'motion':motion.metadata, 'attachment_events':report['attachment_events']})
        phase_index = len(phases)
        if any(item.active for item in filter_items):
            raise RuntimeError('Local hand collision filters remain enabled before the final passive window')
        wait(args.final_passive)
        if step != steps or int(env.world.current_time_step_index)-native_start != steps:
            raise RuntimeError('Native recording step clock mismatch')
        if abs(float(env.world.current_time)-world_start-duration) > 4e-6:
            raise RuntimeError('Native duration mismatch')
        sampling = env._contact_sampling
        if (sampling['sampled_substeps'] != steps
                or sampling['native_end_step_index']-sampling['native_start_step_index'] != steps
                or abs(sampling['end_sim_time_s']-sampling['start_sim_time_s']-duration) > 4.e-6):
            raise RuntimeError('Rigid contact acquisition did not cover every native integration step')
        # The ordinary 20 ms env.step seals its own interval. This diagnostic
        # samples one continuous interval and seals it only after all steps.
        sampling['complete'] = True
        if capture:
            photograph('final')
            report['videos'] = {key: capture.encode_video(camera_id=key,fps=args.media_fps) for key in camera_ids}
        env.stage.Export(str(output/'final.usda'))
        report.update(complete=True,native_steps=step,final_all_constraints_disabled=True,
            final_all_local_collision_filters_disabled=True,
            restored_final_passive_duration_s=args.final_passive)
        code = 0
    except BaseException as error:
        report['error'] = {'type':type(error).__name__,'message':str(error),'traceback':traceback.format_exc()}
        traceback.print_exc()
    finally:
        if filters is not None:
            report['final_active_local_collision_filter_count'] = filters.active_count
        if contact_log:
            contact_log.close()
            report['native_rigid_contacts'] = pin(output/'native_rigid_contacts.jsonl')
            report['native_rigid_contact_sampling'] = dict(env._contact_sampling)
        if driver:
            report['robot_ik'] = driver.receipts
            report['robot_ik_assignment_failures'] = driver.assignment_failures
        if writer:
            report['states'] = writer.close()
        if capture:
            capture.close()
        dump(output/'native_report.json',report)
        print(json.dumps({'event':'fold_finished','complete':report['complete'],'output':str(output)}),flush=True)
        if env:
            if code:
                # Kit fast shutdown derives its process status from the active
                # exception; keep a failed diagnostic active during cleanup.
                try:
                    raise RuntimeError('Native fold failed; inspect '+str(output/'native_report.json'))
                finally:
                    env.close()
            env.close()
    return code


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--release',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--robot-control',choices=('track','parked'),default='track')
    parser.add_argument('--settle',type=float,default=.5)
    parser.add_argument('--fold-duration',type=float,default=5.)
    parser.add_argument('--final-passive',type=float,default=2.)
    parser.add_argument('--clearance',type=float,default=.015)
    parser.add_argument('--wrist-yaw-offset',type=float,default=0.)
    parser.add_argument('--left-wrist-yaw-offset',type=float)
    parser.add_argument('--right-wrist-yaw-offset',type=float)
    parser.add_argument('--approach-height',type=float,default=.12)
    parser.add_argument('--grip-rotation',choices=('fixed','fold_hinge'),default='fold_hinge')
    parser.add_argument('--local-hand-collision-filter',action='store_true')
    parser.add_argument('--ik-seeds',type=Path)
    parser.add_argument('--hold-withdraw-toward-base',type=float,default=.10)
    parser.add_argument('--body-fold-peak-height',type=float)
    parser.add_argument('--fold-peak-height',type=float,default=.06)
    parser.add_argument('--first-leg-grasp-material-x',type=float)
    parser.add_argument('--second-hold-material-region',type=float,nargs=4)
    parser.add_argument('--max-phases',type=int,choices=(1,2,3))
    parser.add_argument('--media-output',type=Path)
    parser.add_argument('--media-fps',type=int,choices=(2,4,8),default=4)
    parser.add_argument('--camera',choices=('overview','task','both'),default='overview')
    raise SystemExit(run(parser.parse_args()))


if __name__ == '__main__':
    main()
