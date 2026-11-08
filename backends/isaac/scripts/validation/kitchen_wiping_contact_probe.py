"""Bounded independent-sponge contact probe; no hand strategy or task score.

Mount the exact frozen task USD, change only the initial sponge location before
reset, and apply a bounded horizontal wrench. A matching-height/material table
proxy supports the real sponge; gravity, rotation and contact remain native.
Canonical furnished-room qualification is performed separately.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import importlib
import json
import os
from pathlib import Path
import traceback

import numpy as np

from bench.isaacsim.release import checked, digest, dump, task_inputs, _frozen_evaluator_module
from bench.paths import OUTPUT_ROOT
from scripts.maintenance.freeze_project_sources import verify_if_snapshot

PROJECT = Path(__file__).resolve().parents[2]


def run(args):
    source = verify_if_snapshot(PROJECT,
        required_modules=('scripts.validation.kitchen_wiping_contact_probe',), entrypoint_file=Path(__file__))
    if source is None or Path.cwd().resolve() != PROJECT:
        raise ValueError('Use the complete frozen source root')
    output = args.output.resolve()
    if not output.is_relative_to(OUTPUT_ROOT):
        raise ValueError('Evidence belongs in shared datasets')
    output.mkdir(parents=True, exist_ok=False)
    report = {'schema': 'kitchen_wiping_contact_probe.v1', 'passed': False,
        'source': source, 'release_manifest_sha256': digest(args.release/'release_manifest.json'),
        'robot_present': False, 'hand_object_qualified': False, 'task_success_claimed': False,
        'object_pose_writes_after_reset': False, 'model_calls': 0,
        'gpu_uuid': os.environ.get('CUDA_VISIBLE_DEVICES'),
        'isaac_allow_unvalidated_driver': os.environ.get('ISAAC_ALLOW_UNVALIDATED_DRIVER'),
        'scope': 'Exact frozen sponge geometry on a table proxy; horizontal tool wrench only. Not robot execution or contact-force calibration.',
        'tool_boundary': {'maximum_force_axis_n': .4, 'vertical_force_n': 0., 'torque_nm': [0., 0., 0.],
                          'post_reset_pose_writes': False, 'post_reset_velocity_writes': False}}
    app = None
    code = 1
    try:
        _, scene, task, _ = task_inputs(args.release, 'surface_wiping')
        meta = deepcopy(task['asset_metadata'])
        center = np.asarray([c['center_world_m'][:2] for c in meta['goals']['soil_cells']
                             if c['patch'] == 'soil_1']).mean(axis=0)
        left, right = center[0]-.08, center[0]+.08
        near_y, far_y = center[1]-.019, center[1]+.019
        start = [float(left), float(near_y), .9003]
        meta['objects'][0]['initial_pose_xyzw'][:3] = start
        task['asset_metadata'] = meta
        frozen = _frozen_evaluator_module(args.release.resolve())
        rules = importlib.import_module(frozen.__package__+'.rules')
        task['evaluation'] = rules.evaluation_configuration('surface_wiping', meta)
        observer = frozen.NativeTaskObserver(task)
        dump(output/'diagnostic_task.json', task)
        from isaacsim import SimulationApp
        app = SimulationApp({'headless': True, 'width': 640, 'height': 480,
            'multi_gpu': False, 'active_cuda_gpus': [0], 'max_gpu_count': 1, 'physics_gpu': 0,
            'fast_shutdown': True, 'extra_args': ['--/rtx/verifyDriverVersion/enabled=false']
            if os.environ.get('ISAAC_ALLOW_UNVALIDATED_DRIVER') == '1' else []},
            experience=str(PROJECT/'bench/apps/workroom.headless.kit'))
        import omni.usd
        from isaacsim.core.api import World
        from pxr import Gf, UsdGeom, UsdPhysics, UsdShade, PhysxSchema
        from bench.simulation.isaacsim_backend.contact_precision import summarize_normal_impulses
        omni.usd.get_context().new_stage()
        stage = omni.usd.get_context().get_stage()
        UsdGeom.SetStageMetersPerUnit(stage, 1.)
        UsdGeom.SetStageUpAxis(stage, 'Z')
        bundle = scene['native_task_bundle']
        checked(bundle['usd_path'], bundle['usd_sha256'])
        root = stage.DefinePrim(bundle['root_prim_path'], 'Xform')
        root.GetReferences().AddReference(bundle['usd_path'], bundle['root_prim_path'])
        body = stage.GetPrimAtPath(meta['actors']['sponge'])
        body.GetAttribute('xformOp:translate').Set(Gf.Vec3d(*start))
        PhysxSchema.PhysxContactReportAPI.Apply(body).CreateThresholdAttr(0.)
        support = '/World/Room/Geometry/worktop/Collider'
        table = UsdGeom.Cube.Define(stage, support)
        table.CreateSizeAttr(1.)
        table.AddTranslateOp().Set(Gf.Vec3d(.45, -.95, .8775))
        table.AddScaleOp().Set(Gf.Vec3f(2.8, .8, .045))
        UsdPhysics.CollisionAPI.Apply(table.GetPrim())
        rigid = UsdPhysics.RigidBodyAPI.Apply(table.GetPrim())
        rigid.CreateKinematicEnabledAttr(True)
        material = UsdShade.Material.Define(stage, '/World/TableContact')
        physics = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
        physics.CreateStaticFrictionAttr(.4)
        physics.CreateDynamicFrictionAttr(.4)
        physics.CreateRestitutionAttr(0.)
        UsdShade.MaterialBindingAPI.Apply(table.GetPrim()).Bind(material, materialPurpose='physics')
        dt = task['evaluation']['physics_dt_s']
        world = World(stage_units_in_meters=1., physics_dt=dt, rendering_dt=1/30,
                      backend='numpy', physics_prim_path='/World/PhysicsScene')
        pc = world.get_physics_context()
        pc.enable_gpu_dynamics(True)
        pc.set_broadphase_type('GPU')
        pc.set_solver_type(scene['isaacsim'].get('solver_type', 'TGS'))
        stage.Export(str(output/'initial.usda'))
        world.reset()
        view = world.physics_sim_view.create_rigid_body_view(meta['actors']['sponge'])
        contacts = world.physics_sim_view.create_rigid_contact_view(meta['actors']['sponge'],
            filter_patterns=[support], max_contact_data_count=8192)
        if view.count != 1 or contacts.sensor_count != 1 or contacts.filter_count != 1:
            raise RuntimeError('Expected one native sponge/table contact pair')
        paths = contacts.filter_paths
        report['native_contact_filter_paths'] = list(paths)
        one = np.array([0], dtype=np.int32)
        start_native_time = float(world.current_time)
        initial_step = int(world.current_time_step_index)
        rows, poses = [], []

        def sample(baseline=False):
            pose = np.asarray(view.get_transforms())[0].copy()
            velocity = np.asarray(view.get_velocities())[0].copy()
            t = float(world.current_time)-start_native_time
            native_contacts = []
            if not baseline:
                forces, _, normals, separation, counts, begins = contacts.get_contact_data(dt=1.)
                count, begin = int(np.asarray(counts)[0, 0]), int(np.asarray(begins)[0, 0])
                if count:
                    f = np.asarray(forces)[begin:begin+count].reshape(-1)
                    n = np.asarray(normals)[begin:begin+count]
                    s = np.asarray(separation)[begin:begin+count].reshape(-1)
                    native_contacts.append({'actor0': meta['actors']['sponge'], 'actor1': support,
                        'point_count': count, 'time_s': t, **summarize_normal_impulses(f, n),
                        'min_separation_m': float(s.min())})
            state = {'position_m': pose[:3].tolist(), 'quaternion_wxyz': [float(pose[6]), *pose[3:6].tolist()],
                     'linear_velocity_m_s': velocity[:3].tolist(), 'angular_velocity_rad_s': velocity[3:].tolist()}
            frame = {'baseline': baseline, 'time_s': t, 'native_time_s': float(world.current_time),
                'native_step': int(world.current_time_step_index), 'physics_dt_s': dt,
                'objects': {'sponge': state}, 'contacts': native_contacts}
            observer.on_frame(frame)
            poses.append([t, *pose, *velocity])
            return frame

        sample(True)
        # Two overlapping strokes cover the first patch, then release. Other
        # patches remain dirty, so this bounded probe must not claim task success.
        plan = [('settle', .4, [left, near_y]), ('forward', 2.4, [right, near_y]),
                ('side_step', .8, [right, far_y]), ('backward', 2.4, [left, far_y]),
                ('release', .6, [left, far_y])]
        report['plan'] = plan
        previous_target = np.asarray(start[:2])
        loaded_steps = 0
        with (output/'native_contact_motion.jsonl').open('x') as stream:
            for name, duration, target in plan:
                target = np.asarray(target)
                for tick in range(round(duration/dt)):
                    pose = np.asarray(view.get_transforms())[0]
                    velocity = np.asarray(view.get_velocities())[0]
                    force = np.zeros(3)
                    if name not in ('settle', 'release'):
                        u = (tick+1)*dt/duration
                        s, sd = u*u*(3-2*u), 6*u*(1-u)/duration
                        desired = previous_target+(target-previous_target)*s
                        desired_velocity = (target-previous_target)*sd
                        force[:2] = np.clip(30*(desired-pose[:2])+1.4*(desired_velocity-velocity[:2]), -.4, .4)
                        view.apply_forces_and_torques_at_position(np.asarray([force], np.float32),
                            np.zeros((1, 3), np.float32), None, one, True)
                    world.step(render=False)
                    frame = sample()
                    state = observer.on_observe()
                    loaded_steps += int(state['detail']['loaded_wiping_contact'])
                    stream.write(json.dumps({'phase': name, 'force_n': force.tolist(),
                        'frame': frame, 'coverage_fraction': state['detail']['coverage_fraction']}, allow_nan=False)+'\n')
                previous_target = target
                print(json.dumps({'phase': name, 'patch_coverage': state['detail']['patch_coverage']}), flush=True)
        np.savez_compressed(output/'native_states.npz', samples=np.asarray(poses))
        state = observer.on_observe()
        final = poses[-1]
        checks = {'complete_native_samples': int(world.current_time_step_index)-initial_step == len(poses)-1,
            'native_observer_valid': state['complete'] and not state['errors'],
            'actual_loaded_contact': loaded_steps > 100,
            'first_patch_cleaned': state['detail']['patch_coverage']['soil_1'] >= .60,
            'other_patches_not_projected_clean': state['detail']['patch_coverage']['soil_3'] == 0.,
            'no_full_task_credit': not state['detail']['cleaning_ready'] and state['released_stable_s'] == 0.,
            'released_tool_supported': abs(final[3]-.90) < .004,
            'released_tool_quiet': np.linalg.norm(final[8:11]) < .025 and np.linalg.norm(final[11:14]) < .25,
            'no_safety_failure': not state['terminal_failure'] and not state['safety_stop']}
        checks = {name: bool(value) for name, value in checks.items()}
        report['final_source'] = verify_if_snapshot(PROJECT,
            required_modules=('scripts.validation.kitchen_wiping_contact_probe',), entrypoint_file=Path(__file__))
        report.update(checks=checks, passed=all(checks.values()), final_state=state,
            actual_loaded_steps=loaded_steps, native_steps=len(poses)-1,
            evidence_sha256={name: digest(output/name) for name in ('initial.usda', 'diagnostic_task.json',
                'native_contact_motion.jsonl', 'native_states.npz')})
        code = 0 if report['passed'] else 1
    except BaseException:
        report['exception'] = traceback.format_exc()
        traceback.print_exc()
    finally:
        report['process_exit_code'] = code
        dump(output/'qualification.json', report)
        print(json.dumps({'passed': report['passed'], 'output': str(output)}), flush=True)
        if app is not None:
            app.close(exit_code=code)
    return code


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--release', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    raise SystemExit(run(parser.parse_args()))
