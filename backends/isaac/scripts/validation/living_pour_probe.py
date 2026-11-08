"""Native PBD pouring under an explicitly separate revolute test fixture.

The kettle remains dynamic. The fixture's bounded applied torque tilts it, never
body pose/velocity writes after reset. This tests vessel/flow/ledger behavior,
not a robot strategy, grip, support-load calibration or runnable task release.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import traceback
import numpy as np
from bench.paths import OUTPUT_ROOT
from scripts.maintenance.freeze_project_sources import verify_if_snapshot

PROJECT = Path(__file__).resolve().parents[2]


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def array(value):
    return np.asarray(value.detach().cpu().numpy() if hasattr(value, 'detach') else value).copy()


def native_inverse(pose):
    pose = np.asarray(pose, float)
    x, y, z, w = pose[3:] / np.linalg.norm(pose[3:])
    r = np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                  [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                  [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])
    inverse = np.eye(4)
    inverse[:3, :3] = r
    inverse[3, :3] = -pose[:3]@r
    return inverse


def run(args):
    verified = verify_if_snapshot(PROJECT, required_modules=('scripts.maintenance.freeze_project_sources',), entrypoint_file=Path(__file__))
    if verified is None or not (PROJECT/'bench/apps/workroom.headless.kit').is_file():
        raise ValueError('Complete frozen source with headless kit is required')
    gpu = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    if not gpu.startswith('GPU-') or ',' in gpu:
        raise ValueError('One physical GPU UUID is required')
    if not 0 < args.physics_dt <= 1/960 or not 0 <= abs(args.aim_offset) <= .15:
        raise ValueError('Invalid diagnostic configuration')
    if not math.isfinite(args.max_angle_deg) or not 0 < args.max_angle_deg <= 85.:
        raise ValueError('Diagnostic tilt must be finite in (0,85] degrees')
    output = args.output.resolve()
    if not output.is_relative_to(OUTPUT_ROOT):
        raise ValueError('Evidence must live under shared datasets')
    output.mkdir(parents=True, exist_ok=False)
    report = {'schema': 'living_fixture_pour_probe.v1', 'source_snapshot': str(PROJECT),
        'snapshot_verified': verified, 'gpu_uuid': gpu, 'physics_dt_s': args.physics_dt,
        'fixture': {'type': 'world-to-dynamic-kettle revolute', 'axis': 'local Y',
                    'max_target_angle_deg': args.max_angle_deg,
                    'max_torque_nm': 2., 'torque_gain_nm_rad': 8., 'damping_nms_rad': .4, 'sleep_threshold_override': 0.,
                    'pivot_local_m': [0, 0, .07], 'pivot_world_z_m': .955,
                    'aim_offset_cross_stream_m': args.aim_offset,
                    'reset_pose_override': True, 'post_reset_body_pose_writes': 0},
        'task_success': False, 'robot_used': False, 'controller_used': False, 'model_calls': 0,
        'support_load_qualified': False, 'passed': False,
        'per_step_usd_save': False, 'vessel_pose_source': 'Native rigid body tensors, never a USD writeback',
        'scope': 'Independent fixture-driven real PBD pouring and geometric spill ledger only'}
    dump(output/'plan.json', report)
    app = world = None
    code = 1
    try:
        from isaacsim import SimulationApp
        allow = os.environ.get('ISAAC_ALLOW_UNVALIDATED_DRIVER') == '1'
        report['unvalidated_driver_override'] = allow
        app = SimulationApp({'headless': True, 'width': 640, 'height': 480, 'multi_gpu': False,
            'active_cuda_gpus': [0], 'max_gpu_count': 1, 'physics_gpu': 0,
            'renderer': 'RayTracedLighting', 'fast_shutdown': True,
            'extra_args': ['--/rtx/verifyDriverVersion/enabled=false'] if allow else []},
            experience=str(PROJECT/'bench/apps/workroom.headless.kit'))
        import carb
        import omni.usd
        import omni.physx
        from omni.physxfabric import get_physx_fabric_interface
        from isaacsim.core.api import World
        from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics, Vt
        import torch
        from scenes.workrooms.living_room.tasks.water_pouring import assets
        from scenes.workrooms.living_room.tasks.water_pouring.liquid_ledger import LiquidLedger
        report['snapshot_verified'] = verify_if_snapshot(PROJECT,
            required_modules=(assets.__name__, LiquidLedger.__module__), entrypoint_file=Path(__file__))
        report['runtime_version'] = (Path(os.environ['ISAAC_PATH'])/'VERSION').read_text().strip()
        settings = carb.settings.get_settings()
        def readback_settings():
            values = {'/physics/'+x: True for x in ('updateToUsd', 'updateParticlesToUsd', 'updateVelocitiesToUsd')}
            values.update({'/physics/suppressReadback': False, '/physics/outputVelocitiesLocalSpace': False})
            for k, v in values.items(): settings.set_bool(k, v)
            if any(settings.get(k) != v for k, v in values.items()):
                raise RuntimeError('Native readback settings did not stick')
            return values
        omni.usd.get_context().new_stage()
        stage = omni.usd.get_context().get_stage()
        world = World(stage_units_in_meters=1., physics_dt=args.physics_dt, rendering_dt=1/30,
                      backend='torch', device='cuda:0', physics_prim_path='/World/PhysicsScene')
        context = world.get_physics_context()
        context.enable_gpu_dynamics(True)
        context.set_broadphase_type('GPU')
        context.set_solver_type('TGS')
        readback_settings()
        api = PhysxSchema.PhysxSceneAPI.Apply(stage.GetPrimAtPath(context.prim_path))
        api.CreateGpuMaxParticleContactsAttr(2**20)
        table = UsdGeom.Cube.Define(stage, '/World/ProbeTable')
        table.CreateSizeAttr(1.)
        table.AddTranslateOp().Set(Gf.Vec3d(.65, 0, .755))
        table.AddScaleOp().Set(Gf.Vec3f(1, 1, .05))
        UsdPhysics.CollisionAPI.Apply(table.GetPrim())
        metadata = assets.author_task(stage, particle_spacing_m=args.particle_spacing)
        report['particle_spacing_m'] = args.particle_spacing
        if args.vessel_sleep_threshold is not None:
            if not math.isfinite(args.vessel_sleep_threshold) or args.vessel_sleep_threshold < 0:
                raise ValueError('Vessel sleep threshold must be nonnegative and finite')
            for info in metadata['containers'].values():
                PhysxSchema.PhysxRigidBodyAPI.Apply(stage.GetPrimAtPath(info['path'])).CreateSleepThresholdAttr(args.vessel_sleep_threshold)
        report['vessel_sleep_threshold_override'] = args.vessel_sleep_threshold
        PhysxSchema.PhysxContactReportAPI.Apply(stage.GetPrimAtPath(
            metadata['containers']['cup']['path'])).CreateThresholdAttr(0.)
        kettle = metadata['containers']['kettle']
        yaw = kettle['initial_yaw_rad']
        direction, lateral = np.array([math.cos(yaw), math.sin(yaw), 0]), np.array([-math.sin(yaw), math.cos(yaw), 0])
        pivot = np.array(metadata['containers']['cup']['initial_position_m'])-.064*direction+args.aim_offset*lateral
        pivot[2] = .955
        base = pivot-[0, 0, .07]
        delta = base-np.array(kettle['initial_position_m'])
        prim = stage.GetPrimAtPath(kettle['path'])
        translate = next(op for op in UsdGeom.Xformable(prim).GetOrderedXformOps() if op.GetOpType() == UsdGeom.XformOp.TypeTranslate)
        translate.Set(Gf.Vec3d(*base))
        PhysxSchema.PhysxRigidBodyAPI.Apply(prim).CreateSleepThresholdAttr(0.)
        points_prim = UsdGeom.Points(stage.GetPrimAtPath(metadata['water']['particles_path']))
        points_prim.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy((np.asarray(points_prim.GetPointsAttr().Get())+delta).astype(np.float32)))
        joint = UsdPhysics.RevoluteJoint.Define(stage, '/World/PourFixture')
        joint.CreateBody1Rel().SetTargets([kettle['path']])
        joint.CreateLocalPos0Attr(Gf.Vec3f(*pivot))
        joint.CreateLocalPos1Attr(Gf.Vec3f(0, 0, .07))
        joint.CreateLocalRot0Attr(Gf.Quatf(math.cos(yaw/2), Gf.Vec3f(0, 0, math.sin(yaw/2))))
        joint.CreateLocalRot1Attr(Gf.Quatf(1.))
        joint.CreateAxisAttr('Y')
        joint.CreateLowerLimitAttr(-2.)
        joint.CreateUpperLimitAttr(90.)
        report['fixture']['pivot_world_m'] = pivot.tolist()
        report['fixture']['reset_kettle_base_m'] = base.tolist()
        report['fixture']['conservative_kettle_body_bottom_z_m'] = float(pivot[2]-math.hypot(.07, .052))
        report['fixture']['cup_rim_z_m'] = float(metadata['containers']['cup']['initial_position_m'][2]+.067)
        if report['fixture']['conservative_kettle_body_bottom_z_m'] < report['fixture']['cup_rim_z_m']+.01:
            raise RuntimeError('Tilt fixture clearance over cup is insufficient')
        dump(output/'asset_metadata.json', metadata)
        app.update(); app.update()
        stage.Export(str(output/'initial.usda'))
        world.reset()
        report['effective_readback_settings'] = readback_settings()
        views = {name: world.physics_sim_view.create_rigid_body_view(info['path']) for name, info in metadata['containers'].items()}
        if any(v.count != 1 for v in views.values()): raise RuntimeError('Native vessel inventory mismatch')
        cup_contact = world.physics_sim_view.create_rigid_contact_view(
            metadata['containers']['cup']['path'],
            filter_patterns=[metadata['tray']['path']+'/Floor'], max_contact_data_count=4096)
        report['native_cup_dry_mass_kg'] = float(array(views['cup'].get_masses()).reshape(-1)[0])
        native_device = str(getattr(views['kettle'].get_transforms(), 'device', 'cpu'))
        report['native_tensor_device'] = native_device
        one = torch.tensor([0], dtype=torch.int32, device=native_device)
        tensor = lambda v: torch.as_tensor(np.asarray(v)[None], dtype=torch.float32, device=native_device)
        com_local = array(views['kettle'].get_coms()).reshape(7)[:3]
        initial_time = float(world.current_time)
        ledger = LiquidLedger(metadata, dt=args.physics_dt, initial_time=initial_time)
        report['initial_native_time_s'] = initial_time
        return_time = None
        return_angle = 0.
        rows, frames, raw_chunk, velocity_chunk = [], [], [], []
        (output/'particle_chunks').mkdir()
        # Tilt slowly. Return when the privately observed cup reaches 90 mL;
        # this diagnostic feedback is disclosed and is not an Agent policy.
        with (output/'samples.jsonl').open('x') as handle:
            for tick in range(1, round(9/args.physics_dt)+1):
                elapsed = (tick-1)*args.physics_dt
                angle = min(args.max_angle_deg, max(0., (elapsed-1.)*30.))
                if return_time is not None:
                    angle = max(0., return_angle-(elapsed-return_time)*45.)
                before = array(views['kettle'].get_transforms())[0]
                velocity = array(views['kettle'].get_velocities())[0]
                rotation = Gf.Rotation(Gf.Quatd(float(before[6]), Gf.Vec3d(*before[3:6].astype(float))))
                up = np.asarray(rotation.TransformDir(Gf.Vec3d(0, 0, 1)))
                native_angle = math.atan2(float(up@direction), float(up[2]))
                torque_scalar = float(np.clip(8*(math.radians(angle)-native_angle)-.4*(velocity[3:]@lateral), -2., 2.))
                com = before[:3]+np.asarray(rotation.TransformDir(Gf.Vec3d(*com_local.astype(float))))
                views['kettle'].apply_forces_and_torques_at_position(tensor([0, 0, 0]), tensor(lateral*torque_scalar), tensor(com), one, True)
                world.step(render=False)
                omni.physx.get_physx_simulation_interface().fetch_results()
                omni.physx.get_physx_interface().update_transformations(False, True, True, False)
                assets.read_water_state(stage, metadata)  # inventory + physical unsmoothed-Points guard
                points = np.asarray(points_prim.GetPointsAttr().Get(), float)
                velocities = np.asarray(points_prim.GetVelocitiesAttr().Get(), float)
                inverse, bodies = {}, {}
                for name, info in metadata['containers'].items():
                    pose = array(views[name].get_transforms())[0]
                    inverse[name] = native_inverse(pose)
                    bodies[name] = {'pose_xyzw': pose.tolist(), 'velocity_linear_angular': array(views[name].get_velocities())[0].tolist()}
                after = bodies['kettle']['pose_xyzw']
                rot_after = Gf.Rotation(Gf.Quatd(float(after[6]), Gf.Vec3d(*after[3:6])))
                up_after = np.asarray(rot_after.TransformDir(Gf.Vec3d(0, 0, 1)))
                angle_after = math.atan2(float(up_after@direction), float(up_after[2]))
                angle_rate = (angle_after-native_angle)/args.physics_dt
                native_rate = float(np.asarray(bodies['kettle']['velocity_linear_angular'][3:])@lateral)
                if abs(angle_rate) > .01 and abs(native_rate) < 1e-7:
                    raise RuntimeError('Native angle changes while angular velocity is zero; observation feedback suspected')
                row = ledger.update(time_s=float(world.current_time), ids=points_prim.GetIdsAttr().Get(),
                    points=points, velocities=velocities, world_to_local=inverse)
                if tick == 1 and row['counts']['kettle'] != ledger.count:
                    raise RuntimeError('Unobserved reset interval did not preserve all water in kettle')
                row.update({'fixture_target_deg': angle, 'actual_angle_before_step_deg': math.degrees(native_angle), 'applied_torque_nm': torque_scalar, 'angle_rate_from_pose_rad_s': angle_rate, 'native_angle_rate_rad_s': native_rate, 'elapsed_s': elapsed+args.physics_dt, 'bodies': bodies})
                row['cup_floor_reaction_n'] = array(cup_contact.get_contact_force_matrix(
                    dt=args.physics_dt)).reshape(-1, 3).sum(axis=0).tolist()
                rows.append(row)
                raw_chunk.append(points.astype(np.float32))
                velocity_chunk.append(velocities.astype(np.float32))
                if len(raw_chunk) == 120:
                    np.savez_compressed(output/'particle_chunks'/f'{tick-119:06d}_{tick:06d}.npz',
                        points_m=np.asarray(raw_chunk), velocities_m_s=np.asarray(velocity_chunk),
                        particle_ids=np.asarray(points_prim.GetIdsAttr().Get(), dtype=np.int64),
                        ticks=np.arange(tick-119, tick+1),
                        dt_s=args.physics_dt, initial_time_s=initial_time)
                    raw_chunk.clear()
                    velocity_chunk.clear()
                handle.write(json.dumps(row, allow_nan=False)+'\n')
                if tick % 120 == 0:
                    frames.append(points.astype(np.float32))
                    handle.flush()
                    print(json.dumps({k: row[k] for k in ('tick', 'elapsed_s', 'current_nominal_volume_ml', 'cumulative_spill_ml')}), flush=True)
                cup_ml = row['current_nominal_volume_ml']['cup']
                if return_time is None and (cup_ml >= 90. or elapsed >= 6.):
                    return_time, return_angle = elapsed, angle
                    report['return_trigger'] = {'elapsed_s': elapsed, 'cup_ml': cup_ml, 'angle_deg': angle}
        if raw_chunk: raise RuntimeError('Incomplete raw particle chunk')
        world.pause()
        # Snapshot once after all measurements: never feed saved USD back into live steps.
        fabric = get_physx_fabric_interface()
        fabric.force_update(args.physics_dt, world.current_time)
        fabric.save_to_usd()
        stage.Flatten().Export(str(output/'final.usda'))
        np.savez_compressed(output/'particle_frames.npz', points_m=np.asarray(frames), sampling_period_s=120*args.physics_dt,
                            particle_ids=np.arange(ledger.count), cumulative_spilled_ids=np.flatnonzero(ledger.spilled))
        final = rows[-1]
        cup_history = [r['current_nominal_volume_ml']['cup'] for r in rows]
        spill = final['cumulative_spill_ml']
        report.update({'status': 'completed', 'observed_native_steps': len(rows), 'final': final,
            'maximum_cup_nominal_volume_ml': max(cup_history),
            'ledger_every_native_step': True, 'raw_particle_positions_saved_every_step': True, 'particle_positions_changed': len({r['positions_sha256'] for r in rows}) > 1,
            'raw_particle_velocities_and_ids_saved_every_step': True,
            'cup_target_interval_met': 90 <= cup_history[-1] <= 110,
            'spill_limit_met': spill <= 5., 'final_outside_unresolved_particles': final['counts']['in_flight_or_unresolved'],
            'final_cup_stable_range_last_second_ml': max(cup_history[-round(1/args.physics_dt):])-min(cup_history[-round(1/args.physics_dt):]),
            'geometric_spill_scope': 'Tray/table/floor interception envelope and escaped domain; no particle contact reports',
            'mass_scope': 'Nominal USD mass partition, not native inverse-mass calibration'})
        report['diagnostic_checks'] = {
            'native_flow_observed': report['particle_positions_changed'] and final['ever_left_kettle_particles'] > 0,
            'inventory_closed_all_steps': all(r['inventory_closed'] for r in rows),
            'upright_return': abs(final['actual_angle_before_step_deg']) < 3.,
            'no_unresolved_outside_water': final['counts']['in_flight_or_unresolved'] == 0,
        }
        dry_rows = [r for r in rows if .5 <= r['elapsed_s'] < 1. and r['counts']['cup'] == 0]
        wet_rows = rows[-round(1/args.physics_dt):]
        # Sleeping bodies can have no reported contact impulse. Do not treat
        # zero-filled sleeping rows as a measured empty-cup support baseline.
        if dry_rows and cup_history[-1] > 0 and args.vessel_sleep_threshold == 0.:
            dry_n = float(np.mean([r['cup_floor_reaction_n'][2] for r in dry_rows]))
            wet_n = float(np.mean([r['cup_floor_reaction_n'][2] for r in wet_rows]))
            expected_n = cup_history[-1] * 1.e-6 * metadata['water']['reference_density_kg_m3'] * 9.81
            report['cup_weight_transfer'] = {'dry_floor_mean_z_n': dry_n,
                'wet_floor_mean_z_n': wet_n, 'increment_z_n': wet_n-dry_n,
                'expected_final_water_weight_n': expected_n,
                'increment_to_expected_ratio': (wet_n-dry_n)/expected_n,
                'scope': 'Same-run empty cup at 0.5–1 s versus final second; interpret only with settled cup and water'}
        else:
            report['cup_weight_transfer'] = {'qualified': False,
                'reason': 'Quantitative comparison requires an explicitly awake cup, empty baseline, and received water'}
        if abs(args.aim_offset) > .05:
            report['diagnostic_checks']['wrong_aim_detected_as_spill'] = spill > 5 and cup_history[-1] < 90
        else:
            report['diagnostic_checks'].update({
                'water_entered_cup': max(cup_history) > 50,
                'cup_target_interval_met': report['cup_target_interval_met'],
                'spill_limit_met': report['spill_limit_met'],
                'cup_settled': report['final_cup_stable_range_last_second_ml'] <= 1.,
                'no_below_rim_leak': final['first_kettle_exit_regions']['below_rim'] == 0,
            })
        report['passed'] = all(report['diagnostic_checks'].values())
        code = 0 if report['passed'] else 1
    except Exception as error:
        report.update({'status': 'failed', 'error': repr(error), 'traceback': traceback.format_exc()})
        traceback.print_exc()
    finally:
        report['requested_exit_code'] = code
        report['requires_external_exit_zero'] = True
        dump(output/'result.json', report)
        if app is not None: app.close(exit_code=code)
    return code


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--physics-dt', type=float, default=1/960)
    parser.add_argument('--aim-offset', type=float, default=0.)
    parser.add_argument('--particle-spacing', type=float, default=.003)
    parser.add_argument('--vessel-sleep-threshold', type=float)
    parser.add_argument('--max-angle-deg', type=float, default=65.)
    raise SystemExit(run(parser.parse_args()))
