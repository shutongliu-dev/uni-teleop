"""Frozen-source, real GPU PBD vessel retention and readback diagnostic.

This is neither a robot strategy nor task scoring. A separate empty-vessel
control can measure the change in rigid floor reaction, without pretending that
particle contact reports are available. Every attempt needs a new output path.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import traceback

import numpy as np

from bench.paths import OUTPUT_ROOT
from scripts.maintenance.freeze_project_sources import verify_if_snapshot

PROJECT = Path(__file__).resolve().parents[2]


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def array(value):
    if hasattr(value, 'detach'):
        value = value.detach().cpu().numpy()
    return np.asarray(value).copy()


def run(args):
    snapshot = verify_if_snapshot(PROJECT, required_modules=('scripts.maintenance.freeze_project_sources',), entrypoint_file=Path(__file__))
    if snapshot is None:
        raise ValueError('Native probe requires a complete source snapshot, not the editable project')
    if not (PROJECT/'bench/apps/workroom.headless.kit').is_file():
        raise ValueError('Freeze the headless experience with --extra bench/apps/workroom.headless.kit')
    gpu = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    if not gpu.startswith('GPU-') or ',' in gpu:
        raise ValueError('Select one approved physical GPU UUID')
    if not 0 < args.physics_dt <= 1/120 or args.seconds < 1.:
        raise ValueError('Invalid diagnostic duration/time step')
    output = args.output.resolve()
    if not str(output).startswith(OUTPUT_ROOT + '/'):
        raise ValueError('Native evidence belongs under shared datasets')
    output.mkdir(parents=True, exist_ok=False)
    report = {'schema': 'living_water_native_probe.v1', 'created_utc': datetime.now(timezone.utc).isoformat(),
        'source_snapshot': str(PROJECT), 'snapshot_verified': snapshot, 'gpu_uuid': gpu,
        'physics_dt_s': args.physics_dt, 'requested_seconds': args.seconds, 'dry_control': args.dry_control,
        'native_accepted': False, 'task_success': False, 'controller_used': False, 'model_calls': 0,
        'scope': 'Independent dynamic vessels, real fluid retention/readback and rigid floor reaction only',
        'status': 'initializing', 'passed': False, 'sync_mode': args.sync_mode}
    dump(output/'plan.json', report)
    app = world = None
    code = 1
    samples = []
    try:
        from isaacsim import SimulationApp
        allow_driver = os.environ.get('ISAAC_ALLOW_UNVALIDATED_DRIVER') == '1'
        app = SimulationApp({'headless': True, 'width': 640, 'height': 480,
            'multi_gpu': False, 'active_cuda_gpus': [0], 'max_gpu_count': 1,
            'physics_gpu': 0, 'renderer': 'RayTracedLighting', 'fast_shutdown': True,
            'extra_args': ['--/rtx/verifyDriverVersion/enabled=false'] if allow_driver else []},
            experience=str(PROJECT/'bench/apps/workroom.headless.kit'))
        import carb
        import omni.physx
        from omni.physxfabric import get_physx_fabric_interface
        from omni.physxfabric.bindings import _physxFabric as fabric_bindings
        import omni.usd
        from isaacsim.core.api import World
        from pxr import Gf, PhysxSchema, Sdf, UsdGeom, UsdPhysics
        from scenes.workrooms.living_room.tasks.water_pouring import assets
        report['snapshot_verified'] = verify_if_snapshot(PROJECT,
            required_modules=('scenes.workrooms.living_room.tasks.water_pouring.assets',),
            entrypoint_file=Path(__file__))
        settings = carb.settings.get_settings()
        if args.capture_ovd:
            (output/'ovd').mkdir()
            settings.set('/persistent/physics/omniPvdOvdRecordingDirectory', str(output/'ovd')+'/')
            settings.set_bool('/physics/omniPvdOutputEnabled', True)
        for name in ('updateToUsd', 'updateVelocitiesToUsd', 'updateParticlesToUsd'):
            settings.set_bool('/physics/'+name, True)
        omni.usd.get_context().new_stage()
        stage = omni.usd.get_context().get_stage()
        world = World(stage_units_in_meters=1., physics_dt=args.physics_dt,
                      rendering_dt=1/30, backend='torch', device='cuda:0',
                      physics_prim_path='/World/PhysicsScene')
        context = world.get_physics_context()
        context.enable_gpu_dynamics(True)
        context.set_broadphase_type('GPU')
        context.set_solver_type('TGS')
        def enforce_readback():
            # Keep the extension loaded: disabling it unloads the headless
            # experience and can tear down particle postprocessing as well.
            values = {'/physics/'+name: True for name in
                      ('updateToUsd', 'updateVelocitiesToUsd', 'updateParticlesToUsd')}
            values.update({'/physics/suppressReadback': False, '/physics/outputVelocitiesLocalSpace': False})
            if args.fabric_update_velocities:
                values[fabric_bindings.SETTING_FABRIC_UPDATE_VELOCITIES] = True
            for name, value in values.items():
                settings.set_bool(name, value)
            effective = {name: settings.get(name) for name in values}
            if effective != values:
                raise RuntimeError('Native USD readback settings did not stick')
            return effective
        report['readback_settings_before_reset'] = enforce_readback()
        physics = stage.GetPrimAtPath(context.prim_path)
        if str(physics.GetPath()) != '/World/PhysicsScene':
            raise RuntimeError('Probe scene path differs from author_task relationship')
        api = PhysxSchema.PhysxSceneAPI.Apply(physics)
        api.CreateEnableCCDAttr(False)
        api.CreateGpuMaxParticleContactsAttr(2**20)
        table = UsdGeom.Cube.Define(stage, '/World/ProbeTable')
        table.CreateSizeAttr(1.)
        table.AddTranslateOp().Set(Gf.Vec3d(.65, 0, .755))
        table.AddScaleOp().Set(Gf.Vec3f(1., 1., .05))
        UsdPhysics.CollisionAPI.Apply(table.GetPrim())
        metadata = assets.author_task(stage, water_mass_kg=args.water_mass_kg,
                                     particle_spacing_m=args.particle_spacing)
        if args.vessel_sleep_threshold is not None:
            if not np.isfinite(args.vessel_sleep_threshold) or args.vessel_sleep_threshold < 0:
                raise ValueError('Vessel sleep threshold must be nonnegative and finite')
            for info in metadata['containers'].values():
                PhysxSchema.PhysxRigidBodyAPI(stage.GetPrimAtPath(info['path'])).CreateSleepThresholdAttr(args.vessel_sleep_threshold)
        report['diagnostic_overrides'] = {'water_mass_kg': args.water_mass_kg,
            'particle_spacing_m': args.particle_spacing,
            'vessel_sleep_threshold': args.vessel_sleep_threshold,
            'note': 'Task-local diagnostic, no mass compensation or scripted fluid force'}
        particle_prim = stage.GetPrimAtPath(metadata['water']['particles_path'])
        system_prim = stage.GetPrimAtPath(metadata['water']['system_path'])
        if args.particle_mode == 'solid':
            # A diagnostic pressure-chain control, never a task-water asset.
            # Preserve the initial 3 mm spacing and mass rather than expanding
            # the particles to the fluid author's larger solid-rest diameter.
            PhysxSchema.PhysxParticleSetAPI(particle_prim).CreateFluidAttr(False)
            PhysxSchema.PhysxParticleSystem(system_prim).CreateSolidRestOffsetAttr(
                metadata['water']['fluid_rest_offset_m'])
        report['diagnostic_overrides']['particle_mode'] = args.particle_mode
        report['diagnostic_overrides']['solid_rest_offset_m'] = float(
            PhysxSchema.PhysxParticleSystem(system_prim).GetSolidRestOffsetAttr().Get())
        if args.particle_iterations is not None:
            if not 1 <= args.particle_iterations <= 255:
                raise ValueError('Particle iterations must be in [1,255]')
            PhysxSchema.PhysxParticleSystem(system_prim).CreateSolverPositionIterationCountAttr(args.particle_iterations)
        report['diagnostic_overrides']['particle_iterations'] = args.particle_iterations
        if args.fluid_boundary_density_scale is not None:
            if not np.isfinite(args.fluid_boundary_density_scale) or not 0. <= args.fluid_boundary_density_scale <= 1.:
                raise ValueError('Fluid boundary density scale must be finite within [0,1]')
            # Local native parser contains this SDK attribute although this
            # runtime schema does not expose it. OVD must confirm consumption.
            system_prim.CreateAttribute('physxParticle:fluidBoundaryDensityScale', Sdf.ValueTypeNames.Float).Set(args.fluid_boundary_density_scale)
        report['diagnostic_overrides']['fluid_boundary_density_scale'] = args.fluid_boundary_density_scale
        report['particle_schema_registration'] = {
            'system_typed': bool(PhysxSchema.PhysxParticleSystem(system_prim)),
            'set_api': bool(PhysxSchema.PhysxParticleSetAPI(particle_prim)),
            'particle_api': bool(PhysxSchema.PhysxParticleAPI(particle_prim)),
            'smoothing_api': bool(PhysxSchema.PhysxParticleSmoothingAPI(system_prim)),
            'isosurface_api': bool(PhysxSchema.PhysxParticleIsosurfaceAPI(system_prim)),
            'particle_enabled': PhysxSchema.PhysxParticleAPI(particle_prim).GetParticleEnabledAttr().Get(),
            'applied': particle_prim.GetAppliedSchemas()}
        dump(output/'asset_metadata.json', metadata)
        report['asset_source_sha256'] = metadata['source_sha256']
        report['actual_physics_scene_path'] = str(physics.GetPath())
        report['runtime_version'] = (Path(os.environ['ISAAC_PATH'])/'VERSION').read_text().strip()
        report['dry_control_mutation'] = 'Remove particle set and particle system before initialization' if args.dry_control else None
        if args.dry_control:
            stage.RemovePrim(metadata['water']['particles_path'])
            stage.RemovePrim(metadata['water']['system_path'])
        for container in metadata['containers'].values():
            PhysxSchema.PhysxContactReportAPI.Apply(stage.GetPrimAtPath(container['path'])).CreateThresholdAttr(0.)
        # The installed particle postprocessing tests require two idle Kit
        # updates for deferred creation before starting simulation.
        app.update()
        app.update()
        report['idle_particle_initialization_updates'] = 2
        stage.Export(str(output/'initial.usda'))
        world.reset()
        report['readback_settings_after_reset'] = enforce_readback()
        sim = world.physics_sim_view
        views = {name: sim.create_rigid_body_view(info['path']) for name, info in metadata['containers'].items()}
        if any(view.count != 1 for view in views.values()):
            raise RuntimeError('Native dynamic vessel body inventory mismatch')
        report['native_body_masses_kg'] = {name: float(array(view.get_masses()).reshape(-1)[0]) for name, view in views.items()}
        report['authored_sleep_thresholds'] = {name: PhysxSchema.PhysxRigidBodyAPI(stage.GetPrimAtPath(info['path'])).GetSleepThresholdAttr().Get()
                                               for name, info in metadata['containers'].items()}
        contact = sim.create_rigid_contact_view(metadata['containers']['kettle']['path'],
            filter_patterns=[metadata['tray']['path']+'/Floor'], max_contact_data_count=4096)
        report['rigid_floor_contact_mapping'] = {'sensors': list(contact.sensor_paths),
                                                'filters': list(contact.filter_paths)}
        initial_pose = {name: array(view.get_transforms())[0].tolist() for name, view in views.items()}
        import torch
        tensor_device = views['kettle'].get_transforms().device
        force_vector = torch.tensor([[0., 0., args.upward_force_n]], dtype=torch.float32, device=tensor_device)
        force_indices = torch.tensor([0], dtype=torch.int32, device=tensor_device)
        kettle_com_local = array(views['kettle'].get_coms())[0, :3]
        if not np.isfinite(args.upward_force_n) or not 0. <= args.upward_force_n <= 8.:
            raise ValueError('Diagnostic upward force must be finite and within [0,8] N')
        report['external_load_protocol'] = {'force_n': args.upward_force_n, 'start_after_measured_s': 1.5,
            'duration_s': .25, 'on': 'kettle only, world +Z at native center of mass',
            'tensor_device': str(tensor_device), 'native_kettle_com_local_m': kettle_com_local.tolist(),
            'scope': 'Independent test actuator, no robot controller, no fluid force/mass compensation'}
        previous_time = float(world.current_time)
        report['initial_native_time_s'] = previous_time
        steps = round(args.seconds/args.physics_dt)
        rows_file = output/'samples.jsonl'
        with rows_file.open('w') as handle:
            for tick in range(1, steps+1):
                load_active = 1.5 <= (tick-1)*args.physics_dt < 1.75
                if load_active and args.upward_force_n:
                    pose = array(views['kettle'].get_transforms())[0].astype(float)
                    world_matrix = Gf.Matrix4d(1.)
                    world_matrix.SetRotate(Gf.Quatd(float(pose[6]), Gf.Vec3d(*pose[3:6])))
                    world_matrix.SetTranslateOnly(Gf.Vec3d(*pose[:3]))
                    world_com = world_matrix.Transform(Gf.Vec3d(*kettle_com_local.astype(float)))
                    position = torch.tensor([list(world_com)], dtype=torch.float32, device=tensor_device)
                    views['kettle'].apply_forces_and_torques_at_position(force_vector, None, position, force_indices, True)
                world.step(render=False)
                velocity_before_sync = array(views['kettle'].get_velocities())[0].tolist()
                omni.physx.get_physx_simulation_interface().fetch_results()
                # Isaac 6.1 SimulationManager.step advances the solver only.
                # Explicitly fetch its physical results to USD, independently
                # of rendering or whether Fabric happens to be enabled.
                omni.physx.get_physx_interface().update_transformations(False, True, True, False)
                if args.sync_mode == 'legacy_save':
                    fabric = get_physx_fabric_interface()
                    fabric.force_update(args.physics_dt, world.current_time)
                    fabric.save_to_usd()
                velocity_after_sync = array(views['kettle'].get_velocities())[0].tolist()
                if args.sync_mode == 'read_only' and not np.allclose(velocity_before_sync, velocity_after_sync, atol=1.e-8, rtol=1.e-6):
                    raise RuntimeError('Readback changed native kettle velocity')
                if tick == 1 and not args.dry_control:
                    report['first_step_schema_stats'] = {
                        'native_stats': omni.physx.get_physxunittests_interface().get_physics_stats(),
                        'physics_gpu': api.GetEnableGPUDynamicsAttr().Get(),
                        'simulation_points_attribute_count': len(particle_prim.GetAttribute('physxParticle:simulationPoints').Get() or []),
                        'unsmoothed_physical_points_count': len(UsdGeom.Points(particle_prim).GetPointsAttr().Get() or []),
                        'all_particle_attributes': [a.GetName() for a in particle_prim.GetAttributes()],
                        'isosurface_children': [str(p.GetPath()) for p in system_prim.GetChildren()]}
                    dump(output/'first_step_diagnostic.json', report['first_step_schema_stats'])
                    stage.Flatten().Export(str(output/'first_step.usda'))
                time = float(world.current_time)
                if abs(time-previous_time-args.physics_dt) > 1.e-7:
                    raise RuntimeError('Native time did not advance by exactly one requested physics step')
                previous_time = time
                bodies = {name: {'pose_xyzw': array(view.get_transforms())[0].tolist(),
                                 'velocity_linear_angular': array(view.get_velocities())[0].tolist()}
                          for name, view in views.items()}
                state = None if args.dry_control else assets.read_water_state(stage, metadata,
                    container_pose_xyzw={name: body['pose_xyzw'] for name, body in bodies.items()})
                if state is not None:
                    raw_points = np.asarray(particle_prim.GetAttribute('points').Get(), dtype=float)
                    raw_velocities = np.asarray(particle_prim.GetAttribute('velocities').Get(), dtype=float)
                    state['center_of_mass_m'] = raw_points.mean(axis=0).tolist()
                    state['mean_velocity_m_s'] = raw_velocities.mean(axis=0).tolist()
                    state['total_momentum_kg_m_s'] = (raw_velocities.sum(axis=0)*metadata['water']['particle_mass_kg']).tolist()
                usd = UsdGeom.XformCache()
                for name, body in bodies.items():
                    p = stage.GetPrimAtPath(metadata['containers'][name]['path'])
                    xyz = np.asarray(usd.GetLocalToWorldTransform(p).ExtractTranslation())
                    body['usd_to_native_translation_error_m'] = float(np.linalg.norm(xyz-body['pose_xyzw'][:3]))
                    if args.sync_mode == 'legacy_save' and body['usd_to_native_translation_error_m'] > 1.e-4:
                        raise RuntimeError('USD vessel transform differs from native rigid body')
                reaction = array(contact.get_contact_force_matrix(args.physics_dt)).reshape(-1, 3).sum(axis=0)
                net_reaction = array(contact.get_net_contact_forces(args.physics_dt)).reshape(-1, 3).sum(axis=0)
                force, position, normal, separation, count, start = map(array, contact.get_contact_data(args.physics_dt))
                indices = [i for n, s in zip(count.ravel(), start.ravel()) for i in range(int(s), int(s+n))]
                detailed = (force[indices].reshape(-1, 1)*normal[indices]).sum(axis=0)
                row = {'tick': tick, 'time_s': time, 'water': state, 'bodies': bodies,
                       'applied_upward_force_n': args.upward_force_n if load_active else 0.,
                       'kettle_velocity_before_sync': velocity_before_sync, 'kettle_velocity_after_sync': velocity_after_sync,
                       'kettle_floor_reaction_n': reaction.tolist(), 'kettle_net_rigid_contact_reaction_n': net_reaction.tolist(),
                       'kettle_floor_detailed_normal_reaction_n': detailed.tolist(), 'floor_contact_count': len(indices)}
                if not np.isfinite(reaction).all():
                    raise RuntimeError('Nonfinite native floor reaction')
                samples.append(row)
                handle.write(json.dumps(row, allow_nan=False)+'\n')
                if tick % 120 == 0:
                    handle.flush()
                    print(json.dumps({'tick': tick, 'time_s': time, 'water': state}), flush=True)
        # Save only after all measured physics steps: live save_to_usd can
        # feed state back into PhysX. No further physics step uses this export.
        world.pause()
        fabric = get_physx_fabric_interface()
        fabric.force_update(args.physics_dt, world.current_time)
        fabric.save_to_usd()
        for _ in range(3):
            app.update()
        stage.Flatten().Export(str(output/'settled_snapshot.usda'))
        raw = None
        if not args.dry_control:
            raw = np.asarray(stage.GetPrimAtPath(metadata['water']['particles_path']).GetAttribute(
                metadata['water']['raw_position_attribute']).Get(), dtype=np.float32)
            np.savez_compressed(output/'settled_particles.npz', positions_m=raw,
                velocities_m_s=np.asarray(UsdGeom.Points(stage.GetPrimAtPath(metadata['water']['particles_path'])).GetVelocitiesAttr().Get()),
                particle_mass_kg=metadata['water']['particle_mass_kg'])
        tail = samples[-round(1/args.physics_dt):]
        reaction_z = np.array([row['kettle_floor_reaction_n'][2] for row in tail])
        expected_mass = .32 + (0. if args.dry_control else args.water_mass_kg)
        retained = args.dry_control or all(row['water']['counts']['outside_vessels'] == 0 and
            row['water']['counts']['kettle'] == metadata['water']['particle_count'] for row in samples)
        raw_updates = args.dry_control or len({row['water']['positions_sha256'] for row in samples}) > 1
        vessel_motion = {name: {'max_translation_from_initial_m': max(float(np.linalg.norm(
            np.array(row['bodies'][name]['pose_xyzw'][:3])-initial_pose[name][:3])) for row in samples),
            'max_final_second_linear_speed_m_s': max(float(np.linalg.norm(row['bodies'][name]['velocity_linear_angular'][:3])) for row in tail),
            'max_final_second_angular_speed_rad_s': max(float(np.linalg.norm(row['bodies'][name]['velocity_linear_angular'][3:])) for row in tail)}
            for name in views}
        report.update({'status': 'completed', 'native_steps': len(samples), 'native_duration_s': previous_time,
            'measured_interval_s': previous_time-report['initial_native_time_s'],
            'initial_pose_xyzw': initial_pose, 'vessel_motion': vessel_motion,
            'fixed_inventory_and_static_retention': retained,
            'raw_particle_positions_changed_during_run': raw_updates,
            'final_water': samples[-1]['water'], 'maximum_outside_particle_count': 0 if args.dry_control else max(row['water']['counts']['outside_vessels'] for row in samples),
            'floor_reaction_last_second': {'mean_z_n': float(reaction_z.mean()), 'min_z_n': float(reaction_z.min()),
                'max_z_n': float(reaction_z.max()), 'expected_weight_n': expected_mass*9.81,
                'relative_weight_error': float(abs(reaction_z.mean()-expected_mass*9.81)/(expected_mass*9.81)),
                'scope': 'Measured rigid-body floor reaction; paired dry control still needed for quantitative liquid-load interpretation'},
            'contact_readout_crosscheck': {
                'net_mean_z_n': float(np.mean([row['kettle_net_rigid_contact_reaction_n'][2] for row in tail])),
                'detailed_normal_mean_z_n': float(np.mean([row['kettle_floor_detailed_normal_reaction_n'][2] for row in tail])),
                'max_net_minus_filtered_n': max(float(np.linalg.norm(np.array(row['kettle_net_rigid_contact_reaction_n'])-row['kettle_floor_reaction_n'])) for row in tail),
                'max_detailed_minus_filtered_n': max(float(np.linalg.norm(np.array(row['kettle_floor_detailed_normal_reaction_n'])-row['kettle_floor_reaction_n'])) for row in tail)},
            'isosurface_mesh_paths': [str(p.GetPath()) for p in stage.Traverse()
                if p.IsA(UsdGeom.Mesh) and str(p.GetPath()).startswith(metadata['water']['system_path'])],
            'snapshot_path': str(output/'settled_snapshot.usda'),
            'snapshot_sha256': hashlib.sha256((output/'settled_snapshot.usda').read_bytes()).hexdigest(),
            'passed': retained and raw_updates and all(v['max_translation_from_initial_m'] < .01
                and v['max_final_second_linear_speed_m_s'] < .02
                and v['max_final_second_angular_speed_rad_s'] < .2 for v in vessel_motion.values()),
            'force_transfer_qualified': False, 'robot_load_or_pouring_qualified': False})
        report['readback_velocity_integrity'] = {
            'max_sync_velocity_change': max(float(np.linalg.norm(np.array(row['kettle_velocity_after_sync'])-row['kettle_velocity_before_sync'])) for row in samples),
            'live_fabric_saves': len(samples) if args.sync_mode == 'legacy_save' else 0,
            'export_save_after_last_physics_step': True}
        report['external_load_response'] = {'max_kettle_rise_m': max(row['bodies']['kettle']['pose_xyzw'][2]-initial_pose['kettle'][2] for row in samples),
            'force_impulse_ns': sum(row['applied_upward_force_n']*args.physics_dt for row in samples),
            'sampled_load_steps': sum(row['applied_upward_force_n'] != 0 for row in samples),
            'interpretation': 'Measure response only; retention pass is not a load-qualification decision'}
        if args.capture_ovd:
            world.stop()
            ovds = sorted(p for p in (output/'ovd').glob('*.ovd') if p.name != 'tmp.ovd')
            if not ovds:
                raise RuntimeError('Native OVD recording was not flushed')
            report['native_ovd_files'] = [{'path': str(p), 'bytes': p.stat().st_size,
                'sha256': hashlib.sha256(p.read_bytes()).hexdigest()} for p in ovds]
        code = 0 if report['passed'] else 1
    except Exception as exc:
        report.update({'status': 'failed', 'error': {'type': type(exc).__name__, 'message': str(exc),
                                                  'traceback': traceback.format_exc()}, 'native_steps': len(samples)})
    finally:
        report['native_shutdown_verified'] = False
        report['requires_external_exit_zero'] = True
        dump(output/'report.json', report)
        print(json.dumps({'event': 'probe_report', 'passed': report['passed'], 'output': str(output)}), flush=True)
        if app is not None:
            app.close(exit_code=code)
    return code


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seconds', type=float, default=3.)
    parser.add_argument('--physics-dt', type=float, default=1/960)
    parser.add_argument('--dry-control', action='store_true')
    parser.add_argument('--water-mass-kg', type=float, default=.25)
    parser.add_argument('--particle-spacing', type=float, default=.003,
                        help='Declared resolution diagnostic in metres; total water mass stays fixed')
    parser.add_argument('--particle-mode', choices=['fluid', 'solid'], default='fluid',
                        help='Solid is an explicit pressure-chain diagnostic, not a water candidate')
    parser.add_argument('--vessel-sleep-threshold', type=float)
    parser.add_argument('--particle-iterations', type=int)
    parser.add_argument('--fluid-boundary-density-scale', type=float)
    parser.add_argument('--capture-ovd', action='store_true')
    parser.add_argument('--upward-force-n', type=float, default=0.)
    parser.add_argument('--fabric-update-velocities', action='store_true')
    parser.add_argument('--sync-mode', choices=['read_only', 'legacy_save'], default='read_only')
    args = parser.parse_args()
    raise SystemExit(run(args))


if __name__ == '__main__':
    main()
