"""Native FEM sponge compression/rebound or bounded-force wiping diagnostic.

The exact frozen two-layer asset remains dynamic. A separate dynamic platen
receives a bounded external wrench; the wiping case applies a bounded force
only to the real scouring layer. Neither case writes foam nodes, rigid poses or
velocities after reset. This is engineering object dynamics, not a robot task.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import traceback

import numpy as np

from bench.isaacsim.release import checked, digest, dump
from bench.paths import load_json, OUTPUT_ROOT
from scenes.workrooms.kitchen.sponge_dynamics import NodeStability, VolumeGeometry
from scenes.workrooms.kitchen.rules import thresholds
from scenes.workrooms.kitchen.wiping import ContactWiping
from scripts.maintenance.freeze_project_sources import verify_if_snapshot
from scripts.validation.sponge_scene_options import apply_velocity_iteration_clamp, velocity_iteration_receipt

PROJECT = Path(__file__).resolve().parents[2]


def smooth(u):
    return u*u*(3.-2.*u)


def native_array(value):
    if hasattr(value, 'detach'):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def probe_inputs(manifest_path):
    manifest = load_json(manifest_path)
    root = manifest_path.parent
    usd, metadata = root/'task.usda', root/'metadata.json'
    checked(usd, manifest['usd_sha256'])
    checked(metadata, manifest['metadata_sha256'])
    return manifest, usd, load_json(metadata)


def run(args):
    source = verify_if_snapshot(PROJECT,
        required_modules=('scripts.validation.kitchen_sponge_dynamics', 'scripts.validation.sponge_scene_options'), entrypoint_file=Path(__file__))
    if source is None or Path.cwd().resolve() != PROJECT:
        raise ValueError('Use the complete frozen source root')
    output = args.output.resolve()
    if not output.is_relative_to(OUTPUT_ROOT):
        raise ValueError('Evidence belongs in shared datasets')
    output.mkdir(parents=True, exist_ok=False)
    report = {'schema': 'kitchen_bonded_sponge_dynamics.v1', 'passed': False,
        'case': args.case, 'physics_dt_s': args.dt, 'source': source,
        'solver_selection': args.solver,
        'external_forces_every_iteration_requested': bool(args.external_forces_every_iteration),
        'rigid_velocity_iterations_override': args.rigid_velocity_iterations,
        'scene_velocity_iterations_requested': args.scene_velocity_iterations,
        'support_contact_offset_m': .002,
        'platen_contact_offset_m': .0002,
        'instrument_rest_offset_m': 0.,
        'asset_manifest': str(args.asset_manifest.resolve()),
        'asset_manifest_sha256': digest(args.asset_manifest),
        'robot_present': False, 'task_success_claimed': False, 'model_calls': 0,
        'post_reset_foam_node_writes': 0, 'post_reset_sponge_pose_or_velocity_writes': 0,
        'gpu_uuid': os.environ.get('CUDA_VISIBLE_DEVICES'),
        'material_bond_disabled_ablation': bool(args.disable_material_bond),
        'scope': 'Native two-layer object dynamics on a material-matched table proxy; no robot operation or calibrated foam claim.'}
    app, code = None, 1
    states, nodes, velocities, times, rows, platen_states = [], [], [], [], [], []
    chunks, archived_frames = [], 0

    def flush_raw_chunk():
        nonlocal archived_frames
        if len(times) == archived_frames:
            return
        first, stop = archived_frames, len(times)
        partial = output/'native_chunks'
        partial.mkdir(exist_ok=True)
        path = partial/f'frames_{len(chunks):06d}.npz'
        state_array = np.asarray(states[first:stop])
        np.savez_compressed(path, time_s=np.asarray(times[first:stop]),
            native_step=np.asarray([row['native_step'] for row in rows[first:stop]]),
            scouring_q=state_array[:, :7], scouring_qd=state_array[:, 7:],
            foam_q=np.asarray(nodes[first:stop]), foam_qd=np.asarray(velocities[first:stop]),
            platen_state=np.asarray(platen_states[first:stop]))
        chunks.append({'path': str(path), 'frames': stop-first,
            'first_frame': first, 'last_frame': stop-1, 'sha256': digest(path)})
        archived_frames = stop
        dump(output/'native_chunks_manifest.json', {'schema': 'native_sponge_probe_chunks.v1',
            'frames': stop, 'chunks': chunks, 'trajectory_complete': False})
    try:
        manifest, usd, meta = probe_inputs(args.asset_manifest)
        if meta['task_id'] != 'surface_wiping' or set(meta.get('volume_bodies', {})) != {'foam'}:
            raise ValueError('Exact bonded sponge asset required')
        foam = meta['volume_bodies']['foam']
        if args.disable_material_bond and args.case != 'support':
            raise ValueError('Material-bond ablation is limited to passive support diagnosis')
        if args.external_forces_every_iteration and args.solver != 'TGS':
            raise ValueError('Per-iteration external forces require TGS')
        geometry = VolumeGeometry(foam['rest_points_m'], foam['tetrahedra'])
        mass = float(foam['total_mass_kg']) + float(meta['objects'][0]['mass_kg'])
        from bench.simulation.isaacsim_backend.render_sync import startup_arguments, verify_settings
        from isaacsim import SimulationApp
        app = SimulationApp({'headless': True, 'width': 640, 'height': 480,
            'multi_gpu': False, 'active_cuda_gpus': [0], 'max_gpu_count': 1, 'physics_gpu': 0,
            'fast_shutdown': True, 'extra_args': startup_arguments() + (
                ['--/rtx/verifyDriverVersion/enabled=false']
                if os.environ.get('ISAAC_ALLOW_UNVALIDATED_DRIVER') == '1' else [])},
            experience=str(PROJECT/'bench/apps/workroom.headless.kit'))
        import carb
        report['render_sync'] = verify_settings(carb.settings.get_settings())
        import omni.usd
        from isaacsim.core.api import World
        from pxr import Gf, UsdGeom, UsdPhysics, UsdShade, PhysxSchema
        from bench.simulation.isaacsim_backend.contact_precision import summarize_normal_impulses
        omni.usd.get_context().new_stage()
        stage = omni.usd.get_context().get_stage()
        UsdGeom.SetStageMetersPerUnit(stage, 1.)
        UsdGeom.SetStageUpAxis(stage, 'Z')
        task_root = stage.DefinePrim(manifest['root_prim_path'], 'Xform')
        task_root.GetReferences().AddReference(str(usd), manifest['root_prim_path'])
        if args.disable_material_bond:
            # Explicit diagnostic runtime override, never a frozen asset edit.
            stage.GetPrimAtPath(foam['attachment']['prim_path']).SetActive(False)
            stage.GetPrimAtPath(foam['interface_collision_filter']['prim_path']).SetActive(False)
        # Translate both material layers before reset, preserving their bond.
        old = np.asarray(meta['objects'][0]['initial_pose_xyzw'][:3])
        center = np.mean([r['center_world_m'][:2] for r in meta['goals']['soil_cells']
                          if r['patch'] == 'soil_1'], axis=0)
        start = np.array([center[0]-.08, center[1]-.019, .9003])
        if args.case in ('compression', 'support'):
            start[:2] = [.60, -.80]
        shift = start-old
        for path in (meta['actors']['sponge'], foam['prim_path']):
            prim = stage.GetPrimAtPath(path)
            translate = prim.GetAttribute('xformOp:translate')
            if not translate or translate.Get() is None:
                raise ValueError('Material-layer root requires an explicit authored translation')
            translate.Set(Gf.Vec3d(*(np.asarray(translate.Get())+shift)))
        support = '/World/TableProxy'
        material = UsdShade.Material.Define(stage, '/World/ProbeContact')
        physics = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
        physics.CreateStaticFrictionAttr(.4)
        physics.CreateDynamicFrictionAttr(.4)
        physics.CreateRestitutionAttr(0.)

        def test_box(path, center, size, dynamic_mass=None, contact_offset=.002):
            cube = UsdGeom.Cube.Define(stage, path)
            cube.CreateSizeAttr(1.)
            cube.AddTranslateOp().Set(Gf.Vec3d(*center))
            cube.AddScaleOp().Set(Gf.Vec3f(*size))
            UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
            collision = PhysxSchema.PhysxCollisionAPI.Apply(cube.GetPrim())
            collision.CreateContactOffsetAttr(contact_offset)
            collision.CreateRestOffsetAttr(0.)
            UsdPhysics.RigidBodyAPI.Apply(cube.GetPrim()).CreateKinematicEnabledAttr(dynamic_mass is None)
            if dynamic_mass is not None:
                inertia = dynamic_mass/12.*np.array([size[1]**2+size[2]**2,
                    size[0]**2+size[2]**2, size[0]**2+size[1]**2])
                mass_api = UsdPhysics.MassAPI.Apply(cube.GetPrim())
                mass_api.CreateMassAttr(dynamic_mass)
                mass_api.CreateCenterOfMassAttr(Gf.Vec3f(0.))
                mass_api.CreateDiagonalInertiaAttr(Gf.Vec3f(*inertia))
            UsdShade.MaterialBindingAPI.Apply(cube.GetPrim()).Bind(material, materialPurpose='physics')
            return cube.GetPrim()

        test_box(support, [.45, -.95, .8775], [2.8, .8, .045])
        body = stage.GetPrimAtPath(meta['actors']['sponge'])
        if args.rigid_velocity_iterations is not None:
            PhysxSchema.PhysxRigidBodyAPI.Apply(body).CreateSolverVelocityIterationCountAttr(args.rigid_velocity_iterations)
        PhysxSchema.PhysxContactReportAPI.Apply(body).CreateThresholdAttr(0.)
        top = .9003+.045
        platen_center = [*start[:2], top+.010]
        if args.case == 'compression':
            test_box('/World/PressPlaten', platen_center, [.070, .044, .008], .1, .0002)
        world = World(stage_units_in_meters=1., physics_dt=args.dt, rendering_dt=1/30,
            backend='torch', device='cuda:0', physics_prim_path='/World/PhysicsScene')
        pc = world.get_physics_context()
        pc.enable_gpu_dynamics(True)
        pc.set_broadphase_type('GPU')
        pc.set_solver_type(args.solver)
        physics_prim = stage.GetPrimAtPath('/World/PhysicsScene')
        report['scene_velocity_iteration_clamp'] = apply_velocity_iteration_clamp(physics_prim, args.scene_velocity_iterations)
        if args.external_forces_every_iteration:
            PhysxSchema.PhysxSceneAPI.Apply(physics_prim).CreateEnableExternalForcesEveryIterationAttr(True)
        report['effective_physics_scene'] = {key: physics_prim.GetAttribute(key).Get()
            for key in ('physxScene:solverType', 'physxScene:timeStepsPerSecond',
                        'physxScene:enableGPUDynamics', 'physxScene:enableExternalForcesEveryIteration',
                        'physxScene:minVelocityIterationCount', 'physxScene:maxVelocityIterationCount',
                        'physxScene:minPositionIterationCount', 'physxScene:maxPositionIterationCount',
                        'physics:gravityMagnitude')}
        report['rigid_solver_usd_readback'] = {key: body.GetAttribute(key).Get()
            for key in ('physxRigidBody:solverPositionIterationCount',
                        'physxRigidBody:solverVelocityIterationCount')}
        stage.Export(str(output/'initial.usda'))
        world.reset()
        report['scene_velocity_iteration_clamp_after_reset'] = velocity_iteration_receipt(physics_prim, args.scene_velocity_iterations)
        sim = world.physics_sim_view
        pad = sim.create_rigid_body_view(meta['actors']['sponge'])
        volume = sim.create_volume_deformable_body_view(foam['prim_path'])
        contact = sim.create_rigid_contact_view(meta['actors']['sponge'],
            filter_patterns=[support], max_contact_data_count=8192)
        platen = sim.create_rigid_body_view('/World/PressPlaten') if args.case == 'compression' else None
        if pad.count != 1 or volume.count != 1 or contact.filter_count != 1:
            raise ValueError('Expected one native rigid layer, foam and table contact pair')
        material_view = sim.create_deformable_material_view(foam['material']['material_path'])
        if material_view.count != 1:
            raise ValueError('Native foam material was not created')
        effective_material = {name: float(native_array(getattr(material_view, getter)()).reshape(-1)[0])
            for name, getter in (('youngs_modulus_pa', 'get_youngs_modulus'),
                                 ('poissons_ratio', 'get_poissons_ratio'),
                                 ('dynamic_friction', 'get_dynamic_friction'))}
        if any(not np.isclose(value, foam['material'][name], rtol=1.e-5, atol=1.e-7)
               for name, value in effective_material.items()):
            raise ValueError('Effective native foam material differs from the frozen declaration')
        native_tets = native_array(volume.get_simulation_element_indices()).reshape(-1, 4)
        if not np.array_equal(native_tets, foam['tetrahedra']):
            raise ValueError('Native tetrahedral topology changed')
        report['native_material'] = effective_material
        report['native_rigid_mass_kg'] = float(native_array(pad.get_masses()).reshape(-1)[0])
        if platen is not None:
            report['native_platen_mass_kg'] = float(native_array(platen.get_masses()).reshape(-1)[0])
        support_view = sim.create_rigid_body_view(support)
        report['native_rigid_collision_offsets_m'] = {
            name: {'prim_paths': list(view.prim_paths),
                   'max_shapes': int(view.max_shapes),
                   'contact': native_array(view.get_contact_offsets()).copy().tolist(),
                   'rest': native_array(view.get_rest_offsets()).copy().tolist()}
            for name, view in [('support', support_view), ('scouring', pad),
                               *([('platen', platen)] if platen is not None else [])]}
        report['composed_collision_offsets_m'] = [
            {'path': str(prim.GetPath()),
             **{key: {'value': prim.GetAttribute('physxCollision:'+key).Get(),
                      'authored': prim.GetAttribute('physxCollision:'+key).HasAuthoredValueOpinion()}
                for key in ('contactOffset', 'restOffset')}}
            for prim in stage.Traverse() if prim.HasAPI(UsdPhysics.CollisionAPI)]
        report['native_volume_paths'] = list(volume.prim_paths)
        report['native_simulation_mesh_paths'] = list(volume.simulation_mesh_prim_paths)
        report['native_collision_mesh_paths'] = list(volume.collision_mesh_prim_paths)
        report['foam_contact_impulses_available'] = False
        dump(output/'native_initialization.json', {**report,
            'effective_asset_override': 'material bond and its contact filter deactivated before reset'
                if args.disable_material_bond else None})
        visual, visual_rows, visual_errors = None, [], []
        media, media_rows = None, []
        if args.media_output:
            from scripts.media.native_press_media import NativePressMedia
            media = NativePressMedia(stage, foam, args.media_output)
        if args.visual_readback:
            from scripts.validation.deformable_visual_probe import NativeSkinProbe
            visual = NativeSkinProbe(stage, foam, output/'visual_readback')

        def sample_visual(label):
            if visual is None:
                return
            try:
                visual_rows.append(visual.sample(label, world, volume, pad))
            except Exception:
                visual_errors.append({'label': label, 'exception': traceback.format_exc()})
                dump(output/'visual_readback_errors.json', visual_errors)

        import torch
        def tensor(value):
            return torch.as_tensor(value, device='cuda:0')
        one = tensor(np.array([0], np.int32))
        report['tensor_frontend'] = 'torch_cuda:0'
        limits = thresholds('surface_wiping')
        wiping = ContactWiping(meta, limits, args.dt)
        stability = NodeStability(limits, args.dt)
        rest = np.asarray(foam['rest_points_m'])
        top_ids = np.flatnonzero((rest[:, 2] >= rest[:, 2].max()-1.e-6)
            & (np.abs(rest[:, 0]) <= .033) & (np.abs(rest[:, 1]) <= .020))
        if len(top_ids) < 3:
            raise ValueError('Compression readback needs frozen central top-surface nodes')
        report['compression_measurement_node_indices'] = top_ids.tolist()
        initial_step, initial_time = int(world.current_time_step_index), float(world.current_time)
        platen_force, platen_torque = np.zeros(3), np.zeros(3)
        platen_target = np.asarray(platen_center)
        loaded_count = 0

        def sample(phase, applied_force, baseline=False):
            nonlocal loaded_count
            pose = native_array(pad.get_transforms())[0].copy()
            velocity = native_array(pad.get_velocities())[0].copy()
            points = native_array(volume.get_simulation_nodal_positions())[0].copy()
            qd = native_array(volume.get_simulation_nodal_velocities())[0].copy()
            try:
                measure = geometry.sample(points, qd)
            except Exception:
                np.savez_compressed(output/'invalid_native_frame.npz', foam_q=points, foam_qd=qd,
                    scouring_q=pose, scouring_qd=velocity,
                    native_step=int(world.current_time_step_index), time_s=float(world.current_time))
                raise
            from scenes.workrooms.kitchen.native_evidence import rotation
            r = np.asarray(rotation([float(pose[6]), *pose[3:6].tolist()]))
            anchors = np.asarray(foam['attachment']['rigid_local_positions_m']) @ r.T + pose[:3]
            measure['maximum_material_bond_error_m'] = float(np.linalg.norm(
                points[np.asarray(foam['attachment']['node_indices'])]-anchors, axis=1).max())
            measure['central_top_height_in_pad_frame_m'] = float(np.median(((points-pose[:3]) @ r)[top_ids, 2]))
            pad_points = np.asarray(meta['objects'][0]['collision_support_points_m']) @ r.T + pose[:3]
            measure['scouring_lowest_z_m'] = float(pad_points[:, 2].min())
            # Read the actual platen pose; no loaded/held sample earns release.
            tool_gap = None
            separated = bool(np.linalg.norm(applied_force) == 0.)
            if platen is not None:
                actual_platen = native_array(platen.get_transforms())[0]
                platen_velocity = native_array(platen.get_velocities())[0]
                platen_states.append(np.concatenate([actual_platen, platen_velocity]))
                all_points = np.concatenate([points, pad_points])
                lower, upper = all_points.min(0), all_points.max(0)
                half = np.array([.035, .022, .004])
                platen_rotation = np.asarray(rotation([float(actual_platen[6]), *actual_platen[3:6].tolist()]))
                world_half = np.abs(platen_rotation) @ half
                delta = np.maximum(np.maximum(actual_platen[:3]-world_half-upper,
                                              lower-(actual_platen[:3]+world_half)), 0.)
                tool_gap = float(np.linalg.norm(delta))
                separated = separated and tool_gap >= .005
            measure['platen_to_composite_bbox_gap_m'] = tool_gap
            measure['stability'] = stability.update(points, fully_separated=separated, baseline=baseline)
            t = float(world.current_time)-initial_time
            impulse = np.zeros(3)
            pairs = []
            if not baseline:
                f, _, n, sep, counts, starts = contact.get_contact_data(dt=1.)
                f, n, sep, counts, starts = map(native_array, (f, n, sep, counts, starts))
                count, begin = int(counts[0, 0]), int(starts[0, 0])
                if count:
                    summary = summarize_normal_impulses(np.asarray(f)[begin:begin+count].reshape(-1),
                                                       np.asarray(n)[begin:begin+count])
                    pairs = [{'actor0': meta['actors']['sponge'], 'actor1': support,
                        'point_count': count, 'min_separation_m': float(np.asarray(sep)[begin:begin+count].min()),
                        'raw_scalar_normal_impulses_ns': np.asarray(f)[begin:begin+count].reshape(-1).tolist(),
                        'raw_normals': np.asarray(n)[begin:begin+count].tolist(),
                        'raw_separations_m': np.asarray(sep)[begin:begin+count].reshape(-1).tolist(), **summary}]
                    impulse = np.asarray(summary['impulse_vector_ns'])
            loaded = impulse[2] > 1.e-10
            loaded_count += int(loaded)
            wipe = wiping.update({'position_m': pose[:3].tolist(),
                'quaternion_wxyz': [float(pose[6]), *pose[3:6].tolist()]}, loaded_contact=loaded, baseline=baseline)
            row = {'phase': phase, 'time_s': t, 'native_step': int(world.current_time_step_index),
                'baseline': baseline, 'force_applied_to_scouring_layer_n': list(applied_force),
                'contacts': pairs, 'normal_table_force_n': float(impulse[2]/args.dt),
                'geometry': measure, 'coverage': wipe}
            if platen is not None:
                row['platen'] = {'pose_xyzw': actual_platen.tolist(), 'velocity': platen_velocity.tolist(),
                    'desired_position_m': platen_target.tolist(),
                    'applied_force_n': platen_force.tolist(), 'applied_torque_nm': platen_torque.tolist()}
            rows.append(row)
            states.append([*pose, *velocity]); nodes.append(points); velocities.append(qd); times.append(t)
            if len(times)-archived_frames >= 256:
                flush_raw_chunk()
            return row

        sample('baseline', [0., 0., 0.], True)
        sample_visual('baseline')
        if args.case == 'compression':
            plan = [('settle', .8), ('press', .8), ('loaded_hold', .4), ('withdraw', .8), ('recover', 1.)]
        elif args.case == 'support':
            plan = [('settle', .2), ('release', .4)]
        else:
            plan = [('settle', .4), ('forward', 2.4), ('side_step', .8), ('backward', 2.4), ('release', .6)]
        report['plan'] = plan
        report['boundary'] = ({'kind': 'independent dynamic test platen with bounded external wrench',
            'mass_kg': .1, 'desired_travel_m': .010, 'desired_maximum_speed_m_s': .01875,
            'position_gain_n_m': 1000., 'velocity_gain_ns_m': 20.,
            'orientation_gain_nm_rad': .2, 'angular_velocity_gain_nms_rad': .01,
            'force_limits_n': [2., 2., 10.], 'torque_limits_nm': [.1, .1, .1],
            'gravity_compensation_n': [0., 0., .981],
            'foam_and_pad_dynamic': True, 'post_reset_platen_state_writes': 0} if platen is not None else
            {'kind': 'bounded horizontal force on independent real scouring layer',
             'maximum_axis_force_n': .4, 'vertical_force_n': 0., 'torque_nm': [0., 0., 0.]})
        if args.case == 'support':
            report['boundary'] = {'kind': 'passive gravity support diagnosis',
                'external_force_n': [0., 0., 0.], 'external_torque_nm': [0., 0., 0.],
                'material_bond_disabled_ablation': bool(args.disable_material_bond)}
        previous_target = start[:2].copy()
        with (output/'native_frames.jsonl').open('x') as stream:
            stream.write(json.dumps(rows[0], allow_nan=False)+'\n')
            for phase, duration in plan:
                target = {'forward': center+[.08, -.019], 'side_step': center+[.08, .019],
                          'backward': center+[-.08, .019]}.get(phase, previous_target)
                for tick in range(round(duration/args.dt)):
                    u = (tick+1)*args.dt/duration
                    force = np.zeros(3)
                    if platen is not None:
                        travel = (.010*smooth(u) if phase == 'press' else .010*(1.-smooth(u))
                                  if phase == 'withdraw' else .010 if phase == 'loaded_hold' else 0.)
                        from scipy.spatial.transform import Rotation
                        pose = native_array(platen.get_transforms())[0]
                        speed = native_array(platen.get_velocities())[0]
                        platen_target = np.asarray(platen_center)-[0., 0., travel]
                        platen_force = np.clip(1000.*(platen_target-pose[:3])-20.*speed[:3]+[0., 0., .981],
                            [-2., -2., -10.], [2., 2., 10.])
                        platen_torque = np.clip(-.2*Rotation.from_quat(pose[3:]).as_rotvec()-.01*speed[3:], -.1, .1)
                        platen.apply_forces_and_torques_at_position(tensor(platen_force[None].astype(np.float32)),
                            tensor(platen_torque[None].astype(np.float32)), None, one, True)
                    elif phase not in ('settle', 'release'):
                        pose = native_array(pad.get_transforms())[0]
                        speed = native_array(pad.get_velocities())[0]
                        desired = previous_target+(target-previous_target)*smooth(u)
                        desired_velocity = (target-previous_target)*6.*u*(1.-u)/duration
                        force[:2] = np.clip(30.*(desired-pose[:2])+1.4*(desired_velocity-speed[:2]), -.4, .4)
                        pad.apply_forces_and_torques_at_position(tensor(force[None, :].astype(np.float32)),
                            tensor(np.zeros((1, 3), np.float32)), None, one, True)
                    world.step(render=False)
                    row = sample(phase, force)
                    stream.write(json.dumps(row, allow_nan=False)+'\n')
                    if (row['geometry']['inverted_tetrahedron_count']
                            or (not args.disable_material_bond
                                and row['geometry']['maximum_material_bond_error_m'] > .004)):
                        stream.flush()
                        raise RuntimeError('Native foam inverted or detached; failed diagnostic stopped with raw evidence')
                previous_target = np.asarray(target)
                sample_visual(phase)
                if media is not None and phase in ('settle', 'loaded_hold', 'recover'):
                    media_rows.append(media.sample(phase, world, volume, pad,
                        extra_rigid_views={'platen': platen} if platen is not None else None))
                print(json.dumps({'phase': phase, 'volume_ratio': row['geometry']['total_volume_ratio'],
                    'coverage': row['coverage']['patch_coverage']}), flush=True)
        states, nodes, velocities, times = map(np.asarray, (states, nodes, velocities, times))
        np.savez_compressed(output/'native_states.npz', time_s=times, scouring_q=states[:, :7],
            scouring_qd=states[:, 7:], foam_q=nodes, foam_qd=velocities,
            platen_state=np.asarray(platen_states),
            rest_points_m=np.asarray(foam['rest_points_m']), tetrahedra=np.asarray(foam['tetrahedra']),
            native_step=np.arange(initial_step, initial_step+len(times)))
        tail = times >= times[-1]-.4-1.e-10
        bottom = states[:, 2]
        height = np.asarray([r['geometry']['central_top_height_in_pad_frame_m'] for r in rows])
        whole_height = nodes[:, :, 2].max(axis=1)-bottom
        reference = np.array([r['phase'] == 'settle' and r['time_s'] >= plan[0][1]-.2 for r in rows])
        baseline_height = float(np.median(height[reference]))
        terminal_height = float(np.median(height[tail]))
        force_reference = float(np.mean([r['normal_table_force_n'] for r, take in zip(rows, reference) if take]))
        force_final = float(np.mean([r['normal_table_force_n'] for r, take in zip(rows, tail) if take]))
        checks = {'full_native_stream': int(world.current_time_step_index)-initial_step == len(times)-1,
            'finite_states': bool(np.isfinite(states).all() and np.isfinite(nodes).all() and np.isfinite(velocities).all()),
            'no_inverted_tetrahedra': all(r['geometry']['inverted_tetrahedron_count'] == 0 for r in rows),
            'no_gross_foam_distortion': all(r['geometry']['minimum_tetrahedron_volume_ratio'] >= .1
                and r['geometry']['minimum_edge_length_ratio'] >= .5
                and r['geometry']['maximum_edge_length_ratio'] <= 1.5 for r in rows),
            'no_foam_table_penetration': float(nodes[:, :, 2].min()) >= .900-.004,
            'supported_final': float(np.max(np.abs(bottom[tail]-.900))) <= .004,
            'released_pad_quiet': float(np.max(np.linalg.norm(states[tail, 7:10], axis=1))) < .025,
            'released_pad_angularly_quiet': float(np.max(np.linalg.norm(states[tail, 10:13], axis=1))) < .25,
            'no_scouring_table_penetration': min(r['geometry']['scouring_lowest_z_m'] for r in rows) >= .900-.004,
            'native_contact_separation_valid': all(c['min_separation_m'] >= -.004 for r in rows for c in r['contacts']),
            'fresh_released_support': sum(r['normal_table_force_n'] > 0 for r, take in zip(rows, tail) if take)
                >= .95*int(tail.sum()),
            'released_foam_quiet': rows[-1]['geometry']['stability']['released_quiet_native_s'] >= .4-1.e-10,
            'gravity_transmitted_to_table': abs(force_final-mass*9.81)/(mass*9.81) <= .15,
            'foam_rebounds_after_release': abs(terminal_height-baseline_height) <= .001}
        if not args.disable_material_bond:
            checks['no_material_bond_separation'] = max(r['geometry']['maximum_material_bond_error_m'] for r in rows) <= .001
        measures = {'mass_kg': mass, 'expected_weight_n': mass*9.81,
            'settled_table_force_n': force_reference, 'released_table_force_n': force_final,
            'settled_height_m': baseline_height, 'released_height_m': terminal_height,
            'whole_foam_global_height_min_max_m': [float(whole_height.min()), float(whole_height.max())],
            'loaded_contact_steps': loaded_count,
            'maximum_material_bond_error_m': max(r['geometry']['maximum_material_bond_error_m'] for r in rows),
            'maximum_native_node_speed_m_s': float(np.linalg.norm(velocities, axis=2).max())}
        if platen is not None:
            loaded = np.array([r['phase'] == 'loaded_hold' for r in rows])
            compression = baseline_height-float(np.median(height[loaded]))
            force_loaded = float(np.mean([r['normal_table_force_n'] for r, take in zip(rows, loaded) if take]))
            checks.update(real_foam_compression=.001 <= compression <= .012,
                pressure_transmitted_to_table=force_loaded > force_reference+.15)
            measures.update(compression_m=compression, pressed_table_force_n=force_loaded)
        elif args.case == 'wiping':
            checks.update(first_patch_cleaned=rows[-1]['coverage']['patch_coverage']['soil_1'] >= .60,
                untouched_third_patch_not_cleaned=rows[-1]['coverage']['patch_coverage']['soil_3'] == 0.)
        report.update(checks={k: bool(v) for k, v in checks.items()}, measurements=measures,
            passed=all(checks.values()), native_steps=len(times)-1, final_coverage=rows[-1]['coverage'],
            evidence_sha256={name: digest(output/name) for name in ('initial.usda', 'native_frames.jsonl', 'native_states.npz')})
        report['visual_readback'] = {'requested': args.visual_readback,
            'complete': bool(visual is not None and not visual_errors and len(visual_rows) == len(plan)+1),
            'samples': visual_rows, 'errors': visual_errors}
        report['native_media'] = {'requested': bool(args.media_output), 'samples': media_rows}
        code = 0 if report['passed'] else 1
    except BaseException:
        report['exception'] = traceback.format_exc()
        traceback.print_exc()
    finally:
        flush_raw_chunk()
        if len(times) and not (output/'native_states.npz').exists():
            state_array = np.asarray(states)
            np.savez_compressed(output/'native_states.npz', time_s=np.asarray(times),
                scouring_q=state_array[:, :7], scouring_qd=state_array[:, 7:],
                foam_q=np.asarray(nodes), foam_qd=np.asarray(velocities),
                platen_state=np.asarray(platen_states),
                rest_points_m=np.asarray(foam['rest_points_m']), tetrahedra=np.asarray(foam['tetrahedra']),
                native_step=np.asarray([row['native_step'] for row in rows]))
            report['partial_native_state_archive'] = {'frames': len(times),
                'sha256': digest(output/'native_states.npz'), 'complete_planned_trajectory': False}
        report['process_exit_code'] = code
        dump(output/'qualification.json', report)
        print(json.dumps({'passed': report['passed'], 'output': str(output)}), flush=True)
        if app is not None:
            app.close(exit_code=code)
    return code


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--asset-manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--case', choices=('compression', 'wiping', 'support'), required=True)
    parser.add_argument('--dt', type=float, choices=(.001, .0005, .00025), default=.0005)
    parser.add_argument('--solver', choices=('TGS', 'PGS'), default='TGS',
        help='Explicit diagnostic solver selection; does not edit the frozen release')
    parser.add_argument('--external-forces-every-iteration', action='store_true',
        help='Explicit TGS numerical ablation; preserves mass, material and controller parameters')
    parser.add_argument('--rigid-velocity-iterations', type=int, choices=(1, 8),
        help='Explicit rigid scouring-layer velocity-iteration diagnostic; omission preserves USD')
    parser.add_argument('--scene-velocity-iterations', type=int, choices=(0,),
        help='Explicit numerical ablation: clamp scene min/max velocity iterations to zero; omission preserves source settings')
    parser.add_argument('--visual-readback', action='store_true')
    parser.add_argument('--media-output', type=Path)
    parser.add_argument('--disable-material-bond', action='store_true',
        help='Passive support ablation only: deactivate material bond and its filter before reset')
    raise SystemExit(run(parser.parse_args()))
