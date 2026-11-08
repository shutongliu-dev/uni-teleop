"""Native free-flight impulse test of the unchanged bonded sponge.

Lift both material layers before reset, then fall .05s, apply .4N world-X at
the actual pad COM for .05s, and release .1s. No render calls or state setters
occur after reset. This is a separate instrument, not robot qualification.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import traceback

import numpy as np
from scipy.spatial.transform import Rotation
from bench.isaacsim.release import digest, dump
from bench.paths import load_json, OUTPUT_ROOT
from scripts.maintenance.freeze_project_sources import verify_if_snapshot
from scripts.validation import sponge_freeflight_metrics as metrics
from scripts.validation.sponge_support_loadcell import (
    load_asset, native, json_value, contact_row, static_collision_bounds,
    bbox_gap, native_offsets, collision_offsets)
from scenes.workrooms.kitchen.sponge_dynamics import VolumeGeometry
from scripts.validation.sponge_scene_options import apply_velocity_iteration_clamp, velocity_iteration_receipt

PROJECT = Path(__file__).resolve().parents[2]
SCENE = '/World/PhysicsScene'
TABLE = '/World/FreeflightTableBoundary'
MODULES = ('scripts.validation.sponge_freeflight', 'scripts.validation.sponge_freeflight_metrics',
           'scripts.validation.sponge_support_loadcell', 'scenes.workrooms.kitchen.sponge_dynamics',
           'scripts.validation.sponge_scene_options')


def author(stage, manifest, usd, meta):
    """CPU-only assembly, called strictly before native reset."""
    from pxr import Gf, Sdf, UsdGeom, UsdPhysics
    UsdGeom.SetStageMetersPerUnit(stage, 1.); UsdGeom.SetStageUpAxis(stage, 'Z')
    root = stage.DefinePrim(manifest['root_prim_path'], 'Xform')
    root.GetReferences().AddReference(str(usd), manifest['root_prim_path'])
    foam = meta['volume_bodies']['foam']; pad_path = meta['actors']['sponge']
    for path in (pad_path, foam['prim_path']):
        attr = stage.GetPrimAtPath(path).GetAttribute('xformOp:translate')
        if not attr or attr.Get() is None:
            raise ValueError('Both material layers need an authored translation')
        attr.Set(Gf.Vec3d(*(np.asarray(attr.Get(), float)+[0., 0., metrics.CONFIG['lift_m']])))
    pad = stage.GetPrimAtPath(pad_path)
    pad.AddAppliedSchema('PhysxContactReportAPI')
    pad.CreateAttribute('physxContactReport:threshold', Sdf.ValueTypeNames.Float, custom=False).Set(0.)
    # Explicit stationary exclusion boundary; it must never contact the load.
    # No plate, spring, calibration mass, or loadcell authoring is imported.
    table = UsdGeom.Cube.Define(stage, TABLE); table.CreateSizeAttr(1.)
    table.AddTranslateOp().Set(Gf.Vec3d(.45, -.95, .8775))
    table.AddScaleOp().Set(Gf.Vec3f(2.8, .8, .045))
    UsdPhysics.CollisionAPI.Apply(table.GetPrim()); table.GetPrim().AddAppliedSchema('PhysxCollisionAPI')
    for name, value in [('contactOffset', .002), ('restOffset', 0.)]:
        table.GetPrim().CreateAttribute('physxCollision:'+name, Sdf.ValueTypeNames.Float, custom=False).Set(value)
    physics = UsdPhysics.Scene.Define(stage, SCENE)
    physics.CreateGravityDirectionAttr(Gf.Vec3f(0., 0., -1.)); physics.CreateGravityMagnitudeAttr(9.81)
    return pad_path



def translate_instrument_frame(stage, manifest, meta, *, center_origin=False):
    """Translate the whole instrument once before reset; default authors nothing.

    Metadata remains in its original source frame. Native archives and bounds
    use the explicitly translated world frame; no tensor state is rewritten.
    """
    receipt = {'center_origin_requested': bool(center_origin), 'applied_translation_m': [0., 0., 0.],
        'mapping': 'native_world_xyz = original_instrument_world_xyz + applied_translation_m',
        'raw_world_frame': 'original_instrument_world', 'metadata_frame': 'unchanged_source_asset_frame',
        'scope': 'Pre-reset common translation only; gravity/forces are vectors and remain unchanged.'}
    if not center_origin:
        return receipt
    from pxr import Gf, Usd, UsdGeom, UsdPhysics
    root = stage.GetPrimAtPath(manifest['root_prim_path']); xform = UsdGeom.Xformable(root)
    if not xform or xform.GetOrderedXformOps() or xform.GetResetXformStack():
        raise ValueError('Origin diagnostic requires the original untransformed task root')
    cache = UsdGeom.XformCache()
    if not np.allclose(np.asarray(cache.GetLocalToWorldTransform(root), float), np.eye(4), atol=1e-12, rtol=0):
        raise ValueError('Origin diagnostic requires identity task ancestors')
    for prim in Usd.PrimRange(root):
        child = UsdGeom.Xformable(prim)
        if child and child.GetResetXformStack():
            raise ValueError('A reset transform stack would escape the common frame translation')
    colliders = [p for p in stage.Traverse() if p.HasAPI(UsdPhysics.CollisionAPI)]
    if any(str(p.GetPath()) != TABLE and not p.GetPath().HasPrefix(root.GetPath()) for p in colliders):
        raise ValueError('All physical geometry must belong to the translated instrument')
    foam = meta['volume_bodies']['foam']; pad = next(x for x in meta['objects'] if x['id'] == 'sponge')
    def world(path, points):
        matrix = np.asarray(cache.GetLocalToWorldTransform(stage.GetPrimAtPath(path)), dtype=float)
        return np.asarray(points, dtype=float) @ matrix[:3, :3] + matrix[3, :3]
    points = np.concatenate((world(foam['simulation_mesh_path'], foam['rest_points_m']),
                             world(meta['actors']['sponge'], pad['collision_support_points_m'])))
    center = .5*(points.min(axis=0)+points.max(axis=0)); shift = -center
    table_attr = stage.GetPrimAtPath(TABLE).GetAttribute('xformOp:translate')
    if not table_attr or table_attr.Get() is None or not np.isfinite(shift).all():
        raise ValueError('Finite table and composite bounds required')
    xform.AddTranslateOp(opSuffix='freeflightOrigin').Set(Gf.Vec3d(*shift))
    table_attr.Set(Gf.Vec3d(*(np.asarray(table_attr.Get(), dtype=float)+shift)))
    receipt.update(applied_translation_m=shift.tolist(), original_lifted_composite_bounds_center_m=center.tolist(),
        raw_world_frame='translated_instrument_world', translated_root=str(root.GetPath()), translated_table=TABLE,
        transform_kind='Pure parent translation; no rotation, scale, child geometry or body-state edit.')
    return receipt


def run(args):
    source = verify_if_snapshot(PROJECT, required_modules=MODULES, entrypoint_file=Path(__file__))
    if source is None or Path.cwd().resolve() != PROJECT:
        raise ValueError('Use the complete frozen source root')
    source_file = PROJECT/'source_snapshot_manifest.json'
    inventory = load_json(source_file)['files_sha256']
    if source_file.stat().st_mode & 0o222 or any((PROJECT/p).stat().st_mode & 0o222 for p in inventory):
        raise ValueError('Snapshot must be sealed read-only before launch')
    output = args.output.resolve()
    if not output.is_relative_to(OUTPUT_ROOT):
        raise ValueError('Evidence belongs in shared datasets')
    output.mkdir(parents=True, exist_ok=False)
    report = {'schema': 'sponge_freeflight.v1', 'passed': False, 'source': source,
        'source_snapshot_sha256': digest(source_file), 'configuration': metrics.CONFIG,
        'solver_requested': args.solver, 'scene_velocity_iterations_requested': args.scene_velocity_iterations, 'asset_manifest': str(args.asset_manifest.resolve()),
        'asset_manifest_sha256': digest(args.asset_manifest), 'gpu_uuid': os.environ.get('CUDA_VISIBLE_DEVICES'),
        'post_reset_pose_velocity_node_writes': 0, 'render_calls': 0, 'model_calls': 0,
        'force_application': 'Explicit world-space native pad COM, world force, zero torque; per-step external wrench only.',
        'scope': 'Contact-free declared-mass inertial consistency, not contact-force calibration, native foam inverse-mass identification, or robot task qualification.'}
    keys = ('time_s', 'native_step', 'pad_q', 'pad_qd', 'foam_q', 'foam_qd',
            'force_world_n', 'force_position_world_m', 'torque_world_nm')
    raw = {k: [] for k in keys}; rows = []; chunks = []; archived = 0; app = None; code = 1
    foam = None

    def flush():
        nonlocal archived
        stop = len(raw['time_s'])
        if stop == archived:
            return
        directory = output/'native_chunks'; directory.mkdir(exist_ok=True)
        path = directory/f'frames_{len(chunks):06d}.npz'
        np.savez_compressed(path, **{k: np.asarray(v[archived:stop]) for k, v in raw.items()})
        chunks.append({'path': str(path), 'first_frame': archived, 'last_frame': stop-1, 'sha256': digest(path)})
        archived = stop
        dump(output/'native_chunks_manifest.json', {'frames': stop, 'chunks': chunks, 'trajectory_complete': stop == 401})

    try:
        manifest, usd, meta = load_asset(args.asset_manifest)
        report['asset_file_sha256'] = {str(p): digest(p) for p in (args.asset_manifest, usd, usd.with_name('metadata.json'))}
        foam = meta['volume_bodies']['foam']; pad_spec = next(x for x in meta['objects'] if x['id'] == 'sponge')
        metrics.lumped_masses(foam)  # Reject invalid rest topology before GPU initialization.
        from isaacsim import SimulationApp
        app = SimulationApp({'headless': True, 'multi_gpu': False, 'active_cuda_gpus': [0],
            'max_gpu_count': 1, 'physics_gpu': 0, 'fast_shutdown': True,
            'extra_args': ['--/rtx/verifyDriverVersion/enabled=false']
            if os.environ.get('ISAAC_ALLOW_UNVALIDATED_DRIVER') == '1' else []},
            experience=str(PROJECT/'bench/apps/workroom.headless.kit'))
        import omni.usd
        import torch
        from isaacsim.core.api import World
        from pxr import UsdPhysics
        omni.usd.get_context().new_stage(); stage = omni.usd.get_context().get_stage()
        pad_path = author(stage, manifest, usd, meta)
        report['coordinate_frame'] = translate_instrument_frame(stage, manifest, meta, center_origin=args.center_origin)
        world = World(stage_units_in_meters=1., physics_dt=metrics.CONFIG['dt_s'], rendering_dt=1/30,
            backend='torch', device='cuda:0', physics_prim_path=SCENE)
        context = world.get_physics_context()
        context.enable_gpu_dynamics(True); context.set_broadphase_type('GPU'); context.set_solver_type(args.solver)
        obstacles = static_collision_bounds(stage, [pad_path, foam['prim_path']])
        if not obstacles or not any(x['path'] == TABLE for x in obstacles):
            raise ValueError('Complete static collision inventory including table boundary required')
        for row in obstacles:
            bounds = np.asarray(row['bounds_m'])
            if (bounds[1] < bounds[0]).any():
                raise ValueError('Empty static collider bound')
            offset = stage.GetPrimAtPath(row['path']).GetAttribute('physxCollision:contactOffset').Get()
            if offset is None or not np.isfinite(offset) or not 0 <= offset <= .002001:
                raise ValueError('Unbounded automatic contact envelope cannot certify free flight')
            row['composed_contact_offset_m'] = float(offset)
        report['static_collision_bounds'] = obstacles
        report['composed_collision_offsets'] = collision_offsets(stage, [manifest['root_prim_path'], TABLE])
        scene = stage.GetPrimAtPath(SCENE)
        report['scene_velocity_iteration_clamp'] = apply_velocity_iteration_clamp(scene, args.scene_velocity_iterations)
        names = ('physxScene:solverType', 'physxScene:enableGPUDynamics', 'physxScene:timeStepsPerSecond',
                 'physxScene:enableExternalForcesEveryIteration', 'physics:gravityMagnitude', 'physics:gravityDirection',
                 'physxScene:minVelocityIterationCount', 'physxScene:maxVelocityIterationCount')
        report['scene_properties'] = {n: json_value(scene.GetAttribute(n).Get()) for n in names}
        stage.Export(str(output/'initial.usda')); world.reset(); sim = world.physics_sim_view
        report['scene_velocity_iteration_clamp_after_reset'] = velocity_iteration_receipt(scene, args.scene_velocity_iterations)
        report['tensor_backend'] = {'device_ordinal': int(sim.device_ordinal), 'device': str(sim.device), 'frontend': 'torch'}
        if int(sim.device_ordinal) < 0:
            raise RuntimeError('GPU tensor simulation required; no CPU fallback')
        pad = sim.create_rigid_body_view(pad_path)
        volume = sim.create_volume_deformable_body_view(foam['prim_path'])
        contact = sim.create_rigid_contact_view(pad_path, max_contact_data_count=8192)
        if pad.count != 1 or volume.count != 1:
            raise ValueError('Exactly one pad and one foam required')
        tets = native(volume.get_simulation_element_indices()).reshape(-1, 4)
        if not np.array_equal(tets, foam['tetrahedra']):
            raise ValueError('Native FEM topology differs from the frozen asset')
        mass = float(native(pad.get_masses()).reshape(-1)[0]); com_local = native(pad.get_coms())[0, :3].astype(float)
        report.update(native_pad_mass_kg=mass, native_pad_com_local_m=com_local.tolist(),
            native_pad_inertias=native(pad.get_inertias()).tolist(), native_pad_offsets=native_offsets(pad),
            native_pad_material_properties=native(pad.get_material_properties()).tolist(),
            native_pad_material_order=['static_friction', 'dynamic_friction', 'restitution'],
            native_volume_paths=list(volume.prim_paths), native_simulation_mesh_paths=list(volume.simulation_mesh_prim_paths),
            native_collision_mesh_paths=list(volume.collision_mesh_prim_paths), native_topology_exact=True,
            native_foam_inverse_masses_available=False, foam_contact_impulses_available=False)
        offsets = report['native_pad_offsets']
        if (not all(offsets[k]['available'] for k in ('get_contact_offsets', 'get_rest_offsets'))
                or not np.allclose(offsets['get_contact_offsets']['values_m'], .0002, atol=1e-8, rtol=0)
                or not np.allclose(offsets['get_rest_offsets']['values_m'], 0., atol=1e-8, rtol=0)):
            raise ValueError('Native pad collision offset readback differs from the asset')
        if not np.isclose(mass, pad_spec['mass_kg'], rtol=2e-6, atol=1e-10):
            raise ValueError('Native pad mass changed')
        np.testing.assert_allclose(com_local, pad_spec['center_of_mass_m'], atol=1e-7, rtol=0)
        mv = sim.create_deformable_material_view(foam['material']['material_path'])
        if mv.count != 1:
            raise ValueError('One native foam material required')
        report['native_material'] = {n: float(native(getattr(mv, g)()).reshape(-1)[0]) for n, g in
            [('youngs_modulus_pa', 'get_youngs_modulus'), ('poissons_ratio', 'get_poissons_ratio'), ('dynamic_friction', 'get_dynamic_friction')]}
        if any(not np.isclose(v, foam['material'][n], rtol=1e-5, atol=1e-7) for n, v in report['native_material'].items()):
            raise ValueError('Native foam material changed')
        body = stage.GetPrimAtPath(foam['prim_path']); padprim = stage.GetPrimAtPath(pad_path)
        report['deformable_properties'] = {n: json_value(body.GetAttribute(n).Get()) for n in
            ('omniphysics:mass', 'omniphysics:kinematicEnabled', 'physxDeformableBody:linearDamping',
             'physxDeformableBody:settlingDamping', 'physxDeformableBody:settlingThreshold', 'physxDeformableBody:sleepThreshold')}
        if not np.isclose(report['deformable_properties']['omniphysics:mass'], foam['total_mass_kg'], rtol=1e-6):
            raise ValueError('Composed declared foam mass changed')
        foam = {**foam, 'damping_receipt': {
            'linear_damping_per_s': float(body.GetAttribute('physxDeformableBody:linearDamping').Get()),
            'settling_damping_per_s': float(body.GetAttribute('physxDeformableBody:settlingDamping').Get()),
            'settling_threshold_m_s': float(body.GetAttribute('physxDeformableBody:settlingThreshold').Get()),
            'pad_linear_damping_per_s': float(padprim.GetAttribute('physxRigidBody:linearDamping').Get()),
            'values_are': 'Composed native-runtime USD values, not independent damping getters'}}
        if padprim.GetAttribute('physics:kinematicEnabled').Get() is not False or body.GetAttribute('omniphysics:kinematicEnabled').Get() is not False:
            raise ValueError('Both layers must be dynamic')
        attachment = stage.GetPrimAtPath(foam['attachment']['prim_path'])
        report['attachment'] = {key: [str(p) for p in attachment.GetRelationship('omniphysics:'+key).GetTargets()] for key in ('src0', 'src1')}
        report['attachment']['enabled'] = attachment.GetAttribute('omniphysics:attachmentEnabled').Get()
        if (not attachment.IsActive() or report['attachment']['enabled'] is not True
                or report['attachment']['src0'] != [foam['simulation_mesh_path']]
                or report['attachment']['src1'] != [pad_path]):
            raise ValueError('Original two-way material attachment required')
        if not stage.GetPrimAtPath(foam['interface_collision_filter']['prim_path']).IsActive():
            raise ValueError('Original material-interface collision filter required')
        if [str(p.GetPath()) for p in stage.Traverse() if p.GetTypeName().endswith('Attachment')] != [str(attachment.GetPath())]:
            raise ValueError('Unexpected additional attachment in free flight')
        if any(p.IsA(UsdPhysics.Joint) for p in stage.Traverse()):
            raise ValueError('Free-flight fixture must not contain a joint or spring')
        gravity = np.array(report['scene_properties']['physics:gravityDirection'])*report['scene_properties']['physics:gravityMagnitude']
        dump(output/'native_initialization.json', report)
        geometry = VolumeGeometry(foam['rest_points_m'], foam['tetrahedra'])
        initial_time = float(world.current_time)
        one = torch.tensor([0], device='cuda:0', dtype=torch.int32)
        tensor = lambda v: torch.as_tensor(v, device='cuda:0', dtype=torch.float32)

        def sample(index, force, position):
            p = native(pad.get_transforms())[0]; v = native(pad.get_velocities())[0]
            q = native(volume.get_simulation_nodal_positions())[0]; qd = native(volume.get_simulation_nodal_velocities())[0]
            values = (float(world.current_time)-initial_time, int(world.current_time_step_index), p, v, q, qd,
                      force, position, np.zeros(3, dtype=np.float32))
            for key, value in zip(keys, values): raw[key].append(value)
            row = {'relative_step': index, 'time_s': values[0], 'native_step': values[1],
                   'phase': 'fall' if index <= 100 else 'pulse' if index <= 200 else 'release'}
            rows.append(row)  # Preserve the raw frame if subsequent readback/geometry fails.
            row['contact'] = contact_row(contact, metrics.CONFIG['dt_s'], baseline=index == 0)
            row['foam'] = geometry.sample(q, qd)
            rotation = Rotation.from_quat(p[3:]); anchors = rotation.apply(foam['attachment']['rigid_local_positions_m'])+p[:3]
            row['bond_error_m'] = float(np.linalg.norm(q[np.asarray(foam['attachment']['node_indices'])]-anchors, axis=1).max())
            points = np.concatenate((q, rotation.apply(pad_spec['collision_support_points_m'])+p[:3]))
            lo, hi = points.min(0), points.max(0)
            row['composite_bounds_m'] = [lo.tolist(), hi.tolist()]
            row['minimum_static_gap_m'] = min(bbox_gap(lo, hi, np.asarray(x['bounds_m'])) for x in obstacles)
            if len(raw['time_s'])-archived >= 100: flush()
            return row

        with (output/'native_frames.jsonl').open('x') as stream:
            row = sample(0, np.zeros(3, np.float32), np.zeros(3, np.float32))
            stream.write(json.dumps(row, allow_nan=False)+'\n')
            for index in range(1, 401):
                force = metrics.force_for_step(index).astype(np.float32)
                pose = native(pad.get_transforms())[0]
                position = (Rotation.from_quat(pose[3:]).apply(com_local)+pose[:3]).astype(np.float32)
                if force.any():
                    result = pad.apply_forces_and_torques_at_position(tensor(force[None]), tensor(np.zeros((1, 3))),
                        tensor(position[None]), one, True)
                    if result is False: raise RuntimeError('External wrench application failed')
                world.step(render=False); row = sample(index, force, position)
                stream.write(json.dumps(row, allow_nan=False)+'\n')
        checks, measured, derived = metrics.evaluate(raw, foam, mass, com_local, gravity)
        checks.update(no_contact=all(r['contact']['point_count'] == 0 and r['contact']['finite']
            and np.linalg.norm(r['contact']['net_contact_force_n']) <= 1e-8 for r in rows),
            all_static_colliders_separated=all(r['minimum_static_gap_m'] > .01 for r in rows),
            material_bond_intact=all(r['bond_error_m'] <= .001 for r in rows),
            full_tet_and_edge_geometry=all(r['foam']['inverted_tetrahedron_count'] == 0
                and r['foam']['minimum_tetrahedron_volume_ratio'] >= .1
                and r['foam']['minimum_edge_length_ratio'] >= .5
                and r['foam']['maximum_edge_length_ratio'] <= 1.5 for r in rows),
            requested_solver=report['scene_properties']['physxScene:solverType'] == args.solver)
        checks['asset_inputs_unchanged'] = all(digest(p) == h for p, h in report['asset_file_sha256'].items())
        np.savez_compressed(output/'derived_momentum.npz', **derived)
        report.update(checks=checks, measurements=measured, passed=all(checks.values()), native_steps=400)
        code = 0 if report['passed'] else 1
    except BaseException:
        report['exception'] = traceback.format_exc(); traceback.print_exc()
    finally:
        try:
            flush()
            arrays = {k: np.asarray(v) for k, v in raw.items()}
            if foam is not None: arrays.update(rest_points_m=np.asarray(foam['rest_points_m']), tetrahedra=np.asarray(foam['tetrahedra']))
            np.savez_compressed(output/'native_states.npz', **arrays)
            with (output/'native_frames.jsonl').open('w') as stream:
                for row in rows: stream.write(json.dumps(row, allow_nan=False)+'\n')
            report['raw_frames'] = len(raw['time_s'])
            report['evidence_sha256'] = {p.name: digest(p) for p in output.iterdir() if p.is_file() and p.name != 'qualification.json'}
        except BaseException:
            report['archive_exception'] = traceback.format_exc(); code = 1; report['passed'] = False
        finally:
            report['process_exit_code'] = code
            dump(output/'qualification.json', report)
            if app is not None: app.close(exit_code=code)
    return code


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--asset-manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--solver', choices=('TGS', 'PGS'), required=True)
    parser.add_argument('--center-origin', action='store_true',
        help='Translate the whole task root and table together before reset so the lifted sponge bounds center is the world origin')
    parser.add_argument('--scene-velocity-iterations', type=int, choices=(0,),
        help='Explicit numerical ablation: clamp scene min/max velocity iterations to zero; omission preserves source settings')
    raise SystemExit(run(parser.parse_args()))
