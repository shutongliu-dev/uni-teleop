"""Original compound spoon force/response diagnostic, with no collider replacement.

Frozen original task USD, target spoon translated +2 m in Z before reset; same
TGS/gravity/dt and original mass/COM/inertia/material/damping. Fall .05 s,
world-COM .4 N X pulse .05 s, release .1 s. Other task bodies stay present and
are recorded for conservative per-frame no-contact bounds. No controllers,
post-reset state writes, mass compensation or contact-force normalization.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import traceback
import numpy as np
from scipy.spatial.transform import Rotation
from bench.isaacsim.release import checked, digest, dump
from bench.paths import load_json, OUTPUT_ROOT
from scripts.maintenance.freeze_project_sources import verify_if_snapshot
from scripts.validation.sponge_freeflight_metrics import force_for_step
from scripts.validation.sponge_support_loadcell import native, json_value, contact_row, native_offsets, bbox_gap

PROJECT = Path(__file__).resolve().parents[2]
SCENE = '/World/PhysicsScene'
MODULES = ('scripts.validation.spoon_freeflight', 'scripts.validation.sponge_freeflight_metrics',
           'scripts.validation.sponge_support_loadcell')
SHIFT = np.array([0., 0., 2.])
CONFIG = {'dt_s': .0005, 'fall_steps': 100, 'pulse_steps': 100, 'release_steps': 200,
          'lift_m': 2., 'force_world_n': [.4, 0., 0.], 'momentum_relative_tolerance': .15,
          'release_relative_tolerance': .05, 'minimum_obstacle_gap_m': .1}


def evaluate(raw, mass, com_local, gravity_z, declared_mass):
    """Use actual native rigid mass, COM and velocities; no synthetic FEM data."""
    shapes = {'time_s': (401,), 'native_step': (401,), 'body_q': (401, 7), 'body_qd': (401, 6),
              'force_world_n': (401, 3), 'force_position_world_m': (401, 3), 'torque_world_nm': (401, 3)}
    data = {k: np.asarray(raw[k], dtype=float) for k in shapes}
    if any(data[k].shape != shape or not np.isfinite(data[k]).all() for k, shape in shapes.items()):
        raise ValueError('Complete finite 401-frame rigid archive required')
    if not np.isclose(mass, declared_mass, rtol=2e-6, atol=1e-10) or not np.isclose(gravity_z, -9.81, atol=1e-5, rtol=0):
        raise ValueError('Original declared spoon mass and gravity required')
    dt = np.diff(data['time_s'])
    if (data['time_s'][0] != 0 or not np.allclose(dt, float(np.float32(CONFIG['dt_s'])), atol=1e-10, rtol=0)
            or not np.array_equal(data['native_step']-data['native_step'][0], np.arange(401))):
        raise ValueError('Missing or irregular native step')
    force = np.asarray([force_for_step(i) for i in range(401)], dtype=np.float32).astype(float)
    if not np.array_equal(force, data['force_world_n']) or np.any(data['torque_world_nm']):
        raise ValueError('Exact world-X force schedule and zero applied torque required')
    rotation = Rotation.from_quat(data['body_q'][:, 3:])
    centers = rotation.apply(np.tile(np.asarray(com_local, dtype=float), (401, 1)))+data['body_q'][:, :3]
    if not np.array_equal(data['force_position_world_m'][1:], centers[:-1].astype(np.float32).astype(float)):
        raise ValueError('Applied native wrench positions must match preceding world COM')
    p = mass*data['body_qd'][:, :3]
    impulse = np.sum(force[101:201]*dt[100:200, None], axis=0)
    dp = p[200]-p[100]
    pulse_error = abs(dp[0]-impulse[0])/impulse[0]
    release_error = abs(p[400, 0]-p[200, 0])/max(abs(p[200, 0]), 1e-15)
    gravity_windows, motion = [], []
    for phase, first, last in [('fall', 0, 100), ('pulse', 100, 200), ('release', 200, 400), ('all', 0, 400)]:
        expected = mass*gravity_z*(data['time_s'][last]-data['time_s'][first]); observed = p[last, 2]-p[first, 2]
        integrated = np.sum(.5*(p[first:last]+p[first+1:last+1])*dt[first:last, None], axis=0)/mass
        displacement = centers[last]-centers[first]
        gravity_windows.append({'phase': phase, 'expected_impulse_ns': float(expected), 'observed_delta_p_ns': float(observed),
            'relative_error': float(abs(observed-expected)/abs(expected)), 'actual_z_displacement_m': float(displacement[2]),
            'q_vs_qd_relative_error': float(abs(displacement[2]-integrated[2])/max(abs(integrated[2]), 1e-15))})
        if phase in ('pulse', 'release'):
            motion.append({'phase': phase, 'actual_x_displacement_m': float(displacement[0]),
                'integrated_x_velocity_m': float(integrated[0]),
                'q_vs_qd_relative_error': float(abs(displacement[0]-integrated[0])/max(abs(integrated[0]), 1e-15))})
    checks = {'finite_complete_native_stream': True, 'same_world_COM_wrench_schedule': True,
        'horizontal_pulse_momentum': bool(pulse_error <= .15), 'release_horizontal_momentum': bool(release_error <= .05),
        'gravity_momentum_all_windows': all(x['relative_error'] <= .15 for x in gravity_windows),
        'actual_gravity_com_motion': all(x['actual_z_displacement_m'] < 0 and x['q_vs_qd_relative_error'] <= .15 for x in gravity_windows),
        'actual_horizontal_com_motion': all(x['actual_x_displacement_m'] > 0 and x['q_vs_qd_relative_error'] <= .15 for x in motion),
        'no_horizontal_momentum_before_pulse': bool(abs(p[100, 0]-p[0, 0]) <= .15*impulse[0]),
        'no_unforced_transverse_momentum': bool(np.max(np.abs(p[:, 1]-p[0, 1])) <= .15*impulse[0]),
        'no_unforced_rotation': bool(np.max(np.linalg.norm(data['body_qd'][:, 3:], axis=1)) <= .001)}
    measurements = {'native_rigid_mass_kg': float(mass), 'declared_mass_kg': float(declared_mass),
        'measured_inertial_mass_kg': float(impulse[0]/(data['body_qd'][200, 0]-data['body_qd'][100, 0])) if data['body_qd'][200, 0] != data['body_qd'][100, 0] else None,
        'measured_delta_v_x_m_s': float(data['body_qd'][200, 0]-data['body_qd'][100, 0]), 'actual_force_pulse_impulse_ns': impulse.tolist(),
        'pulse_delta_p_ns': dp.tolist(), 'pulse_relative_x_error': float(pulse_error), 'release_relative_x_change': float(release_error),
        'gravity_windows': gravity_windows, 'horizontal_motion': motion,
        'damping_correction_applied': False, 'rigid_only_no_fem_mass_approximation': True}
    return checks, measurements, {'time_s': data['time_s'], 'momentum_ns': p, 'com_world_m': centers}


def load_asset(path, actor):
    manifest = load_json(path); usd = path.parent/'task.usda'; meta = path.parent/'metadata.json'
    checked(usd, manifest['usd_sha256']); checked(meta, manifest['metadata_sha256'])
    data = load_json(meta); row = next(x for x in data['objects'] if x['id'] == actor)
    if row.get('category') != 'spoon' or row.get('collision_count') != 220:
        raise ValueError('Original 220-shape spoon required')
    return manifest, usd, data, row


def physical_inventory(stage, root):
    from pxr import Usd
    return {str(p.GetPath()): {'attributes': {a.GetName(): repr(a.Get()) for a in p.GetAttributes()
            if a.GetName() not in ('xformOp:translate', 'physxContactReport:threshold')},
            'relationships': {r.GetName(): [str(x) for x in r.GetTargets()] for r in p.GetRelationships()
                              if r.GetName() != 'physxContactReport:reportPairs' or r.GetTargets()},
            'schemas': [x for x in p.GetAppliedSchemas() if x != 'PhysxContactReportAPI']}
            for p in Usd.PrimRange(stage.GetPrimAtPath(root))}


def author(stage, manifest, usd, metadata, actor):
    from pxr import Gf, Sdf, UsdGeom, UsdPhysics
    UsdGeom.SetStageMetersPerUnit(stage, 1.); UsdGeom.SetStageUpAxis(stage, 'Z')
    root = manifest['root_prim_path']; stage.DefinePrim(root, 'Xform').GetReferences().AddReference(str(usd), root)
    path = metadata['actors'][actor]; prim = stage.GetPrimAtPath(path)
    before = physical_inventory(stage, root)
    translations = {str(p.GetPath()): repr(p.GetAttribute('xformOp:translate').Get()) for p in stage.Traverse() if p.HasAttribute('xformOp:translate')}
    original = np.array(prim.GetAttribute('xformOp:translate').Get(), float)
    prim.GetAttribute('xformOp:translate').Set(Gf.Vec3d(*(original+SHIFT)))
    prim.AddAppliedSchema('PhysxContactReportAPI')
    prim.CreateAttribute('physxContactReport:threshold', Sdf.ValueTypeNames.Float, custom=False).Set(0.)
    if before != physical_inventory(stage, root): raise ValueError('Original composed property graph changed')
    for p, value in translations.items():
        if p != path and repr(stage.GetPrimAtPath(p).GetAttribute('xformOp:translate').Get()) != value:
            raise ValueError('Only target initial translation may change')
    physics = UsdPhysics.Scene.Define(stage, SCENE)
    physics.CreateGravityDirectionAttr(Gf.Vec3f(0., 0., -1.)); physics.CreateGravityMagnitudeAttr(9.81)
    return {'target_path': path, 'original_translation_m': original.tolist(), 'applied_translation_m': SHIFT.tolist(),
            'new_translation_m': (original+SHIFT).tolist(), 'original_composed_property_graph_preserved': True,
            'added_contact_report_instrumentation_only': True, 'other_task_actors_and_statics_preserved': True}


def collision_bounds(stage, actor_paths):
    """Conservative body-local boxes from actual composed collision geometry."""
    from pxr import Gf, UsdGeom, UsdPhysics
    cache = UsdGeom.BBoxCache(0., [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy, UsdGeom.Tokens.guide], useExtentsHint=False)
    xf = UsdGeom.XformCache(); groups = {p: [] for p in actor_paths}; statics = []; paths = {p: [] for p in actor_paths}
    for prim in stage.Traverse():
        if not prim.HasAPI(UsdPhysics.CollisionAPI) or prim.GetAttribute('physics:collisionEnabled').Get() is False: continue
        path = str(prim.GetPath()); owner = next((p for p in actor_paths if path == p or path.startswith(p+'/')), None)
        bounds = cache.ComputeWorldBound(prim).ComputeAlignedRange(); lo, hi = np.array(bounds.GetMin()), np.array(bounds.GetMax())
        if not np.isfinite([lo, hi]).all() or (hi < lo).any(): raise ValueError('Invalid collision bounds: '+path)
        if owner:
            inverse = xf.GetLocalToWorldTransform(stage.GetPrimAtPath(owner)).GetInverse()
            corners = [[x, y, z] for x in [lo[0], hi[0]] for y in [lo[1], hi[1]] for z in [lo[2], hi[2]]]
            groups[owner].extend(np.asarray(inverse.Transform(Gf.Vec3d(*p))) for p in corners); paths[owner].append(path)
        else: statics.append({'path': path, 'bounds_m': [lo.tolist(), hi.tolist()]})
    local = {}
    for path, points in groups.items():
        points = np.asarray(points); lo, hi = points.min(0), points.max(0)
        local[path] = np.asarray([[x, y, z] for x in [lo[0], hi[0]] for y in [lo[1], hi[1]] for z in [lo[2], hi[2]]])
    return local, statics, paths


def run(args):
    source = verify_if_snapshot(PROJECT, required_modules=MODULES, entrypoint_file=Path(__file__))
    if source is None or Path.cwd().resolve() != PROJECT: raise ValueError('Use complete frozen source root')
    sf = PROJECT/'source_snapshot_manifest.json'; inventory = load_json(sf)['files_sha256']
    if sf.stat().st_mode & 0o222 or any((PROJECT/p).stat().st_mode & 0o222 for p in inventory): raise ValueError('Seal source before launch')
    manifest, usd, metadata, spec = load_asset(args.asset_manifest, args.actor)
    output = args.output.resolve()
    if not output.is_relative_to(OUTPUT_ROOT): raise ValueError('Evidence belongs in shared datasets')
    output.mkdir(parents=True, exist_ok=False)
    report = {'schema': 'original_spoon_freeflight.v1', 'passed': False, 'source': source,
        'source_snapshot_sha256': digest(sf), 'asset_manifest': str(args.asset_manifest.resolve()),
        'asset_manifest_sha256': digest(args.asset_manifest), 'asset_usd_sha256': digest(usd), 'actor': args.actor,
        'solver_requested': 'TGS', 'gpu_uuid': os.environ.get('CUDA_VISIBLE_DEVICES'), 'configuration': CONFIG,
        'post_reset_pose_velocity_writes': 0, 'render_calls': 0, 'model_calls': 0, 'controller_changes': False,
        'force_application': 'Native apply_forces_and_torques_at_position, world force at preceding actual world COM, zero torque, is_global=True, then World.step(render=False).',
        'scope': 'Diagnostic of original compound spoon effective inertial response, not contact support acceptance or robot success.',
        'mass_correction_or_contact_normalization': False,
        'declared_mass_properties': {k: spec[k] for k in ('mass_kg', 'center_of_mass_m', 'diagonal_inertia_kg_m2', 'principal_axes_xyzw')}}
    keys = ('time_s', 'native_step', 'body_q', 'body_qd', 'force_world_n', 'force_position_world_m', 'torque_world_nm', 'all_body_q', 'all_body_qd')
    raw = {k: [] for k in keys}; rows = []; chunks = []; saved = 0; app = None; code = 1
    def flush():
        nonlocal saved
        stop = len(raw['time_s'])
        if stop == saved: return
        directory = output/'native_chunks'; directory.mkdir(exist_ok=True); path = directory/f'frames_{len(chunks):06d}.npz'
        np.savez_compressed(path, **{k: np.asarray(v[saved:stop]) for k, v in raw.items()})
        chunks.append({'path': str(path), 'first_frame': saved, 'last_frame': stop-1, 'sha256': digest(path)}); saved = stop
        dump(output/'native_chunks_manifest.json', {'frames': stop, 'chunks': chunks, 'trajectory_complete': stop == 401})
    try:
        from isaacsim import SimulationApp
        app = SimulationApp({'headless': True, 'multi_gpu': False, 'active_cuda_gpus': [0], 'max_gpu_count': 1,
            'physics_gpu': 0, 'fast_shutdown': True, 'extra_args': ['--/rtx/verifyDriverVersion/enabled=false']
            if os.environ.get('ISAAC_ALLOW_UNVALIDATED_DRIVER') == '1' else []}, experience=str(PROJECT/'bench/apps/workroom.headless.kit'))
        import omni.usd
        import torch
        from isaacsim.core.api import World
        omni.usd.get_context().new_stage(); stage = omni.usd.get_context().get_stage()
        report['pre_reset_overrides'] = author(stage, manifest, usd, metadata, args.actor); path = metadata['actors'][args.actor]
        world = World(stage_units_in_meters=1., physics_dt=CONFIG['dt_s'], rendering_dt=1/30,
                      backend='torch', device='cuda:0', physics_prim_path=SCENE)
        context = world.get_physics_context(); context.enable_gpu_dynamics(True); context.set_broadphase_type('GPU'); context.set_solver_type('TGS')
        local, statics, colliders = collision_bounds(stage, list(metadata['actors'].values()))
        if len(colliders[path]) != 220: raise ValueError('Original 220 active spoon collider paths required')
        report.update(body_local_conservative_boxes_m={p: x.tolist() for p, x in local.items()}, static_collision_bounds=statics, active_collision_paths=colliders)
        scene = stage.GetPrimAtPath(SCENE)
        names = ('physxScene:solverType', 'physxScene:enableGPUDynamics', 'physxScene:timeStepsPerSecond',
                 'physxScene:enableExternalForcesEveryIteration', 'physics:gravityMagnitude', 'physics:gravityDirection')
        report['scene_properties'] = {n: json_value(scene.GetAttribute(n).Get()) for n in names}
        body_prim = stage.GetPrimAtPath(path)
        report['body_properties'] = {a.GetName(): json_value(a.Get()) for a in body_prim.GetAttributes() if a.GetName().startswith(('physics:', 'physxRigidBody:'))}
        stage.Export(str(output/'initial.usda')); world.reset(); sim = world.physics_sim_view
        report['tensor_backend'] = {'device_ordinal': int(sim.device_ordinal), 'device': str(sim.device), 'frontend': 'torch'}
        if int(sim.device_ordinal) < 0: raise RuntimeError('Native GPU required')
        body = sim.create_rigid_body_view(path); all_body = sim.create_rigid_body_view(manifest['root_prim_path']+'/Actors/*')
        contact = sim.create_rigid_contact_view(path, max_contact_data_count=8192)
        if body.count != 1 or set(all_body.prim_paths) != set(metadata['actors'].values()): raise ValueError('Native actor path inventory differs')
        report['all_body_paths'] = list(all_body.prim_paths)
        mass = float(native(body.get_masses()).reshape(-1)[0]); com = native(body.get_coms())[0, :3].astype(float); inertia = native(body.get_inertias()).reshape(3, 3)
        rot = Rotation.from_quat(spec['principal_axes_xyzw']).as_matrix(); expected = rot@np.diag(spec['diagonal_inertia_kg_m2'])@rot.T
        np.testing.assert_allclose(mass, spec['mass_kg'], atol=1e-10, rtol=2e-6)
        np.testing.assert_allclose(com, spec['center_of_mass_m'], atol=2e-8, rtol=1e-5)
        np.testing.assert_allclose(inertia, expected, atol=1e-10, rtol=2e-4)
        report.update(native_mass_kg=mass, native_com_local_m=com.tolist(), native_inertia_kg_m2=inertia.tolist(),
                      native_offsets=native_offsets(body), native_material_properties=native(body.get_material_properties()).tolist())
        dump(output/'native_initialization.json', report)
        initial_time = float(world.current_time); one = torch.tensor([0], device='cuda:0', dtype=torch.int32)
        tensor = lambda v: torch.as_tensor(v, device='cuda:0', dtype=torch.float32)
        def sample(index, force, position):
            pose = native(body.get_transforms())[0]; velocity = native(body.get_velocities())[0]
            aq = native(all_body.get_transforms()); av = native(all_body.get_velocities())
            values = (float(world.current_time)-initial_time, int(world.current_time_step_index), pose, velocity, force, position, np.zeros(3, np.float32), aq, av)
            for k, v in zip(keys, values): raw[k].append(v)
            boxes = {}
            for p, q in zip(all_body.prim_paths, aq):
                points = Rotation.from_quat(q[3:]).apply(local[p])+q[:3]; boxes[p] = [points.min(0), points.max(0)]
            lo, hi = boxes[path]; obstacles = [(p, x) for p, x in boxes.items() if p != path]+[(x['path'], np.asarray(x['bounds_m'])) for x in statics]
            gaps = [(bbox_gap(lo, hi, np.asarray(b)), p) for p, b in obstacles]; closest = min(gaps)
            row = {'relative_step': index, 'time_s': values[0], 'native_step': values[1],
                   'contact': contact_row(contact, CONFIG['dt_s'], baseline=index == 0),
                   'body_bounds_m': [lo.tolist(), hi.tolist()], 'minimum_other_collision_gap_m': closest[0], 'closest_other_collider_or_actor': closest[1]}
            rows.append(row)
            if len(raw['time_s'])-saved >= 100: flush()
            return row
        with (output/'native_frames.jsonl').open('x') as stream:
            stream.write(json.dumps(sample(0, np.zeros(3, np.float32), np.zeros(3, np.float32)), allow_nan=False)+'\n')
            for index in range(1, 401):
                force = force_for_step(index).astype(np.float32); pose = native(body.get_transforms())[0]
                position = (Rotation.from_quat(pose[3:]).apply(com)+pose[:3]).astype(np.float32)
                if force.any():
                    ok = body.apply_forces_and_torques_at_position(tensor(force[None]), tensor(np.zeros((1, 3))), tensor(position[None]), one, True)
                    if ok is False: raise RuntimeError('External wrench application failed')
                world.step(render=False); stream.write(json.dumps(sample(index, force, position), allow_nan=False)+'\n')
        checks, measured, derived = evaluate(raw, mass, com, -report['scene_properties']['physics:gravityMagnitude'], spec['mass_kg'])
        checks.update(no_contact=all(r['contact']['point_count'] == 0 and r['contact']['finite'] and np.linalg.norm(r['contact']['net_contact_force_n']) <= 1e-8 for r in rows),
                      all_other_colliders_separated=all(r['minimum_other_collision_gap_m'] > .1 for r in rows), requested_solver=report['scene_properties']['physxScene:solverType'] == 'TGS')
        np.savez_compressed(output/'derived_momentum.npz', **derived)
        report.update(checks=checks, measurements=measured, passed=all(checks.values()), native_steps=400); code = 0 if report['passed'] else 1
    except BaseException:
        report['exception'] = traceback.format_exc(); traceback.print_exc()
    finally:
        try:
            flush(); np.savez_compressed(output/'native_states.npz', **{k: np.asarray(v) for k, v in raw.items()})
            with (output/'native_frames.jsonl').open('w') as stream:
                for row in rows: stream.write(json.dumps(row, allow_nan=False)+'\n')
            report['raw_frames'] = len(raw['time_s'])
            report['evidence_sha256'] = {p.name: digest(p) for p in output.iterdir() if p.is_file() and p.name != 'qualification.json'}
        except BaseException:
            report['archive_exception'] = traceback.format_exc(); code = 1; report['passed'] = False
        finally:
            report['process_exit_code'] = code; dump(output/'qualification.json', report)
            if app is not None: app.close(exit_code=code)
    return code


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--asset-manifest', type=Path, required=True)
    parser.add_argument('--actor', default='spoon_0')
    parser.add_argument('--output', type=Path, required=True)
    raise SystemExit(run(parser.parse_args()))
