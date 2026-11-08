"""Sealed native 79.2 g rigid calibrand for the sponge free-flight force API.

Independent box with exact mass/COM/inertia. Fall .05 s, apply .4 N world-X at
actual COM for .05 s, then release .1 s. No deformable, joint or attachment;
no rendering and no post-reset state writes. This is an instrument calibration.
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
from scripts.validation import rigid_freeflight_metrics as metrics
from scripts.validation.sponge_freeflight_metrics import CONFIG, force_for_step
from scripts.validation.sponge_support_loadcell import native, json_value, contact_row, native_offsets, bbox_gap

PROJECT = Path(__file__).resolve().parents[2]
BODY = '/World/RigidCalibrand'
TABLE = '/World/FreeflightTableBoundary'
SCENE = '/World/PhysicsScene'
MODULES = ('scripts.validation.rigid_freeflight', 'scripts.validation.rigid_freeflight_metrics',
           'scripts.validation.sponge_freeflight_metrics', 'scripts.validation.sponge_support_loadcell')


def author(stage):
    from pxr import Gf, Sdf, UsdGeom, UsdPhysics
    UsdGeom.SetStageMetersPerUnit(stage, 1.); UsdGeom.SetStageUpAxis(stage, 'Z')
    for path, center, size in [(BODY, metrics.CENTER_M, metrics.SIZE_M), (TABLE, [.45, -.95, .8775], [2.8, .8, .045])]:
        box = UsdGeom.Cube.Define(stage, path); box.CreateSizeAttr(1.)
        box.AddTranslateOp().Set(Gf.Vec3d(*center)); box.AddScaleOp().Set(Gf.Vec3f(*size))
        prim = box.GetPrim(); UsdPhysics.CollisionAPI.Apply(prim); prim.AddAppliedSchema('PhysxCollisionAPI')
        for key, val in [('contactOffset', .0002 if path == BODY else .002), ('restOffset', 0.)]:
            prim.CreateAttribute('physxCollision:'+key, Sdf.ValueTypeNames.Float, custom=False).Set(val)
        if path == BODY:
            api = UsdPhysics.RigidBodyAPI.Apply(prim); api.CreateKinematicEnabledAttr(False)
            api.CreateSimulationOwnerRel().SetTargets([Sdf.Path(SCENE)])
            mass = UsdPhysics.MassAPI.Apply(prim); mass.CreateMassAttr(metrics.MASS_KG)
            mass.CreateCenterOfMassAttr(Gf.Vec3f(0., 0., 0.)); mass.CreateDiagonalInertiaAttr(Gf.Vec3f(*metrics.INERTIA_KG_M2))
            mass.CreatePrincipalAxesAttr(Gf.Quatf(1.))
            prim.AddAppliedSchema('PhysxRigidBodyAPI'); prim.AddAppliedSchema('PhysxContactReportAPI')
            for key, value in [('linearDamping', 0.), ('angularDamping', .05)]:
                prim.CreateAttribute('physxRigidBody:'+key, Sdf.ValueTypeNames.Float, custom=False).Set(value)
            for key, value in [('solverPositionIterationCount', 32), ('solverVelocityIterationCount', 1)]:
                prim.CreateAttribute('physxRigidBody:'+key, Sdf.ValueTypeNames.Int, custom=False).Set(value)
            prim.CreateAttribute('physxContactReport:threshold', Sdf.ValueTypeNames.Float, custom=False).Set(0.)
    physics = UsdPhysics.Scene.Define(stage, SCENE)
    physics.CreateGravityDirectionAttr(Gf.Vec3f(0., 0., -1.)); physics.CreateGravityMagnitudeAttr(9.81)


def run(args):
    source = verify_if_snapshot(PROJECT, required_modules=MODULES, entrypoint_file=Path(__file__))
    if source is None or Path.cwd().resolve() != PROJECT:
        raise ValueError('Use a complete frozen source root')
    source_file = PROJECT/'source_snapshot_manifest.json'; inventory = load_json(source_file)['files_sha256']
    if source_file.stat().st_mode & 0o222 or any((PROJECT/p).stat().st_mode & 0o222 for p in inventory):
        raise ValueError('Seal all source files before native launch')
    output = args.output.resolve()
    if not output.is_relative_to(OUTPUT_ROOT):
        raise ValueError('Evidence belongs in shared datasets')
    output.mkdir(parents=True, exist_ok=False)
    report = {'schema': 'rigid_freeflight_calibration.v1', 'passed': False, 'source': source,
        'source_snapshot_sha256': digest(source_file), 'solver_requested': args.solver,
        'gpu_uuid': os.environ.get('CUDA_VISIBLE_DEVICES'), 'configuration': CONFIG,
        'calibrand': {'mass_kg': metrics.MASS_KG, 'size_m': metrics.SIZE_M.tolist(), 'center_m': metrics.CENTER_M.tolist(),
                     'diagonal_inertia_kg_m2': metrics.INERTIA_KG_M2.tolist(), 'com_local_m': [0., 0., 0.],
                     'solver_position_iterations': 32, 'solver_velocity_iterations': 1, 'linear_angular_damping': [0., .05],
                     'iteration_basis': 'Matches awake bonded-FEM island maximum32 (foam32; original rigid pad schema default16).',
                     'damping_basis': 'Original pad has no authored damping override; pinned PhysxSchema110.3.2 defaults linear0 and angular0.05. Explicitly preserved here; no correction subtracted.'},
        'post_reset_pose_velocity_writes': 0, 'render_calls': 0, 'model_calls': 0,
        'force_application': 'Same native apply_forces_and_torques_at_position with world force at preceding actual world COM, zero torque, is_global=True, followed by World.step(render=False).',
        'scope': 'Independent rigid-body force-API calibration; not bonded FEM or robot task acceptance.'}
    keys = ('time_s', 'native_step', 'body_q', 'body_qd', 'force_world_n', 'force_position_world_m', 'torque_world_nm')
    raw = {k: [] for k in keys}; rows = []; chunks = []; archived = 0; app = None; code = 1
    def flush():
        nonlocal archived
        stop = len(raw['time_s'])
        if stop == archived: return
        directory = output/'native_chunks'; directory.mkdir(exist_ok=True)
        path = directory/f'frames_{len(chunks):06d}.npz'
        np.savez_compressed(path, **{k: np.asarray(v[archived:stop]) for k, v in raw.items()})
        chunks.append({'path': str(path), 'first_frame': archived, 'last_frame': stop-1, 'sha256': digest(path)})
        archived = stop
        dump(output/'native_chunks_manifest.json', {'frames': stop, 'chunks': chunks, 'trajectory_complete': stop == 401})
    try:
        from isaacsim import SimulationApp
        app = SimulationApp({'headless': True, 'multi_gpu': False, 'active_cuda_gpus': [0], 'max_gpu_count': 1,
            'physics_gpu': 0, 'fast_shutdown': True, 'extra_args': ['--/rtx/verifyDriverVersion/enabled=false']
            if os.environ.get('ISAAC_ALLOW_UNVALIDATED_DRIVER') == '1' else []}, experience=str(PROJECT/'bench/apps/workroom.headless.kit'))
        import omni.usd
        import torch
        from isaacsim.core.api import World
        from pxr import UsdGeom, UsdPhysics
        omni.usd.get_context().new_stage(); stage = omni.usd.get_context().get_stage(); author(stage)
        world = World(stage_units_in_meters=1., physics_dt=CONFIG['dt_s'], rendering_dt=1/30,
            backend='torch', device='cuda:0', physics_prim_path=SCENE)
        context = world.get_physics_context(); context.enable_gpu_dynamics(True)
        context.set_broadphase_type('GPU'); context.set_solver_type(args.solver)
        colliders = [str(p.GetPath()) for p in stage.Traverse() if p.HasAPI(UsdPhysics.CollisionAPI)]
        if sorted(colliders) != sorted([BODY, TABLE]) or any(p.IsA(UsdPhysics.Joint) or p.GetTypeName().endswith('Attachment') for p in stage.Traverse()):
            raise ValueError('Independent body and stationary table boundary only; no joint/attachment')
        bounds = UsdGeom.BBoxCache(0., [UsdGeom.Tokens.default_], useExtentsHint=False).ComputeWorldBound(stage.GetPrimAtPath(TABLE)).ComputeAlignedRange()
        obstacle = np.array([bounds.GetMin(), bounds.GetMax()], dtype=float)
        if not np.isfinite(obstacle).all() or (obstacle[1] < obstacle[0]).any(): raise ValueError('Invalid static boundary')
        report['static_collision_bounds'] = [{'path': TABLE, 'bounds_m': obstacle.tolist(), 'contact_offset_m': .002}]
        scene = stage.GetPrimAtPath(SCENE)
        names = ('physxScene:solverType', 'physxScene:enableGPUDynamics', 'physxScene:timeStepsPerSecond',
                 'physxScene:enableExternalForcesEveryIteration', 'physics:gravityMagnitude', 'physics:gravityDirection')
        report['scene_properties'] = {n: json_value(scene.GetAttribute(n).Get()) for n in names}
        stage.Export(str(output/'initial.usda')); world.reset(); sim = world.physics_sim_view
        report['tensor_backend'] = {'device_ordinal': int(sim.device_ordinal), 'device': str(sim.device), 'frontend': 'torch'}
        if int(sim.device_ordinal) < 0: raise RuntimeError('Native GPU simulation required')
        body = sim.create_rigid_body_view(BODY); contact = sim.create_rigid_contact_view(BODY, max_contact_data_count=8192)
        if body.count != 1: raise ValueError('Exactly one rigid calibrand required')
        mass = float(native(body.get_masses()).reshape(-1)[0]); com = native(body.get_coms())[0, :3].astype(float)
        inertia = native(body.get_inertias()).reshape(3, 3)
        report.update(native_mass_kg=mass, native_com_local_m=com.tolist(), native_inertia_kg_m2=inertia.tolist(),
                      native_offsets=native_offsets(body), native_material_properties=native(body.get_material_properties()).tolist())
        if not np.isclose(mass, metrics.MASS_KG, rtol=2e-6, atol=1e-10): raise ValueError('Native calibration mass differs')
        np.testing.assert_allclose(com, 0., atol=1e-9, rtol=0)
        np.testing.assert_allclose(inertia, np.diag(metrics.INERTIA_KG_M2), atol=1e-10, rtol=1e-5)
        for key, expected in [('get_contact_offsets', .0002), ('get_rest_offsets', 0.)]:
            row = report['native_offsets'][key]
            if not row['available'] or not np.allclose(row['values_m'], expected, atol=1e-8, rtol=0): raise ValueError('Native collision offset differs')
        prim = stage.GetPrimAtPath(BODY)
        report['body_properties'] = {n: json_value(prim.GetAttribute(n).Get()) for n in ('physics:kinematicEnabled',
            'physxRigidBody:linearDamping', 'physxRigidBody:angularDamping', 'physxRigidBody:solverPositionIterationCount', 'physxRigidBody:solverVelocityIterationCount')}
        if report['body_properties']['physics:kinematicEnabled'] is not False: raise ValueError('Calibration body must be dynamic')
        dump(output/'native_initialization.json', report)
        initial_time = float(world.current_time); one = torch.tensor([0], device='cuda:0', dtype=torch.int32)
        tensor = lambda v: torch.as_tensor(v, device='cuda:0', dtype=torch.float32)
        def sample(index, force, position):
            pose = native(body.get_transforms())[0]; velocity = native(body.get_velocities())[0]
            values = (float(world.current_time)-initial_time, int(world.current_time_step_index), pose, velocity, force, position, np.zeros(3, np.float32))
            for key, value in zip(keys, values): raw[key].append(value)
            row = {'relative_step': index, 'time_s': values[0], 'native_step': values[1]}; rows.append(row)
            row['contact'] = contact_row(contact, CONFIG['dt_s'], baseline=index == 0)
            points = Rotation.from_quat(pose[3:]).apply(metrics.CORNERS_M)+pose[:3]
            lo, hi = points.min(0), points.max(0)
            row['body_bounds_m'] = [lo.tolist(), hi.tolist()]; row['minimum_static_gap_m'] = bbox_gap(lo, hi, obstacle)
            if len(raw['time_s'])-archived >= 100: flush()
            return row
        with (output/'native_frames.jsonl').open('x') as stream:
            row = sample(0, np.zeros(3, np.float32), np.zeros(3, np.float32)); stream.write(json.dumps(row, allow_nan=False)+'\n')
            for index in range(1, 401):
                force = force_for_step(index).astype(np.float32); pose = native(body.get_transforms())[0]
                position = (Rotation.from_quat(pose[3:]).apply(com)+pose[:3]).astype(np.float32)
                if force.any():
                    result = body.apply_forces_and_torques_at_position(tensor(force[None]), tensor(np.zeros((1, 3))), tensor(position[None]), one, True)
                    if result is False: raise RuntimeError('External wrench application failed')
                world.step(render=False); row = sample(index, force, position); stream.write(json.dumps(row, allow_nan=False)+'\n')
        checks, measured, derived = metrics.evaluate(raw, mass, com, -report['scene_properties']['physics:gravityMagnitude'])
        checks.update(no_contact=all(r['contact']['point_count'] == 0 and r['contact']['finite'] and np.linalg.norm(r['contact']['net_contact_force_n']) <= 1e-8 for r in rows),
            all_static_colliders_separated=all(r['minimum_static_gap_m'] > .01 for r in rows),
            requested_solver=report['scene_properties']['physxScene:solverType'] == args.solver)
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
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--solver', choices=('TGS', 'PGS'), required=True)
    raise SystemExit(run(parser.parse_args()))
