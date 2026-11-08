"""Native passive D6 spring calibrand: momentum and relative compression.

Two freely moving rigid cells with a finite-stiffness six-axis material link.
This is a constitutive/numerical diagnostic, not a sponge asset or controller.
Rest targets remain fixed. No state/target writes occur after native reset.
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
from scripts.validation.sponge_support_loadcell import native, json_value

PROJECT = Path(__file__).resolve().parents[2]
SCENE = '/World/PhysicsScene'
BODIES = ('/World/Cells/lower', '/World/Cells/upper')
DT = .0005
MASS = .0792
SIZE = np.array([.120, .078, .0225])
K = 1000.
ZETA = .15
MU = MASS / 4
DAMPING = 2*ZETA*np.sqrt(K*MU)


def author(stage):
    from pxr import Gf, Sdf, UsdGeom, UsdPhysics
    UsdGeom.SetStageMetersPerUnit(stage, 1.)
    UsdGeom.SetStageUpAxis(stage, 'Z')
    inertia = (MASS/2)/12*np.array([SIZE[1]**2+SIZE[2]**2,
                                   SIZE[0]**2+SIZE[2]**2, SIZE[0]**2+SIZE[1]**2])
    for index, path in enumerate(BODIES):
        body = UsdGeom.Xform.Define(stage, path)
        body.AddTranslateOp().Set(Gf.Vec3d(0., 0., 1.+index*SIZE[2]))
        body.AddOrientOp().Set(Gf.Quatf(1.))
        prim = body.GetPrim()
        rb = UsdPhysics.RigidBodyAPI.Apply(prim)
        rb.CreateKinematicEnabledAttr(False)
        rb.CreateSimulationOwnerRel().SetTargets([Sdf.Path(SCENE)])
        mp = UsdPhysics.MassAPI.Apply(prim)
        mp.CreateMassAttr(MASS/2)
        mp.CreateCenterOfMassAttr(Gf.Vec3f(0.))
        mp.CreateDiagonalInertiaAttr(Gf.Vec3f(*inertia))
        mp.CreatePrincipalAxesAttr(Gf.Quatf(1.))
        prim.AddAppliedSchema('PhysxRigidBodyAPI')
        for name, val in [('linearDamping', 0.), ('angularDamping', 0.)]:
            prim.CreateAttribute('physxRigidBody:'+name, Sdf.ValueTypeNames.Float).Set(val)
        for name, val in [('solverPositionIterationCount', 32), ('solverVelocityIterationCount', 1)]:
            prim.CreateAttribute('physxRigidBody:'+name, Sdf.ValueTypeNames.Int).Set(val)
        cube = UsdGeom.Cube.Define(stage, path+'/Collider')
        cube.CreateSizeAttr(1.)
        cube.AddScaleOp().Set(Gf.Vec3f(*SIZE))
        UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
        UsdPhysics.FilteredPairsAPI.Apply(prim).CreateFilteredPairsRel().SetTargets(
            [Sdf.Path(BODIES[1-index])])
    joint = UsdPhysics.Joint.Define(stage, '/World/ElasticLink')
    joint.CreateBody0Rel().SetTargets([Sdf.Path(BODIES[0])])
    joint.CreateBody1Rel().SetTargets([Sdf.Path(BODIES[1])])
    joint.CreateLocalPos0Attr().Set(Gf.Vec3f(0., 0., SIZE[2]/2))
    joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0., 0., -SIZE[2]/2))
    joint.CreateLocalRot0Attr().Set(Gf.Quatf(1.))
    joint.CreateLocalRot1Attr().Set(Gf.Quatf(1.))
    joint.CreateExcludeFromArticulationAttr(True)
    joint.CreateCollisionEnabledAttr(False)
    # USD angular force-drive coefficients are N m / degree, not N m / rad.
    # Small finite angular stiffness; every axis remains soft/unlocked.
    rotation_k = K * (SIZE[0]**2+SIZE[1]**2)/24
    rotation_d = 2*ZETA*np.sqrt(rotation_k*float(inertia.min())/2)
    drives = {}
    for axis in ('transX', 'transY', 'transZ', 'rotX', 'rotY', 'rotZ'):
        angular = axis.startswith('rot')
        k = rotation_k*np.pi/180 if angular else K
        damping = rotation_d*np.pi/180 if angular else DAMPING
        api = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), axis)
        api.CreateTypeAttr('force')
        api.CreateStiffnessAttr(float(k))
        api.CreateDampingAttr(float(damping))
        api.CreateTargetPositionAttr(0.)
        api.CreateTargetVelocityAttr(0.)
        api.CreateMaxForceAttr(float('inf'))
        drives[axis] = {'stiffness': float(k), 'damping': float(damping)}
    physics = UsdPhysics.Scene.Define(stage, SCENE)
    physics.CreateGravityDirectionAttr(Gf.Vec3f(0., 0., -1.))
    physics.CreateGravityMagnitudeAttr(9.81)
    return drives


def run(args):
    source = verify_if_snapshot(PROJECT,
        required_modules=('scripts.validation.elastic_cell_probe',
                          'scripts.validation.sponge_support_loadcell'), entrypoint_file=Path(__file__))
    if source is None or Path.cwd().resolve() != PROJECT:
        raise ValueError('Use an exact complete sealed source root')
    inventory = load_json(PROJECT/'source_snapshot_manifest.json')['files_sha256']
    if any((PROJECT/p).stat().st_mode & 0o222 for p in inventory):
        raise ValueError('Source is not sealed')
    out = args.output.resolve()
    if not out.is_relative_to(OUTPUT_ROOT):
        raise ValueError('Evidence belongs in shared datasets')
    out.mkdir(parents=True, exist_ok=False)
    report = {'schema': 'native_elastic_cell_probe.v1', 'source': source,
        'source_manifest_sha256': digest(PROJECT/'source_snapshot_manifest.json'),
        'gpu_uuid': os.environ.get('CUDA_VISIBLE_DEVICES'), 'mode': args.mode,
        'passed': False, 'body_mass_kg': MASS/2, 'linear_k_n_m': K,
        'linear_damping_ns_m': DAMPING, 'post_reset_state_or_target_writes': 0,
        'scope': 'Independent two-cell passive D6 diagnostic; not a qualified sponge or foam material.'}
    raw = {k: [] for k in ('time', 'native_step', 'q', 'qd', 'forces', 'force_positions')}
    app = None
    code = 1
    try:
        from isaacsim import SimulationApp
        app = SimulationApp({'headless': True, 'multi_gpu': False, 'active_cuda_gpus': [0],
            'max_gpu_count': 1, 'physics_gpu': 0, 'fast_shutdown': True,
            'extra_args': ['--/rtx/verifyDriverVersion/enabled=false']},
            experience=str(PROJECT/'bench/apps/workroom.headless.kit'))
        import omni.usd
        import torch
        from isaacsim.core.api import World
        omni.usd.get_context().new_stage()
        stage = omni.usd.get_context().get_stage()
        report['drives'] = author(stage)
        world = World(stage_units_in_meters=1., physics_dt=DT, rendering_dt=1/30,
                      backend='torch', device='cuda:0', physics_prim_path=SCENE)
        ctx = world.get_physics_context()
        ctx.enable_gpu_dynamics(True)
        ctx.set_broadphase_type('GPU')
        ctx.set_solver_type('TGS')
        stage.Export(str(out/'initial.usda'))
        world.reset()
        sim = world.physics_sim_view
        if int(sim.device_ordinal) < 0:
            raise RuntimeError('Native GPU dynamics required')
        views = [sim.create_rigid_body_view(p) for p in BODIES]
        masses = np.array([native(v.get_masses()).reshape(-1)[0] for v in views])
        np.testing.assert_allclose(masses, MASS/2, rtol=1e-6)
        report['native_masses_kg'] = masses.tolist()
        report['native_inertias'] = [native(v.get_inertias()).tolist() for v in views]
        report['tensor_device'] = str(sim.device)
        one = torch.tensor([0], device='cuda:0', dtype=torch.int32)
        tensor = lambda x: torch.as_tensor(x, device='cuda:0', dtype=torch.float32)
        zero = tensor(np.zeros((1, 3)))
        start = float(world.current_time)
        for step in range(401):
            forces = np.zeros((2, 3), dtype=np.float32)
            positions = np.array([native(v.get_transforms())[0, :3] for v in views])
            if 100 < step <= 200:
                if args.mode == 'impulse': forces[0, 0] = .4
                else: forces[:, 2] = [.4, -.4]
            if step:
                for view, force, position in zip(views, forces, positions):
                    if force.any():
                        result = view.apply_forces_and_torques_at_position(
                            tensor(force[None]), zero, tensor(position[None]), one, True)
                        if result is False: raise RuntimeError('Force API rejected')
                world.step(render=False)
            values = (float(world.current_time)-start, int(world.current_time_step_index),
                      np.array([native(v.get_transforms())[0] for v in views]),
                      np.array([native(v.get_velocities())[0] for v in views]), forces, positions)
            for key, value in zip(raw, values): raw[key].append(value)
        arrays = {k: np.asarray(v) for k, v in raw.items()}
        t, q, qd = arrays['time'], arrays['q'], arrays['qd']
        momentum = np.einsum('b,tbc->tc', masses, qd[:, :, :3])
        observed = momentum[200]-momentum[100]
        expected = np.sum(arrays['forces'][101:201].sum(axis=1)*np.diff(t)[100:200, None], axis=0)
        expected[2] -= masses.sum()*9.81*(t[200]-t[100])
        rotations = Rotation.from_quat(q[:, 0, 3:])
        relative = rotations.inv().apply(q[:, 1, :3]-q[:, 0, :3])
        compression = SIZE[2]-relative[:, 2]
        checks = {'complete_400_steps': bool(len(t)==401 and arrays['native_step'][-1]-arrays['native_step'][0]==400),
                  'finite': bool(all(np.isfinite(a).all() for a in arrays.values())),
                  'total_momentum_error_below_1_percent_pulse': bool(np.linalg.norm(observed-expected)<.0002),
                  'release_momentum_x_drift_below_1_percent_pulse': bool(abs(momentum[-1, 0]-momentum[200, 0])<.0002)}
        if args.mode == 'squeeze':
            checks.update(compression_observed=bool(compression[101:201].max()>.0002),
                          rebound=bool(abs(compression[-1])<.0001),
                          squeeze_x_y_momentum_near_zero=bool(np.linalg.norm(momentum[:, :2],axis=1).max()<1e-5))
        report.update(checks=checks, passed=all(checks.values()),
            observed_pulse_delta_momentum_ns=observed.tolist(), expected_pulse_delta_momentum_ns=expected.tolist(),
            maximum_compression_m=float(compression.max()), final_compression_m=float(compression[-1]),
            raw_frames=len(t))
        code = 0 if report['passed'] else 1
    except BaseException:
        report['exception'] = traceback.format_exc()
        traceback.print_exc()
    finally:
        np.savez_compressed(out/'native_states.npz', **{k: np.asarray(v) for k, v in raw.items()})
        report['evidence_sha256'] = {p.name: digest(p) for p in out.iterdir() if p.is_file()}
        report['process_exit_code'] = code
        dump(out/'qualification.json', report)
        if app is not None: app.close(exit_code=code)
    return code


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('impulse', 'squeeze'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    raise SystemExit(run(parser.parse_args()))
