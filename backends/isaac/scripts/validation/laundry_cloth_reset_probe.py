"""Short, unscored GPU probe separating authored cloth and tensor-reset startup.

Run both modes in separate processes from the same complete frozen source and
explicit task release. No grasp, attachment, prescribed motion, renderer or
task-success evaluator is created. World.reset's internal warmup is observed,
not bypassed; this probe does not claim to clear native solver caches.

An explicit physics_dt_s override supports time-step convergence diagnostics.
Use proportionally more steps to compare the same passive duration; no task
release or evaluator timing is changed by this unscored probe.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import traceback

import numpy as np

from bench.paths import load_json, OUTPUT_ROOT
from bench.simulation.isaacsim_backend.cloth_sleep import (
    SETTLING_THRESHOLD_OPTION, read_surface_sleep_settings, surface_settling_properties,
)
from scripts.maintenance.freeze_project_sources import verify_if_snapshot
from scripts.validation.laundry_cloth_interactions import (
    ConstraintLedger, array, digest, native_evidence,
)

PROJECT = Path(__file__).resolve().parents[2]
MODULES = ('scripts.validation.laundry_cloth_reset_probe',
           'scripts.validation.laundry_cloth_interactions',
           'scenes.workrooms.laundry_room.release')
PHYSICS_OVERRIDE_KEYS = frozenset({
    'physics_dt_s',
    SETTLING_THRESHOLD_OPTION,
    'cloth_iterations', 'cloth_linear_damping', 'cloth_collision_pair_update_frequency',
    'cloth_collision_iteration_multiplier', 'cloth_speculative_ccd',
    'cloth_self_collision_filter_distance_m', 'cloth_contact_offset_m', 'cloth_rest_offset_m',
    'cloth_youngs_modulus_pa', 'cloth_poissons_ratio', 'cloth_surface_bend_stiffness',
    'cloth_elasticity_damping', 'cloth_bend_damping',
})


def diagnostic_inputs(scene, task, *, physics_options=None, initial_native_report=None,
                      reset_mode='tensor-reset'):
    """Explicit counterfactual inputs; never publish them as a task release."""
    from scenes.workrooms.laundry_room.release import validate_release
    scene = deepcopy(scene)
    original = {s['id']: {**scene['isaacsim'], **s.get('isaacsim', {})} for s in scene['cloths']}
    evidence = {'original_release_options_by_cloth': original, 'physics_overrides': {},
                'initial_geometry_override': None}
    if physics_options is not None:
        path = physics_options.resolve()
        options = load_json(path)
        if (not isinstance(options, dict) or set(options)-PHYSICS_OVERRIDE_KEYS
                or any(not isinstance(v, (bool, int, float)) or not np.isfinite(v) for v in options.values())):
            raise ValueError('Diagnostic overrides must be finite supported cloth physics options')
        surface_settling_properties(options)
        if 'physics_dt_s' in options:
            dt = options['physics_dt_s']
            control_dt = float(scene['physics']['control_dt_s'])
            if (isinstance(dt, bool) or not 0. < dt <= .002
                    or not np.isfinite(control_dt) or control_dt <= 0.
                    or abs(control_dt/dt-round(control_dt/dt)) > 1.e-8):
                raise ValueError('Diagnostic physics_dt_s must be positive, at most 2 ms, and divide control_dt_s')
            evidence['physics_timestep_override'] = {
                'source_release_dt_s': float(scene['physics']['dt_s']),
                'effective_dt_s': float(dt), 'control_dt_s': control_dt,
                'scope': 'Passive numerical convergence only; task release and evaluator timing unchanged.'}
            # World reads scene.physics.dt_s, not the cloth/material options.
            scene['physics']['dt_s'] = float(dt)
        scene['isaacsim'].update(options)
        for spec in scene['cloths']:
            spec.setdefault('isaacsim', {}).update(options)
        evidence['physics_overrides'] = {'path': str(path), 'sha256': digest(path), 'options': options}
    if initial_native_report is not None:
        path = initial_native_report.resolve()
        report = load_json(path)
        if (task['id'] != 'towel_folding' or report.get('task_id') != 'towel_folding'
                or report.get('audit_contract') != 'laundry_native_stream.v1'
                or report.get('complete') is not True or report.get('error')
                or [s['id'] for s in scene['cloths']] != ['towel']):
            raise ValueError('Terminal geometry diagnostic requires a complete single-towel folding stream')
        source_release = Path(report['release']['path']).resolve()
        source_manifest = source_release/'release_manifest.json'
        if digest(source_manifest) != report['release']['manifest_sha256']:
            raise ValueError('Source fold release hash differs')
        validate_release(source_release, load_json(source_manifest))
        source_task = load_json(source_release/'task.json')
        source_specs = source_task['asset_metadata']['cloths']
        target_specs = task['asset_metadata']['cloths']
        if (len(source_specs) != 1 or len(target_specs) != 1
                or source_specs[0]['id'] != 'towel' or target_specs[0]['id'] != 'towel'
                or not np.array_equal(source_specs[0]['rest_vertices_m'], target_specs[0]['rest_vertices_m'])
                or not np.array_equal(source_specs[0]['triangles'], target_specs[0]['triangles'])):
            raise ValueError('Terminal geometry source rest shape or topology differs')
        archive = Path(report['state_archive']['path']).resolve()
        if digest(archive) != report['state_archive']['sha256']:
            raise ValueError('Terminal geometry native archive hash differs')
        with np.load(archive, allow_pickle=False) as data:
            if (not np.array_equal(data['towel_triangles'], target_specs[0]['triangles'])
                    or data['towel_q'].shape[0] != report['native_steps']+1
                    or int(data['native_step'][-1]) != report['native_steps']):
                raise ValueError('Terminal geometry native topology or last step differs')
            q = data['towel_q'][-1].copy()
            source_time = float(data['time_s'][-1])
        if q.shape != (len(target_specs[0]['rest_vertices_m']), 3) or not np.isfinite(q).all():
            raise ValueError('Terminal geometry is incomplete or nonfinite')
        scene['cloths'][0]['initial_vertices_m'] = q.tolist()
        evidence['initial_geometry_override'] = {
            'source_report': {'path': str(path), 'sha256': digest(path)},
            'source_state_archive': {'path': str(archive), 'sha256': digest(archive)},
            'source_release': {'path': str(source_release), 'manifest_sha256': digest(source_manifest)},
            'source_native_step': report['native_steps'], 'source_time_s': source_time,
            'velocity_initialization': (
                'All velocities zeroed by tensor setter at diagnostic baseline; original final velocities discarded.'
                if reset_mode == 'tensor-reset' else
                'No velocity state write: native World.reset warmup positions and velocities retained at baseline; original final velocities not imported.'),
            'reset_mode': reset_mode,
            'scope': 'Terminal geometry passive reset, not continuation of the original solver or task credit.'}
    evidence['effective_options_by_cloth'] = {
        s['id']: {**scene['isaacsim'], **s.get('isaacsim', {})} for s in scene['cloths']}
    return scene, evidence


def mesh_diagnostic(q, qd, rest, triangles, reference):
    """Geometric diagnostics only, with explicit nonfinite handling."""
    q, qd, rest, reference = [np.asarray(v, dtype=float) for v in (q, qd, rest, reference)]
    triangles = np.asarray(triangles, dtype=int)
    if q.shape != rest.shape or qd.shape != q.shape or reference.shape != q.shape:
        raise ValueError('Cloth diagnostic arrays differ in shape')
    if not all(np.isfinite(v).all() for v in (q, qd, rest, reference)):
        return {'finite': False}
    edges = np.unique(np.sort(np.concatenate([triangles[:, [0, 1]],
        triangles[:, [1, 2]], triangles[:, [2, 0]]]), axis=1), axis=0)
    lengths = np.linalg.norm(rest[edges[:, 1]]-rest[edges[:, 0]], axis=1)
    ratios = np.linalg.norm(q[edges[:, 1]]-q[edges[:, 0]], axis=1)/lengths
    index = int(np.argmax(ratios))
    return {'finite': True, 'max_edge_stretch_ratio': float(ratios[index]),
        'worst_edge_vertex_ids': edges[index].tolist(),
        'max_speed_m_s': float(np.linalg.norm(qd, axis=1).max()),
        'max_displacement_from_usd_initial_m': float(np.linalg.norm(q-reference, axis=1).max()),
        'z_min_max_m': [float(q[:, 2].min()), float(q[:, 2].max())]}


def solver_readback(stage, specs, scene_options, infos):
    """Reject a frozen backend that silently omits an explicit profile field."""
    mapping = {
        'cloth_iterations': 'physxDeformableBody:solverPositionIterationCount',
        'cloth_linear_damping': 'physxDeformableBody:linearDamping',
        SETTLING_THRESHOLD_OPTION: 'physxDeformableBody:settlingThreshold',
        'cloth_collision_pair_update_frequency': 'physxDeformableBody:collisionPairUpdateFrequency',
        'cloth_collision_iteration_multiplier': 'physxDeformableBody:collisionIterationMultiplier',
        'cloth_speculative_ccd': 'physxDeformableBody:enableSpeculativeCCD',
        'cloth_self_collision_filter_distance_m': 'physxDeformableBody:selfCollisionFilterDistance',
        'cloth_contact_offset_m': 'physxCollision:contactOffset',
        'cloth_rest_offset_m': 'physxCollision:restOffset',
    }
    result = {}
    for name, spec in specs.items():
        prim = stage.GetPrimAtPath(infos[name]['path'])
        options = {**scene_options, **spec.get('isaacsim', {})}
        result[name] = {}
        for key, attr_name in mapping.items():
            if key not in options:
                continue
            attr = prim.GetAttribute(attr_name)
            actual = attr.Get() if attr else None
            expected = options[key]
            if (actual is None or (isinstance(expected, bool) and actual is not expected)
                    or not np.isclose(actual, expected, rtol=1.e-6, atol=1.e-9)):
                raise RuntimeError(f'Native solver profile differs: {name}.{key}: {actual} != {expected}')
            result[name][attr_name] = actual
        # These values may come from the pinned schema fallback. Reading them
        # neither authors a sleep policy nor proves the native actor is asleep.
        result[name]['usd_sleep_settings'] = read_surface_sleep_settings(prim)
        if SETTLING_THRESHOLD_OPTION in options:
            from scripts.validation.audit_laundry_cloth import verify_surface_sleep_settings
            verify_surface_sleep_settings(options, name, result[name])
        material_mapping = {
            'cloth_youngs_modulus_pa': 'omniphysics:youngsModulus',
            'cloth_poissons_ratio': 'omniphysics:poissonsRatio',
            'cloth_surface_bend_stiffness': 'omniphysics:surfaceBendStiffness',
            'cloth_elasticity_damping': 'physxDeformableMaterial:elasticityDamping',
            'cloth_bend_damping': 'physxDeformableMaterial:bendDamping',
        }
        material = stage.GetPrimAtPath(infos[name]['material_path'])
        for key, attr_name in material_mapping.items():
            if key not in options:
                continue
            actual = material.GetAttribute(attr_name).Get()
            if actual is None or not np.isclose(actual, options[key], rtol=1.e-6, atol=1.e-9):
                raise RuntimeError(f'Native material profile differs: {name}.{key}: {actual} != {options[key]}')
            result[name][attr_name] = actual
    return result


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--release', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--reset-mode', required=True, choices=('tensor-reset', 'author-only'))
    p.add_argument('--steps', type=int, default=100)
    p.add_argument('--physics-options', type=Path,
        help='Explicit JSON diagnostic overrides, including physics_dt_s; never a release edit')
    p.add_argument('--initial-native-report', type=Path, help='Author a single folding towel from bound terminal geometry; chosen reset mode controls baseline handling')
    return p


def run(args):
    if not 1 <= args.steps <= 20000:
        raise ValueError('This passive diagnostic permits 1 to 20000 native steps')
    from scenes.workrooms.laundry_room.release import validate_release
    source = verify_if_snapshot(PROJECT, required_modules=MODULES, entrypoint_file=Path(__file__))
    if source is None:
        raise ValueError('A complete frozen project source is required')
    release = args.release.resolve()
    manifest = load_json(release/'release_manifest.json')
    validate_release(release, manifest)
    scene = load_json(release/'scene.json')
    task = load_json(release/'task.json')
    scene, input_evidence = diagnostic_inputs(scene, task,
        physics_options=args.physics_options, initial_native_report=args.initial_native_report,
        reset_mode=args.reset_mode)
    output = args.output.resolve()
    if not output.is_relative_to(OUTPUT_ROOT):
        raise ValueError('Native probe evidence belongs under shared datasets')
    output.mkdir(parents=True, exist_ok=False)
    bind = lambda p: {'path': str(p), 'sha256': digest(p)}
    report = {'schema': 'laundry_cloth_reset_probe.v1', 'complete': False,
        'scope': 'Unscored startup diagnostic; no operation, task credit, robot or calibration claim.',
        'task_id': task['id'], 'reset_mode': args.reset_mode, 'requested_steps': args.steps,
        'task_success_claimed': False, 'physics_accepted': False, 'model_calls': 0,
        'source_snapshot': source, 'source_snapshot_manifest': bind(PROJECT/'source_snapshot_manifest.json'),
        'release': {'path': str(release), 'manifest_sha256': digest(release/'release_manifest.json')},
        'scene_input': bind(release/'scene.json'), 'task_input': bind(release/'task.json'),
        'gpu_uuid': os.environ.get('CUDA_VISIBLE_DEVICES'), 'phases': {},
        'diagnostic_inputs': input_evidence,
        'constraint_policy': 'No attachments, joints, kinematic cloth or prescribed trajectory.',
        'warmup_policy': 'World.reset internal steps are retained and read before any tensor setter.',
        'requested_duration_s': args.steps*float(scene['physics']['dt_s']),
        'post_baseline_position_velocity_write_calls': [0, 0]}
    (output/'diagnostic_scene.json').write_text(json.dumps(scene, indent=2, allow_nan=False)+'\n')
    report['diagnostic_scene'] = bind(output/'diagnostic_scene.json')
    app = ledger = None
    saved, streams = {}, {}
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
        from pxr import Gf, Usd, UsdGeom, UsdPhysics
        from bench.simulation.isaacsim_backend.scene import add_named_cloth
        from bench.simulation.isaacsim_backend.workroom import verify_reference
        omni.usd.get_context().new_stage()
        stage = omni.usd.get_context().get_stage()
        UsdGeom.SetStageMetersPerUnit(stage, 1.)
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        root = UsdGeom.Xform.Define(stage, '/World').GetPrim()
        stage.SetDefaultPrim(root)
        report['room_dependency_hashes'] = verify_reference(scene['workroom'])
        root.GetReferences().AddReference(scene['workroom']['usd_path'], '/World')
        if any(p.IsA(UsdPhysics.Scene) or (p.HasAPI(UsdPhysics.RigidBodyAPI)
                and UsdPhysics.RigidBodyAPI(p).GetRigidBodyEnabledAttr().Get() is not False)
                for p in stage.Traverse()):
            raise RuntimeError('Frozen static room contains an unexpected physics scene or active body')
        table_path = '/World/Geometry/worktop/Collider'
        table = stage.GetPrimAtPath(table_path)
        if not table or not table.HasAPI(UsdPhysics.CollisionAPI) or UsdPhysics.CollisionAPI(table).GetCollisionEnabledAttr().Get() is not True:
            raise RuntimeError('Frozen actual worktop collider is absent or disabled')
        report['support_collider'] = {'path': table_path, 'type': table.GetTypeName(), 'duplicate_table_created': False}
        dt = float(scene['physics']['dt_s'])
        world = World(stage_units_in_meters=1., physics_dt=dt, rendering_dt=.02,
                      backend='torch', device='cuda:0', physics_prim_path='/World/PhysicsScene')
        context = world.get_physics_context()
        context.enable_gpu_dynamics(True)
        context.set_broadphase_type('GPU')
        context.set_solver_type('TGS')
        context.set_gravity(float(scene['physics']['gravity_m_s2'][2]))
        specs = {s['id']: s for s in scene['cloths']}
        infos = {name: add_named_cloth(stage, s, scene['isaacsim']) for name, s in specs.items()}
        paths = {name: info['path'] for name, info in infos.items()}
        ledger = ConstraintLedger(stage, paths, [])
        if ledger.constraint_prims or ledger.read()[1]:
            raise RuntimeError('A passive probe cannot contain constraints or kinematic cloth')
        report['solver_readback'] = solver_readback(stage, specs, scene['isaacsim'], infos)
        for name, info in infos.items():
            mesh = UsdGeom.Mesh(stage.GetPrimAtPath(info['path']))
            transform = UsdGeom.Xformable(mesh).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
            authored = np.asarray([transform.Transform(Gf.Vec3d(*map(float, p)))
                for p in mesh.GetPointsAttr().Get()], dtype=np.float32)
            if not np.allclose(authored, specs[name]['initial_vertices_m'], atol=5.e-7, rtol=0.):
                raise RuntimeError('Authored USD points differ from the explicit release initial vertices')
            saved[name+'_usd_initial_q'] = authored
            saved[name+'_rest_q'] = np.asarray(info['rest_vertices'])
            saved[name+'_triangles'] = np.asarray(info['triangles'])
        stage.Export(str(output/'authored.usda'))
        world.reset()
        views = {name: world.physics_sim_view.create_surface_deformable_body_view(path)
                 for name, path in paths.items()}
        if any(view.count != 1 for view in views.values()):
            raise RuntimeError('Native surface view count differs')

        def capture_phase(phase):
            row = {'absolute_world_step': int(world.current_time_step_index),
                   'absolute_world_time_s': float(world.current_time), 'cloths': {}}
            for name, view in views.items():
                q, qd = array(view.get_simulation_nodal_positions())[0], array(view.get_simulation_nodal_velocities())[0]
                saved[name+'_'+phase+'_q'], saved[name+'_'+phase+'_qd'] = q, qd
                row['cloths'][name] = mesh_diagnostic(q, qd, saved[name+'_rest_q'],
                    saved[name+'_triangles'], saved[name+'_usd_initial_q'])
            report['phases'][phase] = row

        capture_phase('warmup')
        if args.reset_mode == 'tensor-reset':
            for name, view in views.items():
                device = view.get_simulation_nodal_positions().device
                index = torch.tensor([0], dtype=torch.int32, device=device)
                q = torch.tensor(np.asarray(specs[name]['initial_vertices_m'], np.float32)[None], device=device)
                view.set_simulation_nodal_positions(q, index)
                view.set_simulation_nodal_velocities(torch.zeros_like(q), index)
        report['baseline_position_velocity_calls_by_cloth'] = {
            name: ([1, 1] if args.reset_mode == 'tensor-reset' else [0, 0]) for name in views}
        capture_phase('baseline')
        report['native_evidence'] = native_evidence(stage, views, infos, ledger)
        ledger.verify_inventory()
        report['constraint_inventory_method'] = (
            'Full stage inventory before and after startup and at stream end; read cached native '
            'constraint/kinematic flags every step. No stage edits or render updates during the passive stream.')
        evidence = report['native_evidence']
        report['native_physics_dt_s'] = float(world.get_physics_dt())
        if (not str(evidence['tensor_device']).startswith('cuda') or evidence['gpu_dynamics_enabled'] is not True
                or evidence['broadphase_type'] != 'GPU' or not np.allclose(evidence['gravity_m_s2'],
                scene['physics']['gravity_m_s2'], atol=1.e-6) or abs(report['native_physics_dt_s']-dt) > 1.e-12):
            raise RuntimeError('Native GPU, gravity or time step differs')
        for name, row in evidence['cloths'].items():
            if (row['enabled'] is not True or row['disable_gravity'] is not False
                    or row['kinematic_enabled'] is not False or row['self_collision'] is not True
                    or abs(row['surface_thickness_m']-.003) > 1.e-8):
                raise RuntimeError('Native cloth flags or 3 mm thickness differ: '+name)
        start_step, start_time = int(world.current_time_step_index), float(world.current_time)
        report['physics_dt_s'] = dt
        report['relative_stream_zero'] = {'absolute_world_step': start_step, 'absolute_world_time_s': start_time}
        streams = {name+suffix: [] for name in views for suffix in ('_q', '_qd')}
        streams.update({key: [] for key in ('native_step', 'time_s', 'absolute_world_step', 'absolute_world_time_s', 'hidden_constraints')})
        report['per_step_diagnostics'] = []
        for step in range(args.steps+1):
            if step:
                world.step(render=False)
            if int(world.current_time_step_index)-start_step != step:
                raise RuntimeError('Native step counter is not consecutive')
            _, hidden = ledger.read()
            if hidden:
                raise RuntimeError('Unexpected active native constraint')
            row = {'native_step': step, 'cloths': {}}
            for name, view in views.items():
                q, qd = array(view.get_simulation_nodal_positions())[0], array(view.get_simulation_nodal_velocities())[0]
                streams[name+'_q'].append(q)
                streams[name+'_qd'].append(qd)
                row['cloths'][name] = mesh_diagnostic(q, qd, saved[name+'_rest_q'],
                    saved[name+'_triangles'], saved[name+'_usd_initial_q'])
            report['per_step_diagnostics'].append(row)
            for key, value in {'native_step': step, 'time_s': step*dt,
                'absolute_world_step': int(world.current_time_step_index),
                'absolute_world_time_s': float(world.current_time), 'hidden_constraints': hidden}.items():
                streams[key].append(value)
        ledger.verify_inventory()
        report['constraint_inventory_checks'] = ledger.inventory_checks
        report['native_evidence_final'] = native_evidence(stage, views, infos, ledger)
        report['source_snapshot_after_run'] = verify_if_snapshot(PROJECT, required_modules=MODULES, entrypoint_file=Path(__file__))
        report['complete'] = True
        code = 0
    except BaseException as error:
        traceback.print_exc()
        report['error'] = type(error).__name__+': '+str(error)
    finally:
        if ledger is not None:
            try:
                ledger.verify_inventory()
                report['constraint_inventory_checks'] = ledger.inventory_checks
                report['final_inventory_verified'] = True
            except BaseException as error:
                report['final_inventory_verified'] = False
                report['final_inventory_error'] = type(error).__name__+': '+str(error)
                report['complete'] = False
                code = 1
        saved.update({key: np.asarray(value) for key, value in streams.items()})
        if saved:
            np.savez(output/'native_states.npz', **saved)
            report['state_archive'] = bind(output/'native_states.npz')
        (output/'report.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
        print(json.dumps({'complete': report['complete'], 'output': str(output)}), flush=True)
        if app is not None:
            app.close()
    return code


if __name__ == '__main__':
    raise SystemExit(run(parser().parse_args()))
