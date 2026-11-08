"""GPU cloth engineering probes with explicit local vertex attachments.

This is not a robot policy or public Bench action. Formal task runs use small,
declared grasp patches attached to plain Xform anchors, then disable every
attachment. Failed rigid-anchor and nodal-boundary probes remain reproducible.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import traceback

import numpy as np

from bench.paths import load_json, OUTPUT_ROOT
from scripts.maintenance.freeze_project_sources import verify_if_snapshot
from scripts.validation.laundry_initialization import (SCHEMA as INITIALIZATION_SCHEMA,
    MODES as INITIALIZATION_MODES, CountedSurfaceView, authored_arrays,
    authored_control_masks, anchor_targets, same_bits, world_clock_tick)

PROJECT = Path(__file__).resolve().parents[2]
FORMAL_TASKS = ('towel_flattening', 'towel_folding', 'towel_stacking',
                'shirt_stacking', 'trousers_stacking')


def array(value):
    return value.detach().cpu().numpy().copy() if hasattr(value, 'detach') else np.asarray(value).copy()


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def smooth(value):
    value = np.clip(value, 0., 1.)
    return value * value * (3. - 2. * value)


def attachment(stage, path, cloth_path, anchor_path, ids, offsets):
    """Author a native vertex-to-Xform constraint, not nodal state playback."""
    from pxr import Vt
    prim = stage.DefinePrim(path, 'OmniPhysicsVtxXformAttachment')
    prim.GetRelationship('omniphysics:src0').SetTargets([cloth_path])
    prim.GetRelationship('omniphysics:src1').SetTargets([anchor_path])
    prim.GetAttribute('omniphysics:vtxIndicesSrc0').Set(Vt.IntArray(ids))
    prim.GetAttribute('omniphysics:localPositionsSrc1').Set(Vt.Vec3fArray.FromNumpy(np.asarray(offsets, np.float32)))
    prim.GetAttribute('omniphysics:attachmentEnabled').Set(True)
    return prim


class LocalNodalBoundary:
    """Explicit engineering Dirichlet boundary on a small node subset only.

    The installed surface backend has state setters but no nodal kinematic
    target API. Every other vertex is copied from current native state without
    alteration. After release the caller must never call drive again.
    """

    def __init__(self, view, ids, dt):
        import torch
        self.view, self.ids, self.dt = view, np.asarray(ids, dtype=np.int32), float(dt)
        data = view.get_simulation_nodal_positions()
        self.device = data.device
        self.one = torch.tensor([0], dtype=torch.int32, device=self.device)
        self.previous = array(data)[0, self.ids]
        self.writes = 0
        self.uncontrolled_unchanged = True
        self.released = False

    def drive(self, targets):
        import torch
        if self.released:
            raise RuntimeError('No nodal write is permitted after release')
        q = array(self.view.get_simulation_nodal_positions())
        qd = array(self.view.get_simulation_nodal_velocities())
        keep = np.ones(q.shape[1], dtype=bool)
        keep[self.ids] = False
        untouched_q, untouched_qd = q[:, keep].copy(), qd[:, keep].copy()
        targets = np.asarray(targets, dtype=np.float32)
        q[0, self.ids] = targets
        qd[0, self.ids] = (targets-self.previous)/self.dt
        assert np.array_equal(q[:, keep], untouched_q) and np.array_equal(qd[:, keep], untouched_qd)
        self.view.set_simulation_nodal_positions(torch.tensor(q, device=self.device), self.one)
        self.view.set_simulation_nodal_velocities(torch.tensor(qd, device=self.device), self.one)
        self.previous = targets.copy()
        self.writes += 1

    def release(self):
        self.released = True


def run(args):
    source = verify_if_snapshot(PROJECT, required_modules=('scripts.validation.laundry_cloth_interactions',),
                                entrypoint_file=Path(__file__))
    if source is None:
        raise ValueError('Run from a complete frozen project source')
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    report = {'schema': 'laundry_cloth_engineering_probe.v1', 'case': args.case, 'passed': False,
              'source_snapshot': source, 'gpu_uuid': os.environ.get('CUDA_VISIBLE_DEVICES'),
              'physics_dt_s': args.dt, 'robot_present': False, 'model_calls': 0,
              'task_success_claimed': False, 'hand_object_interaction_qualified': False,
              'material_calibrated': False, 'physics_force_calibration_claimed': False,
              'scope': 'Native local grasp-patch attachment/release and thick cloth support feasibility; no robot grasp claim.'}
    app = None
    try:
        from isaacsim import SimulationApp
        app = SimulationApp({'headless': True, 'width': 640, 'height': 480, 'multi_gpu': False,
                             'active_cuda_gpus': [0], 'max_gpu_count': 1, 'physics_gpu': 0,
                             'fast_shutdown': True, 'extra_args': ['--/rtx/verifyDriverVersion/enabled=false']},
                            experience=str(PROJECT / 'bench/apps/workroom.headless.kit'))
        import omni.usd
        import torch
        from isaacsim.core.api import World
        from pxr import Gf, Sdf, UsdGeom, UsdPhysics, PhysxSchema
        from bench.simulation.isaacsim_backend.scene import add_cloth, bind_visual, collision, rigid_material, transform, visual_material

        omni.usd.get_context().new_stage()
        stage = omni.usd.get_context().get_stage()
        world = World(stage_units_in_meters=1., physics_dt=args.dt, rendering_dt=.02,
                      backend='torch', device='cuda:0', physics_prim_path='/World/PhysicsScene')
        context = world.get_physics_context()
        context.enable_gpu_dynamics(True)
        context.set_broadphase_type('GPU')
        context.set_solver_type('TGS')
        table = UsdGeom.Cube.Define(stage, '/World/Table')
        table.CreateSizeAttr(1.)
        transform(table.GetPrim(), (.65, -.55, .86), scale=(1.8, .9, .08))
        collision(table.GetPrim(), rigid_material(stage, 'table', .6), contact_offset=.002, rest_offset=0.)
        bind_visual(table.GetPrim(), visual_material(stage, 'table', (.35, .22, .12)))

        nx, ny = 30, 20
        start_z = .925 if args.case == 'lift_release' else .915
        spec = {'enabled': True, 'cells_xy': [nx, ny], 'size_m': [.6, .4],
                'origin_m': [.35, -.75, start_z], 'mass_kg': .108, 'friction': .6, 'self_contact': True}
        options = {'cloth_thickness_m': .003, 'cloth_contact_offset_m': .004,
                   'cloth_rest_offset_m': .0015, 'cloth_youngs_modulus_pa': args.young,
                   'cloth_surface_bend_stiffness': args.bend, 'cloth_iterations': 64,
                   'cloth_linear_damping': .25, 'cloth_collision_pair_update_frequency': 4,
                   'cloth_collision_iteration_multiplier': 4, 'cloth_speculative_ccd': True}
        cloth = add_cloth(stage, spec, options)
        cloth_paths = {'towel': cloth['path']}
        if args.case == 'two_layer':
            Sdf.CopySpec(stage.GetRootLayer(), cloth['path'], stage.GetRootLayer(), '/World/ClothBase')
            transform(stage.GetPrimAtPath('/World/ClothBase'), (.35, -.75, .904))
            bind_visual(stage.GetPrimAtPath('/World/ClothBase'), visual_material(stage, 'base_cloth', (.25, .42, .6)))
            cloth_paths['towel_base'] = '/World/ClothBase'
        rest_world = np.asarray(cloth['rest_vertices']) + np.asarray(spec['origin_m'])
        patches = [[j*(nx+1)+i for j in (0, 1) for i in (nx-1, nx)],
                   [j*(nx+1)+i for j in (ny-1, ny) for i in (nx-1, nx)]]
        controlled = sorted(sum(patches, []))
        assert len(controlled) / len(rest_world) < .02
        anchors = []
        anchor_ops = []
        origins = []
        for index, ids in enumerate(patches):
            path = '/World/GraspAnchor_' + str(index)
            anchor = UsdGeom.Sphere.Define(stage, path)
            anchor.CreateRadiusAttr(.008)
            center = rest_world[ids].mean(axis=0)
            transform(anchor.GetPrim(), center)
            anchor_ops.append(anchor.GetPrim().GetAttribute('xformOp:translate'))
            if args.control_mode != 'plain_xform':
                body = UsdPhysics.RigidBodyAPI.Apply(anchor.GetPrim())
                body.CreateKinematicEnabledAttr(True)
                UsdPhysics.CollisionAPI.Apply(anchor.GetPrim())
                pairs = UsdPhysics.FilteredPairsAPI.Apply(anchor.GetPrim()).CreateFilteredPairsRel()
                for cloth_path in cloth_paths.values():
                    pairs.AddTarget(cloth_path)
                UsdPhysics.MassAPI.Apply(anchor.GetPrim()).CreateMassAttr(.1)
                PhysxSchema.PhysxRigidBodyAPI.Apply(anchor.GetPrim()).CreateDisableGravityAttr(True)
            bind_visual(anchor.GetPrim(), visual_material(stage, 'anchor', (.8, .12, .06)))
            origins.append(center)
            if args.control_mode != 'nodal':
                anchors.append(attachment(stage, '/World/GraspAttachment_' + str(index), cloth['path'], path,
                                          ids, rest_world[ids] - center))
        report['cloth_spec'] = spec
        report['physics_options'] = options
        report['material'] = cloth['material']
        report['engineering_boundary'] = {'mode': 'native_omniphysics_vertex_xform_attachment',
            'patch_vertex_ids': patches, 'controlled_nodes': len(controlled), 'total_nodes': len(rest_world),
            'controlled_fraction': len(controlled)/len(rest_world), 'anchor_collision_shapes': 2,
            'anchor_cloth_collision_filtered': True,
            'cloth_position_writes_after_reset': False, 'cloth_velocity_writes_after_reset': False,
            'whole_cloth_animation': False, 'release_time_s': 1.0,
            'anchor_motion_api': 'USD xformOp:translate on native kinematic rigid body',
            'native_kinematic_targets_used': False,
            'anchor_motion_limit': 'Pinned GPU backend rejects set_kinematic_targets and its tensor pose writes did not move attachment targets in prototype03. Native USD kinematic animation follows installed PhysX attachment tests; only two independent anchors are animated.',
            'release_method': 'omniphysics:attachmentEnabled=false on both native constraints',
            'post_release_anchor_motion': 'anchors move an additional 0.1m upward while all cloth vertices are dynamic',
            'constraint_stiffness': 'native hard attachment; not a frictional robot grasp'}
        if args.control_mode == 'nodal':
            report['engineering_boundary'].update(mode='engineering_local_nodal_boundary',
                cloth_position_writes_after_reset='only declared 8/651 local grasp nodes while control_enabled',
                cloth_velocity_writes_after_reset='only declared local grasp nodes; finite-difference target velocity',
                anchor_motion_api=None, native_attachments=0,
                release_method='stop every nodal position/velocity setter permanently',
                anchor_motion_limit='No working native surface kinematic-target/attachment-motion API claimed.',
                constraint_stiffness='hard local prescribed boundary, not a robot grasp',
                post_release_anchor_motion='unused anchor targets continue upward for comparison; cloth receives no writes')
        elif args.control_mode == 'plain_xform':
            report['engineering_boundary'].update(mode='native_plain_xform_attachment',
                anchor_collision_shapes=0, anchor_cloth_collision_filtered=False,
                anchor_motion_api='USD xformOp:translate plus IPhysxSimulation.flush_changes before every physics step',
                anchor_motion_limit='Plain non-rigid Xformable anchor: prescribed local attachment boundary, not a robot grasp.',
                anchor_native_body_view=False)
        stage.Export(str(output / 'initial.usda'))
        world.reset()
        sim = world.physics_sim_view
        views = {name: sim.create_surface_deformable_body_view(path) for name, path in cloth_paths.items()}
        anchor_views = ([sim.create_rigid_body_view('/World/GraspAnchor_' + str(i)) for i in range(2)]
                        if args.control_mode != 'plain_xform' else [])
        assert all(view.count == 1 for view in [*views.values(), *anchor_views])
        device = views['towel'].get_simulation_nodal_positions().device
        from omni.physx import get_physx_simulation_interface
        physics_interface = get_physx_simulation_interface()
        one = torch.tensor([0], dtype=torch.int32, device=device)
        boundary = LocalNodalBoundary(views['towel'], controlled, args.dt) if args.control_mode == 'nodal' else None
        series = {name: {'q': [], 'qd': []} for name in views}
        times, controls, anchor_targets, anchor_actual = [], [], [], []
        released = False
        duration = 2.5
        for step in range(round(duration / args.dt)):
            time = step * args.dt
            if time >= 1.0 and not released:
                for prim in anchors:
                    prim.GetAttribute('omniphysics:attachmentEnabled').Set(False)
                if boundary is not None:
                    boundary.release()
                released = True
            dz = .15 * smooth((time - .2) / .6)
            if released:
                dz += .1 * smooth((time - 1.) / .5)
            targets = []
            for index in range(2):
                target = np.r_[origins[index] + [0, 0, dz], [0, 0, 0, 1]].astype(np.float32)
                anchor_ops[index].Set(Gf.Vec3d(*map(float, target[:3])))
                targets.append(target)
            if boundary is not None and not released:
                boundary.drive(rest_world[controlled] + [0, 0, dz])
            if args.control_mode == 'plain_xform':
                physics_interface.flush_changes()
            world.step(render=False)
            times.append((step + 1) * args.dt)
            controls.append(not released)
            anchor_targets.append(targets)
            anchor_actual.append([array(view.get_transforms())[0] for view in anchor_views] if anchor_views else targets)
            for name, view in views.items():
                series[name]['q'].append(array(view.get_simulation_nodal_positions())[0])
                series[name]['qd'].append(array(view.get_simulation_nodal_velocities())[0])
            if step % 1000 == 0:
                print(json.dumps({'event': 'progress', 'step': step + 1, 'time_s': times[-1], 'released': released}), flush=True)
        arrays = {'time_s': np.asarray(times), 'control_enabled': np.asarray(controls),
                  'anchor_targets': np.asarray(anchor_targets), 'anchor_actual': np.asarray(anchor_actual),
                  'controlled_vertex_ids': np.asarray(controlled), 'triangles': cloth['triangles']}
        for name, data in series.items():
            for key, values in data.items():
                arrays[name + '_' + key] = np.asarray(values)
        np.savez_compressed(output / 'native_states.npz', **arrays)
        q, qd = arrays['towel_q'], arrays['towel_qd']
        tail = arrays['time_s'] >= duration - .5
        held = (arrays['time_s'] > .85) & (arrays['time_s'] < .99)
        target_error = []
        for index, ids in enumerate(patches):
            reference = arrays['anchor_targets'] if boundary is not None or args.control_mode == 'plain_xform' else arrays['anchor_actual']
            expected = reference[held, index, :3][:, None, :] + (rest_world[ids] - origins[index])[None, :, :]
            target_error.append(np.linalg.norm(q[held][:, ids, :] - expected, axis=-1).max())
        checks = {'finite_states': all(np.isfinite(value).all() for value in arrays.values()),
                  'local_control_less_than_two_percent': len(controlled)/len(rest_world) < .02,
                  'controlled_patch_tracks_anchor_within_3mm': max(target_error) < .003,
                  'grasp_patch_lifted_at_least_100mm': float(q[held][:, controlled, 2].min() - start_z) > .1,
                  'released_vertices_fall_at_least_80mm': float(q[held][:, controlled, 2].mean()-q[-1, controlled, 2].mean()) > .08,
                  'release_attribute_disabled': all(not prim.GetAttribute('omniphysics:attachmentEnabled').Get() for prim in anchors),
                  'released_rest_500ms': float(np.linalg.norm(qd[tail], axis=-1).max()) < .025,
                  'no_gross_table_penetration': float(q[:, :, 2].min()) >= .898,
                  'all_control_disabled_in_final_500ms': not arrays['control_enabled'][tail].any()}
        metrics = {'max_patch_tracking_error_m': float(max(target_error)),
                   'final_min_max_z_m': [float(q[-1, :, 2].min()), float(q[-1, :, 2].max())],
                   'final_500ms_max_node_speed_m_s': float(np.linalg.norm(qd[tail], axis=-1).max())}
        if boundary is not None:
            report['engineering_boundary'].update(nodal_write_steps=boundary.writes,
                uncontrolled_input_arrays_bitwise_unchanged=boundary.uncontrolled_unchanged,
                released=boundary.released, post_release_nodal_write_steps=0)
        if args.case == 'two_layer':
            base = arrays['towel_base_q']
            separation = q[-1, :, 2] - base[-1, :, 2]
            metrics['final_corresponding_node_z_separation_m'] = {'min': float(separation.min()),
                'median': float(np.median(separation)), 'max': float(separation.max())}
            checks['upper_lower_order_for_all_corresponding_nodes'] = float(separation.min()) > .001
            checks['two_layer_median_separation_2_to_8mm'] = .002 <= float(np.median(separation)) <= .008
            checks['base_released_rest_500ms'] = float(np.linalg.norm(arrays['towel_base_qd'][tail], axis=-1).max()) < .025
        stage.Export(str(output / 'final.usda'))
        checks = {name: bool(value) for name, value in checks.items()}
        report.update(checks=checks, passed=all(checks.values()), metrics=metrics, native_steps=len(times),
                      duration_s=duration, tensor_device=str(device),
                      state_archive={'path': str(output / 'native_states.npz'), 'sha256': digest(output / 'native_states.npz')},
                      limitations=['Hard local attachments prescribe grasp-point motion; not robot pinch/friction proof.',
                                   'Surface thickness and rest/contact offsets are explicit simulation priors, not measured towel calibration.',
                                   'Node-separation checks do not quantify contact forces or prove continuous triangle nonintersection.'])
    except BaseException as error:
        traceback.print_exc()
        report['error'] = {'type': type(error).__name__, 'message': str(error)}
    finally:
        (output / 'qualification.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
        print(json.dumps({'passed': report['passed'], 'output': str(output)}), flush=True)
        if app is not None:
            app.close()
    return 0 if report['passed'] else 1



class ConstraintLedger:
    """Read native-schema constraint attributes; never infer absence from masks.

    Full stage inventories are checked initially and after every render/at end.
    Between them this entrypoint only changes the two registered Xform poses and
    attachmentEnabled attributes; all cached constraints are read every step.
    """

    def __init__(self, stage, cloth_paths, declared):
        self.stage, self.cloth_paths = stage, cloth_paths
        self.declared = declared
        self.constraint_prims = self._inventory()
        self.inventory_checks = 0
        self.verify_inventory()

    def _inventory(self):
        from pxr import UsdPhysics
        return {str(p.GetPath()): p for p in self.stage.Traverse()
                if 'Attachment' in p.GetTypeName() or p.IsA(UsdPhysics.Joint)}

    def actual_attachments(self):
        inverse = {path: name for name, path in self.cloth_paths.items()}
        rows = []
        for row in self.declared:
            prim = self.stage.GetPrimAtPath(row['path'])
            if prim.GetTypeName() != 'OmniPhysicsVtxXformAttachment':
                raise RuntimeError('Registered native vertex attachment is absent')
            cloth = prim.GetRelationship('omniphysics:src0').GetTargets()
            anchor = prim.GetRelationship('omniphysics:src1').GetTargets()
            if len(cloth) != 1 or len(anchor) != 1:
                raise RuntimeError('Native attachment source relationships differ')
            rows.append({'path': str(prim.GetPath()), 'cloth_id': inverse[str(cloth[0])],
                         'vertex_ids': list(prim.GetAttribute('omniphysics:vtxIndicesSrc0').Get()),
                         'anchor_path': str(anchor[0])})
        return rows

    def verify_inventory(self):
        actual = self._inventory()
        if set(actual) != set(self.constraint_prims) or self.actual_attachments() != self.declared:
            raise RuntimeError('Native constraint inventory changed outside the declared controls')
        self.inventory_checks += 1

    def read(self):
        enabled = [bool(self.stage.GetPrimAtPath(row['path']).GetAttribute(
            'omniphysics:attachmentEnabled').Get()) for row in self.declared]
        allowed = {row['path'] for row in self.declared}
        hidden = 0
        for path, prim in self.constraint_prims.items():
            if path in allowed:
                continue
            attr = ('omniphysics:attachmentEnabled' if 'Attachment' in prim.GetTypeName()
                    else 'physics:jointEnabled')
            hidden += int(prim.GetAttribute(attr).Get() is not False)
        for path in self.cloth_paths.values():
            hidden += int(self.stage.GetPrimAtPath(path).GetAttribute('omniphysics:kinematicEnabled').Get() is not False)
        return np.asarray(enabled, dtype=bool), hidden


def native_evidence(stage, views, infos, ledger):
    from pxr import UsdPhysics, PhysxSchema
    from bench.simulation.isaacsim_backend.cloth_sleep import read_surface_sleep_settings
    scene = stage.GetPrimAtPath('/World/PhysicsScene')
    physics = UsdPhysics.Scene(scene)
    physx = PhysxSchema.PhysxSceneAPI(scene)
    direction = np.asarray(physics.GetGravityDirectionAttr().Get(), dtype=float)
    magnitude = float(physics.GetGravityMagnitudeAttr().Get())
    cloths = {}
    for name, info in infos.items():
        prim = stage.GetPrimAtPath(info['path'])
        material = stage.GetPrimAtPath(info['material_path'])
        solver = {f'physxDeformableBody:{key}': prim.GetAttribute(f'physxDeformableBody:{key}').Get()
            for key in ('solverPositionIterationCount', 'linearDamping', 'collisionPairUpdateFrequency',
                        'collisionIterationMultiplier', 'enableSpeculativeCCD', 'settlingThreshold')}
        filter_attr = prim.GetAttribute('physxDeformableBody:selfCollisionFilterDistance')
        if filter_attr.HasAuthoredValueOpinion():
            solver['physxDeformableBody:selfCollisionFilterDistance'] = filter_attr.Get()
        cloths[name] = {'path': info['path'], 'enabled': prim.GetAttribute('omniphysics:deformableBodyEnabled').Get(),
            'schema': 'native_surface_deformable', 'applied_schemas': list(prim.GetAppliedSchemas()),
            'disable_gravity': prim.GetAttribute('physxDeformableBody:disableGravity').Get(),
            'self_collision': prim.GetAttribute('physxDeformableBody:selfCollision').Get(),
            'kinematic_enabled': prim.GetAttribute('omniphysics:kinematicEnabled').Get(),
            'surface_thickness_m': material.GetAttribute('omniphysics:surfaceThickness').Get(),
            'material_attributes': {key: material.GetAttribute(key).Get() for key in info['material']},
            'solver_attributes': solver,
            'usd_sleep_settings': read_surface_sleep_settings(prim),
            'self_collision_filter_mode': 'explicit' if filter_attr.HasAuthoredValueOpinion() else 'runtime_default_unresolved',
            'vertex_count': int(views[name].get_simulation_nodal_positions().shape[1])}
    return {'tensor_device': str(next(iter(views.values())).get_simulation_nodal_positions().device),
        'gpu_dynamics_enabled': physx.GetEnableGPUDynamicsAttr().Get(),
        'broadphase_type': physx.GetBroadphaseTypeAttr().Get(),
        'gravity_m_s2': (direction*magnitude).tolist(), 'cloths': cloths,
        'attachment_inventory': ledger.actual_attachments(),
        'constraint_inventory_prim_paths': sorted(ledger.constraint_prims)}


def run_task(args):
    """Complete native stream in the actual frozen room; scoring is external."""
    initialization = getattr(args, 'initialization', 'tensor-reset')
    if initialization not in INITIALIZATION_MODES:
        raise ValueError('Unknown cloth initialization mode')
    author_only = initialization == 'author-only'
    from scenes.workrooms.laundry_room.release import validate_release
    from bench.simulation.isaacsim_backend.render_sync import startup_arguments
    from scripts.validation.laundry_engineering_motion import build_motion

    source = verify_if_snapshot(PROJECT, required_modules=('scripts.validation.laundry_cloth_interactions',
        'scripts.validation.laundry_engineering_motion', 'scripts.validation.laundry_initialization'), entrypoint_file=Path(__file__))
    if source is None or args.release is None:
        raise ValueError('Formal task needs a frozen source and an immutable --release')
    release = args.release.resolve()
    manifest_path = release/'release_manifest.json'
    manifest = load_json(manifest_path)
    validate_release(release, manifest)
    scene = load_json(release/'scene.json')
    task = load_json(release/'task.json')
    if task['id'] != args.case:
        raise ValueError('Case differs from the selected immutable task release')
    if scene.get('robot', {}).get('enabled', True) is not False:
        raise ValueError('Local cloth engineering recording requires a robot-free engineering release; do not silently omit a released robot')
    # Validate every optional trajectory/duration against frozen task inputs
    # before starting Kit or creating output evidence.
    motion = build_motion(task, scene, fold_align_to_observed_edge=getattr(args, 'fold_edge_alignment', False),
        fold_grip_span_m=getattr(args, 'fold_grip_span_m', None),
        fold_trajectory=getattr(args, 'fold_trajectory', 'arc'),
        fold_peak_height_m=getattr(args, 'fold_peak_height_m', None),
        passive_after_release_s=getattr(args, 'passive_after_release_s', None),
        stack_trajectory=getattr(args, 'stack_trajectory', 'lift'),
        stack_grip_span_m=getattr(args, 'stack_grip_span_m', None),
        stack_grip_layout=getattr(args, 'stack_grip_layout', 'diagonal'),
        garment_profile=getattr(args, 'garment_profile', None))
    dt = float(scene['physics']['dt_s'])
    output = args.output.resolve()
    if not output.is_relative_to(OUTPUT_ROOT):
        raise ValueError('Native evidence belongs under shared datasets')
    output.mkdir(parents=True, exist_ok=False)
    if args.media_output and not args.media_output.resolve().is_relative_to(
            Path(f'{OUTPUT_ROOT}/media/laundry_tasks_20261021')):
        raise ValueError('Task media belongs under the dedicated demo directory')
    binding = lambda path: {'path': str(path), 'sha256': digest(path)}
    report = {'audit_contract': 'laundry_native_stream.v1', 'task_id': task['id'],
        'status': 'initializing', 'complete': False, 'source_snapshot': source,
        'source_snapshot_manifest': binding(PROJECT/'source_snapshot_manifest.json'),
        'release': {'path': str(release), 'manifest_sha256': digest(manifest_path)},
        'scene_input': binding(release/'scene.json'), 'task_input': binding(release/'task.json'),
        'gpu_uuid': os.environ.get('CUDA_VISIBLE_DEVICES'), 'physics_dt_s': dt,
        'robot_present': False, 'model_calls': 0, 'task_success_claimed': False,
        'physics_accepted': False, 'material_calibrated': False,
        'scope': 'Native scripted local grasp-patch operation and passive release; no robot or force-calibration claim.'}
    app = capture = None
    arrays = None
    streams = {}
    exit_code = 1
    try:
        from isaacsim import SimulationApp
        app = SimulationApp({'headless': True, 'width': 1280, 'height': 960,
            'multi_gpu': False, 'active_cuda_gpus': [0], 'max_gpu_count': 1, 'physics_gpu': 0,
            'fast_shutdown': True, 'extra_args': ['--/rtx/verifyDriverVersion/enabled=false', *startup_arguments()]},
            experience=str(PROJECT/'bench/apps/workroom.headless.kit'))
        import omni.usd
        import torch
        from omni.physx import get_physx_simulation_interface
        from isaacsim.core.api import World
        from pxr import Gf, UsdGeom, UsdPhysics
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
        # Laundry v3 is a robot-free static environment. Do not duplicate its table.
        if any(p.IsA(UsdPhysics.Scene) for p in stage.Traverse()):
            raise RuntimeError('Referenced static room unexpectedly contains a PhysicsScene')
        active_bodies = [str(p.GetPath()) for p in stage.Traverse() if p.HasAPI(UsdPhysics.RigidBodyAPI)
                         and UsdPhysics.RigidBodyAPI(p).GetRigidBodyEnabledAttr().Get() is not False]
        if active_bodies:
            raise RuntimeError('Referenced room unexpectedly contains active rigid bodies: '+str(active_bodies))
        table_path = '/World/Geometry/worktop/Collider'
        table = stage.GetPrimAtPath(table_path)
        if not table or not table.HasAPI(UsdPhysics.CollisionAPI) or not UsdPhysics.CollisionAPI(table).GetCollisionEnabledAttr().Get():
            raise RuntimeError('Frozen room worktop collider is missing or disabled')
        report['support_collider'] = {'path': table_path, 'type': table.GetTypeName(),
            'source': scene['workroom']['usd_path'], 'duplicate_table_created': False}
        world = World(stage_units_in_meters=1., physics_dt=dt, rendering_dt=.02,
            backend='torch', device='cuda:0', physics_prim_path='/World/PhysicsScene')
        context = world.get_physics_context()
        context.enable_gpu_dynamics(True)
        context.set_broadphase_type('GPU')
        context.set_solver_type('TGS')
        context.set_gravity(float(scene['physics']['gravity_m_s2'][2]))
        infos = {spec['id']: add_named_cloth(stage, spec, scene['isaacsim']) for spec in scene['cloths']}
        specs = {spec['id']: spec for spec in scene['cloths']}
        paths = {name: info['path'] for name, info in infos.items()}
        initial = np.asarray(specs['towel']['initial_vertices_m'], dtype=np.float32)
        declared, attachments, anchor_ops = [], [], []
        for index, ids in enumerate(motion.patches):
            ids = [int(i) for i in ids]
            anchor_path = '/World/EngineeringGrasp/anchor_'+str(index)
            anchor = UsdGeom.Xform.Define(stage, anchor_path)
            center = np.asarray(motion.origins[index], dtype=float)
            translate = anchor.AddTranslateOp()
            rotate = anchor.AddOrientOp(UsdGeom.XformOp.PrecisionDouble)
            translate.Set(Gf.Vec3d(*map(float, center)))
            rotate.Set(Gf.Quatd(1., Gf.Vec3d(0., 0., 0.)))
            anchor_ops.append((translate, rotate))
            path = '/World/EngineeringGrasp/attachment_'+str(index)
            attachments.append(attachment(stage, path, paths['towel'], anchor_path, ids, initial[ids]-center))
            declared.append({'path': path, 'cloth_id': 'towel', 'vertex_ids': ids, 'anchor_path': anchor_path})
        ledger = ConstraintLedger(stage, paths, declared)
        report['engineering_boundary'] = {'mode': 'native_plain_xform_attachment',
            'attachments': declared, 'native_attachment_paths': [row['path'] for row in declared],
            'nodal_write_steps': 0, 'post_release_nodal_write_steps': 0,
            'cloth_position_writes_after_reset': False, 'cloth_velocity_writes_after_reset': False,
            'whole_cloth_animation': False, 'anchor_collision_shapes': 0,
            'release_time_s': motion.release_time_s, 'motion': motion.metadata,
            'release_method': 'Set every native attachmentEnabled=false then flush_changes before stepping',
            'anchor_motion_api': 'Plain nonrigid Xform translate/orient plus IPhysxSimulation.flush_changes',
            'constraint_inventory_method': 'Initial full stage and every render/end full stage inventory; actual cached constraint/kinematic flags every native step; only declared anchor/attachment attributes mutate between inventory checks.'}
        stage.Export(str(output/'initial.usda'))
        reset_trace = {}
        if author_only:
            # Read the exact exported USD geometry, not a copy of the task spec.
            from pxr import Usd
            asset_specs = {row['id']: row for row in task['asset_metadata']['cloths']}
            authored_stage = Usd.Stage.Open(str(output/'initial.usda'))
            reset_trace.update(authored_arrays(authored_stage, asset_specs))
            warmup_masks = authored_control_masks(authored_stage, asset_specs, declared)
            reset_trace['world_clock_tick_s'] = np.asarray(world_clock_tick(dt), dtype=np.float64)
            reset_trace['world_step_before_reset'] = np.asarray(world.current_time_step_index)
            reset_trace['world_time_before_reset'] = np.asarray(world.current_time)
            report['initialization'] = {'schema': INITIALIZATION_SCHEMA, 'mode': 'author-only',
                'authored_stage': binding(output/'initial.usda'),
                'warmup_policy': 'isaacsim_6_1_world_reset_two_steps',
                'world_clock_tick_s': world_clock_tick(dt),
                'intermediate_warmup_states_recorded': False, 'warmup_operation_credit': False}
        world.reset()
        views = {name: CountedSurfaceView(world.physics_sim_view.create_surface_deformable_body_view(path))
                 for name, path in paths.items()}
        if any(view.count != 1 for view in views.values()):
            raise RuntimeError('Named native cloth body inventory differs')
        # Preserve the native warmup endpoint before any optional legacy setter.
        reset_trace.update(world_step_before_tensor_reset=np.asarray(world.current_time_step_index),
                           world_time_before_tensor_reset=np.asarray(world.current_time))
        for name, view in views.items():
            reset_trace[name+'_before_q'] = array(view.get_simulation_nodal_positions())[0]
            reset_trace[name+'_before_qd'] = array(view.get_simulation_nodal_velocities())[0]
        if author_only:
            reset_trace['world_step_after_warmup'] = np.asarray(world.current_time_step_index)
            reset_trace['world_time_after_warmup'] = np.asarray(world.current_time)
            reset_trace['warmup_anchor_targets'] = anchor_targets(stage, declared)
            enabled, hidden = ledger.read()
            if hidden or not all(enabled):
                raise RuntimeError('Warmup attachment inventory changed')
            for name in views:
                reset_trace[name+'_warmup_q'] = reset_trace[name+'_before_q'].copy()
                reset_trace[name+'_warmup_qd'] = reset_trace[name+'_before_qd'].copy()
                reset_trace[name+'_warmup_control_mask'] = warmup_masks[name]
        else:
            # Historical canonical tensor reset remains the default behavior.
            for name, view in views.items():
                device = view.get_simulation_nodal_positions().device
                index = torch.tensor([0], dtype=torch.int32, device=device)
                q = torch.as_tensor(np.asarray(specs[name]['initial_vertices_m'], np.float32)[None], device=device)
                view.set_simulation_nodal_positions(q, index)
                view.set_simulation_nodal_velocities(torch.zeros_like(q), index)
                reset_trace[name+'_canonical_q'] = array(view.get_simulation_nodal_positions())[0]
                reset_trace[name+'_canonical_qd'] = array(view.get_simulation_nodal_velocities())[0]
        report['reset_position_velocity_calls_by_cloth'] = {name: view.setter_calls.copy() for name, view in views.items()}
        report['native_evidence'] = native_evidence(stage, views, infos, ledger)
        for name, observed in report['native_evidence']['cloths'].items():
            expected_thickness = float(specs[name].get('isaacsim', {}).get(
                'cloth_thickness_m', scene['isaacsim']['cloth_thickness_m']))
            if (observed['enabled'] is not True or observed['disable_gravity'] is not False
                    or observed['self_collision'] is not True or observed['kinematic_enabled'] is not False
                    or not np.isfinite(expected_thickness) or expected_thickness <= 0
                    or abs(observed['surface_thickness_m']-expected_thickness) > 1.e-8):
                raise RuntimeError('Actual native cloth schema flags/thickness differ: '+name+' '+str(observed))
        if (report['native_evidence']['gpu_dynamics_enabled'] is not True
                or report['native_evidence']['broadphase_type'] != 'GPU'
                or not np.allclose(report['native_evidence']['gravity_m_s2'], scene['physics']['gravity_m_s2'], atol=1.e-6)
                or abs(world.get_physics_dt()-dt) > 1.e-12):
            raise RuntimeError('Actual native GPU/gravity/time step differs from the release')
        interface = get_physx_simulation_interface()
        if args.media_output:
            from scripts.media.laundry_cloth_frames import LaundryClothFrameCapture, task_cameras
            capture = LaundryClothFrameCapture(stage=stage, world=world, app=app, cloth_views=views,
                cloth_specs=scene['cloths'], triangles={name: info['triangles'] for name, info in infos.items()},
                cloth_paths=paths, output=args.media_output.resolve(),
                cameras=task_cameras(task['asset_metadata']['target_center_xy_m']))
        streams = {name+suffix: [] for name in views for suffix in ('_q', '_qd', '_control_mask')}
        for key in ('time_s', 'native_step', 'position_write_calls', 'velocity_write_calls',
                    'hidden_constraints', 'attachment_enabled', 'anchor_targets'):
            streams[key] = []
        if author_only:
            streams['absolute_world_step'], streams['absolute_world_time_s'] = [], []
        start_native_step = int(world.current_time_step_index)
        start_native_time = float(world.current_time)
        report['native_world_reset_offset'] = {'absolute_step': start_native_step,
            'absolute_time_s': start_native_time,
            'stream_time_basis': ('Native warmup endpoint is relative frame/step/time zero; two internal setup steps receive no operation credit.' if author_only else
                'Canonical tensor reset is relative frame/step/time zero; media records the unchanged absolute World counters.')}
        if author_only:
            reset_trace['world_step_at_baseline'] = np.asarray(world.current_time_step_index)
            reset_trace['world_time_at_baseline'] = np.asarray(world.current_time)
            reset_trace['baseline_anchor_targets'] = anchor_targets(stage, declared)
            if start_native_step != 2 or abs(start_native_time-2*world_clock_tick(dt)) > 1.e-8:
                raise RuntimeError('Unexpected native World warmup count; author-only requires exactly two setup steps')
            for name, view in views.items():
                reset_trace[name+'_setter_calls'] = np.asarray(view.setter_calls, dtype=np.int64)
                for suffix, getter in (('_q', view.get_simulation_nodal_positions), ('_qd', view.get_simulation_nodal_velocities)):
                    reset_trace[name+'_baseline'+suffix] = array(getter())[0]
                    if not same_bits(reset_trace[name+'_baseline'+suffix], reset_trace[name+'_warmup'+suffix]):
                        raise RuntimeError('Native cloth changed between warmup and operation baseline')
        np.savez_compressed(output/'native_reset_trace.npz', **reset_trace)
        report['initialization_trace'] = binding(output/'native_reset_trace.npz')
        if author_only:
            report['initialization']['trace'] = report['initialization_trace']
        last_setter_calls = np.sum([view.setter_calls for view in views.values()], axis=0)

        def record(step, targets):
            nonlocal last_setter_calls
            current_calls = np.sum([view.setter_calls for view in views.values()], axis=0)
            step_calls = current_calls-last_setter_calls
            last_setter_calls = current_calls.copy()
            enabled, hidden = ledger.read()
            masks = {name: np.zeros(len(specs[name]['initial_vertices_m']), dtype=bool) for name in views}
            for active, row in zip(enabled, declared):
                if active:
                    masks[row['cloth_id']][row['vertex_ids']] = True
            for name, view in views.items():
                streams[name+'_q'].append(array(view.get_simulation_nodal_positions())[0])
                streams[name+'_qd'].append(array(view.get_simulation_nodal_velocities())[0])
                streams[name+'_control_mask'].append(masks[name])
            for key, value in {'time_s': step*dt, 'native_step': step, 'position_write_calls': int(step_calls[0]),
                'velocity_write_calls': int(step_calls[1]), 'hidden_constraints': hidden, 'attachment_enabled': enabled,
                'anchor_targets': targets}.items():
                streams[key].append(value)
            if author_only:
                streams['absolute_world_step'].append(int(world.current_time_step_index))
                streams['absolute_world_time_s'].append(float(world.current_time))
            if hidden or step_calls.any():
                raise RuntimeError('Undeclared native cloth constraint or state setter detected')

        targets = [np.r_[np.asarray(motion.origins[i]), [1., 0., 0., 0.]] for i in range(len(declared))]
        record(0, targets)
        if author_only:
            for name in views:
                for suffix in ('_q', '_qd'):
                    if not same_bits(streams[name+suffix][0], reset_trace[name+'_baseline'+suffix]):
                        raise RuntimeError('Frame zero differs from the bound native warmup baseline')
        if capture:
            capture.capture('step_000000', camera_ids=('task',))
            ledger.verify_inventory()
        released = False
        frame_stride = max(1, round(1./(args.media_fps*dt)))
        for step in range(1, round(motion.duration_s/dt)+1):
            time = (step-1)*dt
            if time >= motion.release_time_s and not released:
                for prim in attachments:
                    prim.GetAttribute('omniphysics:attachmentEnabled').Set(False)
                released = True
            motion.observe_placement_reference(time, streams['towel_q'][-1])
            targets = []
            for index, (translate, rotate) in enumerate(anchor_ops):
                center, quaternion = motion.pose(index, time)
                translate.Set(Gf.Vec3d(*map(float, center)))
                rotate.Set(Gf.Quatd(float(quaternion[0]), Gf.Vec3d(*map(float, quaternion[1:]))))
                targets.append(np.r_[center, quaternion])
            interface.flush_changes()
            world.step(render=False)
            if int(world.current_time_step_index)-start_native_step != step:
                raise RuntimeError('Native World step counter is not consecutive')
            record(step, targets)
            if capture and step % frame_stride == 0:
                capture.capture('step_'+str(step).zfill(6), camera_ids=('task',))
                ledger.verify_inventory()
            if step % 2000 == 0:
                print(json.dumps({'event': 'native_task_progress', 'task': task['id'], 'native_step': step,
                    'time_s': step*dt, 'released': released}), flush=True)
        ledger.verify_inventory()
        if capture:
            capture.capture('final', camera_ids=('room', 'task', 'thickness'))
            ledger.verify_inventory()
            report['video'] = capture.encode_video(camera_id='task', fps=args.media_fps)
        report['native_evidence_final'] = native_evidence(stage, views, infos, ledger)
        report['native_elapsed_s'] = float(world.current_time)-start_native_time
        report['constraint_inventory_checks'] = ledger.inventory_checks
        report['source_snapshot_after_run'] = verify_if_snapshot(PROJECT,
            required_modules=('scripts.validation.laundry_cloth_interactions',
                'scripts.validation.laundry_engineering_motion',
                'scripts.validation.laundry_initialization',
                'scenes.workrooms.laundry_room.release'), entrypoint_file=Path(__file__))
        report['complete'] = True
        report['status'] = 'native_stream_recorded_audit_pending'
        stage.Export(str(output/'final.usda'))
        exit_code = 0
    except BaseException as error:
        traceback.print_exc()
        report['status'] = 'failed'
        report['error'] = {'type': type(error).__name__, 'message': str(error), 'traceback': traceback.format_exc()}
    finally:
        if 'views' in locals():
            report['final_position_velocity_calls_by_cloth'] = {name: view.setter_calls.copy() for name, view in views.items()}
        if streams and streams.get('time_s'):
            arrays = {key: np.asarray(value) for key, value in streams.items()}
            for name, info in infos.items():
                arrays[name+'_triangles'] = np.asarray(info['triangles'], dtype=np.int32)
            np.savez_compressed(output/'native_states.npz', **arrays)
            report['state_archive'] = binding(output/'native_states.npz')
            report['native_steps'] = len(arrays['time_s'])-1
            report['duration_s'] = float(arrays['time_s'][-1])
            report['finite_states'] = all(np.isfinite(a).all() for a in arrays.values())
            tail = arrays['time_s'] >= arrays['time_s'][-1]-.5
            report['final_500ms_max_speed_m_s'] = {name: float(np.linalg.norm(arrays[name+'_qd'][tail], axis=-1).max()) for name in infos}
        if capture is not None:
            capture.close()
        report['limitations'] = ['Engineering hard attachments are not robot grasp or frictional pinch evidence.',
            'Full-stream frozen evaluation and finite-thickness geometric audit are separate; this producer does not grant task success.',
            'Material values are uncalibrated priors; no contact force or continuous-time collision certificate is supplied.']
        if author_only:
            report['limitations'].append('Two internal World.reset warmup steps are setup, not operation credit; intermediate warmup states are not recorded or continuously certified.')
        (output/'native_report.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
        print(json.dumps({'event': 'native_task_finished', 'status': report['status'], 'output': str(output)}), flush=True)
        if app is not None:
            app.close(exit_code=exit_code)
    return exit_code


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', choices=('lift_release', 'two_layer', *FORMAL_TASKS), required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--release', type=Path)
    parser.add_argument('--media-output', type=Path)
    parser.add_argument('--media-fps', type=int, default=8)
    parser.add_argument('--dt', type=float, default=.0005)
    parser.add_argument('--young', type=float, default=6.67e5)
    parser.add_argument('--bend', type=float, default=1000.)
    parser.add_argument('--control-mode', choices=('nodal', 'attachment', 'plain_xform'), default='plain_xform')
    parser.add_argument('--initialization', choices=INITIALIZATION_MODES, default='tensor-reset',
        help='Formal tasks: preserve the legacy canonical tensor reset, or bind the exact two-step native warmup endpoint without cloth setters.')
    parser.add_argument('--fold-edge-alignment', action='store_true')
    parser.add_argument('--fold-grip-span-m', nargs=2, type=float)
    parser.add_argument('--fold-trajectory', choices=('arc', 'bezier'), default='arc')
    parser.add_argument('--stack-trajectory', choices=('lift', 'low-clearance'), default='lift')
    parser.add_argument('--stack-grip-span-m', nargs=2, type=float,
        help='Stack only: material X/Y grasp spans in meters; default keeps historical node indices.')
    parser.add_argument('--stack-grip-layout', choices=('diagonal', 'crease-edge'), default='diagonal',
        help='Stack only: optional leading crease corners; requires explicit material grasp spans.')
    parser.add_argument('--garment-profile', choices=('folded_garment_leading_patches_v1',),
        help='Explicit local multilayer grasp profile for folded shirt/trousers engineering demos only.')
    parser.add_argument('--fold-peak-height-m', type=float,
        help='Only for bezier: nominal peak rise before terminal gap; default 0.16 m.')
    parser.add_argument('--passive-after-release-s', type=float,
        help='Formal tasks: explicit passive simulation duration after release; default unchanged at 6 s.')
    args = parser.parse_args()
    if args.passive_after_release_s is not None and (not np.isfinite(args.passive_after_release_s) or args.passive_after_release_s <= 0):
        parser.error('--passive-after-release-s must be finite and positive')
    if args.fold_peak_height_m is not None and (not np.isfinite(args.fold_peak_height_m) or args.fold_peak_height_m <= 0):
        parser.error('--fold-peak-height-m must be finite and positive')
    if args.fold_peak_height_m is not None and args.fold_trajectory != 'bezier':
        parser.error('--fold-peak-height-m requires --fold-trajectory bezier')
    if args.stack_trajectory != 'lift' and args.case != 'towel_stacking':
        parser.error('--stack-trajectory low-clearance requires towel_stacking')
    if args.stack_grip_span_m is not None and args.case != 'towel_stacking':
        parser.error('--stack-grip-span-m requires towel_stacking')
    if args.stack_grip_layout != 'diagonal' and (args.case != 'towel_stacking' or args.stack_grip_span_m is None):
        parser.error('--stack-grip-layout crease-edge requires towel_stacking and --stack-grip-span-m')
    if args.case != 'towel_folding' and args.fold_trajectory != 'arc':
        parser.error('--fold-trajectory bezier requires towel_folding')
    if args.garment_profile is not None and args.case not in ('shirt_stacking', 'trousers_stacking'):
        parser.error('--garment-profile requires a folded garment task')
    if args.case in ('shirt_stacking', 'trousers_stacking') and args.garment_profile is None:
        parser.error('Folded garment demos require an explicit --garment-profile')
    if args.case not in FORMAL_TASKS and (args.initialization != 'tensor-reset' or args.fold_edge_alignment or args.fold_grip_span_m is not None or args.passive_after_release_s is not None):
        parser.error('Initialization and motion options require a formal cloth task')
    raise SystemExit(run_task(args) if args.case in FORMAL_TASKS else run(args))


if __name__ == '__main__':
    main()
