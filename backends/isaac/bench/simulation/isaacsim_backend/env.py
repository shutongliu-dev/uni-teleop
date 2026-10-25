"""Native Isaac Sim environment: USD stage, PhysX dynamics, RTX rendering.

SimulationApp owns the Kit lifecycle. Only joint drive targets are accepted
during step; all direct state writes are confined to reset. Ground-truth
observations intentionally match the engine-independent SequenceEvaluator.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import traceback

import numpy as np

from .cameras import LEGACY_CAMERAS, aimed_link_matrix, link_camera_matrix, look_at_matrix, resolve_cameras
from .contact_precision import aggregation_metadata, summarize_contact_groups
from .robot_registry import hand_actuator_values
from bench.paths import DATA_ROOT, load_json


# Retained so older importers (`from .env import CAMERAS`) keep working; the
# active camera set is resolved per scene and stored on the env instance.
CAMERAS = LEGACY_CAMERAS


def _np(value):
    if hasattr(value, 'detach'):
        value = value.detach().cpu()
    if hasattr(value, 'numpy'):
        value = value.numpy()
    return np.asarray(value).copy()


def physics_solver_type(options):
    """Select a release-owned solver; omission preserves the existing TGS stage."""
    selected = options.get('solver_type', 'TGS')
    if not isinstance(selected, str) or selected not in ('TGS', 'PGS'):
        raise ValueError('isaacsim.solver_type must be TGS or PGS')
    return selected


def physics_external_forces_every_iteration(options):
    """Validate an explicit TGS force-integration opt-in before Kit starts."""
    key = 'enable_external_forces_every_iteration'
    if key not in options:
        return None
    selected = options[key]
    if type(selected) is not bool:
        raise ValueError(f'isaacsim.{key} must be a bool')
    if selected and physics_solver_type(options) != 'TGS':
        raise ValueError(f'isaacsim.{key}=True requires solver_type TGS')
    return selected


def gpu_tensor_pipeline(sim_view, device):
    """Compatibility guard for the default CUDA state transport profile."""
    from .readback_profile import tensor_pipeline
    return tensor_pipeline(sim_view, device)


def _rotation(q):
    w, x, y, z = np.asarray(q, dtype=float) / np.linalg.norm(q)
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


class FR3DexterousIsaacEnv:
    def __init__(self, scene_spec, *, device='cuda:0', headless=True,
                 render_width=1280, render_height=720):
        self.scene_spec = deepcopy(scene_spec)
        self.device = str(device)
        from .native_task_bundle import task_observer_enabled
        self.task_observer_enabled = task_observer_enabled(self.scene_spec)
        from .readback_profile import CPU_RIGID, resolve_profile
        self.readback_profile_resolution = resolve_profile(self.scene_spec)
        self.native_readback_profile = self.readback_profile_resolution['profile']
        self.physics_device = 'cpu' if self.native_readback_profile == CPU_RIGID else self.device
        self.tensor_device = self.physics_device
        if not self.device.startswith('cuda:'):
            raise ValueError('Select a CUDA device for native RTX rendering')
        self.render_width, self.render_height = int(render_width), int(render_height)
        # Scene-supplied cameras; a scene without a `cameras` block resolves to
        # the legacy overview/task pair, authored and rendered as before.
        self.cameras = resolve_cameras(self.scene_spec)
        self.camera_ids = list(self.cameras)
        self._link_camera_paths = {}
        self._cameras_authored = False
        self.camera_paths = {}
        self.options = self.scene_spec.setdefault('isaacsim', {})
        if (self.options.get('arm_motion_policy') and self.task_observer_enabled
                and self.scene_spec.get('unscored_practice') is not True):
            raise ValueError('Arm motion overrides require disabled task scoring')
        self.solver_type = physics_solver_type(self.options)
        self.enable_external_forces_every_iteration = physics_external_forces_every_iteration(self.options)
        self.dt = float(self.options.get('physics_dt_s', self.scene_spec['physics']['dt_s']))
        if self.dt <= 0 or abs(.02 / self.dt - round(.02 / self.dt)) > 1e-8:
            raise ValueError('physics_dt_s must divide 20 ms control period')
        robot = self.scene_spec['robot']
        from .robot_registry import get_robot_spec
        # A scene without a pinned assembly uses the hand's v1-era assembly.
        self.hand_spec = get_robot_spec(robot)
        self.mimic_joints = self.hand_spec['mimic_joints']
        if robot.get('mount_position_m', self.hand_spec['mount_translation_m']) != self.hand_spec['mount_translation_m'] or robot.get('mount_quaternion_wxyz', self.hand_spec['mount_quaternion_wxyz']) != self.hand_spec['mount_quaternion_wxyz']:
            raise ValueError('Mount must match the selected versioned hand assembly')
        gpu = int(self.device.split(':')[1])
        # Opt-in single-device mode: CUDA_VISIBLE_DEVICES names one GPU and the
        # renderer is chosen by CUDA index, so rendering and physics share that
        # device and no context opens on the host's other (shared) GPUs.
        self.single_cuda_device = os.environ.get('ISAAC_BENCH_SINGLE_CUDA_DEVICE') == '1'
        visible = os.environ.get('CUDA_VISIBLE_DEVICES', '')
        if self.single_cuda_device and (gpu != 0 or not visible or ',' in visible):
            raise ValueError('Single-device mode needs CUDA_VISIBLE_DEVICES with one GPU and device cuda:0')
        gpu_selection = ({'active_cuda_gpus': [gpu], 'max_gpu_count': 1} if self.single_cuda_device
                         else {'active_gpu': gpu})
        self.app = None
        self.world = None
        self._closed = False
        self._render_products = {}
        self._contact_subscription = None
        self._contact_reports = []
        self._contact_errors = []
        self._current_step_time = 0.
        self.experience_path = Path(os.environ.get('ISAAC_BENCH_EXPERIENCE',
            str(Path(__file__).resolve().parents[2] / 'apps/fr3_sharpa.headless.kit')))
        if not self.experience_path.is_file():
            raise FileNotFoundError(f'Isaac bench experience is missing: {self.experience_path}')
        self.allow_unvalidated_driver = os.environ.get('ISAAC_ALLOW_UNVALIDATED_DRIVER') == '1'
        from .render_sync import startup_arguments
        extra_args = startup_arguments()
        if self.allow_unvalidated_driver:
            extra_args.append('--/rtx/verifyDriverVersion/enabled=false')
        # All Isaac / omni / pxr imports must happen after SimulationApp starts.
        from isaacsim import SimulationApp
        self.app = SimulationApp({'headless': bool(headless), 'width': self.render_width,
                                  'height': self.render_height, 'multi_gpu': False,
                                  **gpu_selection, 'physics_gpu': gpu,
                                  'renderer': 'RayTracedLighting', 'fast_shutdown': True,
                                  'extra_args': extra_args}, experience=str(self.experience_path))
        try:
            self._initialize()
        except BaseException:
            self.close()
            raise

    def _initialize(self):
        import carb
        import omni.usd
        import omni.kit.app
        manager = omni.kit.app.get_app().get_extension_manager()
        manager.set_extension_enabled_immediate('isaacsim.core.api', True)
        from isaacsim.core.api import World
        from pxr import PhysxSchema
        from .robot_asset import import_robot
        from .scene import build_scene

        settings = carb.settings.get_settings()
        if self.physics_device == 'cpu':
            settings.set_int('/persistent/physics/numThreads', 0)
        # Keep native solver -> USD synchronization enabled for real rendering.
        settings.set_bool('/physics/updateToUsd', True)
        settings.set_bool('/physics/updateVelocitiesToUsd', True)
        settings.set_bool('/physics/updateParticlesToUsd', True)
        from .render_sync import verify_settings
        self.render_sync = verify_settings(settings)
        assets_root = self.scene_spec['robot'].get('assets_root', f'{DATA_ROOT}/assets')
        self.robot_usd, self.robot_source = import_robot(assets_root, hand_id=self.hand_spec['hand_id'],
                                                         assembly_version=self.scene_spec['robot'].get('assembly_version'),
                                                         instances=self.scene_spec['robot'].get('instances'))
        omni.usd.get_context().new_stage()
        self.stage = omni.usd.get_context().get_stage()
        self.world = World(stage_units_in_meters=1., physics_dt=self.dt, rendering_dt=.02,
                           backend='torch', device=self.physics_device)
        context = self.world.get_physics_context()
        context.enable_gpu_dynamics(self.physics_device != 'cpu')
        context.set_broadphase_type('MBP' if self.physics_device == 'cpu' else 'GPU')
        context.set_solver_type(self.solver_type)
        physics_prim = self.stage.GetPrimAtPath(context.prim_path)
        api = PhysxSchema.PhysxSceneAPI.Apply(physics_prim)
        self._configure_external_forces_every_iteration(api)
        # Swept CCD is incompatible with the GPU dynamics pipeline. Rigid
        # bodies use speculative CCD and the recorded small integration step.
        api.CreateEnableCCDAttr(False)
        api.CreateGpuMaxRigidContactCountAttr(2**20)
        api.CreateGpuMaxRigidPatchCountAttr(2**18)
        api.CreateGpuFoundLostAggregatePairsCapacityAttr(2**14)
        api.CreateGpuTotalAggregatePairsCapacityAttr(2**18)
        self.scene_info = build_scene(self.stage, self.scene_spec, self.robot_usd,
                                      physics_scene_path=context.prim_path)
        from .arm_motion import configure_arm_motion
        self.unrestricted_arm_joints = configure_arm_motion(
            self.stage, self.scene_info['joint_paths'], self.options.get('arm_motion_policy'))
        self._task_readback = None
        if self.scene_info.get('task_layers'):
            from .task_readback import TaskLayerReadback
            self._task_readback = TaskLayerReadback(self.stage, self.scene_info['task_layers'])
        self._configure_task_readback()
        self.has_cloth = self.scene_info['cloth'] is not None
        self._task_reader = None
        if self.scene_info.get('native_task_bundle', {}).get('particle_sets'):
            # Particle isosurfaces are registered by deferred Kit updates.
            self.app.update()
            self.app.update()
        self.world.reset()
        self._configure_task_readback()
        self._initialize_views()
        # A requested pad compliance must be live on exactly the pad colliders (the
        # legacy link binding silently resolved pads to robot_shell; runs/material_binding_01).
        from .robot_materials import runtime_material_check
        self.runtime_robot_material_check = runtime_material_check(
            self.stage, self._sim_view, self._robot_view, self.scene_info)
        self._reset_count = 0
        self.reset()

    def _configure_external_forces_every_iteration(self, api):
        # Omission leaves even the authored USD opinions unchanged for old releases.
        if self.enable_external_forces_every_iteration is not None:
            api.CreateEnableExternalForcesEveryIterationAttr(self.enable_external_forces_every_iteration)

    def _configure_task_readback(self):
        from .readback_profile import GPU_SOLVER_USD_HOST
        if self.native_readback_profile == GPU_SOLVER_USD_HOST:
            import carb
            from .task_readback import configure_particle_export
            configure_particle_export(carb.settings.get_settings())


    def _initialize_views(self):
        # 6.1 initializes a view bound to the explicit stage id. Reuse that
        # native PhysX view: the legacy stage_id=-1 default is not attached.
        # GPU PhysX requires torch/warp; the NumPy frontend is CPU-only.
        self._sim_view = self.world.physics_sim_view
        if self._sim_view is None:
            raise RuntimeError('World did not initialize its stage-bound PhysX tensor view')
        from .readback_profile import tensor_pipeline, validate_gpu_solver_configuration
        self.runtime_physics_tensor_pipeline = tensor_pipeline(
            self._sim_view, self.device, self.native_readback_profile)
        self.tensor_device = str(self._sim_view.device)
        import omni.physx
        context = self.world.get_physics_context()
        prim = self.stage.GetPrimAtPath(context.prim_path)
        self.runtime_readback_configuration = validate_gpu_solver_configuration({
            'profile': self.native_readback_profile,
            'physics_scene_path': str(context.prim_path),
            'gpu_dynamics_enabled': prim.GetAttribute('physxScene:enableGPUDynamics').Get(),
            'broadphase_type': str(prim.GetAttribute('physxScene:broadphaseType').Get()),
            'solver_type': str(prim.GetAttribute('physxScene:solverType').Get()),
            'physics_dt_s': self.dt,
            'native_gpu_override': int(omni.physx.get_physx_interface().get_overwrite_gpu_setting()),
            'actual_readback_suppressed': bool(omni.physx.get_physx_interface().is_readback_suppressed()),
        }, self.native_readback_profile)
        self._native_tensor_device = self.tensor_device
        self._sim_view.set_subspace_roots('/')
        paths = self.scene_info['articulation_path']
        if isinstance(paths, list):
            # Bimanual: one logical robot over one native articulation per arm.
            from .composite_view import CompositeArticulationView
            self._robot_view = CompositeArticulationView([self._sim_view.create_articulation_view(p) for p in paths])
        else:
            self._robot_view = self._sim_view.create_articulation_view(paths)
        if self._robot_view.count != 1:
            raise RuntimeError('PhysX did not create the single robot articulation')
        from .robot_base import runtime_base_check
        self.runtime_robot_base_check = runtime_base_check(self._robot_view, self.scene_info)
        self.joint_names = list(self._robot_view.shared_metatype.dof_names)
        self.body_names = list(self._robot_view.shared_metatype.link_names)
        self.actuated_joint_names = [n for n in self.joint_names if n not in self.mimic_joints]
        if len(self.joint_names) != self.hand_spec['joint_count'] or set(self.joint_names) != set(self.robot_source['joint_limits']):
            raise RuntimeError(f'PhysX joint inventory mismatch: {self.joint_names}')
        self.joint_limits = {n: [self.robot_source['joint_limits'][n]['lower'],
                                self.robot_source['joint_limits'][n]['upper']] for n in self.joint_names}
        from .arm_motion import FLOAT32_MAX
        for name in self.unrestricted_arm_joints:
            # Native positions/targets use float32. Keep public JSON finite while
            # accepting the full representable range, not the original URDF stops.
            self.joint_limits[name] = [-FLOAT32_MAX, FLOAT32_MAX]
        robot_spec = self.scene_spec['robot']
        if 'arms' in self.hand_spec:
            arm = {f'fr3_{side}_joint{i}': value for side in self.hand_spec['arms']
                   for i, value in enumerate(robot_spec.get('arm_home_rad_by_side', self.hand_spec['arm_home_rad_by_side'])[side], start=1)}
        else:
            arm = dict(zip([f'fr3_joint{i}' for i in range(1, 8)], robot_spec['arm_home_rad']))
        arm.update(self.hand_spec.get('home_joint_positions_rad', {}))
        self.home = {n: float(arm.get(n, np.clip(0., *self.joint_limits[n]))) for n in self.joint_names}
        self._one = self._tensor(np.array([0], dtype=np.int32))
        self._targets = np.asarray([[self.home[n] for n in self.joint_names]], dtype=np.float32)
        config = self.scene_spec['robot']
        # Release physics options opt into calibrated hand actuators and arm
        # gravity compensation; absent keys keep the frozen legacy dynamics.
        model = self.options.get('hand_actuator_model')
        actuator = {} if model is None else hand_actuator_values(self.hand_spec['hand_id'], model, self.joint_names)
        hand = lambda name, key, legacy: actuator.get(name, {}).get(key, legacy)
        stiffness = [[0. if n in self.mimic_joints else config['arm_kp'] if n.startswith('fr3_') else hand(n, 'kp', config['hand_kp']) for n in self.joint_names]]
        damping = [[0. if n in self.mimic_joints else config['arm_kd'] if n.startswith('fr3_') else hand(n, 'kd', config['hand_kd']) for n in self.joint_names]]
        efforts = [[self.robot_source['joint_limits'][n]['effort'] for n in self.joint_names]]
        # PhysX dynamic state is on CUDA, while articulation parameters are on
        # the view's parameter device (CPU in this native runtime).
        param_device = self._robot_view.get_dof_stiffnesses().device
        param_one = self._tensor(np.array([0], dtype=np.int32), param_device)
        self._robot_view.set_dof_stiffnesses(self._tensor(np.asarray(stiffness, dtype=np.float32), param_device), param_one)
        self._robot_view.set_dof_dampings(self._tensor(np.asarray(damping, dtype=np.float32), param_device), param_one)
        self._robot_view.set_dof_max_forces(self._tensor(np.asarray(efforts, dtype=np.float32), param_device), param_one)
        if any('armature' in row for row in actuator.values()):
            armature = [[hand(n, 'armature', 0.) for n in self.joint_names]]
            self._robot_view.set_dof_armatures(self._tensor(np.asarray(armature, dtype=np.float32), param_device), param_one)
        if any('joint_friction' in row for row in actuator.values()):
            friction = [[hand(n, 'joint_friction', 0.) for n in self.joint_names]]
            self._robot_view.set_dof_friction_coefficients(self._tensor(np.asarray(friction, dtype=np.float32), param_device), param_one)
        if actuator:
            # The derived USD carries only importer velocity limits; enforce the URDF caps natively.
            velocities = [[self.robot_source['joint_limits'][n]['velocity'] for n in self.joint_names]]
            self._robot_view.set_dof_max_velocities(self._tensor(np.asarray(velocities, dtype=np.float32), param_device), param_one)
        self.runtime_arm_motion = {'policy': self.options.get('arm_motion_policy', 'urdf'),
                                   'unrestricted_joints': list(self.unrestricted_arm_joints)}
        if self.unrestricted_arm_joints:
            velocities = self._robot_view.get_dof_max_velocities().clone()
            indices = [self.joint_names.index(n) for n in self.unrestricted_arm_joints]
            velocities[:, indices] = FLOAT32_MAX
            self._robot_view.set_dof_max_velocities(velocities, param_one)
            # Verify physical limits, rather than merely skipping action validation.
            limits = _np(self._robot_view.get_dof_limits())[0, indices]
            speeds = _np(self._robot_view.get_dof_max_velocities())[0, indices]
            if not (np.all(limits[:, 0] < -1e6) and np.all(limits[:, 1] > 1e6)
                    and np.all(speeds > 1e30)):
                raise RuntimeError('Native simulated arm motion limits were not removed')
            self.runtime_arm_motion['native_verified'] = True
        gravity = self.options.get('arm_gravity_compensation')
        if gravity not in (None, 'physx_feedforward_v1'):
            raise ValueError(f'Unknown arm gravity compensation: {gravity!r}')
        # FR3 compensates gravity and payload internally; apply PhysX's
        # gravity-compensation torques to the seven arm joints every substep.
        self._arm_gravity_mask = None if gravity is None else self._tensor(np.asarray(
            [[1. if n.startswith('fr3_') else 0. for n in self.joint_names]], dtype=np.float32))
        if self.hand_spec.get('hand_armature_kg_m2', 0.):
            # Explicit embodiment assumption: preserves link inertias and passive
            # mimic drives while conditioning their native articulated dynamics.
            armature = [[0. if n.startswith('fr3_') else self.hand_spec['hand_armature_kg_m2'] for n in self.joint_names]]
            self._robot_view.set_dof_armatures(self._tensor(np.asarray(armature, dtype=np.float32), param_device), param_one)
        self._object_views = {name: self._sim_view.create_rigid_body_view(item['path'])
                              for name, item in self.scene_info['objects'].items()}
        for name, view in self._object_views.items():
            if view.count != 1:
                raise RuntimeError(f'PhysX rigid body missing: {name}')
        self.object_names = list(self._object_views)
        if self._task_readback is not None:
            self._task_readback.bind_views(self._object_views)
        self._fixed_actor_view = None
        fixed_paths = self.scene_info.get('kinematic_body_paths', [])
        if fixed_paths:
            from .static_mesh_contacts import FixedActorGuard
            self._sim_view.initialize_kinematic_bodies()
            self._fixed_actor_view = self._sim_view.create_rigid_body_view(fixed_paths)
            actual = list(self._fixed_actor_view.prim_paths)
            if self._fixed_actor_view.count != len(fixed_paths) or set(actual) != set(fixed_paths):
                raise RuntimeError('Native fixed mesh actor inventory differs from authored ownership')
            transforms = _np(self._fixed_actor_view.get_transforms())
            if not hasattr(self, '_fixed_actor_guard'):
                self._fixed_actor_guard = FixedActorGuard(actual, transforms)
            self._fixed_actor_guard.check(actual, transforms)
        sensors = list(self.scene_info['robot_link_paths'].values()) + [x['path'] for x in self.scene_info['objects'].values()]
        # Support surfaces are scene-defined: the legacy pair for a workbench
        # scene, the room floor and solid furniture for a structured room.
        supports = list(self.scene_info.get('support_paths', ['/World/Table', '/World/Ground']))
        filters = sensors + supports + [p for x in self.scene_info['fixtures'].values()
                                        for p in x.get('contact_filter_paths', x['collision_paths'])]
        self._rigid_contacts = self._sim_view.create_rigid_contact_view(
            sensors, filter_patterns=[filters] * len(sensors), max_contact_data_count=32768)
        self._contact_sensor_paths = list(self._rigid_contacts.sensor_paths)
        raw_filters = list(self._rigid_contacts.filter_paths)
        if raw_filters and isinstance(raw_filters[0], str):
            self._contact_filter_paths = [raw_filters] * len(self._contact_sensor_paths)
        else:
            self._contact_filter_paths = [list(row) for row in raw_filters]
        if (len(self._contact_sensor_paths) != self._rigid_contacts.sensor_count or
            len(self._contact_filter_paths) != self._rigid_contacts.sensor_count or
            any(len(row) != self._rigid_contacts.filter_count for row in self._contact_filter_paths)):
            raise RuntimeError('Native contact view path mapping does not match its tensor dimensions')
        self._soft_views, self._soft_initial = {}, {}
        for name, info in self.scene_info.get('soft_bodies', {}).items():
            view = self._sim_view.create_volume_deformable_body_view(info['path'])
            if view.count != 1:
                raise RuntimeError(f'PhysX did not create the volume deformable {name}')
            spec = next(s for s in self.scene_spec['soft_bodies'] if s['id'] == name)
            initial = (info['rest_vertices'] @ _rotation(spec.get('quaternion_wxyz', [1, 0, 0, 0])).T
                       + np.asarray(spec['position_m'])).astype(np.float32)
            actual = _np(view.get_simulation_nodal_positions())[0]
            if actual.shape != initial.shape:
                raise RuntimeError(f'Soft body {name} node topology changed: {actual.shape} != {initial.shape}')
            self._soft_views[name], self._soft_initial[name] = view, initial
        self.soft_body_names = list(self._soft_views)
        self._cloth_view = None
        self.triangles = np.empty((0, 3), dtype=np.int32)
        self._cloth_initial = np.empty((0, 3), dtype=np.float32)
        if self.has_cloth:
            self._cloth_view = self._sim_view.create_surface_deformable_body_view('/World/Cloth')
            if self._cloth_view.count != 1:
                raise RuntimeError('PhysX did not create the required live surface deformable')
            c = self.scene_spec['cloth']
            self.triangles = self.scene_info['cloth']['triangles'].copy()
            self._cloth_initial = (self.scene_info['cloth']['rest_vertices'] @ _rotation(c.get('quaternion_wxyz', [1,0,0,0])).T
                                   + np.asarray(c['origin_m'])).astype(np.float32)
            actual = _np(self._cloth_view.get_simulation_nodal_positions())[0]
            if actual.shape != self._cloth_initial.shape:
                raise RuntimeError(f'Cloth vertex topology changed: {actual.shape} != {self._cloth_initial.shape}')

    def _tensor(self, value, device=None):
        import torch
        selected = getattr(self, 'tensor_device', getattr(self, '_native_tensor_device', self.device)) if device is None else device
        return torch.as_tensor(value, device=selected).contiguous()

    def action_spec(self):
        names = getattr(self, 'actuated_joint_names', self.joint_names)
        return {'type': 'joint_position_targets', 'joint_names': list(names),
                'joint_limits_rad': {n: deepcopy(self.joint_limits[n]) for n in names}, 'duration_s': [.02, .02],
                'arm_joint_names': [n for n in names if n.startswith('fr3_')],
                'hand_joint_names': [n for n in names if not n.startswith('fr3_')],
                'initial_joint_positions_rad': {n: self.home[n] for n in names},
                'state_joint_names': list(self.joint_names), 'physical_joint_count': len(self.joint_names),
                'mimic_joints': deepcopy(getattr(self, 'mimic_joints', {})),
                'hand_id': getattr(self, 'hand_spec', {}).get('hand_id', 'sharpa_w01_right'),
                'joint_count': len(names), 'all_joints_required': True, 'object_state_writeback': False}

    def _validate_action(self, action):
        if not isinstance(action, dict) or set(action) != {'joint_positions_rad', 'duration_s'}:
            raise ValueError('Action contains only complete joint_positions_rad and duration_s; object/cloth writes are reset-only')
        if isinstance(action['duration_s'], bool) or action['duration_s'] != .02:
            raise ValueError('The benchmark control period is fixed at 0.02 s')
        q = action['joint_positions_rad']
        if not isinstance(q, dict) or set(q) != set(getattr(self, 'actuated_joint_names', self.joint_names)):
            raise ValueError('Supply exactly the active named robot joint targets from action_spec')
        for name, value in q.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value):
                raise ValueError(f'Invalid joint target: {name}')
            if not self.joint_limits[name][0] <= value <= self.joint_limits[name][1]:
                raise ValueError(f'Joint target outside URDF limit: {name}')

    def _expand_mimic_targets(self, positions):
        result = dict(positions)
        for name, mimic in getattr(self, 'mimic_joints', {}).items():
            result[name] = result[mimic['joint']] * mimic['multiplier'] + mimic['offset']
            if not self.joint_limits[name][0] <= result[name] <= self.joint_limits[name][1]:
                raise ValueError('Expanded mimic target outside follower limit: ' + name)
        return result

    def reset(self, seed=0, initial_state=None):
        """The only entry point permitted to directly initialize physical state."""
        initial = deepcopy(initial_state or {})
        if set(initial) - {'objects', 'cloth_vertices_m', 'joint_positions_rad', 'soft_body_nodes_m'}:
            raise ValueError('Unknown reset state fields')
        if set(initial.get('soft_body_nodes_m', {})) - set(getattr(self, 'soft_body_names', [])):
            raise ValueError('Reset contains an unknown soft body')
        if set(initial.get('objects', {})) - set(self.object_names):
            raise ValueError('Reset contains an unknown task object')
        task_readback = getattr(self, '_task_readback', None)
        bundle = self.scene_info.get('native_task_bundle')
        if bundle and set(initial.get('objects', {})) & set(bundle['objects']):
            raise ValueError('Referenced mechanism resets must use their frozen assembly poses')
        if (self._reset_count or (task_readback is not None and task_readback.has_particles)
                or (bundle is not None and bundle['particle_sets'])):
            # Stop before restoring source state. Reattach with the declared
            # readback profile and recreate every native tensor handle.
            self.world.stop()
            if task_readback is not None:
                task_readback.restore_before_world_reset()
            if bundle:
                from .native_task_bundle import restore_bundle_before_world_reset
                restore_bundle_before_world_reset(self.stage, bundle)
            self._configure_task_readback()
            self.world.reset()
            self._configure_task_readback()
            self._initialize_views()
        from .workroom_materials import runtime_material_check as check_workroom_materials
        self.runtime_workroom_material_check = check_workroom_materials(
            self.stage, self._sim_view, self.scene_info)
        self._reset_count += 1
        self.seed = int(seed)
        self.time_s, self.last_duration_s = 0., 0.
        self.terminated = self.truncated = False
        self.reset_override = initial
        q = {**{n: self.home[n] for n in self.actuated_joint_names}, **initial.get('joint_positions_rad', {})}
        self._validate_action({'joint_positions_rad': q, 'duration_s': .02})
        q = self._expand_mimic_targets(q)
        self._targets = np.asarray([[q[n] for n in self.joint_names]], dtype=np.float32)
        # World.reset performs a native initialization step. Restore the fixed
        # root explicitly alongside q/qd so no initialization impulse persists.
        from .robot_layout import world_base_poses
        poses = list(world_base_poses(self.scene_spec['robot']).values())
        root = np.asarray([[*pose['position_m'], *pose['quaternion_wxyz'][1:], pose['quaternion_wxyz'][0]]
                           for pose in poses], dtype=np.float32)
        self._robot_view.set_root_transforms(self._tensor(root), self._one)
        self._robot_view.set_root_velocities(self._tensor(np.zeros((len(poses), 6), dtype=np.float32)), self._one)
        self._robot_view.set_dof_positions(self._tensor(self._targets), self._one)
        self._robot_view.set_dof_velocities(self._tensor(np.zeros_like(self._targets)), self._one)
        self._robot_view.set_dof_position_targets(self._tensor(self._targets), self._one)
        self._robot_view.set_dof_velocity_targets(self._tensor(np.zeros_like(self._targets)), self._one)
        self._initial_objects = {}
        reset_objects = {obj['id']: obj for obj in self.scene_spec['objects']}
        reset_objects.update({name: info['initial_state'] for name, info in self.scene_info['objects'].items()
                              if 'task_layer_id' in info})
        if bundle:
            reset_objects.update(bundle['objects'])
        for name, obj in reset_objects.items():
            state = {**obj, **initial.get('objects', {}).get(name, {})}
            pos = np.asarray(state['position_m'], dtype=float)
            quat = np.asarray(state.get('quaternion_wxyz', [1, 0, 0, 0]), dtype=float)
            if pos.shape != (3,) or quat.shape != (4,) or not np.isfinite(pos).all() or not np.isfinite(quat).all() or np.linalg.norm(quat) == 0:
                raise ValueError(f'Invalid reset object pose: {name}')
            quat /= np.linalg.norm(quat)
            tf = np.asarray([[*pos, *quat[[1, 2, 3, 0]]]], dtype=np.float32)
            self._object_views[name].set_transforms(self._tensor(tf), self._one)
            self._object_views[name].set_velocities(self._tensor(np.zeros((1, 6), dtype=np.float32)), self._one)
            self._initial_objects[name] = pos.copy()
        if self.has_cloth:
            points = np.asarray(initial.get('cloth_vertices_m', self._cloth_initial), dtype=np.float32)
            if points.shape != self._cloth_initial.shape or not np.isfinite(points).all():
                raise ValueError('Reset cloth vertices must match the original finite Nx3 mesh')
            self._cloth_view.set_simulation_nodal_positions(self._tensor(points[None, ...]), self._one)
            self._cloth_view.set_simulation_nodal_velocities(self._tensor(np.zeros_like(points)[None, ...]), self._one)
        elif 'cloth_vertices_m' in initial:
            raise ValueError('This task has no cloth')
        for name, view in self._soft_views.items():
            nodes = np.asarray(initial.get('soft_body_nodes_m', {}).get(name, self._soft_initial[name]), dtype=np.float32)
            if nodes.shape != self._soft_initial[name].shape or not np.isfinite(nodes).all():
                raise ValueError(f'Reset soft body {name} nodes must match the original finite Nx3 mesh')
            view.set_simulation_nodal_positions(self._tensor(nodes[None, ...]), self._one)
            view.set_simulation_nodal_velocities(self._tensor(np.zeros_like(nodes)[None, ...]), self._one)
        self._sim_view.update_articulations_kinematic()
        from .robot_base import runtime_base_check
        self.runtime_robot_base_check = runtime_base_check(self._robot_view, self.scene_info)
        self._contact_reports.clear()
        self._contact_errors.clear()
        if task_readback is not None:
            task_readback.invalidate()
        self._begin_contact_sampling(baseline=True)
        if bundle and self.task_observer_enabled:
            from .native_task_bundle import NativeTaskReader
            self._task_reader = NativeTaskReader(self.stage, bundle, self.scene_spec['native_task_bundle'], world=self.world)
            self._sample_task_state([], baseline=True)
        return self.observation()

    def hold_current_targets(self):
        """Cancel motion by holding measured active q; no simulation/state write."""
        q = _np(self._robot_view.get_dof_positions())[0]
        active = {n: float(np.clip(q[self.joint_names.index(n)], *self.joint_limits[n])) for n in self.actuated_joint_names}
        expanded = self._expand_mimic_targets(active)
        self._targets = np.asarray([[expanded[n] for n in self.joint_names]], dtype=np.float32)
        self._robot_view.set_dof_position_targets(self._tensor(self._targets), self._one)
        return {'joint_positions_rad': active, 'duration_s': .02}

    def step(self, action):
        self._validate_action(action)
        if self.terminated:
            raise RuntimeError('Episode terminated; reset before stepping')
        if not self.world.is_playing():
            raise RuntimeError('Native Isaac Sim timeline is not playing; no physics step can be claimed')
        targets = self._expand_mimic_targets(action['joint_positions_rad'])
        self._targets = np.asarray([[targets[n] for n in self.joint_names]], dtype=np.float32)
        self._robot_view.set_dof_position_targets(self._tensor(self._targets), self._one)
        self._contact_reports.clear()
        self._begin_contact_sampling()
        native_start_step = self.world.current_time_step_index
        native_start_time = self.world.current_time
        # No pose, velocity, mesh, constraint, or attachment setters occur here.
        for _ in range(round(.02 / self.dt)):
            self._current_step_time = self.time_s + self.dt
            if getattr(self, '_arm_gravity_mask', None) is not None:
                gravity = self._robot_view.get_gravity_compensation_forces()
                # A non-fixed-base articulation prepends 6 root entries; keep the DOFs.
                gravity = gravity[:, gravity.shape[1] - self._arm_gravity_mask.shape[1]:]
                self._robot_view.set_dof_actuation_forces((gravity * self._arm_gravity_mask).contiguous(), self._one)
            self.world.step(render=False)
            if getattr(self, '_fixed_actor_view', None) is not None:
                self._fixed_actor_guard.check(self._fixed_actor_view.prim_paths,
                                              _np(self._fixed_actor_view.get_transforms()))
            self.time_s += self.dt
            contact_start = len(self._contact_reports)
            self._read_rigid_contacts()
            if getattr(self, '_task_reader', None) is not None:
                self._sample_task_state(self._contact_reports[contact_start:])
        if self.world.current_time_step_index - native_start_step != round(.02 / self.dt):
            raise RuntimeError('Native physics callback count differs from requested integration steps')
        if not math.isclose(self.world.current_time - native_start_time, .02, abs_tol=1.e-8, rel_tol=0):
            raise RuntimeError('Native physics elapsed time differs from 20 ms control interval')
        if self._contact_sampling['sampled_substeps'] != round(.02 / self.dt):
            raise RuntimeError('Native rigid contact sampling missed a physics substep')
        self._contact_sampling['complete'] = True
        self.last_duration_s = .02
        self.truncated = self.time_s >= 120.
        return self.observation()

    def _project_rigid_contacts(self, sample=None):
        # The legacy IPhysxSimulation singleton is unattached under 6.1's
        # multi-stage manager. This view is bound to the actual simulated stage.
        # dt=1 returns impulses (Ns), not forces; tangential impulses are not
        # included by this API and are explicitly not claimed.
        data = self._rigid_contacts.get_contact_data(dt=1.) if sample is None else sample.for_env(self)
        force, _, normal, separation, counts, starts = data
        counts, starts = _np(counts), _np(starts)
        active = np.argwhere(counts > 0)
        if not len(active):
            return []
        used = max(int(starts[i,j]) + int(counts[i,j]) for i,j in active)
        if used >= self._rigid_contacts.max_contact_data_count:
            raise RuntimeError('Native rigid contact buffer reached capacity; evidence may be truncated')
        force, normal, separation = _np(force[:used]).reshape(-1), _np(normal[:used]), _np(separation[:used]).reshape(-1)
        pairs = []
        seen = set()
        for i, j in active:
            actor0, actor1 = self._contact_sensor_paths[i], self._contact_filter_paths[i][j]
            key = tuple(sorted((actor0, actor1)))
            if key in seen:
                continue
            seen.add(key)
            pairs.append((actor0, actor1, int(starts[i,j]), int(counts[i,j])))
        summaries = summarize_contact_groups(force, normal,
            [p[2] for p in pairs], [p[3] for p in pairs])
        rows = []
        for (actor0, actor1, start, count), summary in zip(pairs, summaries):
            rows.append({'time_s': self._current_step_time,
                'actor0': actor0, 'actor1': actor1, 'collider0': None, 'collider1': None,
                'point_count': count, **summary,
                'min_separation_m': float(separation[start:start+count].min()),
                'impulse_components': 'normal_only', 'collider_identity_available': False})
        return rows

    def _sample_task_state(self, contacts, *, baseline=False):
        objects = {name: self._pose(_np(view.get_transforms())[0], _np(view.get_velocities())[0])
                   for name, view in self._object_views.items()}
        self._task_reader.sample(objects=objects, contacts=contacts, time_s=self.time_s,
            native_time_s=self.world.current_time, native_step=self.world.current_time_step_index,
            physics_dt_s=self.dt, baseline=baseline)

    def _begin_contact_sampling(self, *, baseline=False):
        """Record actual acquisitions, including empty-contact substeps."""
        self._contact_sampling = {
            'protocol': 'native_rigid_contact_sampling_v1', 'physics_dt_s': self.dt,
            'start_sim_time_s': float(self.time_s), 'end_sim_time_s': float(self.time_s),
            'native_start_time_s': float(self.world.current_time),
            'native_end_time_s': float(self.world.current_time),
            'native_start_step_index': int(self.world.current_time_step_index),
            'native_end_step_index': int(self.world.current_time_step_index),
            'sampled_substeps': 0, 'complete': baseline,
        }

    def _read_rigid_contacts(self):
        sampling = self._contact_sampling
        step, native_time = int(self.world.current_time_step_index), float(self.world.current_time)
        if (sampling['complete'] or step != sampling['native_end_step_index'] + 1
                or not math.isclose(native_time - sampling['native_end_time_s'], self.dt, abs_tol=1.e-9, rel_tol=0)
                or not math.isclose(self.time_s - sampling['end_sim_time_s'], self.dt, abs_tol=1.e-9, rel_tol=0)
                or not math.isclose(self._current_step_time, self.time_s, abs_tol=1.e-9, rel_tol=0)):
            raise RuntimeError('Native contact acquisition clock is stale, skipped or inconsistent')
        exchange = getattr(self, '_contact_exchange', None)
        public_time = float(self.time_s)
        rows = self._project_rigid_contacts() if exchange is None else exchange.consume(self)
        if (int(self.world.current_time_step_index) != step or float(self.world.current_time) != native_time
                or float(self.time_s) != public_time or float(self._current_step_time) != public_time):
            raise RuntimeError('Native contact acquisition clock changed while reading solver evidence')
        self._contact_reports.extend(rows)
        sampling.update(sampled_substeps=sampling['sampled_substeps'] + 1,
                        end_sim_time_s=float(self.time_s), native_end_time_s=native_time,
                        native_end_step_index=step)

    def capture_state(self):
        q = _np(self._robot_view.get_dof_positions())[0]
        qd = _np(self._robot_view.get_dof_velocities())[0]
        body = _np(self._robot_view.get_link_transforms())[0]
        body_vel = _np(self._robot_view.get_link_velocities())[0]
        # Cloth-only workcells have no task rigid bodies. Preserve the recorded
        # pose/velocity dimensions so concatenation and archive audits stay valid.
        objects = (np.stack([_np(self._object_views[n].get_transforms())[0] for n in self.object_names])
                   if self.object_names else np.empty((0, 7), dtype=body.dtype))
        object_vel = (np.stack([_np(self._object_views[n].get_velocities())[0] for n in self.object_names])
                      if self.object_names else np.empty((0, 6), dtype=body_vel.dtype))
        points = _np(self._cloth_view.get_simulation_nodal_positions())[0] if self.has_cloth else np.empty((0, 3))
        point_vel = _np(self._cloth_view.get_simulation_nodal_velocities())[0] if self.has_cloth else np.empty((0, 3))
        soft = {name: (_np(view.get_simulation_nodal_positions())[0], _np(view.get_simulation_nodal_velocities())[0])
                for name, view in getattr(self, '_soft_views', {}).items()}
        fixed = {}
        if getattr(self, '_fixed_actor_view', None) is not None:
            fixed = {'kinematic_support_q': _np(self._fixed_actor_view.get_transforms()),
                     'kinematic_support_qd': _np(self._fixed_actor_view.get_velocities())}
        task = (self._task_readback.capture_arrays(self.world, self.object_names, objects, object_vel)
                if getattr(self, '_task_readback', None) is not None else {})
        task_particles = (self._task_reader.capture_particles()
                          if getattr(self, '_task_reader', None) is not None else {})
        return {**fixed, **task, **task_particles, 'joint_q': q, 'joint_qd': qd, 'joint_target_q': self._targets[0].copy(),
                **{f'soft_{name}_q': nodes for name, (nodes, _) in soft.items()},
                **{f'soft_{name}_qd': velocities for name, (_, velocities) in soft.items()},
                'body_q': np.concatenate([body, objects]), 'body_qd': np.concatenate([body_vel, object_vel]),
                'object_q': objects, 'object_qd': object_vel,
                'particle_q': points, 'particle_qd': point_vel, 'time_s': np.asarray(self.time_s),
                'native_world_time_s': np.asarray(self.world.current_time),
                'native_physics_step_index': np.asarray(self.world.current_time_step_index)}

    def task_runtime_state(self, state=None):
        """Private same-tick task data; scene-owned adapters interpret its rules."""
        if getattr(self, '_task_readback', None) is None:
            return None
        return self._task_readback.payload(self.capture_state() if state is None else state, self.object_names)

    @staticmethod
    def _pose(q, qd):
        return {'position_m': q[:3].tolist(), 'quaternion_wxyz': q[[6, 3, 4, 5]].tolist(),
                'linear_velocity_m_s': qd[:3].tolist(), 'angular_velocity_rad_s': qd[3:].tolist()}

    def observation(self):
        state = self.capture_state()
        finite = all(np.isfinite(value).all() for value in state.values())
        self.terminated = self.terminated or not finite
        joints = dict(zip(self.joint_names, state['joint_q'].tolist()))
        velocities = dict(zip(self.joint_names, state['joint_qd'].tolist()))
        poses = {name: self._pose(state['body_q'][i], state['body_qd'][i]) for i, name in enumerate(self.body_names)}
        objects = {name: self._pose(state['object_q'][i], state['object_qd'][i]) for i, name in enumerate(self.object_names)}
        robot = {'joint_positions_rad': joints, 'joint_velocities_rad_s': velocities,
                 'joint_targets_rad': dict(zip(self.joint_names, self._targets[0].tolist())),
                 'fingertips': {name: p for name, p in poses.items() if name.endswith('fingertip')}}
        frame_names = ['fr3_link8', 'right_hand_flange', 'right_hand_C_MC', 'left_hand_flange', 'left_hand_C_MC']
        for side in self.hand_spec.get('arms', {}):
            frame_names.extend((f'fr3_{side}_link8', f'{side}_hand_flange', f'{side}_hand_C_MC'))
        for name in frame_names:
            if name in poses:
                robot[name] = poses[name]
        prefix = self.hand_spec['prefix']
        robot['hand_id'] = self.hand_spec['hand_id']
        robot['distal_link_poses'] = {finger: poses[prefix + name] for finger, name in self.hand_spec['fingertip_links'].items() if prefix + name in poses}
        robot['distal_link_pose_semantics'] = 'Rigid-link frame origins; not calibrated contact points.'
        palm = prefix + self.hand_spec['palm_link']
        robot['palm_pose'] = poses.get(palm)
        points = state['particle_q']
        return {'time_s': self.time_s, 'sim_time_s': self.time_s, 'last_duration_s': self.last_duration_s,
                **({'task_runtime': self.task_runtime_state(state)} if getattr(self, '_task_readback', None) is not None else {}),
                **({'task_state': self._task_reader.observe()} if getattr(self, '_task_reader', None) is not None else {}),
                'terminated': self.terminated, 'truncated': self.truncated,
                'joint_positions_rad': joints, 'joint_velocities_rad_s': velocities,
                'robot': robot, 'objects': objects,
                'soft_bodies': {name: self._soft_body_summary(state[f'soft_{name}_q'], state[f'soft_{name}_qd'])
                                for name in getattr(self, 'soft_body_names', [])},
                'cloth': {'enabled': self.has_cloth, 'vertices_m': points.tolist(),
                          'velocities_m_s': state['particle_qd'].tolist(), 'triangles': self.triangles.tolist(),
                          'centroid_m': points.mean(axis=0).tolist() if len(points) else None},
                'contacts': {'source': 'PhysX stage-bound rigid contact tensor; normal impulses',
                             'scope': 'Robot/task rigid-body pairs against robot, objects, table, ground and fixture colliders; every physics substep',
                             'rigid': [dict(row, impulse_vector_ns=list(row['impulse_vector_ns']))
                                       for row in self._contact_reports], 'errors': list(self._contact_errors),
                             'sampling': deepcopy(self._contact_sampling),
                             'cloth_contact_reporting': 'unavailable; not replaced by geometric proximity',
                             'rigid_contact_point_count': sum(x['point_count'] for x in self._contact_reports),
                             'rigid_impulse_magnitude_sum_ns': sum(x['impulse_magnitude_sum_ns'] for x in self._contact_reports)},
                'metrics': {'state_finite': finite, 'success': False, 'success_evaluator': 'external ordered SequenceEvaluator',
                            'object_displacement_m': {n: float(np.linalg.norm(np.asarray(p['position_m']) - self._initial_objects[n])) for n, p in objects.items()}},
                'state_access': 'simulation ground truth; agent observation permissions are separate'}

    def _soft_body_summary(self, nodes, velocities):
        """Per-step deformable state: kinematic summaries always; full nodes only on
        request (``soft_body_nodes_in_observation``). Every step's full nodes stay
        in capture_state for recordings and audits."""
        speeds = np.linalg.norm(velocities, axis=1)
        summary = {'centroid_m': nodes.mean(axis=0).tolist(), 'bounds_min_m': nodes.min(axis=0).tolist(),
                   'bounds_max_m': nodes.max(axis=0).tolist(), 'max_node_speed_m_s': float(speeds.max()),
                   'mean_node_speed_m_s': float(speeds.mean()), 'node_count': int(len(nodes)),
                   'contact_reporting': 'unavailable; PhysX reports no deformable contact impulses'}
        if self.options.get('soft_body_nodes_in_observation', False):
            summary.update(nodes_m=nodes.tolist(), velocities_m_s=velocities.tolist())
        return summary

    def metadata(self):
        from isaacsim.core.version import get_version
        import omni.kit.app
        from .asset_catalog import provenance as catalog_provenance
        from .robot_asset import digest
        info = deepcopy(self.scene_info)
        if info['cloth']:
            info['cloth'].pop('triangles')
            info['cloth'].pop('rest_vertices')
        for soft in info.get('soft_bodies', {}).values():
            for key in ('tets', 'surface_triangles', 'rest_vertices'):
                soft[key] = np.asarray(soft.pop(key)).shape
        source_files = sorted(Path(__file__).parent.glob('*.py'))
        if isinstance(self.robot_usd, dict):
            manifests = {side: Path(path) for side, path in self.robot_source['import_manifests'].items()}
        else:
            manifests = {None: Path(self.robot_source['urdf_path']).parent / 'USD_IMPORT.json'}
        snapshots, usd_files = {}, {}
        for key, import_manifest_path in manifests.items():
            import_manifest = snapshots[key] = load_json(import_manifest_path)
            files = {path: digest(path) for path in import_manifest['derived_files']}
            if files != import_manifest['derived_files']:
                raise RuntimeError('Robot USD dependencies changed since the recorded Isaac import')
            usd_files.update(files)
        if isinstance(self.robot_usd, dict):
            robot_usd = {side: str(path) for side, path in self.robot_usd.items()}
            robot_usd_sha256 = hashlib.sha256(json.dumps({side: digest(path) for side, path in self.robot_usd.items()},
                                                         sort_keys=True).encode()).hexdigest()
            import_record = {side: {'path': str(path), 'sha256': digest(path), 'snapshot': snapshots[side]}
                             for side, path in manifests.items()}
        else:
            robot_usd, robot_usd_sha256 = str(self.robot_usd), digest(self.robot_usd)
            import_record = {'path': str(manifests[None]), 'sha256': digest(manifests[None]), 'snapshot': snapshots[None]}
        return {'robot_model': 'FR3 + ' + self.hand_spec['label'], 'hand_spec': deepcopy(self.hand_spec),
                'physics_engine': ('Isaac Sim native PhysX CPU rigid dynamics' if self.physics_device == 'cpu'
                                   else 'Isaac Sim native PhysX GPU rigid dynamics and surface deformables'),
                'physics_device': self.physics_device,
                'native_task_observer_enabled': self.task_observer_enabled,
                'isaacsim_version': list(get_version()), 'kit_version': omni.kit.app.get_app().get_build_version(),
                'device': self.device, 'physics_dt_s': self.dt, 'control_dt_s': .02,
                'native_physics_tensor_pipeline': deepcopy(self.runtime_physics_tensor_pipeline),
                'native_readback_configuration': deepcopy(self.runtime_readback_configuration),
                'native_readback_profile_resolution': deepcopy(self.readback_profile_resolution),
                'native_state_tensor_device': str(self._native_tensor_device),
                'physics_solver': {
                    'profile_selection': self.solver_type,
                    'usd_readback': self.stage.GetPrimAtPath(self.world.get_physics_context().prim_path)
                        .GetAttribute('physxScene:solverType').Get(),
                    'scope': 'PhysicsScene USD selection; not a native tensor solver-enum query.'},
                'physics_external_forces_every_iteration': {
                    'requested': self.enable_external_forces_every_iteration,
                    'usd_readback': self.stage.GetPrimAtPath(self.world.get_physics_context().prim_path)
                        .GetAttribute('physxScene:enableExternalForcesEveryIteration').Get(),
                    'scope': 'PhysicsScene USD setting; not a native PhysX flag getter.'},
                'gpu_selection': ({'mode': 'single_cuda_visible_device_by_cuda_index',
                                   'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES')}
                                  if self.single_cuda_device else {'mode': 'physical_index'}),
                'experience': {'path': str(self.experience_path), 'sha256': digest(self.experience_path)},
                'driver_verification_disabled': self.allow_unvalidated_driver,
                'driver_support_status': ('Unvalidated host-driver compatibility experiment; not vendor-supported qualification.'
                                          if self.allow_unvalidated_driver else 'Native RTX driver verification enabled.'),
                'scene_spec': deepcopy(self.scene_spec), 'scene_info': info,
                **({'task_layer_readback': self._task_readback.metadata()}
                   if getattr(self, '_task_readback', None) is not None else {}),
                **({'static_mesh_contact_runtime': self._fixed_actor_guard.report()}
                   if hasattr(self, '_fixed_actor_guard') else {}),
                'robot_asset': self.robot_source, 'robot_usd': robot_usd,
                'robot_usd_sha256': robot_usd_sha256,
                'robot_usd_import_manifest': import_record,
                'robot_usd_dependency_sha256': usd_files, 'task_catalog_reader': catalog_provenance(),
                'action_spec': self.action_spec(), 'joint_names': self.joint_names,
                'body_names': self.body_names + self.object_names, 'object_names': self.object_names,
                'native_joint_armatures_kg_m2': dict(zip(self.joint_names, _np(self._robot_view.get_dof_armatures())[0].tolist())),
                'native_joint_stiffnesses': dict(zip(self.joint_names, _np(self._robot_view.get_dof_stiffnesses())[0].tolist())),
                'native_joint_dampings': dict(zip(self.joint_names, _np(self._robot_view.get_dof_dampings())[0].tolist())),
                'native_joint_friction_coefficients': dict(zip(self.joint_names, _np(self._robot_view.get_dof_friction_coefficients())[0].tolist())),
                'native_joint_max_velocities': dict(zip(self.joint_names, _np(self._robot_view.get_dof_max_velocities())[0].tolist())),
                'native_robot_material_properties': _np(self._robot_view.get_material_properties())[0].tolist(),
                'runtime_robot_material_check': deepcopy(getattr(self, 'runtime_robot_material_check', None)),
                'runtime_workroom_material_check': deepcopy(getattr(self, 'runtime_workroom_material_check', None)),
                'runtime_robot_base_check': deepcopy(getattr(self, 'runtime_robot_base_check', None)),
                'arm_gravity_compensation_scope': ('PhysX generalized gravity forces on fr3_joint1-7 (arm and hand payload), '
                    'applied before every physics substep as actuation forces outside the drive effort clamp')
                    if getattr(self, '_arm_gravity_mask', None) is not None else None,
                'hand_actuator_model': self.options.get('hand_actuator_model', 'legacy_uniform_gains'),
                'arm_gravity_compensation': self.options.get('arm_gravity_compensation', 'none'),
                'native_robot_link_masses_kg': dict(zip(self.body_names, _np(self._robot_view.get_masses())[0].tolist())),
                'native_object_masses_kg': {name: float(_np(view.get_masses()).reshape(-1)[0])
                                            for name, view in self._object_views.items()},
                'native_object_inertias_kg_m2': {name: _np(view.get_inertias()).reshape(-1, 3, 3)[0].tolist()
                                                for name, view in self._object_views.items()},
                'native_object_com_pose_xyzw': {name: _np(view.get_coms()).reshape(-1, 7)[0].tolist()
                                                for name, view in self._object_views.items()},
                'capture_state_quaternion': 'xyzw', 'observation_quaternion': 'wxyz',
                'object_pose_writeback_after_reset': False, 'cloth_attachment': False,
                'trained_rl_connected': False, 'manus_hardware_connected': False,
                'material_calibrated': False, 'mount_calibrated': False,
                'capture_state_scope': 'Observed physical arrays; not a complete solver checkpoint.',
                'collision_mode': ('PhysX CPU' if self.physics_device == 'cpu' else 'PhysX GPU')
                                  + '; speculative rigid CCD, swept CCD disabled',
                'contact_evidence': {'source': 'stage-bound PhysX RigidContactView.get_contact_data(dt=1)',
                                     'sensor_paths': self._contact_sensor_paths, 'filter_paths': self._contact_filter_paths,
                                     'capacity_points': self._rigid_contacts.max_contact_data_count,
                                     'impulse_components': 'normal_only', 'collider_identity_available': False,
                                     'cloth_contacts_available': False,
                                     'aggregation': aggregation_metadata()},
                'cameras': deepcopy(self.cameras),
                'render_synchronization': deepcopy(self.render_sync),
                'camera_ids': list(self.camera_ids),
                'camera_prim_paths': dict(self.camera_paths or {name: self._camera_prim_path(name)
                                                                for name in self.cameras}),
                'support_paths': list(self.scene_info.get('support_paths', [])),
                'native_isaacsim_app': True, 'standalone_newton_runtime_used': False,
                'reset_override': deepcopy(self.reset_override),
                'sources_sha256': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files}}

    def render(self, camera='overview', width=None, height=None):
        """Return actual RTX pixels; rendering does not advance physics time."""
        return self.render_many((camera,), width, height)[camera]

    def render_many(self, cameras, width=None, height=None):
        """Capture RGB cameras together after the same non-physics updates."""
        frames = self._capture_render_frames(cameras, width, height, include_depth=False)
        return {name: frame['rgb'] for name, frame in frames.items()}

    def render_rgbd(self, camera='overview', width=None, height=None):
        """Read aligned RGB and metric optical-z depth at one physical state.

        Both annotators share a render product and are read after the same
        rendering updates, with no intervening render or physics step.
        """
        return self._capture_render_frames((camera,), width, height, include_depth=True)[camera]

    def _capture_render_frames(self, cameras, width, height, *, include_depth):
        before = self.capture_state()
        frame = self._render_frames(cameras, width, height, include_depth=include_depth)
        after = self.capture_state()
        if before.keys() != after.keys() or any(
                not np.array_equal(before[key], after[key]) for key in before):
            raise RuntimeError('RGB-D/RGB rendering changed physical state')
        return frame

    def _camera_prim_path(self, camera):
        """Where a camera's prim lives: world-fixed, or under its mount link.

        A wrist camera is a child of the hand link, so it rides the arm with no
        per-step work and cannot be desynchronised from the physical state.
        """
        spec = self.cameras[camera]
        if spec['kind'] == 'world':
            return '/World/Cameras/' + camera
        if camera in self._link_camera_paths:
            return self._link_camera_paths[camera]
        link = spec['mount_link']
        links = self.scene_info['robot_link_paths']
        if link not in links:
            raise ValueError(f'Camera {camera!r} mounts on unknown robot link {link!r}; '
                             f'available: {sorted(links)}')
        path = f'{links[link]}/bench_camera_{camera}'
        self._link_camera_paths[camera] = path
        return path

    def _author_cameras(self):
        """Create every declared camera prim once, after the robot is imported."""
        from pxr import Gf, UsdGeom
        for name, spec in self.cameras.items():
            path = self._camera_prim_path(name)
            if self.stage.GetPrimAtPath(path):
                continue
            camera = UsdGeom.Camera.Define(self.stage, path)
            if spec['kind'] == 'world':
                matrix = look_at_matrix(spec['eye'], spec['target'])
                clipping = tuple(spec.get('clipping_m', (.01, 100.)))
            elif spec.get('aim_target_m') is not None:
                # Aimed wrist camera: its local transform is solved from the mount
                # link's *world* pose, so it frames the workspace rather than
                # depending on the link's own axis convention. The link's world
                # transform is required here - GetLocalTransformation() returns a
                # parent-relative transform and would misplace the camera. It stays
                # a child of the link and rides the hand afterwards.
                link = self.scene_info['robot_link_paths'][spec['mount_link']]
                link_prim = self.stage.GetPrimAtPath(link)
                if not link_prim:
                    raise ValueError(f'Camera {name!r} mount link prim is missing: {link}')
                from pxr import Usd
                link_world = UsdGeom.XformCache(Usd.TimeCode.Default()).GetLocalToWorldTransform(link_prim)
                matrix = aimed_link_matrix(link_world, spec['offset_xyz_m'], spec['aim_target_m'],
                                           float(spec.get('standoff_m', 0.)))
                clipping = tuple(spec['clipping_m'])
            else:
                matrix = link_camera_matrix((spec['offset_rpy_rad'][0], spec['offset_rpy_rad'][1],
                                             spec['offset_rpy_rad'][2], spec['offset_xyz_m']))
                clipping = tuple(spec['clipping_m'])
            UsdGeom.Xformable(camera).AddTransformOp().Set(matrix)
            camera.CreateFocalLengthAttr(float(spec['focal_length_mm']))
            # Missing aperture fields retain the frozen legacy USD defaults.
            if 'horizontal_aperture_mm' in spec:
                camera.CreateHorizontalApertureAttr(spec['horizontal_aperture_mm'])
            if 'vertical_aperture_mm' in spec:
                camera.CreateVerticalApertureAttr(spec['vertical_aperture_mm'])
            camera.CreateClippingRangeAttr(Gf.Vec2f(*clipping))
        return {name: self._camera_prim_path(name) for name in self.cameras}

    def prepare_cameras(self):
        """Author declared camera mounts without allocating render products or stepping."""
        if not self._cameras_authored:
            self.camera_paths = self._author_cameras()
            self._cameras_authored = True
        return dict(self.camera_paths)

    def _render_frames(self, cameras, width, height, *, include_depth):
        import carb
        from .render_sync import verify_settings
        cameras = tuple(cameras)
        if not cameras or len(set(cameras)) != len(cameras):
            raise ValueError('Choose at least one distinct camera')
        for camera in cameras:
            if camera not in self.cameras:
                raise ValueError(f'Unknown camera {camera}')
        self.render_sync = verify_settings(carb.settings.get_settings())
        physics_step_before = self.world.current_time_step_index
        native_time_before = self.world.current_time
        if not self._render_products and not self._cameras_authored:
            self.prepare_cameras()
        products = {camera: self._prepare_render_product(camera, width, height, include_depth=include_depth)
                    for camera in cameras}
        visual_state = None
        if getattr(self, '_task_reader', None) is not None:
            visual_state = self._task_reader.sync_visual_state()
        # World.render updates every active product; prepare all cameras first.
        for _ in range(max(warmup for _, warmup in products.values())):
            self.world.render()
        self.render_sync = verify_settings(carb.settings.get_settings())
        if (self.world.current_time_step_index != physics_step_before or
                self.world.current_time != native_time_before):
            raise RuntimeError('Rendering unexpectedly advanced native PhysX; frame/state mapping is invalid')
        return {camera: self._read_render_product(key, include_depth=include_depth, visual_state=visual_state)
                for camera, (key, _) in products.items()}

    def _prepare_render_product(self, camera, width, height, *, include_depth):
        import omni.replicator.core as rep
        # A scene's own cameras carry their resolution; the legacy pair (no cameras
        # block) has none and keeps rendering at the env's size, as before.
        default_width, default_height = self.cameras[camera].get('resolution',
                                                                 (self.render_width, self.render_height))
        width, height = int(width or default_width), int(height or default_height)
        if width <= 0 or height <= 0:
            raise ValueError('Camera dimensions must be positive')
        key = camera, width, height
        warmup = 1 if self.physics_device == 'cpu' else 2
        if key not in self._render_products:
            path = self._camera_prim_path(camera)
            if not self.stage.GetPrimAtPath(path):
                raise RuntimeError(f'Camera prim was not authored: {path}')
            product = rep.create.render_product(path, (width, height))
            annotator = rep.AnnotatorRegistry.get_annotator('rgb')
            annotator.attach([product])
            calibration = rep.AnnotatorRegistry.get_annotator('camera_params')
            calibration.attach([product])
            self._render_products[key] = product, {'rgb': annotator, 'camera_params': calibration}
            warmup = 8
        product, annotators = self._render_products[key]
        if include_depth and 'depth' not in annotators:
            depth = rep.AnnotatorRegistry.get_annotator('distance_to_image_plane')
            depth.attach([product])
            annotators['depth'] = depth
            warmup = 8
        return key, warmup

    def _read_render_product(self, key, *, include_depth, visual_state):
        from .render_sync import camera_calibration
        _, width, height = key
        _, annotators = self._render_products[key]
        rgb = np.asarray(annotators['rgb'].get_data())
        if rgb.shape not in ((height, width, 3), (height, width, 4)):
            raise RuntimeError(f'Native RTX render product returned invalid image: {rgb.shape}')
        frame = {'rgb': rgb[:, :, :3].astype(np.uint8, copy=True), 'sim_time_s': self.time_s,
                 'camera_calibration': camera_calibration(annotators['camera_params'].get_data(),
                                                          width=width, height=height),
                 'render_synchronization': deepcopy(self.render_sync)}
        if visual_state is not None:
            frame['task_visual_state'] = visual_state
        if include_depth:
            from .depth import DEPTH_SOURCE, validate_camera_z_depth
            depth, statistics = validate_camera_z_depth(
                annotators['depth'].get_data(), width=width, height=height)
            frame.update(depth_m=depth, depth_statistics=statistics, depth_source=DEPTH_SOURCE,
                         depth_sim_time_s=self.time_s, depth_semantics='camera_z',
                         depth_unit='m', depth_invalid_value=0.0, depth_encoding='float32_npy',
                         depth_rgb_aligned=True)
        return frame

    def close(self):
        if self._closed:
            return
        # Kit fast shutdown exits the process directly. Preserve and print an
        # active exception so a failed import/rollout cannot appear as exit 0.
        active_exception = sys.exc_info()
        if active_exception[0] is not None:
            traceback.print_exception(*active_exception)
        exit_code = 1 if active_exception[0] is not None else 0
        self._closed = True
        self._contact_subscription = None
        for product, annotators in self._render_products.values():
            for annotator in annotators.values():
                try:
                    annotator.detach([product])
                except Exception:
                    pass
        if self.world is not None:
            try:
                self.world.stop()
                self.world.clear_instance()
            except Exception:
                pass
        if self.app is not None:
            self.app.close(exit_code=exit_code)

