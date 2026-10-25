"""Generic native task-body, prismatic-coordinate and physical-particle reads.

Particle data comes from official PhysX USD readback with GPU dynamics and
explicit host state transport, never from Fabric or a rendered isosurface. Joint coordinates are derived from native rigid
transforms, not authored joint targets. Task rules belong outside this module.
"""
from __future__ import annotations

from copy import deepcopy
import numpy as np


# Particle sets are not published by omni.physx.fabric in this runtime.
# Readback is selected before native attach; GPU physics remains enabled while
# the state transport uses CPU tensors and the official PhysX USD publisher.
PARTICLE_EXPORT_SETTINGS = {
    '/physics/updateToUsd': True, '/physics/updateParticlesToUsd': True,
    '/physics/updateVelocitiesToUsd': True, '/physics/suppressReadback': False,
    '/physics/outputVelocitiesLocalSpace': False,
}


def configure_particle_export(settings):
    for key, value in PARTICLE_EXPORT_SETTINGS.items():
        settings.set_bool(key, value)


def validate_rigid_synchronization(path, native, position, quaternion_xyzw):
    """Compare the actual rigid publisher to the same-tick native body pose."""
    native, position, q = map(lambda x: np.asarray(x, dtype=float), (native, position, quaternion_xyzw))
    if (native.shape != (7,) or position.shape != (3,) or q.shape != (4,)
            or not all(np.isfinite(x).all() for x in (native, position, q))
            or min(np.linalg.norm(q), np.linalg.norm(native[3:])) < 1.e-12):
        raise RuntimeError('Invalid native/Fabric rigid synchronization data: ' + path)
    distance = float(np.linalg.norm(position-native[:3]))
    angle = float(2*np.arccos(np.clip(abs(q @ native[3:]) /
        (np.linalg.norm(q)*np.linalg.norm(native[3:])), 0., 1.)))
    if distance > 1.e-5 or angle > 1.e-4:
        raise RuntimeError(f'Particle readback lacks same-tick native/Fabric rigid synchronization: '
                           f'{path}, position_error_m={distance}, angle_error_rad={angle}')


def synchronize_particle_output(stage, world, native_poses):
    """Official particle USD export plus independent native/Fabric rigid canary.

    Fabric publishes rigid transforms, but does not publish physical particle
    arrays. Neither this export nor the canary authors rigid state into USD.
    """
    import carb
    import omni.physx
    import omni.physxfabric
    import usdrt
    from pxr import UsdUtils
    settings = carb.settings.get_settings()
    if any(settings.get(key) != value for key, value in PARTICLE_EXPORT_SETTINGS.items()):
        raise RuntimeError('Physical particle USD host readback settings changed')
    if omni.physx.get_physx_interface().is_readback_suppressed():
        raise RuntimeError('Physical particle readback was suppressed at native attach')
    before = (int(world.current_time_step_index), float(world.current_time))
    omni.physx.get_physx_simulation_interface().fetch_results()
    omni.physx.get_physx_interface().update_transformations(False, True, True, False)
    omni.physxfabric.get_physx_fabric_interface().force_update(world.get_physics_dt(), world.current_time)
    fabric = usdrt.Usd.Stage.Attach(UsdUtils.StageCache.Get().GetId(stage).ToLongInt())
    if not native_poses:
        raise RuntimeError('Physical particle readback requires a native rigid synchronization canary')
    for path, native in native_poses.items():
        prim = fabric.GetPrimAtPath(path)
        matrix = usdrt.Rt.Xformable(prim).GetFabricHierarchyWorldMatrixAttr().Get() if prim else None
        if matrix is None:
            raise RuntimeError('Missing Fabric rigid world transform for particle synchronization: ' + path)
        q = matrix.ExtractRotationQuat()
        validate_rigid_synchronization(path, native, matrix.ExtractTranslation(), [*q.GetImaginary(), q.GetReal()])
    if before != (int(world.current_time_step_index), float(world.current_time)):
        raise RuntimeError('Readback must not advance native physics')


def array(value):
    if hasattr(value, 'detach'):
        value = value.detach().cpu()
    return np.asarray(value.numpy() if hasattr(value, 'numpy') else value).copy()


def rotation_xyzw(q):
    q = np.asarray(q, dtype=float)
    if q.shape != (4,) or not np.isfinite(q).all() or np.linalg.norm(q) < 1.e-12:
        raise RuntimeError('Invalid native quaternion')
    x, y, z, w = q / np.linalg.norm(q)
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


def prismatic_state(joint, bodies):
    """Project the two joint anchors, including COM and rotating-axis terms."""
    anchors, speeds, rotations, angular = [], [], [], []
    for n in (0, 1):
        body = bodies[joint[f'body{n}']] if joint[f'body{n}'] else None
        pose = np.asarray(body['pose_xyzw'] if body else [0, 0, 0, 0, 0, 0, 1], dtype=float)
        vel = np.asarray(body['velocity'] if body else np.zeros(6), dtype=float)
        R = rotation_xyzw(pose[3:])
        anchor = pose[:3] + R @ np.asarray(joint[f'local_pos{n}'])
        com = pose[:3] + R @ np.asarray(body['com_local_m'] if body else np.zeros(3))
        anchors.append(anchor)
        speeds.append(vel[:3] + np.cross(vel[3:], anchor-com))
        rotations.append(R @ rotation_xyzw(joint[f'local_rot{n}_xyzw']))
        angular.append(vel[3:])
    axis = rotations[0][:, 'XYZ'.index(joint['axis'])]
    delta = anchors[1]-anchors[0]
    return (float(axis @ delta),
            float(axis @ (speeds[1]-speeds[0]) + np.cross(angular[0], axis) @ delta))


def validate_particle_arrays(points, velocities, ids, count, expected_ids=None):
    points, velocities, ids = np.asarray(points, dtype=float), np.asarray(velocities, dtype=float), np.asarray(ids)
    if points.shape != (count, 3) or velocities.shape != points.shape:
        raise RuntimeError('Physical particle count/velocity inventory changed')
    if (ids.shape != (count,) or ids.dtype.kind not in 'iu' or len(np.unique(ids)) != count
            or (expected_ids is not None and not np.array_equal(ids, expected_ids))):
        raise RuntimeError('Physical particle IDs changed; no reindexing is allowed')
    if not np.isfinite(points).all() or not np.isfinite(velocities).all():
        raise RuntimeError('Nonfinite physical particle state')
    return points.copy(), velocities.copy(), ids.astype(np.int64, copy=True)


class TaskLayerReadback:
    def __init__(self, stage, layers):
        from pxr import UsdGeom, UsdPhysics
        self.stage, self.layers = stage, deepcopy(layers)
        self.body_paths = {name: path for layer in layers for name, path in layer['rigid_bodies'].items()}
        self.joints = [joint for layer in layers for joint in layer['joints']]
        self.particles, self.initial_particles, self.initial_xforms = {}, {}, {}
        self.views, self.coms, self._particle_cache = {}, {}, None
        for name, path in self.body_paths.items():
            prim = stage.GetPrimAtPath(path)
            xform = UsdGeom.Xformable(prim)
            self.initial_xforms[path] = [(op.GetOpName(), op.GetAttr().Get()) for op in xform.GetOrderedXformOps()]
            self.initial_xforms[path].append(('xformOpOrder', prim.GetAttribute('xformOpOrder').Get()))
        for layer in layers:
            for name, item in layer['particle_sets'].items():
                if name in self.particles:
                    raise ValueError('Particle observation IDs must be unique across layers')
                prim = stage.GetPrimAtPath(item['path'])
                system = stage.GetPrimAtPath(item['system_path'])
                if not prim.IsA(UsdGeom.Points):
                    raise ValueError('Particle readback requires a physical Points prim')
                if prim.GetRelationship('physxParticle:particleSystem').GetTargets() != [system.GetPath()]:
                    raise ValueError('Particle set has a different system owner')
                if system.GetAttribute('physxParticleSmoothing:particleSmoothingEnabled').Get() is not False:
                    raise ValueError('Physical particle points must have smoothing explicitly disabled')
                if system.GetAttribute('physxParticleAnisotropy:particleAnisotropyEnabled').Get():
                    raise ValueError('Raw physical particle readback does not support anisotropy')
                mass = UsdPhysics.MassAPI(prim).GetMassAttr().Get()
                if mass is None or not np.isclose(mass, item['total_mass_kg'], atol=1.e-7, rtol=0):
                    raise ValueError('Particle authored mass differs from declared fixed inventory')
                points, velocities, ids = self._usd_particle_arrays(item)
                # Local-to-world mapping for physical point arrays must remain
                # fixed; particle set roots are not robot-attached/emitted data.
                matrix = np.asarray(UsdGeom.XformCache().GetLocalToWorldTransform(prim))
                if not np.allclose(matrix, np.eye(4), atol=1.e-9, rtol=0):
                    raise ValueError('Physical particle Points currently require a world-aligned identity root')
                self.particles[name] = item
                self.initial_particles[name] = (points, velocities, ids)

    @property
    def has_particles(self):
        return bool(self.particles)

    def bind_views(self, object_views):
        self.views = {name: object_views[name] for name in self.body_paths}
        self.coms = {name: array(view.get_coms()).reshape(-1, 7)[0, :3]
                     for name, view in self.views.items()}
        self._particle_cache = None

    def restore_before_world_reset(self):
        """Only called with the World stopped, before a native hard reset."""
        from pxr import Vt
        for path, attrs in self.initial_xforms.items():
            prim = self.stage.GetPrimAtPath(path)
            for name, value in attrs:
                prim.GetAttribute(name).Set(value)
            for name in ('physics:velocity', 'physics:angularVelocity'):
                attr = prim.GetAttribute(name)
                if attr:
                    attr.Set((0., 0., 0.))
        for name, (points, velocities, ids) in self.initial_particles.items():
            prim = self.stage.GetPrimAtPath(self.particles[name]['path'])
            prim.GetAttribute('points').Set(Vt.Vec3fArray.FromNumpy(points.astype(np.float32)))
            prim.GetAttribute('velocities').Set(Vt.Vec3fArray.FromNumpy(velocities.astype(np.float32)))
            prim.GetAttribute('ids').Set(Vt.Int64Array.FromNumpy(ids))
        self._particle_cache = None

    def invalidate(self):
        self._particle_cache = None

    def _usd_particle_arrays(self, item):
        prim = self.stage.GetPrimAtPath(item['path'])
        previous = next((data[2] for name, data in self.initial_particles.items()
                         if self.particles[name]['path'] == item['path']), None)
        return validate_particle_arrays(prim.GetAttribute('points').Get(),
            prim.GetAttribute('velocities').Get(), prim.GetAttribute('ids').Get(), item['count'], previous)

    def _synchronize_particles(self, world):
        synchronize_particle_output(self.stage, world, {
            path: array(self.views[name].get_transforms()).reshape(-1, 7)[0]
            for name, path in self.body_paths.items()})

    def capture_arrays(self, world, object_names, object_q, object_qd):
        bodies = {self.body_paths[name]: {'pose_xyzw': object_q[object_names.index(name)],
                    'velocity': object_qd[object_names.index(name)], 'com_local_m': self.coms[name]}
                  for name in self.body_paths}
        values = [prismatic_state(joint, bodies) for joint in self.joints]
        result = {'task_joint_q': np.asarray([v[0] for v in values]),
                  'task_joint_qd': np.asarray([v[1] for v in values])}
        if self.particles:
            key = (int(world.current_time_step_index), float(world.current_time))
            if self._particle_cache is None or self._particle_cache[0] != key:
                self._synchronize_particles(world)
                self._particle_cache = (key, {name: self._usd_particle_arrays(item)
                                             for name, item in self.particles.items()})
            for name, (points, velocities, ids) in self._particle_cache[1].items():
                result.update({f'task_particle_{name}_q': points.copy(),
                    f'task_particle_{name}_qd': velocities.copy(), f'task_particle_{name}_ids': ids.copy()})
        return result

    def payload(self, state, object_names):
        objects = {}
        for name, path in self.body_paths.items():
            index = object_names.index(name)
            pose, vel = state['object_q'][index], state['object_qd'][index]
            objects[name] = {'path': path, 'position_m': pose[:3].tolist(),
                'quaternion_wxyz': pose[[6, 3, 4, 5]].tolist(),
                'linear_velocity_m_s': vel[:3].tolist(), 'angular_velocity_rad_s': vel[3:].tolist()}
        return {'schema': 'bench_task_runtime_v1',
            'native_physics_step_index': int(state['native_physics_step_index']),
            'native_world_time_s': float(state['native_world_time_s']), 'objects': objects,
            'joints': {joint['path']: {'kind': 'prismatic',
                'position_m': float(state['task_joint_q'][i]), 'velocity_m_s': float(state['task_joint_qd'][i]),
                'source': 'derived_from_native_rigid_transforms_and_COM_velocities'}
                for i, joint in enumerate(self.joints)},
            'particle_sets': {name: {'path': item['path'],
                'ids': state[f'task_particle_{name}_ids'].tolist(),
                'positions_world_m': state[f'task_particle_{name}_q'].tolist(),
                'velocities_world_m_s': state[f'task_particle_{name}_qd'].tolist(),
                'count': len(state[f'task_particle_{name}_ids']), 'expected_count': item['count'],
                'total_mass_kg': item['total_mass_kg'],
                'particle_mass_kg': item['total_mass_kg']/item['count'],
                'mass_source': 'pinned authored fixed particle inventory; not measured load transfer',
                'source': 'PhysX official USD host readback of unsmoothed physical Points; GPU solver, CPU state transport',
                'complete': True} for name, item in self.particles.items()},
            'sampling': {'complete': True, 'physics_steps_added': 0,
                         'rigid_pose_source': 'native PhysX rigid-body tensors',
                         'particle_force_or_weight_feedback_qualified': False}}

    def metadata(self):
        return {'schema': 'bench_task_layer_readback_v1', 'joint_paths': [j['path'] for j in self.joints],
            'particle_sets': deepcopy(self.particles), 'rigid_bodies': deepcopy(self.body_paths),
            'particle_reset': 'Restore pinned authored points while stopped, then hard World reset; initial native initialization step remains explicit.',
            'runtime_state_writes_after_reset': False,
            'particle_export_settings': dict(PARTICLE_EXPORT_SETTINGS),
            'native_readback_profile': 'gpu_solver_usd_host_v1' if self.has_particles else None,
            'particle_force_or_weight_feedback_qualified': False}
