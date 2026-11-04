"""Rigid visual snapshots; not resumable PhysX checkpoints or task evaluation."""
import numpy as np
from .env import _np

CAMERA_ATTRS = ('focalLength', 'horizontalAperture', 'verticalAperture',
                'horizontalApertureOffset', 'verticalApertureOffset', 'clippingRange')


def check_supported(env):
    if env.has_cloth or env._soft_views or getattr(env, '_task_reader', None) is not None:
        raise ValueError('Deferred rendering currently requires rigid scenes with task visual observation off')
    reader = getattr(env, '_task_readback', None)
    if reader is not None and (reader.has_particles or reader.joints):
        raise ValueError('Deferred rendering does not yet support task particles or articulated fixtures')
    if env.native_readback_profile != 'cpu_rigid_v1' or not hasattr(env._robot_view, 'get_root_transforms'):
        raise ValueError('Deferred rendering currently supports a single CPU-rigid robot articulation')
    bundle = env.scene_spec.get('native_task_bundle') or {}
    if bundle.get('particle_sets'):
        raise ValueError('Deferred rendering does not yet support task particle sets')


def manifest(env, cameras):
    from pxr import UsdGeom
    check_supported(env)
    records = {}
    for name in cameras:
        prim = env.stage.GetPrimAtPath(env.camera_paths[name])
        attrs = {}
        for key in CAMERA_ATTRS:
            value = prim.GetAttribute(key).Get()
            attrs[key] = list(value) if key == 'clippingRange' else float(value)
        records[name] = {'resolution': list(env.cameras[name]['resolution']), 'attributes': attrs}
    return {'schema': 'rigid_visual_replay.v1', 'solver_checkpoint': False,
            'joint_names': list(env.joint_names), 'object_names': list(env.object_names),
            'cameras': records, 'action_alignment': 'state_before_action',
            'quaternion_order': 'xyzw', 'physics_dt_s': env.dt, 'control_dt_s': env.control_dt}


def capture(env, cameras):
    from pxr import UsdGeom
    state = env.capture_state()
    state['root_q'] = _np(env._robot_view.get_root_transforms())
    state['root_qd'] = _np(env._robot_view.get_root_velocities())
    cache = UsdGeom.XformCache()
    for name in cameras:
        state['camera_'+name] = np.array(cache.GetLocalToWorldTransform(env.stage.GetPrimAtPath(env.camera_paths[name])))
    return state


def restore(env, state, info):
    """Only for a dedicated rendering pass. Permanently disable further stepping."""
    from pxr import Gf, UsdGeom
    import omni.physx
    import omni.physxfabric
    check_supported(env)
    if list(env.joint_names) != info['joint_names'] or list(env.object_names) != info['object_names']:
        raise ValueError('Replay robot/object inventory differs from archive')
    if not all(np.isfinite(value).all() for value in state.values()):
        raise ValueError('Nonfinite replay state')
    env._visual_replay_only = True
    tensor = lambda value: env._tensor(np.asarray(value, dtype=np.float32))
    env._robot_view.set_root_transforms(tensor(state['root_q']), env._one)
    env._robot_view.set_root_velocities(tensor(state['root_qd']), env._one)
    env._robot_view.set_dof_positions(tensor(state['joint_q'][None]), env._one)
    env._robot_view.set_dof_velocities(tensor(state['joint_qd'][None]), env._one)
    for i, name in enumerate(env.object_names):
        env._object_views[name].set_transforms(tensor(state['object_q'][i:i+1]), env._one)
        env._object_views[name].set_velocities(tensor(state['object_qd'][i:i+1]), env._one)
    if 'kinematic_support_q' in state:
        indices = env._tensor(np.arange(len(state['kinematic_support_q']), dtype=np.int32))
        env._fixed_actor_view.set_transforms(tensor(state['kinematic_support_q']), indices)
    env._sim_view.update_articulations_kinematic()
    omni.physx.get_physx_interface().update_transformations(False, True, True, False)
    omni.physxfabric.get_physx_fabric_interface().force_update(env.dt, env.world.current_time)
    env.time_s = float(state['time_s'])
    for name, camera in info['cameras'].items():
        path = '/World/Cameras/'+name
        prim = UsdGeom.Camera.Define(env.stage, path).GetPrim()
        env.cameras[name] = {'kind': 'world', 'id': name, 'resolution': camera['resolution']}
        env.camera_paths[name] = path
        for key, value in camera['attributes'].items():
            prim.GetAttribute(key).Set(Gf.Vec2f(*value) if key == 'clippingRange' else value)
        xform = UsdGeom.Xformable(prim)
        xform.ClearXformOpOrder()
        xform.AddTransformOp().Set(Gf.Matrix4d(state['camera_'+name].tolist()))
    # Check the native poses as well as the separate no-physics render guard.
    actual = env.capture_state()
    for key in ('joint_q', 'object_q', 'body_q'):
        if not np.allclose(actual[key], state[key], atol=2e-5, rtol=0):
            raise RuntimeError('Visual replay state differs after restore: '+key)
