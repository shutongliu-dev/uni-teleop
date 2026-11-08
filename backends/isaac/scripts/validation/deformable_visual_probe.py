"""Read actual native deformable skin output; never synthesize or drive it.

The simulation owner calls this at selected native boundaries. The helper requests no simulation step, node prescription, or material change.
Immediate invariance checks do not establish the absence of delayed effects;
continued raw native trajectories must also be audited.
Fabric readback is required when the active renderer uses its scene delegate.
"""
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


def _np(value):
    """Read GPU-backed torch/Warp views without numpy coercing a CUDA tensor."""
    if hasattr(value, 'detach'):
        value = value.detach().cpu().numpy()
    elif hasattr(value, 'numpy'):
        value = value.numpy()
    return np.asarray(value)


def rigid_residual(points_world, rest_local, rigid_pose_xyzw):
    """Actual skin displacement after removing the attached base's rigid pose."""
    pose = np.asarray(rigid_pose_xyzw, dtype=float)
    expected = Rotation.from_quat(pose[3:]).apply(rest_local) + pose[:3]
    delta = np.asarray(points_world, dtype=float) - expected
    distance = np.linalg.norm(delta, axis=1)
    return {'max_nonrigid_displacement_m': float(distance.max()),
            'rms_nonrigid_displacement_m': float(np.sqrt(np.mean(distance**2))),
            'p95_nonrigid_displacement_m': float(np.percentile(distance, 95)),
            'minimum_nonrigid_vertical_displacement_m': float(delta[:, 2].min()),
            'maximum_nonrigid_vertical_displacement_m': float(delta[:, 2].max())}


def _world_points(points, matrix):
    points = np.asarray(points, dtype=float)
    # USD matrices act on row vectors; translation is in the fourth row.
    return np.column_stack((points, np.ones(len(points)))) @ np.asarray(matrix, dtype=float)[:, :3]


_FABRIC_COPY_KERNELS = None


def _copy_fabric_selection(selection, row, expected_count, device):
    """Copy Access.Read Fabric buffers into owned Warp arrays, then host memory."""
    import warp as wp
    global _FABRIC_COPY_KERNELS
    if _FABRIC_COPY_KERNELS is None:
        @wp.kernel(enable_backward=False)
        def read_metadata(src: wp.fabricarrayarray(dtype=wp.vec3f),
                          matrices: wp.fabricarray(dtype=wp.mat44d), row: int,
                          count: wp.array(dtype=wp.int32), matrix: wp.array(dtype=wp.mat44d)):
            values = src[row]
            count[0] = values.shape[0]
            matrix[0] = matrices[row]

        @wp.kernel(enable_backward=False)
        def copy_points(src: wp.fabricarrayarray(dtype=wp.vec3f), row: int,
                        destination: wp.array(dtype=wp.vec3f)):
            j = wp.tid()
            destination[j] = src[row, j]
        _FABRIC_COPY_KERNELS = (read_metadata, copy_points)
    points = wp.fabricarrayarray(data=selection, attrib='points')
    matrices = wp.fabricarray(data=selection, attrib='omni:fabric:worldMatrix')
    count = wp.empty(1, dtype=wp.int32, device=device)
    matrix = wp.empty(1, dtype=wp.mat44d, device=device)
    wp.launch(_FABRIC_COPY_KERNELS[0], dim=1, inputs=[points, matrices, row],
              outputs=[count, matrix], device=device)
    # Only these newly allocated outputs are exported to NumPy, never VtArray.
    actual_count = int(count.numpy()[0])
    if actual_count != expected_count:
        raise RuntimeError(f'Native Fabric point count {actual_count} differs from expected {expected_count}')
    copied = wp.empty(actual_count, dtype=wp.vec3f, device=device)
    wp.launch(_FABRIC_COPY_KERNELS[1], dim=actual_count, inputs=[points, row],
              outputs=[copied], device=device)
    return np.array(copied.numpy(), copy=True), np.array(matrix.numpy()[0], copy=True)


def _fabric_points(rt_stage, path, device, expected_count):
    """Request read access explicitly; VtArray buffer export may mark points dirty.

    Fixed USDRT 7.6.3 records changes when a writable Fabric span is obtained,
    even without a value change. SelectPrims Access.Read avoids that span. An
    independent listener also rejects any point/transform changes caused here.
    """
    from usdrt import Rt, Sdf, Usd
    tracker = Rt.ChangeTracker(rt_stage)
    tracker.TrackAttribute('points')
    tracker.TrackAttribute('omni:fabric:worldMatrix')
    tracker.ClearChanges()  # Clears this new listener only, never PhysX events.
    selection = rt_stage.SelectPrims(require_attrs=[
        (Sdf.ValueTypeNames.Point3fArray, 'points', Usd.Access.Read),
        (Sdf.ValueTypeNames.Matrix4d, 'omni:fabric:worldMatrix', Usd.Access.Read)],
        device=str(device), want_paths=True)
    paths = [str(value) for value in selection.GetPaths()]
    if paths.count(path) != 1:
        raise RuntimeError('Expected exactly one native Fabric point prim: '+path)
    points, matrix = _copy_fabric_selection(selection, paths.index(path), expected_count, str(device))
    changed = [str(value) for value in tracker.GetAllChangedAttributes()]
    if changed:
        raise RuntimeError('Native Fabric readback marked tracked attributes changed: '+repr(changed))
    if points.shape != (expected_count, 3) or not np.isfinite(points).all():
        raise RuntimeError('Invalid native Fabric point array')
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise RuntimeError('Invalid native Fabric world matrix: '+path)
    prim = rt_stage.GetPrimAtPath(path)
    return points, _world_points(points, matrix), {
        'native_deformable_tag': bool(prim.HasAttribute('Deformable')),
        'transport': 'native_fabric_access_read_owned_warp_copy',
        'requested_access': 'Read', 'device': str(device),
        'point_count': expected_count, 'tracked_attribute_changes': changed,
        'world_matrix': matrix.tolist()}


def _sync_error(actual_world, tensor_world):
    actual, tensor = np.asarray(actual_world), np.asarray(tensor_world)
    if actual.shape != tensor.shape or actual.ndim != 2 or actual.shape[1] != 3:
        raise RuntimeError('Native point output topology differs from the physics tensor')
    if not np.isfinite(actual).all() or not np.isfinite(tensor).all():
        raise RuntimeError('Nonfinite native point output')
    return float(np.max(np.linalg.norm(actual-tensor, axis=1)))


def _require_current_output(*, fabric_renderer, usd_error, fabric_error):
    """Select the active renderer transport, never whichever error is smaller."""
    error = fabric_error if fabric_renderer else usd_error
    backend = 'Fabric' if fabric_renderer else 'USD'
    if error is None or not np.isfinite(error) or error > 2e-6:
        raise RuntimeError(f'Stale {backend} simulation output; max tensor error={error!r} m')
    return backend


def _flush_native_fabric(interface, *, physics_dt, current_time):
    """Match fixed Isaac 6.1 SimulationContext.render's (dt, time) order.

    Both arguments have seconds units, so swapping them can evade visible-state
    guards while supplying a phase-length timestep to the native output path.
    This is an output flush, not a request to advance simulation time.
    """
    dt, time = float(physics_dt), float(current_time)
    if not np.isfinite([dt, time]).all() or dt <= 0 or time < 0:
        raise ValueError('Native Fabric flush requires a positive dt and nonnegative current time')
    interface.force_update(dt, time)
    return {'physics_dt_s': dt, 'current_time_s': time,
            'positional_argument_order': ['physics_dt_s', 'current_time_s']}


class NativeSkinProbe:
    """Archive native USD and Fabric skins with tensor synchronization checks."""
    def __init__(self, stage, volume_spec, output):
        from pxr import UsdGeom
        self.stage, self.spec = stage, volume_spec
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=False)
        self.bindings = list(volume_spec['visual_bindings'])
        self.rest = {}
        for binding in self.bindings:
            prim = stage.GetPrimAtPath(binding['prim_path'])
            bind = prim.GetAttribute('deformablePose:default:omniphysics:points')
            if not bind or bind.Get() is None:
                raise ValueError('Native skin probe requires an explicit frozen bind pose')
            self.rest[binding['prim_path']] = np.asarray(bind.Get(), dtype=np.float64)
            if len(self.rest[binding['prim_path']]) != len(UsdGeom.Mesh(prim).GetPointsAttr().Get()):
                raise ValueError('Skin bind pose topology differs from the visible mesh')
        self.rows = []

    def sample(self, label, world, volume_view, pad_view):
        """Call after World.step; no World.step/render/reset occurs here."""
        import carb
        import omni.physx
        import omni.usd
        from pxr import UsdGeom
        from bench.isaacsim.release import digest, dump

        if not label.replace('_', '').isalnum():
            raise ValueError('Visual sample label must be a simple identifier')
        path = self.output/(label+'.npz')
        if path.exists():
            raise ValueError('Never overwrite a visual probe sample')
        raw_q = volume_view.get_simulation_nodal_positions()
        device = str(getattr(raw_q, 'device', 'cpu'))
        q_before = _np(raw_q).copy()
        qd_before = _np(volume_view.get_simulation_nodal_velocities()).copy()
        pad_before = _np(pad_view.get_transforms()).copy()
        pad_velocity_before = _np(pad_view.get_velocities()).copy()
        step, time = int(world.current_time_step_index), float(world.current_time)
        arrays = {'simulation_nodes_world_m': q_before[0], 'simulation_node_velocities_m_s': qd_before[0],
                  'rigid_pose_xyzw': pad_before[0], 'rigid_velocity': pad_velocity_before[0]}
        row = {'label': label, 'native_step': step, 'time_s': time, 'status': 'reading',
            'native_output_flush_only': True, 'simulation_writes': 0, 'tensor_device': device,
            'skins': [], 'scope': 'Native visual readback only; immediate state checks do not rule out delayed effects. Audit subsequent raw physics steps.'}
        try:
            settings = carb.settings.get_settings()
            output_settings = {key: settings.get(key) for key in (
                '/app/useFabricSceneDelegate', '/physics/fabricEnabled', '/physics/fabricUpdatePoints',
                '/physics/updateToUsd', '/physics/updateParticlesToUsd', '/physics/updateVelocitiesToUsd')}
            row['native_output_settings'] = output_settings
            fabric_required = bool(output_settings['/app/useFabricSceneDelegate'])
            row['fabric_renderer'] = fabric_required
            rt_stage = None
            if fabric_required:
                if not output_settings['/physics/fabricEnabled'] or not output_settings['/physics/fabricUpdatePoints']:
                    raise RuntimeError('Active Fabric renderer lacks native deformable point output')
                from omni.physxfabric import get_physx_fabric_interface
                from usdrt import Usd as RtUsd
                # enable_fabric(True) disables USD point output. Flush/read the
                # actual GPU transport instead of expecting that cache to move.
                row['fabric_flush_arguments'] = _flush_native_fabric(
                    get_physx_fabric_interface(), physics_dt=world.get_physics_dt(), current_time=time)
                rt_stage = RtUsd.Stage.Attach(omni.usd.get_context().get_stage_id())
            else:
                omni.physx.get_physx_interface().update_transformations(False, True)
            cache = UsdGeom.XformCache()
            sim_prim = self.stage.GetPrimAtPath(self.spec['simulation_mesh_path'])
            usd_sim = _world_points(UsdGeom.PointBased(sim_prim).GetPointsAttr().Get(),
                                    cache.GetLocalToWorldTransform(sim_prim))
            arrays['simulation_usd_world_m'] = usd_sim
            usd_error = _sync_error(usd_sim, q_before[0])
            row['usd_simulation_to_tensor_max_error_m'] = usd_error
            fabric_error = None
            if rt_stage is not None:
                local, actual_world, receipt = _fabric_points(rt_stage, self.spec['simulation_mesh_path'], device, len(q_before[0]))
                arrays['simulation_fabric_local_m'], arrays['simulation_fabric_world_m'] = local, actual_world
                row['simulation_fabric'] = receipt
                fabric_error = _sync_error(actual_world, q_before[0])
                row['fabric_simulation_to_tensor_max_error_m'] = fabric_error
            row['authoritative_output_backend'] = _require_current_output(
                fabric_renderer=fabric_required, usd_error=usd_error, fabric_error=fabric_error)
            for index, binding in enumerate(self.bindings):
                skin_path = binding['prim_path']
                prim = self.stage.GetPrimAtPath(skin_path)
                local = np.array(UsdGeom.PointBased(prim).GetPointsAttr().Get(), copy=True)
                world_points = _world_points(local, cache.GetLocalToWorldTransform(prim))
                arrays[f'skin_{index}_usd_local_m'], arrays[f'skin_{index}_usd_world_m'] = local, world_points
                skin = {'prim_path': skin_path, 'point_count': len(local),
                    'usd': rigid_residual(world_points, self.rest[skin_path], pad_before[0])}
                row['skins'].append(skin)
                if rt_stage is not None:
                    actual, actual_world, receipt = _fabric_points(rt_stage, skin_path, device, len(self.rest[skin_path]))
                    arrays[f'skin_{index}_fabric_local_m'], arrays[f'skin_{index}_fabric_world_m'] = actual, actual_world
                    if actual.shape != self.rest[skin_path].shape:
                        raise RuntimeError('Native Fabric skin topology differs from its bind pose')
                    skin['fabric'] = {**receipt,
                        'usd_world_max_difference_m': _sync_error(actual_world, world_points),
                        **rigid_residual(actual_world, self.rest[skin_path], pad_before[0])}
                    if not receipt['native_deformable_tag']:
                        raise RuntimeError('Fabric skin lacks the native Deformable tag; lazy-loaded USD is insufficient')
            row.update(status='read', native_visual_output_synchronized=True)
            self.rows.append(row)
            return row
        except Exception as exc:
            row.update(status='failed', native_visual_output_synchronized=False,
                       error=f'{type(exc).__name__}: {exc}')
            raise
        finally:
            unchanged = (int(world.current_time_step_index) == step and float(world.current_time) == time
                and np.array_equal(q_before, _np(volume_view.get_simulation_nodal_positions()))
                and np.array_equal(qd_before, _np(volume_view.get_simulation_nodal_velocities()))
                and np.array_equal(pad_before, _np(pad_view.get_transforms()))
                and np.array_equal(pad_velocity_before, _np(pad_view.get_velocities())))
            row['physical_state_bitwise_unchanged'] = bool(unchanged)
            if not unchanged:
                row.update(status='failed', native_visual_output_synchronized=False,
                           error='A read-only native visual flush changed physical state')
            np.savez_compressed(path, **arrays)
            row.update(array_path=str(path), array_sha256=digest(path))
            dump(self.output/(label+'.json'), row)
            if not unchanged:
                raise RuntimeError(row['error'])
