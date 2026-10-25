"""Explicit native solver and state transport profiles; no automatic fallback."""
CUDA_TENSORS = 'cuda_tensors_v1'
CPU_RIGID = 'cpu_rigid_v1'
GPU_SOLVER_USD_HOST = 'gpu_solver_usd_host_v1'


def resolve_profile(scene):
    """Versioned compatibility projection; never rewrites a frozen scene."""
    bundle = scene.get('native_task_bundle') or {}
    bundle_particles = bool(bundle.get('particle_sets'))
    layer_particles = any(layer.get('particle_sets') for layer in scene.get('task_layers', []))
    declared = scene.get('native_readback_profile')
    projected = (declared is None and bundle_particles
                 and bundle.get('schema') == 'native-usd-task-bundle-v1' and not layer_particles)
    profile = GPU_SOLVER_USD_HOST if projected else scene.get('native_readback_profile', CUDA_TENSORS)
    if profile not in (CUDA_TENSORS, GPU_SOLVER_USD_HOST, CPU_RIGID):
        raise ValueError('Unknown native_readback_profile')
    particles = layer_particles or bundle_particles
    if profile == CPU_RIGID and (particles or scene.get('cloth', {}).get('enabled') or scene.get('soft_bodies')):
        raise ValueError('CPU rigid profile cannot simulate particles, cloth or soft bodies')
    if not particles and profile == GPU_SOLVER_USD_HOST:
        raise ValueError('USD host profile requires declared particle readback capability')
    if particles and profile != GPU_SOLVER_USD_HOST:
        raise ValueError('Declared physical particle readback requires explicit gpu_solver_usd_host_v1; Fabric does not publish particle state')
    return {'profile': profile, 'declared_profile': declared,
            'resolution': ('native_bundle_host_readback_projection_v1' if projected
                           else 'explicit_scene_profile' if declared is not None else 'default_cuda_tensors_v1'),
            'bundle_schema': bundle.get('schema'), 'task_layer_particles': layer_particles,
            'bundle_particles': bundle_particles, 'frozen_scene_modified': False}


def select_profile(scene):
    return resolve_profile(scene)['profile']


def tensor_pipeline(sim_view, device, profile=CUDA_TENSORS):
    record = {'requested_device': str(device), 'native_tensor_device': str(sim_view.device),
        'native_tensor_device_ordinal': int(sim_view.device_ordinal),
        'native_parameter_device_ordinal': int(sim_view.param_device_ordinal),
        'cuda_context_available': sim_view.cuda_context is not None,
        'native_readback_profile': profile,
        'state_command_device': str(sim_view.device)}
    if profile == CUDA_TENSORS:
        expected = int(str(device).split(':')[1])
        if record['native_tensor_device_ordinal'] != expected or not record['cuda_context_available']:
            raise RuntimeError('Native PhysX tensor pipeline must remain on the requested CUDA device: ' + str(record))
    elif profile in (GPU_SOLVER_USD_HOST, CPU_RIGID):
        if record['native_tensor_device_ordinal'] != -1 or record['native_tensor_device'] != 'cpu':
            raise RuntimeError('Declared host transport requires native CPU state tensors: ' + str(record))
    else:
        raise ValueError('Unknown native_readback_profile')
    return record


def validate_gpu_solver_configuration(values, profile):
    """Verify the declared solver; a host transport alone never selects CPU physics."""
    if profile == CPU_RIGID:
        if (values['gpu_dynamics_enabled'] is not False or values['broadphase_type'] != 'MBP'
                or values['native_gpu_override'] == 1 or values['actual_readback_suppressed']):
            raise RuntimeError('CPU rigid profile requires native CPU dynamics and host state tensors')
        return dict(values)
    if values['gpu_dynamics_enabled'] is not True or values['broadphase_type'] != 'GPU':
        raise RuntimeError('Native readback profile must preserve GPU dynamics and broadphase')
    if values['native_gpu_override'] == 0:
        raise RuntimeError('Native PhysX override forces CPU simulation')
    expected = profile == CUDA_TENSORS
    if values['actual_readback_suppressed'] is not expected:
        raise RuntimeError('Actual PhysX readback mode differs from declared profile')
    return dict(values)
