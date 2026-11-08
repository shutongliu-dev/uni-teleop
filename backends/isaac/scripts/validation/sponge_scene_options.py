"""Explicit scene-only numerical ablations for independent sponge instruments."""

NAMES = ('physxScene:minVelocityIterationCount', 'physxScene:maxVelocityIterationCount')


def velocity_iteration_receipt(scene, requested):
    values = {name: scene.GetAttribute(name).Get() for name in NAMES}
    if requested is not None and any(value != requested for value in values.values()):
        raise ValueError('Requested scene velocity-iteration clamp is not composed')
    return {'requested': requested, 'composed_values': values,
            'scope': 'Composed USD scene min/max clamp, not a direct native solver iteration getter.',
            'default_preserved': requested is None}


def apply_velocity_iteration_clamp(scene, requested):
    """Omission is byte-preserving; zero explicitly clamps every awake island."""
    if requested is not None and (type(requested) is not int or requested != 0):
        raise ValueError('Only the explicit zero scene velocity-iteration diagnostic is supported')
    if not scene or scene.GetTypeName() != 'PhysicsScene':
        raise ValueError('An existing PhysicsScene is required')
    if requested is not None:
        from pxr import Sdf
        scene.AddAppliedSchema('PhysxSceneAPI')
        for name in NAMES:
            scene.CreateAttribute(name, Sdf.ValueTypeNames.UInt, custom=False,
                                  variability=Sdf.VariabilityUniform).Set(requested)
    return velocity_iteration_receipt(scene, requested)
