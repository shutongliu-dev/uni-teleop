"""Task4's explicit 0.25 ms TGS external-force integration selection.

This is a scene-wide integrator choice. The task assets, material parameters,
robot controls and referee thresholds retain their existing definitions.
Release source provenance hashes this declaration; native evidence belongs to
the resulting frozen release and is not inherited from a diagnostic probe.
"""


PROFILE_ID = 'tower_extraction_tgs_external_forces_v2'


def task_physics_profile():
    """Return a fresh declaration for the task's frozen physics profile."""
    return {
        'profile_id': PROFILE_ID,
        'physics_dt_s': .00025,
        'isaacsim_options': {'physics_dt_s': .00025,
                            'enable_external_forces_every_iteration': True},
        'integrator': {
            'solver_type': 'TGS',
            'control_dt_s': .02,
            'external_force_application':
                'Apply gravity and other external forces during each TGS iteration.',
            'usd_attribute': 'physxScene:enableExternalForcesEveryIteration',
            'scope': 'PhysicsScene; all rigid actors in the task scene.',
        },
        'selection': 'Explicit Task4 0.25 ms TGS external-force integration profile',
        'provenance': {
            'source_module': __name__,
            'reason': 'Select 0.25 ms integration with per-iteration external forces for the 12-layer extraction tower.',
        },
        'qualification_transferred': False,
    }
