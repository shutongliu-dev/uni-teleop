"""Explicit contact-normal and closure-observation policies; no state changes."""

NORMAL_PROFILES = ('facet_v1', 'distance_gradient_v1')
OBSERVATION_POLICIES = ('close_only_v1', 'close_and_hold_v1')


def validate_normal_profile(profile):
    if profile not in NORMAL_PROFILES:
        raise ValueError('Unknown explicit contact normal profile')


def validate_observation_policy(policy, enabled, target_policy):
    if policy not in OBSERVATION_POLICIES:
        raise ValueError('Unknown explicit closure observation policy')
    if policy == 'close_and_hold_v1' and (enabled is not True or target_policy != 'hold_command_v2'):
        raise ValueError('Hold-period observation requires enabled stop and exact hold_command_v2')


WITNESS_PROFILES = ('centroid_v1', 'facet_regions_v1', 'surface_regions_v2')


def validate_witness_profile(profile, normal_profile):
    if profile not in WITNESS_PROFILES:
        raise ValueError('Unknown explicit contact witness profile')
    if profile == 'facet_regions_v1' and normal_profile != 'distance_gradient_v1':
        raise ValueError('Facet-region witnesses require exact distance-gradient normals')
    if profile == 'surface_regions_v2' and normal_profile != 'distance_gradient_v1':
        raise ValueError('Surface-region witnesses require exact distance-gradient normals')
