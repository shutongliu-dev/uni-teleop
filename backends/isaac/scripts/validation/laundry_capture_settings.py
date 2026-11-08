"""Explicit per-operation capture choices for measured regrasp experiments."""
from __future__ import annotations

from scripts.validation.laundry_closure_measurement import (
    PROFILES as CLOSURE_MEASUREMENT_PROFILES, validate_closure_measurement)
from scripts.validation.laundry_contact_observation import validate_normal_profile,validate_observation_policy
from scripts.validation.laundry_contact_stability import PROFILE as PRACTICAL_PROFILE
from scripts.validation.laundry_contact_engagement import PROFILE as ENGAGED_PROFILE

OPTIONAL_KEYS=('closure_measurement_profile','contact_normal_profile','contact_closure_observation_policy',
               'contact_witness_profile')


DEFAULTS = {
    'capture_profile': 'strict_v1',
    'contact_closure_stop': False,
    'contact_closure_target_policy': 'reduce_preload_v1',
    'contact_closure_geometry_profile': 'shared_edge_v1',
}
PROFILES = ('strict_v1', 'local_assist_v2', 'adjacent_patch_v3', 'angled_adjacent_patch_v4',
            'loaded_adjacent_patch_v5', PRACTICAL_PROFILE, ENGAGED_PROFILE)


def validate_loaded_contact(profile, stop, policy, geometry, material_mode):
    """The larger servo-error allowance is confined to a stopped local grasp."""
    if profile == 'loaded_adjacent_patch_v5' and (
            stop is not True or policy != 'hold_command_v2' or
            geometry != 'adjacent_patch_v3' or material_mode != 'captured_patch'):
        raise ValueError('Loaded capture requires enabled stop, hold_command_v2, '
                         'adjacent_patch_v3 stop geometry and captured_patch material')
    if profile in (PRACTICAL_PROFILE, ENGAGED_PROFILE) and (
            stop is not True or policy != 'hold_command_v2' or
            geometry != 'adjacent_patch_v3' or material_mode != 'captured_patch'):
        raise ValueError('Practical surface hold requires enabled stop, hold_command_v2, '
                         'original adjacent_patch_v3 geometry and captured_patch material')


def _validate(values, material_mode):
    validate_normal_profile(values.get('contact_normal_profile','facet_v1'))
    from scripts.validation.laundry_contact_observation import validate_witness_profile
    validate_witness_profile(values.get('contact_witness_profile','centroid_v1'),
                             values.get('contact_normal_profile','facet_v1'))
    validate_observation_policy(values.get('contact_closure_observation_policy','close_only_v1'),
        values['contact_closure_stop'],values['contact_closure_target_policy'])
    if material_mode not in ('captured_patch','shared_edge','captured_pivot_v1'):
        raise ValueError('Unknown local assistance material mode')
    if values.get('closure_measurement_profile','strict_v1') not in CLOSURE_MEASUREMENT_PROFILES:
        raise ValueError('Unknown explicit closure measurement profile')
    if values['capture_profile'] not in PROFILES:
        raise ValueError('Unknown explicit capture profile')
    if values['capture_profile'] in (PRACTICAL_PROFILE, ENGAGED_PROFILE) and (
            values.get('contact_normal_profile') != 'distance_gradient_v1'
            or values.get('contact_witness_profile') not in ('facet_regions_v1', 'surface_regions_v2')):
        raise ValueError('Practical hold requires finite-area true distance-gradient witnesses')
    if type(values['contact_closure_stop']) is not bool:
        raise ValueError('contact_closure_stop must be an explicit boolean')
    policy = values['contact_closure_target_policy']
    if (policy not in ('reduce_preload_v1', 'hold_command_v2') or
            (not values['contact_closure_stop'] and policy != 'reduce_preload_v1')):
        raise ValueError('Explicit contact target policy requires enabled contact closure stop')
    geometry = values['contact_closure_geometry_profile']
    validate_loaded_contact(values['capture_profile'], values['contact_closure_stop'],
                            policy, geometry, material_mode)
    if geometry not in ('shared_edge_v1', 'adjacent_patch_v3', 'angled_adjacent_patch_v4'):
        raise ValueError('Unknown explicit closure geometry profile')
    if geometry != 'shared_edge_v1' and (
            not values['contact_closure_stop'] or material_mode not in ('captured_patch','captured_pivot_v1') or
            values['capture_profile'] != geometry and not (
                values['capture_profile'] in ('loaded_adjacent_patch_v5', PRACTICAL_PROFILE, ENGAGED_PROFILE)
                and geometry == 'adjacent_patch_v3')):
        raise ValueError('Bridge closure geometry requires enabled stop, matching capture and captured-patch material')
    if values['capture_profile'] == 'angled_adjacent_patch_v4' and (
            material_mode not in ('captured_patch','captured_pivot_v1') or
            (values['contact_closure_stop'] and geometry != 'angled_adjacent_patch_v4')):
        raise ValueError('Angled capture requires the full witnessed local patch and matching enabled stop geometry')
    return values


def phase_capture_settings(options, index=None):
    """Keep the validated baseline exact; override only an explicit operation.

    This changes neither the operation duration nor its robot/material identity.
    A fresh assist and a fresh actual capture window are still required each time.
    """
    mode = options.get('attachment_material_mode', 'captured_patch')
    baseline = {key: options.get(key, default) for key, default in DEFAULTS.items()}
    # Preserve the exact legacy dictionary when the orthogonal profile is absent.
    for key in OPTIONAL_KEYS:
        if key in options:baseline[key]=options[key]
    baseline = _validate(baseline, mode)
    if index is None:
        return baseline
    if type(index) is not int or index < 0:
        raise ValueError('A nonnegative execution phase is required')
    override = options.get('phases', {}).get(str(index), {}).get('capture_settings', {})
    if not isinstance(override, dict) or set(override) - (set(DEFAULTS)|set(OPTIONAL_KEYS)):
        raise ValueError('Only explicit gate and closure-stop settings can vary per phase')
    return _validate({**baseline, **override}, mode)


def planned_capture_openings(options, index, settings):
    """Bind the narrow profile to explicit per-hand targets before native startup."""
    profile=settings.get('closure_measurement_profile','strict_v1')
    if profile=='strict_v1':return {}
    phase=options.get('phases',{}).get(str(index),{})
    sides=phase.get('capture_order',phase.get('sides'))
    if (not isinstance(sides,list) or not sides or any(s not in ('left','right') for s in sides)
            or len(set(sides))!=len(sides)):
        raise ValueError('Narrow closure requires explicit capture_order or sides for every phase')
    openings={}
    for side in sides:
        opening=phase.get('hands',{}).get(side,{}).get('open_distance_m')
        validate_closure_measurement(profile,opening)
        openings[side]=float(opening)
    return openings
