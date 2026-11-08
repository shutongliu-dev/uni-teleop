"""Versioned, uncalibrated laundry thickness candidates.

These change the actual FEM material envelope and mass as well as the visual
skin. The reference material density is preserved; collision search distance
and the separation of the two sewn garment panels remain separate quantities.
Historical assets do not implicitly select these profiles.
"""
from copy import deepcopy


PROFILES = {
    'towel_4mm_density_v1': {
        'kind': 'towel', 'reference_thickness_m': .003,
        'physical_thickness_m': .004, 'reference_mass_kg': .108, 'mass_kg': .144},
    'shirt_16mm_density_v1': {
        'kind': 'shirt', 'reference_thickness_m': .0012,
        'physical_thickness_m': .0016, 'reference_mass_kg': .18, 'mass_kg': .24},
    'trousers_18mm_density_v1': {
        'kind': 'trousers', 'reference_thickness_m': .0012,
        'physical_thickness_m': .0018, 'reference_mass_kg': .26, 'mass_kg': .39},
}


def material_profile(profile_id, kind):
    if profile_id not in PROFILES or PROFILES[profile_id]['kind'] != kind:
        raise ValueError('Unknown versioned laundry material profile for this cloth')
    result = deepcopy(PROFILES[profile_id])
    ratio = result['physical_thickness_m']/result['reference_thickness_m']
    result.update(id=profile_id, schema='laundry_material_prior.v1',
        mass_policy='Preserve effective volume density at unchanged material rest area.',
        thickness_ratio=ratio, predicted_bending_thickness_factor=ratio**3,
        material_calibrated=False, native_physics_accepted=False,
        scope='Explicit density-preserving prior; no old native or grasp qualification is inherited.')
    return result


def towel_physics(task_id):
    """Keep each selected towel material/solver family, changing thickness only.

The new mass is bound in the asset. Surface bend stiffness is deliberately
unchanged: the native t^3 effect is measured by the candidate comparison,
not silently compensated to make a folding result resemble the old one.
"""
    if task_id not in ('towel_flattening', 'towel_folding', 'towel_stacking'):
        raise ValueError('A towel task is required')
    flatten = task_id == 'towel_flattening'
    return {'physics_dt_s': .0005, 'cloth_iterations': 64,
        'cloth_linear_damping': .25 if flatten else 2.,
        'cloth_thickness_m': .004, 'cloth_youngs_modulus_pa': 667000.,
        'cloth_poissons_ratio': .3,
        'cloth_surface_bend_stiffness': 1000. if flatten else 100.,
        'cloth_contact_offset_m': .005, 'cloth_rest_offset_m': .002,
        'cloth_collision_pair_update_frequency': 4 if flatten else 8,
        'cloth_collision_iteration_multiplier': 4 if flatten else 8,
        'cloth_speculative_ccd': True,
        'cloth_self_collision_filter_distance_m': .004}
