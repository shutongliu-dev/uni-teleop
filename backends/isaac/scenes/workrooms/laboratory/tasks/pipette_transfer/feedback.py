"""Frozen geometry bindings for the optional conserved liquid payload adapter."""
import math


def payload_configuration(metadata):
    hydraulic = metadata['task_parameters']['hydraulics']
    if hydraulic['schema'] != 'laboratory_piston_finite_volume.v2':
        raise ValueError('Wet payload requires the explicit v2 liquid contract')
    masses = {row['id']: row['mass_kg'] for row in metadata['objects']}
    bindings = []
    for name, vessel in hydraulic['containers'].items():
        bindings.append({
            'body_id': name,
            'volume_path': ['source_ml'] if name == hydraulic['source_id'] else ['receivers_ml', name],
            'dry_mass_kg': masses[name],
            'radius_m': vessel['radius_m'],
            'floor_center_m': [0., 0., vessel['floor_z_m']],
            'axis_body': [0., 0., 1.],
            'maximum_volume_ml': vessel['capacity_ml'],
        })
    bindings.append({
        'body_id': hydraulic['tool_id'], 'volume_path': ['tool_ml'],
        'dry_mass_kg': masses[hydraulic['tool_id']],
        'radius_m': math.sqrt(hydraulic['piston_area_m2'] / math.pi),
        'floor_center_m': [0., 0., .0265], 'axis_body': [0., 0., 1.],
        'maximum_volume_ml': hydraulic['piston_area_m2'] * hydraulic['stroke_m'] * 1.e6,
    })
    return {
        'schema': 'reduced-liquid-payload-v1',
        'state_schema': hydraulic['schema'],
        'density_kg_m3': hydraulic['density_kg_m3'],
        'initial_total_ml': hydraulic['initial_total_ml'],
        'unbound_volume_paths': [['in_flight_ml'], ['cumulative_spill_ml']],
        'bindings': bindings,
    }
