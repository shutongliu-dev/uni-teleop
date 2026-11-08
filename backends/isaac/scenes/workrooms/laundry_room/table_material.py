"""Explicit laundry tabletop prior; historical scenes do not select it implicitly."""
from bench.simulation.isaacsim_backend.workroom_materials import material_options

PROFILE = 'laundry_table_friction_045_v1'
OPTION = 'workroom_surface_materials'


def table_material_options(profile=PROFILE):
    if profile != PROFILE:
        raise ValueError('Unknown laundry table material profile')
    return material_options({'schema': 'workroom-surface-materials-v1', 'surfaces': {
        'table': {'static_friction': .45, 'dynamic_friction': .45, 'restitution': 0.,
                  'friction_combine': 'average', 'restitution_combine': 'average',
                  'parameter_source': PROFILE + ': explicit uncalibrated simulation prior; '
                      'bind the laundry scene legacy table.friction=0.45 to its actual referenced tabletop.'}}})
