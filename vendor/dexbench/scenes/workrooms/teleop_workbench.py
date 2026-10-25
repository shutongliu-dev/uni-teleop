"""Versioned workbench-only practice recipes over immutable released assets.

Selection removes entire scenery prims before native collision registration;
it does not replace task meshes, relocate actors or change physical parameters.
"""
from copy import deepcopy


VERSION = 'teleop_workbench_v1'
GEOMETRY = {
    'mechanical_assembly': ('table_*',),
    'laboratory': ('island_front_*', 'island_rear_*', 'island_end_*',
                   'rear_lab_counter', 'rear_lab_cabinet_*'),
    'kitchen': ('kitchen_bar_end*', 'kitchen_bar_outer_rail'),
    'study': ('desk_top_frame_*', 'desk_pedestal_*', 'rear_shelf_toe',
              'rear_shelf_back', 'rear_shelf_side_*', 'rear_shelf_level_*',
              'rear_shelf_edge_*', 'rear_shelf_divider', 'rear_shelf_lower_*',
              'rear_shelf_pin_*'),
    'card_room': ('table_*',),
    'living_room': ('table_leg_*', 'table_apron_*'),
}
COOKING_GEOMETRY = ('kitchen_counter_*', 'kitchen_base_*', 'kitchen_cabinet_side_*',
                    'kitchen_cooktop', 'kitchen_hob_*', 'kitchen_oven_case')


def apply(scene, definition):
    """Return a separate practice variant; preserve the caller's release recipe."""
    family = definition['family']
    result = deepcopy(scene)
    selection = ['room_floor', 'worktop', *GEOMETRY[family]]
    if family == 'kitchen' and definition['layout']['workspace'] == 'cooking':
        selection.extend(COOKING_GEOMETRY)
    if family == 'living_room' and definition['task_id'] == 'remote_operation':
        selection.extend(('television_*', 'living_tv_*'))
    result['workroom']['geometry_selection'] = selection
    # Removing the roof exposes the task to the original outdoor sun. Use the
    # existing softer lighting preset instead of overexposing white objects.
    result['isaacsim']['workroom_lighting'] = {
        'schema': 'workroom-lighting-v1', 'family': family, 'profile': 'warm'}
    if family == 'kitchen' and definition['layout']['workspace'] == 'bar':
        result['cameras']['overview'].update(
            eye=[.65, -2.65, 2.05], target=[.65, -.55, 1.03], focal_length_mm=24.)
        result['cameras']['wrist'].update(
            offset_xyz_m=[-.14, -.10, .025], aim_target_m=[.50, -.90, .90],
            standoff_m=0.)
    bundle = result.get('native_task_bundle')
    if bundle and family in ('kitchen', 'laboratory'):
        # These authored groups contain optional static dressing, never task
        # fixtures. The backend rejects dynamics in an excluded subtree.
        group = 'Context' if family == 'kitchen' else 'DesktopDressing'
        bundle['excluded_static_roots'] = [bundle['root_prim_path'] + '/' + group]
    result['teleop_scene_variant'] = {
        'version': VERSION, 'family': family, 'task': definition['task_id'],
        'source_release': deepcopy(definition['source_release']),
        'scope': 'Workbench-only practice; altered scenery, not the original benchmark environment.',
    }
    result['description'] = f'{VERSION}: {family}/{definition["task_id"]}'
    return result
