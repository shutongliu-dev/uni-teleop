"""Task-owned static geometry for the loose-bolt fastener candidate.

The threaded SHARPA USDs remain the source of the bolt, driver and fixture.
This library only adds a shallow entry funnel and a low supply rest, so the
task can expose a loose bolt without changing the fixed room or source USDs.
"""
from pathlib import Path

from bench.paths import DATA_ROOT
from scenes.workrooms.mechanical_assembly import assembly_geometry as geometry


DEFAULT_ASSET_ROOT = Path(
    f'{DATA_ROOT}/assets/task_parts/mechanical_assembly/'
    'fastener_assembly/loose_bolt_v5')
ASSET_IDS = {
    'entry_guide': 'mechanical_l4_thread_entry_guide',
    'supply_rest': 'mechanical_l4_bolt_supply_rest',
    'tool_rest': 'mechanical_l4_tool_supply_rest',
}
PARAMETERS = {
    'entry_guide_height_m': .006,
    'entry_guide_inner_radius_bottom_m': .013,
    'entry_guide_inner_radius_top_m': .019,
    'entry_guide_outer_radius_m': .026,
    'supply_rest_size_xy_m': [.090, .180],
    'supply_rest_height_m': .006,
    'supply_rest_corner_chamfer_m': .002,
    'bolt_nest_inner_radius_m': .023,
    'bolt_nest_outer_radius_m': .032,
    'tool_rail_length_m': .220,
    'tool_rail_offset_x_m': .024,
    'tool_rail_width_m': .008,
    'tool_rail_height_m': .014,
    'polygon_segments': geometry.SEGMENTS,
    'design_scope': (
        'A shallow annular lead-in around the authored threaded fixture and a '
        'shallow annular bolt nest and two tool side rails; no thread, snap or magnetic constraint.'),
}


def make_assets():
    p = PARAMETERS
    guide = geometry.make_asset(ASSET_IDS['entry_guide'], dynamic=False,
                                density_kg_m3=7800.)
    geometry.add_revolved(
        guide, 'entry_funnel', [
            [0., p['entry_guide_inner_radius_bottom_m'], p['entry_guide_outer_radius_m']],
            [p['entry_guide_height_m'], p['entry_guide_inner_radius_top_m'],
             p['entry_guide_outer_radius_m']],
        ], color='steel')
    guide.data['geometry_contract'] = {
        'kind': 'shallow_annular_thread_entry_funnel',
        'height_m': p['entry_guide_height_m'],
        'inner_radius_bottom_m': p['entry_guide_inner_radius_bottom_m'],
        'inner_radius_top_m': p['entry_guide_inner_radius_top_m'],
        'outer_radius_m': p['entry_guide_outer_radius_m'],
        'purpose': 'visual and passive coarse centering before the authored thread mouth',
        'thread_constraint': False,
        'pose_snap': False,
    }

    def make_rest(asset_id):
        return geometry.make_asset(asset_id, dynamic=False, density_kg_m3=1200.)

    rest = make_rest(ASSET_IDS['supply_rest'])
    geometry.add_revolved(rest, 'bolt_nest', [
        [0., p['bolt_nest_inner_radius_m'], p['bolt_nest_outer_radius_m']],
        [p['supply_rest_height_m'], p['bolt_nest_inner_radius_m'], p['bolt_nest_outer_radius_m']],
    ], color='plastic_blue')
    rest.data['geometry_contract'] = {
        'kind': 'shallow_annular_bolt_nest',
        'size_xy_m': [2*p['bolt_nest_outer_radius_m']]*2,
        'height_m': p['supply_rest_height_m'],
        'bolt_nest_inner_radius_m': p['bolt_nest_inner_radius_m'],
        'bolt_nest_outer_radius_m': p['bolt_nest_outer_radius_m'],
        'purpose': 'passive lateral guidance around the bolt head; the original table supports the head through the open centre',
        'vertical_support': 'original_workroom_table',
        'task_object_remains_independent': True,
        'thread_constraint': False,
        'pose_snap': False,
    }
    tool_rest = make_rest(ASSET_IDS['tool_rest'])
    size_x, size_y = p['supply_rest_size_xy_m']
    geometry.add_box(tool_rest, 'tool_supply_base',
                     [size_x, size_y, p['supply_rest_height_m']],
                     center=[0., 0., p['supply_rest_height_m'] / 2.],
                     color='plastic_blue')
    for sign in (-1., 1.):
        geometry.add_box(tool_rest, f'tool_side_rail_{"left" if sign < 0 else "right"}',
                         [p['tool_rail_width_m'], p['tool_rail_length_m'], p['tool_rail_height_m']],
                         center=[sign*p['tool_rail_offset_x_m'], 0.,
                                 p['supply_rest_height_m'] + p['tool_rail_height_m']/2.],
                         color='plastic_blue')
    tool_rest.data['geometry_contract'] = {
        'kind': 'tool_supply_tray_with_side_rails',
        'size_xy_m': [size_x, size_y], 'height_m': p['supply_rest_height_m'],
        'tool_rail_length_m': p['tool_rail_length_m'],
        'task_object_remains_independent': True, 'pose_snap': False,
    }
    return [(asset.asset_id, asset) for asset in (guide, rest, tool_rest)]


def prepare_assets(output=DEFAULT_ASSET_ROOT):
    return geometry.write_assets(
        output, make_assets(),
        generator=__name__ + '.prepare_assets', parameters=PARAMETERS,
        source_paths=[__file__, geometry.__file__])
