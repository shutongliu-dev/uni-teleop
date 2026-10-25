"""L2's two broad-headed pegs and genuine chamfered, bottom-seated sockets."""
from pathlib import Path

from bench.paths import DATA_ROOT
from scenes.workrooms.mechanical_assembly import assembly_geometry as geometry
from .definition import TASK

DEFAULT_ASSET_ROOT = Path(f'{DATA_ROOT}/assets/task_parts/mechanical_assembly/peg_insertion/v2')
PARAMETERS = {
    **TASK['design']['candidate_geometry_m'],
    'peg_grip_height': .018, 'workplate_size_xy': [.120, .260], 'workplate_height': .008,
    'socket_outer_diameter': .090, 'supply_cradle_size_xy': [.072, .072],
    'supply_cradle_floor_height': .006, 'supply_cradle_wall_height': .006,
    'supply_cradle_bore_diameter': .036, 'supply_cradle_outer_diameter': .054,
    'polygon_segments': geometry.SEGMENTS,
    'supplemental_geometry_scope': 'Grip height, base thickness, outer walls and supply rests complete unspecified design surfaces; original task dimensions are preserved.',
}
ASSET_IDS = {'peg':'mechanical_l2_peg','workplate':'mechanical_l2_socket_workplate',
             'cradle':'mechanical_l2_supply_cradle'}


def make_assets():
    p=PARAMETERS
    peg=geometry.make_asset(ASSET_IDS['peg'],dynamic=True)
    stem_top=p['peg_length']-p['peg_grip_height']
    geometry.add_revolved(peg,'stem',[[0.,0.,p['peg_diameter']/2],
                                     [stem_top,0.,p['peg_diameter']/2]])
    geometry.add_revolved(peg,'grip',[[stem_top,0.,p['peg_grip_diameter']/2],
                                     [p['peg_length'],0.,p['peg_grip_diameter']/2]],color='plastic_yellow')
    plate=geometry.make_asset(ASSET_IDS['workplate'],dynamic=False,density_kg_m3=7800.)
    geometry.add_box(plate,'solid_bottom_seats',[*p['workplate_size_xy'],p['workplate_height']],
                     center=[0.,0.,p['workplate_height']/2])
    for index,y in enumerate([-.5*p['socket_center_spacing'],.5*p['socket_center_spacing']]):
        bottom=p['workplate_height'];straight=bottom+p['straight_seating_depth'];top=straight+p['socket_guide_depth']
        geometry.add_revolved(plate,f'socket_{index}',
            [[bottom,p['socket_bore_diameter']/2,p['socket_outer_diameter']/2],
             [straight,p['socket_bore_diameter']/2,p['socket_outer_diameter']/2],
             [top,p['socket_entry_diameter']/2,p['socket_outer_diameter']/2]],offset=[0.,y,0.],color='steel')
    plate.data['geometry_contract']={'kind':'two_blind_sockets','bottom_seat_z_m':p['workplate_height'],
        'socket_centers_xy_m':[[0.,-.5*p['socket_center_spacing']],[0.,.5*p['socket_center_spacing']]],
        'radial_running_clearance_m':(p['socket_bore_diameter']-p['peg_diameter'])/2,
        'pair_contact_offset_m':2*geometry.CONTACT_OFFSET_M,'hole_filled_by_global_hull':False}
    cradle=geometry.make_asset(ASSET_IDS['cradle'],dynamic=False,density_kg_m3=1200.)
    floor=p['supply_cradle_floor_height']
    geometry.add_box(cradle,'floor',[*p['supply_cradle_size_xy'],floor],center=[0.,0.,floor/2],color='plastic_green')
    geometry.add_revolved(cradle,'open_wall',[[floor,p['supply_cradle_bore_diameter']/2,p['supply_cradle_outer_diameter']/2],
        [floor+p['supply_cradle_wall_height'],p['supply_cradle_bore_diameter']/2,p['supply_cradle_outer_diameter']/2]],color='plastic_green')
    return [(asset.asset_id,asset) for asset in (peg,plate,cradle)]


def prepare_assets(output=DEFAULT_ASSET_ROOT):
    return geometry.write_assets(output,make_assets(),generator=__name__+'.prepare_assets',parameters=PARAMETERS,
        source_paths=[__file__,Path(__file__).with_name('definition.py'),geometry.__file__])
