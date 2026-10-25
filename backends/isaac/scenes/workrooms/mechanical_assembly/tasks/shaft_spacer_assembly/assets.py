"""L3's three true annular parts and a fixed shaft with a tapered guide tip."""
from pathlib import Path

from bench.paths import DATA_ROOT
from scenes.workrooms.mechanical_assembly import assembly_geometry as geometry
from .definition import TASK

DEFAULT_ASSET_ROOT=Path(f'{DATA_ROOT}/assets/task_parts/mechanical_assembly/shaft_spacer_assembly/v3')
PARAMETERS={**TASK['design']['candidate_geometry_m'],
    'workplate_size_xy':[.120,.120],'workplate_height':.008,
    'supply_rest_size_xy':[.102,.102],'supply_rest_height':.006,
    'polygon_segments':geometry.SEGMENTS,
    'supplemental_geometry_scope':'Workplate and flat supply-rest sizes complete unspecified supports; original shaft, bore and ring dimensions are preserved.'}
ASSET_IDS={name:'mechanical_l3_'+name for name in ('spacer','wide_ring','top_ring','shaft_fixture','supply_rest')}


def make_assets(*, split_shaft_axial=True):
    p=PARAMETERS;assets=[]
    for name,color in [('spacer','plastic_blue'),('wide_ring','plastic_yellow'),('top_ring','plastic_red')]:
        diameter,height=p[name+'_outer_diameter_height']
        part=geometry.make_asset(ASSET_IDS[name],dynamic=True)
        geometry.add_revolved(part,name,[[0.,p['part_lower_entry_diameter']/2,diameter/2],
            [p['part_entry_depth'],p['part_bore_diameter']/2,diameter/2],
            [height,p['part_bore_diameter']/2,diameter/2]],color=color)
        part.data['geometry_contract']={'kind':'through_bore_with_lower_entry_chamfer','height_m':height,
            'minimum_bore_radius_m':p['part_bore_diameter']/2,
            'radial_running_clearance_m':(p['part_bore_diameter']-p['shaft_diameter'])/2,
            'pair_contact_offset_m':2*geometry.CONTACT_OFFSET_M,'hole_filled_by_global_hull':False}
        assets.append(part)
    fixture=geometry.make_asset(ASSET_IDS['shaft_fixture'],dynamic=False,density_kg_m3=7800.)
    floor=p['workplate_height'];straight=floor+p['shaft_exposed_height']-p['shaft_taper_height'];top=floor+p['shaft_exposed_height']
    geometry.add_box(fixture,'shoulder',[*p['workplate_size_xy'],floor],center=[0.,0.,floor/2])
    geometry.add_revolved(fixture,'shaft',[[floor,0.,p['shaft_diameter']/2],
        [straight,0.,p['shaft_diameter']/2],[top,0.,p['shaft_tip_diameter']/2]],color='steel',
        split_axial=split_shaft_axial)
    fixture.data['geometry_contract']={'kind':'fixed_upward_shaft','shoulder_z_m':floor,
        'exposed_height_m':p['shaft_exposed_height'],'shaft_tip_z_m':top,
        'all_parts_stack_height_m':sum(p[n+'_outer_diameter_height'][1] for n in ('spacer','wide_ring','top_ring'))}
    assets.append(fixture)
    rest=geometry.make_asset(ASSET_IDS['supply_rest'],dynamic=False)
    geometry.add_box(rest,'flat_open_rest',[*p['supply_rest_size_xy'],p['supply_rest_height']],
        center=[0.,0.,p['supply_rest_height']/2],color='plastic_green')
    assets.append(rest)
    return [(a.asset_id,a) for a in assets]


def prepare_assets(output=DEFAULT_ASSET_ROOT, *, split_shaft_axial=True):
    parameters={**PARAMETERS,'split_shaft_axial_collision':split_shaft_axial}
    return geometry.write_assets(output,make_assets(split_shaft_axial=split_shaft_axial),generator=__name__+'.prepare_assets',parameters=parameters,
        source_paths=[__file__,Path(__file__).with_name('definition.py'),geometry.__file__])
