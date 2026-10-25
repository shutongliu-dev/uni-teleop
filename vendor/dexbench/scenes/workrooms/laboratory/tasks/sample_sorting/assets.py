"""Four sealed sample bottles and two physical classification trays."""
from scenes.workrooms.laboratory.object_geometry import Author, box, cylinder, pose, tray_parts
from scenes.workrooms.laboratory.visual_details import add_task_visual_details


def author_task(stage, root_path='/World/LaboratoryTask', *, presentation=False):
    a=Author(stage,'sample_sorting',root_path,presentation)
    targets={}
    for color,x in [('blue',.78),('amber',.90)]:
        a.body('tray_'+color,pose(x,.555,.9),tray_parts((.10,.12,.018)),color=color)
    for i,(color,x) in enumerate([('blue',.53),('amber',.58),('amber',.63),('blue',.68)],1):
        name=f'sample_{i:02}'
        tx=.78 if color=='blue' else .90
        slot=-.025 if i<3 else .025
        goal=pose(tx, .555+slot,.9043)
        start=goal if presentation else pose(x,.555,.9003)
        a.body(name,start,[cylinder(.013,.047,(0,0,.0235)),cylinder(.014,.013,(0,0,.0535))],
               .025,color,category=color,nominal_support='tray_'+color if presentation else 'island')
        targets[name]={'fixture':'tray_'+color,'center_m':[tx,.555,.934],
            'half_extents_m':[.046,.056,.032],'goal_pose_xyzw':goal,'category':color}
    a.meta['task_parameters']={'targets':targets,'upright_tolerance_deg':20.,
        'source_surface_z_m':.9,'support_floor_z_m':.904,
        'description':'Sort the four sealed colored sample bottles into the matching physical trays.'}
    return add_task_visual_details(stage, a.meta)
