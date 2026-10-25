"""Long sealed vials, shallow supply cradles and genuinely open rack holes."""
import math
from scenes.workrooms.laboratory.object_geometry import Author, box, cylinder, pose, ring
from scenes.workrooms.laboratory.visual_details import add_task_visual_details


def author_task(stage, root_path='/World/LaboratoryTask', *, presentation=False):
    a=Author(stage,'vial_racking',root_path,presentation)
    # Each hole is a ring supported on a bottom plate, not a painted mark.
    holes=[];rackparts=[box((.26,.11,.006),(0,0,.003))]
    for x in (-.08,0,.08):
        for height in (.025,.060):
            pieces=ring(.012,.022,height,height+.006,24)
            for p in pieces: p.apply_translation((x,0,0))
            rackparts.extend(pieces)
        for side in (-1,1):
            rackparts.append(box((.012,.012,.063),(x,side*.027,.0345)))
    a.body('rack',pose(.72,2.06,.9),rackparts,color='teal')
    cradleparts=[box((.26,.15,.004),(0,0,.002))]
    for x in (-.08,0,.08):
        for dx in (-.013,.013):
            cradleparts.append(box((.004,.11,.012),(x+dx,0,.010)))
    a.body('supply_cradle',pose(.67,.54,.9),cradleparts,color='white')
    for i,x in enumerate((-.08,0,.08),1):
        goal=pose(.72+x,2.06,.9063)
        horizontal=pose(.67+x,.585,.9137,(math.sqrt(.5),0,0,math.sqrt(.5)))
        a.body(f'vial_{i:02}',goal if presentation else horizontal,
               [cylinder(.009,.082,(0,0,.041)),cylinder(.0095,.008,(0,0,.086))],
               .020,'blue',nominal_support='rack' if presentation else 'supply_cradle')
        holes.append({'object_id':f'vial_{i:02}','center_m':[.72+x,2.06,.906],
             'radius_m':.012,'bottom_z_m':.906,'rim_z_m':.966,
             'fixture':'rack','goal_pose_xyzw':goal})
    a.meta['task_parameters']={'holes':holes,'upright_tolerance_deg':10.,
       'insert_depth_min_m':.045,'radial_clearance_m':.0025,
       'description':'Raise each horizontal sealed vial and seat it through an open rear-rack hole.'}
    return add_task_visual_details(stage, a.meta)
