"""Sealed vials standing in a low source rack and a separate receiving rack."""
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
    # A low, genuinely open guide leaves about 65 mm of each vial exposed.
    # The fixture ID is retained; only the destination rack is a scoring support.
    cradleparts=[box((.26,.11,.006),(0,0,.003))]
    for x in (-.08,0,.08):
        pieces=ring(.012,.022,.025,.031,24)
        for p in pieces: p.apply_translation((x,0,0))
        cradleparts.extend(pieces)
        for side in (-1,1):
            cradleparts.append(box((.012,.012,.028),(x,side*.027,.020)))
    a.body('supply_cradle',pose(.67,.54,.9),cradleparts,color='white')
    source_holes=[]
    for i,x in enumerate((-.08,0,.08),1):
        goal=pose(.72+x,2.06,.9063)
        source=pose(.67+x,.54,.9063)
        a.body(f'vial_{i:02}',goal if presentation else source,
               [cylinder(.009,.082,(0,0,.041)),cylinder(.0095,.008,(0,0,.086))],
               .020,'blue',nominal_support='rack' if presentation else 'supply_cradle')
        holes.append({'object_id':f'vial_{i:02}','center_m':[.72+x,2.06,.906],
             'radius_m':.012,'bottom_z_m':.906,'rim_z_m':.966,
             'fixture':'rack','goal_pose_xyzw':goal})
        source_holes.append({'object_id':f'vial_{i:02}','center_m':[.67+x,.54,.906],
             'radius_m':.012,'bottom_z_m':.906,'rim_z_m':.931,
             'fixture':'supply_cradle','initial_pose_xyzw':source})
    a.meta['task_parameters']={'holes':holes,'upright_tolerance_deg':10.,
       'insert_depth_min_m':.045,'radial_clearance_m':.0025,
       'source_holes':source_holes,
       'source_rack':{'fixture':'supply_cradle','base_size_m':[.26,.11,.006],
           'guide_local_z_m':[.025,.031],'post_top_local_z_m':.034,
           'exposed_vial_above_guide_m':.0653,
           'number_plate_local_yz_m':[-.043,.00615]},
       'description':'Lift each upright sealed vial out of the source rack and seat it through an open rear-rack hole.'}
    return add_task_visual_details(stage, a.meta)
