"""A square anti-turn sample bottle and independent coarse helical closure.

Three real helical ramps and three cap lugs implement a short-turn screw closure.
This is a deliberately graspable coarse thread, not a precision ISO thread.
The bottle and cap stay separate free dynamic rigid bodies, with no hidden weld.
"""
import math
from scenes.workrooms.laboratory.object_geometry import (Author,box,cylinder,pose,ring,
                                                        helical_band,sector,tray_parts)
from scenes.workrooms.laboratory.visual_details import add_task_visual_details
PITCH=.018
THREAD_TURN=math.radians(110)
CLOSED_CAP_Z=.075
LUG_ANGLE=math.radians(4)


def author_task(stage, root_path='/World/LaboratoryTask', *, presentation=False):
    a=Author(stage,'cap_placement',root_path,presentation)
    home=[.65,.555,.9003];holder=[.64,2.06,.9]
    holderparts=[box((.10,.10,.006),(0,0,.003))]
    # A square recess prevents bottle rotation without welding or fixing it.
    for dx,dy,sx,sy in ((.029,0,.012,.07),(-.029,0,.012,.07),(0,.029,.07,.012),(0,-.029,.07,.012)):
        holderparts.append(box((sx,sy,.030),(dx,dy,.021)))
    a.body('bottle_holder',pose(*holder),holderparts,color='teal')
    a.body('cap_tray',pose(.86,2.06,.9),tray_parts((.10,.10,.016)),color='amber')
    bottleparts=[box((.044,.044,.052),(0,0,.026)),cylinder(.017,.029,(0,0,.0665))]
    # Each of three starts rises 5.25 mm over 105 degrees, ending in an exit.
    for start in range(3):
        offset=start*2*math.pi/3
        for upper in (0.,.0058):
            bottleparts.append(helical_band(.017,.022,offset,math.radians(105),
                                           .065+upper,PITCH,.0016))
        # Low-end stop blocks accidental clockwise overtravel.
        stop=sector(.017,.022,.064,.070,offset-math.radians(5),offset-math.radians(1))
        bottleparts.append(stop)
    bottlepose=pose(*home)
    a.body('sample_bottle',bottlepose,bottleparts,.09,'white',approximation='sdf',nominal_support='island',
           thread_contact_geometry='three_start_coarse_helical_ramps')
    # Cap wall actually clears both neck and ramps. Its three internal lugs
    # touch the ramp; the solid top seals the neck visually and geometrically.
    capparts=ring(.023,.027,-.009,.011,32)+[cylinder(.027,.003,(0,0,.0125))]
    for start in range(3):
        theta=start*2*math.pi/3+LUG_ANGLE
        capparts.append(sector(.018,.025,-.008,.008- .013,theta-math.radians(2),theta+math.radians(2)))
    cap_pose=pose(home[0],home[1],home[2]+CLOSED_CAP_Z)
    a.body('cap',cap_pose,capparts,.025,'blue',nominal_support='sample_bottle')
    a.meta['task_parameters']={'bottle_home':{'center_m':home,'half_extents_m':[.04,.04,.05]},
        'holder_center_m':[holder[0],holder[1],.906],
        'holder_inner_half_extents_m':[.023,.023,.04],
        'cap_tray_target':{'fixture':'cap_tray','center_m':[.86,2.06,.917],
                           'half_extents_m':[.046,.046,.018]},
        'thread':{'model':'three_start_short_turn_coarse_helical_contact',
            'pitch_m':PITCH,'required_turn_rad':THREAD_TURN,'closed_cap_z_m':CLOSED_CAP_Z,
            'open_cap_z_m':CLOSED_CAP_Z+PITCH*THREAD_TURN/(2*math.pi),
            'cleared_cap_z_m':CLOSED_CAP_Z+.035,'helical_exit_turn_rad':math.radians(103),
            'closed_cap_relative_position_m':[0.,0.,CLOSED_CAP_Z],
            'radial_tolerance_m':.003,'axis_xy_m':home[:2],
            'axis_frame':'bottle local Z; world XY follows the free bottle',
            'closure_tolerance_z_m':.002,'closed_angle_rad':0.,'opening_sign':1},
        'description':'Carry the closed bottle to the square anti-turn socket, unscrew the independent cap, rest it in its tray, screw it back and return the bottle.'}
    return add_task_visual_details(stage, a.meta)
