"""Opt-in source-pose corrections for two household kitchen tasks.

The original geometry, masses, fixtures, goal examples and public rules stay
intact. The two changes have discrete original-hand workspace witnesses;
native support and Agent/Bench evidence belong to separately built releases.
"""
from copy import deepcopy
import math

from pxr import Gf

PROFILE = 'source_access_v1'
TASKS = ('produce_transfer', 'pot_lid_placement')


def apply_source_access(author):
    meta = author.meta
    if meta['task_id'] not in TASKS or meta.get('prop_profile') != 'household_v1':
        raise ValueError('Source access profile requires household produce or pot task')
    meta['source_access_profile'] = PROFILE
    if meta['presentation']:
        return  # Goal presentations are unchanged.
    name = 'tomato_0' if meta['task_id'] == 'produce_transfer' else 'pot'
    row = next(value for value in meta['objects'] if value['id'] == name)
    before = deepcopy(row['initial_pose_xyzw'])
    after = deepcopy(before)
    if name == 'tomato_0':
        after[0] = .23  # 30 mm toward the robot, within the original source board.
    else:
        after[3:] = [0., 0., math.sin(-math.pi/8), math.cos(-math.pi/8)]
    prim = author.stage.GetPrimAtPath(row['body_path'])
    prim.GetAttribute('xformOp:translate').Set(Gf.Vec3d(*after[:3]))
    prim.GetAttribute('xformOp:orient').Set(Gf.Quatf(after[6], Gf.Vec3f(*after[3:6])))
    row['initial_pose_xyzw'] = after
    meta['source_access_changes'] = {name: {'before_pose_xyzw': before, 'after_pose_xyzw': after,
        'scope': 'Source pose only; geometry, inertia, contact materials, robot and goals unchanged.'}}
    meta['layout_revision'] += (' source_access_v1: tomato_0 source X=0.23 m.' if name == 'tomato_0'
                               else ' source_access_v1: source pot yaw=-45 degrees; right handle faces the robot.')
