"""Opt-in robot-free passive settle: filter every robot collider from the rest of the stage.

Only a scene that declares ``isaacsim.robot_collision_mode`` equal to ``MODE`` is
affected; every published scene omits the key and builds unchanged. The
environment keeps its mandatory robot articulation as a host, but a PhysX
collision-group filter removes every robot/scene contact pair, so task objects,
fixtures, room and bundles evolve exactly as on a stage without a robot.
"""
MODE = 'filtered_for_robot_free_settle_v1'
ROOT = '/World/RobotFreeSettle'
ROBOT_GROUP = ROOT+'/Robot'
SCENE_GROUP = ROOT+'/Scene'


def checked_mode(scene):
    mode = scene.get('isaacsim', {}).get('robot_collision_mode')
    if mode is not None and mode != MODE:
        raise ValueError(f'Unknown robot collision mode: {mode!r}')
    return mode


def filter_robot_collisions(stage, robot_path='/World/Robot', world_path='/World'):
    """Author two collision groups: the robot subtree filters everything else under ``world_path``."""
    from pxr import Sdf, UsdPhysics
    if not stage.GetPrimAtPath(robot_path) or stage.GetPrimAtPath(ROOT):
        raise ValueError('Robot-free settle needs the robot host and an unused group root')
    robot = UsdPhysics.CollisionGroup.Define(stage, ROBOT_GROUP)
    scene = UsdPhysics.CollisionGroup.Define(stage, SCENE_GROUP)
    robot.GetCollidersCollectionAPI().CreateIncludesRel().SetTargets([Sdf.Path(robot_path)])
    members = scene.GetCollidersCollectionAPI()
    members.CreateIncludesRel().SetTargets([Sdf.Path(world_path)])
    members.CreateExcludesRel().SetTargets([Sdf.Path(robot_path)])
    robot.CreateFilteredGroupsRel().SetTargets([scene.GetPath()])
    return {'mode': MODE, 'robot_group': ROBOT_GROUP, 'scene_group': SCENE_GROUP,
            'robot_path': robot_path, 'world_path': world_path,
            'scope': 'Every robot/scene collision pair filtered; robot self-collision and scene/scene '
                     'pairs unchanged. Diagnostic settle stages only, never a published scene.'}


def robot_contact(contact, robot_path='/World/Robot'):
    """True when a recorded native contact report involves the robot subtree."""
    return any(str(contact.get(key, '')) == robot_path or str(contact.get(key, '')).startswith(robot_path+'/')
               for key in ('actor0', 'actor1', 'collider0', 'collider1'))
