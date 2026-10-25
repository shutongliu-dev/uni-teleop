"""Compose versioned task objects into the pinned living room for inspection.

The robot installation is a posed reference for scene presentation. The task
objects retain their physical schemas; this is not a runnable bench release.
"""
from copy import deepcopy
import importlib

from bench.paths import DATA_ROOT, load_json

TASK_IDS = ('remote_operation', 'water_pouring', 'tower_building', 'tower_extraction')
CAMERAS = {
    'room': {'eye': [2.35, -1.50, 1.85], 'target': [.42, .25, 1.02], 'focal_length_mm': 26.},
    'workcell': {'eye': [2.25, -1.55, 1.95], 'target': [.65, .00, 1.02], 'focal_length_mm': 28.},
    'tabletop': {'eye': [1.35, -.52, 1.42], 'target': [.65, .00, .86], 'focal_length_mm': 46.},
}


def task_author(task_id):
    if task_id not in TASK_IDS:
        raise ValueError('Unknown living-room task: ' + str(task_id))
    return importlib.import_module(f'scenes.workrooms.living_room.tasks.{task_id}.assets')


def add_room_and_robots(stage, task_id, output_root):
    """Reuse the unchanged parent and original robot meshes at reviewed bases."""
    from pxr import Usd, UsdGeom, UsdPhysics
    from bench.simulation.assets.workrooms.builder import Builder
    from bench.simulation.isaacsim_backend.workroom import verify_reference
    from .scene import environment_reference, ENVIRONMENT_ROOT
    from .dual_preview import add_robot_preview, ROBOT_ROOTS

    if task_id not in TASK_IDS:
        raise ValueError('Unknown living-room task')
    reference = environment_reference()
    verify_reference(reference)
    manifest = load_json(ENVIRONMENT_ROOT / 'manifest.json')
    description = load_json(ENVIRONMENT_ROOT / 'scene_description.json')
    root = UsdGeom.Xform.Define(stage, '/World')
    root.GetPrim().GetReferences().AddReference(manifest['scene'], '/World')
    stage.SetDefaultPrim(root.GetPrim())
    UsdGeom.SetStageMetersPerUnit(stage, 1.)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    builder = Builder(stage, f'{DATA_ROOT}/assets', output_root, reuse_materials=True)
    composition = add_robot_preview(stage, builder)
    if task_id != 'tower_extraction':
        stage.RemovePrim(ROBOT_ROOTS['rear'])
        for prim in list(stage.GetPrimAtPath('/World/Geometry').GetChildren()):
            if prim.GetName().startswith('rear_'):
                stage.RemovePrim(prim.GetPath())
        composition['robots'].pop('rear')
        composition.update(robot_count=1, source_rigid_body_count=43,
            posed_link_count=composition['robots']['front']['posed_link_count'], hand_sides=['right'])
        composition['clearance_audit']['composition_note'] = 'Audited before deleting the unused rear installation'
    composition['scope'] = 'Original robot geometry in a static ready pose; no robot actuation or manipulation qualification'
    # A native PhysicsScene is added later. Remove simulation-only joint/root
    # participation from this presentation layer, leaving source assets intact.
    for record in composition['robots'].values():
        joints = []
        for prim in Usd.PrimRange(stage.GetPrimAtPath(record['root_path'])):
            if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
                prim.RemoveAPI(UsdPhysics.ArticulationRootAPI)
            if prim.IsA(UsdPhysics.Joint):
                joints.append(prim.GetPath())
        for path in joints:
            stage.GetPrimAtPath(path).SetActive(False)
    return {'parent_environment': reference, 'environment_manifest': str(ENVIRONMENT_ROOT / 'manifest.json'),
        'parent_description': description, 'robots': composition,
        'dependencies': {**manifest['dependencies'], **builder.dependencies},
        'cameras': deepcopy(CAMERAS)}


def physical_inventory(stage, root_path='/World/LivingTask'):
    """Read authored dynamics/collision ownership without simulating it."""
    from pxr import Usd, UsdGeom, UsdPhysics
    root = stage.GetPrimAtPath(root_path)
    if not root:
        raise ValueError('Missing task object root')
    bodies, joints, colliders = [], [], []
    for prim in Usd.PrimRange(root):
        path = str(prim.GetPath())
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            api = UsdPhysics.RigidBodyAPI(prim)
            bodies.append({'path': path, 'enabled': api.GetRigidBodyEnabledAttr().Get(),
                'kinematic': api.GetKinematicEnabledAttr().Get(),
                'mass_kg': UsdPhysics.MassAPI(prim).GetMassAttr().Get()})
        if prim.IsA(UsdPhysics.Joint):
            joints.append({'path': path, 'type': prim.GetTypeName()})
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            colliders.append(path)
    if not bodies or not colliders or any(x['enabled'] is False or x['kinematic'] is True for x in bodies):
        raise ValueError('Task objects must contain enabled, nonkinematic rigid bodies and real colliders')
    return {'bodies': bodies, 'joints': joints, 'colliders': colliders,
            'scope': 'Authored schema inventory, not native dynamics qualification'}
