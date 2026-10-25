"""Build an immutable laboratory task asset from complete frozen source, without Isaac."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from bench.isaacsim.release import digest, dump
from bench.paths import DATA_ROOT
from scenes.workrooms.laboratory.composition import ASSET_SCHEMA, ROOT_PRIM, TASK_IDS

PROJECT = Path(__file__).resolve().parents[2]
ASSETS = Path(f'{DATA_ROOT}/assets')


def audit_task(stage, task_id, metadata):
    """Verify dynamic actor inventory, masses, open geometry and fixed fixtures."""
    import math
    from pxr import Usd, UsdGeom, UsdPhysics
    if (UsdGeom.GetStageMetersPerUnit(stage) != 1. or UsdGeom.GetStageUpAxis(stage) != 'Z'
            or metadata.get('task_id') != task_id or metadata.get('root_path', ROOT_PRIM) != ROOT_PRIM):
        raise ValueError('Task stage units or identity differ')
    root = stage.GetPrimAtPath(ROOT_PRIM)
    if not root:
        raise ValueError('Missing laboratory task root')
    bodies, colliders = {}, []
    for prim in Usd.PrimRange(root):
        path = str(prim.GetPath())
        if prim.IsA(UsdPhysics.Scene) or prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            raise ValueError('Laboratory task cannot author a physics world or robot: '+path)
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            api = UsdPhysics.RigidBodyAPI(prim)
            mass = UsdPhysics.MassAPI(prim).GetMassAttr().Get()
            if (api.GetRigidBodyEnabledAttr().Get() is False or api.GetKinematicEnabledAttr().Get()
                    or mass is None or not math.isfinite(mass) or mass <= 0):
                raise ValueError('Task body must be dynamic with explicit positive mass: '+path)
            bodies[path] = float(mass)
        if prim.HasAPI(UsdPhysics.CollisionAPI) and UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get() is not False:
            colliders.append(path)
    if set(bodies) != set(metadata['actors'].values()):
        raise ValueError('Authored body inventory differs from declared actors')
    objects = {row['id']: row for row in metadata['objects']}
    if set(objects) != set(metadata['actors']):
        raise ValueError('Every actor requires exactly one metadata object')
    for name, path in metadata['actors'].items():
        if objects[name]['body_path'] != path or not math.isclose(objects[name]['mass_kg'], bodies[path], abs_tol=1e-7):
            raise ValueError('Authored mass/body differs from object metadata: '+name)
        if not any(p.startswith(path+'/') or p == path for p in colliders):
            raise ValueError('Task object lacks a physical collider: '+name)
    for name, path in metadata['fixtures'].items():
        if not stage.GetPrimAtPath(path) or not any(p == path or p.startswith(path+'/') for p in colliders):
            raise ValueError('Declared fixture lacks physical collision: '+name)
    return {'status': 'passed', 'task_id': task_id, 'dynamic_body_count': len(bodies),
        'dynamic_mass_kg': sum(bodies.values()), 'collider_count': len(colliders),
        'fixture_count': len(metadata['fixtures']), 'bodies_mass_kg': bodies,
        'scope': 'CPU USD structure and authored inventory only; no native contact or manipulation qualification.'}


def build_task_asset(task_id, output, *, presentation=False, desktop_dressing=False):
    from pxr import Usd, UsdGeom
    import importlib
    author_task = importlib.import_module(f'scenes.workrooms.laboratory.tasks.{task_id}.assets').author_task
    from scripts.assets.build_card_room import dependency_inventory
    from scripts.maintenance.freeze_project_sources import verify_if_snapshot
    snapshot = verify_if_snapshot(PROJECT, required_modules=(f'scenes.workrooms.laboratory.tasks.{task_id}.assets',),
        entrypoint_file=Path(__file__))
    if snapshot is None:
        raise ValueError('Build laboratory task assets from complete frozen project source')
    output = Path(output).resolve()
    if task_id not in TASK_IDS or not output.is_relative_to(ASSETS) or output.exists():
        raise ValueError('Choose a fresh laboratory task asset directory under shared assets')
    output.mkdir(parents=True)
    path = output/'task.usda'
    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageMetersPerUnit(stage, 1.)
    UsdGeom.SetStageUpAxis(stage, 'Z')
    metadata = author_task(stage, root_path=ROOT_PRIM, presentation=presentation)
    if desktop_dressing:
        from scenes.workrooms.laboratory.desktop_dressing import author_desktop_dressing
        metadata['desktop_dressing'] = author_desktop_dressing(
            stage, root_path=ROOT_PRIM+'/DesktopDressing', collision=True)
    audit = audit_task(stage, task_id, metadata)
    stage.GetRootLayer().Save()
    dependencies = dependency_inventory(path, {})
    from bench.simulation.isaacsim_backend.native_task_bundle import add_bundle
    from pxr import UsdPhysics
    probe = Usd.Stage.CreateInMemory()
    UsdPhysics.Scene.Define(probe, '/World/physicsScene')
    add_bundle(probe, {'schema': 'native-usd-task-bundle-v1', 'usd_path': str(path),
        'usd_sha256': digest(path), 'root_prim_path': ROOT_PRIM,
        'rigid_bodies': metadata['actors'], 'joints': metadata.get('joints', {}),
        'particle_sets': metadata.get('particle_sets', {})}, physics_scene_path='/World/physicsScene')
    dump(output/'metadata.json', metadata)
    dump(output/'structural_audit.json', audit)
    attribution = 'Laboratory task apparatus: original procedural geometry, CC0-1.0.\nMasses and contact parameters are declared engineering priors.\n'
    (output/'ATTRIBUTION.txt').write_text(attribution)
    manifest = {'schema': ASSET_SCHEMA, 'task_id': task_id, 'root_prim_path': ROOT_PRIM,
        'presentation': bool(presentation), 'reset_scope': ('presentation_diagnostic' if presentation else 'canonical_task_reset'), 'created_utc': datetime.now(timezone.utc).isoformat(),
        'usd_sha256': digest(path), 'metadata_sha256': digest(output/'metadata.json'),
        'structural_audit_sha256': digest(output/'structural_audit.json'),
        'attribution_sha256': digest(output/'ATTRIBUTION.txt'), 'dependencies': dependencies,
        'source_snapshot': snapshot, 'source_root': str(PROJECT),
        'native_physics_accepted': False, 'robot_manipulation_qualified': False}
    dump(output/'manifest.json', manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', choices=TASK_IDS, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--presentation', action='store_true',
        help='Author a distinct diagnostic reset; normal releases reject it')
    parser.add_argument('--desktop-dressing', action='store_true',
        help='Include task-owned fixed desktop supplies with real collision, outside the work area')
    args = parser.parse_args()
    manifest = build_task_asset(args.task, args.output, presentation=args.presentation,
                                desktop_dressing=args.desktop_dressing)
    print(json.dumps({'task_id': args.task, 'output': str(args.output),
        'manifest_sha256': digest(args.output/'manifest.json'), 'usd_sha256': manifest['usd_sha256']}))


if __name__ == '__main__':
    main()
