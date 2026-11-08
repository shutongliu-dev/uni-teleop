"""Build an immutable kitchen task asset from complete frozen source, without Isaac."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from bench.isaacsim.release import digest, dump
from bench.paths import DATA_ROOT
from scenes.workrooms.kitchen.composition import ASSET_SCHEMA, ROOT_PRIM, TASK_IDS

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
        raise ValueError('Missing kitchen task root')
    bodies, colliders, joints = {}, [], {}
    for prim in Usd.PrimRange(root):
        path = str(prim.GetPath())
        if prim.IsA(UsdPhysics.Scene) or prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            raise ValueError('Kitchen task cannot author a physics world or robot: '+path)
        if prim.IsA(UsdPhysics.Joint):
            from bench.simulation.isaacsim_backend.native_task_bundle import passive_elastic_drives
            if prim.GetTypeName() != 'PhysicsJoint' or not metadata.get('elastic_composites'):
                raise ValueError('Kitchen joints require a declared passive elastic composite')
            joints[path] = passive_elastic_drives(prim)
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
    if set(joints) != set(metadata.get('joints', {}).values()):
        raise ValueError('Authored elastic joint inventory differs')
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
    context = metadata.get('contextual_props', {})
    if set(context) & (set(metadata['actors']) | set(metadata['fixtures'])):
        raise ValueError('Context objects must be distinct from task actors and fixtures')
    for name, path in context.items():
        row = metadata['contextual_prop_metadata'][name]
        if (not path.startswith(ROOT_PRIM+'/Context/') or row['body_path'] != path
                or row.get('scoring') is not False or row.get('required') is not False
                or row.get('mass_kg') != 0 or path in bodies
                or not stage.GetPrimAtPath(path)
                or not any(p.startswith(path+'/') for p in colliders)):
            raise ValueError('Context must have declared fixed non-scoring collision: '+name)
    volumes = metadata.get('volume_bodies', {})
    actual_volume_paths = {str(p.GetPath()) for p in Usd.PrimRange(root)
        if p.GetMetadata('apiSchemas') is not None
        and 'OmniPhysicsDeformableBodyAPI' in p.GetMetadata('apiSchemas').GetAppliedItems()}
    if actual_volume_paths != {r['prim_path'] for r in volumes.values()}:
        raise ValueError('Unobserved volume deformable body in kitchen asset')
    for name, row in volumes.items():
        mass = stage.GetPrimAtPath(row['prim_path']).GetAttribute('omniphysics:mass').Get()
        if mass is None or not math.isfinite(mass) or mass <= 0 or not math.isclose(mass, row['total_mass_kg'], abs_tol=1e-7):
            raise ValueError('Volume deformable mass differs from metadata: '+name)
    volume_mass = sum(r['total_mass_kg'] for r in volumes.values())
    return {'status': 'passed', 'task_id': task_id, 'dynamic_body_count': len(bodies),
        'dynamic_mass_kg': sum(bodies.values())+volume_mass, 'collider_count': len(colliders),
        **({'volume_body_count': len(volumes), 'volume_mass_kg': volume_mass,
            'rigid_mass_kg': sum(bodies.values())} if volumes else {}),
        'fixture_count': len(metadata['fixtures']), 'contextual_prop_count': len(context),
        'bodies_mass_kg': bodies,
        'scope': 'CPU USD structure and authored inventory only; no native contact or manipulation qualification.'}


def build_task_asset(task_id, output, *, presentation=False, dressing_profile='legacy', prop_profile='legacy',
                     sponge_physics_profile='rigid', source_access_profile='legacy',
                     utensil_contact_profile='legacy', utensil_grip_profile='legacy'):
    from pxr import Usd, UsdGeom
    from scenes.workrooms.kitchen.assets import author_task
    from scenes.workrooms.kitchen.textures import write_task_textures
    from scripts.assets.build_card_room import dependency_inventory
    from scripts.maintenance.freeze_project_sources import verify_if_snapshot
    snapshot = verify_if_snapshot(PROJECT, required_modules=('scenes.workrooms.kitchen.assets',),
        entrypoint_file=Path(__file__))
    if snapshot is None:
        raise ValueError('Build kitchen task assets from complete frozen project source')
    output = Path(output).resolve()
    if task_id not in TASK_IDS or not output.is_relative_to(ASSETS) or output.exists():
        raise ValueError('Choose a fresh kitchen task asset directory under shared assets')
    output.mkdir(parents=True)
    path = output/'task.usda'
    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageMetersPerUnit(stage, 1.)
    UsdGeom.SetStageUpAxis(stage, 'Z')
    texture_assets = write_task_textures(output)
    metadata = author_task(stage, task_id, root_path=ROOT_PRIM,
        presentation=presentation, texture_assets=texture_assets, dressing_profile=dressing_profile, prop_profile=prop_profile,
        sponge_physics_profile=sponge_physics_profile, source_access_profile=source_access_profile,
        utensil_contact_profile=utensil_contact_profile, utensil_grip_profile=utensil_grip_profile)
    audit = audit_task(stage, task_id, metadata)
    stage.GetRootLayer().Save()
    dependencies = dependency_inventory(path, {})
    # Generic layer validation catches escaping relationships, unpinned resources,
    # scaled bodies and undeclared dynamics before the native loader is reached.
    from bench.simulation.isaacsim_backend.task_layers import mount_task_layers
    from pxr import UsdPhysics
    probe = Usd.Stage.CreateInMemory()
    UsdPhysics.Scene.Define(probe, '/World/physicsScene')
    mount_task_layers(probe, [{'id': 'kitchen', 'usd_path': str(path), 'sha256': digest(path),
        'source_prim_path': ROOT_PRIM, 'target_prim_path': ROOT_PRIM,
        'dependencies_sha256': {p: r['sha256'] for p, r in dependencies.items()},
        'rigid_bodies': metadata['actors'],
        **({'volume_bodies': metadata['volume_bodies']} if metadata.get('volume_bodies') else {})}],
        physics_scene_path='/World/physicsScene')
    dump(output/'metadata.json', metadata)
    dump(output/'structural_audit.json', audit)
    attribution = 'Kitchen task props: original procedural geometry, seeded oak albedo and material design, CC0-1.0.\nWood texture covers 400 by 280 mm; appearance only.\nMasses and friction are engineering priors; not measured material calibration.\n'
    (output/'ATTRIBUTION.txt').write_text(attribution)
    manifest = {'schema': ASSET_SCHEMA, 'task_id': task_id, 'root_prim_path': ROOT_PRIM,
        'presentation': bool(presentation), 'dressing_profile': dressing_profile, 'prop_profile': prop_profile,
        **({'sponge_physics_profile': sponge_physics_profile} if sponge_physics_profile != 'rigid' else {}),
        **({'source_access_profile': source_access_profile} if source_access_profile != 'legacy' else {}),
        **({'utensil_contact_profile': utensil_contact_profile} if utensil_contact_profile != 'legacy' else {}),
        **({'utensil_grip_profile': utensil_grip_profile} if utensil_grip_profile != 'legacy' else {}),
        'reset_scope': ('presentation_diagnostic' if presentation else 'canonical_task_reset'), 'created_utc': datetime.now(timezone.utc).isoformat(),
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
    parser.add_argument('--dressing-profile', choices=('legacy', 'enriched_v1'), default='legacy',
        help='Task-owned contextual kitchen objects; never changes the fixed room')
    parser.add_argument('--prop-profile', choices=('legacy', 'household_v1'), default='legacy')
    parser.add_argument('--sponge-physics-profile', choices=('rigid', 'bonded_fem_v1', 'bonded_fem_coarse_v1', 'elastic_cell_lattice_v1'), default='rigid',
        help='Explicit new physical sponge candidate; requires surface_wiping and household_v1')
    parser.add_argument('--source-access-profile', choices=('legacy', 'source_access_v1'), default='legacy',
        help='Explicit source-pose revision for household produce/pot tasks')
    parser.add_argument('--utensil-contact-profile', choices=('legacy', 'floor_grid_2mm_v1', 'floor_grid_2mm_convex_v1', 'floor_grid_2mm_convex_v2', 'source_grid_1mm_convex_v3', 'floor_grid_2mm_convex_sdf768_v4', 'floor_grid_2mm_fullconvex_v5', 'floor_grid_2mm_fullconvex_v6', 'floor_grid_2mm_splitbowl_sdf768_v7'), default='legacy',
        help='Explicit coplanar tray tessellation for household sorting/clearing')
    parser.add_argument('--utensil-grip-profile', choices=('legacy', 'thick_handle_v1'), default='legacy',
        help='Optional visible thicker handles; preserves food-contact ends and existing layout')
    args = parser.parse_args()
    manifest = build_task_asset(args.task, args.output, presentation=args.presentation,
        dressing_profile=args.dressing_profile, prop_profile=args.prop_profile,
        sponge_physics_profile=args.sponge_physics_profile, source_access_profile=args.source_access_profile,
        utensil_contact_profile=args.utensil_contact_profile, utensil_grip_profile=args.utensil_grip_profile)
    print(json.dumps({'task_id': args.task, 'output': str(args.output),
        'manifest_sha256': digest(args.output/'manifest.json'), 'usd_sha256': manifest['usd_sha256']}))


if __name__ == '__main__':
    main()
