"""Static threaded-tool displays with an explicit selected asset profile.

The current default is the existing SHARPA v3 bundle. The original v1 helper
remains available unchanged for explicitly requested historical previews.
"""
from copy import deepcopy
import hashlib
from pathlib import Path

from bench.paths import DATA_ROOT, load_json


DEFAULT_PROFILE = 'sharpa_v3'
PROFILES = ('sharpa_v3', 'requested_v1')
V3_CATALOG_ROOT = Path(f'{DATA_ROOT}/assets/task_parts/mechanical_workshop/threaded_v2')
V3_CATALOG_SHA256 = 'faf25a316190e3369ad57a50d06243fbc52123c988166bae14cbf17f3062d876'


def _digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _v3_sources(catalog_root):
    from bench.simulation.isaacsim_backend.asset_catalog import load_task_asset
    from .threaded_sharpa_catalog import ASSET_IDS, threaded_sharpa_asset_bindings

    root = Path(catalog_root).resolve()
    manifest_path = root / 'manifest.json'
    if _digest(manifest_path) != V3_CATALOG_SHA256:
        raise ValueError('Selected v3 preview catalog changed: ' + str(manifest_path))
    manifest = load_json(manifest_path)
    dependencies = {str(manifest_path): V3_CATALOG_SHA256}
    expected = threaded_sharpa_asset_bindings()
    assets = {}
    for role, asset_id in ASSET_IDS.items():
        asset = load_task_asset(asset_id, root, verify_hashes=True)
        binding = expected[role]
        if (asset['usd_path'] != binding['path'] or asset['usd_default_prim'] != binding['default_prim']
                or asset['source_dependencies_sha256'].get(binding['path']) != binding['sha256']
                or asset['source_scale'] != binding['source_scale']):
            raise ValueError('Selected v3 preview asset differs from the task binding: ' + role)
        assets[role] = asset
        dependencies.update(asset['source_dependencies_sha256'])
    return assets, dependencies, manifest


def add_sharpa_threaded_preview(stage, builder, *, catalog_root=V3_CATALOG_ROOT):
    """Reference exact v3 sources and disable physics only in the preview layer."""
    from pxr import Gf, Usd, UsdGeom, UsdLux, UsdPhysics

    assets, dependencies, manifest = _v3_sources(catalog_root)
    preview_layer = stage.GetRootLayer()
    if preview_layer.realPath and str(Path(preview_layer.realPath).resolve()) in dependencies:
        raise ValueError('A preview cannot author into a source asset layer')
    for path in dependencies:
        builder.dependency(path)
    positions = {'fixture': [.38, .10], 'bolt': [.38, .28], 'screwdriver': [.35, -.10]}
    rows = {}
    with Usd.EditContext(stage, preview_layer):
        for role, source in assets.items():
            container_path = f'/World/TaskPreview/Threaded/{role}'
            if stage.GetPrimAtPath(container_path):
                raise ValueError('Threaded preview already exists: ' + container_path)
            container = UsdGeom.Xform.Define(stage, container_path)
            quaternion = ([2 ** -.5, 2 ** -.5, 0., 0.] if role == 'screwdriver' else [1., 0., 0., 0.])
            lowest = source['bounds_min_m'][1 if role == 'screwdriver' else 2]
            origin = [*positions[role], .78 - lowest]
            container.AddTranslateOp().Set(Gf.Vec3d(*origin))
            container.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(
                Gf.Quatd(quaternion[0], Gf.Vec3d(*quaternion[1:])))
            asset = stage.DefinePrim(container_path + '/Asset')
            asset.GetReferences().AddReference(source['usd_path'], source['usd_default_prim'])
            # Source transforms, meshes, mass and materials remain referenced.
            # In particular the bolt/fixture already carry scale 1.5.
            disabled = {'rigid_bodies': [], 'colliders': [], 'joints': [], 'lights': []}
            bodies = []
            for prim in Usd.PrimRange(asset, Usd.TraverseInstanceProxies()):
                if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                    bodies.append(prim)
                    UsdPhysics.RigidBodyAPI(prim).CreateRigidBodyEnabledAttr(False)
                    disabled['rigid_bodies'].append(str(prim.GetPath()))
                if prim.HasAPI(UsdPhysics.CollisionAPI):
                    if UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get() is not False:
                        disabled['colliders'].append(str(prim.GetPath()))
                    UsdPhysics.CollisionAPI(prim).CreateCollisionEnabledAttr(False)
                if prim.IsA(UsdPhysics.Joint):
                    UsdPhysics.Joint(prim).CreateJointEnabledAttr(False)
                    disabled['joints'].append(str(prim.GetPath()))
                if prim.HasAPI(UsdLux.LightAPI):
                    UsdLux.LightAPI(prim).CreateIntensityAttr(0.)
                    disabled['lights'].append(str(prim.GetPath()))
            if len(bodies) != 1 or len(disabled['colliders']) != len(source['collision_parts']):
                raise ValueError('v3 preview body/collider inventory differs from its source audit')
            bounds = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ['default', 'render', 'proxy']).ComputeWorldBound(
                container.GetPrim()).ComputeAlignedRange()
            world_bounds = [list(bounds.GetMin()), list(bounds.GetMax())]
            expected_size = source['dimensions_m']
            if role == 'screwdriver':
                expected_size = [expected_size[0], expected_size[2], expected_size[1]]
            if abs(world_bounds[0][2] - .78) > 1.e-7 or any(
                    abs(float(bounds.GetSize()[i]) - expected_size[i]) > 1.e-7 for i in range(3)):
                raise ValueError('v3 preview dimensions or support height differ from the exact source')
            matrix = UsdGeom.XformCache().GetLocalToWorldTransform(bodies[0])
            rows[role] = {
                'source': {'path': source['usd_path'], 'sha256': dependencies[source['usd_path']],
                           'default_prim': source['usd_default_prim'],
                           'bounds_m': [source['bounds_min_m'], source['bounds_max_m']],
                           'source_scale': source['source_scale'], 'additional_scale': [1., 1., 1.]},
                'preview_prim': container_path, 'position_m': origin, 'quaternion_wxyz': quaternion,
                'world_bounds_m': world_bounds,
                'body_world_transform': [[float(matrix[i][j]) for j in range(4)] for i in range(4)],
                'disabled_in_preview_layer_only': disabled,
            }
    # Audit writes against the original pinned dependency inventory, including
    # SOURCE/licenses, parent USD references and original geometry files.
    for path, expected in dependencies.items():
        if _digest(path) != expected:
            raise ValueError('Source dependency changed during preview composition: ' + path)
    return {'schema': 'mechanical_threaded_static_preview.v2', 'profile': 'sharpa_v3',
            'status': 'static_task_owned_preview_only', 'placements': rows,
            'catalog_path': str(Path(catalog_root) / 'manifest.json'),
            'catalog_sha256': V3_CATALOG_SHA256, 'dependencies_sha256': dependencies,
            'preview_layer': preview_layer.identifier,
            'historical_contact_evidence': deepcopy(manifest['historical_acceptance']),
            'physics_steps': 0, 'physics_or_reset_acceptance': False, 'source_assets_modified': False,
            'note': 'Exact existing v3 geometry and source transforms; separate static parts. No insertion, reset, grasp or current-room contact qualification.'}


def add_threaded_tool_preview(stage, builder, *, profile=DEFAULT_PROFILE):
    """Select the current v3 display or explicitly request the preserved v1."""
    if profile == 'sharpa_v3':
        return add_sharpa_threaded_preview(stage, builder)
    if profile == 'requested_v1':
        from .threaded_assets import add_threaded_preview
        return {**add_threaded_preview(stage, builder), 'profile': 'requested_v1'}
    raise ValueError('Unknown threaded preview profile: ' + str(profile))
