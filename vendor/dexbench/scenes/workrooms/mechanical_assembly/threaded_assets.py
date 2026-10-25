"""Pinned shared asset references, never an upstream scene/controller import.

Bounds were measured with usd-core 26.8, default/render/proxy BBoxCache at default
time. Matching SHA256 binds these measurements to the exact shared source USDs.
This module does not start Isaac, write assets, or establish native acceptance.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

from bench.paths import DATA_ROOT


TOOL_ROOT = Path(f'{DATA_ROOT}/assets/task_parts/threaded_tools/v1')
ASSEMBLY_ROOT = Path(f'{DATA_ROOT}/assets/objects/cosigen/046f7bf48865/threaded_assembly')
SOURCE_HASHES = {
    'tool': '33e5cfad3b78de5440dcc908f195a93c8cf354746bae019ccbe5e632c85f6d6f',
    'assembly': 'a7130f6063811d7c2539859a2e7d8fc1d83677fa91b3ae83acdca218dd4032cb',
}
ASSETS = {
    'screwdriver': {
        'root': 'tool', 'relative_path': 'straight_hex_screwdriver_m16.usd',
        'sha256': '98592227db37216372193344d71ed6914760b0793fd303d55877d9f99d366e06',
        'default_prim': '/straight_hex_screwdriver_m16',
        'bounds_m': [[-.017000000923872, -.017000000923872, 0.],
                     [.017000000923872, .017000000923872, .164000004529953]],
        'preview_quaternion_wxyz': [.7071067811865476, .7071067811865476, 0., 0.],
        'preview_origin_above_support_m': .017000000923872,
        'frame': 'Local origin at bit tip; +Z points toward handle. Preview lays the shaft along world -Y.',
    },
    'bolt': {
        'root': 'assembly', 'relative_path': 'robobench/suites/assembly/assets/allen_bolt/allen_bolt_m16.usd',
        'sha256': '8a9da84ecafd33fb764a99bd2767fb22419db0fda02c8624709a66758e90357c',
        'default_prim': '/allen_bolt_m16',
        'bounds_m': [[-.014999999664724, -.014999999664724, -1.490116086078075e-10],
                     [.014999999664724, .014999999664724, .04280000180006027]],
        'preview_quaternion_wxyz': [1., 0., 0., 0.],
        'preview_origin_above_support_m': 1.490116086078075e-10,
        'frame': 'Local origin at thread tip; +Z points toward head. Upright preview is static, not a stable reset claim.',
    },
    'fixture': {
        'root': 'assembly', 'relative_path': 'robobench/suites/assembly/assets/threaded_platform/threaded_platform_m16.usd',
        'sha256': '61e8a9b5a126220c06d9a0b65120f74ed319c2d2988e1cb94dca34b21e4b47e2',
        'default_prim': '/threaded_platform_m16',
        'bounds_m': [[-.07999999821186066, -.05000000074505806, 0.],
                     [.07999999821186066, .05000000074505806, .03799999877810478]],
        'preview_quaternion_wxyz': [1., 0., 0., 0.], 'preview_origin_above_support_m': 0.,
        'frame': 'Local origin on support surface; threaded plate top is at local Z=.038 m.',
    },
}


def threaded_asset_bindings(*, tool_root=TOOL_ROOT, assembly_root=ASSEMBLY_ROOT):
    """Describe exact sources without reading disk or claiming validation."""
    roots = {'tool': Path(tool_root), 'assembly': Path(assembly_root)}
    result = deepcopy(ASSETS)
    for row in result.values():
        row['path'] = str(roots[row['root']] / row['relative_path'])
        row.update(meters_per_unit=1., up_axis='Z', scale=[1., 1., 1.])
    return result


def _verified_json(path, expected):
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != expected:
        raise ValueError(f'Threaded asset source hash differs: {path}')
    return json.loads(data)


def validated_threaded_assets(*, tool_root=TOOL_ROOT, assembly_root=ASSEMBLY_ROOT):
    """Verify pinned provenance, asset files and SHA256SUMS; return preview data.

    An asset binding is not a task release. Original upstream GraspWeldContract,
    force-driven smoke, material overrides and reset helpers are never executed.
    """
    roots = {'tool': Path(tool_root), 'assembly': Path(assembly_root)}
    sources = {name: _verified_json(root / 'SOURCE.json', SOURCE_HASHES[name])
               for name, root in roots.items()}
    verified = {}
    for name, root in roots.items():
        sums = dict((relative, digest) for digest, relative in
                    (line.split(maxsplit=1) for line in (root / 'SHA256SUMS').read_text().splitlines() if line))
        for row in sources[name]['files']:
            relative = row['path']
            path = (root / relative).resolve()
            if not path.is_relative_to(root.resolve()):
                raise ValueError('Asset provenance path escapes its shared root')
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if digest != row['sha256'] or sums.get(relative) != digest:
                raise ValueError(f'Threaded asset file hash differs: {path}')
            verified[str(path)] = digest
    bindings = threaded_asset_bindings(tool_root=tool_root, assembly_root=assembly_root)
    for row in bindings.values():
        if verified.get(str(Path(row['path']).resolve())) != row['sha256']:
            raise ValueError(f'Pinned asset is missing from provenance: {row["path"]}')
    return {
        'schema': 'mechanical_threaded_asset_references.v1', 'assets': bindings,
        'source_manifests': {name: {'path': str(roots[name] / 'SOURCE.json'), 'sha256': digest}
                             for name, digest in SOURCE_HASHES.items()},
        'verified_files_sha256': verified,
        'licenses': {
            'tool': {'path': str(roots['tool'] / 'LICENSE.txt'), 'terms': sources['tool']['license']},
            'assembly': {'path': str(roots['assembly'] / 'provenance/huggingface/LICENSE'),
                         'notice': str(roots['assembly'] / 'provenance/huggingface/NOTICE'),
                         'terms': sources['assembly']['license']},
        },
        'pairing_geometry': deepcopy(sources['tool']['pairing']),
        'status': 'source_hashes_and_offline_bounds_only', 'native_physics_validated': False,
        'preview_scope': 'Task-owned static display at original scale. Never merge these objects into the fixed empty room.',
    }


def add_threaded_preview(stage, builder):
    """Reference original-scale assets in a separate static task preview layer."""
    from pxr import Gf, Usd, UsdGeom, UsdLux, UsdPhysics

    contract = validated_threaded_assets()
    for path in contract['verified_files_sha256']:
        builder.dependency(path)
    for row in contract['source_manifests'].values():
        builder.dependency(row['path'])
    positions = {'fixture': [.38, .10], 'bolt': [.38, .22], 'screwdriver': [.35, -.10]}
    rows = {}
    for name, source in contract['assets'].items():
        container_path = f'/World/TaskPreview/Threaded/{name}'
        container = UsdGeom.Xform.Define(stage, container_path)
        origin = [*positions[name], .78 + source['preview_origin_above_support_m']]
        quaternion = source['preview_quaternion_wxyz']
        container.AddTranslateOp().Set(Gf.Vec3d(*origin))
        container.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(
            Gf.Quatd(quaternion[0], Gf.Vec3d(*quaternion[1:])))
        asset = stage.DefinePrim(container_path + '/Asset')
        asset.GetReferences().AddReference(source['path'], source['default_prim'])
        disabled = {'rigid_bodies': [], 'joints': [], 'lights': []}
        for prim in Usd.PrimRange(asset):
            if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                UsdPhysics.RigidBodyAPI(prim).CreateRigidBodyEnabledAttr(False)
                disabled['rigid_bodies'].append(str(prim.GetPath()))
            if prim.IsA(UsdPhysics.Joint):
                UsdPhysics.Joint(prim).CreateJointEnabledAttr(False)
                disabled['joints'].append(str(prim.GetPath()))
            if prim.HasAPI(UsdLux.LightAPI):
                UsdLux.LightAPI(prim).CreateIntensityAttr(0.)
                disabled['lights'].append(str(prim.GetPath()))
        bounds = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ['default', 'render', 'proxy']).ComputeWorldBound(
            container.GetPrim()).ComputeAlignedRange()
        rows[name] = {
            'source': deepcopy(source), 'preview_prim': container_path,
            'position_m': origin, 'quaternion_wxyz': quaternion,
            'world_bounds_m': [list(bounds.GetMin()), list(bounds.GetMax())],
            'disabled_in_preview_layer_only': disabled,
        }
    return {
        **contract, 'placements': rows,
        'status': 'static_task_owned_preview_only',
        'physics_or_reset_acceptance': False,
        'note': 'Original geometry, units, scale and source files preserved. Bolt displayed separately; no tool insertion or thread engagement claim.',
    }
