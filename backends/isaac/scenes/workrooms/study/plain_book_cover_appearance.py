"""Explicit appearance-only adapter for the drawer and reading-stand books.

The two reviewed pure-color cover meshes get a local visual material. Their
original UV arrays are retained; a shader UV transform maps the broad face to
one large pattern. Independent notebook print and brass decoration stay intact.
No default/historical author calls this opt-in adapter.
"""
from pathlib import Path
import json

import numpy as np

from bench.paths import DATA_ROOT, load_json
from scenes.workrooms.parameterized.spec import validate_parameters
from scenes.workrooms.seed_geometry import file_hash

VERSION = 'study_plain_book_cover_background_v1'
ASSETS = {
    'drawer_retrieval': {
        'manifest': f'{DATA_ROOT}/assets/task_parts/study/drawer_retrieval/v1/manifest.json',
        'sha256': '9307cdd33367a68f31a462e93aa8a51f1836f658e9c8d313b283ec3e57d0688a',
        'actor': 'notebook', 'material': 'green_cover', 'size_xy_m': [.09, .11]},
    'reading_stand': {
        'manifest': f'{DATA_ROOT}/assets/task_parts/study/reading_stand/v1/manifest.json',
        'sha256': '67dc4c8d4be1e1d494df6a2a7b8c0e39fa20ad2bfc06c1d56aca36ec5ba30ca4',
        'actor': 'book', 'material': 'cover', 'size_xy_m': [.14, .17]},
}


def _attributes(prim):
    return {a.GetName(): (str(a.Get()), tuple(map(str, a.GetConnections())), str(a.GetAllMetadata()))
            for a in prim.GetAttributes()}


def apply_plain_book_cover_background(stage, actors, parameters, output_dir):
    """Change only two explicitly identified visual bindings and new shaders."""
    from PIL import Image
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade
    from scenes.workrooms.parameterized.appearance import render_appearance
    validate_parameters(parameters)
    task_id = parameters['task_id']
    if parameters['family'] != 'study' or task_id not in ASSETS:
        raise ValueError('No reviewed plain book cover adapter for this task')
    if parameters['canonical']:
        return {'schema': VERSION, 'canonical': True, 'meshes': [], 'artifacts': []}
    if not stage.GetRootLayer().anonymous:
        raise ValueError('Cover appearance requires an isolated anonymous stage')
    recipe = ASSETS[task_id]
    if file_hash(recipe['manifest']) != recipe['sha256']:
        raise ValueError('Known plain-cover source manifest changed')
    manifest = load_json(Path(recipe['manifest']))
    if file_hash(manifest['initial_usd']) != manifest['initial_usd_sha256']:
        raise ValueError('Known plain-cover source USD changed')
    source = Usd.Stage.Open(manifest['initial_usd'])
    path = '/World/StudyTask/Objects/'+recipe['actor']
    actor = stage.GetPrimAtPath(path)
    if actors.get(recipe['actor']) != path or not actor or not actor.HasAPI(UsdPhysics.RigidBodyAPI):
        raise ValueError('Expected the original task-owned book body')
    meshes = []
    material_path = '/World/StudyTask/Looks/'+recipe['material']
    for face in ('front', 'back'):
        mesh_path = path+'/Geometry/cover_'+face+'/Visual'
        prim = stage.GetPrimAtPath(mesh_path); original = source.GetPrimAtPath(mesh_path)
        if not prim.IsA(UsdGeom.Mesh) or UsdGeom.Imageable(prim).ComputeVisibility() == 'invisible':
            raise ValueError('Expected known visible plain-cover mesh')
        if (_attributes(prim) != _attributes(original) or
                _attributes(prim.GetParent()) != _attributes(original.GetParent())):
            raise ValueError('Plain cover local geometry, UV or transform differs from pinned source')
        material, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()
        if str(material.GetPath()) != material_path:
            raise ValueError('Plain cover material is not the pinned original')
        meshes.append(prim)
    material = UsdShade.Material(stage.GetPrimAtPath(material_path))
    shader = material.ComputeSurfaceSource()[0]
    original_shader = UsdShade.Material(source.GetPrimAtPath(material_path)).ComputeSurfaceSource()[0]
    if shader.GetIdAttr().Get() != 'UsdPreviewSurface' or _attributes(shader.GetPrim()) != _attributes(original_shader.GetPrim()):
        raise ValueError('Original plain-cover shader changed')
    diffuse = shader.GetInput('diffuseColor')
    if diffuse.HasConnectedSource():
        raise ValueError('Known plain cover must use its original untextured color')
    # USD diffuseColor is linear; generated PNG is explicitly sRGB. Preserve
    # its hue/category while adding the already sampled same-color contrast.
    linear = np.asarray(diffuse.Get(), dtype=float)
    srgb = np.where(linear <= .0031308, 12.92*linear, 1.055*linear**(1/2.4)-.055)
    rendered = render_appearance(parameters, base_color=srgb, preserve_category_color=True, size=512)
    destination = path+'/ParameterizedCoverAppearance'
    if stage.GetPrimAtPath(destination):
        raise ValueError('Book appearance already authored')
    output = Path(output_dir).resolve(); output.mkdir(parents=True, exist_ok=False)
    image = output/'cover_background.png'
    with image.open('xb') as stream: Image.fromarray(rendered['albedo']).save(stream, format='PNG')
    clone = UsdShade.Material.Define(stage, destination)
    surface = UsdShade.Shader.Define(stage, destination+'/Surface'); surface.CreateIdAttr('UsdPreviewSurface')
    for old in shader.GetInputs():
        entry = surface.CreateInput(old.GetBaseName(), old.GetTypeName())
        if old.Get() is not None: entry.Set(old.Get())
        if old.HasConnectedSource(): entry.GetAttr().SetConnections(old.GetAttr().GetConnections())
    reader = UsdShade.Shader.Define(stage, destination+'/UV'); reader.CreateIdAttr('UsdPrimvarReader_float2')
    reader.CreateInput('varname', Sdf.ValueTypeNames.Token).Set('st')
    reader.CreateOutput('result', Sdf.ValueTypeNames.Float2)
    transform = UsdShade.Shader.Define(stage, destination+'/UVScale'); transform.CreateIdAttr('UsdTransform2d')
    transform.CreateInput('in', Sdf.ValueTypeNames.Float2).ConnectToSource(reader.ConnectableAPI(), 'result')
    scale = [1/(2*size) for size in recipe['size_xy_m']]
    transform.CreateInput('scale', Sdf.ValueTypeNames.Float2).Set(Gf.Vec2f(*scale))
    transform.CreateOutput('result', Sdf.ValueTypeNames.Float2)
    tex = UsdShade.Shader.Define(stage, destination+'/Albedo'); tex.CreateIdAttr('UsdUVTexture')
    tex.CreateInput('file', Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(str(image)))
    tex.CreateInput('sourceColorSpace', Sdf.ValueTypeNames.Token).Set('sRGB')
    tex.CreateInput('st', Sdf.ValueTypeNames.Float2).ConnectToSource(transform.ConnectableAPI(), 'result')
    tex.CreateOutput('rgb', Sdf.ValueTypeNames.Float3)
    for axis in ('S', 'T'): tex.CreateInput('wrap'+axis, Sdf.ValueTypeNames.Token).Set('repeat')
    surface.GetInput('diffuseColor').ConnectToSource(tex.ConnectableAPI(), 'rgb')
    clone.CreateSurfaceOutput().ConnectToSource(surface.ConnectableAPI(), 'surface')
    for prim in meshes:
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(clone)
    return {'schema': VERSION, 'canonical': False, 'instance_sha256': parameters['instance_sha256'],
        'source_manifest': {'path': recipe['manifest'], 'sha256': recipe['sha256']},
        'source_usd': {'path': manifest['initial_usd'], 'sha256': manifest['initial_usd_sha256']},
        'pattern_id': parameters['appearance']['pattern_id'], 'meshes': [str(p.GetPath()) for p in meshes],
        'pattern_domain': 'one_generated_pattern_field_per_broad_cover_face',
        'uv_policy': 'original_mesh_uv_unchanged_with_explicit_shader_normalization',
        'sampled_pattern_scale_m': parameters['appearance']['pattern_scale_m'],
        'pattern_scale_m_applied': False,
        'scale_scope': 'Whole-face background pattern, not the sampled metric tile; other authored materials retain their recorded metric scale.',
        'new_material': destination, 'original_material': material_path,
        'original_linear_rgb': linear.tolist(), 'shader_uv_scale': scale,
        'artifacts': [{'path': str(image), 'sha256': file_hash(image)}],
        'original_mesh_uv_unchanged': True, 'print_decoration_bindings_unchanged': True,
        'geometry_layout_physics_unchanged': True, 'native_qualified': False}
