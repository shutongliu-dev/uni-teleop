import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pytest

from bench.paths import load_json

pytest.importorskip('pxr')
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

from scenes.workrooms.mechanical_assembly import presentation_materials as pm


@pytest.fixture
def texture_root(tmp_path, monkeypatch):
    for family in ('Metal009', 'Rubber004'):
        folder = tmp_path / family; folder.mkdir()
        files = []
        for suffix in ('Color', 'NormalGL', 'Roughness'):
            path = folder / f'{family}_1K-JPG_{suffix}.jpg'
            path.write_bytes((family + suffix).encode())
            files.append({'file': path.name, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
            if suffix == 'Color':
                # Deliberately fake bytes: these tests exercise source binding and
                # shader math, not image decoding or the real scan measurement.
                monkeypatch.setitem(pm.COLOR_MAP_REFERENCE, family,
                    {**pm.COLOR_MAP_REFERENCE[family], 'sha256': files[-1]['sha256']})
        (folder / 'SOURCE.json').write_text(json.dumps({'assetId': family, 'license': 'CC0-1.0', 'files': files}))
    return tmp_path


@pytest.fixture
def scene(texture_root):
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageMetersPerUnit(stage, 1.)
    root = UsdGeom.Xform.Define(stage, '/Body').GetPrim()
    materials = pm.build_materials(stage, '/Body/PresentationLooks', texture_root)
    return stage, root, materials


def quad(stage, path, *, offset=0.):
    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr().Set([(offset, 0, 0), (offset+.02, 0, 0), (offset+.02, .02, 0), (offset, .02, 0)])
    mesh.CreateFaceVertexCountsAttr().Set([4]); mesh.CreateFaceVertexIndicesAttr().Set([0, 1, 2, 3])
    mesh.CreateSubdivisionSchemeAttr().Set('none')
    return mesh


def color_material(stage, path, color, metallic):
    material = UsdShade.Material.Define(stage, path)
    sh = UsdShade.Shader.Define(stage, path + '/Surface'); sh.CreateIdAttr('UsdPreviewSurface')
    sh.CreateInput('diffuseColor', Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    sh.CreateInput('metallic', Sdf.ValueTypeNames.Float).Set(metallic)
    sh.CreateInput('roughness', Sdf.ValueTypeNames.Float).Set(.4)
    material.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), 'surface')
    return material


def test_material_maps_have_correct_color_spaces_connections_and_dependencies(scene, texture_root):
    stage, _, materials = scene
    assert len(materials) == 15
    dependencies = set()
    for role, material in materials.items():
        source = json.loads(material.GetPrim().GetCustomDataByKey('mechanicalPresentation'))
        assert source['role'] == role
        shader = UsdShade.Shader.Get(stage, material.GetPath().AppendChild('Surface'))
        assert not shader.GetInput('displacement')
        if role == 'marking_light':
            assert not shader.GetInput('diffuseColor').GetAttr().GetConnections()
            continue
        for label, target, output in [('Color', 'diffuseColor', 'rgb'),
                                     ('NormalGL', 'normal', 'rgb'), ('Roughness', 'roughness', 'r')]:
            tex = UsdShade.Shader.Get(stage, material.GetPath().AppendChild(label))
            dependencies.add(tex.GetInput('file').Get().path)
            assert tex.GetInput('sourceColorSpace').Get() == ('sRGB' if label == 'Color' else 'raw')
            assert shader.GetInput(target).GetAttr().GetConnections() == [tex.GetPath().AppendProperty('outputs:' + output)]
            assert tex.GetInput('st').GetAttr().GetConnections() == [material.GetPath().AppendChild('ST').AppendProperty('outputs:result')]
            assert tex.GetInput('wrapS').Get() == tex.GetInput('wrapT').Get() == 'repeat'
            evidence = source['texture_source']['files'][label]
            assert evidence['sha256'] == hashlib.sha256(Path(evidence['path']).read_bytes()).hexdigest()
        normal = UsdShade.Shader.Get(stage, material.GetPath().AppendChild('NormalGL'))
        assert normal.GetInput('scale').Get()[0] <= .3 + 1.e-7
    expected = {str(p) for p in pm.required_texture_files(texture_root) if p.name != 'SOURCE.json'}
    assert dependencies == expected


def test_color_modulation_preserves_linear_mean_tint_with_positive_finite_scale(scene):
    stage, _, materials = scene
    for material in materials.values():
        provenance = json.loads(material.GetPrim().GetCustomDataByKey('mechanicalPresentation'))
        if 'texture_source' not in provenance:
            continue
        evidence = provenance['texture_source']['files']['Color']
        assert evidence['mean_method'] == pm.COLOR_MEAN_METHOD
        mean = np.asarray(evidence['linear_rgb_mean'])
        tex = UsdShade.Shader.Get(stage, material.GetPath().AppendChild('Color'))
        scale = np.asarray(tex.GetInput('scale').Get())[:3]
        bias = np.asarray(tex.GetInput('bias').Get())[:3]
        tint = np.asarray(provenance['color_tint_linear'])
        assert np.isfinite(scale).all() and (scale > 0).all()
        assert np.isfinite(bias).all() and (bias >= 0).all()
        assert np.isfinite(mean).all() and (mean > 0).all()
        assert 0 < provenance['color_blend'] <= 1
        assert mean * scale + bias == pytest.approx(tint, rel=1e-6, abs=1e-8)
        assert np.all(.5 * mean * scale + bias < tint)
        assert np.all(1.5 * mean * scale + bias > tint)


@pytest.mark.parametrize('suffix', ['Color', 'NormalGL', 'Roughness'])
def test_texture_hash_mismatch_does_not_author_materials(texture_root, suffix):
    stage = Usd.Stage.CreateInMemory()
    (texture_root / 'Metal009' / f'Metal009_1K-JPG_{suffix}.jpg').write_bytes(b'changed')
    with pytest.raises(ValueError, match='manifest'):
        pm.build_materials(stage, '/Looks', texture_root)
    assert not stage.GetPrimAtPath('/Looks')


def test_changed_color_and_manifest_cannot_reuse_an_unrelated_mean(texture_root):
    stage = Usd.Stage.CreateInMemory()
    image = texture_root / 'Metal009/Metal009_1K-JPG_Color.jpg'
    image.write_bytes(b'different scan')
    manifest = image.parent / 'SOURCE.json'
    source = load_json(manifest)
    next(entry for entry in source['files'] if entry['file'] == image.name)['sha256'] = hashlib.sha256(image.read_bytes()).hexdigest()
    manifest.write_text(json.dumps(source))
    with pytest.raises(ValueError, match='measured linear mean'):
        pm.build_materials(stage, '/Looks', texture_root)
    assert not stage.GetPrimAtPath('/Looks')


def test_visible_collision_mesh_styled_but_hidden_and_guide_meshes_untouched(scene):
    stage, root, materials = scene
    visible = quad(stage, '/Body/collision'); UsdPhysics.CollisionAPI.Apply(visible.GetPrim())
    hidden = quad(stage, '/Body/Hidden'); hidden.CreateVisibilityAttr().Set('invisible')
    guide = quad(stage, '/Body/Guide'); guide.CreatePurposeAttr().Set('guide')
    report = pm.style_visible_meshes(stage, root.GetPath(), 'steel_brushed', materials)
    assert report['styled_count'] == 1 and len(report['skipped_meshes']) == 2
    assert report['protected_before_sha256'] == report['protected_after_sha256']
    assert UsdGeom.PrimvarsAPI(visible).GetPrimvar('st').GetInterpolation() == 'faceVarying'
    assert not UsdGeom.PrimvarsAPI(hidden).HasPrimvar('st')
    assert not UsdGeom.PrimvarsAPI(guide).HasPrimvar('st')
    assert visible.GetPrim().HasAPI(UsdPhysics.CollisionAPI)


def test_uv_uses_shared_object_coordinates_and_physical_nested_scale(scene):
    stage, root, materials = scene
    UsdGeom.SetStageMetersPerUnit(stage, .01)
    xf = UsdGeom.Xformable(root)
    xf.AddTranslateOp().Set(Gf.Vec3d(10, 20, 30))
    xf.AddRotateZOp().Set(43.)
    xf.AddScaleOp().Set(Gf.Vec3f(3, 3, 3))
    first = quad(stage, '/Body/First')
    second = quad(stage, '/Body/Second')
    UsdGeom.Xformable(second).AddTranslateOp().Set(Gf.Vec3d(.02, 0, 0))
    before = [np.array(m.GetPointsAttr().Get()) for m in (first, second)]
    pm.style_visible_meshes(stage, '/Body', 'steel_brushed', materials)
    a, b = [np.array(UsdGeom.PrimvarsAPI(m).GetPrimvar('st').Get()) for m in (first, second)]
    assert a[1] == pytest.approx(b[0], abs=1e-8)
    assert a[1, 0] == pytest.approx(.02 * .01 * 3 / .08, abs=1e-8)
    for mesh, old in zip((first, second), before):
        assert np.array_equal(np.array(mesh.GetPointsAttr().Get()), old)


def test_preserve_distinguishes_colored_metal_from_colored_polymer(scene):
    stage, _, materials = scene
    red, blue = quad(stage, '/Body/Red'), quad(stage, '/Body/Blue')
    UsdShade.MaterialBindingAPI.Apply(red.GetPrim()).Bind(color_material(stage, '/OldRed', (.8, .1, .04), .8))
    UsdShade.MaterialBindingAPI.Apply(blue.GetPrim()).Bind(color_material(stage, '/OldBlue', (.08, .3, .5), 0.))
    result = pm.style_visible_meshes(stage, '/Body', 'preserve', materials)
    assert [r['role'] for r in result['styled_meshes']] == ['anodized_red', 'polymer_blue']


def test_explicit_role_and_longest_path_override_are_respected(scene):
    stage, _, materials = scene
    quad(stage, '/Body/First'); quad(stage, '/Body/Part/Second')
    report = pm.style_visible_meshes(stage, '/Body', {'*': 'steel_brushed', 'Part': 'rubber_graphite'}, materials)
    assert [r['role'] for r in report['styled_meshes']] == ['steel_brushed', 'rubber_graphite']


def test_physics_material_and_every_geometry_property_preserved(scene):
    stage, root, materials = scene
    mesh = quad(stage, '/Body/collision')
    UsdPhysics.CollisionAPI.Apply(mesh.GetPrim()); UsdPhysics.RigidBodyAPI.Apply(root)
    mass = UsdPhysics.MassAPI.Apply(root); mass.CreateMassAttr().Set(.2)
    mat = UsdShade.Material.Define(stage, '/Physics')
    physics = UsdPhysics.MaterialAPI.Apply(mat.GetPrim()); physics.CreateStaticFrictionAttr().Set(.37)
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(mat, materialPurpose='physics')
    mesh.GetPrim().CreateAttribute('physxCollision:contactOffset', Sdf.ValueTypeNames.Float).Set(.0001)
    signature = pm._protected_digest(root)
    resolved = pm._physics_material(mesh.GetPrim())
    result = pm.style_visible_meshes(stage, '/Body', 'steel_polished', materials)
    assert pm._protected_digest(root) == signature
    assert pm._physics_material(mesh.GetPrim()) == resolved
    assert result['geometry_transforms_physics_unchanged'] is True


def test_allpurpose_physics_material_rejected_before_any_mesh_mutation(scene):
    stage, _, materials = scene
    first = quad(stage, '/Body/First'); second = quad(stage, '/Body/Second')
    mat = UsdShade.Material.Define(stage, '/Physics')
    UsdPhysics.MaterialAPI.Apply(mat.GetPrim()).CreateDynamicFrictionAttr().Set(.4)
    UsdShade.MaterialBindingAPI.Apply(second.GetPrim()).Bind(mat)
    with pytest.raises(ValueError, match='All-purpose binding'):
        pm.style_visible_meshes(stage, '/Body', 'steel_brushed', materials)
    assert not first.GetNormalsAttr().HasAuthoredValue()
    assert not second.GetNormalsAttr().HasAuthoredValue()


def test_cylinder_side_smooth_but_cap_stays_hard():
    n = 32
    points = np.array([(math.cos(a), math.sin(a), z) for z in (0., 1.)
                       for a in np.arange(n) * 2 * math.pi / n])
    faces = [[i, (i + 1) % n, (i + 1) % n + n, i + n] for i in range(n)]
    faces += [list(reversed(range(n))), list(range(n, 2 * n))]
    counts = np.array([len(f) for f in faces]); indices = np.array([v for f in faces for v in f])
    normals, _, degenerate = pm._corner_normals(points, counts, indices)
    assert degenerate == 0
    assert normals[0] == pytest.approx([1, 0, 0], abs=1e-7)
    assert normals[n * 4:n * 5] == pytest.approx(np.tile([0, 0, -1], (n, 1)), abs=1e-7)
    assert normals[n * 5:] == pytest.approx(np.tile([0, 0, 1], (n, 1)), abs=1e-7)
    assert np.linalg.norm(normals, axis=1) == pytest.approx(np.ones(len(normals)))


@pytest.mark.parametrize('degrees,smooth', [(29, True), (31, False), (90, False)])
def test_crease_threshold_preserves_large_angle_edges(degrees, smooth):
    angle = math.radians(degrees)
    p = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, -math.cos(angle), math.sin(angle)]], float)
    normals, faces, _ = pm._corner_normals(p, np.array([3, 3]), np.array([0, 1, 2, 1, 0, 3]))
    assert np.allclose(normals[0], normals[4]) is smooth
    if not smooth:
        assert normals[0] == pytest.approx(faces[0])


def test_indexed_uv_is_replaced_without_changing_mesh_indices(scene):
    stage, _, materials = scene
    mesh = quad(stage, '/Body/Quad')
    uv = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar('st', Sdf.ValueTypeNames.TexCoord2fArray, 'vertex')
    uv.Set([(0, 0)]); uv.SetIndices([0, 0, 0, 0])
    pm.style_visible_meshes(stage, '/Body', 'steel_brushed', materials)
    assert not uv.IsIndexed()
    assert len(uv.Get()) == 4
    assert list(mesh.GetFaceVertexIndicesAttr().Get()) == [0, 1, 2, 3]


def test_overlay_reference_keeps_source_layer_bytes_unchanged(texture_root, tmp_path):
    source_path = tmp_path / 'physical.usda'
    base = Usd.Stage.CreateNew(str(source_path)); UsdGeom.SetStageMetersPerUnit(base, 1.)
    body = UsdGeom.Xform.Define(base, '/Body').GetPrim(); base.SetDefaultPrim(body)
    mesh = quad(base, '/Body/collision'); UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
    base.GetRootLayer().Save(); original = source_path.read_bytes()
    stage = Usd.Stage.CreateInMemory(); UsdGeom.SetStageMetersPerUnit(stage, 1.)
    root = stage.DefinePrim('/Object'); root.GetReferences().AddReference(str(source_path))
    materials = pm.build_materials(stage, '/Object/PresentationLooks', texture_root)
    result = pm.style_visible_meshes(stage, '/Object', 'anodized_yellow', materials)
    assert result['styled_count'] == 1
    assert source_path.read_bytes() == original
    assert not base.GetRootLayer().dirty
    assert len(UsdGeom.Mesh(stage.GetPrimAtPath('/Object/collision')).GetPointsAttr().Get()) == 4


def test_bad_topology_leaves_earlier_mesh_untouched(scene):
    stage, _, materials = scene
    first = quad(stage, '/Body/First'); broken = quad(stage, '/Body/Broken')
    broken.GetFaceVertexIndicesAttr().Set([0, 1, 2, 99])
    with pytest.raises(ValueError, match='topology'):
        pm.style_visible_meshes(stage, '/Body', 'steel_brushed', materials)
    assert not first.GetNormalsAttr().HasAuthoredValue()


def test_historical_zero_area_triangle_is_retained_with_finite_shading(scene):
    stage, _, materials = scene
    mesh = quad(stage, '/Body/Thread')
    mesh.GetFaceVertexCountsAttr().Set([3, 3, 3])
    mesh.GetFaceVertexIndicesAttr().Set([0, 1, 2, 0, 2, 3, 1, 2, 2])
    original = np.array(mesh.GetFaceVertexIndicesAttr().Get())
    result = pm.style_visible_meshes(stage, '/Body', 'steel_brushed', materials)
    assert result['styled_meshes'][0]['existing_zero_area_faces_retained'] == 1
    assert np.array_equal(mesh.GetFaceVertexIndicesAttr().Get(), original)
    assert np.isfinite(mesh.GetNormalsAttr().Get()).all()
    assert np.linalg.norm(mesh.GetNormalsAttr().Get(), axis=1) == pytest.approx(np.ones(9))
