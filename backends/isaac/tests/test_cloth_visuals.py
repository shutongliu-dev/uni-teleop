import numpy as np
import pytest

from bench.simulation.isaacsim_backend.cloth_visuals import closed_skin


def fixture_surface():
    points = np.array([[0., 0., .9], [.2, 0., .9], [.4, 0., .9],
                       [0., .3, .9], [.2, .3, .9], [.4, .3, .9]])
    faces = np.array([[0, 1, 3], [1, 4, 3], [1, 2, 4], [2, 5, 4]])
    return points, faces


def test_thickness_is_closed_manifold_without_changing_native_state():
    points, tri = fixture_surface()
    original = points.copy()
    skin, faces, normals = closed_skin(points, tri, .003)
    np.testing.assert_array_equal(points, original)
    np.testing.assert_allclose(skin[:len(points), 2], .9015)
    np.testing.assert_allclose(skin[len(points):, 2], .8985)
    edges = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    _, counts = np.unique(np.sort(edges, axis=1), axis=0, return_counts=True)
    assert (counts == 2).all()
    np.testing.assert_allclose(np.linalg.norm(normals, axis=1), 1.)
    # Every directed edge has exactly one opposite: all sides face outwards.
    assert set(map(tuple, edges)) == set(map(tuple, edges[:, ::-1]))


def test_folded_skin_follows_local_normals_and_retains_thickness():
    points, tri = fixture_surface()
    points[[2, 5], 0] = .21
    points[[2, 5], 2] += .2
    skin, _, _ = closed_skin(points, tri, .003)
    np.testing.assert_allclose((skin[:len(points)] + skin[len(points):]) / 2,
                               points, atol=1.e-7)
    np.testing.assert_allclose(np.linalg.norm(skin[:len(points)] - skin[len(points):],
                                              axis=1), .003, atol=1.e-7)


@pytest.mark.parametrize('thickness', [0., -.001, float('nan')])
def test_invalid_thickness_rejected(thickness):
    with pytest.raises(ValueError):
        closed_skin(*fixture_surface(), thickness)


def test_degenerate_native_geometry_is_not_silently_rendered_as_healthy():
    points, tri = fixture_surface()
    points[1] = points[0]
    with pytest.raises(ValueError, match='degenerate'):
        closed_skin(points, tri, .003)


def test_texture_readers_declare_standard_usd_output_types(tmp_path):
    """A normal input must not change the texture reader's registered rgb type."""
    from pxr import Sdf, Usd, UsdShade
    from bench.simulation.isaacsim_backend.cloth_visuals import ClosedClothVisual

    textures = {}
    for role in ('normal', 'roughness'):
        path = tmp_path / (role + '.jpg')
        path.write_bytes(b'network-only fixture; no image decoding')
        textures[role] = str(path)
    stage = Usd.Stage.CreateInMemory()
    material = ClosedClothVisual._material(stage, 'test', [.2, .4, .6], textures)
    base = str(material.GetPath())
    surface = UsdShade.Shader(stage.GetPrimAtPath(base + '/Surface'))
    reader = UsdShade.Shader(stage.GetPrimAtPath(base + '/ST'))
    normal = UsdShade.Shader(stage.GetPrimAtPath(base + '/normal'))
    roughness = UsdShade.Shader(stage.GetPrimAtPath(base + '/roughness'))
    assert reader.GetOutput('result').GetTypeName() == Sdf.ValueTypeNames.Float2
    assert normal.GetOutput('rgb').GetTypeName() == Sdf.ValueTypeNames.Float3
    assert roughness.GetOutput('r').GetTypeName() == Sdf.ValueTypeNames.Float
    assert surface.GetInput('normal').GetTypeName() == Sdf.ValueTypeNames.Normal3f
    assert surface.GetInput('normal').GetConnectedSource()[0].GetPath() == normal.GetPath()
    assert normal.GetInput('st').GetConnectedSource()[0].GetPath() == reader.GetPath()
    assert normal.GetInput('sourceColorSpace').Get() == 'raw'
    assert tuple(normal.GetInput('scale').Get()) == (2., 2., 2., 1.)
    assert tuple(normal.GetInput('bias').Get()) == (-1., -1., -1., 0.)
    # Fixing the port type must not silently add an albedo dependency or
    # override the frozen task's uniform color.
    assert not surface.GetInput('diffuseColor').HasConnectedSource()
    assert len([p for p in stage.Traverse() if p.GetAttribute('inputs:file')]) == 2


def test_albedo_preserves_each_linear_dye_mean_and_uses_color_output(tmp_path):
    from PIL import Image
    from pxr import Sdf, Usd, UsdShade
    from bench.simulation.isaacsim_backend.cloth_visuals import ClosedClothVisual

    pixels = np.array([[[8, 100, 200], [240, 180, 70]]], dtype=np.uint8)
    path = tmp_path / 'albedo.png'
    Image.fromarray(pixels).save(path)
    srgb = pixels.astype(float) / 255.
    linear = np.where(srgb <= .04045, srgb / 12.92, ((srgb + .055) / 1.055) ** 2.4)
    for name, color in [('blue', [.12, .36, .54]), ('ochre', [.56, .32, .12])]:
        stage = Usd.Stage.CreateInMemory()
        mat = ClosedClothVisual._material(stage, name, color, {'albedo': str(path)})
        base = str(mat.GetPath())
        texture = UsdShade.Shader(stage.GetPrimAtPath(base + '/albedo'))
        surface = UsdShade.Shader(stage.GetPrimAtPath(base + '/Surface'))
        assert texture.GetInput('sourceColorSpace').Get() == 'sRGB'
        assert texture.GetOutput('rgb').GetTypeName() == Sdf.ValueTypeNames.Float3
        assert surface.GetInput('diffuseColor').GetConnectedSource()[0].GetPath() == texture.GetPath()
        scale = np.asarray(texture.GetInput('scale').Get())[:3]
        np.testing.assert_allclose((linear * scale).mean(axis=(0, 1)), color, rtol=1.e-7)


def test_albedo_cannot_normalize_an_absent_color_channel(tmp_path):
    from PIL import Image
    from bench.simulation.isaacsim_backend.cloth_visuals import linear_albedo_mean

    path = tmp_path / 'black.png'
    Image.new('RGB', (2, 2), (0, 120, 180)).save(path)
    with pytest.raises(ValueError, match='zero mean channel'):
        linear_albedo_mean(path)


@pytest.mark.parametrize('tile', [None, [.367053595, .363200009]])
def test_explicit_texture_scale_keeps_closed_geometry_and_native_mesh_unchanged(tile):
    from pxr import Usd, UsdGeom, Vt
    from bench.simulation.isaacsim_backend.cloth_visuals import ClosedClothVisual

    points, triangles = fixture_surface()
    stage = Usd.Stage.CreateInMemory()
    native = UsdGeom.Mesh.Define(stage, '/World/Cloths/towel')
    native.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(points.astype(np.float32)))
    before = np.asarray(native.GetPointsAttr().Get()).copy()
    visual = ClosedClothVisual(stage, str(native.GetPath()), 'towel', triangles, points,
        .003, [.12, .36, .54], texture_tile_size_m=tile)
    uv = np.asarray(UsdGeom.PrimvarsAPI(visual.mesh).GetPrimvar('st').Get())
    # The legacy default produces exactly the original float32 / .20 UVs.
    expected = points.astype(np.float32)[:, :2] / (
        .20 if tile is None else np.asarray(tile, dtype=np.float32))
    np.testing.assert_array_equal(uv, np.concatenate([expected, expected]))
    np.testing.assert_array_equal(np.asarray(native.GetPointsAttr().Get()), before)
    np.testing.assert_array_equal(np.asarray(visual.mesh.GetPointsAttr().Get()),
                                  closed_skin(before, triangles, .003)[0])
    assert not native.GetPrim().GetAppliedSchemas()
    assert not any('Physics' in name for name in visual.mesh.GetPrim().GetAppliedSchemas())
    assert ('texture_tile_size_m' in visual.metadata) == (tile is not None)


@pytest.mark.parametrize('tile', [.2, [.2], [0., .2], [.2, float('nan')]])
def test_bad_explicit_texture_scale_is_rejected(tile):
    from pxr import Usd
    from bench.simulation.isaacsim_backend.cloth_visuals import ClosedClothVisual

    points, triangles = fixture_surface()
    with pytest.raises(ValueError, match='positive finite XY pair'):
        ClosedClothVisual(Usd.Stage.CreateInMemory(), '/missing', 'towel', triangles,
            points, .003, [.12, .36, .54], texture_tile_size_m=tile)


def test_vertex_normals_survive_deformation_and_usd_roundtrip(tmp_path):
    """Normal buffers remain valid when native Hydra requires vertex data."""
    from pxr import Usd, UsdGeom
    from bench.simulation.isaacsim_backend.cloth_visuals import ClosedClothVisual

    points, triangles = fixture_surface()
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.Mesh.Define(stage, '/World/Cloths/towel')
    visual = ClosedClothVisual(stage, '/World/Cloths/towel', 'towel', triangles,
                               points, .003, [.12, .36, .54])
    assert len(visual.mesh.GetNormalsAttr().Get()) == len(visual.mesh.GetPointsAttr().Get())
    points[[2, 5], 2] += .08
    visual.sync(points)
    path = tmp_path/'normal_roundtrip.usda'
    stage.GetRootLayer().Export(str(path))
    reopened = Usd.Stage.Open(str(path))
    mesh = UsdGeom.Mesh.Get(reopened, visual.path)
    normals = np.asarray(mesh.GetNormalsAttr().Get())
    assert mesh.GetNormalsInterpolation() == UsdGeom.Tokens.vertex
    assert len(normals) != len(mesh.GetFaceVertexIndicesAttr().Get())
    assert len(normals) == len(mesh.GetPointsAttr().Get())
    np.testing.assert_allclose(np.linalg.norm(normals, axis=1), 1., atol=1.e-7)
    assert not UsdGeom.PrimvarsAPI(mesh).GetPrimvar('normals').HasValue()


def test_unchanged_normals_are_dirtied_atomically_with_moving_points():
    """Hydra must see normals dirty even when their numerical value is unchanged."""
    from pxr import Tf, Usd, UsdGeom
    from bench.simulation.isaacsim_backend.cloth_visuals import ClosedClothVisual

    points, triangles = fixture_surface()
    # Binary-exact XY coordinates keep normals bitwise identical on translation.
    points[:, :2] = np.array([[0., 0.], [.25, 0.], [.5, 0.],
                              [0., .25], [.25, .25], [.5, .25]])
    points = points.astype(np.float32)
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.Mesh.Define(stage, '/World/Cloths/towel')
    visual = ClosedClothVisual(stage, '/World/Cloths/towel', 'towel', triangles,
                               points, .003, [.12, .36, .54])
    before = np.asarray(visual.mesh.GetNormalsAttr().Get()).copy()
    notices = []

    def changed(notice, sender):
        notices.append((set(map(str, notice.GetChangedInfoOnlyPaths())),
                        np.asarray(visual.mesh.GetNormalsAttr().Get()).copy()))

    registration = Tf.Notice.Register(Usd.Notice.ObjectsChanged, changed, stage)
    try:
        for offset in (.25, .5):
            visual.sync(points + [offset, 0., 0.])
    finally:
        registration.Revoke()
    assert len(notices) == 2
    for paths, normals in notices:
        assert visual.path + '.points' in paths
        assert visual.path + '.normals' in paths
        np.testing.assert_array_equal(normals, before)
        assert len(normals) == len(visual.mesh.GetPointsAttr().Get())
