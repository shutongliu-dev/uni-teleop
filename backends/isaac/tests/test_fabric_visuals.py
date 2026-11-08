import hashlib
import json

import numpy as np
import pytest

from bench.simulation.isaacsim_backend.fabric_visuals import (
    SkinTopology, cloth_geometry_digest, load_visual_profile, loop_subdivision)


def sheet(z=.9):
    points = np.array([[0., 0., z], [.2, 0., z], [.4, 0., z],
                       [0., .3, z], [.2, .3, z], [.4, .3, z]])
    faces = np.array([[0, 1, 3], [1, 4, 3], [1, 2, 4], [2, 5, 4]])
    return points, faces


def sewn_panels(gap=.0015):
    """Two identical panels sewn on every boundary edge except the y = 0 opening."""
    panel, tri = sheet(0.)
    count = len(panel)
    rest = np.r_[panel + [0., 0., gap], panel - [0., 0., gap]]
    faces = [*tri.tolist(), *(tri[:, ::-1] + count).tolist()]
    edges = np.concatenate([tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]])
    _, inverse, counts = np.unique(np.sort(edges, axis=1), axis=0, return_inverse=True, return_counts=True)
    for a, b in edges[counts[inverse.ravel()] == 1]:
        if panel[a, 1] == 0. and panel[b, 1] == 0.:
            continue
        faces += [[int(b), int(a), int(a + count)], [int(b), int(a + count), int(b + count)]]
    return rest, np.asarray(faces, dtype=np.int32)


def handedness(points, uv, faces):
    e1, e2 = points[faces[:, 1]] - points[faces[:, 0]], points[faces[:, 2]] - points[faces[:, 0]]
    d1, d2 = uv[faces[:, 1]] - uv[faces[:, 0]], uv[faces[:, 2]] - uv[faces[:, 0]]
    tangent = e1 * d2[:, 1:2] - e2 * d1[:, 1:2]
    bitangent = e2 * d1[:, 0:1] - e1 * d2[:, 0:1]
    det = d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0]
    return np.sign(np.einsum('ij,ij->i', np.cross(tangent, bitangent), np.cross(e1, e2)) * np.sign(det))


def frame(rest, atlas=None):
    return {'origin_xy_m': (rest[:, :2].min(axis=0) - .01).tolist(),
            'size_xy_m': (np.ptp(rest[:, :2], axis=0) + .02).tolist(), 'atlas': atlas}


def test_loop_subdivision_is_affine_and_keeps_a_flat_sheet_flat():
    points, tri = sheet()
    operator, children = loop_subdivision(tri, len(points), 1)
    np.testing.assert_allclose(np.asarray(operator.sum(axis=1)).ravel(), 1.)
    assert len(children) == 4 * len(tri) and operator.shape[1] == len(points)
    np.testing.assert_allclose((operator @ points)[:, 2], .9)
    with pytest.raises(ValueError, match='0, 1 or 2'):
        loop_subdivision(tri, len(points), 3)


def test_two_panel_atlas_splits_only_seams_and_keeps_outer_frames_right_handed():
    rest, tri = sewn_panels()
    topology = SkinTopology(tri, len(rest), 1, rest_vertices=rest, atlas='front_back')
    np.testing.assert_array_equal(topology.source[topology.faces], topology.skin_faces)
    uv = topology.uv(rest, frame(rest, 'front_back'))
    assert uv.min() >= 0. and uv.max() <= 1.
    points, normals = topology.render(rest, .0012)
    outer = topology.faces[:len(topology.triangles)]
    centre_z = (topology.operator @ rest[:, 2])[topology.triangles].mean(axis=1)
    np.testing.assert_array_equal(uv[outer].mean(axis=1)[:, 0] >= .5, centre_z < 0.)
    assert (handedness(points, uv, outer) > 0).all()
    # Seam copies share points and smooth normals; the skin itself is unchanged.
    assert len(topology.source) > 2 * topology.count
    skin_points, skin_normals = topology.skin(rest, .0012)
    np.testing.assert_array_equal(points, skin_points[topology.source])
    np.testing.assert_array_equal(normals, skin_normals[topology.source])


def test_two_sided_sheet_shows_a_right_handed_mirror_after_folding():
    points, tri = sheet()
    topology = SkinTopology(tri, len(points), 1, atlas='two_sided')
    uv = topology.uv(points, frame(points, 'two_sided'))
    skin, _ = topology.render(points, .004)
    m = len(topology.triangles)
    top, bottom = topology.faces[:m], topology.faces[m:2 * m]
    assert (uv[top][..., 0] <= .5).all() and (uv[bottom][..., 0] >= .5).all()
    assert (handedness(skin, uv, top) > 0).all() and (handedness(skin, uv, bottom) > 0).all()


def write_profile(tmp_path, rest, tri, dye=(.19, .39, .40), **changes):
    from PIL import Image
    textures = {}
    for role, mode, value in (('albedo', 'RGB', (120, 140, 150)), ('normal', 'RGB', (128, 128, 255)),
                              ('roughness', 'L', 220)):
        Image.new(mode, (4, 4), value).save(tmp_path / (role + '.png'))
        textures[role] = role + '.png'
    profile = {'schema': 'fabric_visual_profile.v1', 'id': 'fixture', 'shader': 'omnisurface',
        'subdivision_levels': 1, 'uv_frame': frame(rest), 'textures': textures,
        'texture_sha256': {role: hashlib.sha256((tmp_path / name).read_bytes()).hexdigest()
                           for role, name in textures.items()},
        'cloth_binding': {'cloth_geometry_sha256': cloth_geometry_digest(rest, tri), 'dye_linear_rgb': list(dye)},
        'omnisurface': {'specular_retro_reflection_weight': .4, 'specular_retro_reflection_color': [.5, .6, .6]},
        'render_only': True, 'physics_changed': False}
    profile.update(changes)
    path = tmp_path / 'profile.json'
    path.write_text(json.dumps(profile))
    return {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def test_profile_loader_pins_every_byte_and_rejects_unknown_inputs(tmp_path):
    points, tri = sheet()
    pin = write_profile(tmp_path, points, tri)
    profile = load_visual_profile(pin)
    assert profile['textures']['albedo'] == str(tmp_path / 'albedo.png')
    with pytest.raises(ValueError, match='Frozen input changed'):
        load_visual_profile({**pin, 'sha256': '0' * 64})
    (tmp_path / 'roughness.png').write_bytes((tmp_path / 'roughness.png').read_bytes() + b'\0')
    with pytest.raises(ValueError, match='Frozen input changed'):
        load_visual_profile(pin)
    cases = (({'shader': 'toon'}, 'Invalid fabric'),
             ({'textures': {'albedo': '../albedo.png'}, 'texture_sha256': {'albedo': '0' * 64}},
              'beside their profile'),
             ({'omnisurface': {'specular_reflection_ior': 9.}}, 'out-of-range'),
             ({'physics_changed': True}, 'Invalid fabric'))
    for index, (changes, message) in enumerate(cases):
        directory = tmp_path / f'case{index}'
        directory.mkdir()
        with pytest.raises(ValueError, match=message):
            load_visual_profile(write_profile(directory, points, tri, **changes))


def test_visual_binds_rest_mesh_and_dye_and_leaves_native_mesh_untouched(tmp_path):
    from pxr import Sdf, Usd, UsdGeom, UsdShade, Vt
    from bench.simulation.isaacsim_backend.fabric_visuals import FabricClothVisual

    points, tri = sheet()
    profile = load_visual_profile(write_profile(tmp_path, points, tri))
    stage = Usd.Stage.CreateInMemory()
    native = UsdGeom.Mesh.Define(stage, '/World/Cloths/towel')
    native.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(points.astype(np.float32)))
    before = np.asarray(native.GetPointsAttr().Get()).copy()
    with pytest.raises(ValueError, match='different cloth rest mesh'):
        FabricClothVisual(stage, '/World/Cloths/towel', 'other', tri, points * 1.01, .003, profile, points)
    with pytest.raises(ValueError, match='frozen cloth colour'):
        FabricClothVisual(stage, '/World/Cloths/towel', 'tinted', tri, points, .003, profile, points,
                          dye_linear_rgb=[.2, .39, .40])
    visual = FabricClothVisual(stage, '/World/Cloths/towel', 'towel', tri, points, .003, profile, points,
                               dye_linear_rgb=[.19, .39, .40])
    mesh = visual.mesh
    count = len(mesh.GetPointsAttr().Get())
    assert len(UsdGeom.PrimvarsAPI(mesh).GetPrimvar('st').Get()) == count
    assert UsdGeom.PrimvarsAPI(mesh).GetPrimvar('st').GetInterpolation() == UsdGeom.Tokens.vertex
    assert len(mesh.GetNormalsAttr().Get()) == count and mesh.GetNormalsInterpolation() == UsdGeom.Tokens.vertex
    np.testing.assert_array_equal(np.asarray(native.GetPointsAttr().Get()), before)
    assert not any('Physics' in name for name in mesh.GetPrim().GetAppliedSchemas())
    shader = UsdShade.Shader(stage.GetPrimAtPath('/World/Looks/fabric_skin_towel/Shader'))
    assert shader.GetSourceAsset('mdl').path == 'OmniSurface.mdl'
    assert shader.GetSourceAssetSubIdentifier('mdl') == 'OmniSurface'
    # Single-channel scalar maps have no alpha: OmniImage must read their red channel.
    assert shader.GetInput('specular_reflection_roughness_image_alpha_mode').Get() == 1
    assert shader.GetInput('diffuse_reflection_color_image').GetAttr().GetColorSpace() == 'sRGB'
    assert shader.GetInput('geometry_normal_image').GetAttr().GetColorSpace() == 'raw'
    assert visual.metadata['profile'] == profile['pin'] and visual.metadata['collision'] is False
    with pytest.raises(ValueError, match='complete finite native cloth surface'):
        visual.sync(points[:-1])
    del Sdf


def test_preview_shader_uses_standard_output_types(tmp_path):
    from pxr import Sdf, Usd, UsdShade
    from bench.simulation.isaacsim_backend.fabric_visuals import FabricClothVisual

    points, tri = sheet()
    profile = load_visual_profile(write_profile(tmp_path, points, tri, shader='preview', ior=1.3))
    stage = Usd.Stage.CreateInMemory()
    material = FabricClothVisual._preview(stage, 'towel', profile['textures'], profile['ior'])
    base = str(material.GetPath())
    surface = UsdShade.Shader(stage.GetPrimAtPath(base + '/Surface'))
    normal = UsdShade.Shader(stage.GetPrimAtPath(base + '/normal'))
    assert normal.GetOutput('rgb').GetTypeName() == Sdf.ValueTypeNames.Float3
    assert surface.GetInput('normal').GetTypeName() == Sdf.ValueTypeNames.Normal3f
    assert surface.GetInput('ior').Get() == pytest.approx(1.3)
    assert surface.GetInput('metallic').Get() == 0.


def test_sync_dirties_points_and_normals_together(tmp_path):
    from pxr import Tf, Usd, UsdGeom
    from bench.simulation.isaacsim_backend.fabric_visuals import FabricClothVisual

    points, tri = sheet()
    points[:, :2] = np.array([[0., 0.], [.25, 0.], [.5, 0.], [0., .25], [.25, .25], [.5, .25]])
    profile = load_visual_profile(write_profile(tmp_path, points, tri))
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.Mesh.Define(stage, '/World/Cloths/towel')
    visual = FabricClothVisual(stage, '/World/Cloths/towel', 'towel', tri, points, .003, profile, points)
    notices = []
    registration = Tf.Notice.Register(Usd.Notice.ObjectsChanged,
        lambda notice, sender: notices.append(set(map(str, notice.GetChangedInfoOnlyPaths()))), stage)
    try:
        visual.sync(points + [.25, 0., 0.])
    finally:
        registration.Revoke()
    assert len(notices) == 1
    assert {visual.path + '.points', visual.path + '.normals'} <= notices[0]
