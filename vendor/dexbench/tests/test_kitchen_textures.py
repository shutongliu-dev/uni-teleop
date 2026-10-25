"""Appearance changes preserve exact contact geometry and reset identity."""
from hashlib import sha256

import numpy as np
import pytest
from pxr import Usd, UsdGeom, UsdPhysics, UsdShade

from scenes.workrooms.kitchen.assets import TASK_IDS, author_task
from scenes.workrooms.kitchen.textures import write_task_textures
from scripts.validation.kitchen_asset_identity import diagnostic_physics_identity


@pytest.mark.parametrize('task_id', TASK_IDS)
def test_texture_changes_no_physical_attributes_or_body_metadata(task_id):
    plain, textured = Usd.Stage.CreateInMemory(), Usd.Stage.CreateInMemory()
    metadata = author_task(plain, task_id)
    assert metadata == author_task(textured, task_id,
        texture_assets={'wood_albedo': 'textures/oak_albedo.png'})
    count = 0
    for prim in plain.Traverse():
        if (prim.HasAPI(UsdPhysics.CollisionAPI) or prim.HasAPI(UsdPhysics.RigidBodyAPI)
                or prim.HasAPI(UsdPhysics.MaterialAPI)):
            count += 1
            other = textured.GetPrimAtPath(prim.GetPath())
            assert prim.GetAllMetadata() == other.GetAllMetadata()
            assert [str(a.GetName()) for a in prim.GetAttributes()] == [str(a.GetName()) for a in other.GetAttributes()]
            for attr in prim.GetAttributes():
                assert attr.Get() == other.GetAttribute(attr.GetName()).Get()
            for relationship in prim.GetRelationships():
                assert relationship.GetTargets() == other.GetRelationship(relationship.GetName()).GetTargets()
    assert count > len(metadata['actors'])


def test_wood_uvs_have_area_on_floor_and_vertical_rim_without_extra_vertices():
    stage = Usd.Stage.CreateInMemory()
    meta = author_task(stage, 'breakfast_serving',
        texture_assets={'wood_albedo': 'textures/oak_albedo.png'})
    path = meta['actors']['tray']
    visual = UsdGeom.Mesh(stage.GetPrimAtPath(path+'/Visual/shell'))
    collision = UsdGeom.Mesh(stage.GetPrimAtPath(path+'/Collision/shell'))
    assert visual.GetPointsAttr().Get() == collision.GetPointsAttr().Get()
    assert visual.GetFaceVertexIndicesAttr().Get() == collision.GetFaceVertexIndicesAttr().Get()
    uv = UsdGeom.PrimvarsAPI(visual).GetPrimvar('st')
    assert uv.GetInterpolation() == 'faceVarying'
    triangle_uv = np.asarray(uv.Get()).reshape(-1, 3, 2)
    a, b = triangle_uv[:, 1]-triangle_uv[:, 0], triangle_uv[:, 2]-triangle_uv[:, 0]
    assert np.all(np.abs(a[:, 0]*b[:, 1]-a[:, 1]*b[:, 0]) > 1e-12)
    # One top plane has one affine mapping: sparse floor triangles cannot form
    # the former displayColor X, regardless of their vertex count.
    points = np.asarray(visual.GetPointsAttr().Get())
    triangles = points[np.asarray(visual.GetFaceVertexIndicesAttr().Get()).reshape(-1, 3)]
    flat = np.ptp(triangles[:, :, 2], axis=1) < 1e-8
    np.testing.assert_allclose(triangle_uv[flat], triangles[flat, :, :2]/[.4, .28], atol=1e-7)
    surface = UsdShade.Shader(stage.GetPrimAtPath('/World/KitchenTask/Materials/wood/Surface'))
    source = surface.GetInput('diffuseColor').GetConnectedSource()[0]
    assert str(source.GetPrim().GetPath()).endswith('/wood/Albedo')
    assert not visual.GetDisplayColorPrimvar().HasAuthoredValue()


def test_relative_texture_identity_is_content_bound_and_reproducible(tmp_path):
    paths, metadata, textures = [], [], []
    for name, presentation in (('canonical', False), ('goal', True)):
        directory = tmp_path/name
        directory.mkdir()
        references = write_task_textures(directory)
        paths.append(directory/'task.usda')
        stage = Usd.Stage.CreateNew(str(paths[-1]))
        metadata.append(author_task(stage, 'breakfast_serving', presentation=presentation,
            texture_assets=references))
        stage.GetRootLayer().Save()
        textures.append(directory/references['wood_albedo'])
    assert sha256(textures[0].read_bytes()).digest() == sha256(textures[1].read_bytes()).digest()
    result = diagnostic_physics_identity(*paths, *metadata)
    assert result['passed'], result
    with pytest.raises(FileExistsError):
        write_task_textures(paths[0].parent)
    # Equal authored relative names must not hide changed dependency content.
    textures[1].write_bytes(textures[1].read_bytes()+b'changed')
    assert not diagnostic_physics_identity(*paths, *metadata)['passed']


@pytest.mark.parametrize('task_id,path_suffix', [
    ('breakfast_serving', 'Actors/tray/Visual/shell'),
    ('utensil_sorting', 'Fixtures/source_pad/Visual/pad'),
    ('pot_lid_placement', 'Fixtures/source_mat/Visual/mat'),
])
def test_large_flat_surfaces_have_planar_normals_without_changing_contact_points(task_id, path_suffix):
    stage = Usd.Stage.CreateInMemory()
    author_task(stage, task_id)
    path = '/World/KitchenTask/'+path_suffix
    visual = UsdGeom.Mesh(stage.GetPrimAtPath(path))
    collision = UsdGeom.Mesh(stage.GetPrimAtPath(path.replace('/Visual/', '/Collision/')))
    assert visual.GetPointsAttr().Get() == collision.GetPointsAttr().Get()
    assert visual.GetFaceVertexIndicesAttr().Get() == collision.GetFaceVertexIndicesAttr().Get()
    assert visual.GetNormalsInterpolation() == 'faceVarying'
    assert collision.GetNormalsInterpolation() == 'vertex'
    points = np.asarray(visual.GetPointsAttr().Get())
    triangles = points[np.asarray(visual.GetFaceVertexIndicesAttr().Get()).reshape(-1, 3)]
    cross = np.cross(triangles[:, 1]-triangles[:, 0], triangles[:, 2]-triangles[:, 0])
    flat_up = (np.ptp(triangles[:, :, 2], axis=1) < 1e-8) & (cross[:, 2] > 1e-10)
    assert flat_up.sum() >= 10
    normals = np.asarray(visual.GetNormalsAttr().Get()).reshape(-1, 3, 3)[flat_up]
    np.testing.assert_allclose(normals, np.broadcast_to([0., 0., 1.], normals.shape), atol=1e-6)
