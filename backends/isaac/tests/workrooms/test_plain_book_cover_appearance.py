"""Only explicit drawer/reading book backgrounds may receive this finish."""
import json
from pathlib import Path

import numpy as np
import pytest

from bench.paths import load_json
from scenes.workrooms.parameterized.spec import sample_parameters
from scenes.workrooms.study import plain_book_cover_appearance as cover


def stage_for(task):
    from pxr import Usd
    recipe = cover.ASSETS[task]
    manifest = load_json(Path(recipe['manifest']))
    return Usd.Stage.Open(Usd.Stage.Open(manifest['initial_usd']).Flatten())


def snapshot(stage):
    return {str(p.GetPath()): (p.GetTypeName(), str(p.GetCustomData()), cover._attributes(p),
        {r.GetName(): tuple(map(str, r.GetTargets())) for r in p.GetRelationships()})
        for p in stage.Traverse()}


@pytest.mark.parametrize('task', cover.ASSETS)
def test_only_two_visual_bindings_change_and_shader_keeps_original_uv(tmp_path, task):
    from pxr import UsdGeom, UsdShade
    stage = stage_for(task); recipe = cover.ASSETS[task]
    path = '/World/StudyTask/Objects/'+recipe['actor']
    params = sample_parameters('study', task, 'single_arm', 9)
    before = snapshot(stage)
    result = cover.apply_plain_book_cover_background(stage, {recipe['actor']: path}, params, tmp_path/'new')
    assert result['pattern_scale_m_applied'] is False
    after = snapshot(stage)
    for mesh in result['meshes']:
        assert after[mesh][3]['material:binding'] == (result['new_material'],)
        after[mesh][3]['material:binding'] = before[mesh][3]['material:binding']
        assert after[mesh] == before[mesh]
    original_rows = {p: row for p,row in after.items() if p in before}
    assert original_rows == before
    assert all(p.startswith(result['new_material']) for p in after.keys()-before.keys())
    shader = UsdShade.Shader(stage.GetPrimAtPath(result['new_material']+'/Surface'))
    original = UsdShade.Material(stage.GetPrimAtPath(result['original_material'])).ComputeSurfaceSource()[0]
    for old in original.GetInputs():
        if old.GetBaseName() != 'diffuseColor':
            current = shader.GetInput(old.GetBaseName())
            assert current.Get() == old.Get()
            assert current.GetAttr().GetConnections() == old.GetAttr().GetConnections()
    # Original Builder UV = local metre coordinates times two. Shader-only
    # scale maps the broad X/Y face to [0,1]^2 without rewriting any primvar.
    mesh = stage.GetPrimAtPath(result['meshes'][0])
    uv = np.asarray(UsdGeom.PrimvarsAPI(mesh).GetPrimvar('st').Get())
    points = np.asarray(UsdGeom.Mesh(mesh).GetPointsAttr().Get())
    normals = np.asarray(UsdGeom.Mesh(mesh).GetNormalsAttr().Get())
    broad = np.abs(normals[:,2]) > .999
    mapped = uv[broad]*result['shader_uv_scale']
    assert np.allclose(mapped, (points[broad,:2]+np.array(recipe['size_xy_m'])/2)/recipe['size_xy_m'], atol=1e-7)
    assert np.ptp(mapped[:,0]) > .95 and np.ptp(mapped[:,1]) > .95
    with pytest.raises(ValueError, match='material is not'):
        cover.apply_plain_book_cover_background(stage, {recipe['actor']: path}, params, tmp_path/'repeat')


@pytest.mark.parametrize('task', cover.ASSETS)
def test_canonical_no_change_and_modified_cover_rejected(tmp_path, task):
    from pxr import UsdGeom
    stage = stage_for(task); recipe = cover.ASSETS[task]
    path = '/World/StudyTask/Objects/'+recipe['actor']; before = snapshot(stage)
    result = cover.apply_plain_book_cover_background(stage, {recipe['actor']:path},
        sample_parameters('study', task, 'single_arm', 0), tmp_path/'zero')
    assert result['canonical'] and snapshot(stage) == before and not (tmp_path/'zero').exists()
    mesh = stage.GetPrimAtPath(path+'/Geometry/cover_front/Visual')
    uv = UsdGeom.PrimvarsAPI(mesh).GetPrimvar('st'); points = uv.Get(); points[0]=(100,100); uv.Set(points)
    with pytest.raises(ValueError, match='local geometry, UV'):
        cover.apply_plain_book_cover_background(stage, {recipe['actor']:path},
            sample_parameters('study', task, 'single_arm', 9), tmp_path/'invalid')
    assert not (tmp_path/'invalid').exists()
