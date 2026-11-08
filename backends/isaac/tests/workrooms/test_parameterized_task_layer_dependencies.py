"""Exercise the same dependency gate used by the native study loader."""
from copy import deepcopy
import hashlib
from pathlib import Path

import pytest
from pxr import Sdf, Usd, UsdGeom, UsdPhysics

from scenes.workrooms.parameterized.usd_authoring import (
    check_task_layer_admission, task_layer_dependencies)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@pytest.fixture
def authored_layer(tmp_path):
    texture = tmp_path/'surface.png'; texture.write_bytes(b'actual texture bytes')
    auxiliary = tmp_path/'second.png'; auxiliary.write_bytes(b'array texture bytes')
    inherited = tmp_path/'inherited.usda'
    lower = Usd.Stage.CreateNew(str(inherited))
    UsdGeom.Xform.Define(lower, '/World/StudyTask')
    lower.GetRootLayer().Save()
    path = tmp_path/'task.usda'; stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageMetersPerUnit(stage, 1.)
    UsdGeom.SetStageUpAxis(stage, 'Z')
    stage.GetRootLayer().subLayerPaths.append('inherited.usda')
    root = UsdGeom.Xform.Define(stage, '/World/StudyTask').GetPrim()
    root.CreateAttribute('singleTexture', Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath('surface.png'))
    root.CreateAttribute('textureArray', Sdf.ValueTypeNames.AssetArray).Set(
        [Sdf.AssetPath('second.png'), Sdf.AssetPath('')])
    body = UsdGeom.Xform.Define(stage, '/World/StudyTask/book').GetPrim()
    UsdPhysics.RigidBodyAPI.Apply(body)
    UsdPhysics.MassAPI.Apply(body).CreateMassAttr(.25)
    collider = UsdGeom.Cube.Define(stage, '/World/StudyTask/book/Collision')
    collider.CreateSizeAttr(.02); UsdPhysics.CollisionAPI.Apply(collider.GetPrim())
    stage.GetRootLayer().Save()
    spec = {'id': 'task', 'usd_path': str(path), 'sha256': sha(path),
        'source_prim_path': '/World/StudyTask', 'target_prim_path': '/World/ImportedTask',
        'rigid_bodies': {'book': '/World/ImportedTask/book'}, 'dependencies_sha256': {}}
    return spec, [path, inherited, texture, auxiliary]


def test_add_actual_dependencies_then_native_admission_without_asset_edits(authored_layer):
    spec, files = authored_layer
    before = {str(path): sha(path) for path in files}
    original = deepcopy(spec)
    with pytest.raises(ValueError, match='Unpinned task layer composition'):
        check_task_layer_admission(spec)
    dependencies = task_layer_dependencies(spec)
    assert spec == original
    assert dependencies == {str(path): before[str(path)] for path in files[1:]}
    patched = {**spec, 'dependencies_sha256': dependencies}
    result = check_task_layer_admission(patched)
    assert result['passed'] and result['observed_bodies'] == ['book']
    assert {str(path): sha(path) for path in files} == before
    # The newly discovered array dependency remains required by the real loader.
    del patched['dependencies_sha256'][str(files[-1])]
    with pytest.raises(ValueError, match='Unpinned task layer asset'):
        check_task_layer_admission(patched)


def test_never_refresh_stale_source_or_existing_dependency_pins(authored_layer):
    spec, files = authored_layer
    spec['dependencies_sha256'] = task_layer_dependencies(spec)
    files[-1].write_bytes(b'replaced texture')
    with pytest.raises(ValueError, match='hash mismatch'):
        task_layer_dependencies(spec)
    spec['sha256'] = '0'*64
    with pytest.raises(ValueError, match='hash mismatch'):
        task_layer_dependencies(spec)


def test_missing_asset_does_not_become_a_valid_declaration(authored_layer):
    spec, files = authored_layer
    files[-1].unlink()
    with pytest.raises(ValueError, match='missing/nonabsolute asset'):
        task_layer_dependencies(spec)


def test_dependency_completion_does_not_waive_body_inventory(authored_layer):
    spec, _ = authored_layer
    spec['dependencies_sha256'] = task_layer_dependencies(spec)
    spec['rigid_bodies'] = {}
    with pytest.raises(ValueError, match='body inventory differs'):
        check_task_layer_admission(spec)
