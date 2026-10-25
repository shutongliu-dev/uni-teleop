"""Offline composition must preserve source materials, dimensions and robot FK."""
import hashlib
from pathlib import Path

import pytest

from bench.paths import DATA_ROOT

pytest.importorskip('pxr.Usd')
from pxr import Usd, UsdGeom, UsdPhysics

from bench.simulation.assets.workrooms.builder import Builder
from scenes.workrooms.mechanical_assembly.preview import add_robot_preview
from scenes.workrooms.mechanical_assembly.threaded_assets import add_threaded_preview
from scenes.workrooms.mechanical_assembly.threaded_preview import (
    V3_CATALOG_ROOT, add_sharpa_threaded_preview, add_threaded_tool_preview)


def test_preview_reuses_environment_materials_and_checks_all_robot_bodies(tmp_path):
    source = Usd.Stage.CreateNew(str(tmp_path / 'source.usda'))
    Builder(source, f'{DATA_ROOT}/assets', tmp_path / 'source')
    source.GetRootLayer().Save()
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.Xform.Define(stage, '/World').GetPrim().GetReferences().AddReference(str(tmp_path / 'source.usda'))
    builder = Builder(stage, f'{DATA_ROOT}/assets', tmp_path / 'preview', reuse_materials=True)
    assert not stage.GetRootLayer().GetPrimAtPath('/World/Looks/steel')
    composition = add_robot_preview(stage, builder)
    assert composition['source_rigid_body_count'] == 43
    assert composition['posed_link_count'] >= 43
    assert composition['physics_steps'] == 0
    assert composition['robot']['base_position_m'] == [-.23, 0., .78]
    for prim in Usd.PrimRange(stage.GetPrimAtPath('/World/TaskRobot')):
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            assert UsdPhysics.RigidBodyAPI(prim).GetRigidBodyEnabledAttr().Get() is False
    assert not stage.GetRootLayer().GetPrimAtPath('/World/Looks/steel')


def test_requested_threaded_assets_keep_original_scale_and_table_support(tmp_path):
    stage = Usd.Stage.CreateInMemory()
    builder = Builder(stage, f'{DATA_ROOT}/assets', tmp_path)
    report = add_threaded_preview(stage, builder)
    assert set(report['placements']) == {'screwdriver', 'bolt', 'fixture'}
    for name, row in report['placements'].items():
        assert row['world_bounds_m'][0][2] == pytest.approx(.78, abs=1.e-7), name
        assert row['source']['scale'] == [1., 1., 1.]
    assert not report['physics_or_reset_acceptance']


def test_current_threaded_preview_references_v3_without_double_scaling_or_source_changes(tmp_path):
    stage = Usd.Stage.CreateInMemory()
    builder = Builder(stage, f'{DATA_ROOT}/assets', tmp_path)
    report = add_threaded_tool_preview(stage, builder)
    assert report['profile'] == 'sharpa_v3'
    expected = {'screwdriver': (1., 3, [.044, .211, .044]),
                'bolt': (1.5, 8, [.045, .045, .0642]),
                'fixture': (1.5, 4, [.24, .15, .057])}
    for name, (scale, collider_count, dimensions) in expected.items():
        row = report['placements'][name]
        assert '/threaded_sharpa/v3/' in row['source']['path']
        assert row['source']['source_scale'] == [scale] * 3
        assert row['source']['additional_scale'] == [1., 1., 1.]
        assert row['world_bounds_m'][0][2] == pytest.approx(.78, abs=1.e-7)
        assert [hi - lo for lo, hi in zip(*row['world_bounds_m'])] == pytest.approx(dimensions, abs=1.e-7)
        disabled = row['disabled_in_preview_layer_only']
        assert len(disabled['rigid_bodies']) == 1
        assert len(disabled['colliders']) == collider_count
        for prim in Usd.PrimRange(stage.GetPrimAtPath(row['preview_prim'])):
            if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                assert UsdPhysics.RigidBodyAPI(prim).GetRigidBodyEnabledAttr().Get() is False
            if prim.HasAPI(UsdPhysics.CollisionAPI):
                assert UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get() is False
        # The new layer authors references/placement/disable opinions only.
        asset_spec = stage.GetRootLayer().GetPrimAtPath(row['preview_prim'] + '/Asset')
        assert list(asset_spec.referenceList.prependedItems)[0].assetPath == row['source']['path']
        source = Usd.Stage.Open(row['source']['path'])
        bodies = [p for p in source.Traverse() if p.HasAPI(UsdPhysics.RigidBodyAPI)]
        assert len(bodies) == 1 and UsdPhysics.RigidBodyAPI(bodies[0]).GetRigidBodyEnabledAttr().Get()
        assert hashlib.sha256(Path(row['source']['path']).read_bytes()).hexdigest() == row['source']['sha256']
    text = stage.GetRootLayer().ExportToString()
    assert 'point3f[] points' not in text and 'faceVertexIndices' not in text
    assert all(path in builder.dependencies for path in report['dependencies_sha256'])
    assert any(path.endswith('DRIVER_LICENSE.txt') for path in report['dependencies_sha256'])
    assert report['physics_steps'] == 0 and not report['physics_or_reset_acceptance']
    assert not report['source_assets_modified']


def test_legacy_threaded_preview_is_only_selected_explicitly(tmp_path):
    stage = Usd.Stage.CreateInMemory()
    report = add_threaded_tool_preview(stage, Builder(stage, f'{DATA_ROOT}/assets', tmp_path),
                                       profile='requested_v1')
    assert report['profile'] == 'requested_v1'
    assert '/threaded_tools/v1/' in report['placements']['screwdriver']['source']['path']
    with pytest.raises(ValueError, match='Unknown threaded preview profile'):
        add_threaded_tool_preview(None, None, profile='latest')


def test_v3_preview_refuses_a_changed_catalog_before_authoring(tmp_path):
    (tmp_path / 'manifest.json').write_text((V3_CATALOG_ROOT / 'manifest.json').read_text() + ' ')
    stage = Usd.Stage.CreateInMemory()
    before = stage.GetRootLayer().ExportToString()
    with pytest.raises(ValueError, match='preview catalog changed'):
        add_sharpa_threaded_preview(stage, None, catalog_root=tmp_path)
    assert stage.GetRootLayer().ExportToString() == before
