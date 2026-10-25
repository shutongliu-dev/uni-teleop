"""Empty study task surfaces must survive authoring and fail closed on intrusion."""
from pathlib import Path

import numpy as np
import pytest

from bench.paths import DATA_ROOT

pytest.importorskip('pxr.Usd')
from pxr import Usd, UsdGeom, UsdPhysics

from bench.simulation.assets.workrooms.study_room import build_study_room, SHELF_BAYS
from bench.simulation.assets.workrooms.common import apply_lighting_profile
from scripts.assets.build_study_room import audit_environment


@pytest.fixture(scope='module')
def study(tmp_path_factory):
    stage = Usd.Stage.CreateInMemory()
    scene = build_study_room(stage, Path(f'{DATA_ROOT}/assets'),
                             tmp_path_factory.mktemp('study_environment'))
    return stage, scene


def test_rectangular_empty_desk_and_open_rear_classification_bays(study):
    stage, scene = study
    audit = audit_environment(stage)
    np.testing.assert_allclose(audit['table_bounds_m'], [[.20, -1., .70], [1.10, 1., .78]], atol=1.e-7)
    assert scene['task_independent'] and scene['objects'] == []
    assert len(audit['shelf_bays']) >= 3
    for bay in audit['shelf_bays']:
        assert bay['empty'] and bay['high'][0] < -.23
        assert bay['high'][2] - bay['low'][2] >= .25
    assert audit['physics_scenes'] == audit['robot_articulations'] == 0


@pytest.mark.parametrize('target', ['desk', 'robot', 'shelf'])
def test_new_decor_cannot_occupy_reserved_operation_space(study, target):
    stage, scene = study
    positions = {'desk': [.65, 0., .95], 'robot': [-.23, 0., 1.0],
                 'shelf': ((np.array(SHELF_BAYS[0]['low']) + SHELF_BAYS[0]['high']) / 2).tolist()}
    path = '/World/Geometry/invalid_decoration'
    with Usd.EditContext(stage, stage.GetSessionLayer()):
        cube = UsdGeom.Cube.Define(stage, path)
        cube.CreateSizeAttr(.06)
        UsdGeom.Xformable(cube).AddTranslateOp().Set(tuple(positions[target]))
        try:
            with pytest.raises(ValueError, match='intrudes'):
                audit_environment(stage)
        finally:
            stage.RemovePrim(path)


def test_visible_shelf_without_enabled_collision_is_rejected(study):
    stage, _ = study
    collider = stage.GetPrimAtPath(SHELF_BAYS[0]['collider_path'])
    with Usd.EditContext(stage, stage.GetSessionLayer()):
        attr = UsdPhysics.CollisionAPI(collider).CreateCollisionEnabledAttr(False)
        try:
            with pytest.raises(ValueError, match='Disabled static support'):
                audit_environment(stage)
        finally:
            attr.Clear()


def test_hidden_guide_collider_cannot_occupy_desk(study):
    stage, _ = study
    path = '/World/Geometry/invalid_group'
    with Usd.EditContext(stage, stage.GetSessionLayer()):
        UsdGeom.Xform.Define(stage, path)
        visual = UsdGeom.Cube.Define(stage, path + '/Visual')
        visual.CreateSizeAttr(.06)
        UsdGeom.Xformable(visual).AddTranslateOp().Set((2., 0., .9))
        collider = UsdGeom.Cube.Define(stage, path + '/Collider')
        collider.CreateSizeAttr(.06)
        UsdGeom.Xformable(collider).AddTranslateOp().Set((.65, 0., .9))
        collider.CreatePurposeAttr('guide')
        collider.CreateVisibilityAttr('invisible')
        UsdPhysics.CollisionAPI.Apply(collider.GetPrim())
        try:
            with pytest.raises(ValueError, match='collider intrudes'):
                audit_environment(stage)
        finally:
            stage.RemovePrim(path)


def test_disabled_floor_is_rejected(study):
    stage, _ = study
    collider = stage.GetPrimAtPath('/World/Geometry/room_floor/Collider')
    with Usd.EditContext(stage, stage.GetSessionLayer()):
        attr = UsdPhysics.CollisionAPI(collider).CreateCollisionEnabledAttr(False)
        try:
            with pytest.raises(ValueError, match='Disabled static support'):
                audit_environment(stage)
        finally:
            attr.Clear()


def test_task_dynamics_cannot_be_baked_into_fixed_room(study):
    stage, _ = study
    with Usd.EditContext(stage, stage.GetSessionLayer()):
        UsdPhysics.Scene.Define(stage, '/World/UnexpectedPhysics')
        try:
            with pytest.raises(ValueError, match='runtime actor'):
                audit_environment(stage)
        finally:
            stage.RemovePrim('/World/UnexpectedPhysics')


def test_lighting_variants_restore_the_authored_baseline(study):
    stage, scene = study
    apply_lighting_profile(stage, 'study', 'low_light')
    restored = apply_lighting_profile(stage, 'study', 'nominal')
    assert restored['lights'] == scene['lighting']['lights']


def _robot_preview(environment, output_root):
    from bench.simulation.assets.workrooms.builder import Builder
    from scenes.workrooms.study.preview import add_robot_preview

    stage = Usd.Stage.CreateInMemory()
    root = UsdGeom.Xform.Define(stage, '/World')
    root.GetPrim().GetReferences().AddReference(environment.GetRootLayer().identifier, '/World')
    builder = Builder(stage, f'{DATA_ROOT}/assets', output_root, reuse_materials=True)
    return stage, add_robot_preview(stage, builder)


def test_real_robot_preview_clears_desk_and_shelf_without_changing_fixed_environment(study, tmp_path):
    environment, _ = study
    before = environment.GetRootLayer().ExportToString()
    _, result = _robot_preview(environment, tmp_path)
    audit = result['clearance_audit']
    assert audit['status'] == 'passed'
    assert audit['desk_collider_count'] == len(audit['desk_checks']) > 0
    assert audit['robot_or_stand_to_desk_shape_aabb_gap_m'] > .001
    assert result['stand']['foot_to_table_edge_m'] == pytest.approx(.25)
    assert result['source_rigid_body_count'] == 43 and result['physics_steps'] == 0
    assert environment.GetRootLayer().ExportToString() == before
    assert not environment.GetPrimAtPath('/World/TaskRobot')


@pytest.mark.parametrize('target', ['robot', 'stand'])
def test_hidden_desk_cabinet_collider_cannot_intrude_into_robot_or_stand(study, tmp_path, target):
    from scenes.workrooms.study.preview import clearance_audit

    environment, _ = study
    stage, _ = _robot_preview(environment, tmp_path)
    if target == 'robot':
        target_prim = next(prim for prim in Usd.PrimRange(stage.GetPrimAtPath('/World/TaskRobot'),
                                                        Usd.TraverseInstanceProxies())
                           if prim.IsA(UsdGeom.Boundable))
    else:
        target_prim = stage.GetPrimAtPath('/World/Geometry/robot_stand_column/Collider')
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ['default', 'render', 'proxy', 'guide'], False, True)
    target_bounds = cache.ComputeWorldBound(target_prim).ComputeAlignedRange()
    center = (np.array(target_bounds.GetMin()) + np.array(target_bounds.GetMax())) / 2
    collider = UsdGeom.Cube.Define(stage, '/World/Geometry/desk_pedestal_intrusion/Collider')
    collider.CreateSizeAttr(.06)
    collider.CreatePurposeAttr('guide')
    collider.CreateVisibilityAttr('invisible')
    UsdGeom.Xformable(collider).AddTranslateOp().Set(tuple(center))
    enabled = UsdPhysics.CollisionAPI.Apply(collider.GetPrim()).CreateCollisionEnabledAttr(False)
    assert clearance_audit(stage)['status'] == 'passed'
    enabled.Set(True)
    with pytest.raises(ValueError, match='robot or stand.*desk collider'):
        clearance_audit(stage)
