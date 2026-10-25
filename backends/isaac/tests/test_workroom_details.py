"""Actual CPU USD geometry checks for the new perimeter details, not physics acceptance."""
from pathlib import Path

import numpy as np
import pytest

from bench.paths import DATA_ROOT

pytest.importorskip('pxr.Usd')
from pxr import Usd, UsdGeom, UsdPhysics

from bench.simulation.assets.workrooms import FAMILIES
from bench.simulation.assets.workrooms.builder import Builder
from bench.simulation.assets.workrooms.common import _room_materials
from bench.simulation.assets.workrooms.interiors import _materials, build_functional_details

ASSETS=Path(f'{DATA_ROOT}/assets')


def _detail_stage(family,tmp_path):
    if not (ASSETS/'fonts/noto_cjk_sc/v1/NotoSansCJKsc-Regular.otf').is_file():
        pytest.skip('Shared label font is not installed')
    stage=Usd.Stage.CreateInMemory()
    b=Builder(stage,ASSETS,tmp_path/family)
    b.family=family
    _room_materials(b,family)
    _materials(b)
    # Only the new detail fragment is built, never six complete room releases.
    # This entry mirrors the referenced commode's published grounded bounds.
    b.records.append(dict(name='tea_chinese_commode',position=[.35,1.64,0.],bounds_size=[1.886,.491,.8483470937983811]))
    build_functional_details(b,family)
    return family,stage


@pytest.fixture(params=FAMILIES)
def detail_stage(request,tmp_path):
    return _detail_stage(request.param,tmp_path)


def test_perimeter_details_leave_workcell_and_robot_volume_clear(detail_stage):
    family,stage=detail_stage
    cache=UsdGeom.BBoxCache(Usd.TimeCode.Default(),['default','render'])
    protected_low=np.array([-.35,-.65,.74])
    protected_high=np.array([1.25,.65,1.75])
    for prim in stage.GetPrimAtPath('/World/Geometry').GetChildren():
        bounds=cache.ComputeWorldBound(prim).ComputeAlignedRange()
        low=np.array(bounds.GetMin());high=np.array(bounds.GetMax())
        assert np.isfinite(low).all() and np.isfinite(high).all(),str(prim.GetPath())
        overlap=np.minimum(high,protected_high)-np.maximum(low,protected_low)
        assert not (overlap>1.e-6).all(),(family,str(prim.GetPath()),low,high)


def test_new_solid_details_have_positive_static_collision_transforms(detail_stage):
    _,stage=detail_stage
    cache=UsdGeom.XformCache(Usd.TimeCode.Default())
    colliders=[p for p in stage.Traverse() if p.HasAPI(UsdPhysics.CollisionAPI)]
    assert len(colliders)>=4
    for prim in colliders:
        transform=np.array(cache.GetLocalToWorldTransform(prim))
        assert np.isfinite(transform).all()
        assert np.linalg.det(transform[:3,:3])>0,str(prim.GetPath())
    assert not any(p.IsA(UsdPhysics.Joint) or p.HasAPI(UsdPhysics.RigidBodyAPI) for p in stage.Traverse())


def test_plate_bottoms_rest_on_drainboard(tmp_path):
    _,stage=_detail_stage('kitchen',tmp_path)
    cache=UsdGeom.BBoxCache(Usd.TimeCode.Default(),['default','render'])
    tray=cache.ComputeWorldBound(stage.GetPrimAtPath('/World/Geometry/dish_rack_drainboard')).ComputeAlignedRange()
    for i in range(3):
        plate=cache.ComputeWorldBound(stage.GetPrimAtPath(f'/World/Geometry/dish_rack_plate{i}')).ComputeAlignedRange()
        assert plate.GetMin()[2]==pytest.approx(tray.GetMax()[2],abs=1.e-6)
