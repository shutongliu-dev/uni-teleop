import numpy as np
import pytest
from pxr import Gf, Usd, UsdGeom, UsdPhysics, Vt
from scenes.workrooms.seed_geometry import bake_body_scale, checked_seed, planar_offset


def test_conditioned_rotated_nested_mesh_bakes_once_and_preserves_collision():
    s=Usd.Stage.CreateInMemory(); b=UsdGeom.Xform.Define(s,'/Body').GetPrim()
    UsdPhysics.RigidBodyAPI.Apply(b); mass=UsdPhysics.MassAPI.Apply(b)
    mass.CreateMassAttr(2.); mass.CreateCenterOfMassAttr(Gf.Vec3f(.01,.02,.03))
    mass.CreateDiagonalInertiaAttr(Gf.Vec3f(.03,.04,.05));mass.CreatePrincipalAxesAttr(Gf.Quatf(1))
    x=UsdGeom.Xform.Define(s,'/Body/Conditioned');x.AddTranslateOp().Set((.02,.04,.06));x.AddRotateZOp().Set(23);x.AddScaleOp().Set((.001,.002,.003))
    m=UsdGeom.Mesh.Define(s,'/Body/Conditioned/Mesh');m.CreatePointsAttr([(-10,-20,-30),(10,20,30),(10,-20,30)])
    m.CreateFaceVertexCountsAttr([3]);m.CreateFaceVertexIndicesAttr([0,1,2]);UsdPhysics.CollisionAPI.Apply(m.GetPrim())
    def points():
        t=UsdGeom.XformCache().GetLocalToWorldTransform(m.GetPrim())
        return np.array([t.Transform(Gf.Vec3d(*p)) for p in m.GetPointsAttr().Get()])
    original=points(); ops=[str(op.Get()) for op in x.GetOrderedXformOps()]
    bake_body_scale(s,'/Body',[.96,1.04,1.])
    assert np.allclose(points(),original*np.array([.96,1.04,1]),atol=1e-8)
    assert [str(op.Get()) for op in x.GetOrderedXformOps()]==ops
    assert m.GetPrim().HasAPI(UsdPhysics.CollisionAPI)
    assert mass.GetMassAttr().Get()==pytest.approx(2*.96*1.04)
    assert list(mass.GetCenterOfMassAttr().Get())==pytest.approx([.0096,.0208,.03])


@pytest.mark.parametrize('bad',[True,-1,10,1.0,None])
def test_seed_rejects_ambiguous_values(bad):
    with pytest.raises(ValueError):checked_seed(bad)


def test_seed_bounds_and_reproducibility():
    assert planar_offset(0,'vial')==([0.,0.],0.)
    assert planar_offset(3,'vial')==planar_offset(3,'vial')
    assert planar_offset(3,'vial')!=planar_offset(3,'rack')
    for seed in range(10):
        xy,yaw=planar_offset(seed,'vial')
        assert max(abs(v) for v in xy)<=.012
        assert abs(yaw)<=np.deg2rad(7)


@pytest.mark.parametrize('scale',[[-1,-1,1],[0,1,1],[1,float('nan'),1]])
def test_invalid_affine_rejected(scale):
    with pytest.raises(ValueError):bake_body_scale(Usd.Stage.CreateInMemory(),'/Body',scale)
