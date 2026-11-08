import numpy as np
import pytest
from scripts.validation import elastic_sponge_compression as probe


def test_fixed_platen_self_weight_and_bounded_loading_are_separate():
    values=np.array([probe.force_for_step(i)[0] for i in range(6001)])
    extra=np.array([probe.force_for_step(i)[1] for i in range(6001)])
    assert extra.min()==-3. and extra.max()==0.
    assert np.allclose(values[:,2],.981+extra,atol=1e-7)
    assert np.all(values[:,:2]==0.)
    assert extra[800]==0. and extra[2000]==extra[2800]==-3. and extra[4000]==extra[-1]==0.
    assert np.all(np.diff(extra[800:2001])<=0.) and np.all(np.diff(extra[2800:4001])>=0.)
    with pytest.raises(ValueError):probe.force_for_step(6001)


def test_platen_is_dynamic_and_guide_connects_only_world_to_platen():
    Usd=pytest.importorskip('pxr.Usd');Physics=pytest.importorskip('pxr.UsdPhysics')
    stage=Usd.Stage.CreateInMemory();receipt=probe.author_platen(stage,'/World/PhysicsScene')
    body=stage.GetPrimAtPath(probe.PLATEN);joint=Physics.PrismaticJoint(stage.GetPrimAtPath(probe.JOINT))
    assert Physics.RigidBodyAPI(body).GetKinematicEnabledAttr().Get() is False
    assert Physics.MassAPI(body).GetMassAttr().Get()==pytest.approx(.1)
    assert joint.GetBody0Rel().GetTargets()==[] and list(map(str,joint.GetBody1Rel().GetTargets()))==[probe.PLATEN]
    assert receipt['passive_guide_stiffness_n_m']==200.
    assert Physics.DriveAPI(joint,'linear').GetTargetPositionAttr().Get()==0.
