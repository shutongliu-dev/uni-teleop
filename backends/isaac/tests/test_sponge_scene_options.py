import pytest
from scripts.validation.sponge_scene_options import apply_velocity_iteration_clamp, velocity_iteration_receipt


def scene():
    Usd=pytest.importorskip('pxr.Usd'); UsdPhysics=pytest.importorskip('pxr.UsdPhysics')
    stage=Usd.Stage.CreateInMemory(); prim=UsdPhysics.Scene.Define(stage,'/World/PhysicsScene').GetPrim()
    return stage,prim


def test_omission_does_not_author_scene_or_change_existing_values():
    stage,prim=scene(); before=stage.GetRootLayer().ExportToString()
    receipt=apply_velocity_iteration_clamp(prim,None)
    assert receipt['default_preserved'] and stage.GetRootLayer().ExportToString()==before
    apply_velocity_iteration_clamp(prim,0); before=stage.GetRootLayer().ExportToString()
    assert apply_velocity_iteration_clamp(prim,None)['composed_values']=={'physxScene:minVelocityIterationCount':0,'physxScene:maxVelocityIterationCount':0}
    assert stage.GetRootLayer().ExportToString()==before


def test_explicit_zero_clamps_min_and_max_without_body_override():
    stage,prim=scene(); receipt=apply_velocity_iteration_clamp(prim,0)
    assert not receipt['default_preserved']
    assert all(v==0 for v in receipt['composed_values'].values())
    text=stage.GetRootLayer().ExportToString()
    assert 'uniform uint physxScene:minVelocityIterationCount = 0' in text
    assert 'uniform uint physxScene:maxVelocityIterationCount = 0' in text
    assert 'physxRigidBody:' not in text
    prim.GetAttribute('physxScene:maxVelocityIterationCount').Set(1)
    with pytest.raises(ValueError,match='not composed'): velocity_iteration_receipt(prim,0)


@pytest.mark.parametrize('bad',[1,-1,True,0.,'0'])
def test_invalid_diagnostic_values_rejected_before_mutation(bad):
    stage,prim=scene(); before=stage.GetRootLayer().ExportToString()
    with pytest.raises(ValueError): apply_velocity_iteration_clamp(prim,bad)
    assert stage.GetRootLayer().ExportToString()==before
