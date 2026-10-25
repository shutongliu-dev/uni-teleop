"""CPU USD checks for reversible room lighting, separate from renderer acceptance."""
from types import SimpleNamespace

import pytest

pytest.importorskip('pxr.Usd')
from pxr import Usd, UsdGeom, UsdLux, UsdPhysics

from bench.simulation.assets.workrooms.common import apply_lighting_profile, light
from bench.simulation.assets.workrooms.layouts import FAMILY_LIGHTING


def _stage(family='laboratory'):
    stage=Usd.Stage.CreateInMemory()
    world=UsdGeom.Xform.Define(stage,'/World')
    stage.SetDefaultPrim(world.GetPrim())
    b=SimpleNamespace(stage=stage,family=family)
    light(b,'Panel',(0,0,2),(0,0,0),(.8,.4),1000.,(1,1,1))
    light(b,'Task',(0,1,1.4),(0,1,.8),(.3,.1),400.,(1,1,1),role='task',temperature=4000.)
    collider=UsdGeom.Cube.Define(stage,'/World/Geometry/Support')
    UsdPhysics.CollisionAPI.Apply(collider.GetPrim())
    return stage


@pytest.mark.parametrize('family',FAMILY_LIGHTING)
def test_variants_restore_nominal_and_preserve_geometry(family):
    stage=_stage(family)
    geometry=stage.GetPrimAtPath('/World/Geometry/Support')
    before=stage.GetRootLayer().ExportToString().split('def "Geometry"')[1]
    nominal=apply_lighting_profile(stage,family)
    low=apply_lighting_profile(stage,family,'low_light')
    assert all(a['intensity']<b['intensity'] for a,b in zip(low['lights'],nominal['lights']))
    assert low['lights'][0]['temperature_k']==FAMILY_LIGHTING[family]['temperature_k']
    warm=apply_lighting_profile(stage,family,'warm')
    assert warm['lights'][0]['temperature_k']<nominal['lights'][0]['temperature_k']
    assert apply_lighting_profile(stage,family)==nominal
    assert geometry.HasAPI(UsdPhysics.CollisionAPI)
    assert stage.GetRootLayer().ExportToString().split('def "Geometry"')[1]==before


def test_profile_on_referenced_room_preserves_source_and_metadata(tmp_path):
    source=tmp_path/'room.usda'
    source_stage=_stage()
    source_stage.GetRootLayer().Export(str(source))
    frozen=source.read_bytes()
    stage=Usd.Stage.CreateInMemory()
    room=stage.DefinePrim('/World/Room','Xform')
    room.GetReferences().AddReference(str(source),'/World')
    result=apply_lighting_profile(stage,'laboratory','daylight',light_root='/World/Room/Lights')
    assert all(x['path'].startswith('/World/Room/Lights/') for x in result['lights'])
    assert result['lights'][0]['nominal_intensity']==1500.
    assert result['lights'][0]['intensity']==pytest.approx(1050.)
    assert source.read_bytes()==frozen
    assert stage.GetPrimAtPath('/World/Room/Lights').GetCustomDataByKey('scence:lightingProfile')=='daylight'


def test_untracked_lights_do_not_silently_receive_guessed_baselines():
    stage=_stage()
    UsdLux.RectLight.Define(stage,'/World/Lights/Unknown')
    with pytest.raises(ValueError,match='nominal profile metadata'):
        apply_lighting_profile(stage,'laboratory')


def test_generated_textures_belong_to_new_output_only(tmp_path):
    from bench.simulation.assets.workrooms.builder import Builder
    assets=tmp_path/'shared_assets'
    output=tmp_path/'new_build'
    builder=Builder(Usd.Stage.CreateInMemory(),assets,output)
    assert builder.generated==output/'generated'/'textures'
    assert builder.generated.is_dir()
    assert not (assets/'scenes').exists()
