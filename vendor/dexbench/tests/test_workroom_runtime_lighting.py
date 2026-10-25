"""Only explicit runtime lighting options may override a referenced room."""
import pytest

from bench.simulation.isaacsim_backend.workroom import apply_runtime_lighting


@pytest.mark.parametrize('config', [
    {}, {'schema': 'wrong', 'family': 'laboratory', 'profile': 'nominal'},
    {'schema': 'workroom-lighting-v1', 'family': 'study', 'profile': 'nominal'},
    {'schema': 'workroom-lighting-v1', 'family': 'laboratory', 'profile': 'nominal', 'extra': True},
])
def test_bad_runtime_lighting_contract_fails_before_stage_edit(config):
    with pytest.raises(ValueError):
        apply_runtime_lighting(None, {'family': 'laboratory'}, config)


def test_no_lighting_option_requires_no_stage_or_asset_builder():
    assert apply_runtime_lighting(None, {'family': 'laboratory'}) is None


def test_runtime_lighting_uses_referenced_root_and_records_effective_values():
    pytest.importorskip('pxr.Usd')
    from pxr import Usd, UsdLux
    stage = Usd.Stage.CreateInMemory()
    light = UsdLux.RectLight.Define(stage, '/World/Room/Lights/Ceiling')
    api = UsdLux.LightAPI(light.GetPrim())
    api.CreateIntensityAttr(100.)
    api.CreateColorTemperatureAttr(6500.)
    prim = light.GetPrim()
    prim.SetCustomDataByKey('scence:lightRole', 'ceiling')
    prim.SetCustomDataByKey('scence:nominalIntensity', 100.)
    prim.SetCustomDataByKey('scence:nominalTemperatureK', 6500.)
    before = stage.GetRootLayer().ExportToString()
    assert apply_runtime_lighting(stage, {'family': 'laboratory'}) is None
    assert stage.GetRootLayer().ExportToString() == before
    config = {'schema': 'workroom-lighting-v1', 'family': 'laboratory', 'profile': 'low_light'}
    result = apply_runtime_lighting(stage, {'family': 'laboratory'}, config)
    assert result['light_root'] == '/World/Room/Lights'
    assert result['profile'] == 'low_light'
    assert result['lights'][0]['intensity'] == api.GetIntensityAttr().Get() < 100.
    config['profile'] = 'nominal'
    apply_runtime_lighting(stage, {'family': 'laboratory'}, config)
    assert api.GetIntensityAttr().Get() == 100.
    config['profile'] = 'unknown'
    with pytest.raises(ValueError, match='Unknown lighting profile'):
        apply_runtime_lighting(stage, {'family': 'laboratory'}, config)
