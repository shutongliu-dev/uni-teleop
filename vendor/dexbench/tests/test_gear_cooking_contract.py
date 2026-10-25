"""An authored asset option must survive composition before native queries."""
from copy import deepcopy
import json

import pytest

from scripts.assets.prepare_gear_sector_weld import WELD0_COOKING_PROFILE
from scripts.validation.gear_sector_sdf_checks import validate_cooking_profile, cooking_schema_readback


def contract():
    profile=deepcopy(WELD0_COOKING_PROFILE)
    return {'parameters':deepcopy(profile['sdf_parameters']),'cooking_profile':profile}


def test_legacy_contract_does_not_require_new_profile(tmp_path):
    assert validate_cooking_profile({},tmp_path) is None


def test_new_candidate_cannot_omit_profile_from_contract(tmp_path):
    (tmp_path/'SOURCE.json').write_text(json.dumps({'cooking_profile':WELD0_COOKING_PROFILE}))
    with pytest.raises(ValueError,match='missing'):
        validate_cooking_profile({},tmp_path)


def test_exact_profile_requires_matching_candidate_provenance(tmp_path):
    value=contract()
    source=tmp_path/'SOURCE.json'
    source.write_text(json.dumps({'cooking_profile':value['cooking_profile']}))
    assert validate_cooking_profile(value,tmp_path)==value['cooking_profile']
    source.write_text('{}')
    with pytest.raises(ValueError,match='provenance'):
        validate_cooking_profile(value,tmp_path)


@pytest.mark.parametrize('damage',['enabled','wrong_api','missing_shape','wrong_storage','bool_zero'])
def test_undeclared_cooking_variants_rejected(tmp_path,damage):
    value=contract();profile=value['cooking_profile']
    if damage=='enabled':profile['weld_tolerance']=1.e-5
    elif damage=='wrong_api':profile['weld_api']='PhysxTriangleMeshSimplificationCollisionAPI'
    elif damage=='missing_shape':profile['shape_paths'].pop()
    elif damage=='wrong_storage':value['parameters']['sdfSubgridResolution']=0
    elif damage=='bool_zero':profile['weld_tolerance']=False
    with pytest.raises(ValueError,match='profile'):
        validate_cooking_profile(value,tmp_path)


class Prim:
    def __init__(self,*,api=True,authored=True,value=0.):
        self.api,self.authored,self.value=api,authored,value
    def GetAttribute(self,name):
        assert name==WELD0_COOKING_PROFILE['weld_attribute']
        return self
    def HasAuthoredValueOpinion(self):return self.authored
    def Get(self):return self.value
    def GetAppliedSchemas(self):return [WELD0_COOKING_PROFILE['weld_api']] if self.api else []


@pytest.mark.parametrize('damage',[{'api':False},{'authored':False},{'value':float('-inf')},{'value':1.e-5}])
def test_composition_cannot_silently_drop_or_override_profile(damage):
    result=cooking_schema_readback(Prim(**damage),WELD0_COOKING_PROFILE)
    assert not result['declared_profile_matches']
    json.dumps(result,allow_nan=False)


def test_explicit_zero_is_retained_without_cooked_topology_claim():
    result=cooking_schema_readback(Prim(),WELD0_COOKING_PROFILE)
    assert result['declared_profile_matches'] and result['value_authored']
    assert result['value']==0.
    assert 'not a direct cooked-mesh' in result['scope']
