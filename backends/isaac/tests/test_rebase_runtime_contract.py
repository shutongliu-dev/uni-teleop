from copy import deepcopy
import hashlib
from pathlib import Path

import pytest

from scripts.maintenance.rebase_runtime_contract import canonical_hash, rebase_runtime_contract


@pytest.fixture
def sample(tmp_path):
    old, new = tmp_path/'old', tmp_path/'new'
    old.mkdir(); new.mkdir()
    for root in (old, new):
        (root/'front.py').write_text('front')
        (root/'native.py').write_text('native')
    inventory = {'front.py': hashlib.sha256(b'front').hexdigest(),
                 str(old/'native.py'): hashlib.sha256(b'native').hexdigest()}
    return old, new, {'runtime_source_inventory': inventory,
        'runtime_code_sha256': canonical_hash(inventory), 'sensor_config_sha256': 'a'*64}


def test_verified_relocation_preserves_every_digest_and_other_contract_field(sample):
    old,new,contract = sample
    result = rebase_runtime_contract(contract, old, new)
    assert result['runtime_source_inventory'] == {
        'front.py': contract['runtime_source_inventory']['front.py'],
        str(new/'native.py'): contract['runtime_source_inventory'][str(old/'native.py')]}
    assert result['sensor_config_sha256'] == contract['sensor_config_sha256']
    assert result['runtime_code_sha256'] != contract['runtime_code_sha256']
    assert len(result['path_rebase']['sources']) == 2


@pytest.mark.parametrize('fault', ['old_bytes','new_bytes','bad_original_hash','escape',
                                  'source_symlink','destination_symlink','external','collision'])
def test_rejects_unverifiable_or_ambiguous_inventory(sample, fault, tmp_path):
    old,new,contract = sample
    if fault == 'old_bytes': (old/'native.py').write_text('changed')
    elif fault == 'new_bytes': (new/'native.py').write_text('changed')
    elif fault == 'bad_original_hash': contract['runtime_code_sha256'] = '0'*64
    elif fault in ('source_symlink','destination_symlink'):
        root = old if fault == 'source_symlink' else new
        (root/'native.py').unlink(); (root/'native.py').symlink_to(old/'front.py')
    else:
        inventory = contract['runtime_source_inventory']
        key = '../front.py' if fault=='escape' else str(new/'native.py')
        inventory[key] = inventory[str(old/'native.py')]
        contract['runtime_code_sha256'] = canonical_hash(inventory)
    with pytest.raises(ValueError):
        rebase_runtime_contract(contract, old, new,
            shared_files=[new/'native.py'] if fault=='collision' else [])


def test_explicit_shared_source_keeps_key_and_still_checks_bytes(sample,tmp_path):
    old,new,contract = sample
    shared=tmp_path/'shared.py'; shared.write_text('shared')
    contract['runtime_source_inventory'][str(shared)] = hashlib.sha256(b'shared').hexdigest()
    contract['runtime_code_sha256'] = canonical_hash(contract['runtime_source_inventory'])
    result=rebase_runtime_contract(contract,old,new,shared_files=[shared])
    assert result['runtime_source_inventory'][str(shared)] == contract['runtime_source_inventory'][str(shared)]
    shared.write_text('changed')
    with pytest.raises(ValueError): rebase_runtime_contract(contract,old,new,shared_files=[shared])


@pytest.mark.parametrize('field', ['implementation_sha256', 'controller_descriptor',
    'system0', 'execution_environment_sha256', 'source_transition',
    'integration_expected_sources', 'unknown_runtime_identity'])
def test_relocation_rejects_opaque_root_sensitive_identities(sample, field):
    old, new, contract = sample
    original = deepcopy(contract)
    contract[field] = {'path': str(old/'native.py'), 'sha256': 'b'*64}
    with pytest.raises(ValueError, match='Unsupported contract fields/schema'):
        rebase_runtime_contract(contract, old, new)
    assert {key: value for key, value in contract.items() if key != field} == original


@pytest.mark.parametrize('value', ['sensor', None, {'path': '/old/sensor.py'}])
def test_optional_content_pin_must_be_a_digest(sample, value):
    old, new, contract = sample
    contract['sensor_config_sha256'] = value
    with pytest.raises(ValueError, match='content pin'):
        rebase_runtime_contract(contract, old, new)


def test_arbitrary_full_contract_schema_is_not_a_source_contract(sample):
    old, new, contract = sample
    contract['schema'] = 'full_episode_contract_v1'
    with pytest.raises(ValueError, match='Unsupported contract fields/schema'):
        rebase_runtime_contract(contract, old, new)


def test_rebased_result_explicitly_has_source_only_scope(sample):
    old, new, contract = sample
    result = rebase_runtime_contract(contract, old, new)
    assert result['schema'] == 'verified_runtime_source_relocation_v1'
    assert result['path_rebase']['full_runtime_contract_generated'] is False
    assert result['path_rebase']['controller_environment_rebuild_required'] is True


@pytest.mark.parametrize('native', ['simulation/isaacsim_runtime.py',
                                  'bench/simulation/isaacsim_runtime.py'])
def test_relocation_preserves_each_frozen_layout_without_executing_sources(tmp_path, native):
    old, new = tmp_path/'old', tmp_path/'new'
    raw = b"raise AssertionError('frozen source must not execute')\n"
    for root in (old, new):
        path = root/native
        path.parent.mkdir(parents=True)
        path.write_bytes(raw)
    inventory = {str(old/native): hashlib.sha256(raw).hexdigest()}
    contract = {'runtime_source_inventory': inventory, 'runtime_code_sha256': canonical_hash(inventory)}
    before = deepcopy(contract)
    result = rebase_runtime_contract(contract, old, new)
    assert result['runtime_source_inventory'] == {str(new/native): inventory[str(old/native)]}
    assert contract == before and (old/native).read_bytes() == (new/native).read_bytes() == raw
    # Identical bytes at a renamed member do not establish a byte-identical
    # snapshot relocation. A source-layout migration requires another contract.
    (new/native).unlink()
    opposite = ('bench/simulation/isaacsim_runtime.py' if native.startswith('simulation/') else
                'simulation/isaacsim_runtime.py')
    replacement = new/opposite
    replacement.parent.mkdir(parents=True, exist_ok=True)
    replacement.write_bytes(raw)
    with pytest.raises(ValueError, match='existing regular file'):
        rebase_runtime_contract(contract, old, new)
