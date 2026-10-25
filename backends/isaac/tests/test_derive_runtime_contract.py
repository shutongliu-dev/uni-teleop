import importlib.util
import json
from pathlib import Path
import sys

import pytest

from bench.paths import load_json


SOURCE = Path(__file__).resolve().parents[1] / 'scripts/maintenance/derive_runtime_contract.py'
spec = importlib.util.spec_from_file_location('derive_runtime_contract_under_test', SOURCE)
tool = importlib.util.module_from_spec(spec); spec.loader.exec_module(tool)
src = tool.sources


def save(path, value):
    path.write_text(json.dumps(value, indent=2)+'\n')


@pytest.fixture
def setup(tmp_path):
    original = tmp_path/'original'
    for tree in src.freeze.SOURCE_TREES:
        (original/tree).mkdir(parents=True)
    names = ['bench/runtime/service.py','bench/control/fingertip.py','bench/simulation/native.py','scripts/isaac_python.sh']
    for name in names:
        path=original/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text('# original\n')
    (original/'scripts/isaac_python.sh').chmod(0o755)
    parent=tmp_path/'parent';src.freeze.freeze(original,parent,('scripts/isaac_python.sh',))
    candidate=tmp_path/'candidate.py';candidate.write_text('# new service\n')
    overlay=tmp_path/'overlay.json'
    data={'schema':'complete_benchmark_source_overlay_v1','parent_manifest_sha256':src._sha_file(parent/src.MANIFEST_NAME),
          'entries':[{'target':tool.SUPPORTED_CHANGE,'expected_parent_sha256':src._sha_file(parent/tool.SUPPORTED_CHANGE),
                      'source_path':str(candidate),'source_sha256':src._sha_file(candidate)}]}
    save(overlay,data);new=tmp_path/'new';src.derive(parent,new,overlay)
    inventory={str(parent/name) if name.startswith('bench/simulation/') else name:src._sha_file(parent/name) for name in names}
    contract={'schema':tool.SCHEMA,'runtime_source_inventory':inventory,
              'runtime_code_sha256':src._canonical_hash(inventory),'sensor_config_sha256':'a'*64}
    old_file=tmp_path/'old_contract.json';save(old_file,contract)
    request={'schema':'explicit_service_runtime_change_request_v1','old_contract_sha256':src._sha_file(old_file),
        'derived_manifest_sha256':src._sha_file(new/src.MANIFEST_NAME),'allowed_source_changes':[
            {'path':tool.SUPPORTED_CHANGE,'old_sha256':src._sha_file(parent/tool.SUPPORTED_CHANGE),
             'new_sha256':src._sha_file(new/tool.SUPPORTED_CHANGE)}]}
    request_file=tmp_path/'request.json';save(request_file,request)
    return {'parent':parent,'new':new,'contract':contract,'old_file':old_file,'request':request,
            'request_file':request_file,'overlay':overlay,'overlay_data':data,'candidate':candidate,'tmp':tmp_path}


def run(s):
    return tool.derive_runtime_contract(s['old_file'],s['new'],s['request_file'])


def repin_contract(s):
    save(s['old_file'],s['contract']);s['request']['old_contract_sha256']=src._sha_file(s['old_file']);save(s['request_file'],s['request'])


def test_strict_service_replacement_and_native_key_relocation(setup):
    result,proof=run(setup)
    assert result['runtime_code_sha256']!=setup['contract']['runtime_code_sha256']
    assert result['sensor_config_sha256']=='a'*64
    assert result['runtime_source_inventory'][tool.SUPPORTED_CHANGE]==src._sha_file(setup['candidate'])
    assert str(setup['new']/'bench/simulation/native.py') in result['runtime_source_inventory']
    assert proof['allowed_source_changes']==setup['request']['allowed_source_changes']
    assert len(proof['sources'])==4 and proof['native_launches']==0
    assert src._sha_bytes(proof['historical_parent_contract_utf8'].encode())==proof['old_contract_file_sha256']


@pytest.mark.parametrize('fault',['old_file_hash','old_runtime_hash','old_runtime_digest','old_bytes','new_bytes',
    'manifest_hash','wrong_allowed_old','wrong_allowed_new','empty_allowlist','duplicate_allowlist','escaped_key',
    'external_key','missing_service','current_implementation_field','current_system0_field','current_environment_field'])
def test_rejects_inconsistent_or_unsupported_inputs(setup,fault):
    s=setup;c=s['contract'];q=s['request']
    if fault=='old_file_hash':s['old_file'].write_text(s['old_file'].read_text()+' ')
    elif fault=='old_runtime_hash':c['runtime_code_sha256']='0'*64;repin_contract(s)
    elif fault=='old_runtime_digest':
        c['runtime_source_inventory'][tool.SUPPORTED_CHANGE]='0'*64;c['runtime_code_sha256']=src._canonical_hash(c['runtime_source_inventory']);repin_contract(s)
    elif fault=='old_bytes':(s['parent']/tool.SUPPORTED_CHANGE).write_text('changed')
    elif fault=='new_bytes':(s['new']/tool.SUPPORTED_CHANGE).write_text('changed')
    elif fault=='manifest_hash':q['derived_manifest_sha256']='0'*64;save(s['request_file'],q)
    elif fault.startswith('wrong_allowed_'):
        q['allowed_source_changes'][0]['old_sha256' if fault.endswith('old') else 'new_sha256']='0'*64;save(s['request_file'],q)
    elif fault=='empty_allowlist':q['allowed_source_changes']=[];save(s['request_file'],q)
    elif fault=='duplicate_allowlist':q['allowed_source_changes']*=2;save(s['request_file'],q)
    elif fault in {'escaped_key','external_key'}:
        key='../escape.py' if fault=='escaped_key' else str(s['candidate'])
        c['runtime_source_inventory'][key]='0'*64;c['runtime_code_sha256']=src._canonical_hash(c['runtime_source_inventory']);repin_contract(s)
    elif fault=='missing_service':
        c['runtime_source_inventory'].pop(tool.SUPPORTED_CHANGE);c['runtime_code_sha256']=src._canonical_hash(c['runtime_source_inventory']);repin_contract(s)
    else:
        c[{'current_implementation_field':'implementation_sha256','current_system0_field':'system0_descriptor','current_environment_field':'execution_environment_sha256'}[fault]]='0'*64;repin_contract(s)
    with pytest.raises(ValueError):run(s)


@pytest.mark.parametrize('kind',['new_root','old_contract_file','request_file','runtime_key'])
def test_path_aliases_rejected(setup,kind):
    s=setup;alias=s['tmp']/'alias'
    if kind=='new_root':alias.symlink_to(s['new'],target_is_directory=True);s['new']=alias
    elif kind in {'old_contract_file','request_file'}:
        key='old_file' if kind=='old_contract_file' else 'request_file';alias.symlink_to(s[key]);s[key]=alias
    else:
        alias.symlink_to(s['parent'],target_is_directory=True)
        old=str(s['parent']/'bench/simulation/native.py');new=str(alias/'bench/simulation/native.py')
        s['contract']['runtime_source_inventory'][new]=s['contract']['runtime_source_inventory'].pop(old)
        s['contract']['runtime_code_sha256']=src._canonical_hash(s['contract']['runtime_source_inventory']);repin_contract(s)
    with pytest.raises(ValueError):run(s)


@pytest.mark.parametrize('declare_other',[False,True])
def test_controller_change_cannot_hide_or_carry_stale_descriptor(setup,declare_other):
    s=setup;other=s['tmp']/'controller.py';other.write_text('# changed controller\n')
    name='bench/control/fingertip.py'
    s['overlay_data']['entries'].append({'target':name,'expected_parent_sha256':src._sha_file(s['parent']/name),
        'source_path':str(other),'source_sha256':src._sha_file(other)})
    save(s['overlay'],s['overlay_data']);s['new']=s['tmp']/'new_with_controller';src.derive(s['parent'],s['new'],s['overlay'])
    s['request']['derived_manifest_sha256']=src._sha_file(s['new']/src.MANIFEST_NAME)
    if declare_other:s['request']['allowed_source_changes'].append({'path':name,'old_sha256':src._sha_file(s['parent']/name),'new_sha256':src._sha_file(other)})
    save(s['request_file'],s['request'])
    with pytest.raises(ValueError,match='complete snapshot|Only service-only'):run(s)


def test_legacy_history_is_retained_only_as_hashed_parent_evidence(setup):
    s=setup;c=s['contract'];c['schema']=tool.LEGACY_SCHEMA
    c.update(source_transition={'historical_service':'old'},integration_expected_sources={tool.SUPPORTED_CHANGE:src._sha_file(s['parent']/tool.SUPPORTED_CHANGE)},
             path_rebase={'historical_root':str(s['parent'])})
    repin_contract(s);result,proof=run(s)
    for key in ('source_transition','integration_expected_sources','path_rebase'):
        assert key not in result and key in proof['historical_only_parent_fields']
    assert json.loads(proof['historical_parent_contract_utf8'])==c
    assert result['schema']==tool.SCHEMA


@pytest.mark.parametrize('field',['api_key','auth','config','implementation_sha256'])
def test_legacy_unknown_top_level_fields_rejected(setup,field):
    setup['contract']['schema']=tool.LEGACY_SCHEMA
    setup['contract'][field]='synthetic-never-archive'
    repin_contract(setup)
    with pytest.raises(ValueError,match='Unsupported contract fields'):run(setup)


@pytest.mark.parametrize('which',['derived_output','parent_proof','original_output'])
def test_cli_must_not_write_into_verified_or_original_roots(setup,monkeypatch,which):
    s=setup;output=s['tmp']/'contract_new.json';proof=s['tmp']/'proof_new.json'
    if which=='derived_output':output=s['new']/'bench/new.py'
    elif which=='parent_proof':proof=s['parent']/'bench/new.py'
    else:output=s['tmp']/'original/bench/new.py'
    monkeypatch.setattr(sys,'argv',['derive_runtime_contract','--contract',str(s['old_file']),
        '--derived-project',str(s['new']),'--request',str(s['request_file']),
        '--output',str(output),'--proof',str(proof)])
    with pytest.raises(ValueError,match='outside parent, derived and original'):tool.main()
    assert not output.exists() and not proof.exists()
    assert src.verify_derivation(s['new'])['derivation_verified']


def test_cli_writes_distinct_new_artifacts_and_pins_request(setup,monkeypatch):
    s=setup;output=s['tmp']/'contract_new.json';proof=s['tmp']/'proof_new.json'
    monkeypatch.setattr(sys,'argv',['derive_runtime_contract','--contract',str(s['old_file']),
        '--derived-project',str(s['new']),'--request',str(s['request_file']),
        '--output',str(output),'--proof',str(proof)])
    tool.main();p=load_json(proof)
    assert src._sha_bytes(p['request_utf8'].encode())==p['request_sha256']
    assert json.loads(p['request_utf8'])==s['request']
    with pytest.raises(ValueError,match='distinct fresh'):tool.main()


def historical_fixture(tmp_path, version, *, service=None):
    """Create synthetic frozen bytes in their original layout, without imports."""
    legacy = version is None or version <= 4
    roots = (('interface', 'bench', 'simulation') if legacy else
             ('interface', 'bench', 'simulation', 'agent', 'scenes') if version == 5 else
             ('bench', 'agent', 'scenes'))
    service = service or ('interface/agent/service.py' if legacy else 'bench/runtime/service.py')
    native = 'simulation/isaacsim_runtime.py' if version != 6 else 'bench/simulation/isaacsim_runtime.py'
    parent, original = tmp_path/'parent', tmp_path/'original'
    original.mkdir()
    for tree in roots:
        (parent/tree).mkdir(parents=True)
    names = [service, native, 'scripts/maintenance/freeze_project_sources.py', 'scripts/isaac_python.sh']
    for name in names:
        path = parent/name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("raise AssertionError('historical snapshot source must never execute')\n")
    (parent/'scripts/isaac_python.sh').chmod(0o755)
    files = {name: src._sha_file(parent/name) for name in names}
    declared_version = None if version is None else f'complete_benchmark_source_verifier_v{version}'
    manifest = {'schema': 'complete_benchmark_source_snapshot_v1', 'source_utility_version': declared_version,
        'source_root': str(original), 'snapshot_root': str(parent), 'files_sha256': files,
        'files_mode_octal': src.freeze.file_modes(parent, files), 'file_mode_policy': src.freeze.MODE_POLICY,
        'suite_source_pins': src.freeze.suite_inventory(files, source_trees=roots,
            include_runtime_guard=declared_version in src.freeze.RUNTIME_GUARD_VERSIONS),
        'explicit_extra_resources': ['scripts/isaac_python.sh'], 'shared_asset_references': {}}
    if version is not None and version >= 5:
        manifest['source_trees'] = list(roots)
    if version is None:
        manifest.pop('source_utility_version')
    save(parent/src.MANIFEST_NAME, manifest)
    candidate = tmp_path/'candidate.py'
    candidate.write_text("raise AssertionError('candidate source must never execute either')\n")
    overlay = tmp_path/'overlay.json'
    save(overlay, {'schema': 'complete_benchmark_source_overlay_v1',
        'parent_manifest_sha256': src._sha_file(parent/src.MANIFEST_NAME), 'entries': [
            {'target': service, 'expected_parent_sha256': files[service],
             'source_path': str(candidate), 'source_sha256': src._sha_file(candidate)}]})
    new = tmp_path/'new'
    src.derive(parent, new, overlay)
    inventory = {service: files[service], str(parent/native): files[native]}
    old_file = tmp_path/'contract.json'
    save(old_file, {'schema': tool.SCHEMA, 'runtime_source_inventory': inventory,
                   'runtime_code_sha256': src._canonical_hash(inventory)})
    request_file = tmp_path/'request.json'
    save(request_file, {'schema': 'explicit_service_runtime_change_request_v1',
        'old_contract_sha256': src._sha_file(old_file), 'derived_manifest_sha256': src._sha_file(new/src.MANIFEST_NAME),
        'allowed_source_changes': [{'path': service, 'old_sha256': files[service],
                                   'new_sha256': src._sha_file(candidate)}]})
    return {'parent': parent, 'new': new, 'old_file': old_file, 'request_file': request_file,
            'service': service, 'native': native, 'version': declared_version}


@pytest.mark.parametrize('version', [None, 1, 2, 3, 4, 5, 6])
def test_parent_layout_selects_service_and_preserves_native_member_spelling(tmp_path, version):
    s = historical_fixture(tmp_path, version)
    before = {path: path.read_bytes() for path in s['parent'].rglob('*') if path.is_file()}
    old_contract = s['old_file'].read_bytes()
    result, proof = run(s)
    assert set(result['runtime_source_inventory']) == {s['service'], str(s['new']/s['native'])}
    assert result['runtime_source_inventory'][s['service']] == src._sha_file(s['new']/s['service'])
    assert result['runtime_source_inventory'][str(s['new']/s['native'])] == src._sha_file(s['parent']/s['native'])
    assert proof['parent_source_utility_version'] == s['version']
    assert proof['supported_service_source'] == s['service']
    assert proof['native_launches'] == proof['model_calls'] == 0
    assert s['old_file'].read_bytes() == old_contract
    assert {path: path.read_bytes() for path in s['parent'].rglob('*') if path.is_file()} == before


@pytest.mark.parametrize('version,wrong_service', [
    (4, 'bench/runtime/service.py'), (5, 'interface/agent/service.py')])
def test_service_allowlist_cannot_be_selected_by_similar_files_existing(tmp_path, version, wrong_service):
    # Both names are legal files inside these layouts; only the declared parent
    # generation determines which service implementation this tool may replace.
    s = historical_fixture(tmp_path, version, service=wrong_service)
    with pytest.raises(ValueError, match='Only service-only replacement'):
        run(s)
