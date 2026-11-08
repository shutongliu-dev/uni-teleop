import json
from pathlib import Path
import pytest

from scenes.workrooms.robot_revision import assets


def fixture(tmp_path,monkeypatch,filename='runtime_inputs_manifest.json',version=assets.VERSION):
    monkeypatch.setattr(assets,'ASSET_ROOT',tmp_path)
    root=tmp_path/'robots/assemblies/fr3_sharpa'/version;root.mkdir(parents=True)
    def file(name,data):
        p=root/name;p.write_text(json.dumps(data));return str(p),assets.file_hash(p)
    urdf,uh=file('physical.urdf',{'test_only':True})
    source,sh=file('SOURCE.json',{'urdf_path':urdf,'urdf_sha256':uh,'mesh_inputs':{},
        'sources':{'/unmounted/historical/report.json':'0'*64}})
    usd,dh=file('robot.usd',{'test_only':True})
    imported,ih=file('USD_IMPORT.json',{'usd_path':usd,'usd_sha256':dh,'derived_files':{usd:dh}})
    value={'schema':assets.MANIFEST_SCHEMAS[filename],'assembly_version':version,
        'hands':{s:{'urdf_path':urdf,'urdf_sha256':uh,'source_path':source,'source_sha256':sh} for s in ('left','right')},
        'assemblies':{s:{'urdf_path':urdf,'urdf_sha256':uh,'source_path':source,'source_sha256':sh,
                        'usd_import_path':imported,'usd_import_sha256':ih} for s in assets.ASSEMBLIES},
        'files_sha256':{urdf:uh,source:sh,usd:dh,imported:ih}}
    if filename=='runtime_inputs_manifest.json':
        value['provenance_manifest']={'path':'/unmounted/archive/revision_manifest.json','sha256':'1'*64}
    path=root/filename;path.write_text(json.dumps(value));return path,value


@pytest.mark.parametrize('filename',list(assets.MANIFEST_SCHEMAS))
@pytest.mark.parametrize('version',sorted(assets.SUPPORTED_VERSIONS))
def test_named_manifest_versions_verify_actual_files_without_reading_old_archive(tmp_path,monkeypatch,filename,version):
    path,value=fixture(tmp_path,monkeypatch,filename,version)
    assert assets.read_asset_manifest(path,expected_sha256=assets.file_hash(path))==value
    Path(value['assemblies']['single_right']['source_path']).write_text('changed metadata')
    with pytest.raises(ValueError,match='immutable digest'):
        assets.read_asset_manifest(path)


@pytest.mark.parametrize('fault',['unknown_name','wrong_schema','bad_provenance','missing_derived','changed_import','bad_digest','historical_report'])
def test_runtime_input_manifest_remains_strict(tmp_path,monkeypatch,fault):
    path,value=fixture(tmp_path,monkeypatch)
    if fault=='unknown_name':path=path.with_name('any_manifest.json')
    elif fault=='wrong_schema':value['schema']=assets.SCHEMA
    elif fault=='bad_provenance':value['provenance_manifest']['sha256']='bad'
    elif fault=='missing_derived':del value['files_sha256'][value['assemblies']['single_right']['usd_import_path'].replace('USD_IMPORT.json','robot.usd')]
    elif fault=='changed_import':Path(value['assemblies']['single_right']['usd_import_path']).write_text('{}')
    elif fault=='historical_report':value['files_sha256']['/unmounted/historical/report.json']='0'*64
    elif fault=='bad_digest':value['files_sha256'][next(iter(value['files_sha256']))]='bad'
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError):assets.read_asset_manifest(path,assets=False)


@pytest.mark.parametrize('fault',['unknown','cross_version'])
def test_manifest_directory_and_declared_version_must_match(tmp_path,monkeypatch,fault):
    path,value=fixture(tmp_path,monkeypatch,version='isaacsim_6_1_l_mount_v2')
    value['assembly_version']='isaacsim_6_1_l_mount_v99' if fault=='unknown' else assets.VERSION
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError):assets.read_asset_manifest(path)


def test_unregistered_version_directory_is_not_admitted_by_matching_its_manifest(tmp_path,monkeypatch):
    path,value=fixture(tmp_path,monkeypatch,version='isaacsim_6_1_l_mount_v99')
    with pytest.raises(ValueError,match='versioned shared asset manifest'):
        assets.read_asset_manifest(path)


@pytest.mark.parametrize('version',sorted(assets.SUPPORTED_VERSIONS))
@pytest.mark.parametrize('kind',['single','dual','instances'])
def test_binding_and_catalog_use_exact_manifest_version(version,kind):
    from copy import deepcopy
    from scenes.workrooms.robot_revision.catalog import _requirements
    mount={'translation_m':[.01,.02,.03],'quaternion_wxyz':[1.,0.,0.,0.],'rpy_rad':[0.,0.,0.]}
    asset={'assembly_version':version,'mounts_by_side':{'left':deepcopy(mount),'right':deepcopy(mount)}}
    if kind=='instances':
        robot={'hand_id':'fr3_sharpa_instances','assembly_version':'namespaced_v1',
            'instances':{s:{'hand_id':'sharpa_w01_right','assembly_version':version} for s in ('front','rear')}}
    else:
        robot={'hand_id':'sharpa_w01_right' if kind=='single' else 'dual_fr3_sharpa_w01',
            'assembly_version':version,'mount_position_m':mount['translation_m'],
            'mount_quaternion_wxyz':mount['quaternion_wxyz'],
            'mounts_by_side':{s:deepcopy(mount) for s in (('right',) if kind=='single' else ('left','right'))}}
    assets.check_robot_binding({'robot':robot},asset)
    assert _requirements({},robot)['assembly_version']==version
    bad=deepcopy(asset);bad['assembly_version']=next(v for v in assets.SUPPORTED_VERSIONS if v!=version)
    with pytest.raises(ValueError):assets.check_robot_binding({'robot':robot},bad)
    bad['assembly_version']='isaacsim_6_1_l_mount_unknown'
    with pytest.raises(ValueError):assets.check_robot_binding({'robot':robot},bad)
