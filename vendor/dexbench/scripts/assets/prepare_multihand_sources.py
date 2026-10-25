"""Reproduce the pinned official hand downloads and isolated Shadow xacro expansion.

Run with system Python after installing xacro 2.1.1 into the named shared tools
folder. Does not install packages into any simulator or project interpreter.
"""
from pathlib import Path
import argparse
import hashlib
import json
import shutil
import sys
import urllib.request
import zipfile

from bench.paths import DATA_ROOT, load_json

SOURCES = [
 ('leap-hand/LEAP_Hand_Sim','150bc3d4b61fd6619193ba5a8ef209f3609ced89','leap/v1_150bc3d'),
 ('Wonikrobotics-git/allegro_hand_ros2_v5','80bd4a88d2c59b8ad0242ec3730302bde61c84fb','allegro/v5_80bd4a8'),
 ('shadow-robot/sr_common','172c3d9c3715ee75eb10dfe4ee57e26a342b4529','shadow/sr_common_172c3d9'),
]


def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()


def prepare(assets):
    for repo,commit,relative in SOURCES:
        root=assets/'robots'/relative;root.mkdir(parents=True,exist_ok=True)
        archive=root/'source.zip';url=f'https://codeload.github.com/{repo}/zip/{commit}'
        if not archive.exists():
            temporary=archive.with_suffix('.partial');urllib.request.urlretrieve(url,temporary);temporary.rename(archive)
        with zipfile.ZipFile(archive) as z:
            if z.testzip(): raise ValueError('Archive CRC failure: '+str(archive))
            if not (root/'upstream').exists():
                for info in z.infolist():
                    if Path(info.filename).is_absolute() or '..' in Path(info.filename).parts: raise ValueError('Unsafe archive path')
                z.extractall(root/'unpack');(root/'unpack'/z.namelist()[0].split('/')[0]).rename(root/'upstream');(root/'unpack').rmdir()
        hashes={str(p.relative_to(root/'upstream')):sha(p) for p in sorted((root/'upstream').rglob('*')) if p.is_file()}
        record={'repository':'https://github.com/'+repo,'commit':commit,'download_url':url,'retrieved_utc_date':'2026-10-11','archive_sha256':sha(archive),'file_sha256':hashes}
        manifest=root/'SOURCE.json'
        if manifest.exists():
            old=load_json(manifest)
            if old['archive_sha256']!=record['archive_sha256'] or old['file_sha256']!=hashes: raise ValueError('Pinned source changed: '+str(root))
        else: manifest.write_text(json.dumps(record,indent=2)+'\n')
    # Only the SRDF and its source license are needed; preserve their commit and hashes.
    srdf_commit='01aca16f5a8d0ff468ff76eabf7c30af2df6c539'
    srdf_root=assets/'robots/shadow/sr_interface_01aca16'
    expected={
      'sr_moveit_hand_config/config/shadowhand.srdf.xacro':'0c9737ea43489038db71f434499ebe0053215012960ceb1ef2b8808459f060ed',
      'sr_moveit_hand_config/config/shadowhands_prefix.srdf.xacro':'3393a986ce107e5999cc0e3c1aa89e279fbee7f8cad22272e4c4eab4350921b6',
      'LICENSE':'6d0c3fdad8f1df2ba711da33c395854a845b98128d83d3f57fac9e70c720d240'}
    records={}
    for relative,expected_hash in expected.items():
        path=srdf_root/relative;path.parent.mkdir(parents=True,exist_ok=True)
        url=f'https://raw.githubusercontent.com/shadow-robot/sr_interface/{srdf_commit}/{relative}'
        if not path.exists():
            temporary=path.with_suffix(path.suffix+'.partial');urllib.request.urlretrieve(url,temporary)
            if sha(temporary)!=expected_hash: raise ValueError('SRDF download hash mismatch: '+relative)
            temporary.rename(path)
        if sha(path)!=expected_hash: raise ValueError('Pinned SRDF file changed: '+relative)
        records[relative]={'url':url,'sha256':expected_hash}
    manifest=srdf_root/'SOURCE.json'
    if not manifest.exists():
        manifest.write_text(json.dumps({'repository':'https://github.com/shadow-robot/sr_interface','commit':srdf_commit,'files':records},indent=2)+'\n')
    elif load_json(manifest)['files']!=records: raise ValueError('SRDF source manifest changed')
    sys.path.insert(0,str(assets/'tools/xacro/2.1.1/site_packages'))
    import xacro
    root=assets/'robots/shadow/sr_common_172c3d9';src=root/'upstream/sr_description';out=root/'derived/E3M5_right_pst';build=out/'xacro_inputs';build.mkdir(parents=True,exist_ok=True)
    for p in src.rglob('*.xacro'):
        q=build/p.relative_to(src);q.parent.mkdir(parents=True,exist_ok=True);q.write_text(p.read_text().replace('$(find sr_description)',str(build)))
    for p in src.rglob('*.yaml'):
        q=build/p.relative_to(src);q.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,q)
    mappings={'hand_type':'hand_e','hand_version':'E3M5','side':'right','tip_sensors':'pst'}
    text=xacro.process_file(str(build/'robots/sr_hand.urdf.xacro'),mappings=mappings).toprettyxml(indent='  ')
    urdf=out/'shadow_right.urdf'
    if urdf.exists() and urdf.read_text()!=text: raise ValueError('Expanded Shadow URDF changed')
    if not urdf.exists(): urdf.write_text(text)
    print(json.dumps({'verified_repositories':len(SOURCES),'shadow_urdf':str(urdf),'sha256':sha(urdf)}))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--assets-root',type=Path,default=Path(f'{DATA_ROOT}/assets'));a=p.parse_args();prepare(a.assets_root)
