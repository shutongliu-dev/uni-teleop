"""Check provenance accounting and the reviewed integration inventory.

--refresh updates current-file hashes after an intentional change. Original
upstream hashes in sources.lock.json are never replaced by integration hashes.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def inventory():
    names = subprocess.check_output(['git','ls-files','--cached','--others','--exclude-standard','-z'],cwd=ROOT).decode().split('\0')
    return {name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
            for name in sorted(set(names)) if name and name!='integration.lock.json' and (ROOT/name).is_file()}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--refresh',action='store_true')
    args=parser.parse_args()
    origin=json.loads((ROOT/'sources.lock.json').read_text())
    migration=json.loads((ROOT/'docs/provenance/module-migration.json').read_text())
    count=0
    for source in origin['sources']:
        base='backends/isaac/' if source['name']=='dexbench' else 'backends/mujoco/'
        for name in source['files_sha256']:
            old=base+name
            if old in migration.get('retired',{}): continue
            target=migration['files'].get(old,old)
            if not (ROOT/target).is_file(): raise SystemExit(f'Unaccounted source file: {old} -> {target}')
            count+=1
    lock=ROOT/'integration.lock.json'
    actual=inventory()
    if args.refresh:
        lock.write_text(json.dumps({'format_version':1,'files_sha256':actual},indent=2)+'\n')
    else:
        expected=json.loads(lock.read_text())['files_sha256']
        changed=[p for p in set(expected)|set(actual) if expected.get(p)!=actual.get(p)]
        if changed: raise SystemExit('Integration inventory differs:\n'+'\n'.join(sorted(changed)))
    print(f'Accounted for {count} original files; verified {len(actual)} integrated files. Original hashes retained separately.')


if __name__=='__main__': main()
