#!/usr/bin/env python3
"""Freeze scene source and render a reproducible collection on one UUID-selected GPU."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

from bench.paths import OUTPUT_ROOT

PROJECT=Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT))

from scripts.maintenance.freeze_project_sources import DEFAULT_EXTRAS, freeze, verify
from bench.simulation.assets.workrooms import FAMILIES, SOURCE_RESOURCES, shared_asset_links

EXTRAS = tuple(dict.fromkeys(DEFAULT_EXTRAS + SOURCE_RESOURCES))


def freeze_sources(output):
    """Freeze the complete bench source and seal it without losing execute bits."""
    source=output/'source'
    manifest=freeze(PROJECT, source, EXTRAS,
                    asset_links=shared_asset_links(PROJECT))
    copied={}
    for relative,digest in manifest['files_sha256'].items():
        target=source/relative
        target.chmod(target.stat().st_mode & ~0o222)
        copied[relative]={'sha256':digest,'mode':oct(target.stat().st_mode & 0o777)}
    (source/'source_snapshot_manifest.json').chmod(0o444)
    verify(source)
    # Preserve the collection's existing compact source-manifest format as well
    # as the complete bench snapshot contract inside source/.
    (output/'source_manifest.json').write_text(json.dumps(copied,indent=2)+'\n')
    return source


def probe_launcher(source, output, env):
    """Check the final direct-exec entrypoint and import root without starting Isaac."""
    code=("import json,sys; from pathlib import Path; import bench.simulation.assets.workrooms as w; "
          "root=Path.cwd(); module=Path(w.__file__).resolve(); "
          "assert module == root/'bench/simulation/assets/workrooms/__init__.py'; "
          "print(json.dumps({'python':sys.executable,'cwd':str(root),'workrooms_module':str(module)}))")
    command=[str(source/'scripts/isaac_python.sh'),'-c',code]
    result=subprocess.run(command,cwd=source,env=env,capture_output=True,text=True)
    record={'command':command,'exit_code':result.returncode,
            'stdout':result.stdout,'stderr':result.stderr,'simulation_started':False}
    (output/'launcher_probe.json').write_text(json.dumps(record,indent=2)+'\n')
    if result.returncode:
        raise RuntimeError('Sealed launcher/import probe failed; see launcher_probe.json')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--families',nargs='+',choices=FAMILIES,default=list(FAMILIES))
    p.add_argument('--gpu',required=True,help='Physical GPU UUID, never a volatile index')
    p.add_argument('--output',type=Path,default=None)
    p.add_argument('--samples',type=int,default=256)
    p.add_argument('--width',type=int,default=1920)
    p.add_argument('--height',type=int,default=1440)
    args=p.parse_args()
    if not args.gpu.startswith('GPU-') or ',' in args.gpu:p.error('One physical GPU UUID required')
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    out=(args.output or Path(OUTPUT_ROOT)/'workroom_collections'/stamp).resolve()
    if out.exists():raise FileExistsError(out)
    source=freeze_sources(out)
    (out/'logs').mkdir();results=[]
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=args.gpu,ISAAC_ALLOW_UNVALIDATED_DRIVER='1',PYTHONDONTWRITEBYTECODE='1',ISAAC_BENCH_SINGLE_CUDA_DEVICE='1')
    probe_launcher(source,out,env)
    for family in args.families:
        verify(source)
        target=out/'renders'/family
        cmd=[str(source/'scripts/isaac_python.sh'),'-m','scripts.media.render_workroom','--family',family,'--output',str(target),'--samples',str(args.samples),'--width',str(args.width),'--height',str(args.height)]
        log=out/'logs'/(family+'.log')
        (out/'logs'/(family+'.launch.json')).write_text(json.dumps({'command':cmd,'source':str(source),'gpu_uuid':args.gpu,'driver_override':True},indent=2)+'\n')
        print(json.dumps({'event':'started','family':family,'output':str(target),'log':str(log)}),flush=True)
        with log.open('w') as handle:r=subprocess.run(cmd,cwd=source,env=env,stdout=handle,stderr=subprocess.STDOUT)
        verify(source)
        errors=[x for x in log.read_text(errors='replace').splitlines() if '[Error]' in x or 'Traceback' in x]
        record={'family':family,'exit_code':r.returncode,'output':str(target),'log':str(log),'native_errors':errors}
        results.append(record);(out/'logs'/(family+'.exit.json')).write_text(json.dumps(record,indent=2)+'\n')
        (out/'results.json').write_text(json.dumps(results,indent=2)+'\n')
        print(json.dumps({'event':'finished','family':family,'exit_code':r.returncode,'errors':len(errors)}),flush=True)
    if any(x['exit_code'] or x['native_errors'] for x in results):sys.exit(1)

if __name__=='__main__':main()
