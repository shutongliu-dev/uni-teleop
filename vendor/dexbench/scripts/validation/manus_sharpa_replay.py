"""Compare recorded MANUS postures with the official optimizer, no live devices.

Requires the external SHARPA optimizer root and isolated Python 3.10 environment.
Outputs are marked non-executable; this is a mapping check, not grasp validation.
"""
import argparse
import json
import os
from pathlib import Path
import select
import subprocess
import time

import numpy as np


def main():
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--session',type=Path,required=True)
    cli.add_argument('--root',type=Path,required=True)
    cli.add_argument('--python',type=Path,required=True)
    cli.add_argument('--output',type=Path,required=True)
    args=cli.parse_args()
    chosen={};scores={'open':float('inf'),'fist':-1.,'pinch':float('inf')}
    for line in (args.session/'frames.jsonl').open():
        row=json.loads(line)
        if not row.get('armed') or not row['input']:continue
        packet=row['input']['tracking'].get('manus')
        if not packet:continue
        nodes={n['semantic']:np.array(n['pose']['position']) for n in packet['nodes']}
        curls=[]
        for finger in ('Index','Middle','Ring','Pinky'):
            a=nodes['Finger'+finger+'/Intermediate']-nodes['Finger'+finger+'/Proximal']
            b=nodes['Finger'+finger+'/Distal']-nodes['Finger'+finger+'/Intermediate']
            curls.append(np.arccos(np.clip(a@b/np.linalg.norm(a)/np.linalg.norm(b),-1,1)))
        curl=float(sum(curls))
        gap=float(np.linalg.norm(nodes['FingerThumb/Tip']-nodes['FingerIndex/Tip']))
        for label,score,better in [('open',curl,curl<scores['open']),('fist',curl,curl>scores['fist']),
                                    ('pinch',gap,gap<scores['pinch'])]:
            if better:scores[label]=score;chosen[label]=row
    if set(chosen)!=set(scores):raise ValueError('No complete active MANUS poses')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    env=dict(os.environ,PYTHONPATH=str(Path(__file__).resolve().parents[2]),
             PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
    log=args.output.with_suffix('.stderr').open('w')
    process=subprocess.Popen([str(args.python),'-B','-m','bench.adapters.manus.sharpa_process',
        '--root',str(args.root),'--replay'],stdin=subprocess.PIPE,stdout=subprocess.PIPE,
        stderr=log,text=True,bufsize=1,env=env)

    def receive():
        if not select.select([process.stdout],[],[],30)[0]:raise RuntimeError('Optimizer output timeout')
        line=process.stdout.readline()
        if not line:raise RuntimeError('Optimizer exited')
        return json.loads(line)
    try:
        result={'kind':'offline_recorded_glove_replay','metadata':receive(),
                'selection':'minimum/maximum four-finger PIP bend and minimum thumb-index tip gap',
                'selection_scores':scores,'poses':{},'formal_evaluation':False,'task_success':None}
        for label,row in chosen.items():
            started=time.monotonic()
            process.stdin.write(json.dumps(row['input']['tracking']['manus'])+'\n');process.stdin.flush()
            candidate=receive()
            if candidate['executable']:raise ValueError('Replay must not produce executable targets')
            result['poses'][label]=dict(candidate=candidate,old_target=row['action']['joint_positions_rad'],
                solve_wall_s=time.monotonic()-started,source_sequence=candidate['sequence'])
            print(label,{n:round(v,3) for n,v in candidate['joint_positions_rad'].items() if 'thumb' in n},flush=True)
        args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    finally:
        process.terminate()
        try:process.wait(timeout=5)
        except subprocess.TimeoutExpired:process.kill();process.wait()
        log.close()


if __name__=='__main__':main()
