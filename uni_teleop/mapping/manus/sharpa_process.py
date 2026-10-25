"""Isolated Python 3.10 official SHARPA optimizer; no robot SDK or device access."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import sys
import threading
import time

from uni_teleop.mapping.manus.sharpa import optimizer_points, kinematic_signature


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--replay',action='store_true',help='Offline old recordings; outputs explicitly non-executable')
    args=parser.parse_args()
    output=os.fdopen(os.dup(sys.stdout.fileno()),'w',buffering=1)
    os.dup2(sys.stderr.fileno(),sys.stdout.fileno())
    root=args.root.resolve()
    os.chdir(root)
    sys.path.insert(0,str(root/'include'))
    from hand_retargeting_optimizer import init_hand_model, HandRetargetingOptimizer
    models=init_hand_model('WAVE')
    names=list(models['right'].joint_names)
    urdf=root/'urdf/right_sharpa_wave/right_sharpa_wave.urdf'
    metadata=dict(protocol='sharpa_optimizer_ready_v1',side='right',joint_names=names,
        engine='official.HandRetargetingOptimizer',
        kinematic_signature=kinematic_signature(urdf),urdf_sha256=hashlib.sha256(urdf.read_bytes()).hexdigest(),
        binaries_sha256={name:hashlib.sha256((root/'include'/name).read_bytes()).hexdigest()
                        for name in ('hand_retargeting_optimizer.so','hand_kinematic_casadi.so')})
    stopped=threading.Event()
    for signum in (signal.SIGINT,signal.SIGTERM):signal.signal(signum,lambda *_:stopped.set())
    state={'latest':None,'error':None,'eof':False}

    def receive():
        try:
            while not stopped.is_set():
                line=sys.stdin.readline(2*1024*1024+1)
                if not line:state['eof']=True;break
                if len(line)>2*1024*1024:raise ValueError('Input JSONL exceeds size limit')
                state['latest']=json.loads(line)
        except (ValueError,OSError) as error:state['error']=str(error)
    threading.Thread(target=receive,daemon=True).start()
    optimizer=HandRetargetingOptimizer(models['right'],'right')
    previous=None;previous_points=None;last_identity=None
    output.write(json.dumps(metadata)+'\n')
    while not stopped.is_set():
        if state['error']:raise ValueError(state['error'])
        packet=state['latest']
        identity=(packet['stream_id'],packet['sequence']) if packet else None
        if packet is not None and identity!=last_identity:
            last_identity=identity
            if packet['device']['hand']!='right':raise ValueError('Require right MANUS glove')
            if args.replay or 0<=time.monotonic_ns()-packet['capture_time_ns']<=150_000_000:
                try:
                    points=optimizer_points(packet)
                except (ValueError,KeyError,TypeError) as error:
                    output.write(json.dumps(dict(protocol='sharpa_optimizer_rejected_v1',
                        sequence=packet['sequence'],reason=str(error)))+'\n')
                    time.sleep(.005)
                    continue
                q=optimizer.optimize_with_casadi(initial_q=previous,raw_keypoints=points,
                                                  keypoints_last=previous_points)
                target=dict(zip(names,map(float,q)))
                if len(target)!=22:raise ValueError('Require 22 optimizer joint targets')
                previous,previous_points=q,points
                if args.replay or 0<=time.monotonic_ns()-packet['capture_time_ns']<=150_000_000:
                    origin={k:packet[k] for k in ('stream_id','sequence','clock_id',
                        'capture_time_ns','calibration_id','topology_id')}
                    origin['device_id']=packet['device']['device_id']
                    output.write(json.dumps(dict(origin,protocol='sharpa_optimizer_v1',
                        executable=not args.replay,joint_positions_rad=target),allow_nan=False)+'\n')
        latest=state['latest']
        latest_identity=(latest['stream_id'],latest['sequence']) if latest else None
        if state['eof'] and last_identity==latest_identity:break
        time.sleep(.005)


if __name__=='__main__':
    main()
