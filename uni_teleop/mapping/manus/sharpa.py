"""Pure SHARPA optimizer boundary; vendor binaries stay outside the repository/Kit."""
from pathlib import Path
import hashlib
import json
import os
import subprocess
import threading
import time
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation

FINGERS = ("thumb", "index", "middle", "ring", "little")


SEMANTICS = ['Hand/Invalid']
for finger in ('Thumb', 'Index', 'Middle', 'Ring', 'Pinky'):
    joints = ('Metacarpal', 'Proximal', 'Distal', 'Tip') if finger == 'Thumb' else (
        'Metacarpal', 'Proximal', 'Intermediate', 'Distal', 'Tip')
    SEMANTICS.extend('Finger'+finger+'/'+joint for joint in joints)


def optimizer_points(packet):
    """Official client convention: wrist-local xyz, Y(-90deg), conjugated wxyz.

    Use semantic names: SDK 3.2 orders thumb first already, unlike the older
    official client's hard-coded [0,21..24,1..20] node permutation.
    """
    if (packet['protocol'] != 'manus_acquisition_v1' or packet['position_unit'] != 'm' or
            packet['node_space'] != 'world' or packet['axes'] != 'right_handed_z_up_x_forward' or
            not packet['glove_tracking_valid']):
        raise ValueError('Unexpected MANUS coordinate/validity contract')
    nodes = {n['semantic']: n for n in packet['nodes']}
    if len(nodes) != len(packet['nodes']) or set(nodes) != set(SEMANTICS):
        raise ValueError('Require the explicit 25 MANUS hand/finger semantics')
    if not all(n['valid'] for n in nodes.values()):
        raise ValueError('Invalid MANUS node')
    points = np.array([nodes[s]['pose']['position']+nodes[s]['pose']['quaternion_wxyz'] for s in SEMANTICS])
    if points.shape != (25,7) or not np.isfinite(points).all() or np.max(abs(np.linalg.norm(points[:,3:],axis=1)-1)) > .001:
        raise ValueError('Require finite xyz and unit wxyz quaternions')
    rotations = Rotation.from_quat(points[:,[4,5,6,3]])
    root = rotations[0].inv()
    turn = Rotation.from_euler('y', -np.pi/2)
    xyz = turn.apply(root.apply(points[:,:3]-points[0,:3]))
    quat = (turn*root*rotations*turn.inv()).as_quat()[:,[3,0,1,2]]
    return np.column_stack((xyz,quat))


def kinematic_signature(path):
    """Ignore mesh/material paths; bind all fixed/active kinematics and limits."""
    records = []
    joints=ET.parse(path).findall('joint')
    children={}
    for joint in joints:
        children.setdefault(joint.find('parent').attrib['link'],[]).append(joint)
    included=[];pending=['right_hand_C_MC']
    while pending:
        branch=children.get(pending.pop(),[])
        included.extend(branch)
        pending.extend(j.find('child').attrib['link'] for j in branch)
    for joint in included:
        record = dict(name=joint.attrib['name'], type=joint.attrib['type'])
        for tag in ('parent','child','origin','axis','limit'):
            node=joint.find(tag)
            if node is not None:
                record[tag]={k:v if tag in ('parent','child') else [float(x) for x in v.split()]
                             for k,v in node.attrib.items()}
        records.append(record)
    return hashlib.sha256(json.dumps(sorted(records,key=lambda r:r['name']),sort_keys=True).encode()).hexdigest()


def candidate_targets(candidate, raw, model, *, now_ns):
    if candidate is None or not 0 <= now_ns-candidate['capture_time_ns'] <= 150_000_000:
        return None
    if candidate['protocol'] != 'sharpa_optimizer_v1' or candidate['executable'] is not True or any(candidate[k] != raw[k] for k in
            ('stream_id','clock_id','calibration_id','topology_id')):
        raise ValueError('SHARPA candidate identity mismatch')
    if (candidate['device_id'] != raw['device']['device_id'] or
            candidate['sequence'] > raw['sequence'] or
            candidate['capture_time_ns'] > raw['capture_time_ns']):
        raise ValueError('SHARPA candidate source mismatch')
    target = candidate['joint_positions_rad']
    if set(target) != set(model.hand.names) or not np.isfinite(list(target.values())).all():
        raise ValueError('SHARPA candidate joint names/values mismatch')
    target = dict(target)
    for finger in FINGERS:
        chain=model.finger_joints[finger]
        names=[j.name for j in chain]
        values=model.hand_posture_prior.project(finger,[target[n] for n in names],
            [j.lower for j in chain],[j.upper for j in chain],names)
        target.update(zip(names,values.tolist()))
    return target


class SharpaOptimizerSource:
    """Latest-only JSONL input/output, preserving acquisition time through the solve."""
    def __init__(self, glove, python, root, log):
        self.glove,self.python,self.root,self.log=glove,python,root,log
        self.latest=self.metadata=self.error=None
        self.process=None
        self.stderr=None
        self.received=self.rejected=0
        self.last_rejection=None
        self.stop=threading.Event()

    def start(self):
        env=os.environ.copy()
        env['PYTHONPATH']=str(Path(__file__).resolve().parents[3])
        env['OPENBLAS_NUM_THREADS']=env['OMP_NUM_THREADS']='1'
        self.stderr=self.log.open('a')
        self.process=subprocess.Popen([str(self.python),'-B','-m','uni_teleop.mapping.manus.sharpa_process',
            '--root',str(self.root)],stdin=subprocess.PIPE,stdout=subprocess.PIPE,
            stderr=self.stderr,text=True,bufsize=1,env=env)
        self.reader=threading.Thread(target=self._read,daemon=True)
        self.writer=threading.Thread(target=self._feed,daemon=True)
        self.reader.start();self.writer.start()

    def _feed(self):
        previous=None
        try:
            while not self.stop.is_set():
                packet=self.glove.latest
                identity=(packet['stream_id'],packet['sequence']) if packet else None
                if packet and identity != previous:
                    self.process.stdin.write(json.dumps(packet,separators=(',',':'))+'\n')
                    self.process.stdin.flush()
                    previous=identity
                self.stop.wait(.01)
        except (OSError,ValueError) as error:
            if not self.stop.is_set():self.error=str(error)

    def _read(self):
        try:
            for line in self.process.stdout:
                if len(line)>2*1024*1024:raise ValueError('Optimizer JSONL exceeds size limit')
                result=json.loads(line)
                if result['protocol']=='sharpa_optimizer_ready_v1':self.metadata=result
                elif result['protocol']=='sharpa_optimizer_rejected_v1':
                    self.rejected+=1
                    self.last_rejection=result['reason']
                elif result['protocol']=='sharpa_optimizer_v1':
                    self.latest=result
                    self.received+=1
        except (ValueError,KeyError) as error:
            self.error=str(error)
        finally:
            if not self.stop.is_set():self.error=self.error or 'SHARPA optimizer exited'

    def health(self):
        return dict(running=self.process is not None and self.process.poll() is None,
            ready=self.metadata is not None,received=self.received,error=self.error,
            rejected=self.rejected,last_rejection=self.last_rejection,
            latest_age_s=(time.monotonic_ns()-self.latest['capture_time_ns'])/1e9 if self.latest else None)

    def close(self):
        self.stop.set()
        if self.process is not None:
            if self.process.poll() is None:
                self.process.terminate()
                try:self.process.wait(timeout=5)
                except subprocess.TimeoutExpired:self.process.kill();self.process.wait()
            self.reader.join(timeout=1);self.writer.join(timeout=1)
        if self.stderr:self.stderr.close()
