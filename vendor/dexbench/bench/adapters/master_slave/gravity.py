"""Leader-only gravity and damping, independent of simulated follower forces.

URDF masses/centres and motor constants are explicit model priors. This module
does not open hardware or establish the physical encoder zero.
"""
import hashlib
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

from bench.control.kinematics import URDFKinematics


class LeaderGravity:
    def __init__(self, urdf):
        self.model=URDFKinematics(urdf)
        if len(self.model.active)!=7 or any(j.kind!='revolute' for j in self.model.active):
            raise ValueError('Leader gravity requires seven revolute joints')
        self.sha256=hashlib.sha256(Path(urdf).read_bytes()).hexdigest()
        self.bodies=[]
        parents={j.child:j for j in self.model.joints}
        for link in ET.parse(urdf).getroot().findall('link'):
            inertial=link.find('inertial')
            if inertial is None:
                raise ValueError(f'Missing leader inertia for {link.get("name")}')
            mass=float(inertial.find('mass').get('value'))
            origin=inertial.find('origin')
            centre=np.fromstring(origin.get('xyz','0 0 0') if origin is not None else '0 0 0',sep=' ')
            if not np.isfinite(mass) or mass<0 or centre.shape!=(3,) or not np.isfinite(centre).all():
                raise ValueError('Invalid leader mass/centre')
            ancestors=[]
            child=link.get('name')
            while child in parents:
                joint=parents[child]
                if joint.name in self.model.names:ancestors.append(self.model.names.index(joint.name))
                child=joint.parent
            self.bodies.append((link.get('name'),mass,centre,ancestors))

    def torque(self, q):
        q=np.asarray(q,dtype=float)
        if q.shape!=(7,) or not np.isfinite(q).all():
            raise ValueError('Require seven finite leader model angles')
        poses=self.model.forward(dict(zip(self.model.names,q)))
        joints=[poses[j.parent]@j.origin for j in self.model.active]
        result=np.zeros(7)
        for name,mass,centre,ancestors in self.bodies:
            world=poses[name][:3,:3]@centre+poses[name][:3,3]
            upward=np.array([0.,0.,mass*9.81])
            for i in ancestors:
                axis=joints[i][:3,:3]@self.model.active[i].axis
                result[i]+=np.dot(axis,np.cross(world-joints[i][:3,3],upward))
        return result

    def currents(self,q,velocity,signs,ma_per_nm,*,gain=1.,damping=.02):
        values=np.array([velocity,signs,ma_per_nm],dtype=float)
        if (values.shape!=(3,7) or not np.isfinite(values).all() or
                not np.all(np.abs(values[1])==1) or np.any(values[2]<=0) or
                not 0<=gain<=1 or not 0<=damping<=1):
            raise ValueError('Invalid leader gravity conversion')
        # velocity is in model coordinates; transpose mapping sends torque to motors.
        return (gain*self.torque(q)-damping*values[0])*values[1]*values[2]
