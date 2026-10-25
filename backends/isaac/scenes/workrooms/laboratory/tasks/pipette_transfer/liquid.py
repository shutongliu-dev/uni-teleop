"""Explicit finite-volume piston pump, driven only by native mechanical state.

This is a reduced physical component, NOT a read-only referee or a PBD fluid.
Its only actuation input is measured piston displacement. It owns water volume,
advects expelled packets ballistically, and deposits them only through actual
vessel openings. It knows no task target, score or expected action sequence.
No rigid/particle writes, simulator handle, controller or target-volume input.
"""
from __future__ import annotations
from copy import deepcopy
from dataclasses import dataclass
import math
import numpy as np

SCHEMA='laboratory_piston_finite_volume.v1'
LIMITATIONS=[
    'Reduced incompressible volume and ideal piston seal; no resolved pressure, bubbles, viscosity or surface tension.',
    'No fluid weight or jet reaction is applied to native rigid bodies.',
    'Vessel contents have horizontal-level capacity approximated by the lowest tilted rim; overflow is spill.',
    'Flight packets have volume and gravity but no packet-to-packet interaction or splash; intercepted liquid settles immediately.',
    'Cylinder liquid-level visuals may synchronize at render points; no resolved jet/splash and no rendering feedback into physics.',
]


def rotation_xyzw(q):
    q=np.asarray(q,dtype=float)
    if q.shape!=(4,) or not np.isfinite(q).all() or abs(np.linalg.norm(q)-1)>1e-3:
        raise ValueError('Expected a finite unit quaternion')
    x,y,z,w=q/np.linalg.norm(q)
    return np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
        [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
        [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])


def _pose(pose):
    pose=np.asarray(pose,dtype=float)
    if pose.shape!=(7,) or not np.isfinite(pose).all():
        raise ValueError('Expected finite world pose xyzw')
    return pose[:3],rotation_xyzw(pose[3:])


@dataclass
class Packet:
    volume_ml:float
    position:np.ndarray
    velocity:np.ndarray


class PistonVolumeModel:
    """Native-independent finite-volume equations used by both runtime and probe."""
    def __init__(self,parameters):
        p=deepcopy(parameters)
        if p.get('schema')!=SCHEMA:
            raise ValueError('Unsupported finite-volume model')
        self.p=p
        self.area=float(p['piston_area_m2']);self.stroke=float(p['stroke_m'])
        if not 0<self.area<.01 or not 0<self.stroke<.5:
            raise ValueError('Invalid physical pump dimensions')
        self.volumes={name:float(c['initial_ml']) for name,c in p['containers'].items()}
        self.initial=sum(self.volumes.values())
        if abs(self.initial-float(p['initial_total_ml']))>1e-8 or any(v<0 for v in self.volumes.values()):
            raise ValueError('Invalid initial conserved liquid inventory')
        self.tool=0.;self.packets=[];self.spill=0.;self.aspirated=0.;self.dispensed=0.
        self.previous_q=float(p['initial_q_m']);self.steps=0;self.last_flow=0.;self.errors=[]

    def _geometry(self,poses):
        if set(poses)!=set(self.volumes):
            raise ValueError('Container pose inventory mismatch')
        geometry={}
        for name,c in self.p['containers'].items():
            pos,R=_pose(poses[name]);vertical=float(R[2,2])
            # Upright cross-section area times conservative low-rim height.
            theta=math.acos(np.clip(vertical,-1,1))
            h=max(0.,c['rim_z_m']-c['floor_z_m']-c['radius_m']*math.tan(theta)) if theta<math.pi/3 else 0.
            capacity=math.pi*c['radius_m']**2*h*1e6
            geometry[name]=(pos,R,c,capacity)
        return geometry

    def step(self,dt_s,*,piston_q_m,tool_pose_xyzw,container_poses,tool_velocity_m_s=(0,0,0)):
        dt=float(dt_s)
        if not math.isfinite(dt) or not 0<dt<=.025:
            raise ValueError('Require one native step of at most 25 ms')
        q=float(piston_q_m)
        if not math.isfinite(q) or not -.001<=q<=self.stroke+.001:
            raise ValueError('Measured piston displacement exceeds mechanical travel')
        q=float(np.clip(q,0,self.stroke))
        tool_pos,Rtool=_pose(tool_pose_xyzw)
        tip=tool_pos+Rtool@np.asarray(self.p['tip_local_m'],float)
        tool_vel=np.asarray(tool_velocity_m_s,dtype=float)
        if tool_vel.shape!=(3,) or not np.isfinite(tool_vel).all():
            raise ValueError('Invalid nozzle translational velocity')
        geo=self._geometry(container_poses)
        for name,(_,_,_,capacity) in geo.items():
            overflow=max(0.,self.volumes[name]-capacity)
            self.volumes[name]-=overflow;self.spill+=overflow
        # Advect already emitted liquid before adding this step's output packet.
        pending=[];g=np.array([0.,0.,-float(self.p['gravity_m_s2'])])
        for packet in self.packets:
            start=packet.position.copy()
            end=start+packet.velocity*dt+.5*g*dt*dt
            captured=None;earliest=2.
            for name,(pos,R,c,capacity) in geo.items():
                if capacity<=0:continue
                a=R.T@(start-pos);b=R.T@(end-pos)
                rim=float(c['rim_z_m'])
                if a[2]>=rim and b[2]<rim:
                    fraction=float((a[2]-rim)/(a[2]-b[2]))
                    impact=a+fraction*(b-a)
                    if np.linalg.norm(impact[:2])<c['radius_m'] and fraction<earliest:
                        captured=name;earliest=fraction
                elif c['floor_z_m']<a[2]<rim and np.linalg.norm(a[:2])<c['radius_m']:
                    # An immersed outlet releases directly inside its vessel.
                    captured=name;earliest=0.
            if captured is not None:
                capacity=geo[captured][3]
                kept=min(packet.volume_ml,max(0.,capacity-self.volumes[captured]))
                self.volumes[captured]+=kept;self.spill+=packet.volume_ml-kept
            elif end[2]<=self.p['spill_plane_z_m']:
                self.spill+=packet.volume_ml
            else:
                packet.position=end;packet.velocity+=g*dt;pending.append(packet)
        self.packets=pending
        displacement=(q-self.previous_q)*self.area*1e6
        self.previous_q=q;self.last_flow=0.
        if displacement>0:
            # Pulling in air increases empty chamber capacity without making water.
            candidates=[]
            for name,(pos,R,c,_) in geo.items():
                local=R.T@(tip-pos)
                level=c['floor_z_m']+self.volumes[name]*1e-6/(math.pi*c['radius_m']**2)
                if (R[2,2]>.5 and np.linalg.norm(local[:2])<c['radius_m']-.0005
                    and c['floor_z_m']+.0002<local[2]<level-.0002):
                    candidates.append(name)
            if len(candidates)>1:
                raise ValueError('Liquid vessels overlap at the pump outlet')
            if candidates:
                name=candidates[0]
                amount=min(displacement,self.volumes[name],max(0.,q*self.area*1e6-self.tool))
                self.volumes[name]-=amount;self.tool+=amount;self.aspirated+=amount;self.last_flow=-amount/dt
        elif displacement<0:
            amount=min(-displacement,self.tool)
            self.tool-=amount;self.dispensed+=amount;self.last_flow=amount/dt
            if amount>0:
                outlet_speed=min(8.,amount*1e-6/(dt*math.pi*self.p['nozzle_radius_m']**2))
                velocity=tool_vel-Rtool[:,2]*outlet_speed
                self.packets.append(Packet(amount,tip.copy(),velocity))
        self.steps+=1
        state=self.state()
        if not state['inventory_closed']:
            raise RuntimeError('Finite-volume water inventory failed to close')
        return state

    def state(self):
        flight=sum(p.volume_ml for p in self.packets)
        accounted=sum(self.volumes.values())+self.tool+flight+self.spill
        residual=self.initial-accounted
        return {'schema':SCHEMA,'source_ml':self.volumes[self.p['source_id']],
            'tool_ml':self.tool,'receivers_ml':{name:self.volumes[name] for name in self.p['receiver_ids']},
            'in_flight_ml':flight,'cumulative_spill_ml':self.spill,'initial_total_ml':self.initial,
            'accounted_total_ml':accounted,'inventory_residual_ml':residual,
            'inventory_closed':bool(abs(residual)<1e-7 and min([*self.volumes.values(),self.tool,flight,self.spill])>=-1e-9),
            'aspirated_ml':self.aspirated,'dispensed_ml':self.dispensed,'outlet_flow_ml_s':self.last_flow,
            'piston_q_m':self.previous_q,'step_count':self.steps,'errors':list(self.errors),
            'physics_model':'native piston displacement coupled to conserved finite volumes and gravity flight',
            'native_water_weight_reaction_applied':False,'limitations':list(LIMITATIONS)}


class ReducedLiquidDynamics:
    """Frozen task physics provider. New instance means a fresh/reset episode."""
    def __init__(self,task):
        metadata=task.get('asset_metadata',task.get('metadata',task))
        self.metadata=deepcopy(metadata)
        parameters=metadata['task_parameters']['hydraulics']
        self.model=PistonVolumeModel(parameters)
        self.p=parameters;self.previous_step=None;self.previous_time=None

    @staticmethod
    def _object_pose(obj):
        w,x,y,z=obj['quaternion_wxyz']
        return [*obj['position_m'],x,y,z,w]

    def on_frame(self,frame):
        step=int(frame['native_step']);now=float(frame['native_time_s']);dt=float(frame['physics_dt_s'])
        objects=frame['objects']
        poses={name:self._object_pose(objects[name]) for name in self.p['containers']}
        tool=self._object_pose(objects[self.p['tool_id']])
        piston=self._object_pose(objects[self.p['plunger_id']])
        pos,R=_pose(tool)
        # Native rigid transforms are the source; authored joint drive targets
        # are intentionally not an input to these equations.
        q=float((R.T@(np.asarray(piston[:3])-pos))[2]-self.p['joint_anchor_z_m'])
        if self.previous_step is None:
            if not frame.get('baseline'):
                raise ValueError('Liquid component requires an initial native baseline')
            if abs(q-self.p['initial_q_m'])>.001:
                raise ValueError('Liquid initial piston position does not match its frozen baseline')
            self.model.previous_q=float(np.clip(q,0,self.p['stroke_m']))
            self.model._geometry(poses)  # validate the complete initial pose inventory
        else:
            if frame.get('baseline') or step!=self.previous_step+1 or abs(now-self.previous_time-dt)>1e-7:
                raise ValueError('Liquid component requires fresh consecutive native frames')
            self.model.step(dt,piston_q_m=q,tool_pose_xyzw=tool,container_poses=poses,
                tool_velocity_m_s=objects[self.p['tool_id']]['linear_velocity_m_s'])
        self.previous_step=step;self.previous_time=now

    def on_observe(self):
        return {**self.model.state(),'native_step':self.previous_step,'native_time_s':self.previous_time}
