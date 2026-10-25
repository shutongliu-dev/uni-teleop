"""Opt-in render-only liquid cylinders from frozen, finite-volume observations.

This module has no physics/simulator handle. It writes only an existing visual
Cylinder's height, local translation Z and visibility at the render sync point.
Collision shapes, rigid poses, material quantities and native steps are untouched.
"""
from collections.abc import Mapping
import math
import re

SCHEMA='cylinder-volume-bindings-v1'


def number(value,lo,hi):
    if isinstance(value,bool) or not isinstance(value,(float,int)) or not math.isfinite(value) or not lo<=value<=hi:
        raise ValueError('Cylinder volume binding numbers must be finite and within the declared range')
    return float(value)


class CylinderVolumeBindings:
    def __init__(self,stage,config):
        self.bindings=[];self._selected={}
        if config is None:return
        from pxr import Sdf,UsdGeom,UsdPhysics
        if (not isinstance(config,dict) or set(config)!={'schema','bindings'} or config['schema']!=SCHEMA
                or not isinstance(config['bindings'],list) or not config['bindings']):
            raise ValueError('Expected cylinder-volume-bindings-v1 with explicit bindings')
        identifiers,paths=set(),set()
        for item in config['bindings']:
            keys={'id','target_prim_path','state_path','cross_section_area_m2','floor_z_m','maximum_height_m'}
            if not isinstance(item,dict) or set(item)!=keys:
                raise ValueError('Invalid cylinder volume binding fields')
            name,path=item['id'],item['target_prim_path']
            if not isinstance(name,str) or not re.fullmatch('[A-Za-z][A-Za-z0-9_]*',name) or name in identifiers:
                raise ValueError('Cylinder binding identifiers must be distinct')
            if (not isinstance(path,str) or not Sdf.Path.IsValidPathString(path) or not Sdf.Path(path).IsAbsolutePath()
                    or not Sdf.Path(path).IsPrimPath() or path in paths):
                raise ValueError('Cylinder binding targets must be distinct absolute prim paths')
            prim=stage.GetPrimAtPath(path)
            if (not prim or not prim.IsA(UsdGeom.Cylinder) or prim.IsInstanceProxy()
                    or prim.HasAPI(UsdPhysics.CollisionAPI) or prim.HasAPI(UsdPhysics.RigidBodyAPI)):
                raise ValueError('Liquid visuals must be existing noncollision, nonrigid Cylinder prims')
            # Leaf-only visuals cannot conceal a collision descendant whose
            # inherited local frame would move with this visual binding.
            if prim.GetChildren():
                raise ValueError('Liquid visual Cylinder must be a leaf prim')
            cylinder=UsdGeom.Cylinder(prim)
            if cylinder.GetAxisAttr().Get()!='Z':
                raise ValueError('Liquid visual Cylinder must use local Z')
            ops=UsdGeom.Xformable(prim).GetOrderedXformOps()
            if len(ops)!=1 or ops[0].GetOpType()!=UsdGeom.XformOp.TypeTranslate or ops[0].IsInverseOp():
                raise ValueError('Liquid visual Cylinder needs one ordinary local translate op')
            translation=ops[0].Get()
            if translation is None or len(translation)!=3 or any(not math.isfinite(v) for v in translation):
                raise ValueError('Liquid visual Cylinder requires finite local translation')
            state_path=item['state_path']
            if not isinstance(state_path,list) or not state_path or any(not isinstance(k,str) or not k for k in state_path):
                raise ValueError('Liquid visual binding requires an explicit state path')
            area=number(item['cross_section_area_m2'],1e-8,1.)
            floor=number(item['floor_z_m'],-10.,10.)
            maximum=number(item['maximum_height_m'],1e-6,2.)
            self.bindings.append({'id':name,'path':path,'keys':tuple(state_path),'area':area,'floor':floor,
                'maximum':maximum,'cylinder':cylinder,'translate':ops[0],'xy':tuple(translation[:2])})
            identifiers.add(name);paths.add(path)

    def sync(self,state,*,native_step):
        from pxr import Gf
        if type(native_step) is not int or native_step<0:
            raise ValueError('A native tick is required for visual liquid synchronization')
        planned=[]
        # Validate every source before any visual mutation; invalid observations
        # cannot leave only half the vessel displays updated.
        for row in self.bindings:
            value=state
            for key in row['keys']:
                if not isinstance(value,Mapping) or key not in value:
                    raise ValueError('Missing liquid visual state path: '+'.'.join(row['keys']))
                value=value[key]
            capacity=row['area']*row['maximum']*1e6
            volume=number(value,0.,capacity+1e-7)
            height=min(row['maximum'],volume*1e-6/row['area'])
            planned.append((row,volume,max(height,1e-6),'inherited' if height>1e-6 else 'invisible'))
        receipts=[]
        for row,volume,height,visibility in planned:
            selected=(height,visibility)
            changed=self._selected.get(row['id'])!=selected
            if changed:
                row['cylinder'].GetHeightAttr().Set(height)
                row['translate'].Set(Gf.Vec3d(*row['xy'],row['floor']+height/2))
                row['cylinder'].CreateVisibilityAttr().Set(visibility)
                self._selected[row['id']]=selected
            receipts.append({'id':row['id'],'target_prim_path':row['path'],'volume_ml':volume,
                'height_m':height,'visibility':visibility,'changed':changed})
        return {'schema':SCHEMA,'source_native_step':native_step,'bindings':receipts,
            'scope':'Only visual height/local-Z/visibility changed; no native physics state or step modification'}
