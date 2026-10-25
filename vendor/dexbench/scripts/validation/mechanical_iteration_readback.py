"""Read-only stage iteration settings and native per-body sleep sidecars."""
import json
import numpy as np

from scripts.validation.validate_assembly_contacts import Observer


def stage_iteration_settings(env):
    from pxr import Usd, UsdPhysics
    result={'rigid_bodies':{},'articulations':{},'physics_scenes':{},
        'scope':'Composed USD settings only; native executed island iteration count is not queried. GPU actors may share an island.'}
    for prim in Usd.PrimRange.Stage(env.stage,Usd.TraverseInstanceProxies(Usd.PrimAllPrimsPredicate)):
        path=str(prim.GetPath())
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            result['rigid_bodies'][path]={k:prim.GetAttribute('physxRigidBody:'+k).Get() for k in (
                'solverPositionIterationCount','solverVelocityIterationCount','sleepThreshold','stabilizationThreshold')}
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            result['articulations'][path]={k:prim.GetAttribute('physxArticulation:'+k).Get() for k in (
                'solverPositionIterationCount','solverVelocityIterationCount','sleepThreshold','stabilizationThreshold')}
        if prim.IsA(UsdPhysics.Scene):
            result['physics_scenes'][path]={k:prim.GetAttribute('physxScene:'+k).Get() for k in (
                'solverType','enableGPUDynamics','enableStabilization','minPositionIterationCount',
                'maxPositionIterationCount','minVelocityIterationCount','maxVelocityIterationCount')}
    return result


class SleepObserver(Observer):
    """Preserve the original observer and append a separate native sleep receipt."""
    def __init__(self,env,output):
        import omni.usd
        from omni.physx import get_physx_simulation_interface
        from pxr import PhysicsSchemaTools
        self.sleep_interface=get_physx_simulation_interface()
        self.sleep_stage_id=omni.usd.get_context().get_stage_id()
        self.sleep_body_ids={name:PhysicsSchemaTools.sdfPathToInt(env.scene_info['objects'][name]['path'])
            for name in env.object_names}
        self.sleep_rows=[]
        self.sleep_stream=(output/'native_sleep.jsonl').open('x')
        super().__init__(env,output)

    def step(self,*args,**kwargs):
        result=super().step(*args,**kwargs)
        before=self.env.capture_state()
        flags={name:bool(self.sleep_interface.is_sleeping(self.sleep_stage_id,path))
            for name,path in self.sleep_body_ids.items()}
        after=self.env.capture_state()
        if not all(np.array_equal(before[k],after[k]) for k in before):
            raise RuntimeError('Native sleep query modified state')
        row={'native_step':self.rows[-1]['native_step'],'time_s':self.rows[-1]['time_s'],
            'sleeping':flags,'state_unchanged_by_query':True}
        self.sleep_rows.append(row)
        self.sleep_stream.write(json.dumps(row,separators=(',',':'))+'\n')
        return result

    def close(self):
        try:
            return super().close()
        finally:
            self.sleep_stream.close()
