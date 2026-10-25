"""Private native observer and task-specific referee, never a state setter."""
from copy import deepcopy
import io
import itertools
import json
from pathlib import Path
import uuid
import zipfile
import numpy as np

from bench.simulation.isaacsim_backend.contact_sample import SubstepContactExchange
from .native_evidence import raw_contacts, rotation


class CollisionClearance:
    """Conservative robot collision bounds moved by actual native body poses."""
    def __init__(self, env):
        from pxr import Usd, UsdGeom, UsdPhysics
        # Importers mark physical-only meshes as guide. CollisionAPI selects
        # physics geometry; render-purpose filtering must not omit its bounds.
        cache=UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_,UsdGeom.Tokens.render,UsdGeom.Tokens.proxy,UsdGeom.Tokens.guide],useExtentsHint=False,ignoreVisibility=True)
        names=list(env.body_names)
        xforms=UsdGeom.XformCache(Usd.TimeCode.Default())
        links=env.scene_info['robot_link_paths'];self.bounds=[]
        for prim in env.stage.Traverse(Usd.TraverseInstanceProxies()):
            path=str(prim.GetPath())
            if not path.startswith('/World/Robot/') or not prim.HasAPI(UsdPhysics.CollisionAPI):continue
            if UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get() is False:continue
            matches=[(name,p) for name,p in links.items() if path==p or path.startswith(p+'/')]
            if not matches:raise ValueError('Robot collider lacks native body mapping: '+path)
            name,bodypath=max(matches,key=lambda x:len(x[1]))
            if name not in names:raise ValueError('Robot collision body is absent from native readback')
            index=names.index(name)
            # A world AABB transformed into the initial body frame encloses the
            # collider. Its rigidly moved enclosure gives a lower distance bound.
            box=cache.ComputeWorldBound(prim).ComputeAlignedRange()
            low,high=np.asarray(box.GetMin()),np.asarray(box.GetMax())
            if not np.isfinite(np.r_[low,high]).all() or np.any(high<low):raise ValueError('Invalid robot collision bounds: '+path)
            corners=np.array(list(itertools.product(*zip(low,high))))
            # USD may still carry the authored ready/default articulation pose.
            # Remove its OWN body transform, never the independently read native
            # pose; applying native poses below then yields a genuine enclosure.
            authored=np.asarray(xforms.GetLocalToWorldTransform(env.stage.GetPrimAtPath(bodypath))).T
            local=(np.c_[corners,np.ones(len(corners))]@np.linalg.inv(authored).T)[:,:3]
            margin=.003
            attr=prim.GetAttribute('physxCollision:contactOffset')
            if attr and attr.Get() is not None:margin=max(margin,float(attr.Get()))
            lo,hi=local.min(axis=0)-margin,local.max(axis=0)+margin
            side=bodypath.split('/')[3]
            if side not in ('front','rear'):raise ValueError('Turn-game collision body lacks instance namespace')
            self.bounds.append((index,side,np.array(list(itertools.product(*zip(lo,hi))))))
        if not self.bounds or {s for _,s,_ in self.bounds}!={'front','rear'}:raise ValueError('Both actual robot collision inventories are required')

    def measure(self,state,objects,size):
        robot=[]
        for i,side,local in self.bounds:
            q=state['body_q'][i]; world=local@rotation([q[6],*q[3:6]]).T+q[:3]
            robot.append((side,world.min(axis=0),world.max(axis=0)))
        result={}
        for name,o in objects.items():
            center=np.asarray(o['position_m']);ext=np.abs(rotation(o['quaternion_wxyz']))@(np.asarray(size)/2)
            lo,hi=center-ext,center+ext
            result[name]={side:min(float(np.linalg.norm(np.maximum(np.maximum(lo-bhi,blo-hi),0)))
                for s,blo,bhi in robot if s==side) for side in ('front','rear')}
        return result


class LivingTaskRuntime:
    def __init__(self,env,scene,task,output,*,evaluator_factory):
        self.env,self.scene,self.task=env,deepcopy(scene),deepcopy(task)
        self.evaluator=evaluator_factory(task)
        self.output=Path(output);self.output.mkdir(parents=True,exist_ok=True)
        self.closed=False;self.episode_id=uuid.uuid4().hex
        self.steps=0;self.actions=0;self.executed=[];self.turn_actor=None
        self.exchange=None;self.original=env.world.step
        self.clearance=CollisionClearance(env) if self.evaluator.game is not None else None
        self.archive=zipfile.ZipFile(self.output/'living_native_evidence.zip','x',compression=zipfile.ZIP_DEFLATED,compresslevel=1)
        self.events=(self.output/'living_task_events.jsonl').open('x')
        self.initial_native_step=int(env.world.current_time_step_index)
        baseline=self._frame(env.capture_state(),[],0.)
        self.evaluator.update(baseline);self._record(baseline)
        if self.evaluator.last_result['status']=='invalid':
            self.archive.close();self.events.close();raise RuntimeError('Invalid living task reset: '+self.evaluator.last_result['reason'])
        self.exchange=SubstepContactExchange(env,self)
        env.world.step=self._step

    def _frame(self,state,rows,time):
        native=self.env.task_runtime_state(state)
        if native.get('schema')!='bench_task_runtime_v1':raise ValueError('Unsupported native task readback')
        if native.get('sampling',{}).get('complete') is not True:raise ValueError('Incomplete native task sampling')
        if int(native['native_physics_step_index'])!=int(state['native_physics_step_index']):raise ValueError('Mixed native task ticks')
        frame={'schema':'living_native_evidence_v1','complete':True,'episode_id':self.episode_id,
            'step_index':self.steps,'time_s':float(time),'native_physics_step_index':int(native['native_physics_step_index']),
            'native_world_time_s':float(native['native_world_time_s']),
            'objects':native['objects'],'joints':native['joints'],'particle_sets':native['particle_sets'],
            'raw_contacts':rows,'robot_action_count':self.actions,'executed_sides':list(self.executed)}
        if self.clearance:
            frame['hand_clearance_m']=self.clearance.measure(state,native['objects'],self.task['evaluation']['reference_configuration']['geometry']['block_size_m'])
        return frame

    def _record(self,frame):
        # All original particle coordinates remain privately recorded. Public
        # feedback contains only task quantities and referee state.
        payload=deepcopy(frame);arrays={}
        for name,row in payload['particle_sets'].items():
            for key in ('ids','positions_world_m','velocities_world_m_s'):
                arrays[name+'_'+key]=np.asarray(row.pop(key))
        arrays['frame_json']=np.asarray(json.dumps(payload,allow_nan=False))
        stream=io.BytesIO();np.savez(stream,**arrays)
        self.archive.writestr(f'{self.steps:09d}.npz',stream.getvalue())

    def _step(self,*args,**kwargs):
        if args or kwargs!={'render':False}:raise RuntimeError('Only existing native physics steps may be observed')
        env=self.env;self.exchange.begin_step(env);result=self.original(*args,**kwargs)
        sample=self.exchange.acquire(env);before=env.capture_state()
        rows=raw_contacts(sample.arrays,env._contact_sensor_paths,env._contact_filter_paths)
        self.steps+=1
        frame=self._frame(before,rows,float(env._current_step_time))
        projected=env._project_rigid_contacts(sample)
        after=env.capture_state()
        if set(before)!=set(after) or any(not np.array_equal(before[k],after[k]) for k in before):
            raise RuntimeError('Task observer changed native state')
        self.exchange.offer_after_readonly(env,sample,projected)
        self.evaluator.update(frame);self._record(frame)
        self.executed=[]
        return result

    def public_state(self):
        game=self.evaluator.game
        if game is None:return None
        raw=game.snapshot();completed=raw['status'] in ('won','draw')
        interrupted=raw['status'] in ('abort','incomplete')
        players={'front':'player_0','rear':'player_1'}
        return {'protocol':'turn_game_state_v1','game_id':self.task['id'],
            'status':'completed' if completed else 'interrupted' if interrupted else 'active',
            'turn_index':sum(raw['completed_turns'].values()),
            'current_player_id':None if completed or interrupted else players[raw['current_side']],
            'players':['player_0','player_1'],'winner_player_id':players.get(raw['winner']),
            'loser_player_id':players.get(raw['loser']),'terminal_reason':('native_referee_rejected' if raw['reason'] and ':' in raw['reason'] else raw['reason']) if completed or interrupted else None,
            'violation_reason':('invalid_evidence' if raw['reason'] and ':' in raw['reason'] else raw['reason']) if raw['status']=='abort' else None,
            'sim_time_s':float(self.env.time_s),'public_rules':deepcopy(self.task['public_rules']),
            'public_progress':{k:deepcopy(raw[k]) for k in ('status','selected_block','selection_locked','completed_turns','collected','legal_blocks')}}

    def score(self):
        result=deepcopy(self.evaluator.last_result)
        state=self.public_state()
        if state is not None:
            result.update(success=state['status']=='completed',turn_index=state['turn_index'],
                winner_player_id=state['winner_player_id'],loser_player_id=state['loser_player_id'],status=state['status'])
        if self.evaluator.game is not None and self.evaluator.game.status=='reset_pending':
            result['reason']='reset_settling_hold_targets'
        return result

    @staticmethod
    def _actor(actor):
        if isinstance(actor,dict):return actor.get('player_id')
        return getattr(actor,'player_id',actor)

    def request(self,command,actor=None):
        error=None
        if self.evaluator.game is None:
            raise ValueError('Ordinary task has no turn intents')
        who=self._actor(actor);side={'player_0':'front','player_1':'rear'}.get(who)
        if command.get('type')=='end_episode':
            self.evaluator.game.finish('budget' if command.get('reason')=='budget' else 'timeout')
        elif command.get('type')=='turn_intent':
            args=command.get('arguments',{})
            if not isinstance(args,dict) or set(args)-{'block_id'}:
                error={'message':'Only block_id is accepted','code':'invalid_intent'}
            else:
                result=self.evaluator.game.command(side,command.get('name'),args.get('block_id'))
                if not result['accepted']:error={'message':result['reason'],'code':'invalid_intent'}
        else:error={'message':'Unsupported task request','code':'invalid_command'}
        event={'command':deepcopy(command),'actor':who,'time_s':float(self.env.time_s),'state':self.public_state()}
        self.events.write(json.dumps(event,allow_nan=False)+'\n');self.events.flush()
        result={'ok':True,'turn_game':self.public_state(),'score':self.score()}
        if error:result['turn_game_error']=error
        return result

    def authorize_step(self,command,actor):
        if self.evaluator.game is None:
            self.actions+=1;return None
        state=self.public_state();who=self._actor(actor)
        if (state['status']!='active' or who!=state['current_player_id'] or not isinstance(actor,dict) or actor.get('turn_index')!=state['turn_index']):
            raise ValueError('Only the current authenticated player may act')
        targets=command.get('joint_positions_rad', {})
        names=list(self.env.joint_names)
        if set(targets)!=set(names):raise ValueError('Complete current robot targets required')
        side={'player_0':'front','player_1':'rear'}[who]
        for index,name in enumerate(names):
            own=name.startswith(('fr3_'+side+'_',side+'_'))
            if (not own or self.evaluator.game.status=='reset_pending') and abs(float(np.float32(targets[name]))-float(self.env._targets[0,index]))>1.e-7:
                raise ValueError('Inactive/reset robot drive targets must remain unchanged')
        changed=any(abs(float(np.float32(targets[name]))-float(self.env._targets[0,index]))>1.e-7 for index,name in enumerate(names))
        self.actions+=1;self.executed=[side] if changed else []
        return state['turn_index']

    def acknowledge_step(self,command,admitted_turn):
        self.executed=[]
        state=self.public_state()
        if state is None:return {}
        return {'turn_game':state,'turn_game_turn_changed':state['turn_index']!=admitted_turn}

    def close(self,reason='budget'):
        if self.closed:return None
        self.closed=True;self.env.world.step=self.original
        if self.evaluator.game is not None and self.evaluator.game.status not in ('won','draw','abort','incomplete'):
            self.evaluator.game.finish(reason if reason in ('timeout','budget','reset','evidence_missing','adapter_failure') else 'adapter_failure')
        receipt=None
        try:
            if self.exchange:receipt=self.exchange.close(self.env,self)
        finally:
            self.archive.close();self.events.close()
        (self.output/'living_sampling_receipt.json').write_text(json.dumps(receipt,indent=2,allow_nan=False)+'\n')
        return receipt


def build_runtime(env,scene,task,output,*,evaluator_factory):
    return LivingTaskRuntime(env,scene,task,output,evaluator_factory=evaluator_factory)
