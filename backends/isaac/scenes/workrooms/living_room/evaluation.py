"""Frozen living-room scoring over private, complete native interval evidence."""
from copy import deepcopy
import math
import numpy as np
from .native_evidence import (rotation, quaternion, cavity_contains, normalized_contacts,
                              stable, support, touched, extrema)


class SequenceEvaluator:
    def __init__(self, task_spec):
        if task_spec.get('schema_version') != 'living-room-task-v1':
            raise ValueError('Unsupported living-room task schema')
        self.task = deepcopy(task_spec)
        self.ident = task_spec['id']
        self.e = self.task['evaluation']
        self.previous = None
        self.frames = []
        self.stable_since = None
        self.progress = 0
        self.held_since = None
        self.pending_button = None
        self.button_down = set()
        self.spilled = set()
        self.spout_particles = set()
        self.initial_particle_ids = None
        self.previous_in_kettle = None
        self.previous_particle_local = None
        self.water_lift_held = False
        self.game = self.tower = None
        if self.ident == 'tower_building':
            from .tasks.tower_building.evaluator import TowerBuildingEvaluator
            self.tower = TowerBuildingEvaluator(self.e['reference_configuration'], trusted_normalized_input=True)
        if self.ident == 'tower_extraction':
            from .tasks.tower_extraction.evaluator import TowerExtractionGame
            self.game = TowerExtractionGame(self.e['reference_configuration'], trusted_normalized_input=True)
        self.last_result = self._result('pending', 'native_evidence_pending')

    def _result(self, status, reason, **fields):
        result = {'task_id': self.ident, 'status': status, 'reason': reason, 'success': status=='success',
            'physics_accepted': False, 'native_manipulation_qualified': False,
            'qualification': deepcopy(self.task['qualification']), **fields}
        if (self.ident == 'tower_building'
                and self.e['reference_configuration'].get('scoring_mode') == 'freeform_height'):
            from .height_public import height_verdict
            reference = fields.get('reference_result', {})
            for key in ('score', 'score_unit', 'current_height_m', 'best_height_m', 'best_height_cm',
                        'stable_duration_s', 'score_valid'):
                if key in reference:
                    result[key] = deepcopy(reference[key])
                elif hasattr(self, 'last_result') and key in self.last_result:
                    result[key] = deepcopy(self.last_result[key])
            result.setdefault('best_height_m', 0.)
            result.setdefault('best_height_cm', 0.)
            result.setdefault('score', result['best_height_m'])
            result.setdefault('score_unit', 'm')
            result.setdefault('current_height_m', 0.)
            result.setdefault('score_valid', True)
            result['success'] = False
            result['referee'] = height_verdict(status, result['best_height_m'])
            if status == 'invalid':
                result['score_valid'] = False
        self.last_result = result
        return deepcopy(result)

    def inspect_conditions(self, observation=None):
        return deepcopy(self.last_result)

    def _validate(self, f):
        if f.get('schema') != 'living_native_evidence_v1' or f.get('complete') is not True:
            raise ValueError('Complete service-owned native evidence is required')
        if set(f['objects']) != set(self.e['body_paths']):
            raise ValueError('Incomplete native rigid inventory')
        dt = self.e['physics_dt_s']
        if self.previous is None:
            if f['step_index'] != 0 or f['time_s'] != 0:
                raise ValueError('Unscored native reset baseline required')
        elif (f['step_index'] != self.previous['step_index']+1 or
              abs(f['time_s']-self.previous['time_s']-dt)>1.e-9 or
              f['episode_id'] != self.previous['episode_id']):
            raise ValueError('Incomplete native sampling or reset changed')
        for obj in f['objects'].values():
            for key,n in [('position_m',3),('linear_velocity_m_s',3),('angular_velocity_rad_s',3)]:
                if np.asarray(obj[key]).shape != (n,) or not np.isfinite(obj[key]).all():
                    raise ValueError('Invalid native rigid state')
            quaternion(obj['quaternion_wxyz'])
        if not isinstance(f['raw_contacts'],list):
            raise ValueError('Raw native contact evidence required')

    def update(self, observation):
        if self.last_result['status'] == 'invalid':
            # Native collection can finish its admitted control window after a
            # task evidence rejection. Preserve the first cause instead of
            # overwriting it with secondary gaps against the last valid frame.
            return deepcopy(self.last_result)
        try:
            self._validate(observation)
            frame=deepcopy(observation)
            frame['contacts']=normalized_contacts(frame['raw_contacts'],self.e['body_paths'],game=self.game is not None,
                contact_actor_paths=self.e.get('contact_actor_paths'))
            if self.game is not None:
                result=self._game(frame)
            elif self.tower is not None:
                result=self._tower(frame)
            elif self.ident=='remote_operation':
                result=self._remote(frame)
            else:
                result=self._water(frame)
            self.previous=frame
            return result
        except (KeyError,ValueError,TypeError,IndexError) as exc:
            if self.game is not None:
                self.game.finish('evidence_missing')
            return self._result('invalid',str(exc))

    def _dwell(self, condition, now):
        if not condition:
            self.stable_since=None
        elif self.stable_since is None:
            self.stable_since=now
        return 0. if self.stable_since is None else now-self.stable_since

    def _remote(self, f):
        objects, contacts, dt=f['objects'],f['contacts'],self.e['physics_dt_s']
        body=objects['remote_body']; r=rotation(body['quaternion_wxyz'])
        joint=f['joints']; metadata=self.e['candidate_metadata']
        coordinates={key: float(joint[spec['joint_path']]['position_m']) for key,spec in metadata['buttons'].items()}
        down={k for k,v in coordinates.items() if v<=metadata['buttons'][k]['press_threshold_candidate_m']}
        released={k for k,v in coordinates.items() if v>=metadata['buttons'][k]['release_threshold_candidate_m']}
        grip_links=set()
        forbidden_support=False
        for row in contacts:
            if 'remote_body' not in (row['a'],row['b']): continue
            other=row['b'] if row['a']=='remote_body' else row['a']
            if other.startswith('robot/') and any(x in other.lower() for x in ('thumb','index','middle','ring','pinky','palm')):
                grip_links.add(other)
            elif other not in ('power_button','confirm_button'):
                forbidden_support=True
        extents=np.abs(r) @ np.array([.0275,.08,.0135])
        held=(len(grip_links)>=2 and not forbidden_support and body['position_m'][2]-extents[2]>.782 and r[2,2]>.95)
        if held:
            if self.held_since is None: self.held_since=f['time_s']
        else: self.held_since=None
        held_ready=self.held_since is not None and f['time_s']-self.held_since>=.05
        new_down=down-self.button_down
        if len(down)>1:
            self.progress=0;self.pending_button=None
        elif new_down:
            key=next(iter(new_down))
            finger=any((c['a']==key+'_button' and c['b'].startswith('robot/') and any(x in c['b'].lower() for x in ('thumb','index','middle','ring','pinky'))) or (c['b']==key+'_button' and c['a'].startswith('robot/') and any(x in c['a'].lower() for x in ('thumb','index','middle','ring','pinky'))) for c in contacts)
            expected=self.task['public_goal']['button_sequence'][self.progress] if self.progress<2 else None
            if held_ready and finger and key==expected:
                self.pending_button=key
            else:
                self.progress=0; self.pending_button=None
        if self.pending_button is not None:
            if not held:
                self.progress=0; self.pending_button=None
            elif self.pending_button in released:
                self.progress+=1; self.pending_button=None
        self.button_down=down|(self.button_down-released)
        holder=objects['holder']; hr=rotation(holder['quaternion_wxyz'])
        local=(np.asarray(body['position_m'])-holder['position_m'])@hr
        relative=hr.T@r
        size=np.array([.0275,.08,.0135]); bounds=np.abs(relative)@size
        inside=(np.all(np.abs(local[:2])+bounds[:2]<=np.asarray(metadata['holder']['opening_xy_m'])/2)
                and relative[2,2]>.995 and relative[1,1]>.995
                and .015 <= local[2] <= .022 and hr[2,2]>.995)
        settled=self.progress==2 and len(released)==2 and inside and stable(objects) and not touched(contacts,set(objects)) and support(contacts,'remote_body',{'holder'},dt) and support(contacts,'holder',{'table'},dt)
        dwell=self._dwell(settled,f['time_s'])
        if body['position_m'][2]<.5 or not (.1<body['position_m'][0]<1.2 and -.55<body['position_m'][1]<.55):
            return self._result('failed','remote_left_workspace')
        return self._result('success' if dwell>=1. else 'active','press_release_and_holder_stability',
                            sequence_progress=self.progress, stable_duration_s=dwell,
                            button_thresholds_calibrated=False)

    def _water(self,f):
        meta=self.e['candidate_metadata']; water=meta['water']; part=f['particle_sets']['water']
        ids=np.asarray(part['ids']); p=np.asarray(part['positions_world_m'],dtype=float);v=np.asarray(part['velocities_world_m_s'],dtype=float)
        if (part.get('complete') is not True or len(ids)!=water['particle_count'] or len(set(ids.tolist()))!=len(ids)
                or p.shape!=(len(ids),3) or v.shape!=p.shape or not np.isfinite(p).all() or not np.isfinite(v).all()
                or abs(float(part['total_mass_kg'])-water['total_mass_kg'])>1.e-12
                or abs(float(part['particle_mass_kg'])-water['particle_mass_kg'])>1.e-15):
            raise ValueError('Incomplete fixed native particle inventory')
        if self.initial_particle_ids is None: self.initial_particle_ids=ids.copy()
        if not np.array_equal(ids,self.initial_particle_ids): raise ValueError('Particle IDs changed')
        masks={};local={}
        for name,spec in meta['containers'].items():
            body=f['objects'][name]
            local[name]=(p-np.asarray(body['position_m']))@rotation(body['quaternion_wxyz'])
            masks[name]=cavity_contains(local[name],spec['inner_profile_z_radius_spout_m'])
        if np.any(masks['cup']&masks['kettle']):raise ValueError('Ambiguous overlapping vessel interiors')
        outside=~(masks['cup']|masks['kettle'])
        # Conservative closed spill region: all exterior particles at/below the
        # tray surface, plus domain escape. Airborne particles remain pending.
        spilled=outside & ((p[:,2]<=meta['tray']['interior_floor_z_m']+.003) |
            (p[:,0]<.1)|(p[:,0]>1.2)|(np.abs(p[:,1])>.55)|(p[:,2]<0))
        self.spilled.update(ids[spilled].tolist())
        if touched(f['contacts'],{'kettle'}) and f['objects']['kettle']['position_m'][2]>.8061:
            self.water_lift_held=True
        if self.previous_in_kettle is not None:
            leaving=self.previous_in_kettle & ~masks['kettle']
            q=self.previous_particle_local
            # Crossing the actual lowered +X spout rim while the kettle is lifted.
            through=(q[:,0]>.04)&(np.abs(q[:,1])<.025)&(q[:,2]>.075)
            if self.water_lift_held and f['objects']['kettle']['position_m'][2]>.8061:
                self.spout_particles.update(ids[leaving&through].tolist())
        self.previous_in_kettle=masks['kettle'].copy();self.previous_particle_local=local['kettle'].copy()
        unit=water['total_mass_kg']/len(ids)/water['reference_density_kg_m3']*1.e6
        cup_ml=float(np.sum(masks['cup'])*unit);spill_ml=len(self.spilled)*unit
        inflight=outside&~spilled
        goals=self.task['public_rules']['goal'];objects=f['objects'];contacts=f['contacts']
        upright=all(rotation(o['quaternion_wxyz'])[2,2]>math.cos(math.radians(5)) for o in objects.values())
        origin=meta['containers']['kettle']['initial_position_m']
        returned=np.linalg.norm(np.asarray(objects['kettle']['position_m'])[:2]-origin[:2])<=.025
        supported=all(support(contacts,k,{'tray','table'},self.e['physics_dt_s']) for k in objects)
        condition=(goals['target_interval_ml'][0]<=cup_ml<=goals['target_interval_ml'][1] and spill_ml<=goals['max_spill_ml']
            and any(i in self.spout_particles for i in ids[masks['cup']].tolist()) and upright and returned and supported
            and not touched(contacts,set(objects)) and stable(objects) and not inflight.any()
            and np.linalg.norm(v,axis=1).max(initial=0)<=.02)
        dwell=self._dwell(condition,f['time_s'])
        # Quantitative two-way load transfer is an explicit unpassed rule gate.
        geometry_success=dwell>=1.
        status='failed' if spill_ml>goals['max_spill_ml'] else 'active'
        return self._result(status,'water_coupling_qualification_pending' if geometry_success else 'native_particle_accounting',
            cup_water_ml=cup_ml,cumulative_spill_ml=spill_ml,stable_duration_s=dwell,
            current_partition_count={**{k:int(v.sum()) for k,v in masks.items()},'in_flight':int(inflight.sum()),'other':int(spilled.sum())},
            geometric_conditions_met=geometry_success,water_coupling_qualified=False)

    def _tower(self,f):
        self.frames.append(f)
        count=round(self.e['sample_period_s']/self.e['physics_dt_s'])
        if f['step_index'] and f['step_index']%count:
            return deepcopy(self.last_result)
        config=self.e['reference_configuration']; states={}
        for ident,obj in f['objects'].items():
            vmax,wmax,_,_=extrema(self.frames,ident)
            states[ident]={k:deepcopy(obj[k]) for k in ('position_m','quaternion_wxyz','linear_velocity_m_s','angular_velocity_rad_s')}
            states[ident]['quaternion_wxyz']=quaternion(obj['quaternion_wxyz'])
            states[ident].update(interval_max_linear_speed_m_s=vmax,interval_max_angular_speed_rad_s=wmax)
        intervals=self.frames[1:] if len(self.frames)>1 else []
        pairs={tuple(sorted((r['a'],r['b']))) for x in intervals for r in x['contacts']}
        contacts=[]
        for a,b in sorted(pairs):
            per=[[r for r in x['contacts'] if {r['a'],r['b']}=={a,b}] for x in intervals]
            last=next(rows[-1] for rows in reversed(per) if rows)
            z=[sum(r['impulse_on_a_n_s'][2]*(1 if r['a']==a else -1) for r in rows)/self.e['physics_dt_s'] for rows in per]
            target=a if min(z,default=0)>0 else b if max(z,default=0)<0 else None
            if target not in states:target=None
            force=min((abs(v) for v in z),default=0) if target else 0.
            contacts.append({'a':a,'b':b,'point_m':last['point_m'],'separation_m':min(r['separation_m'] for rows in per for r in rows),
                'minimum_upward_force_n':force,'support_for':target,'continuous':bool(per and all(per))})
        obs={'schema':'living_tower_normalized_observation_v1','configuration_sha256':config['configuration_sha256'],
            'sampling_complete':True,'contacts_complete':True,'physics_valid':True,'robot_only_actions':True,'object_pose_writes':False,
            'episode_id':f['episode_id'],'sample_index':f['step_index']//count,'robot_action_count':f.get('robot_action_count',0),
            'budget_exhausted':False,'time_s':f['time_s'],'interval_start_s':self.frames[0]['time_s'],'objects':states,'contacts':contacts}
        result=self.tower.update(obs);self.frames=[f]
        return self._result(result['status'],result['reason'],reference_result=result)

    def _game(self,f):
        config=self.e['reference_configuration'];g=config['geometry'];h=g['block_size_m'][2]
        center=np.array([*g['center_xy_m'],g['table_top_z_m']]);dt=0. if self.previous is None else self.e['physics_dt_s']
        blocks={}
        for ident,obj in f['objects'].items():
            frames=[f] if self.previous is None else [self.previous,f]
            vmax,wmax,zmin,tilt=extrema(frames,ident)
            clearance=f['hand_clearance_m'][ident]
            blocks[ident]={'center_h':((np.asarray(obj['position_m'])-center)/h).tolist(),
                'quaternion_wxyz':quaternion(obj['quaternion_wxyz']),
                'linear_velocity_h_s':(np.asarray(obj['linear_velocity_m_s'])/h).tolist(),
                'angular_velocity_rad_s':obj['angular_velocity_rad_s'],
                'hand_distance_h':{s:float(clearance[s])/h for s in ('front','rear')},
                'interval_max_linear_speed_h_s':vmax/h,'interval_max_angular_speed_rad_s':wmax,
                'interval_min_center_z_h':(zmin-center[2])/h,'interval_max_tilt_rad':tilt}
        contacts=[]
        for r in f['contacts']:
            v=np.asarray(r['impulse_on_a_n_s']);m=float(np.linalg.norm(v))
            contacts.append({'a':r['a'],'b':r['b'],'normal_z_on_a':float(v[2]/m) if m else 0.,
                'normal_impulse_n_s':m,'point_h':((np.asarray(r['point_m'])-center)/h).tolist(),
                'separation_h':r['separation_m']/h,'supported_interval_s':dt if m else 0.,
                'minimum_normal_force_n':m/dt if dt else 0.})
        result=self.game.observe({'schema':'living_tower_extraction_normalized_observation.v1',
            'stream_id':f['episode_id'],'episode_id':f['episode_id'],'reset_id':f['episode_id'],
            'configuration_sha256':config['configuration_sha256'],'tick':f['step_index'],'time_s':f['time_s'],
            'blocks':blocks,'contacts':contacts,'executed_sides':f.get('executed_sides',[])})
        return self._result('success' if result['status'] in ('won','draw') else result['status'],result['reason'],game=result)
