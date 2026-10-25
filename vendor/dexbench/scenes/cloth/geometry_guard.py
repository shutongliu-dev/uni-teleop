"""Private cloth physics observer with separate diagnostic trajectory tracking.

ClothPhysicsObserver accepts arbitrary Agent joint commands: it has no reference
trajectory and never returns geometry, targets or teacher state to the Agent.
AbortGuard retains the original 5 mm diagnostic tracking gate by default.
No threshold, solver setting, state setter or integration step is changed here.
"""
from pathlib import Path
import copy,difflib,hashlib,json,time
import numpy as np
from bench.evaluation.cloth.cloth_robot_collision import cooked_piece_geometry,apply_transform,planes_in_body,transform_xyzw,state_digest
from bench.convex_geometry import convex_triangle_audit
from bench.evaluation.cloth.cloth_self_intersection_precomputed import self_intersections, StaticTriangleTopology
from scenes.cloth.layer_identity import FreshUsdcLayerFingerprint, PROTOCOL as LAYER_IDENTITY_PROTOCOL
from bench.simulation.isaacsim_backend.contact_precision import summarize_normal_impulses
from bench.simulation.isaacsim_backend.contact_sample import SubstepContactExchange, PROTOCOL as CONTACT_SAMPLE_PROTOCOL

PENETRATION=.0001
class GeometryStop(RuntimeError):
 def __init__(self,record):self.record=record;super().__init__('Readonly diagnostic stop: '+record['kind'])
def arr(x):
 if hasattr(x,'detach'):x=x.detach()
 if hasattr(x,'cpu'):x=x.cpu()
 if hasattr(x,'numpy'):x=x.numpy()
 return np.asarray(x).copy()
def dump(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def layer_hashes(env):return FreshUsdcLayerFingerprint()(env)
def archive_layer_snapshot(env,directory):
 """Read USD text once per layer; archive private UTF-8 evidence, never save USD."""
 directory=Path(directory);directory.mkdir(parents=True,exist_ok=False);rows=[];hashes={}
 for layer in env.stage.GetUsedLayers():
  identifier=layer.identifier
  if identifier in hashes:raise ValueError('Duplicate USD layer identifier')
  payload=layer.ExportToString().encode('utf-8');digest=hashlib.sha256(payload).hexdigest()
  # Layer identifiers are metadata, never filesystem paths or filename fragments.
  name=hashlib.sha256(identifier.encode('utf-8')).hexdigest()+'.usda.txt'
  path=directory/name;path.write_bytes(payload);hashes[identifier]=digest
  rows.append({'identifier':identifier,'sha256':digest,'byte_count':len(payload),'path':str(path.resolve())})
 manifest={'schema':'private_usd_layer_text_snapshot_v1','encoding':'utf-8','layers':rows}
 dump(directory/'manifest.json',manifest)
 return hashes,manifest

def archive_layer_differences(start,end,directory):
 """Complete unified diffs including added/removed layers; no ignored fields."""
 directory=Path(directory);directory.mkdir(exist_ok=False)
 before={row['identifier']:row for row in start['layers']};after={row['identifier']:row for row in end['layers']};changes=[]
 for identifier in sorted(set(before)|set(after)):
  a=before.get(identifier);b=after.get(identifier)
  if a is not None and b is not None and a['sha256']==b['sha256']:continue
  old=b'' if a is None else Path(a['path']).read_bytes();new=b'' if b is None else Path(b['path']).read_bytes()
  for row,payload in ((a,old),(b,new)):
   if row is not None and hashlib.sha256(payload).hexdigest()!=row['sha256']:raise ValueError('Layer evidence changed before diff')
  # splitlines/keepends preserves every source line, including a missing final LF.
  lines=difflib.unified_diff(old.decode('utf-8').splitlines(keepends=True),new.decode('utf-8').splitlines(keepends=True),fromfile='start/'+identifier,tofile='end/'+identifier)
  text=''
  for line in lines:
   text+=line if line.endswith('\n') else line+'\n\\ No newline at end of file\n'
  payload=text.encode('utf-8');path=directory/(hashlib.sha256(identifier.encode('utf-8')).hexdigest()+'.diff');path.write_bytes(payload)
  changes.append({'identifier':identifier,'change':'added' if a is None else 'removed' if b is None else 'changed','start':a,'end':b,'diff_path':str(path.resolve()),'diff_sha256':hashlib.sha256(payload).hexdigest(),'diff_byte_count':len(payload)})
 manifest={'schema':'private_usd_layer_complete_diff_v1','changed_layer_count':len(changes),'changes':changes,'ignored_identifiers_or_fields':[],'scope':'Private read-only USD evidence; strict original stage equality remains required.'}
 dump(directory/'manifest.json',manifest)
 return manifest

def overlap(lo,hi,other_lo,other_hi):return bool(np.all(hi>=other_lo) and np.all(lo<=other_hi))
def corners(bounds):return np.array([[bounds[k,0],bounds[j,1],bounds[i,2]] for i in (0,1) for j in (0,1) for k in (0,1)])
def rigid_failure(rows,robot_paths,mount_path):
 for row in rows:
  if not np.isfinite(row['min_separation_m']) or not np.isfinite(row['impulse_magnitude_sum_ns']):return {'kind':'invalid_rigid_contact','contact':row}
  if row['min_separation_m']<-.002:return {'kind':'rigid_separation_exceeds_original2mm','contact':row}
  a,b=row['actor0'],row['actor1'];robot=a if a in robot_paths else b if b in robot_paths else None;other=b if robot==a else a
  if robot is not None and robot!=mount_path and other=='/World/Table' and row['impulse_magnitude_sum_ns']>1e-8:return {'kind':'moving_robot_table_solved_contact','contact':row}
 return None

def raw_rigid_contacts(env,sample=None):
 data=env._rigid_contacts.get_contact_data(dt=1.) if sample is None else sample.for_env(env)
 force,_,normal,separation,counts,starts=data
 counts,starts=arr(counts),arr(starts);ids=np.argwhere(counts>0)
 if not len(ids):return []
 used=max(int(starts[i,j])+int(counts[i,j]) for i,j in ids)
 if used>=env._rigid_contacts.max_contact_data_count:raise ValueError('Rigid contact buffer reached capacity')
 force,normal,separation=arr(force[:used]).reshape(-1),arr(normal[:used]),arr(separation[:used]).reshape(-1)
 if not np.isfinite(force).all() or not np.isfinite(normal).all() or not np.isfinite(separation).all():raise ValueError('Nonfinite rigid contact raw arrays')
 rows=[];seen=set()
 for i,j in ids:
  a,b=env._contact_sensor_paths[i],env._contact_filter_paths[i][j];key=tuple(sorted((a,b)))
  if key in seen:continue
  seen.add(key);start,n=int(starts[i,j]),int(counts[i,j]);sl=slice(start,start+n)
  rows.append({'actor0':a,'actor1':b,'point_count':n,'min_separation_m':float(separation[sl].min()),**summarize_normal_impulses(force[sl],normal[sl])})
 return rows

class GeometryModel:
 def __init__(self,export,meta,triangles,initial_cloth,task_pieces,pad_local=None):
  if export.get('status')!='complete' or export.get('visual_mesh_fallback') is not False or export.get('issues'):raise ValueError('Verified native cook export required')
  self.names=meta['body_names'];self.robot=[];self.tris=np.asarray(triangles);self.pad_local=None if pad_local is None else np.asarray(pad_local,float);self.pad_index=None if pad_local is None else self.names.index('right_index_elastomer');self.tasks=[]
  initial=np.asarray(initial_cloth,dtype=float)
  if self.tris.ndim!=2 or self.tris.shape[1]!=3 or not np.issubdtype(self.tris.dtype,np.integer) or self.tris.min()<0 or self.tris.max()>=len(initial):raise ValueError('Invalid cloth topology')
  edges=np.sort(np.concatenate([self.tris[:,[0,1]],self.tris[:,[1,2]],self.tris[:,[2,0]]]),axis=1);self.edges=np.unique(edges,axis=0);self.rest_lengths=np.linalg.norm(initial[self.edges[:,0]]-initial[self.edges[:,1]],axis=1)
  if np.any(self.rest_lengths<=0):raise ValueError('Degenerate reference edges')
  for c in export['colliders']:
   if c['status']!='verified_cooking_representation':raise ValueError('Unchecked collider')
   body=self.names.index(c['body_name']);m=c['mesh_to_body_column_matrix']
   for p in c['pieces']:
    g=cooked_piece_geometry(p);v=apply_transform(g['conservative_outer_vertices'],m);self.robot.append({'body':body,'name':c['body_name'],'path':c['path'],'v':v,'planes':planes_in_body(g['conservative_outer_planes'],m),'inner':planes_in_body(g['conservative_inner_planes'],m),'bounds':np.stack([v.min(0),v.max(0)])})
  for x in task_pieces:
   if 'planes' not in x:raise ValueError('Unsupported task shape: do not substitute visual hull')
   t={k:np.asarray(v,float) if k in ('position','rotation','offset','local_rotation','planes','bounds') else v for k,v in x.items()};self.tasks.append(t)
  self.table=next(x for x in self.tasks if x['name']=='table');self.table_top=float(self.table['position'][2]+self.table['bounds'][1,2]);self.table_lo=self.table['position']+self.table['bounds'][0];self.table_hi=self.table['position']+self.table['bounds'][1]
  self.topology=StaticTriangleTopology(self.tris)
  self._cloth_key=None;self._cloth_result=None;self.last_profile={}
 def cloth_quality(self,raw_vertices):
  # This cache proves identity of every cloth coordinate, shape and dtype. It
  # never reuses a geometric verdict after motion, rounding or an approximate match.
  raw=np.ascontiguousarray(raw_vertices);key=(raw.shape,raw.dtype.str,raw.tobytes())
  if key==self._cloth_key:
   self.last_profile['cloth_quality_exact_state_cache_hit']=True
   return copy.deepcopy(self._cloth_result)
  self.last_profile['cloth_quality_exact_state_cache_hit']=False
  vertices=np.asarray(raw,dtype=float);tri=vertices[self.tris];lengths=np.linalg.norm(vertices[self.edges[:,0]]-vertices[self.edges[:,1]],axis=1);ratio=float((lengths/self.rest_lengths).max());area=np.linalg.norm(np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]),axis=1)
  if ratio>1.25:result={'kind':'cloth_edge_ratio_exceeds_original1_25','maximum':ratio}
  elif np.any(area<1e-12):result={'kind':'cloth_degenerate_triangle','triangles':np.flatnonzero(area<1e-12).tolist()}
  else:
   hit,candidates=self_intersections(vertices,self.tris,topology=self.topology)
   result={'kind':'cloth_self_intersection','pairs':hit} if hit else {'kind':None,'max_edge_ratio':ratio,'self_intersection_broadphase_candidates':int(candidates)}
  self._cloth_key=key;self._cloth_result=copy.deepcopy(result);return result
 def check(self,state,expected_pad):
  if self.pad_local is None:raise ValueError('Tracking requires an explicitly declared pad marker')
  expected=np.asarray(expected_pad,float)
  if expected.shape!=(3,) or not np.isfinite(expected).all():raise ValueError('Tracking requires a finite pad reference')
  started=time.perf_counter();self.last_profile={'cloth_quality_exact_state_cache_hit':None,'task_pieces_aabb_certified_clear':0,'task_pieces_triangle_checked':0,'robot_pieces_aabb_certified_clear':0,'robot_pieces_triangle_checked':0}
  try:return self._check(state,expected_pad)
  finally:self.last_profile['total_geometry_wall_s']=time.perf_counter()-started
 def _check(self,state,expected_pad):
  physical=self._check_physics(state)
  if physical['kind'] is not None:return physical
  pad=apply_transform(self.pad_local[None],transform_xyzw(np.asarray(state['body_q'])[self.pad_index]))[0];error=float(np.linalg.norm(pad-np.asarray(expected_pad)))
  if error>.005:return {'kind':'original5mm_tracking_exceeded','error_m':error,'pad_m':pad.tolist(),'reference_pad_m':list(expected_pad)}
  return {**physical,'pad_error_m':error}
 def check_physics(self,state):
  started=time.perf_counter();self.last_profile={'cloth_quality_exact_state_cache_hit':None,'task_pieces_aabb_certified_clear':0,'task_pieces_triangle_checked':0,'robot_pieces_aabb_certified_clear':0,'robot_pieces_triangle_checked':0}
  try:return self._check_physics(state)
  finally:self.last_profile['total_geometry_wall_s']=time.perf_counter()-started
 def _check_physics(self,state):
  if any(not np.isfinite(x).all() for x in state.values()):return {'kind':'nonfinite_state'}
  vertices=np.asarray(state['particle_q'],dtype=float);poses=np.asarray(state['body_q'],dtype=float);started=time.perf_counter()
  quality=self.cloth_quality(state['particle_q']);self.last_profile['cloth_quality_wall_s']=time.perf_counter()-started
  if quality['kind'] is not None:return quality
  lo,hi=vertices.min(0),vertices.max(0)
  for p in self.tasks:
   if p['object_index'] is None:position,rotation=p['position'],p['rotation']
   else:m=transform_xyzw(state['object_q'][p['object_index']]);position,rotation=m[:3,3],m[:3,:3]
   world=(corners(p['bounds'])@p['local_rotation'].T+p['offset'])@rotation.T+position
   if not overlap(lo,hi,world.min(0),world.max(0)):
    self.last_profile['task_pieces_aabb_certified_clear']+=1;continue
   self.last_profile['task_pieces_triangle_checked']+=1
   local=((vertices-position)@rotation-p['offset'])@p['local_rotation'];test=convex_triangle_audit(local,self.tris,p['planes'],p['bounds'],PENETRATION)
   if not test['all_triangles_certified_clear_within_tolerance']:return {'kind':'cloth_task_solid_penetration','piece':p['name'],**test}
  transforms={};local={};minimum_robot_gap=float('inf')
  for p in self.robot:
   body=p['body']
   if body not in transforms:transforms[body]=transform_xyzw(poses[body])
   m=transforms[body];world=apply_transform(p['v'],m);wlo,whi=world.min(0),world.max(0)
   if p['name']!='fr3_link0' and overlap(wlo[:2],whi[:2],self.table_lo[:2],self.table_hi[:2]):
    gap=float(wlo[2]-self.table_top);minimum_robot_gap=min(minimum_robot_gap,gap)
    if gap<=0.:return {'kind':'moving_robot_table_geometry_not_clear','body':p['name'],'minimum_table_plane_gap_m':gap,'classification':'conservative_AABB_plus_support_plane_abort_not_exact_full_convex_overlap'}
   if not overlap(lo,hi,wlo,whi):
    self.last_profile['robot_pieces_aabb_certified_clear']+=1;continue
   self.last_profile['robot_pieces_triangle_checked']+=1
   if body not in local:local[body]=apply_transform(vertices,np.linalg.inv(m))
   outer=convex_triangle_audit(local[body],self.tris,p['planes'],p['bounds'],PENETRATION)
   if not outer['all_triangles_certified_clear_within_tolerance']:
    inner=convex_triangle_audit(local[body],self.tris,p['inner'],p['bounds'],PENETRATION)
    return {'kind':'cloth_robot_penetration' if not inner['all_triangles_certified_clear_within_tolerance'] else 'cloth_robot_representation_unchecked','body':p['name'],'collider':p['path'],'outer':outer,'inner':inner}
  return {**quality,'minimum_robot_table_gap_m':minimum_robot_gap if np.isfinite(minimum_robot_gap) else None}

class AbortGuard:
 def __init__(self,env,model,requested_pad,output,*,mode='tracking'):
  if mode not in ('tracking','physical'):raise ValueError('Unknown observer mode')
  if mode=='physical' and requested_pad is not None:raise ValueError('Physical observer must not accept a control trajectory')
  reference=None if mode=='physical' else np.asarray(requested_pad,float)
  if mode=='tracking' and (reference.ndim!=2 or reference.shape[1]!=3 or not len(reference) or not np.isfinite(reference).all()):raise ValueError('Finite nonempty tracking trajectory required')
  self.mode=mode;self.env=env;self.model=model;self.reference=reference;self.original=env.world.step;self.callback=self.wrapped;self.start_step=int(env.world.current_time_step_index);self.start_time=float(env.world.current_time);self.last_step=self.start_step;self.last_time=self.start_time;self.dt=float(env.world.get_physics_dt());self.steps=0;self.rows=[];self.full_states=[];self.rigid=[];self.trip=None;self.closed=False;self.attached=False;self.output=Path(output);self.robot_paths=set(env.scene_info['robot_link_paths'].values());self.mount=env.scene_info['robot_link_paths']['fr3_link0'];self.started=time.monotonic();self.layer_evidence=self.output.with_name(self.output.name+'_layers');self.layers_start,self.layer_start_manifest=archive_layer_snapshot(env,self.layer_evidence/'start')
 def attach(self):
  state=self.env.capture_state();self.inspect(state,initial=True)
  self.env.world.step=self.callback;self.attached=True
 def inspect(self,state,initial=False):
  inspect_started=time.perf_counter();timing={}
  index=int(state['native_physics_step_index']);clock=float(state['native_world_time_s']);before=state_digest(state);sample_layers=layer_hashes(self.env) if self.steps%40==0 else None
  # Original20ms reference timing; nativefloat32 callback dt remains separate.
  expected=None
  if self.mode=='tracking':
   f=self.steps/40.;a=int(np.floor(f));b=min(a+1,len(self.reference)-1)
   if a>=len(self.reference):raise ValueError('Physical execution exceeded frozen reference')
   expected=(1-(f-a))*self.reference[a]+(f-a)*self.reference[b]
  rigid=[];rigid_completed=False;contact_sample=None;public_rows=None
  try:
   section_started=time.perf_counter()
   try:verdict=self.model.check(state,expected) if self.mode=='tracking' else self.model.check_physics(state)
   finally:timing['geometry_wall_s']=time.perf_counter()-section_started
   if not initial:
    section_started=time.perf_counter()
    try:
     exchange=getattr(self,'contact_exchange',None)
     if exchange is not None:contact_sample=exchange.acquire(self.env)
     rigid=raw_rigid_contacts(self.env,contact_sample);rigid_completed=True;rf=rigid_failure(rigid,self.robot_paths,self.mount);verdict=rf if rf is not None else verdict
     if contact_sample is not None:public_rows=self.env._project_rigid_contacts(contact_sample)
    finally:timing['rigid_contact_getter_wall_s']=time.perf_counter()-section_started
  except Exception as exc:verdict={'kind':'required_geometry_or_contact_unchecked','error_type':type(exc).__name__,'error':str(exc)}
  section_started=time.perf_counter()
  after=state_digest(self.env.capture_state());same=before==after and index==int(self.env.world.current_time_step_index) and clock==float(self.env.world.current_time)
  layer_same=True if sample_layers is None else sample_layers==layer_hashes(self.env)
  if not same or not layer_same:verdict={'kind':'readonly_state_or_layer_changed'}
  timing['readonly_verification_wall_s']=time.perf_counter()-section_started
  row={'sample_index':len(self.rows),'observed_steps':self.steps,'native_step_index':index,'native_time_s':clock,'public_time_s_at_callback':float(state['time_s']),'state_sha256_before':before,'state_sha256_after':after,'readonly_preserved':same and layer_same,'layer_check':sample_layers is not None,'layer_identity_protocol':LAYER_IDENTITY_PROTOCOL if sample_layers is not None else None,'rigid_contact_getter_completed':rigid_completed,'geometry':verdict,'geometry_profile':copy.deepcopy(getattr(self.model,'last_profile',{})),'timing':timing}
  self.rows.append(row);self.full_states.append({k:np.asarray(v).copy() for k,v in state.items()});self.rigid.append(rigid)
  timing['inspect_wall_s']=time.perf_counter()-inspect_started
  if verdict['kind'] is not None:
   self.trip=row;raise GeometryStop({'kind':verdict['kind'],'sample_index':row['sample_index'],'observed_steps':self.steps,'native_step_index':index,'native_time_s':clock,'geometry':verdict})
  if contact_sample is not None:self.contact_exchange.offer_after_readonly(self.env,contact_sample,public_rows)
 def wrapped(self,*args,**kwargs):
  if self.closed or args or kwargs!={'render':False}:raise ValueError('Guard permits only existing world.step(render=False)')
  if int(self.env.world.current_time_step_index)!=self.last_step or float(self.env.world.current_time)!=self.last_time:raise ValueError('Unobserved native advance')
  wrapped_started=time.perf_counter();result=self.original(*args,**kwargs);original_wall=time.perf_counter()-wrapped_started;self.steps+=1;index=int(self.env.world.current_time_step_index);clock=float(self.env.world.current_time)
  if index!=self.last_step+1 or not np.isfinite(clock) or abs(clock-self.last_time-self.dt)>1e-10:raise ValueError('Unexpected native substep interval')
  self.last_step,self.last_time=index,clock;section_started=time.perf_counter();state=self.env.capture_state();capture_wall=time.perf_counter()-section_started
  try:self.inspect(state)
  finally:
   if self.rows and self.rows[-1]['observed_steps']==self.steps:self.rows[-1]['timing']={**self.rows[-1]['timing'],'original_world_step_and_sampler_wall_s':original_wall,'initial_state_capture_wall_s':capture_wall,'complete_wrapped_step_wall_s':time.perf_counter()-wrapped_started}
  return result
 def close(self):
  if self.closed:raise ValueError('Already sealed')
  self.closed=True;owner_ok=not self.attached or self.env.world.step==self.callback
  if self.attached and owner_ok:self.env.world.step=self.original
  self.output.mkdir(exist_ok=False);dump(self.output/'trace.json',self.rows);dump(self.output/'rigid_contacts_per_substep.json',self.rigid)
  if self.full_states:np.savez_compressed(self.output/'states.npz',**{k:np.stack([s[k] for s in self.full_states]) for k in self.full_states[0]})
  timing={}
  for key in sorted({k for r in self.rows for k in r['timing']}):
   values=[r['timing'][key] for r in self.rows if key in r['timing']];timing[key]={'count':len(values),'sum_s':sum(values),'mean_s':sum(values)/len(values),'maximum_s':max(values)}
  layers_end,layer_end_manifest=archive_layer_snapshot(self.env,self.layer_evidence/'end')
  layer_changes=archive_layer_differences(self.layer_start_manifest,layer_end_manifest,self.layer_evidence/'diffs')
  result={'schema':'abort_only_substep_geometry_guard_v2','existing_steps_observed':self.steps,'samples_including_initial':len(self.rows),'physical_steps_added':0,'native_substep_remainder_modulo40':self.steps%40,'all_observer_reads_preserved':all(x['readonly_preserved'] for x in self.rows),'world_step_restored':owner_ok,'stage_layers_equal_start_end':self.layers_start==layers_end,'stage_layer_evidence':{'start_manifest':str((self.layer_evidence/'start'/'manifest.json').resolve()),'end_manifest':str((self.layer_evidence/'end'/'manifest.json').resolve()),'diff_manifest':str((self.layer_evidence/'diffs'/'manifest.json').resolve()),'changed_layer_count':layer_changes['changed_layer_count']},'trigger':self.trip,'all_sampled_geometry_passed':self.trip is None and bool(self.rows),'wall_s':time.monotonic()-self.started,'check_timing':timing,'coverage':{'geometry_every_existing_native_step':len(self.rows)==self.steps+1,'geometry_requested_interval_s':self.dt,'raw_rigid_contact_samples_completed':sum(r['rigid_contact_getter_completed'] for r in self.rows),'layer_interval_native_steps':40,'layer_interval_identity_protocol':LAYER_IDENTITY_PROTOCOL,'layer_endpoint_identity_protocol':'complete_utf8_usda_sha256_v1','layer_identity_semantics':'Fresh full USDC bytes; stricter than text for NaN payloads. Original complete start/end text archives and equality remain required.','cloth_quality_exact_state_cache_hits':sum(r['geometry_profile'].get('cloth_quality_exact_state_cache_hit') is True for r in self.rows),'cache_scope':'Only self-intersection/edge/area on identical complete cloth shape+dtype+bytes; robot/task poses and contacts always reread. AABB is conservative broadphase, nearby triangles exact clip. No temporal decimation.'},'scope':'Private abort-only diagnostic, not publiccontroller feedback, calibrated forces, continuousCCD or a guarantee before the first violatingstep. Initial rigid contact response is not evaluated until first real step.'};dump(self.output/'report.json',result);return result

class ClothPhysicsObserver(AbortGuard):
 """Physics checks only; primitive tracking is a separate controller contract."""
 def __init__(self,env,model,output):
  self.observer_failure=None;self.unacknowledged_state=None;self.contact_exchange=None
  super().__init__(env,model,None,output,mode='physical')
 def _failed(self,exc):
  if self.observer_failure is not None:return
  self.observer_failure={'error_type':type(exc).__name__,'last_observed_native_step':self.last_step}
  try:self.unacknowledged_state={k:np.asarray(v).copy() for k,v in self.env.capture_state().items()}
  except Exception as failure:self.observer_failure['state_capture_error_type']=type(failure).__name__
 def attach(self):
  try:
   super().attach()
   self.contact_exchange=SubstepContactExchange(self.env,self)
  except BaseException as exc:self._failed(exc);raise
 def wrapped(self,*args,**kwargs):
  if self.observer_failure is not None:raise RuntimeError('Observer already failed; no further physics step permitted')
  try:
   if self.contact_exchange is None:raise RuntimeError('Physical contact observer is not registered')
   self.contact_exchange.begin_step(self.env)
   return super().wrapped(*args,**kwargs)
  except BaseException as exc:
   if self.contact_exchange is not None:self.contact_exchange.invalidate()
   self._failed(exc);raise
 def close(self):
  try:result=super().close()
  except BaseException:
   if self.contact_exchange is not None:
    self.contact_exchange.invalidate();self.contact_exchange.close(self.env,self)
   raise
  contact_delivery=(self.contact_exchange.close(self.env,self) if self.contact_exchange is not None else
      {'protocol':CONTACT_SAMPLE_PROTOCOL,'acquisitions':0,'deliveries':0,'all_samples_consumed_once':False,'raw_arrays_reused_across_native_steps':False,'state_captures_reused':False})
  clock_matches=(int(self.env.world.current_time_step_index)==self.last_step and float(self.env.world.current_time)==self.last_time)
  valid=(self.observer_failure is None and self.trip is None and clock_matches
      and result['world_step_restored'] and result['stage_layers_equal_start_end']
      and result['all_observer_reads_preserved'] and result['coverage']['geometry_every_existing_native_step']
      and result['coverage']['raw_rigid_contact_samples_completed']==self.steps
      and contact_delivery['all_samples_consumed_once'] and contact_delivery['acquisitions']==self.steps==contact_delivery['deliveries'])
  if self.unacknowledged_state is not None:np.savez_compressed(self.output/'unacknowledged_state.npz',**self.unacknowledged_state)
  result.update(schema='cloth_agent_substep_physics_observer_v1',contact_delivery=contact_delivery,
      status='passed' if valid else 'failed',observer_failure=self.observer_failure,
      native_clock_matches_last_sample=clock_matches,
      unacknowledged_state_saved=self.unacknowledged_state is not None,
      diagnostic_trajectory_tracking_checked=False,
      task_success_claimed=False,
      scope='Private cloth/robot/task geometry and rigid contacts at every existing native step; no control trajectory or task score. No calibrated forces or continuous CCD claim.')
  dump(self.output/'report.json',result)
  return result
