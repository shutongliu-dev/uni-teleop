"""Opt-in edge entry through original wrist and finger position targets."""
from __future__ import annotations

import numpy as np

from scripts.validation.laundry_engineering_motion import minimum_jerk


TANGENT_PROFILES = {'single_panel_30mm_v1': .030, 'local_panel_20mm_v2': .020}


def validate_tangent_profile(profile):
    if not isinstance(profile, str) or profile not in TANGENT_PROFILES:
        raise ValueError('Unknown explicit single-panel material tangent profile')
    return TANGENT_PROFILES[profile]


def observed_garment_offset(spec, current_q, pick, material_direction, offset_m, *,
                            profile='single_panel_30mm_v1'):
    """Map a local sewn-panel material direction into fresh observed geometry."""
    if (isinstance(offset_m,bool) or not isinstance(offset_m,(int,float)) or
            not np.isfinite(offset_m) or not -.010<=offset_m<=.010):
        raise ValueError('Explicit pinch inward offset must be finite and within +/-10mm')
    radius=validate_tangent_profile(profile)
    rest=np.asarray(spec['rest_vertices_m'],float);q=np.asarray(current_q,float)
    direction=np.asarray(material_direction,float)
    if (q.shape!=rest.shape or q.ndim!=2 or q.shape[1]!=3 or not np.isfinite(q).all()
            or direction.shape!=(2,) or not np.isfinite(direction).all() or np.linalg.norm(direction)<.5):
        raise ValueError('Complete actual garment geometry and a material inward direction required')
    seeds=np.asarray(pick['seed_vertex_ids'],int)
    seed=pick.get('natural_corner_vertex_id')
    seed=int(seeds[np.argmax(q[seeds,2])]) if seed is None else int(seed)
    # Current flat sewn assets have two distinct rest panels. Never fit through
    # both coincident XY layers or through remote folded material.
    distance=np.linalg.norm(rest[:,:2]-rest[seed,:2],axis=1)
    ids=np.flatnonzero((distance<=radius+1.e-9)&(distance>1.e-8)&
                      (abs(rest[:,2]-rest[seed,2])<1.e-7))
    x=rest[ids,:2]-rest[seed,:2];y=q[ids]-q[seed]
    if len(ids)<3 or np.linalg.matrix_rank(x)!=2:
        raise ValueError('A resolved local single-panel material tangent is required')
    mapping=np.linalg.lstsq(x,y,rcond=None)[0]
    rms=float(np.sqrt(np.mean(np.sum((x@mapping-y)**2,axis=1))))
    world=direction/np.linalg.norm(direction)@mapping
    if rms>.003 or np.linalg.norm(world[:2])<.25:
        raise ValueError('Observed garment edge is too folded or uncertain for planar entry')
    inward=world[:2]/np.linalg.norm(world[:2])
    result={**pick,'inward_world_xy':inward.tolist(),
        'pinch_tcp_xy_m':(np.asarray(pick['edge_world_m'])[:2]+offset_m*inward).tolist(),
        'pinch_inward_offset_m':float(offset_m),
        'pinch_direction_basis':{'kind':'actual_local_single_panel_material_tangent',
            'seed_vertex_id':seed,'neighbour_vertex_ids':ids.tolist(),'fit_rms_m':rms,
            'maximum_material_radius_m':radius,'rest_panel_z_m':float(rest[seed,2]),
            'material_direction_xy':direction.tolist(),
            'source':'fresh actual cloth q; original material identities; XY projection'}}
    if profile != 'single_panel_30mm_v1':
        original_ids=np.flatnonzero((distance<=.030+1.e-9)&(distance>1.e-8)&
                                   (abs(rest[:,2]-rest[seed,2])<1.e-7))
        original_x=rest[original_ids,:2]-rest[seed,:2]
        original_y=q[original_ids]-q[seed]
        original_mapping=np.linalg.lstsq(original_x,original_y,rcond=None)[0]
        original_rms=float(np.sqrt(np.mean(np.sum((original_x@original_mapping-original_y)**2,axis=1))))
        original_world=direction/np.linalg.norm(direction)@original_mapping
        result['pinch_direction_basis'].update(profile=profile,
            original_30mm_fit={'neighbour_vertex_ids':original_ids.tolist(),
                'fit_rms_m':original_rms,'world_material_direction':original_world.tolist(),
                'planning_guard_passed':bool(original_rms<=.003 and np.linalg.norm(original_world[:2])>=.25)},
            scope='A smaller original single-panel tangent estimates entry at a curved material edge. '
                  'The same rank, neighbour count,3mm fit and XY-direction guards apply; actual '
                  'finite-area finger capture, penetration and native stability remain separately required.')
    return result


def edge_feed_plan(pick, setting):
    """A narrow open hand starts outside and ends slightly inside a real edge."""
    if 'closure_inward_travel_m' not in setting:
        return None
    travel=setting['closure_inward_travel_m'];start=setting.get('pinch_inward_offset_m')
    if (isinstance(travel,bool) or not isinstance(travel,(int,float)) or
            isinstance(start,bool) or not isinstance(start,(int,float)) or
            not np.isfinite([travel,start]).all() or not 0.<travel<=.012 or
            not -.010<=start<=0. or not 0.<=start+travel<=.010):
        raise ValueError('Explicit edge feed must start outside and move at most12mm to within10mm inside')
    opening=setting.get('open_distance_m',.060);closing=setting.get('close_distance_m',.010)
    if not .020<=opening<=.030 or not 0.<closing<opening:
        raise ValueError('Edge feed requires an explicit20–30mm open pinch and smaller closed gap')
    if setting.get('closure_lift_m',0.) or setting.get('capture_rise_m',0.):
        raise ValueError('Initial edge feed keeps its verified wrist height fixed')
    if 'pinch_direction_basis' not in pick:
        raise ValueError('Edge feed requires a freshly observed material edge direction')
    direction=np.asarray(pick['inward_world_xy'],float)
    if direction.shape!=(2,) or not np.isfinite(direction).all() or abs(np.linalg.norm(direction)-1.)>1.e-8:
        raise ValueError('Observed unit XY inward direction required')
    return {'schema':'laundry_observed_edge_feed.v1','start_inward_offset_m':float(start),
        'end_inward_offset_m':float(start+travel),'travel_m':float(travel),
        'translation_world_m':np.r_[travel*direction,0.].tolist(),
        'time_law':'same minimum-jerk fraction as original finger closure',
        'source':pick['pinch_direction_basis'],'cloth_state_written':False,
        'scope':'Planned bounded edge entry, not a contact-force or captured-material certificate.'}


def closure_displacement(plan, amount):
    if not np.isfinite(amount) or not 0.<=amount<=1.:
        raise ValueError('Closure fraction must lie in[0,1]')
    delta=np.zeros(3)
    if 'closure_feed' in plan:
        feed=plan['closure_feed'];vector=np.asarray(feed['translation_world_m'],float)
        if (feed.get('schema')!='laundry_observed_edge_feed.v1' or vector.shape!=(3,) or
                not np.isfinite(vector).all() or abs(vector[2])>1.e-12 or
                not 0.<np.linalg.norm(vector[:2])<=.012):
            raise ValueError('Bounded horizontal original-wrist feed required')
        delta+=amount*vector
    start=plan.get('closure_lift_start_fraction',0.)
    f=amount if start==0 else minimum_jerk(float(np.clip((amount-start)/(1-start),0.,1.)),0.,1.)
    delta[2]+=plan['closure_lift_m']*f
    return delta


def closure_wrists(base, edges, amount, *, feed_fractions=None):
    wrists={side:matrix.copy() for side,matrix in base.items()}
    for edge in edges:
        p=edge['contact_plan'];u=(feed_fractions or {}).get(edge['side'],amount) if 'closure_feed' in p else amount
        wrists[edge['side']][:3,3]+=closure_displacement(p,u)
    return wrists


class ClosureFeedProgress:
    """An observed finger stop also stops further feed, at the last sent target."""
    def __init__(self):
        self.last={};self.stopped={};self.events=[]

    def fractions(self, edges, assists, amount, time_s):
        result={}
        for edge in edges:
            if 'closure_feed' not in edge['contact_plan']:continue
            side=edge['side'];assist=next(a for a in assists if a.side==side)
            latched=assist.closure_stop is not None and assist.closure_stop.latched
            if latched and side not in self.stopped:
                self.stopped[side]=self.last.get(side,0.)
                self.events.append({'side':side,'time_s':float(time_s),
                    'last_submitted_feed_fraction':self.stopped[side],
                    'translation_world_m':(np.asarray(edge['contact_plan']['closure_feed']['translation_world_m'])*self.stopped[side]).tolist(),
                    'reason':'Measured finger closure stop freezes further wrist feed; no actual pose or cloth state is written.'})
            result[side]=self.stopped.get(side,float(amount))
            self.last[side]=result[side]
        return result


def preflight_edge_feed(base, edges, seed, solve):
    """Sequential20ms coupled wrist/finger IK; no simulator or target writes."""
    current=dict(seed);names=set(current);rows=[]
    for i in range(101):
        u=minimum_jerk(i*.02,0.,2.)
        current=solve(closure_wrists(base,edges,u),u,current)
        if set(current)!=names or not np.isfinite(list(current.values())).all():
            raise ValueError('Edge-entry IK changed the finite original joint inventory')
        rows.append({'time_s':i*.02,'closure_fraction':u,'joint_positions_rad':dict(current)})
    return rows
