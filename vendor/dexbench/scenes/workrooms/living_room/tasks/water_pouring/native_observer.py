"""Service-owned, read-only particle/contact aggregation. No simulator imports."""
from collections.abc import Mapping
from copy import deepcopy
import hashlib
import math
import uuid
import numpy as np

from .rules import configuration_hash


def cavity_contains(local_points, profile):
    """Analytic physical cavity classification; points on a wall stay uncredited."""
    points = np.asarray(local_points, dtype=float)
    theta = np.arctan2(points[:, 1], points[:, 0])
    weight = np.maximum(0., np.cos(theta)) ** 12
    rows = [row for row in profile if row[1] > 0]
    radius = np.hypot(points[:, 0], points[:, 1])
    valid = np.zeros(len(points), dtype=bool)
    for (z0, r0, s0), (z1, r1, s1) in zip(rows, rows[1:]):
        lo, hi = z0-.2*s0*weight, z1-.2*s1*weight
        fraction = (points[:, 2]-lo) / (hi-lo)
        allowed = (r0+s0*weight)*(1-fraction) + (r1+s1*weight)*fraction
        valid |= ((points[:, 2] > lo) & (points[:, 2] <= hi) & (radius < allowed))
    return valid


def native_inverse(row):
    q = np.asarray(row['quaternion_wxyz'], float)
    p = np.asarray(row['position_m'], float)
    if p.shape != (3,) or q.shape != (4,) or not np.isfinite(p).all() or not np.isfinite(q).all() or abs(np.linalg.norm(q)-1.) > 1.e-4:
        raise ValueError('Invalid native vessel world pose')
    w,x,y,z = q/np.linalg.norm(q)
    r = np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
                  [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
                  [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])
    return r, -p@r


class NativeTaskObserver:
    def __init__(self, configuration):
        self.c = deepcopy(configuration['evaluation'])
        if self.c['contract_sha256'] != configuration_hash(self.c):
            raise ValueError('Water observer configuration digest mismatch')
        self.dt, self.n = self.c['physics_dt_s'], self.c['particle_count']
        self.reset_id = uuid.uuid4().hex
        self.previous = self.latest = self.window_start = None
        self.spilled = np.zeros(self.n, bool)
        self.left = np.zeros(self.n, bool)
        self.exit_regions = dict.fromkeys(('front_rim','other_rim','below_rim'), 0)
        self.max_lift = 0.
        self.last_support = {}
        self.window_count = 0
        self.window_support = dict.fromkeys(('kettle','cup'), math.inf)
        self.window_robot = dict.fromkeys(('kettle','cup'), 0.)
        self.error = None

    def _consume(self, frame):
        if not isinstance(frame, Mapping):
            raise ValueError('Native frame must be a read-only mapping')
        first = self.previous is None
        if bool(frame['baseline']) != first:
            raise ValueError('Reset/baseline changed within an episode')
        stamp, native_time, step = float(frame['time_s']), float(frame['native_time_s']), frame['native_step']
        if not np.isfinite([stamp,native_time]).all() or type(step) is not int or step < 0:
            raise ValueError('Invalid native clocks')
        if abs(float(frame['physics_dt_s'])-self.dt) > 1.e-12:
            raise ValueError('Native dt differs from frozen task')
        if not first and (step != self.previous[2]+1 or abs(native_time-self.previous[1]-self.dt) > 1.e-7
                          or abs(stamp-self.previous[0]-self.dt) > 1.e-7):
            raise ValueError('Missing/repeated native particle interval')
        particle = frame['particle_sets'][self.c['particle_set_id']]
        points, velocities, ids = (np.asarray(particle[k]) for k in ('positions_m','velocities_m_s','ids'))
        if points.shape != (self.n,3) or velocities.shape != points.shape or ids.shape != (self.n,):
            raise ValueError('Particle inventory shape changed')
        if not np.isfinite(points).all() or not np.isfinite(velocities).all() or not np.array_equal(ids,np.arange(self.n)):
            raise ValueError('Particle inventory/IDs/nonfinite state')
        total_mass = float(particle['total_mass_kg'])
        if not math.isfinite(total_mass) or abs(total_mass-self.n*self.c['particle_mass_kg']) > 1.e-6:
            raise ValueError('Physical particle mass declaration changed')
        bodies, classes, local, occupied = {}, {}, {}, np.zeros(self.n,bool)
        for name in ('kettle','cup'):
            row = frame['objects'][name]
            body = {k:np.asarray(row[k],float).tolist() for k in
                    ('position_m','quaternion_wxyz','linear_velocity_m_s','angular_velocity_rad_s')}
            if any(len(body[k]) != 3 or not np.isfinite(body[k]).all() for k in ('linear_velocity_m_s','angular_velocity_rad_s')):
                raise ValueError('Invalid native vessel velocity')
            r,t = native_inverse(body)
            local[name] = points@r+t
            mask = cavity_contains(local[name],self.c['containers'][name]['inner_profile_z_radius_spout_m'])
            if np.any(mask & occupied):
                raise ValueError('Vessel cavities overlap')
            classes[name], occupied = mask, occupied | mask
            bodies[name] = body
        outside = ~occupied
        tray, envelope = self.c['tray'], self.c['geometry_spill_envelope_m']
        bounds = np.asarray(tray['bounds_xy_m'])
        over = ((points[:,:2] >= bounds[0]) & (points[:,:2] <= bounds[1])).all(axis=1)
        expanded = ((points[:,:2] >= bounds[0]-envelope) & (points[:,:2] <= bounds[1]+envelope)).all(axis=1)
        wall = ((points[:,:2] <= bounds[0]+tray['wall_thickness_m']+envelope) |
                (points[:,:2] >= bounds[1]-tray['wall_thickness_m']-envelope)).any(axis=1)
        table = np.asarray(self.c['table_bounds_xy_m']); domain = np.asarray(self.c['task_domain_m'])
        over_table = ((points[:,:2] >= table[0]) & (points[:,:2] <= table[1])).all(axis=1)
        hits = ((over & (points[:,2] <= tray['interior_floor_z_m']+envelope)) |
                (expanded & wall & (points[:,2] <= tray['rim_top_z_m']+envelope)) |
                (over_table & (points[:,2] <= self.c['table_top_z_m']+envelope)) |
                (points[:,2] <= envelope) | ((points < domain[0]) | (points > domain[1])).any(axis=1)) & outside
        self.spilled |= hits
        first_exit = ~classes['kettle'] & ~self.left
        q = local['kettle']; high = q[:,2] >= .075
        front = high & (q[:,0] > .025) & (np.abs(q[:,1]) < .04)
        for name,mask in {'front_rim':front,'other_rim':high & ~front,'below_rim':~high}.items():
            self.exit_regions[name] += int(np.count_nonzero(first_exit & mask))
        self.left |= ~classes['kettle']
        classes.update(spilled_outside=outside & self.spilled, in_flight_or_unresolved=outside & ~self.spilled)
        counts = {name:int(np.count_nonzero(mask)) for name,mask in classes.items()}
        if sum(counts.values()) != self.n:
            raise ValueError('Particle compartments do not close')
        self.max_lift = max(self.max_lift,bodies['kettle']['position_m'][2]-self.c['containers']['kettle']['initial_position_m'][2])
        support, robot = dict.fromkeys(('kettle','cup'),0.), dict.fromkeys(('kettle','cup'),0.)
        for contact in frame['contacts']:
            impulse = float(contact['impulse_magnitude_sum_ns'])
            if not math.isfinite(impulse) or impulse < 0:
                raise ValueError('Invalid native contact impulse')
            for actor,other in ((contact['actor0'],contact['actor1']),(contact['actor1'],contact['actor0'])):
                for name,path in self.c['vessel_actor_paths'].items():
                    if actor == path or actor.startswith(path+'/'):
                        if any(other == p or other.startswith(p.rstrip('/')+'/') for p in self.c['support_actor_prefixes']):
                            support[name] += impulse/self.dt
                        if any(other.startswith(p) for p in self.c['robot_actor_prefixes']):
                            robot[name] += impulse/self.dt
        for name,body in bodies.items():
            mode = 'current_native_contacts'
            if support[name] > self.c['thresholds']['minimum_support_force_n']:
                self.last_support[name] = (deepcopy(body),support[name])
            elif name in self.last_support:
                previous,force = self.last_support[name]
                displacement = np.linalg.norm(np.asarray(body['position_m'])-previous['position_m'])
                angle = 2*math.acos(min(1.,abs(np.dot(body['quaternion_wxyz'],previous['quaternion_wxyz']))))
                th = self.c['thresholds']
                if (displacement <= th['sleep_support_position_tolerance_m'] and angle <= th['sleep_support_orientation_tolerance_rad']
                    and np.linalg.norm(body['linear_velocity_m_s']) <= th['maximum_linear_speed_m_s']
                    and np.linalg.norm(body['angular_velocity_rad_s']) <= th['maximum_angular_speed_rad_s']):
                    support[name],mode = force,'last_contact_held_at_unchanged_pose'
                else:
                    self.last_support.pop(name,None)
            body['support_mode'] = mode
            self.window_support[name] = min(self.window_support[name],support[name])
            self.window_robot[name] = max(self.window_robot[name],robot[name])
        self.latest = {'schema':'living_water_observation_v1','contract_sha256':self.c['contract_sha256'],
            'reset_id':self.reset_id,'time_s':stamp,'errors':[], 'water':{'particle_count':self.n,
                'ids_sha256':hashlib.sha256(ids.astype('<i8',copy=False).tobytes()).hexdigest(),
                'counts':counts,'cumulative_unique_spilled_particles':int(self.spilled.sum()),
                'ever_left_kettle_particles':int(self.left.sum()),'first_kettle_exit_regions':dict(self.exit_regions)},
            'vessels':bodies,'activity':{'max_kettle_lift_m':self.max_lift}}
        if first:
            self.window_start = (stamp,native_time,step)
            self.window_support = dict.fromkeys(('kettle','cup'),math.inf)
            self.window_robot = dict.fromkeys(('kettle','cup'),0.)
        else:
            self.window_count += 1
        self.previous = stamp,native_time,step

    def on_frame(self, frame):
        if self.error:
            return
        try:
            self._consume(frame)
        except (KeyError,TypeError,ValueError,IndexError,OverflowError) as error:
            self.error = str(error)

    def on_observe(self):
        if self.error or self.latest is None:
            return {'schema':'living_water_observation_v1','contract_sha256':self.c['contract_sha256'],
                    'reset_id':self.reset_id,'errors':[self.error or 'No native baseline']}
        result = deepcopy(self.latest)
        start,end = self.window_start,self.previous
        result['sampling'] = {'start_step':start[2],'end_step':end[2],
            'start_time_s':start[1],'end_time_s':end[1],'sampled_substeps':self.window_count,'physics_dt_s':self.dt}
        for name,body in result['vessels'].items():
            body['support_force_min_n'] = self.window_support[name] if math.isfinite(self.window_support[name]) else 0.
            body['robot_contact_force_max_n'] = self.window_robot[name]
        self.window_start = self.previous
        self.window_count = 0
        self.window_support = dict.fromkeys(('kettle','cup'),math.inf)
        self.window_robot = dict.fromkeys(('kettle','cup'),0.)
        return result
