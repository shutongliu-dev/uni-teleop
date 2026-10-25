"""Pure native observation validation shared by frozen mechanical task scorers."""
import math
import numpy as np

EPS = 1.e-9


def number(value, name):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(name + ' must be finite')
    return float(value)


def vector(value, count, name):
    if not isinstance(value, (list, tuple)) or len(value) != count:
        raise ValueError(name + ' has wrong dimension')
    return np.array([number(v, name) for v in value])


def rotation(value):
    q = vector(value, 4, 'quaternion')
    norm = np.linalg.norm(q)
    if abs(norm-1.) > 1.e-4:
        raise ValueError('orientation must be a unit wxyz quaternion')
    w, x, y, z = q/norm
    return np.array([[1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)],
                     [2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)],
                     [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)]])


def angle(a, b):
    return math.acos(float(np.clip((np.trace(a.T @ b)-1)/2, -1., 1.)))


def within(path, prefix):
    return path == prefix or path.startswith(prefix+'/')


def read_native_contact_window(obs, *, time, first, physics_dt_s, control_period_s,
                               previous_time=None, native_step=None, native_time=None):
    """Validate complete receipts and every normal-contact row independently.

    Returns (rows, native_end_step, native_end_time); does not mutate state or
    infer task support. Consumers retain returned counters for the next call.
    Empty rows can be valid no-contact samples, never evidence of support.
    """
    time = number(time,'window time')
    physics_dt_s = number(physics_dt_s,'physics dt')
    control_period_s = number(control_period_s,'control period')
    if (type(first) is not bool or physics_dt_s<=0 or control_period_s<=0
            or abs(control_period_s/physics_dt_s-round(control_period_s/physics_dt_s))>EPS):
        raise ValueError('Positive integral native sampling contract required')
    if (abs(number(obs['time_s'],'observation time')-time)>EPS
            or abs(number(obs['sim_time_s'],'simulation time')-time)>EPS):
        raise ValueError('Observation clocks disagree')
    if not first:
        previous_time = number(previous_time,'previous observation time')
        native_time = number(native_time,'previous native time')
        if type(native_step) is not int or native_step<0 or abs(time-previous_time-control_period_s)>EPS:
            raise ValueError('Consecutive observation/native counters required')
    contacts = obs['contacts']
    if not isinstance(contacts,dict):
        raise ValueError('Native contact payload must be a mapping')
    if contacts.get('source') != 'PhysX stage-bound rigid contact tensor; normal impulses' or contacts.get('errors') != []:
        raise ValueError('Native contact source or errors invalid')
    sampling = contacts['sampling']
    if not isinstance(sampling,dict):
        raise ValueError('Native sampling receipt must be a mapping')
    if sampling.get('protocol') != 'native_rigid_contact_sampling_v1' or sampling.get('complete') is not True:
        raise ValueError('Complete native sampling receipt required')
    if number(sampling['physics_dt_s'],'physics dt') != physics_dt_s:
        raise ValueError('Native physics dt differs')
    for key in ('sampled_substeps','native_start_step_index','native_end_step_index'):
        if type(sampling[key]) is not int or sampling[key] < 0:
            raise ValueError('Native step counters must be nonnegative integers')
    start,end,ns,ne = [number(sampling[k],k) for k in
        ('start_sim_time_s','end_sim_time_s','native_start_time_s','native_end_time_s')]
    count = 0 if first else round(control_period_s/physics_dt_s)
    duration = 0. if first else control_period_s
    if (min(start,end,ns,ne) < -EPS or sampling['sampled_substeps'] != count
            or sampling['native_end_step_index']-sampling['native_start_step_index'] != count
            or abs(end-time)>EPS or abs(end-start-duration)>EPS or abs(ne-ns-duration)>EPS
            or abs(number(obs['last_duration_s'],'duration')-duration)>EPS):
        raise ValueError('Incomplete native coverage or inconsistent clocks')
    rows = contacts['rigid']
    if not isinstance(rows,list):
        raise ValueError('Native rows must be a list')
    if first:
        if abs(time)>EPS or rows:
            raise ValueError('First observation must be unsampled reset baseline')
    elif (abs(start-previous_time)>EPS or sampling['native_start_step_index'] != native_step
            or abs(ns-native_time)>EPS):
        raise ValueError('Native windows must be consecutive')
    seen = set()
    for row in rows:
        if not isinstance(row,dict):
            raise ValueError('Native contact row must be a mapping')
        a,b = row['actor0'],row['actor1']
        if not isinstance(a,str) or not isinstance(b,str) or a==b:
            raise ValueError('Invalid native actor pair')
        tick = round((number(row['time_s'],'contact time')-start)/physics_dt_s)
        if first or not 1<=tick<=round(control_period_s/physics_dt_s) or abs(row['time_s']-start-tick*physics_dt_s)>EPS:
            raise ValueError('Stale, future or off-grid native contact')
        key = (tick,*sorted((a,b)))
        if key in seen: raise ValueError('Duplicate native contact pair')
        seen.add(key)
        impulse = number(row['impulse_magnitude_sum_ns'],'normal impulse')
        vec = vector(row['impulse_vector_ns'],3,'normal impulse vector')
        vector_norm = math.hypot(*vec)
        separation = number(row['min_separation_m'],'separation')
        normal_error = number(row['normal_length_max_abs_error'],'normal error')
        n,active = row['point_count'],row['nonzero_impulse_normal_count']
        # Aggregates are float64 point-vector sums. A dimensionless relative
        # allowance covers reduction roundoff without creating an absolute
        # impulse floor that many zero/tiny rows could accumulate as support.
        if (type(n) is not int or n<=0 or type(active) is not int or not 0<=active<=n
                or impulse<0 or not math.isfinite(vector_norm) or vector_norm>impulse*(1.+1.e-9)
                or (impulse==0. and (active!=0 or np.any(vec!=0.))) or impulse>0 and active==0
                or not 0<=normal_error<=32*2**-23 or row.get('impulse_components')!='normal_only'):
            raise ValueError('Invalid native normal-impulse evidence')
    return rows,sampling['native_end_step_index'],ne
