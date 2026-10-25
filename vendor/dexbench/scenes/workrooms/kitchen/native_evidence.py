"""Read-only kitchen geometry and contact evidence helpers; no simulator access."""
import itertools
import math
import numpy as np

EPS = 1.e-8


def number(value, label='number'):
    if isinstance(value, bool):
        raise ValueError(label+' must be numeric')
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(label+' must be finite')
    return result


def vector(value, length, label='vector'):
    if len(value) != length:
        raise ValueError(label+' has wrong dimension')
    return tuple(number(x, label) for x in value)


def norm(value):
    return math.sqrt(sum(x*x for x in value))


def subtract(a, b):
    return tuple(x-y for x, y in zip(a, b))


def rotation(q):
    w, x, y, z = q
    return ((1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)),
            (2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)),
            (2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)))


def rotate(r, p):
    return tuple(sum(a*b for a, b in zip(row, p)) for row in r)


def angle(a, b):
    return 2*math.acos(min(1., abs(sum(x*y for x, y in zip(a, b)))))


def state(row):
    q = vector(row['quaternion_wxyz'], 4, 'quaternion')
    length = norm(q)
    if abs(length-1.) > 1.e-4:
        raise ValueError('Native quaternion must be normalized wxyz')
    return {'position_m': vector(row['position_m'], 3, 'position'),
        'quaternion_wxyz': tuple(x/length for x in q),
        'linear_velocity_m_s': vector(row['linear_velocity_m_s'], 3, 'linear velocity'),
        'angular_velocity_rad_s': vector(row['angular_velocity_rad_s'], 3, 'angular velocity')}


def fixed_state(pose):
    p = vector(pose, 7, 'authored pose xyzw')
    return state({'position_m': p[:3], 'quaternion_wxyz': (p[6], *p[3:6]),
        'linear_velocity_m_s': (0., 0., 0.), 'angular_velocity_rad_s': (0., 0., 0.)})


def world_points(obj, bounds, support_points=None):
    points = list(itertools.product(*zip(*bounds))) if support_points is None else support_points
    return np.asarray(points, dtype=float) @ np.asarray(rotation(obj['quaternion_wxyz'])).T + obj['position_m']


def local_points(points, reference):
    return (np.asarray(points)-reference['position_m']) @ np.asarray(rotation(reference['quaternion_wxyz']))


def contained(points, bounds, tolerance=0., radius=None):
    points = np.asarray(points)
    if radius is not None:
        return bool(np.all(np.linalg.norm(points[:, :2], axis=1) <= radius+tolerance))
    lo, hi = np.asarray(bounds)
    return bool(np.all(points[:, :2] >= lo-tolerance) and np.all(points[:, :2] <= hi+tolerance))


def upright(obj, tolerance):
    return rotation(obj['quaternion_wxyz'])[2][2] >= math.cos(tolerance)


def still(obj, limits):
    return (norm(obj['linear_velocity_m_s']) <= limits['maximum_linear_speed_m_s']
        and norm(obj['angular_velocity_rad_s']) <= limits['maximum_angular_speed_rad_s'])


def near(a, b, position, orientation):
    return (norm(subtract(a['position_m'], b['position_m'])) <= position
        and angle(a['quaternion_wxyz'], b['quaternion_wxyz']) <= orientation)


def relative_state(obj, reference):
    # Relative position and orientation are used only to bound contact memory.
    inverse = tuple(zip(*rotation(reference['quaternion_wxyz'])))
    a = reference['quaternion_wxyz']
    b = obj['quaternion_wxyz']
    w, x, y, z = a[0], -a[1], -a[2], -a[3]
    v, i, j, k = b
    return {'position_m': rotate(inverse, subtract(obj['position_m'], reference['position_m'])),
        'quaternion_wxyz': (w*v-x*i-y*j-z*k, w*i+x*v+y*k-z*j,
                           w*j-x*k+y*v+z*i, w*k+x*j-y*i+z*v)}


class Contacts:
    """Keep all contacts, including zero-load manifolds; never infer force scale."""
    def __init__(self, rows, *, time_s, actors, fixtures, robot_prefix, limits):
        self.rows = []
        self.touched = set()
        self.robot_touched = set()
        self.penetration = False
        self.floor_touched = set()
        self.actor_paths = {**fixtures, **actors}
        seen = set()
        def identify(path):
            if not isinstance(path, str) or not path.startswith('/'):
                raise ValueError('Absolute native contact actor paths required')
            for name, root in sorted(self.actor_paths.items(), key=lambda kv: -len(kv[1])):
                if path == root or path.startswith(root+'/'):
                    return name
            return path
        for raw in rows:
            original_a, original_b = raw['actor0'], raw['actor1']
            if original_a == original_b:
                raise ValueError('Native self contact pair')
            key = tuple(sorted((original_a, original_b)))
            if key in seen:
                raise ValueError('Duplicated native contact actor pair')
            seen.add(key)
            if abs(number(raw['time_s'], 'contact time')-time_s) > EPS:
                raise ValueError('Contact is from another native step')
            count = raw['point_count']
            impulse = number(raw['impulse_magnitude_sum_ns'], 'contact impulse')
            separation = number(raw['min_separation_m'], 'contact separation')
            if type(count) is not int or count <= 0 or impulse < 0:
                raise ValueError('Malformed native contact summary')
            signed = vector(raw['impulse_vector_ns'], 3, 'signed native impulse')
            if norm(signed) > impulse+1.e-8:
                raise ValueError('Signed contact resultant exceeds summed point impulse')
            a, b = identify(original_a), identify(original_b)
            relevant = a in actors or b in actors
            if relevant and separation < -limits['maximum_penetration_m']:
                self.penetration = True
            for name, other in ((a, original_b), (b, original_a)):
                if name in actors:
                    self.touched.add(name)
                    if other.startswith(robot_prefix):
                        self.robot_touched.add(name)
                    if any(part in other.lower() for part in ('/floor', '/ground')):
                        self.floor_touched.add(name)
            self.rows.append({'a': a, 'b': b, 'original_a': original_a, 'original_b': original_b,
                'positive': impulse > limits['minimum_contact_impulse_ns'], 'separation_m': separation,
                'impulse_vector_ns': signed, 'impulse_magnitude_sum_ns': impulse})

    def pair(self, first, second, positive=True):
        return any({row['a'], row['b']} == {first, second}
            and (not positive or row['positive']) for row in self.rows)

    def upward(self, body, support, minimum):
        return any({row['a'], row['b']} == {body, support}
            and row['impulse_vector_ns'][2]*(1 if row['a'] == body else -1)
                > max(minimum, .2*row['impulse_magnitude_sum_ns'])
            for row in self.rows)

    def external(self, body, allowed, robot_allowed=False):
        for row in self.rows:
            if body not in (row['a'], row['b']):
                continue
            other = row['b'] if body == row['a'] else row['a']
            path = row['original_b'] if body == row['a'] else row['original_a']
            if other not in allowed and not (robot_allowed and path.startswith('/World/Robot/')):
                return True
        return False
