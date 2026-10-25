"""Small geometry and signed-contact helpers for frozen laboratory observers."""
import itertools
import math
import numpy as np

EPS = 1.e-7


def number(value):
    if type(value) not in (float, int) or not math.isfinite(value):
        raise ValueError('Finite native scalar required')
    return float(value)


def rotation(q):
    q = np.asarray(q, float)
    if q.shape != (4,) or not np.isfinite(q).all() or abs(np.linalg.norm(q)-1) > 1.e-4:
        raise ValueError('Unit native quaternion required')
    w,x,y,z = q
    return np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
                     [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
                     [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])


def state(row):
    result = {}
    for key, length in (('position_m', 3), ('quaternion_wxyz', 4),
                        ('linear_velocity_m_s', 3), ('angular_velocity_rad_s', 3)):
        values = np.asarray(row[key], float)
        if values.shape != (length,) or not np.isfinite(values).all():
            raise ValueError('Complete finite native body state required')
        result[key] = values.tolist()
    rotation(result['quaternion_wxyz'])
    return result


def points(obj, bounds):
    return (np.asarray(list(itertools.product(*zip(*bounds)))) @
            rotation(obj['quaternion_wxyz']).T + obj['position_m'])


def local(point, obj):
    return (np.asarray(point)-obj['position_m']) @ rotation(obj['quaternion_wxyz'])


def angle(a, b):
    return 2*math.acos(min(1.,abs(float(np.dot(a['quaternion_wxyz'], b['quaternion_wxyz'])))))


def distance(a, b):
    return float(np.linalg.norm(np.asarray(a['position_m'])-b['position_m']))


def fixed(pose):
    return {'position_m': list(pose[:3]), 'quaternion_wxyz': [pose[6], *pose[3:6]],
            'linear_velocity_m_s': [0.,0.,0.], 'angular_velocity_rad_s': [0.,0.,0.]}


def upright(obj, tolerance):
    return float(rotation(obj['quaternion_wxyz'])[2,2]) >= math.cos(tolerance)


class Contacts:
    def __init__(self, rows, paths, actors, time_s, minimum):
        self.upward, self.pairs, self.robot, self.floor = set(), set(), set(), set()
        self.minimum_separation = 0.
        ordered = sorted(paths.items(), key=lambda x: -len(x[1]))

        def identify(path):
            if not isinstance(path, str) or not path.startswith('/'):
                raise ValueError('Absolute contact path required')
            for name, root in ordered:
                if path == root or path.startswith(root+'/'):
                    return name
            return path

        seen = set()
        for row in rows:
            a,b = row['actor0'],row['actor1']
            key = tuple(sorted((a,b)))
            if key in seen or a == b:
                raise ValueError('Duplicate or self contact')
            seen.add(key)
            if abs(number(row['time_s'])-time_s) > EPS:
                raise ValueError('Stale contact')
            impulse = number(row['impulse_magnitude_sum_ns'])
            separation = number(row['min_separation_m'])
            vector = np.asarray(row['impulse_vector_ns'], float)
            if (impulse < 0 or vector.shape != (3,) or not np.isfinite(vector).all()
                    or np.linalg.norm(vector) > impulse+1.e-8):
                raise ValueError('Invalid signed contact impulse')
            ia,ib = identify(a),identify(b)
            if ia in actors or ib in actors:
                self.minimum_separation = min(self.minimum_separation, separation)
            if impulse > minimum:
                self.pairs.add(frozenset((ia,ib)))
                if vector[2] > max(minimum,.2*impulse):
                    self.upward.add((ia,ib))
                if -vector[2] > max(minimum,.2*impulse):
                    self.upward.add((ib,ia))
            for name,other,path in ((ia,ib,b),(ib,ia,a)):
                if name not in actors:
                    continue
                if path.startswith('/World/Robot/'):
                    self.robot.add(name)
                if other.startswith('environment_floor'):
                    self.floor.add(name)
