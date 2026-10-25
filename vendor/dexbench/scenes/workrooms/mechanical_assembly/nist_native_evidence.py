"""Read-only geometry and native signed-contact checks for NIST adaptations."""
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
    w, x, y, z = q
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


def state(row):
    result = {}
    for key, size in (('position_m', 3), ('quaternion_wxyz', 4),
                      ('linear_velocity_m_s', 3), ('angular_velocity_rad_s', 3)):
        values = np.asarray(row[key], float)
        if values.shape != (size,) or not np.isfinite(values).all():
            raise ValueError('Complete finite native body state required')
        result[key] = values.tolist()
    rotation(result['quaternion_wxyz'])
    return result


def fixed(pose):
    return dict(position_m=list(pose[:3]), quaternion_wxyz=[pose[6], *pose[3:6]],
                linear_velocity_m_s=[0., 0., 0.], angular_velocity_rad_s=[0., 0., 0.])


def local(point, obj):
    return (np.asarray(point)-obj['position_m']) @ rotation(obj['quaternion_wxyz'])


def points(obj, bounds):
    return (np.asarray(list(itertools.product(*zip(*bounds)))) @
            rotation(obj['quaternion_wxyz']).T + obj['position_m'])


def distance(a, b):
    return float(np.linalg.norm(np.asarray(a['position_m'])-b['position_m']))


def angle(a, b):
    return 2*math.acos(min(1., abs(float(np.dot(a['quaternion_wxyz'], b['quaternion_wxyz'])))))


def route_in_order(polyline, gates):
    """Find strictly ordered intersections along the actual piecewise centerline."""
    cursor = -EPS
    hits = []
    for gate in gates:
        center, radius = np.asarray(gate['center_m']), gate['radius_m']
        found = None
        for index, (start, end) in enumerate(zip(polyline[:-1], polyline[1:])):
            if index+1 < cursor:
                continue
            delta, rel = end-start, start-center
            length2 = float(np.dot(delta, delta))
            if length2 <= EPS*EPS:
                if np.linalg.norm(rel) <= radius and index > cursor+EPS:
                    found = float(index)
                    break
                continue
            b, c = 2*float(np.dot(rel, delta)), float(np.dot(rel, rel))-radius*radius
            discriminant = b*b-4*length2*c
            if discriminant < 0:
                continue
            lower = max(0., (-b-math.sqrt(discriminant))/(2*length2), cursor-index+2*EPS)
            upper = min(1., (-b+math.sqrt(discriminant))/(2*length2))
            if lower <= upper+EPS:
                found = index+lower
                break
        if found is None:
            return False, hits
        cursor = found
        hits.append(float(found))
    return True, hits


def enclosed(polyline, target, plane_tolerance):
    """A planar loop must wind around and clear the complete pulley footprint."""
    vertices = np.asarray(polyline)
    center = np.asarray(target['center_m'])
    if np.max(np.abs(vertices[:, 2]-center[2])) > plane_tolerance:
        return False
    relative = vertices[:, :2]-center[:2]
    following = np.roll(relative, -1, axis=0)
    cross = relative[:, 0]*following[:, 1]-relative[:, 1]*following[:, 0]
    dot = np.einsum('ij,ij->i', relative, following)
    winding = float(np.arctan2(cross, dot).sum())
    if abs(abs(winding)-2*math.pi) > .1:
        return False
    for start, end in zip(relative, following):
        d = end-start
        alpha = float(np.clip(-np.dot(start, d)/max(float(np.dot(d, d)), 1.e-15), 0., 1.))
        if np.linalg.norm(start+alpha*d) < target['radius_m']-EPS:
            return False
    return True


def simple_loop(polyline):
    """Reject transverse crossings of nonadjacent belt centerline segments."""
    xy = np.asarray(polyline)[:, :2]
    count = len(xy)

    def cross(a, b):
        return a[0]*b[1]-a[1]*b[0]

    for i in range(count):
        a, b = xy[i], xy[(i+1) % count]
        for j in range(i+2, count):
            if i == 0 and j == count-1:
                continue
            c, d = xy[j], xy[(j+1) % count]
            denominator = cross(b-a, d-c)
            if abs(denominator) <= 1.e-12:
                continue
            t, u = cross(c-a, d-c)/denominator, cross(c-a, b-a)/denominator
            if EPS < t < 1-EPS and EPS < u < 1-EPS:
                return False
    return True


class Contacts:
    def __init__(self, rows, paths, actors, time_s, minimum, maximum_contact_gap=.0005):
        self.upward, self.touching, self.robot, self.floor = set(), set(), set(), set()
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
            a, b = row['actor0'], row['actor1']
            key = tuple(sorted((a, b)))
            if key in seen or a == b:
                raise ValueError('Duplicate or self contact')
            seen.add(key)
            if abs(number(row['time_s'])-time_s) > EPS:
                raise ValueError('Stale native contact')
            impulse = number(row['impulse_magnitude_sum_ns'])
            separation = number(row['min_separation_m'])
            vector = np.asarray(row['impulse_vector_ns'], float)
            if (impulse < 0 or vector.shape != (3,) or not np.isfinite(vector).all()
                    or np.linalg.norm(vector) > impulse+1.e-8):
                raise ValueError('Invalid signed native contact impulse')
            ia, ib = identify(a), identify(b)
            if ia in actors or ib in actors:
                self.minimum_separation = min(self.minimum_separation, separation)
            # The asset's 0.2 mm contact offset per collider permits a 0.4 mm
            # speculative envelope. Keep a modest 0.5 mm geometric allowance;
            # zero-load native manifolds can prove mating, never loaded support.
            if separation <= maximum_contact_gap:
                self.touching.add(frozenset((ia, ib)))
            if impulse > minimum:
                if vector[2] > max(minimum, .2*impulse):
                    self.upward.add((ia, ib))
                if -vector[2] > max(minimum, .2*impulse):
                    self.upward.add((ib, ia))
            for name, other, path in ((ia, ib, b), (ib, ia, a)):
                if name in actors:
                    if path.startswith('/World/Robot/'):
                        self.robot.add(name)
                    if other.startswith('environment_floor'):
                        self.floor.add(name)
