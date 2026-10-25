"""Scene-owned, read-only conversion of complete native samples to task units."""
import math
import numpy as np


def rotation(q):
    q = np.asarray(q, dtype=float)
    if q.shape != (4,) or not np.isfinite(q).all() or abs(np.linalg.norm(q)-1) > 1.e-4:
        raise ValueError('Finite native quaternion required')
    w, x, y, z = q/np.linalg.norm(q)
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


def quaternion(q):
    q = np.asarray(q, dtype=float)
    rotation(q)
    return (q/np.linalg.norm(q)).tolist()


def cavity_contains(local_points, profile):
    """Same analytical cavity as authored source, including lowered +X spout."""
    points = np.asarray(local_points, dtype=float)
    theta = np.arctan2(points[:, 1], points[:, 0])
    weight = np.maximum(0., np.cos(theta))**12
    rows = [row for row in profile if row[1] > 0]
    radius = np.hypot(points[:, 0], points[:, 1])
    valid = np.zeros(len(points), dtype=bool)
    for (z0, r0, s0), (z1, r1, s1) in zip(rows, rows[1:]):
        lo, hi = z0-.2*s0*weight, z1-.2*s1*weight
        fraction = (points[:, 2]-lo)/(hi-lo)
        allowed = (r0+s0*weight)*(1-fraction)+(r1+s1*weight)*fraction
        valid |= (points[:, 2] > lo) & (points[:, 2] <= hi) & (radius < allowed)
    return valid


def raw_contacts(arrays, sensors, filters):
    """Canonical actor pairs retain signed scalar-times-normal and every point.

    Native sensor/filter aliases may repeat a directed pair. They must be exactly
    equal. Reversed pairs must agree after reversing the impulse direction.
    """
    impulses, points, normals, separations, counts, starts = [np.asarray(v) for v in arrays]
    if counts.shape != starts.shape or counts.shape != (len(sensors), len(filters[0])):
        raise ValueError('Incomplete native contact mapping')
    result, seen = [], {}
    for i, j in zip(*np.nonzero(counts)):
        count, start = int(counts[i,j]), int(starts[i,j])
        if count < 0 or start < 0 or start+count >= len(impulses):
            raise ValueError('Native contact buffer overflow or invalid slice')
        a, b = sensors[i], filters[i][j]
        if a == b:
            raise ValueError('Native self-pair')
        sign = 1 if a < b else -1
        pair = tuple(sorted((a,b)))
        rows = []
        for k in range(start, start+count):
            vector = float(np.asarray(impulses[k]).reshape(-1)[0])*np.asarray(normals[k],dtype=float)*sign
            row = (np.asarray(points[k],dtype=float), vector, float(np.asarray(separations[k]).reshape(-1)[0]))
            if not np.isfinite(np.r_[row[0],row[1],row[2]]).all():
                raise ValueError('Nonfinite native contact')
            rows.append(row)
        # Point order can differ in reverse views; match by complete geometry.
        fingerprint = sorted(tuple(np.r_[p,v,s]) for p,v,s in rows)
        if pair in seen:
            if seen[pair] != fingerprint:
                raise ValueError('Mirrored native contact views disagree')
            continue
        seen[pair] = fingerprint
        result.extend({'a':pair[0], 'b':pair[1], 'point_m':p.tolist(),
                       'impulse_on_a_n_s':v.tolist(), 'separation_m':s} for p,v,s in rows)
    return result


def actor(path, body_paths, *, game=False, contact_actor_paths=None):
    for ident, root in body_paths.items():
        if path == root or path.startswith(root+'/'):
            return ident
    for ident, root in (contact_actor_paths or {}).items():
        if path == root or path.startswith(root+'/'):
            return ident
    if path.startswith('/World/Robot/'):
        pieces = path.split('/')
        side = pieces[3] if pieces[3] in ('front','rear') else 'front'
        leaf = path.rsplit('/',1)[-1]
        hand = not leaf.startswith('fr3_') and 'panda' not in leaf
        return ('hand/' if game and hand else 'robot/')+side if game else 'robot/'+leaf
    if path.startswith('/World/LivingTask/CatchTray/'):
        return 'tray'
    # The pinned living-room table's authored collider is named worktop.
    # Match that exact room subtree, not arbitrary props containing the word.
    worktop = '/World/Room/Geometry/worktop'
    if path == worktop or path.startswith(worktop+'/'):
        return 'table'
    low = path.lower()
    if '/floor' in low or '/ground' in low or '/rug' in low:
        return 'floor'
    if game:
        raise ValueError('Unmapped task contact actor: '+path)
    return 'external/'+path.strip('/').replace('/','_')


def normalized_contacts(rows, body_paths, *, game=False, contact_actor_paths=None):
    out=[]
    for row in rows:
        if not any(row[k] == p or row[k].startswith(p+'/') for p in body_paths.values() for k in ('a','b')):
            continue
        out.append({**row, 'a':actor(row['a'],body_paths,game=game,contact_actor_paths=contact_actor_paths),
                    'b':actor(row['b'],body_paths,game=game,contact_actor_paths=contact_actor_paths)})
    return out


def stable(objects, *, linear=.01, angular=.05):
    return all(np.linalg.norm(r['linear_velocity_m_s']) <= linear and
               np.linalg.norm(r['angular_velocity_rad_s']) <= angular for r in objects.values())


def support(rows, body, others, dt):
    return any(((r['a']==body and r['b'] in others and r['impulse_on_a_n_s'][2]>1.e-6*dt)
             or (r['b']==body and r['a'] in others and r['impulse_on_a_n_s'][2]<-1.e-6*dt))
             and -.001 <= r['separation_m'] <= .002 for r in rows)


def touched(rows, names, prefix='robot/'):
    return any((r['a'] in names and r['b'].startswith(prefix)) or
               (r['b'] in names and r['a'].startswith(prefix)) for r in rows)


def extrema(frames, ident):
    states=[f['objects'][ident] for f in frames]
    vs=[float(np.linalg.norm(s['linear_velocity_m_s'])) for s in states]
    ws=[float(np.linalg.norm(s['angular_velocity_rad_s'])) for s in states]
    for a,b,fa,fb in zip(states,states[1:],frames,frames[1:]):
        dt=fb['time_s']-fa['time_s']
        vs.append(float(np.linalg.norm(np.asarray(a['position_m'])-b['position_m'])/dt))
        qa,qb=quaternion(a['quaternion_wxyz']),quaternion(b['quaternion_wxyz'])
        # Bound the exact normalized endpoints handed to the scene referee.
        # A BLAS dot product can round differently from its scalar dot, and
        # suppressing tiny rotations can understate their interval speed.
        dot=min(1.,abs(sum(x*y for x,y in zip(qa,qb))))
        ws.append(2*math.acos(dot)/dt)
    return max(vs),max(ws),min(s['position_m'][2] for s in states),max(math.acos(np.clip(rotation(s['quaternion_wxyz'])[2,2],-1,1)) for s in states)
