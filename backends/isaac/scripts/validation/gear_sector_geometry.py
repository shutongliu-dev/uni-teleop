"""Exact closed radial gear sectors and their predeclared CPU witnesses."""
from pathlib import Path
import hashlib, json, math
import numpy as np
import trimesh


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def clip_polygon(vertices, normal):
    out=[]
    for a,b in zip(vertices, np.roll(vertices,-1,axis=0)):
        da,db=float(a@normal),float(b@normal)
        # Original exact on-plane coordinates may carry float64 sin(pi) roundoff.
        if abs(da)<2.e-16: da=0.
        if abs(db)<2.e-16: db=0.
        if da>=0: out.append(a)
        if (da<0 and db>0) or (da>0 and db<0):
            out.append(a+(b-a)*(da/(da-db)))
    return np.asarray(out)

def ear_triangles(vertices, loop, normal):
    # Radial coordinate plus Z is a stable 2D coordinate frame for vertical caps.
    radial=np.array([-normal[1],normal[0],0.])
    projected=np.c_[vertices@radial, vertices[:,2]]
    ids=list(loop)
    p=projected[ids]
    cross=lambda a,b:a[...,0]*b[...,1]-a[...,1]*b[...,0]
    area=np.sum(cross(p,np.roll(p,-1,axis=0)))
    if area<0: ids.reverse()
    faces=[]
    while len(ids)>3:
        found=False
        for k in range(len(ids)):
            tri=[ids[k-1],ids[k],ids[(k+1)%len(ids)]]
            a,b,c=projected[tri]
            turn=float(cross(b-a,c-a))
            if turn<=1.e-20: continue
            others=[i for i in ids if i not in tri]
            q=projected[others]
            if len(q) and np.any((cross(b-a,q-a)>=-1.e-20)&(cross(c-b,q-b)>=-1.e-20)&(cross(a-c,q-c)>=-1.e-20)):
                continue
            faces.append(tri); ids.pop(k);found=True;break
        if not found: raise ValueError('No nondegenerate ear in cap: '+str(len(ids)))
    faces.append(ids)
    faces=np.array(faces)
    n=np.cross(vertices[faces[0,1]]-vertices[faces[0,0]],vertices[faces[0,2]]-vertices[faces[0,0]])
    if n@normal<0: faces=faces[:,::-1]
    return faces

def sector(mesh,index,count):
    if count not in (4, 8, 16) or not 0 <= index < count:
        raise ValueError('A declared radial partition and valid sector index are required')
    a,b=2*math.pi*np.array([index,index+1])/count
    normals=[np.array([-math.sin(a),math.cos(a),0.]),np.array([math.sin(b),-math.cos(b),0.])]
    vertices=[];faces=[];parents=[]
    for i,triangle in enumerate(mesh.triangles):
        polygon=triangle
        for normal in normals:
            if len(polygon): polygon=clip_polygon(polygon,normal)
        if len(polygon)<3:continue
        for j in range(1,len(polygon)-1):
            tri=polygon[[0,j,j+1]]
            if np.linalg.norm(np.cross(tri[1]-tri[0],tri[2]-tri[0]))<=1.e-23:continue
            start=len(vertices);vertices.extend(tri);faces.append([start,start+1,start+2]);parents.append(i)
    result=trimesh.Trimesh(vertices=vertices,faces=faces,process=False)
    result.merge_vertices(digits_vertex=13)
    exterior_count=len(result.faces)
    parent=np.array(parents)
    exterior_area=np.bincount(parent,weights=result.area_faces,minlength=len(mesh.faces))
    counts=np.bincount(result.edges_unique_inverse)
    boundary=result.edges_unique[counts==1]
    adjacency={}
    for i,j in boundary:
        adjacency.setdefault(i,[]).append(j);adjacency.setdefault(j,[]).append(i)
    if any(len(v)!=2 for v in adjacency.values()):
        raise ValueError('Nonloop boundary '+str({k:len(v) for k,v in adjacency.items() if len(v)!=2}))
    loops=[];remaining=set(adjacency)
    while remaining:
        begin=min(remaining);loop=[begin];prev=None;current=begin
        while True:
            following=next(k for k in adjacency[current] if k!=prev)
            if following==begin:break
            if following in loop:raise ValueError('Self-intersecting boundary graph')
            loop.append(following);prev,current=current,following
        remaining.difference_update(loop);loops.append(loop)
    if len(loops)!=2:raise ValueError('Sector must have exactly two cut loops')
    capfaces=[]
    for loop in loops:
        distance=[np.max(abs(result.vertices[loop]@n)) for n in normals]
        chosen=int(np.argmin(distance))
        if distance[chosen]>1.e-12:raise ValueError('Cap not on declared radial plane')
        capfaces.extend(ear_triangles(result.vertices,loop,-normals[chosen]))
    result=trimesh.Trimesh(vertices=result.vertices,faces=np.vstack([result.faces,np.asarray(capfaces)]),process=False)
    if not result.is_watertight or not result.is_winding_consistent or result.volume<=0 or result.euler_number!=2 or result.body_count!=1:
        raise ValueError('Invalid closed sector')
    # Actual source facets are clipped, not resampled; cap triangles are the only added faces.
    source_plane_error=float(np.max(abs(np.einsum('ij,ikj->ik',mesh.face_normals[parent],result.triangles[:exterior_count]-mesh.triangles[parent,:1]))))
    return result,exterior_area,source_plane_error,exterior_count

def orientation(vertices):
    # Compare deterministic geometric frames, not a native-results-dependent search.
    candidates=[]
    for degrees in np.arange(0.,90.00001,.05):
        angle=math.radians(float(degrees));c,s=math.cos(angle),math.sin(angle)
        rotation=np.array([[c,-s,0.],[s,c,0.],[0.,0.,1.]])
        local=vertices@rotation
        extent=np.ptp(local,axis=0)
        candidates.append((float(max(extent)),float(degrees),extent))
    best=min(candidates,key=lambda x:(x[0],x[1]))
    return {'local_frame_yaw_deg':best[1], 'local_aabb_extents_m':best[2].tolist(),
            'max_extent_m':best[0], 'spacing_at_1024_m':best[0]/1024,
            'spacing_at_1250_m':best[0]/1250}


def yaw_matrix(degrees):
    a = math.radians(degrees)
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])


def nearest_surface(mesh, points):
    """Exact triangle distance with dimensionless kernels for tiny ledges."""
    triangles = mesh.triangles
    origins = triangles[:, 0]
    scales = np.linalg.norm(triangles - np.roll(triangles, 1, axis=1), axis=2).max(axis=1)
    normalized = (triangles - origins[:, None]) / scales[:, None, None]
    low, high = triangles.min(axis=1), triangles.max(axis=1)
    distances, nearest = [], []
    for p in np.asarray(points):
        # Any triangle vertex is a valid upper bound. AABB pruning retains
        # every triangle which could improve that bound.
        bound = np.linalg.norm(mesh.vertices - p, axis=1).min()
        box_distance = np.linalg.norm(np.maximum(np.maximum(low-p, p-high), 0.), axis=1)
        indices = np.flatnonzero(box_distance <= bound+1.e-12)
        q = trimesh.triangles.closest_point(normalized[indices],
            (p-origins[indices])/scales[indices, None])
        q = q*scales[indices, None]+origins[indices]
        d = np.linalg.norm(q-p, axis=1)
        k = int(np.argmin(d))
        distances.append(float(d[k])); nearest.append(q[k])
    return np.asarray(distances), np.asarray(nearest)


def signed_surface_distance(mesh, points):
    """CPU winding number sign; independent of SDF storage/interpolation."""
    points = np.asarray(points)
    distance, _ = nearest_surface(mesh, points)
    inside = []
    for point in points:
        a, b, c = np.moveaxis(mesh.triangles-point, 1, 0)
        na, nb, nc = (np.linalg.norm(v, axis=1) for v in (a, b, c))
        numerator = np.einsum('ij,ij->i', a, np.cross(b, c))
        denominator = (na*nb*nc + np.einsum('ij,ij->i', a, b)*nc
            + np.einsum('ij,ij->i', b, c)*na + np.einsum('ij,ij->i', c, a)*nb)
        winding = np.arctan2(numerator, denominator).sum()/(2*math.pi)
        inside.append(abs(winding) > .5)
    return distance*np.where(inside, -1., 1.)


def piece_calibration(mesh, index, count, exterior_count):
    """Five actual exterior planes, away from the new internal radial cuts."""
    a, b = 2*math.pi*np.array([index, index+1])/count
    cut_normals = np.array([[-math.sin(a), math.cos(a), 0.],
                            [math.sin(b), -math.cos(b), 0.]])
    selected, normals = [], []
    priorities = np.where(mesh.face_normals[:exterior_count, 2] > .9, 0,
        np.where(mesh.face_normals[:exterior_count, 2] < -.9, 1, 2))
    order = np.lexsort((-mesh.area_faces[:exterior_count], priorities))
    for face in order:
        center, normal = mesh.triangles_center[face], mesh.face_normals[face]
        if (center @ cut_normals.T).min() < .002:
            continue
        if any(np.dot(normal, n) > .98 for n in normals):
            continue
        points = np.array([center-.0001*normal, center+.0001*normal])
        expected = np.array([-.0001, .0001])
        actual = signed_surface_distance(mesh, points)
        if np.max(abs(actual-expected)) > 1.e-10:
            continue
        selected.append(points); normals.append(normal)
        if len(selected) == 5:
            break
    if len(selected) != 5:
        raise ValueError('Five independent known exterior planes are required')
    points = np.concatenate(selected)
    return {'points_body_m': points.tolist(),
        'expected_signed_distance_m': np.tile([-.0001, .0001], 5).tolist(),
        'expected_gradient_body': np.repeat(normals, 2, axis=0).tolist(),
        'minimum_distance_to_cut_plane_m': float(np.min(points @ cut_normals.T)),
        'cpu_signed_distance_m': signed_surface_distance(mesh, points).tolist()}


def partition(mesh, count=8):
    """Construct every sector, checking exact source-facet coverage and integrals."""
    parts, rows = [], []
    areas = np.zeros(len(mesh.faces))
    for index in range(count):
        piece, area, error, exterior = sector(mesh, index, count)
        frame = orientation(piece.vertices)
        calibration = piece_calibration(piece, index, count, exterior)
        rows.append({'index': index, 'exterior_face_count': exterior,
            'cap_face_count': len(piece.faces)-exterior, 'parent_plane_error_m': error,
            'calibration': calibration, **frame})
        parts.append(piece); areas += area
    volume = sum(p.volume for p in parts)
    com = sum(p.volume*p.center_mass for p in parts)/volume
    inertia = np.zeros((3, 3))
    for p in parts:
        d = p.center_mass-mesh.center_mass
        inertia += p.moment_inertia+p.volume*(np.eye(3)*(d@d)-np.outer(d, d))
    checks = {'closed_single_positive_sectors': all(p.is_watertight and p.is_winding_consistent
            and p.euler_number == 2 and p.body_count == 1 and p.volume > 0 for p in parts),
        'source_facet_coverage': bool(np.max(abs(areas-mesh.area_faces)) < 1.e-16),
        'volume': bool(abs(volume-mesh.volume) < 1.e-15),
        'com': bool(np.linalg.norm(com-mesh.center_mass) < 1.e-12),
        'body_inertia': bool(np.max(abs(inertia-mesh.moment_inertia)) < 1.e-15),
        'original_face_planes': max(r['parent_plane_error_m'] for r in rows) < 1.e-12}
    if not all(checks.values()):
        raise ValueError('Exact radial partition proof failed: '+str(checks))
    return parts, {'passed': True, 'checks': checks, 'sectors': rows,
        'volume_error_m3': float(abs(volume-mesh.volume)),
        'com_error_m': float(np.linalg.norm(com-mesh.center_mass)),
        'unit_density_body_inertia_error_m5': float(np.max(abs(inertia-mesh.moment_inertia))),
        'max_source_face_area_partition_error_m2': float(np.max(abs(areas-mesh.area_faces))),
        'scope': 'Exact CPU partition only; cooked geometry, compound contacts and task remain unqualified.'}
