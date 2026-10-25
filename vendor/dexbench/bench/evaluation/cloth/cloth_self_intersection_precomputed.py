"""Read-only recorded cloth topology, deformation, and self-intersection audit.

Triangle candidates use bounding spheres then exact AABBs. Segment/
triangle intersection handles non-coplanar pairs; projected separating axes cover
coplanar pairs. Structural shared vertices/edges are allowed, but intersections
away from those vertices/edges are detected, including for neighboring faces.
This is a sampled-frame audit, not a continuous collision certificate.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree


def vector_norm(values, axis):
    if axis not in (-1, values.ndim-1) or values.shape[-1] not in (2, 3):
        return np.linalg.norm(values, axis=axis)
    squared = values * values
    total = squared[..., 0] + squared[..., 1]
    if values.shape[-1] == 3:
        total = total + squared[..., 2]
    return np.sqrt(total)


def segment_triangle_hit(p, q, tri, eps=1e-10, return_points=False, prepared=None):
    d = q-p
    if prepared is None:
        e1, e2 = tri[:, 1]-tri[:, 0], tri[:, 2]-tri[:, 0]
        edge_scale = vector_norm(e1,axis=1)*vector_norm(e2,axis=1)
    else:
        e1, e2, edge_scale = prepared
    h = np.cross(d, e2)
    a = np.einsum('ij,ij->i', e1, h)
    det_scale=edge_scale*vector_norm(d,axis=1)
    valid = np.abs(a) > np.maximum(det_scale*1e-12,1e-30)
    f = np.divide(1., a, out=np.zeros_like(a), where=valid)
    s = p-tri[:, 0]
    u = f*np.einsum('ij,ij->i', s, h)
    z = np.cross(s, e1)
    v = f*np.einsum('ij,ij->i', d, z)
    t = f*np.einsum('ij,ij->i', e2, z)
    hit = valid & (u >= -eps) & (v >= -eps) & (u+v <= 1+eps) & (t >= -eps) & (t <= 1+eps)
    return (hit, p+t[:,None]*d) if return_points else hit


def coplanar_overlap(a, b, normal, strict=False):
    drop = int(np.argmax(np.abs(normal)))
    aa, bb = np.delete(a, drop, axis=1), np.delete(b, drop, axis=1)
    edges = np.concatenate((np.roll(aa, -1, axis=0)-aa, np.roll(bb, -1, axis=0)-bb))
    axes = np.stack((-edges[:, 1], edges[:, 0]), axis=1)
    axes /= np.maximum(vector_norm(axes,axis=1)[:,None],1e-30)
    pa, pb = aa@axes.T, bb@axes.T
    if strict:
        return bool(np.all(np.minimum(pa.max(0),pb.max(0))-np.maximum(pa.min(0),pb.min(0)) > 1e-9))
    # Exact touching is counted as potential intersection for non-neighbors.
    return not np.any((pa.max(0) < pb.min(0)-1e-12) | (pb.max(0) < pa.min(0)-1e-12))


def coplanar_overlap_batch(a,b,normal,strict):
    result=np.zeros(len(a),dtype=bool)
    drop=np.argmax(np.abs(normal),axis=1)
    for axis in range(3):
        ids=np.flatnonzero(drop==axis)
        if not len(ids):continue
        aa,bb=np.delete(a[ids],axis,axis=2),np.delete(b[ids],axis,axis=2)
        # SAT is a conjunction of six edge axes. Evaluate A's three first;
        # only surviving pairs need B's axes. Every retained projection uses
        # the original elementwise arithmetic and tolerances, in original order.
        # Nonfinite inputs or non-default error handling take the original path.
        limit = np.sqrt(np.finfo(aa.dtype).max)/4 if aa.dtype in (np.float32,np.float64) else 0.
        full = (aa.dtype not in (np.float32,np.float64)
                or aa.dtype != bb.dtype
                or np.geterr() != dict(divide='warn', over='warn', under='ignore', invalid='warn')
                or not ((np.abs(aa)<=limit).all() and (np.abs(bb)<=limit).all()))
        edges=np.roll(aa,-1,axis=1)-aa
        if full:
            edges=np.concatenate((edges,np.roll(bb,-1,axis=1)-bb),axis=1)
        axes=np.stack((-edges[:,:,1],edges[:,:,0]),axis=2)
        axes/=np.maximum(vector_norm(axes,axis=2)[:,:,None],1e-30)
        pa=aa[:,:,0,None]*axes[:,None,:,0]+aa[:,:,1,None]*axes[:,None,:,1]
        pb=bb[:,:,0,None]*axes[:,None,:,0]+bb[:,:,1,None]*axes[:,None,:,1]
        overlap=np.minimum(triangle_max(pa),triangle_max(pb))-np.maximum(triangle_min(pa),triangle_min(pb))
        keep=np.where(strict[ids],(overlap>1e-9).all(1),(overlap>=-1e-12).all(1))
        if not full:
            pending=np.flatnonzero(keep)
            if len(pending):
                aaa,bbb=aa[pending],bb[pending]
                edges=np.roll(bbb,-1,axis=1)-bbb
                axes=np.stack((-edges[:,:,1],edges[:,:,0]),axis=2)
                axes/=np.maximum(vector_norm(axes,axis=2)[:,:,None],1e-30)
                pa=aaa[:,:,0,None]*axes[:,None,:,0]+aaa[:,:,1,None]*axes[:,None,:,1]
                pb=bbb[:,:,0,None]*axes[:,None,:,0]+bbb[:,:,1,None]*axes[:,None,:,1]
                overlap=np.minimum(triangle_max(pa),triangle_max(pb))-np.maximum(triangle_min(pa),triangle_min(pb))
                keep[pending]=np.where(strict[ids[pending]],(overlap>1e-9).all(1),(overlap>=-1e-12).all(1))
        result[ids]=keep
    return result


# Frozen topology metadata only. Geometry and the broadphase are recomputed for
# every call; sharing a vertex never causes a candidate triangle pair to be skipped.
class StaticTriangleTopology:
    def __init__(self, triangles):
        value = np.asarray(triangles)
        if value.ndim != 2 or value.shape[1] != 3 or not np.issubdtype(value.dtype, np.integer):
            raise ValueError('Expected integer triangle topology')
        self.triangles = value.copy()
        self.triangles.setflags(write=False)
        self.n = len(value)
        incident = {}
        for face, tri in enumerate(value):
            for vertex in set(tri.tolist()):
                incident.setdefault(vertex, []).append(face)
        masks = {}
        for face, tri in enumerate(value):
            for local_index, vertex in enumerate(tri):
                for other in incident[vertex]:
                    if other > face:
                        key = face * self.n + other
                        masks[key] = masks.get(key, 0) | (1 << local_index)
        self.keys = np.array(sorted(masks), dtype=np.int64)
        self.masks = np.array([masks[key] for key in self.keys], dtype=np.uint8)
        self.keys.setflags(write=False)
        self.masks.setflags(write=False)

    def shared_boundary(self, triangles, pairs):
        # Fail closed if caller topology changed, including in-place mutation.
        if not np.array_equal(triangles, self.triangles):
            raise ValueError('Cached triangle topology changed')
        keys = pairs[:, 0] * self.n + pairs[:, 1]
        mask = np.zeros(len(pairs), dtype=np.uint8)
        if len(self.keys):
            indices = np.searchsorted(self.keys, keys)
            valid = indices < len(self.keys)
            safe = np.minimum(indices, len(self.keys)-1)
            valid &= self.keys[safe] == keys
            mask[valid] = self.masks[indices[valid]]
        # These tables also reproduce argmax behavior for the no-shared case:
        # first=0 and last=2, whose distance is ignored by shared_count == 0.
        counts = np.array([0, 1, 1, 2, 1, 2, 2, 3], dtype=np.uint8)
        first = np.array([0, 0, 1, 0, 2, 0, 1, 0], dtype=np.uint8)
        last = np.array([2, 0, 1, 1, 2, 2, 2, 2], dtype=np.uint8)
        return counts[mask], first[mask], last[mask]


def triangle_min(values):
    """Exactly three vertices, same left-to-right reduction and NaN propagation."""
    return np.minimum(np.minimum(values[:, 0], values[:, 1]), values[:, 2])


def triangle_max(values):
    return np.maximum(np.maximum(values[:, 0], values[:, 1]), values[:, 2])


def self_intersections(vertices, triangles, *, topology=None):
    xyz = vertices[triangles]
    centers = xyz.mean(1)
    radius = triangle_max(vector_norm(xyz-centers[:, None], axis=2))
    if topology is not None and not np.array_equal(triangles, topology.triangles):
        raise ValueError('Cached triangle topology changed')
    xyz_min, xyz_max = triangle_min(xyz), triangle_max(xyz)
    pairs = cKDTree(centers).query_pairs(2*radius.max()+1e-10, output_type='ndarray')
    if not len(pairs):
        return [], 0
    pairs = pairs[vector_norm(centers[pairs[:, 0]]-centers[pairs[:, 1]], axis=1) <= radius[pairs].sum(1)+1e-10]
    a, b = xyz[pairs[:, 0]], xyz[pairs[:, 1]]
    keep = ((xyz_min[pairs[:, 0]] <= xyz_max[pairs[:, 1]]+1e-10) & (xyz_min[pairs[:, 1]] <= xyz_max[pairs[:, 0]]+1e-10)).all(1)
    pairs, a, b = pairs[keep], a[keep], b[keep]
    if topology is None:
        ta,tb=triangles[pairs[:,0]],triangles[pairs[:,1]]
        shared_a=(ta[:,:,None]==tb[:,None,:]).any(2)
        shared_count=shared_a.sum(1)
        shared_first=np.argmax(shared_a,axis=1)
        shared_last=2-np.argmax(shared_a[:,::-1],axis=1)
    else:
        shared_count, shared_first, shared_last = topology.shared_boundary(triangles, pairs)
    hit = np.zeros(len(pairs), bool)
    # A strict common coordinate makes all six segment determinants exactly zero.
    # This requires the entire vertex array to share one exact coordinate;
    # mixed/nonplanar batches use the original
    # segment code unchanged. Keep every pair, including structural neighbors,
    # for the original normals/coplanar/SAT logic below. No tolerance is involved.
    axis_flat = (vertices == vertices[0]).all(axis=0).any()
    # Use original arithmetic when NumPy's error mode explicitly requests raise.
    skip_segments = (vertices.dtype == np.float64 and bool(axis_flat) and
                     'raise' not in np.geterr().values())
    if not skip_segments:
        start=a[np.arange(len(a)),shared_first]
        end=a[np.arange(len(a)),shared_last]
        segment=end-start
        segment_sq=np.einsum('ij,ij->i',segment,segment)
        def away_from_structural_boundary(points):
            rel=points-start
            along=np.divide(np.einsum('ij,ij->i',rel,segment),segment_sq,out=np.zeros(len(a)),where=segment_sq>1e-20)
            closest=start+np.clip(along,0.,1.)[:,None]*segment
            return (shared_count==0) | (vector_norm(points-closest,axis=1)>1e-7)
        def prepare(tri):
            e1, e2 = tri[:, 1]-tri[:, 0], tri[:, 2]-tri[:, 0]
            return e1, e2, vector_norm(e1,axis=1)*vector_norm(e2,axis=1)
        prepared_a, prepared_b = prepare(a), prepare(b)
        for j in range(3):
            cross,points=segment_triangle_hit(a[:, j],a[:, (j+1)%3],b,return_points=True,prepared=prepared_b)
            hit |= cross & away_from_structural_boundary(points)
            cross,points=segment_triangle_hit(b[:, j],b[:, (j+1)%3],a,return_points=True,prepared=prepared_a)
            hit |= cross & away_from_structural_boundary(points)
    na = np.cross(a[:, 1]-a[:, 0], a[:, 2]-a[:, 0])
    nb = np.cross(b[:, 1]-b[:, 0], b[:, 2]-b[:, 0])
    la, lb = vector_norm(na, axis=1), vector_norm(nb, axis=1)
    nonzero = (la > 1e-14) & (lb > 1e-14)
    dot = np.zeros(len(pairs))
    dot[nonzero] = np.abs(np.einsum('ij,ij->i', na[nonzero], nb[nonzero]) / (la[nonzero]*lb[nonzero]))
    plane_dist = np.zeros(len(pairs))
    plane_dist[nonzero] = np.abs(np.einsum('ij,ij->i', b[nonzero, 0]-a[nonzero, 0], na[nonzero]))/la[nonzero]
    coplanar = nonzero & (dot > 1-1e-10) & (plane_dist < 1e-9)
    ids=np.flatnonzero(coplanar & ~hit)
    hit[ids]=coplanar_overlap_batch(a[ids],b[ids],na[ids],shared_count[ids]>0)
    return pairs[hit].tolist(), len(pairs)


def audit(states_path, triangles_path):
    states = np.load(states_path)
    vertices = states['particle_q'].astype(np.float64)
    triangles = np.load(triangles_path).astype(np.int64)
    edges = np.sort(np.concatenate((triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]])), axis=1)
    edges, edge_index, edge_count = np.unique(edges, axis=0, return_inverse=True, return_counts=True)
    edge_faces = [[] for _ in edges]
    for k, e in enumerate(edge_index):
        edge_faces[e].append(k % len(triangles))
    adjacent = np.array([f for f in edge_faces if len(f) == 2])
    ref_xyz = vertices[0, triangles]
    ref_area = vector_norm(np.cross(ref_xyz[:, 1]-ref_xyz[:, 0], ref_xyz[:, 2]-ref_xyz[:, 0]), axis=1)
    ref_len = vector_norm(vertices[0, edges[:, 0]]-vertices[0, edges[:, 1]], axis=1)
    rows=[]
    for i, verts in enumerate(vertices):
        xyz=verts[triangles]
        n=np.cross(xyz[:, 1]-xyz[:, 0], xyz[:, 2]-xyz[:, 0])
        area=vector_norm(n, axis=1)
        n=n/np.maximum(area[:, None], 1e-30)
        cosine=np.einsum('ij,ij->i', n[adjacent[:, 0]], n[adjacent[:, 1]])
        length=vector_norm(verts[edges[:, 0]]-verts[edges[:, 1]], axis=1)
        hits, candidates=self_intersections(verts, triangles)
        rows.append({'frame':i, 'time_s':float(states['time_s'][i]),
                     'intersection_pairs':hits,'candidate_pairs':candidates,
                     'area_ratio_min':float((area/ref_area).min()),
                     'area_ratio_max':float((area/ref_area).max()),
                     'edge_ratio_min':float((length/ref_len).min()),
                     'edge_ratio_max':float((length/ref_len).max()),
                     'minimum_adjacent_normal_dot':float(cosine.min()),
                     'adjacent_fold_over_90_deg_count':int((cosine<0).sum()),
                     'degenerate_triangle_count':int((area<1e-12).sum())})
    return {'schema_version':'recorded-cloth-self-intersection-v1',
            'states_path':str(states_path),'triangles_path':str(triangles_path),
            'method':'Sphere/AABB broadphase; segment-triangle crossings plus coplanar 2D separating axes. Includes shared-vertex/edge pairs; only intersections within 1e-7 m of their structural common boundary are ignored. Coplanar structural neighbors require positive interior overlap.',
            'limitations':['Recorded frames only, no continuous-between-frame guarantee.',
                           'Does not certify self-contact thickness.',
                           'Numerical boundary tolerance 1e-7 m; degeneracy and adjacent dihedral separately reported.'],
            'frames':len(rows),'vertices':int(vertices.shape[1]),'triangles':len(triangles),
            'boundary_edges':int((edge_count==1).sum()),'nonmanifold_edges':int((edge_count>2).sum()),
            'frames_with_intersection':sum(bool(r['intersection_pairs']) for r in rows),
            'intersection_pairs_total':sum(len(r['intersection_pairs']) for r in rows),
            'degenerate_triangle_count_total':sum(r['degenerate_triangle_count'] for r in rows),
            'minimum_area_ratio':min(r['area_ratio_min'] for r in rows),
            'maximum_edge_ratio':max(r['edge_ratio_max'] for r in rows),
            'rows':rows}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--states',type=Path,required=True)
    p.add_argument('--triangles',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    result=audit(args.states,args.triangles)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='rows'},indent=2))


if __name__=='__main__':
    main()
