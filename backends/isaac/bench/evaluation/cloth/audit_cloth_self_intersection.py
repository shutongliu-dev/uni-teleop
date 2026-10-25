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


def segment_triangle_hit(p, q, tri, eps=1e-10, return_points=False):
    d = q-p
    e1, e2 = tri[:, 1]-tri[:, 0], tri[:, 2]-tri[:, 0]
    h = np.cross(d, e2)
    a = np.einsum('ij,ij->i', e1, h)
    det_scale=np.linalg.norm(e1,axis=1)*np.linalg.norm(e2,axis=1)*np.linalg.norm(d,axis=1)
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
    axes /= np.maximum(np.linalg.norm(axes,axis=1)[:,None],1e-30)
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
        edges=np.concatenate((np.roll(aa,-1,axis=1)-aa,np.roll(bb,-1,axis=1)-bb),axis=1)
        axes=np.stack((-edges[:,:,1],edges[:,:,0]),axis=2)
        axes/=np.maximum(np.linalg.norm(axes,axis=2)[:,:,None],1e-30)
        pa,pb=np.einsum('nvi,nai->nva',aa,axes),np.einsum('nvi,nai->nva',bb,axes)
        overlap=np.minimum(pa.max(1),pb.max(1))-np.maximum(pa.min(1),pb.min(1))
        result[ids]=np.where(strict[ids],(overlap>1e-9).all(1),(overlap>=-1e-12).all(1))
    return result


def self_intersections(vertices, triangles):
    xyz = vertices[triangles]
    centers = xyz.mean(1)
    radius = np.linalg.norm(xyz-centers[:, None], axis=2).max(1)
    pairs = cKDTree(centers).query_pairs(2*radius.max()+1e-10, output_type='ndarray')
    if not len(pairs):
        return [], 0
    pairs = pairs[np.linalg.norm(centers[pairs[:, 0]]-centers[pairs[:, 1]], axis=1) <= radius[pairs].sum(1)+1e-10]
    a, b = xyz[pairs[:, 0]], xyz[pairs[:, 1]]
    keep = ((a.min(1) <= b.max(1)+1e-10) & (b.min(1) <= a.max(1)+1e-10)).all(1)
    pairs, a, b = pairs[keep], a[keep], b[keep]
    ta,tb=triangles[pairs[:,0]],triangles[pairs[:,1]]
    shared_a=(ta[:,:,None]==tb[:,None,:]).any(2)
    shared_count=shared_a.sum(1)
    shared_first=np.argmax(shared_a,axis=1)
    shared_last=2-np.argmax(shared_a[:,::-1],axis=1)
    start=a[np.arange(len(a)),shared_first]
    end=a[np.arange(len(a)),shared_last]
    segment=end-start
    segment_sq=np.einsum('ij,ij->i',segment,segment)
    def away_from_structural_boundary(points):
        rel=points-start
        along=np.divide(np.einsum('ij,ij->i',rel,segment),segment_sq,out=np.zeros(len(a)),where=segment_sq>1e-20)
        closest=start+np.clip(along,0.,1.)[:,None]*segment
        return (shared_count==0) | (np.linalg.norm(points-closest,axis=1)>1e-7)
    hit = np.zeros(len(pairs), bool)
    for j in range(3):
        cross,points=segment_triangle_hit(a[:, j],a[:, (j+1)%3],b,return_points=True)
        hit |= cross & away_from_structural_boundary(points)
        cross,points=segment_triangle_hit(b[:, j],b[:, (j+1)%3],a,return_points=True)
        hit |= cross & away_from_structural_boundary(points)
    na = np.cross(a[:, 1]-a[:, 0], a[:, 2]-a[:, 0])
    nb = np.cross(b[:, 1]-b[:, 0], b[:, 2]-b[:, 0])
    la, lb = np.linalg.norm(na, axis=1), np.linalg.norm(nb, axis=1)
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
    ref_area = np.linalg.norm(np.cross(ref_xyz[:, 1]-ref_xyz[:, 0], ref_xyz[:, 2]-ref_xyz[:, 0]), axis=1)
    ref_len = np.linalg.norm(vertices[0, edges[:, 0]]-vertices[0, edges[:, 1]], axis=1)
    rows=[]
    for i, verts in enumerate(vertices):
        xyz=verts[triangles]
        n=np.cross(xyz[:, 1]-xyz[:, 0], xyz[:, 2]-xyz[:, 0])
        area=np.linalg.norm(n, axis=1)
        n=n/np.maximum(area[:, None], 1e-30)
        cosine=np.einsum('ij,ij->i', n[adjacent[:, 0]], n[adjacent[:, 1]])
        length=np.linalg.norm(verts[edges[:, 0]]-verts[edges[:, 1]], axis=1)
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
