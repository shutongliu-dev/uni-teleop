"""Offline full-cloth-surface audit against a fixed finite cylinder and optional table.

Imports NumPy only, never a simulator. Barycentric lattice samples include all
vertices and, for even n, every edge midpoint, plus explicit face centroids.
A conservative 1-Lipschitz SDF certificate covers the unsampled triangle area:
min(surface SDF) >= min(sample SDF) - max_triangle_edge / subdivisions.
The bound uses the diameter of each lattice subtriangle as a covering radius.
A negative bound with no negative samples is unresolved, not penetration.
"""
from pathlib import Path
from collections import Counter
import argparse
import hashlib
import json
import numpy as np

from bench.paths import load_json


def cylinder_sdf(points, center, axis, radius, half_length):
    rel = points - center
    axial = np.einsum('...i,i->...', rel, axis)
    radial = np.linalg.norm(rel - axial[..., None] * axis, axis=-1)
    d = np.stack((radial - radius, np.abs(axial) - half_length), axis=-1)
    return np.minimum(np.max(d, axis=-1), 0) + np.linalg.norm(np.maximum(d, 0), axis=-1)


def box_sdf(points, center, half):
    d = np.abs(points - center) - half
    return np.minimum(np.max(d, axis=-1), 0) + np.linalg.norm(np.maximum(d, 0), axis=-1)


def grid_triangles(cells):
    nx, ny = cells
    tris = []
    for y in range(ny):
        for x in range(nx):
            a = y * (nx + 1) + x
            b, c, d = a + 1, a + nx + 2, a + nx + 1
            tris.extend(((a, b, d), (b, c, d)))
    return np.asarray(tris, dtype=int)


def barycentric_lattice(n):
    lattice = [[i/n, j/n, (n-i-j)/n] for i in range(n+1) for j in range(n-i+1)]
    return np.unique(np.concatenate((np.asarray(lattice), [[1/3, 1/3, 1/3]])), axis=0)


def summarize_surface(vertices, tris, sdf, subdivisions):
    tri = vertices[tris]
    bary = barycentric_lattice(subdivisions)
    xyz = np.einsum('sv,tvc->tsc', bary, tri)
    distances = sdf(xyz)
    triangle_sample_min = distances.min(axis=1)
    diameters = np.max(np.linalg.norm(tri[:, [1, 2, 0]] - tri, axis=-1), axis=1)
    lower_bound = triangle_sample_min - diameters / subdivisions
    worst_tri, worst_sample = np.unravel_index(int(np.argmin(distances)), distances.shape)
    return {
        'minimum_sampled_surface_sdf_m': float(distances.min()),
        'minimum_vertex_sdf_m': float(sdf(vertices).min()),
        'minimum_certified_surface_sdf_lower_bound_m': float(lower_bound.min()),
        'triangle_count_with_sampled_penetration': int((triangle_sample_min < -1e-8).sum()),
        'triangle_count_with_certified_nonpenetration': int((lower_bound >= 0).sum()),
        'triangle_count_unresolved_by_sampling_bound': int(((triangle_sample_min >= -1e-8) & (lower_bound < 0)).sum()),
        'all_triangles_certified_outside': bool((lower_bound >= 0).all()),
        'worst_triangle_id': int(worst_tri),
        'worst_sample_barycentric': bary[worst_sample].tolist(),
        'worst_sample_position_m': xyz[worst_tri, worst_sample].tolist(),
    }


def topology_report(tris, vertex_count):
    counter = Counter(tuple(sorted((int(a), int(b)))) for t in tris for a,b in zip(t, np.roll(t, -1)))
    edges = np.asarray(list(counter), int)
    adj = [set() for _ in range(vertex_count)]
    for a,b in edges:
        adj[a].add(b); adj[b].add(a)
    unseen = set(range(vertex_count)); components=[]
    while unseen:
        stack=[unseen.pop()]; size=0
        while stack:
            a=stack.pop();size+=1
            neighbors=adj[a]&unseen;unseen-=neighbors;stack.extend(neighbors)
        components.append(size)
    return {'vertices': vertex_count, 'triangles': len(tris), 'unique_edges': len(edges),
            'boundary_edges': sum(v==1 for v in counter.values()),
            'nonmanifold_edges': sum(v>2 for v in counter.values()),
            'connected_component_sizes': components}, edges


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--states', type=Path, required=True, help='NPZ with particle_q [T,N,3] and time_s [T]')
    parser.add_argument('--scene', type=Path, help='Optional scene JSON supplies cloth.cells_xy and table box')
    parser.add_argument('--observations', type=Path, help='Optional JSONL supplies triangles and verifies constant topology')
    parser.add_argument('--cells', type=int, nargs=2, default=[40,44])
    parser.add_argument('--center', type=float, nargs=3, help='Override scene cylinder center')
    parser.add_argument('--axis-vector', type=float, nargs=3, help='Override scene cylinder axis')
    parser.add_argument('--radius', type=float, help='Override scene cylinder radius')
    parser.add_argument('--half-length', type=float, help='Override scene cylinder half length')
    parser.add_argument('--subdivisions', type=int, default=12)
    parser.add_argument('--frame-stride', type=int, default=1)
    parser.add_argument('--output', type=Path, required=True)
    args=parser.parse_args()
    scene=load_json(args.scene) if args.scene else {}
    cylinder=scene.get('cloth_support_cylinder',{})
    cylinder_source='scene.cloth_support_cylinder' if cylinder else 'CLI/default geometry'
    cylinder_overrides={name:getattr(args,name) for name in ('center','axis_vector','radius','half_length') if getattr(args,name) is not None}
    args.center=args.center if args.center is not None else cylinder.get('center_m',[.53,.10,.85])
    args.radius=args.radius if args.radius is not None else float(cylinder.get('radius_m',.08))
    args.half_length=args.half_length if args.half_length is not None else float(cylinder.get('half_length_m',.30))
    if args.axis_vector is None:
        axis_name=cylinder.get('axis','x')
        if axis_name not in ('x','y','z'):parser.error('scene cylinder axis must be x, y, or z')
        args.axis_vector=np.eye(3)[('x','y','z').index(axis_name)].tolist()
    if args.radius<=0 or args.half_length<=0 or args.subdivisions<1 or args.frame_stride<1:
        parser.error('radius, half-length, subdivisions, frame-stride must be positive')
    axis=np.asarray(args.axis_vector, float)
    if not np.isfinite(axis).all() or np.linalg.norm(axis)==0:parser.error('axis-vector must be finite and nonzero')
    axis/=np.linalg.norm(axis)
    center=np.asarray(args.center, float)
    if not np.isfinite(center).all():parser.error('center must be finite')
    states=np.load(args.states)
    q=states['particle_q'].astype(float)
    times=states['time_s'] if 'time_s' in states else np.arange(len(q),dtype=float)
    if q.ndim!=3 or q.shape[-1]!=3 or not np.isfinite(q).all():parser.error('particle_q must be finite [T,N,3]')
    cells=scene.get('cloth',{}).get('cells_xy',args.cells)
    tris=grid_triangles(cells)
    topology_constant=None
    if args.observations:
        first=None;topology_constant=True;count=0
        with args.observations.open() as f:
            for line in f:
                ob=json.loads(line);current=ob['cloth']['triangles'];count+=1
                if first is None:first=current
                topology_constant &= current==first
        tris=np.asarray(first,int)
        if count!=len(q):parser.error('Observation count differs from states')
    if len(q[0]) != (cells[0]+1)*(cells[1]+1) and not args.observations:
        parser.error('Grid cells do not match recorded particle count; provide correct --cells/--scene or --observations')
    if tris.min()<0 or tris.max()>=len(q[0]):parser.error('Triangle indices outside recorded particles')
    topology,edges=topology_report(tris,len(q[0]));topology['constant_across_observations']=topology_constant
    rest_lengths=np.linalg.norm(q[0,edges[:,1]]-q[0,edges[:,0]],axis=-1)
    if (rest_lengths<=0).any():parser.error('Initial edge length is zero')
    rest_tri=q[0,tris]
    rest_areas=np.linalg.norm(np.cross(rest_tri[:,1]-rest_tri[:,0],rest_tri[:,2]-rest_tri[:,0]),axis=-1)/2
    if (rest_areas<=0).any():parser.error('Initial triangle area is zero')
    functions={'cylinder':lambda p:cylinder_sdf(p,center,axis,args.radius,args.half_length)}
    table=scene.get('table')
    if table:
        half=np.asarray(table.get('half_extents_m',np.asarray(table.get('size_m',[1.,.9,.05]))/2),float)
        c=np.asarray(table.get('position_m',[.5,0,float(table.get('height_m',0))-half[2]]),float)
        functions['table']=lambda p:box_sdf(p,c,half)
    indices=sorted(set(range(0,len(q),args.frame_stride))|{len(q)-1})
    frames=[]
    for fi in indices:
        tri=q[fi,tris]
        ratios=np.linalg.norm(q[fi,edges[:,1]]-q[fi,edges[:,0]],axis=-1)/rest_lengths
        area=np.linalg.norm(np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]),axis=-1)/2
        frames.append({'index':fi,'time_s':float(times[fi]),
                       'edge_length_ratio_min':float(ratios.min()),'edge_length_ratio_max':float(ratios.max()),
                       'area_ratio_min':float((area/rest_areas).min()),'area_ratio_max':float((area/rest_areas).max()),
                       'surfaces':{name:summarize_surface(q[fi],tris,fn,args.subdivisions) for name,fn in functions.items()}})
    summary={}
    for name in functions:
        selected=[f['surfaces'][name] for f in frames]
        summary[name]={
            'minimum_sampled_surface_sdf_m':min(f['minimum_sampled_surface_sdf_m'] for f in selected),
            'minimum_vertex_sdf_m':min(f['minimum_vertex_sdf_m'] for f in selected),
            'minimum_certified_surface_sdf_lower_bound_m':min(f['minimum_certified_surface_sdf_lower_bound_m'] for f in selected),
            'frames_with_sampled_surface_penetration':sum(f['triangle_count_with_sampled_penetration']>0 for f in selected),
            'maximum_triangles_with_sampled_penetration':max(f['triangle_count_with_sampled_penetration'] for f in selected),
            'frames_certified_without_surface_penetration':sum(f['all_triangles_certified_outside'] for f in selected),
            'all_audited_surfaces_certified_outside':all(f['all_triangles_certified_outside'] for f in selected),
        }
    provenance={str(p.resolve()):hashlib.sha256(p.read_bytes()).hexdigest() for p in [args.states,args.scene,args.observations,Path(__file__)] if p}
    report={'schema_version':'offline_cloth_cylinder_surface_audit_v1','method':__doc__,
            'input_sha256':provenance,'cylinder':{'center_m':center.tolist(),'axis':axis.tolist(),'radius_m':args.radius,'half_length_m':args.half_length,'static':True,'configuration_source':cylinder_source,'explicit_cli_overrides':cylinder_overrides},
            'subdivisions':args.subdivisions,'samples_per_triangle':len(barycentric_lattice(args.subdivisions)),
            'recorded_frame_count':len(q),'audited_frame_count':len(frames),'frame_stride':args.frame_stride,
            'topology':topology,'summary':summary,'deformation':{
                'minimum_edge_length_ratio':min(f['edge_length_ratio_min'] for f in frames),
                'maximum_edge_length_ratio':max(f['edge_length_ratio_max'] for f in frames),
                'minimum_area_ratio':min(f['area_ratio_min'] for f in frames),
                'maximum_area_ratio':max(f['area_ratio_max'] for f in frames)},'frames':frames,
            'limitations':['Only recorded timestamps are audited; no certification between frames.','Fixed analytic capped cylinder and table box must match simulation/visual geometry.','No self-intersection, arm/hand, or noncolliding furniture check is included.','Sampled penetration is a lower bound on the true deepest penetration; a nonnegative Lipschitz lower bound certifies the full triangle.','Initial edges/areas define reported deformation reference; they must be undeformed rest geometry for material strain.']}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2))
    print(json.dumps({'output':str(args.output),'audited_frames':len(frames),'summary':summary,'deformation':report['deformation']},indent=2))


if __name__=='__main__':main()
