"""Conservative true-shape clearance certificates after an AABB broad phase.

A positive support-plane gap between two convex outer hulls lower-bounds their
shape distance. Directions may be proposed numerically, but acceptance always
uses explicit support extrema and the unchanged distance gate minus a guard.
"""
import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation
from scipy.optimize import minimize,LinearConstraint


def projection_gap(moving, obstacle, normal):
    normal=np.asarray(normal,float);normal=normal/np.linalg.norm(normal)
    return float((moving@normal).min()-(obstacle@normal).max()),normal


def direction_hint(moving, obstacle):
    a,b=ConvexHull(moving).equations,ConvexHull(obstacle).equations
    matrix=np.block([[a[:,:3],np.zeros((len(a),3))],[np.zeros((len(b),3)),b[:,:3]]]);bound=np.r_[-a[:,3],-b[:,3]]
    def fun(z):d=z[:3]-z[3:];return .5*float(d@d)
    def jac(z):d=z[:3]-z[3:];return np.r_[d,-d]
    answer=minimize(fun,np.r_[moving.mean(0),obstacle.mean(0)],jac=jac,
        constraints=[LinearConstraint(matrix,-np.inf,bound)],method='SLSQP',options={'maxiter':300,'ftol':1e-16})
    d=answer.x[:3]-answer.x[3:]
    return d/np.linalg.norm(d) if np.linalg.norm(d)>1e-12 else None


def certify_motion(clouds, obstacles, *, threshold=.005, guard=1e-6, prefilter=.01):
    if not 0<guard<threshold<prefilter:raise ValueError('Positive ordered guard, distance gate and broad phase required')
    if not clouds or not obstacles:raise ValueError('Complete body and non-support obstacle inventory required')
    count=len(clouds[0]);certs=[];failures=[];min_bound=float('inf');queries=0;tested=0
    if any(x.ndim!=3 or x.shape[0]!=count or x.shape[2]!=3 or not np.isfinite(x).all() for x in clouds):raise ValueError('Complete finite native world collision vertices required')
    for oi,points in enumerate(obstacles):
        points=np.asarray(points,float);hull=ConvexHull(points);points=points[hull.vertices]
        normals=np.unique(np.round(ConvexHull(points).equations[:,:3],12),axis=0);normals/=np.linalg.norm(normals,axis=1)[:,None]
        low,high=points.min(0),points.max(0)
        for bi,world in enumerate(clouds):
            lo,hi=world.min(1),world.max(1);vector=np.maximum(low-hi,0)-np.maximum(lo-high,0)
            gap=np.linalg.norm(vector,axis=1);near=np.flatnonzero(gap<prefilter)
            if len(near)<count:min_bound=min(min_bound,float(gap[gap>=prefilter].min())-guard)
            for frame in near:
                vertices=world[frame];values=(vertices@normals.T).min(0)-(points@normals.T).max(0);i=int(values.argmax());value=float(values[i]);normal=normals[i]
                # An AABB separation vector is another valid support-plane
                # candidate; always verify it on actual vertices as well.
                if gap[frame]>0:
                    candidate=-vector[frame]/gap[frame];v,n=projection_gap(vertices,points,candidate)
                    if v>value:value,normal=v,n
                if value-guard<threshold:
                    queries+=1;candidate=direction_hint(vertices,points)
                    if candidate is not None:
                        v,n=projection_gap(vertices,points,candidate);normals=np.vstack([normals,n])
                        if v>value:value,normal=v,n
                certified=value-guard;min_bound=min(min_bound,certified);tested+=1
                certs.append((int(frame),bi,oi,value,*normal.tolist()))
                if certified<threshold:failures.append({'frame':int(frame),'body_index':bi,'obstacle_index':oi,'certified_lower_bound_m':certified})
    report={'passed':not failures,'frames':count,'body_count':len(clouds),'obstacle_count':len(obstacles),
        'threshold_m':threshold,'numerical_guard_m':guard,'aabb_prefilter_m':prefilter,
        'global_certified_lower_bound_m':min_bound,'support_plane_certificates':tested,'direction_hint_queries':queries,
        'failed_certificates':failures,'method':'Actual collision mesh vertices; convex outer-hull support-plane separation. Every candidate direction is checked by min(n·moving)-max(n·obstacle)-guard >= unchanged threshold. Far pairs use AABB lower bounds minus guard. QP is only a direction hint, never an acceptance oracle.'}
    return report,{'certificates':np.asarray(certs,float).reshape(-1,7)}


def certify_stage(stage,specs,poses,obstacle_rows,*,excluded_paths=(),threshold=.005):
    from pxr import UsdGeom,UsdPhysics,Gf
    cache=UsdGeom.XformCache();clouds=[];obstacles=[];obstacle_paths=[]
    for j,spec in enumerate(specs):
        local=[];body=stage.GetPrimAtPath(spec['body_path'])
        for path in spec['collider_paths']:
            prim=stage.GetPrimAtPath(path)
            if not prim.HasAPI(UsdPhysics.CollisionAPI) or prim.GetAttribute('physics:approximation').Get()!='convexHull':raise ValueError('Native member convex collision meshes required')
            matrix,reset=cache.ComputeRelativeTransform(prim,body)
            if reset:raise ValueError('Unexpected collider transform reset')
            local.extend(matrix.Transform(Gf.Vec3d(*map(float,p))) for p in UsdGeom.Mesh(prim).GetPointsAttr().Get())
        local=np.asarray(local,float);q=np.asarray(poses[:,j],float)
        world=Rotation.from_quat(np.repeat(q[:,3:],len(local),axis=0)).apply(np.tile(local,(len(q),1))).reshape(len(q),len(local),3)+q[:,None,:3]
        clouds.append(world)
    for row in obstacle_rows:
        path=row['path']
        if path in excluded_paths:continue
        prim=stage.GetPrimAtPath(path)
        if not prim.HasAPI(UsdPhysics.CollisionAPI):raise ValueError('Expected actual static collision primitive')
        if not prim.IsA(UsdGeom.Mesh) or prim.GetAttribute('physics:approximation').Get() not in ('none','convexHull'):
            raise ValueError('Exact triangle or convex static mesh required; unsupported approximations fail closed')
        matrix=cache.GetLocalToWorldTransform(prim);points=np.array([matrix.Transform(Gf.Vec3d(*map(float,p))) for p in UsdGeom.Mesh(prim).GetPointsAttr().Get()])
        obstacles.append(points);obstacle_paths.append(path)
    result,arrays=certify_motion(clouds,obstacles,threshold=threshold)
    result.update(body_paths=[s['body_path'] for s in specs],obstacle_paths=obstacle_paths,
        static_mesh_scope='Outer convex hull encloses each original static triangle mesh; separation from this hull is a conservative lower bound on true separation.')
    return result,arrays
