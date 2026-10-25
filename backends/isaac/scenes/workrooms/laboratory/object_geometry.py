"""Original laboratory contact geometry, SI units and explicit physical priors.

No room, robot, PhysicsScene or controller is authored here. Hollow structures
use actual separated convex pieces; no convex hull closes an intended opening.
"""
from __future__ import annotations
import math
import numpy as np
import trimesh
from pxr import Gf, Sdf, UsdGeom, UsdPhysics, UsdShade

COLORS = {'white': (.80,.85,.86), 'blue': (.055,.30,.63), 'amber': (.82,.42,.06),
          'teal': (.04,.37,.34), 'steel': (.45,.50,.53), 'black': (.08,.09,.10)}
PHYSICS_PRIOR = {'static_friction': .30, 'dynamic_friction': .25, 'restitution': .01,
    'contact_offset_m': .0002, 'rest_offset_m': 0.,
    'mass_source': 'Declared plausible synthetic laboratory object masses, not measured.',
    'material_source': 'Declared engineering dry-contact prior, not precision calibration.'}


def schema(prim, name):
    # CPU usd-core does not register PhysX APIs. Preserve the authored list,
    # otherwise applying a second schema silently erases an unknown first API.
    authored = prim.GetMetadata('apiSchemas')
    schemas = list(authored.GetAppliedItems()) if authored is not None else []
    if name not in schemas:
        prim.SetMetadata('apiSchemas', Sdf.TokenListOp.CreateExplicit(schemas+[name]))


def attr(prim, name, kind, value):
    prim.CreateAttribute(name, kind).Set(value)


def box(size, center=(0,0,0)):
    mesh = trimesh.creation.box(extents=size)
    mesh.apply_translation(center)
    return mesh


def cylinder(radius, height, center=(0,0,0), sections=32):
    mesh = trimesh.creation.cylinder(radius=radius, height=height, sections=sections)
    mesh.apply_translation(center)
    return mesh


def sector(inner, outer, z0, z1, angle0, angle1):
    """Closed convex annular chord segment (small enough angles stay convex)."""
    vertices = [[r*math.cos(a),r*math.sin(a),z] for z in (z0,z1)
                for r,a in ((inner,angle0),(outer,angle0),(outer,angle1),(inner,angle1))]
    return trimesh.convex.convex_hull(np.asarray(vertices))


def ring(inner, outer, z0, z1, segments=32):
    return [sector(inner,outer,z0,z1,2*math.pi*i/segments,2*math.pi*(i+1)/segments)
            for i in range(segments)]


def helical_sector(inner, outer, angle0, angle1, z0, z1, thickness):
    vertices = [[r*math.cos(a),r*math.sin(a),z+dz] for dz in (0,thickness)
                for a,z in ((angle0,z0),(angle1,z1)) for r in (inner,outer)]
    return trimesh.convex.convex_hull(np.asarray(vertices))



def helical_band(inner, outer, start, turn, base_z, pitch, thickness, segments=48):
    """Continuous closed helical ribbon without compound end-face seams."""
    vertices=[]
    for i in range(segments+1):
        t=turn*i/segments;angle=start+t;z=base_z+pitch*t/(2*math.pi)
        vertices.extend([[r*math.cos(angle),r*math.sin(angle),z+dz]
                         for r,dz in ((inner,0),(outer,0),(outer,thickness),(inner,thickness))])
    faces=[]
    for i in range(segments):
        for k in range(4):
            a=4*i+k;b=4*i+(k+1)%4;c=4*(i+1)+(k+1)%4;d=4*(i+1)+k
            faces.extend([(a,b,c),(a,c,d)])
    faces.extend([(0,2,1),(0,3,2)])
    end=4*segments;faces.extend([(end,end+1,end+2),(end,end+2,end+3)])
    mesh=trimesh.Trimesh(vertices=vertices,faces=faces,process=True)
    mesh.fix_normals()
    return mesh

def tray_parts(size=(.12,.12,.018), floor=.004, wall=.004):
    x,y,z=size
    return [box((x,y,floor),(0,0,floor/2)),
            box((wall,y,z),((x-wall)/2,0,z/2)),box((wall,y,z),(-(x-wall)/2,0,z/2)),
            box((x,wall,z),(0,(y-wall)/2,z/2)),box((x,wall,z),(0,-(y-wall)/2,z/2))]


def pose(x,y,z, quaternion=(0,0,0,1)):
    return [float(x),float(y),float(z),*[float(v) for v in quaternion]]


class Author:
    def __init__(self, stage, task_id, root, presentation):
        self.stage, self.root = stage, root
        UsdGeom.Xform.Define(stage, root)
        self.materials={}
        for name,color in COLORS.items():
            material=UsdShade.Material.Define(stage,root+'/Materials/'+name)
            shader=UsdShade.Shader.Define(stage,root+'/Materials/'+name+'/Shader')
            shader.CreateIdAttr('UsdPreviewSurface')
            shader.CreateInput('diffuseColor',Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
            shader.CreateInput('roughness',Sdf.ValueTypeNames.Float).Set(.38)
            material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(),'surface')
            self.materials[name]=material
        self.contact=UsdShade.Material.Define(stage,root+'/Materials/Contact')
        physical=UsdPhysics.MaterialAPI.Apply(self.contact.GetPrim())
        physical.CreateStaticFrictionAttr(PHYSICS_PRIOR['static_friction'])
        physical.CreateDynamicFrictionAttr(PHYSICS_PRIOR['dynamic_friction'])
        physical.CreateRestitutionAttr(PHYSICS_PRIOR['restitution'])
        self.meta={'schema':'laboratory_task_assets.v1','task_id':task_id,'root_path':root,
            'presentation':bool(presentation),'actors':{},'objects':[],'fixtures':{},
            'fixture_metadata':{},'joints':{},'particle_sets':{},'task_parameters':{},
            'source':{'kind':'original_procedural','license':'CC0-1.0','external_assets':[]},
            'physics_prior':dict(PHYSICS_PRIOR),'physics_accepted':False,
            'qualification':'Authored candidate; hand-object interaction belongs to data collection.'}

    def body(self,name,world_pose,parts,mass=0.,color='white',approximation='convexHull',**extras):
        path=self.root+('/Actors/' if mass else '/Fixtures/')+name
        root=UsdGeom.Xform.Define(self.stage,path)
        root.AddTranslateOp().Set(Gf.Vec3d(*world_pose[:3]))
        q=world_pose[3:]
        root.AddOrientOp().Set(Gf.Quatf(q[3],Gf.Vec3f(*q[:3])))
        colliders=[]
        for i,part in enumerate(parts):
            mesh=UsdGeom.Mesh.Define(self.stage,path+f'/Shape_{i:03}')
            mesh.CreatePointsAttr([Gf.Vec3f(*point) for point in part.vertices])
            mesh.CreateFaceVertexCountsAttr([3]*len(part.faces))
            mesh.CreateFaceVertexIndicesAttr(part.faces.reshape(-1).tolist())
            mesh.CreateSubdivisionSchemeAttr('none')
            UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(self.materials[color])
            UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
            UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr(approximation)
            if approximation=='sdf':
                schema(mesh.GetPrim(),'PhysxSDFMeshCollisionAPI')
                attr(mesh.GetPrim(),'physxSDFMeshCollision:sdfResolution',Sdf.ValueTypeNames.Int,512)
                attr(mesh.GetPrim(),'physxSDFMeshCollision:sdfSubgridResolution',Sdf.ValueTypeNames.Int,6)
                attr(mesh.GetPrim(),'physxSDFMeshCollision:sdfMargin',Sdf.ValueTypeNames.Float,.01)
                attr(mesh.GetPrim(),'physxSDFMeshCollision:sdfNarrowBandThickness',Sdf.ValueTypeNames.Float,.02)
            schema(mesh.GetPrim(),'PhysxCollisionAPI')
            attr(mesh.GetPrim(),'physxCollision:contactOffset',Sdf.ValueTypeNames.Float,.0002)
            attr(mesh.GetPrim(),'physxCollision:restOffset',Sdf.ValueTypeNames.Float,0.)
            UsdShade.MaterialBindingAPI(mesh.GetPrim()).Bind(self.contact,materialPurpose='physics')
            colliders.append(str(mesh.GetPath()))
        combined=trimesh.util.concatenate(parts)
        bounds=combined.bounds
        entry={'id':name,'body_path':path,'mass_kg':float(mass),
               'initial_position_m':world_pose[:3],'initial_pose_xyzw':world_pose,
               'orientation_xyzw':world_pose[3:],'half_extents_m':((bounds[1]-bounds[0])/2).tolist(),
               'local_bounds_m':bounds.tolist(),'collision_approximation':approximation,'collider_paths':colliders,**extras}
        if mass:
            UsdPhysics.RigidBodyAPI.Apply(root.GetPrim()).CreateRigidBodyEnabledAttr(True)
            UsdPhysics.RigidBodyAPI(root.GetPrim()).CreateKinematicEnabledAttr(False)
            volumes=np.asarray([abs(p.volume) for p in parts]); weights=mass*volumes/volumes.sum()
            com=sum(w*np.asarray(p.center_mass) for w,p in zip(weights,parts))/mass
            inertia=np.zeros((3,3))
            for w,p in zip(weights,parts):
                d=np.asarray(p.center_mass)-com
                inertia+=np.asarray(p.moment_inertia)*(w/abs(p.volume))+w*(np.dot(d,d)*np.eye(3)-np.outer(d,d))
            # Shapes here have nearly symmetric inertia; explicitly use full
            # principal axes rather than assigning an AABB inertia surrogate.
            eig,axes=np.linalg.eigh(inertia)
            if np.linalg.det(axes)<0: axes[:,0]*=-1
            transform=np.eye(4);transform[:3,:3]=axes
            quat=trimesh.transformations.quaternion_from_matrix(transform)
            api=UsdPhysics.MassAPI.Apply(root.GetPrim())
            api.CreateMassAttr(mass);api.CreateCenterOfMassAttr(Gf.Vec3f(*com))
            api.CreateDiagonalInertiaAttr(Gf.Vec3f(*eig))
            api.CreatePrincipalAxesAttr(Gf.Quatf(float(quat[0]),Gf.Vec3f(*quat[1:])))
            schema(root.GetPrim(),'PhysxRigidBodyAPI')
            attr(root.GetPrim(),'physxRigidBody:sleepThreshold',Sdf.ValueTypeNames.Float,0.)
            attr(root.GetPrim(),'physxRigidBody:enableSpeculativeCCD',Sdf.ValueTypeNames.Bool,True)
            entry.update(center_of_mass_m=com.tolist(),diagonal_inertia_kg_m2=eig.tolist())
            self.meta['actors'][name]=path;self.meta['objects'].append(entry)
        else:
            self.meta['fixtures'][name]=path;self.meta['fixture_metadata'][name]=entry
        return entry
