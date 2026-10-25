"""NIST-derived rigid assembly parts and explicit passive-link flexible candidates.

Original STL vertices retain their source scale (millimetres converted to SI).
Only rigid coordinate changes and documented sub-micrometre seam welding occur.
Flexible materials are articulated engineering approximations, not calibrated
continuum models. This module never authors a robot, controller or PhysicsScene.
"""
from __future__ import annotations
import hashlib
import json
import math
from pathlib import Path
import numpy as np
import trimesh
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade
from bench.paths import DATA_ROOT, load_json
from scenes.workrooms.laboratory.object_geometry import Author as GeometryAuthor
from scenes.workrooms.laboratory.object_geometry import box, cylinder, ring, pose, tray_parts, schema, attr

SOURCE_ROOT = Path(f'{DATA_ROOT}/assets/nist/assembly_task_boards/20261020_source_snapshot')
ASSET_ROOT = Path(f'{DATA_ROOT}/assets/task_parts/mechanical_assembly')
ROOT_PRIM = '/World/NistTask'
TASK_IDS = ('nist_connector_insertion', 'nist_belt_drive_assembly', 'nist_cable_routing', 'nist_wire_harness')
SOURCE_URL = 'https://www.nist.gov/el/intelligent-systems-division-73500/robotic-grasping-and-manipulation-assembly/assembly'
SCHEMA = 'mechanical-nist-task-asset.v1'
ASSET_VERSIONS = {'nist_connector_insertion':'v3','nist_cable_routing':'v5','nist_belt_drive_assembly':'v5','nist_wire_harness':'v4'}
TABLE_Z = .78
PLATE_Z = .790


def _hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _q(rotation):
    mat=np.eye(4);mat[:3,:3]=rotation
    q=trimesh.transformations.quaternion_from_matrix(mat)
    return [float(q[1]),float(q[2]),float(q[3]),float(q[0])]


def _r(q):
    return trimesh.transformations.quaternion_matrix([q[3],*q[:3]])[:3,:3]


def _direction_q(direction):
    direction=np.asarray(direction,float);direction/=np.linalg.norm(direction)
    # Local X follows the physical cable/tape centreline.
    z=np.array([0.,0.,1.])
    if abs(np.dot(z,direction))>.99:z=np.array([0.,1.,0.])
    y=np.cross(z,direction);y/=np.linalg.norm(y);z=np.cross(direction,y)
    return _q(np.column_stack((direction,y,z)))


class Author(GeometryAuthor):
    def __init__(self,stage,task_id,root=ROOT_PRIM,source_root=SOURCE_ROOT):
        super().__init__(stage,task_id,root,False)
        self.source_root=Path(source_root);self.used={};self.entries={}
        self.meta.update(schema=SCHEMA,root_prim_path=root)
        self.meta['source']={'kind':'nist_cad_derived_plus_declared_simulation_fixtures',
            'source_url':SOURCE_URL,'source_root':str(self.source_root),
            'source_units':'millimetres','scale_to_m':.001,'external_assets':[],
            'license':'NIST source and source-specific third-party notices retained; see ATTRIBUTION.txt'}
        self.meta['physics_prior']['mass_source']='Declared engineering masses; not vendor-measured.'
        self.meta['physics_prior']['material_source']='Declared dry-contact simulation prior; not calibrated NIST material data.'
        self.meta['diagnostic_goal_poses']={}
        for name,color in [('red',(.65,.035,.025)),('green',(.025,.34,.11))]:
            m=UsdShade.Material.Define(stage,root+'/Materials/'+name)
            s=UsdShade.Shader.Define(stage,root+'/Materials/'+name+'/Shader');s.CreateIdAttr('UsdPreviewSurface')
            s.CreateInput('diffuseColor',Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
            s.CreateInput('roughness',Sdf.ValueTypeNames.Float).Set(.48)
            m.CreateSurfaceOutput().ConnectToSource(s.ConnectableAPI(),'surface');self.materials[name]=m

    def body(self,name,world_pose,parts,mass=0.,color='white',approximation='convexHull',**extras):
        entry=super().body(name,world_pose,parts,mass,color,approximation,**extras)
        analytic_boxes=[]
        for i,part in enumerate(parts):
            # Exact axis-aligned procedural boxes only. Do not turn curved,
            # hollow, threaded or imported CAD surfaces into bounding boxes.
            vertices=np.asarray(part.vertices)
            if len(vertices)!=8 or len(part.faces)!=12:
                continue
            if not all(len(np.unique(np.round(vertices[:,axis],11)))==2 for axis in range(3)):
                continue
            low,high=part.bounds;path=entry['collider_paths'][i]
            self.stage.RemovePrim(path)
            shape=UsdGeom.Cube.Define(self.stage,path);shape.CreateSizeAttr(1.)
            shape.AddTranslateOp().Set(Gf.Vec3d(*((low+high)/2)))
            shape.AddScaleOp().Set(Gf.Vec3f(*(high-low)))
            UsdShade.MaterialBindingAPI.Apply(shape.GetPrim()).Bind(self.materials[color])
            UsdPhysics.CollisionAPI.Apply(shape.GetPrim())
            UsdShade.MaterialBindingAPI(shape.GetPrim()).Bind(self.contact,materialPurpose='physics')
            schema(shape.GetPrim(),'PhysxCollisionAPI')
            attr(shape.GetPrim(),'physxCollision:contactOffset',Sdf.ValueTypeNames.Float,.0002)
            attr(shape.GetPrim(),'physxCollision:restOffset',Sdf.ValueTypeNames.Float,0.)
            analytic_boxes.append(i)
        if analytic_boxes:
            entry['analytic_box_shape_indices']=analytic_boxes
            entry['collision_approximation']='analytic_box' if len(analytic_boxes)==len(parts) else 'mixed_convex_analytic_box'
            entry['primitive_representation_note']='Exact procedural box dimensions; avoids native thin-convex far-face contacts. Original CAD is unchanged.'
        self.entries[name]=entry
        return entry

    def source_mesh(self,board,name,*,matrix=None,origin=None,weld_mm=None):
        folder=self.source_root/'extracted'/f'atb{board}_stl'
        matches=list(folder.rglob(name))
        if len(matches)!=1:raise ValueError(f'Expected one source {board}/{name}; got {len(matches)}')
        path=matches[0];mesh=trimesh.load(path,force='mesh',process=True)
        rec={'path':str(path),'sha256':_hash(path),'board':board,'units':'mm','scale_to_m':.001,
             'source_vertex_count':len(mesh.vertices),'source_face_count':len(mesh.faces),
             'source_bounds_mm':mesh.bounds.tolist(),'source_watertight':bool(mesh.is_watertight)}
        if weld_mm is not None:
            before=mesh.vertices.copy();volume=mesh.volume
            mesh.merge_vertices(digits_vertex=int(round(-math.log10(weld_mm))))
            rec['seam_weld']={'grid_mm':weld_mm,'vertices_after':len(mesh.vertices),
                'volume_relative_change':float(abs(mesh.volume-volume)/max(abs(volume),1e-12)),
                'reason':'Original STL had four unpaired seam edges; coincident source coordinates differ below0.1micrometre.'}
        if matrix is not None:
            matrix=np.asarray(matrix,float);mesh.apply_transform(matrix);rec['rigid_transform_raw']=matrix.tolist()
        if origin is not None:
            mesh.apply_translation(-np.asarray(origin,float));rec['origin_after_transform_mm']=list(map(float,origin))
        mesh.apply_scale(.001)
        if not mesh.is_watertight:raise ValueError(f'Open source collision mesh: {name}')
        self.used[str(path)]=rec
        return mesh

    def centered_source(self,board,name,*,matrix=None,weld_mm=None):
        m=self.source_mesh(board,name,matrix=matrix,weld_mm=weld_mm)
        center=(m.bounds[0]+m.bounds[1])/2
        m.apply_translation(-center)
        self.used[next(reversed(self.used))]['additional_centering_m']=center.tolist()
        return m

    def plate(self,*,size=(.48,.74,.010),center=(.40,0.,.785)):
        self.body('workplate',pose(*center),[box(size)],color='white',
                  geometry_origin='Original task-owned compact rearrangement; no source board scaling.')

    def seating(self,actor,target,contacts,*,axis_only=False,tol=(.006,.006,.008)):
        q=target[3:]
        return {'id':actor+'_seated','actor':actor,'target_position_m':target[:3],
            'target_quaternion_wxyz':[q[3],*q[:3]],'position_tolerance_m':list(tol),
            'orientation_tolerance_rad':.35,'orientation_axis_only':axis_only,
            'orientation_axis_local':[0.,0.,1.],'support_ids':list(contacts),
            'contact_ids':list(contacts)}

    def joint(self,name,first,second,anchor_world,*,kind='SphericalJoint'):
        if first not in self.meta['actors'] or second not in self.meta['actors']:
            raise ValueError('Passive joints connect two explicitly declared dynamic bodies')
        path=self.root+'/Joints/'+name
        joint=(UsdPhysics.SphericalJoint if kind=='SphericalJoint' else UsdPhysics.FixedJoint).Define(self.stage,path)
        joint.CreateBody0Rel().SetTargets([self.meta['actors'][first]])
        joint.CreateBody1Rel().SetTargets([self.meta['actors'][second]])
        for i,actor in enumerate((first,second)):
            e=self.entries[actor];q=e['initial_pose_xyzw'][3:]
            local=_r(q).T@(np.asarray(anchor_world)-e['initial_pose_xyzw'][:3])
            getattr(joint,f'CreateLocalPos{i}Attr')().Set(Gf.Vec3f(*local))
            getattr(joint,f'CreateLocalRot{i}Attr')().Set(Gf.Quatf(q[3],Gf.Vec3f(*[-v for v in q[:3]])))
        joint.CreateCollisionEnabledAttr(False)
        joint.CreateExcludeFromArticulationAttr(True)
        self.meta['joints'][name]=path
        # Filtering is only for the two permanently adjacent bodies.
        for actor,other in ((first,second),(second,first)):
            p=self.stage.GetPrimAtPath(self.meta['actors'][actor])
            rel=UsdPhysics.FilteredPairsAPI.Apply(p).CreateFilteredPairsRel()
            old=list(rel.GetTargets());target=Sdf.Path(self.meta['actors'][other])
            rel.SetTargets(old+([] if target in old else [target]))
        return path

    def chain(self,prefix,nodes,*,radius=.00175,width=None,thickness=.0025,color='black',mass=.035,closed=False):
        nodes=np.asarray(nodes,float);pairs=list(zip(nodes[:-1],nodes[1:]))
        if closed:pairs.append((nodes[-1],nodes[0]))
        ids=[]
        for i,(p0,p1) in enumerate(pairs):
            name=f'{prefix}_{i:03}';length=float(np.linalg.norm(p1-p0));q=_direction_q(p1-p0)
            if width is None:
                mesh=trimesh.creation.capsule(radius=radius,height=length,count=[8,8])
                # Trimesh capsule extends from0 to height before centering.
                mesh.apply_translation(-mesh.bounds.mean(0));mesh.apply_transform(trimesh.geometry.align_vectors([0,0,1],[1,0,0]))
            else:mesh=box((length,thickness,width))
            entry=self.body(name,pose(*((p0+p1)/2),q),[mesh],mass/len(pairs),color,
                link_length_m=length,flexible_group=prefix,
                physical_model='Passive articulated segment; no pose playback or motor')
            # Small convex-mesh contact candidates are retained in v1/v2.
            # Use exact intended primitive dimensions, as for laboratory grains;
            # no hull cooking, inflation, solver/gain or mass changes.
            shape_path=entry['collider_paths'][0]
            self.stage.RemovePrim(shape_path)
            if width is None:
                shape=UsdGeom.Capsule.Define(self.stage,shape_path)
                shape.CreateAxisAttr('X');shape.CreateHeightAttr(length);shape.CreateRadiusAttr(radius)
                entry['collision_approximation']='analytic_capsule'
                entry['analytic_geometry']={'axis':'X','cylinder_length_m':length,'radius_m':radius}
            else:
                shape=UsdGeom.Cube.Define(self.stage,shape_path);shape.CreateSizeAttr(1.)
                shape.AddScaleOp().Set(Gf.Vec3f(length,thickness,width))
                entry['collision_approximation']='analytic_box'
                entry['analytic_geometry']={'size_m':[length,thickness,width]}
            UsdShade.MaterialBindingAPI.Apply(shape.GetPrim()).Bind(self.materials[color])
            UsdPhysics.CollisionAPI.Apply(shape.GetPrim())
            UsdShade.MaterialBindingAPI(shape.GetPrim()).Bind(self.contact,materialPurpose='physics')
            schema(shape.GetPrim(),'PhysxCollisionAPI')
            attr(shape.GetPrim(),'physxCollision:contactOffset',Sdf.ValueTypeNames.Float,.0002)
            attr(shape.GetPrim(),'physxCollision:restOffset',Sdf.ValueTypeNames.Float,0.)
            ids.append(name)
        for i in range(len(ids)-1):self.joint(f'{prefix}_joint_{i:03}',ids[i],ids[i+1],nodes[i+1])
        if closed:self.joint(f'{prefix}_joint_{len(ids)-1:03}',ids[-1],ids[0],nodes[0])
        return ids

    def finish(self):
        self.meta['source']['external_assets']=list(self.used.values())
        self.meta['qualification']='Authored physical candidate; no robot manipulation or native acceptance claimed.'
        return self.meta


def _connector_task(a):
    a.plate(size=(.30,.56,.010))
    goals=[]
    # These two pairs were exported in a shared assembled coordinate frame.
    # Reverse Z to present a horizontal upward-facing receptacle.
    for name,male,female,center,port,supply,mass in (
        ('usb','USB_Male.STL','USB_Female.STL',(311.9105,345.8883,42.71591949),(.43,-.14,.834),(.30,-.14),.025),
        ('bnc','BNC_Male.STL','BNC_Female.STL',(311.9105,138.8883,66.3959198),(.43,.14,.814),(.30,.14),.018)):
        T=np.eye(4);T[:3,:3]=np.diag([1.,-1.,-1.]);T[:3,3]=-T[:3,:3]@np.asarray(center)
        fm=a.source_mesh(1,female,matrix=T);mm=a.source_mesh(1,male,matrix=T)
        # Source USB port42mm body, BNC port21.17mm body: lower face rests on a pedestal.
        base=port[2]+float(fm.bounds[0,2]);height=base-PLATE_Z
        supports=[box((.043,.055,max(.001,height)),(0,0,-height/2))]
        a.body(name+'_pedestal',pose(port[0],port[1],base),supports,color='teal')
        a.body(name+'_socket',pose(*port),[fm],color='steel',approximation='sdf',source_pair_frame_preserved=True)
        initial=pose(supply[0],supply[1],PLATE_Z+.0003-float(mm.bounds[0,2]))
        a.body(name+'_plug',initial,[mm],mass,'blue' if name=='usb' else 'steel',approximation='sdf',nominal_support='workplate')
        target=pose(*port)
        if name=='bnc':
            # Source coframe has bayonet-pin interference. Align source entry
            # slots90degrees and withdraw2.5mm; about8mm actual insertion
            # remains. No locked-bayonet/electrical-continuity claim.
            target=pose(port[0],port[1],port[2]+.0025,(0.,0.,-math.sqrt(.5),math.sqrt(.5)))
        a.meta['diagnostic_goal_poses'][name+'_plug']=target
        goals.append(a.seating(name+'_plug',target,[name+'_socket'],axis_only=name=='bnc'))
    a.meta['task_parameters']={'seating_goals':goals,'flexible_groups':[],
        'required_moved_ids':['usb_plug','bnc_plug'],
        'bnc_entry_configuration':{'source_axial_withdrawal_m':.0025,'world_yaw_degrees':-90.,'approximate_insertion_depth_m':.008,'source_sampling':'4000 male surface samples; no source-female interior samples at this entry configuration; native fit separately required.'},
        'adaptation':'ATB1 connector insertion subset; BNC uses a partially inserted slot-aligned entry, no electrical continuity or BNC bayonet-lock test; other ATB1 operations duplicate existing Tasks2/4/5.'}


def _atb2_rotation(a):
    path=a.source_root/'atb2_board_frame.json'
    if not path.is_file():raise FileNotFoundError(path)
    transform=np.asarray(load_json(path)['raw_to_board_mm'])
    # The OBB frame normal points into the original assembly. Rotate180°aboutX.
    flip=np.diag([1.,-1.,-1.,1.]);return flip@transform


def _belt_outline(centers,radii,count=64,segment_length=.675/64):
    """Equal-chord nodes, plus closure residual for a source-length belt.

    A polyline samples the original pulley radii densely. Walking exact rigid
    link lengths gives anchor-identical reset/goal loops after centre-spacing
    bisection. Equal arc samples would instead stretch curved link anchors.
    """
    from scipy.spatial import ConvexHull
    angles=np.linspace(0,2*math.pi,720,endpoint=False)
    pts=np.vstack([np.asarray(c)+np.c_[np.cos(angles),np.sin(angles)]*r for c,r in zip(centers,radii)])
    hull=pts[ConvexHull(pts).vertices]
    edges=np.roll(hull,-1,axis=0)-hull;lens=np.linalg.norm(edges,axis=1)
    cumulative=np.r_[0.,np.cumsum(lens)];perimeter=float(cumulative[-1])
    edge_index=0;point=hull[0].copy();nodes=[point.copy()]
    for step in range(count):
        origin=point.copy()
        while True:
            j=edge_index%len(hull);v=hull[j];edge=edges[j]
            # Following a convex outline by a short chord has one forward
            # intersection, even when one span crosses several tiny facets.
            end=v+edge
            if np.linalg.norm(end-origin)<segment_length-1e-12:
                edge_index+=1;continue
            d=v-origin;aa=float(np.dot(edge,edge));bb=2*float(np.dot(d,edge));cc=float(np.dot(d,d)-segment_length**2)
            root=(-bb+math.sqrt(max(0.,bb*bb-4*aa*cc)))/(2*aa)
            if root<-1e-9 or root>1+1e-9:
                edge_index+=1;continue
            root=float(np.clip(root,0.,1.));point=v+root*edge
            progress=(edge_index//len(hull))*perimeter+cumulative[j]+root*lens[j]
            break
        if step<count-1:nodes.append(point.copy())
    return np.asarray(nodes),perimeter,float(progress-perimeter)


def _belt_task(a):
    a.plate()
    rot=_atb2_rotation(a)
    small=a.centered_source(2,'Task Board belt drive Assembled - _TTPT32T5250-A-H10_b-1.STL',matrix=rot,weld_mm=.0001)
    large=a.centered_source(2,'Task Board belt drive Assembled - _TTPT60T5250-A-P12_b-1.STL',matrix=rot)
    # Source 675mm T5 belt,25.4mm mesh width. Articulated straight spans
    # approximate rubber bending; no tooth-lock or stretch servo is introduced.
    count=64;length=.675
    # Cross sections of unscaled source pulley meshes: groove tooth-tip radii
    #25.050/47.325mm, flanges27.5/49.5mm. Chord sag plus half tape thickness
    # leaves0.1–0.25mm radial clearance, with tape inside the axial flanges.
    groove_radii=[.025050015,.047325008,.015]
    centerline_radii=[.02695,.04898,.0173]
    x=.455;idler_x=.385
    low,high=.17,.26
    for _ in range(50):
        sep=(low+high)/2;centers=[[x,-sep/2],[x,sep/2],[idler_x,0.]]
        _,perim,residual=_belt_outline(centers,centerline_radii,count)
        if residual>0:low=sep
        else:high=sep
    outline,perim,residual=_belt_outline(centers,centerline_radii,count)
    if abs(residual)>1e-8:raise ValueError('Belt equal-chord closure failed')
    target_z=PLATE_Z+.0175
    poses={}
    for name,center,mesh,radius,mass,supply in (
        ('drive_pulley',centers[0],small,.00475,.085,None),
        ('timing_pulley',centers[1],large,.00570,.19,[.29,.265]),
        ('tensioner_idler',centers[2],None,.00370,.028,[.28,-.28])):
        if mesh is None:
            parts=ring(.0042,.015,-.013,.013,48)
            z=PLATE_Z+.0145
        else:parts=[mesh];z=PLATE_Z+.0015-float(mesh.bounds[0,2])
        target=pose(center[0],center[1],z);poses[name]=target
        shaft=[cylinder(radius,.026,(0,0,.0145)),cylinder(radius+.003,.0015,(0,0,.00075))]
        a.body(name+'_shaft',pose(center[0],center[1],PLATE_Z),shaft,color='steel')
        initial=target if supply is None else pose(supply[0],supply[1],PLATE_Z+.0003-min(p.bounds[0,2] for p in parts))
        a.body(name,initial,parts,mass,'steel',approximation='sdf' if mesh is not None else 'convexHull',nominal_support='workplate')
        a.meta['diagnostic_goal_poses'][name]=target
    # Equal chord loop: total physical segment length is675mm exactly.
    radius=length/(2*count*math.sin(math.pi/count));angles=np.linspace(0,2*math.pi,count,endpoint=False)
    nodes=np.c_[.280+radius*np.cos(angles),-.055+radius*np.sin(angles),np.full(count,PLATE_Z+.0127+.0003)]
    ids=a.chain('belt_link',nodes,width=.0254,thickness=.0025,color='black',mass=.065,closed=True)
    goals_nodes=np.c_[outline,np.full(count,target_z)]
    for i,name in enumerate(ids):
        p0,p1=goals_nodes[i],goals_nodes[(i+1)%count]
        a.meta['diagnostic_goal_poses'][name]=pose(*((p0+p1)/2),_direction_q(p1-p0))
    goals=[a.seating(n,poses[n],[n+'_shaft'],axis_only=True,tol=(.005,.005,.006)) for n in ['drive_pulley','timing_pulley','tensioner_idler']]
    for goal in goals:goal['support_ids'].append('workplate')
    a.meta['task_parameters']={'seating_goals':goals,'required_moved_ids':['timing_pulley','tensioner_idler'],
        'flexible_groups':[{'id':'timing_belt','actor_ids':ids,'closed':True,
            'maximum_neighbor_distance_m':.015,'support_ids':['workplate','drive_pulley','timing_pulley','tensioner_idler'],
            'minimum_support_fraction':.02,'plane_tolerance_m':.025,
            'required_mating_contact_ids':['drive_pulley','timing_pulley'],
            'enclosed_targets':[{'center_m':[c[0],c[1],target_z],'radius_m':r*.65} for c,r in zip(centers,[.0275,.0495,.015])],
            'perimeter_range_m':[.63,.71]}],
        'belt':{'source_designation':'TTBU675T5-250','nominal_length_m':length,'width_m':.0254,'segment_count':count,
            'target_outline_perimeter_m':perim,'target_centerline_m':goals_nodes.tolist(),
            'source_groove_tooth_tip_radii_m':groove_radii,'centerline_wrap_radii_m':centerline_radii,
            'source_pulley_axial_groove_half_width_m':.014,'goal_uses_actual_axial_groove':True,
            'approximation':'Passive articulated closed tape; no continuum strain, calibrated tension or timing-tooth transmission claim.'},
        'adaptation':'ATB2 timing-pulley/belt/tensioner subset. Independent idler is installed on a real post to extend belt path. Collars/threading omitted as duplicates.'}


def _tube_mesh(a):
    mesh=a.source_mesh(3,'thin round cable tube .STL')
    center=mesh.bounds.mean(0);mesh.apply_translation(-center)
    R=np.array([[1,0,0],[0,0,-1],[0,1,0.]])
    T=np.eye(4);T[:3,:3]=R;mesh.apply_transform(T)
    return mesh


def _cable_task(a):
    a.plate()
    tube=_tube_mesh(a);guide_z=PLATE_Z-float(tube.bounds[0,2])
    guides=[(.40,-.105,guide_z),(.40,.095,guide_z)]
    for i,c in enumerate(guides):a.body(f'guide_{i}',pose(*c),[tube.copy()],color='green' if i==0 else 'amber',approximation='sdf')
    # Two original-size cable-end fixtures; open clip and tube route is reversible.
    for i,y in enumerate([-.205,.195]):
        a.body(f'route_post_{i}',pose(.365,y,PLATE_Z),[cylinder(.006,.022,(0,0,.011))],color='steel')
    stereo=a.source_mesh(3,'Stereo Plug.STL');origin=np.array([*stereo.bounds.mean(0)[:2],stereo.bounds[0,2]])
    stereo.apply_translation(-origin)
    R=np.array([[1,0,0],[0,0,1],[0,-1,0.]]) # localZ points world+Y
    q=_q(R);count=42;wire_len=.56
    nodes=np.c_[np.full(count+1,.27),np.linspace(-.305,.255,count+1),np.full(count+1,PLATE_Z+.00175+.0003)]
    ids=a.chain('cable_link',nodes,mass=.023)
    # Cable has a physical weighted start plug, not a forbidden static-body joint.
    start='cable_start';a.body(start,pose(nodes[0,0],nodes[0,1],PLATE_Z+.0048),[box((.012,.012,.009))],.018,'black')
    a.joint('cable_start_join',start,ids[0],nodes[0],kind='FixedJoint')
    end='cable_end';a.body(end,pose(nodes[-1,0],nodes[-1,1],PLATE_Z+.0053,q),[stereo],.018,'black',approximation='sdf')
    a.joint('cable_end_join',ids[-1],end,nodes[-1],kind='FixedJoint')
    # True open cylindrical socket for the3.5mm audio pin; no convex hull closure.
    socket_y=.309;socket_z=guide_z
    sockets=ring(.00215,.007,-.006,.006,32)
    for m in sockets:m.apply_transform(trimesh.geometry.align_vectors([0,0,1],[0,1,0]))
    a.body('audio_socket',pose(.40,socket_y,socket_z),sockets,color='steel')
    a.body('audio_socket_stand',pose(.40,socket_y,PLATE_Z),[box((.024,.024,socket_z-PLATE_Z-.007),(0,0,(socket_z-PLATE_Z-.007)/2))],color='teal')
    target=pose(.40,socket_y-.054,socket_z,q)
    allids=[start,*ids,end]
    delta=np.asarray(target[:3])-a.entries[end]['initial_pose_xyzw'][:3]
    for name in allids:
        old=a.entries[name]['initial_pose_xyzw'];a.meta['diagnostic_goal_poses'][name]=pose(*(np.asarray(old[:3])+delta),old[3:])
    # A reset-only route witness with measured path arc length; actual production
    # actions still manipulate only robot joints. Diagnostic joints must be
    # rebuilt from these nodes by an engineering author before stepping.
    route=np.array([[.27,-.305,socket_z],[.385,-.23,socket_z],[.40,-.105,socket_z],[.40,.095,socket_z],[.40,target[1],socket_z]])
    seat=a.seating(end,target,['audio_socket'],axis_only=True,tol=(.006,.008,.006))
    # The imported body's origin is at the cable tail, 54 mm behind the pin
    # mating region. Natural tilt about a still-inserted pin moves that origin
    # by several millimetres; measure insertion at the physical pin instead.
    seat['actor_local_point_m']=[0.,0.,.054]
    seat['target_position_m']=[.40,socket_y,socket_z]
    seat['position_reference_note']='Original source pin mating region, 54 mm from the source tail origin; socket center is the target.'
    seat['support_via_flexible_group']='audio_cable'
    a.meta['task_parameters']={'seating_goals':[seat],
        'required_moved_ids':[end],
        'flexible_groups':[{'id':'audio_cable','actor_ids':allids,'closed':False,
            'maximum_neighbor_distance_m':.025,'support_ids':['workplate','guide_0','guide_1','audio_socket'],
            'minimum_support_fraction':.02,'route_gates':[{'center_m':list(c),'radius_m':.0095} for c in guides]}],
        'cable':{'diameter_m':.0035,'length_m':wire_len,'source_designation':'EMSOD-1.5 / ATB3 thin audio cable',
            'approximation':'Passive spherical-link cable, nonadjacent self-collision enabled, no continuum bending calibration.',
            'diagnostic_route_waypoints_m':route.tolist()},
        'adaptation':'One original-size audio cable and two source tubes; routing/correct-end insertion retained. No hand control or pin electrical continuity claim.'}


def _harness_task(a):
    a.plate(size=(.48,.78,.010))
    # Actual ATB4 connector body. Keep its mesh open recesses under SDF.
    mesh=a.centered_source(4,'Wire harness - AT06_2S_LED1220-2 AT06_2S_LED1220_3-1.STL')
    # Source18mm body is positioned vertically as a terminal endpoint; the
    # project fixture is a broad removable nest, not a magnetic connector latch.
    half=mesh.extents/2;terminal_z=PLATE_Z+float(half[2])+.0003
    goals=[];groups=[];moved=[];allactors=[]
    for color,x,goalx in [('red',.255,.41),('blue',.31,.46)]:
        count=30;nodes=np.c_[np.full(count+1,x),np.linspace(-.205,.205,count+1),np.full(count+1,PLATE_Z+.0015+.0003)]
        ids=a.chain(color+'_link',nodes,radius=.0015,color=color,mass=.018)
        endpoints=[]
        for side,index,y in [('a',0,-.213),('b',count,.213)]:
            name=color+'_end_'+side
            bodypose=pose(x,y,terminal_z)
            a.body(name,bodypose,[mesh.copy()],.012,color,approximation='sdf')
            anchor=nodes[index]
            a.joint(name+'_wire',name,ids[0] if side=='a' else ids[-1],anchor,kind='FixedJoint')
            endpoints.append(name);moved.append(name)
            target=pose(goalx,y,terminal_z+.003)
            # The terminal nest has a real cable exit in its inward-facing
            # wall. An unbroken wall here crosses the attached wire and makes
            # the fixed endpoint joint pull two bodies through solid material.
            outer=(float(mesh.extents[0])+.009,float(mesh.extents[1])+.009,.012)
            floor=.003;wall=.003;exit_width=.006
            parts=tray_parts(outer,floor=floor,wall=wall)
            del parts[3 if side=='a' else 4]
            pillar_width=(outer[0]-exit_width)/2
            wall_y=(1 if side=='a' else -1)*(outer[1]-wall)/2
            for sign in [-1,1]:
                parts.append(box((pillar_width,wall,outer[2]),
                    (sign*(exit_width+pillar_width)/2,wall_y,outer[2]/2)))
            a.body(name+'_terminal',pose(goalx,y,PLATE_Z),parts,color=color,
                cable_exit={'direction_local':[0,1 if side=='a' else -1,0],
                    'width_m':exit_width,'bottom_above_plate_m':floor,
                    'open_top':True,'wire_diameter_m':.003,
                    'note':'Only the wire-facing wall is split; floor and remaining walls retain original nest dimensions.'})
            goals.append(a.seating(name,target,[name+'_terminal'],tol=(.006,.006,.009)))
            a.meta['diagnostic_goal_poses'][name]=target
        groupids=[endpoints[0],*ids,endpoints[1]];allactors+=groupids
        for name in groupids:
            old=a.entries[name]['initial_pose_xyzw'];a.meta['diagnostic_goal_poses'][name]=pose(old[0]+goalx-x,old[1],old[2]+.003,old[3:])
        groups.append({'id':color+'_lead','actor_ids':groupids,'closed':False,
            'maximum_neighbor_distance_m':.045,'support_ids':['workplate','bundle_retainer_0','bundle_retainer_1',endpoints[0]+'_terminal',endpoints[1]+'_terminal'],
            'minimum_support_fraction':.02,'route_gates':[{'center_m':[.435,y,PLATE_Z+.009],'radius_m':.04} for y in [-.06,.06]]})
    for i,y in enumerate([-.06,.06]):
        # Common open retainer retains routing while permitting vertical release.
        parts=[box((.09,.014,.003),(0,0,.0015)),box((.004,.014,.018),(-.043,0,.009)),box((.004,.014,.018),(.043,0,.009))]
        a.body('bundle_retainer_'+str(i),pose(.435,y,PLATE_Z),parts,color='black')
    traycenter=[.40,.325,PLATE_Z]
    a.body('collection_tray',pose(*traycenter),tray_parts((.35,.12,.025),floor=.003,wall=.003),color='teal')
    a.meta['task_parameters']={'seating_goals':goals,'flexible_groups':groups,'required_moved_ids':moved,
        'removal':{'actor_ids':allactors,'center_m':[.40,.325,PLATE_Z+.060],
            'half_extents_m':[.171,.056,.057],'support_ids':['collection_tray']},
        'adaptation':'ATB4 two-color prepared-lead subset: correct terminal mapping, shared open retainer routing, released assembly then removal to tray. No crimping, hidden snap, electrical-current or cable-tie tightening claim.',
        'harness':{'lead_count':2,'wire_diameter_m':.003,'wire_length_m':.410,
            'source_designation':'ATB4 color-coordinated harness with AT06-2S housings',
            'approximation':'Passive spherical-link conductors with source connector shells; synthetic open terminal nests permit reversible removal.'}}


def author_task(stage,task_id,root_path=ROOT_PRIM,*,source_root=SOURCE_ROOT,presentation=False):
    if task_id not in TASK_IDS:raise ValueError(task_id)
    if presentation:raise ValueError('This builder authors canonical reset only; use separately recorded diagnostic poses')
    if stage.GetPrimAtPath(root_path):raise ValueError('Task root must be fresh')
    UsdGeom.SetStageMetersPerUnit(stage,1.);UsdGeom.SetStageUpAxis(stage,'Z')
    a=Author(stage,task_id,root_path,source_root)
    {_id:_fn for _id,_fn in zip(TASK_IDS,(_connector_task,_belt_task,_cable_task,_harness_task))}[task_id](a)
    return a.finish()


def structural_audit(stage,meta):
    joints=[];errors=[];goal_gaps=[]
    objects={x['body_path']:x for x in meta['objects']}
    for name,path in meta['joints'].items():
        joint=UsdPhysics.Joint(stage.GetPrimAtPath(path));targets=[joint.GetBody0Rel().GetTargets(),joint.GetBody1Rel().GetTargets()]
        if any(len(x)!=1 for x in targets):errors.append('joint endpoint:'+name);continue
        if any(str(x[0]) not in meta['actors'].values() for x in targets):errors.append('undeclared endpoint:'+name)
        world=[]
        for i,t in enumerate(targets):
            mat=UsdGeom.Xformable(stage.GetPrimAtPath(t[0])).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
            pos=getattr(joint,f'GetLocalPos{i}Attr')().Get();world.append(np.array(mat.Transform(Gf.Vec3d(*pos))))
        gap=float(np.linalg.norm(world[0]-world[1]));joints.append({'id':name,'anchor_gap_m':gap})
        if gap>1e-6:errors.append('anchor mismatch:'+name)
        goals=[]
        for i,t in enumerate(targets):
            obj=objects[str(t[0])];p=meta['diagnostic_goal_poses'].get(obj['id'],obj['initial_pose_xyzw'])
            local=getattr(joint,f'GetLocalPos{i}Attr')().Get();goals.append(np.asarray(p[:3])+_r(p[3:])@np.asarray(local))
        goal_gap=float(np.linalg.norm(goals[0]-goals[1]));goal_gaps.append(goal_gap)
        if goal_gap>1e-6:errors.append('goal anchor mismatch:'+name)
    for obj in meta['objects']:
        if obj['mass_kg']<=0 or min(obj['diagonal_inertia_kg_m2'])<=0:errors.append('mass:'+obj['id'])
        if not all(np.isfinite(obj['initial_pose_xyzw'])):errors.append('pose:'+obj['id'])
    return {'schema':'mechanical_nist_structural_audit.v1','passed':not errors,'errors':errors,
        'dynamic_body_count':len(meta['actors']),'fixture_count':len(meta['fixtures']),
        'joint_count':len(joints),'joints':joints,'maximum_goal_anchor_gap_m':max(goal_gaps,default=0.),'native_physics_accepted':False,
        'scope':'Authored topology/units/mass/anchors only; native collision and task feasibility require separate evidence.'}


def build_assets(task_id,output_dir=None,*,source_root=SOURCE_ROOT):
    output=Path(output_dir) if output_dir else ASSET_ROOT/task_id/ASSET_VERSIONS.get(task_id,'v1')
    if output.exists() and any(output.iterdir()):raise FileExistsError(f'Refusing to rewrite immutable asset: {output}')
    output.mkdir(parents=True,exist_ok=True)
    stage=Usd.Stage.CreateNew(str(output/'task.usda'));meta=author_task(stage,task_id,source_root=source_root)
    stage.GetRootLayer().Save();audit=structural_audit(stage,meta)
    (output/'metadata.json').write_text(json.dumps(meta,indent=2,sort_keys=True)+'\n')
    (output/'structural_audit.json').write_text(json.dumps(audit,indent=2,sort_keys=True)+'\n')
    attribution=('NIST Assembly Task Boards; National Institute of Standards and Technology.\n'
        +SOURCE_URL+'\nRetrieved2026-10-20; source page updated2026-10-01.\n'
        +'Original downloads and source-specific notices are preserved at '+str(source_root)+'.\n'
        +'NIST policies: https://www.nist.gov/copyrights-disclaimers and https://www.nist.gov/open/license .\n'
        +'No archive-level license file was supplied. Commercial part identifiers and CAD authorship are preserved; this is not a claim that every third-party asset is CC0.\n'
        +'2026-10-20 modifications: millimetre-to-metre conversion; rigid local frame changes; documented STL seam welding; compact fixtures and finite rigid-link flexible approximations.\n'
        +'This derivative is an adapted simulation task, not a NIST-certified implementation or measurement.\n')
    (output/'ATTRIBUTION.txt').write_text(attribution)
    dependencies={item['path']:{'sha256':item['sha256']} for item in meta['source']['external_assets']}
    for name in ['provenance.json','provenance_sources/source_manifest.json','provenance_sources/nist_copyrights_disclaimers.html','provenance_sources/nist_data_license_policy.html']:
        p=Path(source_root)/name
        if p.is_file():dependencies[str(p)]={'sha256':_hash(p)}
    manifest={'schema':SCHEMA,'task_id':task_id,'root_prim_path':ROOT_PRIM,'presentation':False,
        'usd_path':str(output/'task.usda'),'metadata_path':str(output/'metadata.json'),
        'usd_sha256':_hash(output/'task.usda'),'metadata_sha256':_hash(output/'metadata.json'),
        'structural_audit_sha256':_hash(output/'structural_audit.json'),'attribution_sha256':_hash(output/'ATTRIBUTION.txt'),
        'files_sha256':{name:_hash(output/name) for name in ['task.usda','metadata.json','structural_audit.json','ATTRIBUTION.txt']},
        'dependencies':dependencies,'source_root':str(source_root),'physics_accepted':False,
        'task_policy_success_claimed':False,'status':'prepared_unvalidated'}
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n')
    return manifest
