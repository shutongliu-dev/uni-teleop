#!/usr/bin/env python3
"""Inspect a composed USD room independently of its builder implementation.

Run with the shared CPU OpenUSD 26.8 environment. The audit reads real composed
geometry, transforms, material bindings and resolved asset paths. It does not
qualify manipulation, visual realism, or dynamic contact behavior.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import glob
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys

from pxr import Ar, Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade, UsdUtils

from bench.paths import load_json


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def safe(name):
    return re.sub(r'[^A-Za-z0-9_]', '_', str(name))


def finite(values):
    return all(math.isfinite(float(x)) for x in values)


def determinant(matrix):
    a = [[float(matrix[i][j]) for j in range(3)] for i in range(3)]
    return (a[0][0]*(a[1][1]*a[2][2]-a[1][2]*a[2][1])
            -a[0][1]*(a[1][0]*a[2][2]-a[1][2]*a[2][0])
            +a[0][2]*(a[1][0]*a[2][1]-a[1][1]*a[2][0]))


def bounds(stage, cache, path):
    prim = stage.GetPrimAtPath(path)
    if not prim:
        raise ValueError('Required geometry is absent: '+str(path))
    extent = cache.ComputeWorldBound(prim).ComputeAlignedRange()
    if extent.IsEmpty():
        raise ValueError('Geometry has an empty visible world bound: '+str(path))
    cached_lo, cached_hi = list(map(float, extent.GetMin())), list(map(float, extent.GetMax()))
    # BBoxCache legitimately uses an authored mesh extent. Verify against the
    # actual point buffer as well, so stale extents cannot certify a wrong table.
    lo, hi = [math.inf]*3, [-math.inf]*3
    xforms=UsdGeom.XformCache(Usd.TimeCode.Default())
    measured_points=0
    for child in Usd.PrimRange(prim,Usd.TraverseInstanceProxies()):
        imageable=UsdGeom.Imageable(child)
        if imageable and (imageable.ComputeVisibility()=='invisible'
                or imageable.ComputePurpose() not in ('default','render')):
            continue
        if child.IsA(UsdGeom.Mesh):
            matrix=xforms.GetLocalToWorldTransform(child)
            points=UsdGeom.Mesh(child).GetPointsAttr().Get()
            if points is None:continue
            for point in points:
                transformed=matrix.Transform(point)
                for i in range(3):
                    value=float(transformed[i]);lo[i]=min(lo[i],value);hi[i]=max(hi[i],value)
                measured_points+=1
        elif any(child.IsA(shape) for shape in (UsdGeom.Cube,UsdGeom.Cylinder,
                UsdGeom.Sphere,UsdGeom.Cone,UsdGeom.Capsule)):
            child_extent=cache.ComputeWorldBound(child).ComputeAlignedRange()
            if not child_extent.IsEmpty():
                for i in range(3):
                    lo[i]=min(lo[i],float(child_extent.GetMin()[i]))
                    hi[i]=max(hi[i],float(child_extent.GetMax()[i]))
    if not finite(lo+hi):
        lo,hi=cached_lo,cached_hi
    if not finite(lo+hi):
        raise ValueError('Geometry has nonfinite world bounds: '+str(path))
    return {'path': str(path), 'min_m':lo, 'max_m':hi,
            'size_m':[hi[i]-lo[i] for i in range(3)],
            'bbox_cache_min_m':cached_lo,'bbox_cache_max_m':cached_hi,
            'actual_mesh_vertices_measured':measured_points}


def visual_bound(stage, cache, name):
    return bounds(stage, cache, '/World/Geometry/'+name+'/Visual')


def xy_overlap(a, b):
    return [min(a['max_m'][i], b['max_m'][i])-max(a['min_m'][i], b['min_m'][i])
            for i in (0,1)]


def construction_checks(stage, cache):
    """Physical dimensions come from USD meshes, not scene_description constants."""
    table=visual_bound(stage,cache,'worktop')
    mount=visual_bound(stage,cache,'robot_mounting_plate')
    column=visual_bound(stage,cache,'robot_column')
    foot=visual_bound(stage,cache,'robot_anchor_plate')
    floor=visual_bound(stage,cache,'room_floor')
    dimensions={'measured':table,'required_width_m':1.,'required_depth_m':1.,
                'required_top_z_m':.78,'tolerance_m':1.e-5}
    dimensions['passed']=(abs(table['size_m'][0]-1.)<=1.e-5
        and abs(table['size_m'][1]-1.)<=1.e-5 and abs(table['max_m'][2]-.78)<=1.e-5)
    links=[]
    for lower,upper in [(floor,foot),(foot,column),(column,mount)]:
        gap=upper['min_m'][2]-lower['max_m'][2]
        overlap=xy_overlap(lower,upper)
        links.append({'lower':lower['path'],'upper':upper['path'],
            'vertical_gap_m':gap,'xy_overlap_m':overlap,
            'passed':-.006<=gap<=1.e-5 and min(overlap)>0})
    # Use the complete support footprint, including its real floor anchor plate.
    support_max_x=max(item['max_m'][0] for item in (mount,column,foot))
    gap=table['min_m'][0]-support_max_x
    separation={'table_near_x_m':table['min_m'][0],
        'support_outer_x_m':support_max_x,'measured_gap_m':gap,
        'required_gap_m':.10,'tolerance_m':1.e-5,
        'mount_to_worktop_gap_m':table['min_m'][0]-mount['max_m'][0],
        'passed':abs(gap-.10)<=1.e-5}
    base_candidates=[p for p in stage.Traverse() if p.GetName()=='fr3_link0_visual']
    if len(base_candidates)!=1:
        robot={'passed':False,'error':'Expected exactly one actual FR3 base visual',
               'candidates':[str(p.GetPath()) for p in base_candidates]}
    else:
        base=bounds(stage,cache,base_candidates[0].GetPath())
        vertical_gap=base['min_m'][2]-mount['max_m'][2]
        overlap=xy_overlap(base,mount)
        robot={'measured_base_visual':base,'mount_top_m':mount['max_m'][2],
            'vertical_gap_m':vertical_gap,'xy_overlap_m':overlap,
            'passed':abs(vertical_gap)<=.001 and min(overlap)>0,
            'scope':'Actual base mesh touches the visible mounting plate; no inferred load test.'}
    return {'workspace':dimensions,
        'support_geometry':{'mount':mount,'column':column,'foot':foot,'floor':floor},
        'support_continuity':links,'table_support_separation':separation,'robot_mount':robot,
        'passed':dimensions['passed'] and all(x['passed'] for x in links)
            and separation['passed'] and robot['passed']}


def collider_checks(stage):
    xforms=UsdGeom.XformCache(Usd.TimeCode.Default())
    counts=Counter(); failures=[]; walls=[]
    for prim in Usd.PrimRange.Stage(stage,Usd.TraverseInstanceProxies()):
        if not prim.HasAPI(UsdPhysics.CollisionAPI):
            continue
        path=str(prim.GetPath()); counts['all']+=1
        matrix=xforms.GetLocalToWorldTransform(prim)
        det=determinant(matrix)
        invalid=[]
        if not finite([matrix[i][j] for i in range(4) for j in range(4)]) or det<=0:
            invalid.append('nonfinite, reflected or singular transform')
        measurements={'path':path,'type':str(prim.GetTypeName()),'world_determinant':det}
        if prim.IsA(UsdGeom.Cube):
            counts['cube']+=1
            size=float(UsdGeom.Cube(prim).GetSizeAttr().Get())
            dimensions=[size*math.sqrt(sum(float(matrix[i][j])**2 for j in range(3))) for i in range(3)]
            measurements.update(cube_size=size,world_edge_lengths_m=dimensions)
            if not finite([size]+dimensions) or min([size]+dimensions)<=0:
                invalid.append('nonpositive or nonfinite primitive dimension')
        for shape in (UsdGeom.Cylinder,UsdGeom.Capsule,UsdGeom.Cone,UsdGeom.Sphere):
            if prim.IsA(shape):
                obj=shape(prim);radius=float(obj.GetRadiusAttr().Get())
                measurements['radius']=radius
                if not math.isfinite(radius) or radius<=0:
                    invalid.append('invalid primitive radius')
                if hasattr(obj,'GetHeightAttr'):
                    height=float(obj.GetHeightAttr().Get()); measurements['height']=height
                    if not math.isfinite(height) or height<=0:invalid.append('invalid primitive height')
        if path.startswith('/World/Geometry/wall_'):
            counts['wall']+=1;walls.append(measurements)
        if invalid:failures.append({**measurements,'errors':invalid})
    if not walls:failures.append({'error':'No actual wall colliders found'})
    return {'counts':dict(counts),'wall_colliders':walls,'failures':failures,
            'passed':not failures,'scope':'Dimensions and transforms, not native PhysX cooking/contact acceptance.'}


def environment_joint_checks(stage):
    """A joint between two imported disabled rigid bodies cannot constrain either."""
    def dynamic_owner(path):
        prim=stage.GetPrimAtPath(path)
        while prim and not prim.IsPseudoRoot():
            if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                body=UsdPhysics.RigidBodyAPI(prim)
                return {'path':str(prim.GetPath()),'rigid_body_enabled':bool(body.GetRigidBodyEnabledAttr().Get())}
            prim=prim.GetParent()
        return {'path':None,'rigid_body_enabled':False}
    records=[];failures=[]
    for prim in Usd.PrimRange.Stage(stage,Usd.TraverseInstanceProxies()):
        if not prim.IsA(UsdPhysics.Joint) or str(prim.GetPath()).startswith('/World/Robot/'):
            continue
        joint=UsdPhysics.Joint(prim);enabled=bool(joint.GetJointEnabledAttr().Get())
        owners=[]
        for relation in (joint.GetBody0Rel(),joint.GetBody1Rel()):
            targets=relation.GetTargets()
            owners.append(dynamic_owner(targets[0]) if targets else {'path':None,'rigid_body_enabled':False})
        invalid=enabled and not any(owner['rigid_body_enabled'] for owner in owners)
        record={'path':str(prim.GetPath()),'joint_enabled':enabled,'body_owners':owners,
                'active_between_two_static_bodies':invalid}
        records.append(record)
        if invalid:failures.append(record)
    return {'passed':not failures,'joint_count':len(records),'joints':records,'failures':failures,
            'scope':'Imported environment joint configuration; dynamic robot joints are audited separately.'}


def material_has_texture(material):
    if not material:return False
    for prim in Usd.PrimRange(material.GetPrim()):
        for attr in prim.GetAttributes():
            value=attr.Get()
            if isinstance(value,Sdf.AssetPath):
                if Path(value.path).suffix.lower() in {'.png','.jpg','.jpeg','.exr','.tif','.tiff','.dds','.tga'}:
                    return True
    return False


def mesh_checks(stage, imported_roots):
    counts=Counter(); missing_normals=[]; missing_uv=[]; missing_material=[]; malformed=[]
    for prim in Usd.PrimRange.Stage(stage,Usd.TraverseInstanceProxies()):
        if not prim.IsA(UsdGeom.Mesh):continue
        imageable=UsdGeom.Imageable(prim)
        if imageable.ComputeVisibility()=='invisible' or imageable.ComputePurpose() not in ('default','render'):
            counts['hidden_or_proxy_meshes']+=1;continue
        path=str(prim.GetPath());mesh=UsdGeom.Mesh(prim)
        imported=path.startswith('/World/Robot/') or any(path.startswith(root+'/') for root in imported_roots)
        counts['visible_meshes']+=1
        if imported:counts['imported_visible_meshes']+=1
        points=mesh.GetPointsAttr().Get();indices=mesh.GetFaceVertexIndicesAttr().Get();sizes=mesh.GetFaceVertexCountsAttr().Get()
        if points is None or indices is None or sizes is None:
            malformed.append({'path':path,'error':'missing points or topology'})
        elif not len(points) or any(not finite(p) for p in points):
            malformed.append({'path':path,'error':'empty or nonfinite vertex array'})
        elif sum(sizes)!=len(indices) or any(n<3 for n in sizes) or any(i<0 or i>=len(points) for i in indices):
            malformed.append({'path':path,'error':'invalid face vertex indices/counts'})
        normals=mesh.GetNormalsAttr().Get()
        if normals is None or not len(normals):
            missing_normals.append(path)
            if imported:counts['imported_missing_authored_normals']+=1
        else:
            counts['authored_normals']+=1
            if any(not finite(n) for n in normals):malformed.append({'path':path,'error':'nonfinite authored normals'})
        uv=UsdGeom.PrimvarsAPI(prim).FindPrimvarWithInheritance('st')
        uv_value=uv.Get() if uv else None
        # Some importers use a non-st UV primvar; retain this distinction in counts.
        alternative=[p for p in UsdGeom.PrimvarsAPI(prim).GetPrimvars()
            if p.GetTypeName() in (Sdf.ValueTypeNames.TexCoord2fArray,Sdf.ValueTypeNames.Float2Array)
            and p.Get() is not None]
        has_uv=(uv_value is not None and len(uv_value)>0) or bool(alternative)
        if not has_uv:
            missing_uv.append(path)
            if imported:counts['imported_missing_uv']+=1
        else:counts['has_uv']+=1
        material,_=UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()
        if not material:
            missing_material.append(path)
            if imported:counts['imported_missing_material']+=1
        elif material_has_texture(material) and not has_uv:
            counts['textured_material_without_mesh_uv']+=1
    return {'counts':dict(counts),'missing_authored_normals':missing_normals,
        'missing_uv':missing_uv,'missing_material':missing_material,'malformed':malformed,
        'passed':not malformed and not missing_material,
        'notes':['Absent authored normals are reported; valid USD meshes may generate normals at render time.',
                 'Missing UVs are reported separately; a constant or procedurally mapped material may not need them.',
                 'This records bindings and geometry; native image inspection must evaluate appearance.']}


def existing_asset(path):
    value=str(path)
    if Ar.IsPackageRelativePath(value):value=Ar.SplitPackageRelativePathOuter(value)[0]
    if '<UDIM>' in value:return bool(glob.glob(value.replace('<UDIM>','[0-9][0-9][0-9][0-9]')))
    return Path(value).is_file()


def resolve_runtime_mdl(authored, search_dirs):
    """Resolve a real native MDL module file, without ignoring missing modules."""
    path=Path(str(authored))
    if path.suffix.lower()!='.mdl' or path.is_absolute() or '..' in path.parts:
        return None
    for directory in search_dirs:
        candidate=(directory/path).resolve()
        if candidate.is_file() and candidate.is_relative_to(directory.resolve()):
            return {'authored':str(authored),'resolved':str(candidate),
                    'sha256':sha256(candidate),'bytes':candidate.stat().st_size,
                    'search_directory':str(directory.resolve()),
                    'resolution':'Verified local Isaac Sim MDL module search directory'}
    return None


def dependency_checks(scene_path, stage, mdl_search_dirs=()):
    layers,assets,unresolved=UsdUtils.ComputeAllDependencies(str(scene_path))
    layer_records=[];asset_records=[];failures=[];mdl_resolved={};still_unresolved=[]
    for layer in layers:
        p=layer.realPath or layer.identifier
        exists=existing_asset(p)
        layer_records.append({'path':p,'exists':exists})
        if not exists:failures.append({'kind':'layer','path':p})
    for p in assets:
        p=str(p);exists=existing_asset(p)
        asset_records.append({'path':p,'exists':exists})
        if not exists:failures.append({'kind':'asset','path':p})
    for p in unresolved:
        resolved=resolve_runtime_mdl(p,mdl_search_dirs)
        if resolved:
            mdl_resolved[str(p)]=resolved
        else:
            still_unresolved.append(str(p))
            failures.append({'kind':'unresolved_dependency','path':str(p)})
    # Inspect authored file inputs too. ComputeAllDependencies includes layer arcs
    # and payloads; this catches texture paths authored directly on shader inputs.
    attribute_files=[]
    for prim in stage.Traverse():
        for attr in prim.GetAttributes():
            value=attr.Get()
            if not isinstance(value,Sdf.AssetPath) or not value.path:continue
            candidate=value.resolvedPath
            if not candidate:
                stack=attr.GetPropertyStack()
                candidate=Sdf.ComputeAssetPathRelativeToLayer(stack[0].layer,value.path) if stack else value.path
                candidate=str(Ar.GetResolver().Resolve(candidate)) or candidate
            exists=existing_asset(candidate)
            mdl_fallback=None
            if not exists:
                mdl_fallback=resolve_runtime_mdl(value.path,mdl_search_dirs)
                if mdl_fallback:
                    mdl_resolved[value.path]=mdl_fallback
                    candidate=mdl_fallback['resolved'];exists=True
            attribute_files.append({'attribute':str(attr.GetPath()),'authored':value.path,
                                    'resolved':candidate,'exists':exists,
                                    'runtime_mdl_resolution':mdl_fallback})
            if not exists:failures.append({'kind':'asset_attribute','attribute':str(attr.GetPath()),
                                          'path':candidate})
    return {'layers':layer_records,'assets':asset_records,'asset_attributes':attribute_files,
            'usd_default_resolver_unresolved':[str(x) for x in unresolved],
            'verified_runtime_mdl_modules':list(mdl_resolved.values()),
            'mdl_search_directories':[str(p.resolve()) for p in mdl_search_dirs],
            'unresolved':still_unresolved,'failures':failures,'passed':not failures,
            'scope':'USD layers and file dependencies exist; MDL module presence is verified locally, not compiled by this CPU audit.'}


def import_grounding(stage,cache,description):
    results=[];roots=[]
    for item in description.get('geometry',[]):
        if item.get('kind') not in ('asset','mesh_asset'):continue
        path='/World/Geometry/'+safe(item['name']);roots.append(path)
        actual=bounds(stage,cache,path)
        expected=float(item['position'][2]);delta=actual['min_m'][2]-expected
        results.append({'name':item['name'],'source':item.get('source'),
            'requested_support_z_m':expected,'actual_visible_bounds':actual,
            'bottom_offset_m':delta,'tolerance_m':1.e-4,'passed':abs(delta)<=1.e-4})
    return {'items':results,'count':len(results),'passed':all(x['passed'] for x in results),
        'scope':'Imported visible geometry bottom vs its declared support plane; not arbitrary-object contact validation.'},roots


def compact_layout_checks(stage,cache,description,grounding):
    """Measure the room shell, saved cameras and explicitly classified furniture."""
    spec=description['layout']['room']
    floor=visual_bound(stage,cache,'room_floor')
    ceiling=visual_bound(stage,cache,'room_ceiling')
    table=visual_bound(stage,cache,'worktop')
    xmin,ymin=floor['min_m'][:2];xmax,ymax=floor['max_m'][:2]
    floor_z=floor['max_m'][2];ceiling_z=ceiling['min_m'][2]
    measured=[xmin,xmax,ymin,ymax]
    requested=list(map(float,spec['bounds']))
    area=(xmax-xmin)*(ymax-ymin)
    room_errors=[]
    if not 14.<=area<=18.:room_errors.append('actual interior floor area is outside 14 to 18 square metres')
    if len(requested)!=4 or any(abs(a-b)>1e-4 for a,b in zip(measured,requested)):
        room_errors.append('actual floor vertices disagree with declared room bounds')
    if abs(ceiling_z-floor_z-float(spec['height']))>1e-4:
        room_errors.append('actual clear height disagrees with declared room height')
    room={'floor':floor,'ceiling':ceiling,'measured_bounds_xy_m':measured,
          'declared_bounds_xy_m':requested,'floor_area_m2':area,'required_area_range_m2':[14.,18.],
          'clear_height_m':ceiling_z-floor_z,'passed':not room_errors,'errors':room_errors}
    xforms=UsdGeom.XformCache(Usd.TimeCode.Default());cameras=[]
    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Camera):continue
        eye=list(map(float,xforms.GetLocalToWorldTransform(prim).ExtractTranslation()))
        distances=[eye[0]-xmin,xmax-eye[0],eye[1]-ymin,ymax-eye[1],
                   eye[2]-floor_z,ceiling_z-eye[2]]
        camera=UsdGeom.Camera(prim);focal=camera.GetFocalLengthAttr().Get()
        cameras.append({'path':str(prim.GetPath()),'actual_world_eye_m':eye,
                        'inside_clear_room':min(distances)>.001,
                        'signed_distances_to_room_faces_m':distances,
                        'focal_length_mm':focal,
                        'passed':min(distances)>.001 and focal is not None and float(focal)>0})
    camera_check={'items':cameras,'count':len(cameras),'passed':bool(cameras) and all(c['passed'] for c in cameras),
                  'scope':'Saved USD camera centres inside measured room shell; not a frustum or occlusion test.'}
    if not cameras:camera_check['error']='No actual saved USD cameras were found'

    # Classification identifies which real meshes form a piece of furniture;
    # declarations never provide the measured bounds or a collision verdict.
    groups=[r for r in description.get('geometry',[]) if r.get('kind')=='interior_group']
    known={r['actual_visible_bounds']['path']:r['actual_visible_bounds'] for r in grounding['items']}
    furniture=[];excluded=[];failures=[];seen=set()
    roles={'floor_furniture','wall_fixture','architectural','supported_decor'}
    for group in groups:
        name=group.get('name');role=group.get('role');paths=group.get('prim_paths',[])
        if role not in roles or (role=='floor_furniture' and not paths):
            failures.append({'name':name,'error':'missing or invalid interior role / prim_paths'});continue
        if role!='floor_furniture':
            excluded.append({'name':name,'role':role,'prim_paths':paths});continue
        parts=[];errors=[]
        for path in paths:
            if path in seen:errors.append('furniture prim appears in more than one floor group: '+path)
            seen.add(path)
            try:
                part=known.get(path)
                if part is None:part=bounds(stage,cache,path)
                parts.append(part)
            except ValueError as exc:errors.append(str(exc))
        if not parts:
            failures.append({'name':name,'error':'furniture group has no measurable geometry','details':errors});continue
        lo=[min(p['min_m'][i] for p in parts) for i in range(3)]
        hi=[max(p['max_m'][i] for p in parts) for i in range(3)]
        actual={'min_m':lo,'max_m':hi,'size_m':[hi[i]-lo[i] for i in range(3)]}
        wall_clearances=[lo[0]-xmin,xmax-hi[0],lo[1]-ymin,ymax-hi[1]]
        overlap=xy_overlap(actual,table)
        if min(wall_clearances)<-.002:errors.append('visible floor furniture extends beyond measured room boundary')
        if min(overlap)>.002:errors.append('floor furniture footprint overlaps the actual worktable footprint')
        record={'name':name,'role':role,'prim_paths':paths,'actual_visible_bounds':actual,
                'signed_wall_clearances_m':wall_clearances,'worktable_xy_overlap_m':overlap,
                'tolerance_m':.002,'passed':not errors,'errors':errors}
        furniture.append(record)
        if errors:failures.append({'name':name,'errors':errors})
    if not groups:failures.append({'error':'No explicit interior_group classifications were found'})
    if not furniture:failures.append({'error':'No floor_furniture groups were measured'})
    furniture_check={'items':furniture,'count':len(furniture),'excluded_groups':excluded,
        'failures':failures,'passed':not failures,
        'scope':'Visible floor-furniture AABB containment and worktable footprint separation; wall fixtures, architectural elements and supported decor explicitly excluded. No pairwise mesh-contact test.'}
    return {'room':room,'cameras':camera_check,'furniture':furniture_check,
            'passed':room['passed'] and camera_check['passed'] and furniture_check['passed']}


def joint_frame(joint, side):
    position=getattr(joint,'GetLocalPos'+str(side)+'Attr')().Get()
    rotation=getattr(joint,'GetLocalRot'+str(side)+'Attr')().Get()
    matrix=Gf.Matrix4d(1.)
    matrix.SetRotate(Gf.Rotation(Gf.Quatd(rotation)))
    matrix.SetTranslateOnly(Gf.Vec3d(position))
    return matrix


def robot_pose_checks(stage):
    """Compare the reopened visible geometry with its saved physics joint state."""
    transforms=UsdGeom.XformCache(Usd.TimeCode.Default())
    records=[];failures=[];roots=[];scenes=[]
    robot_root=Sdf.Path('/World/Robot');excluded_environment_joints=[]
    axes={'X':Gf.Vec3d(1,0,0),'Y':Gf.Vec3d(0,1,0),'Z':Gf.Vec3d(0,0,1)}
    for prim in stage.Traverse():
        if prim.IsA(UsdPhysics.Scene):scenes.append(str(prim.GetPath()))
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI):roots.append(str(prim.GetPath()))
        if not prim.IsA(UsdPhysics.RevoluteJoint):continue
        if not prim.GetPath().HasPrefix(robot_root):
            excluded_environment_joints.append(str(prim.GetPath()))
            continue
        joint=UsdPhysics.RevoluteJoint(prim);name=prim.GetName()
        state=prim.GetAttribute('state:angular:physics:position').Get()
        drive=UsdPhysics.DriveAPI(prim,'angular')
        target=drive.GetTargetPositionAttr().Get() if drive else None
        max_force=drive.GetMaxForceAttr().Get() if drive else None
        raw_schemas=prim.GetMetadata('apiSchemas')
        record={'name':name,'path':str(prim.GetPath()),'saved_position_deg':state,
            'drive_target_deg':target,'drive_max_force':max_force,
            'authored_api_schemas':list(raw_schemas.GetAppliedItems()) if raw_schemas else []}
        errors=[]
        if state is None or not math.isfinite(float(state)):errors.append('missing/nonfinite saved joint position')
        if target is None or max_force is None or not finite([target,max_force]) or max_force<=0:
            errors.append('missing/invalid hold drive')
        if not prim.HasAPI(UsdPhysics.DriveAPI,'angular'):errors.append('angular drive API absent')
        if not any('JointStateAPI:angular' in schema for schema in record['authored_api_schemas']):
            errors.append('angular JointState API absent')
        frames=[]
        for side in (0,1):
            targets=getattr(joint,'GetBody'+str(side)+'Rel')().GetTargets()
            body=stage.GetPrimAtPath(targets[0]) if len(targets)==1 else None
            if not body:
                errors.append('missing body'+str(side));break
            frames.append(joint_frame(joint,side)*transforms.GetLocalToWorldTransform(body))
        if len(frames)==2 and state is not None:
            relative=frames[1]*frames[0].GetInverse()
            rotation=relative.ExtractRotation();axis=axes[str(joint.GetAxisAttr().Get())]
            angle=float(rotation.GetAngle())
            dot=float(Gf.Dot(rotation.GetAxis(),axis))
            measured=angle*(1 if dot>=0 else -1)
            difference=math.degrees(math.atan2(math.sin(math.radians(measured-float(state))),
                                               math.cos(math.radians(measured-float(state)))))
            displacement=(frames[0].ExtractTranslation()-frames[1].ExtractTranslation()).GetLength()
            record.update(measured_geometry_position_deg=measured,
                geometry_minus_saved_position_deg=difference,
                joint_frame_origin_distance_m=float(displacement),axis_alignment_abs=abs(dot))
            if abs(difference)>.12:errors.append('visible link geometry does not match saved joint position')
            if displacement>.001:errors.append('joint frame origins separated by over 1 mm')
            if angle>.5 and abs(dot)<.999:errors.append('relative link rotation is not about the joint axis')
        record['passed']=not errors;record['errors']=errors;records.append(record)
        if errors:failures.append({'joint':name,'errors':errors})
    if len(roots)!=1:failures.append({'error':'Expected one articulation root','roots':roots})
    if len(scenes)!=1:failures.append({'error':'Expected one PhysicsScene','scenes':scenes})
    if len(records)!=29:failures.append({'error':'Expected FR3 + SHARPA 29 revolute joints','count':len(records)})
    return {'passed':not failures,'physics_scenes':scenes,'articulation_roots':roots,
        'robot_root':str(robot_root),'excluded_environment_revolute_joints':excluded_environment_joints,
        'revolute_joint_count':len(records),'joints':records,'failures':failures,
        'geometry_angle_tolerance_deg':.12,'joint_frame_distance_tolerance_m':.001,
        'pure_usd_play_tested':False,
        'scope':'Saved API configuration and actual reopened link transforms only; native initialization and unsupported script gravity compensation are not replayed.'}


def audit(scene_path, mdl_search_dirs=()):
    scene_path=scene_path.resolve();stage=Usd.Stage.Open(str(scene_path))
    if not stage:raise RuntimeError('USD stage could not be opened')
    cache=UsdGeom.BBoxCache(Usd.TimeCode.Default(),['default','render'],useExtentsHint=False)
    description_path=scene_path.parent/'scene_description.json'
    description=load_json(description_path) if description_path.exists() else {}
    result={'schema':'scence.independent_usd_audit.v1',
        'created_at_utc':datetime.now(timezone.utc).isoformat(),
        'audit_implementation':{'path':str(Path(__file__).resolve()),'sha256':sha256(__file__)},
        'scene':{'path':str(scene_path),'sha256':sha256(scene_path),
                 'meters_per_unit':UsdGeom.GetStageMetersPerUnit(stage),'up_axis':str(UsdGeom.GetStageUpAxis(stage))},
        'scope':'Independent composed geometry and dependency audit; no task qualification.',
        'task_qualification':None,'manipulation_validated':False}
    checks={}
    checks['units']={'passed':UsdGeom.GetStageMetersPerUnit(stage)==1. and UsdGeom.GetStageUpAxis(stage)=='Z'}
    checks['construction']=construction_checks(stage,cache)
    checks['colliders']=collider_checks(stage)
    checks['environment_joints']=environment_joint_checks(stage)
    checks['import_grounding'],roots=import_grounding(stage,cache,description)
    if description.get('layout',{}).get('room'):
        checks['compact_layout']=compact_layout_checks(stage,cache,description,checks['import_grounding'])
    checks['meshes']=mesh_checks(stage,roots)
    checks['dependencies']=dependency_checks(scene_path,stage,mdl_search_dirs)
    if description_path.exists():
        result['scene_description']={'path':str(description_path),'sha256':sha256(description_path)}
    else:
        checks['import_grounding']['passed']=False
        checks['import_grounding']['error']='Adjacent scene_description.json missing; imported asset support targets unverified.'
    stability_path=scene_path.parent/'stability.json'
    if stability_path.exists():
        stability=load_json(stability_path)
        result['native_passive_stability']={'path':str(stability_path),'sha256':sha256(stability_path),
            'reported_status':stability.get('status'),'scope':stability.get('scope'),
            'actual_native_steps':stability.get('actual_native_steps'),
            'max_root_translation_m':stability.get('max_root_translation_m'),
            'max_joint_deviation_rad':stability.get('max_joint_deviation_rad'),
            'note':'Associated native evidence; this CPU audit does not rerun dynamics.'}
        checks['saved_robot_pose']=robot_pose_checks(stage)
    else:result['native_passive_stability']=None
    result['checks']=checks
    result['status']='passed' if all(item['passed'] for item in checks.values()) else 'failed'
    result['failed_checks']=[name for name,item in checks.items() if not item['passed']]
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scene',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--mdl-search-dir',action='append',type=Path,
        help='Verified native MDL search directory; may be repeated. Defaults to the running Isaac Sim core Base directory.')
    args=parser.parse_args()
    try:
        mdl_dirs=args.mdl_search_dir if args.mdl_search_dir is not None else [
            Path(os.environ['ISAAC_PATH'])/'kit/mdl/core/Base']
        result=audit(args.scene,mdl_dirs)
    except Exception as exc:
        import traceback
        result={'schema':'scence.independent_usd_audit.v1','status':'error',
                'scene':str(args.scene),'error':repr(exc),'traceback':traceback.format_exc(),
                'task_qualification':None,'manipulation_validated':False}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'status':result['status'],'output':str(args.output),
                      'failed_checks':result.get('failed_checks',[]),'error':result.get('error')}))
    return 0 if result['status']=='passed' else 1


if __name__=='__main__':
    sys.exit(main())
