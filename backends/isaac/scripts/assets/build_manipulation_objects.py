"""Create original reusable manipulation objects with explicit compound physics.

No downloaded geometry. Density-derived mass/inertia are simulation parameters,
not measurements. Collision unions preserve container openings and handle holes.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import numpy as np
import trimesh

from bench.paths import DATA_ROOT

DEFAULT = Path(f'{DATA_ROOT}/assets/task_parts/manipulation_collection/v2')


def box(center, size, color, **extra):
    return {'geometry': 'box', 'position_m': center, 'half_extents_m': (np.array(size)/2).tolist(), 'color': color, **extra}


def cylinder(center, radius, height, color, **extra):
    return {'geometry': 'cylinder', 'position_m': center, 'radius_m': radius, 'half_height_m': height/2, 'color': color, **extra}


def ring(radius, height, thickness, center_z, color, segments=24):
    # Tangent boxes approximate a circular wall without sealing the aperture.
    pieces = []
    for i in range(segments):
        angle = 2*math.pi*i/segments
        radial = radius-thickness/2
        p = box([radial*math.cos(angle),radial*math.sin(angle),center_z],
                [thickness, 2*radial*math.tan(math.pi/segments)*1.01, height],color)
        p['quaternion_xyzw'] = [0,0,math.sin(angle/2),math.cos(angle/2)]
        pieces.append(p)
    return pieces


def geometry(part):
    if part['geometry'] == 'box':
        return trimesh.creation.box(extents=2*np.array(part['half_extents_m']))
    return trimesh.creation.cylinder(radius=part['radius_m'], height=2*part['half_height_m'], sections=48)


def build(root=DEFAULT):
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    blue=[.16,.38,.57];steel=[.55,.58,.62];tan=[.55,.31,.13];rubber=[.10,.13,.14];cream=[.82,.78,.65]
    cup=[cylinder([0,0,.003],.037,.006,blue)]+ring(.04,.085,.006,.0485,blue)
    cup += [box([.055,0,.085],[.032,.013,.012],blue),box([.071,0,.051],[.012,.013,.080],blue),box([.055,0,.017],[.032,.013,.012],blue)]
    tray=[box([0,0,.004],[.25,.18,.008],steel)]
    tray += [box([0,s*.086,.029],[.25,.008,.05],steel) for s in (-1,1)]
    tray += [box([s*.121,0,.029],[.008,.164,.05],steel) for s in (-1,1)]
    catalog={
      'ceramic_mug': (cup, .32, 'dynamic', 'Hollow cup with three-piece open handle; grasp body or handle.'),
      'small_bowl': ([cylinder([0,0,.003],.053,.006,cream)]+ring(.058,.04,.007,.025,cream),.24,'dynamic','Open cylindrical bowl; aperture not convex-filled.'),
      'capped_bottle': ([cylinder([0,0,.065],.030,.13,blue),cylinder([0,0,.142],.020,.024,rubber)],.22,'dynamic','Rigid capped bottle; no fluid.'),
      'steel_can': ([cylinder([0,0,.040],.033,.080,steel),cylinder([0,0,.082],.034,.004,steel)],.16,'dynamic','Closed tin can.'),
      'wooden_block': ([box([0,0,.023],[.075,.045,.046],tan)],.10,'dynamic','Rectangular wooden block.'),
      'rubber_mallet': ([box([0,0,.095],[.024,.027,.19],tan),box([0,0,.175],[.11,.052,.050],rubber)],.36,'dynamic','Two-piece mallet; rigid handle and rubber head friction assumption.'),
      'screwdriver': ([cylinder([0,0,.045],.017,.090,rubber),cylinder([0,0,.12],.004,.060,steel),box([0,0,.155],[.010,.003,.016],steel)],.10,'dynamic','Coaxial screwdriver; compound shaft preserves narrow geometry.'),
      'storage_tray': (tray,1.0,'static','Open tray fixture with bottom and four independent walls.'),
      'cloth_drape_cylinder': ([cylinder([0,0,.10],.060,.20,steel)],1.0,'static','Vertical round support for cloth draping.'),
      'parts_rack': ([box([0,0,.008],[.24,.14,.016],rubber)]+[cylinder([x,0,.043],.010,.070,steel) for x in (-.08,0,.08)],1.0,'static','Three protruding pegs on a base; each has collision.'),
    }
    assets={}
    for name,(parts,mass,mode,description) in catalog.items():
        meshes=[]; volumes=[]; centers=[]; inertias=[]
        for part in parts:
            mesh=geometry(part); q=part.get('quaternion_xyzw',[0,0,0,1]);R=trimesh.transformations.quaternion_matrix([q[3],*q[:3]])[:3,:3]
            volumes.append(float(mesh.volume));centers.append(part['position_m']); inertias.append(R@mesh.moment_inertia@R.T)
            mesh.apply_transform(np.block([[R,np.array(part['position_m'])[:,None]],[np.array([[0,0,0,1]])]]));meshes.append(mesh)
        weights=np.array(volumes)*mass/sum(volumes); centers=np.array(centers)
        com=np.average(centers,axis=0,weights=weights)
        tensor=sum(I*(mass/sum(volumes))+w*(np.dot(c-com,c-com)*np.eye(3)-np.outer(c-com,c-com)) for I,w,c in zip(inertias,weights,centers))
        if np.max(np.abs(tensor-np.diag(np.diag(tensor))))>1e-7:
            # Symmetry-breaking cup handle uses the full COM tensor's diagonal
            # approximation, explicitly reported, never called measured inertia.
            approximation='Diagonal approximation to compound uniform-density inertia at COM; off-diagonal terms recorded.'
        else: approximation='Analytic compound uniform-density inertia at COM.'
        # Dynamic root centered on COM; static root stays at fixture bottom.
        origin=com if mode=='dynamic' else np.zeros(3)
        for part,mesh in zip(parts,meshes):
            part['position_m']=(np.array(part['position_m'])-origin).tolist();mesh.apply_translation(-origin)
            part['roughness']=.6 if name not in ('steel_can','storage_tray','cloth_drape_cylinder') else .3
            part['metallic']=.8 if name in ('steel_can','storage_tray','cloth_drape_cylinder') else .05
        combined=trimesh.util.concatenate(meshes);path=root/(name+'.obj'); data=trimesh.exchange.obj.export_obj(combined,include_normals=False,include_color=False)
        if path.exists() and path.read_text()!=data: raise FileExistsError('Use a new version: '+str(path))
        path.write_text(data)
        bounds=combined.bounds.tolist()
        assets[name]={'name':name,'description':description,'default_motion':mode,'mesh_path':path.name,'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                      'mass_kg':mass,'center_of_mass_m':(com-origin).tolist(),'inertia_diag_kg_m2':np.diag(tensor).tolist(),
                      'full_inertia_tensor_kg_m2':tensor.tolist(),'inertia_derivation':approximation,
                      'visual_parts':parts,'collision_parts':parts,'bounds_m':bounds,
                      'collision_model':'Union of separately colliding analytic solids; not whole-object convex hull.',
                      'materials_calibrated':False}
    manifest={'schema_version':'original-manipulation-assets-v2','source':'Original procedural workcell assets; no external mesh dependency',
              'generator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'units':'m, kg, s','license':'CC0-1.0',
              'physics_limitations':['Density and friction are assumed, not measured. Compound mass sums include small wall overlaps.','Mug/bowl are rigid; no fluid or fracture model.'], 'assets':assets}
    (root/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    (root/'LICENSE.txt').write_text('Original procedural geometry dedicated to the public domain under CC0 1.0 Universal.\nhttps://creativecommons.org/publicdomain/zero/1.0/\n')
    return manifest


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=DEFAULT);a=p.parse_args();m=build(a.output);print(json.dumps({'output':str(a.output),'assets':list(m['assets'])}))
