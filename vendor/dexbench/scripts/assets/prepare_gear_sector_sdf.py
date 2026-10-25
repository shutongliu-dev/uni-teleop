"""Author a new eight-sector SDF1250 gear catalog; preserve all physical priors."""
from copy import deepcopy
import argparse
import json
import math
from pathlib import Path
import numpy as np

from bench.isaacsim.release import digest, dump
from bench.paths import load_json
from scripts.assets.prepare_gear_sdf import prepare as prepare_surface_base, GEARS
from scripts.validation.gear_sector_geometry import partition, yaw_matrix

SECTOR_COUNT = 8
PARAMETERS = {'sdfResolution': 1250, 'sdfSubgridResolution': 6,
    'sdfBitsPerSubgridPixel': 'BitsPerPixel16', 'sdfEnableRemeshing': False,
    'sdfTriangleCountReductionFactor': 1.}
ASSET_SUFFIX = '_sdf1250_sectors08'


def prepare(output):
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade
    import trimesh
    output = Path(output).resolve()
    # This creates a new directory only. Intermediate authoring is never
    # published/frozen; the original single-SDF candidates remain untouched.
    prepare_surface_base(output)
    base = load_json(output/'SOURCE.json')
    manifest = load_json(output/'manifest.json')
    surface = load_json(output/'surface/cpu_geometry.json')
    entries, reports = {}, {}
    piece_root = output/'sectors'
    piece_root.mkdir()
    cached = {}
    for name in GEARS:
        source_path = Path(surface['assets'][name]['mesh_path'])
        source_mesh = trimesh.load(source_path, force='mesh', process=False)
        key = digest(source_path)
        if key not in cached:
            cached[key] = partition(source_mesh, SECTOR_COUNT)
        parts, proof = cached[key]
        proof = deepcopy(proof)
        usd_path = output/(name+'.usda')
        stage = Usd.Stage.Open(str(usd_path))
        # Save body/material/visual authored state for exact preservation checks.
        preserved = {str(p.GetPath()): {a.GetName(): a.Get() for a in p.GetAttributes()}
            for p in stage.Traverse() if str(p.GetPath()) != '/Gear/collision'}
        stage.RemovePrim('/Gear/collision')
        physical = UsdShade.Material(stage.GetPrimAtPath('/Gear/physics_material'))
        asset = deepcopy(manifest['assets'][name+'_sdf1024'])
        asset['collision_parts'] = []
        pieces = []
        for part, row in zip(parts, proof['sectors'], strict=True):
            index = row['index']
            rotation = yaw_matrix(row['local_frame_yaw_deg'])
            local = trimesh.Trimesh(vertices=part.vertices@rotation, faces=part.faces, process=False)
            path = '/Gear/collision_'+str(index).zfill(2)
            geom = UsdGeom.Mesh.Define(stage, path)
            geom.CreatePointsAttr(local.vertices.tolist())
            geom.CreateFaceVertexCountsAttr([3]*len(local.faces))
            geom.CreateFaceVertexIndicesAttr(local.faces.reshape(-1).tolist())
            geom.CreateSubdivisionSchemeAttr('none')
            geom.CreateVisibilityAttr('invisible')
            geom.AddRotateZOp(UsdGeom.XformOp.PrecisionDouble).Set(row['local_frame_yaw_deg'])
            prim = geom.GetPrim()
            UsdPhysics.CollisionAPI.Apply(prim).CreateCollisionEnabledAttr(True)
            UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr('sdf')
            previous = prim.GetMetadata('apiSchemas')
            prim.SetMetadata('apiSchemas', Sdf.TokenListOp.CreateExplicit([
                *previous.GetAppliedItems(), 'PhysxCollisionAPI', 'PhysxSDFMeshCollisionAPI']))
            for key, value in {'contactOffset': .0001, 'restOffset': 0.}.items():
                prim.CreateAttribute('physxCollision:'+key, Sdf.ValueTypeNames.Float, custom=False).Set(value)
            for key, value in PARAMETERS.items():
                kind = {bool: Sdf.ValueTypeNames.Bool, int: Sdf.ValueTypeNames.Int,
                    float: Sdf.ValueTypeNames.Float, str: Sdf.ValueTypeNames.Token}[type(value)]
                prim.CreateAttribute('physxSDFMeshCollision:'+key, kind, custom=False).Set(value)
            UsdShade.MaterialBindingAPI.Apply(prim).Bind(physical, UsdShade.Tokens.weakerThanDescendants, 'physics')
            mesh_path = piece_root/(name+'_'+str(index).zfill(2)+'.obj')
            mesh_path.write_text(trimesh.exchange.obj.export_obj(local, digits=17))
            proof_path = piece_root/(name+'_'+str(index).zfill(2)+'_evidence.npz')
            cap_mask = np.arange(len(part.faces)) >= row['exterior_face_count']
            cap_faces = part.faces[cap_mask]
            cap_edges = np.sort(cap_faces[:, [[0,1],[1,2],[2,0]]].reshape(-1,2), axis=1)
            unique, counts = np.unique(cap_edges, axis=0, return_counts=True)
            seam_edges = unique[counts == 1]
            c = row['calibration']
            np.savez_compressed(proof_path, vertices_body_m=part.vertices,
                vertices_local_m=local.vertices, faces=part.faces, cap_mask=cap_mask,
                seam_edges=seam_edges, mesh_to_body_rotation=rotation,
                calibration_points_local_m=np.asarray(c['points_body_m'])@rotation,
                calibration_gradients_local=np.asarray(c['expected_gradient_body'])@rotation,
                calibration_expected_m=np.asarray(c['expected_signed_distance_m']))
            angle = math.radians(row['local_frame_yaw_deg'])/2
            asset['collision_parts'].append({'usd_prim_path': path, 'approximation': 'sdf',
                'geometry': 'usd_sdf', 'position_m': [0.,0.,0.],
                'quaternion_xyzw': [0.,0.,math.sin(angle),math.cos(angle)]})
            for file in (mesh_path, proof_path):
                asset['source_dependencies_sha256'][str(file)] = digest(file)
            pieces.append({'index': index, 'usd_prim_path': path, 'mesh_path': str(mesh_path),
                'evidence_path': str(proof_path), 'mesh_to_body_rotation': rotation.tolist(),
                **row})
        stage.GetRootLayer().Save()
        reopened = Usd.Stage.Open(str(usd_path))
        for path, attributes in preserved.items():
            prim = reopened.GetPrimAtPath(path)
            if not prim or {a.GetName(): a.Get() for a in prim.GetAttributes()} != attributes:
                raise ValueError('Original body/material/visual changed: '+path)
        maximum_roundoff = 0.
        for part, piece in zip(parts, pieces, strict=True):
            geom = UsdGeom.Mesh(reopened.GetPrimAtPath(piece['usd_prim_path']))
            rotation = np.asarray(piece['mesh_to_body_rotation'])
            local = np.asarray(geom.GetPointsAttr().Get(), dtype=float)
            matrix = np.asarray(UsdGeom.Xformable(geom).GetLocalTransformation())
            if not np.allclose(matrix[:3,:3], rotation.T, rtol=0., atol=1.e-14):
                raise ValueError('Inverse local-frame transform changed')
            if not np.array_equal(geom.GetFaceVertexIndicesAttr().Get(), part.faces.reshape(-1)):
                raise ValueError('Original clipped topology changed')
            error = float(np.linalg.norm(local@rotation.T-part.vertices, axis=1).max())
            maximum_roundoff = max(error, maximum_roundoff)
        if maximum_roundoff > 1.e-8:
            raise ValueError('USD local coordinates exceed float32 roundoff budget')
        asset.update(label=name+'_eight_sector_sdf1250_candidate', sector_pieces=pieces,
            qualification='Unvalidated eight-sector geometry candidate; no dynamics/task qualification')
        asset['physics_assumptions']['surface'] = 'Exact clipped original union in eight closed radial sectors; buried caps and compound contact risks require separate native evidence.'
        asset['physics_assumptions']['scope'] = 'SDF1250 sector geometry and contact dynamics remain unvalidated.'
        asset['source_dependencies_sha256'][str(usd_path)] = digest(usd_path)
        entries[name+ASSET_SUFFIX] = asset
        reports[name] = {'source_mesh': str(source_path), 'source_mesh_sha256': digest(source_path),
            'partition_proof': proof, 'usd_mesh_to_body_roundoff_m': maximum_roundoff,
            'body_material_visual_authored_attributes_preserved': True,
            'pieces': pieces, 'usd_sha256': digest(usd_path)}
    dump(output/'cpu_sector_geometry.json', {'passed': True, 'assets': reports,
        'scope': 'CPU partition and authoring only; no cooked geometry or physics qualification.'})
    provenance = {'schema': 'gear_sector_sdf_candidate.v1',
        'source_root': base['source_root'], 'source_manifest_sha256': base['source_manifest_sha256'],
        'base_surface_generator_sha256': base['generator_sha256'],
        'generator_sha256': digest(__file__),
        'sector_geometry_generator_sha256': digest(Path(__file__).parents[1]/'validation/gear_sector_geometry.py'),
        'sector_count': SECTOR_COUNT, 'sdf_parameters': PARAMETERS,
        'cpu_sector_geometry_sha256': digest(output/'cpu_sector_geometry.json'),
        'native_geometry_validated': False, 'physics_accepted': False, 'task_success_claimed': False}
    dump(output/'SOURCE.json', provenance)
    for asset in entries.values():
        for path in (output/'SOURCE.json', output/'cpu_sector_geometry.json'):
            asset['source_dependencies_sha256'][str(path)] = digest(path)
    dump(output/'manifest.json', {'schema': manifest['schema'], 'assets': entries,
        'license': manifest['license'], 'qualification': 'Independent unvalidated engineering candidate'})
    return {'asset_root': str(output), 'manifest_sha256': digest(output/'manifest.json'),
        'native_geometry_validated': False, 'physics_accepted': False}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.output)))
