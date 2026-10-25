"""Compare native convex contact surfaces with their authored convex cells."""
import numpy as np


def convex_surface_error(source, cooked):
    """Symmetric outward halfspace error; redundant coplanar vertices are valid."""
    from scipy.spatial import ConvexHull
    source, cooked = np.asarray(source, float), np.asarray(cooked, float)
    if (source.ndim != 2 or cooked.ndim != 2 or source.shape[1] != 3 or cooked.shape[1] != 3
            or not np.isfinite(source).all() or not np.isfinite(cooked).all()):
        raise ValueError('Finite source and cooked convex vertices are required')
    planes = ConvexHull(source).equations
    actual = ConvexHull(cooked).equations
    return float(max(0., (cooked@planes[:, :3].T+planes[:, 3]).max(),
                     (source@actual[:, :3].T+actual[:, 3]).max()))


def check_convex_geometry(env, *, include_vertices=False):
    from pxr import UsdGeom, UsdPhysics, UsdUtils, PhysicsSchemaTools
    from omni.physx import get_physx_cooking_interface
    from omni.physx.bindings._physx import PhysxCollisionRepresentationResult
    before = env.capture_state()
    stage_id = UsdUtils.StageCache.Get().GetId(env.stage).ToLongInt()
    cooking = get_physx_cooking_interface()
    records = {}
    for group in ('objects', 'fixtures'):
        for name, info in env.scene_info[group].items():
            for path in info['collision_paths']:
                prim = env.stage.GetPrimAtPath(path)
                if UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr().Get() != 'convexHull':
                    raise ValueError('This geometry probe requires explicit convex cells: '+path)
                pieces, status = [], []
                def callback(result, convexes):
                    status.append(result == PhysxCollisionRepresentationResult.RESULT_VALID)
                    for convex in convexes:
                        pieces.append({'vertices': [list(v) for v in convex.vertices],
                            'polygon_count': len(convex.polygons),
                            'maximum_vertices_per_polygon': max(p.num_vertices for p in convex.polygons)})
                cooking.request_convex_collision_representation(stage_id=stage_id,
                    collision_prim_id=PhysicsSchemaTools.sdfPathToInt(prim.GetPath()),
                    run_asynchronously=False, on_result=callback)
                if status != [True] or len(pieces) != 1:
                    raise ValueError('Expected one native cooked convex: '+path)
                piece = pieces[0]
                source = np.asarray(UsdGeom.Mesh(prim).GetPointsAttr().Get())
                # USD stores row-vector transforms. Compare in parent-space
                # metres so mesh-local conditioning cannot change the tolerance.
                linear = np.asarray(UsdGeom.Xformable(prim).GetLocalTransformation(),float)[:3,:3]
                error = convex_surface_error(source @ linear, np.asarray(piece['vertices']) @ linear)
                topology = (len(piece['vertices']) <= 64 and piece['polygon_count'] <= 64
                            and piece['maximum_vertices_per_polygon'] <= 32)
                records[path] = {'passed': bool(error <= 20e-6 and topology),
                    'body': name, 'native_vertex_count': len(piece['vertices']),
                    'native_polygon_count': piece['polygon_count'],
                    'native_maximum_vertices_per_polygon': piece['maximum_vertices_per_polygon'],
                    'maximum_surface_halfspace_error_m': error,
                    'tolerance_m': 20e-6, 'gpu_convex_topology_envelope': bool(topology)}
                if include_vertices:
                    from pxr import PhysxSchema
                    api = PhysxSchema.PhysxConvexHullCollisionAPI(prim)
                    records[path].update(source_vertices_mesh_local=source.tolist(),
                        native_vertices_mesh_local=piece['vertices'],
                        mesh_local_to_parent_linear=linear.tolist(),
                        resolved_min_thickness_m=api.GetMinThicknessAttr().Get(),
                        resolved_hull_vertex_limit=api.GetHullVertexLimitAttr().Get())
    after = env.capture_state()
    unchanged = all(np.array_equal(before[k], after[k]) for k in before)
    if not unchanged:
        raise ValueError('Native cooking observation changed physics state')
    return {'passed': bool(records) and all(r['passed'] for r in records.values()),
        'colliders': records, 'physical_state_unchanged': unchanged,
        'source': 'PhysX native convex collision representation versus USD convex-cell vertices',
        'scope': 'Per-cell boundary and topology; simulation motion independently tests the assembled union.'}
