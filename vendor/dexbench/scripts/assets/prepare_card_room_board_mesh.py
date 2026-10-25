"""Author an opt-in exact-box-surface triangle mesh for the task0 board.

Only CPU mesh authoring occurs. The original board, visuals and physical
material remain pinned. Native mesh cooking, contacts and manipulation are
unqualified until separately checked. No scene or release is activated here.
"""
from copy import deepcopy
import argparse
import json
from pathlib import Path
import shutil

import numpy as np

from bench.isaacsim.release import digest, dump
from bench.paths import DATA_ROOT
from bench.simulation.isaacsim_backend.asset_catalog import load_task_asset, load_obj_mesh

SOURCE = Path(f'{DATA_ROOT}/assets/task_parts/board_games/tic_tac_toe_3x3/v3')
LOWER = np.array([-.23, -.23, 0.])
UPPER = np.array([.23, .23, .012])


def validate_original(asset):
    expected = {'geometry': 'box', 'position_m': [0, 0, .006],
                'half_extents_m': [.23, .23, .006],
                'quaternion_xyzw': [0., 0., 0., 1.], 'scale': [1, 1, 1]}
    parts = asset['collision_parts']
    if len(parts) != 1 or any(parts[0].get(k) != v for k, v in expected.items()):
        raise ValueError('Board candidate requires the original exact unscaled box')
    if asset['dynamic'] is not False or asset['mass_kg'] != 0.:
        raise ValueError('Board source must remain an immovable fixture')


def box_surface(cell_size_m=.01):
    """Six coplanar rectangular grids sharing every edge vertex; no bevels."""
    if type(cell_size_m) not in (int, float) or cell_size_m not in (.005, .01):
        raise ValueError('Explicit 5 mm or 10 mm board surface cells required')
    cells = np.ceil((UPPER - LOWER)/cell_size_m - 1e-10).astype(int)
    axes = [np.linspace(lo, hi, n+1) for lo, hi, n in zip(LOWER, UPPER, cells)]
    vertices, indices, faces = [], {}, []

    def vertex(index):
        index = tuple(index)
        if index not in indices:
            indices[index] = len(vertices)
            vertices.append([axes[i][index[i]] for i in range(3)])
        return indices[index]

    for fixed in range(3):
        u, v = [i for i in range(3) if i != fixed]
        for side in (0, int(cells[fixed])):
            desired = -1 if side == 0 else 1
            for i in range(cells[u]):
                for j in range(cells[v]):
                    ids = []
                    for du, dv in ((0, 0), (1, 0), (1, 1), (0, 1)):
                        index = [0, 0, 0]
                        index[fixed], index[u], index[v] = side, i+du, j+dv
                        ids.append(vertex(index))
                    for triangle in ((ids[0], ids[1], ids[2]), (ids[0], ids[2], ids[3])):
                        a, b, c = np.asarray([vertices[k] for k in triangle])
                        if np.cross(b-a, c-a)[fixed]*desired < 0:
                            triangle = (triangle[0], triangle[2], triangle[1])
                        faces.append(triangle)
    return np.asarray(vertices), np.asarray(faces, dtype=np.int32)


def geometry_review(vertices, faces):
    import trimesh
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    triangles = np.asarray(vertices)[faces]
    on_face = np.any(np.all(np.isclose(triangles, LOWER, atol=1e-14, rtol=0), axis=1)
                     | np.all(np.isclose(triangles, UPPER, atol=1e-14, rtol=0), axis=1), axis=1)
    if (not mesh.is_watertight or not mesh.is_winding_consistent or not mesh.is_volume
            or mesh.euler_number != 2 or len(mesh.split(only_watertight=False)) != 1
            or not np.all(on_face) or not np.array_equal(mesh.bounds, [LOWER, UPPER])
            or not np.isclose(mesh.volume, np.prod(UPPER-LOWER), atol=1e-15, rtol=0)):
        raise ValueError('Board grid must preserve the complete closed box exterior')
    return {'vertices': len(vertices), 'triangles': len(faces),
            'bounds_m': mesh.bounds.tolist(), 'volume_m3': float(mesh.volume),
            'watertight': True, 'outward_consistent': True, 'connected_components': 1,
            'all_triangles_on_original_box_faces': True,
            'maximum_float32_vertex_error_m': float(np.linalg.norm(vertices-vertices.astype(np.float32), axis=1).max()),
            'native_cooked_geometry_verified': False}


def prepare(output, *, source=SOURCE, cell_size_m=.01):
    output, source = Path(output).resolve(), Path(source).resolve()
    if output.exists():
        raise FileExistsError('Use a new immutable board candidate directory')
    original = load_task_asset('board', source, verify_hashes=True)
    validate_original(original)
    vertices, faces = box_surface(cell_size_m)
    geometry = geometry_review(vertices, faces)
    asset_id = 'board_triangle_grid'+str(round(cell_size_m*1000))+'mm'
    output.mkdir(parents=True)
    obj = output/'collision_surface.obj'
    with obj.open('w') as stream:
        stream.write('# Exact original box boundary; coplanar grid triangulation only.\n')
        for vertex in vertices:
            stream.write('v '+' '.join(format(float(x), '.17g') for x in vertex)+'\n')
        for face in faces:
            stream.write('f '+' '.join(str(int(x)+1) for x in face)+'\n')
    dependencies = {str(source/'manifest.json'): digest(source/'manifest.json'),
                    str(source/'LICENSE.txt'): digest(source/'LICENSE.txt'),
                    str(obj): digest(obj), original['mesh_path']: original['sha256']}
    dependencies.update({part['mesh_path']: part['sha256'] for part in original['visual_parts']})
    shutil.copyfile(source/'LICENSE.txt', output/'LICENSE.txt')
    dependencies[str(output/'LICENSE.txt')] = digest(output/'LICENSE.txt')
    provenance = {'schema': 'card-room-board-triangle-candidate.v1',
        'source_root': str(source), 'source_asset_id': 'board', 'source_files_modified': False,
        'generator_sha256': digest(__file__), 'source_collision_parts': deepcopy(original['collision_parts']),
        'geometry': geometry, 'cell_size_limit_m': cell_size_m,
        'original_visual_parts': deepcopy(original['visual_parts']),
        'source_visual_points_triangles_materials_preserved': True,
        'material_prior': {'static_friction': .4, 'dynamic_friction': .4, 'restitution': 0., 'friction_combine': 'average'},
        'contact_offset_m': .001, 'rest_offset_m': 0., 'dependencies_sha256': dependencies,
        'body_mode': 'Existing kinematic_mesh_v1 adapter makes this mesh a fixed fixture and guards its pose.',
        'contact_actor_path': '/World/Fixtures/board/collision_000',
        'physics_accepted': False, 'native_runs': 0, 'model_calls': 0,
        'scope': 'Coplanar triangulation preserves closed box surface; native cooking and pair dispatch require separate evidence.'}
    dump(output/'SOURCE.json', provenance)
    dependencies[str(output/'SOURCE.json')] = digest(output/'SOURCE.json')
    asset = deepcopy(original)
    for key in ('asset_id', 'asset_root', 'manifest_path', 'manifest_sha256'):
        asset.pop(key, None)
    asset.update(label='Task0 board exact-box-surface triangle candidate',
        collision_parts=[{'geometry': 'triangle_mesh', 'mesh_path': str(obj), 'sha256': digest(obj),
                          'position_m': [0., 0., 0.], 'quaternion_xyzw': [0., 0., 0., 1.],
                          'scale': [1, 1, 1], 'label': 'solid_board', 'classification': 'physics'}],
        source_dependencies_sha256=dependencies,
        bounds_min_m=LOWER.tolist(), bounds_max_m=UPPER.tolist(), dimensions_m=(UPPER-LOWER).tolist(),
        material_prior=provenance['material_prior'], contact_offset_m=.001, rest_offset_m=0.,
        qualification='Prepared unvalidated; native contacts and cooked surface require separate review')
    dump(output/'manifest.json', {'schema_version': 'scene_family_asset_library_v1',
         'asset_version': output.name, 'units': 'metres', 'quaternion_order': 'xyzw',
         'assets': {asset_id: asset}, 'license': 'CC0-1.0', 'generator_sha256': digest(__file__)})
    load_task_asset(asset_id, output, verify_hashes=True)
    return {'asset_root': str(output), 'asset_id': asset_id, 'manifest_sha256': digest(output/'manifest.json'),
            'geometry': geometry, 'physics_accepted': False}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--source', type=Path, default=SOURCE)
    parser.add_argument('--cell-size-m', type=float, choices=(.005, .01), default=.01)
    args = parser.parse_args()
    print(json.dumps(prepare(args.output, source=args.source, cell_size_m=args.cell_size_m), indent=2))
