"""Frozen-source native proof of original mesh contacts on fixed kinematic actors.

Three perforated supports exercise triangle meshes, a disabled-body compound,
convex decomposition and SDF. Cubes must rest on their solid rims while paired
cubes fall through the original holes to the floor. No geometry approximation
replacement or mid-episode physical state writes are permitted.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time
import traceback

from bench.isaacsim.release import dump
from scripts.maintenance.freeze_project_sources import verify_if_snapshot
from scripts.validation.validate_scenario import seal_report_before_shutdown
from scripts.validation.validate_workrooms import THRESHOLDS, assess_objects, check_window, plain, record, support_contact
from bench.simulation.isaacsim_backend.workroom import SCHEMA, SUPPORT_PATHS, digest
from bench.simulation.isaacsim_backend.static_mesh_contacts import MODE, geometry_signature


def _slab(x0, x1, y0, y1):
    points = [(x, y, z) for z in (-.02, .02) for y in (y0, y1) for x in (x0, x1)]
    faces = [(0, 2, 3, 1), (4, 5, 7, 6), (0, 1, 5, 4),
             (2, 6, 7, 3), (0, 4, 6, 2), (1, 3, 7, 5)]
    indices = [v for a, b, c, d in faces for v in (a, b, c, a, c, d)]
    return points, indices


def _closed_ring():
    """One watertight square annulus: 16 shared vertices, 32 outward triangles."""
    outer = [(-.25, -.25), (.25, -.25), (.25, .25), (-.25, .25)]
    inner = [(-.10, -.10), (.10, -.10), (.10, .10), (-.10, .10)]
    points = [(x, y, z) for z in (-.02, .02) for x, y in outer + inner]
    quads = []
    for i in range(4):
        j = (i + 1) % 4
        quads.extend([(8+i, 8+j, 12+j, 12+i),  # top annulus, +Z
                      (i, 4+i, 4+j, j),        # bottom annulus, -Z
                      (i, j, 8+j, 8+i),        # outer wall
                      (4+i, 12+i, 12+j, 4+j)]) # hole wall
    return points, [v for a, b, c, d in quads for v in (a, b, c, a, c, d)]


def _mesh(name, approximation, slabs, indent='        ', translation=None):
    points, indices = [], []
    for slab in slabs:
        p, i = _slab(*slab)
        indices.extend(x + len(points) for x in i)
        points.extend(p)
    if approximation == 'sdf':
        # Touching disconnected boxes leave internal faces and duplicate
        # vertices; SDF needs the actual watertight boundary of their union.
        points, indices = _closed_ring()
    schemas = ['PhysicsCollisionAPI', 'PhysicsMeshCollisionAPI']
    if approximation == 'sdf':
        schemas.append('PhysxSDFMeshCollisionAPI')
    if approximation == 'convexDecomposition':
        schemas.append('PhysxConvexDecompositionCollisionAPI')
    text = [f'{indent}def Mesh "{name}" (prepend apiSchemas = {json.dumps(schemas)}) {{',
            f'{indent}    point3f[] points = [{", ".join(str(p) for p in points)}]',
            f'{indent}    int[] faceVertexCounts = {json.dumps([3]*(len(indices)//3))}',
            f'{indent}    int[] faceVertexIndices = {json.dumps(indices)}',
            f'{indent}    uniform token subdivisionScheme = "none"',
            f'{indent}    uniform token physics:approximation = "{approximation}"',
            f'{indent}    bool physics:collisionEnabled = true',
            f'{indent}    bool doubleSided = true']
    if translation is not None:
        text.extend([f'{indent}    double3 xformOp:translate = {translation}',
                     f'{indent}    uniform token[] xformOpOrder = ["xformOp:translate"]'])
    if approximation == 'sdf':
        text.append(f'{indent}    uniform int physxSDFMeshCollision:sdfResolution = 128')
    if approximation == 'convexDecomposition':
        # Pin cooking in the synthetic source: the SDK default voxel hull can
        # inflate this 40 mm rim by 5.45 mm. Projection retains the source face
        # planes; the runtime adapter must preserve this authored setting.
        text.append(f'{indent}    bool physxConvexDecompositionCollision:shrinkWrap = true')
    text.append(f'{indent}}}')
    return '\n'.join(text)


def make_probe_scene(output):
    """Write only diagnostic input data; safe before SimulationApp/pxr imports."""
    from scenes.workrooms import scene
    spec = scene('mechanical_assembly')
    template = deepcopy(spec['objects'][0])
    half = deepcopy(spec['workroom_diagnostics']['cube_table']['half_extents_m'])
    folder = output / 'fixture_source'
    folder.mkdir()
    usd = folder / 'scene.usda'
    chunks = ['#usda 1.0', '(defaultPrim = "World"; metersPerUnit = 1; upAxis = "Z")',
              'def Xform "World" {', '    def Xform "Robot" {}', '    def Xform "Geometry" {']
    for name, position, scale in [('worktop', (-3., 0., .76), (.5, .5, .02)),
                                  ('room_floor', (0., 0., -.02), (4., 3., .02))]:
        chunks.append(f'''        def Xform "{name}" {{
            def Cube "Collider" (prepend apiSchemas = ["PhysicsCollisionAPI"]) {{
                double size = 2
                double3 xformOp:translate = {position}
                double3 xformOp:scale = {scale}
                uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:scale"]
            }}
        }}''')
    spec['objects'], spec['workroom_diagnostics'] = [], {}
    for index, (name, approximation) in enumerate([('triangle_compound', 'none'),
                                                   ('convex', 'convexDecomposition'), ('sdf', 'sdf')]):
        x = .45 + index*.75
        # Keep the authored mesh near its local centre for stable mass-volume
        # integration. The Xform reproduces the same world-space support.
        slabs = [(-.25, .25, -.25, -.10), (-.25, .25, .10, .25),
                 (-.25, -.10, -.10, .10), (.10, .25, -.10, .10)]
        translation = (x, 0., .78)
        if approximation == 'none':
            chunks.append(f'        def Xform "{name}" (prepend apiSchemas = ["PhysicsRigidBodyAPI"]) {{\n'
                          '            bool physics:rigidBodyEnabled = false\n'
                          f'            double3 xformOp:translate = {translation}\n'
                          '            uniform token[] xformOpOrder = ["xformOp:translate"]')
            chunks.extend(_mesh(f'slab{i}', approximation, [slab], indent='            ') for i, slab in enumerate(slabs))
            chunks.append('        }')
        else:
            chunks.append(_mesh(name, approximation, slabs, translation=translation))
        actor = '/World/Room/Geometry/' + name
        for suffix, y, top, support in [('rim', -.175, .8, actor), ('hole', 0., 0., SUPPORT_PATHS['floor'])]:
            object_id = name + '_' + suffix
            position = [x, y, 1.05]
            spec['objects'].append({**deepcopy(template), 'id': object_id, 'position_m': position})
            spec['workroom_diagnostics'][object_id] = {
                'half_extents_m': half, 'initial_position_m': position,
                'support_top_z_m': top, 'expected_support_path': support}
    chunks.extend(['    }', '}'])
    usd.write_text('\n'.join(chunks) + '\n')
    manifest = folder / 'scene_description.json'
    dump(manifest, {'family': 'static_mesh_probe', 'dependencies': {}})
    spec['workroom'] = {'schema': SCHEMA, 'family': 'static_mesh_probe', 'version': 'diagnostic_v1',
        'source_prim': '/World', 'usd_path': str(usd), 'usd_sha256': digest(usd),
        'dependency_manifest_path': str(manifest), 'dependency_manifest_sha256': digest(manifest)}
    spec['workroom_provenance']['source_file_hashes'] = {str(usd): digest(usd), str(manifest): digest(manifest)}
    spec['isaacsim']['static_mesh_contact_mode'] = MODE
    spec['description'] = 'Unscored native original-mesh contact probe: perforated supports.'
    spec['workroom_provenance']['scope'] = 'Synthetic native diagnostic; task release only supplies unchanged robot and cube inputs.'
    return spec


def assess_hole_crossings(initial, rows, diagnostics, paths):
    """Require a vertical passage through each opening, not a roll off its rim."""
    import numpy as np
    results = {}
    for name, spec in diagnostics.items():
        if not name.endswith('_hole'):
            continue
        start = np.array(initial['objects'][name]['position_m'])
        support = '/World/Room/Geometry/' + name.removesuffix('_hole')
        # A bounding sphere is conservative under any cube orientation.
        radius = float(np.linalg.norm(spec['half_extents_m']))
        crossing = None
        maximum_xy, impulse = 0., 0.
        for index, row in enumerate(rows):
            observation = row['observation']
            position = np.array(observation['objects'][name]['position_m'])
            maximum_xy = max(maximum_xy, float(np.linalg.norm(position[:2]-start[:2])))
            impulse += support_contact(observation, paths[name], support)['impulse_ns']
            if position[2] + radius < .76:
                crossing = index
                break
        results[name] = {'passed': crossing is not None and maximum_xy <= .025 and impulse <= 1.e-6,
                         'fully_below_support_row': crossing, 'maximum_xy_displacement_m': maximum_xy,
                         'mesh_contact_impulse_before_passage_ns': impulse}
    return {'passed': bool(results) and all(x['passed'] for x in results.values()), 'objects': results}


def native_shape_stats(env):
    """Read the explicit simulated stage/PhysicsScene, never a default singleton."""
    import omni.physx
    from omni.physx.bindings._physx import PhysicsSceneStats
    from isaacsim.core.experimental.utils import stage as stage_utils
    from pxr import PhysicsSchemaTools, UsdPhysics
    scenes = [p for p in env.stage.Traverse() if p.IsA(UsdPhysics.Scene)]
    if len(scenes) != 1:
        raise RuntimeError('Probe requires exactly one native PhysicsScene')
    stats = PhysicsSceneStats()
    stage_id = stage_utils.get_stage_id(env.stage)
    path = PhysicsSchemaTools.sdfPathToInt(scenes[0].GetPath())
    ok = omni.physx.get_physx_statistics_interface().get_physx_scene_statistics(stage_id, path, stats)
    if not ok:
        raise RuntimeError('Could not read native shape statistics for the simulated stage')
    return {name: int(getattr(stats, name)) for name in ('nb_trimesh_shapes', 'nb_convex_shapes',
        'nb_box_shapes', 'nb_kinematic_rigids', 'nb_articulations', 'nb_dynamic_rigids')}


def run(args):
    root = Path(__file__).absolute().parents[2]
    output = args.output.absolute()
    output.mkdir(parents=True, exist_ok=False)
    started = datetime.now(timezone.utc).isoformat()
    env, error, checks = None, None, {}
    steps = 0
    plan = {'schema': 'static_mesh_contact_probe_plan_v1', 'model_calls': 0,
            'device': args.device, 'gpu_uuid': os.environ.get('CUDA_VISIBLE_DEVICES'),
            'driver_override': os.environ.get('ISAAC_ALLOW_UNVALIDATED_DRIVER'),
            'physics_mode': MODE, 'episodes': 2, 'settle_control_steps_per_episode': 120,
            'minimum_native_triangle_mesh_shapes': 5, 'minimum_native_convex_shapes': 4,
            'maximum_support_drift_m': 1.e-6, 'maximum_support_angle_rad': 1.e-5,
            'object_thresholds': deepcopy(THRESHOLDS),
            'maximum_hole_passage_xy_displacement_m': .025,
            'maximum_hole_passage_mesh_impulse_ns': 1.e-6,
            'source_root': str(root), 'validator_sha256': digest(__file__),
            'scope': 'Native mesh contact, hole preservation and fixed actor checks; no task/model score.'}
    dump(output/'plan.json', plan)
    try:
        # The snapshot guard checks modules actually loaded in this process.
        # These imports are pure Python; only constructing the env loads Kit.
        from scenes import workrooms
        from bench.simulation.isaacsim_backend.env import FR3DexterousIsaacEnv
        verification = verify_if_snapshot(root, required_modules=(
            'scripts.validation.validate_static_mesh_contacts', 'scenes.workrooms',
            'bench.simulation.isaacsim_backend.env', 'bench.simulation.isaacsim_backend.static_mesh_contacts'),
            entrypoint_file=Path(__file__))
        if verification is None:
            raise ValueError('Native contact proof requires a complete frozen source snapshot')
        dump(output/'runtime_imports.json', verification)
        uuid = os.environ.get('CUDA_VISIBLE_DEVICES', '')
        if (args.device != 'cuda:0' or os.environ.get('ISAAC_BENCH_SINGLE_CUDA_DEVICE') != '1'
                or not uuid.startswith('GPU-') or ',' in uuid):
            raise ValueError('Use one physical GPU UUID with single-device mode and cuda:0')
        spec = make_probe_scene(output)
        dump(output/'scene.json', spec)
        env = FR3DexterousIsaacEnv(spec, device=args.device, render_width=320, render_height=240)
        action = {'joint_positions_rad': env.action_spec()['initial_joint_positions_rad'], 'duration_s': .02}
        paths = {name: row['path'] for name, row in env.scene_info['objects'].items()}
        expected_actors = {'/World/Room/Geometry/' + n for n in ('triangle_compound', 'convex', 'sdf')}
        if set(env.scene_info['kinematic_body_paths']) != expected_actors:
            raise RuntimeError('Unexpected original mesh actor inventory')
        diagnostic = spec['workroom_diagnostics']
        first_rest = None
        start_wall = time.monotonic()
        with (output/'physics.jsonl').open('x') as stream:
            for episode in range(2):
                initial = env.reset(seed=0)
                dump(output/f'episode_{episode}_reset.json', record(env, initial, episode, 'reset'))
                previous = plain(env.capture_state())
                rows = []
                for index in range(120):
                    observation = env.step(action)
                    row = record(env, observation, episode, 'settle', action)
                    check_window(observation, row['state'], previous, env.dt)
                    previous = row['state']
                    stream.write(json.dumps(row, allow_nan=False, separators=(',', ':')) + '\n')
                    rows.append(row)
                    steps += 1
                    if (index+1) % 30 == 0:
                        stream.flush()
                        print(json.dumps({'event': 'mesh_probe_progress', 'episode': episode,
                                          'control_steps': steps}), flush=True)
                checks[f'episode_{episode}'] = assess_objects(initial, rows, diagnostic, paths)
                checks[f'episode_{episode}_holes'] = assess_hole_crossings(initial, rows, diagnostic, paths)
                rest = {name: row['position_m'] for name, row in rows[-1]['observation']['objects'].items()}
                if first_rest is None:
                    first_rest = rest
                else:
                    import numpy as np
                    error_m = max(float(np.linalg.norm(np.array(rest[n])-first_rest[n])) for n in rest)
                    checks['reset_repeatability'] = {'passed': error_m <= .002, 'maximum_rest_position_error_m': error_m}
        stats = native_shape_stats(env)
        dump(output/'native_shape_statistics.json', stats)
        checks['native_shapes'] = {'passed': stats['nb_trimesh_shapes'] >= 5 and stats['nb_convex_shapes'] >= 4
            and stats['nb_kinematic_rigids'] == 3, 'statistics': stats,
            'scope': 'Stage-bound cooked shape counts plus per-support rim/through-hole dynamics.'}
        contract = env.scene_info['static_mesh_contacts']
        unchanged = all(geometry_signature(env.stage.GetPrimAtPath(path)) == before
                        for path, before in contract['geometry_sha256'].items())
        source_unchanged = digest(spec['workroom']['usd_path']) == spec['workroom']['usd_sha256']
        checks['original_geometry'] = {'passed': unchanged and source_unchanged,
            'runtime_geometry_unchanged': unchanged, 'source_file_unchanged': source_unchanged,
            'mesh_approximations': contract['mesh_approximations']}
        guard = env._fixed_actor_guard.report()
        checks['immovable_every_substep'] = {'passed': guard['checked_batches'] >= steps*round(.02/env.dt), **guard}
        dump(output/'metadata.json', env.metadata())
        dump(output/'performance.json', {'control_steps': steps, 'wall_time_s': time.monotonic()-start_wall,
            'native_physics_steps': steps*round(.02/env.dt), 'batch_actor_count': 3})
        dump(output/'runtime_imports_final.json', verify_if_snapshot(root, required_modules=(
            'scripts.validation.validate_static_mesh_contacts', 'bench.simulation.isaacsim_backend.env'),
            entrypoint_file=Path(__file__)))
    except BaseException as exc:
        error = {'type': type(exc).__name__, 'detail': str(exc), 'traceback': traceback.format_exc()}
        print(json.dumps({'event': 'mesh_probe_error', **error}), flush=True)
    report = {'schema': 'native_static_mesh_contact_probe_v1',
        'passed': error is None and bool(checks) and all(x['passed'] for x in checks.values()),
        'error': error, 'checks': plain(checks), 'model_calls': 0, 'control_steps': steps,
        'object_state_writeback_after_reset': False, 'task_success_claimed': False,
        'started_utc': started, 'finished_utc': datetime.now(timezone.utc).isoformat(),
        'plan_sha256': digest(output/'plan.json'),
        'artifacts_sha256': {str(p.relative_to(output)): digest(p) for p in output.rglob('*') if p.is_file()}}
    return seal_report_before_shutdown(output, report, env)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args(argv)
    if not run(args)['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
