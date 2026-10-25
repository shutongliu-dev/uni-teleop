"""Compose the pinned room, common robot installation and requested tool display.

Writes a separate static preview; never changes the room, robot or tool assets.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from bench.paths import DATA_ROOT, load_json
from scripts.assets.build_card_room import dependency_inventory, digest

PROJECT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--environment', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    from scenes.workrooms.mechanical_assembly.threaded_preview import DEFAULT_PROFILE, PROFILES, add_threaded_tool_preview
    parser.add_argument('--threaded-profile', choices=PROFILES, default=DEFAULT_PROFILE,
                        help='Exact tool asset selection; current default is the user-selected SHARPA v3 bundle')
    args = parser.parse_args()
    from pxr import Usd, UsdGeom
    from scripts.maintenance.freeze_project_sources import verify_if_snapshot
    from bench.simulation.assets.workrooms.builder import Builder
    from scenes.workrooms.mechanical_assembly.preview import add_robot_preview
    snapshot = verify_if_snapshot(PROJECT, required_modules=('scenes.workrooms.mechanical_assembly.preview',),
                                  entrypoint_file=Path(__file__))
    if snapshot is None:
        parser.error('Prepare from a complete frozen source snapshot')
    manifest_path = args.environment / 'manifest.json'
    manifest = load_json(manifest_path)
    if manifest.get('schema') != 'scence.mechanical_workshop_asset.v1':
        raise ValueError('Expected a published task-independent mechanical environment, not a preview or legacy room')
    description_path = args.environment / 'scene_description.json'
    if digest(description_path) != manifest['description_sha256'] or digest(manifest['scene']) != manifest['scene_sha256']:
        raise ValueError('Environment changed')
    for path, record in manifest['dependencies'].items():
        if digest(path) != record['sha256']:
            raise ValueError('Environment dependency changed: ' + path)
    description = load_json(description_path)
    if (description['family'] != 'mechanical_assembly'
            or description.get('schema') != 'scence.mechanical_workshop_environment.v1'
            or description.get('task_independent') is not True
            or description.get('variant') != manifest.get('variant')
            or description.get('objects') != []):
        raise ValueError('Expected a task-independent mechanical workshop with an empty work surface')
    from scripts.assets.build_mechanical_workshop import audit_environment
    environment_audit = audit_environment(Usd.Stage.Open(manifest['scene']))
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError('Preserve previous preview; choose a new output')
    output.mkdir(parents=True)
    stage_path = output / 'scene.usda'
    stage = Usd.Stage.CreateNew(str(stage_path))
    root = UsdGeom.Xform.Define(stage, '/World')
    root.GetPrim().GetReferences().AddReference(manifest['scene'], '/World')
    builder = Builder(stage, f'{DATA_ROOT}/assets', output, reuse_materials=True)
    composition = add_robot_preview(stage, builder)
    # Selected tool assets are composed separately from the fixed room.
    composition['threaded_tool_display'] = add_threaded_tool_preview(stage, builder, profile=args.threaded_profile)
    snapshot = verify_if_snapshot(PROJECT, required_modules=(
        'scenes.workrooms.mechanical_assembly.preview',
        'scenes.workrooms.mechanical_assembly.robot_layout',
        'scenes.workrooms.mechanical_assembly.threaded_preview',
        'scenes.workrooms.mechanical_assembly.threaded_sharpa_catalog'
            if args.threaded_profile == 'sharpa_v3' else 'scenes.workrooms.mechanical_assembly.threaded_assets'),
        entrypoint_file=Path(__file__))
    stage.GetRootLayer().Save()
    dependencies = dependency_inventory(stage_path, {**manifest['dependencies'], **builder.dependencies,
        str(manifest_path): {}, str(description_path): {}},
        runtime_builtins=description.get('runtime_builtin_dependencies'))
    description.update(preview_composition=composition, dependencies=dependencies,
        variant='mechanical_workcell_static_preview', task_independent=False,
        scope='Static installation and requested screw/tool display; no task release or native dynamics')
    def write(name, value):
        (output / name).write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    write('scene_description.json', description)
    write('manifest.json', {'schema': 'mechanical_workcell_static_preview.v1',
        'created_utc': datetime.now(timezone.utc).isoformat(), 'scene': str(stage_path),
        'scene_sha256': digest(stage_path), 'description_sha256': digest(output / 'scene_description.json'),
        'environment_manifest': str(manifest_path), 'environment_manifest_sha256': digest(manifest_path),
        'source_snapshot': snapshot, 'source_root': str(PROJECT), 'dependencies': dependencies,
        'parent_environment_audit': environment_audit,
        'threaded_asset_profile': args.threaded_profile,
        'physics_steps': 0, 'native_physics_accepted': False, 'scope': description['scope']})
    print(json.dumps({'preview': str(output), 'scene_sha256': digest(stage_path),
                      'robot_links': composition['source_rigid_body_count']}))


if __name__ == '__main__':
    main()
