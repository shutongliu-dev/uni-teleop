"""Compose the original single-arm robot in a separate static study preview."""
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
    parser.add_argument('--environment', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--assets-root', type=Path, default=Path(f'{DATA_ROOT}/assets'))
    args = parser.parse_args()
    from pxr import Usd, UsdGeom
    from bench.simulation.assets.workrooms.builder import Builder
    from scenes.workrooms.study.preview import add_robot_preview
    from scripts.assets.build_study_room import audit_environment
    from scripts.media.render_workroom import validate_camera_geometry
    from scripts.maintenance.freeze_project_sources import verify_if_snapshot
    snapshot = verify_if_snapshot(PROJECT, required_modules=('scenes.workrooms.study.preview',),
                                  entrypoint_file=Path(__file__))
    if snapshot is None:
        parser.error('Prepare from a complete frozen source snapshot')
    environment = args.environment.resolve()
    manifest_path = environment / 'manifest.json'
    manifest = load_json(manifest_path)
    if manifest.get('schema') != 'scence.study_room_asset.v1':
        raise ValueError('Expected a versioned fixed study environment')
    description_path = environment / 'scene_description.json'
    if digest(description_path) != manifest['description_sha256'] or digest(manifest['scene']) != manifest['scene_sha256']:
        raise ValueError('Fixed study environment changed')
    audit_path = environment / 'structural_audit.json'
    if digest(audit_path) != manifest['structural_audit_sha256']:
        raise ValueError('Fixed study structural audit changed')
    for path, item in manifest['dependencies'].items():
        if digest(path) != item['sha256']:
            raise ValueError('Study dependency changed: ' + path)
    description = load_json(description_path)
    if description.get('family') != 'study' or not description.get('task_independent') or description.get('objects') != []:
        raise ValueError('Expected a study with empty task surfaces')
    audit = audit_environment(Usd.Stage.Open(manifest['scene']))
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError('Preserve previous previews; choose a new output')
    output.mkdir(parents=True)
    path = output / 'scene.usda'
    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.Xform.Define(stage, '/World').GetPrim().GetReferences().AddReference(manifest['scene'], '/World')
    builder = Builder(stage, args.assets_root.resolve(), output, reuse_materials=True)
    composition = add_robot_preview(stage, builder)
    composition['camera_geometry_check'] = validate_camera_geometry(stage, description['cameras'])
    snapshot = verify_if_snapshot(PROJECT, required_modules=(
        'scenes.workrooms.study.preview', 'scenes.workrooms.mechanical_assembly.preview',
        'scenes.workrooms.mechanical_assembly.robot_layout',
        'bench.simulation.assets.workrooms.study_room'), entrypoint_file=Path(__file__))
    stage.GetRootLayer().Save()
    dependencies = dependency_inventory(path, {**manifest['dependencies'], **builder.dependencies,
        str(manifest_path): {}, str(description_path): {}, str(audit_path): {}},
        runtime_builtins=description.get('runtime_builtin_dependencies'))
    description.update(preview_composition=composition, dependencies=dependencies,
        variant=manifest['variant'] + '_robot_preview', task_independent=False,
        scope='Static study with original robot geometry; no task objects, controller execution or physics qualification')

    def write(name, value):
        (output / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')

    write('scene_description.json', description)
    write('manifest.json', {'schema': 'study_workcell_static_preview.v1',
        'created_utc': datetime.now(timezone.utc).isoformat(), 'scene': str(path),
        'scene_sha256': digest(path), 'description_sha256': digest(output / 'scene_description.json'),
        'environment_manifest': str(manifest_path), 'environment_manifest_sha256': digest(manifest_path),
        'source_snapshot': snapshot, 'source_root': str(PROJECT), 'dependencies': dependencies,
        'parent_environment_audit': audit, 'physics_steps': 0,
        'native_physics_accepted': False, 'scope': description['scope']})
    print(json.dumps({'preview': str(output), 'robot_links': composition['source_rigid_body_count'],
                      'scene_sha256': digest(path)}))


if __name__ == '__main__':
    main()
