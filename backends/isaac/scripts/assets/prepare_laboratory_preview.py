#!/usr/bin/env python3
"""Reference a published laboratory and add a separate static robot installation."""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

from bench.paths import DATA_ROOT, load_json

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT))

from scripts.assets.build_card_room import dependency_inventory, digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--environment', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--cameras-json', type=Path,
                        help='Optional JSON mapping overriding room, workcell or tabletop cameras')
    args = parser.parse_args()
    from pxr import Usd, UsdGeom
    from bench.simulation.assets.workrooms.builder import Builder
    from scenes.workrooms.laboratory.preview import PREVIEW_CAMERAS, add_robot_preview, audit_environment
    from scripts.media.render_workroom import validate_camera_geometry
    from scripts.maintenance.freeze_project_sources import verify_if_snapshot
    modules = ('scenes.workrooms.laboratory.preview', 'scenes.workrooms.laboratory.robot_layout')
    snapshot = verify_if_snapshot(PROJECT, required_modules=modules, entrypoint_file=Path(__file__))
    if snapshot is None:
        parser.error('Prepare from a complete frozen source snapshot')
    manifest_path = args.environment.resolve() / 'manifest.json'
    manifest = load_json(manifest_path)
    if manifest.get('schema') != 'scence.laboratory_asset.v1':
        raise ValueError('Expected a published task-independent laboratory')
    description_path = manifest_path.with_name('scene_description.json')
    if digest(description_path) != manifest['description_sha256'] or digest(manifest['scene']) != manifest['scene_sha256']:
        raise ValueError('Parent laboratory environment changed')
    for path, record in manifest['dependencies'].items():
        if digest(path) != record['sha256']:
            raise ValueError('Laboratory environment dependency changed: ' + path)
    description = load_json(description_path)
    if (description.get('family') != 'laboratory'
            or description.get('schema') != 'scence.laboratory_environment.v1'
            or description.get('task_independent') is not True
            or description.get('variant') != manifest.get('variant')
            or description.get('objects') != []):
        raise ValueError('Expected a task-independent laboratory with no movable task objects')
    environment_audit = audit_environment(Usd.Stage.Open(manifest['scene']), description)
    cameras = deepcopy(PREVIEW_CAMERAS)
    if args.cameras_json:
        overrides = load_json(args.cameras_json)
        if not isinstance(overrides, dict) or not set(overrides) <= cameras.keys():
            raise ValueError('Camera overrides must map room, workcell or tabletop definitions')
        cameras.update(overrides)
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError('Preserve previous preview; choose a new output')
    output.mkdir(parents=True)
    stage_path = output / 'scene.usda'
    stage = Usd.Stage.CreateNew(str(stage_path))
    root = UsdGeom.Xform.Define(stage, '/World')
    root.GetPrim().GetReferences().AddReference(manifest['scene'], '/World')
    builder = Builder(stage, f'{DATA_ROOT}/assets', output, reuse_materials=True)
    composition = add_robot_preview(stage, builder, environment=description)
    camera_audit = validate_camera_geometry(stage, cameras)
    shared_modules = ('scenes.workrooms.mechanical_assembly.robot_layout',
                      'scenes.workrooms.mechanical_assembly.preview')
    snapshot = verify_if_snapshot(PROJECT, required_modules=(*modules, *shared_modules),
        entrypoint_file=Path(__file__))
    stage.GetRootLayer().Save()
    explicit = {**manifest['dependencies'], **builder.dependencies,
                str(manifest_path): {}, str(description_path): {}}
    if args.cameras_json:
        explicit[str(args.cameras_json.resolve())] = {}
    dependencies = dependency_inventory(stage_path, explicit,
        runtime_builtins=description.get('runtime_builtin_dependencies'))
    description.setdefault('cameras', {}).update(cameras)
    description.update(preview_composition=composition, dependencies=dependencies,
        variant='laboratory_central_single_arm_static_preview', task_independent=False,
        scope='Static laboratory and FR3/right-SHARPA installation; no task release or native dynamics')

    def write(name, value):
        (output / name).write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')

    write('scene_description.json', description)
    write('manifest.json', {'schema': 'laboratory_static_preview.v1',
        'created_utc': datetime.now(timezone.utc).isoformat(), 'scene': str(stage_path),
        'scene_sha256': digest(stage_path), 'description_sha256': digest(output / 'scene_description.json'),
        'environment_manifest': str(manifest_path), 'environment_manifest_sha256': digest(manifest_path),
        'source_snapshot': snapshot, 'source_root': str(PROJECT), 'dependencies': dependencies,
        'parent_environment_audit': environment_audit, 'preview_camera_audit': camera_audit,
        'physics_steps': 0, 'native_physics_accepted': False, 'scope': description['scope']})
    print(json.dumps({'preview': str(output), 'scene_sha256': digest(stage_path),
        'robot_links': composition['source_rigid_body_count'], 'clearance_audit': composition['clearance_audit']}))


if __name__ == '__main__':
    main()
