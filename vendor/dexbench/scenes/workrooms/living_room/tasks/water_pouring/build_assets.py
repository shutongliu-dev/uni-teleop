"""Build a fresh water-pouring object candidate from frozen source on CPU.

Run from the snapshot root with ``python -m``. This authors USD only: it does
not start Isaac, settle liquid, publish a task, or change catalog selection.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

from pxr import Sdf, Usd, UsdGeom

from bench.paths import DATA_ROOT
from scripts.assets.build_card_room import dependency_inventory
from scripts.maintenance.freeze_project_sources import verify_if_snapshot
from scenes.workrooms.living_room.task_scene import physical_inventory
from . import assets


PROJECT = Path(__file__).resolve().parents[5]
ASSET_ROOT = Path(f'{DATA_ROOT}/assets/task_parts/living_room/water_pouring')
REQUIRED_MODULES = (assets.__name__, 'scenes.workrooms.living_room.task_scene',
                    'scripts.assets.build_card_room')


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def build_assets(output, *, particle_spacing_m=assets.PARTICLE_SPACING_M,
                 vessel_sleep_threshold=None):
    """Return the compatible object manifest; failed partial builds stay put."""
    requested = Path(output).absolute()
    output = requested.resolve()
    allowed = ASSET_ROOT.resolve()
    if output == allowed or not output.is_relative_to(allowed):
        raise ValueError('Choose a new version below the shared water_pouring asset root')
    if requested.is_symlink() or output.exists():
        raise FileExistsError('Preserve existing assets; choose a fresh output: '+str(requested))
    if vessel_sleep_threshold is not None and (
            isinstance(vessel_sleep_threshold, bool)
            or not isinstance(vessel_sleep_threshold, (int, float))
            or not math.isfinite(vessel_sleep_threshold) or vessel_sleep_threshold < 0):
        raise ValueError('Vessel sleep threshold must be finite and nonnegative')
    # Validate inventory/spacing before creating any output directory.
    assets.initial_particles([0., 0., 0.], 0., particle_spacing_m=particle_spacing_m)
    snapshot = verify_if_snapshot(PROJECT, required_modules=REQUIRED_MODULES,
                                  entrypoint_file=Path(__file__))
    if snapshot is None:
        raise ValueError('Build from a complete frozen source snapshot, not an editable checkout')
    source_manifest = PROJECT/'source_snapshot_manifest.json'
    source_manifest_sha = digest(source_manifest)

    stage = Usd.Stage.CreateInMemory()
    UsdGeom.Xform.Define(stage, '/World')
    UsdGeom.SetStageMetersPerUnit(stage, 1.)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    stage.SetDefaultPrim(stage.GetPrimAtPath('/World'))
    metadata = assets.author_task(stage, presentation=False, particle_spacing_m=particle_spacing_m)
    parameters = {'particle_spacing_m': particle_spacing_m,
        'vessel_sleep_threshold': vessel_sleep_threshold,
        'sleep_threshold_units': 'mass-normalized kinetic energy (m2/s2)',
        'native_simulation_run': False}
    if vessel_sleep_threshold is not None:
        for info in metadata['containers'].values():
            stage.GetPrimAtPath(info['path']).CreateAttribute(
                'physxRigidBody:sleepThreshold', Sdf.ValueTypeNames.Float, custom=False).Set(vessel_sleep_threshold)
    metadata['asset_build'] = parameters
    # Keep embedded metadata and the adjacent metadata.json consistent after
    # the explicit builder-owned sleep override; author_task itself is intact.
    stage.GetPrimAtPath(metadata['root_path']).SetCustomDataByKey(
        'candidateMetadata', json.dumps(metadata, sort_keys=True))
    inventory = physical_inventory(stage)
    output.mkdir(parents=True, exist_ok=False)
    asset_path = output/'initial.usda'
    if not stage.Export(str(asset_path)):
        raise RuntimeError('Failed to export initial task USD')
    dump(output/'metadata.json', metadata)
    explicit = [entry['path'] if isinstance(entry, dict) else entry
                for entry in metadata.get('dependencies', [])]
    dependencies = dependency_inventory(asset_path, explicit)
    verified_after = verify_if_snapshot(PROJECT, required_modules=REQUIRED_MODULES,
                                        entrypoint_file=Path(__file__))
    if verified_after is None or digest(source_manifest) != source_manifest_sha:
        raise ValueError('Frozen source identity changed during the asset build')
    manifest = {'schema': 'living_task_objects_candidate.v1', 'task_id': 'water_pouring',
        'source_root': str(PROJECT), 'source_snapshot': verified_after,
        'source_snapshot_manifest_path': str(source_manifest),
        'source_snapshot_manifest_sha256': source_manifest_sha,
        'initial_usd': str(asset_path), 'initial_usd_sha256': digest(asset_path),
        'metadata_sha256': digest(output/'metadata.json'), 'inventory': inventory,
        'dependencies': dependencies, 'build_parameters': parameters,
        'native_manipulation_qualified': False, 'release_status': 'scene_candidate_not_bench_release',
        'scope': 'CPU-authored initial objects only; native liquid dynamics and task qualification require separate evidence'}
    # The manifest is the completion marker. Never overwrite a prior version
    # or remove a failed partial build to make it appear unused.
    dump(output/'manifest.json', manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--particle-spacing', type=float, default=assets.PARTICLE_SPACING_M)
    parser.add_argument('--vessel-sleep-threshold', type=float,
                        help='Optional mass-normalized threshold; 0 explicitly disables vessel sleep')
    args = parser.parse_args()
    manifest = build_assets(args.output, particle_spacing_m=args.particle_spacing,
                            vessel_sleep_threshold=args.vessel_sleep_threshold)
    print(json.dumps({'output': str(Path(manifest['initial_usd']).parent),
        'status': manifest['release_status'], 'manifest_sha256': digest(Path(manifest['initial_usd']).with_name('manifest.json'))}))


if __name__ == '__main__':
    main()
