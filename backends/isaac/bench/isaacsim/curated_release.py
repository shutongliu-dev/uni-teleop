"""Prepare task-only workcells as new releases; never rewrite an old release.

The four elementary tasks share one cube workbench. The four sequential tasks
use their own minimal inventories. Only context object instances and visual
decoration are removed. Task JSON, evaluator code, physics profile and retained
asset specifications are copied unchanged from the selected source release.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import tempfile

from bench.isaacsim.release import digest, dump, validate
from bench.paths import DATA_ROOT, load_json
from bench.simulation.isaacsim_backend.scene_variants import CURATED_LAYOUTS, curate_scene

RELEASE_ROOT = Path(f'{DATA_ROOT}/datasets/fr3_sharpa_bench/isaacsim/releases')
SOURCES = {
    'simple': RELEASE_ROOT / 'simple_manipulation/v4_humanlike',
    'sequential': RELEASE_ROOT / 'sequential_workcell/v1',
}
OUTPUTS = {
    'simple': RELEASE_ROOT / 'simple_manipulation/v5_curated',
    'sequential': RELEASE_ROOT / 'sequential_workcell/v2_curated',
}
TASK_LAYOUT = {
    '01_cube_push': 'cube_workbench',
    '02_cube_lift_hold': 'cube_workbench',
    '03_cube_pick_place': 'cube_workbench',
    '04_cube_grasp_hold': 'cube_workbench',
    '01_ordered_kitting': 'ordered_kitting',
    '02_loose_gear_assembly': 'loose_gear_assembly',
    '03_lined_tray_loading': 'lined_tray_loading',
    '04_cover_and_secure': 'cover_and_secure',
}


def specifications(source):
    """Read and validate immutable source inputs; return scenes for CPU review."""
    source = Path(source).resolve()
    manifest = validate(source)
    result = []
    for entry in manifest['tasks']:
        task_id = entry['id']
        if task_id not in TASK_LAYOUT:
            raise ValueError('No reviewed curated inventory for task: ' + task_id)
        original = load_json(source / entry['scene'])
        scene = curate_scene(original, TASK_LAYOUT[task_id])
        result.append({'entry': deepcopy(entry), 'scene': scene,
                       'task': load_json(source / entry['task'])})
    return manifest, result


def prepare(source, output):
    source, output = Path(source).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError('Choose a new release path; destination exists: ' + str(output))
    manifest, specs = specifications(source)
    if output.is_relative_to(source) or source.is_relative_to(output):
        raise ValueError('Output and source release trees must be separate')
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix='.' + output.name + '.preparing-', dir=output.parent))
    try:
        files = {}
        for relative, item in manifest['files'].items():
            destination = staging / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source / relative, destination)
            files[relative] = {'sha256': digest(destination), 'source': str(source / relative),
                               'source_sha256': item['sha256']}
        entries, inventories = [], []
        for row in specs:
            entry, scene = row['entry'], row['scene']
            relative = entry['scene']
            dump(staging / relative, scene)
            files[relative]['sha256'] = digest(staging / relative)
            entry['scene_sha256'] = files[relative]['sha256']
            if digest(staging / entry['task']) != entry['task_sha256']:
                raise ValueError('Task changed during curated release preparation')
            entries.append(entry)
            variant = scene['scene_variant']
            inventories.append({
                'task_id': entry['id'], 'layout_id': variant['layout_id'],
                'title_zh': variant['title_zh'], 'arrangement_zh': variant['arrangement_zh'],
                'objects': deepcopy(scene['objects']), 'fixtures': deepcopy(scene.get('fixtures', [])),
                'cloth_enabled': bool(scene.get('cloth', {}).get('enabled')),
                'removed_instance_ids': variant['removed_instance_ids'],
                'retained_asset_specs_unchanged': True,
                'task_sha256_unchanged': entry['task_sha256'],
            })
        shutil.copy2(source / manifest['physics_profile'], staging / manifest['physics_profile'])
        shutil.copy2(source / 'release_manifest.json', staging / 'source_release_manifest.json')
        provenance = {
            'schema_version': 'isaacsim-curated-layout-provenance-v1',
            'generator': {'path': str(Path(__file__)), 'sha256': digest(__file__)},
            'scene_variants_source': {'path': str(Path(__file__).parents[2] / 'bench/simulation/isaacsim_backend/scene_variants.py'),
                                     'sha256': digest(Path(__file__).parents[2] / 'bench/simulation/isaacsim_backend/scene_variants.py')},
            'source_release': str(source), 'source_release_sha256': digest(source / 'release_manifest.json'),
            'source_release_modified': False,
            'scene_stratum': 'curated_v1',
            'changes': ['Remove unrelated object/fixture instances by explicit task inventory.',
                        'Disable the all-purpose visual workcell detail mesh package.'],
            'preserved': ['Task instructions, goals, thresholds and evaluator files byte-for-byte.',
                          'Robot, table, cloth, physics profile and every retained asset specification.',
                          'Source asset files, catalogs, licenses and dependency inventories.'],
            'removed_shared_asset_files': [],
            'physics_accepted': False,
            'task_policy_success_claimed': False,
            'layouts': inventories,
        }
        dump(staging / 'curation_provenance.json', provenance)
        files['curation_provenance.json'] = {'sha256': digest(staging / 'curation_provenance.json')}
        curated_manifest = {
            'schema_version': 'isaacsim-curated-task-release-v1',
            'release_id': manifest['release_id'] + '_curated_v1',
            'status': 'prepared_unvalidated', 'prepared_at_utc': datetime.now(timezone.utc).isoformat(),
            'source_release': str(source), 'source_release_sha256': digest(source / 'release_manifest.json'),
            'source_status': manifest['status'], 'original_release_modified': False,
            'tasks': entries, 'files': files,
            'physics_profile': manifest['physics_profile'], 'physics_profile_sha256': manifest['physics_profile_sha256'],
            'asset_catalog': deepcopy(manifest['asset_catalog']),
            'scene_stratum': 'curated_v1', 'physics_accepted': False, 'task_policy_success_claimed': False,
            'purpose': 'Task-only scene comparison; retain this scene stratum when reporting results.',
        }
        dump(staging / 'release_manifest.json', curated_manifest)
        validate(staging)
        staging.rename(output)
        return curated_manifest
    except BaseException:
        shutil.rmtree(staging)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--family', required=True, choices=sorted(SOURCES))
    parser.add_argument('--source', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--preview', action='store_true', help='Print retained inventories without creating a release')
    args = parser.parse_args()
    source = args.source or SOURCES[args.family]
    output = args.output or OUTPUTS[args.family]
    if args.preview:
        _, specs = specifications(source)
        value = {'source': str(source), 'proposed_output': str(output), 'layouts': [
            {'task_id': row['entry']['id'], 'layout': CURATED_LAYOUTS[row['scene']['scene_variant']['layout_id']],
             'removed_instance_ids': row['scene']['scene_variant']['removed_instance_ids']}
            for row in specs]}
    else:
        result = prepare(source, output)
        value = {'release': str(output), 'status': result['status'], 'tasks': [row['id'] for row in result['tasks']]}
    print(json.dumps(value, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
