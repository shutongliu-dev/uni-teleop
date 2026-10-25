"""Export scene candidates and separately freeze supported task releases."""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
import shutil
import tempfile

from bench.isaacsim.release import digest, dump, validate
from .registry import DEFAULT_REGISTRY, generate
from bench.paths import DATA_ROOT, load_json


def _destination(output, source):
    output, source = Path(output).resolve(), Path(source).resolve()
    if output.exists():
        raise FileExistsError('Choose a new output directory: ' + str(output))
    if output.is_relative_to(source) or source.is_relative_to(output):
        raise ValueError('Source and destination trees must be separate')
    if output.is_relative_to(f'{DATA_ROOT}/assets'):
        raise ValueError('Scenario recordings/releases belong in datasets, not shared assets')
    output.parent.mkdir(parents=True, exist_ok=True)
    return output, Path(tempfile.mkdtemp(prefix='.'+output.name+'.preparing-', dir=output.parent))


def _provenance(row):
    here = Path(__file__).parent
    return {'schema': 'bench_scenario_provenance.v1', **deepcopy(row['identity']),
            'source_release': deepcopy(row['source_release']), 'source_release_modified': False,
            'generator_sources_sha256': {str(path): digest(path) for path in sorted(here.glob('*.py'))},
            'initial_state_write_timing': 'reset_only_from_exported_scene',
            'seed_behavior': 'Seed sampled concrete object poses before freezing; runtime reset seed does not resample them.',
            'comparison_rule': 'Compare controllers only on the same scene/task/split/seed and release manifest SHA256.',
            'native_stability_verified': False, 'task_policy_success_claimed': False,
            'task_qualified': False, 'episode_request_defaults': deepcopy(row['episode_request_defaults']),
            'score_thresholds_changed': False, 'target_asset_changed': False,
            'unsupported_tasks': 'Sorting/tool-use/cloth scoring is not inferred from scene geometry.'}


def export_scene(scene_id, split, seed, output, *, task_id=None, registry_path=DEFAULT_REGISTRY):
    row = generate(scene_id, split, seed, task_id=task_id, registry_path=registry_path)
    output, staging = _destination(output, row['source_release']['path'])
    try:
        dump(staging/'scene.json', row['scene'])
        if row['task'] is not None:
            dump(staging/'task.json', row['task'])
        dump(staging/'registry.json', row['registry'])
        dump(staging/'geometry_validation.json', row['validation'])
        dump(staging/'provenance.json', _provenance(row))
        manifest = {'schema': 'bench_scenario_export.v1', **row['identity'],
                    'scene': 'scene.json', 'task': 'task.json' if row['task'] else None,
                    'episode_request_defaults': deepcopy(row['episode_request_defaults']),
                    'task_qualified': False,
                    'files_sha256': {str(p.relative_to(staging)): digest(p) for p in sorted(staging.iterdir())},
                    'runtime_asset_files_sha256': row['validation']['asset_dependencies_sha256'],
                    'load_command': 'scripts/isaac_python.sh -m bench.simulation.isaacsim_backend.smoke --scene <export>/scene.json --output <new-output> --device <cuda:N>',
                    'native_execution_performed': False}
        dump(staging/'scenario_manifest.json', manifest)
        if output.exists():
            raise FileExistsError(output)
        staging.rename(output)
        return manifest
    except BaseException:
        shutil.rmtree(staging)
        raise


def prepare_release(scene_id, split, seed, output, *, task_id, registry_path=DEFAULT_REGISTRY):
    """Freeze one scene/task/seed with the unchanged source evaluator/profile.

    Task initial-position fields and initial-relative goals are synchronized in
    generation. Unsupported scene/task pairs fail before any output is created.
    Historical scenes, tasks, scores and shared assets are never rewritten.
    """
    row = generate(scene_id, split, seed, task_id=task_id, registry_path=registry_path)
    if row['task'] is None:
        raise ValueError('This scene has no supported task-release adapter')
    source = Path(row['source_release']['path'])
    old = validate(source)
    output, staging = _destination(output, source)
    try:
        files = {}
        # Only the selected task plus the original frozen evaluator dependencies
        # enter this release. Old curation reports stay in their original release.
        for relative, item in old['files'].items():
            if not relative.startswith('evaluator/'):
                continue
            destination = staging/relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source/relative, destination)
            if digest(destination) != item['sha256']:
                raise ValueError('Source evaluator changed while copying')
            files[relative] = {'sha256': item['sha256'], 'source': str(source/relative)}
        if 'evaluator/sequence_evaluator.py' not in files:
            raise ValueError('Missing frozen source evaluator')
        profile = old['physics_profile']
        shutil.copy2(source/profile, staging/profile)
        if digest(staging/profile) != old['physics_profile_sha256']:
            raise ValueError('Source physics profile changed while copying')
        shutil.copy2(source/'release_manifest.json', staging/'source_release_manifest.json')
        if digest(staging/'source_release_manifest.json') != row['source_release']['sha256']:
            raise ValueError('Source release changed during derivation')
        entry = deepcopy(row['source_task'])
        entry['scene'] = f'tasks/{task_id}/scene.json'
        entry['task'] = f'tasks/{task_id}/task.json'
        for kind in ('scene', 'task'):
            dump(staging/entry[kind], row[kind])
            entry[kind+'_sha256'] = digest(staging/entry[kind])
        asset_hashes = load_json(source/'asset_file_hashes.json')
        for name, expected in row['validation']['asset_dependencies_sha256'].items():
            if name in asset_hashes and asset_hashes[name]['sha256'] != expected:
                raise ValueError('Scenario asset conflicts with the source release')
            asset_hashes[name] = {'sha256': expected}
        dump(staging/'asset_file_hashes.json', asset_hashes)
        dump(staging/'scenario_registry.json', row['registry'])
        dump(staging/'scenario_provenance.json', _provenance(row))
        dump(staging/'scene_geometry_validation.json', row['validation'])
        for path in staging.rglob('*'):
            if path.is_file():
                files.setdefault(str(path.relative_to(staging)), {'sha256': digest(path)})
        manifest = {'schema_version': 'isaacsim-scenario-task-release-v1',
            'release_id': f'scenario_{scene_id}_{task_id}_{split}_{seed}',
            'status': 'prepared_unvalidated', 'prepared_at_utc': datetime.now(timezone.utc).isoformat(),
            'source_release': str(source), 'source_release_sha256': row['source_release']['sha256'],
            'original_release_modified': False, 'tasks': [entry], 'files': files,
            'physics_profile': profile, 'physics_profile_sha256': old['physics_profile_sha256'],
            'asset_catalog': deepcopy(old['asset_catalog']),
            'additional_asset_catalogs': deepcopy(row['registry']['catalogs']),
            'scenario': row['identity'], 'scene_stratum': row['scene']['scene_variant']['ranking_stratum'],
            'physics_accepted': False, 'task_policy_success_claimed': False,
            'task_qualified': False, 'episode_request_defaults': deepcopy(row['episode_request_defaults']),
            'purpose': 'Explicit initial-state generalization candidate; requires its own native qualification.'}
        dump(staging/'release_manifest.json', manifest)
        validate(staging)
        if digest(source/'release_manifest.json') != row['source_release']['sha256']:
            raise ValueError('Source release changed during preparation')
        if output.exists():
            raise FileExistsError(output)
        staging.rename(output)
        return manifest
    except BaseException:
        shutil.rmtree(staging)
        raise
