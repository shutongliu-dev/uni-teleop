"""Independently check a native gallery against the current selectable-task catalog.

This reads immutable reports and images; it does not render or qualify physics.
Run from the catalog worktree being reviewed, with an explicit output file.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess

from PIL import Image

from bench.paths import load_json
from scenes.workrooms.catalog import task_definition
from scenes.workrooms.robot_configurations import CONFIGURATION_IDS, ELIGIBLE_TASKS
from scripts.maintenance.freeze_project_sources import verify


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check(gallery, receipts, review_paths, output):
    if not __debug__:
        raise RuntimeError('Run validation without Python optimization; assertions are required')
    if output.exists():
        raise FileExistsError('Preserve earlier audit outputs')
    data = load_json(gallery)
    assert data['html_sha256'] == sha(gallery.with_suffix('.html'))
    assert data['physics_steps'] == data['agent_calls'] == 0
    assert data['native_rtx'] and not data['pixels_modified'] and not data['task_success_claimed']
    expected = {(family, task, config) for family, tasks in ELIGIBLE_TASKS.items()
                for task in tasks for config in CONFIGURATION_IDS}
    task_count = sum(map(len, ELIGIBLE_TASKS.values()))
    image_count = 2*len(expected)
    rows = {(r['family'], r['task_id'], r['configuration_id']): r for r in data['tasks']}
    assert len(rows) == len(data['tasks']) == len(expected)
    assert set(rows) == expected and data['task_count'] == task_count and data['image_count'] == image_count
    source_reports, snapshots = {}, {}
    for source in data['sources']:
        path = Path(source['path'])
        assert sha(path) == source['sha256']
        report = load_json(path)
        assert report['physics_steps'] == report['agent_calls'] == 0 and not report['errors']
        if report.get('schema') == 'catalog_task_png_gallery_recovered.v1':
            original = Path(report['parent_report'])
            assert sha(original) == report['parent_report_sha256']
            assert report['tasks'] == load_json(original)['tasks']
            assert report['native_process_exit_code'] == 143 and report['terminated_process_gone']
            assert report['status'] == 'completed_artifacts'
        else:
            assert report['status'] == 'completed'
        source_reports[str(path)] = report
        root = Path(report['source_snapshot']['snapshot_root'])
        if str(root) not in snapshots:
            snapshots[str(root)] = {**verify(root), 'manifest_sha256': sha(root / 'source_snapshot_manifest.json')}
    run_records = load_json(receipts)['runs']
    assert {r['report'] for r in run_records} == set(source_reports)
    for receipt in run_records:
        report = source_reports[receipt['report']]
        assert sha(receipt['report']) == receipt['report_sha256']
        assert sha(receipt['log']) == receipt['log_sha256']
        recovered = report.get('schema') == 'catalog_task_png_gallery_recovered.v1'
        assert receipt['exit_code'] == (143 if recovered else 0)
        assert receipt['process_gone']
        errors = [line for line in Path(receipt['log']).read_text(errors='replace').splitlines() if '[Error]' in line]
        assert errors == receipt['native_error_lines'] == []
    pins, images, replacements = [], {}, []
    for (family, task, configuration), row in sorted(rows.items()):
        definition = task_definition(family, task, configuration)
        selected = definition['source_release']
        assert selected['path'] == row['release']
        assert selected['manifest_sha256'] == row['release_manifest_sha256']
        assert sha(Path(row['release']) / 'release_manifest.json') == selected['manifest_sha256']
        assert row['physics_steps'] == row['agent_calls'] == 0 and row['status'] == 'rendered'
        assert row['robot_pose']['maximum_world_matrix_error'] < 1e-7
        matches = [r for report in source_reports.values() for r in report['tasks']
                   if (r['family'], r['task_id'], r['configuration_id']) == (family, task, configuration)
                   and all(r['images'][view]['sha256'] == row['images'][view]['sha256'] for view in ('overview', 'task'))]
        assert len(matches) == 1
        for field in ('release', 'release_manifest_sha256', 'source_root', 'robot_pose'):
            assert row[field] == matches[0][field]
        assert set(row['images']) == {'overview', 'task'}
        for view, image in row['images'].items():
            path = Path(image['path'])
            assert sha(path) == image['sha256'] and image['native_rtx']
            assert image['actual_samples_per_pixel'] >= 64 and image['pending_stage_loads'] == 0
            with Image.open(path) as decoded:
                assert decoded.format == 'PNG' and decoded.size == (1920, 1440) and decoded.mode in ('RGB', 'RGBA')
                decoded.verify()
            assert str(path) not in images
            images[str(path)] = image['sha256']
        if row.get('superseded_presentation'):
            replacements.append({'family': family, 'task_id': task, 'configuration_id': configuration,
                                 'superseded': row['superseded_presentation'], 'selected_images': row['images']})
        pins.append({'family': family, 'task_id': task, 'configuration_id': configuration, **selected})
    assert len(images) == image_count
    reviewed = {}
    for path in review_paths:
        if path.suffix == '.jsonl':
            records = [json.loads(line) for line in path.read_text().splitlines()]
        else:
            records = load_json(path)['images']
        for record in records:
            if record.get('visual_inspection') is False or record.get('manually_inspected') is False:
                continue
            file = record.get('path', record.get('image_path'))
            assert sha(file) == record['sha256']
            if file in images:
                assert images[file] == record['sha256']
                reviewed[file] = record
    root = Path(__file__).resolve().parents[2]
    result = {'schema': 'task_configuration_gallery_audit.v1', 'status': 'passed',
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'catalog_root': str(root), 'catalog_git_commit': subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip(),
        'checker_path': str(Path(__file__).resolve()), 'checker_sha256': sha(__file__),
        'gallery': str(gallery.with_suffix('.html')), 'gallery_sha256': sha(gallery.with_suffix('.html')),
        'manifest': str(gallery), 'manifest_sha256': sha(gallery),
        'task_count': task_count, 'configuration_count': len(expected), 'image_count': image_count,
        'configuration_counts_by_family': dict(Counter(key[0] for key in rows)),
        'exact_current_catalog_pins': pins, 'all_images_hash_size_native_spp_checked': True,
        'frozen_sources_reverified': snapshots, 'run_receipts': str(receipts), 'run_receipts_sha256': sha(receipts),
        'presentation_replacements': replacements, 'manual_review_selected_image_count': len(reviewed),
        'manual_review_selected_task_count': len({tuple(Path(p).parts[-4:-2]) for p in reviewed}),
        'manual_review_configuration_counts': dict(Counter(Path(p).parts[-2] for p in reviewed)),
        'manual_review_selected_images': reviewed,
        'manual_review_evidence': [{'path': str(p), 'sha256': sha(p)} for p in review_paths],
        'physics_steps': 0, 'agent_calls': 0, 'pixels_modified': False, 'manipulation_or_physics_success_claimed': False,
        'limitations': ['Static canonical authored reset; no settling or task actions.',
            'Manual inspection covers the explicitly recorded sample, not every selected image.',
            'Recovered completed frames retain interrupted-process exit 143; older frames and failed attempts remain on disk.',
            'Laboratory opposed overview uses presentation-only wider optics; public episode cameras remain unchanged.',
            'Water replacement uses latest native particle-surface presentation with original release and physics unchanged.']}
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    return {key: result[key] for key in ('status', 'task_count', 'configuration_count', 'image_count', 'manual_review_selected_image_count')}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gallery', type=Path, required=True)
    parser.add_argument('--receipts', type=Path, required=True)
    parser.add_argument('--reviews', type=Path, nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(check(args.gallery, args.receipts, args.reviews, args.output)))


if __name__ == '__main__':
    main()
