from copy import deepcopy
import json

import pytest

from bench.isaacsim.release import digest, dump
from bench.paths import load_json
from scenes.workrooms.catalog import list_tasks
from scenes.workrooms.robot_configurations import ELIGIBLE_TASKS
from scripts.media.render_task_gallery import selected_definitions
from scripts.media.package_task_configurations import build, CONFIGURATIONS


def test_catalog_configuration_scope():
    catalog = list_tasks()
    rows = selected_definitions(catalog, configurations=tuple(CONFIGURATIONS))
    # 26 rigid tasks (including the three mechanism tasks) and five laundry tasks.
    eligible = {(family, task) for family, tasks in ELIGIBLE_TASKS.items() for task in tasks}
    assert {(row['family'], row['task_id']) for _, row in rows} - eligible == {
        ('laundry_room', task) for task in ('towel_flattening', 'towel_folding',
            'towel_stacking', 'shirt_folding', 'trousers_folding')}
    assert len(rows) == 93
    assert len({row['source_release']['manifest_sha256'] for _, row in rows}) == 93
    assert 'kitchen' not in {row['family'] for _, row in rows}
    # Both robot profiles publish the same installation choices for each task.
    original = list_tasks(robot_profile='original')
    assert ([(row['family'], row['task_id'], tuple(row.get('robot_configurations', ()))) for row in original]
            == [(row['family'], row['task_id'], tuple(row.get('robot_configurations', ()))) for row in catalog])
    laundry = [row for _, row in rows if row['family'] == 'laundry_room']
    assert {(row['task_id'], row['configuration_id']) for row in laundry} == {
        (task, configuration) for task in ('towel_flattening', 'towel_folding',
            'towel_stacking', 'shirt_folding', 'trousers_folding') for configuration in CONFIGURATIONS}
    assert len(selected_definitions(catalog)) == len(catalog) == 51
    no_nist = selected_definitions(catalog, configurations=tuple(CONFIGURATIONS),
        excluded_tasks=('nist_connector_insertion', 'nist_belt_drive_assembly',
                        'nist_cable_routing', 'nist_wire_harness'))
    assert len(no_nist) == 81
    with pytest.raises(ValueError, match='Unknown'):
        selected_definitions(catalog, task_ids=['no_such_task'])


def example_report(tmp_path):
    release = tmp_path/'release'
    release.mkdir()
    dump(release/'release_manifest.json', {'identity': 'frozen-example'})
    report = {'status': 'completed', 'errors': [], 'physics_steps': 0, 'agent_calls': 0,
              'source_snapshot': {'example': True}, 'gpu_uuid': 'GPU-example',
              'samples': 64, 'tasks': []}
    for configuration in CONFIGURATIONS:
        images = {}
        for name in ('overview', 'task'):
            path = tmp_path/f'{configuration}_{name}.png'
            path.write_bytes(b'opaque native image bytes')
            images[name] = {'path': str(path), 'sha256': digest(path),
                            'native_rtx': True, 'actual_samples_per_pixel': 64}
        report['tasks'].append({'family': 'study', 'task_id': 'book_stacking',
            'configuration_id': configuration, 'status': 'rendered', 'title': 'Stack books',
            'difficulty': 2, 'release': str(release),
            'release_manifest_sha256': digest(release/'release_manifest.json'), 'images': images})
    path = tmp_path/'report.json'
    dump(path, report)
    return path, report


def test_gallery_checks_complete_triplets_and_image_hashes(tmp_path):
    path, report = example_report(tmp_path)
    built = build([path], tmp_path/'index.html')
    assert built['configurations'] == 3
    index = load_json(tmp_path/'index.json')
    assert index['image_count'] == 6 and not index['pixels_modified']
    assert 'data-view="task"' in (tmp_path/'index.html').read_text()
    with pytest.raises(FileExistsError):
        build([path], tmp_path/'index.html')
    with pytest.raises(ValueError, match='Duplicate'):
        build([path, path], tmp_path/'duplicate.html')
    partial = deepcopy(report)
    partial['tasks'].pop()
    dump(path, partial)
    with pytest.raises(ValueError, match='all three'):
        build([path], tmp_path/'partial.html')
    dump(path, report)
    (tmp_path/'single_arm_task.png').write_bytes(b'changed image')
    with pytest.raises(ValueError, match='does not match'):
        build([path], tmp_path/'tampered.html')


def test_gallery_recovers_only_exact_completed_rows_from_interrupted_process(tmp_path):
    path, report = example_report(tmp_path)
    report['status'] = 'initializing'
    dump(path, report)
    recovered = {**deepcopy(report), 'schema': 'catalog_task_png_gallery_recovered.v1',
        'status': 'completed_artifacts', 'parent_report': str(path),
        'parent_report_sha256': digest(path), 'terminated_process_gone': True,
        'native_process_exit_code': 143}
    receipt = tmp_path/'recovered.json'
    dump(receipt, recovered)
    built = build([receipt], tmp_path/'recovered_gallery.html')
    assert built['configurations'] == 3
    recovered['tasks'][0]['title'] = 'Different unbound row'
    dump(receipt, recovered)
    with pytest.raises(ValueError, match='differ from'):
        build([receipt], tmp_path/'forged.html')


def test_preferred_framing_requires_same_release_and_explicit_report(tmp_path):
    path, report = example_report(tmp_path)
    framed = deepcopy(report)
    for row in framed['tasks']:
        row['presentation_framing'] = {'overview': {'focal_length_mm': 20.}}
    second = tmp_path/'framing.json'
    dump(second, framed)
    build([path, second], tmp_path/'framing_gallery.html', [second])
    selected = load_json(tmp_path/'framing_gallery.json')
    assert len(selected['tasks']) == 3
    assert all(row['superseded_presentation']['report'] == str(path)
               for row in selected['tasks'])
    with pytest.raises(ValueError, match='Duplicate'):
        build([path, second], tmp_path/'ambiguous.html')
    with pytest.raises(ValueError, match='part of'):
        build([path], tmp_path/'absent.html', [second])
    framed['tasks'][0]['release_manifest_sha256'] = 'changed'
    dump(second, framed)
    with pytest.raises(ValueError, match='different frozen release'):
        build([path, second], tmp_path/'changed_release.html', [second])
