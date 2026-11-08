"""Private archive fixtures are synthetic; no test claims native physics success."""
from copy import deepcopy
import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest

from bench.paths import load_json
from bench.runtime.native_elastic_archive import NativeElasticArchive, canonical


def frame(step):
    names = ['sponge']+['cell_'+str(i) for i in range(27)]
    objects = {name: {'position_m': [i*.01, step*.001, 0.], 'quaternion_wxyz': [1., 0., 0., 0.],
        'linear_velocity_m_s': [0., 2., 0.], 'angular_velocity_rad_s': [0., 0., .1]}
        for i, name in enumerate(names)}
    return {'native_step': 100+step, 'baseline': step == 0, 'native_time_s': 1.+step*.0005,
        'time_s': step*.0005, 'physics_dt_s': .0005, 'objects': objects,
        'robot_body_q': np.array([[1., 0., 0., 0., 0., 0., 1.], [-1., 0., 0., 0., 0., 0., 1.]]),
        'joints': {'link_'+str(i): {'kind': 'elastic_d6', 'anchor_delta_world_m': [step*.00001, 0., 0.]}
                   for i in range(63)},
        'elastic_robot_geometry': {'complete': True, 'fully_separated': step != 2,
                                  'method': 'native_cooked_rigid_composite_geometry', 'clearance_m': .005},
        'contacts': [] if step == 0 else [{'time_s': step*.0005, 'actor0': '/World/Task/sponge',
            'actor1': '/World/Tray', 'impulse_vector_ns': [0., 0., .0004]}]}


def reader(monkeypatch):
    import scripts.maintenance.freeze_project_sources as freeze
    monkeypatch.setattr(freeze, 'verify_if_snapshot', lambda *a, **k: {'verified': True, 'scope': 'synthetic_unit_fixture'})
    baseline = frame(0)
    cells = [{'id': name, 'body_path': '/World/Task/'+name, 'composite_rest_center_m': [0., 0., 0.]}
             for name in baseline['objects'] if name != 'sponge']
    composite = {'parent_rigid_body': 'sponge', 'member_ids': list(baseline['objects']),
                 'total_mass_kg': .0792, 'cells': cells, 'links': [],
                 'visual_skin_bindings': [{'large_visual_only_field': 'x'*1000000}]}
    return SimpleNamespace(elastic_info={'sponge': composite}, last_elastic_frame=baseline,
        elastic_frame_sink=None, volume_robot_body_names=['left', 'right'],
        volume_robot_export={'fixture': 'synthetic_robot'}, elastic_collider_export={'fixture': 'synthetic_components'},
        info={'usd_path': '/synthetic/task.usda', 'usd_sha256': 'a'*64,
              'objects': {name: {'path': '/World/Task/'+name} for name in baseline['objects']},
              'joints': {name: {'kind': 'elastic_d6'} for name in baseline['joints']}},
        spec={'observer': {'path': '/synthetic/observer.py', 'sha256': 'b'*64,
                           'files': {'observer.py': 'b'*64}, 'configuration': {'asset_metadata': {'elastic_composites': composite}}}})


def load(path):
    return load_json(path)


def test_all_28_native_states_contacts_and_joint_json_are_copied_and_sealed(monkeypatch, tmp_path):
    native = reader(monkeypatch)
    archive = NativeElasticArchive(tmp_path/'archive', native, chunk_size=2)
    for step in (1, 2, 3):
        sample = frame(step)
        native.elastic_frame_sink(sample)
        sample['objects']['sponge']['position_m'][0] = 999.  # No archive aliases.
    archive.close(actual_native_step=103, reason='fixture_finish')
    archive.close(actual_native_step=999, reason='ignored_second_close')
    manifest, metadata = load(archive.path/'manifest.json'), load(archive.path/'metadata.json')
    assert manifest['complete_native_step_coverage']
    assert manifest['runtime_source_snapshot_verified']
    assert not manifest['physics_or_task_success_claimed']
    assert manifest['frame_count'] == 4 and len(manifest['chunks']) == 2
    arrays = [np.load(archive.path/chunk['path'], allow_pickle=False) for chunk in manifest['chunks']]
    q = np.concatenate([a['object_q'] for a in arrays])
    assert q.shape == (4, 28, 7)
    assert np.concatenate([a['object_qd'] for a in arrays]).shape == (4, 28, 6)
    assert np.concatenate([a['robot_body_q'] for a in arrays]).shape == (4, 2, 7)
    sponge = metadata['object_names'].index('sponge')
    np.testing.assert_allclose(q[:, sponge, :3], [[0., step*.001, 0.] for step in range(4)])
    np.testing.assert_array_equal(np.concatenate([a['baseline'] for a in arrays]), [True, False, False, False])
    assert len(json.loads(str(arrays[0]['joints_json'][1]))) == 63
    assert json.loads(str(arrays[0]['contacts_json'][1])) == frame(1)['contacts']
    assert json.loads(str(arrays[1]['elastic_robot_geometry_json'][0]))['fully_separated'] is False
    assert native.elastic_frame_sink is None
    assert all(not path.stat().st_mode & 0o222 for path in archive.path.iterdir())
    for chunk in manifest['chunks']:
        assert hashlib.sha256((archive.path/chunk['path']).read_bytes()).hexdigest() == chunk['sha256']
    assert hashlib.sha256((archive.path/'metadata.json').read_bytes()).hexdigest() == manifest['metadata_sha256']
    with pytest.raises(RuntimeError, match='closed'):
        archive.append(frame(4))


def test_metadata_binds_full_configuration_without_copying_skin(monkeypatch, tmp_path):
    native = reader(monkeypatch)
    archive = NativeElasticArchive(tmp_path/'archive', native)
    metadata = load(archive.path/'metadata.json')
    assert metadata['source'] == 'actual_native_tensor_getters'
    assert archive.chunk_size == 256
    assert 'visual_skin_bindings' not in metadata['elastic_composites_physical']['sponge']
    assert len((archive.path/'metadata.json').read_bytes()) < 20000
    expected = hashlib.sha256(canonical(native.spec['observer']['configuration']).encode()).hexdigest()
    assert metadata['observer_configuration_sha256'] == expected
    assert metadata['task_usd_sha256'] == 'a'*64
    assert metadata['skin_binding_saved'] is False
    archive.close(actual_native_step=100, reason='fixture_baseline')


@pytest.mark.parametrize('fault', ['lost_tick', 'repeat_tick', 'changed_dt', 'missing_member',
                                 'missing_joint', 'nonfinite_velocity', 'nonunit_robot', 'boolean_step'])
def test_corrupt_frame_preserves_partial_evidence_and_marks_incomplete(monkeypatch, tmp_path, fault):
    native = reader(monkeypatch)
    archive = NativeElasticArchive(tmp_path/'archive', native)
    bad = frame(1)
    if fault == 'lost_tick': bad['native_step'] += 1
    elif fault == 'repeat_tick': bad['native_step'] -= 1
    elif fault == 'changed_dt': bad['physics_dt_s'] *= 2
    elif fault == 'missing_member': bad['objects'].pop('cell_26')
    elif fault == 'missing_joint': bad['joints'].pop('link_62')
    elif fault == 'nonfinite_velocity': bad['objects']['cell_26']['linear_velocity_m_s'][0] = float('nan')
    elif fault == 'nonunit_robot': bad['robot_body_q'][1, 6] = 2.
    elif fault == 'boolean_step': bad['native_step'] = True
    with pytest.raises(ValueError):
        native.elastic_frame_sink(bad)
    archive.close(actual_native_step=101, reason='fixture_failure')
    manifest = load(archive.path/'manifest.json')
    assert not manifest['complete_native_step_coverage']
    assert manifest['frame_count'] == 1 and len(manifest['errors']) == 1
    assert np.load(archive.path/manifest['chunks'][0]['path'], allow_pickle=False)['object_q'].shape == (1, 28, 7)


def test_terminal_gap_is_not_silently_filled_and_second_sink_is_rejected(monkeypatch, tmp_path):
    native = reader(monkeypatch)
    archive = NativeElasticArchive(tmp_path/'archive', native)
    with pytest.raises(ValueError, match='already has'):
        NativeElasticArchive(tmp_path/'second', native)
    assert not (tmp_path/'second').exists()
    archive.close(actual_native_step=102, reason='fixture_gap')
    assert not load(archive.path/'manifest.json')['complete_native_step_coverage']


def test_bad_baseline_is_rejected_before_making_output(monkeypatch, tmp_path):
    native = reader(monkeypatch)
    native.last_elastic_frame = frame(1)
    with pytest.raises(ValueError, match='baseline'):
        NativeElasticArchive(tmp_path/'archive', native)
    assert not (tmp_path/'archive').exists()
    assert native.elastic_frame_sink is None
