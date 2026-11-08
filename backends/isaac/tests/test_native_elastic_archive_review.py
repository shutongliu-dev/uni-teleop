"""Synthetic archive integrity tests; cooking/observer doubles are not native evidence."""
from copy import deepcopy
import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest

from bench.paths import load_json
from bench.runtime.native_elastic_archive import NativeElasticArchive
from bench.evaluation.cloth.cloth_robot_collision import state_digest
from bench.simulation.isaacsim_backend.native_task_bundle import joint_positions
import scripts.validation.native_elastic_archive_review as review


def write(path, value):
    if path.exists():
        path.chmod(0o644)
    path.write_text(json.dumps(value, sort_keys=True)+'\n')


def read(path):
    return load_json(path)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def archived(monkeypatch, tmp_path):
    """Exercise real archive serialization/replay; stub only unavailable native services."""
    geometry = {'schema': 'native_rigid_composite_robot_geometry.v1', 'complete': True,
                'method': 'native_cooked_rigid_composite_geometry', 'clearance_m': .005,
                'fully_separated': True, 'component_convex_pieces_checked': 2}
    class GeometryDouble:
        def __init__(self, *args):
            pass
        def check(self, objects, robot):
            return deepcopy(geometry)
    class ObserverDouble:
        def on_frame(self, frame):
            self.observed = {'native_step': frame['native_step'], 'complete': True, 'errors': [],
                'safety_stop': None, 'native_support_by_actor': {'sponge': {'supported': True}},
                'detail': {'elastic_geometry': {'quiet': True, 'fully_separated': True}}}
        def on_observe(self):
            return deepcopy(self.observed)
    import scripts.maintenance.freeze_project_sources as freeze
    monkeypatch.setattr(freeze, 'verify_if_snapshot', lambda *a, **k: {
        'verified': True, 'runtime_imports_verified': True, 'snapshot_root': str(tmp_path/'synthetic_source'),
        'scope': 'Synthetic test double, not frozen native provenance'})
    monkeypatch.setattr(review, 'RigidCompositeRobotGeometry', GeometryDouble)
    monkeypatch.setattr(review, 'reanalyse_export', lambda value: deepcopy(value))
    monkeypatch.setattr(review, 'load_observer', lambda spec: ObserverDouble())
    definitions = {'link': {'kind': 'elastic_d6', 'body0': 'sponge', 'body1': 'cell',
        'anchor0_m': [0., 0., .004], 'anchor1_m': [0., 0., 0.],
        'local_rotation0_wxyz': [1., 0., 0., 0.], 'local_rotation1_wxyz': [1., 0., 0., 0.], 'drives': {}}}
    monkeypatch.setattr(review, 'add_bundle', lambda *a, **k: {'joints': deepcopy(definitions)})
    actors = {'sponge': '/World/Task/sponge', 'cell': '/World/Task/cell'}
    composite = {'parent_rigid_body': 'sponge', 'member_ids': list(actors), 'links': [],
                 'cells': [{'id': 'cell', 'body_path': actors['cell']}]}
    metadata = {'actors': actors, 'fixtures': {'sponge_tray': '/World/Task/Tray'},
                'objects': [{'id': 'sponge', 'mass_kg': .0396}, {'id': 'cell', 'mass_kg': .0396}],
                'elastic_composites': {'sponge': composite}}
    task = {'asset_metadata': metadata, 'evaluation': {'thresholds': {
        'stable_s': .02, 'stability_pose_window_s': .01, 'maximum_linear_speed_m_s': .025,
        'maximum_angular_speed_rad_s': .15, 'maximum_penetration_m': .004}}}
    bundle = {'usd_path': '/synthetic/task.usda', 'usd_sha256': 'a'*64,
        'elastic_composites': metadata['elastic_composites'], 'observer': {
            'path': '/synthetic/observer.py', 'sha256': 'b'*64, 'files': {'observer.py': 'b'*64},
            'class_name': 'ObserverDouble', 'configuration': {'metadata': metadata}}}
    runtime = {'scene_spec': {'native_task_bundle': bundle}, 'physics_dt_s': .01,
               'object_names': ['sponge', 'cell'], 'body_names': ['robot', 'sponge', 'cell']}
    def sample(i):
        objects = {name: {'position_m': [0., 0., .004 if name == 'cell' else 0.],
            'quaternion_wxyz': [1., 0., 0., 0.], 'linear_velocity_m_s': [0., 0., 0.],
            'angular_velocity_rad_s': [0., 0., 0.]} for name in actors}
        return {'native_step': 100+i, 'baseline': i == 0, 'native_time_s': 1.+i*.01,
            'time_s': i*.01, 'physics_dt_s': .01, 'objects': objects,
            'robot_body_q': np.array([[2., 0., 0., 0., 0., 0., 1.]]),
            'joints': joint_positions(objects, definitions), 'elastic_robot_geometry': deepcopy(geometry),
            'contacts': [] if i == 0 else [{'actor0': actors['sponge'], 'actor1': '/World/Task/Tray',
                'time_s': i*.01, 'min_separation_m': 0., 'point_count': 1,
                'impulse_vector_ns': [0., 0., .0792*9.81*.01], 'impulse_magnitude_sum_ns': .0792*9.81*.01}]}
    q = np.array([[0., 0., 0., 0., 0., 0., 1.], [0., 0., .004, 0., 0., 0., 1.]])
    endpoints = {'native_physics_step_index': np.array([100, 104]),
        'object_q': np.stack([q, q]), 'object_qd': np.zeros((2, 2, 6)),
        'body_q': np.stack([np.vstack((sample(0)['robot_body_q'], q))]*2)}
    np.savez(tmp_path/'states.npz', **endpoints)
    baseline_hash = state_digest({key: value[0] for key, value in endpoints.items()})
    cooked = {'state_hash_before': baseline_hash, 'state_hash_after': baseline_hash,
              'scope': 'Synthetic geometry double'}
    native = SimpleNamespace(elastic_info=metadata['elastic_composites'], last_elastic_frame=sample(0),
        elastic_frame_sink=None, volume_robot_body_names=['robot'], volume_robot_export=deepcopy(cooked),
        elastic_collider_export=deepcopy(cooked), spec=bundle,
        info={'usd_path': bundle['usd_path'], 'usd_sha256': bundle['usd_sha256'], 'joints': definitions,
              'objects': {name: {'path': path} for name, path in actors.items()}})
    archive = NativeElasticArchive(tmp_path/'native_elastic', native, chunk_size=2)
    for i in range(1, 5):
        native.elastic_frame_sink(sample(i))
    archive.close(actual_native_step=104, reason='finish_request')
    records = []
    for i in (0, 4):
        observer = ObserverDouble(); observer.on_frame(sample(i))
        records.append({'observation': {'sim_time_s': i*.01, 'task_state': observer.on_observe()}})
    (tmp_path/'physics.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in records))
    return tmp_path, runtime, task


def assess(archived):
    private, runtime, task = archived
    evidence = {}
    result = review.review_native_elastic(private, runtime, task, evidence,
                                         expected_source=private/'synthetic_source')
    return result, evidence


def reseal_metadata(private, mutate):
    root = private/'native_elastic'
    metadata = read(root/'metadata.json'); mutate(metadata); write(root/'metadata.json', metadata)
    manifest = read(root/'manifest.json'); manifest['metadata_sha256'] = sha(root/'metadata.json')
    write(root/'manifest.json', manifest)


def test_valid_synthetic_structure_replays_and_binds_each_input(archived):
    result, evidence = assess(archived)
    assert result['passed'], result
    assert result['frames'] == 5 and result['endpoint_frames_verified'] == 2
    assert result['normal_z_to_weight_ratio'] == pytest.approx(1.)
    assert str((archived[0]/'physics.jsonl').resolve()) in evidence
    assert result['native_execution_performed'] is False


@pytest.mark.parametrize('fault', ['source', 'joint_definition', 'robot_order', 'object_path'])
def test_resealed_wrong_source_or_physical_bindings_are_rejected(archived, fault):
    def mutate(metadata):
        if fault == 'source': metadata['source'] = 'rendered_or_reconstructed'
        elif fault == 'joint_definition': metadata['native_joint_definitions']['link']['anchor0_m'][0] = .2
        elif fault == 'robot_order': metadata['robot_body_names'] = ['another_robot']
        elif fault == 'object_path': metadata['object_paths']['cell'] = '/World/Other'
    reseal_metadata(archived[0], mutate)
    result, _ = assess(archived)
    assert not result['passed'] and result['errors']


@pytest.mark.parametrize('fault', ['unknown_file', 'duplicate_entry', 'escaping_path', 'missing_file'])
def test_unknown_duplicate_or_unsafe_chunk_inventory_is_rejected(archived, fault):
    root = archived[0]/'native_elastic'
    manifest = read(root/'manifest.json')
    if fault == 'unknown_file':
        np.savez(root/'frames_999999.npz', native_step=np.array([123]))
    elif fault == 'duplicate_entry': manifest['chunks'].append(deepcopy(manifest['chunks'][0]))
    elif fault == 'escaping_path': manifest['chunks'][0]['path'] = '../states.npz'
    elif fault == 'missing_file': manifest['chunks'][0]['path'] = 'absent.npz'
    write(root/'manifest.json', manifest)
    result, _ = assess(archived)
    assert not result['passed'] and result['errors']


@pytest.mark.parametrize('fault', ['lost_tick', 'no_normal_force'])
def test_self_consistently_rehashed_bad_frames_cannot_pass(archived, fault):
    root = archived[0]/'native_elastic'; manifest = read(root/'manifest.json')
    for chunk in manifest['chunks']:
        path = root/chunk['path']
        with np.load(path, allow_pickle=False) as data:
            arrays = {key: data[key] for key in data.files}
        if fault == 'lost_tick' and chunk['first_native_step'] == 102:
            arrays['native_step'][0] = 103
            chunk['first_native_step'] = 103
        if fault == 'no_normal_force':
            contacts = []
            for row in arrays['contacts_json']:
                values = json.loads(str(row))
                for contact in values:
                    contact['impulse_vector_ns'] = [0., 0., 0.]
                    contact['impulse_magnitude_sum_ns'] = 0.
                contacts.append(json.dumps(values))
            arrays['contacts_json'] = np.asarray(contacts)
        path.chmod(0o644); np.savez_compressed(path, **arrays); chunk['sha256'] = sha(path)
    write(root/'manifest.json', manifest)
    result, _ = assess(archived)
    assert not result['passed']
    if fault == 'no_normal_force':
        assert result['normal_z_to_weight_ratio'] == 0.
    else:
        assert result['errors']
