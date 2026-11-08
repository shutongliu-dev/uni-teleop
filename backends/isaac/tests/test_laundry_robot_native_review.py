"""Recorded evidence replay rejects gaps, mixed clocks and altered verdicts."""
from copy import deepcopy
import io
import json
import zipfile

import numpy as np
import pytest

from scripts.validation.review_laundry_robot_native import replay


def test_randomized_http_replay_resolves_selected_seed_not_root_seed_zero(tmp_path, monkeypatch):
    from scripts.validation.review_laundry_robot_native import selected_inputs
    import bench.isaacsim.release as releases
    calls = []

    def selected(root, task_id, *, seed):
        calls.append((root, task_id, seed))
        return {}, {'actual_seed': seed}, {'reset_case': {
            'schema': 'laundry_reset_case.v1', 'task_id': task_id, 'seed': seed}}, {}

    monkeypatch.setattr(releases, 'task_inputs', selected)
    task, scene = selected_inputs({'task_id': 'shirt_folding', 'seed': 5}, tmp_path,
                                  {'schema': 'laundry_randomized_release.v1'})
    assert calls == [(tmp_path, 'shirt_folding', 5)]
    assert scene['actual_seed'] == task['reset_case']['seed'] == 5


@pytest.mark.parametrize('seed', [True, 10, -1, None])
def test_randomized_http_replay_rejects_ambiguous_seed(tmp_path, seed):
    from scripts.validation.review_laundry_robot_native import selected_inputs
    with pytest.raises(ValueError, match='seed'):
        selected_inputs({'task_id': 'shirt_folding', 'seed': seed}, tmp_path,
                        {'schema': 'laundry_randomized_release.v1'})


class Evaluator:
    def observe(self, frame):
        # Distinct q values make archive substitution observable in the result.
        return {'success': False, 'referee': {'state': 'ongoing'},
                'detail': {'operation_seen': False, 'operation_displacement_seen': False},
                'actual_z': float(frame['cloths']['towel']['vertices_m'][0, 2])}


def fixture(tmp_path, *, edit_state=None, edit_trace=None, duplicate=False):
    topology = {'towel': [[0, 1, 2]]}
    states = {'laundry_observed_time_s': np.array([0., .0005, .001]),
        'laundry_native_step': np.arange(3), 'native_physics_step_index': np.arange(10, 13),
        'native_world_time_s': np.array([.2, .2005, .201]),
        'joint_q': np.zeros((3, 58)), 'joint_qd': np.zeros((3, 58)),
        'body_q': np.zeros((3, 2, 7)), 'body_qd': np.zeros((3, 2, 6)),
        'cloth_towel_q': np.array([[[0., 0., 1.+i*.00001]]*3 for i in range(3)]),
        'cloth_towel_qd': np.zeros((3, 3, 3))}
    task = {'asset_metadata': {'cloths': [{'id': 'towel', 'triangles': topology['towel'],
                                         'rest_vertices_m': [[0, 0, 0]]*3}]},
            'evaluation': {'physics_dt_s': .0005}}
    rows = []
    for i in range(3):
        row = {'schema': 'laundry_native_frame.v1', 'complete': True,
            'native_step': i, 'time_s': i*.0005, 'control_mode': 'robot_joint_targets',
            'native_physics_step_index': i+10, 'native_world_time_s': .2+i*.0005,
            'controlled_vertices': {'towel': []},
            'physics': {'complete': True, 'valid': True, 'gravity_enabled': True,
                'hidden_constraints': 0, 'self_intersections': 0, 'intercloth_intersections': 0,
                'max_penetration_m': 0.},
            'geometry_diagnostics': {'finite_thickness_compression_checked': True,
                'self_intersections': 0, 'intercloth_intersections': 0,
                'max_cloth_envelope_penetration_m': 0., 'table_envelope_penetration_m': 0.},
            'robot_interaction': {'complete': True, 'contact_force_measured': False,
                'robot_cloth_penetration_checked': True, 'robot_cloth_geometry_uncertain': False,
                'robot_cloth_penetration_m': 0., 'hand_near_cloth': False, 'fully_separated': True}}
        row['score'] = Evaluator().observe({'cloths': {'towel': {'vertices_m': states['cloth_towel_q'][i]}}})
        rows.append(row)
    if edit_state:
        edit_state(states)
    if edit_trace:
        edit_trace(rows)
    archive_path, trace_path = tmp_path/'states.zip', tmp_path/'trace.jsonl'
    stream = io.BytesIO()
    np.savez(stream, **states)
    with zipfile.ZipFile(archive_path, 'w') as archive:
        archive.writestr('topology.json', json.dumps(topology))
        archive.writestr('robot_colliders.json', json.dumps({'status': 'complete', 'issues': [],
            'visual_mesh_fallback': False, 'state_hash_before': 'a'*64, 'state_hash_after': 'a'*64,
            'layer_hashes_before': {'root': 'b'*64}, 'layer_hashes_after': {'root': 'b'*64}}))
        archive.writestr('states/000000.npz', stream.getvalue())
        if duplicate:
            archive.writestr('states/000000.npz', stream.getvalue())
    trace_path.write_text(''.join(json.dumps(row)+'\n' for row in rows))
    return archive_path, trace_path, task


def test_native_archive_replay_preserves_original_geometry_and_scores(tmp_path):
    archive, trace, task = fixture(tmp_path)
    result = replay(archive, trace, Evaluator(), task)
    assert result['samples_checked'] == 3
    assert result['native_steps'] == 2
    assert result['native_geometry_recomputed'] is False
    assert result['recorded_geometry_summary']['fully_separated_frames'] == 3


@pytest.mark.parametrize('fault', ['gap', 'mixed_time', 'missing_frame', 'extra_frame', 'altered_score', 'q_substitution',
                                    'physics_substitution', 'controlled_nodes'])
def test_replay_rejects_incomplete_or_contradictory_recorded_evidence(tmp_path, fault):
    def edit_state(states):
        if fault == 'gap':
            states['native_physics_step_index'][1] += 1
        if fault == 'mixed_time':
            states['laundry_observed_time_s'][1] += .001
        if fault == 'q_substitution':
            states['cloth_towel_q'][1, 0, 2] += .1
    def edit_trace(rows):
        if fault == 'missing_frame':
            rows.pop()
        if fault == 'extra_frame':
            rows.append(deepcopy(rows[-1]))
        if fault == 'altered_score':
            rows[1]['score']['actual_z'] += .1
        if fault == 'physics_substitution':
            rows[1]['physics']['max_penetration_m'] = .01
        if fault == 'controlled_nodes':
            rows[1]['controlled_vertices']['towel'] = [0]
    archive, trace, task = fixture(tmp_path, edit_state=edit_state, edit_trace=edit_trace)
    with pytest.raises(ValueError):
        replay(archive, trace, Evaluator(), task)


def test_native_archive_rejects_duplicate_members(tmp_path):
    with pytest.warns(UserWarning, match='Duplicate'):
        archive, trace, task = fixture(tmp_path, duplicate=True)
    with pytest.raises(ValueError, match='Duplicate'):
        replay(archive, trace, Evaluator(), task)


def test_single_arm_replay_uses_declared_29_dof_and_rejects_hidden_dual_inventory(tmp_path):
    def single(states):
        states['joint_q'] = states['joint_q'][:, :29]
        states['joint_qd'] = states['joint_qd'][:, :29]
    archive, trace, task = fixture(tmp_path, edit_state=single)
    task.update(robot_configuration_id='single_arm', active_sides=['right'])
    result = replay(archive, trace, Evaluator(), task)
    assert len(result['joint_span_rad']) == 29
    task['active_sides'] = ['left', 'right']
    with pytest.raises(ValueError, match='active robot sides'):
        replay(archive, trace, Evaluator(), task)
    task.update(robot_configuration_id='dual_opposed')
    with pytest.raises(ValueError, match='joint inventory'):
        replay(archive, trace, Evaluator(), task)


def test_configured_selection_binds_base_and_selected_release_independently(tmp_path):
    from scripts.validation.review_laundry_robot_native import verify_selection, sha
    base, selected = tmp_path/'base', tmp_path/'selected'
    for path in (base, selected):
        path.mkdir()
        (path/'release_manifest.json').write_text(path.name)
        (path/'task.json').write_text(json.dumps({'id': 'towel_folding'}))
    task = {'id': 'towel_folding', 'robot_configuration_id': 'single_arm'}
    published = {'task_id': task['id'], 'configuration_id': 'single_arm',
                 'task_release_sha256': sha(selected/'release_manifest.json')}
    q = {'task_release': str(base), 'release_manifest_sha256': sha(base/'release_manifest.json'),
         'selected_release_manifest_sha256': published['task_release_sha256'],
         'selected_task': published, 'created': {'spec': deepcopy(published)},
         'configuration_selection': {'family': 'laundry_room', 'task_id': task['id'],
             'configuration_id': 'single_arm', 'source_release': {'path': str(selected),
                 'manifest_sha256': published['task_release_sha256']}}}
    verify_selection(q, selected, task)
    for mutated in ('selected_sha', 'created_configuration', 'base_sha', 'source_path'):
        bad = deepcopy(q)
        if mutated == 'selected_sha': bad['selected_release_manifest_sha256'] = 'a'*64
        elif mutated == 'created_configuration': bad['created']['spec']['configuration_id'] = 'dual_opposed'
        elif mutated == 'base_sha': bad['release_manifest_sha256'] = 'a'*64
        else: bad['configuration_selection']['source_release']['path'] = str(base)
        with pytest.raises(ValueError):
            verify_selection(bad, selected, task)
