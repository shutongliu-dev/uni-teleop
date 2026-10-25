"""Recorded source pins select audit layouts; filesystem lookalikes are not evidence."""
import json
from pathlib import Path

import numpy as np
import pytest

from bench.isaacsim import audit


LAYOUTS = ('simulation', 'bench/simulation')
ENV_SOURCE = 'class Environment:\n    def step(self):\n        self.robot.set_dof_position_targets([])\n'


def write_json(path, value):
    path.write_text(json.dumps(value))


def recording(tmp_path, layout):
    root = tmp_path / 'episode'
    root.mkdir()
    # The recording belongs to a different, now unavailable source checkout.
    original = tmp_path / 'removed_source_checkout'
    files = {'bench/isaacsim/run.py': '# recorded launcher\n',
             layout + '/isaacsim_backend/env.py': ENV_SOURCE,
             layout + '/isaacsim_backend/scene.py': '# recorded scene\n',
             layout + '/isaacsim_backend/robot_asset.py': '# recorded asset loader\n'}
    pins = {}
    for relative, text in files.items():
        snapshot = root / 'source_snapshot' / relative
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        snapshot.write_text(text)
        pins[str(original / relative)] = audit.digest(snapshot)
    write_json(root / 'source_hashes.json', pins)
    write_json(root / 'result.json', {'source_files_changed_during_run': []})
    context = {'metadata': {'joint_names': ['joint'],
                           'action_spec': {'joint_limits_rad': {'joint': [-1, 1]}}},
               'states': {'joint_target_q': np.asarray([[0.], [.1]])},
               'actions': [{'duration_s': .02, 'joint_positions_rad': {'joint': .1}}]}
    return root, original, pins, context


@pytest.mark.parametrize('layout', LAYOUTS)
def test_archived_sources_and_action_inspection_use_the_recorded_layout(tmp_path, layout):
    root, original, pins, context = recording(tmp_path, layout)
    report = audit.verify_sources(root)
    assert report['source_snapshots_verified'] == 4
    assert report['simulator_source_layout'] == layout
    assert report['recorded_project_root'] == str(original)
    assert set(report['current_worktree_differences']) == set(pins)
    assert audit.check_actions(root, context)['captured_step_setters'] == ['set_dof_position_targets']


@pytest.mark.parametrize('layout', LAYOUTS)
def test_missing_declared_environment_never_falls_back_to_other_layout(tmp_path, layout):
    root, _, _, context = recording(tmp_path, layout)
    (root / 'source_snapshot' / layout / 'isaacsim_backend/env.py').unlink()
    other = next(name for name in LAYOUTS if name != layout)
    decoy = root / 'source_snapshot' / other / 'isaacsim_backend/env.py'
    decoy.parent.mkdir(parents=True)
    decoy.write_text(ENV_SOURCE)
    with pytest.raises(audit.MissingEvidence, match='Missing file'):
        audit.check_actions(root, context)


@pytest.mark.parametrize('layout', LAYOUTS)
def test_action_audit_rejects_changed_captured_source_without_a_prior_source_check(tmp_path, layout):
    root, _, _, context = recording(tmp_path, layout)
    source = root / 'source_snapshot' / layout / 'isaacsim_backend/env.py'
    source.write_text(ENV_SOURCE + '# unpinned mutation\n')
    with pytest.raises(ValueError, match='differs from launch pin'):
        audit.check_actions(root, context)


@pytest.mark.parametrize('partial', [True, False])
def test_mixed_layouts_are_rejected_even_when_the_second_layout_is_incomplete(tmp_path, partial):
    root, original, pins, context = recording(tmp_path, LAYOUTS[0])
    names = ['env.py'] if partial else ['env.py', 'scene.py', 'robot_asset.py']
    for name in names:
        pins[str(original / 'bench/simulation/isaacsim_backend' / name)] = 'a' * 64
    write_json(root / 'source_hashes.json', pins)
    with pytest.raises(ValueError, match='Mixed simulator source layouts'):
        audit.verify_sources(root)
    with pytest.raises(ValueError, match='Mixed simulator source layouts'):
        audit.check_actions(root, context)


def test_duplicate_json_source_keys_are_not_silently_overwritten(tmp_path):
    root, _, pins, _ = recording(tmp_path, LAYOUTS[0])
    key, digest = next(iter(pins.items()))
    serialized = json.dumps(pins)
    (root / 'source_hashes.json').write_text(serialized[:-1] + ',' + json.dumps(key) + ':' + json.dumps(digest) + '}')
    with pytest.raises(ValueError, match='Duplicate source pin'):
        audit.verify_sources(root)


def test_two_recorded_source_roots_are_rejected(tmp_path):
    root, _, pins, _ = recording(tmp_path, LAYOUTS[0])
    pins[str(tmp_path / 'other_checkout/bench/isaacsim/run.py')] = 'a' * 64
    write_json(root / 'source_hashes.json', pins)
    with pytest.raises(ValueError, match='Ambiguous recorded rollout source root'):
        audit.verify_sources(root)


@pytest.mark.parametrize('bad_path', ['bench/../simulation/extra.py', 'bench//extra.py'])
def test_noncanonical_source_aliases_are_rejected(tmp_path, bad_path):
    root, original, pins, _ = recording(tmp_path, LAYOUTS[0])
    pins[str(original) + '/' + bad_path] = 'a' * 64
    write_json(root / 'source_hashes.json', pins)
    with pytest.raises(ValueError, match='not canonical and absolute'):
        audit.verify_sources(root)


def test_unpinned_scene_does_not_complete_a_partial_source_manifest(tmp_path):
    root, original, pins, _ = recording(tmp_path, LAYOUTS[0])
    del pins[str(original / 'simulation/isaacsim_backend/scene.py')]
    write_json(root / 'source_hashes.json', pins)
    with pytest.raises(audit.MissingEvidence, match='scene.py'):
        audit.verify_sources(root)


def self_collision_proof(tmp_path, root, pins):
    robot, kit, kit_version, log = (tmp_path / name for name in ('robot.usd', 'experience.kit', 'kit_version.txt', 'probe.log'))
    for path in (robot, kit, kit_version, log):
        path.write_text(path.name)
    scene = tmp_path / '01_ordered_kitting_scene.json'
    write_json(scene, {'isaacsim': {'robot_self_collisions': True}})
    metadata = {'robot_usd_sha256': audit.digest(robot), 'robot_usd_dependency_sha256': {},
                'scene_info': {'robot_self_collision': {'enabled': True, 'excluded_link_pairs': [], 'cross_finger_exclusions': []}},
                'experience': {'path': str(kit), 'sha256': audit.digest(kit)},
                'physics_dt_s': .0005, 'native_isaacsim_app': True,
                'isaacsim_version': ['6.1.0', 'rc', 'build'], 'kit_version': 'test-kit'}
    write_json(root / 'metadata.json', metadata)
    joint_names = [f'joint_{index}' for index in range(29)]
    proof = {'original_authored_values': {'legacy_physx': True, 'shared_newton': False},
             'deliberately_closed_hand_reset_q': dict.fromkeys(joint_names, 0.), 'modes': []}
    for legacy in (False, True):
        for shared in (False, True):
            pairs = {'/World/Robot/right_thumb_DP | /World/Robot/right_index_DP':
                     {'point_count': 1, 'normal_impulse_ns': .1, 'max_penetration_m': .001}} if legacy else {}
            proof['modes'].append({'legacy_physx': legacy, 'shared_newton': shared,
                'authored_after_reset': {'legacy_physx': legacy, 'shared_newton': shared},
                'self_pairs': pairs, 'total_self_points': int(legacy),
                'total_self_normal_impulse_ns': .1 if legacy else 0., 'max_joint_speed_rad_s': 1.})
    proof_path = tmp_path / 'probe.json'
    write_json(proof_path, proof)
    provenance = {'result_sha256': audit.digest(proof_path), 'process_exit_code': 0,
        'runtime_provenance_postrun': {'sdk_files_sha256': {},
            'kit_version_reference': {'source': str(kit_version), 'source_sha256': audit.digest(kit_version), 'value': 'test-kit'},
            'probe_log': str(log), 'probe_log_sha256': audit.digest(log),
            'isaac_version_file_value': '6.1.0-rc+build', 'collection_scope': 'fixture'},
        'files_sha256': {**pins, str(scene): audit.digest(scene), str(kit): audit.digest(kit)},
        'robot_usd': str(robot), 'robot_usd_sha256': audit.digest(robot), 'robot_dependencies_sha256': {},
        'joint_limits': {name: {'lower': -1., 'upper': 1.} for name in joint_names}}
    provenance_path = tmp_path / 'self_collision_schema_probe_provenance.json'
    write_json(provenance_path, provenance)
    return proof_path, provenance_path, provenance


@pytest.mark.parametrize('layout', LAYOUTS)
def test_self_collision_probe_binds_to_archived_backend_pins(tmp_path, layout):
    root, _, pins, _ = recording(tmp_path, layout)
    proof_path, _, _ = self_collision_proof(tmp_path, root, pins)
    report = audit.audit_self_collision_probe(proof_path, [root])
    assert report['status'] == 'pass'
    assert report['matched_episodes'] == 1


@pytest.mark.parametrize('missing_from', ['episode', 'probe', 'both'])
def test_self_collision_probe_never_accepts_absent_backend_pins_as_equal(tmp_path, missing_from):
    root, original, pins, _ = recording(tmp_path, LAYOUTS[1])
    proof_path, provenance_path, provenance = self_collision_proof(tmp_path, root, pins)
    key = str(original / 'bench/simulation/isaacsim_backend/env.py')
    if missing_from in ('episode', 'both'):
        del pins[key]
        write_json(root / 'source_hashes.json', pins)
    if missing_from in ('probe', 'both'):
        del provenance['files_sha256'][key]
        write_json(provenance_path, provenance)
    with pytest.raises((ValueError, audit.MissingEvidence), match='(env.py|source hash unavailable)'):
        audit.audit_self_collision_probe(proof_path, [root])
