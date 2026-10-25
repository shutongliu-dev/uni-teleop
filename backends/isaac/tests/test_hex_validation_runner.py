"""Native Hex runner aggregation checks with explicit CPU doubles."""
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path

import pytest

from scenes.board_games.hex import HexMatch
from scenes.board_games.hex.release import task_specification
from scripts.validation.validate_hex_task import (ReservePreflightObserver, load_release, physical_checks,
                                                  preflight_camera_ids, table_contact_paths)
from scripts.validation.validate_scenario import digest
from test_hex_native_observer import FakeEnv, contact, ROBOT

TABLE = '/World/Room/Geometry/worktop/Collider'


def reserve_env():
    match = HexMatch()
    env = FakeEnv(match)
    env.scene_info['workbench'] = {'kind': 'referenced_workroom', 'supports': {'table': TABLE}}
    for paths in env._contact_filter_paths:
        paths.append(TABLE)
    env.next_rows = [contact(name=name, other=TABLE) for name in match.inventory]
    initial = {'objects': {name: {'position_m': match.geometry.reserve_position(name.split('_')[0], int(name[-2:]))}
                           for name in match.inventory}}
    return env, match, initial


def test_each_native_substep_must_support_all_50_tokens_for_full_dwell():
    env, match, initial = reserve_env()
    observer = ReservePreflightObserver(env, match, initial).attach()
    for _ in range(31): env.advance()
    assert observer.preflight_summary()['passed']
    assert observer.samples == env.acquisitions == 31
    assert not observer.interference
    preflight = physical_checks(observer, match, expected_samples=31, phase='preflight')
    env.next_rows = [row for row in env.next_rows if row['actor0'] != '/World/Objects/red_00']
    env.advance()
    report = observer.preflight_summary()
    assert not report['passed']
    assert report['tokens']['red_00']['stable_supported_dwell_s'] == 0.
    assert report['tokens']['blue_00']['passed']
    post = physical_checks(observer, match, expected_samples=32, phase='after_model_joint_step')
    assert preflight['reserve_support']['passed'] and preflight['reserve_support']['phase'] == 'preflight'
    assert not post['reserve_support']['passed']
    assert post['native_observer']['passed'] and post['native_observer']['expected_samples'] == 32
    observer.close()


def test_robot_interference_is_recorded_and_token_contact_resets_dwell():
    env, match, initial = reserve_env()
    observer = ReservePreflightObserver(env, match, initial).attach()
    for _ in range(31): env.advance()
    env.next_rows.append(contact('red_00', ROBOT))
    env.advance()
    report = observer.preflight_summary()
    assert not report['passed']
    assert not report['tokens']['red_00']['checks']['released_by_both_robots']
    pairs = list(observer.interference.values())
    assert pairs[0]['kind'] == 'robot_token' and pairs[0]['positive_substeps'] == 1
    assert pairs[0]['impulse_ns'] > 0
    observer.close()


def test_original_table_actor_mapping_is_resolved_exactly():
    actor = '/World/Room/Geometry/worktop'
    assert table_contact_paths({'workbench': {'supports': {'table': TABLE}},
                                'collider_actor_paths': {TABLE: actor}}) == (actor,)
    with pytest.raises(ValueError, match='original table'):
        table_contact_paths({'workbench': {'kind': 'referenced_workroom'}})


def fake_release(tmp_path):
    import shutil
    import scenes.board_games.hex as package
    match = HexMatch()
    initial = {'joint_positions_rad': {'q': 0.}}
    scene = {'objects': [{'id': name} for name in sorted(match.inventory)],
             'robot': {'hand_id': 'dual_fr3_sharpa_w01'},
             'hex_task': {'schema': 'hex_native_task.v1', 'initial_state': initial}}
    for name, data in [('scene.json', scene), ('task.json', task_specification()), ('initial_state.json', initial)]:
        (tmp_path/name).write_text(json.dumps(data))
    frozen = tmp_path/'evaluator/hex'
    frozen.mkdir(parents=True)
    for path in Path(package.__file__).parent.glob('*.py'):
        shutil.copy2(path, frozen/path.name)
    manifest = {'schema': 'robot_hex_task_release.v1', 'files_sha256': {
        name: digest(tmp_path/name) for name in ('scene.json', 'task.json', 'initial_state.json')}}
    (tmp_path/'release_manifest.json').write_text(json.dumps(manifest))
    return manifest


def test_release_loader_accepts_scene_default_objects_and_checks_frozen_initial_payload(tmp_path):
    fake_release(tmp_path)
    scene, task, initial, geometry, thresholds, source = load_release(tmp_path)
    assert len(scene['objects']) == 50 and initial == scene['hex_task']['initial_state']
    (tmp_path/'initial_state.json').write_text('{}')
    with pytest.raises(ValueError, match='Frozen release file mismatch'):
        load_release(tmp_path)


def test_release_loader_rejects_unhashed_or_outside_release_input(tmp_path):
    manifest = fake_release(tmp_path)
    del manifest['files_sha256']['initial_state.json']
    (tmp_path/'release_manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='explicit initial_state'):
        load_release(tmp_path)


def test_release_loader_rejects_a_frozen_evaluator_different_from_live_code(tmp_path):
    fake_release(tmp_path)
    (tmp_path/'evaluator/hex/settlement.py').write_text('# different evaluator\n')
    with pytest.raises(ValueError, match='Actual imported Hex evaluator differs'):
        load_release(tmp_path)


def test_release_loader_rejects_an_imported_evaluator_module_outside_package(tmp_path, monkeypatch):
    import scenes.board_games.hex.settlement as settlement
    fake_release(tmp_path)
    monkeypatch.setattr(settlement, '__file__', str(tmp_path/'evaluator/hex/settlement.py'))
    with pytest.raises(ValueError, match='imported outside the verified package'):
        load_release(tmp_path)


def test_native_calibration_is_converted_to_optical_camera_axes():
    import numpy as np
    from bench.geometry import quaternion_matrix
    from scripts.validation.validate_hex_task import public_camera_calibration
    native = {'world_to_view': np.eye(4).tolist(), 'resolution': [640, 480],
              'intrinsics_px': {'fx': 500., 'fy': 510., 'cx': 320., 'cy': 240.}}
    result = public_camera_calibration(native)
    assert result['intrinsics_3x3'] == [500., 0., 320., 0., 510., 240., 0., 0., 1.]
    assert np.allclose(quaternion_matrix(result['world_from_camera']['quaternion_wxyz']), np.diag([1., -1., -1.]))
    assert result['depth_scale_m'] == .01


def test_smoke_template_is_a_bounded_one_milliradian_owned_wrist_step():
    from scenes.board_games.hex import ActuatorOwnership
    from scripts.validation.validate_hex_task import bounded_smoke_command
    owner = ActuatorOwnership('player_0', 'left', ('fr3_left_joint7',))
    result = bounded_smoke_command(owner, {'fr3_left_joint7': 1.}, {'fr3_left_joint7': [-1., 1.]})
    assert result['joint_positions'] == {'fr3_left_joint7': .999}
    assert result['duration_s'] == .02 and result['robot_id'] == 'left'


@pytest.mark.parametrize('requested, expected', [
    (['room'], ['room', 'task', 'wrist_left', 'wrist_right']),
    (['wrist_right', 'task', 'task'], ['wrist_right', 'task', 'wrist_left']),
    (None, ['overview', 'task', 'room', 'wrist_left', 'wrist_right']),
])
def test_model_smoke_camera_selection_always_requires_all_three_task_views(requested, expected):
    available = ['overview', 'task', 'room', 'wrist_left', 'wrist_right']
    assert preflight_camera_ids(requested, available, model_smoke=True) == expected


def test_missing_required_camera_is_rejected_before_model_smoke():
    with pytest.raises(ValueError, match='wrist_right'):
        preflight_camera_ids(['room'], ['room', 'task', 'wrist_left'], model_smoke=True)
    assert preflight_camera_ids(['room'], ['room', 'task'], model_smoke=False) == ['room']


@pytest.mark.parametrize('post_step_fault', [None, 'support', 'integrity', 'interference',
                                           'unsupported', 'sample_gap', 'observer_failure'])
def test_model_smoke_uses_equal_public_images_separate_calls_and_one_owned_step(
        tmp_path, monkeypatch, post_step_fault):
    import numpy as np
    from types import SimpleNamespace
    from agent.model_client import responses, transport
    from scripts.validation.validate_hex_task import model_smoke
    from test_hex_agent_bridge import response

    class ObserverDouble:
        samples = 1000
        failed = False
        unsupported_samples = 0
        support_passed = True
        def __init__(self):
            self.interference = {}
        def summary(self):
            return {'samples': self.samples, 'failed': self.failed,
                    'unsupported_samples': self.unsupported_samples}
        def preflight_summary(self):
            return {'passed': self.support_passed}

    observer = ObserverDouble()
    match = HexMatch()

    class NativeDouble:
        dt = .0005
        time_s = .5
        camera_ids = ['task', 'wrist_left', 'wrist_right']
        actuated_joint_names = ([f'fr3_{side}_joint{i}' for side in ('left', 'right') for i in range(1, 8)]
                               + [f'{side}_hand{i}' for side in ('left', 'right') for i in range(22)])
        joint_names = actuated_joint_names
        step_calls = 0
        def __init__(self):
            self.q = np.zeros(58)
        def action_spec(self):
            return {'joint_limits_rad': {name: [-1., 1.] for name in self.joint_names}}
        def capture_state(self):
            return {'joint_q': self.q.copy(), 'time_s': np.asarray(self.time_s),
                    'native_world_time_s': np.asarray(self.time_s),
                    'native_physics_step_index': np.asarray(round(self.time_s/self.dt))}
        def render_rgbd(self, camera):
            return {'rgb': np.arange(48, dtype=np.uint8).reshape(4, 4, 3), 'depth_m': np.ones((4, 4)),
                    'sim_time_s': self.time_s, 'camera_calibration': {'world_to_view': np.eye(4).tolist(),
                    'resolution': [4, 4], 'intrinsics_px': {'fx': 5., 'fy': 5., 'cx': 2., 'cy': 2.}}}
        def _validate_action(self, command):
            assert set(command['joint_positions_rad']) == set(self.joint_names)
            assert command['duration_s'] == .02
            assert all(-1 <= value <= 1 for value in command['joint_positions_rad'].values())
        def step(self, command):
            self._validate_action(command)
            self.step_calls += 1
            before = self.time_s
            self.time_s += .02
            self.last_command = command
            self.q = np.array([command['joint_positions_rad'][name] for name in self.joint_names])
            observer.samples += 40
            if post_step_fault == 'support':
                observer.support_passed = False
            elif post_step_fault == 'integrity':
                match.invalid_reason = 'unexpected_token_motion'
            elif post_step_fault == 'interference':
                observer.interference[('robot', 'table')] = {'kind': 'robot_table'}
            elif post_step_fault == 'unsupported':
                observer.unsupported_samples = 1
            elif post_step_fault == 'sample_gap':
                observer.samples -= 1
            elif post_step_fault == 'observer_failure':
                observer.failed = True
            return {'sim_time_s': self.time_s, 'contacts': {'errors': [], 'sampling': {
                'complete': True, 'sampled_substeps': 40, 'native_start_step_index': round(before/self.dt),
                'native_end_step_index': round(self.time_s/self.dt), 'start_sim_time_s': before,
                'end_sim_time_s': self.time_s}}}

    env = NativeDouble()
    calls = []
    class ModelDouble:
        def __init__(self, config, **kwargs): pass
        def public_metadata(self): return {'protocol': 'responses', 'base_url': 'https://example.test/v1'}
        def close(self): pass
        def complete(self, **kwargs):
            assert env.step_calls == 0
            request = json.loads(kwargs['prompt'])['request']
            calls.append((request, kwargs['images']))
            color = request['public_observation']['color']
            return response({'placement_intent': {'type': 'placement_intent', 'piece_id': color+'_00', 'q': 3, 'r': 3},
                             'joint_targets': request['allowed_joint_command']})
    monkeypatch.setattr(responses, 'ResponsesClient', ModelDouble)
    monkeypatch.setattr(transport.ModelGatewayConfig, 'from_file', lambda path: object())
    # This double tests two-context dispatch, not native optical visibility.
    monkeypatch.setattr('scripts.validation.validate_hex_task.camera_visibility',
                        lambda *args: {'passed': True, 'scope': 'CPU dispatch double only'})
    scene = {'hex_task': {'players': [{'player_id': 'player_0', 'robot_id': 'left'},
                                      {'player_id': 'player_1', 'robot_id': 'right'}]}}
    args = SimpleNamespace(micu_credentials=tmp_path/'not-read.json', model_timeout_seconds=180., model_max_output_tokens=8192)
    result = model_smoke(env, scene, match, observer, tmp_path, args,
                         dict.fromkeys(env.joint_names, 0.))
    assert result['passed'] is (post_step_fault is None)
    assert len(calls) == 2 and env.step_calls == 1
    checks = result['post_command_physical_checks']
    assert checks['native_observer']['expected_samples'] == 1040
    assert all(row['phase'] == 'after_model_joint_step' for row in checks.values())
    failed_check = {'support': 'reserve_support', 'integrity': 'game_integrity',
                    'interference': 'robot_interference', 'unsupported': 'native_observer',
                    'sample_gap': 'native_observer', 'observer_failure': 'native_observer'}
    assert all(row['passed'] == (key != failed_check.get(post_step_fault)) for key, row in checks.items())
    assert calls[0][0]['public_observation']['player_id'] == 'player_1'
    assert calls[1][0]['public_observation']['player_id'] == 'player_0'
    assert calls[0][0]['public_observation']['snapshot'] == calls[1][0]['public_observation']['snapshot']
    assert calls[0][1] == calls[1][1] and len(calls[0][1]) == 2
    assert result['native_dispatch']['opponent_targets_preserved']
    assert env.last_command['joint_positions_rad']['fr3_left_joint7'] == .001
    assert all(env.last_command['joint_positions_rad'][name] == 0 for name in env.joint_names
               if name != 'fr3_left_joint7')
    assert match.pending.player_id == 'player_0' and not match.committed
    assert not result['logical_move_committed'] and not result['completed_game_claimed']
