"""Frozen-source native Hex initialization, support and dual-robot preflight.

The default engineering preflight makes no model calls or placement claims.
After the explicit reset it sends bounded joint holds and consumes existing
native contacts through HexNativeObserver. The optional model smoke admits
one tiny owned-joint command after preflight passes. Full contact frames are
not duplicated in the progress log.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, fields
from datetime import datetime, timezone
import json
import io
import math
import os
from pathlib import Path
import traceback
from uuid import uuid4

import numpy as np

from bench.paths import OUTPUT_ROOT, load_json
from scenes.board_games.hex import HexGeometry, HexMatch, HexNativeObserver
from scenes.board_games.hex.settlement import SettlementThresholds
from bench.isaacsim.release import dump
from scripts.maintenance.freeze_project_sources import verify_if_snapshot
from scripts.validation.validate_scenario import digest, seal_report_before_shutdown
from scripts.validation.validate_workrooms import check_window, plain


REQUIRED_FILES = ('scene.json', 'task.json', 'initial_state.json')
MODEL_SMOKE_CAMERAS = ('task', 'wrist_left', 'wrist_right')


def verify_evaluator_sources(release, task):
    """The actually imported evaluator must equal both frozen inventories."""
    import importlib
    package = importlib.import_module('scenes.board_games.hex')
    source_root = Path(package.__file__).resolve().parent
    frozen_root = Path(release)/'evaluator/hex'
    expected = task['sources_sha256']
    actual = {p.name: digest(p) for p in source_root.glob('*.py')}
    frozen = {p.name: digest(p) for p in frozen_root.glob('*.py')}
    if actual != expected or frozen != expected:
        raise ValueError('Actual imported Hex evaluator differs from the frozen task evaluator')
    for name in expected:
        module_name = 'scenes.board_games.hex' + ('' if name == '__init__.py' else '.'+Path(name).stem)
        module = importlib.import_module(module_name)
        if Path(module.__file__).resolve() != source_root/name:
            raise ValueError('Hex evaluator module imported outside the verified package: '+module_name)
    return {'source_root': str(source_root), 'sources_sha256': actual,
            'matches_task_and_frozen_evaluator': True}


def load_release(release):
    release = Path(release).resolve()
    manifest_path = release/'release_manifest.json'
    manifest = load_json(manifest_path)
    if manifest.get('schema') != 'robot_hex_task_release.v1':
        raise ValueError('Expected a prepared robot_hex_task_release.v1 release')
    hashes = manifest['files_sha256']
    if not set(REQUIRED_FILES).issubset(hashes):
        raise ValueError('Release must hash scene.json, task.json and explicit initial_state.json')
    for name, expected in hashes.items():
        path = (release/name).resolve()
        if not path.is_relative_to(release) or digest(path) != expected:
            raise ValueError('Frozen release file mismatch: '+name)
    for name, expected in manifest.get('asset_files_sha256', {}).items():
        if digest(name) != expected:
            raise ValueError('Frozen Hex asset mismatch: '+name)
    scene, task, initial = (load_json(release/name) for name in REQUIRED_FILES)
    if task.get('schema') != 'robot_hex_task.v1' or scene.get('hex_task', {}).get('schema') != 'hex_native_task.v1':
        raise ValueError('Missing explicit Hex task or native scene identity')
    evaluator = verify_evaluator_sources(release, task)
    geometry = HexGeometry(**{f.name: task['geometry'][f.name] for f in fields(HexGeometry)})
    thresholds = SettlementThresholds(**{f.name: task['settlement'][f.name] for f in fields(SettlementThresholds)})
    expected = HexMatch(geometry, thresholds=thresholds).inventory
    if ({row['id'] for row in scene['objects']} != expected or len(scene['objects']) != 50
            or not set(initial.get('objects', {})).issubset(expected)):
        raise ValueError('Scene must contain all 50 named tokens; reset may only override their poses')
    if scene['hex_task'].get('initial_state') != initial:
        raise ValueError('Scene and frozen initial_state.json disagree')
    if scene['robot'].get('hand_id') != 'dual_fr3_sharpa_w01':
        raise ValueError('Hex preflight requires both independent FR3 + SHARPA robots')
    return scene, task, initial, geometry, thresholds, {'manifest': str(manifest_path),
        'manifest_sha256': digest(manifest_path), 'input_sha256': {n: hashes[n] for n in REQUIRED_FILES},
        'actual_evaluator_verification': evaluator}


def table_contact_paths(scene_info):
    workbench = scene_info['workbench']
    path = workbench.get('supports', {}).get('table')
    if path is None:
        if workbench.get('kind') == 'referenced_workroom':
            raise ValueError('Referenced workroom does not identify its original table support')
        path = '/World/Table'
    actor = scene_info.get('collider_actor_paths', {}).get(path, path)
    return (actor,)


class ReservePreflightObserver(HexNativeObserver):
    """Aggregate reserve stability from the very same complete native substeps."""

    def __init__(self, env, match, initial):
        super().__init__(env, match)
        self.table_paths = table_contact_paths(env.scene_info)
        sensors, filters = self._mapping
        if any(not set(self.table_paths).issubset(filters[sensors.index(path)]) for path in self._paths):
            raise RuntimeError('Every token must observe its actual original table support actor')
        self.initial_positions = {name: np.asarray(initial['objects'][name]['position_m'], float)
                                  for name in match.inventory}
        self.stable_since = dict.fromkeys(match.inventory)
        self.stable_duration = dict.fromkeys(match.inventory, 0.)
        self.support_impulse = dict.fromkeys(match.inventory, 0.)
        self.maximum_drift = dict.fromkeys(match.inventory, 0.)
        self.last_stats = {}
        self.rejections = Counter()
        self.interference = {}

    def _observe(self, rows, sampled_before):
        super()._observe(rows, sampled_before)
        frame = self.last_frame
        support = dict.fromkeys(self.match.inventory, 0.)
        limits, geometry = self.match.verifier.thresholds, self.match.geometry
        for row in rows:
            pair = (row['actor0'], row['actor1'])
            impulse = row.get('impulse_magnitude_sum_ns')
            if not isinstance(impulse, (int, float)) or not math.isfinite(impulse) or impulse < 0:
                raise RuntimeError('Invalid native contact impulse during preflight')
            for index, path in enumerate(pair):
                name = self._path_piece.get(path)
                if name is not None and pair[1-index] in self.table_paths:
                    support[name] += impulse
            robot_paths = [p for p in pair if p in self._robot_paths]
            if robot_paths and impulse > limits.maximum_robot_contact_impulse_ns:
                if len(robot_paths) == 2:
                    owners = [p.split('/')[3] for p in robot_paths]
                    kind = 'robot_self' if owners[0] == owners[1] else 'between_robots'
                else:
                    other = pair[1] if pair[0] in self._robot_paths else pair[0]
                    kind = ('robot_table' if other in self.table_paths else
                            'robot_token' if other in self._path_piece else 'robot_environment')
                key = tuple(sorted(pair))
                item = self.interference.setdefault(key, {'actors': list(key), 'kind': kind,
                    'positive_substeps': 0, 'impulse_ns': 0., 'maximum_substep_impulse_ns': 0.,
                    'minimum_separation_m': None})
                item['positive_substeps'] += 1
                item['impulse_ns'] += impulse
                item['maximum_substep_impulse_ns'] = max(item['maximum_substep_impulse_ns'], impulse)
                separation = row.get('min_separation_m')
                if isinstance(separation, (int, float)) and math.isfinite(separation):
                    item['minimum_separation_m'] = (separation if item['minimum_separation_m'] is None
                                                    else min(item['minimum_separation_m'], separation))
        for name, piece in frame.pieces.items():
            position = np.asarray(piece.position_m)
            q = np.asarray(piece.quaternion_wxyz)
            cosine = float(np.clip(1-2*(q[1]**2+q[2]**2)/(q@q), -1, 1))
            tilt = math.acos(cosine)
            speed = float(np.linalg.norm(piece.linear_velocity_m_s))
            angular = float(np.linalg.norm(piece.angular_velocity_rad_s))
            drift = float(np.linalg.norm(position[:2]-self.initial_positions[name][:2]))
            bottom = (position[2] - geometry.piece_base_height_m/2*cosine
                      - geometry.piece_radius_m*math.sin(tilt))
            self.support_impulse[name] += support[name]
            self.maximum_drift[name] = max(self.maximum_drift[name], drift)
            checks = {'complete_native_contacts': frame.contacts_complete,
                'supported_by_original_table': support[name] > limits.minimum_support_impulse_ns,
                'linear_speed': speed <= limits.maximum_linear_speed_m_s,
                'angular_speed': angular <= limits.maximum_angular_speed_rad_s,
                'upright': tilt <= limits.maximum_tilt_rad,
                'reserve_xy_preserved': drift <= .003,
                'base_bottom_at_table': abs(bottom-geometry.table_top_z_m) <= limits.bottom_z_tolerance_m,
                'released_by_both_robots': piece.all_robot_contact_impulse_ns <= limits.maximum_robot_contact_impulse_ns,
                'reported_penetration': piece.maximum_penetration_m is not None
                    and piece.maximum_penetration_m <= limits.maximum_penetration_m}
            if all(checks.values()):
                if self.stable_since[name] is None:
                    self.stable_since[name] = frame.simulation_time_s
                self.stable_duration[name] = frame.simulation_time_s-self.stable_since[name]
            else:
                self.stable_since[name] = None
                self.stable_duration[name] = 0.
                self.rejections.update(key for key, ok in checks.items() if not ok)
            self.last_stats[name] = {'checks': checks, 'position_m': list(piece.position_m),
                'linear_speed_m_s': speed, 'angular_speed_rad_s': angular, 'tilt_rad': tilt,
                'bottom_z_error_m': float(bottom-geometry.table_top_z_m),
                'xy_drift_m': drift, 'maximum_xy_drift_m': self.maximum_drift[name],
                'stable_supported_dwell_s': self.stable_duration[name],
                'table_support_impulse_ns': self.support_impulse[name],
                'maximum_reported_penetration_m': piece.maximum_penetration_m}

    def preflight_summary(self):
        dwell = self.match.verifier.thresholds.dwell_s
        tokens = {name: {**row, 'passed': all(row['checks'].values())
                         and self.stable_duration[name]+1.e-12 >= dwell}
                  for name, row in self.last_stats.items()}
        return {'passed': len(tokens) == 50 and all(row['passed'] for row in tokens.values()),
                'token_count': len(tokens), 'tokens': tokens, 'required_dwell_s': dwell,
                'table_contact_paths': list(self.table_paths), 'failed_substep_conditions': dict(self.rejections),
                'scope': 'Every consecutive native substep in the terminal reserve dwell; no placement or grasp score'}


def physical_checks(observer, match, *, expected_samples, phase):
    """Take a fresh acceptance snapshot for the specified physical phase."""
    summary = observer.summary()
    return {
        'reserve_support': {**observer.preflight_summary(), 'phase': phase},
        'game_integrity': {'passed': match.invalid_reason is None,
                           'invalid_reason': match.invalid_reason, 'phase': phase},
        'robot_interference': {'passed': not observer.interference, 'phase': phase,
            'scope': 'Positive native robot contacts with either robot, tokens, table or other environment actors',
            'pairs': list(observer.interference.values())},
        'native_observer': {**summary, 'phase': phase, 'expected_samples': expected_samples,
            'passed': not summary['failed'] and summary['unsupported_samples'] == 0
                      and summary['samples'] == expected_samples},
    }


def preflight_camera_ids(requested, available, *, model_smoke):
    """Optional extra views cannot omit the model smoke's visibility gates."""
    selected = list(dict.fromkeys(requested or available))
    if model_smoke:
        selected.extend(camera for camera in MODEL_SMOKE_CAMERAS if camera not in selected)
    missing = set(selected)-set(available)
    if missing:
        raise ValueError('Required preflight cameras unavailable: '+', '.join(sorted(missing)))
    return selected


def camera_visibility(env, camera, frame, state):
    """Private depth-backed target visibility, using actual RTX calibration."""
    from scenes.workrooms.observations import expected_camera_world
    calibration = frame['camera_calibration']
    view, projection = np.asarray(calibration['world_to_view']), np.asarray(calibration['projection'])
    depth = np.asarray(frame['depth_m'])
    width, height = calibration['resolution']
    meta = env.scene_spec['hex_task']
    geometry = HexGeometry(**meta['geometry'])

    def sample(point):
        point = np.r_[point, 1.] @ view
        clip = point @ projection
        if point[2] >= 0 or clip[3] <= 0:
            return {'visible': False, 'reason': 'behind_camera'}
        pixel = np.rint([(clip[0]/clip[3]+1)*width/2, (1-clip[1]/clip[3])*height/2]).astype(int)
        x, y = pixel
        if not (1 <= x < width-1 and 1 <= y < height-1):
            return {'visible': False, 'reason': 'outside_frame', 'pixel': pixel.tolist()}
        measured = float(np.median(depth[y-1:y+2, x-1:x+2]))
        expected = float(-point[2])
        return {'visible': bool(np.isfinite(measured) and abs(measured-expected) < .012),
                'pixel': pixel.tolist(), 'expected_depth_m': expected, 'measured_depth_m': measured}

    if camera == 'task':
        cells = {geometry.cell_label(q, r): sample([*geometry.cell_center(q, r)[:2],
                                                   geometry.board_top_z_m+.0001])
                 for q in range(7) for r in range(7)}
        return {'passed': all(row['visible'] for row in cells.values()), 'cells': cells,
                'scope': 'All 49 empty cell centres have actual board depth; not a filled-board perception score.'}
    if camera.startswith('wrist_'):
        side = camera.removeprefix('wrist_')
        color = next(p['color'] for p in meta['players'] if p['robot_id'] == side)
        piece_id = f"{color}_{meta['home_design'][side]['reserve_index']:02d}"
        pose = state['object_q'][env.object_names.index(piece_id)]
        from scenes.board_games.hex.native_observer import _rotate_xyzw
        point = pose[:3]+_rotate_xyzw(pose[3:], np.array([0., 0., geometry.piece_base_height_m/2+geometry.handle_height_m]))
        target = sample(point)
        wrist = state['body_q'][env.body_names.index(side+'_hand_flange')]
        expected = expected_camera_world(env.cameras[camera], {'position_m': wrist[:3],
                                                               'quaternion_wxyz': wrist[[6, 3, 4, 5]]})
        actual = np.linalg.inv(view)
        error = float(np.max(np.abs(expected-actual)))
        return {'passed': target['visible'] and error < .001, 'piece_id': piece_id,
                'target_handle_top': target, 'maximum_mount_matrix_error': error,
                'scope': 'Own ready-to-grasp token top visibility and fixed mount pose; no full-board wrist claim.'}
    return {'passed': True, 'scope': 'Context view; global board visibility is checked in task camera.'}


def capture_cameras(env, output, selected):
    from PIL import Image
    result = []
    for camera in selected:
        if camera not in env.camera_ids:
            raise ValueError('Unknown scene camera: '+camera)
        before = env.capture_state()
        frame = env.render_rgbd(camera)
        after = env.capture_state()
        rgb, depth = np.asarray(frame['rgb']), np.asarray(frame['depth_m'])
        valid = np.isfinite(depth) & (depth > 0)
        checks = {'native_state_preserved': all(np.array_equal(v, after[k]) for k, v in before.items()),
                  'rgb_has_information': rgb.ndim == 3 and rgb.shape[2] == 3 and float(rgb.std()) >= 3.,
                  'depth_available': float(valid.mean()) >= .05,
                  'actual_camera_calibration': bool(frame.get('camera_calibration'))}
        visibility = camera_visibility(env, camera, frame, after)
        checks['task_information_visible'] = visibility['passed']
        Image.fromarray(rgb).save(output/(camera+'.png'))
        np.save(output/(camera+'_depth.npy'), depth, allow_pickle=False)
        metadata = {k: v for k, v in frame.items() if k not in ('rgb', 'depth_m')}
        dump(output/(camera+'.json'), plain(metadata))
        row = {'camera': camera, 'passed': all(checks.values()), 'checks': checks,
               'rgb_std': float(rgb.std()), 'valid_depth_fraction': float(valid.mean()), 'visibility': visibility}
        result.append(row)
        print(json.dumps({'event': 'camera', **{k: v for k, v in row.items() if k != 'visibility'}}), flush=True)
    return result


def public_camera_calibration(native, *, depth_scale_m=.01):
    """Optical +X right,+Y down,+Z forward, from actual RTX view/projection."""
    from bench.geometry import matrix_quaternion
    world_from_optical = np.linalg.inv(np.asarray(native['world_to_view'], float)).T @ np.diag([1., -1., -1., 1.])
    k = native['intrinsics_px']
    width, height = native['resolution']
    return {'intrinsics_3x3': [k['fx'], 0., k['cx'], 0., k['fy'], k['cy'], 0., 0., 1.],
            **{key: k[key] for key in ('fx', 'fy', 'cx', 'cy')}, 'width': width, 'height': height,
            'world_from_camera': {'frame_id': 'world', 'position_m': world_from_optical[:3, 3].tolist(),
                                  'quaternion_wxyz': matrix_quaternion(world_from_optical[:3, :3]).tolist()},
            'depth_scale_m': depth_scale_m}


def bounded_smoke_command(owner, targets, limits):
    """One 1 mrad own-wrist step, explicitly bounded before model admission."""
    name = f'fr3_{owner.robot_id}_joint7'
    if name not in owner.joint_names:
        raise ValueError('Smoke owner must identify its own FR3 wrist joint')
    old = float(targets[name])
    low, high = limits[name]
    delta = .001 if old+.001 <= high else -.001
    value = old+delta
    if not low <= value <= high:
        raise ValueError('No bounded 1 mrad smoke wrist step')
    return {'type': 'joint_targets', 'robot_id': owner.robot_id,
            'joint_positions': {name: value}, 'duration_s': .02}


def model_smoke(env, scene, match, observer, output, args, held_targets):
    """Two separate public model contexts; one admitted own-joint native step."""
    from PIL import Image
    from scenes.board_games.hex import ActuatorOwnership, HexPlayerProtocol
    from scripts.validation.hex_model_smoke import HexAgentBridge
    from agent.model_client.responses import ResponsesClient, MICU_EXTERNAL_USER_AGENT
    from agent.model_client.transport import ModelGatewayConfig

    folder = output/'model_smoke'
    folder.mkdir(exist_ok=False)
    camera = 'task'
    frame = env.render_rgbd(camera)
    rgb, depth = frame['rgb'], frame['depth_m']
    calibration = public_camera_calibration(frame['camera_calibration'])
    valid = np.isfinite(depth) & (depth > 0)
    codes = np.zeros(depth.shape, dtype=np.uint8)
    codes[valid] = np.clip(np.rint(depth[valid]/calibration['depth_scale_m']), 1, 255).astype(np.uint8)
    images = []
    for name, pixels in [('rgb.png', rgb), ('depth_10mm.png', codes)]:
        stream = io.BytesIO()
        Image.fromarray(pixels).save(stream, format='PNG')
        raw = stream.getvalue()
        (folder/name).write_bytes(raw)
        images.append(raw)
    np.save(folder/'depth_m.npy', depth, allow_pickle=False)
    snapshot = {'snapshot_id': 'hex-smoke-'+uuid4().hex, 'simulation_time_s': float(env.time_s),
        'rgb': 'model_smoke/rgb.png',
        'depth': 'model_smoke/depth_m.npy (float32 meters); attached image 1: model_smoke/depth_10mm.png, '
                 'grayscale 10 mm per code, 0 invalid, saturated at 2.55 m', 'calibration': calibration}
    dump(folder/'snapshot.json', snapshot)
    dump(folder/'native_camera_metadata.json', {k: v for k, v in frame.items() if k not in ('rgb', 'depth_m')})
    owners = tuple(ActuatorOwnership(row['player_id'], row['robot_id'], tuple(
        name for name in env.actuated_joint_names
        if name.startswith((f"fr3_{row['robot_id']}_", f"{row['robot_id']}_"))))
        for row in scene['hex_task']['players'])
    if any(len(owner.joint_names) != 29 for owner in owners):
        raise ValueError('Each model context must own exactly one 29-DOF robot')
    protocol = HexPlayerProtocol(match, owners)
    limits = env.action_spec()['joint_limits_rad']
    templates = {owner.player_id: bounded_smoke_command(owner, held_targets, limits) for owner in owners}
    for template in templates.values():
        env._validate_action({'joint_positions_rad': {**held_targets, **template['joint_positions']},
                              'duration_s': template['duration_s']})
    config = ModelGatewayConfig.from_file(args.micu_credentials)
    clients = {owner.player_id: ResponsesClient(config, user_agent=MICU_EXTERNAL_USER_AGENT,
                                               timeout_s=args.model_timeout_seconds) for owner in owners}
    bridge = HexAgentBridge(protocol, clients, model='gpt-6-astra', reasoning_effort='max',
                            max_output_tokens=args.model_max_output_tokens)
    calls_before = env.capture_state()
    preflight_samples = observer.samples
    native_dispatch = {}
    try:
        active = match.rules.current_player
        waiting = next(player for player in match.rules.players if player != active)
        print(json.dumps({'event': 'model_request', 'player_id': waiting, 'readiness': True}), flush=True)
        ready = bridge.propose(waiting, snapshot, joint_command=templates[waiting], images=tuple(images), readiness=True)
        dump(folder/'readiness.json', ready.evidence())
        print(json.dumps({'event': 'model_request', 'player_id': active, 'readiness': False}), flush=True)
        proposal = bridge.propose(active, snapshot, joint_command=templates[active], images=tuple(images))
        dump(folder/'active_proposal.json', proposal.evidence())
        calls_after = env.capture_state()
        if any(not np.array_equal(v, calls_after[k]) for k, v in calls_before.items()):
            raise RuntimeError('Physical state changed while waiting for model responses')

        def execute(admitted):
            if admitted['player_id'] != active or admitted['duration_s'] != .02:
                raise ValueError('Smoke execution requires exactly the active player and one control interval')
            merged = {'joint_positions_rad': {**held_targets, **admitted['joint_positions']}, 'duration_s': .02}
            env._validate_action(merged)
            before = env.capture_state()
            observed = env.step(merged)
            after = env.capture_state()
            check_window(observed, after, before, env.dt)
            np.savez_compressed(folder/'before_native_command.npz', **before)
            np.savez_compressed(folder/'after_native_command.npz', **after)
            native_dispatch.update(command=merged, admitted_owned_joint_names=sorted(admitted['joint_positions']),
                native_substeps=int(after['native_physics_step_index']-before['native_physics_step_index']),
                simulation_duration_s=float(after['time_s']-before['time_s']),
                measured_joint_delta_rad={name: float(after['joint_q'][env.joint_names.index(name)]
                    -before['joint_q'][env.joint_names.index(name)]) for name in admitted['joint_positions']},
                opponent_targets_preserved=all(merged['joint_positions_rad'][name] == held_targets[name]
                    for owner in owners if owner.player_id != active for name in owner.joint_names))

        execution = bridge.execute(proposal, execute_joint_targets=execute, published_snapshot=snapshot)
        post_checks = physical_checks(observer, match,
            expected_samples=preflight_samples+round(.02/env.dt), phase='after_model_joint_step')
        summary = {'schema': 'hex_live_two_model_smoke.v1', 'passed': execution['joint_command_dispatched']
            and native_dispatch['opponent_targets_preserved'] and not match.committed
            and all(row['passed'] for row in post_checks.values()),
            'transport': clients[active].public_metadata(), 'bridge': bridge.summary(),
            'readiness': ready.evidence(), 'active_execution': execution, 'native_dispatch': native_dispatch,
            'physics_paused_during_model_calls': True, 'native_observer': observer.summary(),
            'post_command_physical_checks': post_checks,
            'rgb_and_depth_image_bytes_identical_for_players': True,
            'logical_move_committed': bool(match.committed), 'completed_game_claimed': False,
            'model_manipulation_score_claimed': False,
            'depth_image_encoding': {'meters_per_code': .01, 'invalid_code': 0, 'saturation_m': 2.55,
                                     'full_native_depth_artifact': 'depth_m.npy'}}
        post_frames = capture_cameras(env, folder, MODEL_SMOKE_CAMERAS)
        summary['post_command_cameras'] = post_frames
        summary['passed'] = summary['passed'] and all(row['passed'] for row in post_frames)
        dump(folder/'result.json', plain(summary))
        return summary
    finally:
        dump(folder/'bridge_summary.json', bridge.summary())
        for player in match.rules.players:
            dump(folder/(player+'_public_history.json'), bridge.history(player))
        for client in clients.values():
            client.close()


def run(args):
    output = args.output.resolve()
    if not output.is_relative_to(OUTPUT_ROOT):
        raise ValueError('Native validation output belongs under shared datasets')
    output.mkdir(parents=True, exist_ok=False)
    env = observer = None
    error, checks, frames = None, {}, []
    steps = 0
    try:
        root = args.project_root.resolve()
        if root != Path(__file__).resolve().parents[2]:
            raise ValueError('--project-root must be the actual frozen runner source root')
        from bench.simulation.isaacsim_backend.env import FR3DexterousIsaacEnv
        modules = ('scripts.validation.validate_hex_task', 'bench.simulation.isaacsim_backend.env',
                   'scenes.board_games.hex.native_observer', 'scenes.board_games.hex.settlement')
        if args.model_smoke:
            if args.micu_credentials is None:
                raise ValueError('--model-smoke requires the explicitly supplied --micu-credentials file')
            import importlib
            for name in ('scripts.validation.hex_model_smoke', 'agent.model_client.responses'):
                importlib.import_module(name)
                modules += (name,)
        elif args.micu_credentials is not None:
            raise ValueError('Credentials require explicit --model-smoke')
        verification = verify_if_snapshot(root, required_modules=modules, entrypoint_file=Path(__file__))
        if verification is None:
            raise ValueError('A complete frozen source snapshot is required')
        dump(output/'runtime_imports.json', verification)
        uuid = os.environ.get('CUDA_VISIBLE_DEVICES', '')
        if (not uuid.startswith('GPU-') or ',' in uuid
                or os.environ.get('ISAAC_BENCH_SINGLE_CUDA_DEVICE') != '1' or args.device != 'cuda:0'):
            raise ValueError('Select one physical GPU UUID in single CUDA mode with device cuda:0')
        scene, task, initial, geometry, thresholds, source = load_release(args.release)
        count = round(args.settle_seconds/.02)
        if (not math.isfinite(args.settle_seconds) or abs(count*.02-args.settle_seconds) > 1.e-9
                or not thresholds.dwell_s+.02 <= args.settle_seconds <= 10.):
            raise ValueError('Settle duration must be a 20ms multiple, exceed the dwell, and be at most 10s')
        dump(output/'plan.json', {'schema': 'hex_native_preflight_plan.v1',
            'created_utc': datetime.now(timezone.utc).isoformat(), 'project_root': str(root),
            'release': str(args.release.resolve()), 'release_sources': source,
            'gpu_uuid': uuid, 'driver_override': os.environ.get('ISAAC_ALLOW_UNVALIDATED_DRIVER'),
            'device': args.device, 'settle_seconds': args.settle_seconds, 'seed': args.seed,
            'resolution': [args.width, args.height], 'maximum_model_calls': 2 if args.model_smoke else 0,
            'model_smoke_requested': args.model_smoke,
            'task_qualification': False, 'thresholds': asdict(thresholds),
            'additional_preflight_thresholds': {'maximum_reserve_xy_drift_m': .003},
            'validator_sha256': digest(__file__)})
        env = FR3DexterousIsaacEnv(scene, device=args.device, render_width=args.width, render_height=args.height)
        observed = env.reset(seed=args.seed, initial_state=initial)
        np.savez_compressed(output/'reset_state.npz', **env.capture_state())
        dump(output/'reset_observation.json', observed)
        match = HexMatch(geometry, thresholds=thresholds)
        expected_initial = {'objects': {row['id']: {**row, **initial.get('objects', {}).get(row['id'], {})}
                                       for row in scene['objects']}}
        observer = ReservePreflightObserver(env, match, expected_initial).attach()
        action = {'joint_positions_rad': {**env.action_spec()['initial_joint_positions_rad'],
                                         **initial.get('joint_positions_rad', {})}, 'duration_s': .02}
        env._validate_action(action)
        dump(output/'hold_command.json', action)
        previous = env.capture_state()
        with (output/'progress.jsonl').open('x') as stream:
            for steps in range(1, count+1):
                observed = env.step(action)
                state = env.capture_state()
                check_window(observed, state, previous, env.dt)
                previous = state
                row = {'control_step': steps, 'simulation_time_s': env.time_s,
                       'native_substeps_observed': observer.samples,
                       'minimum_token_dwell_s': min(observer.stable_duration.values()),
                       'interfering_pairs': len(observer.interference),
                       'unsupported_native_samples': observer.unsupported_samples}
                stream.write(json.dumps(row, allow_nan=False)+'\n')
                if steps % 10 == 0 or steps == count:
                    stream.flush()
                    print(json.dumps({'event': 'settle', **row}), flush=True)
        checks.update(physical_checks(observer, match,
            expected_samples=round(args.settle_seconds/env.dt), phase='preflight_before_optional_model_step'))
        from bench.simulation.isaacsim_backend.robot_base import runtime_base_check
        base = runtime_base_check(env._robot_view, env.scene_info)
        checks['dual_fixed_bases'] = {'passed': base['all_fixed_base'] and len(base['articulations']) == 2
            and all(r['dof_count'] == 29 and r['link_count'] == 43 and r.get('root_pose_verified')
                    for r in base['articulations']), **base}
        selected_cameras = preflight_camera_ids(args.camera, env.camera_ids, model_smoke=args.model_smoke)
        frames = capture_cameras(env, output, selected_cameras)
        checks['cameras'] = {'passed': bool(frames) and all(row['passed'] for row in frames), 'frames': frames}
        if args.model_smoke and all(row['passed'] for row in checks.values()):
            checks['model_smoke'] = model_smoke(env, scene, match, observer, output, args,
                                               action['joint_positions_rad'])
        np.savez_compressed(output/'final_state.npz', **env.capture_state())
        dump(output/'final_observation.json', env.observation())
        dump(output/'metadata_final.json', env.metadata())
        dump(output/'scene_info.json', env.scene_info)
    except BaseException as exc:
        # The optional model bridge exposes only a finite, sanitized error
        # contract; provider bodies, credentials and exception chains stay out.
        public_error = getattr(exc, 'public_metadata', None)
        error = ({'type': type(exc).__name__, **public_error()} if callable(public_error) else
                 {'type': type(exc).__name__, 'detail': str(exc), 'traceback': traceback.format_exc()})
        print(json.dumps({'event': 'error', **error}), flush=True)
    finally:
        if observer is not None:
            try:
                final_observer = observer.close()
                dump(output/'observer.json', final_observer)
                checks['final_observer_context'] = {'passed': not final_observer['failed']
                    and final_observer['final_context_verified'] is True,
                    'closure_checks': final_observer['closure_checks']}
            except BaseException as exc:
                error = error or {'type': type(exc).__name__, 'detail': str(exc), 'traceback': traceback.format_exc()}
    bridge_file = output/'model_smoke/bridge_summary.json'
    model_calls = (sum(row['model_calls'] for row in load_json(bridge_file)['contexts'])
                   if bridge_file.is_file() else 0)
    total_steps = observer.samples//round(.02/env.dt) if observer is not None else steps
    report = {'schema': 'hex_native_preflight.v1',
        'passed': error is None and bool(checks) and all(row['passed'] for row in checks.values()),
        'checks': checks, 'error': error, 'control_steps': total_steps,
        'preflight_control_steps': steps, 'model_calls': model_calls,
        'model_smoke_requested': args.model_smoke,
        'task_qualification': False, 'placement_success_claimed': False,
        'object_state_writes_after_reset': False, 'joint_targets_only_after_reset': True,
        'artifacts_sha256': {str(p.relative_to(output)): digest(p) for p in output.rglob('*') if p.is_file()}}
    return seal_report_before_shutdown(output, plain(report), env)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root', type=Path, required=True)
    parser.add_argument('--release', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--settle-seconds', type=float, default=.5)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--width', type=int, default=960)
    parser.add_argument('--height', type=int, default=720)
    parser.add_argument('--camera', action='append',
                        help='Repeat to select scene camera IDs; default all; model smoke always requires task and both wrists')
    parser.add_argument('--model-smoke', action='store_true',
                        help='After passing preflight, make two isolated model calls and one bounded own-joint step')
    parser.add_argument('--micu-credentials', type=Path, help='External explicit Micu credential profile; never copied')
    parser.add_argument('--model-timeout-seconds', type=float, default=180.)
    parser.add_argument('--model-max-output-tokens', type=int, default=8192)
    report = run(parser.parse_args())
    if not report['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
