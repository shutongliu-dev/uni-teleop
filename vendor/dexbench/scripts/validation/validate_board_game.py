"""Frozen-source native three-level board-game support, ready pose and RGB-D checks.

No model calls or placement claims. After reset only bounded joint holds are
sent; every native physics substep is observed through the frozen evaluator.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, fields
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import traceback

import numpy as np

from bench.paths import OUTPUT_ROOT, load_json
from scenes.board_games.tabletop import BoardGameGeometry, BoardGameMatch, BoardGameNativeObserver, BoardGameRules, verify_evaluator_sources
from scenes.board_games.tabletop.settlement import SettlementThresholds
from scenes.board_games.tabletop.release import verify_public_rule_contract
from bench.isaacsim.release import dump
from scripts.maintenance.freeze_project_sources import verify_if_snapshot
from scripts.validation.validate_scenario import digest, seal_report_before_shutdown
from scripts.validation.validate_workrooms import check_window, plain


REQUIRED_FILES = ('scene.json', 'task.json', 'initial_state.json')
REQUIRED_CAMERAS = ('task', 'wrist_left', 'wrist_right', 'overview', 'room')


def load_release(release):
    release = Path(release).resolve()
    manifest_path = release/'release_manifest.json'
    manifest = load_json(manifest_path)
    if manifest.get('schema') != 'robot_board_game_task_release.v1':
        raise ValueError('Expected a prepared robot_board_game_task_release.v1 release')
    hashes = manifest['files_sha256']
    if not set(REQUIRED_FILES).issubset(hashes):
        raise ValueError('Release must hash scene.json, task.json and explicit initial_state.json')
    for name, expected in hashes.items():
        path = (release/name).resolve()
        if not path.is_relative_to(release) or digest(path) != expected:
            raise ValueError('Frozen release file mismatch: '+name)
    for name, expected in manifest.get('asset_files_sha256', {}).items():
        if digest(name) != expected:
            raise ValueError('Frozen board-game asset mismatch: '+name)
    scene, task, initial = (load_json(release/name) for name in REQUIRED_FILES)
    if task.get('schema') != 'robot_board_game_task.v1' or scene.get('board_game_task', {}).get('schema') != 'board_game_native_task.v1':
        raise ValueError('Missing explicit board-game task or native scene identity')
    evaluator = verify_evaluator_sources(release, task)
    public_rules = verify_public_rule_contract(task, release=release)
    geometry = BoardGameGeometry(**{f.name: task['geometry'][f.name] for f in fields(BoardGameGeometry)})
    thresholds = SettlementThresholds(**{f.name: task['settlement'][f.name] for f in fields(SettlementThresholds)})
    def normalized(value):
        return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))
    metadata = scene['board_game_task']
    if manifest.get('task_id') != geometry.game_id or task.get('task_id') != geometry.game_id:
        raise ValueError('Manifest, task and geometry identity disagree')
    if normalized(task['geometry']) != normalized(geometry.specification()):
        raise ValueError('Task geometry specification disagrees with evaluated geometry')
    if normalized(metadata['geometry']) != normalized(asdict(geometry)):
        raise ValueError('Scene and evaluated geometry disagree')
    if normalized(task['rules']) != normalized(BoardGameRules(geometry).specification()):
        raise ValueError('Frozen rules disagree with evaluated rules')
    players = [{'player_id': 'player_0', 'color': 'black', 'robot_id': 'left'},
               {'player_id': 'player_1', 'color': 'white', 'robot_id': 'right'}]
    if metadata.get('players') != players or metadata.get('board_fixture_id') != 'board':
        raise ValueError('Scene players or board fixture identity disagree')
    if [row['id'] for row in scene['fixtures']].count('board') != 1:
        raise ValueError('Exactly one physical board fixture is required')
    board = next(row for row in scene['fixtures'] if row['id'] == 'board')
    expected_position = [*geometry.center_xy_m, geometry.table_top_z_m]
    expected_rotation = [math.cos(geometry.board_yaw_rad/2), 0., 0., math.sin(geometry.board_yaw_rad/2)]
    if (board.get('asset_id') != 'board'
            or not np.allclose(board.get('position_m', []), expected_position, atol=1e-10, rtol=0)
            or not np.allclose(board.get('quaternion_wxyz', [1., 0., 0., 0.]), expected_rotation, atol=1e-10, rtol=0)):
        raise ValueError('Physical board asset or transform differs from evaluated geometry')
    if any(row.get('asset_id') != 'token_'+row['id'].split('_')[0] for row in scene['objects']):
        raise ValueError('Token asset color differs from its inventory identity')
    expected = BoardGameMatch(geometry, thresholds=thresholds).inventory
    if ({row['id'] for row in scene['objects']} != expected or len(scene['objects']) != sum(geometry.inventory_counts.values())
            or not set(initial.get('objects', {})).issubset(expected)):
        raise ValueError('Scene must contain the exact named game inventory; reset may only override their poses')
    expected_supports = {f'{color}_{index:02d}': {'fixture_id': geometry.reserve_fixture_id(color, index),
        'surface_z_m': geometry.reserve_surface_z_m} for color, count in geometry.inventory_counts.items() for index in range(count)}
    if normalized(metadata.get('reserve_supports')) != normalized(expected_supports):
        raise ValueError('Reserve supports disagree with frozen supply-station geometry')
    fixtures = {row['id']: row for row in scene['fixtures']}
    if len(fixtures) != len(scene['fixtures']):
        raise ValueError('Fixture identities must be unique')
    for name in geometry.reserve_stations():
        fixture = fixtures.get(name, {})
        if (fixture.get('asset_id') != name
                or fixture.get('position_m') != [0, 0, 0]
                or fixture.get('quaternion_wxyz', [1, 0, 0, 0]) != [1, 0, 0, 0]):
            raise ValueError('Reserve fixture must retain its authored world transform')
    for row in scene['objects']:
        color, index = row['id'].rsplit('_', 1)
        if not np.allclose(row.get('position_m', []), geometry.reserve_position(color, int(index)), atol=1e-10, rtol=0):
            raise ValueError('Default reserve position differs from frozen geometry')
    if scene['board_game_task'].get('initial_state') != initial:
        raise ValueError('Scene and frozen initial_state.json disagree')
    if scene['robot'].get('hand_id') != 'dual_fr3_sharpa_w01':
        raise ValueError('board-game preflight requires both independent FR3 + SHARPA robots')
    return scene, task, initial, geometry, thresholds, {'manifest': str(manifest_path),
        'manifest_sha256': digest(manifest_path), 'input_sha256': {n: hashes[n] for n in REQUIRED_FILES},
        'actual_evaluator_verification': evaluator, 'public_rule_verification': public_rules}


def table_contact_paths(scene_info):
    workbench = scene_info['workbench']
    path = workbench.get('supports', {}).get('table')
    if path is None:
        if workbench.get('kind') == 'referenced_workroom':
            raise ValueError('Referenced workroom does not identify its original table support')
        path = '/World/Table'
    actor = scene_info.get('collider_actor_paths', {}).get(path, path)
    return (actor,)


def reserve_support_spec(env, match):
    """Resolve each frozen reserve surface to its actual native contact actors.

    Older table-only scenes retain the original table contract. New stations
    declare all token supports explicitly; a contact with another tray cannot
    satisfy a token's assigned support check.
    """
    meta = getattr(env, 'scene_spec', {}).get('board_game_task', {})
    declarations = meta.get('reserve_supports')
    if declarations is None:
        paths = table_contact_paths(env.scene_info)
        return {name: {'paths': paths, 'surface_z_m': match.geometry.table_top_z_m,
                       'fixture_id': None} for name in match.inventory}
    if not isinstance(declarations, dict) or set(declarations) != match.inventory:
        raise ValueError('Reserve supports must identify the complete inventory')
    supports = {}
    actors = env.scene_info.get('collider_actor_paths', {})
    for name, declaration in declarations.items():
        if not isinstance(declaration, dict) or set(declaration) != {'fixture_id', 'surface_z_m'}:
            raise ValueError('Invalid reserve support declaration')
        fixture_id, height = declaration['fixture_id'], declaration['surface_z_m']
        if (not isinstance(fixture_id, str) or not fixture_id.startswith('reserve_')
                or type(height) not in (float, int) or not math.isfinite(height)):
            raise ValueError('Invalid reserve support identity or height')
        fixture = env.scene_info.get('fixtures', {}).get(fixture_id)
        if not isinstance(fixture, dict):
            raise ValueError('Missing native reserve support fixture: '+fixture_id)
        paths = fixture.get('contact_filter_paths', fixture.get('collision_paths'))
        root = '/World/Fixtures/'+fixture_id
        if not paths or any(not isinstance(p, str) or (p != root and not p.startswith(root+'/'))
                            for p in paths):
            raise ValueError('Reserve support needs its own native contact actors')
        paths = tuple(dict.fromkeys(actors.get(p, p) for p in paths))
        supports[name] = {'paths': paths, 'surface_z_m': float(height), 'fixture_id': fixture_id}
    return supports


class ReservePreflightObserver(BoardGameNativeObserver):
    """Aggregate reserve stability from the very same complete native substeps."""

    def __init__(self, env, match, initial):
        super().__init__(env, match)
        self.table_paths = table_contact_paths(env.scene_info)
        self.reserve_supports = reserve_support_spec(env, match)
        sensors, filters = self._mapping
        for path in self._paths:
            name = self._path_piece[path]
            if not set(self.reserve_supports[name]['paths']).issubset(filters[sensors.index(path)]):
                raise RuntimeError('Every token must observe its declared reserve support actors')
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
                if name is not None and pair[1-index] in self.reserve_supports[name]['paths']:
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
                'supported_by_declared_reserve_surface': support[name] > limits.minimum_support_impulse_ns,
                'linear_speed': speed <= limits.maximum_linear_speed_m_s,
                'angular_speed': angular <= limits.maximum_angular_speed_rad_s,
                'upright': tilt <= limits.maximum_tilt_rad,
                'reserve_xy_preserved': drift <= .003,
                'base_bottom_at_reserve_surface': abs(bottom-self.reserve_supports[name]['surface_z_m']) <= limits.bottom_z_tolerance_m,
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
                'bottom_z_error_m': float(bottom-self.reserve_supports[name]['surface_z_m']),
                'declared_support': self.reserve_supports[name],
                'xy_drift_m': drift, 'maximum_xy_drift_m': self.maximum_drift[name],
                'stable_supported_dwell_s': self.stable_duration[name],
                'reserve_support_impulse_ns': self.support_impulse[name],
                'maximum_reported_penetration_m': piece.maximum_penetration_m}

    def preflight_summary(self):
        dwell = self.match.verifier.thresholds.dwell_s
        tokens = {name: {**row, 'passed': all(row['checks'].values())
                         and self.stable_duration[name]+1.e-12 >= dwell}
                  for name, row in self.last_stats.items()}
        return {'passed': len(tokens) == len(self.match.inventory) and all(row['passed'] for row in tokens.values()),
                'token_count': len(tokens), 'tokens': tokens, 'required_dwell_s': dwell,
                'table_contact_paths': list(self.table_paths), 'reserve_supports': self.reserve_supports,
                'failed_substep_conditions': dict(self.rejections),
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


def preflight_camera_ids(requested, available):
    """Qualification always includes both wrists and the three world views."""
    selected = list(dict.fromkeys(requested or available))
    selected.extend(camera for camera in REQUIRED_CAMERAS if camera not in selected)
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
    meta = env.scene_spec['board_game_task']
    geometry = BoardGameGeometry(**meta['geometry'])

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

    def cells_in_view():
        return {geometry.cell_label(q, r): sample([*geometry.cell_center(q, r)[:2],
                                                   geometry.board_top_z_m+.0001])
                for q in range(geometry.size) for r in range(geometry.size)}

    def handles_in_view(names):
        from scenes.board_games.tabletop.native_observer import _rotate_xyzw
        pieces = {}
        for name in sorted(names):
            pose = state['object_q'][env.object_names.index(name)]
            point = pose[:3]+_rotate_xyzw(pose[3:], np.array(
                [0., 0., geometry.piece_base_height_m/2+geometry.handle_height_m]))
            pieces[name] = sample(point)
        return pieces

    if camera == 'task':
        cells, pieces = cells_in_view(), handles_in_view(env.object_names)
        return {'passed': bool(pieces) and all(row['visible'] for row in (*cells.values(), *pieces.values())),
                'cells': cells, 'reserve_handle_tops': pieces,
                'scope': 'All empty cell centres and all reset inventory handle tops have actual surface depth; no filled-board perception score.'}
    if camera.startswith('wrist_'):
        side = camera.removeprefix('wrist_')
        probes = {str(i): sample(point) for i, point in enumerate(meta['home_design'][side]['wrist_probe_points_world'])}
        color = next(player['color'] for player in meta['players'] if player['robot_id'] == side)
        pieces = handles_in_view([name for name in env.object_names if name.startswith(color+'_')])
        cells = cells_in_view()
        wrist = state['body_q'][env.body_names.index(side+'_hand_flange')]
        expected = expected_camera_world(env.cameras[camera], {'position_m': wrist[:3],
                                                               'quaternion_wxyz': wrist[[6, 3, 4, 5]]})
        actual = np.linalg.inv(view)
        error = float(np.max(np.abs(expected-actual)))
        visible = bool(probes) and bool(pieces) and all(p['visible'] for p in (*probes.values(), *cells.values(), *pieces.values()))
        return {'passed': visible and error < .001,
                'workspace_probes': probes, 'cells': cells, 'own_reserve_handle_tops': pieces,
                'maximum_mount_matrix_error': error,
                'scope': 'Actual depth of all empty cell centres and own reserve handle tops at ready; fixed wrist extrinsics, no grasp or filled-board claim.'}
    return {'passed': True, 'scope': 'Context view; global board visibility is checked in task camera.'}


def ready_pose_check(env, scene, initial):
    """Compare actual native joints and flange poses with the frozen ready target."""
    from bench.geometry import quaternion_matrix
    state = env.capture_state()
    joints = np.asarray(state['joint_q'])
    if (len(env.joint_names) != 58 or len(set(env.joint_names)) != 58
            or joints.shape != (58,) or not np.isfinite(joints).all()):
        raise ValueError('Ready pose requires 58 finite native joint coordinates')
    if set(initial.get('joint_positions_rad', {})) != set(env.joint_names):
        raise ValueError('Ready state must explicitly cover every native robot joint')
    homes = scene['board_game_task']['home_design']
    if set(homes) != {'left', 'right'}:
        raise ValueError('Ready pose requires both robot targets')
    if not all(math.isfinite(value) for value in initial['joint_positions_rad'].values()):
        raise ValueError('Non-finite ready joint target')
    errors = {name: abs(float(state['joint_q'][env.joint_names.index(name)])-value)
              for name, value in initial['joint_positions_rad'].items()}
    if not all(math.isfinite(error) for error in errors.values()):
        raise ValueError('Non-finite native ready joints')
    wrists = {}
    for side, home in homes.items():
        actual = state['body_q'][env.body_names.index(side+'_hand_flange')]
        if abs(float(np.linalg.norm(actual[3:]))-1.) > .001:
            raise ValueError('Invalid native wrist quaternion')
        rotation = quaternion_matrix(actual[[6, 3, 4, 5]])
        target = np.asarray(home['wrist_rotation_matrix'])
        position = np.asarray(home['wrist_position_m'])
        if (target.shape != (3, 3) or position.shape != (3,)
                or not np.isfinite(target).all() or not np.isfinite(position).all()
                or not np.isfinite(actual).all()):
            raise ValueError('Invalid native or target wrist pose')
        if not np.allclose(target.T @ target, np.eye(3), atol=1e-6, rtol=0) or abs(np.linalg.det(target)-1.) > 1e-6:
            raise ValueError('Ready target wrist rotation must be orthonormal')
        angle = math.acos(float(np.clip((np.trace(target.T @ rotation)-1)/2, -1, 1)))
        distance = float(np.linalg.norm(actual[:3]-np.asarray(home['wrist_position_m'])))
        wrists[side] = {'position_error_m': distance, 'rotation_error_rad': angle,
                        'actual_position_m': actual[:3].tolist(),
                        'passed': distance < .005 and angle < .03}
    return {'passed': max(errors.values()) < .02 and all(row['passed'] for row in wrists.values()),
            'maximum_joint_error_rad': max(errors.values()), 'joint_errors_rad': errors,
            'wrists': wrists, 'limits': {'joint_error_rad': .02, 'wrist_position_m': .005, 'wrist_rotation_rad': .03},
            'scope': 'Actual native ready pose after settling; no approach or grasp qualification'}


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
        modules = ('scripts.validation.validate_board_game', 'bench.simulation.isaacsim_backend.env',
                   'scenes.board_games.tabletop.native_observer', 'scenes.board_games.tabletop.settlement')
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
        dump(output/'plan.json', {'schema': 'board_game_native_preflight_plan.v1',
            'created_utc': datetime.now(timezone.utc).isoformat(), 'project_root': str(root),
            'release': str(args.release.resolve()), 'release_sources': source,
            'gpu_uuid': uuid, 'driver_override': os.environ.get('ISAAC_ALLOW_UNVALIDATED_DRIVER'),
            'device': args.device, 'settle_seconds': args.settle_seconds, 'seed': args.seed,
            'resolution': [args.width, args.height], 'maximum_model_calls': 0,
            'model_smoke_requested': False,
            'task_qualification': False, 'thresholds': asdict(thresholds),
            'additional_preflight_thresholds': {'maximum_reserve_xy_drift_m': .003},
            'validator_sha256': digest(__file__)})
        env = FR3DexterousIsaacEnv(scene, device=args.device, render_width=args.width, render_height=args.height)
        observed = env.reset(seed=args.seed, initial_state=initial)
        np.savez_compressed(output/'reset_state.npz', **env.capture_state())
        dump(output/'reset_observation.json', observed)
        match = BoardGameMatch(geometry, thresholds=thresholds)
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
            expected_samples=round(args.settle_seconds/env.dt), phase='ready_pose_hold'))
        from bench.simulation.isaacsim_backend.robot_base import runtime_base_check
        base = runtime_base_check(env._robot_view, env.scene_info)
        checks['dual_fixed_bases'] = {'passed': base['all_fixed_base'] and len(base['articulations']) == 2
            and all(r['dof_count'] == 29 and r['link_count'] == 43 and r.get('root_pose_verified')
                    for r in base['articulations']), **base}
        checks['ready_pose'] = ready_pose_check(env, scene, initial)
        selected_cameras = preflight_camera_ids(args.camera, env.camera_ids)
        frames = capture_cameras(env, output, selected_cameras)
        checks['cameras'] = {'passed': bool(frames) and all(row['passed'] for row in frames), 'frames': frames}
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
    model_calls = 0
    total_steps = observer.samples//round(.02/env.dt) if observer is not None else steps
    report = {'schema': 'board_game_native_preflight.v1',
        'passed': error is None and bool(checks) and all(row['passed'] for row in checks.values()),
        'checks': checks, 'error': error, 'control_steps': total_steps,
        'preflight_control_steps': steps, 'model_calls': model_calls,
        'model_smoke_requested': False,
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
                        help='Repeat to select scene camera IDs; all five qualification cameras are always required')
    report = run(parser.parse_args())
    if not report['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
