"""Native L1 rule diagnostics with explicit reset fixtures and joint holds only.

These four fixtures test the physical referee, not robot grasp/place skill or
the release's default initialization. No model is called. An external launcher
must retain stdout/stderr and require a zero process exit after report sealing.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict, replace
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import traceback

import numpy as np

from bench.paths import OUTPUT_ROOT
from scenes.board_games.tabletop import BoardGameMatch, BoardGameNativeObserver
from scenes.board_games.tabletop.settlement import NativeFrame, PieceEvidence
from bench.isaacsim.release import dump
from scripts.maintenance.freeze_project_sources import verify_if_snapshot
from scripts.validation.validate_board_game import load_release
from scripts.validation.validate_scenario import digest, seal_report_before_shutdown
from scripts.validation.validate_workrooms import check_window, plain


CASE_IDS = ('offset_within_cell', 'crosses_cell_edge', 'extra_piece_on_board',
            'seeded_piece_gravity_recovery')


def diagnostic_cases(geometry, thresholds, initial):
    """Derive explicitly disclosed pose overrides; never mutate release inputs."""
    if geometry.game_id != 'tic_tac_toe_3x3' or geometry.shape != 'square' or geometry.size != 3:
        raise ValueError('This bounded diagnostic suite supports the L1 3x3 release only')
    slack = geometry.pitch_m/2-geometry.piece_radius_m-thresholds.edge_clearance_m
    if slack <= 0:
        raise ValueError('L1 cell has no positive footprint placement slack')

    def pose(q, r, *, dx=0., gap=.004):
        x, y = geometry.cell_local_xy(q, r)
        return {'position_m': [*geometry.local_to_world_xy(x+dx, y),
                               geometry.cell_center(q, r)[2]+gap],
                'quaternion_wxyz': [1., 0., 0., 0.]}

    definitions = (
        ('offset_within_cell', ('player_0', 'black_00', 1, 1),
         {'black_00': pose(1, 1, dx=slack/2)}, [], True, None),
        ('crosses_cell_edge', ('player_0', 'black_00', 1, 1),
         {'black_00': pose(1, 1, dx=geometry.pitch_m/2)}, [], False,
         'piece_footprint_outside_cell'),
        ('extra_piece_on_board', ('player_0', 'black_00', 0, 0),
         {'black_00': pose(0, 0), 'white_00': pose(2, 2)}, [], False,
         'unrequested_piece_on_board:white_00'),
        ('seeded_piece_gravity_recovery', ('player_1', 'white_00', 1, 1),
         {'black_00': pose(0, 0, gap=.012), 'white_00': pose(1, 1)},
         [{'player_id': 'player_0', 'piece_id': 'black_00', 'q': 0, 'r': 0,
           'source': 'diagnostic_fixture_not_native_commit'}], True,
         'committed_piece_needs_recovery:black_00:'),
    )
    result = []
    for name, intent, overrides, seed, commits, reason in definitions:
        reset = deepcopy(initial)
        reset.setdefault('objects', {})
        for piece_id, override in overrides.items():
            reset['objects'][piece_id] = {**reset['objects'].get(piece_id, {}), **override}
        cells = {intent[1]: list(intent[2:])}
        cells.update({row['piece_id']: [row['q'], row['r']] for row in seed})
        if name == 'extra_piece_on_board':
            cells['white_00'] = [2, 2]
        result.append({'case_id': name, 'initial_state': reset, 'object_pose_overrides': overrides,
            'diagnostic_cells': cells,
            'logical_seed': seed, 'intent': dict(zip(('player_id', 'piece_id', 'q', 'r'), intent)),
            'expected_native_commits': int(commits), 'required_reason_prefix': reason,
            'scope': 'diagnostic_reset_fixture; no robot placement or default-initialization qualification',
            'recovery_scope': ('Observe gravity restoration of a synthetic logical seed' if seed else
                'Observe rejection and retryability only; no physical clearing/retry is performed')})
    return result


def diagnostic_match(case, geometry, thresholds):
    match = BoardGameMatch(geometry, thresholds=thresholds)
    for seed in case['logical_seed']:
        if seed.get('source') != 'diagnostic_fixture_not_native_commit':
            raise ValueError('Logical fixture seeds must be explicitly disclosed')
        # Test-fixture-only logical setup, before observation or physics. No
        # production setter, native commit event or physical placement is claimed.
        match.rules.play(seed['player_id'], seed['q'], seed['r'])
        match._committed[seed['piece_id']] = (seed['q'], seed['r'])
    match.begin_move(**case['intent'])
    return match


def assess_case(case, geometry, thresholds, records, observer, commits, *, expected_samples):
    """Replay saved private native evidence and require the intended diagnosis."""
    replay = diagnostic_match(case, geometry, thresholds)
    identical = True
    reasons, commit_rows, stable_start, joint_dwell = [], [], None, []
    complete = True
    for row in records:
        raw = row['frame']
        frame = NativeFrame(**{**raw, 'pieces': {name: PieceEvidence(**piece)
                                               for name, piece in raw['pieces'].items()}})
        complete &= frame.contacts_complete and not row['unsupported']
        # Independently measure the complete consecutive stable interval before
        # a commit. Include every seeded/committed piece and the pending token.
        pending = replay.pending
        cells = replay.committed
        if pending is not None:
            cells[pending.piece_id] = (pending.q, pending.r)
        stable = (frame.contacts_complete and set(frame.pieces) == replay.inventory
                  and all(replay.verifier._valid_piece(p) for p in frame.pieces.values())
                  and all(replay.verifier.stability_rejection(frame.pieces[name], *cell) is None
                          for name, cell in cells.items())
                  and not any(name not in cells and piece.board_support_impulse_ns > thresholds.minimum_support_impulse_ns
                              for name, piece in frame.pieces.items()))
        if not stable:
            stable_start = None
        elif stable_start is None:
            stable_start = frame.simulation_time_s
        actual = replay.observe(frame)
        identical &= asdict(actual) == row['result']
        reasons.append(actual.reason)
        if actual.committed:
            commit_rows.append(row)
            joint_dwell.append(0. if stable_start is None else frame.simulation_time_s-stable_start)
    expected_commits = case['expected_native_commits']
    prefix = case['required_reason_prefix']
    last = records[-1]['frame']['pieces'] if records else {}
    targets = case['object_pose_overrides']
    supported = bool(last) and all(
        last[name]['board_support_impulse_ns'] > thresholds.minimum_support_impulse_ns
        and last[name]['all_robot_contact_impulse_ns'] <= thresholds.maximum_robot_contact_impulse_ns
        for name in targets)
    physical_rest = bool(last)
    for name, cell in case['diagnostic_cells'].items():
        if name not in last:
            physical_rest = False
            continue
        evidence = PieceEvidence(**last[name])
        if case['case_id'] == 'crosses_cell_edge':
            # Assessment only: remove the intended XY rule violation to check
            # the unchanged native height/tilt/contact/velocity rest evidence.
            evidence = replace(evidence, position_m=(*geometry.cell_center(*cell)[:2], evidence.position_m[2]))
        physical_rest &= replay.verifier.stability_rejection(evidence, *cell) is None
    status = replay.public_status()
    checks = {
        'complete_native_samples': (len(records) == expected_samples and complete
            and observer['samples'] == expected_samples and observer['unsupported_samples'] == 0),
        'native_context': (observer['closed'] and not observer['failed']
            and observer['final_context_verified'] is True
            and observer['physical_steps_added'] == 0 and observer['physical_state_setters_called'] == 0),
        'saved_results_replay_identically': identical,
        'native_commits_match': (len(commit_rows) == expected_commits == observer['commits']
            and commits == replay.commit_log),
        'full_joint_stable_dwell': all(dwell+1.e-12 >= thresholds.dwell_s for dwell in joint_dwell),
        'required_rejection_observed': prefix is None or any(reason.startswith(prefix) for reason in reasons),
        'diagnostic_tokens_finally_board_supported_and_released': supported,
        'diagnostic_tokens_finally_at_physical_rest': physical_rest,
        'gravity_fall_observed': bool(last) and all(
            pose['position_m'][2]-last[name]['position_m'][2] >= .002 for name, pose in targets.items()),
        'no_forfeit_or_terminal': (not replay.terminal and replay.invalid_reason is None
            and replay.rules.winner is None and status['outcome'] == 'ongoing' and status['retryable']),
        'only_expected_turn_changes': replay.rules.turn_index == len(case['logical_seed'])+expected_commits,
        'rejection_remains_recoverable': expected_commits == 1 or (
            replay.pending is not None and bool(reasons) and reasons[-1].startswith(prefix)),
        'no_evidence_faults': not any(row['result']['evidence_unavailable'] for row in records),
    }
    if case['logical_seed']:
        first_commit = next((i for i, row in enumerate(records) if row['result']['committed']), len(records))
        checks['recovery_precedes_commit'] = any(reason.startswith(prefix) for reason in reasons[:first_commit])
    if case['case_id'] == 'offset_within_cell':
        name = case['intent']['piece_id']
        center = np.asarray(geometry.cell_center(case['intent']['q'], case['intent']['r'])[:2])
        requested_offset = np.linalg.norm(np.asarray(targets[name]['position_m'][:2])-center)
        checks['off_center_position_retained'] = bool(last) and np.linalg.norm(
            np.asarray(last[name]['position_m'][:2])-center) >= requested_offset/2
    # Numeric geometry checks may produce numpy.bool_, which the native JSON
    # writer rejects. Normalize at this report boundary, before artifact writes.
    return plain({'passed': all(checks.values()), 'checks': checks,
        'native_commits': commits, 'joint_stable_dwell_at_commits_s': joint_dwell,
        'final_public_status': status, 'final_turn_index': replay.rules.turn_index,
        'samples': len(records), 'required_dwell_s': thresholds.dwell_s,
        'logical_seed_count_not_native_commits': len(case['logical_seed'])})


def run_case(env, case, geometry, thresholds, output, *, seed, count):
    output.mkdir()
    dump(output/'case.json', case)
    observed = env.reset(seed=seed, initial_state=case['initial_state'])
    states = [env.capture_state()]
    np.savez_compressed(output/'reset_state.npz', **states[0])
    dump(output/'reset_observation.json', observed)
    dump(output/'scene_info.json', env.scene_info)
    for name, pose in case['object_pose_overrides'].items():
        actual = states[0]['object_q'][env.object_names.index(name)]
        expected = [*pose['position_m'], *pose['quaternion_wxyz'][1:], pose['quaternion_wxyz'][0]]
        if not np.allclose(actual, expected, atol=1.e-5, rtol=0):
            raise RuntimeError('Actual reset pose differs from declared diagnostic fixture: '+name)
    match = diagnostic_match(case, geometry, thresholds)
    observer = BoardGameNativeObserver(env, match, frame_jsonl=output/'native_frames.jsonl').attach()
    action = {'joint_positions_rad': {**env.action_spec()['initial_joint_positions_rad'],
                                     **case['initial_state'].get('joint_positions_rad', {})}, 'duration_s': .02}
    env._validate_action(action)
    dump(output/'hold_command.json', action)
    try:
        with (output/'contacts.jsonl').open('x') as contacts, (output/'progress.jsonl').open('x') as progress:
            for index in range(1, count+1):
                observed = env.step(action)
                state = env.capture_state()
                check_window(observed, state, states[-1], env.dt)
                states.append(state)
                contacts.write(json.dumps(plain({'control_step': index, 'time_s': env.time_s,
                                                 'contacts': observed['contacts']}), allow_nan=False)+'\n')
                row = {'control_step': index, 'time_s': env.time_s, 'samples': observer.samples,
                       'native_commits': observer.commits, **match.public_status()}
                progress.write(json.dumps(row, allow_nan=False)+'\n')
                if index % 10 == 0 or index == count:
                    print(json.dumps({'event': 'rule_diagnostic', 'case_id': case['case_id'], **row}), flush=True)
    finally:
        summary = observer.close()
        dump(output/'observer.json', summary)
        dump(output/'commits.json', match.commit_log)
        np.savez_compressed(output/'states.npz', **{name: np.stack([s[name] for s in states]) for name in states[0]})
        np.savez_compressed(output/'final_state.npz', **env.capture_state())
        dump(output/'final_observation.json', env.observation())
        dump(output/'metadata_final.json', env.metadata())
    records = [json.loads(line) for line in (output/'native_frames.jsonl').read_text().splitlines()]
    report = assess_case(case, geometry, thresholds, records, summary, match.commit_log,
                         expected_samples=count*round(.02/env.dt))
    report.update(case_id=case['case_id'], control_steps=count,
                  artifacts_sha256={str(p.relative_to(output)): digest(p) for p in output.iterdir() if p.is_file()})
    dump(output/'case_report.json', report)
    return report


def control_count(duration, dwell):
    if not math.isfinite(duration):
        raise ValueError('Duration must be finite')
    count = round(duration/.02)
    if abs(count*.02-duration) > 1.e-9 or not dwell+.10 <= duration <= 2.:
        raise ValueError('Diagnostic duration must be a 20ms multiple between dwell+100ms and 2s')
    return count


def run(args):
    output = args.output.resolve()
    if not output.is_relative_to(OUTPUT_ROOT):
        raise ValueError('Native rule diagnostic artifacts belong under shared datasets')
    output.mkdir(parents=True, exist_ok=False)
    env, error, reports = None, None, []
    try:
        root = args.project_root.resolve()
        if root != Path(__file__).resolve().parents[2]:
            raise ValueError('--project-root must be the actual frozen runner source root')
        from bench.simulation.isaacsim_backend.env import FR3DexterousIsaacEnv
        modules = ('scripts.validation.validate_board_game_rules', 'scripts.validation.validate_board_game',
                   'bench.simulation.isaacsim_backend.env', 'scenes.board_games.tabletop.native_observer',
                   'scenes.board_games.tabletop.settlement')
        verification = verify_if_snapshot(root, required_modules=modules, entrypoint_file=Path(__file__))
        if verification is None:
            raise ValueError('A complete frozen source snapshot is required')
        dump(output/'runtime_imports.json', verification)
        uuid = os.environ.get('CUDA_VISIBLE_DEVICES', '')
        if (not uuid.startswith('GPU-') or ',' in uuid or args.device != 'cuda:0'
                or os.environ.get('ISAAC_BENCH_SINGLE_CUDA_DEVICE') != '1'):
            raise ValueError('Select one physical GPU UUID in single CUDA mode with device cuda:0')
        scene, task, initial, geometry, thresholds, source = load_release(args.release)
        cases = diagnostic_cases(geometry, thresholds, initial)
        count = control_count(args.duration, thresholds.dwell_s)
        dump(output/'plan.json', {'schema': 'board_game_native_rules_plan.v1',
            'created_utc': datetime.now(timezone.utc).isoformat(), 'project_root': str(root),
            'release': str(args.release.resolve()), 'release_sources': source,
            'validator_sha256': digest(__file__), 'gpu_uuid': uuid,
            'driver_override': os.environ.get('ISAAC_ALLOW_UNVALIDATED_DRIVER'),
            'seed': args.seed, 'duration_per_case_s': args.duration, 'thresholds': asdict(thresholds),
            'case_ids': list(CASE_IDS), 'model_calls': 0, 'diagnostic_reset_overrides': True,
            'logical_seed_is_not_native_placement': True,
            'default_release_initialization_preflight': False, 'camera_qualification': False})
        env = FR3DexterousIsaacEnv(scene, device=args.device, render_width=320, render_height=240)
        for case in cases:
            reports.append(run_case(env, case, geometry, thresholds, output/case['case_id'],
                                    seed=args.seed, count=count))
    except BaseException as exc:
        error = {'type': type(exc).__name__, 'detail': str(exc), 'traceback': traceback.format_exc()}
        print(json.dumps({'event': 'error', **error}), flush=True)
    report = {'schema': 'board_game_native_rules_diagnostic.v1',
        'passed': error is None and len(reports) == len(CASE_IDS) and all(r['passed'] for r in reports),
        'cases': reports, 'error': error, 'model_calls': 0, 'task_qualification': False,
        'placement_success_claimed': False, 'robot_grasp_or_place_qualified': False,
        'object_state_writes_after_reset': False, 'joint_targets_only_after_reset': True,
        'scope': 'L1 referee diagnostics under disclosed reset overrides, not default release preflight',
        'limitations': ['No robot grasp, transport, release action or game completion is exercised',
            'The seeded committed token is a synthetic logical fixture, not a recorded prior placement',
            'Rejected fixtures remain in place; only retryability is checked, not robot correction',
            'No images or model observations are qualified; raw evidence is private evaluator data',
            'External process exit receipt and native error log review are required'],
        'artifacts_sha256': {str(p.relative_to(output)): digest(p) for p in output.rglob('*') if p.is_file()}}
    return seal_report_before_shutdown(output, plain(report), env)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root', type=Path, required=True)
    parser.add_argument('--release', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--duration', type=float, default=.6)
    parser.add_argument('--seed', type=int, default=0)
    report = run(parser.parse_args())
    if not report['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
