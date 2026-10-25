"""CPU evidence doubles for the diagnostic harness; no native qualification."""
from copy import deepcopy
from dataclasses import asdict, replace
import json

import pytest

from scenes.board_games.tabletop import SettlementThresholds, geometry_for
from scenes.board_games.tabletop.settlement import NativeFrame, PieceEvidence
from scripts.validation.validate_board_game_rules import (
    CASE_IDS, assess_case, control_count, diagnostic_cases, diagnostic_match)


def fixtures():
    geometry, limits = geometry_for('tic_tac_toe_3x3'), SettlementThresholds()
    initial = {'joint_positions_rad': {'left_joint': 0.2}, 'objects': {
        f'{color}_{i:02d}': {'position_m': list(geometry.reserve_position(color, i)),
                           'quaternion_wxyz': [1., 0., 0., 0.]}
        for color, count in geometry.inventory_counts.items() for i in range(count)}}
    return geometry, limits, initial


def evidence(position, **changes):
    return replace(PieceEvidence(tuple(position), (1., 0., 0., 0.), (0., 0., 0.),
                                 (0., 0., 0.), 0., 0., 0.), **changes)


def synthetic_records(case, geometry, limits):
    """Deterministic test evidence, deliberately not a simulator substitute."""
    match = diagnostic_match(case, geometry, limits)
    records = []
    for i in range(60):
        pieces = {name: evidence(row['position_m']) for name, row in case['initial_state']['objects'].items()}
        for name, pose in case['object_pose_overrides'].items():
            p = list(pose['position_m'])
            # Mimic an initial unsupported interval followed by resting support.
            if i >= 5:
                p[2] = geometry.board_top_z_m+geometry.piece_base_height_m/2
            pieces[name] = evidence(p, board_support_impulse_ns=.001 if i >= 5 else 0.)
        frame = NativeFrame(i+100, (i+1)*.01, .01, pieces, True)
        result = match.observe(frame)
        records.append({'frame': asdict(frame), 'result': asdict(result), 'unsupported': []})
    summary = {'samples': len(records), 'unsupported_samples': 0, 'closed': True, 'failed': False,
        'final_context_verified': True, 'physical_steps_added': 0, 'physical_state_setters_called': 0,
        'commits': len(match.commit_log)}
    return records, summary, match.commit_log


def test_fixture_overrides_preserve_release_and_robot_and_expose_logical_seed():
    geometry, limits, initial = fixtures()
    before = deepcopy(initial)
    cases = diagnostic_cases(geometry, limits, initial)
    assert tuple(case['case_id'] for case in cases) == CASE_IDS
    assert initial == before
    for case in cases:
        assert case['initial_state']['joint_positions_rad'] == initial['joint_positions_rad']
        for name in initial['objects']:
            if name not in case['object_pose_overrides']:
                assert case['initial_state']['objects'][name] == initial['objects'][name]
        match = diagnostic_match(case, geometry, limits)
        assert not match.commit_log and match.rules.turn_index == len(case['logical_seed'])
    offset = cases[0]['object_pose_overrides']['black_00']['position_m']
    crossed = cases[1]['object_pose_overrides']['black_00']['position_m']
    assert geometry.footprint_clearance_m(1, 1, offset[:2], geometry.piece_radius_m) > limits.edge_clearance_m
    assert geometry.footprint_clearance_m(1, 1, crossed[:2], geometry.piece_radius_m) < 0
    cases[-1]['logical_seed'][0].pop('source')
    with pytest.raises(ValueError, match='disclosed'):
        diagnostic_match(cases[-1], geometry, limits)


@pytest.mark.parametrize('game_id', ['connect_four_5x5', 'hex7x7'])
def test_other_level_cannot_be_silently_diagnosed_as_l1(game_id):
    with pytest.raises(ValueError, match='L1'):
        diagnostic_cases(geometry_for(game_id), SettlementThresholds(), {})


@pytest.mark.parametrize('case_index', range(4))
def test_saved_native_shape_evidence_replays_intended_case(case_index):
    geometry, limits, initial = fixtures()
    case = diagnostic_cases(geometry, limits, initial)[case_index]
    records, observer, commits = synthetic_records(case, geometry, limits)
    report = assess_case(case, geometry, limits, records, observer, commits, expected_samples=60)
    assert report['passed'], report
    # Exercise the actual JSON artifact boundary; the off-centre norm check
    # previously returned numpy.bool_ and failed only after native completion.
    assert all(type(value) is bool for value in report['checks'].values())
    assert json.loads(json.dumps(report, allow_nan=False)) == report
    assert report['final_public_status']['outcome'] == 'ongoing'
    assert report['final_public_status']['retryable']
    if case_index == 3:
        assert len(commits) == 1 and commits[0]['piece_id'] == 'white_00'
        assert report['logical_seed_count_not_native_commits'] == 1
        assert report['joint_stable_dwell_at_commits_s'] == pytest.approx([.3])


@pytest.mark.parametrize('corruption', ['missing_frame', 'false_commit', 'missing_support',
                                      'wrong_commit_log', 'not_closed', 'unsupported', 'no_offset'])
def test_diagnostic_acceptance_fails_closed_on_modified_evidence(corruption):
    geometry, limits, initial = fixtures()
    case = diagnostic_cases(geometry, limits, initial)[0]
    records, observer, commits = synthetic_records(case, geometry, limits)
    if corruption == 'missing_frame':
        records.pop(20)
    elif corruption == 'false_commit':
        records[10]['result']['committed'] = True
    elif corruption == 'missing_support':
        records[-1]['frame']['pieces']['black_00']['board_support_impulse_ns'] = 0.
    elif corruption == 'wrong_commit_log':
        commits[0]['piece_id'] = 'black_01'
    elif corruption == 'not_closed':
        observer['closed'] = False
    elif corruption == 'unsupported':
        records[0]['unsupported'] = ['missing_contact_separation']
    elif corruption == 'no_offset':
        records[-1]['frame']['pieces']['black_00']['position_m'] = geometry.cell_center(1, 1)
    assert not assess_case(case, geometry, limits, records, observer, commits, expected_samples=60)['passed']


def test_seeded_commit_cannot_credit_time_before_joint_recovery():
    geometry, limits, initial = fixtures()
    case = diagnostic_cases(geometry, limits, initial)[3]
    records, observer, commits = synthetic_records(case, geometry, limits)
    # A late disturbance must restart the complete common dwell. Retain the old
    # logged commit to verify the harness detects a falsely early acceptance.
    records[30]['frame']['pieces']['black_00']['linear_velocity_m_s'] = (.03, 0., 0.)
    report = assess_case(case, geometry, limits, records, observer, commits, expected_samples=60)
    assert not report['passed']
    assert not report['checks']['saved_results_replay_identically']
    assert not report['checks']['native_commits_match']


@pytest.mark.parametrize('duration', [float('nan'), float('inf'), -.2, .3, .601, 3.])
def test_duration_rejects_invalid_or_unbounded_native_work(duration):
    with pytest.raises(ValueError):
        control_count(duration, .3)


def test_default_duration_is_bounded_and_exceeds_dwell():
    assert control_count(.6, .3) == 30
