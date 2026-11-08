"""Synthetic consumer/negative tests, not native manipulation qualification."""
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path

import pytest

from bench.paths import DATA_ROOT, load_json
from scenes.workrooms.card_room.expansion_referee import ExpansionNativeFrame, ExpansionObjectEvidence
from scenes.workrooms.card_room.practice.cases import GYM_IDS, STRATA, build_case, parent_id, selections
from scenes.workrooms.card_room.practice.referee import PracticeReferee
from scenes.workrooms.card_room.practice.release import _task


def metadata(gym):
    path = Path(f'{DATA_ROOT}/assets/task_parts/board_games')/parent_id(gym)/'scene_v2/metadata.json'
    if not path.is_file():
        pytest.skip('Original shared scene-v2 inputs unavailable')
    return load_json(path)


def make(gym='polyomino_territory_practice', stratum=None, side='left', variant=0):
    meta = metadata(gym)
    stratum = stratum or STRATA[parent_id(gym)][0]
    case = build_case(gym, {'active_side': side, 'case_id': f'{stratum}_{variant:02d}',
        'split': 'train' if variant < 4 else 'validation'}, meta)
    task = _task(gym, {'asset_metadata': meta}, case)
    return PracticeReferee(task)


def baseline(ref, index=1):
    return ExpansionNativeFrame(index, index*.0005, .0005,
        {name: ExpansionObjectEvidence(tuple(p['position_m']), tuple(p['quaternion_wxyz']), (0., 0., 0.), (0., 0., 0.),
            ref.helper._masses[name]*9.81*.0005, {'player_0': 0., 'player_1': 0.}, 0., {},
            {'player_0': 0., 'player_1': 0.}) for name, p in ref.case['reset'].items()}, True)


def advance(frame, **objects):
    return replace(frame, substep_index=frame.substep_index+1,
        simulation_time_s=frame.simulation_time_s+frame.dt_s, objects=frame.objects | objects)


def target_objects(ref, frame, *, contact):
    objects = {}
    for i, target in enumerate(ref.stage['targets']):
        name = target['object_id']
        p = target['position_m']
        if p is None:
            bounds = target['region_bounds_xy_m']
            p = [(bounds[0][0]+bounds[1][0])/2+(i-.5)*.08,
                 (bounds[0][1]+bounds[1][1])/2, .816]
        q = target['quaternion_wxyz'] or ([0., 1., 0., 0.] if target['orientation'] == 'face_up' else [1., 0., 0., 0.])
        objects[name] = replace(frame.objects[name], position_m=tuple(p), quaternion_wxyz=tuple(q),
            robot_contact_impulses_ns={'player_0': float(contact and ref.player == 'player_0'),
                'player_1': float(contact and ref.player == 'player_1')})
    return objects


def finish_stage(ref, frame):
    frame = advance(frame, **target_objects(ref, frame, contact=True))
    ref.observe(frame)
    frame = advance(frame, **target_objects(ref, frame, contact=False))
    ref.observe(frame)
    for _ in range(600):
        frame = advance(frame)
        ref.observe(frame)
    return frame


@pytest.mark.parametrize('gym', GYM_IDS)
def test_full_curricula_bind_distinct_real_case_content_without_hidden_public_fields(gym):
    meta = metadata(gym)
    cases = [build_case(gym, selection, meta) for selection in selections(gym)]
    assert len(cases) == len({c['case_fingerprint'] for c in cases}) == 48
    assert {c['public']['active_side'] for c in cases} == {'left', 'right'}
    for case in cases:
        public = json.dumps(case['public'])
        assert 'private_reset_pairs' not in public and 'case_fingerprint' not in public and 'private_case_sha256' not in public
        if parent_id(gym) == 'memory_pairs':
            assert all(f'"pair_{i}"' not in public for i in range(6))
        ref = PracticeReferee(_task(gym, {'asset_metadata': meta}, case))
        frame = baseline(ref)
        ref.observe(frame)
        for _ in range(4):
            frame = advance(frame)
            ref.observe(frame)
        assert not ref.score()['success']
        assert ref.state == 'ongoing'


@pytest.mark.parametrize('side', ['left', 'right'])
@pytest.mark.parametrize('gym,stratum', [(g, s) for g in GYM_IDS[:3] for s in STRATA[parent_id(g)]])
def test_only_actual_hand_evidence_and_fresh_released_stage_windows_can_confirm(gym, stratum, side):
    ref = make(gym, stratum, side)
    frame = baseline(ref)
    ref.observe(frame)
    count = len(ref.public['goal']['stages'])
    for _ in range(count):
        frame = finish_stage(ref, frame)
    assert ref.score()['success'], ref.score()
    assert len(ref.confirmations) == count
    assert all(c['simulation_time_s']-c['stable_start_s'] >= .3-1e-10 for c in ref.confirmations)
    assert ref.score()['formal_game_result'] is False


@pytest.mark.parametrize('fault', ['no_contact', 'floor', 'nonhand_only', 'wrong_side', 'gap', 'penetration', 'inventory'])
def test_false_physical_success_is_rejected(fault):
    ref = make()
    frame = baseline(ref)
    ref.observe(frame)
    objects = target_objects(ref, frame, contact=fault != 'no_contact')
    name = ref.stage['targets'][0]['object_id']
    obj = objects[name]
    if fault == 'nonhand_only':
        obj = replace(obj, robot_nonhand_contact_impulses_ns={'player_0': 1., 'player_1': 0.})
    if fault == 'wrong_side':
        obj = replace(obj, robot_contact_impulses_ns={'player_0': 0., 'player_1': 1.})
    if fault == 'penetration':
        obj = replace(obj, maximum_penetration_m=.004)
    objects[name] = obj
    frame = advance(frame, **objects)
    if fault == 'gap':
        frame = replace(frame, substep_index=frame.substep_index+1)
    ref.observe(frame)
    released = target_objects(ref, frame, contact=False)
    if fault == 'floor':
        released[name] = replace(released[name], deck_support_impulse_ns=0., other_static_support_impulse_ns=1.)
    if fault == 'inventory':
        other = 'white_mono'
        released[other] = replace(frame.objects[other], position_m=(.65, 0., .817))
    frame = advance(frame, **released)
    for _ in range(610):
        ref.observe(frame)
        frame = advance(frame)
    assert not ref.score()['success']


def test_three_hundred_ms_needs_endpoints_and_fresh_contact_each_stage():
    ref = make(stratum='multi_piece_adjacent')
    frame = baseline(ref)
    ref.observe(frame)
    frame = advance(frame, **target_objects(ref, frame, contact=True)); ref.observe(frame)
    frame = advance(frame, **target_objects(ref, frame, contact=False)); ref.observe(frame)
    for _ in range(599):
        frame = advance(frame); ref.observe(frame)
    assert ref.stage_index == 0
    frame = advance(frame); ref.observe(frame)
    assert ref.stage_index == 1
    frame = advance(frame, **target_objects(ref, frame, contact=False))
    for _ in range(610):
        ref.observe(frame); frame = advance(frame)
    assert not ref.score()['success']


@pytest.mark.parametrize('stratum', STRATA['disk_curling'])
@pytest.mark.parametrize('side', ['left', 'right'])
def test_curling_needs_contact_release_crossing_support_and_optional_obstacle(stratum, side):
    ref = make('disk_curling_practice', stratum, side)
    frame = baseline(ref); ref.observe(frame)
    name = ref.stage['targets'][0]['object_id']
    direction = 1 if side == 'left' else -1
    obj = replace(frame.objects[name], position_m=(frame.objects[name].position_m[0], -direction*.31, .815),
        linear_velocity_m_s=(0., direction*.08, 0.), robot_contact_impulses_ns={ref.player: .01, ref.other: 0.})
    frame = advance(frame, **{name: obj}); ref.observe(frame)
    obj = replace(obj, position_m=(obj.position_m[0], -direction*.28, .815), robot_contact_impulses_ns={'player_0': 0., 'player_1': 0.})
    frame = advance(frame, **{name: obj}); ref.observe(frame)
    obj = replace(obj, position_m=(.65, 0., .815), linear_velocity_m_s=(0., 0., 0.))
    frame = advance(frame, **{name: obj})
    if stratum == 'obstacle_contact':
        paths = ref.task['asset_metadata']['actors']
        ref.contact_rows = [{'actor0': paths[name], 'actor1': paths[ref.stage['obstacle_ids'][0]],
            'time_s': frame.simulation_time_s, 'impulse_magnitude_sum_ns': .001}]
    ref.observe(frame)
    ref.contact_rows = ()
    for _ in range(600):
        frame = advance(frame); ref.observe(frame)
    assert ref.score()['success'], ref.score()


def test_budget_interruption_is_incomplete_not_game_win_or_physical_failure():
    ref = make()
    ref.interrupt('budget')
    assert ref.terminal and not ref.score()['success']
    assert ref.score()['referee']['state'] == 'ongoing'
    assert ref.score()['formal_game_result'] is False


@pytest.mark.parametrize('fault', ['nonhand', 'second_push', 'late_release', 'missing_obstacle', 'floor_score'])
def test_curling_physical_rule_negative_cases(fault):
    stratum = 'obstacle_contact' if fault == 'missing_obstacle' else 'target_score'
    ref = make('disk_curling_practice', stratum)
    frame = baseline(ref); ref.observe(frame)
    name = ref.stage['targets'][0]['object_id']
    obj = replace(frame.objects[name], position_m=(.65, -.31, .815),
        linear_velocity_m_s=(0., .08, 0.), robot_contact_impulses_ns={'player_0': .01, 'player_1': 0.})
    if fault == 'nonhand':
        obj = replace(obj, robot_nonhand_contact_impulses_ns={'player_0': .01, 'player_1': 0.})
    frame = advance(frame, **{name: obj}); ref.observe(frame)
    obj = replace(obj, position_m=(.65, -.20 if fault == 'late_release' else -.28, .815),
        robot_contact_impulses_ns={'player_0': .01 if fault == 'late_release' else 0., 'player_1': 0.})
    frame = advance(frame, **{name: obj}); ref.observe(frame)
    obj = replace(obj, position_m=(.65, 0., .815), linear_velocity_m_s=(0., 0., 0.),
        robot_contact_impulses_ns={'player_0': .01 if fault == 'second_push' else 0., 'player_1': 0.})
    if fault == 'floor_score':
        obj = replace(obj, position_m=(.65, 0., .01), deck_support_impulse_ns=0., other_static_support_impulse_ns=1.)
    frame = advance(frame, **{name: obj})
    for _ in range(605):
        ref.observe(frame); frame = advance(frame)
    assert not ref.score()['success']


def test_launch_is_a_declared_subskill_without_falsely_claiming_line_crossing():
    ref = make('disk_curling_practice', 'launch')
    frame = baseline(ref); ref.observe(frame)
    name = ref.stage['targets'][0]['object_id']
    obj = replace(frame.objects[name], position_m=(.52, -.32, .815),
        linear_velocity_m_s=(0., .08, 0.), robot_contact_impulses_ns={'player_0': .01, 'player_1': 0.})
    frame = advance(frame, **{name: obj}); ref.observe(frame)
    obj = replace(obj, position_m=(.52, -.30, .815), linear_velocity_m_s=(0., 0., 0.),
        robot_contact_impulses_ns={'player_0': 0., 'player_1': 0.})
    frame = advance(frame, **{name: obj})
    for _ in range(601):
        ref.observe(frame); frame = advance(frame)
    assert ref.score()['success'] and ref.shot['crossed'] is False
