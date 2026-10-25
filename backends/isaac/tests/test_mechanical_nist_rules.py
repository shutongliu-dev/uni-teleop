"""Adversarial CPU referee receipts; no native physics qualification is implied."""
from copy import deepcopy
import math

import numpy as np
import pytest

from scenes.workrooms.mechanical_assembly import nist_rules as rules
from scenes.workrooms.mechanical_assembly.nist_evaluation import NativeTaskObserver, SequenceEvaluator
from scenes.workrooms.mechanical_assembly.nist_native_evidence import (
    Contacts, enclosed, fixed, route_in_order, simple_loop,
)


def fixture(task_id='nist_connector_insertion', goal_reset=False):
    """Small geometry fixture, independent of authoring and external assets."""
    targets, goals, groups = {}, [], []
    fixtures = {'workplate': '/World/NistTask/workplate',
                'collection_tray': '/World/NistTask/collection_tray'}

    def seat(name, target):
        targets[name] = list(target)
        socket = name+'_socket'
        fixtures[socket] = '/World/NistTask/'+socket
        goals.append(dict(id=name, actor=name, target_position_m=list(target),
            target_quaternion_wxyz=[1., 0., 0., 0.],
            position_tolerance_m=[.006, .006, .008], orientation_tolerance_rad=.35,
            support_ids=[socket], contact_ids=[socket]))

    if task_id == 'nist_connector_insertion':
        seat('usb_plug', [0., .12, .8])
        seat('bnc_plug', [.06, .12, .8])
        required = list(targets)
    elif task_id == 'nist_belt_drive_assembly':
        fixtures.update({name: '/World/NistTask/'+name
                         for name in ('drive_pulley', 'timing_pulley')})
        ids = ['belt_'+str(i) for i in range(4)]
        for name, point in zip(ids, [[-.10, -.06, .8], [.10, -.06, .8],
                                    [.10, .06, .8], [-.10, .06, .8]]):
            targets[name] = point
        groups.append(dict(id='belt', actor_ids=ids, closed=True,
            maximum_neighbor_distance_m=.25, support_ids=['workplate'],
            required_mating_contact_ids=['drive_pulley', 'timing_pulley'],
            minimum_support_fraction=.04, perimeter_range_m=[.60, .70],
            enclosed_targets=[dict(center_m=[-.035, 0., .8], radius_m=.015),
                              dict(center_m=[.035, 0., .8], radius_m=.015)]))
        required = [ids[0]]
    else:
        colors = ['cable'] if task_id == 'nist_cable_routing' else ['red', 'blue']
        for index, color in enumerate(colors):
            ids = [color+'_'+str(i) for i in range(4)]
            dy = index*.025
            polyline = [[0., 0.+dy, .8], [0., .10+dy, .8],
                        [.10, .10+dy, .8], [.10, .18+dy, .8]]
            targets.update(zip(ids, polyline))
            seat(ids[-1], polyline[-1])
            if task_id == 'nist_wire_harness':
                seat(ids[0], polyline[0])
            groups.append(dict(id=color, actor_ids=ids, closed=False,
                maximum_neighbor_distance_m=.15, support_ids=['workplate'],
                minimum_support_fraction=.04,
                route_gates=[dict(center_m=[0., .11, .8], radius_m=.035),
                             dict(center_m=[.10, .11, .8], radius_m=.035)]))
        required = [g['actor_ids'][-1] for g in groups]
    actors = {name: '/World/NistTask/'+name for name in targets}
    objects = []
    for name, point in targets.items():
        initial = list(point)
        if not goal_reset:
            initial[1] -= .3
        objects.append(dict(id=name, body_path=actors[name], mass_kg=.02,
            local_bounds_m=[[-.003]*3, [.003]*3],
            initial_pose_xyzw=initial+[0., 0., 0., 1.]))
    parameters = dict(seating_goals=goals, flexible_groups=groups, required_moved_ids=required)
    if task_id == 'nist_wire_harness':
        parameters['removal'] = dict(actor_ids=list(actors), center_m=[.40, .04, .8],
            half_extents_m=[.16, .16, .05], support_ids=['collection_tray'])
    meta = dict(task_id=task_id, actors=actors, objects=objects, fixtures=fixtures,
        fixture_metadata={name: dict(initial_pose_xyzw=[0., 0., .78, 0., 0., 0., 1.])
                          for name in fixtures}, task_parameters=parameters)
    config = rules.evaluation_configuration(task_id, meta, physics_dt_s=.002,
        room_support_paths={'floor': ['/World/Geometry/floor']})
    task = dict(schema_version='mechanical-nist-task-v1', id=task_id, episode_seed=0,
        evaluation=config, public_rules=rules.public_rules(task_id), asset_metadata=meta)
    frame = dict(baseline=True, time_s=0., native_time_s=0., native_step=0, physics_dt_s=.002,
        objects={row['id']: fixed(row['initial_pose_xyzw']) for row in objects}, contacts=[])
    observer, evaluator = NativeTaskObserver(task), SequenceEvaluator(task)
    observer.on_frame(frame)
    evaluator.update(dict(time_s=0., task_state=observer.on_observe()))
    return task, frame, observer, evaluator, targets


def place(frame, targets):
    for name, target in targets.items():
        frame['objects'][name]['position_m'] = list(target)


def contacts(task, mode='goal', mating=True):
    p, paths = task['asset_metadata']['task_parameters'], task['asset_metadata']['actors']
    rows = []
    fixtures = task['asset_metadata']['fixtures']
    if mode == 'goal':
        rows.extend((paths[g['actor']], fixtures[g['support_ids'][0]]) for g in p['seating_goals'])
        rows.extend((paths[g['actor_ids'][1]], fixtures['workplate']) for g in p['flexible_groups'])
        if mating:
            for group in p['flexible_groups']:
                rows.extend((paths[group['actor_ids'][i]], fixtures.get(target, paths.get(target)))
                            for i, target in enumerate(group.get('required_mating_contact_ids', [])))
    elif mode == 'tray':
        rows.extend((paths[g['actor_ids'][1]], fixtures['collection_tray']) for g in p['flexible_groups'])
    return [dict(actor0=a, actor1=b, impulse_magnitude_sum_ns=.001,
        impulse_vector_ns=[0., 0., .001], min_separation_m=0., time_s=0.) for a, b in rows]


def advance(frame, observer, evaluator=None, duration=.4, rows=None, transform=None, budget=False):
    steps = round(duration/frame['physics_dt_s'])
    result = None
    for _ in range(steps):
        step = frame['native_step']+1
        t = step*frame['physics_dt_s']
        frame.update(baseline=False, native_step=step, time_s=t, native_time_s=t)
        frame['contacts'] = [dict(row, time_s=t) for row in rows or []]
        if transform:
            transform(frame)
        observer.on_frame(frame)
        if evaluator is not None and step % round(.02/frame['physics_dt_s']) == 0:
            result = evaluator.update(dict(time_s=t, task_state=observer.on_observe(),
                                           budget_exhausted=budget))
    return result


@pytest.mark.parametrize('task_id', rules.TASK_IDS)
def test_frozen_contracts_and_canonical_reset_are_unscored(task_id):
    task, frame, observer, evaluator, _ = fixture(task_id)
    summary = observer.on_observe()
    assert summary['complete'], summary['errors']
    assert summary['released_stable_s'] == 0 and summary['events'] == []
    assert evaluator.inspect_conditions()['status'] == 'ongoing'
    assert rules.public_rules(task_id)['source_board'] == rules.TASK_IDS.index(task_id)+1
    task['evaluation']['thresholds']['stable_s'] = 0.
    with pytest.raises(ValueError, match='differ'):
        NativeTaskObserver(task)


@pytest.mark.parametrize('task_id', rules.TASK_IDS)
def test_goal_reset_and_passive_support_never_award_task_credit(task_id):
    task, frame, observer, evaluator, _ = fixture(task_id, goal_reset=True)
    result = advance(frame, observer, evaluator, rows=contacts(task))
    assert result['status'] == 'ongoing'
    assert not observer.on_observe()['detail']['required_movement']
    assert observer.on_observe()['released_stable_s'] == 0.


@pytest.mark.parametrize('task_id', rules.TASK_IDS[:3])
def test_actual_goal_release_and_dwell_are_required(task_id):
    task, frame, observer, evaluator, targets = fixture(task_id)
    place(frame, targets)
    result = advance(frame, observer, evaluator)
    assert result['status'] == 'ongoing'
    assert observer.on_observe()['released_stable_s'] == 0.
    result = advance(frame, observer, evaluator, duration=.2, rows=contacts(task))
    assert result['status'] == 'ongoing'
    result = advance(frame, observer, evaluator, duration=.12, rows=contacts(task))
    assert result['success'] is True
    assert not result['physics_accepted'] and not result['native_manipulation_qualified']


def test_zero_load_manifold_is_contact_but_not_support():
    task, frame, observer, evaluator, targets = fixture()
    place(frame, targets)
    rows = contacts(task)
    for row in rows:
        row.update(impulse_magnitude_sum_ns=0., impulse_vector_ns=[0., 0., 0.])
    result = advance(frame, observer, evaluator, rows=rows)
    assert result['status'] == 'ongoing'
    seats = observer.on_observe()['detail']['seating']
    assert all(s['mating_contact'] and not s['support'] for s in seats.values())


def test_wrong_socket_mapping_can_be_retried_without_terminal_loss():
    task, frame, observer, evaluator, targets = fixture()
    place(frame, dict(zip(targets, reversed(list(targets.values())))))
    result = advance(frame, observer, evaluator, rows=contacts(task))
    assert result['status'] == 'ongoing' and result['referee']['recoverable']
    place(frame, targets)
    result = advance(frame, observer, evaluator, rows=contacts(task))
    assert result['success']


def test_robot_contact_interrupts_the_complete_native_release_dwell():
    task, frame, observer, evaluator, targets = fixture()
    place(frame, targets)
    advance(frame, observer, evaluator, duration=.24, rows=contacts(task))
    robot = dict(actor0=task['asset_metadata']['actors']['usb_plug'],
        actor1='/World/Robot/hand', impulse_magnitude_sum_ns=0.,
        impulse_vector_ns=[0., 0., 0.], min_separation_m=0.)
    advance(frame, observer, duration=.002, rows=contacts(task)+[robot])
    assert observer.on_observe()['released_stable_s'] == 0.
    # Resume on the original 20ms control grid.
    result = advance(frame, observer, evaluator, duration=.198, rows=contacts(task))
    assert result['status'] == 'ongoing'
    result = advance(frame, observer, evaluator, duration=.12, rows=contacts(task))
    assert result['success']


def test_continuous_motion_inside_tolerance_does_not_count_as_stable():
    task, frame, observer, evaluator, targets = fixture()
    place(frame, targets)

    def jitter(sample):
        sample['objects']['usb_plug']['position_m'][0] = .005*math.sin(sample['time_s']*50)

    result = advance(frame, observer, evaluator, rows=contacts(task), transform=jitter, duration=.6)
    assert result['status'] == 'ongoing'
    assert observer.on_observe()['released_stable_s'] < .3


def test_ordered_route_rejects_shortcut_and_reversed_gate_order():
    line = np.array([[0., 0., 0.], [1., 0., 0.], [1., 1., 0.]])
    gates = [dict(center_m=[.8, 0., 0.], radius_m=.1),
             dict(center_m=[1., .8, 0.], radius_m=.1)]
    assert route_in_order(line, gates)[0]
    assert not route_in_order(line, gates[::-1])[0]
    assert not route_in_order(np.array([[0., 0., 0.], [1., 1., 0.]]), gates)[0]


def test_endpoint_seated_but_cable_shortcut_cannot_pass():
    task, frame, observer, evaluator, targets = fixture('nist_cable_routing')
    place(frame, targets)
    frame['objects']['cable_1']['position_m'] = [.033, .06, .8]
    frame['objects']['cable_2']['position_m'] = [.066, .12, .8]
    result = advance(frame, observer, evaluator, rows=contacts(task))
    assert result['status'] == 'ongoing'
    assert not observer.on_observe()['detail']['flexible_groups']['cable']['ordered_route']


def restart_with_metadata(task, frame):
    task['evaluation'] = rules.evaluation_configuration(task['id'], task['asset_metadata'],
        physics_dt_s=.002, room_support_paths={'floor': ['/World/Geometry/floor']})
    observer, evaluator = NativeTaskObserver(task), SequenceEvaluator(task)
    observer.on_frame(frame)
    evaluator.update(dict(time_s=0., task_state=observer.on_observe()))
    return observer, evaluator


def test_cable_seating_uses_world_target_of_declared_local_mating_point():
    from scenes.workrooms.mechanical_assembly.nist_native_evidence import rotation
    task, frame, _, _, targets = fixture('nist_cable_routing')
    goal = task['asset_metadata']['task_parameters']['seating_goals'][0]
    goal['actor_local_point_m'] = [0., 0., .054]
    goal['target_position_m'] = [*targets['cable_3'][:2], targets['cable_3'][2]+.054]
    observer, evaluator = restart_with_metadata(task, frame)
    place(frame, targets)
    tilt = .145
    q = [math.cos(tilt/2), math.sin(tilt/2), 0., 0.]
    frame['objects']['cable_3']['quaternion_wxyz'] = q
    frame['objects']['cable_3']['position_m'] = (
        np.asarray(goal['target_position_m'])-rotation(q) @ np.array(goal['actor_local_point_m'])).tolist()
    # The remote CAD origin moves beyond 6mm while the actual pin point stays
    # at its world socket target. Adding the target offset twice would fail.
    assert abs(frame['objects']['cable_3']['position_m'][1]-targets['cable_3'][1]) > .006
    result = advance(frame, observer, evaluator, rows=contacts(task))
    assert result['success']
    assert max(observer.on_observe()['detail']['seating']['cable_3']['position_error_m']) < 1.e-12


@pytest.mark.parametrize('condition', ['missing_socket', 'touch_without_load', 'supported', 'broken_chain'])
def test_cable_endpoint_support_fallback_requires_mating_intact_chain_and_loaded_support(condition):
    task, frame, _, _, targets = fixture('nist_cable_routing')
    goal = task['asset_metadata']['task_parameters']['seating_goals'][0]
    goal['support_via_flexible_group'] = 'cable'
    observer, evaluator = restart_with_metadata(task, frame)
    place(frame, targets)
    rows = contacts(task)
    rows[0].update(impulse_magnitude_sum_ns=0., impulse_vector_ns=[0., 0., 0.])
    if condition == 'missing_socket':
        rows.pop(0)
    elif condition == 'touch_without_load':
        rows[1].update(impulse_magnitude_sum_ns=0., impulse_vector_ns=[0., 0., 0.])
    elif condition == 'broken_chain':
        frame['objects']['cable_0']['position_m'][0] = -.3
    result = advance(frame, observer, evaluator, rows=rows)
    seat = observer.on_observe()['detail']['seating']['cable_3']
    assert not seat['direct_support']
    assert result['success'] is (condition == 'supported')
    if condition == 'supported':
        assert seat['mating_contact'] and seat['support_via_flexible_group']
    else:
        assert result['status'] == 'ongoing' and result['referee']['recoverable']


@pytest.mark.parametrize('invalid', ['unknown', 'closed', 'not_endpoint'])
def test_declared_flexible_seat_support_is_restricted_to_its_own_open_cable_endpoint(invalid):
    task, frame, _, _, _ = fixture('nist_cable_routing')
    p = task['asset_metadata']['task_parameters']
    goal = p['seating_goals'][0]
    goal['support_via_flexible_group'] = 'missing' if invalid == 'unknown' else 'cable'
    if invalid == 'closed':
        p['flexible_groups'][0]['closed'] = True
    elif invalid == 'not_endpoint':
        goal['actor'] = 'cable_1'
    with pytest.raises(ValueError, match='endpoint'):
        restart_with_metadata(task, frame)


def test_belt_must_enclose_both_pulleys_with_reasonable_length_and_intact_loop():
    task, frame, observer, evaluator, targets = fixture('nist_belt_drive_assembly')
    place(frame, targets)
    frame['objects']['belt_2']['position_m'] = [.10, -.025, .8]
    frame['objects']['belt_3']['position_m'] = [-.10, -.025, .8]
    result = advance(frame, observer, evaluator, rows=contacts(task))
    assert result['status'] == 'ongoing'
    assert not observer.on_observe()['detail']['flexible_groups']['belt']['enclosed_targets']
    place(frame, targets)
    assert advance(frame, observer, evaluator, rows=contacts(task))['success']
    assert not simple_loop(np.array([[0., 0., 0.], [1., 1., 0.], [0., 1., 0.], [1., 0., 0.]]))
    assert not enclosed(np.array([[0., 0., 0.], [1., 0., 0.], [1., 1., 0.], [0., 1., 0.]]),
                        dict(center_m=[.5, .5, .05], radius_m=.1), .02)


def test_belt_plate_support_and_one_wheel_contact_cannot_replace_both_mates():
    task, frame, observer, evaluator, targets = fixture('nist_belt_drive_assembly')
    place(frame, targets)
    rows = contacts(task, mating=False)
    result = advance(frame, observer, evaluator, rows=rows)
    detail = observer.on_observe()['detail']['flexible_groups']['belt']
    assert detail['support'] and detail['enclosed_targets'] and detail['intact']
    assert detail['required_mating_contacts'] == {'drive_pulley': False, 'timing_pulley': False}
    assert result['status'] == 'ongoing' and result['referee']['recoverable']
    first_mate = contacts(task)[-2]
    # Native zero-load contact is enough for pairing, not support or tension.
    first_mate.update(impulse_magnitude_sum_ns=0., impulse_vector_ns=[0., 0., 0.])
    result = advance(frame, observer, evaluator, rows=rows+[first_mate])
    assert result['status'] == 'ongoing'
    assert observer.on_observe()['detail']['flexible_groups']['belt']['required_mating_contacts'] == {
        'drive_pulley': True, 'timing_pulley': False}
    second_mate = contacts(task)[-1]
    second_mate.update(impulse_magnitude_sum_ns=0., impulse_vector_ns=[0., 0., 0.])
    assert advance(frame, observer, evaluator, rows=rows+[first_mate, second_mate])['success']


def test_belt_contact_does_not_count_distant_speculative_pair_and_can_sleep():
    task, frame, observer, evaluator, targets = fixture('nist_belt_drive_assembly')
    place(frame, targets)
    rows = contacts(task)
    rows[-1]['min_separation_m'] = .001
    assert advance(frame, observer, evaluator, rows=rows)['status'] == 'ongoing'
    rows[-1]['min_separation_m'] = .0003
    advance(frame, observer, evaluator, duration=.04, rows=rows)
    # Sleeping native pairs may disappear; unchanged paired poses preserve the
    # actual earlier contact evidence without requiring a continuous impulse.
    assert advance(frame, observer, evaluator, duration=.28, rows=contacts(task, mating=False))['success']


def test_plate_support_cannot_substitute_for_declared_socket_mating():
    task, frame, _, _, targets = fixture()
    for goal in task['asset_metadata']['task_parameters']['seating_goals']:
        goal['support_ids'] = ['workplate']
    task['evaluation'] = rules.evaluation_configuration(task['id'], task['asset_metadata'],
        physics_dt_s=.002, room_support_paths={'floor': ['/World/Geometry/floor']})
    observer, evaluator = NativeTaskObserver(task), SequenceEvaluator(task)
    observer.on_frame(frame)
    evaluator.update(dict(time_s=0., task_state=observer.on_observe()))
    place(frame, targets)
    rows = contacts(task)
    result = advance(frame, observer, evaluator, rows=rows)
    assert result['status'] == 'ongoing'
    assert all(row['geometry'] and row['support'] and not row['mating_contact']
               for row in observer.on_observe()['detail']['seating'].values())
    for goal in task['asset_metadata']['task_parameters']['seating_goals']:
        rows.append(dict(actor0=task['asset_metadata']['actors'][goal['actor']],
            actor1=task['asset_metadata']['fixtures'][goal['contact_ids'][0]],
            impulse_magnitude_sum_ns=0., impulse_vector_ns=[0., 0., 0.],
            min_separation_m=0.))
    assert advance(frame, observer, evaluator, rows=rows)['success']


def test_harness_requires_assembly_before_removal_and_support_for_each_lead():
    task, frame, observer, evaluator, targets = fixture('nist_wire_harness')
    removed = {name: [p[0]+.35, p[1]-.06, p[2]] for name, p in targets.items()}
    place(frame, removed)
    result = advance(frame, observer, evaluator, rows=contacts(task, 'tray'))
    assert result['status'] == 'ongoing' and not observer.on_observe()['events']
    place(frame, targets)
    result = advance(frame, observer, evaluator, rows=contacts(task))
    summary = observer.on_observe()
    assert result['status'] == 'ongoing' and summary['detail']['assembly_confirmed']
    assert len(summary['events']) == 1 and summary['phase'] == 'removing'
    place(frame, removed)
    result = advance(frame, observer, evaluator, rows=contacts(task, 'tray')[:1])
    assert result['status'] == 'ongoing'
    assert not observer.on_observe()['detail']['removal']['tray_support']['blue']
    result = advance(frame, observer, evaluator, rows=contacts(task, 'tray'))
    assert result['success']


@pytest.mark.parametrize('fault', ['step', 'clock', 'missing_body', 'stale_contact'])
def test_missing_or_corrupt_native_samples_are_unknown_not_physical_failure(fault):
    task, frame, observer, evaluator, _ = fixture()
    frame.update(baseline=False, native_step=1, time_s=.002, native_time_s=.002)
    if fault == 'step':
        frame['native_step'] = 2
    elif fault == 'clock':
        frame['native_time_s'] = .005
    elif fault == 'missing_body':
        frame['objects'].pop('usb_plug')
    else:
        frame['contacts'] = contacts(task)
    observer.on_frame(frame)
    result = evaluator.update(dict(time_s=.002, task_state=observer.on_observe()))
    assert result['status'] == 'unknown' and not result['physical_failure']
    assert result['referee']['terminal']


def test_expired_budget_is_terminal_incomplete_even_if_a_later_goal_arrives():
    from bench.protocol.task_referee import project_referee_state
    task, frame, observer, evaluator, targets = fixture()
    result = advance(frame, observer, evaluator, duration=.02, budget=True)
    assert result['status'] == 'incomplete' and result['terminal']
    assert result['reason'] == 'budget_exhausted'
    assert project_referee_state(result, task['public_rules']['referee'])['state'] == 'ongoing'
    place(frame, targets)
    assert advance(frame, observer, evaluator, rows=contacts(task)) == result
    assert not result['physical_failure']


def test_floor_contact_and_physical_safety_have_distinct_latched_outcomes():
    for kind, expected in [('floor', 'task_failure'), ('penetration', 'safety_stop')]:
        task, frame, observer, evaluator, _ = fixture()
        row = contacts(task)[0]
        if kind == 'floor':
            row['actor1'] = '/World/Room/Geometry/floor'
        else:
            row['min_separation_m'] = -.01
        result = advance(frame, observer, evaluator, duration=.02, rows=[row])
        assert result['status'] == expected
        assert result['physical_failure'] == (kind == 'floor')
        assert advance(frame, observer, evaluator, duration=.02) == result


def test_loaded_support_witness_survives_sleep_but_not_an_airborne_relocation():
    task, frame, observer, evaluator, targets = fixture()
    place(frame, targets)
    advance(frame, observer, evaluator, duration=.04, rows=contacts(task))
    result = advance(frame, observer, evaluator, duration=.32)
    assert result['success']
    # Separate evaluator to inspect loss before terminal success.
    task, frame, observer, evaluator, targets = fixture()
    place(frame, targets)
    advance(frame, observer, evaluator, duration=.04, rows=contacts(task))
    frame['objects']['usb_plug']['position_m'][2] += .007
    result = advance(frame, observer, evaluator, duration=.4)
    assert result['status'] == 'ongoing'
    assert not observer.on_observe()['detail']['seating']['usb_plug']['support']


def test_frozen_evaluator_cannot_accept_forged_growing_dwell():
    task, _, observer, evaluator, _ = fixture()
    receipt = deepcopy(observer.on_observe())
    receipt.update(time_s=.02, native_time_s=.02, native_step=10, baseline=False, released_stable_s=.3)
    result = evaluator.update(dict(time_s=.02, task_state=receipt))
    assert result['status'] == 'unknown'


def test_signed_contact_support_direction_is_not_reversed():
    row = dict(actor0='/World/A', actor1='/World/B', time_s=0.,
        impulse_vector_ns=[0., 0., -.001], impulse_magnitude_sum_ns=.001, min_separation_m=0.)
    evidence = Contacts([row], {'a': '/World/A', 'b': '/World/B'}, {'a', 'b'}, 0., 1.e-10)
    assert evidence.upward == {('b', 'a')}
