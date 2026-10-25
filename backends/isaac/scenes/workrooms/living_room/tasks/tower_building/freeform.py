"""Stable-height scoring over complete trusted physical observations.

The score is retained across collapses. There is no target arrangement, block
count or binary success height. Native authentication remains the caller's job.
"""
from __future__ import annotations

from copy import deepcopy

from .evaluator import EPS, angle, boxes_penetrate, distance
from .task import _bounds

VERSION = 'living_tower_building_freeform_height_evaluator_v1'


def _result(evaluator, reason, *, status='incomplete', terminal=False,
            current_height=0., stable_s=0., blocks=None, components=None):
    evaluator.terminal = terminal
    evaluator.last_result = {
        'evaluator': VERSION, 'task_id': 'tower_building', 'scoring_mode': 'freeform_height',
        'status': status, 'reason': reason, 'terminal': terminal,
        'success': False, 'reference_success': False, 'native_qualified': False,
        'score': evaluator.best_stable_height_m, 'score_unit': 'm', 'score_valid': status != 'invalid',
        'current_height_m': current_height, 'current_height_cm': current_height*100.,
        'best_stable_height_m': evaluator.best_stable_height_m,
        'best_stable_height_cm': evaluator.best_stable_height_m*100.,
        'best_height_m': evaluator.best_stable_height_m,
        'best_height_cm': evaluator.best_stable_height_m*100.,
        'best_stable_height_time_s': evaluator.best_stable_height_time_s,
        'best_stable_block_ids': list(evaluator.best_stable_block_ids),
        'stable_duration_s': stable_s,
        'required_stable_duration_s': evaluator.rules['goal']['stable_duration_s'],
        'inventory_block_count': len(evaluator.ids),
        'blocks': deepcopy(blocks or {}), 'assemblies': deepcopy(components or []),
        'evidence_scope': 'Conditional reference score over complete trusted contact and motion intervals; separate native and manipulation evidence required',
    }
    return deepcopy(evaluator.last_result)


def _support_edge(evaluator, name, row, states, bounds):
    """Use upward physical support receipts without imposing a layer or yaw."""
    t = evaluator.t
    if (not row['continuous'] or row['force'] <= t['minimum_support_force_n']
            or not -t['penetration_m']-EPS <= row['separation'] <= t['support_gap_m']+EPS
            or not evaluator._point_inside(states[name], row['point'])):
        return False
    other, point = row['other'], row['point']
    if other == 'table':
        table, top = evaluator.g['table_bounds_xy_m'], evaluator.g['table_top_z_m']
        return (abs(point[2]-top) <= t['contact_point_m']+EPS
                and abs(bounds[name][0][2]-top) <= t['support_gap_m']+EPS
                and all(table[0][axis] <= point[axis] <= table[1][axis] for axis in (0, 1)))
    return other in states and evaluator._point_inside(states[other], point)


def _assemblies(evaluator, states, contacts):
    size, table = evaluator.g['block_size_m'], evaluator.g['table_bounds_xy_m']
    bounds = {name: _bounds(state, size) for name, state in states.items()}
    blocks = {name: {'reasons': [], 'supported_to_table': False} for name in evaluator.ids}
    neighbours = {name: set() for name in evaluator.ids}
    supports = {name: set() for name in evaluator.ids}
    for name in evaluator.ids:
        lo, hi = bounds[name]
        if (lo[2] < evaluator.g['table_top_z_m']-evaluator.t['penetration_m']-EPS
                or any(lo[axis] < table[0][axis]-EPS or hi[axis] > table[1][axis]+EPS for axis in (0, 1))):
            blocks[name]['reasons'].append('outside_table_scoring_volume')
        for row in contacts[name]:
            other = row['other']
            if other in neighbours:
                neighbours[name].add(other)
                neighbours[other].add(name)
            elif other != 'table':
                blocks[name]['reasons'].append('tray_robot_floor_or_external_contact')
            if row['separation'] < -evaluator.t['penetration_m']-EPS:
                blocks[name]['reasons'].append('excessive_contact_penetration')
            if _support_edge(evaluator, name, row, states, bounds):
                supports[name].add(other)
    # Unreported intersections still invalidate the affected physical assembly.
    for index, name in enumerate(evaluator.ids):
        for other in evaluator.ids[index+1:]:
            first, second = bounds[name], bounds[other]
            possible_overlap = all(min(first[1][axis], second[1][axis])-max(first[0][axis], second[0][axis])
                                   > evaluator.t['penetration_m']+EPS for axis in range(3))
            if possible_overlap and boxes_penetrate(states[name], states[other], size, evaluator.t['penetration_m']):
                for actor in (name, other):
                    blocks[actor]['reasons'].append('geometric_block_penetration')
                neighbours[name].add(other)
                neighbours[other].add(name)
    rooted = {name for name, sources in supports.items() if 'table' in sources}
    while True:
        enlarged = rooted | {name for name, sources in supports.items() if sources & rooted}
        if enlarged == rooted:
            break
        rooted = enlarged
    for name in evaluator.ids:
        blocks[name]['supported_to_table'] = name in rooted
        if name not in rooted:
            blocks[name]['reasons'].append('continuous_contact_support_chain_required')
        blocks[name]['reasons'] = sorted(set(blocks[name]['reasons']))
    components, unseen = [], set(evaluator.ids)
    while unseen:
        pending, group = [min(unseen)], set()
        while pending:
            name = pending.pop()
            if name in group:
                continue
            group.add(name)
            pending.extend(neighbours[name]-group)
        unseen -= group
        components.append({
            'block_ids': sorted(group),
            'height_m': max(bounds[name][1][2] for name in group)-evaluator.g['table_top_z_m'],
            'eligible': all(not blocks[name]['reasons'] for name in group),
        })
    return blocks, components


def update_height(evaluator, observation):
    if evaluator.terminal:
        return deepcopy(evaluator.last_result)
    try:
        time, index, actions, episode = evaluator._sampling(observation)
        if not isinstance(observation['objects'], dict) or set(observation['objects']) != set(evaluator.ids):
            raise ValueError('Every physical block must be observed exactly once with no extras')
        states = {name: evaluator._state(observation['objects'][name]) for name in evaluator.ids}
        contacts = evaluator._contacts(observation)
        previous = evaluator.previous
        first = previous is None
        dt = 0. if first else time-evaluator.time
        for name, state in states.items():
            if first:
                initial = evaluator.task['initial_objects'][name]
                if (distance(state['position_m'], initial['position_m']) > evaluator.t['initial_position_m']
                        or angle(state['quaternion_wxyz'], initial['quaternion_wxyz']) > evaluator.t['initial_angle_rad']):
                    raise ValueError('Reset does not match the frozen physical tray supply')
            elif (distance(state['position_m'], previous[name]['position_m'])/dt > state['vmax']+1.e-6
                    or angle(state['quaternion_wxyz'], previous[name]['quaternion_wxyz'])/dt > state['wmax']+1.e-6):
                raise ValueError('Interval speed bound contradicts observed motion')
        evaluator.previous, evaluator.time, evaluator.index = states, time, index
        evaluator.action_count, evaluator.episode = actions, episode
        budget = evaluator.task['budget']
        exhausted = (observation['budget_exhausted']
            or (budget['max_sim_time_s'] is not None and time >= budget['max_sim_time_s'])
            or (budget['max_robot_actions'] is not None and actions >= budget['max_robot_actions']))
        if first:
            return _result(evaluator, 'budget_exhausted' if exhausted else 'reset_baseline_not_scored', terminal=exhausted)
        blocks, components = _assemblies(evaluator, states, contacts)
        windows, current_height, longest = {}, 0., 0.
        for component in components:
            ids = tuple(component['block_ids'])
            component['stable_duration_s'] = 0.
            if not component['eligible']:
                continue
            current_height = max(current_height, component['height_m'])
            if any(states[name]['vmax'] > evaluator.t['linear_speed_m_s']
                    or states[name]['wmax'] > evaluator.t['angular_speed_rad_s'] for name in ids):
                component['eligible'] = False
                component['reason'] = 'still_moving'
                continue
            window = evaluator.height_windows.get(ids)
            drift = window is not None and any(
                max(distance(frame[name]['position_m'], window['anchor'][name]['position_m']) for frame in (previous, states))
                + states[name]['vmax']*dt/2 > evaluator.t['stable_position_range_m']+EPS
                or max(angle(frame[name]['quaternion_wxyz'], window['anchor'][name]['quaternion_wxyz']) for frame in (previous, states))
                + states[name]['wmax']*dt/2 > evaluator.t['stable_angle_range_rad']+EPS
                for name in ids)
            if window is None or drift:
                window = {'start_s': time, 'anchor': {name: deepcopy(states[name]) for name in ids}}
            windows[ids] = window
            dwell = time-window['start_s']
            component['stable_duration_s'] = dwell
            longest = max(longest, dwell)
            if (dwell+EPS >= evaluator.rules['goal']['stable_duration_s']
                    and component['height_m'] > evaluator.best_stable_height_m+EPS):
                evaluator.best_stable_height_m = component['height_m']
                evaluator.best_stable_height_time_s = time
                evaluator.best_stable_block_ids = list(ids)
        evaluator.height_windows = windows
        return _result(evaluator, 'budget_exhausted' if exhausted else 'height_score_active',
            terminal=exhausted, current_height=current_height, stable_s=longest, blocks=blocks, components=components)
    except (KeyError, TypeError, ValueError, IndexError, OverflowError, AttributeError) as exc:
        evaluator.height_windows = {}
        return _result(evaluator, str(exc), status='invalid', terminal=True)
