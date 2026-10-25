"""Synthetic CPU contracts only: no native bodies, controller or dimensions chosen."""
from copy import deepcopy
import math

import pytest

from scenes.workrooms.living_room.tasks.tower_extraction.evaluator import TowerExtractionGame
from scenes.workrooms.living_room.tasks.tower_extraction.rules import OBSERVATION_SCHEMA, public_rules, starting_side
from scenes.workrooms.living_room.tasks.tower_extraction.task import task_configuration

# Deliberately synthetic test dimensions, never a task asset/release selection.
GEOMETRY = {'block_size_m': [.12, .035, .02]}


class Stream:
    """Numerical fixtures, with explicit contact construction rather than flags."""
    def __init__(self, *, seed=0, turns=2, trusted=True, geometry=None, collection_trays=None):
        self.config = task_configuration(geometry or GEOMETRY, seed=seed, max_turns_per_player=turns,
                                         collection_trays=collection_trays)
        self.game = TowerExtractionGame(self.config, trusted_normalized_input=trusted)
        self.tick = -1
        self.time = -.02
        self.blocks = {bid: {'center_h': list(b['center_h']),
                            'quaternion_wxyz': list(b['quaternion_wxyz']),
                            'linear_velocity_h_s': [0., 0., 0.],
                            'angular_velocity_rad_s': [0., 0., 0.],
                            'interval_max_linear_speed_h_s': 0.,
                            'interval_max_angular_speed_rad_s': 0.,
                            'interval_min_center_z_h': b['center_h'][2],
                            'interval_max_tilt_rad': 0.,
                            'hand_distance_h': {'front': 10., 'rear': 10.}}
                       for bid, b in self.config['blocks'].items()}
        self.removed = set()
        self.collection_sides = {}
        self.hand_contacts = []
        self.executed = []
        self.unsupported = set()
        self.extra_contacts = []
        self.last_blocks = None

    @staticmethod
    def contact(a, b, normal=1., impulse=.1):
        return {'a': a, 'b': b, 'normal_z_on_a': normal, 'normal_impulse_n_s': impulse,
                'point_h': [0., 0., 0.], 'separation_h': 0., 'supported_interval_s': 0.,
                'minimum_normal_force_n': 0.}

    def sample(self):
        self.tick += 1
        self.time = round(self.tick * .02, 10)
        for bid, block in self.blocks.items():
            block['interval_min_center_z_h'] = block['center_h'][2]
            w, x, y, z = block['quaternion_wxyz']
            block['interval_max_tilt_rad'] = math.acos(max(-1., min(1., 1 - 2 * (x*x + y*y))))
            block['interval_max_linear_speed_h_s'] = math.sqrt(sum(v*v for v in block['linear_velocity_h_s']))
            block['interval_max_angular_speed_rad_s'] = math.sqrt(sum(v*v for v in block['angular_velocity_rad_s']))
            if self.last_blocks is not None:
                previous = self.last_blocks[bid]
                speed = math.sqrt(sum((a-b)**2 for a, b in zip(block['center_h'], previous['center_h']))) / .02
                dot = abs(sum(a*b for a, b in zip(block['quaternion_wxyz'], previous['quaternion_wxyz'])))
                angular = 2 * math.acos(min(1., dot)) / .02
                block['interval_max_linear_speed_h_s'] = max(block['interval_max_linear_speed_h_s'], speed)
                block['interval_max_angular_speed_rad_s'] = max(block['interval_max_angular_speed_rad_s'], angular)
                block['interval_min_center_z_h'] = min(block['center_h'][2], previous['center_h'][2])
                pq = previous['quaternion_wxyz']
                previous_tilt = math.acos(max(-1., min(1., 1 - 2 * (pq[1]**2 + pq[2]**2))))
                block['interval_max_tilt_rad'] = max(block['interval_max_tilt_rad'], previous_tilt)
        contacts = []
        for bid, initial in self.config['blocks'].items():
            if bid in self.unsupported:
                continue
            layer = initial['layer']
            if bid in self.removed or layer == 0:
                support = 'table'
                if bid in self.removed and self.config.get('collection_trays'):
                    support = self.config['collection_trays'][self.collection_sides[bid]]['actor_id']
            else:
                support = next((candidate for candidate, b in self.config['blocks'].items()
                                if b['layer'] == layer - 1 and candidate not in self.removed), None)
            if support is not None:
                c = self.contact(bid, support)
                p = self.blocks[bid]['center_h']
                if support not in self.blocks:
                    c['point_h'] = [*p[:2], p[2] - .5]
                else:
                    other = self.blocks[support]['center_h']
                    # Adjacent layers have perpendicular strips; their centre-
                    # line intersection lies on both physical contact faces.
                    x, y = (other[0], p[1]) if initial['layer'] % 2 == 0 else (p[0], other[1])
                    c['point_h'] = [x, y, p[2] - .5]
                c['supported_interval_s'] = .02 if self.tick else 0.
                c['minimum_normal_force_n'] = 1.
                contacts.append(c)
        # Zero load is still native contact: selection must lock at first touch.
        contacts += [self.contact(bid, 'hand/' + side, 0., 0.) for side, bid in self.hand_contacts]
        contacts += deepcopy(self.extra_contacts)
        self.last_blocks = deepcopy(self.blocks)
        return {'schema': OBSERVATION_SCHEMA, 'stream_id': 'fixture', 'episode_id': 'episode',
                'configuration_sha256': self.config['configuration_sha256'],
                'reset_id': 'reset', 'tick': self.tick, 'time_s': self.time,
                'blocks': deepcopy(self.blocks), 'contacts': contacts, 'executed_sides': list(self.executed)}

    def step(self, n=1):
        for _ in range(n):
            result = self.game.observe(self.sample())
        return result

    def ready(self):
        self.step(51)
        assert self.game.status == 'await_intent'
        return self

    def touch(self, bid='block_03'):
        side = self.game.current_side
        assert self.game.command(side, 'select', bid)['accepted']
        self.hand_contacts = [(side, bid)]
        self.step()
        self.hand_contacts = []
        assert self.game.selection_locked

    def place(self, bid=None, *, side=None, slot_index=None):
        bid = bid or self.game.selected
        side = side or self.game.current_side
        if slot_index is None:
            slot_index = self.game.turns[side]
        slot = self.config['collection_slots'][side][slot_index]
        g = self.config['geometry']; h = g['block_size_m'][2]
        p = slot['position_m']
        self.blocks[bid]['center_h'] = [(p[0]-g['center_xy_m'][0])/h,
                                       (p[1]-g['center_xy_m'][1])/h,
                                       (p[2]-g['table_top_z_m'])/h if self.config.get('collection_trays') else .5]
        yaw = slot['yaw_rad']
        self.blocks[bid]['quaternion_wxyz'] = [math.cos(yaw/2), 0., 0., math.sin(yaw/2)]
        self.removed.add(bid)
        self.collection_sides[bid] = side

    def extract(self, bid):
        self.touch(bid)
        self.place(bid)
        return self.step(52)  # Motion tick, then a full 1 s supported dwell.


def test_factory_requires_dimensions_and_preserves_planned_boundary():
    with pytest.raises(ValueError):
        task_configuration({})
    c = task_configuration(GEOMETRY)
    assert c['rules']['max_turns_per_player'] == 6
    assert len(c['collection_slots']['front']) == len(c['collection_slots']['rear']) == 6
    assert c['status'] == 'planned' and c['native_adapter'] is None
    assert c['source_release'] is None and c['native_execution_available'] is False
    with pytest.raises(ValueError, match='capacity'):
        task_configuration({'block_size_m': [.8, .25, .1]})
    with pytest.raises(ValueError, match='differs'):
        TowerExtractionGame({**c, 'native_execution_available': True})


def test_seed_and_equal_limit_validation():
    assert [starting_side(i) for i in range(4)] == ['front', 'rear', 'front', 'rear']
    for seed in (-1, True, 1.2):
        with pytest.raises(ValueError):
            task_configuration(GEOMETRY, seed=seed)
    for kwargs in ({'max_turns_per_player': 7}, {'max_turns_per_player': True},
                   {'protected_bottom_layers': 0}, {'thresholds': {'stable_duration_s': .9}},
                   {'thresholds': {'max_observation_gap_s': .1}}, {'thresholds': {'unknown': 1}}):
        with pytest.raises(ValueError):
            public_rules(GEOMETRY, **kwargs)


def test_collection_capacity_is_mirrored_nonoverlapping_and_within_table():
    c = task_configuration(GEOMETRY)
    front, rear = c['collection_slots']['front'], c['collection_slots']['rear']
    for a, b in zip(front, rear):
        assert a['position_m'][0] + b['position_m'][0] == pytest.approx(1.3)
        assert a['position_m'][1] == pytest.approx(-b['position_m'][1])
    slots = front + rear
    for i, a in enumerate(slots):
        for b in slots[i+1:]:
            assert any(a['bounds_h'][1][j] < b['bounds_h'][0][j] or b['bounds_h'][1][j] < a['bounds_h'][0][j]
                       for j in (0, 1))
    for slot in slots:
        p = slot['position_m']
        assert .15 < p[0] - .06 < p[0] + .06 < 1.15
        assert -.5 < p[1] - .0175 < p[1] + .0175 < .5


def test_reset_needs_full_second_support_and_default_adapter_is_unavailable():
    s = Stream()
    s.step(50)
    assert s.game.status == 'reset_pending'
    assert not s.game.command('front', 'select', 'block_03')['accepted']
    s.step()
    assert s.game.status == 'await_intent' and s.game.turns == {'front': 0, 'rear': 0}
    s = Stream(trusted=False)
    assert s.step()['reason'] == 'trusted_adapter_unavailable'
    assert s.game.winner is None
    s = Stream()
    s.unsupported = {'block_17'}
    assert s.step(501)['reason'] == 'invalid_reset_not_stable'


def test_invalid_reset_never_awards_winner():
    for change in ('collapse', 'hand', 'execute'):
        s = Stream()
        if change == 'collapse':
            s.blocks['block_09']['center_h'][2] -= 1.
        elif change == 'hand':
            s.hand_contacts = [('front', 'block_03')]
        else:
            s.executed = ['front']
        result = s.step()
        assert result['status'] == 'abort' and result['winner'] is None


def test_protected_layers_and_rejected_commands_do_not_take_turn():
    s = Stream().ready()
    assert s.game.legal_blocks() == [f'block_{n:02d}' for n in range(3, 15)]
    for bid in ('block_00', 'block_15'):
        assert not s.game.command('front', 'select', bid)['accepted']
    assert not s.game.command('rear', 'select', 'block_03')['accepted']
    assert not s.game.command('front', 'select')['accepted']
    assert not s.game.command('front', 'push', 'block_03')['accepted']
    assert s.game.current_side == 'front' and s.game.turns == {'front': 0, 'rear': 0}
    # A partial original top layer exposes the next highest complete layer.
    s.blocks['block_15']['center_h'][0] += .5
    s.step()
    assert s.game.legal_blocks() == [f'block_{n:02d}' for n in range(3, 12)]


def test_selection_can_change_before_touch_and_locks_on_zero_impulse_contact():
    s = Stream().ready()
    assert s.game.command('front', 'select', 'block_03')['accepted']
    assert s.game.command('front', 'select', 'block_04')['accepted']
    assert s.game.command('front', 'cancel')['accepted']
    s.touch('block_03')
    for action in ('push', 'pull', 'regrasp', 'release', 'wait'):
        assert s.game.command('front', action, 'block_03')['accepted']
    assert not s.game.command('front', 'select', 'block_04')['accepted']
    assert not s.game.command('front', 'cancel')['accepted']
    assert not s.game.command('front', 'pull', 'block_04')['accepted']


def test_only_supported_own_collection_and_release_dwell_can_handoff():
    s = Stream().ready()
    s.touch()
    s.place(side='rear')
    assert s.step(70)['completed_turns']['front'] == 0
    s.place(side='front')
    s.unsupported = {'block_03'}
    assert s.step(70)['completed_turns']['front'] == 0
    s.unsupported = set()
    s.blocks['block_03']['hand_distance_h']['front'] = .05
    assert s.step(70)['completed_turns']['front'] == 0
    s.blocks['block_03']['hand_distance_h']['front'] = 10.
    s.step(50)
    assert s.game.turns['front'] == 0
    s.step()
    assert s.game.turns['front'] == 1 and s.game.current_side == 'rear'


def test_intent_without_physical_contact_cannot_score():
    s = Stream().ready()
    s.game.command('front', 'select', 'block_03')
    s.place()
    s.step(70)
    assert not s.game.selection_locked and s.game.turns['front'] == 0


def test_pose_changes_and_reported_velocity_reset_dwell():
    s = Stream().ready()
    s.touch()
    s.place()
    s.step(30)
    s.blocks['block_06']['center_h'][0] += .01  # Reported velocity falsely remains zero.
    s.step()
    assert s.game.status == 'active'
    s.step(50)
    assert s.game.turns['front'] == 0
    s.step()
    assert s.game.turns['front'] == 1


@pytest.mark.parametrize('collapse', ['drop', 'tilt'])
def test_pre_handoff_collapse_after_contact_is_current_players_loss(collapse):
    s = Stream().ready()
    s.touch()
    s.place()
    s.step(40)
    if collapse == 'drop':
        s.blocks['block_09']['center_h'][2] -= 1.
    else:
        s.blocks['block_09']['quaternion_wxyz'] = [math.cos(.5), math.sin(.5), 0., 0.]
    result = s.step()
    assert result['status'] == 'won' and result['loser'] == 'front' and result['winner'] == 'rear'
    assert result['completed_turns'] == {'front': 0, 'rear': 0}


def test_spontaneous_collapse_after_handoff_has_no_causal_loser_even_after_hold():
    s = Stream().ready()
    assert s.extract('block_03')['current_side'] == 'rear'
    s.game.command('rear', 'wait')
    s.executed = ['rear']  # Native execution alone does not establish tower contact.
    s.step()
    s.executed = []
    s.blocks['block_09']['center_h'][2] -= 1.
    result = s.step()
    assert result['status'] == 'abort' and result['reason'] == 'abort_no_causal_evidence'
    assert result['winner'] is None and result['loser'] is None


@pytest.mark.parametrize('interference', ['hand', 'robot', 'execution'])
def test_noncurrent_physical_interference_aborts_even_on_collapse_tick(interference):
    s = Stream().ready()
    s.touch()
    if interference == 'hand':
        s.hand_contacts = [('rear', 'block_09')]
    elif interference == 'robot':
        s.extra_contacts = [s.contact('block_09', 'robot/rear', 0., 0.)]
    else:
        s.executed = ['rear']
    s.blocks['block_09']['center_h'][2] -= 1.
    result = s.step()
    assert result['reason'] == 'noncurrent_physical_interference' and result['winner'] is None


def test_undeclared_direct_block_contact_aborts():
    s = Stream().ready()
    s.game.command('front', 'select', 'block_03')
    s.hand_contacts = [('front', 'block_04')]
    assert s.step()['reason'] == 'undeclared_block_contact'


@pytest.mark.parametrize('damage', ['missing_block', 'missing_tick', 'duplicate_tick', 'time_gap', 'nan',
                                  'bad_quaternion', 'boolean_claim', 'reset', 'unknown_contact', 'boolean_time', 'hash'])
def test_corrupt_or_incomplete_evidence_aborts_without_a_winner(damage):
    s = Stream().ready()
    obs = s.sample()
    if damage == 'missing_block':
        obs['blocks'].pop('block_17')
    elif damage == 'missing_tick':
        obs['tick'] += 1
    elif damage == 'duplicate_tick':
        obs['tick'] -= 1
    elif damage == 'time_gap':
        obs['time_s'] += .5
    elif damage == 'nan':
        obs['blocks']['block_03']['center_h'][0] = float('nan')
    elif damage == 'bad_quaternion':
        obs['blocks']['block_03']['quaternion_wxyz'] = [2., 0., 0., 0.]
    elif damage == 'boolean_claim':
        obs['stable'] = True
    elif damage == 'reset':
        obs['reset_id'] = 'other_reset'
    elif damage == 'unknown_contact':
        obs['contacts'].append(s.contact('block_03', 'unknown_body'))
    elif damage == 'hash':
        obs['configuration_sha256'] = '0' * 64
    else:
        obs['time_s'] = True
    result = s.game.observe(obs)
    assert result['status'] == 'abort' and result['winner'] is None and result['loser'] is None
    assert s.game.observe(s.sample()) == result  # Terminal state is latched.


def test_budget_is_incomplete_even_with_unequal_score_and_terminal_is_latched():
    s = Stream().ready()
    s.extract('block_03')
    result = s.game.finish('budget')
    assert result['status'] == 'incomplete' and result['winner'] is None
    assert result['completed_turns'] == {'front': 1, 'rear': 0}
    assert s.game.finish('reset') == result


def test_balanced_turn_limit_draw_requires_both_sides():
    s = Stream(turns=1).ready()
    assert s.extract('block_03')['status'] == 'await_intent'
    result = s.extract('block_06')
    assert result['status'] == 'draw' and result['reason'] == 'balanced_turn_limit'
    assert result['completed_turns'] == {'front': 1, 'rear': 1}
    assert result['winner'] is None and result['loser'] is None


def test_exhausted_targets_require_supported_stable_tower_before_draw():
    s = Stream().ready()
    # Synthetic displacements make every layer above layer 1 incomplete;
    # protecting the now-highest complete layer leaves no selectable target.
    for layer in range(2, 6):
        s.blocks[f'block_{layer * 3:02d}']['center_h'][0] += .5
    s.step(50)
    assert s.game.legal_blocks() == [] and s.game.status == 'await_intent'
    s.step(2)
    assert s.game.status == 'draw' and s.game.reason == 'legal_targets_exhausted'


@pytest.mark.parametrize('tamper', ['parent', 'slot', 'block', 'rule', 'hash'])
def test_recomputed_hash_cannot_bypass_canonical_configuration(tamper):
    import hashlib
    import json
    c = task_configuration(GEOMETRY)
    if tamper == 'parent':
        c['parent_environment']['usd_sha256'] = 'f' * 64
    elif tamper == 'slot':
        c['collection_slots']['front'][0]['position_m'][0] += .1
    elif tamper == 'block':
        c['blocks']['block_00']['center_h'][0] += 1.
    elif tamper == 'rule':
        c['rules']['loss'] = 'The other player always loses'
    else:
        c['configuration_sha256'] = 'f' * 64
    if tamper != 'hash':
        c.pop('configuration_sha256')
        c['configuration_sha256'] = hashlib.sha256(json.dumps(c, sort_keys=True,
            separators=(',', ':'), allow_nan=False).encode()).hexdigest()
    with pytest.raises(ValueError, match='differs'):
        TowerExtractionGame(c, trusted_normalized_input=True)


@pytest.mark.parametrize('damage', ['off_face', 'skip_layer', 'penetration', 'coverage', 'force', 'interval_motion'])
def test_false_support_or_hidden_interval_motion_resets_handoff(damage):
    s = Stream().ready()
    s.touch()
    s.place()
    s.step(30)
    obs = s.sample()
    contact = next(c for c in obs['contacts'] if c['a'] == 'block_09')
    if damage == 'off_face':
        contact['point_h'][0] += 100.
    elif damage == 'skip_layer':
        contact['b'] = 'block_00'
    elif damage == 'penetration':
        c = s.contact('block_00', 'block_01', normal=0., impulse=0.)
        c['separation_h'] = -.2
        obs['contacts'].append(c)
    elif damage == 'coverage':
        contact['supported_interval_s'] = .01
    elif damage == 'force':
        contact['minimum_normal_force_n'] = 0.
    else:
        obs['blocks']['block_09']['interval_max_linear_speed_h_s'] = 1.
    result = s.game.observe(obs)
    assert result['status'] == 'active' and result['completed_turns']['front'] == 0
    s.step(50)
    assert s.game.turns['front'] == 0
    s.step()
    assert s.game.turns['front'] == 1


@pytest.mark.parametrize('damage', ['speed', 'minimum_height', 'maximum_tilt', 'coverage', 'force_integral', 'impossible_extrema'])
def test_self_contradictory_interval_summaries_abort(damage):
    s = Stream().ready()
    s.blocks['block_06']['center_h'][0] += .01
    obs = s.sample()
    block = obs['blocks']['block_06']
    if damage == 'speed':
        block['interval_max_linear_speed_h_s'] = 0.
    elif damage == 'minimum_height':
        block['interval_min_center_z_h'] += 1.
    elif damage == 'maximum_tilt':
        block['interval_max_tilt_rad'] = -.1
    elif damage == 'coverage':
        obs['contacts'][0]['supported_interval_s'] = .1
    elif damage == 'impossible_extrema':
        block['interval_min_center_z_h'] -= 1.
    else:
        obs['contacts'][0]['minimum_normal_force_n'] = 100.
    result = s.game.observe(obs)
    assert result['status'] == 'abort' and result['winner'] is None


def test_first_contact_and_collapse_same_interval_have_no_causal_order():
    s = Stream().ready()
    s.game.command('front', 'select', 'block_03')
    s.hand_contacts = [('front', 'block_03')]
    s.blocks['block_09']['center_h'][2] -= 1.
    result = s.step()
    assert result['status'] == 'abort' and result['reason'] == 'abort_no_causal_evidence'


@pytest.mark.parametrize('illegal', ['undeclared', 'became_protected'])
def test_illegal_contact_precedes_same_interval_collapse(illegal):
    s = Stream().ready()
    if illegal == 'undeclared':
        s.game.command('front', 'select', 'block_03')
        s.hand_contacts = [('front', 'block_04')]
    else:
        s.game.command('front', 'select', 'block_12')
        s.blocks['block_15']['center_h'][0] += .5
        s.step()
        s.hand_contacts = [('front', 'block_12')]
    s.blocks['block_09']['center_h'][2] -= 1.
    result = s.step()
    assert result['status'] == 'abort' and result['winner'] is None
    assert result['reason'] == ('undeclared_block_contact' if illegal == 'undeclared' else 'protected_target_contact')


def test_target_that_becomes_protected_before_touch_cannot_be_extracted():
    s = Stream().ready()
    s.game.command('front', 'select', 'block_12')
    s.blocks['block_15']['center_h'][0] += .5
    s.step()
    s.hand_contacts = [('front', 'block_12')]
    assert s.step()['reason'] == 'protected_target_contact'


def test_interval_collapse_then_rebound_is_retained():
    s = Stream().ready()
    s.touch()
    obs = s.sample()
    obs['blocks']['block_09']['interval_min_center_z_h'] -= 1.
    obs['blocks']['block_09']['interval_max_linear_speed_h_s'] = 100.
    result = s.game.observe(obs)
    assert result['status'] == 'won' and result['loser'] == 'front'


def test_small_collection_offset_is_allowed_with_remaining_slot_gap():
    s = Stream().ready()
    s.touch()
    s.place()
    h = s.config['geometry']['block_size_m'][2]
    margin = max(h * .25, s.config['geometry']['in_layer_gap_m'])
    s.blocks['block_03']['center_h'][0] += margin / 8 / h
    s.blocks['block_03']['center_h'][1] += margin / 8 / h
    assert s.step(52)['completed_turns']['front'] == 1
    slots = s.config['collection_slots']['front']
    assert slots[1]['bounds_h'][0][0] - slots[0]['bounds_h'][1][0] == pytest.approx(margin / 2 / h)


TRAY_GEOMETRY = {'block_size_m': [.20, .045, .030], 'layers': 12, 'in_layer_gap_m': .010}


def collection_trays():
    return {side: {'actor_id': f'{side}_collection_tray_floor',
                   'wall_actor_ids': [f'{side}_collection_tray_wall_{wall}'
                                      for wall in ('neg_x', 'pos_x', 'neg_y', 'pos_y')],
                   'interior_bounds_xy_m': [[.29, low_y], [1.01, high_y]],
                   'floor_z_m': .786, 'slot_pitch_xy_m': [.24, .06]}
            for side, low_y, high_y in (('front', -.48, -.17), ('rear', .17, .48))}


def tray_stream(**kwargs):
    return Stream(geometry=TRAY_GEOMETRY, collection_trays=collection_trays(), **kwargs)


def test_legacy_no_tray_configuration_hashes_are_unchanged():
    # Hashed in portable form, so the values do not depend on where the data package lives.
    assert task_configuration(GEOMETRY)['configuration_sha256'] == '388c92e05c95d3c54c42fd2891b1fd4e861636652895189e061fc5adc69365da'
    assert task_configuration({'block_size_m': [.15, .045, .025]})['configuration_sha256'] == '8c22edacf9093a9e753143e8666eb431dfd58fb7b9f9fbb61fc77acfe902485a'


def test_tall_tower_collection_slots_are_inside_symmetric_physical_trays():
    trays = collection_trays()
    config = task_configuration(TRAY_GEOMETRY, collection_trays=trays)
    assert len(config['blocks']) == 36
    assert config['rules']['max_turns_per_player'] == 15
    assert 'tray' in config['rules']['collection']
    assert len(config['collection_slots']['front']) == len(config['collection_slots']['rear']) == 15
    first = config['collection_slots']['front'][0]
    assert first['position_m'] == pytest.approx([.41, -.445, .801])
    assert config['collection_slots']['front'][-1]['position_m'] == pytest.approx([.89, -.205, .801])
    assert config['blocks']['block_00']['center_h'][2] == pytest.approx(.5)
    for a, b in zip(config['collection_slots']['front'], config['collection_slots']['rear']):
        assert [a['position_m'][i]+b['position_m'][i] for i in (0, 1)] == pytest.approx([1.3, 0.])
    for side in ('front', 'rear'):
        low, high = trays[side]['interior_bounds_xy_m']
        for slot in config['collection_slots'][side]:
            assert slot['support_actor_id'] == trays[side]['actor_id']
            for bounds in slot['bounds_h']:
                world_xy = [bounds[i] * .03 + [.65, 0.][i] for i in (0, 1)]
                assert all(low[i] <= world_xy[i] <= high[i] for i in (0, 1))
    assert TowerExtractionGame(config).config == config


@pytest.mark.parametrize('damage', ['capacity', 'symmetry', 'tower_overlap', 'outside_table',
                                  'actor_collision', 'wall_collision', 'floor_height', 'pitch', 'nonfinite'])
def test_invalid_collection_tray_geometry_fails_before_release(damage):
    trays = collection_trays()
    if damage == 'capacity':
        trays['front']['interior_bounds_xy_m'] = [[.49, -.48], [.81, -.17]]
        trays['rear']['interior_bounds_xy_m'] = [[.49, .17], [.81, .48]]
    elif damage == 'symmetry':
        trays['rear']['interior_bounds_xy_m'][1][1] -= .01
    elif damage == 'tower_overlap':
        trays['front']['interior_bounds_xy_m'][1][1] = -.05
    elif damage == 'outside_table':
        trays['front']['interior_bounds_xy_m'][0][0] = .10
    elif damage == 'actor_collision':
        trays['front']['actor_id'] = 'table'
    elif damage == 'wall_collision':
        trays['front']['wall_actor_ids'].append(trays['rear']['actor_id'])
    elif damage == 'floor_height':
        trays['front']['floor_z_m'] = .77
    elif damage == 'pitch':
        trays['front']['slot_pitch_xy_m'][0] = .19
    else:
        trays['front']['interior_bounds_xy_m'][0][0] = math.nan
    with pytest.raises(ValueError):
        task_configuration(TRAY_GEOMETRY, collection_trays=trays)


def test_real_collection_floor_contacts_allow_both_players_to_handoff():
    stream = tray_stream(turns=1).ready()
    first = stream.extract('block_03')
    assert first['completed_turns'] == {'front': 1, 'rear': 0}
    assert first['current_side'] == 'rear'
    second = stream.extract('block_06')
    assert second['completed_turns'] == {'front': 1, 'rear': 1}
    assert second['status'] == 'draw' and second['reason'] == 'balanced_turn_limit'


@pytest.mark.parametrize('actor', ['table', 'floor', 'front_collection_tray_wall_neg_y',
                                 'rear_collection_tray_floor'])
def test_collection_cannot_substitute_table_world_floor_walls_or_other_tray(actor):
    stream = tray_stream().ready()
    stream.touch()
    stream.place()
    for _ in range(60):
        obs = stream.sample()
        contact = next(c for c in obs['contacts'] if c['a'] == 'block_03')
        contact['b'] = actor
        stream.game.observe(obs)
    assert stream.game.status == 'active'
    assert stream.game.turns['front'] == 0


@pytest.mark.parametrize('damage', ['coverage', 'unloaded', 'off_floor', 'penetration'])
def test_tray_collection_requires_complete_loaded_floor_contact_without_penetration(damage):
    stream = tray_stream().ready()
    stream.touch()
    stream.place()
    stream.step(30)
    obs = stream.sample()
    contact = next(c for c in obs['contacts'] if c['a'] == 'block_03')
    if damage == 'coverage':
        contact['supported_interval_s'] = .01
    elif damage == 'unloaded':
        contact['minimum_normal_force_n'] = 0.
    elif damage == 'off_floor':
        contact['point_h'][2] += .1
    else:
        contact['separation_h'] = -.1
    assert stream.game.observe(obs)['status'] == 'active'
    stream.step(50)
    assert stream.game.turns['front'] == 0
    assert stream.step()['completed_turns']['front'] == 1


def test_tower_support_remains_rooted_at_the_table_with_collection_trays():
    stream = tray_stream()
    for _ in range(501):
        obs = stream.sample()
        for contact in obs['contacts']:
            if contact['b'] == 'table':
                contact['b'] = 'front_collection_tray_floor'
        stream.game.observe(obs)
    assert stream.game.status == 'abort'
    assert stream.game.reason == 'invalid_reset_not_stable'
    assert stream.game.winner is None


def test_previously_collected_blocks_keep_their_own_tray_floor_requirement():
    stream = tray_stream().ready()
    stream.extract('block_03')
    stream.touch('block_06')
    stream.place()
    for _ in range(60):
        obs = stream.sample()
        next(c for c in obs['contacts'] if c['a'] == 'block_03')['b'] = 'front_collection_tray_wall_neg_y'
        stream.game.observe(obs)
    assert stream.game.status == 'active'
    assert stream.game.turns == {'front': 1, 'rear': 0}
