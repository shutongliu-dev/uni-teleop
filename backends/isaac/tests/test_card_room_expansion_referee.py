"""Synthetic native-boundary consumer tests, never robot/dynamics qualification."""
from copy import deepcopy
from dataclasses import replace
import json
import math

import pytest

from bench.protocol.card_games import CardGameState, CardGameEventPage, EVENT_ADAPTER
from scenes.workrooms.card_room.game_designs import design_for
from scenes.workrooms.card_room.expansion_rules import IllegalMove, rotated_cells
from scenes.workrooms.card_room.expansion_referee import (
    ExpansionMatch, ExpansionNativeFrame, ExpansionObjectEvidence, PLAYERS,
    public_rule_contract,
)


def metadata(task):
    design = design_for(task)
    result = {'task_id': task, 'dimensions': design,
              'actors': {p['id']: '/World/CardGame/Actors/'+p['id'] for p in design['pieces']},
              # Synthetic consumer fixture parameters, not estimated asset masses.
              'masses_kg': {p['id']: .1 for p in design['pieces']},
              'initial_positions_m': {p['id']: p['initial_position_m'] for p in design['pieces']},
              'initial_orientations_wxyz': {p['id']: p['initial_orientation_wxyz'] for p in design['pieces']}}
    if task == 'memory_pairs':
        result['private_reset_pairs'] = {f'card_{i:02d}': f'pair_{i//2}' for i in range(12)}
    return result


class Rig:
    dt = .01

    def __init__(self, task, *, baseline=True):
        self.meta = metadata(task)
        self.match = ExpansionMatch(task, asset_metadata=self.meta)
        self.objects = {}
        self.step = -1
        for p in self.meta['dimensions']['pieces']:
            position = tuple(p['initial_position_m'][i]-(.0006 if i == 2 else 0.) for i in range(3))
            self.objects[p['id']] = ExpansionObjectEvidence(
                position, (1., 0., 0., 0.), (0., 0., 0.), (0., 0., 0.), .1*9.81*self.dt,
                {player: 0. for player in PLAYERS}, 0.,
                robot_nonhand_contact_impulses_ns={player: 0. for player in PLAYERS})
        self.validate_public()
        if baseline:
            self.tick()

    def validate_public(self):
        CardGameState.model_validate(self.match.public_state())
        for event in self.match.events_after():
            EVENT_ADAPTER.validate_python(event)
        CardGameEventPage.model_validate(self.match.events_since())

    def tick(self, **frame_fields):
        self.step += 1
        frame = ExpansionNativeFrame(self.step, self.step*self.dt, self.dt, deepcopy(self.objects), True)
        result = self.match.observe(replace(frame, **frame_fields))
        self.validate_public()
        return result

    def modify(self, name, **kwargs):
        self.objects[name] = replace(self.objects[name], **kwargs)

    def hand(self, name, *, player=None, arm=False, **kwargs):
        player = player or self.match.current_player
        self.modify(name, robot_contact_impulses_ns={p: .001 if p == player else 0. for p in PLAYERS},
                    robot_nonhand_contact_impulses_ns={p: .001 if arm and p == player else 0. for p in PLAYERS}, **kwargs)
        return self.tick()

    def release(self, *names):
        for name in names:
            self.modify(name, robot_contact_impulses_ns={p: 0. for p in PLAYERS},
                        robot_nonhand_contact_impulses_ns={p: 0. for p in PLAYERS},
                        linear_velocity_m_s=(0., 0., 0.), angular_velocity_rad_s=(0., 0., 0.))

    def settle(self):
        result = None
        for _ in range(31):
            result = self.tick()
        return result

    def intent(self, **action):
        result = self.match.admit_intent(self.match.current_player, action)
        self.validate_public()
        return result

    def reveal(self, name, *, quaternion=(0., 1., 0., 0.)):
        self.intent(type='reveal_card', card_id=name)
        self.hand(name, quaternion_wxyz=quaternion)
        self.release(name)
        return self.settle()

    def collect(self, a, b, x=.455):
        self.intent(type='collect_pair')
        color = 'black' if self.match.current_player == 'player_0' else 'white'
        y = -.355 if color == 'black' else .355
        self.hand(a, position_m=(x, y, .816))
        self.hand(b, position_m=(x, y, .836), deck_support_impulse_ns=0.,
                  object_support_impulses_ns={a: .1*9.81*self.dt})
        self.release(a, b)
        return self.settle()

    def start_shot(self, name):
        self.intent(type='shoot_disk', piece_id=name)
        sign = 1 if self.match.current_player == 'player_0' else -1
        self.hand(name, linear_velocity_m_s=(0., sign*.1, 0.))
        return sign

    def finish_shot(self, name, xy, *, static=0.):
        sign = 1 if self.match.current_player == 'player_0' else -1
        x = self.objects[name].position_m[0]
        self.hand(name, position_m=(x, -sign*.245, .815), linear_velocity_m_s=(0., sign*.1, 0.))
        self.release(name)
        self.modify(name, position_m=(*xy, .815 if not static else .1),
                    deck_support_impulse_ns=0. if static else .1*9.81*self.dt,
                    other_static_support_impulse_ns=static)
        return self.settle()


@pytest.mark.parametrize('task', ['polyomino_territory', 'barrier_race', 'memory_pairs', 'disk_curling'])
def test_metadata_baseline_and_public_private_boundary(task):
    rig = Rig(task, baseline=False)
    assert rig.match.phase == 'awaiting_native_baseline'
    assert rig.match.public_state()['event_cursor'] == -1
    assert 'private_reset_pairs' not in json.dumps(public_rule_contract(task))
    assert 'position_m' not in json.dumps(rig.match.public_state())
    bad = metadata(task)
    bad['masses_kg'].pop(next(iter(bad['masses_kg'])))
    with pytest.raises(ValueError):
        ExpansionMatch(task, asset_metadata=bad)
    rig.tick()
    assert rig.match.phase != 'awaiting_native_baseline'


def test_memory_map_is_bound_to_actual_asset_and_hidden_until_revealed():
    meta = metadata('memory_pairs')
    with pytest.raises(ValueError):
        ExpansionMatch('memory_pairs', asset_metadata=meta, private_pairs={'card_00': 'different'})
    rig = Rig('memory_pairs')
    assert 'pair_0' not in json.dumps(rig.match.public_state())
    assert rig.match.events_after() == []
    assert rig.reveal('card_00').committed
    assert rig.match.turn_index == 1 and rig.match.current_player == 'player_0'
    assert rig.match.public_state()['round_index'] == 0
    events = rig.match.events_after()
    assert [e['type'] for e in events] == ['card_revealed', 'operation_committed']
    assert events[0]['details'] == {'card_id': 'card_00', 'symbol': 'pair_0', 'expected': True}
    assert 'pair_1' not in json.dumps(events)
    assert rig.match.events_since(-1, 1)['has_more']
    assert rig.match.events_since(0, 1)['next_cursor'] == 1
    events[0]['details']['symbol'] = 'mutated'
    assert rig.match.events_after()[0]['details']['symbol'] == 'pair_0'


@pytest.mark.parametrize('action', [
    {'type': 'place_piece', 'piece_id': 'white_mono', 'anchor': [0, 0]},
    {'type': 'place_piece', 'piece_id': 'black_mono', 'anchor': [True, 0]},
    {'type': 'place_piece', 'piece_id': [], 'anchor': [0, 0]},
    {'type': 'place_piece', 'piece_id': 'black_mono', 'anchor': [0, 0], 'physics_valid': True},
    {'type': 'pass'}, {'type': 'reveal_card', 'card_id': 'card_00'},
])
def test_illegal_requests_are_atomic(action):
    rig = Rig('polyomino_territory')
    before = deepcopy({**rig.match.__dict__, 'rules': rig.match.rules.__dict__})
    with pytest.raises(IllegalMove):
        rig.match.admit_intent('player_0', action)
    assert {**rig.match.__dict__, 'rules': rig.match.rules.__dict__} == before


def poly_target(rig, *, hand=True, arm=False, x=.52):
    rig.intent(type='place_piece', piece_id='black_mono', anchor=[0, 0])
    if hand:
        rig.hand('black_mono', arm=arm, position_m=(x, -.13, .817))
    else:
        rig.modify('black_mono', position_m=(x, -.13, .817))
        rig.tick()
    rig.release('black_mono')


def test_poly_requires_real_own_hand_and_full_endpoint_dwell():
    rig = Rig('polyomino_territory')
    poly_target(rig)
    for _ in range(30):
        assert not rig.tick().committed
    assert rig.match.turn_index == 0
    assert rig.tick().committed
    assert rig.match.turn_index == 1 and rig.match.current_player == 'player_1'
    assert rig.match.rules.scores() == {'black': 1, 'white': 0}


def test_late_contact_resets_entire_release_window_and_deadline_wins_over_commit():
    rig = Rig('polyomino_territory')
    poly_target(rig)
    for _ in range(29): rig.tick()
    rig.hand('black_mono')
    rig.release('black_mono')
    for _ in range(30): assert not rig.tick().committed
    rig.match.interrupt('time_limit')
    assert not rig.tick().committed
    assert rig.match.turn_index == 0 and rig.match.winner is None
    assert rig.match.public_state()['status']['outcome'] == 'truncated'


@pytest.mark.parametrize('kind', ['no_contact', 'arm_only', 'floor_support', 'cell_overhang', 'robot_still_contacting'])
def test_poly_negative_evidence_cannot_commit(kind):
    rig = Rig('polyomino_territory')
    poly_target(rig, hand=kind != 'no_contact', arm=kind == 'arm_only', x=.524 if kind == 'cell_overhang' else .52)
    if kind == 'floor_support':
        rig.modify('black_mono', deck_support_impulse_ns=0., other_static_support_impulse_ns=.1*9.81*rig.dt)
    if kind == 'robot_still_contacting':
        rig.hand('black_mono')
    assert not rig.settle().committed
    assert rig.match.turn_index == 0 and not rig.match.terminal


def test_unrequested_inventory_displacement_and_committed_cell_overhang_require_recovery():
    rig = Rig('polyomino_territory')
    poly_target(rig)
    initial = rig.objects['white_mono']
    rig.modify('white_mono', position_m=(.9, .3, .817))
    assert rig.tick().recovery_required
    rig.objects['white_mono'] = initial
    assert rig.settle().committed
    rig.modify('black_mono', position_m=(.524, -.13, .817))
    assert rig.tick().recovery_required  # Within generic pose tolerance, outside occupied cell.


@pytest.mark.parametrize('fault', ['gap', 'duplicate', 'source', 'contacts', 'missing_body', 'nan', 'unknown_penetration', 'missing_robot_ownership'])
def test_bad_native_stream_ends_incomplete_without_winner(fault):
    rig = Rig('polyomino_territory')
    overrides = {}
    if fault == 'gap': overrides['substep_index'] = 3
    if fault == 'duplicate': overrides.update(substep_index=0, simulation_time_s=0.)
    if fault == 'source': overrides['source'] = 'model_generated'
    if fault == 'contacts': overrides['contacts_complete'] = False
    if fault == 'missing_body': rig.objects.pop('black_mono')
    if fault == 'nan': rig.modify('black_mono', position_m=(math.nan, 0., 0.))
    if fault == 'unknown_penetration': rig.modify('black_mono', maximum_penetration_m=None)
    if fault == 'missing_robot_ownership': rig.modify('black_mono', robot_nonhand_contact_impulses_ns={})
    result = rig.tick(**overrides)
    assert result.evidence_unavailable
    assert rig.match.terminal and rig.match.winner is None and not rig.match.rules.draw
    assert rig.match.public_state()['status']['end_reason'] == 'missing_physical_evidence'


def test_physics_failure_and_budget_interrupts_are_not_rule_losses():
    rig = Rig('barrier_race')
    rig.modify('black_pawn', maximum_penetration_m=.004)
    rig.tick()
    assert rig.match.public_state()['status']['end_reason'] == 'physical_failure'
    assert rig.match.rules.physics_valid is False and rig.match.winner is None
    for reason in ('cancelled', 'runtime_error', 'time_limit', 'step_limit'):
        other = Rig('memory_pairs')
        other.match.interrupt(reason)
        other.validate_public()
        assert other.match.winner is None and not other.match.rules.draw


def test_barrier_pawn_and_wall_confirm_real_geometry_with_operation_epochs():
    rig = Rig('barrier_race')
    rig.intent(type='move_pawn', destination=[2, 1])
    rig.hand('black_pawn', position_m=(.65, -.075, .8485), quaternion_wxyz=(0., 0., 0., 1.))
    rig.release('black_pawn')
    assert rig.settle().committed  # Cylindrical pawn yaw is immaterial.
    rig.intent(type='place_barrier', piece_id='white_barrier_0', anchor=[1, 1], orientation='vertical')
    # Correct slot but wrong physical orientation must not become a logical wall.
    rig.hand('white_barrier_0', position_m=(.6125, -.0375, .8335))
    rig.release('white_barrier_0')
    assert not rig.settle().committed
    rig.hand('white_barrier_0', quaternion_wxyz=(math.sqrt(.5), 0., 0., math.sqrt(.5)))
    rig.release('white_barrier_0')
    assert rig.settle().committed
    assert rig.match.turn_index == 2 and rig.match.rules.walls['white_barrier_0']['orientation'] == 'vertical'


def test_barrier_opposite_edge_winner_only_after_final_physical_confirmation():
    rig = Rig('barrier_race')
    for color, cell in [('black', (2, 1)), ('white', (2, 3)), ('black', (2, 2)),
                        ('white', (2, 1)), ('black', (2, 3)), ('white', (2, 0))]:
        name = color+'_pawn'
        rig.intent(type='move_pawn', destination=list(cell))
        rig.hand(name, position_m=(.65+(cell[0]-2)*.075, (cell[1]-2)*.075, .8485))
        assert rig.match.winner is None
        rig.release(name)
        assert rig.settle().committed
    assert rig.match.winner == 'player_1' and rig.match.turn_index == 6


def test_memory_match_real_collection_stack_and_same_player_next_epoch():
    rig = Rig('memory_pairs')
    assert rig.reveal('card_00', quaternion=(0., 0., 1., 0.)).committed  # Either real flip axis.
    assert rig.reveal('card_01').committed
    assert rig.match.phase == 'collect_match'
    assert rig.collect('card_00', 'card_01').committed
    assert rig.match.turn_index == 3 and rig.match.rules.turn_index == 1
    assert rig.match.current_player == 'player_0' and rig.match.rules.scores()['black'] == 1
    assert len(rig.match.events_after()) == 5


def test_memory_mismatch_needs_both_real_returns_before_switch():
    rig = Rig('memory_pairs')
    rig.reveal('card_00'); rig.reveal('card_02')
    rig.intent(type='flip_back')
    rig.hand('card_00', quaternion_wxyz=(1., 0., 0., 0.)); rig.release('card_00')
    assert not rig.settle().committed
    assert rig.match.current_player == 'player_0'
    rig.hand('card_02', quaternion_wxyz=(1., 0., 0., 0.)); rig.release('card_02')
    assert rig.settle().committed
    assert rig.match.turn_index == 3 and rig.match.current_player == 'player_1'
    assert rig.match.phase == 'reveal_first'


@pytest.mark.parametrize('with_intent', [False, True])
def test_unexpected_reveal_is_public_durable_and_cannot_be_cancelled_away(with_intent):
    rig = Rig('memory_pairs')
    if with_intent:
        rig.intent(type='reveal_card', card_id='card_00')
    rig.hand('card_02', quaternion_wxyz=(0., 1., 0., 0.))
    assert rig.match.terminal and rig.match.winner is None
    event = rig.match.events_after()[0]
    assert event['details'] == {'card_id': 'card_02', 'symbol': 'pair_1', 'expected': False}
    assert rig.match.public_state()['status']['end_reason'] == 'rule_violation'
    with pytest.raises(IllegalMove): rig.match.cancel_intent('player_0')
    assert rig.match.events_after()[0] == event


def test_expected_reveal_cannot_be_replaced_or_cancelled_after_information_is_public():
    rig = Rig('memory_pairs')
    rig.intent(type='reveal_card', card_id='card_00')
    rig.hand('card_00', quaternion_wxyz=(0., 1., 0., 0.))
    with pytest.raises(IllegalMove): rig.match.cancel_intent('player_0')
    with pytest.raises(IllegalMove): rig.match.admit_intent('player_0', {'type': 'reveal_card', 'card_id': 'card_01'})
    assert rig.match.events_after()[0]['details']['symbol'] == 'pair_0'
    rig.hand('card_00', quaternion_wxyz=(1., 0., 0., 0.))
    with pytest.raises(IllegalMove): rig.match.cancel_intent('player_0')
    with pytest.raises(IllegalMove): rig.match.admit_intent('player_0', {'type': 'reveal_card', 'card_id': 'card_01'})


@pytest.mark.parametrize('angle', [91., 120., 160.])
def test_partial_flip_exposes_identity_before_flat_placement_and_cannot_be_erased(angle):
    rig = Rig('memory_pairs')
    rig.intent(type='reveal_card', card_id='card_00')
    radians = math.radians(angle)/2
    rig.hand('card_00', quaternion_wxyz=(math.cos(radians), math.sin(radians), 0., 0.))
    assert rig.match.events_after()[0]['type'] == 'card_revealed'
    assert rig.match.turn_index == 0
    rig.hand('card_00', quaternion_wxyz=(1., 0., 0., 0.))
    with pytest.raises(IllegalMove): rig.match.cancel_intent('player_0')
    with pytest.raises(IllegalMove): rig.match.admit_intent('player_0', {'type': 'reveal_card', 'card_id': 'card_01'})
    assert rig.match.events_after()[0]['details']['symbol'] == 'pair_0'


def test_unrequested_partial_flip_is_broadcast_before_rule_violation():
    rig = Rig('memory_pairs')
    rig.hand('card_02', quaternion_wxyz=(.5, math.sqrt(.75), 0., 0.))
    assert rig.match.events_after()[0]['details'] == {'card_id': 'card_02', 'symbol': 'pair_1', 'expected': False}
    assert rig.match.public_state()['status']['end_reason'] == 'rule_violation'
    assert rig.match.winner is None


def test_memory_six_pairs_complete_with_supported_two_layer_collection_and_latched_winner():
    rig = Rig('memory_pairs')
    for pair in range(6):
        a, b = f'card_{pair*2:02d}', f'card_{pair*2+1:02d}'
        assert rig.reveal(a).committed
        assert rig.reveal(b).committed
        assert rig.collect(a, b, .455+pair*.078).committed
    assert rig.match.winner == 'player_0' and rig.match.turn_index == 18
    before = rig.match.public_state()
    rig.match.interrupt('time_limit')
    assert rig.match.public_state() == before


@pytest.mark.parametrize('fault', ['no_deck_root', 'cycle', 'outside_zone'])
def test_memory_collection_requires_actual_acyclic_support_in_own_area(fault):
    rig = Rig('memory_pairs')
    rig.reveal('card_00'); rig.reveal('card_01')
    rig.intent(type='collect_pair')
    x = .455 if fault != 'outside_zone' else .9
    rig.hand('card_00', position_m=(x, -.355, .816))
    rig.hand('card_01', position_m=(x, -.355, .836), deck_support_impulse_ns=0.,
             object_support_impulses_ns={'card_00': .00981})
    if fault in ('no_deck_root', 'cycle'):
        rig.modify('card_00', deck_support_impulse_ns=0.,
                   object_support_impulses_ns={'card_01': .00981} if fault == 'cycle' else {})
    rig.release('card_00', 'card_01')
    assert not rig.settle().committed
    assert rig.match.rules.scores()['black'] == 0


def test_disk_preparation_can_lift_and_regrasp_but_shot_has_one_effective_contact():
    rig = Rig('disk_curling')
    rig.intent(type='shoot_disk', piece_id='black_disk_0')
    rig.hand('black_disk_0', position_m=(.515, -.34, .9), deck_support_impulse_ns=0.,
             linear_velocity_m_s=(0., .1, 0.))
    assert rig.match.phase == 'preparing'
    rig.release('black_disk_0'); rig.tick()
    rig.hand('black_disk_0', position_m=(.515, -.34, .815), deck_support_impulse_ns=.00981,
             linear_velocity_m_s=(0., .1, 0.))
    assert rig.match.phase == 'propelling'
    assert rig.finish_shot('black_disk_0', (.65, 0.)).committed
    assert rig.match.current_player == 'player_1' and rig.match.rules.played == ['black_disk_0']


@pytest.mark.parametrize('fault', ['second_push', 'still_contacting_after_line', 'other_disk_contact', 'arm_push', 'opponent_push'])
def test_disk_physical_shot_violations_never_award_winner(fault):
    rig = Rig('disk_curling')
    rig.start_shot('black_disk_0')
    if fault == 'second_push':
        rig.release('black_disk_0'); rig.tick(); rig.hand('black_disk_0')
    elif fault == 'still_contacting_after_line':
        rig.hand('black_disk_0', position_m=(.515, -.23, .815))
    elif fault == 'other_disk_contact': rig.hand('white_disk_0')
    elif fault == 'arm_push': rig.hand('black_disk_0', arm=True)
    else: rig.hand('black_disk_0', player='player_1')
    assert rig.match.public_state()['status']['end_reason'] == 'shot_rule_violation'
    assert rig.match.turn_index == 0 and rig.match.winner is None and not rig.match.rules.draw


def test_disk_needs_real_propulsion_and_free_crossing_and_cannot_erase_launch():
    rig = Rig('disk_curling')
    rig.intent(type='shoot_disk', piece_id='black_disk_0')
    rig.hand('black_disk_0')
    rig.release('black_disk_0')
    assert not rig.settle().committed
    rig.hand('black_disk_0', linear_velocity_m_s=(0., .1, 0.))
    with pytest.raises(IllegalMove): rig.match.cancel_intent('player_0')
    rig.release('black_disk_0')
    assert not rig.settle().committed  # Never crossed the line.
    assert rig.match.turn_index == 0


def test_disk_eight_real_confirmations_score_final_all_disk_positions():
    rig = Rig('disk_curling')
    for i in range(4):
        for color, y in (('black', 0.), ('white', .1)):
            name = f'{color}_disk_{i}'
            rig.start_shot(name)
            assert rig.finish_shot(name, (.53+.08*i, y)).committed
    assert rig.match.turn_index == 8 and rig.match.rules.final_scores == {'black': 10, 'white': 6}
    assert rig.match.winner == 'player_0'


@pytest.mark.parametrize('support', [0., .00981])
def test_off_deck_disk_only_settles_with_measured_known_static_support(support):
    rig = Rig('disk_curling')
    rig.start_shot('black_disk_0')
    rig.hand('black_disk_0', position_m=(.515, -.245, .815))
    rig.release('black_disk_0')
    rig.modify('black_disk_0', position_m=(1.2, 0., .1), deck_support_impulse_ns=0.,
               other_static_support_impulse_ns=support)
    assert rig.settle().committed == bool(support)
    assert rig.match.rules.score_positions({'black_disk_0': (1.2, 0.)}, on_deck={'black_disk_0': False})['black'] == 0


def test_disk_below_target_needs_floor_support_and_is_not_scoring_eligible():
    rig = Rig('disk_curling')
    rig.start_shot('black_disk_0')
    rig.hand('black_disk_0', position_m=(.515, -.245, .815))
    rig.release('black_disk_0')
    rig.modify('black_disk_0', position_m=(.65, 0., .1), deck_support_impulse_ns=0.,
               other_static_support_impulse_ns=.00981)
    assert rig.settle().committed
    assert rig.match.rules.final_positions['black_disk_0'] == (.65, 0.)
    assert rig.match.rules.final_on_deck == {'black_disk_0': False}


@pytest.mark.parametrize('support', [0., .00981])
def test_sideways_off_deck_disk_needs_real_support_but_not_flat_orientation(support):
    rig = Rig('disk_curling')
    rig.start_shot('black_disk_0')
    rig.hand('black_disk_0', position_m=(.515, -.245, .815))
    rig.release('black_disk_0')
    rig.modify('black_disk_0', position_m=(1.2, 0., .1),
               quaternion_wxyz=(math.sqrt(.5), math.sqrt(.5), 0., 0.),
               deck_support_impulse_ns=0., other_static_support_impulse_ns=support)
    assert rig.settle().committed == bool(support)
    if support:
        assert rig.match.rules.final_on_deck == {'black_disk_0': False}


def test_upside_down_flat_disk_remains_scoring_eligible():
    rig = Rig('disk_curling')
    rig.intent(type='shoot_disk', piece_id='black_disk_0')
    rig.hand('black_disk_0', quaternion_wxyz=(0., 1., 0., 0.), linear_velocity_m_s=(0., .1, 0.))
    assert rig.match.phase == 'propelling'
    assert rig.finish_shot('black_disk_0', (.65, 0.)).committed
    assert rig.match.rules.final_on_deck == {'black_disk_0': True}


def test_pass_intent_is_not_logical_commit_and_waits_for_safe_native_frame():
    rig = Rig('polyomino_territory')
    # Confirm an explicit legal history that traps black's remaining 2x2 piece.
    history = [('black_tri_L', (3, 0), 180), ('white_tet_square', (4, 3), 0),
               ('black_mono', (1, 2), 0), ('white_tri_line', (1, 0), 0),
               ('black_tet_L', (1, 3), 0), ('white_tet_L', (2, 1), 270),
               ('black_domino', (0, 4), 90), ('white_domino', (4, 5), 0),
               ('black_tri_line', (3, 3), 90), ('white_mono', (0, 2), 0)]
    for name, anchor, angle in history:
        rig.intent(type='place_piece', piece_id=name, anchor=list(anchor), rotation_deg=angle)
        cells = rotated_cells(rig.match.rules.pieces[name]['cells'], angle)
        center = [anchor[i]+max(c[i] for c in cells)/2 for i in (0, 1)]
        radians = math.radians(angle)/2
        rig.hand(name, position_m=(.65+(center[0]-2.5)*.052, (center[1]-2.5)*.052, .817),
                 quaternion_wxyz=(math.cos(radians), 0., 0., math.sin(radians)))
        rig.release(name)
        assert rig.settle().committed
    rig.intent(type='pass')
    assert rig.match.turn_index == 10
    saved = rig.objects['white_mono']
    rig.modify('white_mono', deck_support_impulse_ns=0.)
    assert not rig.tick().committed
    rig.objects['white_mono'] = saved
    assert rig.tick().committed
    assert rig.match.turn_index == 11 and rig.match.current_player == 'player_1'
