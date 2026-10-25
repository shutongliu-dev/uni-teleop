"""Load-bearing pinch for grasp_hold (v8 release); CPU contract tests, not physics acceptance."""
import pytest

from scenes.manipulation.contracts import make_success_contract
from scenes.manipulation.evaluator import SimpleTaskEvaluator
from scenes.manipulation.fidelity_release import pinch_force
from test_simple_tasks import Trajectory, task


class PassingHumanlike:
    """Stand-in for the frozen humanlike heuristic so only the pinch rule is tested."""
    def reset(self):
        pass

    def update(self, observation):
        return {'valid': True, 'passed': True, 'instantaneous_pass': True, 'grasp_attempted': True, 'reason': 'test'}


def pad_rows(time, forces):
    rows = []
    for finger, force in forces.items():
        impulse = force * .02 / 40
        for tick in range(1, 41):
            rows.append({'actor0': '/World/Objects/cube', 'actor1': f'/World/Robot/right_{finger}_elastomer',
                         'time_s': time - .02 + tick * .0005, 'point_count': 1,
                         'impulse_magnitude_sum_ns': impulse, 'min_separation_m': -.0001,
                         'impulse_vector_ns': [0., 0., impulse], 'nonzero_impulse_normal_count': 1,
                         'normal_length_max_abs_error': 0., 'impulse_components': 'normal_only'})
    return rows


def grasp_run(force=.4265, hold=1.):
    run = Trajectory('push')
    spec = task('grasp_hold')
    spec['evaluation'].update(initial_position_m=[.45, .1, .78], goal_position_m=[.45, .1, .78], hold_s=hold)
    if force is not None:
        spec['evaluation']['minimum_pinch_normal_force_n'] = force
    spec['evaluation']['humanlike_grasp'] = {}
    import scenes.manipulation.humanlike as module
    saved = module.HumanlikeGraspEvaluator
    module.HumanlikeGraspEvaluator = lambda config: PassingHumanlike()
    try:
        run.evaluator = SimpleTaskEvaluator(spec)
    finally:
        module.HumanlikeGraspEvaluator = saved
    run.observe()
    return run


def press(run, forces, steps):
    result = None
    for _ in range(steps):
        result = run.observe(hand=True, edits=lambda obs: obs['contacts']['rigid'].extend(
            pad_rows(obs['time_s'], forces)))
    return result


def test_published_threshold_is_weight_over_twice_average_friction():
    assert pinch_force(.05, .5, .65) == pytest.approx(.05 * 9.81 / (2 * .575))
    assert round(pinch_force(.05, .5, .65), 4) == .4265


def test_firm_pinch_succeeds_after_the_full_hold():
    run = grasp_run()
    result = press(run, {'thumb': 1., 'index': 1.}, 49)
    assert not result['success']
    result = press(run, {'thumb': 1., 'index': 1.}, 3)
    assert result['success'] and result['hold_mean_pinch_normal_force_n'] == pytest.approx(1.)


def test_light_touch_never_succeeds():
    # slot_0013-like: 0.06 N thumb, 0.21 N index, held for far longer than the hold.
    result = press(grasp_run(), {'thumb': .06, 'index': .21}, 120)
    assert not result['success'] and result['pinch_normal_force_n'] == pytest.approx(.06)


def test_tripod_counts_index_and_middle_together():
    result = press(grasp_run(), {'thumb': .9, 'index': .45, 'middle': .45}, 60)
    assert result['success']


def test_a_window_below_half_resets_the_hold():
    run = grasp_run()
    press(run, {'thumb': 1., 'index': 1.}, 40)
    press(run, {'thumb': .1, 'index': 1.}, 1)
    result = press(run, {'thumb': 1., 'index': 1.}, 40)
    assert not result['success']
    assert press(run, {'thumb': 1., 'index': 1.}, 12)['success']


def test_shell_load_does_not_count_as_pinch():
    run = grasp_run()
    result = None
    for _ in range(80):
        result = run.observe(hand=True, robot_actor='/World/Robot/right_index_MP', impulse=.0005,
            edits=lambda obs: obs['contacts']['rigid'].extend(pad_rows(obs['time_s'], {'thumb': 1.})))
    assert not result['success'] and result['pinch_normal_force_n'] == 0.


def test_without_the_option_grasp_scoring_is_unchanged():
    result = press(grasp_run(force=None, hold=.3), {'thumb': .06, 'index': .21}, 20)
    assert result['success'] and 'pinch_normal_force_n' not in result


def test_threshold_is_rejected_outside_grasp_hold():
    value = task('lift_hold')
    value['evaluation']['minimum_pinch_normal_force_n'] = .4
    with pytest.raises(ValueError, match='only to grasp_hold'):
        SimpleTaskEvaluator(value)


@pytest.mark.parametrize('bad', [0., -1., float('nan'), True])
def test_threshold_must_be_positive_finite(bad):
    value = task('push')
    value['evaluation']['minimum_pinch_normal_force_n'] = bad
    with pytest.raises(ValueError):
        SimpleTaskEvaluator(value)


def test_v3_contract_publishes_the_pinch_and_v2_contracts_do_not():
    value = task('grasp_hold')
    value['evaluation'].update(minimum_pinch_normal_force_n=.4265, hold_s=1.)
    contract = make_success_contract(value, .002)
    assert contract['protocol'] == 'simple_manipulation_success_v3'
    assert contract['terminal']['minimum_pinch_normal_force_n'] == .4265
    assert contract['terminal']['hold_s'] == 1.
    legacy = make_success_contract(task('grasp_hold'), .002)
    assert legacy['protocol'] == 'simple_manipulation_success_v2'
    assert 'minimum_pinch_normal_force_n' not in legacy['terminal']


ALLEGRO_PADS = {'thumb': ['allegro_link_15_0_tip', 'allegro_link_15_0'], 'index': ['allegro_link_3_0_tip', 'allegro_link_3_0'],
                'middle': ['allegro_link_7_0_tip', 'allegro_link_7_0']}


def pinch_only_run(links=ALLEGRO_PADS):
    """A hand without the SHARPA humanlike contract: the pinch-force rule certifies the grasp."""
    run = Trajectory('push')
    spec = task('grasp_hold')
    spec['evaluation'].update(initial_position_m=[.45, .1, .78], goal_position_m=[.45, .1, .78], hold_s=1.,
                              minimum_pinch_normal_force_n=.4265, pinch_links=links)
    run.evaluator = SimpleTaskEvaluator(spec)
    run.observe()
    return run


def link_rows(time, forces):
    rows = []
    for link, force in forces.items():
        impulse = force * .02 / 40
        for tick in range(1, 41):
            rows.append({'actor0': '/World/Objects/cube', 'actor1': f'/World/Robot/{link}',
                         'time_s': time - .02 + tick * .0005, 'point_count': 1,
                         'impulse_magnitude_sum_ns': impulse, 'min_separation_m': -.0001,
                         'impulse_vector_ns': [0., 0., impulse], 'nonzero_impulse_normal_count': 1,
                         'normal_length_max_abs_error': 0., 'impulse_components': 'normal_only'})
    return rows


def press_links(run, forces, steps):
    result = None
    for _ in range(steps):
        result = run.observe(hand=True, edits=lambda obs: obs['contacts']['rigid'].extend(link_rows(obs['time_s'], forces)))
    return result


def test_pinch_rule_alone_certifies_a_non_sharpa_grasp():
    result = press_links(pinch_only_run(), {'allegro_link_15_0_tip': 1., 'allegro_link_3_0': .6, 'allegro_link_7_0_tip': .5}, 60)
    assert result['success'] and result['hold_mean_pinch_normal_force_n'] == pytest.approx(1.)
    assert 'humanlike_grasp' not in result


def test_pinch_links_ignore_non_pad_hand_contact():
    # Palm contact is hand contact but never a pinch channel.
    result = press_links(pinch_only_run(), {'allegro_palm_link': 2., 'allegro_link_15_0_tip': 1.}, 80)
    assert not result['success'] and result['pinch_normal_force_n'] == pytest.approx(0.)


def test_pinch_links_must_be_complete_and_disjoint():
    with pytest.raises(ValueError, match='pinch_links'):
        pinch_only_run({'thumb': ['a'], 'index': ['b']})
    with pytest.raises(ValueError, match='pinch_links'):
        pinch_only_run({'thumb': ['a'], 'index': ['a'], 'middle': ['c']})


def test_left_hand_links_are_hand_contacts():
    assert SimpleTaskEvaluator._hand_actor('/World/Robot/Left/left_index_elastomer')
    assert not SimpleTaskEvaluator._hand_actor('/World/Robot/Left/fr3_left_link7')
    assert not SimpleTaskEvaluator._hand_actor('/World/Robot/fr3_hand_adapter')
