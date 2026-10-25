"""Published nominal inputs and explicit design-only quantity proposals."""
import pytest

from scenes.workrooms.catalog import task_definition
from scenes.workrooms.living_room.tasks.water_pouring.rules import public_rules
from scenes.workrooms.living_room.tasks.water_pouring.task import task_configuration


def test_quantity_overrides_reach_public_goal_without_fabricating_native_reset():
    plan = task_configuration(seed=11, quantities={
        'initial_water_ml': 300, 'target_water_ml': 120, 'target_tolerance_ml': 8,
        'cup_usable_capacity_ml': 250, 'kettle_usable_capacity_ml': 400, 'max_spill_ml': 0})
    assert plan['public_goal']['target_interval_ml'] == [112, 128]
    assert plan['public_goal'] == plan['rules']['goal']
    assert plan['public_goal']['max_spill_ml'] == 0
    assert plan['initial_state'] is None and plan['native_execution_available'] is False
    assert plan['source_release'] is None and plan['requires_new_release'] is True
    assert plan['configuration_scope'] == 'design_only_proposal'
    assert plan['rules']['status'] == plan['public_goal']['status'] == 'design_only_proposal'
    metadata = task_definition('living_room', 'water_pouring')
    assert metadata['task_number'] == 2 and metadata['user_confirmed_task']
    assert metadata['status'] == 'implemented'
    assert metadata['evaluator']['implementation'].endswith('.water_pouring.evaluator')
    assert metadata['evaluator']['reference_implementation'].endswith('.SequenceEvaluator')
    with pytest.raises(ValueError, match='Unknown task'):
        task_definition('living_room', 'glasses_case')


@pytest.mark.parametrize('quantities', [
    {'target_water_ml': 500},  # Cup overflow and insufficient source water.
    {'initial_water_ml': 100},  # Cannot provide the full target interval.
    {'cup_usable_capacity_ml': 100},
    {'kettle_usable_capacity_ml': 200},
    {'target_tolerance_ml': 100},  # Empty cup cannot satisfy the goal.
    {'max_spill_ml': 90},
    {'max_spill_ml': -1},
    {'initial_water_ml': True},
    {'target_water_ml': float('nan')},
    {'target_water_ml': float('inf')},
    {'target_water_ml': '100'},
    {'unrecognized_quantity_ml': 10},
    [],
])
def test_impossible_or_ambiguous_water_specs_are_rejected(quantities):
    with pytest.raises(ValueError):
        task_configuration(quantities=quantities)


def test_plan_and_public_rules_do_not_share_mutable_quantities():
    original = public_rules()
    plan = task_configuration()
    plan['public_goal']['target_interval_ml'][0] = -10
    assert plan['rules']['goal'] == original['goal']
    plan['rules']['fluid']['forbidden_shortcuts'].clear()
    assert public_rules() == original


def test_explicit_nominal_quantities_use_same_real_250ml_inventory():
    configuration = task_configuration(seed=11, quantities={'initial_water_ml': 250})
    assert configuration['status'] == 'implemented'
    assert configuration['native_execution_available'] is True
    assert configuration['public_goal']['initial_water_ml'] == 250
    assert configuration['public_goal']['target_interval_ml'] == [90, 110]
    assert configuration['task']['asset_metadata']['water']['total_mass_kg'] == .25
    assert configuration['task']['asset_metadata']['water']['particle_count'] == 488
    assert configuration['task']['asset_metadata']['water']['particle_spacing_m'] == .008
    assert configuration['task']['episode_seed'] == 11


def test_even_goal_only_quantity_change_is_unreleased_proposal():
    configuration = task_configuration(quantities={'target_water_ml': 95})
    assert configuration['native_execution_available'] is False
    assert configuration['source_release'] is None
    assert 'scene' not in configuration and 'task' not in configuration
    assert configuration['public_goal']['target_interval_ml'] == [85, 105]
