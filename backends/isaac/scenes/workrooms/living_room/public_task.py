"""Declared static living-room rules; no native observations enter this module."""
from collections.abc import Mapping
import math
from bench.protocol.task_referee import project_task_rules


REMOTE_NUMBERS = ('press_position_m release_position_m joint_lower_m joint_upper_m joint_tolerance_m '
    'minimum_contact_impulse_ns maximum_penetration_m held_before_press_s stable_s '
    'maximum_linear_speed_m_s maximum_angular_speed_rad_s maximum_stable_position_range_m '
    'maximum_stable_orientation_range_rad holder_xy_tolerance_m holder_bottom_tolerance_m '
    'holder_orientation_tolerance_rad minimum_handheld_clearance_m max_time_s max_actions').split()
WATER_NUMBERS = ('dwell_s maximum_observation_gap_s maximum_unresolved_ml maximum_cup_volume_range_ml '
    'maximum_linear_speed_m_s maximum_angular_speed_rad_s maximum_tilt_rad failure_tilt_rad '
    'return_xy_radius_m return_z_tolerance_m minimum_lift_m minimum_spout_exit_ml minimum_support_force_n '
    'maximum_robot_contact_force_n maximum_stable_position_range_m maximum_stable_orientation_range_rad '
    'sleep_support_position_tolerance_m sleep_support_orientation_tolerance_rad').split()
QUANTITIES = ('initial_water_ml target_water_ml target_tolerance_ml max_spill_ml '
              'cup_usable_capacity_ml kettle_usable_capacity_ml').split()


def select(value, schema):
    """Recursive allowlist with strict leaf types; unknown fields are discarded."""
    if isinstance(schema, dict):
        if not isinstance(value, Mapping):
            raise ValueError('Declared public object is missing')
        return {key: select(value[key], child) for key, child in schema.items() if key in value}
    if isinstance(schema, list):
        if not isinstance(value, (tuple, list)) or len(value) > 100:
            raise ValueError('Declared public list is invalid')
        return [select(item, schema[0]) for item in value]
    if schema == 'number':
        if type(value) not in (int, float) or not math.isfinite(value):
            raise ValueError('Declared public number must be finite')
        return value
    if type(value) is not schema or (schema is str and (not value or len(value) > 4096)):
        raise ValueError('Declared public field has the wrong type')
    return value


def project_manipulation_task(task):
    if not isinstance(task, Mapping) or task.get('schema_version') != 'living-room-manipulation-task-v1':
        raise ValueError('Unsupported static manipulation task schema')
    identity = task.get('id')
    if identity not in ('remote_operation', 'water_pouring'):
        raise ValueError('Unknown living-room manipulation task')
    referee = project_task_rules(task)
    if referee is not None:
        return referee
    remote = identity == 'remote_operation'
    schema = {key: str for key in ('schema', 'status', 'instruction_zh', 'budget_exhaustion',
              'evidence_fault', 'qualification', 'evidence_scope', 'overfill')}
    schema.update(competitive=bool, stages=[str], failure=[str], randomization=[str],
        success={'all_of': [str], 'reset_credit': bool, 'stable_duration_s': 'number',
                 'thresholds': {key: 'number' for key in WATER_NUMBERS}})
    if remote:
        schema.update(button_sequence=[str], thresholds={key: 'number' for key in REMOTE_NUMBERS},
            buttons={'ids': [str], 'mechanism': str, 'press_evidence': [str], 'no_credit': [str]},
            recovery={'wrong_button': str, 'table_drop': str, 'table_assisted_flip': bool, 'regrasp': bool})
    else:
        schema.update(recovery=[str],
            goal={**{key: 'number' for key in QUANTITIES}, 'target_interval_ml': ['number'], 'status': str},
            measurement={**{key: str for key in ('quantity_basis', 'cup_water', 'water_difference', 'spill', 'spill_recovery',
                'conservation', 'missing_inventory', 'public_feedback', 'support_and_release')},
                'unresolved_allowance_ml': 'number'})
    rules = select(task.get('public_rules'), schema)
    expected = 'living_room_' + identity + '_rules_v2'
    if rules.get('schema') != expected or rules.get('competitive') is not False or not rules.get('instruction_zh'):
        raise ValueError('Public rules differ from declared task identity')
    goal_schema = {key: str for key in ('operation', 'frame_id', 'description', 'object_id')}
    goal_schema.update({key: 'number' for key in QUANTITIES})
    goal_schema.update(target_interval_ml=['number'], button_sequence=[str])
    goal = select(task.get('public_goal'), goal_schema)
    if not all(goal.get(key) for key in ('operation', 'frame_id', 'description')):
        raise ValueError('Static task intent must be explicit')
    title = select(task.get('title_zh'), str)
    steps = select(task.get('steps'), [{'id': str, 'instruction_zh': str}])
    if not steps or any(not row.get('instruction_zh') for row in steps):
        raise ValueError('Static task instructions must be explicit')
    seed = task.get('episode_seed', 0)
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError('Episode seed must be an unsigned 32-bit integer')
    return {'task_id': identity, 'task_kind': 'manipulation', 'create_endpoint': 'POST /episodes',
            'instruction': title + '。' + '；'.join(row['instruction_zh'] for row in steps),
            'goal': goal, 'public_rules': rules, 'episode_seed': seed}
