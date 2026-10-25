"""CPU contracts for the loose-bolt fastener candidate."""
from copy import deepcopy
import math

import numpy as np
import pytest

from scenes.workrooms.mechanical_assembly.tasks.fastener_assembly.evaluator import FastenerEvaluator
from scenes.workrooms.mechanical_assembly.tasks.fastener_assembly.rules import public_rules
from scenes.workrooms.mechanical_assembly.tasks.fastener_assembly.task import build_loose_task_inputs, build_task_inputs


@pytest.fixture(scope='module')
def task_templates():
    return build_loose_task_inputs(), build_task_inputs()


def _observation(task, time_s, *, first, contacts=None):
    objects = {}
    for name, pose in task['evaluation']['initial_objects'].items():
        row = deepcopy(pose)
        row.update(linear_velocity_m_s=[0., 0., 0.], angular_velocity_rad_s=[0., 0., 0.])
        objects[name] = row
    contacts = list(contacts or [])
    start = 0. if first else time_s - .02
    sampling = {
        'protocol': 'native_rigid_contact_sampling_v1', 'physics_dt_s': .0005,
        'start_sim_time_s': start, 'end_sim_time_s': time_s,
        'native_start_time_s': start, 'native_end_time_s': time_s,
        'native_start_step_index': round(start/.0005),
        'native_end_step_index': round(time_s/.0005),
        'sampled_substeps': 0 if first else 40, 'complete': True,
    }
    return {
        'time_s': time_s, 'sim_time_s': time_s,
        'last_duration_s': 0. if first else .02,
        'metrics': {'state_finite': True}, 'terminated': False,
        'objects': objects,
        'contacts': {
            'source': 'PhysX stage-bound rigid contact tensor; normal impulses',
            'errors': [], 'rigid': contacts, 'sampling': sampling,
        },
    }


def test_loose_reset_has_task_owned_supply_and_entry_guide(task_templates):
    values = task_templates[0]
    scene, task = values['scene'], values['task']
    assert task['public_rules']['reset_variant'] == 'loose_bolt'
    assert {row['id'] for row in scene['objects']} == {'bolt', 'screwdriver'}
    assert {'thread_entry_guide', 'bolt_supply_rest'} <= {row['id'] for row in scene['fixtures']}
    assert task['reset_conditions']['pose_snap'] is False
    assert task['evaluation']['engagement_requires_post_reset_contacts'] is True
    assert task['evaluation']['entry_guide_actor'].endswith('/thread_entry_guide')
    bolt = next(row for row in scene['objects'] if row['id'] == 'bolt')
    tool = next(row for row in scene['objects'] if row['id'] == 'screwdriver')
    fixture = next(row for row in scene['fixtures'] if row['id'] == 'threaded_fixture')
    assert bolt['position_m'][:2] != fixture['position_m'][:2]
    assert tool['position_m'][1] == 0.36
    assert task['reset_conditions']['tool_supply_position_m'][1] == 0.36
    assert scene['mechanical_task_provenance']['tool_supply_layout']['fixture_clearance_y_m'] > .07


def test_loose_evaluator_keeps_desktop_pickup_recoverable_and_unscored(task_templates):
    task = task_templates[0]['task']
    evaluator = FastenerEvaluator(task)
    result = evaluator.update(_observation(task, 0., first=True))
    assert result['reason'] == 'reset_baseline_not_scored'
    for index in range(1, 4):
        result = evaluator.update(_observation(task, index*.02, first=False))
    assert not result['success']
    assert not result['settled_baseline_recorded']
    assert result['irrecoverable_reason'] is None
    assert result['reason'] == 'waiting_for_passive_settlement'


class LooseTrace:
    """Synthetic slow pickup and engagement; not physical manipulation evidence."""
    def __init__(self, task_templates):
        self.task = deepcopy(task_templates[0]['task'])
        self.engaged_poses = task_templates[1]['task']['evaluation']['initial_objects']
        self.evaluator = FastenerEvaluator(self.task)
        self.poses = deepcopy(self.task['evaluation']['initial_objects'])
        self.index = 0
        self.evaluator.update(_observation(self.task, 0., first=True))

    def observation(self, *, pairs=(), impulse=.001, poses=None):
        self.index += 1
        time = self.index*.02
        actors = self.task['evaluation']['actors']
        rows = []
        for tick in range(1, 41):
            for a, b in pairs:
                pair_impulse = impulse if any(actor.startswith('/World/Robot/') for actor in (a, b)) else .001
                rows.append({'time_s': time-.02+tick*.0005,
                    'actor0': actors.get(a, a), 'actor1': actors.get(b, b),
                    'point_count': 1, 'nonzero_impulse_normal_count': int(pair_impulse > 0.),
                    'impulse_magnitude_sum_ns': pair_impulse, 'impulse_vector_ns': [0., 0., pair_impulse],
                    'normal_length_max_abs_error': 0., 'min_separation_m': -.00001,
                    'impulse_components': 'normal_only'})
        observation = _observation(self.task, time, first=False, contacts=rows)
        next_poses = deepcopy(self.poses if poses is None else poses)
        for name, pose in next_poses.items():
            state = observation['objects'][name]
            state.update(deepcopy(pose))
            state['linear_velocity_m_s'] = ((np.array(pose['position_m'])-
                self.poses[name]['position_m'])/.02).tolist()
        self.poses = next_poses
        return observation

    def emit(self, **options):
        return self.evaluator.update(self.observation(**options))

    def engage(self, *, contact=True):
        start = deepcopy(self.poses)
        goal = self.engaged_poses
        for fraction in np.linspace(0., 1., 501)[1:]:
            poses = {}
            for name, initial in start.items():
                q0 = np.array(initial['quaternion_wxyz'])
                q1 = np.array(goal[name]['quaternion_wxyz'])
                if np.dot(q0, q1) < 0.: q1 = -q1
                q = (1.-fraction)*q0+fraction*q1
                poses[name] = {
                    'position_m': ((1.-fraction)*np.array(initial['position_m'])+
                        fraction*np.array(goal[name]['position_m'])).tolist(),
                    'quaternion_wxyz': (q/np.linalg.norm(q)).tolist()}
            result = self.emit(poses=poses, pairs=[('bolt', '/World/Robot/right_index_DP')])
            assert result['referee']['state'] == 'ongoing'
            assert not result['settled_baseline_recorded']
            assert result['contact_backed_bolt_turn_rad'] == 0.
        return self.emit(pairs=ENGAGED_PAIRS if contact else [])


ENGAGED_PAIRS = [('bolt', 'threaded_fixture'), ('screwdriver', 'bolt')]
HAND_BOLT = ('bolt', '/World/Robot/right_index_DP')


@pytest.mark.parametrize('impulse', [0., 1.e-8, .001])
def test_pickup_contact_resets_settlement_even_after_engagement(task_templates, impulse):
    trace = LooseTrace(task_templates)
    trace.engage()
    for _ in range(23): trace.emit(pairs=ENGAGED_PAIRS)
    assert trace.evaluator.settle_elapsed == pytest.approx(.48)
    result = trace.emit(pairs=[*ENGAGED_PAIRS, HAND_BOLT], impulse=impulse)
    assert result['referee']['state'] == 'ongoing'
    assert trace.evaluator.engaged_seen and trace.evaluator.settle_elapsed == 0.
    assert not result['settled_baseline_recorded'] and result['progress'] == 0.
    for _ in range(24): result = trace.emit(pairs=ENGAGED_PAIRS)
    assert not result['settled_baseline_recorded']
    result = trace.emit(pairs=ENGAGED_PAIRS)
    assert result['settled_baseline_recorded'] and result['progress'] == 0.
    assert result['bolt_turn_rad'] == result['contact_backed_bolt_turn_rad'] == 0.
    result = trace.emit(pairs=[*ENGAGED_PAIRS, HAND_BOLT])
    assert result['referee']['state'] == 'task_failure'
    assert result['irrecoverable_reason'] == 'direct_robot_bolt_contact'


@pytest.mark.parametrize('target, reason', [
    ('bolt', 'direct_robot_bolt_contact'), ('screwdriver', 'non_hand_robot_tool_contact')])
def test_setup_never_allows_non_hand_robot_contact(task_templates, target, reason):
    trace = LooseTrace(task_templates)
    result = trace.emit(pairs=[(target, '/World/Robot/fr3_link5')])
    assert result['referee']['state'] == 'task_failure'
    assert result['irrecoverable_reason'] == reason


@pytest.mark.parametrize('damage', ['bad_contact', 'bad_motion', 'speed', 'penetration'])
def test_pickup_permission_does_not_bypass_native_evidence_or_safety(task_templates, damage):
    trace = LooseTrace(task_templates)
    observation = trace.observation(pairs=[HAND_BOLT])
    if damage == 'bad_contact':
        observation['contacts']['rigid'][-1]['impulse_magnitude_sum_ns'] = float('nan')
    elif damage == 'bad_motion':
        observation['objects']['bolt']['quaternion_wxyz'] = [0., 0., 0., 0.]
    elif damage == 'speed':
        observation['objects']['bolt']['linear_velocity_m_s'][0] = 1.
    else:
        observation['contacts']['rigid'][-1]['min_separation_m'] = -.01
    result = trace.evaluator.update(observation)
    assert result['referee']['state'] == ('evaluator_invalid' if damage.startswith('bad_') else 'safety_stop')
    assert result['irrecoverable_reason'] is None and not result['success']


@pytest.mark.parametrize('damage', ['axis', 'tilt'])
def test_first_engagement_window_enforces_thread_axis(task_templates, damage):
    trace = LooseTrace(task_templates)
    trace.engage(contact=False)
    # Move without engagement first, so the next contact window is quiet and
    # only the original axis/tilt gate determines its task outcome.
    for _ in range(25):
        intermediate = deepcopy(trace.poses)
        if damage == 'axis': intermediate['bolt']['position_m'][0] += .002/25
        else:
            half = math.radians(6./25)/2
            w, x, y, z = intermediate['bolt']['quaternion_wxyz']
            c, s = math.cos(half), math.sin(half)
            intermediate['bolt']['quaternion_wxyz'] = [c*w-s*x, c*x+s*w, c*y-s*z, c*z+s*y]
        result = trace.emit(poses=intermediate, pairs=[HAND_BOLT])
        assert result['referee']['state'] == 'ongoing'
    result = trace.emit(pairs=ENGAGED_PAIRS)
    assert result['referee']['state'] == 'task_failure'
    assert result['irrecoverable_reason'] == 'bolt_left_thread_axis'


def test_loose_pickup_then_tool_advance_and_released_stability_can_complete(task_templates):
    trace = LooseTrace(task_templates)
    trace.engage()
    for _ in range(24): result = trace.emit(pairs=ENGAGED_PAIRS)
    assert result['settled_baseline_recorded'] and result['contact_backed_bolt_turn_rad'] == 0.
    baseline = deepcopy(trace.poses)
    final_turn = math.pi/6
    for turn in np.linspace(0., final_turn, 151)[1:]:
        turn = float(turn)
        poses = deepcopy(baseline)
        for pose in poses.values():
            w, x, y, z = pose['quaternion_wxyz']
            c, s = math.cos(-turn/2), math.sin(-turn/2)
            pose['quaternion_wxyz'] = [c*w-s*z, c*x-s*y, c*y+s*x, c*z+s*w]
            pose['position_m'][2] -= .003*turn/(2*math.pi)
        result = trace.emit(poses=poses, pairs=[*ENGAGED_PAIRS,
            ('screwdriver', '/World/Robot/right_index_DP')])
        assert result['referee']['state'] == 'ongoing', trace.evaluator.invalid
    assert result['contact_backed_bolt_turn_rad'] == pytest.approx(final_turn)
    for _ in range(24): result = trace.emit(pairs=ENGAGED_PAIRS)
    assert not result['success']
    result = trace.emit(pairs=ENGAGED_PAIRS)
    assert result['success'] and result['release_stable_s'] == pytest.approx(.5)


def test_loose_rules_publish_the_setup_contact_boundary_without_changing_preinserted():
    loose = public_rules(reset_variant='loose_bolt')
    assert loose['schema'] == 'mechanical_fastener_rules_v5'
    assert 'before the settled baseline' in loose['success']['requirements'][-1]
    assert public_rules()['schema'] == 'mechanical_fastener_rules_v4'

