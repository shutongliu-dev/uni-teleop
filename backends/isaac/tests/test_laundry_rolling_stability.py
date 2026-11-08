"""Practical stability uses real samples in the final window, not cumulative drift."""
import numpy as np
import pytest

from scenes.workrooms.laundry_room.composition import build_task_inputs
from scenes.workrooms.laundry_room.evaluation import SequenceEvaluator, _ReleasedWindow
from scenes.workrooms.laundry_room.rules import PRACTICAL_PROFILE
from scenes.workrooms.laundry_room.terminal_window import terminal_window


def test_whole_towel_slow_drift_keeps_success_when_the_final_window_is_stable():
    task = build_task_inputs('towel_flattening', physics_options={'physics_dt_s': .002},
                             rule_profile=PRACTICAL_PROFILE)['task']
    cloth = task['asset_metadata']['cloths'][0]
    initial = np.asarray(cloth['initial_vertices_m'])
    target = np.asarray(cloth['rest_vertices_m'])+[.65, -.55, .9015]
    triangles = np.asarray(cloth['triangles'])
    scorer = SequenceEvaluator(task)
    positions, velocities, masks, scores = [], [], [], []
    for step in range(801):
        # 10 mm/s continues beyond 15 mm total drift, while every last
        # 500 ms moves only 5 mm and remains within the target placement.
        points = initial.copy() if step == 0 else target+[max(step-2, 0)*.002*.01, 0., 0.]
        velocity = np.tile([.01 if step >= 2 else 0., 0., 0.], (len(points), 1))
        control = [0, 1] if step == 1 else []
        frame = {'schema': 'laundry_native_frame.v1', 'complete': True,
            'time_s': step*.002, 'native_step': step,
            'control_mode': 'engineering_local_grasp', 'controlled_vertices': {'towel': control},
            'physics': {'complete': True, 'valid': True, 'gravity_enabled': True,
                'hidden_constraints': 0, 'self_intersections': 0,
                'intercloth_intersections': 0, 'max_penetration_m': 0.},
            'cloths': {'towel': {'enabled': True, 'vertices_m': points,
                'velocities_m_s': velocity, 'triangles': triangles}}}
        scores.append(scorer.observe(frame))
        positions.append(points)
        velocities.append(velocity)
        mask = np.zeros(len(points), dtype=bool)
        mask[control] = True
        masks.append(mask)
    arrays = {'native_step': np.arange(801), 'time_s': np.arange(801)*.002,
        'towel_q': np.asarray(positions), 'towel_qd': np.asarray(velocities),
        'towel_control_mask': np.asarray(masks), 'towel_triangles': triangles}
    terminal = terminal_window(arrays, scorer)
    assert np.linalg.norm(positions[-1]-positions[2], axis=1).min() > .015
    assert terminal['passed'] and terminal['max_node_drift_m'] == pytest.approx(.005)
    assert all(score['success'] for score in scores[253:])
    assert scores[-1]['released_stable_s'] == pytest.approx(.5)


def test_returning_to_the_anchor_does_not_hide_an_in_window_excursion():
    window = _ReleasedWindow(.5, .01, .015)
    for step in range(51):
        # The final and initial points coincide, but a real intermediate
        # sample moves 20 mm away. Checking endpoints alone would pass.
        displacement = .02*(1-abs(step-25)/25)
        valid, duration = window.add(step, step*.01,
                                     {'towel': np.array([[displacement, 0., 0.]])})
    assert not valid
    assert duration == 0.


def test_aabb_corner_that_was_never_observed_cannot_reject_the_window():
    window = _ReleasedWindow(.5, .01, .015)
    points = ([0., 0., 0.], [.012, 0., 0.], [0., .012, 0.])
    assert np.linalg.norm([.012, .012, 0.]) > .015
    for step, point in enumerate(points):
        valid, duration = window.add(step, step*.01, {'towel': np.array([point])})
        assert valid
        assert duration == pytest.approx(step*.01)


def test_expired_point_inside_a_partial_chunk_cannot_reject_the_window():
    window = _ReleasedWindow(.05, .01, .015)
    for step in range(7):
        point = [.1, 0., 0.] if step == 0 else [0., 0., 0.]
        valid, duration = window.add(step, step*.01, {'towel': np.array([point])})
    # Step 0 still contributes to its 50 ms chunk bounds, but is outside
    # the actual [step 1, step 6] window and must be ignored by exact replay.
    assert valid
    assert duration == pytest.approx(.05)
