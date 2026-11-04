from copy import deepcopy
import unittest
from bench.simulation.isaacsim_backend.timing import control_timing
from bench.evaluation.rigid_safety import RigidContactSafetyGuard
from uni_teleop.backends.isaac.contact_guard import TeleopContactGuard


class ControlTimingTests(unittest.TestCase):
    def test_default_and_explicit_practice_period(self):
        scene = {'physics': {'dt_s': .0005}}
        self.assertEqual(control_timing(scene), (.0005, .02))
        scene['isaacsim'] = {'physics_dt_s': .004, 'control_dt_s': .04}
        with self.assertRaises(ValueError):
            control_timing(scene)
        scene['unscored_practice'] = True
        self.assertEqual(control_timing(scene), (.004, .04))
        for key, value in [('physics_dt_s', .003), ('physics_dt_s', 0),
                           ('control_dt_s', float('nan')), ('control_dt_s', .2)]:
            invalid = deepcopy(scene)
            invalid['isaacsim'][key] = value
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                control_timing(invalid)

    def test_contact_guard_accepts_ten_substeps_but_rejects_missing_substep(self):
        observation = {'sim_time_s': .04, 'contacts': {
            'source': 'PhysX stage-bound rigid contact tensor; normal impulses',
            'errors': [], 'rigid': [], 'sampling': {
                'complete': True, 'protocol': 'native_rigid_contact_sampling_v1',
                'start_sim_time_s': 0., 'end_sim_time_s': .04,
                'native_start_time_s': 0., 'native_end_time_s': .04,
                'physics_dt_s': .004, 'native_start_step_index': 0,
                'native_end_step_index': 10, 'sampled_substeps': 10}}}
        self.assertIsNone(TeleopContactGuard(control_dt=.04).observe(observation))
        # Benchmark default continues requiring its original 20 ms window.
        with self.assertRaises(ValueError):
            RigidContactSafetyGuard().observe(observation)
        observation['contacts']['sampling']['sampled_substeps'] = 9
        with self.assertRaises(ValueError):
            TeleopContactGuard(control_dt=.04).observe(observation)
