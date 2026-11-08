import copy
import unittest

import numpy as np

from scripts.validation.laundry_arc_options import phase_peak_heights, arc_height_report
from scripts.validation.laundry_captured_garment_fold import captured_arc, bezier_position


class ArcOptionsTests(unittest.TestCase):
    def test_legacy_values_and_report_are_unchanged(self):
        options = {'phases': {'0': {'hands': {'left': {'open_distance_m': .06}}}}}
        before = copy.deepcopy(options)
        heights = phase_peak_heights(options, 3, .12)
        self.assertEqual(heights, [.12, .12, .12])
        self.assertEqual(arc_height_report(options, heights), {})
        self.assertEqual(options, before)

    def test_two_short_sleeves_and_full_body_have_independent_heights(self):
        options = {'phases': {'0': {'peak_height_m': .06}, '1': {'peak_height_m': .06}}}
        heights = phase_peak_heights(options, 3, .12)
        self.assertEqual(heights, [.06, .06, .12])
        self.assertEqual(arc_height_report(options, heights)['phase_arc_settings'], [
            {'phase_index': 0, 'peak_height_m': .06, 'source': 'phase_override'},
            {'phase_index': 1, 'peak_height_m': .06, 'source': 'phase_override'},
            {'phase_index': 2, 'peak_height_m': .12, 'source': 'cli'}])

    def test_invalid_later_overrides_are_rejected_even_for_one_phase_probe(self):
        for value in (True, False, None, '.06', [], .039, .151, float('nan'), float('inf')):
            with self.subTest(value=value), self.assertRaises(ValueError):
                phase_peak_heights({'phases': {'2': {'peak_height_m': value}}}, 1, .12)
        for options in ({'phases': []}, {'phases': {'00': {}}}, {'phases': {'-1': {}}},
                        {'phases': {'0': None}}):
            with self.assertRaises(ValueError): phase_peak_heights(options, 1, .12)

    def test_height_only_changes_arc_interior_not_observed_pickup_or_placement(self):
        q = np.array([[-.1, 0., .902], [-.1, .01, .902],
                      [.1, 0., .903], [.1, .01, .903]])
        task = {'asset_metadata': {'design': {}, 'cloths': [{
            'unfolded_reference_vertices_m': q.tolist(), 'physical_thickness_m': .0016}]}}
        phase = {'reference_fold': [0, 0., -1, .007]}
        before = q.copy()
        default = captured_arc(task, phase, q, [0, 1], peak=.12)
        reduced = captured_arc(task, phase, q, [0, 1], peak=.06)
        np.testing.assert_array_equal(np.asarray(default['controls_m'])[[0, 2]],
                                      np.asarray(reduced['controls_m'])[[0, 2]])
        self.assertAlmostEqual(bezier_position(default['controls_m'], .5)[2]
                               - bezier_position(reduced['controls_m'], .5)[2], .06)
        np.testing.assert_array_equal(q, before)


if __name__ == '__main__': unittest.main()
