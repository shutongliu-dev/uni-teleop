"""CPU harness-contract checks. Fake observations are not physics acceptance."""
from argparse import Namespace
from types import SimpleNamespace

import pytest

from scripts.validation import validate_static_mesh_contacts as probe


@pytest.mark.parametrize('contact_evidence', [True, False])
def test_probe_consumes_assessor_report_and_preserves_initial_evidence(tmp_path, monkeypatch, contact_evidence):
    from bench.simulation.isaacsim_backend import env as env_module
    actors = ['/World/Room/Geometry/' + name for name in ('triangle_compound', 'convex', 'sdf')]
    object_path, support_path = '/World/Objects/cube', probe.SUPPORT_PATHS['floor']
    def pose(z, velocity=0.):
        return {'position_m': [.45, 0., z], 'quaternion_wxyz': [1., 0., 0., 0.],
                'linear_velocity_m_s': [0., 0., velocity], 'angular_velocity_rad_s': [0., 0., 0.]}
    class FakeEnv:
        dt = .0005
        def __init__(self, *args, **kwargs):
            self.scene_info = {'objects': {'triangle_compound_hole': {'path': object_path}},
                'kinematic_body_paths': actors, 'static_mesh_contacts': {'geometry_sha256': {}, 'mesh_approximations': {}}}
            self._fixed_actor_guard = SimpleNamespace(report=lambda: {'checked_batches': 9600})
        def action_spec(self):
            return {'initial_joint_positions_rad': {}}
        def reset(self, seed):
            self.count = 0
            return {'objects': {'triangle_compound_hole': pose(1.05)}}
        def capture_state(self):
            return {}
        def step(self, action):
            self.count += 1
            contact = [{'actor0': object_path, 'actor1': support_path,
                'impulse_magnitude_sum_ns': .01, 'min_separation_m': -.0001, 'time_s': self.count*.02}]
            return {'objects': {'triangle_compound_hole': pose(.03 if self.count > 3 else 1.-self.count*.2,
                                            -.2 if self.count <= 3 else 0.)},
                    'contacts': {'rigid': contact if contact_evidence else []}}
        def metadata(self):
            return {}
    def make_spec(output):
        source = output/'synthetic.usda'
        source.write_text('#usda 1.0\n')
        return {'workroom': {'usd_path': str(source), 'usd_sha256': probe.digest(source)},
                'workroom_diagnostics': {'triangle_compound_hole': {'half_extents_m': [.016, .02, .03],
                    'support_top_z_m': 0., 'expected_support_path': support_path}}}
    monkeypatch.setattr(probe, 'make_probe_scene', make_spec)
    def verify_loaded(*args, required_modules, **kwargs):
        import sys
        assert all(name in sys.modules for name in required_modules)
        return {'verified_for_harness_test': True}
    monkeypatch.setattr(probe, 'verify_if_snapshot', verify_loaded)
    monkeypatch.setattr(env_module, 'FR3DexterousIsaacEnv', FakeEnv)
    monkeypatch.setattr(probe, 'record', lambda env, obs, episode, phase, command=None:
                        {'observation': obs, 'state': {}, 'episode': episode, 'phase': phase})
    monkeypatch.setattr(probe, 'check_window', lambda *a: None)
    monkeypatch.setattr(probe, 'native_shape_stats', lambda env:
                        {'nb_trimesh_shapes': 5, 'nb_convex_shapes': 4, 'nb_kinematic_rigids': 3})
    def seal_json(output, report, env):
        import json
        # Reproduce the real seal boundary, including nested NumPy reductions.
        json.dumps(report, allow_nan=False)
        return report
    monkeypatch.setattr(probe, 'seal_report_before_shutdown', seal_json)
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', 'GPU-test-harness-only')
    monkeypatch.setenv('ISAAC_BENCH_SINGLE_CUDA_DEVICE', '1')
    result = probe.run(Namespace(output=tmp_path/'probe', device='cuda:0'))
    assert result['error'] is None
    assert result['passed'] is contact_evidence
    assert result['checks']['episode_0']['passed'] is contact_evidence
    assert result['checks']['episode_1']['passed'] is contact_evidence
    assert result['control_steps'] == 240
    assert (tmp_path/'probe'/'episode_0_reset.json').is_file()
    assert (tmp_path/'probe'/'episode_1_reset.json').is_file()


def test_hole_passage_rejects_rolling_off_replacement_hull():
    from copy import deepcopy
    name = 'triangle_compound_hole'
    object_path = '/World/Objects/' + name
    initial = {'objects': {name: {'position_m': [.45, 0., 1.05]}}}
    diagnostics = {name: {'half_extents_m': [.016, .02, .03]}}
    rows = [{'observation': {'objects': {name: {'position_m': [.45, 0., z]}},
                            'contacts': {'rigid': []}}} for z in (.95, .8, .7, .5)]
    assert probe.assess_hole_crossings(initial, rows, diagnostics, {name: object_path})['passed']
    moved = deepcopy(rows)
    moved[1]['observation']['objects'][name]['position_m'][0] += .2
    assert not probe.assess_hole_crossings(initial, moved, diagnostics, {name: object_path})['passed']
    contacted = deepcopy(rows)
    contacted[1]['observation']['contacts']['rigid'].append({'actor0': object_path,
        'actor1': '/World/Room/Geometry/triangle_compound', 'impulse_magnitude_sum_ns': .001,
        'min_separation_m': 0., 'time_s': .04})
    assert not probe.assess_hole_crossings(initial, contacted, diagnostics, {name: object_path})['passed']
