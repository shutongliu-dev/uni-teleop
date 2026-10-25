from types import SimpleNamespace
import numpy as np

from scripts.validation.mechanical_ring_coverage import reset_receipt, support_topology
from scripts.validation.validate_mechanical_ring_sdf import stack_impulse_comparison


def test_reverse_stack_support_is_not_canonical_goal():
    names = ['spacer', 'wide_ring', 'top_ring']
    masses = {'spacer': 1., 'wide_ring': 2., 'top_ring': 3.}
    info = {'objects': {name: {'path': '/World/Objects/'+name} for name in names}}
    contacts = []
    for name, support, supported_mass in [('top_ring', '/World/Fixtures/shaft_fixture/body', 6.),
            ('wide_ring', '/World/Objects/top_ring', 3.), ('spacer', '/World/Objects/wide_ring', 1.)]:
        contacts.append({'actor0': '/World/Objects/'+name, 'actor1': support,
            'normal_impulses_ns': [supported_mass*9.81*.0005], 'normals': [[0., 0., 1.]]})
    row = {'raw_contacts': contacts, 'native_contact_impulse_crosscheck': {
        name: {'net_tensor_ns': [0., 0., mass*9.81*.0005],
               'pair_matrix_sum_ns': [0., 0., mass*9.81*.0005]} for name, mass in masses.items()}}
    reverse = stack_impulse_comparison([row]*1000, info, masses, .0005, support_topology({'id': 'negative'}))
    canonical = stack_impulse_comparison([row]*1000, info, masses, .0005, support_topology({'id': 'centered'}))
    assert reverse['quantitative_weight_balance_passed']
    assert not canonical['quantitative_weight_balance_passed']
    assert support_topology({'id': 'nominal_supply'}) is None


def test_native_reset_audit_rejects_drift_and_velocity():
    state = {'object_q': np.array([[.4, 0., .891, 0., 0., 0., 1.]]), 'object_qd': np.zeros((1, 6))}
    env = SimpleNamespace(object_names=['spacer'], capture_state=lambda: {k: v.copy() for k, v in state.items()})
    case = {'reset_objects': {'spacer': {'position_m': [.4, 0., .891], 'quaternion_wxyz': [1., 0., 0., 0.]}}}
    assert reset_receipt(env, case)['passed']
    state['object_q'][0, 2] += .001
    assert not reset_receipt(env, case)['passed']
    state['object_q'][0, 2] -= .001
    state['object_qd'][0, 2] = .1
    assert not reset_receipt(env, case)['passed']
