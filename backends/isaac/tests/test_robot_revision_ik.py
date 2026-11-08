from types import SimpleNamespace

import numpy as np

from scripts.diagnostics.check_bimanual_options import solve_target
from scripts.validation.laundry_configuration_workspace import _arm_chains


class Arm:
    names = ['q']

    def __init__(self):
        self.calls = []
        self.poses = {}

    def inverse(self, target, tip, seed, **options):
        self.calls.append((np.array(target), dict(seed), options))
        value = float(len(self.calls))
        self.poses[value] = np.array(target)
        return {'converged': True, 'joint_positions_rad': {'q': value}}

    def forward(self, joints, base):
        return {'tip': self.poses[joints['q']]}


def model():
    return SimpleNamespace(arm=Arm(), names=['q','finger'], arm_tip='tip',
        arm_base=np.eye(4), wrist_offset=np.eye(4),
        public_geometry=lambda *a: {'fingertip_positions_m': {'middle': [0.,0.,0.]}})


def test_opt_in_restarts_colliding_converged_solution_at_identical_target():
    own = model()
    checker = SimpleNamespace(boxes=lambda side,q:q['q'],
                              check=lambda boxes: {'clear': boxes['right'] >= 2.})
    initial = {'q':0., 'finger':.25}
    seed = {'offset_rad':1.2, 'arm_joint_positions_rad': {'q':.5}}
    result = solve_target(own,initial,[.1,.2,.3],'right',0.,'above',checker,{},
                          restart_seeds=lambda model,q: [seed])
    assert result['clearance_passed'] and len(own.arm.calls)==2
    assert result['joint_positions_rad']['finger']==initial['finger']
    assert not result['original_rejected_candidate']['clearance_passed']
    first,second = own.arm.calls
    assert np.array_equal(first[0],second[0])
    assert first[2]==second[2]==dict(iterations=160,multi_starts=0,
        position_tolerance=.0002,rotation_tolerance=.004)
    assert second[1]==seed['arm_joint_positions_rad']


def test_default_clear_candidate_never_calls_restart():
    own = model()
    def forbidden(*_):
        raise AssertionError('No restarts are needed after a clear first solution')
    result = solve_target(own,{'q':0.,'finger':0.},[.1,.2,.3],'right',0.,'above',
                          restart_seeds=forbidden)
    assert result['clearance_passed'] and len(own.arm.calls)==1
    assert 'original_rejected_candidate' not in result


def test_laundry_three_endpoints_remain_coupled_and_restarts_lazy():
    own = model(); requested=[]
    initial = {'q':0.,'finger':.25}
    seed={'offset_rad':1.2,'arm_joint_positions_rad':{'q':.5}}
    def restart(model,q):
        requested.append(dict(q));return [seed]
    wrist=np.eye(4); wrist[:3,3]=[.1,.2,.3]
    runs=_arm_chains(own,initial,wrist,np.zeros(3),np.array([.1,.2,.3]),restart)
    first,calls,record=next(runs)
    assert calls==3 and record is None and not requested
    assert [r['rise_m'] for r in first]==[.10,0.,.05]
    second,calls,record=next(runs)
    assert calls==3 and record==seed and requested[0]['finger']==.25
    assert own.arm.calls[3][1]==seed['arm_joint_positions_rad']
    for a,b in zip(own.arm.calls[:3],own.arm.calls[3:]):
        assert np.array_equal(a[0],b[0]) and a[2]==b[2]
    assert max(r['actual_pinch_error_m'] for r in second)<1e-14
