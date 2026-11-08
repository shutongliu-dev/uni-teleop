import numpy as np
import pytest

from scripts.validation.laundry_captured_garment_fold import rebound_descent_wrists,preflight_rebound_descent


def test_descent_starts_at_old_hover_not_shifted_new_hover():
    start=np.eye(4);start[:3,3]=[.4308,-.3578,1.0356]
    contact=start.copy();contact[:3,3]=[.4342,-.3569,.9276]
    before=start.copy();end_before=contact.copy()
    np.testing.assert_array_equal(rebound_descent_wrists({'left':start},{'left':contact},0.)['left'],start)
    np.testing.assert_array_equal(rebound_descent_wrists({'left':start},{'left':contact},1.)['left'],contact)
    middle=rebound_descent_wrists({'left':start},{'left':contact},.5)['left']
    np.testing.assert_allclose(middle[:3,3],(start[:3,3]+contact[:3,3])/2)
    assert start[2,3]!=pytest.approx(contact[2,3]+.10)
    np.testing.assert_array_equal(start,before);np.testing.assert_array_equal(contact,end_before)


def test_preflight_continues_seed_for_76_points_and_preserves_other_hand():
    start=np.eye(4);start[2,3]=1.03;contact=start.copy();contact[2,3]=.922
    seed={'left_joint':1.03,'held_right_finger':.52};calls=[]
    def solve(wrists,warm):
        calls.append(dict(warm));assert set(wrists)=={'left'}
        result={**warm,'left_joint':float(wrists['left'][2,3])}
        warm['held_right_finger']=99.  # Cannot mutate the original seed.
        return result
    report=preflight_rebound_descent({'left':start},{'left':contact},seed,solve)
    assert report['sample_count']==len(calls)==76
    assert seed=={'left_joint':1.03,'held_right_finger':.52}
    for i,row in enumerate(report['samples']):
        assert row['time_s']==pytest.approx(i*.02)
        assert row['joint_targets_rad']['held_right_finger']==.52
        if i:assert calls[i]==report['samples'][i-1]['joint_targets_rad']
    assert report['samples'][0]['fraction']==0.
    assert report['samples'][-1]['fraction']==1.


def test_rebind_rejects_wrist_flip_or_changed_batch_before_descent():
    start=np.eye(4);changed=start.copy();changed[:2,:2]*=-1
    for end,amount in [({'left':changed},.2),({'right':start},.2),({'left':start},float('nan'))]:
        with pytest.raises(ValueError):rebound_descent_wrists({'left':start},end,amount)


def test_preflight_failure_cannot_be_marked_as_completed_path():
    start=np.eye(4);end=start.copy();end[2,3]=-.10;seed={'joint':0.};calls=[]
    def fail(wrists,warm):
        calls.append(wrists['left'].copy())
        if len(calls)==24:raise ValueError('IK limit reached')
        return warm
    with pytest.raises(ValueError,match='IK limit'):
        preflight_rebound_descent({'left':start},{'left':end},seed,fail)
    assert len(calls)==24 and seed=={'joint':0.}
    for invalid in ({'extra':0.},{'joint':float('nan')}):
        with pytest.raises(ValueError,match='joint inventory'):
            preflight_rebound_descent({'left':start},{'left':end},seed,lambda *_:invalid)
