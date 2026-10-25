"""Fail-closed sector geometry witnesses, including buried-cut counterexamples."""
import numpy as np
import pytest
from bench.paths import load_json
from scripts.validation.gear_sector_sdf_checks import (
    root_bracket, gap_metrics, occupancy_metrics, legacy_union_metrics, QueryJournal)


def plane(offset, direction=1.):
    def query(points):
        points=np.asarray(points)
        return np.c_[np.tile([direction,0.,0.],(len(points),1)),direction*(points[:,0]-offset)]
    return query


@pytest.mark.parametrize('gap,passed', [(0.,True),(15.e-6,True),(-15.e-6,True),(25.e-6,False),(-25.e-6,False)])
def test_both_gap_and_overlap_are_bounded(gap,passed):
    center=np.zeros((1,3));normal=np.array([[1.,0.,0.]])
    left=root_bracket(plane(0.),center,normal,increasing=True)
    right=root_bracket(plane(gap,-1.),center,normal,increasing=False)
    result=gap_metrics(left,right)
    assert result['passed'] is passed
    assert result['signed_gap_m'][0]==pytest.approx(gap,abs=4.e-9)


def test_invalid_bracket_and_nonfinite_queries_fail():
    center=np.zeros((1,3));normal=np.array([[1.,0.,0.]])
    no_root=root_bracket(plane(.002),center,normal,increasing=True)
    invalid=root_bracket(lambda p:np.full((len(p),4),np.nan),center,normal,increasing=False)
    assert not no_root['qualified'].any() and not invalid['qualified'].any()
    assert gap_metrics(no_root,invalid)['passed'] is False


def test_occupancy_preserves_cut_zeros_but_bounds_void_and_fill():
    # A buried radial seam has zero min-piece distance and negative body distance.
    result=occupancy_metrics([0.,-40.e-6,40.e-6],[0.,-40.e-6,40.e-6],[-.009,-.0001,40.e-6])
    assert result['passed'] and result['cpu_internal_interface_zero_count']==1
    assert result['unambiguous_inside_count']==1 and result['unambiguous_outside_count']==1
    assert not occupancy_metrics([25.e-6],[0.],[-.009])['passed']
    assert not occupancy_metrics([-5.e-6],[40.e-6],[40.e-6])['passed']
    # The 20um near-boundary band reports wrong signs without claiming definite membership.
    ambiguous=occupancy_metrics([-5.e-6],[10.e-6],[10.e-6])
    assert ambiguous['passed'] and ambiguous['strict_native_wrong_sign_outside_count']==1
    assert ambiguous['membership_ambiguous_within_20um_of_body_or_cut_count']==1
    assert not occupancy_metrics([np.nan],[0.],[-.009])['passed']


def test_legacy_original_body_metric_is_not_called_unchanged():
    raw=np.array([[0.,1.,0.,0.],[0.,0.,1.,.0005]])
    result=legacy_union_metrics(raw,[-.0005,.0005],[[0.,0.,-1.],[0.,0.,1.]])
    assert not result['original_union_metric_passed']
    assert result['original_union_distance_failed_indices']==[0]
    assert result['used_for_sector_acceptance'] is False


def test_query_journal_keeps_completed_and_failed_requests(tmp_path):
    calls=[]
    def query(points):
        calls.append(1)
        if len(calls)==2:raise RuntimeError('native query failed')
        return np.c_[points,np.zeros(len(points))]
    journal=QueryJournal(query,tmp_path/'raw')
    journal(np.zeros((2,3)))
    with pytest.raises(RuntimeError):journal(np.ones((3,3)))
    assert np.load(tmp_path/'raw/0000_response.npz')['raw'].shape==(2,4)
    assert np.load(tmp_path/'raw/0001_request.npz')['points'].shape==(3,3)
    assert not load_json(tmp_path/'raw/0001_receipt.json')['completed']


def test_root_bracket_rejects_finite_distance_with_nan_gradient():
    def query(points):
        raw=plane(0.)(points);raw[:,1]=np.nan;return raw
    result=root_bracket(query,np.zeros((1,3)),np.array([[1.,0.,0.]]),increasing=True)
    assert not result['qualified'].any()


def test_derivative_calibration_rejects_nan_in_unused_gradient_column(tmp_path):
    from scripts.validation.gear_sector_sdf_checks import finite_layout
    directions=np.vstack([np.eye(3),-np.eye(3)])
    points=directions*1.001
    calls=[]
    def query(points):
        radius=np.linalg.norm(points,axis=1)
        raw=np.c_[points/radius[:,None],radius-1.]
        calls.append(1)
        if len(calls)>1:raw[:,0]=np.nan
        return raw
    journal=QueryJournal(query,tmp_path/'queries')
    with pytest.raises(ValueError,match='finite Nx4'):
        finite_layout(journal,points,np.full(6,.001),directions)
    assert np.isnan(np.load(tmp_path/'queries/0001_response.npz')['raw'][:,0]).all()


def test_transient_bad_raw_receipt_cannot_be_erased_by_later_recovery(tmp_path):
    count=[]
    def query(points):
        count.append(1)
        raw=plane(0.)(points)
        if len(count)==1:raw[:,2]=np.nan
        return raw
    journal=QueryJournal(query,tmp_path/'transient')
    journal(np.zeros((1,3)))
    assert np.isfinite(journal(np.zeros((1,3)))).all()
    assert not journal.all_receipts_finite
    assert journal.invalid_receipt_indices==[0]
    assert load_json(tmp_path/'transient/0000_receipt.json')['finite_Nx4'] is False
