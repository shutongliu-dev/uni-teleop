from types import SimpleNamespace
import numpy as np
import pytest
from scripts.validation.kitchen_contact_snapshot import capture_contact_snapshot


def fixture_env():
    view=SimpleNamespace(max_contact_data_count=8)
    view.get_contact_data=lambda dt: (np.array([[.7],[0.],[0.]]),np.zeros((3,3)),np.array([[0.,0.,1.]]*3),np.zeros((3,1)),np.array([[1]]),np.array([[0]]))
    view.get_net_contact_forces=lambda dt: np.array([[0.,0.,.1]])
    view.get_contact_force_matrix=lambda dt: np.array([[[0.,0.,.7]]])
    return SimpleNamespace(_rigid_contacts=view,dt=.0005,time_s=1.,_contact_sensor_paths=['body'],_contact_filter_paths=[['support']],world=SimpleNamespace(current_time_step_index=2000,current_time=1.),capture_state=lambda:{'q':np.array([1.])})


def test_raw_api_disagreement_is_preserved_without_rescaling(tmp_path):
    env=fixture_env();p=tmp_path/'raw.npz';r=capture_contact_snapshot(env,p)
    with np.load(p) as data:
        assert data['scalar_impulses'][0,0]==.7 and data['net_impulses'][0,2]==.1
        assert data['pair_impulses'][0,0,2]==.7
        assert data['scalar_impulses'].shape==(1,1)
    assert r['state_sha256_before']==r['state_sha256_after']


def test_snapshot_rejects_native_advance(tmp_path):
    env=fixture_env()
    def net(dt):
        env.world.current_time_step_index+=1
        return np.array([[0.,0.,.1]])
    env._rigid_contacts.get_net_contact_forces=net
    with pytest.raises(RuntimeError,match='physical state or time'):
        capture_contact_snapshot(env,tmp_path/'raw.npz')
