import numpy as np
import pytest
from scripts.validation.elastic_sponge_metrics import composite,freeflight,support


def free_raw():
    n=401;m=np.array([.00792,.07128]);dt=float(np.float32(.0005));t=np.arange(n)*dt
    f=np.zeros((n,3),np.float32);f[101:201,0]=.4
    v=np.zeros((n,2,6));v[:,:,2]=(-9.81*t)[:,None];v[:,:,0]=(np.cumsum(f[:,0].astype(float))*dt/m.sum())[:,None]
    q=np.zeros((n,2,7));q[:,:,6]=1.;q[:,:,2]=1.2;q[:,1,1]=.02
    q[1:,:,:3]+=np.cumsum(.5*(v[:-1,:,:3]+v[1:,:,:3])*dt,axis=0)
    pos=np.zeros((n,3));pos[1:]=q[:-1,0,:3].astype(np.float32)
    return {'time_s':t,'native_step':np.arange(n),'body_q':q,'body_qd':v,'force_world_n':f,'force_position_world_m':pos},m,np.zeros((2,3))


def test_true_member_momentum_passes_and_external_impulse_loss_fails():
    raw,m,c=free_raw();checks,measure,_=freeflight(raw,m,c,0)
    assert all(checks.values()) and measure['pulse_relative_x_error']<1e-12
    raw['body_qd'][:,:,0]*=.6
    checks,_,_=freeflight(raw,m,c,0)
    assert not checks['pulse_momentum_within_15_percent']


def test_all_member_inventory_required_and_force_point_is_native_com():
    raw,m,c=free_raw()
    with pytest.raises(ValueError,match='inventory'):composite(raw,m[:1],c[:1])
    raw['force_position_world_m'][150,1]+=.01
    with pytest.raises(ValueError,match='native padCOM'):freeflight(raw,m,c,0)


def support_raw(mass,members):
    n=1201;t=np.arange(n)*float(np.float32(.0005));q=np.tile([.6,-.8,.89-(.2+mass)*9.81/2000,0,0,0,1.],(n,1))
    b=np.zeros((n,members,7));b[:,:,6]=1.;b[:,:,2]=.93
    return {'time_s':t,'native_step':np.arange(n),'plate_q':q,'plate_qd':np.zeros((n,6)),'body_q':b,'body_qd':np.zeros((n,members,6))}


def test_calibrated_plate_uses_all_native_mass_and_quiet_members():
    c,e,_=support(support_raw(0.,0),[],np.empty((0,3)))
    assert all(c.values());empty={'passed':True,'measurements':e}
    masses=np.full(28,.0792/28);raw=support_raw(.0792,28)
    checks,m,_=support(raw,masses,np.zeros((28,3)),empty)
    assert all(checks.values()) and np.isclose(m['tare_full_balance_n'],.776952)
    raw['body_qd'][-1,-1,0]=1.
    checks,_,_=support(raw,masses,np.zeros((28,3)),empty)
    assert not checks['all_native_body_linear_quiet']
    raw=support_raw(.13,28)
    checks,_,_=support(raw,masses,np.zeros((28,3)),empty)
    assert not checks['tare_weight_within_15_percent']
