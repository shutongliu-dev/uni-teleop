import numpy as np

from scripts.validation.laundry_material_audit import freefall_dynamics


def test_detects_declared_damped_gravity_before_contact_only():
    dt=.002;t=np.arange(101)*dt;damping=2.;v=np.zeros(len(t))
    for i in range(100):v[i+1]=(1-damping*dt)*(v[i]-9.81*dt)
    z=.96+np.cumsum(v)*dt
    # Force contact-like velocity change at 80 ms; excluded by height envelope.
    minimum=np.where(t<.08,.95,.902)
    v[t>=.08]=.03
    result=freefall_dynamics(t,minimum,v,z,.5*v*v,table_z=.9,thickness=.004,contact_offset=.005,damping=damping)
    assert result['first_contact_envelope_time_s']==.08
    assert result['damped_gravity_step_velocity_error_max_m_s']<1.e-12
    assert result['freefall_window_s'][1]<.08
    assert result['post_contact_upward_com_velocity_max_m_s']==.03


def test_reports_no_freefall_for_supported_cloth():
    t=np.arange(101)*.002
    result=freefall_dynamics(t,np.full(101,.902),np.zeros(101),np.full(101,.902),np.zeros(101),table_z=.9,thickness=.004,contact_offset=.005,damping=2.)
    assert result['sample_count']==0
    assert 'fitted_com_acceleration_m_s2' not in result
