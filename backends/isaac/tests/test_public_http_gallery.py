import numpy as np
import pytest

from scripts.validation.public_http_smoke import bind_gallery_native_samples


def test_gallery_binds_observation_time_to_real_native_index_and_archive(tmp_path):
    path=tmp_path/'states.npz'
    np.savez(path,time_s=[0.,.02,.04],native_world_time_s=[.004,.024,.044],native_physics_step_index=[2,12,22])
    snapshots=[{'sim_time_s':0.},{'sim_time_s':.04000000000000001}]
    bind_gallery_native_samples(snapshots,path)
    assert snapshots[0]['native_sample']['sample_index']==0
    assert snapshots[1]['native_sample']['native_physics_step_index']==22
    assert snapshots[1]['native_sample']['native_world_time_s']==.044
    assert len(snapshots[0]['native_sample']['archive_sha256'])==64
    with pytest.raises(RuntimeError,match='unique recorded native sample'):
        bind_gallery_native_samples([{'sim_time_s':.03}],path)
