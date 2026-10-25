from scripts.validation.run_mechanical_response_probe import reduce_contacts


def contact(**overrides):
    return dict(dict(actor0='tool', actor1='part', point_count=2,
        impulse_magnitude_sum_ns=1., impulse_vector_ns=[0., 0., -1.],
        min_separation_m=.0001, time_s=.0005), **overrides)


def test_contact_windows_preserve_zero_load_touching_and_positive_separation():
    rows = [contact(), contact(time_s=.001, impulse_magnitude_sum_ns=0.,
                              impulse_vector_ns=[0., 0., 0.])]
    result, = reduce_contacts(rows)
    assert result['active_samples'] == 1
    assert result['touching_samples'] == 2
    assert result['active_time_s'] == [.0005]
    assert result['min_separation_m'] == .0001
    assert result['impulse_vector_ns'] == [0., 0., -1.]


def test_contacts_keep_directed_vectors_and_count_substeps_not_manifolds():
    result = reduce_contacts([contact(), contact(), contact(actor0='part', actor1='tool',
        impulse_vector_ns=[0., 0., 1.], min_separation_m=-.0002)])
    assert len(result) == 2
    assert result[0]['active_samples'] == 1
    assert result[0]['normal_impulse_sum_ns'] == 2.
    assert result[0]['impulse_vector_ns'] == [0., 0., -2.]
    assert result[1]['impulse_vector_ns'] == [0., 0., 1.]
    assert result[1]['min_separation_m'] == -.0002


def test_active_times_distinguish_simultaneous_chain_from_same_window():
    a, b = reduce_contacts([contact(time_s=.0005),
        contact(actor0='part', actor1='fixture', time_s=.001)])
    assert not set(a['active_time_s']) & set(b['active_time_s'])
