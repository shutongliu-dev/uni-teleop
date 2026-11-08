from copy import deepcopy
import builtins
import json
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.validation import laundry_captured_towel_tasks as towel
from scripts.validation import laundry_captured_garment_fold as runner


@pytest.fixture
def material():
    x, y = np.meshgrid(np.linspace(-.3, .3, 61), np.linspace(-.2, .2, 41))
    rest = np.column_stack([x.ravel(), y.ravel(), np.zeros(x.size)])
    task = {'id': 'towel_flattening', 'asset_metadata': {
        'physical_thickness_m': .004, 'table_top_z_m': .9,
        'reset_profile': 'towel_partial_edge_fold_v1',
        'cloths': [{'id': 'towel', 'rest_vertices_m': rest.tolist()}]}}
    return task, {'cloths': [{'id': 'towel'}]}, rest + [.56, -.53, .902]


def options(indices, order=None, phase='0'):
    setting = {'edge_indices': indices}
    if order is not None:
        setting['capture_order'] = order
    return {'phases': {phase: setting}}


def test_explicit_left_is_exact_original_second_material_pick_and_keeps_alignment(material, monkeypatch):
    task, scene, q = material
    monkeypatch.setattr(towel, 'pinch_plan', lambda *a, **k: {})
    defaults = {'phases': {'0': {'capture_order': ['left', 'right']}}}
    selected = options([1], ['left'])
    before = deepcopy((task, scene, selected)); original_q = q.copy()
    legacy = towel.prepare_phase(None, task, scene, q, 0, defaults)
    phase = towel.prepare_phase(None, task, scene, q, 0, selected)
    assert phase['edges'] == [legacy['edges'][1]]
    assert phase['edges'][0]['seed_vertex_ids'] == [2440]
    assert phase['selected_edge_indices'] == [1]
    assert not phase['all_phase_edges_selected']
    assert phase['partial_boundary_use']['partial']
    assert runner.capture_batches(phase, selected) == [['left']]
    assert len(towel.phase_edges(task)) == 2
    align = towel.prepare_phase(None, task, scene, q, 1, selected)
    assert [e['seed_vertex_ids'] for e in align['edges']] == [[60], [2500]]
    assert 'partial_boundary_use' not in align
    assert 'all_phase_edges_selected' not in legacy
    assert runner.planned_duration(2, selected, 2.) == runner.planned_duration(2, defaults, 2.) - 6.5
    assert (task, scene, selected) == before
    np.testing.assert_array_equal(q, original_q)


def test_rebind_stays_with_selected_boundary_and_original_side_mapping(material, monkeypatch):
    task, scene, q = material
    monkeypatch.setattr(towel, 'pinch_plan', lambda *a, **k: {})
    opt = options([1], ['left'])
    got = towel.prepare_phase(None, task, scene, q, 0, opt, only_sides=['left'])
    assert [e['side'] for e in got['edges']] == ['left']
    with pytest.raises(ValueError, match='unselected'):
        towel.prepare_phase(None, task, scene, q, 0, opt, only_sides=['right'])
    opt['phases']['0'].update(sides=['left', 'right'], capture_order=['right'])
    got = towel.prepare_phase(None, task, scene, q, 0, opt)
    assert got['edges'][0]['seed_vertex_ids'] == [2440]
    assert got['edges'][0]['side'] == 'right'


@pytest.mark.parametrize('indices', [[], [True], [1.], [2], [-1], [1, 1], None, '1'])
def test_bad_selection_rejected_before_geometry(material, monkeypatch, indices):
    task, scene, q = material
    monkeypatch.setattr(towel, 'towel_picks', lambda *a, **k: pytest.fail('No planning before validation'))
    with pytest.raises(ValueError, match='edge_indices'):
        towel.prepare_phase(None, task, scene, q, 0, options(indices))


@pytest.mark.parametrize('name,phase,order', [
    ('towel_folding', '0', ['left']), ('towel_stacking', '0', ['left']),
    ('towel_flattening', '1', ['left']), ('towel_flattening', '2', ['left']),
    ('towel_flattening', '0', ['right']), ('towel_flattening', '0', ['left', 'right'])])
def test_subset_scope_and_capture_order_are_explicit(name, phase, order):
    with pytest.raises(ValueError):
        towel.validate_options({'id': name}, options([1], order, phase))


def test_native_rejects_invalid_later_boundary_selection_before_simulator(tmp_path, monkeypatch, material):
    import scripts.maintenance.freeze_project_sources as freeze
    import bench.isaacsim.release as release_module
    task, scene, _ = material
    scene.update(named_cloth_initialization='author-only', robot={'hand_id': 'dual_fr3_sharpa_w01'})
    release = tmp_path / 'release'; release.mkdir()
    for name, value in [('task.json', task), ('scene.json', scene), ('initial_state.json', {})]:
        (release / name).write_text(json.dumps(value))
    option_file = tmp_path / 'options.json'
    option_file.write_text(json.dumps(options([1], ['left'], '1')))
    monkeypatch.setattr(freeze, 'verify_if_snapshot', lambda *a, **k: {'verified': True})
    monkeypatch.setattr(release_module, 'validate', lambda _: {'task_id': task['id']})
    original_import = builtins.__import__
    def guarded(name, *a, **k):
        if name == 'bench.simulation.isaacsim_backend.env':
            pytest.fail('Invalid selection must fail before native startup')
        return original_import(name, *a, **k)
    monkeypatch.setattr(builtins, '__import__', guarded)
    out = tmp_path / 'native'
    with pytest.raises(ValueError, match='initial flap unfolding'):
        runner.native(SimpleNamespace(release=release, options=option_file, output=out, max_phases=2), task_adapter=towel)
    assert not out.exists()
