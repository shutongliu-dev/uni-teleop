import hashlib
import json
from copy import deepcopy

import numpy as np
import pytest

from scripts.validation.laundry_cloth_reset_probe import diagnostic_inputs, mesh_diagnostic, parser, solver_readback


def test_probe_metrics_detect_local_initialization_impulse():
    rest = np.array([[0., 0., 0.], [.02, 0., 0.], [0., .02, 0.]])
    q = rest.copy()
    q[1, 2] = .05
    qd = np.zeros_like(q)
    qd[1, 2] = 50.
    value = mesh_diagnostic(q, qd, rest, [[0, 1, 2]], rest)
    assert value['max_edge_stretch_ratio'] == pytest.approx(np.sqrt(.02**2+.05**2)/.02)
    assert value['worst_edge_vertex_ids'] == [0, 1]
    assert value['max_speed_m_s'] == 50.
    assert value['max_displacement_from_usd_initial_m'] == .05


def test_probe_nonfinite_state_remains_reportable():
    q = np.zeros((3, 3))
    q[0, 0] = np.nan
    assert mesh_diagnostic(q, np.zeros_like(q), np.zeros_like(q), [[0, 1, 2]], q) == {'finite': False}


def test_probe_cli_requires_explicit_mode_and_release():
    with pytest.raises(SystemExit):
        parser().parse_args(['--output', '/tmp/not-a-run'])
    args = parser().parse_args(['--release', '/explicit/release', '--output', '/evidence', '--reset-mode', 'author-only'])
    assert args.steps == 100
    assert args.reset_mode == 'author-only'


def test_probe_rejects_ignored_explicit_filter():
    class Attribute:
        def Get(self):
            return None
    class Prim:
        def GetAttribute(self, name):
            return Attribute()
    class Stage:
        def GetPrimAtPath(self, path):
            return Prim()
    with pytest.raises(RuntimeError, match='Native solver profile differs'):
        solver_readback(Stage(), {'towel': {}}, {'cloth_self_collision_filter_distance_m': .003},
                        {'towel': {'path': '/World/Cloths/towel'}})


def test_diagnostic_override_is_explicit_and_does_not_mutate_release(tmp_path):
    path = tmp_path/'physics.json'
    path.write_text(json.dumps({'cloth_linear_damping': 20., 'cloth_bend_damping': .05}))
    scene = {'isaacsim': {'cloth_linear_damping': 2.},
             'cloths': [{'id': 'towel', 'isaacsim': {'cloth_linear_damping': 2.}}]}
    changed, evidence = diagnostic_inputs(scene, {}, physics_options=path)
    assert changed['cloths'][0]['isaacsim']['cloth_linear_damping'] == 20.
    assert scene['cloths'][0]['isaacsim']['cloth_linear_damping'] == 2.
    assert evidence['original_release_options_by_cloth']['towel']['cloth_linear_damping'] == 2.
    assert evidence['effective_options_by_cloth']['towel']['cloth_bend_damping'] == .05
    path.write_text(json.dumps({'cloth_thickness_m': .001}))
    with pytest.raises(ValueError, match='supported cloth physics'):
        diagnostic_inputs(scene, {}, physics_options=path)


def test_timestep_override_reaches_world_input_and_preserves_release_and_task(tmp_path):
    path = tmp_path/'physics.json'
    path.write_text(json.dumps({'physics_dt_s': .00025}))
    scene = {'physics': {'dt_s': .0005, 'control_dt_s': .02},
             'isaacsim': {'physics_dt_s': .0005, 'cloth_thickness_m': .003},
             'cloths': [{'id': 'towel', 'initial_vertices_m': [[0., 0., .9]],
                        'isaacsim': {'physics_dt_s': .0005}}]}
    task = {'evaluation': {'physics_dt_s': .0005, 'thresholds': {'stable_s': .5}}}
    before_scene, before_task = deepcopy(scene), deepcopy(task)
    changed, evidence = diagnostic_inputs(scene, task, physics_options=path, reset_mode='author-only')
    assert changed['physics']['dt_s'] == .00025
    assert changed['isaacsim']['physics_dt_s'] == .00025
    assert evidence['effective_options_by_cloth']['towel']['physics_dt_s'] == .00025
    assert evidence['physics_timestep_override']['source_release_dt_s'] == .0005
    assert evidence['physics_timestep_override']['effective_dt_s'] == .00025
    assert evidence['original_release_options_by_cloth']['towel']['physics_dt_s'] == .0005
    assert changed['cloths'][0]['initial_vertices_m'] == scene['cloths'][0]['initial_vertices_m']
    assert changed['isaacsim']['cloth_thickness_m'] == .003
    assert scene == before_scene and task == before_task
    assert 8000*changed['physics']['dt_s'] == 4000*scene['physics']['dt_s'] == 2.


@pytest.mark.parametrize('dt', [True, 0., -.00025, .0025, .0003])
def test_timestep_override_rejects_invalid_or_incompatible_step(tmp_path, dt):
    path = tmp_path/'physics.json'
    path.write_text(json.dumps({'physics_dt_s': dt}))
    scene = {'physics': {'dt_s': .0005, 'control_dt_s': .02}, 'isaacsim': {}, 'cloths': []}
    with pytest.raises(ValueError, match='Diagnostic physics_dt_s'):
        diagnostic_inputs(scene, {}, physics_options=path, reset_mode='author-only')
    assert scene['physics']['dt_s'] == .0005


def test_bound_terminal_geometry_rejects_tampered_archive(tmp_path, monkeypatch):
    from scenes.workrooms.laundry_room import release as release_module
    monkeypatch.setattr(release_module, 'validate_release', lambda *args: None)
    rest = [[0., 0., 0.], [.02, 0., 0.], [0., .02, 0.]]
    tri = [[0, 1, 2]]
    task = {'id': 'towel_folding', 'asset_metadata': {'cloths': [
        {'id': 'towel', 'rest_vertices_m': rest, 'triangles': tri}]}}
    source = tmp_path/'release'
    source.mkdir()
    (source/'task.json').write_text(json.dumps(task))
    (source/'release_manifest.json').write_text('{}')
    archive = tmp_path/'states.npz'
    q = np.asarray(rest)+[.8, -.55, .9015]
    np.savez(archive, towel_q=np.array([q, q+[0., 0., .01]]), towel_triangles=tri,
             native_step=[0, 1], time_s=[0., .0005])
    digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    report = {'task_id': 'towel_folding', 'audit_contract': 'laundry_native_stream.v1',
        'complete': True, 'native_steps': 1,
        'release': {'path': str(source), 'manifest_sha256': digest(source/'release_manifest.json')},
        'state_archive': {'path': str(archive), 'sha256': digest(archive)}}
    report_path = tmp_path/'native_report.json'
    report_path.write_text(json.dumps(report))
    scene = {'isaacsim': {}, 'cloths': [{'id': 'towel', 'initial_vertices_m': q.tolist()}]}
    changed, evidence = diagnostic_inputs(scene, task, initial_native_report=report_path)
    assert np.allclose(changed['cloths'][0]['initial_vertices_m'], q+[0., 0., .01])
    assert scene['cloths'][0]['initial_vertices_m'] == q.tolist()
    assert evidence['initial_geometry_override']['source_native_step'] == 1
    _, passive = diagnostic_inputs(scene, task, initial_native_report=report_path, reset_mode='author-only')
    assert passive['initial_geometry_override']['reset_mode'] == 'author-only'
    assert passive['initial_geometry_override']['velocity_initialization'].startswith('No velocity state write')
    assert 'zeroed' not in passive['initial_geometry_override']['velocity_initialization']
    archive.write_bytes(archive.read_bytes()+b'tampered')
    with pytest.raises(ValueError, match='archive hash differs'):
        diagnostic_inputs(scene, task, initial_native_report=report_path)
