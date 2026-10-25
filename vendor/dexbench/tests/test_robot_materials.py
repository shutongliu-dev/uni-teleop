"""Material intent must survive USD binding and actual body-local PhysX shapes."""
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from bench.paths import load_json
from scripts.diagnostics.probe_robot_materials import (
    BIMANUAL_COMPLIANCE, bimanual_source, compliant_readback, enabled_spring_values, probe_scene,
)
from bench.simulation.isaacsim_backend import robot_materials as materials


REQUEST = {'values': {'pad_friction': .8, 'shell_friction': .2}, 'pad_compliance': None,
           'binding': {'mode': 'collider_instance_v1'},
           'resolved_collider_materials': {'/World/PhysicsMaterials/robot_pad': 1,
                                           '/World/PhysicsMaterials/robot_shell': 1}}


def link(name, friction, shape_count=1):
    path = '/World/Robot/'+name
    kind = materials.material_kind(name)
    return {'link_name': name, 'link_path': path, 'kind': kind,
            'native_view_count': 1, 'native_shape_count': shape_count,
            'usd_colliders': [{'link_name': name, 'link_path': path, 'kind': kind,
                               'collider_path': path+'/Collision',
                               'material_path': '/World/PhysicsMaterials/'+kind}],
            'shapes': [{'shape_index': i, 'material_properties': [friction, friction, 0.]} for i in range(shape_count)]}


def fixture():
    links = [link('right_index_elastomer', .8), link('right_index_DP', .2)]
    return links, [shape['material_properties'] for item in links for shape in item['shapes']]


def test_legacy_equal_friction_is_unchanged_but_pad_only_request_fails():
    assert materials.material_options({'robot_contact_materials': {'pad_friction': .5, 'shell_friction': .5}}) == {
        'binding': None, 'pad_compliance': None}
    with pytest.raises(ValueError, match='legacy link binding'):
        materials.material_options({'robot_contact_materials': REQUEST['values']})
    assert materials.material_options({'robot_contact_materials': REQUEST['values'],
                                      'robot_material_binding': 'collider_instance_v1'})['pad_compliance'] is None


@pytest.mark.parametrize('invalid', [float('nan'), float('inf'), -.1, True, '0.5'])
def test_friction_must_be_physical_finite_number(invalid):
    with pytest.raises(ValueError, match='finite non-negative'):
        materials.material_options({'robot_contact_materials': {'pad_friction': invalid, 'shell_friction': .5},
                                    'robot_material_binding': 'collider_instance_v1'})


def test_actual_link_shape_mapping_passes_and_all_shapes_are_checked():
    links, aggregate = fixture()
    record = materials.check_link_materials(links, REQUEST, aggregate)
    assert record['passed'] and record['pad_shapes'] == record['shell_shapes'] == 1
    assert record['compliant_shapes'] == 0 and record['links'][0]['link_name'] == 'right_index_elastomer'
    # One USD mesh can cook into several native shapes. Every body-local shape
    # is checked without pretending the mesh traversal defines shape order.
    links[0] = link('right_index_elastomer', .8, shape_count=2)
    assert materials.check_link_materials(links, REQUEST, aggregate+[aggregate[0]])['pad_shapes'] == 2
    links[0]['shapes'][1]['material_properties'] = [.2, .2, 0.]
    with pytest.raises(RuntimeError, match='shape 1'):
        materials.check_link_materials(links, REQUEST, aggregate+[aggregate[1]])


def test_swapped_pad_shell_values_fail_even_with_identical_global_counts():
    links, aggregate = fixture()
    links[0]['shapes'][0]['material_properties'], links[1]['shapes'][0]['material_properties'] = aggregate[::-1]
    with pytest.raises(RuntimeError, match='Native material mismatch'):
        materials.check_link_materials(links, REQUEST, aggregate)


def test_equal_native_values_cannot_hide_wrong_usd_binding():
    links = [link('right_index_elastomer', .5), link('right_index_DP', .5)]
    request = deepcopy(REQUEST)
    request['values'] = {'pad_friction': .5, 'shell_friction': .5}
    links[0]['usd_colliders'][0]['material_path'] = '/World/PhysicsMaterials/robot_shell'
    with pytest.raises(RuntimeError, match='USD collider material resolution'):
        materials.check_link_materials(links, request, [[.5, .5, 0.]]*2)


def test_rigid_request_rejects_unintended_compliance_and_nonfinite_readback():
    for restitution in (-5000., -1e-9, float('nan')):
        links, aggregate = fixture()
        links[0]['shapes'][0]['material_properties'][2] = restitution
        with pytest.raises(RuntimeError, match='Native material mismatch'):
            materials.check_link_materials(links, REQUEST, aggregate)


def test_shape_padding_or_missing_coverage_is_rejected():
    links, aggregate = fixture()
    links[0]['native_view_count'] = 2
    with pytest.raises(RuntimeError, match='single-body shape count'):
        materials.check_link_materials(links, REQUEST, aggregate)
    links, aggregate = fixture()
    with pytest.raises(RuntimeError, match='cover the articulation'):
        materials.check_link_materials(links, REQUEST, aggregate+[[0., 0., 0.]])


def test_runtime_reader_uses_exact_body_identity_and_never_assumes_articulation_order(monkeypatch):
    links, aggregate = fixture()
    monkeypatch.setattr(materials, 'collider_material_inventory', lambda *_: [item['usd_colliders'][0] for item in links])
    robot = SimpleNamespace(get_material_properties=lambda: np.asarray([aggregate[::-1]]),
                            link_paths=[[item['link_path'] for item in links]])
    views = {item['link_path']: SimpleNamespace(count=1, max_shapes=1, prim_paths=[item['link_path']],
              get_material_properties=lambda item=item: np.array([[s['material_properties'] for s in item['shapes']]]))
             for item in links}
    sim = SimpleNamespace(create_rigid_body_view=lambda path: views[path])
    info = {'robot_path': '/World/Robot', 'robot_contact_materials': REQUEST}
    record = materials.runtime_material_check(None, sim, robot, info)
    assert record['passed'] and record['robot_shapes'] == 2
    robot.link_paths = robot.link_paths[0]
    assert materials.runtime_material_check(None, sim, robot, info) == record
    views[links[0]['link_path']].prim_paths = ['/World/WrongLink']
    with pytest.raises(RuntimeError, match='one exact link'):
        materials.runtime_material_check(None, sim, robot, info)


def test_diagnostic_override_does_not_modify_source_or_enable_compliance():
    source = {'isaacsim': {'physics_dt_s': .0005,
                          'robot_contact_materials': {'pad_friction': .5, 'shell_friction': .5}}}
    before = deepcopy(source)
    probe = probe_scene(source, 'differentiated')
    assert source == before
    assert probe['isaacsim']['robot_contact_materials'] == REQUEST['values']
    assert probe['isaacsim'].get('robot_pad_compliance') is None
    assert probe_scene(source, 'default')['isaacsim']['robot_contact_materials'] == before['isaacsim']['robot_contact_materials']


def test_probe_leaves_failed_receipt_when_native_constructor_terminates_python(tmp_path):
    # Kit fast shutdown bypasses Python exception/finally handlers. Reproduce
    # that process boundary without starting Isaac or a GPU.
    script = """
import os, pathlib, sys
from bench.isaacsim.release import dump
from scripts.diagnostics.probe_robot_materials import record_incomplete_initialization
output = pathlib.Path(sys.argv[1])
plan = {'mode': 'differentiated', 'scope': 'test native construction boundary'}
dump(output/'plan.json', plan)
record_incomplete_initialization(output, plan)
os._exit(1)
"""
    child = subprocess.run([sys.executable, '-c', script, str(tmp_path)],
                           cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True)
    assert child.returncode == 1, child.stderr
    report = load_json(tmp_path/'report.json')
    assert report['passed'] is False and report['provisional'] is True
    assert report['status'] == 'native_initialization_completion_not_observed'
    assert report['native_shutdown_verified'] is False and report['error'] is None


def test_bimanual_probe_preserves_pinned_release_scene_and_original_compliance():
    source, evidence = bimanual_source()
    before = deepcopy(source)
    spec = probe_scene(source, 'bimanual_release')
    assert spec == source == before and spec is not source
    assert spec['isaacsim']['robot_pad_compliance'] == BIMANUAL_COMPLIANCE
    assert spec['robot']['hand_id'] == 'dual_fr3_sharpa_w01'
    assert evidence['scene_modified'] is False and evidence['task_id'] == '05_bimanual_box_lift'
    source['isaacsim']['robot_pad_compliance']['damping'] = 0.
    with pytest.raises(ValueError, match='preserve the pinned release'):
        probe_scene(source, 'bimanual_release')


def test_bimanual_compliance_readback_records_per_link_values_and_detects_wrong_damping():
    links, _ = fixture()
    links[0]['shapes'][0]['material_properties'][2] = -20000.
    views = {}
    for item in links:
        values = np.array([[[20000., 40.]]] if item['kind'] == 'robot_pad' else [[[0., 0.]]])
        views[item['link_path']] = SimpleNamespace(count=1, max_shapes=1, prim_paths=[item['link_path']],
            get_compliant_material_properties=lambda values=values: (values, np.zeros((1, 1, 2), dtype=int)))
    sim = SimpleNamespace(create_rigid_body_view=lambda path: views[path])
    record = compliant_readback(sim, {'links': links}, BIMANUAL_COMPLIANCE)
    assert record['passed'] and record['links'][0]['shapes'][0]['damping'] == 40.
    changed_unused = deepcopy(record)
    changed_unused['links'][1]['shapes'][0]['stiffness'] = '+Infinity'
    changed_unused['links'][1]['raw_values']['data_hex'] = 'unused-shell-values'
    assert enabled_spring_values(changed_unused) == enabled_spring_values(record)
    changed_unused['links'][0]['shapes'][0]['damping'] = 20.
    assert enabled_spring_values(changed_unused) != enabled_spring_values(record)
    views[links[0]['link_path']].get_compliant_material_properties = lambda: (
        np.array([[[20000., 20.]]]), np.zeros((1, 1, 2), dtype=int))
    assert not compliant_readback(sim, {'links': links}, BIMANUAL_COMPLIANCE)['passed']


def test_compliant_getter_nonfinite_values_remain_exact_audit_evidence():
    item = link('fr3_left_link0', .5)
    raw = np.array([[[float('inf'), float('nan')]]], dtype=np.float32)
    view = SimpleNamespace(count=1, max_shapes=1, prim_paths=[item['link_path']],
        get_compliant_material_properties=lambda: (raw, np.zeros((1, 1, 2), dtype=np.uint8)))
    result = compliant_readback(SimpleNamespace(create_rigid_body_view=lambda path: view), {'links': [item]}, BIMANUAL_COMPLIANCE)
    assert result['passed']
    shape = result['links'][0]['shapes'][0]
    assert shape['stiffness'] == '+Infinity' and shape['damping'] == 'NaN'
    assert not shape['spring_enabled_by_native_restitution']
    evidence = result['links'][0]['raw_values']
    assert bytes.fromhex(evidence['data_hex']) == raw.tobytes()
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize('values,modes', [([float('nan'), 40.], [0, 0]),
                                        ([20000., float('inf')], [0, 0]),
                                        ([20000., 40.], [0, 1])])
def test_enabled_pad_spring_rejects_nonfinite_or_wrong_combine_mode(values, modes):
    item = link('left_index_elastomer', .5)
    item['shapes'][0]['material_properties'][2] = -20000.
    view = SimpleNamespace(count=1, max_shapes=1, prim_paths=[item['link_path']],
        get_compliant_material_properties=lambda: (np.array([[values]], dtype=np.float32),
                                                   np.array([[modes]], dtype=np.uint8)))
    record = compliant_readback(SimpleNamespace(create_rigid_body_view=lambda path: view),
                                {'links': [item]}, BIMANUAL_COMPLIANCE)
    assert not record['passed']
