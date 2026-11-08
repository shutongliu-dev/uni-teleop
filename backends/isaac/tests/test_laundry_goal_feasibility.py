"""Input isolation and raw-evidence failure tests, not native qualification."""
from copy import deepcopy
import json
import numpy as np
import pytest

from scripts.validation.laundry_goal_feasibility import (
    PROFILE, checked, derive_inputs, load_arrays, load_candidate, passive_run_checks, pin, strain, write_json)


def inputs():
    rest = [[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]]
    flat = [[0., 0., .91], [1., 0., .91], [0., 1., .91]]
    goal = [[0., 0., .91], [0., 0., 1.91], [0., 1., .91]]
    meta = {'id': 'towel', 'rest_vertices_m': rest, 'triangles': [[0, 1, 2]],
            'folded_reference_vertices_m': goal}
    task = {'id': 'shirt_folding', 'evaluation': {'rule_profile': PROFILE, 'physics_dt_s': .002},
            'asset_metadata': {'cloths': [meta]}}
    scene = {'named_cloth_initialization': 'author-only', 'robot': {'hand_id': 'original'},
        'cloth': {'enabled': False}, 'cloths': [{**deepcopy(meta), 'initial_vertices_m': flat,
            'origin_m': [0., 0., 0.], 'quaternion_wxyz': [1., 0., 0., 0.],
            'mass_kg': .24, 'friction': .45, 'isaacsim': {'cloth_thickness_m': .0016}}],
        'physics': {'dt_s': .002}, 'isaacsim': {'cloth_surface_bend_stiffness': .01}}
    initial = {'cloths_vertices_m': {'towel': deepcopy(flat)}, 'joint_positions_rad': {'joint': .25}}
    return task, scene, initial


def test_goal_only_changes_authored_positions_and_preserves_original_inputs():
    task, scene, initial = inputs()
    before = deepcopy((task, scene, initial))
    candidate, reset, changes = derive_inputs(task, scene, initial, 'analytic-goal')
    assert (task, scene, initial) == before
    assert len(changes) == 2
    assert candidate['cloths'][0]['initial_vertices_m'] == task['asset_metadata']['cloths'][0]['folded_reference_vertices_m']
    assert reset['cloths_vertices_m']['towel'] == candidate['cloths'][0]['initial_vertices_m']
    candidate['cloths'][0]['initial_vertices_m'] = scene['cloths'][0]['initial_vertices_m']
    reset['cloths_vertices_m'] = initial['cloths_vertices_m']
    assert candidate == scene and reset == initial


def test_unfolded_is_exact_control_and_surface_override_is_explicit_only():
    task, scene, initial = inputs()
    candidate, reset, changes = derive_inputs(task, scene, initial, 'unfolded')
    assert candidate == scene and reset == initial and changes == []
    surface = {'schema': 'workroom-surface-materials-v1', 'surfaces': {'table': {
        'static_friction': .4, 'dynamic_friction': .4, 'restitution': 0.,
        'friction_combine': 'average', 'restitution_combine': 'average',
        'parameter_source': 'explicit uncalibrated test prior'}}}
    candidate, reset, changes = derive_inputs(task, scene, initial, 'unfolded', surface)
    assert candidate['isaacsim'].pop('workroom_surface_materials') == surface
    assert candidate == scene and reset == initial
    assert changes == ['scene.isaacsim.workroom_surface_materials']


def test_explicit_geometry_candidate_never_rewrites_original_scoring_reference():
    task, scene, initial = inputs(); old = deepcopy((task, scene, initial))
    points = np.asarray(task['asset_metadata']['cloths'][0]['folded_reference_vertices_m'])
    points[:, 0] += .003
    candidate, reset, changes = derive_inputs(task, scene, initial, 'geometry-candidate', candidate_vertices=points)
    assert (task, scene, initial) == old and len(changes) == 2
    assert candidate['cloths'][0]['initial_vertices_m'] == points.tolist()
    assert reset['cloths_vertices_m']['towel'] == points.tolist()
    with pytest.raises(ValueError, match='required only'):
        derive_inputs(task, scene, initial, 'unfolded', candidate_vertices=points)
    with pytest.raises(ValueError, match='required only'):
        derive_inputs(task, scene, initial, 'geometry-candidate')
    with pytest.raises(ValueError, match='identical topology'):
        derive_inputs(task, scene, initial, 'geometry-candidate', candidate_vertices=points[:2])
    points[0, 0] = np.nan
    with pytest.raises(ValueError, match='Finite original target'):
        derive_inputs(task, scene, initial, 'geometry-candidate', candidate_vertices=points)


def test_candidate_requires_matching_task_generator_and_exact_saved_vertices(tmp_path):
    task_path = tmp_path/'task.json'; task_path.write_text('{}')
    generator = tmp_path/'generator.py'; generator.write_text('# static diagnostic\n')
    vertices = tmp_path/'candidate.npy'; np.save(vertices, np.eye(3))
    report = {'schema': 'laundry_legal_goal_reset_geometry_research.v1',
        'native_simulation_executed': False, 'task_success': False,
        'original_task': pin(task_path), 'script': pin(generator), 'tool': pin(generator),
        'candidates': [{'name': 'test', 'reset_coordinates': pin(vertices)}]}
    evidence = tmp_path/'evidence.json'; evidence.write_text(json.dumps(report))
    points, provenance = load_candidate(vertices, evidence, task_path)
    assert np.array_equal(points, np.eye(3)) and provenance['candidate_name'] == 'test'
    other_task = tmp_path/'other.json'; other_task.write_text('{"different":true}')
    with pytest.raises(ValueError, match='different original task'):
        load_candidate(vertices, evidence, other_task)
    generator.write_text('# changed\n')
    with pytest.raises(ValueError, match='Changed pinned input'):
        load_candidate(vertices, evidence, task_path)
    generator.write_text('# static diagnostic\n')
    np.save(vertices, np.ones((3, 3)))
    with pytest.raises(ValueError, match='exactly one pinned'):
        load_candidate(vertices, evidence, task_path)


@pytest.mark.parametrize('change,match', [
    (lambda t, s, i: s['cloths'][0]['triangles'].append([0, 2, 1]), 'rest topology'),
    (lambda t, s, i: s.update(named_cloth_initialization='setters'), 'authored'),
    (lambda t, s, i: i['cloths_vertices_m']['towel'][0].__setitem__(2, .8), 'positions differ'),
    (lambda t, s, i: s['robot'].update(enabled=False), 'installed robot'),
    (lambda t, s, i: t['evaluation'].update(rule_profile='new-rule'), 'v4 task'),
])
def test_unbound_or_changed_physical_inputs_fail_closed(change, match):
    task, scene, initial = inputs()
    change(task, scene, initial)
    with pytest.raises(ValueError, match=match):
        derive_inputs(task, scene, initial, 'analytic-goal')


def test_geometric_strain_is_rigid_invariant_and_detects_compression():
    rest = np.asarray([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]])
    tri = np.array([[0, 1, 2]])
    rotated = np.asarray([[1., 2., 3.], [1., 2., 4.], [1., 3., 3.]])
    assert strain(rest, rotated, tri)['principal_deviation_rest_area_p95'] == pytest.approx(0.)
    compressed = rest.copy(); compressed[:, 0] *= .5
    got = strain(rest, compressed, tri)
    assert got['principal_deviation_rest_area_p95'] == pytest.approx(.5)
    assert got['rest_area_fraction_above_5pct_strain'] == 1.
    assert got['triangle_area_ratio_min'] == pytest.approx(.5)


def test_changed_pins_and_existing_output_are_rejected(tmp_path):
    target = tmp_path/'record.json'; write_json(target, {'x': 1})
    record = pin(target)
    with pytest.raises(FileExistsError):
        write_json(target, {'x': 2})
    target.write_text('{}')
    with pytest.raises(ValueError, match='Changed pinned input'):
        checked(record)


def test_native_missing_steps_and_shape_truncation_are_rejected(tmp_path):
    files = {}
    for key, value in [('native_step', np.array([0, 1, 3])), ('time_s', np.array([0., .002, .006]))]:
        path = tmp_path/(key+'.npy'); np.save(path, value)
        files[key] = {**pin(path), 'shape': list(value.shape)}
    native = {'states': {'valid_samples': 3, 'files': files}}
    with pytest.raises(ValueError, match='Missing or inconsistent'):
        load_arrays(native, tuple(files))
    native['states']['valid_samples'] = 4
    with pytest.raises(ValueError, match='Truncated'):
        load_arrays(native, tuple(files))


@pytest.mark.parametrize('failure', ['incomplete', 'padded', 'attachment', 'write', 'controlled'])
def test_terminal_shape_cannot_qualify_incomplete_or_assisted_passive_probe(failure):
    native = {'complete': True, 'states': {'complete': True, 'allocated_samples': 251},
        'native_steps': 250, 'mode': 'settle', 'actual_attachment_prims': [],
        'attachments_created': 0, 'cloth_position_velocity_writes': 0,
        'task_success_claimed': False}
    arrays = {'time_s': np.arange(251)*.002, 'control_mask': np.zeros((251, 3), dtype=np.uint8)}
    assert all(passive_run_checks(native, arrays).values())
    # Terminal geometry is deliberately irrelevant to this provenance gate.
    if failure == 'incomplete':
        native['complete'] = False
    elif failure == 'padded':
        native['states']['allocated_samples'] = 1501
    elif failure == 'attachment':
        native['actual_attachment_prims'] = ['/World/UnauthorizedAttachment']
    elif failure == 'write':
        native['cloth_position_velocity_writes'] = 1
    else:
        arrays['control_mask'][0, 0] = 1
    assert not all(passive_run_checks(native, arrays).values())
