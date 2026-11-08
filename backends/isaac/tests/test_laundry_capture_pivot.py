"""A full measured patch may hold at one shared vertex without moving its arc."""
import copy
import hashlib
import sys
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial import ConvexHull

from scripts.validation.laundry_capture_gate import CaptureGate
from scripts.validation.laundry_capture_pivot import select_captured_pivot
from scripts.validation.laundry_contact_assist import CapturedPatchAssist, capture_constraint_vertices


@pytest.fixture
def capture():
    q = np.array([[-.005, 0, .9], [0, -.005, .9], [0, .005, .9], [.005, 0, .9]])
    tri = np.array([[0, 1, 2], [1, 3, 2]])
    bodies = np.array([[0, 0, .9, 0, 0, 0, 1], [-.006, 0, .9, 0, 0, 0, 1],
                       [.006, 0, .9, 0, 0, 0, 1]], float)
    names = ['right_hand_wrist', 'right_thumb_fingertip', 'right_index_fingertip']
    vertices = np.array([[x, y, z] for x in (-.002, .002) for y in (-.002, .002)
                         for z in (-.002, .002)])
    hull = ConvexHull(vertices)
    geometry = SimpleNamespace(body_names=names, surface_pair=('DP', 'DP'), _variants=[], pads={
        'right': {f: {'body': i, 'body_name': names[i], 'inner': vertices,
                      'inner_planes': hull.equations, 'faces': hull.simplices}
                  for f, i in (('thumb', 1), ('index', 2))}})
    witness = {'thumb': {'triangle_id': 0, 'point_m': [-.002, 0, .9],
                         'surface_point_m': [-.004, 0, .9]},
               'index': {'triangle_id': 1, 'point_m': [.002, 0, .9],
                         'surface_point_m': [.004, 0, .9]}, 'opposing_normal_dot': -1.}
    diagnostic = {'side': 'right', 'distal_group_severe_penetration': False,
        'contact_force_measured': False, 'surface_pair': {'thumb': 'DP', 'index': 'DP'},
        'multilayer_diagnostic': {'plausible': True, 'paired_witnesses': [witness]},
        'state_identity': {'body_q_float64_sha256': hashlib.sha256(bodies.astype('<f8').tobytes()).hexdigest(),
                           'cloth_q_float64_sha256': hashlib.sha256(q.astype('<f8').tobytes()).hexdigest()}}
    gate = CaptureGate('right', tri, q, names, ['right_thumb_IP', 'right_index_DIP'],
                       profile='adjacent_patch_v3')
    for time in np.linspace(0, .1, 6):
        result = gate.observe(float(time), diagnostic, bodies, [0, 0], [0, 0], [0, 0], .06, cloth_q=q)
    assert result['capture_candidate']
    return SimpleNamespace(q=q, rest=q.copy(), tri=tri, bodies=bodies, geometry=geometry,
                           diagnostic=diagnostic, result=result)


def select(c):
    return select_captured_pivot(c.result, c.tri, c.rest, c.q, c.bodies, c.diagnostic,
                                 c.geometry, side='right', time_s=.1)


def test_pivot_requires_matching_explicit_contact_normal_profile(capture):
    c=capture;c.result['contact_normal_profile']='distance_gradient_v1'
    with pytest.raises(ValueError,match='contact-normal'):select(c)
    c.diagnostic['contact_normal_profile']='distance_gradient_v1'
    _,report=select(c)
    assert report['contact_normal_profile']=='distance_gradient_v1'


def test_pivot_preserves_identity_of_facet_region_evidence(capture):
    c = capture
    c.result['contact_normal_profile'] = c.diagnostic['contact_normal_profile'] = 'distance_gradient_v1'
    c.result['contact_witness_profile'] = 'facet_regions_v1'
    with pytest.raises(ValueError, match='contact-witness'):
        select(c)
    c.diagnostic['contact_witness_profile'] = 'facet_regions_v1'
    ids, report = select(c)
    assert ids == [1] and report['contact_witness_profile'] == 'facet_regions_v1'


def test_selects_true_common_vertex_with_deterministic_tie_and_reports_discretization(capture):
    c = capture
    before = copy.deepcopy(c)
    ids, report = select(c)
    assert ids == [1]  # Equal midpoint distances: the lower original ID wins.
    assert report['common_witness_vertex_candidates'] == [1, 2]
    assert report['trajectory_reference_material_ids'] == [0, 1, 2, 3]
    assert report['distance_to_witness_midpoint_m'] == pytest.approx(.005)
    assert report['pad_witness_axis_lateral_m'] == pytest.approx(.005)
    assert report['selected_vertex_two_finger_contact_proven'] is False
    assert set(report['actual_cooked_pad_distances']) == {'thumb', 'index'}
    assert all(v['signed_inner_surface_distance_m'] > .004 for v in report['actual_cooked_pad_distances'].values())
    np.testing.assert_array_equal(c.q, before.q)
    np.testing.assert_array_equal(c.bodies, before.bodies)
    assert c.result == before.result
    # Moving the actual witnessed midpoint selects the other real shared node.
    c.result['evidence']['witnesses']['index']['point_m'][1] = .002
    assert select(c)[0] == [2]


def material_reference_fixture(c):
    from scripts.validation.laundry_pivot_reference import bind_pivot_reference, PROFILE
    c.rest = np.array([[.34, 0, .9], [.35, -.005, .9], [.35, .005, .9], [.36, .005, .9]])
    gate = CaptureGate('right', c.tri, c.rest, c.geometry.body_names,
                      ['right_thumb_IP', 'right_index_DIP'], profile='adjacent_patch_v3')
    for time in np.linspace(0, .1, 6):
        c.result = gate.observe(float(time), c.diagnostic, c.bodies, [0, 0], [0, 0], [0, 0], .06,
                               cloth_q=c.q)
    task = {'asset_metadata': {'cloths': [{'rest_vertices_m': c.rest.tolist(),
            'boundary_features': {'right_cuff': [0, 1, 2, 3]}}]}}
    edge = {'natural_corner_vertex_id': 3, 'feature': 'right_cuff', 'seed_vertex_ids': [3]}
    reference = bind_pivot_reference(task, edge, {'pivot_material_reference': PROFILE})
    return reference


def test_material_rank_uses_only_actual_common_vertices_after_full_capture(capture):
    c = capture
    reference = material_reference_fixture(c)
    original = select(c)
    assert original[0] == [1]
    assert select_captured_pivot(c.result, c.tri, c.rest, c.q, c.bodies, c.diagnostic,
        c.geometry, side='right', time_s=.1, material_reference=None) == original
    selected, report = select_captured_pivot(c.result, c.tri, c.rest, c.q, c.bodies, c.diagnostic,
        c.geometry, side='right', time_s=.1, material_reference=reference)
    assert selected == [2]
    assert report['trajectory_reference_material_ids'] == original[1]['trajectory_reference_material_ids']
    assert report['material_reference_within_original_35mm_neighbourhood'] is True
    assert report['distance_to_material_reference_m'] == pytest.approx(0.)
    assert report['selected_vertex_two_finger_contact_proven'] is False
    # The original common patch and original full-capture gate remain mandatory.
    c.result['capture_candidate'] = False
    with pytest.raises(ValueError, match='complete positive'):
        select_captured_pivot(c.result, c.tri, c.rest, c.q, c.bodies, c.diagnostic,
            c.geometry, side='right', time_s=.1, material_reference=reference)


def test_valid_material_descriptor_remote_from_actual_patch_is_rejected(capture):
    c = capture
    reference = material_reference_fixture(c)
    c.rest = np.vstack([c.rest, [.40, .005, .9]])
    c.q = np.vstack([c.q, [1., 1., 1.]])
    c.diagnostic['state_identity']['cloth_q_float64_sha256'] = hashlib.sha256(c.q.astype('<f8').tobytes()).hexdigest()
    reference.update(natural_corner_vertex_id=4, natural_corner_rest_m=c.rest[4].tolist(),
                     rest_point_m=[.39, .005, .9])
    with pytest.raises(ValueError, match='captured 35mm neighbourhood'):
        select_captured_pivot(c.result, c.tri, c.rest, c.q, c.bodies, c.diagnostic,
            c.geometry, side='right', time_s=.1, material_reference=reference)


def test_attached_material_reference_cannot_be_changed(capture):
    c = capture
    reference = material_reference_fixture(c)
    a = CapturedPatchAssist.__new__(CapturedPatchAssist)
    a.attachment_material_mode, a.rest, a.active = 'captured_pivot_v1', c.rest, False
    a.set_pivot_material_reference(reference)
    reference['rest_point_m'][0] += 1.
    assert a.pivot_material_reference['rest_point_m'][0] == pytest.approx(.35)
    a.active = True
    with pytest.raises(ValueError, match='while attached'):
        a.set_pivot_material_reference(None)


@pytest.mark.parametrize('mutation', [
    lambda c: c.result.update(capture_candidate=False),
    lambda c: c.result.update(continuous_duration_s=.08),
    lambda c: c.result.update(capture_profile='loaded_adjacent_patch_v5'),
    lambda c: c.result['evidence'].update(last_time_s=.08),
    lambda c: c.result.update(vertex_ids=[1, 2]),
    lambda c: c.result.update(vertex_ids=[0, 1, 2, 2]),
    lambda c: c.result.update(vertex_ids=[0, 1, 2, 100]),
    lambda c: c.diagnostic.update(state_identity={}),
    lambda c: c.diagnostic.update(side='left'),
    lambda c: c.diagnostic.update(distal_group_severe_penetration=True),
    lambda c: c.q.__setitem__((1, 0), .5),
    lambda c: c.bodies.__setitem__((0, 0), .01),
    lambda c: c.result['evidence'].update(material_topology={}),
    lambda c: c.result['evidence'].update(witnesses={}),
    lambda c: c.result['evidence']['witnesses']['thumb'].update(point_m=[1, 0, .9]),
    lambda c: c.result['evidence']['witnesses']['index'].update(triangle_id=0),
])
def test_partial_forged_stale_or_unwitnessed_input_cannot_create_pivot(capture, mutation):
    mutation(capture)
    with pytest.raises(ValueError):
        select(capture)


@pytest.mark.parametrize('explicit_material_reference', [False, True])
def test_disconnected_actual_triangles_have_no_pivot_fallback(capture, explicit_material_reference):
    c = capture
    reference = material_reference_fixture(c) if explicit_material_reference else None
    c.q = np.concatenate([c.q, c.q[[0, 1]]])
    c.rest = c.q.copy()
    c.tri[1] = [3, 4, 5]
    c.diagnostic['state_identity']['cloth_q_float64_sha256'] = hashlib.sha256(c.q.astype('<f8').tobytes()).hexdigest()
    with pytest.raises(ValueError, match='shared by both'):
        select_captured_pivot(c.result, c.tri, c.rest, c.q, c.bodies, c.diagnostic,
            c.geometry, side='right', time_s=.1, material_reference=reference)


def test_single_bridge_uses_its_actual_common_vertex_and_keeps_all_five_reference_nodes(capture):
    c = capture
    c.q = np.array([[-.01, 0, .9], [-.005, -.005, .9], [0, 0, .9],
                    [.005, -.005, .9], [.01, 0, .9]])
    c.rest = c.q.copy()
    c.tri = np.array([[0, 1, 2], [1, 3, 2], [2, 3, 4]])
    witness = c.diagnostic['multilayer_diagnostic']['paired_witnesses'][0]
    for finger, face in [('thumb', 0), ('index', 2)]:
        witness[finger].update(triangle_id=face, point_m=c.q[c.tri[face]].mean(0).tolist())
    c.diagnostic['state_identity']['cloth_q_float64_sha256'] = hashlib.sha256(c.q.astype('<f8').tobytes()).hexdigest()
    gate = CaptureGate('right', c.tri, c.rest, c.geometry.body_names,
                       ['right_thumb_IP', 'right_index_DIP'], profile='adjacent_patch_v3')
    for time in np.linspace(0, .1, 6):
        c.result = gate.observe(float(time), c.diagnostic, c.bodies, [0, 0], [0, 0], [0, 0], .06,
                                cloth_q=c.q)
    ids, report = select(c)
    assert ids == [2]
    assert report['trajectory_reference_material_ids'] == [0, 1, 2, 3, 4]
    assert report['material_topology']['bridge_triangle_id'] == 1


def test_original_modes_keep_their_exact_ids_and_reference(capture):
    c = capture
    for mode, expected in [('captured_patch', [0, 1, 2, 3]), ('shared_edge', [1, 2])]:
        assert capture_constraint_vertices(c.result, c.tri, mode) == expected
        a = CapturedPatchAssist.__new__(CapturedPatchAssist)
        a.attachment_material_mode, a.ids, a.captured_patch_ids = mode, expected, [0, 1, 2, 3]
        assert a.trajectory_reference_ids() == expected
    with pytest.raises(ValueError):
        capture_constraint_vertices(c.result, c.tri, 'captured_pivot_v1')  # Rich fresh evidence is required.


@pytest.mark.parametrize('use_material_reference', [False, True])
def test_single_constraint_mask_actual_wrist_offset_and_release(capture, monkeypatch, use_material_reference):
    c = capture
    reference = material_reference_fixture(c) if use_material_reference else None
    expected_id = 2 if use_material_reference else 1
    a = CapturedPatchAssist.__new__(CapturedPatchAssist)
    a.attachment_material_mode = 'captured_pivot_v1'
    a.pivot_material_reference = reference
    a.triangles, a.rest, a.geometry = c.tri, c.rest, c.geometry
    a.side, a.last_capture_diagnostic = 'right', c.diagnostic
    a.thumb_index, a.index_index = 1, 2
    a.namespace, a.cloth_id, a.wrist_index = '/World/MeasuredCaptureAssist/right', 'towel', 0
    a.env = SimpleNamespace(stage=object(), scene_info={'cloths': {'towel': {'path': '/World/towel'}}})
    a.events, a.ids, a.captured_patch_ids, a.prim, a.active = [], [], [], None, False
    attrs = {}
    class Attribute:
        def __init__(self, value): self.value = value
        def Get(self): return self.value
        def Set(self, value): self.value = value
    def attach(stage, path, cloth, anchor, ids, offsets):
        attrs.update({'omniphysics:attachmentEnabled': Attribute(True),
                      'omniphysics:vtxIndicesSrc0': Attribute(list(ids))})
        return SimpleNamespace(GetAttribute=lambda name: attrs[name])
    op = SimpleNamespace(Set=lambda value: None)
    xform = SimpleNamespace(AddTranslateOp=lambda _: op, AddOrientOp=lambda _: op)
    monkeypatch.setitem(sys.modules, 'pxr', SimpleNamespace(UsdGeom=SimpleNamespace(
        Xform=SimpleNamespace(Define=lambda *_: xform), XformOp=SimpleNamespace(PrecisionDouble=0))))
    physx = SimpleNamespace(get_physx_simulation_interface=lambda: SimpleNamespace(flush_changes=lambda: None))
    monkeypatch.setitem(sys.modules, 'omni', SimpleNamespace(physx=physx))
    monkeypatch.setitem(sys.modules, 'omni.physx', physx)
    monkeypatch.setattr('scripts.validation.laundry_cloth_interactions.attachment', attach)
    monkeypatch.setattr(a, '_pose', lambda value: setattr(a, 'anchor_pose', np.asarray(value).copy()))
    original = c.q.copy()
    a._enable(.1, c.result, c.bodies, c.q)
    assert a.ids == [expected_id] and a.trajectory_reference_ids() == [0, 1, 2, 3]
    np.testing.assert_array_equal(a.mask(4), np.arange(4) == expected_id)
    np.testing.assert_allclose(a.events[0]['fixed_local_offsets_m'], [[0, .005 if use_material_reference else -.005, 0]], atol=1.e-15)
    assert a.events[0]['pivot_selection']['selected_vertex_id'] == expected_id
    moved = c.bodies.copy(); moved[0, :3] += [.01, .02, .03]
    a.update('fold', .102, {'robot_body_q': moved}, c.q, [])
    np.testing.assert_array_equal(a.anchor_pose, moved[0])
    a.update('open', .104, {'robot_body_q': moved}, c.q, [])
    assert not a.active and attrs['omniphysics:attachmentEnabled'].Get() is False
    np.testing.assert_array_equal(a.mask(4), np.zeros(4))
    np.testing.assert_array_equal(c.q, original)
    assert [e['event'] for e in a.events] == ['enable', 'disable']
