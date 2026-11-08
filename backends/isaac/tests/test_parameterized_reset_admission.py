import json
import math

import numpy as np
import pytest

pxr = pytest.importorskip('pxr')
from pxr import Gf, Usd, UsdGeom, UsdPhysics  # noqa: E402

from scenes.workrooms.parameterized import reset_admission as admission  # noqa: E402


MECHANICAL_TASK = {'evaluation': {'maximum_reported_penetration_m': .0005}}


def _stage():
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageMetersPerUnit(stage, 1.)
    UsdGeom.SetStageUpAxis(stage, 'Z')
    UsdGeom.Xform.Define(stage, '/World')
    # Worktop top at z=0.78 spanning x in [0, 1]; the floor never supports.
    _cube(stage, '/World/Room/Geometry/worktop/Collider', (.5, 0., .76), (1., 1., .04))
    _cube(stage, '/World/Room/Geometry/room_floor/Collider', (.5, 0., -.05), (4., 4., .1))
    return stage


def _cube(stage, path, center, size, yaw_deg=0.):
    cube = UsdGeom.Cube.Define(stage, path)
    cube.CreateSizeAttr(1.)
    xf = UsdGeom.Xformable(cube)
    xf.AddTranslateOp().Set(Gf.Vec3d(*center))
    if yaw_deg:
        xf.AddRotateZOp().Set(float(yaw_deg))
    xf.AddScaleOp().Set(Gf.Vec3f(*size))
    UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
    return cube


def _body(stage, path, origin, com=(0., 0., 0.)):
    """Rigid body frame at the world origin; children are authored in world coordinates."""
    xform = UsdGeom.Xform.Define(stage, path)
    UsdPhysics.RigidBodyAPI.Apply(xform.GetPrim())
    world_com = [o + c for o, c in zip(origin, com)]
    UsdPhysics.MassAPI.Apply(xform.GetPrim()).CreateCenterOfMassAttr(Gf.Vec3f(*world_com))
    return path


def _box_mesh(stage, path, half, center, approximation='sdf'):
    mesh = UsdGeom.Mesh.Define(stage, path)
    hx, hy, hz = half
    mesh.CreatePointsAttr([Gf.Vec3f(x, y, z) for x in (-hx, hx) for y in (-hy, hy) for z in (-hz, hz)])
    # Closed outward-wound box in itertools.product corner order.
    faces = [[0, 1, 3, 2], [4, 6, 7, 5], [0, 4, 5, 1], [2, 3, 7, 6], [0, 2, 6, 4], [1, 5, 7, 3]]
    mesh.CreateFaceVertexCountsAttr([4]*6)
    mesh.CreateFaceVertexIndicesAttr([i for face in faces for i in face])
    mesh.CreateExtentAttr([Gf.Vec3f(-hx, -hy, -hz), Gf.Vec3f(hx, hy, hz)])
    UsdGeom.Xformable(mesh).AddTranslateOp().Set(Gf.Vec3d(*center))
    UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
    UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr(approximation)
    return mesh


def _evaluate(stage, bodies, layout=(), task=None, origins=None):
    record = admission.evaluate(stage, bodies=bodies, origins=origins or {}, layout=list(layout),
                                task=task or MECHANICAL_TASK)
    json.dumps(admission._finite(record), allow_nan=False)
    return record


def _rules(record):
    return sorted({row['rule'] for row in record['reasons']})


def test_com_beyond_worktop_edge_is_rejected_and_inside_is_admitted():
    stage = _stage()
    # A 100 mm gear-like slab straddling the x=0 edge, centre of mass 20 mm beyond it.
    _body(stage, '/World/Objects/gear', (-.02, 0., .7802), com=(0., 0., .01))
    _cube(stage, '/World/Objects/gear/collision_000', (-.02, 0., .7902), (.1, .1, .02))
    record = _evaluate(stage, {'gear': '/World/Objects/gear'})
    assert _rules(record) == ['S_com_outside_support']
    assert record['measurements']['support']['gear']['com_margin_m'] == pytest.approx(-.02, abs=1e-9)

    stage = _stage()
    _body(stage, '/World/Objects/gear', (.06, 0., .7802), com=(0., 0., .01))
    _cube(stage, '/World/Objects/gear/collision_000', (.06, 0., .7902), (.1, .1, .02))
    record = _evaluate(stage, {'gear': '/World/Objects/gear'})
    assert record['passed'] and record['measurements']['minimum_com_margin_m'] > .009


def test_side_neighbour_at_same_height_is_not_support():
    """Two yawed tubes side by side: AABB slivers overlap, colliders do not touch.

    The tube beyond the edge has only the floor below.  The historical near-
    support proxy accepted its upper piece against the neighbour's body top.
    """
    stage = _stage()
    for name, x, y in (('inside', .985, 0.), ('outside', 1.040, .03)):
        _body(stage, f'/World/Objects/{name}', (x, y, .7803), com=(0., 0., .03))
        _cube(stage, f'/World/Objects/{name}/body', (x, y, .8053), (.04, .04, .05), yaw_deg=45.)
        _cube(stage, f'/World/Objects/{name}/cap', (x, y, .8603), (.04, .04, .06), yaw_deg=45.)
    record = _evaluate(stage, {'inside': '/World/Objects/inside', 'outside': '/World/Objects/outside'})
    support = record['measurements']['support']
    assert support['inside']['status'] == 'supported' and support['inside']['com_margin_m'] > 0
    assert support['outside']['status'] == 'unsupported'
    assert _rules(record) == ['S_unsupported']


def test_short_drop_is_allowed_but_air_above_floor_is_not():
    stage = _stage()
    _body(stage, '/World/Objects/plug', (.5, 0., .7903), com=(0., 0., .01))
    _cube(stage, '/World/Objects/plug/shape', (.5, 0., .8003), (.02, .02, .02))
    record = _evaluate(stage, {'plug': '/World/Objects/plug'})
    assert record['passed'] and record['measurements']['support']['plug']['status'] == 'drop'


def test_evaluator_reset_bounds():
    task = {**MECHANICAL_TASK, 'asset_metadata': {'recovery_workspace_bounds_m': [[-1., -1.35, .6], [2., -.55, 1.8]]}}
    stage = _stage()
    record = _evaluate(stage, {}, task=task, origins={'salt': [.36, -.5368, .9045], 'pepper': [.17, -.74, .97]})
    assert _rules(record) == ['R1_outside_referee_bounds']
    assert record['measurements']['bounds_margin_m']['salt'] == pytest.approx(-.0132)
    record = _evaluate(stage, {}, task=task, origins={'salt': [.36, -.5532, .9045]})
    assert record['passed']


def _ring_world(ring_x, layout_offset):
    stage = _stage()
    # Fixed 120 x 120 x 8 mm shoulder on the worktop; a 6 mm supply rest moves with the ring.
    _cube(stage, '/World/Fixtures/shaft_fixture/collision_000', (.4, 0., .784), (.12, .12, .008))
    _cube(stage, '/World/Fixtures/ring_rest/collision_000', (ring_x, -.1, .783), (.10, .30, .006))
    _body(stage, '/World/Objects/wide_ring', (ring_x, 0., .786), com=(0., 0., .01))
    _box_mesh(stage, '/World/Objects/wide_ring/collision', (.0425, .0425, .01), (ring_x, 0., .796))
    layout = [{'id': 'loose_supply', 'paths': ['/World/Objects/wide_ring', '/World/Fixtures/ring_rest'],
               'translation_xy_m': [layout_offset, 0.], 'yaw_rad': .3 if layout_offset else 0.,
               'pivot_m': [ring_x, 0., .786]}]
    return stage, layout


def test_sdf_ring_pressed_into_fixed_shoulder_is_rejected():
    stage, layout = _ring_world(.35, .1)
    record = _evaluate(stage, {'wide_ring': '/World/Objects/wide_ring'}, layout)
    assert _rules(record) == ['R3_penetration']
    contact = record['measurements']['interpenetration']
    assert contact['maximum_depth_m'] == pytest.approx(.002, abs=2e-5)
    assert contact['threshold_m'] == pytest.approx(.00025)
    assert contact['worst_pair'] == ['/World/Objects/wide_ring/collision', '/World/Fixtures/shaft_fixture/collision_000']


def test_sdf_ring_clear_of_shoulder_is_admitted():
    stage, layout = _ring_world(.22, .1)
    record = _evaluate(stage, {'wide_ring': '/World/Objects/wide_ring'}, layout)
    assert record['passed']
    assert record['measurements']['interpenetration']['maximum_depth_m'] == 0.


def test_unchanged_relation_is_not_a_layout_candidate():
    # Same overlap, but the layout moved nothing: canonical relations are not rechecked.
    stage, layout = _ring_world(.35, 0.)
    record = _evaluate(stage, {'wide_ring': '/World/Objects/wide_ring'}, layout)
    assert record['measurements']['interpenetration']['candidates'] == 0
    assert 'R3_penetration' not in _rules(record)


def test_convex_pair_depth_and_open_triangle_surface():
    stage = _stage()
    _cube(stage, '/World/Fixtures/plate/collision_000', (.5, 0., .7845), (.2, .2, .009))
    _body(stage, '/World/Objects/gear', (.42, 0., .7802), com=(0., 0., .01))
    _cube(stage, '/World/Objects/gear/collision_000', (.42, 0., .7902), (.1, .1, .02))
    layout = [{'id': 'g', 'paths': ['/World/Objects/gear'], 'translation_xy_m': [.1, 0.], 'yaw_rad': .2}]
    record = _evaluate(stage, {'gear': '/World/Objects/gear'}, layout)
    assert 'R3_penetration' in _rules(record)
    assert record['measurements']['interpenetration']['maximum_depth_m'] == pytest.approx(.0088, abs=1e-4)
    # An open triangle surface has no inside; only its own samples inside the other solid count.
    open_shape = {'kind': 'mesh', 'points': admission._sphere_points(.05, 0., 'Z'), 'faces': np.asarray([[0, 1, 2]])}
    assert admission._depth_inside(open_shape, open_shape['points']) == 0.


def test_jointed_bodies_are_skipped_and_records_are_deterministic():
    stage = _stage()
    _body(stage, '/World/Objects/drawer', (-.05, 0., .79), com=(0., 0., 0.))
    _cube(stage, '/World/Objects/drawer/shape', (-.05, 0., .79), (.1, .1, .02))
    joint = UsdPhysics.PrismaticJoint.Define(stage, '/World/Joints/slide')
    joint.CreateBody1Rel().SetTargets(['/World/Objects/drawer'])
    first = _evaluate(stage, {'drawer': '/World/Objects/drawer'})
    second = _evaluate(stage, {'drawer': '/World/Objects/drawer'})
    assert first['passed'] and first['measurements']['support']['drawer']['status'] == 'jointed'
    assert json.dumps(admission._finite(first), sort_keys=True) == json.dumps(admission._finite(second), sort_keys=True)


def test_not_applicable_families_and_canonical_seed():
    assert admission.check_reset_admission({}, [], family='laundry_room', seed=3)['applicable'] is False
    assert admission.check_reset_admission({}, [], family='kitchen', seed=0)['applicable'] is False


def test_rejection_carries_json_safe_measurements(monkeypatch):
    stage, layout = _ring_world(.35, .1)
    monkeypatch.setattr('scripts.validation.workroom_randomized_workspace.composed_support_stage',
                        lambda scene: (stage, {}))
    values = {'scene': {}, 'task': MECHANICAL_TASK,
              'actual_body_poses_m': {'wide_ring': {'prim_path': '/World/Objects/wide_ring',
                                                    'position_m': [.35, 0., .786]}}}
    with pytest.raises(admission.ResetAdmissionError) as raised:
        admission.check_reset_admission(values, layout, family='mechanical_assembly', seed=3)
    assert isinstance(raised.value, ValueError)
    assert 'R3_penetration' in str(raised.value)
    json.dumps(raised.value.measurements, allow_nan=False)
    assert not math.isnan(raised.value.measurements['measurements']['interpenetration']['maximum_depth_m'])
