import numpy as np
import pytest
import hashlib
import json

from scripts.media.render_laundry_material_review import (
    apply_variant, invariant_digest, recorded_task_cameras, verify_recorded_skin,
)


def test_replay_camera_uses_hash_bound_frozen_task_target(tmp_path):
    path = tmp_path / 'task.json'
    path.write_text(json.dumps({'asset_metadata': {'target_center_xy_m': [.95, -.55]}}))
    entry = {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    (tmp_path / 'native_report.json').write_text(json.dumps({'task_input': entry}))
    cameras, source = recorded_task_cameras(tmp_path)
    np.testing.assert_allclose(cameras['thickness']['target'], [.95, -.55, .91])
    assert source['task_input'] == entry
    path.write_text(json.dumps({'asset_metadata': {'target_center_xy_m': [.65, -.55]}}))
    with pytest.raises(ValueError, match='differs from the recorded'):
        recorded_task_cameras(tmp_path)


def fixture_recording(tmp_path):
    from pxr import Sdf, Usd, UsdGeom
    from bench.simulation.isaacsim_backend.cloth_visuals import ClosedClothVisual
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.Xform.Define(stage, '/World')
    native = UsdGeom.Mesh.Define(stage, '/World/Cloths/towel')
    native.GetPrim().CreateAttribute('omniphysics:mass', Sdf.ValueTypeNames.Float).Set(.108)
    points = np.array([[0., 0., .9], [.2, 0., .9], [0., .2, .9]], dtype=np.float32)
    tri = np.array([[0, 1, 2]], dtype=np.int32)
    texture = tmp_path / 'normal.jpg'
    texture.write_bytes(b'network-only fixture')
    ClosedClothVisual(stage, native.GetPath(), 'towel', tri, points, .003,
                      [.2, .4, .6], {'normal': str(texture)}, points)
    archive = tmp_path / 'native_states.npz'
    np.savez(archive, towel_q=points[None], towel_qd=np.zeros_like(points)[None], towel_triangles=tri)
    return stage, archive


def test_static_endpoint_must_match_actual_recording(tmp_path):
    from pxr import UsdGeom, Vt
    stage, archive = fixture_recording(tmp_path)
    checked = verify_recorded_skin(stage, archive)
    assert checked['towel']['max_skin_reconstruction_error_m'] == 0.
    mesh = UsdGeom.Mesh(stage.GetPrimAtPath('/World/ClothVisuals/towel'))
    points = np.asarray(mesh.GetPointsAttr().Get()).copy()
    points[:, 2] += .01
    mesh.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(points))
    with pytest.raises(ValueError, match='differs from archived native endpoint'):
        verify_recorded_skin(stage, archive)


def test_material_override_guard_detects_unrelated_physics_edit(tmp_path):
    stage, archive = fixture_recording(tmp_path)
    before = invariant_digest(stage)
    apply_variant(stage, verify_recorded_skin(stage, archive), 'types', None)
    assert invariant_digest(stage) == before
    stage.GetPrimAtPath('/World/Cloths/towel').GetAttribute('omniphysics:mass').Set(.2)
    assert invariant_digest(stage) != before
