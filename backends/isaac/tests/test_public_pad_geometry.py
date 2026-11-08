"""Original source pad geometry in public virtual-tip frames; CPU only."""
from copy import deepcopy
import hashlib
import io
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import pytest
import trimesh

from bench.control.bimanual import BimanualFingertipPolicy
from bench.control.fingertip import Fr3SharpaGeometry
from bench.geometry import quaternion_matrix, rpy_matrix, transform
from bench.paths import DATA_ROOT
from bench.protocol.robot import FingerGuide
from bench.runtime.pad_geometry import source_pad_geometries, _mesh_vertices


@pytest.fixture(scope='module')
def sides():
    return BimanualFingertipPolicy(f'{DATA_ROOT}/assets', [0., 0., .77],
                                   assembly_version='isaacsim_6_1_v2').geometry.sides


@pytest.mark.parametrize('side', ['left', 'right'])
def test_static_pad_transform_matches_original_fk_and_source_mesh_in_four_postures(sides, side):
    model = sides[side]
    pads = source_pad_geometries(model)
    assert set(pads) == {'thumb', 'index', 'middle', 'ring', 'little'}
    root = ET.parse(model.hand_path).getroot()
    links = {link.get('name'): link for link in root.findall('link')}
    for amount in (0., .25, .65, 1.):
        q = {j.name: (1 - amount) * j.lower + amount * j.upper for j in model.hand.active}
        fk = model.hand.forward(q)
        for finger, pad in pads.items():
            guide = FingerGuide(joint_names=[j.name for j in model.finger_joints[finger]],
                                tip_link=model.fingertip_links[finger], pad_geometry=pad)
            p = guide.pad_geometry
            tip = guide.tip_link
            actual = np.linalg.inv(fk[tip]) @ fk[p.pad_link]
            advertised = transform(p.tip_from_pad.position_m, quaternion_matrix(p.tip_from_pad.quaternion_wxyz))
            assert np.max(np.abs(advertised - actual)) < 2.e-14
            collision = links[p.pad_link].find('collision')
            mesh = collision.find('geometry/mesh')
            path = (Path(model.hand_path).parent / mesh.get('filename')).resolve()
            raw = path.read_bytes()
            vertices = np.asarray(trimesh.load(io.BytesIO(raw), file_type='stl', force='mesh', process=False).vertices)
            origin = collision.find('origin')
            xyz = np.fromstring(origin.get('xyz', '0 0 0'), sep=' ')
            rpy = np.fromstring(origin.get('rpy', '0 0 0'), sep=' ')
            scale = np.fromstring(mesh.get('scale', '1 1 1'), sep=' ')
            # Independent route: compose the full forward kinematics to the
            # physical pad, then invert the public tip pose for every mesh point.
            matrix = np.linalg.inv(fk[tip]) @ fk[p.pad_link] @ transform(xyz, rpy_matrix(rpy))
            points = (matrix @ np.c_[vertices * scale, np.ones(len(vertices))].T).T[:, :3]
            assert np.allclose(p.bounds_in_tip_frame_m, [points.min(0), points.max(0)], atol=2.e-14, rtol=0.)
            assert p.source_hashes['collision_mesh_0'] == hashlib.sha256(raw).hexdigest()
            assert p.native_contact_geometry_verified is False
            assert DATA_ROOT not in guide.model_dump_json()
            assert 'vertices' not in guide.model_dump_json()


def test_source_cache_cannot_be_modified_through_public_result(sides):
    model = sides['left']
    one = source_pad_geometries(model)
    hits = _mesh_vertices.cache_info().hits
    one['thumb']['bounds_in_tip_frame_m'][0][0] = 100.
    two = source_pad_geometries(model)
    assert two['thumb']['bounds_in_tip_frame_m'][0][0] < 0.
    assert _mesh_vertices.cache_info().hits > hits
    # A changed URDF must not silently reuse a populated geometry cache.
    wrong = deepcopy(model)
    wrong.asset_sha256[str(wrong.hand_path)] = '0' * 64
    with pytest.raises(ValueError, match='pinned URDF'):
        source_pad_geometries(wrong)


def test_changed_mesh_cannot_reuse_cached_geometry_or_original_source_claim(sides, monkeypatch):
    model = sides['left']
    source_pad_geometries(model)
    read = Path.read_bytes
    def changed(path):
        raw = read(path)
        if path.name == 'elastomer_HB1_TH.STL':
            return raw[:-1] + bytes([raw[-1] ^ 1])
        return raw
    monkeypatch.setattr(Path, 'read_bytes', changed)
    with pytest.raises(ValueError, match='original robot input manifest'):
        source_pad_geometries(model)


def test_non_sharpa_and_legacy_finger_guides_omit_the_optional_field():
    legacy = FingerGuide(joint_names=['index'], tip_link='index_tip')
    assert 'pad_geometry' not in legacy.model_dump(exclude_unset=True)
    from types import SimpleNamespace
    assert source_pad_geometries(SimpleNamespace(embodiment=SimpleNamespace(sharpa_joint_semantics=False))) == {}


def test_renamed_instances_share_source_dimensions_without_losing_frame_identity():
    from bench.control.instance_embodiment import load_instance
    embodiment = load_instance(f'{DATA_ROOT}/assets', 'front', {
        'hand_id': 'sharpa_w01_right', 'assembly_version': 'isaacsim_6_1_v2',
        'base_position_m': [0., 0., .77], 'base_quaternion_wxyz': [1., 0., 0., 0.]})
    model = Fr3SharpaGeometry(f'{DATA_ROOT}/assets', [0., 0., 0.], embodiment=embodiment)
    pads = source_pad_geometries(model)
    assert len(pads) == 5
    assert pads['index']['pad_link'] == 'front_index_elastomer'
    assert pads['index']['tip_from_pad']['frame_id'] == 'front_index_fingertip'


def test_pad_frame_and_bounds_are_validated(sides):
    model = sides['left']
    pad = source_pad_geometries(model)['index']
    with pytest.raises(ValueError, match='tip frame'):
        FingerGuide(joint_names=['left_index_MCP_FE'], tip_link='wrong_tip', pad_geometry=pad)
    pad['bounds_center_in_tip_frame_m'][0] += .01
    with pytest.raises(ValueError, match='midpoint'):
        FingerGuide(joint_names=['left_index_MCP_FE'], tip_link='left_index_fingertip', pad_geometry=pad)

