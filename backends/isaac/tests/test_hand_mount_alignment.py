"""FR3 flange -> hand mount convention (flange_axis_centred_v1), checked on the
actual versioned assemblies with URDF FK and the hand's own collision meshes."""
import re
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import pytest
import trimesh

from bench.geometry import rpy_matrix
from bench.control.kinematics import URDFKinematics
from bench.paths import load_json
from bench.simulation.isaacsim_backend.robot_registry import get_hand_spec
from bench.simulation.isaacsim_backend.robot_asset import DEFAULT_ASSETS, DERIVED_RELATIVE

# Audit data: the hand link whose wrist-end face mates with the flange/adapter,
# its fingers (tip links) and the MCP flexion joints used to find the palm normal.
HANDS = {
    'sharpa_w01_right': ('right_hand_wrist', r'right_(index|middle|ring|pinky)_MCP_FE'),
    'allegro_v5_right': ('allegro_palm_link', r'allegro_joint_(1|5|9)_0'),
    'leap_v1_right': ('leap_palm_lower', r'leap_(1|5|9)'),
    'shadow_hand_e_right': ('rh_forearm', r'rh_(FF|MF|RF|LF)J3'),
}


def assembly(hand_id):
    spec = get_hand_spec(hand_id)
    if hand_id == 'sharpa_w01_right':
        directory = DEFAULT_ASSETS / DERIVED_RELATIVE
    else:
        directory = DEFAULT_ASSETS / 'robots/assemblies' / ('fr3_' + hand_id) / spec['assembly_version']
    source = load_json(directory / 'SOURCE.json')
    return Path(source['urdf_path']), source, spec


def collision_points(link, urdf_dir):
    points = []
    for element in link.findall('collision'):
        T = np.eye(4)
        origin = element.find('origin')
        if origin is not None:
            T[:3, 3] = np.fromstring(origin.get('xyz', '0 0 0'), sep=' ')
            T[:3, :3] = rpy_matrix(np.fromstring(origin.get('rpy', '0 0 0'), sep=' '))
        shape = element.find('geometry')[0]
        if shape.tag == 'mesh':
            mesh = trimesh.load(urdf_dir / shape.get('filename'), force='mesh', process=False)
            P = np.asarray(mesh.vertices) * np.fromstring(shape.get('scale', '1 1 1'), sep=' ')
        elif shape.tag == 'cylinder':
            r, L = float(shape.get('radius')), float(shape.get('length'))
            a = np.linspace(0, 2 * np.pi, 64, endpoint=False)
            ring = np.c_[r * np.cos(a), r * np.sin(a)]
            P = np.r_[np.c_[ring, np.full(64, -L / 2)], np.c_[ring, np.full(64, L / 2)]]
        elif shape.tag == 'box':
            P = np.array([[x, y, z] for x in (-.5, .5) for y in (-.5, .5) for z in (-.5, .5)]) * np.fromstring(shape.get('size'), sep=' ')
        else:
            pytest.skip(f'unsupported collision primitive {shape.tag}')
        points.append((T[:3, :3] @ P.T).T + T[:3, 3])
    return np.vstack(points)


@pytest.mark.parametrize('hand_id', sorted(HANDS))
def test_hand_mount_follows_flange_axis_convention(hand_id):
    urdf, source, spec = assembly(hand_id)
    face_link, mcp = HANDS[hand_id]
    kin = URDFKinematics(urdf)
    poses = kin.forward({})
    flange = np.linalg.inv(poses['fr3_link8'])
    in_flange = lambda link, P: (flange @ poses[link] @ np.c_[P, np.ones(len(P))].T).T[:, :3]
    prefix = spec['prefix']
    tips = {f: prefix + link for f, link in spec['fingertip_links'].items()}
    palm = prefix + spec['palm_link']
    parent = {j.child: j.parent for j in kin.joints}

    def finger_base(link):
        while parent.get(link) not in (palm, None):
            link = parent[link]
        return link
    fingers = [f for f in tips if f != 'thumb']
    tip = lambda P, f: (flange @ P[tips[f]])[:3, 3]
    direction = np.mean([tip(poses, f) - (flange @ poses[finger_base(tips[f])])[:3, 3] for f in fingers], axis=0)
    direction /= np.linalg.norm(direction)
    sign = {j.name: 1. if abs(j.upper) >= abs(j.lower) else -1. for j in kin.active}
    flexed = kin.forward({n: .15 * sign[n] for n in kin.names if re.fullmatch(mcp, n)})
    curl = np.mean([tip(flexed, f) - tip(poses, f) for f in fingers], axis=0)
    normal = curl - (curl @ direction) * direction
    normal /= np.linalg.norm(normal)
    thumb = (tip(poses, 'thumb') - (flange @ poses[palm])[:3, 3]) @ np.cross(direction, normal)
    assert np.degrees(np.arccos(direction[2])) < 2.     # fingers along flange +z
    assert np.degrees(np.arccos(normal[0])) < 2.        # palm normal along +x
    assert thumb > 0                                    # thumb toward +y
    links = {link.get('name'): link for link in ET.parse(urdf).getroot().findall('link')}
    face = in_flange(face_link, collision_points(links[face_link], urdf.parent))
    seat = face[:, 2].min()
    slab = face[face[:, 2] < seat + .001]
    centre = (slab[:, :2].min(0) + slab[:, :2].max(0)) / 2
    adapter = spec['adapter']
    expected_seat = adapter['thickness_m'] if adapter else 0.
    # SHARPA's with_flange collision starts 0.5 mm off its flange face.
    assert abs(seat - expected_seat) <= .0006
    assert np.linalg.norm(centre) <= .0005
    if adapter:
        assert source['mount']['adapter_mass_included'] is True
        assert 'fr3_hand_adapter' in links
        assert source['mount']['adapter']['top_face_z_m'] == pytest.approx(expected_seat)
