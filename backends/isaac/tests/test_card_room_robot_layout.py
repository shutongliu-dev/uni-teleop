"""Real URDF geometry checks for waiting poses, not native physics acceptance."""
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import pytest
import trimesh

from bench.paths import DATA_ROOT, load_json
from scenes.board_games.tabletop import geometry_for
from bench.control.bimanual import BimanualGeometry
from bench.geometry import rpy_matrix, transform
from scenes.workrooms.card_room.robot_layout import (ready_configuration, arm_base_poses,
    ROBOT_BASE_DISTANCE_M, ROBOT_BASE_HEIGHT_M)
from scenes.workrooms.observations import expected_camera_world

RELEASE = Path(f'{DATA_ROOT}/datasets/fr3_sharpa_bench/isaacsim/releases/board_games/hex7x7/v6')
GAMES = ('tic_tac_toe_3x3', 'connect_four_5x5', 'hex7x7')
PITCHES = {'tic_tac_toe_3x3': .14, 'connect_four_5x5': .115, 'hex7x7': .09}


def expanded_geometry(game_id):
    return replace(geometry_for(game_id), pitch_m=PITCHES[game_id])


@pytest.fixture(scope='module')
def robot():
    path = RELEASE/'scene.json'
    if not path.exists():
        pytest.skip('Pinned real robot release unavailable')
    robot = load_json(path)['robot']
    robot['arm_base_poses'] = arm_base_poses(expanded_geometry(GAMES[0]))
    return robot


@pytest.fixture(scope='module')
def model(robot):
    return BimanualGeometry(robot['assets_root'], robot['base_position_m'],
        robot['base_quaternion_wxyz'], assembly_version=robot['assembly_version'],
        arm_base_poses=robot['arm_base_poses'])


@pytest.mark.parametrize('game_id', GAMES)
def test_ready_is_bounded_symmetric_and_leaves_scene_unmodified(robot, model, game_id):
    before = deepcopy(robot)
    initial, homes, cameras = ready_configuration(robot, expanded_geometry(game_id))
    assert robot == before
    assert set(initial['joint_positions_rad']) == set(model.names)
    for side in ('left', 'right'):
        single = model.sides[side]
        q = {n: initial['joint_positions_rad'][n] for n in single.names}
        single.checked_joints(q, enforce_limits=True)
        assert homes[side]['minimum_arm_joint_margin_rad'] > .30
        assert homes[side]['position_error_m'] < .0001
        assert homes[side]['rotation_error_rad'] < .002
        assert homes[side]['collision_qualified'] is False
        assert 'reserve_index' not in homes[side]
        for finger in ('index', 'middle', 'ring', 'pinky'):
            assert abs(q[f'{side}_{finger}_MCP_AA']) < .1
            assert .2 < q[f'{side}_{finger}_PIP'] < .8
        assert not cameras['wrist_'+side]['aim']
        assert 'aim_target_m' not in cameras['wrist_'+side]
    left = [initial['joint_positions_rad'][n] for n in model.sides['left'].arm.names]
    right = [initial['joint_positions_rad'][n] for n in model.sides['right'].arm.names]
    assert right == pytest.approx(left, abs=1e-12)


def test_inventory_and_game_changes_do_not_move_ready_robots_or_camera(robot):
    results = [ready_configuration(robot, expanded_geometry(game)) for game in GAMES]
    assert results[0][0] == results[1][0] == results[2][0]
    assert results[0][2] == results[1][2] == results[2][2]


def collision_meshes(model, initial):
    """Actual source collision meshes with all local transforms, no visual proxies."""
    result = {}
    for side, single in model.sides.items():
        q = initial['joint_positions_rad']
        arm_poses = single.arm.forward(q, single.arm_base)
        wrist = arm_poses[single.arm_tip] @ single.wrist_offset
        hand_poses = single.hand.forward(q, wrist)
        rows = []
        for path, poses in ((single.arm_path, arm_poses), (single.hand_path, hand_poses)):
            for link in ET.parse(path).getroot().findall('link'):
                name = link.get('name')
                if name not in poses:
                    continue
                for collision in link.findall('collision'):
                    shape = collision.find('geometry/mesh')
                    assert shape is not None, 'Audit must explicitly handle every source collider'
                    origin = collision.find('origin')
                    xyz = np.fromstring(origin.get('xyz', '0 0 0') if origin is not None else '0 0 0', sep=' ')
                    rpy = np.fromstring(origin.get('rpy', '0 0 0') if origin is not None else '0 0 0', sep=' ')
                    source = (path.parent/shape.get('filename')).resolve()
                    mesh = trimesh.load(source, force='mesh', process=False)
                    vertices = np.asarray(mesh.vertices)*np.fromstring(shape.get('scale', '1 1 1'), sep=' ')
                    pose = poses[name] @ transform(xyz, rpy_matrix(rpy))
                    world = vertices @ pose[:3, :3].T + pose[:3, 3]
                    rows.append((name, world, np.asarray(mesh.faces)))
        result[side] = rows
    return result


@pytest.fixture(scope='module')
def meshes(robot, model):
    initial, _, _ = ready_configuration(robot, expanded_geometry(GAMES[0]))
    return collision_meshes(model, initial)


@pytest.mark.parametrize('game_id', GAMES)
def test_source_collision_envelopes_clear_both_boards_and_each_other(meshes, game_id):
    geometry = expanded_geometry(game_id)
    board_half = geometry.board_size_xy_m[1]/2
    bounds = {}
    for side, sign in (('left', -1), ('right', 1)):
        parts = meshes[side]
        assert len(parts) >= 35
        vertices = np.vstack([v for _, v, _ in parts])
        hand = np.vstack([v for name, v, _ in parts if not name.startswith('fr3_')])
        bounds[side] = (vertices.min(0), vertices.max(0))
        # Source mesh envelopes leave the central board strip entirely clear.
        assert (sign*vertices[:, 1]).min() > board_half+.03
        assert hand[:, 2].min() > geometry.table_top_z_m+geometry.piece_height_m+.30
        # Movable links remain in the room's previously audited clear volume.
        assert vertices[:, 0].min() > -.05 and vertices[:, 0].max() < 1.35
        assert np.abs(vertices[:, 1]).max() < 1.
        assert vertices[:, 2].max() < 1.95
    assert bounds['right'][0][1]-bounds['left'][1][1] > .60


def segment_hit(eye, point, meshes):
    """Triangle segment intersection; source mesh occlusion only."""
    direction = point-eye
    for rows in meshes.values():
        for name, vertices, faces in rows:
            if (np.any(vertices.max(0) < np.minimum(eye, point))
                    or np.any(vertices.min(0) > np.maximum(eye, point))):
                continue
            triangles = vertices[faces]
            v0 = triangles[:, 0]
            e1, e2 = triangles[:, 1]-v0, triangles[:, 2]-v0
            h = np.cross(direction, e2)
            a = np.einsum('ij,ij->i', e1, h)
            valid = np.abs(a) > 1e-12
            f = np.divide(1., a, out=np.zeros_like(a), where=valid)
            s = eye-v0
            u = f*np.einsum('ij,ij->i', s, h)
            q = np.cross(s, e1)
            v, t = f*(q @ direction), f*np.einsum('ij,ij->i', e2, q)
            if np.any(valid & (u >= 0) & (v >= 0) & (u+v <= 1)
                      & (t > 1e-5) & (t < 1-1e-5)):
                return name
    return None


@pytest.mark.parametrize('game_id', GAMES)
def test_fixed_wrist_optics_project_near_board_cells_without_source_mesh_occlusion(robot, model, meshes, game_id):
    initial, homes, cameras = ready_configuration(robot, expanded_geometry(game_id))
    for side in ('left', 'right'):
        single = model.sides[side]
        public = single.public_geometry({n: initial['joint_positions_rad'][n] for n in single.names})
        spec = cameras['wrist_'+side]
        camera = expected_camera_world(spec, public['wrist'])
        for point in homes[side]['wrist_probe_points_world']:
            xyz = (np.r_[point, 1.] @ np.linalg.inv(camera))[:3]
            assert xyz[2] < -.1
            half = np.array([spec['horizontal_aperture_mm'], spec['vertical_aperture_mm']])/2
            ndc = xyz[:2]/-xyz[2]*spec['focal_length_mm']/half
            assert np.abs(ndc).max() < .70
            assert segment_hit(camera[3, :3], np.asarray(point), meshes) is None


def test_ready_rejects_unreviewed_base_transform(robot):
    altered = deepcopy(robot)
    altered['arm_base_poses']['left']['position_m'][0] += .1
    with pytest.raises(ValueError, match='reviewed opposing'):
        ready_configuration(altered, expanded_geometry(GAMES[0]))


def convex_projection_distance_lower_bound(first, second):
    """Certified separation of source convex envelopes on measured face axes.

    A positive projection gap is a lower bound on every triangle-pair distance.
    A nonpositive result is unresolved overlap, not a penetration measurement.
    PhysX cooking and contact offsets still require the independent native run.
    """
    from scipy.spatial import ConvexHull
    coarse = np.maximum(first.min(0)-second.max(0), second.min(0)-first.max(0)).max()
    if coarse > .025:
        return float(coarse)
    one, two = ConvexHull(first), ConvexHull(second)
    axes = np.vstack((np.eye(3), one.equations[:, :3], two.equations[:, :3]))
    axes /= np.linalg.norm(axes, axis=1)[:, None]
    a, b = first[one.vertices] @ axes.T, second[two.vertices] @ axes.T
    return float(np.maximum(a.min(0)-b.max(0), b.min(0)-a.max(0)).max())


def test_original_hand_meshes_clear_nonadjacent_arm_links(meshes):
    # The v1 ready pose failed this independent guard: its left index_DP
    # intersected link2 by a 1.107 mm source-envelope projection, reproduced as
    # 1.095 mm native contact penetration.  Environment-only guards missed it.
    for side, parts in meshes.items():
        arm = [(name, v) for name, v, _ in parts if name.startswith('fr3_')
               and not name.endswith(('_link7', '_link8', '_base'))]
        hand = [(name, v) for name, v, _ in parts if not name.startswith('fr3_')]
        for hand_name, hand_vertices in hand:
            for arm_name, arm_vertices in arm:
                distance = convex_projection_distance_lower_bound(hand_vertices, arm_vertices)
                assert distance > .025, (hand_name, arm_name, distance)
        # Link7's wrist attachment is expected to adjoin the fixed hand wrist
        # and metacarpal body; the articulated fingers must nevertheless clear it.
        forearm = [(name, v) for name, v, _ in parts if name.endswith('_link7')]
        fingers = [(name, v) for name, v in hand if name not in (side+'_hand_wrist', side+'_hand_C_MC')]
        for hand_name, hand_vertices in fingers:
            for arm_name, arm_vertices in forearm:
                assert convex_projection_distance_lower_bound(hand_vertices, arm_vertices) > .025, (hand_name, arm_name)


def test_setback_bases_are_shared_with_pedestal_mount_targets():
    geometry = expanded_geometry(GAMES[2])
    bases = arm_base_poses(geometry)
    assert ROBOT_BASE_DISTANCE_M == pytest.approx(.60)
    assert ROBOT_BASE_HEIGHT_M == pytest.approx(.90)
    for side, sign in (('left', -1), ('right', 1)):
        assert bases[side]['position_m'] == pytest.approx([.65, sign*.60, .90])
        assert abs(bases[side]['position_m'][1])-.52 == pytest.approx(.08)
        assert np.linalg.norm(bases[side]['quaternion_wxyz']) == pytest.approx(1.)


@pytest.mark.parametrize('game_id', GAMES)
def test_fixed_wrist_views_cover_lateral_supply_stations_and_board(robot, model, meshes, game_id):
    geometry = expanded_geometry(game_id)
    initial, _, cameras = ready_configuration(robot, geometry)
    for side, color in (('left', 'black'), ('right', 'white')):
        single = model.sides[side]
        public = single.public_geometry({n: initial['joint_positions_rad'][n] for n in single.names})
        spec = cameras['wrist_'+side]
        camera = expected_camera_world(spec, public['wrist'])
        points = [[*geometry.reserve_position(color, i)[:2], geometry.reserve_surface_z_m+geometry.piece_height_m]
                  for i in range(geometry.inventory_counts[color])]
        points += [[*geometry.cell_center(q, r)[:2], geometry.board_top_z_m+.0001]
                   for q in range(geometry.size) for r in range(geometry.size)]
        for point in points:
            xyz = (np.r_[point, 1.] @ np.linalg.inv(camera))[:3]
            assert xyz[2] < -.1
            half = np.array([spec['horizontal_aperture_mm'], spec['vertical_aperture_mm']])/2
            ndc = xyz[:2]/-xyz[2]*spec['focal_length_mm']/half
            assert np.abs(ndc).max() < .96
            assert segment_hit(camera[3, :3], np.asarray(point), meshes) is None
