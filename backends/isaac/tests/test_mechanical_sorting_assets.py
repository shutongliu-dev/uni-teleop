import json
import numpy as np
import pytest

from bench.paths import load_json
from scenes.workrooms.mechanical_assembly.tasks.parts_sorting.assets import prepare_assets
from scenes.workrooms.mechanical_assembly.tasks.parts_sorting.task import build_task_inputs, contact_probe_inputs


@pytest.fixture(scope='module')
def library(tmp_path_factory):
    root = tmp_path_factory.mktemp('sorting')/'v1'
    prepare_assets(root)
    return root


def test_shapes_have_integrated_mass_and_open_trays(library):
    import trimesh
    manifest = load_json(library/'manifest.json')
    assert len(manifest['assets']) == 6
    for name in ('block', 'cylinder', 'ring'):
        row = manifest['assets']['sorting_'+name]
        assert row['mass_kg'] > 0 and min(row['inertia_diag_kg_m2']) > 0
        assert row['mass_kg'] == pytest.approx(row['collision_volume_m3']*2700)
        assert np.linalg.norm(row['center_of_mass_m']) < 1e-10
        tray = manifest['assets']['sorting_tray_'+name]
        assert len(tray['collision_parts']) == 5 and not tray['dynamic']
    ring = manifest['assets']['sorting_ring']
    cylinder = manifest['assets']['sorting_cylinder']
    assert len(cylinder['collision_parts']) == 4
    for piece in cylinder['collision_parts']:
        mesh = trimesh.load_mesh(library/piece['mesh_path'], process=False)
        assert len(mesh.vertices) <= 64
        planes = np.c_[mesh.face_normals, -np.einsum('ij,ij->i', mesh.face_normals, mesh.triangles_center)]
        assert len(np.unique(np.round(planes, 9), axis=0)) <= 32
    assert cylinder['mass_kg'] == pytest.approx(96/2*np.sin(2*np.pi/96)*.03**2*.045*2700)
    assert len(ring['collision_parts']) == 96
    for piece in ring['collision_parts']:
        mesh = trimesh.load_mesh(library/piece['mesh_path'], process=False)
        # Every actual halfspace convex piece lies outside the clear bore.
        inner = np.asarray(mesh.vertices)
        radial = np.linalg.norm(inner[:, :2], axis=1)
        chord = radial.min()*np.cos(np.pi/96)
        assert chord >= .0175-1e-9
    assert manifest['assets']['sorting_block']['mass_kg'] == pytest.approx(.060*.060*.045*2700)


def test_task_and_gravity_probe_have_separate_resets_and_no_score_claim(library):
    task = build_task_inputs(asset_root=library)
    probe = contact_probe_inputs(asset_root=library)
    assert {o['id'] for o in task['scene']['objects']} == {'block', 'cylinder', 'ring'}
    assert all(o['position_m'][0] == .27 for o in task['scene']['objects'])
    assert all(o['position_m'][0] == .55 for o in probe['scene']['objects'])
    assert not probe['diagnostic']['manipulation_score_claimed']
    assert task['scene']['robot'] == probe['scene']['robot']
    assert task['scene']['isaacsim'] == probe['scene']['isaacsim']
    assert task['initial_state'] == probe['initial_state']
    with pytest.raises(FileExistsError):
        prepare_assets(library)


def test_supply_parts_and_matching_trays_keep_separate_lanes(library):
    value = build_task_inputs(asset_root=library)
    scene, task = value['scene'], value['task']
    by_id = {row['id']: row for row in scene['objects'] + scene['fixtures']}
    bounds = []
    for name, lane_y in [('block', -.17), ('cylinder', 0.), ('ring', .17)]:
        obj, tray = by_id[name], by_id['tray_'+name]
        assert obj['position_m'][:2] == [.27, lane_y]
        assert tray['position_m'][:2] == [.55, lane_y]
        assert task['goals'][name]['center_xy_m'] == tray['position_m'][:2]
        assert task['evaluation']['initial_objects'][name]['position_m'] == obj['position_m']
        vertices = np.asarray(task['evaluation']['geometry_vertices_local_m'][name])
        world = vertices + obj['position_m']
        bounds.append((world.min(axis=0), world.max(axis=0)))
    # The initial source geometry must not overlap another loose part.
    for index, (low, high) in enumerate(bounds):
        for other_low, other_high in bounds[index+1:]:
            assert np.any(high < other_low) or np.any(other_high < low)
