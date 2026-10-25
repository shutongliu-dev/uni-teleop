"""Real exported solids and five frozen reset compositions; CPU only."""
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import trimesh

from bench.paths import DATA_ROOT, load_json
from scenes.workrooms.mechanical_assembly.distractor_assets import (
    prepare_assets, NUT_ID, WASHER_ID, BOLT_ID, STEEL_DENSITY_KG_M3)
from scenes.workrooms.mechanical_assembly.presentation_reset import remove_supply_pads
from scenes.workrooms.mechanical_assembly.table_distractors import add_distractors


RULES_PARENT_RELEASES = Path(f'{DATA_ROOT}/datasets/fr3_sharpa_bench/development/mechanical_referee_20261018/rules_v1/releases')


@pytest.fixture(scope='module')
def assets(tmp_path_factory):
    root = tmp_path_factory.mktemp('table_hardware')/'v1'
    prepare_assets(output=root)
    return root,load_json(root/'manifest.json')


def _solid_angle(mesh,point):
    vectors = mesh.triangles-np.asarray(point)
    a,b,c = vectors[:,0],vectors[:,1],vectors[:,2]
    la,lb,lc = np.linalg.norm(vectors,axis=2).T
    numerator = np.einsum('ij,ij->i',a,np.cross(b,c))
    denominator = la*lb*lc + np.einsum('ij,ij->i',a,b)*lc
    denominator += np.einsum('ij,ij->i',a,c)*lb + np.einsum('ij,ij->i',b,c)*la
    return float(np.sum(2*np.arctan2(numerator,denominator)))


def test_exported_holes_hex_flats_and_geometry_mass(assets):
    root,manifest = assets
    assert set(manifest['assets']) == {NUT_ID,WASHER_ID,BOLT_ID}
    for asset_id,record in manifest['assets'].items():
        mesh = trimesh.load_mesh(root/record['mesh_path'],process=True)
        assert mesh.is_watertight and mesh.is_winding_consistent
        assert record['dynamic'] is True and not record['mass_properties_are_measured']
        assert record['mass_kg'] == pytest.approx(mesh.volume*STEEL_DENSITY_KG_M3,rel=2.e-7)
        assert record['center_of_mass_m'] == pytest.approx(mesh.center_mass,abs=2.e-10)
        assert record['inertia_diag_kg_m2'] == pytest.approx(np.diag(mesh.moment_inertia)*STEEL_DENSITY_KG_M3,rel=2.e-7)
        assert record['inertia_method'] != 'Uniform-density bounding-box approximation, not measured inertia'
        collision_volume = 0.
        for part in record['collision_parts']:
            cell = trimesh.load_mesh(root/part['mesh_path'],process=True)
            assert cell.is_watertight and cell.is_convex
            collision_volume += cell.volume
            if asset_id in (NUT_ID,WASHER_ID):
                # Independently test exported convex cells against the whole
                # central bore, including angular midpoint samples.
                from scipy.spatial import ConvexHull
                equations = ConvexHull(cell.vertices).equations
                for height in np.linspace(mesh.bounds[0,2]+1.e-5,mesh.bounds[1,2]-1.e-5,5):
                    angles = np.arange(97)*2*np.pi/97
                    samples = np.c_[.00649*np.cos(angles),.00649*np.sin(angles),np.full(97,height)]
                    assert not np.any(np.all(samples@equations[:,:3].T+equations[:,3] <= 0.,axis=1))
        assert collision_volume == pytest.approx(mesh.volume,rel=2.e-7)
        if asset_id in (NUT_ID,WASHER_ID):
            assert mesh.euler_number == 0  # One real through hole in the closed shell.
            z = .005 if asset_id == NUT_ID else .00125
            assert abs(_solid_angle(mesh,[0.,0.,z])) < 1.e-10
            assert abs(_solid_angle(mesh,[.008,0.,z])) == pytest.approx(4*np.pi,abs=1.e-9)
        if asset_id == NUT_ID:
            normals = np.c_[np.cos(np.pi/6+np.arange(6)*np.pi/3),np.sin(np.pi/6+np.arange(6)*np.pi/3)]
            assert (mesh.vertices[:,:2]@normals.T).max(axis=0) == pytest.approx(np.full(6,.0095),abs=1.e-10)
        if asset_id == BOLT_ID:
            assert mesh.bounds[:,2] == pytest.approx([0.,.032],abs=1.e-10)
            assert len(record['visual_parts']) == 2 and len(record['collision_parts']) == 5
    assert load_json(root/'SOURCE.json')['license'] == 'CC0-1.0'
    assert (root/'LICENSE.txt').is_file()
    with pytest.raises(FileExistsError): prepare_assets(root)


def _inputs(task_id):
    root = RULES_PARENT_RELEASES/task_id
    scene,task = (load_json(root/name) for name in ('scene.json','task.json'))
    if task_id != 'gear_train_assembly': scene,task,_ = remove_supply_pads(scene,task)
    return scene,task


@pytest.mark.parametrize('task_id',['parts_sorting','peg_insertion','shaft_spacer_assembly','fastener_assembly','gear_train_assembly'])
def test_five_real_resets_unscored_nonoverlap_and_actual_bottom(assets,task_id):
    root,_ = assets
    scene,task = _inputs(task_id)
    before = deepcopy((scene,task))
    changed,new_task,report = add_distractors(scene,task,asset_root=root)
    assert (scene,task) == before
    assert changed['objects'][:len(scene['objects'])] == scene['objects']
    assert changed['fixtures'] == scene['fixtures']
    assert changed['robot'] == scene['robot'] and changed['physics'] == scene['physics']
    assert changed['isaacsim'] == scene['isaacsim']
    assert new_task['evaluation'] == task['evaluation']
    assert new_task['public_rules'] == task['public_rules'] and new_task['initial_state'] == task['initial_state']
    ids = set(changed['task_distractors'])
    assert len(ids) == 5 and ids.isdisjoint(new_task['evaluation']['initial_objects'])
    assert set(new_task['distractors']['ids']) == ids and new_task['distractors']['scored'] is False
    all_bounds = []
    for spec in changed['objects'][-5:]:
        assert spec['id'] in ids
        evidence = report['items'][spec['id']]
        bounds = np.array(evidence['collision_bounds_world_m']);all_bounds.append(bounds)
        assert bounds[0,2] == pytest.approx(.7802,abs=1.e-10)
        assert np.all(bounds[0,:2] >= [.17,-.48]) and np.all(bounds[1,:2] <= [1.13,.48])
        assert min(evidence['xy_gaps_m'].values()) >= .020
        assert 'robot_stand' in evidence['xy_gaps_m']
    for first in range(5):
        for second in range(first):
            a,b=all_bounds[first],all_bounds[second]
            separation=np.maximum(0.,np.maximum(a[0,:2]-b[1,:2],b[0,:2]-a[1,:2]))
            assert np.linalg.norm(separation) >= .020
    # Loose, separated groups replace the old long presentation row. Check
    # the spatial intent as well as the unchanged 20 mm collision margin.
    centers = np.array([(bounds[0]+bounds[1])/2 for bounds in all_bounds])
    assert sum(centers[:,1] < -.08) == 3 and sum(centers[:,1] > .08) == 2
    assert min(item['minimum_existing_xy_gap_m'] for item in report['items'].values()) < .070
    if task_id == 'fastener_assembly':
        # The source screwdriver has cm/Y fallback layer metadata but is a
        # 211 mm tool with metre point values. Referencing it must preserve the
        # physical placement and source scale, not shrink/rotate by metadata.
        from scenes.workrooms.mechanical_assembly.table_distractors import _bounds
        screwdriver = next(row for row in scene['objects'] if row['id']=='screwdriver')
        bounds,_ = _bounds(screwdriver)
        # This frozen reset has a small tool tilt; it adds about 4.5 microns
        # to the world AABB height without changing the source tool dimensions.
        assert bounds[1,2]-bounds[0,2] == pytest.approx(.211,abs=1.e-5)
    assert not report['native_validation_claimed']
    with pytest.raises(ValueError,match='already declared'):
        add_distractors(changed,new_task,root)


def test_occupied_reserved_area_rejected_without_mutation(assets):
    root,_ = assets
    scene,task = _inputs('peg_insertion')
    peg = next(row for row in scene['objects'] if row['id']=='peg_0')
    peg['position_m'][:2] = [.965,-.235]
    task['evaluation']['initial_objects']['peg_0']['position_m'] = deepcopy(peg['position_m'])
    before = deepcopy((scene,task))
    with pytest.raises(ValueError,match='insufficient collision-bounds clearance'):
        add_distractors(scene,task,root)
    assert (scene,task) == before
