"""Backend-independent safety and frozen-task alignment, not physics acceptance."""
from copy import deepcopy
from pathlib import Path
import xml.etree.ElementTree as ET
import subprocess
import sys

import numpy as np
import pytest

from bench.paths import DATA_ROOT, load_json
from bench.simulation.isaacsim_backend.env import FR3DexterousIsaacEnv
from bench.simulation.isaacsim_backend.geometry import cloth_grid
from bench.simulation.isaacsim_backend.robot_asset import assemble_urdf, collision_configuration

RELEASE = Path(f'{DATA_ROOT}/datasets/fr3_sharpa_bench/sequential_workcell/v1')


def test_real_subprocess_task_assets_do_not_import_newton():
    script = '''
import importlib.abc, sys
class RejectNewton(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'newton' or fullname.startswith('newton.') or fullname == 'bench.simulation.newton_backend':
            raise AssertionError('Forbidden physics import: ' + fullname)
sys.meta_path.insert(0, RejectNewton())
from bench.simulation.isaacsim_backend.env import FR3DexterousIsaacEnv
from bench.simulation.isaacsim_backend.asset_catalog import load_task_asset, load_task_catalog, load_obj_mesh
catalog = load_task_catalog()
assert len(catalog['assets']) == 11
for name in catalog['assets']:
    asset = load_task_asset(name, verify_hashes=True)
    vertices, faces = load_obj_mesh(asset['mesh_path'])
    assert len(vertices) and len(faces)
assert not any(x == 'newton' or x.startswith('newton.') or x == 'bench.simulation.newton_backend' for x in sys.modules)
print('11 real task assets loaded; Newton import denied and never attempted')
'''
    result = subprocess.run([sys.executable, '-c', script], cwd=Path(__file__).resolve().parents[1],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert '11 real task assets loaded' in result.stdout


@pytest.fixture
def contract():
    # Deliberately do not start a simulator: these checks cover action ingress.
    env = FR3DexterousIsaacEnv.__new__(FR3DexterousIsaacEnv)
    root = Path(f'{DATA_ROOT}/assets/robots/assemblies/fr3_sharpa/isaacsim_6_1_v1/SOURCE.json')
    limits = load_json(root)['joint_limits']
    env.joint_names = list(limits)
    env.joint_limits = {n: [v['lower'], v['upper']] for n, v in limits.items()}
    home = {n: float(np.clip(0., *env.joint_limits[n])) for n in env.joint_names}
    env.home = home
    return env, {'joint_positions_rad': home, 'duration_s': .02}


def test_action_spec_matches_native_complete_joint_ingress(contract):
    env, action = contract
    spec = env.action_spec()
    env._validate_action(action)
    assert len(spec['arm_joint_names']) == 7
    assert len(spec['hand_joint_names']) == 22
    assert set(spec['arm_joint_names'] + spec['hand_joint_names']) == set(env.joint_names)


def test_self_collision_requires_explicit_override_without_mutating_frozen_scene():
    scene = load_json(RELEASE / 'tasks/01_ordered_kitting/scene.json')
    assert collision_configuration(scene)['enabled'] is False
    scene['isaacsim'] = {'robot_self_collisions': True, 'filter_thumb_cmc_structure': True}
    config = collision_configuration(scene)
    assert scene['robot']['self_collisions'] is False
    assert config['enabled'] is True
    assert config['excluded_link_pairs'] == [['right_hand_C_MC', 'right_thumb_MC']]
    assert config['cross_finger_exclusions'] == []


@pytest.mark.parametrize('field', ['objects', 'cloth_vertices_m', 'reset_override', 'body_q'])
def test_control_rejects_physical_state_writeback(contract, field):
    env, action = contract
    action[field] = {}
    with pytest.raises(ValueError):
        env._validate_action(action)


def test_control_rejects_missing_nonfinite_and_out_of_limit_targets(contract):
    env, action = contract
    env._validate_action(action)
    for value in [float('nan'), float('inf'), True, 99.]:
        bad = deepcopy(action)
        bad['joint_positions_rad']['fr3_joint1'] = value
        with pytest.raises(ValueError):
            env._validate_action(bad)
    del action['joint_positions_rad']['fr3_joint1']
    with pytest.raises(ValueError):
        env._validate_action(action)


@pytest.mark.parametrize('task', ['03_lined_tray_loading', '04_cover_and_secure'])
def test_cloth_topology_matches_recorded_frozen_task(task):
    folder = RELEASE / 'tasks' / task
    scene = load_json(folder / 'scene.json')
    task_spec = load_json(folder / 'task.json')
    points, triangles = cloth_grid(scene['cloth'])
    assert np.array_equal(triangles, np.load(folder / 'validation/initial/triangles.npy'))
    assert np.all(points[task_spec['vertex_groups']['near_edge'], 1] <= scene['cloth']['size_m'][1] * .11)
    assert np.all(points[task_spec['vertex_groups']['far_edge'], 1] >= scene['cloth']['size_m'][1] * .89)


def test_combined_urdf_keeps_one_tree_and_manufacturer_dofs(tmp_path):
    path, source = assemble_urdf(output=tmp_path)
    root = ET.parse(path).getroot()
    links = {x.get('name') for x in root.findall('link')}
    parents = {x.find('child').get('link'): x.find('parent').get('link') for x in root.findall('joint')}
    assert links - set(parents) == {'base'}
    assert len(source['joint_limits']) == 29
    for start in parents:
        name, seen = start, set()
        while name in parents:
            assert name not in seen
            seen.add(name)
            name = parents[name]
        assert name == 'base'
    assert parents['right_hand_flange'] == 'fr3_link8'
    assert all(Path(x.get('filename')).is_file() for x in root.findall('.//mesh'))
