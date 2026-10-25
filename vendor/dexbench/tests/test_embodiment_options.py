"""Opt-in embodiment physics options; CPU checks with a fake articulation view, not physics acceptance."""
import math
from pathlib import Path

import numpy as np
import pytest

from bench.paths import DATA_ROOT, load_json
from bench.simulation.isaacsim_backend import robot_registry
from bench.simulation.isaacsim_backend.robot_registry import (HAND_ACTUATOR_MODELS, SHARPA_ACTUATOR_SOURCE,
                                                        hand_actuator_values)

ASSETS = Path(f'{DATA_ROOT}/assets')
HAND = ['right_thumb_CMC_FE', 'right_thumb_CMC_AA', 'right_thumb_MCP_FE', 'right_thumb_MCP_AA', 'right_thumb_IP',
        *[f'right_{f}_{j}' for f in ('index', 'middle', 'ring') for j in ('MCP_FE', 'MCP_AA', 'PIP', 'DIP')],
        'right_pinky_CMC', *[f'right_pinky_{j}' for j in ('MCP_FE', 'MCP_AA', 'PIP', 'DIP')]]


def test_position_mode_model_covers_every_hand_joint_and_never_the_arm():
    values = hand_actuator_values('sharpa_w01_right', 'sharpa_official_position_mode_v1',
                                  [f'fr3_joint{i}' for i in range(1, 8)] + HAND)
    assert set(values) == set(HAND) and len(values) == 22
    assert all(set(row) == {'kp', 'kd', 'armature'} for row in values.values())
    assert values['right_index_DIP']['kp'] == pytest.approx(.904127400716439)
    assert values['right_thumb_CMC_AA']['kd'] == pytest.approx(.45163971158982275)


def test_friction_is_only_in_the_explicit_probe_model():
    assert 'joint_friction' not in HAND_ACTUATOR_MODELS['sharpa_official_position_mode_v1']
    probe = hand_actuator_values('sharpa_w01_right', 'sharpa_official_position_mode_joint_friction_probe_v1', HAND)
    assert probe['right_index_MCP_FE']['joint_friction'] == pytest.approx(.104)


@pytest.mark.parametrize('hand,model', [('allegro_v5_right', 'sharpa_official_position_mode_v1'),
                                        ('sharpa_w01_right', 'uniform_guess')])
def test_unknown_model_or_hand_is_rejected(hand, model):
    with pytest.raises(ValueError):
        hand_actuator_values(hand, model, HAND)


def test_registry_table_matches_the_pinned_vendor_usd():
    path = ASSETS / SHARPA_ACTUATOR_SOURCE['path']
    if not path.is_file():
        pytest.skip('Pinned SHARPA upstream USD is unavailable')
    import hashlib, re
    assert hashlib.sha256(path.read_bytes()).hexdigest() == SHARPA_ACTUATOR_SOURCE['sha256']
    text = path.read_text()
    values = hand_actuator_values('sharpa_w01_right', 'sharpa_official_position_mode_v1', HAND)
    for name, body in re.findall(r'over "(right_[A-Za-z_]+)"\s*\{(.*?)\n        \}', text, re.S):
        field = lambda key: float(re.search(r'float ' + re.escape(key) + r' = ([-0-9.eE]+)', body).group(1))
        if 'drive:angular:physics:stiffness' not in body:
            continue
        # The USD drive gains are per degree; the registry is per radian.
        assert values[name]['kp'] == pytest.approx(field('drive:angular:physics:stiffness') * 180 / math.pi)
        assert values[name]['kd'] == pytest.approx(field('drive:angular:physics:damping') * 180 / math.pi)
        assert values[name]['armature'] == pytest.approx(field('physxJoint:armature'))


def test_legacy_registry_defaults_are_unchanged():
    spec = robot_registry.get_hand_spec('sharpa_w01_right')
    assert spec['hand_kp'] == 3. and spec['hand_kd'] == .1
    assert 'hand_armature_kg_m2' not in spec


def test_v8_release_options_are_explicit_and_scene_bytes_unchanged(tmp_path):
    from scenes.manipulation.fidelity_release import PHYSICS_OPTIONS, SOURCE, prepare
    if not SOURCE.is_dir():
        pytest.skip('Frozen v7 source release is unavailable')
    manifest = prepare(SOURCE, tmp_path / 'v8')
    profile = load_json(tmp_path / 'v8' / manifest['physics_profile'])
    for key, value in PHYSICS_OPTIONS.items():
        assert profile['isaacsim_options'][key] == value
    for entry in manifest['tasks']:
        assert (tmp_path / 'v8' / entry['scene']).read_bytes() == (SOURCE / entry['scene']).read_bytes()
    grasp = load_json(tmp_path / 'v8/tasks/04_cube_grasp_hold/task.json')
    assert grasp['evaluation']['minimum_pinch_normal_force_n'] == pytest.approx(.4265)
    assert grasp['public_goal']['workspace']['table_top_z_m'] == pytest.approx(.75)
    push = load_json(tmp_path / 'v8/tasks/01_cube_push/task.json')
    old = load_json(SOURCE / 'tasks/01_cube_push/task.json')
    assert push['evaluation'] == old['evaluation']


def test_v8_tasks_route_projects_every_task_with_public_facts(tmp_path):
    from types import SimpleNamespace
    from fastapi.testclient import TestClient
    from scenes.manipulation.fidelity_release import SOURCE, prepare
    from bench.runtime.server import create_app
    if not SOURCE.is_dir():
        pytest.skip('Frozen v7 source release is unavailable')
    prepare(SOURCE, tmp_path / 'v8')
    manager = SimpleNamespace(release=tmp_path / 'v8', shutdown=lambda: None)
    with TestClient(create_app(service=manager)) as client:
        response = client.get('/tasks')
    assert response.status_code == 200
    rows = {row['task_id']: row['goal'] for row in response.json()}
    assert len(rows) == 4
    for task_id, goal in rows.items():
        assert goal['workspace']['table_top_z_m'] == pytest.approx(.75)
        assert goal['object_mass_kg'] == pytest.approx(.05)
        assert ('push_grasp_scoring' in goal['grasp_constraints']) == (task_id == '01_cube_push')
    assert rows['04_cube_grasp_hold']['success_contract']['terminal']['minimum_pinch_normal_force_n'] == pytest.approx(.4265)
    assert 'maximum_speed' not in response.text


@pytest.mark.parametrize('root', [0, 6])
def test_gravity_feedforward_uses_dof_columns_for_fixed_or_floating_root(root):
    # PhysX prepends 6 root entries unless the articulation is fixed-base.
    from test_contact_sampling_contract import fake_env

    class Tensor(np.ndarray):
        def contiguous(self):
            return self
        def clone(self):
            return self.copy()
    env = fake_env()
    applied, dofs = [], 29
    gravity = np.arange(root + dofs, dtype=np.float32).reshape(1, -1).view(Tensor)
    env._robot_view.get_gravity_compensation_forces = lambda: gravity
    env._robot_view.set_dof_actuation_forces = lambda forces, indices: applied.append(forces.clone())
    mask = np.zeros((1, dofs), dtype=np.float32).view(Tensor)
    mask[0, :7] = 1.
    env._arm_gravity_mask = mask
    env.step({'joint_positions_rad': {}})
    assert len(applied) == 40 and applied[0].shape == (1, dofs)
    assert applied[0][0, :7].tolist() == gravity[0, root:root + 7].tolist()
    assert float(abs(applied[0][0, 7:]).sum()) == 0.
