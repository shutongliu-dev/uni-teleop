"""New full-SE3 mounting uses the supplied physical chain, including its offset."""
import hashlib
import io
from pathlib import Path
import zipfile

import numpy as np
import pytest

from bench.control.kinematics import URDFKinematics
from bench.geometry import quaternion_matrix, transform
from bench.paths import DATA_ROOT, load_json
from bench.simulation.isaacsim_backend.l_mount import MOUNTS, UPLOAD_SHA256, VERSION
from bench.simulation.isaacsim_backend.robot_registry import get_hand_spec, scene_with_hand


def test_instance_public_mounts_keep_two_right_hands_and_actual_names():
    from bench.simulation.isaacsim_backend.l_mount import public_mount
    robot = {'hand_id': 'fr3_sharpa_instances', 'assembly_version': 'namespaced_v1',
        'instances': {name: {'hand_id': 'sharpa_w01_right', 'assembly_version': VERSION,
            'base_position_m': [0., 0., 0.], 'base_quaternion_wxyz': [1., 0., 0., 0.]}
            for name in ('front', 'rear')}}
    mounts = public_mount(robot)['mounts_by_side']
    assert mounts == {name: MOUNTS['right'] for name in ('front', 'rear')}
    robot['instances']['rear']['assembly_version'] = 'isaacsim_6_1_v2'
    assert public_mount(robot) is None
    assert public_mount({'hand_id': 'sharpa_w01_right', 'assembly_version': 'isaacsim_6_1_v2'}) is None


def uploaded_urdf():
    """The uploaded L-mount URDF: from the user's archive when present, else from its extraction
    in the data package, which RAW_SOURCE.json binds to the same archive and member."""
    member = 'franka_sharpa_l_mount/fr3_sharpa_right_l_mount.urdf'
    archive = Path(f'{DATA_ROOT}/assets/robot_uploads/20261104/franka_sharpa_l_mount.zip')
    if archive.is_file():
        assert hashlib.sha256(archive.read_bytes()).hexdigest() == UPLOAD_SHA256
        with zipfile.ZipFile(archive) as z:
            return z.read(member)
    inventory = load_json(Path(f'{DATA_ROOT}/assets/robots/assemblies/fr3_sharpa_l_mount/upload_20261104/RAW_SOURCE.json'))
    assert inventory['archive']['sha256'] == UPLOAD_SHA256
    path, row = next((path, row) for path, row in inventory['files'].items() if row['archive_member'] == member)
    data = Path(path).read_bytes()
    assert hashlib.sha256(data).hexdigest() == row['sha256']
    return data


def test_output_frame_reproduces_uploaded_palm_at_arbitrary_arm_configurations():
    chain = URDFKinematics(io.BytesIO(uploaded_urdf()))
    mount = MOUNTS['right']
    offset = transform(mount['translation_m'], quaternion_matrix(mount['quaternion_wxyz']))
    flange_to_palm = transform([0., 0., .035], np.eye(3))
    rng = np.random.default_rng(941)
    for _ in range(17):
        q = {j.name: float(rng.uniform(j.lower, j.upper)) for j in chain.active}
        poses = chain.forward(q)
        assert np.allclose(poses['fr3_link8'] @ offset @ flange_to_palm,
                           poses['right_hand_C_MC'], atol=1e-12, rtol=0.)


def test_left_mount_reflects_both_frames_without_an_improper_rotation():
    mirror = np.diag([1., -1., 1.])
    right, left = MOUNTS['right'], MOUNTS['left']
    a, b = (quaternion_matrix(m['quaternion_wxyz']) for m in (right, left))
    assert np.allclose(b, mirror @ a @ mirror, atol=1e-14, rtol=0.)
    assert np.linalg.det(a) == pytest.approx(1.)
    assert np.linalg.det(b) == pytest.approx(1.)
    assert np.allclose(left['translation_m'], mirror @ right['translation_m'], atol=1e-14)


def test_new_scene_declares_both_distinct_mounts_and_preserves_old_pins():
    spec = get_hand_spec('dual_fr3_sharpa_w01', VERSION)
    scene = scene_with_hand({'robot': {}}, 'dual_fr3_sharpa_w01', VERSION)
    assert scene['robot']['mounts_by_side'] == MOUNTS
    assert spec['hand_frame_yaw_rad'] is None
    assert spec['mounts_by_side']['left'] != spec['mounts_by_side']['right']
    with pytest.raises(ValueError, match='cannot re-pin'):
        scene_with_hand({'robot': {'hand_id': 'sharpa_w01_right', 'assembly_version': 'isaacsim_6_1_v2'}},
                        'sharpa_w01_right', VERSION)


def test_agent_wording_uses_actual_interface_instead_of_old_identity():
    from agent.model_client.runner import mount_prompt, MOUNT_V2_SENTENCES
    prompt = '\n'.join(old for old, _ in MOUNT_V2_SENTENCES)
    actual = mount_prompt(prompt, {'mount_convention': 'l_adapter_20261104_v1'})
    assert 'identically mounted' not in actual
    assert '35 mm' in actual and 'L adapter output' in actual


@pytest.fixture(scope='module')
def prepared_dual():
    from bench.control.factory import public_robot_geometry
    assets = f'{DATA_ROOT}/assets'
    if not (Path(assets)/'robots/assemblies/dual_fr3_sharpa_w01'/VERSION/'SOURCE_left.json').is_file():
        pytest.skip('New robot source assemblies are not installed')
    return public_robot_geometry({'assets_root': assets, 'base_position_m': [0., 0., 0.],
        'hand_id': 'dual_fr3_sharpa_w01', 'assembly_version': VERSION})


def test_prepared_geometry_pads_and_structural_pairs_are_source_bound(prepared_dual):
    from bench.runtime.pad_geometry import source_pad_geometries
    from bench.simulation.isaacsim_backend.robot_asset import collision_configuration
    scene = scene_with_hand({'robot': {}}, 'dual_fr3_sharpa_w01', VERSION)
    pairs = collision_configuration(scene)['excluded_link_pairs']
    assert len(pairs) == 4
    for side, model in prepared_dual.sides.items():
        assert len(model.arm.names) == 7 and len(model.hand.names) == 22
        assert len(source_pad_geometries(model)) == 5
        assert [f'fr3_{side}_link7', f'fr3_{side}_l_adapter'] in pairs
        assert [f'{side}_hand_C_MC', f'{side}_thumb_MC'] in pairs


def test_runtime_checks_physical_inputs_without_reading_archived_reports(prepared_dual, monkeypatch):
    import json
    from bench.simulation.isaacsim_backend import l_mount
    model = prepared_dual.sides['right']
    source = load_json(model.arm_path.parent/'SOURCE_right.json')
    inputs = l_mount.runtime_source_inputs(source, f'{DATA_ROOT}/assets')
    assert set(source['mesh_inputs']) <= inputs.keys()
    assert source['import_urdf_path'] in inputs and str(model.hand_path) in inputs
    assert not any(p.endswith(('.zip', '.py', '.log', 'simulation_report.json')) for p in inputs)
    archived = set(source['sources']) - inputs.keys()
    assert any(p.endswith('simulation_report.json') for p in archived)
    original = l_mount._sha
    def physical_only(path):
        assert str(path) not in archived, 'Startup read an archival-only dependency'
        return original(path)
    monkeypatch.setattr(l_mount, '_sha', physical_only)
    l_mount.prepared_assembly(f'{DATA_ROOT}/assets', side='right')


@pytest.mark.parametrize('side', ['left', 'right'])
def test_new_reference_profile_certifies_whole_amount_interval(prepared_dual, side):
    from bench.control.hand_presets import generate_hand_preset, hand_preset_catalog
    from bench.protocol.hand_presets import HandPresetResult
    model = prepared_dual.sides[side]
    catalog = hand_preset_catalog(prepared_dual)
    assert catalog.library_version == 'sharpa_l_mount_preposes_v1' and catalog.sides == ['left', 'right']
    endpoint = generate_hand_preset(prepared_dual, {'side': side, 'preset': 'pinch_prepose'})
    by_child = {j.child: j for j in model.hand.joints}
    speed = {}
    for finger, tip in model.fingertip_links.items():
        radius = total = 0.
        while tip in by_child:
            joint = by_child[tip]
            total += abs(endpoint.hand_posture_rad.get(joint.name, 0.)) * radius
            radius += np.linalg.norm(joint.origin[:3, 3])
            tip = joint.parent
        speed[finger] = total
    minimum = np.inf
    for amount in np.linspace(0., 1., 257):
        row = generate_hand_preset(prepared_dual, {'side': side, 'preset': 'pinch_prepose', 'amount': float(amount)})
        points = list(row.fingertip_poses.items())
        for index, (finger, point) in enumerate(points):
            for other, other_point in points[:index]:
                bound = np.linalg.norm(np.array(point.position_m) - other_point.position_m)
                minimum = min(minimum, bound - (speed[finger] + speed[other]) / 512.)
        assert not row.validation.collision_checked and not row.validation.native_grasp_qualified
    assert minimum >= row.validation.all_amount_virtual_tip_separation_lower_bound_m
    assert HandPresetResult.model_validate_json(endpoint.model_dump_json()) == endpoint
