"""Embodiment capabilities, provenance and physical asset semantics, without GPU."""
from copy import deepcopy
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
import pytest
from bench.paths import DATA_ROOT, load_json
from bench.simulation.isaacsim_backend.robot_registry import get_hand_spec, list_hands, scene_with_hand
from bench.simulation.isaacsim_backend.robot_asset import collision_configuration
from bench.simulation.isaacsim_backend.multihand_asset import assemble_hand_urdf
from bench.simulation.isaacsim_backend.env import FR3DexterousIsaacEnv
from bench.simulation.isaacsim_backend.scene_variants import apply_scene_variant
from bench.simulation.isaacsim_backend.asset_catalog import load_task_asset

SCENE=Path(f'{DATA_ROOT}/datasets/fr3_sharpa_bench/isaacsim/releases/sequential_workcell/v1/tasks/01_ordered_kitting/scene.json')


def test_capabilities_distinguish_active_and_physical_joints():
    specs={x['hand_id']:x for x in list_hands()}
    assert {k:(v['joint_count'],v['actuated_joint_count']) for k,v in specs.items()}=={
      'sharpa_w01_right':(29,29),'allegro_v5_right':(23,23),'leap_v1_right':(23,23),'shadow_hand_e_right':(31,27),
      'dual_fr3_sharpa_w01':(58,58)}
    assert specs['shadow_hand_e_right']['hardware_actuator_equivalent'] is False
    with pytest.raises(ValueError):get_hand_spec('guess')


@pytest.mark.parametrize('hand_id',['allegro_v5_right','leap_v1_right','shadow_hand_e_right'])
def test_assemblies_preserve_tree_meshes_and_valid_physical_limits(hand_id,tmp_path):
    path,source=assemble_hand_urdf(hand_id,output=tmp_path)
    root=ET.parse(path).getroot();links={x.get('name') for x in root.findall('link')}
    parents={j.find('child').get('link'):j.find('parent').get('link') for j in root.findall('joint')}
    assert links-set(parents)=={'base'}
    assert len(source['joint_limits'])==get_hand_spec(hand_id)['joint_count']
    for start in parents:
        seen=set();node=start
        while node in parents:
            assert node not in seen;seen.add(node);node=parents[node]
        assert node=='base'
    assert all(Path(m.get('filename')).is_file() for m in root.findall('.//mesh'))
    assert len(root.findall('.//mimic'))==(4 if hand_id=='shadow_hand_e_right' else 0)
    for v in source['joint_limits'].values():assert v['lower']<v['upper'] and v['effort']>0 and v['velocity']>0


def test_shadow_followers_are_not_exposed_as_independent_actions(tmp_path):
    _,source=assemble_hand_urdf('shadow_hand_e_right',output=tmp_path)
    env=FR3DexterousIsaacEnv.__new__(FR3DexterousIsaacEnv)
    env.hand_spec=get_hand_spec('shadow_hand_e_right');env.mimic_joints=env.hand_spec['mimic_joints']
    env.joint_names=list(source['joint_limits']);env.actuated_joint_names=[n for n in env.joint_names if n not in env.mimic_joints]
    env.joint_limits={n:[v['lower'],v['upper']] for n,v in source['joint_limits'].items()};env.home={n:float(np.clip(0,*env.joint_limits[n])) for n in env.joint_names}
    spec=env.action_spec();q=spec['initial_joint_positions_rad'];q['rh_FFJ2']=.4
    env._validate_action({'joint_positions_rad':q,'duration_s':.02})
    assert env._expand_mimic_targets(q)['rh_FFJ1']==.4
    assert len(spec['joint_names'])==27 and len(spec['state_joint_names'])==31
    q['rh_FFJ1']=.2
    with pytest.raises(ValueError):env._validate_action({'joint_positions_rad':q,'duration_s':.02})


def test_scene_variants_keep_task_inputs_and_all_added_physics():
    scene=load_json(SCENE);original=deepcopy(scene);variant=apply_scene_variant(scene,'enriched_v2')
    assert scene==original
    assert variant['objects'][:len(scene['objects'])]==scene['objects']
    assert variant['fixtures'][:len(scene['fixtures'])]==scene['fixtures']
    for item in variant['objects'][len(scene['objects']):]+variant['fixtures'][len(scene['fixtures']):]:
        asset=load_task_asset(item['asset_id'],item['asset_root'],verify_hashes=True)
        assert asset['collision_parts'] and asset['visual_parts'] and asset['mass_kg']>0
        assert min(asset['inertia_diag_kg_m2'])>0
    with pytest.raises(ValueError):apply_scene_variant(variant,'enriched_v2')


def test_allegro_structural_filters_do_not_disable_cross_finger_collisions():
    scene=scene_with_hand(load_json(SCENE),'allegro_v5_right');config=collision_configuration(scene)
    assert config['enabled'] and len(config['excluded_link_pairs'])==4
    assert all(a=='allegro_palm_link' for a,b in config['excluded_link_pairs'])
    assert config['cross_finger_exclusions']==[] and config['exclusion_source']['sha256']


def test_shadow_structural_filters_and_conditioning_are_explicit():
    spec=get_hand_spec('shadow_hand_e_right')
    config=collision_configuration(scene_with_hand(load_json(SCENE),spec['hand_id']))
    assert config['enabled'] and len(config['excluded_link_pairs'])==7
    assert config['cross_finger_exclusions']==[]
    assert spec['hand_armature_kg_m2']==.0001 and 'not measured' in spec['armature_status']


def test_leap_home_uses_upstream_gym_to_urdf_order():
    import ast
    root=Path(f'{DATA_ROOT}/assets/robots/leap/v1_150bc3d/upstream/leapsim')
    tree=ast.parse((root/'tasks/leap_hand_grasp.py').read_text())
    poses=[ast.literal_eval(n.value) for n in ast.walk(tree) if isinstance(n,ast.Assign) and isinstance(n.value,ast.List) and len(n.value.elts)==16]
    line=next(x for x in (root/'cfg/task/LeapHandRot.yaml').read_text().splitlines() if x.strip().startswith('sim_to_real_indices:'))
    indices=ast.literal_eval(line.split(':',1)[1].strip())
    actual=get_hand_spec('leap_v1_right')['home_joint_positions_rad']
    assert actual=={f'leap_{i}':poses[0][indices[i]] for i in range(12,16)}


def test_scene_pins_the_assembly_version_of_its_hand():
    from bench.simulation.isaacsim_backend.robot_registry import DEFAULT_HAND
    base = load_json(SCENE)
    leap = scene_with_hand(base, 'leap_v1_right')
    assert leap['robot']['assembly_version'] == get_hand_spec('leap_v1_right')['assembly_version']
    assert scene_with_hand(leap, 'leap_v1_right') == leap                 # idempotent
    stale = deepcopy(leap)
    stale['robot']['assembly_version'] = 'isaacsim_6_1_v5'
    with pytest.raises(ValueError, match='pins leap_v1_right assembly'):
        scene_with_hand(stale, 'leap_v1_right')
    # Another hand's pin is an explicit embodiment change, not a conflict.
    assert scene_with_hand(stale, 'allegro_v5_right')['robot']['assembly_version'] == get_hand_spec('allegro_v5_right')['assembly_version']
    # The default SHARPA scene is unchanged (no new key; frozen v8 scene hashes stay valid).
    assert 'assembly_version' not in scene_with_hand(base, DEFAULT_HAND)['robot']


def test_leap_thumb_pip_palm_exclusion_is_an_audited_release_opt_in(tmp_path):
    import hashlib
    from bench.simulation.isaacsim_backend.robot_asset import LEAP_THUMB_PIP_PALM_AUDIT
    base = load_json(SCENE)
    leap = scene_with_hand(base, 'leap_v1_right')
    # Runtime scene_with_hand never adds it: frozen v9 r1/r2 LEAP scenes keep their physics.
    assert 'filter_leap_thumb_pip_palm' not in leap['isaacsim']
    assert collision_configuration(leap)['excluded_link_pairs'] == []
    opted = deepcopy(leap)
    opted['isaacsim'].update(get_hand_spec('leap_v1_right')['release_collision_filters'])
    assert scene_with_hand(opted, 'leap_v1_right')['isaacsim']['filter_leap_thumb_pip_palm'] is True   # preserved
    config = collision_configuration(opted)
    assert config['excluded_link_pairs'] == [['leap_palm_lower', 'leap_thumb_pip']]
    assert config['exclusion_source']['audit']['sha256'] == LEAP_THUMB_PIP_PALM_AUDIT['sha256']
    audit = Path(LEAP_THUMB_PIP_PALM_AUDIT['path'])
    if audit.is_file():
        assert hashlib.sha256(audit.read_bytes()).hexdigest() == LEAP_THUMB_PIP_PALM_AUDIT['sha256']
        gap = np.asarray(load_json(audit)['gap_m'])
        q12 = np.asarray(load_json(audit)['leap_12_rad'])
        interior = (q12 > -.2) & (q12 < 1.8)
        assert gap[interior].min() > .001            # the true meshes never meet inside the range
    for bad in (scene_with_hand(base, 'allegro_v5_right'), deepcopy(opted)):
        bad['isaacsim']['filter_leap_thumb_pip_palm'] = True
        if bad['robot']['hand_id'] == 'leap_v1_right':
            bad['isaacsim']['robot_self_collisions'] = False
        with pytest.raises(ValueError, match='thumb_pip/palm'):
            collision_configuration(bad)


def test_leap_exclusion_reaches_the_runtime_scene_through_the_release_profile(tmp_path):
    # task_inputs() replaces scene['isaacsim'] with the release physics profile, and the
    # worker then applies scene_with_hand: the opt-in must survive that exact path.
    from bench.isaacsim.release import task_inputs
    from scenes.manipulation.embodiment_release import SOURCE, prepare
    if not SOURCE.is_dir():
        pytest.skip('v8 source release not present')
    prepare('leap_v1_right', output=tmp_path / 'leap')
    _, scene, _, _ = task_inputs(tmp_path / 'leap', '02_cube_lift_hold')
    runtime = scene_with_hand(scene, 'leap_v1_right')
    assert collision_configuration(runtime)['excluded_link_pairs'] == [['leap_palm_lower', 'leap_thumb_pip']]
    prepare('allegro_v5_right', output=tmp_path / 'allegro')
    _, scene, _, _ = task_inputs(tmp_path / 'allegro', '02_cube_lift_hold')
    assert 'filter_leap_thumb_pip_palm' not in scene['isaacsim']
    old = Path(f'{DATA_ROOT}/datasets/fr3_sharpa_bench/isaacsim/releases/simple_manipulation/v9_embodiments_r2/leap_v1_right')
    if old.is_dir():                                   # frozen r2 keeps its physics
        _, scene, _, _ = task_inputs(old, '02_cube_lift_hold')
        assert collision_configuration(scene_with_hand(scene, 'leap_v1_right'))['excluded_link_pairs'] == []


def test_versioned_assembly_files_are_created_once(tmp_path):
    # Every import used to rewrite the identical URDF in place; a concurrent import then
    # hashed a truncated file (dual_contact_replay_04: 'provenance mismatch: urdf_sha256').
    import os
    from bench.simulation.isaacsim_backend.robot_asset import write_once
    path = tmp_path / 'robot.urdf'
    write_once(path, b'<robot/>\n', 'changed')
    os.utime(path, (1, 1))
    write_once(path, b'<robot/>\n', 'changed')
    assert path.stat().st_mtime == 1 and path.read_bytes() == b'<robot/>\n'
    with pytest.raises(FileExistsError, match='changed'):
        write_once(path, b'<robot name="other"/>\n', 'changed')
    assert sorted(p.name for p in tmp_path.iterdir()) == ['robot.urdf']
    from bench.simulation.isaacsim_backend.robot_asset import assemble_urdf
    first = assemble_urdf(output=tmp_path / 'sharpa')[0]
    os.utime(first, (1, 1))
    assemble_urdf(output=tmp_path / 'sharpa')
    assert first.stat().st_mtime == 1
