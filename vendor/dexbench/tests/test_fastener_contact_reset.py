"""CPU provenance and fixture-frame guards, not native contact acceptance."""
from copy import deepcopy
from pathlib import Path
import shutil

import numpy as np
import pytest

from bench.geometry import quaternion_matrix
from bench.isaacsim.release import task_inputs
from bench.paths import DATA_ROOT
from scenes.workrooms.mechanical_assembly.tasks.fastener_assembly.definition import LOOSE_BOLT_RELEASE
from scripts.validation.mechanical_fastener_contact_reset import (prepare_contact_reset,
    passive_support_pairs)
from scripts.validation.mechanical_fastener_passive_checks import passive_phase_window
from test_fastener_passive_checks import evidence, PATHS


BASE = Path(f'{DATA_ROOT}/datasets/fr3_sharpa_bench/development/mechanical_task_validation_20261018/fastener')
SOURCES = [BASE/'task_release_sdf1024_convex_partition_pgs_phase_reset_candidate01',
           BASE/'task_release_sdf1024_convex_partition_pgs_phase_reset_halfdt_candidate01']


@pytest.fixture(scope='module')
def canonical():
    _, scene, task, _ = task_inputs(LOOSE_BOLT_RELEASE['path'], 'fastener_assembly')
    return scene, task


def actor(scene, role):
    return next(row for row in scene['objects']+scene['fixtures'] if row['id'] == role)


def test_normal_invocation_keeps_canonical_loose_reset(canonical):
    scene, task = canonical
    selected, diagnostic, receipt = prepare_contact_reset(scene, task)
    assert (selected, diagnostic, receipt) == (scene, task, None)
    assert selected is not scene and diagnostic is not task
    assert actor(selected, 'bolt')['position_m'][:2] == [.56, -.16]


@pytest.mark.parametrize('pair_name', ['bolt_table', 'bolt_supply', 'bolt_fixture'])
def test_actual_bolt_support_gap_cannot_be_bypassed_by_pair_label(pair_name):
    rows, config = evidence()
    supports = {'tool_bolt': ('tool', 'bolt'), pair_name: ('bolt', 'fixture')}
    # Keep positive support in each 20 ms aggregate while missing pressure
    # for 19.5 ms: the frozen 1 ms continuous-gap gate must still fail.
    for i, row in enumerate(rows):
        if i % 40:
            row['raw_contacts'][1]['normal_impulses_ns'] = [0.]
    result = passive_phase_window(rows, 'final_passive', PATHS, .0005,
                                  config, support_pairs=supports)
    assert result['checks']['both_support_pairs_every_20ms']
    assert result['maximum_bolt_support_gap_s'] == pytest.approx(.0195)
    assert result['bolt_support_pair'] == pair_name
    assert not result['checks']['fixture_support_gap']
    assert not result['passed']


@pytest.mark.parametrize('source', SOURCES)
def test_pinned_engaged_reset_copies_only_two_poses_and_passive_references(canonical, source):
    scene, task = canonical
    before = deepcopy((scene, task))
    inputs_before = {p: p.read_bytes() for root in [source, Path(LOOSE_BOLT_RELEASE['path'])]
                     for p in root.iterdir() if p.is_file()}
    selected, diagnostic, receipt = prepare_contact_reset(scene, task, source)
    _, historical, _, _ = task_inputs(source, 'fastener_assembly')
    expected = deepcopy(scene)
    for role in ('screwdriver', 'bolt'):
        pose = {k: actor(historical, role)[k] for k in ('position_m', 'quaternion_wxyz')}
        actor(expected, role).update(pose)
        assert diagnostic['evaluation']['initial_objects'][role] == pose
    assert selected == expected
    assert selected['isaacsim'] == scene['isaacsim']
    assert selected['isaacsim']['solver_type'] == 'PGS'
    assert selected['isaacsim']['physics_dt_s'] == .0005
    assert diagnostic['initial_state'] == task['initial_state']
    assert diagnostic['public_rules'] == task['public_rules']
    assert (scene, task) == before
    assert inputs_before == {p: p.read_bytes() for p in inputs_before}
    assert receipt['source_manifest_sha256'] in {
        '924b9f4dec7eb1437b60d2292cb0ca7c454b243b5b26647086a4591642a06b5d',
        '94c67e82adbb4c2cb35d7522ea58df665b833b1c821d21ed76780d1ad542983a'}
    assert set(receipt['compatible_rigid_assets']) == {'bolt', 'screwdriver', 'threaded_fixture'}
    assert receipt['selected_isaacsim_options'] == scene['isaacsim']
    assert receipt['original_releases_modified'] is False
    assert receipt['object_pose_velocity_writes_after_reset'] == 0
    assert all(receipt[k] is False for k in ('task_success_claimed',
        'robot_manipulation_qualified', 'physics_accepted', 'prior_acceptance_transfers'))


def test_engaged_mapping_preserves_pose_relative_to_fixture(canonical):
    scene, task = deepcopy(canonical)
    fixture = actor(scene, 'threaded_fixture')
    fixture['position_m'] = [.45, .12, .80]
    fixture['quaternion_wxyz'] = [np.cos(.3), 0., 0., np.sin(.3)]
    selected, _, _ = prepare_contact_reset(scene, task, SOURCES[0])
    _, historical, _, _ = task_inputs(SOURCES[0], 'fastener_assembly')
    old_fixture = actor(historical, 'threaded_fixture')
    current_r = quaternion_matrix(fixture['quaternion_wxyz'])
    for role in ('screwdriver', 'bolt'):
        old, new = actor(historical, role), actor(selected, role)
        assert np.allclose(current_r.T@(np.array(new['position_m'])-fixture['position_m']),
                           np.array(old['position_m'])-old_fixture['position_m'], atol=1e-14)
        assert np.allclose(current_r.T@quaternion_matrix(new['quaternion_wxyz']),
                           quaternion_matrix(old['quaternion_wxyz']), atol=1e-14)


def test_unpinned_source_is_rejected_before_loading(canonical, tmp_path):
    (tmp_path/'release_manifest.json').write_text('{}')
    with pytest.raises(ValueError, match='pinned native18/native19'):
        prepare_contact_reset(*canonical, tmp_path)


def test_modified_pinned_release_payload_is_rejected(canonical, tmp_path):
    source = SOURCES[0]
    # Keep the exact allowed manifest while corrupting one of its frozen inputs.
    corrupted = tmp_path/'corrupted'
    shutil.copytree(source, corrupted, copy_function=shutil.copyfile)
    (corrupted/'scene.json').write_text('{}')
    with pytest.raises(ValueError, match='Frozen input changed'):
        prepare_contact_reset(*canonical, corrupted)


@pytest.mark.parametrize('damage,reason', [
    ('profile', 'physics profile'), ('robot', 'robot and constraints'),
    ('ready', 'ready pose'), ('material', 'actor physics'),
    ('shape', 'rigid shape'), ('pose', 'unit quaternions'),
    ('threshold', 'thresholds'), ('task_variant', 'loose-bolt target')])
def test_contact_reset_rejects_incompatible_target(canonical, damage, reason):
    scene, task = deepcopy(canonical)
    if damage == 'profile':
        scene['isaacsim']['robot_contact_materials']['pad_friction'] += .1
    elif damage == 'robot':
        scene['robot']['base_position_m'][0] += .1
    elif damage == 'ready':
        task['initial_state']['joint_positions_rad']['fr3_joint1'] += .01
    elif damage == 'material':
        actor(scene, 'bolt')['usd_physics_material_fallback']['static_friction'] += .1
    elif damage == 'shape':
        actor(scene, 'bolt')['asset_root'] = f'{DATA_ROOT}/assets/task_parts/threaded_sharpa/sdf1024_candidate_20261018'
    elif damage == 'pose':
        actor(scene, 'threaded_fixture')['quaternion_wxyz'] = [2., 0., 0., 0.]
    elif damage == 'threshold':
        task['evaluation']['maximum_reported_penetration_m'] += .001
    else:
        task['evaluation']['reset_variant'] = 'preinserted'
    with pytest.raises(ValueError, match=reason):
        prepare_contact_reset(scene, task, SOURCES[0])


@pytest.mark.parametrize('engaged', [False, True])
def test_explicit_supports_match_canonical_table_or_engaged_fixture(canonical, engaged):
    scene, task = canonical
    config = deepcopy(task['evaluation'])
    config['tool_rest_actor'] = 'tool_supply'
    pairs = passive_support_pairs(scene, config, PATHS, contact_reset=engaged)
    if engaged:
        assert pairs == {'tool_bolt': ('tool', 'bolt'), 'bolt_fixture': ('bolt', 'fixture')}
    else:
        assert pairs == {'tool_supply': ('tool', 'tool_supply'),
                         'bolt_table': ('bolt', '/World/Room/Geometry/worktop/Collider')}
    rows, synthetic_config = evidence()
    for row in rows:
        row['raw_contacts'] = [{'actor0': a, 'actor1': b,
            'normal_impulses_ns': [.001], 'separations_m': [-1e-6]} for a, b in pairs.values()]
    assert passive_phase_window(rows, 'final_passive', PATHS, .0005,
                                synthetic_config, support_pairs=pairs)['passed']
    if not engaged:
        wrong = {**pairs, 'bolt_table': ('bolt', 'bolt_supply')}
        assert not passive_phase_window(rows, 'final_passive', PATHS, .0005,
                                        synthetic_config, support_pairs=wrong)['passed']


@pytest.mark.parametrize('pairs', [{}, {'a': None, 'b': ['bolt', 'fixture']},
    {'a': ['tool', 'bolt'], 'b': ['tool', 'fixture']}])
def test_explicit_supports_require_each_movable_body(pairs):
    rows, config = evidence()
    with pytest.raises(ValueError, match='one explicit support pair'):
        passive_phase_window(rows, 'final_passive', PATHS, .0005, config, support_pairs=pairs)
