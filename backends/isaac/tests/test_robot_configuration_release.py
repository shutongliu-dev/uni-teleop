"""Derivative release counterexamples against a real frozen CPU probe."""
from copy import deepcopy
import json
from pathlib import Path
import shutil

import pytest

from bench.isaacsim.release import digest, validate
from bench.paths import DATA_ROOT, load_json
from scenes.workrooms.catalog import task_definition
from scenes.workrooms.robot_configuration_release import _compose, task_inputs, KINEMATIC_ROBOT_KEYS
from scenes.workrooms.robot_configuration_release import TERMINAL_SCORING_INHERITANCE
from scenes.runtime import terminal_scoring_contract


PROBE = Path(f'{DATA_ROOT}/datasets/fr3_sharpa_bench/development/'
             'bimanual_options_20261023/probe_releases/parts_sorting/dual_same_side_v2')


def read(root, relative):
    return load_json(root/relative)


def write(root, relative, value):
    (root/relative).write_text(json.dumps(value, indent=2, ensure_ascii=False)+'\n')


@pytest.fixture
def probe(tmp_path):
    if not (PROBE/'release_manifest.json').is_file():
        pytest.skip('The explicit bimanual CPU publication probe is unavailable')
    root = tmp_path/'release'
    shutil.copytree(PROBE, root)
    return root


def configuration_base(family, task_id):
    """The unrandomized task release that the published configurations compose.

    Parameterized seed cases and the L-mount revision wrap a configuration
    release; a configuration is never composed from those wrappers.
    """
    bases = set()
    for profile in ('original', 'l_mount'):
        definition = task_definition(family, task_id, robot_profile=profile)
        manifest = {'base_release': definition['robot_configurations']['dual_same_side']['source_release']}
        while manifest.get('schema') != 'workroom_robot_configuration_release.v1':
            manifest = read(Path(manifest['base_release']['path']), 'release_manifest.json')
        bases.add(manifest['base_release']['path'])
    base, = bases  # Both robot profiles bind the same original task world.
    return Path(base)


def rehash(root):
    manifest = read(root, 'release_manifest.json')
    for relative in manifest['files']:
        manifest['files'][relative]['sha256'] = digest(root/relative)
    manifest['files_sha256'] = {p: row['sha256'] for p, row in manifest['files'].items()}
    for entry in manifest['tasks']:
        for key in ('scene', 'task', 'initial_state'):
            entry[key+'_sha256'] = digest(root/entry[key])
    manifest['physics_profile_sha256'] = digest(root/'physics_profile.json')
    write(root, 'release_manifest.json', manifest)


def test_real_derivative_keeps_original_task_and_physics(probe):
    manifest = validate(probe, assets=False)
    base = Path(manifest['base_release']['path'])
    scene, task = read(probe, 'scene.json'), read(probe, 'task.json')
    original_scene, original_task = read(base, 'scene.json'), read(base, 'task.json')
    assert scene['objects'] == original_scene['objects']
    assert scene['isaacsim'] == original_scene['isaacsim']
    assert task['evaluation'] == original_task['evaluation']
    assert task['active_sides'] == ['left', 'right']
    assert len(task['initial_state']['joint_positions_rad']) == 58
    assert digest(probe/'physics_profile.json') == digest(base/'physics_profile.json')
    assert 'left_wrist' in scene['cameras'] and 'right_wrist' in scene['cameras']
    assert manifest['physics_accepted'] is False


@pytest.mark.parametrize('kind', ['goal', 'physics', 'object', 'scorer', 'new_gain'])
def test_rehashed_changes_cannot_escape_derivative_scope(probe, kind):
    scene = read(probe, 'scene.json')
    if kind == 'goal':
        task = read(probe, 'task.json')
        task['evaluation']['stable_s'] = 0.
        write(probe, 'task.json', task)
    elif kind == 'physics':
        scene['isaacsim']['physics_dt_s'] *= 2
        profile = read(probe, 'physics_profile.json')
        profile['isaacsim_options'] = deepcopy(scene['isaacsim'])
        write(probe, 'physics_profile.json', profile)
        write(probe, 'scene.json', scene)
    elif kind == 'object':
        scene['objects'][0]['position_m'][0] += .1
        write(probe, 'scene.json', scene)
    elif kind == 'scorer':
        path = probe/'evaluator/sequence_evaluator.py'
        path.write_text(path.read_text()+'\n# rehashed unreviewed scoring edit\n')
    elif kind == 'new_gain':
        configuration = read(probe, 'robot_configuration.json')
        configuration['robot']['new_control_gain'] = 1000000.
        scene['robot'] = deepcopy(configuration['robot'])
        write(probe, 'robot_configuration.json', configuration)
        write(probe, 'scene.json', scene)
    rehash(probe)
    with pytest.raises(ValueError):
        validate(probe, assets=False)


def test_undeclared_python_dependency_is_rejected(probe):
    (probe/'evaluator/unreviewed_helper.py').write_text('side_effect = True\n')
    with pytest.raises(ValueError, match='[Uu]ndeclared|inventory|dependencies'):
        validate(probe, assets=False)


def test_rehashed_original_provenance_cannot_be_rewritten(probe):
    provenance = read(probe, 'source_provenance.json')
    provenance['invented_original_physics_acceptance'] = True
    write(probe, 'source_provenance.json', provenance)
    rehash(probe)
    with pytest.raises(ValueError, match='provenance'):
        validate(probe, assets=False)


def test_evaluator_parent_symlink_is_rejected(probe):
    (probe/'evaluator').rename(probe/'evaluator_payload')
    (probe/'evaluator').symlink_to('evaluator_payload', target_is_directory=True)
    with pytest.raises(ValueError, match='escapes'):
        validate(probe, assets=False)


def test_snapshot_reference_requires_real_matching_digest(probe):
    manifest = read(probe, 'release_manifest.json')
    manifest['source_snapshot']['sha256'] = '0'*64
    write(probe, 'release_manifest.json', manifest)
    with pytest.raises(ValueError):
        validate(probe, assets=False)


@pytest.mark.parametrize('family,task_id', [
    ('mechanical_assembly', 'nist_wire_harness'), ('laboratory', 'pipette_transfer'),
    ('living_room', 'remote_operation'), ('living_room', 'water_pouring'),
    ('living_room', 'tower_building'), ('card_room', 'task0'), ('study', 'drawer_retrieval')])
def test_seed_binding_keeps_new_hands_and_frozen_hooks(probe, tmp_path, family, task_id):
    # The probe configuration is the original-mount robot. The catalog's own
    # source release is now a frozen seed-case wrapper (seeds 0..9 only), so
    # compose from the task release that the real configurations bind.
    base = configuration_base(family, task_id)
    configuration = read(probe, 'robot_configuration.json')
    configuration.update(family=family, task_id=task_id)
    # This is a parser/scorer fixture, never a proposed physical installation.
    # Preserve each family's exact gains/options and reuse only tested geometry.
    robot = deepcopy(read(base, 'scene.json')['robot'])
    for key in KINEMATIC_ROBOT_KEYS:
        if key in configuration['robot']:
            robot[key] = deepcopy(configuration['robot'][key])
        else:
            robot.pop(key, None)
    configuration['robot'] = robot
    scene, task, initial, sources, _ = _compose(base, configuration)
    root = tmp_path/'binding'
    (root/'evaluator').mkdir(parents=True)
    for relative, source in sources.items():
        destination = root/'evaluator'/relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(source)
    for relative, value in [('scene.json', scene), ('task.json', task), ('initial_state.json', initial)]:
        write(root, relative, value)
    manifest = {'task_id': task_id, 'base_release': {'schema': read(base, 'release_manifest.json')['schema']},
                'tasks': [{'id': task_id}]}
    write(root, 'release_manifest.json', manifest)
    _, rebound_scene, rebound, reset = task_inputs(root, task_id, manifest, seed=17)
    # Every base family here has an episode binder; it must actually run.
    assert rebound.get('episode_seed', rebound.get('seed')) == 17
    assert rebound['active_sides'] == ['left', 'right']
    assert rebound['initial_state'] == initial == reset
    assert len(reset['joint_positions_rad']) == 58
    for name, hook in rebound_scene.get('native_task_bundle', {}).items():
        if name in ('observer', 'reduced_dynamics'):
            assert hook['configuration'] == rebound
            assert Path(hook['path']).is_relative_to(root)
            assert digest(hook['path']) == hook['sha256']
            assert hook['files'] == {name: digest(root/'evaluator'/name) for name in sources}
    if task_id == 'remote_operation':
        assert rebound['evaluation']['hand_link_prefixes'] == ['left_', 'right_']
    if task_id == 'task0':
        assert rebound['episode_seed'] == 17
        assert rebound['public_goal']['target_cells'] != task['public_goal']['target_cells']
    if family in ('study',) or task_id == 'tower_building':
        assert rebound_scene['scenario']['seed'] == 17


@pytest.mark.parametrize('configuration_id', ['dual_same_side', 'dual_opposed'])
def test_tower_derivation_inherits_contract_without_rewriting_legacy_releases(tmp_path, configuration_id):
    # Legacy configuration releases were derived from the original-mount source release.
    definition = task_definition('living_room', 'tower_building', robot_profile='original')
    source = Path(definition['source_release']['path'])
    frozen = Path(definition['robot_configurations'][configuration_id]['source_release']['path'])
    before = {name: digest(frozen/name) for name in (
        'scene.json', 'task.json', 'rules.json', 'release_manifest.json', 'source_provenance.json')}
    manifest = validate(frozen, assets=False)
    assert 'terminal_scoring_inheritance' not in manifest
    configuration = read(frozen, 'robot_configuration.json')
    old_scene, old_task, _, _, _ = _compose(source, configuration, inherit_terminal_scoring=False)
    assert old_scene == read(frozen, 'scene.json') and old_task == read(frozen, 'task.json')
    assert 'terminal_scoring_contract' not in old_task

    scene, task, _, sources, _ = _compose(source, configuration)
    contract = terminal_scoring_contract(read(source, 'task.json'))
    assert task['terminal_scoring_contract'] == contract == terminal_scoring_contract(old_task)
    assert task['evaluation'] == read(source, 'task.json')['evaluation']
    assert {key: value for key, value in task.items() if key != 'terminal_scoring_contract'} == old_task
    for name in ('observer', 'reduced_dynamics'):
        if name in scene.get('native_task_bundle', {}):
            assert scene['native_task_bundle'][name]['configuration'] == task
    assert sources == {str(path.relative_to(frozen/'evaluator')): path.read_text()
                       for path in (frozen/'evaluator').rglob('*.py')}

    # A temporary validator fixture exercises the new format. It is not a
    # published release or new qualification of the original source snapshot.
    fixture = tmp_path/'with_contract'
    shutil.copytree(frozen, fixture)
    write(fixture, 'scene.json', scene)
    write(fixture, 'task.json', task)
    manifest['terminal_scoring_inheritance'] = TERMINAL_SCORING_INHERITANCE
    write(fixture, 'release_manifest.json', manifest)
    provenance = read(fixture, 'source_provenance.json')
    provenance['robot_configuration_derivation']['terminal_scoring_inheritance'] = TERMINAL_SCORING_INHERITANCE
    write(fixture, 'source_provenance.json', provenance)
    rehash(fixture)
    validate(fixture, assets=False)
    rebound = task_inputs(fixture, 'tower_building', read(fixture, 'release_manifest.json'), seed=17)[2]
    assert rebound['terminal_scoring_contract'] == contract

    # Rehashing a changed declaration cannot change its base's score semantics.
    task['terminal_scoring_contract']['score_unit'] = 'cm'
    write(fixture, 'task.json', task)
    rehash(fixture)
    with pytest.raises(ValueError, match='Unapproved task or scene change'):
        validate(fixture, assets=False)
    assert before == {name: digest(frozen/name) for name in before}


@pytest.mark.parametrize('marker', [None, False, True, 'base_effective_contract.v2'])
def test_unknown_terminal_scoring_inheritance_is_rejected(probe, marker):
    manifest = read(probe, 'release_manifest.json')
    manifest['terminal_scoring_inheritance'] = marker
    write(probe, 'release_manifest.json', manifest)
    with pytest.raises(ValueError, match='Unknown.*terminal scoring inheritance'):
        validate(probe, assets=False)
