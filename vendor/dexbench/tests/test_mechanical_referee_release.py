"""Rule-only publication preserves a pinned parent; no asset rebuild or GPU."""
from copy import deepcopy
import json
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace

import pytest

from bench.isaacsim.release import checked, digest, load_evaluator, validate
from bench.paths import DATA_ROOT, load_json
from bench.protocol.task_referee import project_referee_state, project_task_rules
from scenes.workrooms.mechanical_assembly import referee_release, rules


PARENT = Path(f'{DATA_ROOT}/datasets/fr3_sharpa_bench/development/'
    'mechanical_task_validation_20261018/sorting/task_release_sdf1024_layout_v1')
PARENT_SHA256 = 'cfb877eb5f4e9b5dfb22a6672b7a4cf7958aeb7e4c18083da2be18ce07268769'


def read_json(path):
    return load_json(path)


def file_bytes(root):
    return {str(path.relative_to(root)): path.read_bytes()
        for path in root.rglob('*') if path.is_file()}


@pytest.fixture
def parent(tmp_path, monkeypatch, request):
    # Pin an actual old release, without importing authoring defaults or ever
    # modifying shared evidence. Mutation tests operate only on this copy.
    parent_path, parent_hash = getattr(request, 'param', (PARENT, PARENT_SHA256))
    if not (parent_path/'release_manifest.json').is_file():
        pytest.skip('Pinned historical release is not installed in this workspace')
    checked(parent_path/'release_manifest.json', parent_hash)
    manifest = read_json(parent_path/'release_manifest.json')
    original = {relative: (parent_path/relative).read_bytes()
        for relative in [*manifest['files'], 'release_manifest.json']}
    source = tmp_path/'parent'
    for relative, content in original.items():
        path = source/relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    accepted_validations = []

    def validate_without_rehashing_shared_assets(root):
        # Production revise_rules still runs full validate. This test retains
        # all release/input/scorer checks while avoiding repeated large hashes.
        result = validate(root, assets=False)
        accepted_validations.append(Path(root).resolve())
        return result

    monkeypatch.setattr(referee_release, 'DATASETS', tmp_path)
    monkeypatch.setattr(referee_release, 'validate', validate_without_rehashing_shared_assets)
    value = SimpleNamespace(source=source, output=tmp_path/'revised', manifest=manifest,
        manifest_sha256=parent_hash, original=original, accepted_validations=accepted_validations)
    yield value
    assert {relative: (parent_path/relative).read_bytes() for relative in original} == original


def revise(parent, **options):
    return referee_release.revise_rules(parent.source, parent.output,
        source_manifest_sha256=options.get('source_manifest_sha256', parent.manifest_sha256))


def assert_unpublished(parent):
    assert not parent.output.exists()
    assert not list(parent.output.parent.glob('.'+parent.output.name+'.preparing-*'))


def test_rule_revision_preserves_every_physical_input_and_loads_its_frozen_contract(parent):
    original_task = read_json(parent.source/'task.json')
    assert 'referee' not in original_task['public_rules']
    manifest = revise(parent)
    output = parent.output
    task = read_json(output/'task.json')
    for name in referee_release.UNCHANGED:
        assert (output/name).read_bytes() == parent.original[name]
        assert manifest['files'][name] == parent.manifest['files'][name]
    assert task['initial_state'] == original_task['initial_state']
    assert task['evaluation'] == original_task['evaluation']
    changed_fields = {'public_rules', 'public_rules_sha256', 'public_referee_contract_sha256', 'public_rules_file'}
    assert {key: value for key, value in task.items() if key not in changed_fields} == {
        key: value for key, value in original_task.items() if key not in changed_fields}
    assert task['public_rules'] == read_json(output/'rules.json') == rules.public_rules('parts_sorting')
    assert manifest['asset_files_sha256'] == parent.manifest['asset_files_sha256']
    assert manifest['asset_catalog'] == parent.manifest['asset_catalog']
    assert manifest['physics_accepted'] is manifest['task_policy_success_claimed'] is False
    assert manifest['status'] == 'prepared_unvalidated'
    assert manifest['rule_parent_release'] == {'path': str(parent.source), 'manifest_sha256': PARENT_SHA256}

    public = project_task_rules(task)
    evaluator_type = load_evaluator(output)
    evaluator = evaluator_type(task)
    state = project_referee_state(evaluator.inspect_conditions(), public['public_rules'])
    assert state['state'] == 'ongoing' and not state['terminal']
    # In-process validation intentionally caches by manifest hash, so this
    # instance can retain staging filenames after the atomic rename. A fresh
    # worker must load all frozen dependencies from the published directory.
    probe = '''import json, sys
from pathlib import Path
from bench.isaacsim.release import load_evaluator
from bench.protocol.task_referee import project_referee_state, project_task_rules
root = Path(sys.argv[1])
task = json.loads((root/'task.json').read_text())
cls = load_evaluator(root)
state = project_referee_state(cls(task).inspect_conditions(), project_task_rules(task)['public_rules'])
print(json.dumps({'evaluator': cls.update.__code__.co_filename,
    'referee': cls._verdict.__globals__['verdict'].__code__.co_filename, 'state': state}))
'''
    result = subprocess.run([sys.executable, '-B', '-c', probe, str(output)],
        cwd=Path(__file__).resolve().parents[1], check=True, capture_output=True, text=True, timeout=60)
    loaded = json.loads(result.stdout)
    assert loaded['state'] == state
    assert all(Path(loaded[name]).resolve().is_relative_to(output) for name in ('evaluator', 'referee'))
    assert 'evaluator/referee.py' in manifest['files']
    assert digest(output/'evaluator/referee.py') == manifest['files']['evaluator/referee.py']['sha256']
    provenance = read_json(output/'source_provenance.json')['rule_revision']
    assert provenance['physical_inputs_byte_identical'] == {
        name: parent.manifest['files'][name]['sha256'] for name in referee_release.UNCHANGED}
    assert provenance['evaluation_parameters_unchanged'] is True
    assert provenance['new_native_qualification_claimed'] is False
    assert len(parent.accepted_validations) == 2
    assert {relative: (parent.source/relative).read_bytes() for relative in parent.original} == parent.original

    # Even a real completed new release cannot be overwritten on a second call.
    published = file_bytes(output)
    with pytest.raises(ValueError, match='fresh referee release'):
        revise(parent)
    assert file_bytes(output) == published


@pytest.mark.parametrize('when', ['before_scene_copy', 'after_scene_copy'])
def test_source_mutation_during_copy_is_rejected_and_staging_is_removed(parent, monkeypatch, when):
    real_copy = shutil.copyfile
    source_scene = parent.source/'scene.json'
    scene_copied = mutated = False

    def mutate_scene():
        value = read_json(source_scene)
        value['isaacsim']['physics_dt_s'] *= 2.
        source_scene.write_text(json.dumps(value))

    def copy_with_source_race(source, destination, *args, **kwargs):
        nonlocal scene_copied, mutated
        source = Path(source)
        if when == 'before_scene_copy' and source == source_scene and not mutated:
            mutate_scene()
            mutated = True
        result = real_copy(source, destination, *args, **kwargs)
        if source == source_scene:
            scene_copied = True
        elif (when == 'after_scene_copy' and scene_copied and not mutated
                and source.is_relative_to(parent.source)):
            # Another parent file is being copied. Staging already has the
            # correct scene, so only the final parent recheck can catch this.
            mutate_scene()
            mutated = True
        return result

    monkeypatch.setattr(referee_release.shutil, 'copyfile', copy_with_source_race)
    with pytest.raises(ValueError, match='Frozen input changed: .*scene.json') as error:
        revise(parent)
    assert mutated
    assert_unpublished(parent)
    if when == 'before_scene_copy':
        assert len(parent.accepted_validations) == 1
        assert '.preparing-' in str(error.value)
    else:
        # Both original preflight and completed staging validate succeeded;
        # rejection specifically names the changed parent, not a new scene.
        assert len(parent.accepted_validations) == 2
        assert str(source_scene) in str(error.value)
        assert '.preparing-' not in str(error.value)


@pytest.mark.parametrize('kind', ['empty_directory', 'occupied_directory', 'file'])
def test_existing_output_is_never_reused_or_modified(parent, kind):
    if kind == 'file':
        parent.output.write_bytes(b'keep existing file')
        before = parent.output.read_bytes()
    else:
        parent.output.mkdir()
        if kind == 'occupied_directory': (parent.output/'keep.txt').write_bytes(b'keep existing data')
        before = file_bytes(parent.output)
    with pytest.raises(ValueError, match='fresh referee release'):
        revise(parent)
    assert not parent.accepted_validations
    assert (parent.output.read_bytes() if kind == 'file' else file_bytes(parent.output)) == before
    assert not list(parent.output.parent.glob('.'+parent.output.name+'.preparing-*'))


def test_output_created_by_another_publisher_during_copy_is_preserved(parent, monkeypatch):
    real_copy = shutil.copyfile
    created = False

    def create_competing_output(source, destination, *args, **kwargs):
        nonlocal created
        if not created:
            parent.output.mkdir()
            (parent.output/'other-publication.txt').write_bytes(b'preserve concurrent output')
            created = True
        return real_copy(source, destination, *args, **kwargs)

    monkeypatch.setattr(referee_release.shutil, 'copyfile', create_competing_output)
    with pytest.raises(FileExistsError):
        revise(parent)
    assert created and len(parent.accepted_validations) == 2
    assert file_bytes(parent.output) == {'other-publication.txt': b'preserve concurrent output'}
    assert not list(parent.output.parent.glob('.'+parent.output.name+'.preparing-*'))


def test_wrong_parent_manifest_identity_is_rejected_before_copy(parent):
    with pytest.raises(ValueError, match='Frozen input changed: .*release_manifest.json'):
        revise(parent, source_manifest_sha256='0'*64)
    assert not parent.accepted_validations
    assert_unpublished(parent)


def test_rule_revision_cannot_sneak_in_changed_scoring_thresholds(parent, monkeypatch):
    original_thresholds = rules.thresholds

    def changed_thresholds(task_id):
        value = deepcopy(original_thresholds(task_id))
        value['minimum_stable_s'] += .1
        return value

    monkeypatch.setattr(rules, 'thresholds', changed_thresholds)
    with pytest.raises(ValueError, match='cannot change frozen physical or task thresholds'):
        revise(parent)
    assert len(parent.accepted_validations) == 1
    assert_unpublished(parent)


@pytest.mark.parametrize('parent', [
    (Path(f'{DATA_ROOT}/datasets/fr3_sharpa_bench/development/'
          'mechanical_task_validation_20261019/fastener/loose_bolt_candidate_v7'),
     'efdd235408aeb9f682273b7a248f87ef19da8eff1cf6d9f77d797e162f7d5424'),
    (Path(f'{DATA_ROOT}/datasets/fr3_sharpa_bench/development/'
          'mechanical_referee_20261018/rules_v1/releases/fastener_assembly'),
     '490a0ee47eb1132691515ff874d71cbd6ae71c02b8ccd36ebc4be4bb871456ca'),
], ids=['loose_bolt', 'preinserted'], indirect=True)
def test_fastener_rule_revision_preserves_reset_variant_and_physics(parent):
    from scenes.workrooms.mechanical_assembly.tasks.fastener_assembly.rules import public_rules

    original_task = read_json(parent.source/'task.json')
    variant = original_task['public_rules'].get('reset_variant', 'preinserted')
    manifest = revise(parent)
    task = read_json(parent.output/'task.json')
    assert task['public_rules'] == public_rules(30., reset_variant=variant)
    assert task['reset_conditions'] == original_task['reset_conditions']
    assert task['evaluation'] == original_task['evaluation']
    assert load_evaluator(parent.output)(task).reset_variant == variant
    assert read_json(parent.output/'rules.json') == task['public_rules']
    for name in referee_release.UNCHANGED:
        assert (parent.output/name).read_bytes() == parent.original[name]
        assert manifest['files'][name] == parent.manifest['files'][name]
    if variant == 'preinserted':
        assert task['public_rules'] == original_task['public_rules']
    else:
        assert task['public_rules']['schema'] == 'mechanical_fastener_rules_v5'
        assert original_task['public_rules']['schema'] == 'mechanical_fastener_rules_v4'
    assert {relative: (parent.source/relative).read_bytes() for relative in parent.original} == parent.original
