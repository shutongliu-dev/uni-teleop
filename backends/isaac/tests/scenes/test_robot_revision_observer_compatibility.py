"""Frozen laundry observer adaptation must preserve every physical/task input."""
from copy import deepcopy
import json

import pytest

from scenes.workrooms.robot_revision import release
from scenes.workrooms.robot_revision import laundry_observer as compatibility
from test_robot_revision_release import publication, original_robot_catalog


def patched(publication, tmp_path):
    old, old_manifest, parent, revision, _ = publication('laundry_room', 'towel_flattening', 'dual_opposed')
    new = tmp_path/'patched_observer'
    manifest = release.prepare_release(parent, new, revision=revision,
        robot_asset_manifest=old_manifest['robot_asset_manifest']['path'],
        configuration_id=old_manifest['configuration_id'], observer_compatibility=compatibility.VERSION)
    return old, old_manifest, new, manifest, parent


def test_opt_in_observer_derivation_keeps_legacy_and_all_ten_physical_seeds(publication, tmp_path):
    from bench.isaacsim.release import validate, task_inputs
    old, before, new, after, parent = patched(publication, tmp_path)
    assert 'observer_compatibility' not in before
    assert validate(old) == before and validate(new) == after
    record = after['observer_compatibility']
    assert record['version'] == compatibility.VERSION
    assert record['parent_sha256'] == compatibility.ORIGINAL_SHA256
    assert record['derived_sha256'] == compatibility.PATCHED_SHA256
    assert set(record['sources_sha256']) == set(release.OBSERVER_COMPATIBILITY_SOURCES)
    assert (old/compatibility.OBSERVER_PATH).read_bytes() == (parent/compatibility.OBSERVER_PATH).read_bytes()
    assert (new/compatibility.OBSERVER_PATH).read_bytes() == compatibility.patch_observer(
        (parent/compatibility.OBSERVER_PATH).read_bytes())
    assert {name for name in before['files_sha256'] if before['files_sha256'][name] != after['files_sha256'][name]} == {
        compatibility.OBSERVER_PATH}
    for seed in range(10):
        assert task_inputs(old, 'towel_flattening', seed=seed) == task_inputs(new, 'towel_flattening', seed=seed)
    assert all(after[k] is False for k in ('physics_accepted', 'task_policy_success_claimed', 'inherited_qualification'))


@pytest.mark.parametrize('fault', ['observer', 'parent_hash', 'source_hash', 'unknown_version', 'missing_record'])
def test_rehashed_observer_or_provenance_cannot_expand_the_patch(publication, tmp_path, fault):
    _, _, new, manifest, _ = patched(publication, tmp_path)
    manifest = deepcopy(manifest)
    if fault == 'observer':
        target = new/compatibility.OBSERVER_PATH
        target.write_bytes(target.read_bytes()+b'\n# unauthorized observer edit\n')
        digest = release.file_hash(target)
        manifest['files_sha256'][compatibility.OBSERVER_PATH] = digest
        manifest['files'][compatibility.OBSERVER_PATH] = {'sha256': digest}
        manifest['observer_compatibility']['derived_sha256'] = digest
    elif fault == 'parent_hash':
        manifest['observer_compatibility']['parent_sha256'] = '0'*64
    elif fault == 'source_hash':
        manifest['observer_compatibility']['sources_sha256'][release.OBSERVER_COMPATIBILITY_SOURCES[0]] = '0'*64
    elif fault == 'unknown_version':
        manifest['observer_compatibility']['version'] = 'unreviewed.v99'
    else:
        del manifest['observer_compatibility']
    (new/'release_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    with pytest.raises(ValueError):
        release.validate_release(new, manifest)


def test_observer_patch_is_not_available_to_other_task_families(publication, tmp_path):
    _, manifest, parent, revision, _ = publication('mechanical_assembly', 'parts_sorting')
    with pytest.raises(ValueError, match='Unsupported native observer'):
        release.prepare_release(parent, tmp_path/'wrong_family', revision=revision,
            robot_asset_manifest=manifest['robot_asset_manifest']['path'],
            configuration_id=manifest['configuration_id'], observer_compatibility=compatibility.VERSION)
