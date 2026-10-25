import importlib.util
import json
import os
from pathlib import Path
import subprocess

import pytest

from bench.paths import load_json


SOURCE = Path(__file__).resolve().parents[1] / "scripts/maintenance/freeze_project_sources.py"
spec = importlib.util.spec_from_file_location("freeze_project_sources", SOURCE)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def legacy_fixture_layout(root, manifest):
    """Restore the real old roots in a synthetic fixture, never a sealed run."""
    (root / "interface").mkdir()
    (root / "bench/simulation").rename(root / "simulation")
    if (root / "bench/apps").exists():
        (root / "bench/apps").rename(root / "apps")
    for tree in ("agent", "scenes"):
        (root / tree).rmdir()
    manifest.pop("source_trees", None)
    for field in ("files_sha256", "files_mode_octal", "suite_source_pins"):
        if field in manifest:
            manifest[field] = {name.replace("bench/simulation/", "simulation/", 1).replace("bench/apps/", "apps/", 1): value
                               for name, value in manifest[field].items()}
    manifest['explicit_extra_resources'] = [name.replace('bench/apps/', 'apps/', 1)
                                            for name in manifest['explicit_extra_resources']]


@pytest.fixture
def project(tmp_path):
    root = tmp_path / "project"
    for tree in module.SOURCE_TREES:
        (root / tree).mkdir(parents=True)
    for name in ("bench/runtime/server.py", "bench/control/experimental/candidate.py",
                 "bench/simulation/native.py", "scripts/isaac_python.sh"):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# source\n")
    (root / "scripts/isaac_python.sh").chmod(0o755)
    config = root / "configs/credentials.json"
    config.parent.mkdir()
    config.write_text('{"key":"test-secret-never-copy"}')
    return root


def test_complete_tree_isolated_from_edits_and_credentials(project, tmp_path):
    target = tmp_path / "snapshot"
    result = module.freeze(project, target, ("scripts/isaac_python.sh",))
    assert "bench/control/experimental/candidate.py" in result["suite_source_pins"]
    assert not (target / "configs").exists()
    (project / "bench/control/experimental/new.py").write_text("# new source\n")
    (project / "bench/runtime/server.py").write_text("# edited\n")
    assert module.verify(target)["verified"]
    assert not (target / "bench/control/experimental/new.py").exists()
    with pytest.raises(FileExistsError):
        module.freeze(project, target, ("scripts/isaac_python.sh",))


def test_detects_added_source_in_snapshot(project, tmp_path):
    target = tmp_path / "snapshot"
    module.freeze(project, target, ("scripts/isaac_python.sh",))
    (target / "bench/added.py").write_text("# added\n")
    with pytest.raises(ValueError, match="inventory changed"):
        module.verify(target)


@pytest.mark.parametrize("tree", ["agent", "scenes"])
def test_v6_pins_module_roots_and_detects_changes(project, tmp_path, tree):
    relative = f"{tree}/example.py"
    (project / relative).write_text("# source in the separated module\n")
    target = tmp_path / "snapshot"
    result = module.freeze(project, target, ("scripts/isaac_python.sh",))
    assert result["source_trees"] == list(module.SOURCE_TREES)
    assert result["suite_source_pins"][relative] == result["files_sha256"][relative]
    (target / relative).write_text("# changed after freeze\n")
    with pytest.raises(ValueError, match="inventory changed"):
        module.verify(target)


def test_v4_without_new_module_directories_verifies_without_rewriting(project, tmp_path):
    helper = project / module.RUNTIME_GUARD_SOURCE
    helper.parent.mkdir(parents=True, exist_ok=True)
    helper.write_text("# historical guard fixture\n")
    target = tmp_path / "snapshot"
    result = module.freeze(project, target, ("scripts/isaac_python.sh",))
    result["source_utility_version"] = "complete_benchmark_source_verifier_v4"
    legacy_fixture_layout(target, result)
    manifest = target / "source_snapshot_manifest.json"
    manifest.write_text(json.dumps(result))
    before = manifest.read_bytes()
    assert module.verify(target)["verified"]
    assert manifest.read_bytes() == before
    (target / "agent").mkdir()
    (target / "agent/hidden.py").write_text("# not included in a v4 inventory\n")
    with pytest.raises(ValueError, match="outside versioned layout"):
        module.verify(target)


def test_v6_cannot_omit_module_roots_from_declared_layout(project, tmp_path):
    target = tmp_path / "snapshot"
    result = module.freeze(project, target, ("scripts/isaac_python.sh",))
    result["source_trees"].remove("agent")
    (target / "source_snapshot_manifest.json").write_text(json.dumps(result))
    with pytest.raises(ValueError, match="versioned layout"):
        module.verify(target)


def test_directory_alias_cannot_hide_sources(project, tmp_path):
    hidden = tmp_path / "hidden"
    hidden.mkdir()
    (hidden / "candidate.py").write_text("# hidden\n")
    (project / "bench/aliased").symlink_to(hidden, target_is_directory=True)
    with pytest.raises(ValueError, match="Symlink"):
        module.freeze(project, tmp_path / "snapshot", ("scripts/isaac_python.sh",))


@pytest.mark.parametrize("destination_kind", ["parent_symlink", "parent_dotdot"])
def test_destination_alias_cannot_write_into_editable_tree(project, tmp_path, destination_kind):
    destination = project / "bench" / "snapshot"
    if destination_kind == "parent_symlink":
        alias = tmp_path / "project_alias"
        alias.symlink_to(project, target_is_directory=True)
        requested = alias / "bench" / "snapshot"
    else:
        sibling = tmp_path / "sibling"
        sibling.mkdir()
        requested = sibling / ".." / project.name / "bench" / "snapshot"
    with pytest.raises(ValueError, match="outside the editable"):
        module.freeze(project, requested, ("scripts/isaac_python.sh",))
    assert not destination.exists()


def test_script_directory_alias_cannot_silently_omit_helpers(project, tmp_path):
    external = tmp_path / "helpers"
    external.mkdir()
    (external / "helper.py").write_text("# helper source\n")
    (project / "scripts" / "alias").symlink_to(external, target_is_directory=True)
    with pytest.raises(ValueError, match="Symlink"):
        module.freeze(project, tmp_path / "snapshot", ("scripts/isaac_python.sh",))
    assert not (tmp_path / "snapshot").exists()


def test_regular_python_helpers_are_frozen_and_verified(project, tmp_path):
    helper = project / "scripts" / "maintenance" / "helper.py"
    helper.parent.mkdir()
    helper.write_text("# helper source\n")
    target = tmp_path / "snapshot"
    result = module.freeze(project, target, ("scripts/isaac_python.sh",))
    assert "scripts/maintenance/helper.py" in result["files_sha256"]
    assert "scripts/maintenance/helper.py" not in result["suite_source_pins"]
    (target / "scripts/maintenance/helper.py").write_text("# edited\n")
    with pytest.raises(ValueError, match="inventory changed"):
        module.verify(target)


def test_changed_source_while_copying_has_no_completed_manifest(project, tmp_path, monkeypatch):
    target = tmp_path / "snapshot"
    copy = module.shutil.copy2
    changed = False

    def mutate(source, destination):
        nonlocal changed
        value = copy(source, destination)
        if not changed:
            (project / "bench/added.py").write_text("# race\n")
            changed = True
        return value

    monkeypatch.setattr(module.shutil, "copy2", mutate)
    with pytest.raises(ValueError, match="Source changed"):
        module.freeze(project, target, ("scripts/isaac_python.sh",))
    assert not (target / "source_snapshot_manifest.json").exists()


@pytest.mark.parametrize("extra", ["../outside.py", "/etc/passwd", "configs/credentials.json",
                                   "configs/local.json", "credentials/local.json", ".env.d/local.json"])
def test_unsafe_or_credential_extras_refused(project, tmp_path, extra):
    with pytest.raises(ValueError):
        module.freeze(project, tmp_path / "snapshot", ("scripts/isaac_python.sh", extra))


def test_configuration_path_is_refused_before_content_is_read(project, tmp_path, monkeypatch):
    forbidden = project / "configs" / "local.json"
    forbidden.write_text("must not be read")
    read = Path.read_bytes

    def guarded(path):
        assert path != forbidden, "Configuration was read before rejection"
        return read(path)

    monkeypatch.setattr(Path, "read_bytes", guarded)
    with pytest.raises(ValueError, match="Configuration extras"):
        module.freeze(project, tmp_path / "snapshot", ("scripts/isaac_python.sh", "configs/local.json"))
    assert not (tmp_path / "snapshot").exists()


@pytest.fixture
def asset_link(project, tmp_path, monkeypatch):
    shared = tmp_path / "shared_assets"
    target = shared / "robots" / "hand"
    target.mkdir(parents=True)
    (target / "mesh.stl").write_text("shared asset; never copied\n")
    link = project / "bench/simulation/assets/hand"
    link.parent.mkdir()
    link.symlink_to(target, target_is_directory=True)
    monkeypatch.setattr(module, "SHARED_ASSET_ROOT", shared)
    return link, target


def test_registered_asset_reference_preserved_without_copying(project, tmp_path, asset_link, monkeypatch):
    link, asset = asset_link
    read = Path.read_bytes

    def guarded(path):
        assert not path.is_relative_to(asset), "Asset contents must not be copied or read"
        return read(path)

    monkeypatch.setattr(Path, "read_bytes", guarded)
    target = tmp_path / "snapshot"
    result = module.freeze(project, target, ("scripts/isaac_python.sh",),
                           asset_links=("bench/simulation/assets/hand",))
    copied = target / "bench/simulation/assets/hand"
    assert copied.is_symlink() and copied.resolve() == asset
    record = result["shared_asset_references"]["bench/simulation/assets/hand"]
    assert record["original_link_target"] == str(asset)
    assert record["resolved_target"] == str(asset)
    assert record["snapshot_link_target"] == str(asset)
    assert all("mesh.stl" not in name for name in result["files_sha256"])
    assert module.verify(target)["shared_asset_references"] == 1


def test_relative_asset_link_keeps_same_resolved_target(project, tmp_path, asset_link):
    link, asset = asset_link
    link.unlink()
    original = os.path.relpath(asset, link.parent)
    link.symlink_to(original, target_is_directory=True)
    target = tmp_path / "snapshot"
    result = module.freeze(project, target, ("scripts/isaac_python.sh",),
                           asset_links=("bench/simulation/assets/hand",))
    assert result["shared_asset_references"]["bench/simulation/assets/hand"]["original_link_target"] == original
    assert (target / "bench/simulation/assets/hand").resolve() == asset
    assert module.verify(target)["verified"]


def test_unregistered_shared_asset_alias_still_rejected(project, tmp_path, asset_link):
    with pytest.raises(ValueError, match="Symlink"):
        module.freeze(project, tmp_path / "snapshot", ("scripts/isaac_python.sh",))


@pytest.mark.parametrize("case", ["outside_shared", "missing_target", "code_path"])
def test_registered_asset_cannot_escape_shared_root_or_hide_code(project, tmp_path, asset_link, case):
    link, asset = asset_link
    relative = "bench/simulation/assets/hand"
    link.unlink()
    if case == "outside_shared":
        external = tmp_path / "external"
        external.mkdir()
        link.symlink_to(external, target_is_directory=True)
    elif case == "missing_target":
        link.symlink_to(asset / "missing", target_is_directory=True)
    else:
        link = project / "bench" / "aliased_code"
        link.symlink_to(asset, target_is_directory=True)
        relative = "bench/aliased_code"
    with pytest.raises(ValueError, match="Asset"):
        module.freeze(project, tmp_path / "snapshot", ("scripts/isaac_python.sh",),
                       asset_links=(relative,))
    assert not (tmp_path / "snapshot").exists()


def test_verify_rejects_changed_target_and_new_unregistered_alias(project, tmp_path, asset_link):
    link, asset = asset_link
    target = tmp_path / "snapshot"
    module.freeze(project, target, ("scripts/isaac_python.sh",), asset_links=("bench/simulation/assets/hand",))
    copied = target / "bench/simulation/assets/hand"
    other = asset.parent / "other"
    other.mkdir()
    copied.unlink()
    copied.symlink_to(other, target_is_directory=True)
    with pytest.raises(ValueError, match="reference changed"):
        module.verify(target)
    copied.unlink()
    copied.symlink_to(asset, target_is_directory=True)
    (target / "bench/simulation/assets/unregistered").symlink_to(asset, target_is_directory=True)
    with pytest.raises(ValueError, match="Symlink"):
        module.verify(target)


def test_retargeted_source_asset_during_copy_does_not_seal_manifest(project, tmp_path, asset_link, monkeypatch):
    link, asset = asset_link
    other = asset.parent / "other"
    other.mkdir()
    original_copy = module.shutil.copy2
    changed = False

    def retarget(source, destination):
        nonlocal changed
        result = original_copy(source, destination)
        if not changed:
            link.unlink()
            link.symlink_to(other, target_is_directory=True)
            changed = True
        return result

    monkeypatch.setattr(module.shutil, "copy2", retarget)
    target = tmp_path / "snapshot"
    with pytest.raises(ValueError, match="Source changed"):
        module.freeze(project, target, ("scripts/isaac_python.sh",),
                       asset_links=("bench/simulation/assets/hand",))
    assert not (target / "source_snapshot_manifest.json").exists()


def test_read_only_seal_keeps_entrypoint_directly_executable(project, tmp_path):
    # A tiny shell fixture exercises OS exec permissions without starting Isaac.
    (project / "scripts/isaac_python.sh").write_text("#!/bin/sh\nprintf 'fixture probe ok\\n'\n")
    target = tmp_path / "sealed"
    result = module.freeze(project, target, ("scripts/isaac_python.sh",))
    for name in result["files_sha256"]:
        path = target / name
        path.chmod(path.stat().st_mode & ~0o222)
    assert module.verify(target)["file_modes_verified"] is True
    assert (target / "scripts/isaac_python.sh").stat().st_mode & 0o777 == 0o555
    probe = subprocess.run([str(target / "scripts/isaac_python.sh")], check=True,
                           capture_output=True, text=True)
    assert probe.stdout == "fixture probe ok\n"


@pytest.mark.parametrize("name,mode", [("scripts/isaac_python.sh", 0o444),
                                      ("bench/runtime/server.py", 0o755),
                                      ("scripts/isaac_python.sh", 0o777)])
def test_changed_modes_cannot_pass_byte_only_verification(project, tmp_path, name, mode):
    target = tmp_path / "snapshot"
    module.freeze(project, target, ("scripts/isaac_python.sh",))
    (target / name).chmod(mode)
    with pytest.raises(ValueError, match="file mode changed"):
        module.verify(target)


def test_nonexecutable_native_launcher_refused_before_copy(project, tmp_path):
    (project / "scripts/isaac_python.sh").chmod(0o644)
    target = tmp_path / "snapshot"
    with pytest.raises(ValueError, match="must remain executable"):
        module.freeze(project, target, ("scripts/isaac_python.sh",))
    assert not target.exists()


def test_legacy_snapshot_mode_limits_and_new_contract_requirement(project, tmp_path):
    target = tmp_path / "snapshot"
    module.freeze(project, target, ("scripts/isaac_python.sh",))
    path = target / "source_snapshot_manifest.json"
    data = load_json(path)
    data.pop("files_mode_octal"); data.pop("file_mode_policy")
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="Missing file mode contract"):
        module.verify(target)
    data["source_utility_version"] = "complete_benchmark_source_verifier_v2"
    legacy_fixture_layout(target, data)
    path.write_text(json.dumps(data))
    assert module.verify(target)["file_modes_verified"] is False
    (target / "scripts/isaac_python.sh").chmod(0o444)
    with pytest.raises(ValueError, match="must remain executable"):
        module.verify(target)


def test_chmod_during_freeze_leaves_no_manifest(project, tmp_path, monkeypatch):
    copy = module.shutil.copy2
    def change_mode(source, destination):
        result = copy(source, destination)
        (project / "scripts/isaac_python.sh").chmod(0o555)
        return result
    monkeypatch.setattr(module.shutil, "copy2", change_mode)
    target = tmp_path / "snapshot"
    with pytest.raises(ValueError, match="Source changed"):
        module.freeze(project, target, ("scripts/isaac_python.sh",))
    assert not (target / "source_snapshot_manifest.json").exists()


def test_duplicate_manifest_keys_are_not_silently_discarded(project, tmp_path):
    target = tmp_path / "snapshot"
    module.freeze(project, target, ("scripts/isaac_python.sh",))
    path = target / "source_snapshot_manifest.json"
    path.write_text(path.read_text().replace('"snapshot_root":', '"snapshot_root": "wrong", "snapshot_root":'))
    with pytest.raises(ValueError, match="Duplicate manifest key"):
        module.verify(target)


@pytest.mark.parametrize('version', [None, 'complete_benchmark_source_verifier_v2',
                                   'complete_benchmark_source_verifier_v3'])
def test_v4_pins_guard_while_historical_suite_membership_remains_verifiable(project, tmp_path, version):
    helper = project/module.RUNTIME_GUARD_SOURCE
    helper.parent.mkdir(parents=True, exist_ok=True)
    helper.write_text('# pinned runtime guard fixture\n')
    target = tmp_path/'snapshot'
    result = module.freeze(project, target, ('scripts/isaac_python.sh',))
    assert module.RUNTIME_GUARD_SOURCE in result['suite_source_pins']
    assert module.verify(target)['file_modes_verified'] is True
    # Construct historical-format fixtures; no real sealed manifest is edited.
    result['suite_source_pins'].pop(module.RUNTIME_GUARD_SOURCE)
    legacy_fixture_layout(target, result)
    if version is None:
        result.pop('source_utility_version')
    else:
        result['source_utility_version'] = version
    if version != 'complete_benchmark_source_verifier_v3':
        result.pop('files_mode_octal'); result.pop('file_mode_policy')
    (target/'source_snapshot_manifest.json').write_text(json.dumps(result))
    verified = module.verify(target)
    assert verified['file_modes_verified'] is (version == 'complete_benchmark_source_verifier_v3')
