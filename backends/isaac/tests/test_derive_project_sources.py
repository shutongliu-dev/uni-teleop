import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from test_freeze_project_sources import legacy_fixture_layout

from bench.paths import load_json


SOURCE = Path(__file__).resolve().parents[1] / "scripts/maintenance/derive_project_sources.py"
spec = importlib.util.spec_from_file_location("derive_project_sources", SOURCE)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


@pytest.fixture
def fixture(tmp_path):
    source = tmp_path / "source"
    for tree in module.freeze.SOURCE_TREES:
        (source / tree).mkdir(parents=True)
    for name in ("bench/adapters/__init__.py", "bench/runtime/server.py", "bench/__init__.py",
                 "bench/controller.py", "bench/simulation/__init__.py", "bench/simulation/native.py",
                 "scripts/helper.py", "scripts/isaac_python.sh", "bench/apps/native.kit"):
        path = source / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# original\n")
        # This fixture exercises preservation of group/world read bits even
        # when pytest itself was launched under a private 0077 umask.
        path.chmod(0o644)
    (source / "scripts/isaac_python.sh").chmod(0o755)
    parent = tmp_path / "parent"
    module.freeze.freeze(source, parent, ("scripts/isaac_python.sh", "bench/apps/native.kit"))
    candidate = tmp_path / "candidate.py"
    candidate.write_text("VALUE = 2\n")
    overlay = tmp_path / "overlay.json"
    data = {"schema": "complete_benchmark_source_overlay_v1",
            "parent_manifest_sha256": module._sha_file(parent / module.MANIFEST_NAME),
            "entries": [{"target": "bench/runtime/server.py",
                         "expected_parent_sha256": module._sha_file(parent / "bench/runtime/server.py"),
                         "source_path": str(candidate), "source_sha256": module._sha_file(candidate)}]}
    overlay.write_text(json.dumps(data))
    return source, parent, candidate, overlay, data, tmp_path / "derived"


def write(fixture):
    fixture[3].write_text(json.dumps(fixture[4]))


def test_copy_selected_sources_once_preserves_parent_and_has_new_identity(fixture, monkeypatch):
    source, parent, candidate, overlay, data, target = fixture
    parent_bytes = {str(p.relative_to(parent)): (p.read_bytes(), p.stat().st_mode) for p in parent.rglob("*") if p.is_file()}
    added = dict(data["entries"][0], target="bench/experimental/new.py", expected_parent_sha256=None)
    data["entries"].append(added); write(fixture)
    selected = []
    copy = module._copy_selected
    def record(src, dst, digest):
        selected.append((src, dst)); copy(src, dst, digest)
    monkeypatch.setattr(module, "_copy_selected", record)
    manifest = module.derive(parent, target, overlay)
    assert len(selected) == len(manifest["files_sha256"])
    assert [src for src, dst in selected if dst == target / "bench/runtime/server.py"] == [candidate]
    assert (target / "bench/runtime/server.py").read_bytes() == candidate.read_bytes()
    assert all((parent / name).read_bytes() == raw and (parent / name).stat().st_mode == mode for name, (raw, mode) in parent_bytes.items())
    assert set(parent_bytes) == {str(p.relative_to(parent)) for p in parent.rglob("*") if p.is_file()}
    assert module.freeze.verify(target)["verified"]
    assert module.verify_derivation(target)["source_content_changed"]
    provenance = load_json(target / module.PROVENANCE_NAME)
    assert provenance["parent_content_identity_sha256"] != provenance["derived_content_identity_sha256"]
    assert provenance["runtime_contract_generated"] is False
    assert manifest["derivation"]["runtime_contract_rebuild_required"] is True
    assert not any(p.name == "configs" for p in target.rglob("*"))
    with pytest.raises(FileExistsError):
        module.derive(parent, target, overlay)


@pytest.mark.parametrize("kind", ["parent_hash", "candidate_hash", "previous_hash", "add_existing", "replace_missing", "duplicate", "unknown_field"])
def test_explicit_hash_and_operation_boundaries(fixture, kind):
    _, parent, _, overlay, data, target = fixture
    row = data["entries"][0]
    if kind == "parent_hash": data["parent_manifest_sha256"] = "0" * 64
    elif kind == "candidate_hash": row["source_sha256"] = "0" * 64
    elif kind == "previous_hash": row["expected_parent_sha256"] = "0" * 64
    elif kind == "add_existing": row["expected_parent_sha256"] = None
    elif kind == "replace_missing": row["target"] = "bench/missing.py"
    elif kind == "duplicate": data["entries"].append(dict(row))
    else: row["allow_mismatch"] = True
    write(fixture)
    with pytest.raises(ValueError): module.derive(parent, target, overlay)
    assert not target.exists()


@pytest.mark.parametrize("target_name", ["../escape.py", "/tmp/escape.py", "bench//alias.py", "bench/./alias.py",
    "bench/../alias.py", "bench\\alias.py", "configs/local.json", "bench/credentials.py", "bench/configs/a.py",
    "bench/apps/new.kit", "scripts/new.sh", "random/new.py", "bench/data.npy", "source_snapshot_manifest.json"])
def test_invalid_target_rejected_before_candidate_read(fixture, monkeypatch, target_name):
    _, parent, candidate, overlay, data, target = fixture
    data["entries"][0]["target"] = target_name; data["entries"][0]["expected_parent_sha256"] = None; write(fixture)
    read = Path.read_bytes
    def guard(path):
        assert path != candidate
        return read(path)
    monkeypatch.setattr(Path, "read_bytes", guard)
    with pytest.raises(ValueError): module.derive(parent, target, overlay)
    assert not target.exists()


@pytest.mark.parametrize("kind", ["source_link", "source_parent_link", "source_dotdot", "hardlink", "parent_link", "output_link", "output_dotdot", "output_inside_source"])
def test_source_and_destination_aliases_fail(fixture, tmp_path, kind):
    source, parent, candidate, overlay, data, target = fixture
    row = data["entries"][0]
    if kind == "source_link":
        alias = tmp_path / "alias.py"; alias.symlink_to(candidate); row["source_path"] = str(alias)
    elif kind == "source_parent_link":
        alias = tmp_path / "alias"; alias.symlink_to(tmp_path, target_is_directory=True); row["source_path"] = str(alias / candidate.name)
    elif kind == "source_dotdot": row["source_path"] = str(tmp_path / "parent" / ".." / candidate.name)
    elif kind == "hardlink": os.link(candidate, tmp_path / "linked.py")
    elif kind == "parent_link":
        alias = tmp_path / "alias"; alias.symlink_to(parent, target_is_directory=True); parent = alias
    elif kind == "output_link":
        alias = tmp_path / "alias"; alias.symlink_to(source, target_is_directory=True); target = alias / "new"
    elif kind == "output_dotdot": target = str(tmp_path / "parent" / ".." / "derived")
    else: target = source / "new"
    write(fixture)
    with pytest.raises(ValueError): module.derive(parent, target, overlay)
    assert not fixture[-1].exists()


@pytest.mark.parametrize("relative", ["configs/local.py", "secrets.py", ".env", "credentials/safe.py", "api_key.py", "client.key"])
def test_unsafe_source_name_refused_without_reading(fixture, tmp_path, monkeypatch, relative):
    _, parent, _, overlay, data, target = fixture
    bad = tmp_path / relative; bad.parent.mkdir(parents=True, exist_ok=True); bad.write_text("not to read")
    data["entries"][0]["source_path"] = str(bad); write(fixture)
    read = Path.read_bytes
    def guard(path):
        assert path != bad
        return read(path)
    monkeypatch.setattr(Path, "read_bytes", guard)
    with pytest.raises(ValueError): module.derive(parent, target, overlay)
    assert not target.exists()


@pytest.mark.parametrize("mutation", ["parent_file", "parent_added", "parent_manifest", "candidate", "overlay"])
def test_inputs_changed_during_copy_leave_no_completed_manifest(fixture, monkeypatch, mutation):
    _, parent, candidate, overlay, data, target = fixture
    copy = module._copy_selected; once = False
    def mutate(src, dst, digest):
        nonlocal once
        copy(src, dst, digest)
        if once: return
        once = True
        if mutation == "parent_file": (parent / "bench/runtime/server.py").write_text("changed")
        elif mutation == "parent_added": (parent / "bench/new.py").write_text("changed")
        elif mutation == "parent_manifest": (parent / module.MANIFEST_NAME).write_text("changed")
        elif mutation == "candidate": candidate.write_text("changed")
        else: overlay.write_text("changed")
    monkeypatch.setattr(module, "_copy_selected", mutate)
    with pytest.raises(ValueError): module.derive(parent, target, overlay)
    assert target.exists() and not (target / module.MANIFEST_NAME).exists()


def test_existing_extra_may_be_replaced_but_not_new_resource(fixture):
    _, parent, candidate, overlay, data, target = fixture
    data["entries"][0]["target"] = "bench/apps/native.kit"
    data["entries"][0]["expected_parent_sha256"] = module._sha_file(parent / "bench/apps/native.kit"); write(fixture)
    result = module.derive(parent, target, overlay)
    assert result["explicit_extra_resources"] == ["scripts/isaac_python.sh", "bench/apps/native.kit"]
    assert (target / "bench/apps/native.kit").read_bytes() == candidate.read_bytes()


@pytest.mark.parametrize("tree", ["agent", "scenes"])
def test_new_layout_overlay_includes_agent_and_scene_sources(fixture, tree):
    _, parent, candidate, overlay, data, target = fixture
    relative = f"{tree}/added.py"
    data["entries"][0].update(target=relative, expected_parent_sha256=None)
    write(fixture)
    result = module.derive(parent, target, overlay)
    assert result["source_trees"] == list(module.freeze.SOURCE_TREES)
    assert result["suite_source_pins"][relative] == module._sha_file(candidate)
    assert module.verify_derivation(target)["offline_parent_transition_verified"]


def test_legacy_overlay_cannot_add_an_uninventoried_module(fixture, monkeypatch):
    _, parent, candidate, overlay, data, target = fixture
    path = parent / module.MANIFEST_NAME
    manifest = load_json(path)
    manifest["source_utility_version"] = "complete_benchmark_source_verifier_v4"
    legacy_fixture_layout(parent, manifest)
    path.write_text(json.dumps(manifest))
    data["parent_manifest_sha256"] = module._sha_file(path)
    data["entries"][0].update(target="agent/hidden.py", expected_parent_sha256=None)
    write(fixture)
    read = Path.read_bytes

    def guarded(path):
        assert path != candidate, "Invalid layout must be refused before reading an overlay"
        return read(path)

    monkeypatch.setattr(Path, "read_bytes", guarded)
    with pytest.raises(ValueError, match="Only source-tree Python"):
        module.derive(parent, target, overlay)
    assert not target.exists()


def test_candidate_changed_after_its_copy_still_refused(fixture, monkeypatch):
    _, parent, candidate, overlay, _, target = fixture
    copy = module._copy_selected
    def mutate(src, dst, digest):
        copy(src, dst, digest)
        if src == candidate:
            candidate.write_text("VALUE = 3\n")
    monkeypatch.setattr(module, "_copy_selected", mutate)
    with pytest.raises(ValueError, match="Candidate changed while deriving"):
        module.derive(parent, target, overlay)
    assert not (target / module.MANIFEST_NAME).exists()


def test_addition_cannot_treat_existing_file_as_directory(fixture):
    _, parent, _, overlay, data, target = fixture
    data["entries"][0].update(target="bench/controller.py/new.py", expected_parent_sha256=None); write(fixture)
    with pytest.raises(ValueError, match="collide with parent directories"):
        module.derive(parent, target, overlay)
    assert not target.exists()


def test_asset_reference_preserved_and_cannot_be_overlaid(fixture, tmp_path, monkeypatch):
    source, parent, candidate, overlay, data, target = fixture
    assets = tmp_path / "assets"; hand = assets / "robot"; hand.mkdir(parents=True)
    (hand / "mesh.stl").write_text("not copied")
    monkeypatch.setattr(module.freeze, "SHARED_ASSET_ROOT", assets)
    link = source / "bench/simulation/assets/hand"; link.parent.mkdir(); link.symlink_to(hand, target_is_directory=True)
    parent = tmp_path / "parent_assets"
    module.freeze.freeze(source, parent, ("scripts/isaac_python.sh", "bench/apps/native.kit"), asset_links=("bench/simulation/assets/hand",))
    data["parent_manifest_sha256"] = module._sha_file(parent / module.MANIFEST_NAME); write(fixture)
    read = Path.read_bytes
    def guard(path):
        assert not path.is_relative_to(hand)
        return read(path)
    monkeypatch.setattr(Path, "read_bytes", guard)
    module.derive(parent, target, overlay)
    assert (target / "bench/simulation/assets/hand").is_symlink()
    assert (target / "bench/simulation/assets/hand").resolve() == hand
    assert module.freeze.verify(target)["shared_asset_references"] == 1
    data["entries"][0].update(target="bench/simulation/assets/hand/sneak.py", expected_parent_sha256=None); write(fixture)
    with pytest.raises(ValueError, match="shared asset reference"):
        module.derive(parent, tmp_path / "bad_derived", overlay)


def test_asset_target_changed_during_derivation_has_no_seal(fixture, tmp_path, monkeypatch):
    source, _, _, overlay, data, target = fixture
    assets = tmp_path / "assets"; hand = assets / "robot"; hand.mkdir(parents=True)
    other = assets / "other"; other.mkdir()
    monkeypatch.setattr(module.freeze, "SHARED_ASSET_ROOT", assets)
    link = source / "bench/simulation/assets/hand"; link.parent.mkdir(); link.symlink_to(hand, target_is_directory=True)
    parent = tmp_path / "parent_assets"
    module.freeze.freeze(source, parent, ("scripts/isaac_python.sh", "bench/apps/native.kit"), asset_links=("bench/simulation/assets/hand",))
    data["parent_manifest_sha256"] = module._sha_file(parent / module.MANIFEST_NAME); write(fixture)
    copy = module._copy_selected; changed = False
    def retarget(src, dst, digest):
        nonlocal changed
        copy(src, dst, digest)
        if not changed:
            reference = parent / "bench/simulation/assets/hand"
            reference.unlink(); reference.symlink_to(other, target_is_directory=True); changed = True
    monkeypatch.setattr(module, "_copy_selected", retarget)
    with pytest.raises(ValueError, match="reference changed"):
        module.derive(parent, target, overlay)
    assert not (target / module.MANIFEST_NAME).exists()


@pytest.mark.parametrize("change", ["provenance", "runtime_identity", "added_source"])
def test_derived_verification_detects_changes(fixture, change):
    _, parent, _, overlay, _, target = fixture
    module.derive(parent, target, overlay)
    if change == "provenance": (target / module.PROVENANCE_NAME).write_text("{}")
    elif change == "added_source": (target / "bench/extra.py").write_text("added")
    else:
        path = target / module.MANIFEST_NAME; data = load_json(path)
        data["derivation"]["runtime_source_identity_sha256"] = "0" * 64; path.write_text(json.dumps(data))
    with pytest.raises(ValueError): module.verify_derivation(target)


def test_derived_module_imports_from_actual_root(fixture):
    _, parent, _, overlay, _, target = fixture
    module.derive(parent, target, overlay)
    script = (
        "import importlib.util, json; import bench.runtime.server; "
        f"s=importlib.util.spec_from_file_location('verifier', {str(module._VERIFIER_PATH)!r}); "
        "m=importlib.util.module_from_spec(s); s.loader.exec_module(m); "
        f"r=m.verify_runtime_imports(__import__('pathlib').Path({str(target)!r}), required_modules=('bench.runtime.server',)); "
        "assert bench.runtime.server.VALUE == 2; print(json.dumps(r))"
    )
    completed = subprocess.run([sys.executable, "-c", script], cwd=target, env={**os.environ, "PYTHONPATH": str(target)},
                               text=True, capture_output=True, check=True)
    result = json.loads(completed.stdout)
    assert result["runtime_imports_verified"]
    assert result["imported_project_modules"]["bench.runtime.server"]["path"] == str(target / "bench/runtime/server.py")


def test_duplicate_json_keys_refused(fixture):
    _, parent, _, overlay, data, target = fixture
    overlay.write_text(json.dumps(data).replace('"entries":', '"entries": [], "entries":', 1))
    with pytest.raises(ValueError, match="Duplicate JSON key"): module.derive(parent, target, overlay)
    assert not target.exists()


def rewrite_local_checksums(target, change):
    """Attack only local metadata; no claim to bypass external READY pins."""
    provenance_path = target / module.PROVENANCE_NAME
    manifest_path = target / module.MANIFEST_NAME
    provenance = load_json(provenance_path)
    manifest = load_json(manifest_path)
    change(provenance, manifest)
    provenance_path.write_text(json.dumps(provenance, indent=2) + '\n')
    manifest['derivation']['provenance_sha256'] = module._sha_file(provenance_path)
    manifest['derivation']['runtime_source_identity_sha256'] = module._canonical_hash({
        'snapshot_root': str(target),
        'content_identity_sha256': manifest['derivation']['content_identity_sha256'],
        'provenance_sha256': manifest['derivation']['provenance_sha256']})
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')


@pytest.mark.parametrize('legacy', [False, True])
@pytest.mark.parametrize('case', ['parent_root_mismatch', 'selected_overlay_mismatch', 'embedded_overlay_mismatch'])
def test_root_review_contradictions_rejected_with_local_hashes_recomputed(fixture, case, legacy):
    _, parent, _, overlay, _, target = fixture
    module.derive(parent, target, overlay)
    def mutate(provenance, manifest):
        if legacy:
            provenance['schema'] = module.LEGACY_VERSION
            provenance.pop('parent_manifest_utf8'); provenance.pop('overlay_utf8')
        if case == 'parent_root_mismatch':
            provenance['parent_snapshot_root'] += '_different'
        elif case == 'selected_overlay_mismatch':
            provenance['selected_overlays'][0]['target'] = 'bench/controller.py'
        else:
            provenance['overlay']['entries'][0]['target'] = 'bench/controller.py'
    rewrite_local_checksums(target, mutate)
    with pytest.raises(ValueError):
        module.verify_derivation(target)


@pytest.mark.parametrize('case', ['candidate_sha', 'operation', 'old_sha', 'source_path', 'undeclared_change',
    'removed_parent_file', 'parent_document_bytes', 'overlay_document_bytes', 'ancestry', 'parent_content_identity'])
def test_offline_parent_overlay_inventory_semantics_bound(fixture, case):
    _, parent, _, overlay, _, target = fixture
    module.derive(parent, target, overlay)
    def mutate(provenance, manifest):
        entry = provenance['overlay']['entries'][0]
        record = provenance['selected_overlays'][0]
        if case == 'candidate_sha':
            entry['source_sha256'] = '0' * 64; record['source_sha256'] = '0' * 64
        elif case == 'operation': record['operation'] = 'add'
        elif case == 'old_sha':
            entry['expected_parent_sha256'] = None
            record['expected_parent_sha256'] = None; record['operation'] = 'add'
        elif case == 'source_path': record['source_path'] += '_other'
        elif case == 'undeclared_change':
            path = target / 'bench/controller.py'; path.write_text('CHANGED = True\n')
            manifest['files_sha256']['bench/controller.py'] = module._sha_file(path)
        elif case == 'removed_parent_file':
            (target / 'bench/controller.py').unlink(); manifest['files_sha256'].pop('bench/controller.py')
        elif case == 'parent_document_bytes': provenance['parent_manifest_utf8'] += ' '
        elif case == 'overlay_document_bytes': provenance['overlay_utf8'] += ' '
        elif case == 'ancestry': manifest['source_root_ancestors'] = []
        else: provenance['parent_content_identity_sha256'] = '0' * 64
        if case in {'candidate_sha', 'old_sha'}:
            # Also update original-overlay representation to exercise the actual
            # parent/derived transition check, not just embedded-document hash.
            provenance['overlay_utf8'] = json.dumps(provenance['overlay'])
            provenance['overlay_sha256'] = module._sha_bytes(provenance['overlay_utf8'].encode())
        manifest['suite_source_pins'] = module.freeze.suite_inventory(manifest['files_sha256'])
        identity = module._canonical_hash({'files_sha256': manifest['files_sha256'], 'shared_asset_targets': {}})
        manifest['derivation']['content_identity_sha256'] = identity
        provenance['derived_content_identity_sha256'] = identity
    rewrite_local_checksums(target, mutate)
    with pytest.raises(ValueError): module.verify_derivation(target)


def test_v2_verifies_offline_after_original_parent_and_candidates_removed(fixture):
    source, parent, candidate, overlay, _, target = fixture
    module.derive(parent, target, overlay)
    shutil.rmtree(source); shutil.rmtree(parent); candidate.unlink(); overlay.unlink()
    report = module.verify_derivation(target)
    assert report['derivation_verified'] is True
    assert report['offline_parent_transition_verified'] is True


def test_legacy_v1_is_explicitly_limited_even_when_self_hashes_match(fixture):
    _, parent, _, overlay, _, target = fixture
    module.derive(parent, target, overlay)
    def downgrade(provenance, manifest):
        provenance['schema'] = module.LEGACY_VERSION
        provenance.pop('parent_manifest_utf8'); provenance.pop('overlay_utf8')
    rewrite_local_checksums(target, downgrade)
    result = module.verify_derivation(target)
    assert result['verified'] is True  # freeze source inventory only
    assert result['derivation_verified'] is False
    assert result['offline_parent_transition_verified'] is False
    assert 'LIMITED legacy v1' in result['derivation_verification_scope']


def test_derivation_preserves_sealed_modes_under_restrictive_umask(fixture):
    _, parent, _, overlay, _, target = fixture
    # The file hashes/manifest stay unchanged when the parent is sealed.
    for path in parent.rglob("*"):
        if path.is_file():
            path.chmod(path.stat().st_mode & ~0o222)
    previous = os.umask(0o077)
    try:
        result = module.derive(parent, target, overlay)
    finally:
        os.umask(previous)
    assert result["files_mode_octal"]["scripts/isaac_python.sh"] == "0555"
    assert result["files_mode_octal"]["bench/controller.py"] == "0444"
    assert module.verify_derivation(target)["file_modes_verified"] is True


def test_legacy_parent_remains_derivable_without_editing_its_manifest(fixture):
    _, parent, _, overlay, data, target = fixture
    path = parent / module.MANIFEST_NAME
    old = load_json(path)
    old.pop("files_mode_octal"); old.pop("file_mode_policy"); old.pop("source_utility_version")
    legacy_fixture_layout(parent, old)
    path.write_text(json.dumps(old)); before = path.read_bytes()
    data["parent_manifest_sha256"] = module._sha_file(path); write(fixture)
    module.derive(parent, target, overlay)
    assert module.verify_derivation(target)["file_modes_verified"] is True
    assert path.read_bytes() == before


def test_candidate_mode_change_during_copy_has_no_manifest(fixture, monkeypatch):
    _, parent, candidate, overlay, _, target = fixture
    copy = module._copy_selected
    def mutate(src, dst, digest):
        copy(src, dst, digest)
        if src == candidate:
            candidate.chmod(0o755)
    monkeypatch.setattr(module, "_copy_selected", mutate)
    with pytest.raises(ValueError, match="changed during copying"):
        module.derive(parent, target, overlay)
    assert not (target / module.MANIFEST_NAME).exists()


def test_overlay_cannot_remove_native_launcher_execute_permission(fixture):
    _, parent, _, overlay, data, target = fixture
    data["entries"][0]["target"] = "scripts/isaac_python.sh"
    data["entries"][0]["expected_parent_sha256"] = module._sha_file(parent / "scripts/isaac_python.sh")
    write(fixture)
    with pytest.raises(ValueError, match="must remain executable"):
        module.derive(parent, target, overlay)
    assert not (target / module.MANIFEST_NAME).exists()


def test_derived_mode_contract_bound_to_selection_provenance(fixture):
    _, parent, _, overlay, _, target = fixture
    module.derive(parent, target, overlay)
    path = target / module.MANIFEST_NAME
    manifest = load_json(path)
    manifest["files_mode_octal"]["bench/runtime/server.py"] = "0755"
    (target / "bench/runtime/server.py").chmod(0o755)
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="differs from selected source provenance"):
        module.verify_derivation(target)
