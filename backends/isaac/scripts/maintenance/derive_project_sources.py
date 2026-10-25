#!/usr/bin/env python3
"""Derive a fresh source snapshot from verified parent bytes and explicit overlays.

The parent and candidates are read only. Every destination file is copied once
from its selected source; an interrupted derivation has no completed manifest.
The resulting freeze-v2-compatible snapshot has a new source/runtime identity.
It does not generate or authorize a benchmark runtime contract or native run.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import stat
import sys


_VERIFIER_PATH = Path(__file__).resolve().with_name("freeze_project_sources.py")
_SPEC = importlib.util.spec_from_file_location("source_derivation_freeze_verifier", _VERIFIER_PATH)
freeze = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(freeze)
VERSION = "explicit_source_derivation_v2"
LEGACY_VERSION = "explicit_source_derivation_v1"
MANIFEST_NAME = "source_snapshot_manifest.json"
PROVENANCE_NAME = "source_derivation.json"


def _sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha_file(path: Path) -> str:
    return _sha_bytes(path.read_bytes())


def _hash(value, label):
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(f"Expected lowercase SHA256: {label}")
    return value


def _safe_parts(path: Path):
    for part in path.parts:
        name = part.lower()
        if name in {"config", "configs", "configuration", "configurations"}:
            raise ValueError(f"Configuration path is not an allowed source: {path}")
        if (any(word in name for word in ("credential", "secret", ".env", "password", "passwd", "api_key", "apikey"))
                or name in {".netrc", "id_rsa", "id_ed25519", "token", "tokens", "auth.json"}
                or Path(name).suffix in {".pem", ".key", ".p12", ".pfx"}):
            raise ValueError(f"Credential-like path is not an allowed source: {path}")


def _absolute_name(value) -> Path:
    """Validate a historical path spelling without requiring its old file."""
    raw = os.fspath(value)
    path = Path(raw)
    if not path.is_absolute() or str(path) != raw or ".." in path.parts:
        raise ValueError(f"Expected canonical absolute path without aliases: {raw}")
    _safe_parts(path)
    return path


def _absolute(value, *, must_exist=True) -> Path:
    path = _absolute_name(value)
    if path.resolve(strict=must_exist) != path:
        raise ValueError(f"Aliased absolute path: {path}")
    for ancestor in (path, *path.parents):
        if ancestor.is_symlink():
            raise ValueError(f"Symlink alias in path: {path}")
    return path


def _relative(value) -> str:
    if not isinstance(value, str):
        raise ValueError("Target paths must be strings")
    path = Path(value)
    if (not value or path.is_absolute() or str(path) != value or ".." in path.parts
            or value in {".", MANIFEST_NAME, PROVENANCE_NAME} or "\\" in value):
        raise ValueError(f"Expected canonical relative source path: {value}")
    _safe_parts(path)
    return value


def _regular(path: Path):
    _absolute(path)
    info = path.stat()
    if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
        raise ValueError(f"Expected regular source without hardlink aliases: {path}")
    return info


def _json(raw: bytes):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"Duplicate JSON key: {key}")
            result[key] = value
        return result
    return json.loads(raw, object_pairs_hook=pairs)


def _allowed(relative: str, extras: set[str], source_trees=freeze.SOURCE_TREES):
    path = Path(_relative(relative))
    if relative not in extras and not (len(path.parts) > 1 and path.parts[0] in (*source_trees, "scripts") and path.suffix == ".py"):
        raise ValueError(f"Only source-tree Python or an existing parent extra may be selected: {relative}")


def _parent(parent: Path, expected_sha: str):
    manifest_path = parent / MANIFEST_NAME
    _regular(manifest_path)
    raw = manifest_path.read_bytes()
    if _sha_bytes(raw) != expected_sha:
        raise ValueError("Parent manifest SHA256 changed or does not match overlay")
    manifest = _json(raw)
    source_trees = freeze.snapshot_source_trees(manifest)
    files = manifest.get("files_sha256")
    extras = manifest.get("explicit_extra_resources")
    assets = manifest.get("shared_asset_references", {})
    if not isinstance(files, dict) or not isinstance(extras, list) or not isinstance(assets, dict):
        raise ValueError("Malformed complete parent inventory")
    if len(set(extras)) != len(extras):
        raise ValueError("Duplicate parent extra resources")
    for name in extras:
        _relative(name)
        if name not in files:
            raise ValueError(f"Parent extra is absent from inventory: {name}")
    for name, digest in files.items():
        _allowed(name, set(extras), source_trees); _hash(digest, name)
    for name in assets:
        _relative(name)
    # Inspect names/aliases before a verifier could read forbidden source data.
    actual_names = set(extras)
    for tree in (*source_trees, "scripts"):
        actual_names.update(freeze.source_tree_files(parent, tree, required=tree != "scripts", asset_links=set(assets)))
    for name in actual_names:
        _allowed(name, set(extras), source_trees); _regular(parent / name)
    if actual_names != set(files):
        raise ValueError("Parent complete source inventory changed")
    freeze.verify(parent)
    return manifest


def _copy_selected(source: Path, destination: Path, expected_sha: str):
    info = _regular(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    _absolute(destination, must_exist=False)
    # O_EXCL/O_NOFOLLOW avoids overwriting an injected destination alias. An
    # incomplete output remains evidence, never an authorized snapshot.
    source_fd = os.open(source, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        current = os.fstat(source_fd)
        if (current.st_dev, current.st_ino, current.st_nlink) != (info.st_dev, info.st_ino, 1):
            raise ValueError(f"Source identity changed before copying: {source}")
        with os.fdopen(source_fd, "rb", closefd=False) as stream:
            raw = stream.read()
        if _sha_bytes(raw) != expected_sha:
            raise ValueError(f"Selected source bytes changed while copying: {source}")
        target_fd = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, stat.S_IMODE(info.st_mode))
        with os.fdopen(target_fd, "wb") as stream:
            stream.write(raw)
            stream.flush()
            # os.open applies the caller's umask; preserve the selected source's
            # permissions explicitly, including the native launcher's execute bits.
            os.fchmod(stream.fileno(), stat.S_IMODE(info.st_mode))
    finally:
        os.close(source_fd)


def _canonical_hash(value):
    return _sha_bytes(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode())


def derive(parent, destination, overlay_file) -> dict:
    parent = _absolute(parent)
    destination = _absolute(destination, must_exist=False)
    overlay_file = _absolute(overlay_file)
    if destination.exists():
        raise FileExistsError(destination)
    if not parent.is_dir():
        raise ValueError("Parent snapshot must be a directory")
    _regular(overlay_file)
    overlay_raw = overlay_file.read_bytes()
    overlay = _json(overlay_raw)
    if (not isinstance(overlay, dict) or set(overlay) != {"schema", "parent_manifest_sha256", "entries"}
            or overlay["schema"] != "complete_benchmark_source_overlay_v1"
            or not isinstance(overlay["entries"], list) or not overlay["entries"]):
        raise ValueError("Expected explicit nonempty source overlay v1")
    parent_sha = _hash(overlay["parent_manifest_sha256"], "parent manifest")
    manifest = _parent(parent, parent_sha)
    source_trees = freeze.snapshot_source_trees(manifest)
    parent_manifest_raw = (parent / MANIFEST_NAME).read_bytes()
    if _sha_bytes(parent_manifest_raw) != parent_sha:
        raise ValueError("Parent manifest changed before offline provenance capture")
    original_roots = manifest.get("source_root_ancestors", [])
    if not isinstance(original_roots, list):
        raise ValueError("Malformed parent source root ancestry")
    protected_roots = {parent, _absolute(manifest["source_root"], must_exist=False)}
    protected_roots.update(_absolute(root, must_exist=False) for root in original_roots)
    if any(destination.is_relative_to(root) or root.is_relative_to(destination) for root in protected_roots):
        raise ValueError("Derived snapshot must be disjoint from parent and original source roots")
    files = manifest["files_sha256"]
    extras = set(manifest["explicit_extra_resources"])
    assets = manifest.get("shared_asset_references", {})
    selected = {name: (parent / name, digest) for name, digest in files.items()}
    records, destinations = [], set()
    fields = {"target", "expected_parent_sha256", "source_path", "source_sha256"}
    for entry in overlay["entries"]:
        if not isinstance(entry, dict) or set(entry) != fields:
            raise ValueError("Overlay entry must name target, expected parent SHA, absolute source and SHA")
        name = _relative(entry["target"])
        if name in destinations:
            raise ValueError(f"Duplicate overlay target: {name}")
        destinations.add(name)
        _allowed(name, extras, source_trees)
        target = Path(name)
        if any(target.is_relative_to(Path(link)) or Path(link).is_relative_to(target) for link in assets):
            raise ValueError(f"Overlay intersects registered shared asset reference: {name}")
        previous = entry["expected_parent_sha256"]
        if previous is not None:
            _hash(previous, "expected parent")
        if previous != files.get(name) or (name in files and previous is None):
            raise ValueError(f"Overlay parent SHA does not match existing/addition status: {name}")
        source = _absolute(entry["source_path"])
        _regular(source)
        candidate_sha = _hash(entry["source_sha256"], "candidate")
        if _sha_file(source) != candidate_sha:
            raise ValueError(f"Candidate SHA256 mismatch: {source}")
        selected[name] = (source, candidate_sha)
        records.append({"target": name, "expected_parent_sha256": previous,
                        "source_path": str(source), "source_sha256": candidate_sha,
                        "operation": "add" if previous is None else "replace",
                        "content_changed": previous != candidate_sha})
    names = [Path(name) for name in selected]
    if any(a != b and a in b.parents for a in names for b in names):
        raise ValueError("Selected file paths collide with parent directories")
    before_assets = freeze.asset_references(parent, tuple(assets), source_trees=source_trees)
    tool_sources = {str(Path(__file__).resolve()): _sha_file(Path(__file__).resolve()),
                    str(_VERIFIER_PATH): _sha_file(_VERIFIER_PATH)}
    expected_files = {name: digest for name, (_, digest) in sorted(selected.items())}
    expected_modes = {name: format(stat.S_IMODE(_regular(source).st_mode), "04o")
                      for name, (source, _) in sorted(selected.items())}
    destination.mkdir(parents=True, exist_ok=False)
    for tree in source_trees:
        (destination / tree).mkdir()
    for name, (source, digest) in sorted(selected.items()):
        _copy_selected(source, destination / name, digest)
    for name, reference in before_assets.items():
        link = destination / name
        link.parent.mkdir(parents=True, exist_ok=True)
        _absolute(link, must_exist=False)
        link.symlink_to(reference["resolved_target"], target_is_directory=Path(reference["resolved_target"]).is_dir())
    expected_assets = {name: {"link_target": reference["resolved_target"], "resolved_target": reference["resolved_target"]}
                       for name, reference in before_assets.items()}
    # Revalidate every input after all output writes. This includes unchanged
    # parent inventory, overlay manifest, candidates, verifier and shared refs.
    _parent(parent, parent_sha)
    _regular(overlay_file)
    if overlay_file.read_bytes() != overlay_raw:
        raise ValueError("Overlay manifest changed while deriving snapshot")
    for record in records:
        source = Path(record["source_path"]); _regular(source)
        if _sha_file(source) != record["source_sha256"]:
            raise ValueError(f"Candidate changed while deriving snapshot: {source}")
    if any(_sha_file(Path(path)) != digest for path, digest in tool_sources.items()):
        raise ValueError("Derivation utility or verifier changed")
    current = freeze.inventory(destination, tuple(manifest["explicit_extra_resources"]), tuple(assets),
                               source_trees=source_trees)
    current_modes = freeze.file_modes(destination, current)
    final_selected_modes = {name: format(stat.S_IMODE(_regular(source).st_mode), "04o")
                            for name, (source, _) in sorted(selected.items())}
    if (current != expected_files
            or freeze.asset_references(parent, tuple(assets), source_trees=source_trees) != before_assets
            or freeze.asset_references(destination, tuple(assets), source_trees=source_trees) != expected_assets
            or current_modes != expected_modes or final_selected_modes != expected_modes):
        raise ValueError("Derived inventory/assets changed during copying")
    freeze._verify_native_entrypoint(destination, current)
    content_changed = current != files
    asset_identity = {name: row["resolved_target"] for name, row in expected_assets.items()}
    content_identity = _canonical_hash({"files_sha256": current, "shared_asset_targets": asset_identity})
    parent_content_identity = _canonical_hash({"files_sha256": files, "shared_asset_targets": asset_identity})
    provenance = {"schema": VERSION, "created_at_utc": datetime.now(timezone.utc).isoformat(),
                  "parent_snapshot_root": str(parent), "parent_manifest_sha256": parent_sha,
                  "parent_manifest_utf8": parent_manifest_raw.decode("utf-8"),
                  "overlay_path": str(overlay_file), "overlay_sha256": _sha_bytes(overlay_raw),
                  "overlay_utf8": overlay_raw.decode("utf-8"),
                  "overlay": overlay, "selected_overlays": records, "tool_sources_sha256": tool_sources,
                  "parent_and_candidates_verified_before_after": True,
                  "parent_content_identity_sha256": parent_content_identity,
                  "derived_content_identity_sha256": content_identity, "source_content_changed": content_changed,
                  "selected_files_mode_octal": expected_modes,
                  "native_launches": 0, "model_calls": 0, "runtime_imports_verified": False,
                  "runtime_contract_generated": False, "runtime_contract_rebuild_required": True,
                  "byte_identical_path_rebase_is_not_a_contract_for_this_derivation": True,
                  "runtime_policy": "Generate a new contract from the selected derived source bytes and actual launch paths. Never accept a parent runtime hash for changed code; this source identity is not a native runtime contract."}
    provenance_raw = (json.dumps(provenance, indent=2, allow_nan=False) + "\n").encode()
    with (destination / PROVENANCE_NAME).open("xb") as stream:
        stream.write(provenance_raw)
    version = (freeze.VERIFIER_VERSION if source_trees == freeze.SOURCE_TREES
               else freeze.V5_VERIFIER_VERSION if source_trees == freeze.V5_SOURCE_TREES
               else "complete_benchmark_source_verifier_v4")
    result = {"schema": "complete_benchmark_source_snapshot_v1", "source_utility_version": version,
              **({"source_trees": list(source_trees)} if source_trees != freeze.LEGACY_SOURCE_TREES else {}),
              "created_at_utc": provenance["created_at_utc"], "source_root": str(parent),
              "source_root_ancestors": sorted(str(root) for root in protected_roots),
              "snapshot_root": str(destination), "files_sha256": current,
              "files_mode_octal": current_modes, "file_mode_policy": freeze.MODE_POLICY,
              "suite_source_pins": freeze.suite_inventory(current, source_trees=source_trees),
              "explicit_extra_resources": manifest["explicit_extra_resources"],
              "shared_asset_references": {name: {"original_link": name, "original_link_target": ref["link_target"],
                  "resolved_target": ref["resolved_target"], "snapshot_link_target": ref["resolved_target"]}
                  for name, ref in before_assets.items()},
              "shared_asset_scope": manifest.get("shared_asset_scope"),
              "credential_files_selected": False, "embedded_secret_scan_performed": False,
              "credential_handling": "Configuration and credential-like paths refused before content selection; source authors remain responsible for embedded credentials.",
              "native_validation_performed": False, "runtime_imports_verified": False,
              "derivation": {"provenance_file": PROVENANCE_NAME, "provenance_sha256": _sha_bytes(provenance_raw),
                  "parent_manifest_sha256": parent_sha, "source_content_changed": content_changed,
                  "content_identity_sha256": content_identity,
                  "runtime_source_identity_sha256": _canonical_hash({"snapshot_root": str(destination),
                     "content_identity_sha256": content_identity, "provenance_sha256": _sha_bytes(provenance_raw)}),
                  "runtime_contract_rebuild_required": True},
              "usage": "Use this root for every process. Verify source inventory and actual runtime imports; generate a new benchmark runtime contract before execution."}
    with (destination / MANIFEST_NAME).open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False); stream.write("\n")
    try:
        verify_derivation(destination)
    except Exception:
        # This file was created by this call, never a pre-existing manifest.
        # Preserve the failed output/provenance but not a completed seal.
        (destination / MANIFEST_NAME).unlink()
        raise
    return result


def verify_derivation(destination) -> dict:
    destination = _absolute(destination)
    result = freeze.verify(destination)
    manifest = _json((destination / MANIFEST_NAME).read_bytes())
    derivation = manifest.get("derivation", {})
    if derivation.get("provenance_file") != PROVENANCE_NAME:
        raise ValueError("Missing explicit source derivation provenance")
    path = destination / PROVENANCE_NAME
    _regular(path)
    raw = path.read_bytes()
    if _sha_bytes(raw) != derivation.get("provenance_sha256"):
        raise ValueError("Source derivation provenance changed")
    provenance = _json(raw)
    if (manifest.get("files_mode_octal") is not None
            and provenance.get("selected_files_mode_octal") != manifest["files_mode_octal"]):
        raise ValueError("Derived file mode contract differs from selected source provenance")
    version = provenance.get("schema")
    if version not in {VERSION, LEGACY_VERSION} or provenance.get("parent_manifest_sha256") != derivation.get("parent_manifest_sha256"):
        raise ValueError("Source derivation parent identity mismatch")
    _verify_overlay_records(manifest, provenance)
    if version == VERSION:
        _verify_offline_transition(manifest, provenance)
    content_identity = _canonical_hash({"files_sha256": manifest["files_sha256"],
        "shared_asset_targets": {name: row["resolved_target"] for name, row in manifest["shared_asset_references"].items()}})
    runtime_identity = _canonical_hash({"snapshot_root": str(destination),
        "content_identity_sha256": content_identity, "provenance_sha256": _sha_bytes(raw)})
    changed = provenance.get("parent_content_identity_sha256") != content_identity
    if (derivation.get("content_identity_sha256") != content_identity
            or provenance.get("derived_content_identity_sha256") != content_identity
            or derivation.get("runtime_source_identity_sha256") != runtime_identity
            or derivation.get("source_content_changed") is not changed
            or provenance.get("source_content_changed") is not changed
            or derivation.get("runtime_contract_rebuild_required") is not True
            or provenance.get("runtime_contract_rebuild_required") is not True
            or provenance.get("runtime_contract_generated") is not False):
        raise ValueError("Derived source/runtime identity or contract policy mismatch")
    complete = version == VERSION
    return {**result, "derivation_verified": complete, "source_content_changed": changed,
            "runtime_source_identity_sha256": runtime_identity, "runtime_contract_generated": False,
            "offline_parent_transition_verified": complete,
            "derivation_verification_scope": "Full offline parent/overlay/inventory consistency; authenticity still requires external manifest pins."
                if complete else "LIMITED legacy v1: source inventory, self-hashes and record consistency only; original parent transition and overlay byte digest cannot be verified offline."}


def _verify_overlay_records(manifest: dict, provenance: dict):
    """Checks available in both versions, without reading historical sources."""
    parent_root = str(_absolute_name(provenance.get("parent_snapshot_root", "")))
    if parent_root != manifest["source_root"]:
        raise ValueError("Provenance parent root differs from manifest source_root")
    _absolute_name(provenance.get("overlay_path", ""))
    overlay = provenance.get("overlay")
    if (not isinstance(overlay, dict) or set(overlay) != {"schema", "parent_manifest_sha256", "entries"}
            or overlay["schema"] != "complete_benchmark_source_overlay_v1"
            or overlay["parent_manifest_sha256"] != provenance["parent_manifest_sha256"]
            or not isinstance(overlay["entries"], list) or not overlay["entries"]):
        raise ValueError("Embedded overlay identity or schema is inconsistent")
    _hash(overlay["parent_manifest_sha256"], "parent manifest")
    _hash(provenance.get("overlay_sha256"), "overlay")
    records = provenance.get("selected_overlays")
    if not isinstance(records, list) or len(records) != len(overlay["entries"]):
        raise ValueError("Embedded overlay and selected records disagree")
    seen = set()
    for entry, record in zip(overlay["entries"], records):
        if not isinstance(entry, dict) or set(entry) != {"target", "expected_parent_sha256", "source_path", "source_sha256"}:
            raise ValueError("Malformed embedded overlay entry")
        name = _relative(entry["target"])
        _allowed(name, set(manifest["explicit_extra_resources"]), freeze.snapshot_source_trees(manifest))
        if name in seen:
            raise ValueError("Duplicate embedded overlay target")
        seen.add(name)
        _absolute_name(entry["source_path"])
        new_sha = _hash(entry["source_sha256"], "candidate")
        old_sha = entry["expected_parent_sha256"]
        if old_sha is not None:
            _hash(old_sha, "expected parent")
        expected = {**entry, "operation": "add" if old_sha is None else "replace",
                    "content_changed": old_sha != new_sha}
        if not isinstance(record, dict) or record != expected or type(record.get("content_changed")) is not bool:
            raise ValueError("Selected overlay record differs from embedded overlay/operation")
        if manifest["files_sha256"].get(name) != new_sha:
            raise ValueError("Selected overlay SHA is not the derived target file SHA")


def _embedded_document(provenance, text_key, digest_key):
    value = provenance.get(text_key)
    if not isinstance(value, str):
        raise ValueError(f"Missing offline original document: {text_key}")
    raw = value.encode("utf-8")
    if _sha_bytes(raw) != _hash(provenance.get(digest_key), digest_key):
        raise ValueError(f"Offline document SHA mismatch: {text_key}")
    return _json(raw)


def _verify_offline_transition(manifest: dict, provenance: dict):
    """Reconstruct the exact selected source inventory from retained documents.

    These are consistency checks, not signatures. External READY/manifest pins
    remain required to authenticate which parent and overlay were authorized.
    Historical parent/candidate/overlay paths are never opened here.
    """
    parent = _embedded_document(provenance, "parent_manifest_utf8", "parent_manifest_sha256")
    overlay = _embedded_document(provenance, "overlay_utf8", "overlay_sha256")
    if overlay != provenance["overlay"]:
        raise ValueError("Embedded overlay differs from original hashed overlay bytes")
    if (not isinstance(parent, dict) or parent.get("schema") != "complete_benchmark_source_snapshot_v1"
            or parent.get("snapshot_root") != provenance["parent_snapshot_root"]):
        raise ValueError("Original parent manifest root/schema does not match provenance")
    source_trees = freeze.snapshot_source_trees(parent)
    if freeze.snapshot_source_trees(manifest) != source_trees:
        raise ValueError("Derived source layout differs from parent")
    files = parent.get("files_sha256")
    extras = parent.get("explicit_extra_resources")
    assets = parent.get("shared_asset_references", {})
    if not isinstance(files, dict) or not isinstance(extras, list) or not isinstance(assets, dict):
        raise ValueError("Invalid offline parent inventory")
    if len(set(extras)) != len(extras) or extras != manifest["explicit_extra_resources"]:
        raise ValueError("Derived explicit extras differ from parent")
    for name in extras:
        _relative(name)
        if name not in files:
            raise ValueError("Offline parent extra missing from its inventory")
    for name, digest in files.items():
        _allowed(name, set(extras), source_trees); _hash(digest, name)
    if parent.get("suite_source_pins") != freeze.suite_inventory(files,
            source_trees=freeze.snapshot_source_trees(parent),
            include_runtime_guard=parent.get("source_utility_version") in freeze.RUNTIME_GUARD_VERSIONS):
        raise ValueError("Offline parent suite inventory mismatch")
    expected_assets = {}
    asset_root = freeze.asset_reference_root(source_trees)
    for name, reference in assets.items():
        link = Path(_relative(name))
        if link == asset_root or not link.is_relative_to(asset_root) or link.suffix == ".py":
            raise ValueError("Invalid offline parent asset reference")
        target = str(_absolute_name(reference["resolved_target"]))
        if reference.get("original_link") != name or reference.get("snapshot_link_target") != target:
            raise ValueError("Offline parent asset reference identity mismatch")
        expected_assets[name] = {"original_link": name, "original_link_target": target,
                                "resolved_target": target, "snapshot_link_target": target}
    if manifest.get("shared_asset_references", {}) != expected_assets:
        raise ValueError("Derived asset references differ from parent")
    roots = parent.get("source_root_ancestors", [])
    if not isinstance(roots, list):
        raise ValueError("Invalid offline parent ancestry")
    ancestors = {str(_absolute_name(root)) for root in roots}
    ancestors.update([str(_absolute_name(parent["source_root"])), parent["snapshot_root"]])
    if manifest.get("source_root_ancestors") != sorted(ancestors):
        raise ValueError("Derived source ancestry differs from parent")
    expected_files = dict(files)
    for entry in overlay["entries"]:
        name = entry["target"]
        old = entry["expected_parent_sha256"]
        if old != files.get(name) or (name in files and old is None):
            raise ValueError("Overlay expected old SHA/add operation contradicts parent inventory")
        _allowed(name, set(extras), source_trees)
        if any(Path(name).is_relative_to(Path(link)) or Path(link).is_relative_to(Path(name)) for link in assets):
            raise ValueError("Overlay target intersects offline parent asset reference")
        expected_files[name] = entry["source_sha256"]
    if expected_files != manifest["files_sha256"]:
        raise ValueError("Derived inventory is not exactly parent plus explicit overlays")
    parent_identity = _canonical_hash({"files_sha256": files,
        "shared_asset_targets": {name: reference["resolved_target"] for name, reference in assets.items()}})
    if provenance.get("parent_content_identity_sha256") != parent_identity:
        raise ValueError("Offline parent content identity mismatch")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--overlay", required=True, help="Explicit source overlay v1 JSON")
    args = parser.parse_args()
    result = derive(args.parent, args.output, args.overlay)
    print(json.dumps({"snapshot_root": result["snapshot_root"], "files": len(result["files_sha256"]),
                      "shared_asset_references": len(result["shared_asset_references"]),
                      "derivation": result["derivation"], "native_validation_performed": False}))


if __name__ == "__main__":
    main()
