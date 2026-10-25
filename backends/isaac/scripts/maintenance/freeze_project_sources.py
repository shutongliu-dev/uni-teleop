#!/usr/bin/env python3
"""Copy a complete benchmark source inventory before preparing an experiment.

Run the experiment's launcher, frontend and native worker from the returned root.
Freezing only the imported frontend subset is insufficient: suite.py also checks
every Python source under bench/, agent/ and scenes/.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import sys


SOURCE_TREES = ("bench", "agent", "scenes")
V5_SOURCE_TREES = ("interface", "bench", "simulation", "agent", "scenes")
LEGACY_SOURCE_TREES = ("interface", "bench", "simulation")
ALL_SOURCE_TREES = V5_SOURCE_TREES
RUNTIME_TREES = (*ALL_SOURCE_TREES, "scripts", "uni_teleop")
VERIFIER_VERSION = "complete_benchmark_source_verifier_v6"
V5_VERIFIER_VERSION = "complete_benchmark_source_verifier_v5"
MODE_CONTRACT_VERSIONS = {"complete_benchmark_source_verifier_v3", "complete_benchmark_source_verifier_v4",
                          V5_VERIFIER_VERSION, VERIFIER_VERSION}
RUNTIME_GUARD_VERSIONS = {"complete_benchmark_source_verifier_v4", V5_VERIFIER_VERSION, VERIFIER_VERSION}
RUNTIME_GUARD_SOURCE = "scripts/maintenance/freeze_project_sources.py"
MODE_POLICY = "exact_or_write_bits_removed_v1"
DEFAULT_EXTRAS = (
    "scripts/isaac_python.sh",
    "bench/apps/fr3_sharpa.headless.kit",
    "scripts/assets/manipulation_scene_layouts.json",
    "scenes/catalog.json",
    "scenes/workrooms/current_release.json",
    "pyproject.toml",
)


# Registered asset links resolve into the data package; None means its assets/ root.
SHARED_ASSET_ROOT = None


def _data_assets():
    from bench.paths import data_root  # Imported only when a snapshot registers asset links.
    return data_root() / 'assets'

def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def snapshot_source_trees(manifest: dict) -> tuple[str, ...]:
    """Read a versioned layout; old snapshots retain their original source roots."""
    version = manifest.get("source_utility_version")
    if version in (VERIFIER_VERSION, V5_VERIFIER_VERSION):
        trees = SOURCE_TREES if version == VERIFIER_VERSION else V5_SOURCE_TREES
        if manifest.get("source_trees") != list(trees):
            raise ValueError("Snapshot source_trees does not match the versioned layout")
        return trees
    if version not in {
        None,
        "complete_benchmark_source_verifier_v1", "complete_benchmark_source_verifier_v2",
        "complete_benchmark_source_verifier_v3", "complete_benchmark_source_verifier_v4",
    }:
        raise ValueError("Unsupported source verifier version")
    return LEGACY_SOURCE_TREES


def asset_reference_root(source_trees: tuple[str, ...]) -> Path:
    """The shared-asset whitelist follows the declared source layout exactly."""
    if source_trees == SOURCE_TREES:
        return Path("bench/simulation/assets")
    if source_trees in (V5_SOURCE_TREES, LEGACY_SOURCE_TREES):
        return Path("simulation/assets")
    raise ValueError("Unsupported source layout for shared asset references")


def suite_inventory(files: dict[str, str], *, include_runtime_guard: bool = True,
                    source_trees: tuple[str, ...] = SOURCE_TREES) -> dict[str, str]:
    # v6 follows suite.source_pins: Python plus its explicit launch dependencies.
    # Preserve older manifests' inclusion of resources under their source roots.
    return {name: value for name, value in files.items()
            if (name.split("/", 1)[0] in source_trees
                and (source_trees != SOURCE_TREES or name.endswith(".py")))
            or name == "scripts/isaac_python.sh"
            or (include_runtime_guard and name == RUNTIME_GUARD_SOURCE)}


def file_modes(root: Path, files) -> dict[str, str]:
    """Record permissions separately from byte identities, including executables."""
    return {name: format(stat.S_IMODE(regular_file(root, name).stat().st_mode), "04o")
            for name in sorted(files)}


def verify_file_modes(root: Path, files, manifest: dict) -> bool:
    modes = manifest.get("files_mode_octal")
    if modes is None:
        if manifest.get("source_utility_version") in MODE_CONTRACT_VERSIONS:
            raise ValueError("Missing file mode contract")
        return False  # Historical byte-only manifests make no permission claim.
    if (not isinstance(modes, dict) or set(modes) != set(files)
            or manifest.get("file_mode_policy") != MODE_POLICY):
        raise ValueError("File mode contract does not match source inventory")
    actual = file_modes(root, files)
    for name, value in modes.items():
        if (not isinstance(value, str) or len(value) != 4
                or any(c not in "01234567" for c in value)):
            raise ValueError(f"Invalid file mode contract: {name}")
        expected = int(value, 8)
        if int(actual[name], 8) not in (expected, expected & ~0o222):
            raise ValueError(f"Source file mode changed: {name}")
    return True


def _read_manifest(path: Path) -> dict:
    def unique_pairs(pairs):
        value = {}
        for key, item in pairs:
            if key in value:
                raise ValueError(f"Duplicate manifest key: {key}")
            value[key] = item
        return value
    value = json.loads(path.read_text(), object_pairs_hook=unique_pairs)
    if not isinstance(value, dict):
        raise ValueError("Expected snapshot manifest object")
    return value


def regular_file(root: Path, relative: str) -> Path:
    name = Path(relative)
    if (not isinstance(relative, str) or name.is_absolute() or ".." in name.parts
            or not name.parts or str(name) != relative or "\\" in relative):
        raise ValueError(f"Expected a relative source path: {relative}")
    path = root / name
    # Do not hide code behind symlinks that the suite's rglob does not traverse.
    for parent in (path, *path.parents):
        if parent == root:
            break
        if parent.is_symlink():
            raise ValueError(f"Symlink in source inventory: {relative}")
    if not stat.S_ISREG(path.stat().st_mode):
        raise ValueError(f"Expected a regular file: {relative}")
    return path


def asset_references(root: Path, asset_links: tuple[str, ...], *,
                     source_trees: tuple[str, ...] = SOURCE_TREES) -> dict[str, dict]:
    """Inspect explicitly registered shared asset references without copying them."""
    if len(set(asset_links)) != len(asset_links):
        raise ValueError("Duplicate shared asset link registration")
    asset_root = asset_reference_root(source_trees)
    references = {}
    for relative in asset_links:
        name = Path(relative)
        if (str(name) != relative or "\\" in relative or name.is_absolute() or ".." in name.parts
                or name == asset_root or not name.is_relative_to(asset_root) or name.suffix == ".py"):
            raise ValueError(f"Asset link must be below {asset_root}/: {relative}")
        path = root / name
        if not path.is_symlink():
            raise ValueError(f"Registered asset is not a symlink: {relative}")
        for parent in path.parents:
            if parent == root:
                break
            if parent.is_symlink():
                raise ValueError(f"Asset link has an aliased parent: {relative}")
        try:
            target = path.resolve(strict=True)
        except (FileNotFoundError, RuntimeError) as error:
            raise ValueError(f"Asset link target does not exist: {relative}") from error
        allowed = (SHARED_ASSET_ROOT or _data_assets()).resolve(strict=True)
        if target == allowed or not target.is_relative_to(allowed) or target.suffix == ".py":
            raise ValueError(f"Asset target is not a shared asset resource: {relative}")
        if not (target.is_dir() or target.is_file()):
            raise ValueError(f"Asset target is not a regular resource: {relative}")
        references[str(name)] = {"link_target": os.readlink(path),
                                 "resolved_target": str(target)}
    return references


def source_tree_files(root: Path, tree: str, *, required: bool = True,
                      asset_links: set[str] = frozenset()) -> set[str]:
    directory = root / tree
    if not required and not directory.exists() and not directory.is_symlink():
        return set()
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError(f"Missing regular source directory (or symlink): {tree}")
    paths = set()
    for path in directory.rglob("*"):
        if path.is_symlink():
            if str(path.relative_to(root)) in asset_links:
                continue
            raise ValueError(f"Symlink in source tree: {path.relative_to(root)}")
        if path.suffix == ".py":
            paths.add(str(path.relative_to(root)))
    return paths


def inventory(root: Path, extras: tuple[str, ...], asset_links: tuple[str, ...] = (), *,
              source_trees: tuple[str, ...] = SOURCE_TREES) -> dict[str, str]:
    if any(not isinstance(name, str) for name in extras) or len(set(extras)) != len(extras):
        raise ValueError("Expected distinct explicit extra resource names")
    registered = set(asset_references(root, asset_links, source_trees=source_trees))
    paths: set[str] = set(extras)
    for tree in source_trees:
        paths.update(source_tree_files(root, tree, asset_links=registered))
    for tree in set(ALL_SOURCE_TREES) - set(source_trees):
        if source_tree_files(root, tree, required=False):
            raise ValueError(f"Source inventory changed outside versioned layout: {tree}")
    # Native/controller launch helpers are code, not runtime assets. Configs and
    # credentials are never recursively copied; extra resources must be named.
    paths.update(source_tree_files(root, "scripts", required=False))
    paths.update(source_tree_files(root, "uni_teleop", required=False))
    for name in extras:
        # Do not infer whether a configuration is safe from its basename: e.g.
        # configs/local.json may hold credentials. Reject these paths before IO.
        if any(part.lower() in {"config", "configs", "configuration", "configurations"}
               for part in Path(name).parts):
            raise ValueError(f"Configuration extras are not source resources: {name}")
    for name in paths:
        if any(word in part.lower() for part in Path(name).parts
               for word in ("credential", "secret", ".env")):
            raise ValueError(f"Credential-like extra is not a source resource: {name}")
    return {name: digest(regular_file(root, name)) for name in sorted(paths)}


def freeze(source: Path, destination: Path, extras=DEFAULT_EXTRAS, *, asset_links=()) -> dict:
    # Resolve existing parent aliases and '..' before checking containment. A
    # lexical absolute path can otherwise copy into the editable tree via alias.
    source, destination = source.resolve(strict=True), destination.resolve(strict=False)
    if destination.is_relative_to(source) or source.is_relative_to(destination):
        raise ValueError("Snapshot must live outside the editable source tree")
    extras = tuple(extras)
    asset_links = tuple(asset_links)
    before_assets = asset_references(source, asset_links)
    before = inventory(source, extras, asset_links)
    before_modes = file_modes(source, before)
    _verify_native_entrypoint(source, before)
    destination.mkdir(parents=True, exist_ok=False)
    for tree in SOURCE_TREES:
        (destination / tree).mkdir()
    # On interruption leave the partial directory as evidence. No valid manifest
    # is written until both inventories and every copied byte agree.
    for relative in before:
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(regular_file(source, relative), target)
    # Resolve relative source link targets to the same absolute shared asset in
    # the relocated snapshot; never duplicate reusable robot meshes or datasets.
    for relative, reference in before_assets.items():
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.symlink_to(reference["resolved_target"],
                          target_is_directory=Path(reference["resolved_target"]).is_dir())
    after = inventory(source, extras, asset_links)
    copied = inventory(destination, extras, asset_links)
    expected_assets = {name: {"link_target": ref["resolved_target"],
                              "resolved_target": ref["resolved_target"]}
                       for name, ref in before_assets.items()}
    if (before != after or before != copied
            or file_modes(source, before) != before_modes
            or file_modes(destination, copied) != before_modes
            or asset_references(source, asset_links) != before_assets
            or asset_references(destination, asset_links) != expected_assets):
        raise ValueError("Source changed during snapshot; directory is incomplete")
    result = {
        "schema": "complete_benchmark_source_snapshot_v1",
        "source_utility_version": VERIFIER_VERSION,
        "source_trees": list(SOURCE_TREES),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_root": str(source),
        "snapshot_root": str(destination),
        "files_sha256": copied,
        "files_mode_octal": before_modes,
        "file_mode_policy": MODE_POLICY,
        "suite_source_pins": suite_inventory(copied),
        "explicit_extra_resources": list(extras),
        "shared_asset_references": {
            name: {"original_link": name, "original_link_target": ref["link_target"],
                   "resolved_target": ref["resolved_target"],
                   "snapshot_link_target": ref["resolved_target"]}
            for name, ref in before_assets.items()},
        "shared_asset_scope": "Registered references only; shared asset contents are neither copied nor content-frozen by this source utility. Runtime asset hashes remain separately required.",
        "credential_files_selected": False,
        "embedded_secret_scan_performed": False,
        "credential_handling": "Configuration extras and credential-like paths are rejected without reading them; selected source files must not contain embedded credentials.",
        "native_validation_performed": False,
        "usage": "Prepare and execute all processes from snapshot_root; keep credentials external. Verify before and after execution.",
    }
    (destination / "source_snapshot_manifest.json").write_text(
        json.dumps(result, indent=2) + "\n")
    return result


def _verify_native_entrypoint(root: Path, files):
    name = "scripts/isaac_python.sh"
    if name in files:
        path = regular_file(root, name)
        if not path.stat().st_mode & 0o111 or not os.access(path, os.X_OK):
            raise ValueError("Native entrypoint scripts/isaac_python.sh must remain executable")


def verify(destination: Path) -> dict:
    requested = Path(destination).absolute()
    destination = requested.resolve(strict=True)
    if requested != destination:
        raise ValueError("Snapshot root must not use aliases")
    manifest = _read_manifest(regular_file(destination, "source_snapshot_manifest.json"))
    if manifest.get("schema") != "complete_benchmark_source_snapshot_v1":
        raise ValueError("Unsupported snapshot manifest")
    if manifest.get("snapshot_root") != str(destination):
        raise ValueError("Manifest snapshot_root does not match the verified directory")
    source_root = manifest.get("source_root")
    if not isinstance(source_root, str) or not Path(source_root).is_absolute():
        raise ValueError("Invalid manifest source_root")
    source = Path(source_root)
    if (str(source) != source_root or ".." in source.parts
            or source.resolve(strict=False) != source
            or destination.is_relative_to(source) or source.is_relative_to(destination)):
        raise ValueError("Manifest source and snapshot roots must be disjoint without aliases")
    extras = manifest.get("explicit_extra_resources")
    files = manifest.get("files_sha256")
    if (not isinstance(extras, list) or any(not isinstance(name, str) for name in extras)
            or len(set(extras)) != len(extras) or not isinstance(files, dict)):
        raise ValueError("Malformed complete source inventory")
    assets = manifest.get("shared_asset_references", {})
    if not isinstance(assets, dict):
        raise ValueError("Invalid shared asset reference manifest")
    trees = snapshot_source_trees(manifest)
    actual_assets = asset_references(destination, tuple(assets), source_trees=trees)
    for name, reference in assets.items():
        if (not isinstance(reference, dict) or reference.get("original_link") != name
                or actual_assets[name]["link_target"] != reference.get("snapshot_link_target")
                or actual_assets[name]["resolved_target"] != reference.get("resolved_target")):
            raise ValueError(f"Shared asset reference changed: {name}")
    current = inventory(destination, tuple(extras), tuple(assets), source_trees=trees)
    if current != files:
        raise ValueError("Snapshot source inventory changed")
    if manifest.get("suite_source_pins") != suite_inventory(current,
            source_trees=trees,
            include_runtime_guard=manifest.get("source_utility_version") in RUNTIME_GUARD_VERSIONS):
        raise ValueError("Manifest suite_source_pins does not match the complete source inventory")
    modes_verified = verify_file_modes(destination, current, manifest)
    _verify_native_entrypoint(destination, current)
    return {"verified": True, "snapshot_root": str(destination), "files": len(current),
            "shared_asset_references": len(assets), "verifier_version": VERIFIER_VERSION,
            "file_modes_verified": modes_verified, "runtime_imports_verified": False,
            "native_validation_performed": False}


def verify_runtime_imports(destination: Path, *, required_modules: tuple[str, ...],
                           entrypoint_file: Path | None = None) -> dict:
    """Verify loaded sources and package paths in this process, without imports.

    Call after importing the host/worker's required modules. Evidence applies at
    this call only, not to other processes, future imports or native execution.
    """
    result = verify(destination)
    root = Path(result["snapshot_root"])
    manifest = _read_manifest(regular_file(root, "source_snapshot_manifest.json"))
    runtime_trees = (*snapshot_source_trees(manifest), "scripts", "uni_teleop")
    if Path.cwd().resolve() != root:
        raise ValueError("Running process cwd does not match snapshot_root")
    if (not required_modules or isinstance(required_modules, str)
            or any(not isinstance(name, str) or name.split(".", 1)[0] not in runtime_trees
                   or any(not part.isidentifier() for part in name.split("."))
                   for name in required_modules)):
        raise ValueError("Declare nonempty required project module names")
    # `python -m package.module` executes that module under __main__, without
    # necessarily creating sys.modules['package.module']. Verify its real file
    # using the canonical name in __spec__; do not import a second copy as proof.
    main_module = sys.modules.get("__main__")
    main_name = getattr(getattr(main_module, "__spec__", None), "name", None)
    project_main = (isinstance(main_name, str)
                    and main_name.split(".", 1)[0] in RUNTIME_TREES)
    missing = [name for name in required_modules
               if name not in sys.modules and not (project_main and name == main_name)]
    if missing:
        raise ValueError(f"Required modules are not loaded in this process: {missing}")
    files = manifest["files_sha256"]
    imported, packages = {}, {}
    loaded = list(sys.modules.items())
    if project_main:
        loaded.append((main_name, main_module))
    for name, module in loaded:
        if name.split(".", 1)[0] not in RUNTIME_TREES:
            continue
        if module is None:
            raise ValueError(f"Project module is not initialized: {name}")
        file = getattr(module, "__file__", None)
        if file is not None:
            path = Path(file).absolute()
            if path.resolve(strict=True) != path or not path.is_relative_to(root):
                raise ValueError(f"Loaded project module is outside snapshot_root or aliased: {name}: {file}")
            relative = str(path.relative_to(root))
            if relative not in files or digest(regular_file(root, relative)) != files[relative]:
                raise ValueError(f"Loaded project module is not a verified source: {name}: {file}")
            imported[name] = {"path": str(path), "sha256": files[relative]}
        module_paths = getattr(module, "__path__", None)
        if module_paths is not None:
            paths = [Path(path).absolute() for path in module_paths]
            expected = root.joinpath(*name.split("."))
            if not paths or any(path != expected or path.resolve(strict=True) != path for path in paths):
                raise ValueError(f"Loaded project package search path is outside its snapshot package: {name}")
            packages[name] = [str(path) for path in paths]
        if file is None and module_paths is None:
            raise ValueError(f"Project module has no verifiable origin: {name}")
    entrypoint = None
    if entrypoint_file is not None:
        path = Path(entrypoint_file).absolute()
        if path.resolve(strict=True) != path or not path.is_relative_to(root):
            raise ValueError("Running entrypoint is outside snapshot_root or aliased")
        relative = str(path.relative_to(root))
        if relative not in files or digest(regular_file(root, relative)) != files[relative]:
            raise ValueError("Running entrypoint is not a verified source")
        entrypoint = {"path": str(path), "sha256": files[relative]}
    return {**result, "runtime_imports_verified": True, "pid": os.getpid(),
            "cwd": str(Path.cwd()), "required_modules": list(required_modules),
            "imported_project_modules": imported, "project_package_paths": packages,
            "entrypoint": entrypoint,
            "runtime_scope": "This process at this call; does not verify another process or future imports."}


def verify_if_snapshot(project: Path, *, required_modules: tuple[str, ...],
                       entrypoint_file: Path) -> dict | None:
    """Enforce import isolation for frozen launches, leave editable runs explicit.

    Inspect the entrypoint's source root and the actual cwd. Otherwise starting
    in a snapshot while accidentally importing editable code could skip the
    guard. Even a dangling manifest symlink counts as a marker and must fail
    verification, never fall back to the editable path.
    """
    project = Path(project).absolute()
    marker = project / "source_snapshot_manifest.json"
    cwd_marker = Path.cwd() / "source_snapshot_manifest.json"
    if not (marker.exists() or marker.is_symlink()):
        if cwd_marker.exists() or cwd_marker.is_symlink():
            raise ValueError("Frozen working directory loaded an entrypoint outside its snapshot")
        return None
    return verify_runtime_imports(project, required_modules=required_modules,
                                  entrypoint_file=entrypoint_file)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    create = commands.add_parser("create")
    create.add_argument("--source", type=Path, required=True)
    create.add_argument("--output", type=Path, required=True)
    create.add_argument("--extra", action="append", default=[])
    create.add_argument("--asset-link", action="append", default=[],
                        help="Explicit bench/simulation/assets/ symlink to preserve as a shared asset reference")
    check = commands.add_parser("verify")
    check.add_argument("snapshot", type=Path)
    args = parser.parse_args()
    if args.command == "create":
        extras = tuple(dict.fromkeys((*DEFAULT_EXTRAS, *args.extra)))
        value = freeze(args.source, args.output, extras,
                       asset_links=tuple(args.asset_link))
        print(json.dumps({"snapshot_root": value["snapshot_root"],
                          "files": len(value["files_sha256"]),
                          "shared_asset_references": len(value["shared_asset_references"]),
                          "native_validation_performed": False}))
    else:
        print(json.dumps(verify(args.snapshot)))


if __name__ == "__main__":
    main()
