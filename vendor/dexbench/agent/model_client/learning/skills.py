"""Content-addressed, initially empty learning libraries owned by a trusted host.

These file tools validate controlled requests and preserve provenance. They do not
isolate generated Python, prove bench success, or authorize arbitrary host access.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import os
from pathlib import Path, PurePosixPath
import re
import stat
import sys
from typing import Any
import uuid

from .journal import (
    Journal, JournalError, atomic_write_bytes, atomic_write_json, canonical_json,
    checked_directory, fsync_directory, read_bytes, read_json, rename_noreplace,
    strict_json_loads,
)
from bench.paths import OUTPUT_ROOT

DEFAULT_LEARNING_ROOT = Path(f"{OUTPUT_ROOT}/skill_learning")
PROTOCOL = "agent-skill-bundle-v1"
AUTHOR_DIRS = ("skills", "modules", "parameters", "notes", "tests")
MAX_FILES = 512
MAX_FILE_BYTES = 4 * 1024 * 1024
MAX_BUNDLE_BYTES = 16 * 1024 * 1024


class SkillLibraryError(JournalError):
    """Invalid library state or request."""


def _identifier(value: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", value):
        raise SkillLibraryError("invalid identifier")
    return value


def _payload_path(value: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        raise SkillLibraryError("invalid payload path")
    parts = value.split("/")
    if any(part in ("", ".", "..") or not re.fullmatch(r"[A-Za-z0-9_.-]+", part) for part in parts):
        raise SkillLibraryError("invalid payload path")
    if value != "catalog.json" and (len(parts) < 2 or parts[0] not in AUTHOR_DIRS):
        raise SkillLibraryError("path is outside the author content directories")
    if PurePosixPath(value).suffix not in (".py", ".json", ".md", ".txt"):
        raise SkillLibraryError("only Python, JSON and UTF-8 text assets are supported")
    return value


def _hash(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _bundle_hash(manifest: dict[str, Any]) -> str:
    return _hash(PROTOCOL.encode() + b"\0" + canonical_json(manifest))


def _digest(value: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise SkillLibraryError("invalid bundle hash")
    return value


def _dependencies(value: Any) -> dict[str, str]:
    if not isinstance(value, dict):
        raise SkillLibraryError("dependencies must name exact versions")
    for name, version in value.items():
        if (not isinstance(name, str) or not name.isidentifier() or not isinstance(version, str)
                or not version or re.search(r"[<>=!*~\s]", version)):
            raise SkillLibraryError("dependencies must name exact import roots and versions")
    return value


def _continuous_evidence(evidence: dict[str, Any]) -> None:
    """Validate host-classified scores without inventing a binary success gate."""
    from pydantic import TypeAdapter, ValidationError
    from bench.protocol.task_scoring import Number, Unit

    fields = {"task_score", "task_score_unit", "task_score_valid", "result_valid_for_scoring"}
    if not fields.issubset(evidence):
        raise SkillLibraryError("continuous validation evidence fields missing")
    if (evidence["task_outcome"] != "unknown" or evidence.get("task_success") is not None
            or type(evidence["task_score_valid"]) is not bool
            or type(evidence["result_valid_for_scoring"]) is not bool
            or evidence["task_score_valid"] != evidence["result_valid_for_scoring"]):
        raise SkillLibraryError("invalid continuous validation evidence")
    try:
        TypeAdapter(Number | None).validate_python(evidence["task_score"])
        TypeAdapter(Unit | None).validate_python(evidence["task_score_unit"])
    except ValidationError as exc:
        raise SkillLibraryError("invalid continuous score or unit") from exc
    if evidence["task_score"] is not None and evidence["task_score_unit"] is None:
        raise SkillLibraryError("continuous score requires its declared unit")
    if evidence["task_score_valid"] and (
            evidence["task_score"] is None or evidence["task_score_unit"] is None
            or evidence["execution_status"] not in ("completed", "agent_budget")):
        raise SkillLibraryError("continuous score lacks valid execution evidence")
    if evidence["valid"] and not evidence["result_valid_for_scoring"]:
        raise SkillLibraryError("invalid continuous score cannot qualify validation")


def _file_map(root: Path) -> dict[str, bytes]:
    checked_directory(root)
    result: dict[str, bytes] = {}
    total = 0
    def walk_error(error: OSError) -> None:
        raise error
    for directory, dirs, files in os.walk(root, followlinks=False, onerror=walk_error):
        for name in dirs:
            if not stat.S_ISDIR((Path(directory) / name).lstat().st_mode):
                raise SkillLibraryError("linked or special content directory")
            rel = (Path(directory) / name).relative_to(root).as_posix()
            parts = rel.split("/")
            if parts[0] not in AUTHOR_DIRS or any(
                part in (".", "..") or not re.fullmatch(r"[A-Za-z0-9_.-]+", part)
                for part in parts
            ):
                raise SkillLibraryError("unexpected content directory")
        for name in files:
            path = Path(directory) / name
            rel = _payload_path(path.relative_to(root).as_posix())
            data = read_bytes(path, limit=MAX_FILE_BYTES)
            total += len(data)
            if total > MAX_BUNDLE_BYTES or len(result) >= MAX_FILES:
                raise SkillLibraryError("bundle quota exceeded")
            result[rel] = data
    if "catalog.json" not in result:
        raise SkillLibraryError("catalog.json is required")
    return result


def _inventory(files: dict[str, bytes]) -> list[dict[str, Any]]:
    return [{"path": path, "size": len(data), "sha256": _hash(data)}
            for path, data in sorted(files.items())]


def _revision(files: dict[str, bytes]) -> str:
    return _hash(b"agent-author-v1\0" + canonical_json(_inventory(files)))


def _validate_payload(files: dict[str, bytes], dependencies: dict[str, str]) -> dict[str, Any]:
    trees: dict[str, ast.Module] = {}
    for path, data in files.items():
        try:
            source = data.decode("utf-8")
        except UnicodeError as exc:
            raise SkillLibraryError("payload must be UTF-8") from exc
        if path.endswith(".py"):
            try:
                trees[path] = ast.parse(source, filename=path)
            except SyntaxError as exc:
                raise SkillLibraryError(f"invalid Python syntax in {path}:{exc.lineno}") from exc
        elif path.endswith(".json"):
            strict_json_loads(data)
    catalog = strict_json_loads(files["catalog.json"])
    if not isinstance(catalog, dict) or set(catalog) != {"skills"} or not isinstance(catalog["skills"], list):
        raise SkillLibraryError("catalog must contain a skills list")
    identifiers = set()
    for skill in catalog["skills"]:
        if not isinstance(skill, dict) or set(skill) != {
            "skill_id", "module", "entrypoint", "description", "tags", "input_schema", "output_schema"
        }:
            raise SkillLibraryError("invalid skill catalog entry")
        identifier = _identifier(skill["skill_id"])
        if identifier in identifiers:
            raise SkillLibraryError("duplicate skill ID")
        identifiers.add(identifier)
        module = _payload_path(skill["module"])
        if module not in trees or not isinstance(skill["entrypoint"], str):
            raise SkillLibraryError("skill entrypoint module is missing")
        definitions = [node for node in trees[module].body if isinstance(node, ast.FunctionDef)
                       and node.name == skill["entrypoint"]]
        if len(definitions) != 1:
            raise SkillLibraryError("entrypoint must be one top-level synchronous function")
        args = definitions[0].args
        if ([arg.arg for arg in args.posonlyargs + args.args] != ["api", "args", "state"]
                or args.vararg or args.kwarg or args.kwonlyargs or args.defaults):
            raise SkillLibraryError("entrypoint must have signature run(api, args, state)")
        if (not isinstance(skill["description"], str) or not isinstance(skill["tags"], list)
                or any(not isinstance(tag, str) for tag in skill["tags"])
                or not isinstance(skill["input_schema"], dict)
                or not isinstance(skill["output_schema"], dict)):
            raise SkillLibraryError("invalid skill metadata")
    local_modules = {path[:-3].replace("/", ".") for path in trees}
    local_packages = {".".join(module.split(".")[:i]) for module in local_modules
                      for i in range(1, len(module.split(".")))}
    local_packages |= {module[:-9] for module in local_modules if module.endswith(".__init__")}
    allowed = set(sys.stdlib_module_names) | set(dependencies)

    def module_exists(module: str) -> bool:
        return module in local_modules or module in local_packages

    def check_import(module: str) -> None:
        root = module.split(".")[0]
        if root in AUTHOR_DIRS:
            if not module_exists(module):
                raise SkillLibraryError(f"missing local dependency: {module}")
        elif root not in allowed:
            raise SkillLibraryError(f"undeclared dependency: {root}")

    for path, tree in trees.items():
        package = path[:-3].replace("/", ".").split(".")[:-1]
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    check_import(alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.level:
                    if node.level > len(package):
                        raise SkillLibraryError("relative import escapes bundle")
                    base = package[:len(package) - node.level + 1]
                    module = ".".join(base + ([node.module] if node.module else []))
                else:
                    module = node.module or ""
                check_import(module)
                if node.module is None or module in local_packages:
                    for alias in node.names:
                        if alias.name == "*":
                            continue
                        submodule = module + "." + alias.name
                        init_file = module.replace(".", "/") + "/__init__.py"
                        if module.split(".")[0] in AUTHOR_DIRS and not module_exists(submodule) and init_file not in trees:
                            raise SkillLibraryError(f"missing local dependency: {submodule}")
    # This is a static source check, not a sandbox: runtime dynamic import and
    # resource access must be controlled by the separately implemented executor.
    return catalog


class SkillLibrary:
    def __init__(self, path: Path | str, *, expected_contract: dict[str, Any] | None = None):
        self.path = checked_directory(path)
        self.host = checked_directory(self.path / "host")
        self.author = checked_directory(self.path / "author")
        self.bundles = checked_directory(self.path / "bundles")
        self.staging = checked_directory(self.host / "staging")
        self.experiment = read_json(self.host / "experiment.json")
        if (not isinstance(self.experiment, dict) or self.experiment.get("schema") != "agent-learning-v1"
                or not isinstance(self.experiment.get("contract"), dict)):
            raise SkillLibraryError("invalid experiment contract")
        self.contract = self.experiment["contract"]
        _dependencies(self.contract.get("dependencies", {}))
        if expected_contract is not None and canonical_json(expected_contract) != canonical_json(self.contract):
            raise SkillLibraryError("experiment contract mismatch")
        self.journal = Journal(self.host / "journal")
        events = self.journal.read()
        if not events or events[0]["kind"] != "experiment_created" or events[0]["payload"] != {
            "experiment_sha256": _hash(canonical_json(self.experiment))
        }:
            raise SkillLibraryError("experiment provenance mismatch")

    @classmethod
    def create(cls, experiment_id: str, *, learning_root: Path | str = DEFAULT_LEARNING_ROOT,
               contract: dict[str, Any]) -> SkillLibrary:
        _identifier(experiment_id)
        if not isinstance(contract, dict):
            raise SkillLibraryError("contract must be an object")
        canonical_json(contract)
        _dependencies(contract.get("dependencies", {}))
        if not isinstance(contract.get("compatibility", {}), dict):
            raise SkillLibraryError("compatibility must be an object")
        root = checked_directory(learning_root, create=True)
        target = root / experiment_id
        if target.exists() or target.is_symlink():
            raise FileExistsError(target)
        temporary = root / (".initializing-" + uuid.uuid4().hex)
        temporary.mkdir(mode=0o700)
        for name in ("author", "bundles", "episodes", "host", "audit"):
            (temporary / name).mkdir(mode=0o700)
        for name in AUTHOR_DIRS:
            (temporary / "author" / name).mkdir(mode=0o700)
        for name in ("journal", "slots", "attempts", "evidence", "staging"):
            (temporary / "host" / name).mkdir(mode=0o700)
        atomic_write_json(temporary / "author" / "catalog.json", {"skills": []}, replace=False)
        experiment = {"schema": "agent-learning-v1", "experiment_id": experiment_id, "contract": contract}
        atomic_write_json(temporary / "host" / "experiment.json", experiment, replace=False)
        Journal(temporary / "host" / "journal").append("experiment_created", {
            "experiment_sha256": _hash(canonical_json(experiment))
        }, event_id="experiment-created")
        for directory, _, _ in os.walk(temporary, topdown=False):
            fsync_directory(directory)
        rename_noreplace(temporary, target)
        fsync_directory(root)
        return cls(target, expected_contract=contract)

    def author_revision(self) -> str:
        with self.journal.transaction():
            return _revision(_file_map(self.author))

    def _require_practice(self, phase: str) -> None:
        if phase != "practice":
            raise SkillLibraryError("long-term writes require the practice phase")

    def write_author(self, path: str, content: str | bytes, *, expected_revision: str,
                     phase: str = "practice") -> str:
        self._require_practice(phase)
        path = _payload_path(path)
        data = content.encode("utf-8") if isinstance(content, str) else content
        if not isinstance(data, bytes) or len(data) > MAX_FILE_BYTES:
            raise SkillLibraryError("invalid author content or file quota")
        try:
            data.decode("utf-8")
        except UnicodeError as exc:
            raise SkillLibraryError("author content must be UTF-8") from exc
        with self.journal.transaction():
            files = _file_map(self.author)
            if _revision(files) != expected_revision:
                raise SkillLibraryError("stale author revision")
            files[path] = data
            if len(files) > MAX_FILES or sum(map(len, files.values())) > MAX_BUNDLE_BYTES:
                raise SkillLibraryError("author quota exceeded")
            destination = self.author / path
            checked_directory(destination.parent, create=True)
            atomic_write_bytes(destination, data)
            revision = _revision(files)
            self.journal.append("author_changed", {"previous_revision": expected_revision,
                "revision": revision, "path": path, "sha256": _hash(data), "operation": "write"})
            return revision

    def delete_author(self, path: str, *, expected_revision: str, phase: str = "practice") -> str:
        self._require_practice(phase)
        path = _payload_path(path)
        if path == "catalog.json":
            raise SkillLibraryError("catalog cannot be deleted")
        with self.journal.transaction():
            files = _file_map(self.author)
            if _revision(files) != expected_revision:
                raise SkillLibraryError("stale author revision")
            if path not in files:
                raise FileNotFoundError(path)
            (self.author / path).unlink()
            fsync_directory((self.author / path).parent)
            del files[path]
            revision = _revision(files)
            self.journal.append("author_changed", {"previous_revision": expected_revision,
                "revision": revision, "path": path, "operation": "delete"})
            return revision

    def read_author(self, path: str) -> bytes:
        with self.journal.transaction():
            return read_bytes(self.author / _payload_path(path), limit=MAX_FILE_BYTES)

    def _candidate(self, event_id: str) -> dict[str, Any]:
        events = {event["event_id"]: event for event in self.journal.read()}
        visited: set[str] = set()
        active: set[str] = set()

        def check(identifier: str) -> dict[str, Any]:
            event = events.get(identifier)
            if event is None or event["kind"] != "candidate_committed":
                raise SkillLibraryError("candidate provenance is missing")
            if identifier in active:
                raise SkillLibraryError("cyclic candidate provenance")
            if identifier not in visited:
                active.add(identifier)
                self._verify_origin(event["payload"])
                for parent in event["payload"].get("parents", []):
                    check(parent)
                if any(source not in events for source in event["payload"].get("source_events", [])):
                    raise SkillLibraryError("candidate source evidence is missing")
                active.remove(identifier)
                visited.add(identifier)
            return event
        return check(event_id)

    def _verify_origin(self, payload: dict[str, Any]) -> None:
        imported = payload.get("import")
        if imported is None:
            return
        digest = _digest(imported["origin_sha256"])
        origin = read_json(self.host / "origins" / (digest + ".json"))

        def validate(value: dict[str, Any], expected: str, depth: int = 0) -> None:
            if depth > 32 or _hash(canonical_json(value)) != expected:
                raise SkillLibraryError("import origin hash mismatch or excessive ancestry depth")
            if not isinstance(value, dict) or not isinstance(value.get("journal"), list):
                raise SkillLibraryError("invalid import provenance")
            previous = None
            ids = set()
            for sequence, event in enumerate(value["journal"], 1):
                if not isinstance(event, dict) or set(event) != {
                    "sequence", "event_id", "kind", "transaction_id", "payload", "previous_hash", "event_hash"
                }:
                    raise SkillLibraryError("invalid imported journal event")
                body = {key: entry for key, entry in event.items() if key != "event_hash"}
                if (event["sequence"] != sequence or event["previous_hash"] != previous
                        or _hash(b"agent-journal-v1\0" + canonical_json(body)) != event["event_hash"]
                        or event["event_id"] in ids):
                    raise SkillLibraryError("imported journal chain mismatch")
                ids.add(event["event_id"])
                previous = event["event_hash"]
                if event["kind"] == "candidate_committed" and event["payload"].get("import"):
                    key = event["payload"]["import"]["origin_sha256"]
                    ancestor = value.get("ancestor_origins", {}).get(key)
                    if ancestor is None:
                        raise SkillLibraryError("missing transitive import provenance")
                    validate(ancestor, key, depth + 1)
            chain = value["journal"]
            candidates = [event for event in chain if event["event_id"] == value["candidate_event_id"]
                          and event["kind"] == "candidate_committed"]
            if (not chain or chain[0]["kind"] != "experiment_created"
                    or chain[0]["payload"] != {"experiment_sha256": _hash(canonical_json(value["experiment"]))}
                    or len(candidates) != 1
                    or candidates[0]["payload"]["bundle_hash"] != value["bundle_hash"]):
                raise SkillLibraryError("import candidate origin mismatch")
        validate(origin, digest)
        if (origin["candidate_event_id"] != imported["source_candidate_event_id"]
                or origin["experiment"]["experiment_id"] != imported["source_experiment_id"]
                or origin["bundle_hash"] != imported["source_bundle_hash"]
                or origin["bundle_hash"] != payload["bundle_hash"]):
            raise SkillLibraryError("import origin binding mismatch")

    def verify(self, bundle_hash: str) -> dict[str, Any]:
        directory = self.bundles / _digest(bundle_hash)
        checked_directory(directory)
        if {path.name for path in directory.iterdir()} != {"manifest.json", "payload"}:
            raise SkillLibraryError("unexpected bundle contents")
        raw = read_bytes(directory / "manifest.json")
        manifest = strict_json_loads(raw)
        if (not isinstance(manifest, dict) or set(manifest) != {
                "protocol", "files", "catalog", "dependencies", "compatibility"
            } or manifest["protocol"] != PROTOCOL or raw != canonical_json(manifest)
                or _bundle_hash(manifest) != bundle_hash):
            raise SkillLibraryError("bundle manifest hash or schema mismatch")
        files = _file_map(directory / "payload")
        if _inventory(files) != manifest["files"]:
            raise SkillLibraryError("bundle payload hash mismatch")
        catalog = _validate_payload(files, _dependencies(manifest["dependencies"]))
        if catalog != manifest["catalog"] or not isinstance(manifest["compatibility"], dict):
            raise SkillLibraryError("bundle catalog or compatibility mismatch")
        return manifest

    def read_bundle_file(self, candidate_event_id: str, path: str) -> bytes:
        with self.journal.transaction():
            candidate = self._candidate(candidate_event_id)
            bundle_hash = candidate["payload"]["bundle_hash"]
            self.verify(bundle_hash)
            return read_bytes(self.bundles / bundle_hash / "payload" / _payload_path(path), limit=MAX_FILE_BYTES)

    def _seal_stage(self, stage: Path, files: dict[str, bytes], manifest: dict[str, Any]) -> None:
        stage.mkdir(mode=0o700)
        payload = stage / "payload"
        payload.mkdir(mode=0o700)
        for path, data in sorted(files.items()):
            destination = payload / path
            checked_directory(destination.parent, create=True)
            atomic_write_bytes(destination, data, replace=False, mode=0o400)
        atomic_write_bytes(stage / "manifest.json", canonical_json(manifest), replace=False, mode=0o400)
        for directory, _, _ in os.walk(stage, topdown=False):
            # Moving a directory between parents can require write permission on
            # the directory itself. Seal its root after the atomic rename, before
            # committing the candidate that makes it eligible for execution.
            os.chmod(directory, 0o700 if Path(directory) == stage else 0o500)
            fsync_directory(directory)
        fsync_directory(self.staging)

    def _complete_prepared(self, event: dict[str, Any]) -> dict[str, Any]:
        transaction_id = event["transaction_id"]
        payload = event["payload"]
        bundle_hash = payload["bundle_hash"]
        destination = self.bundles / _digest(bundle_hash)
        stage = self.staging / _identifier(transaction_id)
        if destination.exists():
            self.verify(bundle_hash)
        elif stage.exists():
            # Validate the complete staged tree before making it visible.
            manifest = read_json(stage / "manifest.json")
            if (_bundle_hash(manifest) != bundle_hash
                    or _inventory(_file_map(stage / "payload")) != manifest["files"]):
                raise SkillLibraryError("prepared stage is corrupt")
            rename_noreplace(stage, destination)
            fsync_directory(self.bundles)
            fsync_directory(self.staging)
            os.chmod(destination, 0o500)
            fsync_directory(destination)
            self.verify(bundle_hash)
        else:
            raise SkillLibraryError("prepared publication has neither stage nor bundle")
        os.chmod(destination, 0o500)
        fsync_directory(destination)
        self._verify_origin(payload)
        return self.journal.append("candidate_committed", {
            **payload, "prepared_event_id": event["event_id"],
        }, event_id="candidate-" + transaction_id, transaction_id=transaction_id)

    def publish(self, expected_revision: str, *, transaction_id: str, reason: str,
                parents: list[str] | None = None, source_events: list[str] | None = None,
                phase: str = "practice") -> dict[str, Any]:
        self._require_practice(phase)
        _identifier(transaction_id)
        if not isinstance(reason, str) or not reason:
            raise SkillLibraryError("candidate change reason is required")
        for references in (parents, source_events):
            if references is not None and (not isinstance(references, list)
                    or any(not isinstance(item, str) or not item for item in references)
                    or len(set(references)) != len(references)):
                raise SkillLibraryError("provenance references must be unique event ID lists")
        request = {"author_revision": expected_revision, "reason": reason,
                   "parents": parents or [], "source_events": source_events or [], "import": None}
        with self.journal.transaction():
            events = self.journal.read()
            for event in events:
                if event["transaction_id"] == transaction_id and event["kind"] == "publication_prepared":
                    if any(event["payload"][key] != value for key, value in request.items()):
                        raise SkillLibraryError("publication transaction reused with different request")
                    return self._complete_prepared(event)
            for parent in request["parents"]:
                self._candidate(parent)
            event_ids = {event["event_id"] for event in events}
            if any(event_id not in event_ids for event_id in request["source_events"]):
                raise SkillLibraryError("source event does not belong to this library")
            files = _file_map(self.author)
            if _revision(files) != expected_revision:
                raise SkillLibraryError("stale author revision")
            dependencies = _dependencies(self.contract.get("dependencies", {}))
            catalog = _validate_payload(files, dependencies)
            manifest = {"protocol": PROTOCOL, "files": _inventory(files), "catalog": catalog,
                        "dependencies": dependencies, "compatibility": self.contract.get("compatibility", {})}
            bundle_hash = _bundle_hash(manifest)
            stage = self.staging / transaction_id
            if stage.exists() or stage.is_symlink():
                raise SkillLibraryError("uncommitted staging exists; recover before publishing")
            if not (self.bundles / bundle_hash).exists():
                self._seal_stage(stage, files, manifest)
            else:
                self.verify(bundle_hash)
            prepared = self.journal.append("publication_prepared", {**request, "bundle_hash": bundle_hash},
                event_id="prepared-" + transaction_id, transaction_id=transaction_id)
            return self._complete_prepared(prepared)

    def recover(self) -> dict[str, Any]:
        """Resume only recorded publications; quarantine uncommitted stages."""
        with self.journal.transaction():
            events = self.journal.read()
            prepared = {event["transaction_id"]: event for event in events
                        if event["kind"] == "publication_prepared"}
            committed = {event["transaction_id"] for event in events if event["kind"] == "candidate_committed"}
            resumed = []
            for transaction_id, event in prepared.items():
                if transaction_id not in committed:
                    resumed.append(self._complete_prepared(event)["event_id"])
            quarantine = checked_directory(self.host / "quarantine", create=True)
            quarantined = []
            for stage in self.staging.iterdir():
                # Duplicate complete content is retained as an audit artifact too.
                if not stat.S_ISDIR(stage.lstat().st_mode):
                    raise SkillLibraryError("invalid staged object")
                target = quarantine / (stage.name + "-" + uuid.uuid4().hex)
                rename_noreplace(stage, target)
                fsync_directory(quarantine)
                fsync_directory(self.staging)
                quarantined.append(target.name)
                self.journal.append("stage_quarantined", {"stage": stage.name, "quarantine": target.name})
            known = {event["payload"]["bundle_hash"] for event in prepared.values()}
            orphans = sorted(path.name for path in self.bundles.iterdir() if path.name not in known)
            selected = self.selected()
            return {"resumed_candidates": resumed, "quarantined_stages": quarantined,
                    "orphan_bundles_not_authorized": orphans, "selected": selected}

    def record_validation(self, candidate_event_id: str, evidence: dict[str, Any], *,
                          transaction_id: str) -> dict[str, Any]:
        _identifier(transaction_id)
        with self.journal.transaction():
            candidate = self._candidate(candidate_event_id)
            bundle_hash = candidate["payload"]["bundle_hash"]
            self.verify(bundle_hash)
            if not isinstance(evidence, dict) or not {
                "source", "protocol_id", "slot_id", "attempt_id", "task_outcome", "valid",
                "execution_status", "cost"
            }.issubset(evidence):
                raise SkillLibraryError("validation evidence fields missing")
            if (evidence["source"] != "bench" or evidence["task_outcome"] not in ("success", "failure", "unknown")
                    or type(evidence["valid"]) is not bool
                    or not all(isinstance(evidence[key], str) and evidence[key]
                               for key in ("protocol_id", "slot_id", "attempt_id", "execution_status"))):
                raise SkillLibraryError("invalid validation evidence")
            if evidence.get("task_scoring_mode") == "continuous":
                _continuous_evidence(evidence)
            elif evidence["valid"] and evidence["task_outcome"] == "unknown":
                raise SkillLibraryError("unknown outcome is not valid task evidence")
            payload = {"candidate_event_id": candidate_event_id, "bundle_hash": bundle_hash,
                       "evidence": evidence}
            event = self.journal.append("validation_recorded", payload,
                event_id="validation-" + transaction_id, transaction_id=transaction_id)
            # Evidence is immutable and independent of the bundle. The journal is
            # authoritative; a missing sidecar can be reconstructed from this event.
            evidence_path = self.host / "evidence" / (event["event_id"] + ".json")
            if evidence_path.exists():
                if read_json(evidence_path) != event:
                    raise SkillLibraryError("validation sidecar mismatch")
            else:
                atomic_write_json(evidence_path, event, replace=False)
            if evidence["valid"]:
                self.journal.append("candidate_tested", {"candidate_event_id": candidate_event_id,
                    "bundle_hash": bundle_hash, "validation_event_id": event["event_id"]},
                    event_id="tested-" + transaction_id, transaction_id=transaction_id)
            return event

    def select(self, candidate_event_id: str, *, validation_event_ids: list[str],
               gate: dict[str, Any], transaction_id: str) -> dict[str, Any]:
        _identifier(transaction_id)
        with self.journal.transaction():
            candidate = self._candidate(candidate_event_id)
            bundle_hash = candidate["payload"]["bundle_hash"]
            self.verify(bundle_hash)
            if (not isinstance(gate, dict) or gate.get("approved") is not True
                    or not isinstance(gate.get("protocol_id"), str) or not gate["protocol_id"]
                    or not validation_event_ids):
                raise SkillLibraryError("selection requires an explicit bench protocol gate")
            events = self.journal.read()
            by_id = {event["event_id"]: event for event in events}
            for event_id in validation_event_ids:
                event = by_id.get(event_id)
                if (not event or event["kind"] != "validation_recorded"
                        or event["payload"]["candidate_event_id"] != candidate_event_id
                        or not event["payload"]["evidence"]["valid"]
                        or event["payload"]["evidence"]["protocol_id"] != gate["protocol_id"]):
                    raise SkillLibraryError("selection evidence does not qualify this candidate")
            request = {"candidate_event_id": candidate_event_id, "bundle_hash": bundle_hash,
                       "validation_event_ids": validation_event_ids, "gate": gate}
            old = by_id.get("selected-" + transaction_id)
            if old:
                if any(old["payload"].get(key) != value for key, value in request.items()):
                    raise SkillLibraryError("selection transaction reused with different request")
                return {"bundle_hash": old["payload"]["bundle_hash"],
                        "candidate_event_id": old["payload"]["candidate_event_id"],
                        "selection_event_id": old["event_id"]}
            previous = self.selected()
            event = self.journal.append("candidate_selected", {**request, "previous": previous},
                event_id="selected-" + transaction_id, transaction_id=transaction_id)
            selected = {"bundle_hash": bundle_hash, "candidate_event_id": candidate_event_id,
                        "selection_event_id": event["event_id"]}
            atomic_write_json(self.host / "selected.json", selected)
            return selected

    def selected(self) -> dict[str, Any] | None:
        with self.journal.transaction():
            selected = None
            for event in self.journal.read():
                if event["kind"] == "candidate_selected":
                    selected = {"bundle_hash": event["payload"]["bundle_hash"],
                                "candidate_event_id": event["payload"]["candidate_event_id"],
                                "selection_event_id": event["event_id"]}
            if selected is not None:
                self._candidate(selected["candidate_event_id"])
                self.verify(selected["bundle_hash"])
            cache = self.host / "selected.json"
            if cache.exists() or cache.is_symlink():
                cached = read_json(cache)
                if cached != selected:
                    # A previous valid pointer is expected if the host crashed
                    # after the journal commit but before replacing this cache.
                    valid_old = any(event["kind"] == "candidate_selected" and cached == {
                        "bundle_hash": event["payload"]["bundle_hash"],
                        "candidate_event_id": event["payload"]["candidate_event_id"],
                        "selection_event_id": event["event_id"]
                    } for event in self.journal.read())
                    if not valid_old:
                        raise SkillLibraryError("selected cache has no matching journal provenance")
                    atomic_write_json(cache, selected)
            elif selected is not None:
                atomic_write_json(cache, selected, replace=False)
            return selected

    def import_candidate(self, source: SkillLibrary, source_candidate_event_id: str, *,
                         transaction_id: str, provenance: dict[str, Any],
                         phase: str = "practice") -> dict[str, Any]:
        """Trusted-host operation on a preauthorized source object, not a model path.

        Imports immutable content and the complete source event chain. It leaves
        author unchanged and never inherits source validation/selection rights.
        """
        self._require_practice(phase)
        _identifier(transaction_id)
        if source.path == self.path:
            raise SkillLibraryError("source must be a separate library")
        if not isinstance(provenance, dict) or not {"scenes", "phase", "cost", "licenses"}.issubset(provenance):
            raise SkillLibraryError("import provenance requires scenes, phase, cost and licenses")
        if provenance["phase"] not in ("practice", "declared_prior"):
            raise SkillLibraryError("evaluation experience cannot be imported as practice")
        # A retry resolves the original transaction before looking at a source
        # journal that may have acquired additional costs/evidence in the meantime.
        with self.journal.transaction():
            for event in self.journal.read():
                if event["transaction_id"] == transaction_id and event["kind"] == "publication_prepared":
                    imported = event["payload"].get("import")
                    if imported is None:
                        raise SkillLibraryError("transaction belongs to a different operation")
                    self._verify_origin(event["payload"])
                    saved = read_json(self.host / "origins" / (imported["origin_sha256"] + ".json"))
                    if (saved["experiment"] != source.experiment
                            or saved["candidate_event_id"] != source_candidate_event_id
                            or saved["declaration"] != provenance):
                        raise SkillLibraryError("import transaction reused with different request")
                    return self._complete_prepared(event)
        # Lock ordering avoids deadlocks when two hosts import in opposite directions;
        # lock acquisition is nonblocking, and only trusted hosts call this method.
        with source.journal.transaction():
            candidate = source._candidate(source_candidate_event_id)
            bundle_hash = candidate["payload"]["bundle_hash"]
            manifest = source.verify(bundle_hash)
            files = _file_map(source.bundles / bundle_hash / "payload")
            source_chain = source.journal.read()
            ancestors = {}
            for event in source_chain:
                if event["kind"] == "candidate_committed" and event["payload"].get("import"):
                    origin_key = event["payload"]["import"]["origin_sha256"]
                    source._verify_origin(event["payload"])
                    ancestors[origin_key] = read_json(source.host / "origins" / (origin_key + ".json"))
            origin = {"experiment": source.experiment, "candidate_event_id": source_candidate_event_id,
                      "bundle_hash": bundle_hash, "journal": source_chain, "declaration": provenance,
                      "ancestor_origins": ancestors}
            origin_bytes = canonical_json(origin)
            if len(origin_bytes) > 16 * 1024 * 1024:
                raise SkillLibraryError("import provenance exceeds storage quota")
            origin_hash = _hash(origin_bytes)
        with self.journal.transaction():
            request = {"author_revision": None, "reason": "explicit immutable import", "parents": [],
                       "source_events": [], "import": {"origin_sha256": origin_hash,
                       "source_experiment_id": source.experiment["experiment_id"],
                       "source_candidate_event_id": source_candidate_event_id,
                       "source_bundle_hash": bundle_hash}}
            for event in self.journal.read():
                if event["transaction_id"] == transaction_id and event["kind"] == "publication_prepared":
                    if any(event["payload"][key] != value for key, value in request.items()):
                        raise SkillLibraryError("import transaction reused with different request")
                    return self._complete_prepared(event)
            origins = checked_directory(self.host / "origins", create=True)
            origin_path = origins / (origin_hash + ".json")
            if origin_path.exists():
                if read_json(origin_path) != origin:
                    raise SkillLibraryError("import origin mismatch")
            else:
                atomic_write_json(origin_path, origin, replace=False)
            if not (self.bundles / bundle_hash).exists():
                self._seal_stage(self.staging / transaction_id, files, manifest)
            else:
                self.verify(bundle_hash)
            prepared = self.journal.append("publication_prepared", {**request, "bundle_hash": bundle_hash},
                event_id="prepared-" + transaction_id, transaction_id=transaction_id)
            return self._complete_prepared(prepared)

    def status(self) -> dict[str, Any]:
        with self.journal.transaction():
            events = self.journal.read()
            candidates = [event for event in events if event["kind"] == "candidate_committed"]
            return {"experiment_id": self.experiment["experiment_id"],
                    "author_revision": self.author_revision(), "candidate_count": len(candidates),
                    "selected": self.selected(), "event_count": len(events),
                    "runtime_isolation_implemented": False}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    init = commands.add_parser("init")
    init.add_argument("experiment_id")
    init.add_argument("--learning-root", type=Path, default=DEFAULT_LEARNING_ROOT)
    init.add_argument("--contract", type=Path, required=True)
    for name in ("status", "recover", "publish", "verify"):
        command = commands.add_parser(name)
        command.add_argument("library", type=Path)
        if name == "publish":
            command.add_argument("--revision", required=True)
            command.add_argument("--transaction-id", required=True)
            command.add_argument("--reason", required=True)
        elif name == "verify":
            command.add_argument("bundle_hash")
    args = parser.parse_args(argv)
    if args.command == "init":
        result = SkillLibrary.create(args.experiment_id, learning_root=args.learning_root,
                                     contract=read_json(args.contract)).status()
    else:
        library = SkillLibrary(args.library)
        if args.command == "publish":
            result = library.publish(args.revision, transaction_id=args.transaction_id, reason=args.reason)
        elif args.command == "verify":
            result = library.verify(args.bundle_hash)
        else:
            result = getattr(library, args.command)()
    print(canonical_json(result).decode())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
